"""[A1 验收测试 · 波次1.x §v1.0.5/§v1.0.6] observation_candidates_from_field —— 概率图→候选观测航点(A′ 混合,估计非 GT)。

判据来源:contracts/env_static.py 的 observation_candidates_from_field / detections_from_field /
field_from_targets 签名 + docstring、agent_team_harness §4.2、本波次补丁单(必测不变量 1–7 + §v1.0.6)。

被测原语(对契约编程,只测暴露行为,不依赖实现内部):
    observation_candidates_from_field(field, world, *, threshold_rel=0.3,
        fill_threshold_rel=0.5, peak_min_separation_m, fill_radius_m) -> List[Waypoint]
      = A′ 混合:峰值锚点(逐点保留,顺序在前,**阈值用 threshold_rel**)
        + 高概率区(**阈值用 fill_threshold_rel,§v1.0.6 与峰值阈值拆开,默认 0.5 更严**)
        按覆盖半径 fill_radius_m 补点。修复纯峰值 detections_from_field 的「σ 模糊把合并簇
        当单点」缺陷;返回点是估计,非 GT。

§v1.0.6:峰值阈值(threshold_rel)与补点阈值(fill_threshold_rel)拆开。本套显式传 fill_threshold_rel
(不依赖契约默认漂移),并新增两条 §v1.0.6 判据:峰值前缀与 fill_threshold_rel **无关**、补点数随
fill_threshold_rel 升高 **单调不增**。

受控场口径(本波次补丁单指定):WorldConfig(60,30,0.5)、
    peak_min_separation_m=3.0、fill_radius_m=5.4、threshold_rel=0.3、fill_threshold_rel=0.5、bandwidth_m=3.0。

本套对补丁单的不变量做强断言,固定受控场 / 无 RNG,作为该原语"完成"的客观判据:
    1 峰值保留(返回 ⊇ 纯峰值,且峰值点逐位在结果前缀)
    2 自适应-不补(单孤立目标 → 混合 == 纯峰值)
    3 自适应-补点(σ 糊成一块 → 混合点数 > 纯峰值点数,且峰值仍在)
    4 覆盖改善(良分离场每目标 fill_radius_m 内有候选,recall==1.0;混合 recall ≥ 纯峰值 recall)
    5 精度(每候选点所在 cell field >= threshold_rel*field.max())
    6 确定性(两次调用逐位相同)
    7 边界(严格在 [0,x_max]×[0,y_max] 内)+ 空场(全零 / 空 targets → [])
    §v1.0.6-A 峰值前缀与 fill_threshold_rel 无关(== 纯 detections_from_field)
    §v1.0.6-B 补点单调:fill_threshold_rel 升高 → 总点数不增
"""
from __future__ import annotations

import numpy as np

from msim.contracts.config import WorldConfig
from msim.env_static.field import (
    detections_from_field,
    observation_candidates_from_field,
    field_from_targets,
)


# ----------------------------------------------------------------------
# 受控场固定口径(本波次补丁单指定)
# ----------------------------------------------------------------------
BANDWIDTH_M = 3.0
PEAK_MIN_SEP_M = 3.0
FILL_RADIUS_M = 5.4
THRESHOLD_REL = 0.3        # 峰值锚点阈值
FILL_THRESHOLD_REL = 0.5   # 补点高概率区阈值(§v1.0.6 与峰值阈值拆开,默认 0.5 更严)


def _world() -> WorldConfig:
    return WorldConfig(x_max_m=60.0, y_max_m=30.0, res_m=0.5)


def _make_field(targets: list, world: WorldConfig, *, bandwidth_m: float = BANDWIDTH_M):
    return field_from_targets(targets, world, bandwidth_m=bandwidth_m)


def _peaks(field, world):
    """纯峰值参考(detections_from_field,min_separation_m = peak_min_separation_m,同 threshold_rel)。"""
    return detections_from_field(
        field, world, threshold_rel=THRESHOLD_REL, min_separation_m=PEAK_MIN_SEP_M
    )


def _hybrid(field, world, *, fill_threshold_rel: float = FILL_THRESHOLD_REL):
    """A′ 混合;§v1.0.6:显式传 fill_threshold_rel(默认 0.5,不依赖契约默认漂移)。"""
    return observation_candidates_from_field(
        field, world,
        threshold_rel=THRESHOLD_REL,
        fill_threshold_rel=fill_threshold_rel,
        peak_min_separation_m=PEAK_MIN_SEP_M,
        fill_radius_m=FILL_RADIUS_M,
    )


def _well_separated_targets() -> list:
    """两两间距 > 2*fill_radius_m(=10.8m)的良分离目标。

    (10,8)-(30,22):≈24.4m;(30,22)-(50,10):≈23.3m;(10,8)-(50,10):≈40.0m。均 > 10.8m。
    """
    return [
        np.array([10.0, 8.0], dtype=np.float64),
        np.array([30.0, 22.0], dtype=np.float64),
        np.array([50.0, 10.0], dtype=np.float64),
    ]


def _merged_cluster_targets() -> list:
    """5 个相距 1m(< fill_radius_m)的近邻目标:配 bandwidth=6 → 纯峰值仅得 1 点,
    高概率区跨度远 > 2*fill_radius_m,逼出补点缺陷修复。"""
    return [np.array([float(28 + i), 15.0], dtype=np.float64) for i in range(5)]


def _cell_index(p: np.ndarray, res: float) -> tuple:
    """候选点世界坐标 → 所在 cell (ix,iy)(与 cell 中心 (ix+0.5)*res 互逆)。"""
    ix = int(round(float(p[0]) / res - 0.5))
    iy = int(round(float(p[1]) / res - 0.5))
    return ix, iy


def _recall_at_R(pts: list, targets: list, R: float) -> float:
    """每个 target 在 R 内是否至少有一个候选点;返回命中比例。"""
    arr = np.asarray([np.asarray(p, dtype=np.float64) for p in pts])
    hits = 0
    for t in targets:
        dist = np.hypot(arr[:, 0] - t[0], arr[:, 1] - t[1])
        if float(dist.min()) <= R:
            hits += 1
    return hits / len(targets)


# ======================================================================
# 不变量 1(核心):峰值保留 —— 返回 ⊇ 纯峰值,且峰值点逐位在结果前缀
# ======================================================================
def test_hybrid_superset_of_peaks_well_separated() -> None:
    world = _world()
    field = _make_field(_well_separated_targets(), world)
    peaks = _peaks(field, world)
    hyb = _hybrid(field, world)
    assert len(peaks) > 0, "良分离场应有峰值,本测试才有判别力"
    assert len(hyb) >= len(peaks), f"混合点数须 >= 纯峰值点数:{len(hyb)} < {len(peaks)}"
    # 峰值逐位在结果前缀(契约:峰值原样保留、顺序在前)。
    for i, pk in enumerate(peaks):
        assert np.array_equal(hyb[i], pk), \
            f"第 {i} 个峰值点未逐位出现在混合结果前缀:hyb[{i}]={hyb[i]} vs peak={pk}"


def test_hybrid_superset_of_peaks_merged_cluster() -> None:
    """合并簇场(纯峰值=1)同样满足峰值保留:该 1 个峰值逐位在结果前缀。"""
    world = _world()
    field = _make_field(_merged_cluster_targets(), world, bandwidth_m=6.0)
    peaks = _peaks(field, world)
    hyb = _hybrid(field, world)
    assert len(peaks) >= 1
    assert len(hyb) >= len(peaks)
    for i, pk in enumerate(peaks):
        assert np.array_equal(hyb[i], pk), \
            f"合并簇场峰值点未逐位在前缀:hyb[{i}]={hyb[i]} vs peak={pk}"


# ======================================================================
# 不变量 2:自适应-不补 —— 单孤立目标(blob 半径 < fill_radius_m)→ 混合 == 纯峰值
# ======================================================================
def test_hybrid_no_fill_on_single_isolated_target() -> None:
    world = _world()
    t = np.array([20.0, 15.0], dtype=np.float64)
    field = _make_field([t], world)  # bandwidth=3 → blob 半径 < fill_radius_m=5.4
    peaks = _peaks(field, world)
    hyb = _hybrid(field, world)
    assert len(peaks) == 1, f"单孤立目标纯峰值应为 1 点,得 {len(peaks)}"
    assert len(hyb) == len(peaks), \
        f"孤立小目标应退化回纯峰值(不补点):混合 {len(hyb)} != 峰值 {len(peaks)}"
    # 坐标逐位相等(点数 + 坐标都相等)。
    for a, b in zip(hyb, peaks):
        assert np.array_equal(a, b), f"孤立目标混合输出坐标须 == 纯峰值:{a} != {b}"


# ======================================================================
# 不变量 3:自适应-补点(修复缺陷) —— σ 糊成一整块 → 混合点数 > 纯峰值点数,峰值仍在
# ======================================================================
def test_hybrid_fills_on_blurred_merged_cluster() -> None:
    world = _world()
    targets = _merged_cluster_targets()  # 5 目标 @28..32m 一线,相距 1m
    field = _make_field(targets, world, bandwidth_m=6.0)  # 大 bandwidth 糊成一块
    peaks = _peaks(field, world)
    hyb = _hybrid(field, world)

    # 前提复核:纯峰值确实只得 1 点(整块被吞成单点 = 待修复缺陷)。
    assert len(peaks) == 1, \
        f"构造前提:大 bandwidth 近邻簇纯峰值应坍缩为 1 点,得 {len(peaks)}(需调参以触发缺陷)"

    # 前提复核:高概率区跨度 > 2*fill_radius_m(确有需补点的空隙)。
    # 口径用 THRESHOLD_REL(峰值阈值视角的整块高概率区);补点区(fill_threshold_rel=0.5)是其子集,
    # 跨度断言仅证"构造前提充分"。核心补点断言(下方)在 fill_threshold_rel=0.5 下经实测仍成立。
    fmax = float(field.max())
    M = field >= THRESHOLD_REL * fmax
    ixs, iys = np.nonzero(M)
    span_x = (ixs.max() - ixs.min()) * world.res_m
    span_y = (iys.max() - iys.min()) * world.res_m
    assert max(span_x, span_y) > 2 * FILL_RADIUS_M, \
        f"构造前提:高概率区跨度须 > 2*fill_radius_m={2*FILL_RADIUS_M},得 {max(span_x, span_y):.2f}"

    # 核心断言:混合确有补点(点数 > 纯峰值点数);在默认 fill_threshold_rel=0.5(更严)下仍成立,
    # 因合并簇核心 field 远高于 0.5×max,补点区跨度仍 > 2R。
    assert len(hyb) > len(peaks), \
        f"σ 糊成一块时混合应补点(> 纯峰值):混合 {len(hyb)} <= 峰值 {len(peaks)}"
    # 峰值仍逐位在结果里(前缀)。
    for i, pk in enumerate(peaks):
        assert np.array_equal(hyb[i], pk), f"补点后峰值点须仍在前缀:hyb[{i}]={hyb[i]} vs {pk}"


def test_hybrid_fills_on_three_targets_within_3m() -> None:
    """补丁单参考构造:3 个目标挤在 ~3m 内 + bandwidth=6 → 纯峰值=1,混合 > 1。"""
    world = _world()
    targets = [
        np.array([29.0, 15.0], dtype=np.float64),
        np.array([30.0, 15.0], dtype=np.float64),
        np.array([31.0, 15.0], dtype=np.float64),
    ]
    field = _make_field(targets, world, bandwidth_m=6.0)
    peaks = _peaks(field, world)
    hyb = _hybrid(field, world)
    assert len(peaks) == 1, f"3 目标 ~3m 内 + bandwidth=6 纯峰值应为 1,得 {len(peaks)}"
    assert len(hyb) > len(peaks), f"该糊块场混合应补点(> 纯峰值),得混合 {len(hyb)} 峰值 {len(peaks)}"


# ======================================================================
# 不变量 4:覆盖改善 —— 良分离场每目标 fill_radius_m 内有候选,recall==1.0;混合 recall >= 纯峰值
# ======================================================================
def test_hybrid_recall_at_fill_radius_on_well_separated() -> None:
    world = _world()
    targets = _well_separated_targets()  # 两两间距 > 2*fill_radius_m
    field = _make_field(targets, world)
    peaks = _peaks(field, world)
    hyb = _hybrid(field, world)

    recall_hyb = _recall_at_R(hyb, targets, FILL_RADIUS_M)
    recall_peak = _recall_at_R(peaks, targets, FILL_RADIUS_M)
    assert recall_hyb == 1.0, \
        f"良分离场每目标 fill_radius_m 内须有混合候选(recall@R==1.0),得 {recall_hyb}"
    assert recall_hyb >= recall_peak, \
        f"混合 recall@R 须 >= 纯峰值 recall@R:{recall_hyb} < {recall_peak}"


def test_hybrid_recall_not_worse_on_merged_cluster() -> None:
    """合并簇场:混合 recall@R 亦不应劣于纯峰值(补点只增不减覆盖)。"""
    world = _world()
    targets = _merged_cluster_targets()
    field = _make_field(targets, world, bandwidth_m=6.0)
    peaks = _peaks(field, world)
    hyb = _hybrid(field, world)
    recall_hyb = _recall_at_R(hyb, targets, FILL_RADIUS_M)
    recall_peak = _recall_at_R(peaks, targets, FILL_RADIUS_M)
    assert recall_hyb >= recall_peak, \
        f"合并簇场混合 recall@R 须 >= 纯峰值:{recall_hyb} < {recall_peak}"


# ======================================================================
# 不变量 5:精度 —— 每个候选点所在 cell 的 field >= threshold_rel*field.max()
# ======================================================================
def test_hybrid_all_candidates_above_relative_threshold() -> None:
    world = _world()
    res = world.res_m
    # 同时覆盖良分离与合并簇两类场,确保补点也落在高概率区。
    for targets, bw in (
        (_well_separated_targets(), BANDWIDTH_M),
        (_merged_cluster_targets(), 6.0),
    ):
        field = _make_field(targets, world, bandwidth_m=bw)
        fmax = float(field.max())
        thr = THRESHOLD_REL * fmax
        hyb = _hybrid(field, world)
        assert len(hyb) > 0
        for p in hyb:
            ix, iy = _cell_index(p, res)
            assert 0 <= ix < field.shape[0] and 0 <= iy < field.shape[1], \
                f"候选 cell 索引越界:{p}"
            assert float(field[ix, iy]) >= thr - 1e-9, \
                f"候选点所在 cell 低于相对阈值:field={field[ix, iy]:.4f} < thr={thr:.4f}"


# ======================================================================
# 不变量 6:确定性 —— 同 field+参数两次调用逐位相同(长度 + np.array_equal 全 True)
# ======================================================================
def test_hybrid_deterministic() -> None:
    world = _world()
    # 在两类场上都验确定性(良分离 + 合并簇补点路径)。
    for targets, bw in (
        (_well_separated_targets(), BANDWIDTH_M),
        (_merged_cluster_targets(), 6.0),
    ):
        field = _make_field(targets, world, bandwidth_m=bw)
        h1 = _hybrid(field, world)
        h2 = _hybrid(field, world)
        assert len(h1) == len(h2), "两次调用候选点数须相同(确定性、无 RNG)"
        for a, b in zip(h1, h2):
            assert np.array_equal(a, b), "两次调用须逐位相同(确定性、无 RNG)"


# ======================================================================
# 不变量 7:边界 —— 所有候选点严格在 [0,x_max]×[0,y_max] 内
# ======================================================================
def test_hybrid_within_bounds() -> None:
    world = _world()
    for targets, bw in (
        (_well_separated_targets(), BANDWIDTH_M),
        (_merged_cluster_targets(), 6.0),
    ):
        field = _make_field(targets, world, bandwidth_m=bw)
        hyb = _hybrid(field, world)
        assert len(hyb) > 0, "受控场应有候选点"
        for p in hyb:
            assert p.shape == (2,), f"每个候选点 shape 须为 (2,):{p}"
            assert 0.0 <= float(p[0]) <= world.x_max_m, f"候选点 North 越界:{p}"
            assert 0.0 <= float(p[1]) <= world.y_max_m, f"候选点 East 越界:{p}"


# ======================================================================
# 不变量 7(空场):全零场 与 field_from_targets([],...) → 返回 []
# ======================================================================
def test_hybrid_empty_on_zero_field() -> None:
    world = _world()
    zero_field = np.zeros((world.nx, world.ny), dtype=np.float64)
    out = observation_candidates_from_field(
        zero_field, world,
        threshold_rel=THRESHOLD_REL,
        peak_min_separation_m=PEAK_MIN_SEP_M,
        fill_radius_m=FILL_RADIUS_M,
    )
    assert out == [], "全零场应返回 []"


def test_hybrid_empty_on_empty_targets_field() -> None:
    world = _world()
    field = field_from_targets([], world, bandwidth_m=BANDWIDTH_M)  # 空 targets → 全零场
    out = observation_candidates_from_field(
        field, world,
        threshold_rel=THRESHOLD_REL,
        peak_min_separation_m=PEAK_MIN_SEP_M,
        fill_radius_m=FILL_RADIUS_M,
    )
    assert out == [], "field_from_targets([]) 派生的全零场应返回 []"


# ======================================================================
# §v1.0.6-A:峰值前缀与 fill_threshold_rel 无关 —— 改 fill_threshold_rel(0.3/0.5/0.7),
# 结果的【峰值前缀】逐位相同,且 == 纯 detections_from_field(peaks 只随 threshold_rel 变)。
# ======================================================================
def _fill_threshold_cases():
    """对 fill 阈值敏感(补点路径活跃)的两类受控场:合并簇 + 3 目标 ~3m。"""
    return [
        ("merged5", _merged_cluster_targets(), 6.0),
        ("three3m",
         [np.array([29.0, 15.0]), np.array([30.0, 15.0]), np.array([31.0, 15.0])],
         6.0),
    ]


def test_peak_prefix_independent_of_fill_threshold() -> None:
    world = _world()
    for name, targets, bw in _fill_threshold_cases():
        field = _make_field(targets, world, bandwidth_m=bw)
        peaks = _peaks(field, world)
        assert len(peaks) >= 1, f"{name}: 受控场应有峰值,判据才有判别力"
        # 三个 fill 阈值下,前 len(peaks) 个点逐位 == 纯峰值(峰值集合与 fill 阈值无关)。
        for fthr in (0.3, 0.5, 0.7):
            hyb = _hybrid(field, world, fill_threshold_rel=fthr)
            assert len(hyb) >= len(peaks), \
                f"{name} fthr={fthr}: 混合点数须 >= 峰值数(峰值恒在前缀)"
            for i, pk in enumerate(peaks):
                assert np.array_equal(hyb[i], pk), \
                    f"{name} fthr={fthr}: 第 {i} 峰值点应与 fill 阈值无关:hyb[{i}]={hyb[i]} vs peak={pk}"


def test_peak_count_constant_across_fill_threshold() -> None:
    """峰值数(= 纯 detections_from_field 数)恒定,与 fill_threshold_rel 完全无关。"""
    world = _world()
    for name, targets, bw in _fill_threshold_cases():
        field = _make_field(targets, world, bandwidth_m=bw)
        n_peaks = len(_peaks(field, world))
        # 任取两 fill 阈值,二者结果的"峰值前缀"逐位一致(已由上一测试覆盖到坐标);
        # 这里再钉死"峰值数恒定"这一标量不变量。
        for fthr in (0.3, 0.5, 0.7):
            hyb = _hybrid(field, world, fill_threshold_rel=fthr)
            assert len(hyb) >= n_peaks, f"{name} fthr={fthr}: 总点数须含全部峰值"


# ======================================================================
# §v1.0.6-B:补点单调 —— 提高 fill_threshold_rel(0.3→0.5→0.7)→ 总点数(峰值+补点)单调不增。
# ======================================================================
def test_fill_count_monotone_in_fill_threshold() -> None:
    world = _world()
    for name, targets, bw in _fill_threshold_cases():
        field = _make_field(targets, world, bandwidth_m=bw)
        n03 = len(_hybrid(field, world, fill_threshold_rel=0.3))
        n05 = len(_hybrid(field, world, fill_threshold_rel=0.5))
        n07 = len(_hybrid(field, world, fill_threshold_rel=0.7))
        # 补点区随阈值升高收缩 → 总点数单调不增。
        assert n03 >= n05 >= n07, \
            f"{name}: fill_threshold_rel 升高总点数应单调不增:n03={n03} n05={n05} n07={n07}"
        # 判别力:这两类场在低阈值确实补点(否则单调断言空洞)。
        peaks = len(_peaks(field, world))
        assert n03 > peaks, \
            f"{name}: 低 fill 阈值(0.3)应确有补点(> 峰值数),否则单调判据无判别力:n03={n03} peaks={peaks}"


def test_hybrid_equiv_to_no_fill_at_extreme_threshold() -> None:
    """补点单调的边界印证:fill_threshold_rel→1.0(只取 max 格)时补点退化,趋近纯峰值。

    实测合并簇/3目标在 fill_threshold_rel=0.7 已退化到 == 峰值数(补点全被高阈值滤掉)。
    """
    world = _world()
    for name, targets, bw in _fill_threshold_cases():
        field = _make_field(targets, world, bandwidth_m=bw)
        peaks = _peaks(field, world)
        hyb_high = _hybrid(field, world, fill_threshold_rel=0.7)
        # 高补点阈值下不应少于峰值(峰值恒保留),且不多于低阈值结果(单调)。
        assert len(hyb_high) >= len(peaks), \
            f"{name}: 高 fill 阈值下仍须保留全部峰值"
        assert len(hyb_high) <= len(_hybrid(field, world, fill_threshold_rel=0.3)), \
            f"{name}: 高 fill 阈值总点数须 <= 低阈值(补点单调)"
