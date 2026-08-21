"""[A1 验收测试 · 波次1.x §v1.0.4] detections_from_field —— 概率图→离散候选检测点(估计,非 GT)。

判据来源:contracts/env_static.py 的 detections_from_field / field_from_targets 签名 + docstring、
agent_team_harness §4.2、本波次补丁单(必测不变量 1–7)。

被测原语(对契约编程,只测暴露行为,不依赖实现内部):
    field_from_targets(targets, world, *, bandwidth_m)              -> Field        (造受控场)
    detections_from_field(field, world, *, threshold_rel=0.3,
                          min_separation_m)                          -> List[Waypoint] (有损逆过程)

受控场口径(本波次补丁单指定):WorldConfig(60,30,0.5)、bandwidth_m=3.0、min_separation_m=3.0。
检测点是【估计/检测(estimate, NOT ground truth)】:可合并近邻、漏阈下弱峰 —— 本套用受控场
对"确定性/边界/召回/精度/单调/有损诚实/空场"做强断言,作为该原语"完成"的客观判据。
"""
from __future__ import annotations

import numpy as np

from msim.contracts.config import WorldConfig
from msim.env_static.field import detections_from_field, field_from_targets


# ----------------------------------------------------------------------
# 受控场固定口径(本波次补丁单指定)
# ----------------------------------------------------------------------
BANDWIDTH_M = 3.0
MIN_SEP_M = 3.0
THRESHOLD_REL = 0.3


def _world() -> WorldConfig:
    # 小场降体量(nx=120, ny=60),仍走真实公式。
    return WorldConfig(x_max_m=60.0, y_max_m=30.0, res_m=0.5)


def _well_separated_targets() -> list:
    """两两间距 > 2*min_separation_m 的良分离目标(补丁单指定 (10,8),(30,22),(50,10))。"""
    return [
        np.array([10.0, 8.0], dtype=np.float64),
        np.array([30.0, 22.0], dtype=np.float64),
        np.array([50.0, 10.0], dtype=np.float64),
    ]


def _make_field(targets: list, world: WorldConfig):
    return field_from_targets(targets, world, bandwidth_m=BANDWIDTH_M)


def _cell_index(p: np.ndarray, res: float) -> tuple:
    """检测点世界坐标 → 其所在 cell 的 (ix,iy)(与 cell 中心 (ix+0.5)*res 互逆)。"""
    ix = int(round(float(p[0]) / res - 0.5))
    iy = int(round(float(p[1]) / res - 0.5))
    return ix, iy


# ======================================================================
# 不变量 1:确定性 —— 同 field+同参数两次调用逐位相同
# ======================================================================
def test_detections_deterministic() -> None:
    world = _world()
    field = _make_field(_well_separated_targets(), world)
    d1 = detections_from_field(field, world, threshold_rel=THRESHOLD_REL, min_separation_m=MIN_SEP_M)
    d2 = detections_from_field(field, world, threshold_rel=THRESHOLD_REL, min_separation_m=MIN_SEP_M)
    assert len(d1) == len(d2), "两次调用检测点数须相同(确定性、无 RNG)"
    for a, b in zip(d1, d2):
        assert np.array_equal(a, b), "两次调用须逐位相同(确定性、无 RNG)"


# ======================================================================
# 不变量 2:边界 —— 所有检测点严格在 [0,x_max]×[0,y_max] 内
# ======================================================================
def test_detections_within_bounds() -> None:
    world = _world()
    field = _make_field(_well_separated_targets(), world)
    dets = detections_from_field(field, world, threshold_rel=THRESHOLD_REL, min_separation_m=MIN_SEP_M)
    assert len(dets) > 0, "良分离目标场应抽到检测点"
    for p in dets:
        assert p.shape == (2,), "每个检测点 shape 须为 (2,)"
        assert 0.0 <= float(p[0]) <= world.x_max_m, f"检测点 North 越界:{p}"
        assert 0.0 <= float(p[1]) <= world.y_max_m, f"检测点 East 越界:{p}"


# ======================================================================
# 不变量 3:召回(受控强断言) —— 良分离目标每个都被 min_separation_m 内命中,recall==1.0
# ======================================================================
def test_detections_recall_on_well_separated_targets() -> None:
    world = _world()
    targets = _well_separated_targets()
    field = _make_field(targets, world)
    dets = detections_from_field(field, world, threshold_rel=THRESHOLD_REL, min_separation_m=MIN_SEP_M)
    assert len(dets) >= len(targets), "良分离目标至少应被各自抽出一个峰"
    det_arr = np.asarray([np.asarray(p, dtype=np.float64) for p in dets])
    hits = 0
    for t in targets:
        dist = np.hypot(det_arr[:, 0] - t[0], det_arr[:, 1] - t[1])
        assert float(dist.min()) <= MIN_SEP_M, \
            f"目标 {t} 的 min_separation_m 内无检测点(漏检),最近 {float(dist.min()):.3f}"
        hits += 1
    recall = hits / len(targets)
    assert recall == 1.0, f"良分离目标召回须为 1.0,得 {recall}"


# ======================================================================
# 不变量 4:精度 —— 每个检测点所在 cell 的 field >= threshold_rel*field.max()
# ======================================================================
def test_detections_above_relative_threshold() -> None:
    world = _world()
    res = world.res_m
    field = _make_field(_well_separated_targets(), world)
    fmax = float(field.max())
    thr = THRESHOLD_REL * fmax
    dets = detections_from_field(field, world, threshold_rel=THRESHOLD_REL, min_separation_m=MIN_SEP_M)
    assert len(dets) > 0
    for p in dets:
        ix, iy = _cell_index(p, res)
        assert 0 <= ix < field.shape[0] and 0 <= iy < field.shape[1], f"cell 索引越界:{p}"
        assert float(field[ix, iy]) >= thr - 1e-9, \
            f"检测点所在 cell 低于相对阈值:field={field[ix, iy]:.4f} < thr={thr:.4f}"


# ======================================================================
# 不变量 5①:有损/诚实 —— 单孤立目标 → 恰 1 点且距该目标 <= res_m
# ======================================================================
def test_detections_single_isolated_target_exactly_one() -> None:
    world = _world()
    res = world.res_m
    t = np.array([20.0, 15.0], dtype=np.float64)
    field = _make_field([t], world)
    dets = detections_from_field(field, world, threshold_rel=THRESHOLD_REL, min_separation_m=MIN_SEP_M)
    assert len(dets) == 1, f"单孤立目标应恰抽 1 点,得 {len(dets)}"
    dist = float(np.hypot(dets[0][0] - t[0], dets[0][1] - t[1]))
    assert dist <= res + 1e-9, f"单孤立目标检测点应落在该目标 res_m 内,距 {dist:.4f} > res {res}"


# ======================================================================
# 不变量 5②:有损/诚实 —— 两目标间距 < min_separation_m(相距 1m)→ 合并为 1 点(证明是估计、会失真)
# ======================================================================
def test_detections_close_pair_merges_to_one() -> None:
    world = _world()
    a = np.array([20.0, 15.0], dtype=np.float64)
    b = np.array([21.0, 15.0], dtype=np.float64)  # 相距 1m < min_separation_m=3m
    field = _make_field([a, b], world)
    dets = detections_from_field(field, world, threshold_rel=THRESHOLD_REL, min_separation_m=MIN_SEP_M)
    assert len(dets) == 1, \
        f"间距 < min_separation_m 的两目标应被合并为 1 点(有损估计,非精确等于 GT),得 {len(dets)}"


# ======================================================================
# 不变量 6:单调 —— threshold_rel 升高 → 检测点数不增(并构造一例严格下降)
# ======================================================================
def test_detections_monotone_in_threshold() -> None:
    world = _world()
    # 含强弱不一的多簇叠加场:升阈应逐步滤掉较弱峰。
    targets = [
        np.array([8.0, 6.0], dtype=np.float64),
        np.array([10.0, 8.0], dtype=np.float64),
        np.array([12.0, 6.0], dtype=np.float64),
        np.array([30.0, 22.0], dtype=np.float64),
        np.array([50.0, 10.0], dtype=np.float64),
        np.array([52.0, 12.0], dtype=np.float64),
        np.array([25.0, 5.0], dtype=np.float64),
    ]
    field = _make_field(targets, world)
    thresholds = [0.2, 0.3, 0.4, 0.6, 0.8]
    counts = [
        len(detections_from_field(field, world, threshold_rel=thr, min_separation_m=MIN_SEP_M))
        for thr in thresholds
    ]
    for lo, hi, c_lo, c_hi in zip(thresholds[:-1], thresholds[1:], counts[:-1], counts[1:]):
        assert c_hi <= c_lo, \
            f"升阈检测点数不应增:thr {lo}->{hi} 计数 {c_lo}->{c_hi}"
    # 直接命中补丁单口径:0.2 -> 0.6 不增。
    n02 = len(detections_from_field(field, world, threshold_rel=0.2, min_separation_m=MIN_SEP_M))
    n06 = len(detections_from_field(field, world, threshold_rel=0.6, min_separation_m=MIN_SEP_M))
    assert n06 <= n02, f"threshold_rel 0.2->0.6 检测点数应不增,得 {n02}->{n06}"
    # 该受控场应能体现严格下降(确认单调测试有判别力,非恒等通过)。
    assert counts[-1] < counts[0], "受控场应使升阈检测点数严格下降(确保单调断言有判别力)"


# ======================================================================
# 不变量 7:空场 —— 全零场 与 field_from_targets([], ...) 均返回 []
# ======================================================================
def test_detections_empty_on_zero_field() -> None:
    world = _world()
    zero_field = np.zeros((world.nx, world.ny), dtype=np.float64)
    assert detections_from_field(zero_field, world, threshold_rel=THRESHOLD_REL, min_separation_m=MIN_SEP_M) == [], \
        "全零场应返回 []"


def test_detections_empty_on_empty_targets_field() -> None:
    world = _world()
    field = field_from_targets([], world, bandwidth_m=BANDWIDTH_M)  # 空目标 → 全零场
    assert detections_from_field(field, world, threshold_rel=THRESHOLD_REL, min_separation_m=MIN_SEP_M) == [], \
        "field_from_targets([]) 派生的全零场应返回 []"
