"""[A1 验收测试 · 波次1.x §v1.0.5/§v1.0.6/§v1.0.10] A′ 候选提取参数 config 旋钮 —— 守住单一真相源。

判据来源:
  - contracts/config.py:ObsCandidateConfig(threshold_rel, fill_threshold_rel,
    peak_min_separation_m, fill_radius_m)、SimConfig.obs_candidate、
    resolve_obs_candidate_params(cfg) -> (threshold_rel, fill_threshold_rel, peak_sep, fill_r)
    的签名 + docstring(None 半径回退:peak_sep=0.5×footprint_lateral、fill=0.9×footprint_lateral)。
  - §v1.0.6:A0 把"峰值阈值"(threshold_rel)与"补点阈值"(fill_threshold_rel,更严)拆开,
    resolve_obs_candidate_params 由 3 元组改为 **4 元组**;fill_threshold_rel **不随 footprint 变**。
  - §v1.0.10:默认阈值调严:threshold_rel=0.5(峰值锚点)、fill_threshold_rel=0.7(补点),全局生效。
  - eval/runner.py:default_detections(cfg, ...) 经 resolve_obs_candidate_params 取默认,
    R 默认 = cfg.obs_candidate.fill_radius_m(None → 0.9×footprint_lateral=5.4m),并跟随该旋钮。

本套对契约/公开行为编程(不读实现内部),固定 seed/受控场,无 RNG 偶发:
    1 解析默认       resolve_obs_candidate_params(SimConfig()) == (0.5, 0.7, 3.0, 5.4)
    2 解析覆盖       ObsCandidateConfig(0.4,0.6,2.0,2.7) → (0.4, 0.6, 2.0, 2.7)
    3 None 半径跟随  换 footprint_lateral_m → peak_sep/fill=0.5×/0.9×新值;thr/fill_thr 不随 footprint 变
    4 旋钮影响检测   固定 seed 受控场:R 调小(5.4→2.7)→ default_detections 候选点数变多,
                     且两版均在场内、非空、确定性(同 cfg 两次逐位一致)。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from msim.contracts.config import (
    ObsCandidateConfig,
    SensorConfig,  # noqa: F401  (锚点导入:确认 config 公开面稳定)
    SimConfig,
    WorldConfig,
    resolve_obs_candidate_params,
)
from msim.eval.runner import default_detections


# 默认 SimConfig 的物理常数(锚定断言):footprint_lateral=6 → peak_sep=3.0、fill=5.4。
_DEFAULT_FOOTPRINT_LATERAL = 6.0
_EXPECTED_DEFAULT_PEAK_SEP = 0.5 * _DEFAULT_FOOTPRINT_LATERAL   # 3.0
_EXPECTED_DEFAULT_FILL = 0.9 * _DEFAULT_FOOTPRINT_LATERAL       # 5.4
# §v1.0.6:峰值/补点两个阈值(均与 footprint 无关的纯比例)。
# §v1.0.10:默认值调严:峰值阈值 0.5,补点阈值 0.7。
_EXPECTED_DEFAULT_THRESHOLD = 0.5
_EXPECTED_DEFAULT_FILL_THRESHOLD = 0.7


def _controlled_cfg() -> SimConfig:
    """受控小场 + 锁定 seed:WorldConfig(60,30,0.5),其余默认(footprint_lateral=6→默认 R=5.4)。"""
    return replace(SimConfig(seed=0), world=WorldConfig(x_max_m=60.0, y_max_m=30.0, res_m=0.5))


# ======================================================================
# 必测 1:解析默认 —— resolve_obs_candidate_params(SimConfig()) == (0.5, 0.7, 3.0, 5.4)
# §v1.0.6:4 元组(peak 阈值、fill 阈值、peak_sep 3.0、fill_r 5.4)。
# §v1.0.10:默认峰值阈值 0.5、补点阈值 0.7。
# ======================================================================
def test_resolve_default_params() -> None:
    cfg = SimConfig()
    # 前提复核:默认 footprint_lateral=6,本测试的 (3.0,5.4) 才成立。
    assert cfg.sensor.footprint_lateral_m == _DEFAULT_FOOTPRINT_LATERAL, \
        f"默认 footprint_lateral_m 应为 {_DEFAULT_FOOTPRINT_LATERAL},得 {cfg.sensor.footprint_lateral_m}"
    resolved = resolve_obs_candidate_params(cfg)
    # §v1.0.6:必须是 4 元组(防退回 3 元组旧契约)。
    assert len(resolved) == 4, f"resolve_obs_candidate_params 应返回 4 元组,得长度 {len(resolved)}"
    thr, fill_thr, peak_sep, fill_r = resolved
    assert thr == _EXPECTED_DEFAULT_THRESHOLD, f"默认 threshold_rel 应为 0.5(§v1.0.10),得 {thr}"
    assert fill_thr == _EXPECTED_DEFAULT_FILL_THRESHOLD, \
        f"默认 fill_threshold_rel 应为 0.7(§v1.0.10 补点阈值),得 {fill_thr}"
    assert peak_sep == _EXPECTED_DEFAULT_PEAK_SEP, \
        f"默认 peak_sep 应为 0.5×footprint=3.0,得 {peak_sep}"
    assert fill_r == _EXPECTED_DEFAULT_FILL, \
        f"默认 fill_radius 应为 0.9×footprint=5.4,得 {fill_r}"
    # 整体元组逐项相等(防字段顺序/类型漂移)。
    assert resolved == (0.5, 0.7, 3.0, 5.4)
    # 约定:补点阈值 >= 峰值阈值(fill 更严)。
    assert fill_thr >= thr, "fill_threshold_rel 约定 >= threshold_rel(补点更严)"


# ======================================================================
# 必测 2:解析覆盖 —— 四参全显式覆盖时原样透传,不触发任何 footprint 回退
# ======================================================================
def test_resolve_override_all_params() -> None:
    cfg = replace(
        SimConfig(),
        obs_candidate=ObsCandidateConfig(
            threshold_rel=0.4,
            fill_threshold_rel=0.6,
            peak_min_separation_m=2.0,
            fill_radius_m=2.7,
        ),
    )
    assert resolve_obs_candidate_params(cfg) == (0.4, 0.6, 2.0, 2.7), \
        "四参全显式覆盖时应原样返回 (0.4, 0.6, 2.0, 2.7),不回退 footprint 默认"


# ======================================================================
# 必测 3:None 半径跟随 footprint —— 换 footprint_lateral_m,peak_sep/fill = 0.5×/0.9× 新值;
# 而 threshold_rel / fill_threshold_rel 是纯比例,**不随 footprint 变**。
# ======================================================================
def test_none_radii_follow_footprint() -> None:
    # ObsCandidateConfig 用默认(两半径均 None)→ 必须回退到挂 footprint 的物理默认。
    cfg = SimConfig()
    assert cfg.obs_candidate.peak_min_separation_m is None
    assert cfg.obs_candidate.fill_radius_m is None

    for new_fl in (4.0, 8.0, 10.0):
        new_sensor = replace(cfg.sensor, footprint_lateral_m=new_fl)
        cfg_fl = replace(cfg, sensor=new_sensor)
        thr, fill_thr, peak_sep, fill_r = resolve_obs_candidate_params(cfg_fl)
        # 两个阈值是纯比例,不随 footprint 变。
        assert thr == _EXPECTED_DEFAULT_THRESHOLD, f"threshold_rel 不应随 footprint 变:得 {thr}"
        assert fill_thr == _EXPECTED_DEFAULT_FILL_THRESHOLD, \
            f"fill_threshold_rel 不应随 footprint 变:得 {fill_thr}"
        # 两个半径随 footprint 回退。
        assert peak_sep == 0.5 * new_fl, \
            f"None peak_sep 应随新 footprint={new_fl} → 0.5×={0.5*new_fl},得 {peak_sep}"
        assert fill_r == 0.9 * new_fl, \
            f"None fill_radius 应随新 footprint={new_fl} → 0.9×={0.9*new_fl},得 {fill_r}"


def test_none_radii_independent_of_explicit_threshold() -> None:
    """仅显式给 threshold_rel(其余默认)时,半径仍随 footprint 回退,两个阈值透传/默认。
    §v1.0.10:fill_threshold_rel 默认值为 0.7;threshold_rel 此处显式传入 0.5 透传。
    """
    cfg = SimConfig()
    cfg2 = replace(
        cfg,
        sensor=replace(cfg.sensor, footprint_lateral_m=8.0),
        obs_candidate=ObsCandidateConfig(threshold_rel=0.5),  # fill_threshold_rel 默认 0.7(§v1.0.10)、两半径默认 None
    )
    # threshold 透传 0.5;fill_threshold_rel 取默认 0.7(§v1.0.10);None 半径随 footprint=8 → 4.0/7.2。
    assert resolve_obs_candidate_params(cfg2) == (0.5, 0.7, 4.0, 7.2), \
        "threshold 透传 + fill_threshold 默认 0.7(§v1.0.10) + None 半径随 footprint=8 → (0.5, 0.7, 4.0, 7.2)"


def test_explicit_fill_threshold_passthrough() -> None:
    """§v1.0.6:仅显式给 fill_threshold_rel,其余默认 → 仅该项变,半径仍随 footprint 回退。
    §v1.0.10:threshold_rel 默认值为 0.5;此处显式传入 fill_threshold_rel=0.7(与新默认相同,透传验证)。
    """
    cfg = replace(SimConfig(), obs_candidate=ObsCandidateConfig(fill_threshold_rel=0.7))
    # footprint=6(默认)→ peak_sep=3.0、fill=5.4;threshold_rel 默认 0.5(§v1.0.10);fill_threshold 显式 0.7。
    assert resolve_obs_candidate_params(cfg) == (0.5, 0.7, 3.0, 5.4), \
        "仅 fill_threshold_rel=0.7 时应得 (0.5, 0.7, 3.0, 5.4)(§v1.0.10:threshold_rel 默认 0.5)"


# ======================================================================
# 必测 4:旋钮影响 default_detections —— R 调小 → 候选更密(点数变多);在场内、非空、确定性
# ======================================================================
def _all_within_bounds(pts, world: WorldConfig) -> bool:
    for p in pts:
        a = np.asarray(p, dtype=np.float64)
        if a.shape != (2,):
            return False
        if not (0.0 <= float(a[0]) <= world.x_max_m):
            return False
        if not (0.0 <= float(a[1]) <= world.y_max_m):
            return False
    return True


def _bitwise_equal(p_list, q_list) -> bool:
    if len(p_list) != len(q_list):
        return False
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(p_list, q_list))


def test_fill_radius_knob_increases_candidate_count() -> None:
    cfg_default = _controlled_cfg()  # 默认旋钮 → R=5.4
    # 仅改 R(其余字段保持默认:fill_threshold_rel=0.5、threshold_rel=0.3)。
    cfg_small_r = replace(
        cfg_default, obs_candidate=ObsCandidateConfig(fill_radius_m=2.7)
    )

    # 前提复核:旋钮确实把解析出的 R 改了(5.4 → 2.7),其余维持默认(4 元组的第 4 项 = fill_r)。
    assert resolve_obs_candidate_params(cfg_default)[3] == 5.4
    assert resolve_obs_candidate_params(cfg_small_r)[3] == 2.7

    d_default = default_detections(cfg_default)
    d_small_r = default_detections(cfg_small_r)

    # 非空(本受控场对两个 R 都该产生候选,断言才有判别力)。
    assert len(d_default) > 0, "默认 R 受控场应有候选点"
    assert len(d_small_r) > 0, "小 R 受控场应有候选点"

    # 核心:R 调小 → 覆盖补点更密 → 候选点数严格变多。
    assert len(d_small_r) > len(d_default), \
        f"R 调小(5.4→2.7)候选点数应变多:R=2.7 得 {len(d_small_r)} <= R=5.4 得 {len(d_default)}"

    # 两版候选都严格在场内。
    world = cfg_default.world
    assert _all_within_bounds(d_default, world), "默认 R 候选点须在 [0,x_max]×[0,y_max] 内"
    assert _all_within_bounds(d_small_r, world), "小 R 候选点须在 [0,x_max]×[0,y_max] 内"


def test_default_detections_deterministic_per_cfg() -> None:
    """确定性:同一 cfg(含旋钮)两次 default_detections 逐位一致(固定 seed、无 RNG 漂移)。"""
    cfg_default = _controlled_cfg()
    cfg_small_r = replace(
        cfg_default, obs_candidate=ObsCandidateConfig(fill_radius_m=2.7)
    )
    for cfg in (cfg_default, cfg_small_r):
        a = default_detections(cfg)
        b = default_detections(cfg)
        assert _bitwise_equal(a, b), \
            f"同 cfg 两次 default_detections 须逐位相同(确定性):R={resolve_obs_candidate_params(cfg)[3]}"


# ======================================================================
# §v1.0.6 追补:default_detections 跟随 cfg.obs_candidate.fill_threshold_rel(补点单调)
# 提高补点阈值(0.3→0.5→0.7)→ 补点区收缩 → 总候选点数单调不增(峰值前缀恒定)。
# ======================================================================
def test_default_detections_follows_fill_threshold_knob_monotone() -> None:
    cfg = _controlled_cfg()
    cfg_03 = replace(cfg, obs_candidate=ObsCandidateConfig(fill_threshold_rel=0.3))
    cfg_05 = replace(cfg, obs_candidate=ObsCandidateConfig(fill_threshold_rel=0.5))
    cfg_07 = replace(cfg, obs_candidate=ObsCandidateConfig(fill_threshold_rel=0.7))

    # 前提复核:旋钮确实改了解析出的 fill_threshold_rel(4 元组第 2 项)。
    assert resolve_obs_candidate_params(cfg_03)[1] == 0.3
    assert resolve_obs_candidate_params(cfg_05)[1] == 0.5
    assert resolve_obs_candidate_params(cfg_07)[1] == 0.7

    n03 = len(default_detections(cfg_03))
    n05 = len(default_detections(cfg_05))
    n07 = len(default_detections(cfg_07))

    # 提高补点阈值 → 总点数单调不增(补点只减不增,峰值恒定)。
    assert n03 >= n05 >= n07, \
        f"fill_threshold_rel 升高 → 候选点数应单调不增:n03={n03} n05={n05} n07={n07}"
    # 各版本均在场内、非空(峰值至少存在 → 非空,判据有判别力)。
    assert n07 > 0, "最严补点阈值下峰值锚点仍应使候选非空"
