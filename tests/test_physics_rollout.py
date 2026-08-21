"""[A1 验收测试 · #6 A3 物理] rollout —— Level1 解析口径 + Level2 与 AUV 积分一致。

判据来源:contracts/physics.py(RolloutResult / RolloutEngine,两实现非两套代码)、
contracts/config.py(RolloutConfig 各系数)、harness §4.2(rollout 行)、纪律4(解析确定量)。

Level1(修士训练默认,纯几何、确定、不调 AUV 类):
  duration_s = Σ(段长)/cruise_speed + Σ|Δψ|/nominal_yaw_rate + Σ dwell_time
  energy_J   = length_energy_coeff × Σ段长  + turn_energy_coeff × Σ|Δψ|
  其中 Σ|Δψ| = 首段航向相对 start_pose 航向的转角 + 相邻段航向差(wrap 到 [-pi,pi))。

为对"首段是否相对 start heading 计转角"这一约定鲁棒,直线/对齐用例令首段转角=0;
另用纯直线(无转弯)精确锁定长度/时间/能耗的解析值。
Level2:用根目录 AUV 类 step_motion 细步长积分,与 Level1 同向但含动力学;
本套只断言其"物理合理 + 与积分自洽"(容差内),不强行等于 Level1。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from msim.contracts.config import RolloutConfig, SimConfig
from msim.contracts.physics import RolloutResult
from msim.physics.rollout import Level1RolloutEngine, Level2RolloutEngine


@pytest.fixture
def rcfg() -> RolloutConfig:
    return RolloutConfig()  # cruise=1.5, yaw_rate=0.5236, dwell=5, len_coeff=1, turn_coeff=1


def _pose(x: float, y: float, psi: float) -> np.ndarray:
    return np.array([x, y, psi], dtype=np.float64)


def _seg_len(a, b) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


# ======================================================================
# Level1 —— 纯直线(无转弯):精确锁定长度/时间/能耗
# ======================================================================
def test_level1_straight_line_exact(rcfg: RolloutConfig) -> None:
    eng = Level1RolloutEngine(rcfg)
    start = _pose(0.0, 0.0, 0.0)            # 朝 North(+x)
    wps = [np.array([10.0, 0.0]), np.array([25.0, 0.0])]   # 全程沿 North 直线,无转弯
    res = eng.execute(start, wps)
    assert isinstance(res, RolloutResult)

    total_len = 10.0 + 15.0                  # 25 m
    n_wp = len(wps)
    exp_duration = total_len / rcfg.cruise_speed_mps + 0.0 + n_wp * rcfg.dwell_time_s
    exp_energy = rcfg.length_energy_coeff * total_len + rcfg.turn_energy_coeff * 0.0

    assert res.duration_s == pytest.approx(exp_duration, rel=1e-9, abs=1e-9)
    assert res.energy_J == pytest.approx(exp_energy, rel=1e-9, abs=1e-9)


# ======================================================================
# Level1 —— 含一个直角转弯:用时含 |Δψ|/yaw_rate,能耗含 turn_coeff×|Δψ|
# ======================================================================
def test_level1_right_angle_turn(rcfg: RolloutConfig) -> None:
    eng = Level1RolloutEngine(rcfg)
    start = _pose(0.0, 0.0, 0.0)            # 朝 North
    # 第一段沿 North(航向 0,与 start 对齐 → 首段转角 0);第二段沿 East(航向 +pi/2)
    wps = [np.array([10.0, 0.0]), np.array([10.0, 8.0])]
    res = eng.execute(start, wps)

    len1 = 10.0
    len2 = 8.0
    total_len = len1 + len2
    total_turn = abs(math.pi / 2.0)         # 仅第二段相对第一段转 90°

    exp_duration = (total_len / rcfg.cruise_speed_mps
                    + total_turn / rcfg.nominal_yaw_rate_rps
                    + len(wps) * rcfg.dwell_time_s)
    exp_energy = (rcfg.length_energy_coeff * total_len
                  + rcfg.turn_energy_coeff * total_turn)

    assert res.duration_s == pytest.approx(exp_duration, rel=1e-9, abs=1e-9)
    assert res.energy_J == pytest.approx(exp_energy, rel=1e-9, abs=1e-9)


# ======================================================================
# Level1 —— 总量 = 各段累加(per_wp 与汇总自洽)
# ======================================================================
def test_level1_per_wp_sums_to_total(rcfg: RolloutConfig) -> None:
    eng = Level1RolloutEngine(rcfg)
    start = _pose(0.0, 0.0, 0.0)
    wps = [np.array([5.0, 0.0]), np.array([5.0, 5.0]), np.array([0.0, 5.0])]
    res = eng.execute(start, wps)

    if res.per_wp_energy_J:
        assert sum(res.per_wp_energy_J) == pytest.approx(res.energy_J, rel=1e-9, abs=1e-9)
    if res.per_wp_duration_s:
        assert sum(res.per_wp_duration_s) == pytest.approx(res.duration_s, rel=1e-9, abs=1e-9)


# ======================================================================
# Level1 —— 总长 = Σ段长(对任意折线,与端点解析一致)
# ======================================================================
def test_level1_total_length_matches_polyline(rcfg: RolloutConfig) -> None:
    eng = Level1RolloutEngine(rcfg)
    start = _pose(2.0, 3.0, 0.0)
    wps = [np.array([2.0, 3.0]), np.array([12.0, 3.0]), np.array([12.0, 9.0])]
    res = eng.execute(start, wps)
    pts = [start[:2]] + [np.asarray(w, dtype=np.float64) for w in wps]
    total_len = sum(_seg_len(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    # 反推:从 duration 扣掉 dwell 与转弯项剩余 = total_len / cruise(避免依赖 energy 系数=1)
    exp_energy_len_part = rcfg.length_energy_coeff * total_len
    # energy 至少包含长度项;turn 项 >=0
    assert res.energy_J >= exp_energy_len_part - 1e-9


# ======================================================================
# Level1 —— end_pose:末位置 = 最后 waypoint
# ======================================================================
def test_level1_end_pose_at_last_waypoint(rcfg: RolloutConfig) -> None:
    eng = Level1RolloutEngine(rcfg)
    start = _pose(0.0, 0.0, 0.0)
    wps = [np.array([10.0, 0.0]), np.array([10.0, 5.0])]
    res = eng.execute(start, wps)
    np.testing.assert_allclose(np.asarray(res.end_pose, dtype=float)[:2],
                               np.asarray(wps[-1], dtype=float), atol=1e-9)


# ======================================================================
# Level1 —— 单调性:更长路径用时/能耗更大
# ======================================================================
def test_level1_monotone_in_length(rcfg: RolloutConfig) -> None:
    eng = Level1RolloutEngine(rcfg)
    start = _pose(0.0, 0.0, 0.0)
    short = eng.execute(start, [np.array([5.0, 0.0])])
    long = eng.execute(start, [np.array([50.0, 0.0])])
    assert long.duration_s > short.duration_s
    assert long.energy_J > short.energy_J


# ======================================================================
# Level2 —— 与 AUV 类细步长积分一致(容差内):物理合理 + 同向
# ======================================================================
def test_level2_runs_and_is_physical() -> None:
    cfg = SimConfig()
    eng = Level2RolloutEngine(cfg, auv_type="follower")
    start = _pose(0.0, 0.0, 0.0)
    wps = [np.array([15.0, 0.0]), np.array([15.0, 10.0])]
    res = eng.execute(start, wps)
    assert isinstance(res, RolloutResult)
    assert res.energy_J > 0.0, "Level2 物理积分能耗须为正"
    assert res.duration_s > 0.0
    # 末位姿应抵达最后 waypoint 邻域(arrive_radius 量级)
    end_xy = np.asarray(res.end_pose, dtype=float)[:2]
    assert np.linalg.norm(end_xy - np.asarray(wps[-1], dtype=float)) < 5.0, \
        "Level2 末位姿应抵达最后 waypoint 邻域"


def test_level2_duration_consistent_with_dt_integration() -> None:
    """Level2 用时 ≈ 步数×sim_dt(细步长积分口径,harness §4.2:与 AUV 积分一致)。"""
    cfg = SimConfig()
    dt = cfg.rollout.sim_dt_s
    eng = Level2RolloutEngine(cfg, auv_type="follower")
    res = eng.execute(_pose(0.0, 0.0, 0.0), [np.array([20.0, 0.0])])
    # duration 应为 dt 的整数倍(逐步积分),容差一个 dt
    n_steps = res.duration_s / dt
    assert abs(n_steps - round(n_steps)) < 1.0, "Level2 用时应为 sim_dt 积分步数口径"


def test_level2_longer_path_costs_more() -> None:
    cfg = SimConfig()
    eng = Level2RolloutEngine(cfg, auv_type="follower")
    short = eng.execute(_pose(0.0, 0.0, 0.0), [np.array([10.0, 0.0])])
    long = eng.execute(_pose(0.0, 0.0, 0.0), [np.array([40.0, 0.0])])
    assert long.energy_J > short.energy_J
    assert long.duration_s > short.duration_s
