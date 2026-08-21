"""[A1 验收测试 · #6 A3 物理] CameraSensorModel(p_d/p_fa/footprint/corridor_width)
+ Follower 物理封装 + Leader 能耗不进 reward 的结构性约束。

判据来源:contracts/physics.py(CameraSensorModel.corridor_width_m = lateral×1.0;
§v1.0.7:走廊宽 = footprint_lateral,与 grid cell 严格对齐;
FollowerProtocol;LeaderActorProtocol.energy_J_cumulative 注释"不进 reward")、
contracts/mdp.py(reward 公式只含 per_follower_energy)、harness §4.2。
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from msim.contracts.config import SimConfig
from msim.contracts.physics import (
    CameraSensorModel,
    FollowerProtocol,
    LeaderActorProtocol,
)
from msim.physics.follower import Follower


# ======================================================================
# CameraSensorModel —— 走廊宽 = footprint 横向 × 1.0(§v1.0.7,契约 property,已冻结)
# ======================================================================
def test_corridor_width_is_lateral_times_1p0() -> None:
    cam = CameraSensorModel(footprint_lateral_m=6.0)
    assert cam.corridor_width_m == pytest.approx(6.0)


def test_corridor_width_scales_with_lateral() -> None:
    for lat in [4.0, 6.0, 10.0, 12.5]:
        cam = CameraSensorModel(footprint_lateral_m=lat)
        assert cam.corridor_width_m == pytest.approx(lat), \
            f"corridor_width 须严格 = {lat}×1.0"


def test_camera_default_detection_params() -> None:
    cam = CameraSensorModel()
    # "接近完美但不完美":高检出、低虚警(契约默认)
    assert 0.0 < cam.p_fa < cam.p_d <= 1.0
    assert cam.p_d == pytest.approx(0.95)
    assert cam.p_fa == pytest.approx(0.05)


# ======================================================================
# Follower —— 物理封装暴露 sensor / pose / energy_norm,可 reset
# ======================================================================
def test_follower_has_camera_sensor() -> None:
    f = Follower(SimConfig())
    assert isinstance(f.sensor, CameraSensorModel)
    # sensor 参数从 SimConfig.sensor 注入
    cfg = SimConfig()
    assert f.sensor.footprint_lateral_m == pytest.approx(cfg.sensor.footprint_lateral_m)
    assert f.sensor.p_d == pytest.approx(cfg.sensor.p_d)


def test_follower_reset_and_pose() -> None:
    f = Follower(SimConfig())
    start = np.array([3.0, 4.0, 0.0], dtype=np.float64)
    f.reset(start)
    np.testing.assert_allclose(np.asarray(f.pose, dtype=float)[:2], start[:2], atol=1e-9)


def test_follower_energy_norm_in_unit_interval() -> None:
    f = Follower(SimConfig())
    f.reset(np.array([0.0, 0.0, 0.0]))
    e = f.energy_norm
    assert 0.0 <= e <= 1.0, "energy_norm 须 ∈ [0,1]"
    assert e == pytest.approx(1.0), "初始(满电)energy_norm 应为 1.0"


# ======================================================================
# Leader 能耗不进 reward —— 结构性约束
# ======================================================================
def test_leader_protocol_exposes_cumulative_energy_bypass() -> None:
    """LeaderActorProtocol 暴露 energy_J_cumulative(旁路统计);
    而 reward 公式(contracts/mdp.py step docstring)只含 per_follower_energy → Leader 不进 reward。
    """
    # Protocol 必须声明 energy_J_cumulative(旁路)
    assert hasattr(LeaderActorProtocol, "energy_J_cumulative")
    # FollowerProtocol 才是 reward 能耗来源(暴露 energy_norm)
    assert hasattr(FollowerProtocol, "energy_norm")


def test_reward_formula_excludes_leader_energy() -> None:
    """从 contracts/mdp.py 的 step docstring 锁定:reward 不含 leader 能耗项。"""
    from msim.contracts.mdp import CoopSurveyEnvProtocol
    doc = CoopSurveyEnvProtocol.step.__doc__ or ""
    assert "per_follower_energy" in doc, "reward 能耗项应基于 per_follower_energy"
    # 不出现 leader 能耗进 reward 的措辞(防回归:Leader 能耗误入 reward)
    assert "leader_energy" not in doc.lower().replace(" ", "_")
