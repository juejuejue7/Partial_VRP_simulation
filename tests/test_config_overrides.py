"""[A1 验收测试 · #5 A2 地基] config.apply_overrides —— 字段覆盖正确。

判据来源:contracts/config.py + config/config.py(apply_overrides docstring:
泛型 dataclass 字段覆盖,dataclasses.replace,不 import AUVConfig → 无反向依赖)。

异构差异的唯一注入路径:SimConfig.{leader,follower}_overrides(纯 dict)合并进
一个 AUVConfig 实例,返回新配置。验收:被覆盖字段生效、未覆盖字段保持默认、原对象不被改写。
"""
from __future__ import annotations

import dataclasses

import pytest

from msim.config.config import apply_overrides
from msim.contracts.config import SimConfig
from msim.physics.auv_dynamics import AUVConfig


def test_apply_overrides_sets_named_fields() -> None:
    base = AUVConfig()
    leader_ov = SimConfig().leader_overrides
    out = apply_overrides(base, leader_ov)
    for k, v in leader_ov.items():
        assert getattr(out, k) == pytest.approx(v), f"override 字段 {k} 未生效"


def test_apply_overrides_keeps_other_fields_default() -> None:
    base = AUVConfig()
    out = apply_overrides(base, {"cruise_speed": 2.0})
    # 仅 cruise_speed 改变,其它通用物理常数保持
    assert out.cruise_speed == pytest.approx(2.0)
    assert out.m_u == base.m_u
    assert out.I_z == base.I_z
    assert out.battery_capacity_J == base.battery_capacity_J


def test_apply_overrides_does_not_mutate_base() -> None:
    base = AUVConfig()
    orig_speed = base.cruise_speed
    _ = apply_overrides(base, {"cruise_speed": 9.9})
    assert base.cruise_speed == pytest.approx(orig_speed), "apply_overrides 不应原地改写 base"


def test_apply_overrides_returns_auvconfig() -> None:
    base = AUVConfig()
    out = apply_overrides(base, {"battery_capacity_J": 1.0e7})
    assert dataclasses.is_dataclass(out)
    assert out.battery_capacity_J == pytest.approx(1.0e7)


def test_leader_vs_follower_overrides_differ() -> None:
    sim = SimConfig()
    leader = apply_overrides(AUVConfig(), sim.leader_overrides)
    follower = apply_overrides(AUVConfig(), sim.follower_overrides)
    # Leader 声呐高功耗、容量更大(契约默认值)
    assert leader.sensor_static_power_W > follower.sensor_static_power_W
    assert leader.battery_capacity_J > follower.battery_capacity_J
