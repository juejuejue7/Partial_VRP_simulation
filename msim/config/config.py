"""[第0层][A2] config 实现。

dataclass 直接复用 contracts.config(数据契约即实现);本模块负责行为函数 apply_overrides。
"""
from __future__ import annotations

import dataclasses
from typing import Dict

from msim.contracts.config import (  # re-export 冻结的数据契约
    RewardConfig,
    RolloutConfig,
    SensorConfig,
    SimConfig,
    WindowConfig,
    WorldConfig,
)

__all__ = [
    "WorldConfig", "WindowConfig", "SensorConfig", "RewardConfig",
    "RolloutConfig", "SimConfig", "apply_overrides",
]


def apply_overrides(base, overrides: Dict[str, float]):
    """泛型 dataclass 字段覆盖(dataclasses.replace),不 import AUVConfig → 无反向依赖。

    physics 层调用本函数把 SimConfig.*_overrides 注入 AUVConfig(异构唯一注入路径)。

    实现要点:
    - 纯 dict → dataclass 字段合并,不对 base 的具体类型做任何假设(泛型)。
    - 仅覆盖 base 上确实存在的字段;overrides 出现未知键 → 抛 KeyError(防静默拼写错)。
    - dataclasses.replace 返回新实例,base 不被原地修改(frozen 友好,可复现)。
    """
    if not dataclasses.is_dataclass(base):
        raise TypeError(f"apply_overrides 需要 dataclass 实例,收到 {type(base)!r}")

    valid = {f.name for f in dataclasses.fields(base)}
    unknown = set(overrides) - valid
    if unknown:
        raise KeyError(
            f"overrides 含 base 不存在的字段 {sorted(unknown)};合法字段 {sorted(valid)}"
        )
    return dataclasses.replace(base, **overrides)
