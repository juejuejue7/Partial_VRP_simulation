"""[第3层][A5] 粗网格标量动作 → 世界 waypoint(纯几何)。实现 contracts.mdp.decode_action。

不做可达性检查、避障、吸附(architecture §4.5)。内部委托 geometry 实现
`msim.geometry.window.coarse_index_to_world`(A2 拥有,纯几何),本模块只做 SimConfig
子配置(window/sensor)的取用与转发。

本模块重定义 contracts.mdp.decode_action 的**实现**(覆盖 NIE 骨架)。
"""
from __future__ import annotations

from msim.contracts.config import SimConfig
from msim.contracts.geometry import GridProtocol, WindowRegion
from msim.contracts.types import Waypoint
from msim.geometry.window import coarse_index_to_world  # A2 实现(§v1.0.2 import 路径)

__all__ = ["decode_action"]


def decode_action(action_idx: int, window: WindowRegion,
                  grid: GridProtocol, cfg: SimConfig) -> Waypoint:
    """离散动作 action_idx ∈ [0, Hc*Wc) → 窗口内该粗格中心的世界 waypoint。

    纯几何映射:**不做**可达性检查、避障、吸附。委托 geometry.coarse_index_to_world,
    其形状由 (cfg.window, cfg.sensor) 经 coarse_grid_shape 推导(Q5:footprint 覆盖整格)。

    grid 参数为契约签名一部分(粗↔细一致性载体),纯几何解码不需要它(window 位姿+尺度
    即可定位粗格中心),保留以满足契约。
    """
    return coarse_index_to_world(int(action_idx), window, cfg.window, cfg.sensor)
