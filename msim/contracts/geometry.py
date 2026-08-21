"""
第0/1层几何契约:栅格变换(grid)、观测窗口(window)、粗网格形状。

坐标系/索引/数组轴序的唯一真相源在 contracts/types.py,本文件不重复定义,只引用。
验收判据(harness §4.2):
- grid:栅格→世界→栅格往返误差为 0;粗↔细映射可逆。
- window:窗口始终在 Leader 后方;随 Leader 前进单调推进;边界裁剪正确。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable

import numpy as np

from .config import SensorConfig, WindowConfig, WorldConfig
from .types import Cell, CellSet, Pose, Position


# ======================================================================
# grid —— 栅格 ↔ 世界(NED)双向变换 + 粗↔细
# ======================================================================
@runtime_checkable
class GridProtocol(Protocol):
    """细网格坐标系。实现住 geometry/grid.py(A2)。"""

    nx: int          # 沿 North 列数
    ny: int          # 沿 East 行数
    res_m: float

    def world_to_cell(self, xy: Position) -> Cell:
        """世界坐标 [x_N, y_E] → cell (ix, iy);floor 量化。越界由实现 clip 或报错(契约:clip 到边界)。"""
        ...

    def cell_to_world(self, cell: Cell) -> Position:
        """cell (ix, iy) → 该 cell **中心**的世界坐标 [x_N, y_E]。"""
        ...

    def in_bounds(self, cell: Cell) -> bool:
        """cell 是否落在 [0,nx)×[0,ny)。"""
        ...


# ======================================================================
# window —— Leader 后方滑动观测窗口
# ======================================================================
@dataclass(frozen=True)
class WindowRegion:
    """Leader 后方的矩形观测区域(随 Leader 实际动力学位姿滑动)。

    几何:以 Leader 当前位姿为锚,沿 -heading(正后方)延伸 look_back_m、
    横向 ±width_m/2 的矩形。窗口编码"观测资格"(Follower 只能去已扫过、安全水域)。
    纯数据 dataclass(A2 不可改本类,纪律2);几何查询见下方独立函数(geometry/window.py 实现)。
    """
    leader_pose: Pose          # 生成本窗口时的 Leader 位姿 [x_N, y_E, psi]
    look_back_m: float
    width_m: float


def get_window(leader_pose: Pose, cfg: WindowConfig) -> WindowRegion:
    """由 Leader 实际位姿生成当前观测窗口。

    约束(harness 验收):窗口始终在 Leader 后方;Leader 前进时窗口单调推进;
    超出场地边界时正确裁剪。
    """
    raise NotImplementedError  # [A2 波次1]


def window_contains_world(window: WindowRegion, xy: Position) -> bool:
    """世界坐标点是否在窗口内(裁剪概率场、"永久错过"判定)。"""
    raise NotImplementedError  # [A2 波次1]


def window_contains_cell(window: WindowRegion, cell: Cell, grid: GridProtocol) -> bool:
    """cell 中心是否在窗口内。"""
    raise NotImplementedError  # [A2 波次1]


def window_cells(window: WindowRegion, grid: GridProtocol) -> CellSet:
    """窗口覆盖的细网格 cell 集合(概率场裁剪/降采样)。"""
    raise NotImplementedError  # [A2 波次1]


# ======================================================================
# 粗网格 —— 形状由 window÷footprint 推导(Q5:保证相机 footprint 覆盖整格)
# ======================================================================
def coarse_grid_shape(window: WindowConfig, sensor: SensorConfig) -> Tuple[int, int]:
    """返回 (Hc, Wc):

        Wc = ceil(look_back_m / footprint_along_m)     # 沿测线格数
        Hc = ceil(width_m     / footprint_lateral_m)   # 横向格数

    设计意图(Q5):每个粗格 ≤ 相机 footprint,使 Follower 抵达粗格中心拍照即可覆盖整格。
    action_space = Discrete(Hc*Wc),n 在 env 构造时按此算出。
    """
    raise NotImplementedError  # [A2 波次1]


def coarse_index_to_world(coarse_idx: int, window: WindowRegion,
                          cfg: WindowConfig, sensor: SensorConfig) -> Position:
    """粗网格标量动作索引 → 窗口内该粗格 **中心** 的世界坐标 waypoint。

    纯几何映射(Q5/动作契约):不做可达性检查、避障、吸附。
    coarse_idx ∈ [0, Hc*Wc);内部解为 (行=横向, 列=沿测线),结合 window 位姿投回世界。
    """
    raise NotImplementedError  # [A2 波次1]
