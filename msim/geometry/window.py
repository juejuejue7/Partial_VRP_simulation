"""[第1层][A2] Leader 后方滑动观测窗口。实现 contracts.geometry 的 window 接口。

验收(harness §4.2):窗口始终在 Leader 后方;随 Leader 前进单调推进;边界裁剪正确。

实现落点说明:contracts/geometry.py 的 get_window / window_* / coarse_* 为 NIE 骨架
(契约层只定签名)。本模块(A2 拥有)重定义同名**实现**,复用 contracts 的 WindowRegion
纯数据 dataclass(纪律2:不改该类)。A1 测试 / A5 从 `msim.geometry.window` import 本实现。

几何约定(NED,axis 序 = [North=x, East=y]):
- 前进单位向量  fwd = (cos psi, sin psi)
- 后方单位向量  back = -fwd
- 横向单位向量  lat = (-sin psi, cos psi)   # Leader 左手(East 侧)为正
- 窗口 = 以 Leader 位姿为锚,沿 back 延伸 [0, look_back_m]、横向 [-width/2, +width/2] 的矩形。
  ⇒ 任意窗口内点都在 Leader 正后方(沿 fwd 投影 <= 0),Leader 前进时窗口锚随之前移 → 单调推进。
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from msim.contracts.config import SensorConfig, WindowConfig
from msim.contracts.geometry import GridProtocol, WindowRegion  # 纯数据 dataclass + 协议,复用
from msim.contracts.types import FLOAT, Cell, CellSet, Pose, Position

__all__ = [
    "WindowRegion", "get_window",
    "window_contains_world", "window_contains_cell", "window_cells",
    "coarse_grid_shape", "coarse_index_to_world",
]


# --- 内部:由 Leader 朝向构造后方/横向单位向量 ----------------------------
def _frame(psi: float) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (back, lat) 单位向量(NED axis 序 [North, East])。"""
    c, s = math.cos(psi), math.sin(psi)
    back = np.array([-c, -s], dtype=FLOAT)   # -fwd
    lat = np.array([-s, c], dtype=FLOAT)     # Leader 左手(East 侧)为正
    return back, lat


def _local_coords(window: WindowRegion, xy: Position) -> Tuple[float, float]:
    """世界点 → 窗口局部坐标 (s=沿后方距离, t=横向偏移)。"""
    lx, ly, psi = float(window.leader_pose[0]), float(window.leader_pose[1]), float(window.leader_pose[2])
    back, lat = _frame(psi)
    d = np.array([float(xy[0]) - lx, float(xy[1]) - ly], dtype=FLOAT)
    s = float(d @ back)
    t = float(d @ lat)
    return s, t


# ======================================================================
# get_window —— 由 Leader 实际位姿生成当前观测窗口
# ======================================================================
def get_window(leader_pose: Pose, cfg: WindowConfig) -> WindowRegion:
    """由 Leader 实际位姿生成当前观测窗口。

    约束:窗口始终在 Leader 后方;Leader 前进时窗口单调推进;边界裁剪由
    window_cells / window_contains_world 在查询时按场地与窗口几何完成。
    """
    return WindowRegion(
        leader_pose=np.asarray(leader_pose, dtype=FLOAT).copy(),
        look_back_m=float(cfg.look_back_m),
        width_m=float(cfg.width_m),
    )


# ======================================================================
# 点 / cell 隶属判定
# ======================================================================
def window_contains_world(window: WindowRegion, xy: Position) -> bool:
    """世界坐标点是否在窗口内(裁剪概率场、"永久错过"判定)。

    判定:局部坐标 s ∈ [0, look_back_m] 且 |t| <= width_m/2。
    s>=0 即"在 Leader 后方或正侧",s<0(前方)一律 False → 保证窗口恒在后方。
    """
    s, t = _local_coords(window, xy)
    half_w = window.width_m / 2.0
    # 含 1e-9 容差吸收浮点误差(边界点稳定归属)。
    return (-1e-9 <= s <= window.look_back_m + 1e-9) and (abs(t) <= half_w + 1e-9)


def window_contains_cell(window: WindowRegion, cell: Cell, grid: GridProtocol) -> bool:
    """cell 中心是否在窗口内。"""
    center = grid.cell_to_world(cell)
    return window_contains_world(window, center)


def window_cells(window: WindowRegion, grid: GridProtocol) -> CellSet:
    """窗口覆盖的细网格 cell 集合(概率场裁剪/降采样)。

    边界裁剪:遍历窗口 AABB 内的 cell,只收"中心在窗口内 且 in_bounds"者
    → 超出场地边界的 cell 自动剔除(harness:边界裁剪正确)。
    """
    lx, ly = float(window.leader_pose[0]), float(window.leader_pose[1])
    psi = float(window.leader_pose[2])
    back, lat = _frame(psi)
    half_w = window.width_m / 2.0

    # 窗口四角(局部 → 世界)求轴对齐包围盒,缩小遍历范围。
    corners = []
    for s in (0.0, window.look_back_m):
        for t in (-half_w, half_w):
            corners.append(np.array([lx, ly], dtype=FLOAT) + s * back + t * lat)
    corners = np.asarray(corners)
    # corners = np.rint(corners)  # 吸收浮点误差
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)

    c_lo = grid.world_to_cell(np.array([x_min, y_min], dtype=FLOAT))
    c_hi = grid.world_to_cell(np.array([x_max, y_max], dtype=FLOAT))
    ix0, ix1 = min(c_lo[0], c_hi[0]), max(c_lo[0], c_hi[0])
    iy0, iy1 = min(c_lo[1], c_hi[1]), max(c_lo[1], c_hi[1])

    cells: CellSet = set()
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            cell = (ix, iy)
            if grid.in_bounds(cell) and window_contains_cell(window, cell, grid):
                cells.add(cell)
    return cells


# ======================================================================
# 粗网格 —— 形状由 window÷footprint 推导(Q5:相机 footprint 覆盖整格)
# ======================================================================
def coarse_grid_shape(window: WindowConfig, sensor: SensorConfig) -> Tuple[int, int]:
    """返回 (Hc, Wc):

        Wc = ceil(look_back_m / footprint_along_m)     # 沿测线格数(列)
        Hc = ceil(width_m     / footprint_lateral_m)   # 横向格数(行)

    设计意图(Q5):每个粗格 <= 相机 footprint,Follower 抵达粗格中心拍照即覆盖整格。
    """
    Wc = math.ceil(window.look_back_m / sensor.footprint_along_m)
    Hc = math.ceil(window.width_m / sensor.footprint_lateral_m)
    return (int(Hc), int(Wc))


def coarse_index_to_world(coarse_idx: int, window: WindowRegion,
                          cfg: WindowConfig, sensor: SensorConfig) -> Position:
    """粗网格标量动作索引 → 窗口内该粗格 **中心** 的世界坐标 waypoint。

    纯几何映射(Q5/动作契约):不做可达性检查、避障、吸附。
    coarse_idx ∈ [0, Hc*Wc);解为 (row=横向, col=沿测线),结合 window 位姿投回世界。

    粗格中心局部坐标:
        s_c = (col + 0.5) * (look_back_m / Wc)        # 沿后方
        t_c = -width/2 + (row + 0.5) * (width_m / Hc) # 横向
    则 world = leader_xy + s_c*back + t_c*lat。s_c∈(0,look_back),|t_c|<width/2
    ⇒ 中心必落在窗口内(window_contains_world == True),粗↔细可逆。
    """
    Hc, Wc = coarse_grid_shape(cfg, sensor)
    n = Hc * Wc
    if not (0 <= coarse_idx < n):
        raise IndexError(f"coarse_idx {coarse_idx} 越界 [0,{n})")

    row = coarse_idx // Wc          # 横向(0..Hc-1)
    col = coarse_idx % Wc           # 沿测线(0..Wc-1)

    cell_along = window.look_back_m / Wc
    cell_lateral = window.width_m / Hc
    s_c = (col + 0.5) * cell_along
    t_c = -window.width_m / 2.0 + (row + 0.5) * cell_lateral

    lx, ly = float(window.leader_pose[0]), float(window.leader_pose[1])
    psi = float(window.leader_pose[2])
    back, lat = _frame(psi)
    world = np.array([lx, ly], dtype=FLOAT) + s_c * back + t_c * lat
    return world.astype(FLOAT)
