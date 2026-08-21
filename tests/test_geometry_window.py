"""[A1 验收测试 · #5 A2 地基] window —— 后方/单调推进/边界裁剪 + 粗网格形状。

判据来源:contracts/geometry.py(WindowRegion / get_window / window_contains_* /
window_cells / coarse_grid_shape / coarse_index_to_world)与 harness §4.2(window 行)。

窗口几何(契约 docstring):以 Leader 位姿为锚,沿 -heading(正后方)延伸 look_back_m、
横向 ±width_m/2 的矩形。约束:窗口始终在 Leader 后方;Leader 前进时窗口单调推进;
超界正确裁剪。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from msim.contracts.config import SensorConfig, WindowConfig, WorldConfig
from msim.contracts.geometry import WindowRegion
from msim.geometry.window import (
    coarse_grid_shape,
    coarse_index_to_world,
    get_window,
    window_cells,
    window_contains_cell,
    window_contains_world,
)
from msim.geometry.grid import Grid


@pytest.fixture
def wincfg() -> WindowConfig:
    return WindowConfig()  # look_back=30, width=20, advance_threshold=5


@pytest.fixture
def grid() -> Grid:
    return Grid(WorldConfig())


def _pose(x: float, y: float, psi: float) -> np.ndarray:
    return np.array([x, y, psi], dtype=np.float64)


# ----------------------------------------------------------------------
# get_window:把 WindowConfig 的尺度搬进 WindowRegion,锚在 Leader 位姿
# ----------------------------------------------------------------------
def test_get_window_carries_config(wincfg: WindowConfig) -> None:
    pose = _pose(100.0, 50.0, 0.0)
    w = get_window(pose, wincfg)
    assert isinstance(w, WindowRegion)
    np.testing.assert_allclose(np.asarray(w.leader_pose, dtype=float), pose)
    assert w.look_back_m == pytest.approx(wincfg.look_back_m)
    assert w.width_m == pytest.approx(wincfg.width_m)


# ----------------------------------------------------------------------
# 窗口始终在 Leader 后方:正后方点入窗,正前方点出窗(heading=+North)
# ----------------------------------------------------------------------
def test_window_is_behind_leader_heading_north(wincfg: WindowConfig) -> None:
    # Leader 在 (100,50),朝 North(psi=0)。后方 = North 减小方向。
    pose = _pose(100.0, 50.0, 0.0)
    w = get_window(pose, wincfg)

    behind = np.array([100.0 - 10.0, 50.0])      # 正后方 10m,在 look_back 内
    ahead = np.array([100.0 + 10.0, 50.0])       # 正前方 10m
    assert window_contains_world(w, behind) is True, "正后方点应在窗口内"
    assert window_contains_world(w, ahead) is False, "正前方点不应在窗口内"


def test_window_is_behind_leader_heading_east(wincfg: WindowConfig) -> None:
    # Leader 朝 East(psi=pi/2)。后方 = East 减小方向。
    pose = _pose(100.0, 50.0, math.pi / 2.0)
    w = get_window(pose, wincfg)

    behind = np.array([100.0, 50.0 - 10.0])      # East 方向后方
    ahead = np.array([100.0, 50.0 + 10.0])
    assert window_contains_world(w, behind) is True
    assert window_contains_world(w, ahead) is False


# ----------------------------------------------------------------------
# 沿测线深度边界:刚好 look_back 内/外
# ----------------------------------------------------------------------
def test_window_lookback_depth_boundary(wincfg: WindowConfig) -> None:
    pose = _pose(100.0, 50.0, 0.0)
    w = get_window(pose, wincfg)
    inside = np.array([100.0 - (wincfg.look_back_m - 0.5), 50.0])
    outside = np.array([100.0 - (wincfg.look_back_m + 0.5), 50.0])
    assert window_contains_world(w, inside) is True
    assert window_contains_world(w, outside) is False, "超过 look_back 深度应出窗"


# ----------------------------------------------------------------------
# 横向边界:±width/2 内/外
# ----------------------------------------------------------------------
def test_window_lateral_boundary(wincfg: WindowConfig) -> None:
    pose = _pose(100.0, 50.0, 0.0)
    w = get_window(pose, wincfg)
    half = wincfg.width_m / 2.0
    # 后方 10m、横向偏移
    inside = np.array([90.0, 50.0 + (half - 0.5)])
    outside = np.array([90.0, 50.0 + (half + 0.5)])
    assert window_contains_world(w, inside) is True
    assert window_contains_world(w, outside) is False, "超过 ±width/2 横向应出窗"


# ----------------------------------------------------------------------
# 随 Leader 前进单调推进:Leader 沿 +North 前进后,新窗覆盖的最前沿单调前移
# ----------------------------------------------------------------------
def test_window_monotone_advance(wincfg: WindowConfig) -> None:
    pose0 = _pose(50.0, 50.0, 0.0)
    pose1 = _pose(60.0, 50.0, 0.0)   # 前进 10m
    w0 = get_window(pose0, wincfg)
    w1 = get_window(pose1, wincfg)

    # 前进后,原先在 w0 最前沿之前(更靠前)的点应被 w1 纳入而非 w0
    # 取一个介于两 Leader 位置之间、紧贴 pose0 后方的点
    p = np.array([50.0 - 1.0, 50.0])  # pose0 后方 1m
    assert window_contains_world(w0, p) is True
    # 该点相对 pose1 是后方 11m,仍在 look_back(30)内 → w1 也含它(窗口在推进而非跳变丢格)
    assert window_contains_world(w1, p) is True

    # 而 pose1 前方的点两窗都不含(窗口未越到 Leader 前方)
    ahead1 = np.array([60.0 + 5.0, 50.0])
    assert window_contains_world(w0, ahead1) is False
    assert window_contains_world(w1, ahead1) is False


# ----------------------------------------------------------------------
# window_contains_cell 与 window_contains_world 在 cell 中心上一致
# ----------------------------------------------------------------------
def test_window_contains_cell_matches_world(wincfg: WindowConfig, grid: Grid) -> None:
    pose = _pose(100.0, 50.0, 0.0)
    w = get_window(pose, wincfg)
    for cell in [(450, 250), (480, 240), (520, 260)]:
        center = grid.cell_to_world(cell)
        assert window_contains_cell(w, cell, grid) == window_contains_world(w, center)


# ----------------------------------------------------------------------
# window_cells:返回集合中每个 cell 中心都在窗口内;且非空
# ----------------------------------------------------------------------
def test_window_cells_subset_of_contains(wincfg: WindowConfig, grid: Grid) -> None:
    pose = _pose(100.0, 50.0, 0.0)
    w = get_window(pose, wincfg)
    cells = window_cells(w, grid)
    assert len(cells) > 0, "窗口内应含若干 cell"
    # 每个返回的 cell 中心都应被 contains 判为真(自洽)
    for cell in list(cells)[:500]:
        assert window_contains_cell(w, cell, grid), f"{cell} 在 window_cells 中却 contains=False"
    # 窗口面积 ≈ look_back×width,栅格数量量级核对(允许边界裁剪)
    approx_n = (wincfg.look_back_m * wincfg.width_m) / (grid.res_m ** 2)
    assert len(cells) <= approx_n * 1.2 + 10


# ----------------------------------------------------------------------
# 边界裁剪:Leader 贴近场地边缘时窗口越界部分被裁剪(返回 cell 全部 in_bounds)
# ----------------------------------------------------------------------
def test_window_cells_clipped_to_bounds(wincfg: WindowConfig, grid: Grid) -> None:
    # Leader 在原点附近朝 North,后方窗口会越出 x<0 / y 边界
    pose = _pose(2.0, 2.0, 0.0)
    w = get_window(pose, wincfg)
    cells = window_cells(w, grid)
    for cell in cells:
        assert grid.in_bounds(cell), f"窗口 cell 越界未裁剪:{cell}"


# ======================================================================
# 粗网格形状(coarse_grid_shape):Wc=ceil(look_back/along), Hc=ceil(width/lateral)
# ======================================================================
def test_coarse_grid_shape_formula() -> None:
    win = WindowConfig(look_back_m=30.0, width_m=20.0)
    sen = SensorConfig(footprint_along_m=4.0, footprint_lateral_m=6.0)
    Hc, Wc = coarse_grid_shape(win, sen)
    assert Wc == math.ceil(30.0 / 4.0) == 8, f"Wc={Wc}"
    assert Hc == math.ceil(20.0 / 6.0) == 4, f"Hc={Hc}"


def test_coarse_grid_shape_exact_division() -> None:
    win = WindowConfig(look_back_m=24.0, width_m=18.0)
    sen = SensorConfig(footprint_along_m=4.0, footprint_lateral_m=6.0)
    Hc, Wc = coarse_grid_shape(win, sen)
    assert Wc == 6
    assert Hc == 3


# ----------------------------------------------------------------------
# coarse_index_to_world:纯几何,索引 ∈ [0,Hc*Wc),返回窗口内粗格中心
# ----------------------------------------------------------------------
def test_coarse_index_to_world_in_window(wincfg: WindowConfig) -> None:
    sen = SensorConfig()
    Hc, Wc = coarse_grid_shape(wincfg, sen)
    pose = _pose(100.0, 50.0, 0.0)
    w = get_window(pose, wincfg)
    for idx in range(Hc * Wc):
        wp = np.asarray(coarse_index_to_world(idx, w, wincfg, sen), dtype=np.float64)
        assert wp.shape == (2,)
        # 每个粗格中心应落在(裁剪前的)窗口几何内
        assert window_contains_world(w, wp), f"粗格 {idx} 中心 {wp} 不在窗口内"


def test_coarse_indices_distinct(wincfg: WindowConfig) -> None:
    sen = SensorConfig()
    Hc, Wc = coarse_grid_shape(wincfg, sen)
    pose = _pose(100.0, 50.0, 0.0)
    w = get_window(pose, wincfg)
    pts = [tuple(np.round(np.asarray(coarse_index_to_world(i, w, wincfg, sen)), 6))
           for i in range(Hc * Wc)]
    assert len(set(pts)) == Hc * Wc, "不同粗格索引应映射到不同 waypoint"
