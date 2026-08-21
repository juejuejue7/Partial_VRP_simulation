"""[A1 验收测试 · #5 A2 地基] grid —— 栅格↔世界往返 + 粗↔细可逆。

判据来源:contracts/types.py(坐标系唯一真相源)与 contracts/geometry.py、
agent_team_harness §4.2(grid 行)。

契约数学定义(逐字对照 types.py):
  world_to_cell(xy) = (floor(x/res), floor(y/res))
  cell_to_world((ix,iy)) = ((ix+0.5)*res, (iy+0.5)*res)   # cell 中心
  不变量:world_to_cell(cell_to_world(c)) == c   (往返无损)
  越界 world_to_cell:clip 到边界(contracts/geometry.py world_to_cell docstring)。

实现到位前应红(NotImplementedError);实现正确后应全绿——这是 A2 grid 的完成判据。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from msim.contracts.config import WorldConfig
from msim.geometry.grid import Grid


@pytest.fixture
def world() -> WorldConfig:
    # 默认场地:200×100 m,res=0.2 → nx=1000, ny=500
    return WorldConfig()


@pytest.fixture
def grid(world: WorldConfig) -> Grid:
    return Grid(world)


# ----------------------------------------------------------------------
# 基础形状/分辨率字段(契约:Grid 暴露 nx/ny/res_m)
# ----------------------------------------------------------------------
def test_grid_shape_fields(grid: Grid, world: WorldConfig) -> None:
    assert grid.nx == world.nx == 1000
    assert grid.ny == world.ny == 500
    assert grid.res_m == world.res_m == pytest.approx(0.2)


# ----------------------------------------------------------------------
# cell_to_world:返回 cell 中心
# ----------------------------------------------------------------------
def test_cell_to_world_returns_center(grid: Grid) -> None:
    res = grid.res_m
    for cell in [(0, 0), (1, 0), (0, 1), (3, 7), (999, 499)]:
        xy = np.asarray(grid.cell_to_world(cell), dtype=np.float64)
        expected = np.array([(cell[0] + 0.5) * res, (cell[1] + 0.5) * res])
        assert xy.shape == (2,)
        np.testing.assert_allclose(xy, expected, atol=1e-12)


# ----------------------------------------------------------------------
# world_to_cell:floor 量化(逐字对照契约公式)
# ----------------------------------------------------------------------
def test_world_to_cell_floor(grid: Grid) -> None:
    res = grid.res_m
    cases = {
        (0.0, 0.0): (0, 0),
        (0.19, 0.0): (0, 0),          # 同格内
        (0.2, 0.0): (1, 0),           # 跨格边界(floor)
        (0.21, 0.41): (1, 2),
        (1.5, 3.5): (int(math.floor(1.5 / res)), int(math.floor(3.5 / res))),
    }
    for xy, expected in cases.items():
        got = tuple(int(v) for v in grid.world_to_cell(np.asarray(xy, dtype=np.float64)))
        assert got == expected, f"world_to_cell({xy}) = {got}, expected {expected}"


# ----------------------------------------------------------------------
# 往返无损:world_to_cell(cell_to_world(c)) == c  (harness §4.2 核心判据)
# ----------------------------------------------------------------------
def test_roundtrip_cell_world_cell_lossless(grid: Grid) -> None:
    rng = np.random.default_rng(0)
    cells = [(0, 0), (1, 1), (999, 499), (123, 456), (500, 250)]
    cells += [(int(rng.integers(0, grid.nx)), int(rng.integers(0, grid.ny)))
              for _ in range(200)]
    for c in cells:
        xy = grid.cell_to_world(c)
        back = tuple(int(v) for v in grid.world_to_cell(xy))
        assert back == c, f"往返破损:{c} -> {tuple(xy)} -> {back}"


# ----------------------------------------------------------------------
# in_bounds 语义:[0,nx)×[0,ny)
# ----------------------------------------------------------------------
def test_in_bounds(grid: Grid) -> None:
    assert grid.in_bounds((0, 0)) is True
    assert grid.in_bounds((grid.nx - 1, grid.ny - 1)) is True
    assert grid.in_bounds((-1, 0)) is False
    assert grid.in_bounds((0, -1)) is False
    assert grid.in_bounds((grid.nx, 0)) is False
    assert grid.in_bounds((0, grid.ny)) is False


# ----------------------------------------------------------------------
# 越界 world_to_cell:clip 到边界(契约 docstring:clip 到边界)
# ----------------------------------------------------------------------
def test_world_to_cell_clips_out_of_range(grid: Grid) -> None:
    # 负坐标 → clip 到 (0,0)
    c_neg = tuple(int(v) for v in grid.world_to_cell(np.array([-5.0, -5.0])))
    assert grid.in_bounds(c_neg), f"负坐标未 clip:{c_neg}"
    assert c_neg == (0, 0)

    # 远超上界 → clip 到 (nx-1, ny-1)
    big = np.array([1e6, 1e6])
    c_big = tuple(int(v) for v in grid.world_to_cell(big))
    assert grid.in_bounds(c_big), f"超界未 clip:{c_big}"
    assert c_big == (grid.nx - 1, grid.ny - 1)


# ----------------------------------------------------------------------
# 轴序对齐世界坐标轴序(第0轴=North=x,第1轴=East=y);防坐标错位
# ----------------------------------------------------------------------
def test_axis_order_north_first(grid: Grid) -> None:
    # 沿 North(+x)移动只动 ix;沿 East(+y)移动只动 iy
    c_x = tuple(int(v) for v in grid.world_to_cell(np.array([10.0, 0.0])))
    c_y = tuple(int(v) for v in grid.world_to_cell(np.array([0.0, 10.0])))
    assert c_x[1] == 0 and c_x[0] > 0, "North 移动应只增大 ix(第0轴)"
    assert c_y[0] == 0 and c_y[1] > 0, "East 移动应只增大 iy(第1轴)"
