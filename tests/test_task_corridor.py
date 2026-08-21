"""[A1 验收测试 · #7 A4 任务/价值] corridor —— 走廊宽=横向×1.0、覆盖格集与解析解一致。

判据来源:contracts/task.py(corridor_cells)、contracts/physics.py
(CameraSensorModel.corridor_width_m = lateral×1.0;§v1.0.7:走廊宽 = footprint_lateral,
与 grid cell 严格对齐)、harness §4.2(corridor 行,不漏/不多)。

解析解约定(覆盖判据 · butt cap,A0 §v1.0.2 冻结):设线段 P0→P1,单位方向 t̂、
法向 n̂,长 L。cell 中心 c 覆盖 ⟺ 纵向投影 s=(c−P0)·t̂ 满足 0≤s≤L(投影超出端点
即不覆盖,**无圆帽**)且横向 |(c−P0)·n̂| ≤ width/2。butt cap(平头)而非 round cap
(胶囊):round cap 端部圆帽会高估 value,违"不 overclaim"。退化段(L<eps)退化为
点的方格邻域判据。A1 据此构造 ground truth 集合。
"""
from __future__ import annotations

import numpy as np
import pytest

from msim.contracts.config import WorldConfig
from msim.contracts.physics import CameraSensorModel
from msim.geometry.grid import Grid
from msim.task.corridor import corridor_cells


@pytest.fixture
def grid() -> Grid:
    # 小细网格:20×10 m,res=1.0 → nx=20, ny=10,cell 中心在 (i+0.5, j+0.5)
    return Grid(WorldConfig(x_max_m=20.0, y_max_m=10.0, res_m=1.0))


def _ref_corridor(grid: Grid, a: np.ndarray, b: np.ndarray, width_m: float) -> set:
    """解析参考(butt cap · A0 §v1.0.2):cell 中心 c 覆盖 ⟺ 纵向投影 s=(c−P0)·t̂∈[0,L]
    (投影超端即不覆盖,无圆帽)且横向 |(c−P0)·n̂| ≤ width/2。退化段→点的方格邻域。"""
    half = width_m / 2.0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    seg = b - a
    seg_len = float(np.hypot(*seg))
    out = set()
    eps = 1e-9
    if seg_len < eps:
        # 退化线段:点邻域(点到 cell 中心欧氏距 ≤ width/2),保持原方格判据。
        for ix in range(grid.nx):
            for iy in range(grid.ny):
                c = np.asarray(grid.cell_to_world((ix, iy)), dtype=np.float64)
                if float(np.hypot(*(c - a))) <= half + eps:
                    out.add((ix, iy))
        return out

    t_hat = seg / seg_len                       # 单位方向
    n_hat = np.array([-t_hat[1], t_hat[0]])      # 单位法向
    for ix in range(grid.nx):
        for iy in range(grid.ny):
            c = np.asarray(grid.cell_to_world((ix, iy)), dtype=np.float64)
            d = c - a
            s = float(d @ t_hat)                 # 纵向投影
            lateral = abs(float(d @ n_hat))      # 横向距离
            # butt cap:投影必须落在 [0,L](端部平头,无圆帽),且横向在半宽内。
            if -eps <= s <= seg_len + eps and lateral <= half + eps:
                out.add((ix, iy))
    return out


# ======================================================================
# 走廊宽度严格 = footprint 横向 × 1.0(§v1.0.7)
# ======================================================================
def test_corridor_width_used_is_lateral_times_1p0(grid: Grid) -> None:
    cam = CameraSensorModel(footprint_lateral_m=6.0)
    width = cam.corridor_width_m
    assert width == pytest.approx(6.0)
    # 用该 width 调 corridor_cells 不应抛错且返回非空
    a = np.array([2.0, 5.0]); b = np.array([18.0, 5.0])
    cells = corridor_cells(a, b, width, grid)
    assert len(cells) > 0


# ======================================================================
# 轴对齐(沿 North)线段:覆盖格集与解析解逐元素一致(不漏/不多)
# ======================================================================
def test_axis_aligned_north_matches_analytic(grid: Grid) -> None:
    a = np.array([2.0, 5.0]); b = np.array([15.0, 5.0])   # 沿 North,East=5
    width = 4.0   # half=2 → 中心 y∈[3,7] 的格被覆盖(cell 中心 y=iy+0.5)
    got = corridor_cells(a, b, width, grid)
    exp = _ref_corridor(grid, a, b, width)
    assert got == exp, f"覆盖格集不一致:漏 {exp - got},多 {got - exp}"


# ======================================================================
# 轴对齐(沿 East)线段
# ======================================================================
def test_axis_aligned_east_matches_analytic(grid: Grid) -> None:
    a = np.array([10.0, 1.0]); b = np.array([10.0, 9.0])  # 沿 East,North=10
    width = 3.0
    got = corridor_cells(a, b, width, grid)
    exp = _ref_corridor(grid, a, b, width)
    assert got == exp, f"覆盖格集不一致:漏 {exp - got},多 {got - exp}"


# ======================================================================
# 斜线段:覆盖格集与解析解一致
# ======================================================================
def test_diagonal_matches_analytic(grid: Grid) -> None:
    a = np.array([3.0, 2.0]); b = np.array([14.0, 8.0])
    width = 2.5
    got = corridor_cells(a, b, width, grid)
    exp = _ref_corridor(grid, a, b, width)
    assert got == exp, f"斜线覆盖格集不一致:漏 {exp - got},多 {got - exp}"


# ======================================================================
# 宽度单调:更宽走廊覆盖是更窄走廊的超集
# ======================================================================
def test_wider_corridor_superset(grid: Grid) -> None:
    a = np.array([2.0, 5.0]); b = np.array([15.0, 5.0])
    narrow = corridor_cells(a, b, 2.0, grid)
    wide = corridor_cells(a, b, 6.0, grid)
    assert narrow.issubset(wide), "更宽走廊应包含更窄走廊的全部覆盖格"


# ======================================================================
# 覆盖格全部 in_bounds(落栅格无越界)
# ======================================================================
def test_corridor_cells_in_bounds(grid: Grid) -> None:
    a = np.array([0.5, 0.5]); b = np.array([19.5, 9.5])
    cells = corridor_cells(a, b, 4.0, grid)
    for c in cells:
        assert grid.in_bounds(c), f"走廊覆盖格越界:{c}"


# ======================================================================
# 退化线段(零长):返回点邻域(半径 width/2 内的格),与解析一致
# ======================================================================
def test_degenerate_segment(grid: Grid) -> None:
    p = np.array([10.0, 5.0])
    width = 3.0
    got = corridor_cells(p, p, width, grid)
    exp = _ref_corridor(grid, p, p, width)
    assert got == exp
