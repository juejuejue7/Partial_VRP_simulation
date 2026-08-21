"""[第0层][A2] 栅格 ↔ 世界(NED)双向变换 + 粗↔细。实现 contracts.geometry.GridProtocol。

坐标系/索引/轴序见 contracts/types.py(唯一真相源)。验收:往返无损、粗细可逆。

不变量(harness §4.2):
- world_to_cell(cell_to_world(c)) == c        # 栅格→世界→栅格往返误差为 0
- cell_to_world((ix,iy)) 返回 cell **中心** = ((ix+0.5)*res, (iy+0.5)*res)
- world_to_cell(xy) = (floor(x/res), floor(y/res)),越界 clip 到 [0,nx)×[0,ny)
"""
from __future__ import annotations

import math

import numpy as np

from msim.contracts.config import WorldConfig
from msim.contracts.types import FLOAT, Cell, Position


class Grid:
    """细网格坐标系。满足 GridProtocol。"""

    def __init__(self, world: WorldConfig) -> None:
        self.nx: int = world.nx          # 沿 North(x)的列数
        self.ny: int = world.ny          # 沿 East(y)的行数
        self.res_m: float = world.res_m

    def world_to_cell(self, xy: Position) -> Cell:
        """世界坐标 [x_N, y_E] → cell (ix, iy);floor 量化,越界 clip 到边界。"""
        ix = math.floor(float(xy[0]) / self.res_m)
        iy = math.floor(float(xy[1]) / self.res_m)
        # clip 到 [0, nx)×[0, ny):契约规定越界裁剪而非报错。
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return (ix, iy)

    def cell_to_world(self, cell: Cell) -> Position:
        """cell (ix, iy) → 该 cell **中心** 的世界坐标 [x_N, y_E]。"""
        ix, iy = cell
        x = (ix + 0.5) * self.res_m
        y = (iy + 0.5) * self.res_m
        return np.array([x, y], dtype=FLOAT)

    def in_bounds(self, cell: Cell) -> bool:
        """cell 是否落在 [0,nx)×[0,ny)。"""
        ix, iy = cell
        return 0 <= ix < self.nx and 0 <= iy < self.ny
