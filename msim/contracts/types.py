"""
公共类型别名 + 坐标系约定(全项目坐标系唯一真相源)。

==================================================================
坐标系(NED 水平面,与根目录 AUV 类 eta=[x,y,psi] 严格一致)
==================================================================
- x = North(指北),y = East(指东),原点在场地一角 (North=0, East=0)。
- psi = heading,从 North(+x)起、向 East(+y)为正;wrap 到 [-pi, pi)。
  (与 AUV_dynamics 的 dx=u cosψ - v sinψ / dy=u sinψ + v cosψ 自洽,无需改 AUV 类。)
- 默认场地:North 为长边(主测线方向),East 为短边;具体数值见 WorldConfig。

==================================================================
栅格离散化(坐标错位是全项目静默 bug 高发区,此处定死,全员不得各自解读)
==================================================================
- cell 索引 = (ix, iy):ix 沿 North(x),iy 沿 East(y)。
- 数组存储 = field[ix, iy](第0轴=North=x,第1轴=East=y;轴序==坐标序)。
  ⚠ 注意与图像/imshow 习惯 (row=y) 相反:本项目轴序对齐世界坐标轴序,可视化时自行转置。
- world_to_cell(xy) = (floor(x/res), floor(y/res))
- cell_to_world((ix,iy)) = ((ix+0.5)*res, (iy+0.5)*res)   # 返回 cell 中心
- 不变量:world_to_cell(cell_to_world(c)) == c(往返无损,harness 验收判据)。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:  # 仅静态类型,不在运行期产生硬依赖
    import numpy.typing as npt

# --- 标量 / 索引 ---------------------------------------------------------
Cell = Tuple[int, int]          # (ix, iy):ix 沿 North,iy 沿 East
CellSet = set                   # set[Cell];走廊覆盖/已计划覆盖的集合运算载体

# --- 连续量(numpy 数组,float64,NED) -----------------------------------
# 运行期统一用 np.ndarray;别名仅作可读性注释,不做强校验。
Pose = np.ndarray               # shape (3,)  = [x_N, y_E, psi]
Position = np.ndarray           # shape (2,)  = [x_N, y_E]
Waypoint = np.ndarray           # shape (2,)  = [x_N, y_E]

# --- 场 / 地图 -----------------------------------------------------------
Field = np.ndarray              # shape (nx, ny) float64 in [0,1];field[ix, iy];静态先验
CoverageMask = np.ndarray       # shape (nx, ny) bool;"已计划覆盖"场(reward 不重复计数用)
CoarseMap = np.ndarray          # shape (Hc, Wc) float32;窗口概率场降采样(obs["map"])

# --- 全局数值约定 --------------------------------------------------------
FLOAT = np.float64              # 位姿/坐标/物理量
OBS_FLOAT = np.float32          # 喂给 RL 的 obs


def wrap_to_pi(angle: float) -> float:
    """把角度规范到 [-pi, pi)(与 AUV_dynamics._wrap_to_pi 同义,供契约层复用)。"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
