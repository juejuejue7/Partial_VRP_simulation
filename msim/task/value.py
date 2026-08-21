"""[第2b层][A4] 先验期望熵降 + 不重复计数 + 已计划覆盖场。

实现 contracts.task.expected_entropy_reduction / mark_planned。
两不变量:不重复计数(planned_coverage);走廊宽 = footprint×0.9。
planned_coverage 更新(reward 用,修士也做)≠ belief 回写(修士空操作)。

单格期望熵降(bit)= 互信息 I(H; z)= H(p) − E_z[H(p_post)](A0 冻结 DECISIONS,Q6):
  p   = field[ix,iy]                          # 先验目标概率
  似然 P(z=1|H1)=p_d、P(z=1|H0)=p_fa
  q1  = p·p_d + (1−p)·p_fa,  q0 = 1 − q1     # 边际测量分布
  p1  = p·p_d/q1            (z=1 后验)
  p0  = p·(1−p_d)/q0        (z=0 后验)
  ER  = H(p) − [q1·H(p1) + q0·H(p0)]          # H 为二元熵(unit=bit → log2)
  0·log0 取 0;p∈{0,1} 或退化分母时该项贡献 0。
**开环**:只用先验 + 传感器模型,不读真实测量。**只读**:绝不改 field / planned_coverage。
"""
from __future__ import annotations

import math

import numpy as np

from msim.contracts.physics import CameraSensorModel
from msim.contracts.types import CellSet, CoverageMask, Field

__all__ = ["expected_entropy_reduction", "mark_planned"]


def _binary_entropy(q: float, log_fn) -> float:
    """二元熵 H(q) = −q·log q − (1−q)·log(1−q);0·log0 取 0(q∈{0,1} → 0)。"""
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * log_fn(q) - (1.0 - q) * log_fn(1.0 - q)


def _cell_eig(p: float, p_d: float, p_fa: float, log_fn) -> float:
    """单格期望熵降 = H(p) − E_z[H(p_post)] = 互信息 I(H; z)。"""
    q1 = p * p_d + (1.0 - p) * p_fa
    q0 = 1.0 - q1
    eig = _binary_entropy(p, log_fn)
    if q1 > 0.0:
        post1 = p * p_d / q1
        eig -= q1 * _binary_entropy(post1, log_fn)
    if q0 > 0.0:
        post0 = p * (1.0 - p_d) / q0
        eig -= q0 * _binary_entropy(post0, log_fn)
    return eig


def expected_entropy_reduction(cells: CellSet, planned_coverage: CoverageMask,
                               field: Field, sensor: CameraSensorModel,
                               *, unit: str = "bit") -> float:
    """对 cells 中 planned_coverage==False 的格累加先验期望熵降并求和。

    不重复计数:planned_coverage[ix,iy]==True 的格跳过(贡献 0)。
    **只读**:不修改 planned_coverage / field(更新由 env 显式调用 mark_planned / belief)。
    unit="bit" → log2(默认);其它字符串视为自然对数(nat)。
    """
    log_fn = math.log2 if unit == "bit" else math.log

    p_d = sensor.p_d
    p_fa = sensor.p_fa

    total = 0.0
    for (ix, iy) in cells:
        if planned_coverage[ix, iy]:
            continue  # 不重复计数:已计划覆盖的格贡献 0
        p = float(field[ix, iy])
        total += _cell_eig(p, p_d, p_fa, log_fn)
    return total


def mark_planned(planned_coverage: CoverageMask, cells: CellSet) -> None:
    """把 cells 并入"已计划覆盖"场(in-place 置 True)。

    env 在每段走廊结算后**显式**调用;唯一置 True 的入口。
    修士也执行(不破开环:只记是否计划拍过,不记结果)。
    """
    for (ix, iy) in cells:
        planned_coverage[ix, iy] = True
