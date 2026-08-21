"""
第2b层任务契约:走廊覆盖、期望熵降、BeliefUpdater(信息回写开关)、能耗均衡。

本层确定性但逻辑最微妙(bug 最隐蔽,A1 加倍测试)。两条价值不变量:
- 不重复计数:同一格被任一 Follower 走廊重复覆盖,期望熵降只计一次(靠 planned_coverage)。
- 走廊宽度 = footprint 横向宽度 × 0.9。

修士/博士唯一区别收敛于此:BeliefUpdater 注入点。
- planned_coverage(已计划覆盖场)更新 ≠ belief.update(地图回写)。二者绝不混为一谈:
  * planned_coverage 修士也更新,只记"哪些 cell 被计划拍过",不记测量结果 → 不破开环 → 供 reward 不重复计数。
  * belief.update 修士注入 NullUpdater(空操作,field 逐位不变);博士注入 BayesianUpdater(回写后验)。
"""
from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .geometry import GridProtocol
from .physics import CameraSensorModel
from .types import CellSet, CoverageMask, Field, Waypoint


# ======================================================================
# corridor —— 走廊覆盖(footprint×0.9 扩带 → 细网格覆盖格集)
# ======================================================================
def corridor_cells(seg_start: Waypoint, seg_end: Waypoint,
                   width_m: float, grid: GridProtocol) -> CellSet:
    """把线段 [seg_start → seg_end] 按 width_m 扩成带,落栅格,返回覆盖的 cell 集合。

    width_m 由调用方传 CameraSensorModel.corridor_width_m。
    验收(harness):扩带宽度精确;落栅格的覆盖格集与解析解一致(不漏格/不多格)。
    """
    raise NotImplementedError  # [A4 波次1]


# ======================================================================
# value —— 先验期望熵降 + 不重复计数
# ======================================================================
def expected_entropy_reduction(cells: CellSet, planned_coverage: CoverageMask,
                               field: Field, sensor: CameraSensorModel,
                               *, unit: str = "bit") -> float:
    """对 cells 中 **尚未被任何 Follower 计划覆盖**(planned_coverage==False)的格,
    累加先验期望熵降并求和。

    单格熵降 = H(p) - E_z[H(p_post)],其中后验 p_post 由相机模型 (p_d, p_fa) 对
    测量结果 z 求期望得到(修士只用先验 + 传感器模型,不用真实测量 → 开环)。
    unit="bit"(log2)。**只读**:不修改 planned_coverage / field(更新由 env 显式调用)。
    验收(harness):不重复计数;公式与手算小例一致。
    """
    raise NotImplementedError  # [A4 波次1]


def mark_planned(planned_coverage: CoverageMask, cells: CellSet) -> None:
    """把 cells 并入"已计划覆盖"场(in-place 置 True)。

    env 在每段走廊结算后**显式**调用。修士也执行(不破开环:只记是否计划拍过,不记结果)。
    """
    raise NotImplementedError  # [A4 波次1]


# ======================================================================
# belief —— 信息回写开关(修士/博士唯一注入点)
# ======================================================================
@runtime_checkable
class BeliefUpdater(Protocol):
    """value 持有其一个实例;每段走廊结算后调 update。"""

    def update(self, field: Field, observed_cells: CellSet,
               sensor: CameraSensorModel,
               measurements: Optional[object] = None) -> None:
        """把观测结果回写概率场 field(in-place)。

        NullUpdater(修士):空操作,field 逐位不变(开环性证明,harness 验收)。
        BayesianUpdater(博士):用 measurements 做贝叶斯后验回写。
        measurements 修士恒为 None。
        """
        ...


# ======================================================================
# energy_balance —— 跨 Follower 能耗均衡惩罚(min-max 主形式)
# ======================================================================
def balance_penalty(per_follower_energy_J: Sequence[float]) -> float:
    """跨 Follower 能耗均衡惩罚。主形式 = min-max:返回 max_i E_i。

    进 reward 时乘 RewardConfig.balance_weight;weight=0 即 balance-off 消融(非 if 分支)。
    须在覆盖量可比前提下偏好 max 最小的分配(均衡与效率分离,见 project_summary §8)。
    """
    raise NotImplementedError  # [A4 波次1]
