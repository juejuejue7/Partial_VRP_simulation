"""[第2b层][A4] 信息回写开关 —— 修士/博士的【唯一】注入点。

- NullUpdater(修士):空操作,field 逐位不变(开环性,harness 验收)。
- BayesianUpdater(博士):贝叶斯后验回写。
全代码库"修士 vs 博士"差异收敛于此一处;其它位置出现 if master/doctor 即违规。
"""
from __future__ import annotations

from typing import Optional

from msim.contracts.physics import CameraSensorModel
from msim.contracts.task import BeliefUpdater  # Protocol
from msim.contracts.types import CellSet, Field

__all__ = ["BeliefUpdater", "NullUpdater", "BayesianUpdater"]


class NullUpdater:
    """修士:空操作。满足 BeliefUpdater。"""

    def update(self, field: Field, observed_cells: CellSet,
               sensor: CameraSensorModel, measurements: Optional[object] = None) -> None:
        # 开环性:绝不修改 field。此实现即契约语义(NullUpdater 调用后地图逐位不变)。
        return None


class BayesianUpdater:
    """博士:真实观测回写贝叶斯后验。满足 BeliefUpdater。"""

    def update(self, field: Field, observed_cells: CellSet,
               sensor: CameraSensorModel, measurements: Optional[object] = None) -> None:
        raise NotImplementedError  # [博士阶段;修士不启用]
