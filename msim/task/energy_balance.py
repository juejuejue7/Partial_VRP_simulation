"""[第2b层][A4] 跨 Follower 能耗均衡惩罚(min-max 主形式)。

实现 contracts.task.balance_penalty。on/off 消融 = RewardConfig.balance_weight 0/非0
(由调用方乘权重,本函数内**不做** if 分支判断 on/off)。
"""
from __future__ import annotations

from typing import Sequence

__all__ = ["balance_penalty"]


def balance_penalty(per_follower_energy_J: Sequence[float]) -> float:
    """跨 Follower 能耗均衡惩罚。主形式 = min-max:返回 max_i E_i。

    空序列返回 0.0(A0 冻结 DECISIONS §v1.0.2 A4-Q3:防 max([]) 崩;
    语义中性:无 Follower 无失衡)。同总能耗下 max 更小 → 更均衡分配更优。
    """
    if len(per_follower_energy_J) == 0:
        return 0.0
    return float(max(per_follower_energy_J))
