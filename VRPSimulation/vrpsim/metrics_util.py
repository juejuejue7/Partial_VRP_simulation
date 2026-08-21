"""[第2层] 跨场景共用的指标计算件。定义在 `contracts/metrics.py`,这里只实现。

放在这里而不是各场景各写一份的理由:⑥ 負荷均衡 四列要在 `mission` /
`lawnmower` / `twophase` 三个场景里给出**逐位相同的口径**。抄三份必然漂移。
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from msim.contracts.types import FLOAT

__all__ = ["jain_index", "imbalance_frac", "balance_metrics"]


def jain_index(x: Sequence[float]) -> float:
    """Jain 公平性指数 = (Σx)² / (n · Σx²) ∈ (0, 1]。

    1.0 = 完全均衡;1/n = 全压在一台身上。
    ⚠ **n 小的时候很钝**:两台 1084 s vs 942 s(差 13%)也能给出 0.995。
      想看出差别请配合 `imbalance_frac` 一起读 —— 这正是 D19 两列并存的理由。
    """
    a = np.asarray(list(x), dtype=FLOAT)
    if a.size == 0:
        return 1.0
    ss = float((a ** 2).sum())
    if ss <= 0.0:
        return 1.0
    return float(a.sum() ** 2 / (a.size * ss))


def imbalance_frac(x: Sequence[float]) -> float:
    """相对不均衡 = (max − min) / max ∈ [0, 1]。0 = 完全均衡。

    比 Jain 灵敏得多:同样是两台 1084 / 942 s,这里给出 0.131 而 Jain 给 0.995。
    全为 0(还没开始跑)时返回 0.0。
    """
    a = np.asarray(list(x), dtype=FLOAT)
    if a.size == 0:
        return 0.0
    hi = float(a.max())
    if hi <= 0.0:
        return 0.0
    return float((hi - float(a.min())) / hi)


def balance_metrics(distances_m: Sequence[float],
                    busy_times_s: Sequence[float]) -> Dict[str, float]:
    """⑥ 負荷均衡 四列(D19)。

    **为什么距离与时间要分开量**:min-max VRP 均衡的是**距离**,而任务完成时刻
    由**最慢那台的时间**决定。停留时间(`dwell_time_s`)占比一高,两者就会分叉 ——
    实测二段式切成 [18 点 452.1 m, 4 点 451.2 m]:距离不均衡仅 0.2%,
    时间不均衡却有 13.1%。把两列并排放进表里,这个隐藏假设就自己显形了。

    `busy_times_s` = 各机非 IDLE(TRANSIT + DWELL)的时长,不是任务总时长 ——
    用总时长的话所有机都一样,量不出任何东西。
    """
    bt = np.asarray(list(busy_times_s), dtype=FLOAT)
    return {
        "jain_fairness": jain_index(distances_m),
        "jain_fairness_time": jain_index(busy_times_s),
        "load_imbalance_distance_frac": imbalance_frac(distances_m),
        "load_imbalance_time_frac": imbalance_frac(busy_times_s),
        # ② 稼働率の**絶対量**(2026-08-15 人工指定)。割合ではなく秒数 ——
        # 電池・耐久の律速はこちらで決まる。同じ `busy_times_s` から出すので
        # 上の `*_time*` 二列と口径がずれようがない。
        "max_busy_time_s": float(bt.max()) if bt.size else 0.0,
    }
