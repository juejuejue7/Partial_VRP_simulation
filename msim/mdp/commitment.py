"""[第3层][A5] 承诺队列 + 解析确定量(剩余距离 / 可用时间)。

纪律4:开环下这两量是承诺队列**解析**计算的确定量,**非估计**。命名与注释体现确定性。
口径与 `Level1RolloutEngine.execute` / `eval.metrics.path_cost` 逐项一致(单一真相源):
- 段长 L_k = ‖w_k − w_{k-1}‖,w_{-1}=current_pos;
- 转角 Σ|Δψ| = Σ_{k≥1} |wrap(s_k − s_{k-1})|(**只累计相邻段间转角**,忽略首段转入角,§v1.0.1);
- 用时 = L/cruise + Σ|Δψ|/nominal_yaw_rate + dwell × 剩余 waypoint 数。

本模块重定义 contracts.mdp 中两解析函数的**实现**(覆盖 NIE 骨架);FollowerCommitment
为纯数据 dataclass,从 contracts 直接复用(纪律2:不改契约类)。A1 测试 / env 从
`msim.mdp.commitment` import 本实现。
"""
from __future__ import annotations

from typing import List

import numpy as np

from msim.contracts.config import RolloutConfig
from msim.contracts.mdp import FollowerCommitment  # 纯数据 dataclass,复用(纪律2)
from msim.contracts.types import FLOAT, Position, wrap_to_pi

__all__ = [
    "FollowerCommitment",
    "resolve_remaining_distance",
    "resolve_available_time",
]


def _remaining_points(commitment: FollowerCommitment) -> List[np.ndarray]:
    """承诺队列中**尚未走完**的 waypoint(cursor 起,含 cursor)。

    cursor 之前的点已走完(is_busy 语义:cursor<len 即忙)。返回 (2,) float 数组列表。
    """
    return [
        np.asarray(commitment.queue[k], dtype=FLOAT)[:2]
        for k in range(commitment.cursor, len(commitment.queue))
    ]


def resolve_remaining_distance(current_pos: Position,
                               commitment: FollowerCommitment) -> float:
    """剩余执行距离 d_rem(承诺队列**解析**计算的确定量,**非估计**):

        d = ‖pos − w_cursor‖ + Σ_k ‖w_k − w_{k+1}‖   (队列内相邻剩余点)

    pos = current_pos;w_cursor..w_last = 队列内 cursor 起的剩余点。
    无剩余点(队列空或已走完)→ 0.0。
    """
    pts = _remaining_points(commitment)
    if len(pts) == 0:
        return 0.0

    p = np.asarray(current_pos, dtype=FLOAT)[:2]
    seq = [p] + pts
    total = 0.0
    for k in range(1, len(seq)):
        d = seq[k] - seq[k - 1]
        total += float(np.hypot(d[0], d[1]))
    return total


def resolve_available_time(current_pos: Position, commitment: FollowerCommitment,
                           cfg: RolloutConfig) -> float:
    """预计可用时间 t_avail(承诺队列**解析**计算的确定量,**非估计**):

        t = d_rem / cruise_speed
          + Σ |Δψ_k| / nominal_yaw_rate         (转弯时间近似;Q3:转弯计入时间)
          + dwell_time × 剩余 waypoint 数
    与 Level 1 rollout.duration_s 口径逐项一致(单一真相源)。

    Σ|Δψ| 口径(§v1.0.1):设 p=current_pos、剩余点 w_cursor..w_last,
    段方向 s_0=dir(p→w_cursor)、s_k=dir(w_{k-1}→w_k);
    **只累计相邻段间转角** Σ_{k≥1} |wrap(s_k − s_{k-1})|。
    因本函数只有 Position、无朝向,**不含**首段转入角(s_0 不参与转角);
    Level 1 execute 同口径忽略 start_pose.psi 的首段转入角,二者逐项一致。

    无剩余点 → 0.0。零长段方向不确定,沿用前一段方向(不引入虚假转角)。
    """
    pts = _remaining_points(commitment)
    n = len(pts)
    if n == 0:
        return 0.0

    p = np.asarray(current_pos, dtype=FLOAT)[:2]
    seq = [p] + pts  # 位置序列:current_pos, w_cursor, ..., w_last

    # --- 段长 + 段方向(零长段沿用前序方向,与 rollout._segment_geometry 一致)---
    total_len = 0.0
    headings: List[float] = []
    for k in range(1, len(seq)):
        d = seq[k] - seq[k - 1]
        seg_len = float(np.hypot(d[0], d[1]))
        total_len += seg_len
        if seg_len > 0.0:
            headings.append(float(np.arctan2(d[1], d[0])))  # atan2(East, North)
        else:
            headings.append(headings[-1] if headings else 0.0)

    # --- 相邻段间转角(首段不计转入角,§v1.0.1)---
    total_turn = 0.0
    for k in range(1, len(headings)):
        total_turn += abs(wrap_to_pi(headings[k] - headings[k - 1]))

    return (
        total_len / cfg.cruise_speed_mps
        + total_turn / cfg.nominal_yaw_rate_rps
        + cfg.dwell_time_s * n
    )
