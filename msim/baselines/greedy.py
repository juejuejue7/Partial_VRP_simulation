"""[第5层][A6] 朴素贪心基线(纯 numpy,无外部求解器)。

弱基线:不优化全局 min-max,只逐目标做局部贪心。用途是给"近似最优"(ortools_vrp)
和"精确解"(exact_milp)提供一个下界对照(性能地板),凸显学习/优化方法的增益。

策略(min-max 感知的最近邻分配):
  反复取一个尚未分配的目标,指派给"分配后增量代价 + 当前负载"最小的那台 Follower,
  再以其最近邻接续延展路径。负载用当前路径能耗近似 → 倾向把新目标给更空闲的 Follower。

⚠ 措辞:贪心解既非近似最优也非 ground truth,仅作朴素对照基线。
"""
from __future__ import annotations

from typing import List

import numpy as np

from msim.contracts.config import RolloutConfig, SimConfig
from msim.contracts.types import Waypoint
from msim.eval.metrics import path_cost


def _start_poses(cfg: SimConfig) -> List[np.ndarray]:
    """N 台 Follower 的出发位姿(场地短边 East 方向均布,航向指 North)。

    旁路基线无 env;约定一个确定的初始布阵以保证可复现。航向 psi=0(指 North)。
    """
    n = cfg.n_followers
    y = cfg.world.y_max_m
    # East 方向均匀铺开:y_i = (i+0.5)/n * y_max;x=0(场地起始端)
    return [
        np.array([0.0, (i + 0.5) / n * y, 0.0], dtype=float) for i in range(n)
    ]


def solve_greedy(cfg: SimConfig, targets: List[Waypoint]) -> List[List[Waypoint]]:
    """贪心分配:返回 N 条 Follower 路径(每条 = 该 Follower 的 waypoint 序列)。

    targets: 待观测目标点列表(世界坐标 [x_N, y_E])。无目标 → N 条空路径。
    纯 numpy,确定可复现;代价口径用 eval.metrics.path_cost(Level 1 解析)。
    """
    n = cfg.n_followers
    rc: RolloutConfig = cfg.rollout
    starts = _start_poses(cfg)
    routes: List[List[np.ndarray]] = [[] for _ in range(n)]

    if len(targets) == 0:
        return routes

    pts = [np.asarray(t, dtype=float)[:2] for t in targets]
    remaining = set(range(len(pts)))

    # 每台 Follower 的"末端位姿"(用于增量代价)与当前能耗负载。
    tails: List[np.ndarray] = [sp.copy() for sp in starts]
    loads: List[float] = [0.0] * n

    while remaining:
        best = None  # (新负载, follower_idx, target_idx)
        for j in remaining:
            tgt = pts[j]
            for i in range(n):
                # 该目标接到 follower i 末端后的整段增量能耗。
                inc_e, _ = path_cost(tails[i], [tgt], rc)
                new_load = loads[i] + inc_e
                cand = (new_load, i, j)
                if best is None or cand < best:
                    best = cand
        new_load, i_sel, j_sel = best
        routes[i_sel].append(pts[j_sel])
        loads[i_sel] = new_load
        # 更新末端位姿:航向取最后一段朝向。
        prev = tails[i_sel]
        d = pts[j_sel] - prev[:2]
        if float(np.hypot(d[0], d[1])) >= 1e-12:
            psi = float(np.arctan2(d[1], d[0]))  # atan2(East, North)
        else:
            psi = float(prev[2])
        tails[i_sel] = np.array([pts[j_sel][0], pts[j_sel][1], psi], dtype=float)
        remaining.discard(j_sel)

    return routes
