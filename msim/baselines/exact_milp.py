"""[第5层][A6] 小实例 MILP 精确解 —— 真正的 ground truth(仅小实例可解)。

与 ortools_vrp 的"近似最优"对立:本模块用混合整数线性规划(pulp + CBC)求 min-max
VRP 的**全局最优**。在小实例(目标数受 max_targets 限制)上,CBC 求到的整数最优即
ground truth,可作为评判近似器/学习策略的真值上界参照。

模型(开放路径 min-max VRP,异构起点,MTZ 子环消除):
  变量 x[i,a,b] ∈ {0,1}:车 i 是否走弧 a→b(a,b ∈ {起点i} ∪ 目标集)
       u[j]:MTZ 序号(消子环);L:最长单车路径长(min-max 目标)
  目标 min L
  约束 每个目标恰被一辆车进出一次;车从自己起点出发;单车路径长 ≤ L;MTZ。
  弧代价 = 纯路径长度(与 ortools_vrp 同口径,保证可比)。
  评估时再用 eval.metrics.path_cost 完整口径折算 per-follower 能耗。

⚠ pulp/CBC 延迟 import(波次0 骨架不依赖)。仅小实例可解(NP-hard,大实例会超时)。
"""
from __future__ import annotations

from typing import List

import numpy as np

from msim.contracts.config import SimConfig
from msim.contracts.types import Waypoint

from msim.baselines.greedy import _start_poses

# 小实例守卫:目标数超过此值则拒绝(MILP 会指数爆炸,不再是"精确"可解)。
DEFAULT_MAX_TARGETS = 8


def solve_exact_minmax(
    cfg: SimConfig,
    targets: List[Waypoint],
    *,
    max_targets: int = DEFAULT_MAX_TARGETS,
    time_limit_s: float = 30.0,
) -> List[List[Waypoint]]:
    """小实例 min-max VRP 精确解(ground truth)。返回 N 条 Follower 路径。

    targets: 目标点 [x_N, y_E] 列表。无目标 → N 条空路径。
    超过 max_targets → ValueError(精确 MILP 仅对小实例有效,拒绝大实例避免伪精确)。
    pulp/CBC 在函数内延迟 import。
    """
    import pulp

    n = cfg.n_followers
    starts = _start_poses(cfg)
    if len(targets) == 0:
        return [[] for _ in range(n)]
    if len(targets) > max_targets:
        raise ValueError(
            f"exact_milp 仅解小实例:目标数 {len(targets)} > max_targets={max_targets}。"
            f"大实例请用 ortools_vrp(近似最优)。"
        )

    pts = [np.asarray(t, dtype=float)[:2] for t in targets]
    T = len(pts)

    # 节点编号:0..n-1 = 车起点;n..n+T-1 = 目标。
    start_node = list(range(n))
    tgt_node = list(range(n, n + T))
    coords = [sp[:2] for sp in starts] + pts

    def d(a: int, b: int) -> float:
        return float(np.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1]))

    prob = pulp.LpProblem("minmax_vrp_exact", pulp.LpMinimize)

    # x[i][a][b] = 车 i 走弧 a→b。允许弧:从车 i 起点 或 任一目标 → 任一其它目标。
    # 开放路径:不建到达任何起点的弧(不计返程)。
    x = {}
    for i in range(n):
        x[i] = {}
        src_nodes = [start_node[i]] + tgt_node
        for a in src_nodes:
            x[i][a] = {}
            for b in tgt_node:
                if a == b:
                    continue
                x[i][a][b] = pulp.LpVariable(f"x_{i}_{a}_{b}", cat="Binary")

    L = pulp.LpVariable("L", lowBound=0)  # min-max:最长单车路径
    prob += L  # 目标 min L

    # --- 每个目标恰好被进入一次(被某辆车从某节点访问)---
    for b in tgt_node:
        prob += (
            pulp.lpSum(
                x[i][a][b]
                for i in range(n)
                for a in x[i]
                if b in x[i][a]
            )
            == 1
        )

    for i in range(n):
        # --- 流守恒:车 i 进入目标 b 的次数 = 离开 b 的次数(目标非终点时)---
        for b in tgt_node:
            inflow = pulp.lpSum(x[i][a][b] for a in x[i] if b in x[i][a])
            outflow = pulp.lpSum(x[i][b][c] for c in x[i].get(b, {}))
            # 开放路径:离开 ≤ 进入(可在某目标收尾);且只有进入过才能离开。
            prob += outflow <= inflow
        # --- 车 i 从自己起点最多出发一条弧(可不出动)---
        prob += pulp.lpSum(x[i][start_node[i]][c] for c in x[i][start_node[i]]) <= 1
        # --- 单车路径长 ≤ L(min-max 链接)---
        prob += (
            pulp.lpSum(
                d(a, b) * x[i][a][b]
                for a in x[i]
                for b in x[i][a]
            )
            <= L
        )

    # --- MTZ 子环消除(对目标节点;聚合所有车的弧)---
    u = {b: pulp.LpVariable(f"u_{b}", lowBound=1, upBound=T) for b in tgt_node}
    for a in tgt_node:
        for b in tgt_node:
            if a == b:
                continue
            x_ab = pulp.lpSum(
                x[i][a][b] for i in range(n) if a in x[i] and b in x[i][a]
            )
            prob += u[a] - u[b] + T * x_ab <= T - 1

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s)
    prob.solve(solver)

    # --- 解析解:逐车从起点沿弧重建访问顺序 ---
    routes: List[List[np.ndarray]] = [[] for _ in range(n)]
    for i in range(n):
        cur = start_node[i]
        visited = set()
        while True:
            nxt = None
            for b in x[i].get(cur, {}):
                v = x[i][cur][b].value()
                if v is not None and v > 0.5 and b not in visited:
                    nxt = b
                    break
            if nxt is None:
                break
            routes[i].append(pts[nxt - n])
            visited.add(nxt)
            cur = nxt
    return routes
