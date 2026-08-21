"""[第2层] Leader 侧规划器 —— 窗口内可用点选取 + min-max VRP + 截断 + 投影补点。

为什么不直接用 `msim.baselines.ortools_vrp.solve_minmax_vrp`
--------------------------------------------------------------------------------
那个函数的车辆起点来自 `msim.baselines.greedy._start_poses(cfg)` —— 一个由
`SimConfig` 推出的**固定布阵**(`x=0, y=(i+0.5)/n*y_max`),喂不进 Follower 的
实时位置。本任务每次重解都要从两台 Follower 的**当前**位置起算,故在这里按
**同一套建模**重写一个接受显式起点的版本:

    - N 台 Follower = N 辆车,各自从自己当前位置出发;
    - 开放路径(end 复用 start,返程弧代价置 0),不计返程;
    - min-max 由 `SetGlobalSpanCostCoefficient` 实现,系数同 msim 的 1000;
    - 距离整数化比例同 msim(1 m -> 100)。

这样两边的解质量口径一致,只有起点来源不同。CLAUDE.md 硬纪律 5:不改 msim 任何文件。

无 ortools(或求解失败)时退化到确定性贪心,不让仿真崩。
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from msim.contracts.geometry import WindowRegion
from msim.contracts.types import FLOAT
from msim.geometry.window import window_contains_world

from .contracts.mission import WP_NONE, Assignment, MissionConfig

__all__ = ["available_pool", "project_to_window_front", "solve_minmax_vrp_from",
           "plan_round"]

_DIST_SCALE = 100        # 1 m -> 100(厘米),同 msim.baselines.ortools_vrp


# ======================================================================
# 1. 可规划池
# ======================================================================
def available_pool(targets_ned: np.ndarray, region: WindowRegion, *,
                   visited: Sequence[bool], occupied: Sequence[int],
                   cfg: MissionConfig) -> np.ndarray:
    """当前可被规划的目标下标。

    池 = 落在窗口内
         ∩ (未访问 | cfg.revisit_visited)
         ∩ 没有任何一台**正在前往**

    "占用"的口径由 D15 收口(关闭待裁决 6):只有每台当前正在前往的那一个点被冻结
    (D8 明文),已下发未走到的队列尾每轮全部释放回池。因为新时序里两台每轮一起被
    重新规划,min-max VRP 一次解出的两条路线天然互斥,不需要也不该再额外预留。

    `occupied[j]` = 占用目标 j 的 follower_id;没人占用则为 WP_NONE。
    """
    tg = np.asarray(targets_ned, dtype=FLOAT).reshape(-1, 2)
    pool: List[int] = []
    for j, p in enumerate(tg):
        if visited[j] and not cfg.revisit_visited:
            continue
        if int(occupied[j]) != WP_NONE:
            continue                    # 有人正前往,冻结
        if not window_contains_world(region, p):
            continue
        pool.append(j)
    return np.asarray(pool, dtype=int)


# ======================================================================
# 2. 投影补点(规格 §4)
# ======================================================================
def project_to_window_front(position_ned, region: WindowRegion,
                            world_east_max_m: float) -> np.ndarray:
    """把 Follower 当前位置投影到窗口**前边界**(North = Leader 所在纬度)。

    前边界是 North = leader_north 上、横跨窗口宽度的一段。投影 = 保持 East 不变、
    把 North 抬到前边界;East 再夹进窗口横向范围与场地范围,免得投出界。

    该点用来把落后的 Follower 拉回 Leader 附近,**不是观测目标**:
    不占用、不计覆盖(规格 §4)。
    """
    p = np.asarray(position_ned, dtype=FLOAT).reshape(2)
    leader_n = float(region.leader_pose[0])
    leader_e = float(region.leader_pose[1])
    half = float(region.width_m) / 2.0

    east = float(np.clip(p[1], leader_e - half, leader_e + half))
    east = float(np.clip(east, 0.0, world_east_max_m))
    return np.array([leader_n, east], dtype=FLOAT)


# ======================================================================
# 3. min-max VRP(显式起点)
# ======================================================================
def solve_minmax_vrp_from(starts_ned: Sequence[np.ndarray],
                          targets_ned: Sequence[np.ndarray], *,
                          time_limit_s: float = 1.0,
                          span_coefficient: int = 1000,
                          solver: str = "ortools") -> Tuple[List[List[int]], str]:
    """从给定起点出发的 min-max 开放路径 VRP。

    返回 (routes, solver_used);routes[i] 是车 i 要访问的 **targets 下标**序列。
    solver="greedy" 或 ortools 不可用/无解 → 确定性贪心退化。
    """
    n = len(starts_ned)
    tg = [np.asarray(t, dtype=FLOAT).reshape(2) for t in targets_ned]
    if n == 0:
        return [], solver
    if len(tg) == 0:
        return [[] for _ in range(n)], "trivial"

    if solver == "ortools":
        routes = _solve_ortools(starts_ned, tg, time_limit_s, span_coefficient)
        if routes is not None:
            return routes, "ortools"
    return _solve_greedy(starts_ned, tg), "greedy"


def _solve_ortools(starts, tg, time_limit_s: float,
                   span_coefficient: int) -> Optional[List[List[int]]]:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        return None

    n = len(starts)
    n_t = len(tg)
    coords = [np.asarray(s, dtype=FLOAT).reshape(2) for s in starts] + tg

    def dist(a: int, b: int) -> int:
        if b < n:                       # 回到任意 depot → 开放路径,不计返程
            return 0
        d = float(np.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1]))
        return int(round(d * _DIST_SCALE))

    manager = pywrapcp.RoutingIndexManager(n + n_t, n, list(range(n)), list(range(n)))
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(i: int, j: int) -> int:
        return dist(manager.IndexToNode(i), manager.IndexToNode(j))

    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    span = float(np.max([np.max(np.abs(c)) for c in coords])) if coords else 1.0
    horizon = int(_DIST_SCALE * max(1.0, span) * 4.0 * (n_t + 1))
    routing.AddDimension(transit_idx, 0, horizon, True, "Distance")
    routing.GetDimensionOrDie("Distance").SetGlobalSpanCostCoefficient(span_coefficient)

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search.time_limit.FromMilliseconds(int(max(1.0, time_limit_s * 1000)))

    sol = routing.SolveWithParameters(search)
    if sol is None:
        return None

    routes: List[List[int]] = [[] for _ in range(n)]
    for veh in range(n):
        idx = routing.Start(veh)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node >= n:
                routes[veh].append(node - n)
            idx = sol.Value(routing.NextVar(idx))
    return routes


def _solve_greedy(starts, tg) -> List[List[int]]:
    """确定性 min-max 感知最近邻(同 msim greedy 的思路,只是起点显式给)。"""
    n = len(starts)
    routes: List[List[int]] = [[] for _ in range(n)]
    tails = [np.asarray(s, dtype=FLOAT).reshape(2).copy() for s in starts]
    loads = [0.0] * n
    remaining = set(range(len(tg)))

    while remaining:
        best = None
        for j in sorted(remaining):                 # sorted → 平局时确定
            for i in range(n):
                d = float(np.linalg.norm(tg[j] - tails[i]))
                cand = (loads[i] + d, i, j)
                if best is None or cand < best:
                    best = cand
        load, i, j = best
        routes[i].append(j)
        tails[i] = tg[j].copy()
        loads[i] = load
        remaining.discard(j)
    return routes


# ======================================================================
# 4. 一轮联合规划(规格 §4;D15 改为一次产出全部 Follower 的序列)
# ======================================================================
def plan_round(*, t_plan_s: float, t_deliver_s: float,
               follower_positions: Sequence[np.ndarray],
               targets_ned: np.ndarray, waypoint_ids: np.ndarray,
               region: WindowRegion, visited: Sequence[bool],
               occupied: Sequence[int], world_east_max_m: float,
               cfg: MissionConfig) -> Tuple[List[Assignment], float]:
    """一轮联合规划:**一次**解出全部 Follower 的序列。

    步骤(规格 §4,D15 时序):
      1. 取窗口内未被占用的可规划点作为池;
      2. 用两台 Follower 的位置作起点解一次 min-max VRP;
      3. **每台各取自己那条路线**,各截到 `max_sequence_len` 个;
      4. 某台不足 `max_sequence_len` 个时,把该台位置投影到窗口前边界,
         投影点作为它序列的**最后一个** waypoint(不占用)。

    与 D15 之前的差别:旧版只取请求方那条路线、把另一条算完就扔。现在两条都用上,
    而 min-max VRP 的路线划分天然互斥 ⇒ **同一轮内不可能把同一个 waypoint 派给两台**。

    参数口径(不对称,D15 明文裁决):
      `follower_positions` 已由承诺队列解析推算到 **t_plan_s**(收齐回复的时刻);
      `visited` / `occupied` 则是 **t_deliver_s**(广播落地时刻)的口径 —— 那一跳的
      飞行时间里队列还在推进,按落地时刻冻结才与 `Follower.assign()` 保留的队首对齐。

    返回 `(每台一个 Assignment, 求解实测挂钟秒数)`。
    ⚠ 挂钟耗时只是诊断量,**不进入仿真时间轴**(仿真侧的余量是 `cfg.plan_solve_s`)。
    """
    starts = [np.asarray(p, dtype=FLOAT).reshape(2) for p in follower_positions]
    pool = available_pool(targets_ned, region, visited=visited, occupied=occupied,
                          cfg=cfg)

    solver_used = "trivial"
    routes: List[List[int]] = [[] for _ in starts]
    solve_wall_s = 0.0
    if pool.size:
        t0 = time.perf_counter()
        routes, solver_used = solve_minmax_vrp_from(
            starts, [targets_ned[j] for j in pool],
            time_limit_s=cfg.vrp_time_limit_s,
            span_coefficient=cfg.vrp_span_coefficient,
            solver=cfg.solver)
        solve_wall_s = time.perf_counter() - t0

    pool_ids = tuple(int(waypoint_ids[j]) for j in pool)
    out: List[Assignment] = []
    for i in range(len(starts)):
        picked = [int(pool[k]) for k in routes[i][: cfg.max_sequence_len]]
        pts = [np.asarray(targets_ned[j], dtype=FLOAT).reshape(2) for j in picked]

        has_proj = len(picked) < cfg.max_sequence_len
        if has_proj:
            pts.append(project_to_window_front(starts[i], region, world_east_max_m))

        points = (np.asarray(pts, dtype=FLOAT).reshape(-1, 2) if pts
                  else np.zeros((0, 2), dtype=FLOAT))
        out.append(Assignment(
            t_s=float(t_plan_s), follower_id=i,
            wp_ids=tuple(int(waypoint_ids[j]) for j in picked),
            points_ned=points, has_projection=bool(has_proj),
            pool_ids=pool_ids, solver=solver_used,
            leader_north_m=float(region.leader_pose[0]),
            t_deliver_s=float(t_deliver_s)))
    return out, float(solve_wall_s)
