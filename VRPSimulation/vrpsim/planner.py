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

__all__ = ["available_pool", "project_to_window_front",
           "project_to_window_front_multi", "solve_minmax_vrp_from", "plan_round"]

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


def _spread_1d(vals: Sequence[float], min_sep: float,
               lo: float, hi: float) -> List[float]:
    """把一维坐标推开到彼此至少 `min_sep`,**保持原有先后次序**,尽量留在 [lo, hi]。

    保序(而不是重排)是关键:交换次序会让两台的航路交叉,反而在半路上撞得更近。
    区间装不下 n 台时(`(n-1)*min_sep > hi-lo`)退化为均分 —— 这是几何硬约束,
    只能尽力而为,调用方应当知道此时拿不到 `min_sep`。
    """
    n = len(vals)
    if n == 0:
        return []
    if n == 1:
        return [float(np.clip(vals[0], lo, hi))]

    order = sorted(range(n), key=lambda i: float(vals[i]))
    v = [float(vals[i]) for i in order]

    if (n - 1) * min_sep > hi - lo:                      # 窗口太窄, 均分
        step = (hi - lo) / (n - 1)
        v = [lo + k * step for k in range(n)]
    else:
        for k in range(1, n):                            # 前向推开
            v[k] = max(v[k], v[k - 1] + min_sep)
        if v[-1] > hi:                                   # 越过右界 -> 往回压
            v[-1] = hi
            for k in range(n - 2, -1, -1):
                v[k] = min(v[k], v[k + 1] - min_sep)
        if v[0] < lo:                                    # 再兜一次左界
            v[0] = lo
            for k in range(1, n):
                v[k] = max(v[k], v[k - 1] + min_sep)

    out = [0.0] * n
    for k, i in enumerate(order):
        out[i] = v[k]
    return out


def project_to_window_front_multi(positions_ned: Sequence[np.ndarray],
                                  region: WindowRegion, world_east_max_m: float,
                                  *, min_sep_m: float,
                                  active: Optional[Sequence[bool]] = None
                                  ) -> List[Optional[np.ndarray]]:
    """一次性给多台 Follower 生成投影点,并保证彼此横向至少隔开 `min_sep_m`。

    为什么不能逐台独立投影(2026-08-24 修复):`project_to_window_front` 只看**自己**
    的 East,窗口里没目标时两台都被送到前边界,East 各自夹进同一个窗口区间 ⇒ 位置
    相近甚至完全相同。投影点又**不占用**(`WP_PROJECTION`),min-max VRP 的互斥保证
    只覆盖真实目标、管不到它们。sparse_2 实测:间距 <2 m 的 489 拍里 476 拍是
    "两台队首都是投影点",**没有一拍**是两台都奔真实目标 —— 撞车风险全部来自这里。

    `active[i]=False` 的台这一轮不需要投影点(它有真实目标可去),返回 None,
    也不参与去冲突 —— 它的落点是真实目标,不在前边界这条线上。

    ⚠ 只保证**落点**间隔 >= `min_sep_m`,不保证**途中**。两台从不同方位飞向同一条
      前边界的路上仍可能更近:sparse_2 实测(ortools)落点严格 2.000 m 时,途中最近
      1.14 m、87 拍(0.38%)低于 2 m。本仿真不做航路级避碰(开环、无航迹协商)。

    ⚠ 【待裁决 2026-08-24】把间隔改成"按窗口宽 / 台数均分份额"(2 台 /100 m 窗口
      ⇒ 50 m)可以把 <2 m 的拍数压到 **0**(sparse_2 min 30.0 m、Mothra min 7.05 m),
      且时长与航程同时更优。但它同时让 Mothra 在**关掉等待机制**时也达成 100% 覆盖,
      直接推翻 D16 的支撑证据(`test_leader_waiting_is_what_makes_full_coverage_possible`
      断言"关掉等待必漏点")。那是策略层变更而非 bug 修复,需人工裁决,未决前按
      `min_sep_m` 的固定值走。实测对照见该测试附近的注释与会话记录。
    """
    n = len(positions_ned)
    act = [True] * n if active is None else [bool(x) for x in active]
    leader_n = float(region.leader_pose[0])
    leader_e = float(region.leader_pose[1])
    half = float(region.width_m) / 2.0
    lo = max(leader_e - half, 0.0)
    hi = min(leader_e + half, float(world_east_max_m))

    idx = [i for i in range(n) if act[i]]
    easts = [float(np.clip(np.asarray(positions_ned[i], dtype=FLOAT).reshape(2)[1],
                           lo, hi)) for i in idx]
    spread = _spread_1d(easts, float(min_sep_m), lo, hi)

    out: List[Optional[np.ndarray]] = [None] * n
    for i, e in zip(idx, spread):
        out[i] = np.array([leader_n, e], dtype=FLOAT)
    return out


# ======================================================================
# 3. min-max VRP(显式起点)
# ======================================================================
def solve_minmax_vrp_from(starts_ned: Sequence[np.ndarray],
                          targets_ned: Sequence[np.ndarray], *,
                          time_limit_s: float = 30.0,
                          span_coefficient: int = 1000,
                          exact_max_targets: int = 13,
                          gls_lambda: float = 100.0,
                          solver: str = "exact") -> Tuple[List[List[int]], str]:
    """从给定起点出发的 min-max 开放路径 VRP。

    返回 (routes, solver_used);routes[i] 是车 i 要访问的 **targets 下标**序列。
    `solver_used` 如实反映实际走的分支(`exact` / `ortools` / `greedy` / `trivial`),
    调用方会把它记进 `Assignment.solver`,事后可审计 —— 不静默混用。

    分流口径(D21,见 `contracts/mission.py` 求解器一节):
      solver="exact"   2 台车且 1 <= len(targets) <= exact_max_targets → 精确最优;
                       否则退 ortools(再不行退贪心)。
      solver="ortools" 一律 ortools(消融对照用)。
      solver="greedy"  一律确定性贪心(无 ortools 环境的退化,测试用)。
    """
    n = len(starts_ned)
    tg = [np.asarray(t, dtype=FLOAT).reshape(2) for t in targets_ned]
    if n == 0:
        return [], solver
    if len(tg) == 0:
        return [[] for _ in range(n)], "trivial"

    if solver == "exact" and n == 2 and len(tg) <= int(exact_max_targets):
        return _solve_exact(starts_ned, tg, span_coefficient), "exact"

    if solver in ("exact", "ortools"):
        routes = _solve_ortools(starts_ned, tg, time_limit_s, span_coefficient,
                                gls_lambda)
        if routes is not None:
            return routes, "ortools"
    return _solve_greedy(starts_ned, tg), "greedy"


def _held_karp(start, tg, span_penalty_unused=None):
    """一次位掩码 DP,给出**每个**子集的最短开放路径(长度 + 访问次序)。

    dp[S][j] = 从 start 出发、恰好访问完集合 S、停在 j 的最短路程。
    开放路径 ⇒ 终点不限,故 L(S) = min_j dp[S][j]。
    返回 (best_len[S], best_end[S], parent[S][j]),后两者供回溯次序用。
    """
    k = len(tg)
    d0 = [float(np.hypot(*(tg[j] - start))) for j in range(k)]
    d = [[float(np.hypot(*(tg[a] - tg[b]))) for b in range(k)] for a in range(k)]

    size = 1 << k
    inf = float("inf")
    dp = [[inf] * k for _ in range(size)]
    par = [[-1] * k for _ in range(size)]
    for j in range(k):
        dp[1 << j][j] = d0[j]
    for s in range(1, size):
        row = dp[s]
        for j in range(k):
            cur = row[j]
            if cur == inf or not (s >> j & 1):
                continue
            dj = d[j]
            for m in range(k):
                if s >> m & 1:
                    continue
                ns = s | (1 << m)
                cand = cur + dj[m]
                if cand < dp[ns][m]:
                    dp[ns][m] = cand
                    par[ns][m] = j

    best_len = [0.0] * size
    best_end = [-1] * size
    for s in range(1, size):
        row = dp[s]
        e = min(range(k), key=row.__getitem__)
        best_len[s] = row[e]
        best_end[s] = e
    return best_len, best_end, par


def _backtrack(mask: int, best_end, par) -> List[int]:
    if mask == 0:
        return []
    seq: List[int] = []
    j = best_end[mask]
    s = mask
    while j != -1:
        seq.append(j)
        nj = par[s][j]
        s ^= (1 << j)
        j = nj
    seq.reverse()
    return seq


def _solve_exact(starts, tg, span_coefficient: int) -> List[List[int]]:
    """2 台车 min-max 开放路径 VRP 的**精确**解,复杂度 O(2^k·k²)。

    目标函数与 `_solve_ortools` 逐字对齐:Σ弧代价 + span系数·max路程
    (`SetGlobalSpanCostCoefficient(c)` 加的是 c·(max end_cumul − min start_cumul),
    而 `fix_start_cumul_to_zero=True` ⇒ start_cumul 恒 0)。两个分支同一口径,
    换求解器不会顺带换掉优化目标。

    两层:
      内层 每台车跑一次 Held-Karp —— **一次 DP 就给出所有 2^k 个子集**的最优路程;
      外层 枚举 2^k 种分工,查表取 Σ + c·max 最小者。
    故不是 O(4^k)。平局时取 mask 较小者 ⇒ 构造性确定,不需要随机种子。
    """
    s0 = np.asarray(starts[0], dtype=FLOAT).reshape(2)
    s1 = np.asarray(starts[1], dtype=FLOAT).reshape(2)
    k = len(tg)
    len0, end0, par0 = _held_karp(s0, tg)
    len1, end1, par1 = _held_karp(s1, tg)

    full = (1 << k) - 1
    c = float(span_coefficient)
    best_obj = float("inf")
    best_mask = 0
    for mask in range(1 << k):
        a = len0[mask]
        b = len1[full ^ mask]
        o = a + b + c * (a if a > b else b)
        if o < best_obj:
            best_obj = o
            best_mask = mask
    return [_backtrack(best_mask, end0, par0),
            _backtrack(full ^ best_mask, end1, par1)]


def _solve_ortools(starts, tg, time_limit_s: float,
                   span_coefficient: int,
                   gls_lambda: float = 100.0) -> Optional[List[List[int]]]:
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
    # ⚠ 默认 0.1 会让 GLS 失效:span 系数把目标值放大约 1e3,而 GLS 的惩罚加在
    #   弧代价上,量级差 1e4 ⇒ 惩罚改变不了邻域排序 ⇒ GLS 退化成贪心下降
    #   (实测其解与 GREEDY_DESCENT 逐位相同)。要紧的是 span 与 λ 的**比值**:
    #   1000/0.1 失败,1000/100 与 10/0.1 都成功。见 contracts/mission.py D21。
    search.guided_local_search_lambda_coefficient = float(gls_lambda)
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
            exact_max_targets=cfg.vrp_exact_max_targets,
            gls_lambda=cfg.vrp_gls_lambda,
            solver=cfg.solver)
        solve_wall_s = time.perf_counter() - t0

    pool_ids = tuple(int(waypoint_ids[j]) for j in pool)

    # 投影点必须**一起**算:逐台独立投影会把两台送到同一个点(投影点不占用,
    # min-max VRP 的互斥只管真实目标)。见 `project_to_window_front_multi`。
    picked_all = [[int(pool[k]) for k in routes[i][: cfg.max_sequence_len]]
                  for i in range(len(starts))]
    need_proj = [len(p) < cfg.max_sequence_len for p in picked_all]
    proj_pts = project_to_window_front_multi(
        starts, region, world_east_max_m,
        min_sep_m=cfg.projection_min_separation_m, active=need_proj)

    out: List[Assignment] = []
    for i in range(len(starts)):
        picked = picked_all[i]
        pts = [np.asarray(targets_ned[j], dtype=FLOAT).reshape(2) for j in picked]

        has_proj = need_proj[i]
        if has_proj:
            pts.append(proj_pts[i])

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
