"""[第5层][A6] min-max VRP 强基线 —— 近似最优(NP-hard,非 ground truth)。

min-max 多车辆路径问题(minimize 最长单车路径)是 NP-hard;OR-Tools 的元启发式
(GUIDED_LOCAL_SEARCH 等)给出**近似最优**解,不是真值标签。真正的 ground truth
仅由 exact_milp 在小实例上给出。

建模:
  - N 台 Follower = N 辆车,各自从自己的起点 depot 出发(异构起点,开放路径不计返程)。
  - 目标点 = 待访问节点,每点必须被某辆车访问一次。
  - 弧代价 = 纯路径长度(整数化);min-max 通过 routing 的全局跨度系数
    (SetGlobalSpanCostCoefficient)实现:惩罚"最长路径"使各车均衡。
  - 评估时再用 eval.metrics.path_cost 完整口径(含转弯+停留)折算 per-follower 能耗。

⚠ ortools 延迟 import(波次0 骨架不依赖)。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from msim.contracts.config import SimConfig
from msim.contracts.types import Waypoint

# 起点布阵与 greedy 复用同一约定,保证基线间可比。
from msim.baselines.greedy import _start_poses

# 距离整数化比例:routing 用整数代价,放大后再取整以保留厘米级精度。
_DIST_SCALE = 100  # 1 m -> 100(厘米)


def solve_minmax_vrp(
    cfg: SimConfig,
    targets: List[Waypoint],
    *,
    time_limit_s: Optional[float] = None,
) -> List[List[Waypoint]]:
    """min-max VRP 近似最优:返回 N 条 Follower 路径(NP-hard,近似最优非 ground truth)。

    targets: 待观测目标点 [x_N, y_E] 列表。无目标 → N 条空路径。
    time_limit_s: 元启发式时间预算;None 时按实例规模自适应(小实例 1s,大实例 5s)。
    ortools 在函数内延迟 import;无 ortools 时抛 ImportError 由调用方处理。
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = cfg.n_followers
    starts = _start_poses(cfg)  # N 个起点位姿 [x,y,psi]
    if len(targets) == 0:
        return [[] for _ in range(n)]

    pts = [np.asarray(t, dtype=float)[:2] for t in targets]
    n_targets = len(pts)

    # ---- 节点编排:前 N 个 = 各车起点 depot,其后 = 目标节点 ----
    # node[0..N-1]   : 车 i 的起点
    # node[N..N+T-1] : 目标 j (= node N+j)
    start_nodes = list(range(n))
    end_nodes = list(range(n))  # 开放路径:end 复用 start(返程弧代价置 0)
    n_nodes = n + n_targets

    coords: List[np.ndarray] = [sp[:2] for sp in starts] + pts

    def dist(a: int, b: int) -> int:
        # 起点(depot)之间或回到任意 depot 的弧 → 0(开放路径不计返程/空驶)。
        if b < n:  # 进入某个 depot(收尾返程)→ 不计代价
            return 0
        d = float(np.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1]))
        return int(round(d * _DIST_SCALE))

    manager = pywrapcp.RoutingIndexManager(n_nodes, n, start_nodes, end_nodes)
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(from_index: int, to_index: int) -> int:
        return dist(manager.IndexToNode(from_index), manager.IndexToNode(to_index))

    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # ---- min-max:对单车累计距离维度施加全局跨度系数 ----
    horizon = int(_DIST_SCALE * (cfg.world.x_max_m + cfg.world.y_max_m) * (n_targets + 1))
    routing.AddDimension(transit_idx, 0, horizon, True, "Distance")
    distance_dim = routing.GetDimensionOrDie("Distance")
    # 大系数 → 优先压低最长路径(min-max),而非仅最小化总和。
    distance_dim.SetGlobalSpanCostCoefficient(1000)

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    if time_limit_s is None:
        time_limit_s = 5.0 if n_targets > 12 else 1.0
    search.time_limit.FromMilliseconds(int(max(1.0, time_limit_s * 1000)))

    solution = routing.SolveWithParameters(search)
    if solution is None:
        # 退化:近似器未给解时回退贪心,保证 runner 不崩(仍标注为 VRP 槽位)。
        from msim.baselines.greedy import solve_greedy

        return solve_greedy(cfg, targets)

    routes: List[List[np.ndarray]] = [[] for _ in range(n)]
    for veh in range(n):
        idx = routing.Start(veh)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node >= n:  # 目标节点
                routes[veh].append(pts[node - n])
            idx = solution.Value(routing.NextVar(idx))
    return routes
