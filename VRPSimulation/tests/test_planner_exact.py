"""精确解求解器(Held-Karp)与求解器口径的验收测试。

为什么需要精确解 —— 实测证据(2026-08-25, dense_1, 169 轮逐轮对照)
================================================================================
`SetGlobalSpanCostCoefficient(1000)` 把目标函数放大约三个数量级、变成由 max路程
主导,而 ortools 的 GLS 惩罚加在**弧代价**上、强度由
`guided_local_search_lambda_coefficient`(默认 0.1)控制。两者量级差 1e4 ⇒ 惩罚
永远不足以让任何邻域解在增广目标下显得更优 ⇒ **GLS 退化成普通贪心下降**
(实测:GLS 与 GREEDY_DESCENT 的解逐位相同),爬到第一个局部最优就停,剩下的
预算纯空转。后果是 169 轮里 14 轮(8.3%)非最优,最差一轮目标值 +43.5%;
且**加时间无效**(某轮 k=11 给到 30 s 仍 +38.9%)。

池规模实测上限 k=13(七个场景 775 次求解无一超过),这个规模可以直接精确求解:
外层枚举 2^k 种分工,内层 Held-Karp 一次 DP 就给出所有子集的最短开放路径,
总复杂度 O(2^k·k²)。实测耗时 k=12 → 171 ms, k=13 → 378 ms, k=15 → 1.6 s
(每 +1 个点约 ×2.15),相对 15 s 规划周期有 40 倍余量。

⚠ 本文件测的是**求解器口径**,不是仿真行为:哪些情况必须给出全局最优、哪些
情况允许退回 ortools、以及退化必须可审计(返回的 solver 名要如实反映)。
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from vrpsim.contracts.mission import MissionConfig
from vrpsim.planner import solve_minmax_vrp_from

SPAN = 1000.0


# ======================================================================
# 参照实现:暴力枚举(只用于测试,复杂度 (k+1)·k!,故 k <= 6)
# ======================================================================
def _route_len(start, tg, route) -> float:
    p = np.asarray(start, dtype=float)
    s = 0.0
    for j in route:
        s += float(np.hypot(*(tg[j] - p)))
        p = tg[j]
    return s


def _objective(starts, tg, routes) -> float:
    """与 ortools 建模一致:Σ弧代价 + span系数·max路程。

    `SetGlobalSpanCostCoefficient(c)` 加的是 c·(max end_cumul − min start_cumul),
    而 `AddDimension(..., fix_start_cumul_to_zero=True)` ⇒ start_cumul 恒 0,
    故等于 c·max路程。
    """
    lens = [_route_len(starts[i], tg, r) for i, r in enumerate(routes)]
    return sum(lens) + SPAN * max(lens)


def _brute_force_optimum(starts, tg) -> float:
    """穷举所有 (分工 × 次序),返回最优目标值。2 台车,k <= 6。"""
    k = len(tg)
    best = float("inf")
    for perm in itertools.permutations(range(k)):
        for cut in range(k + 1):
            best = min(best, _objective(starts, tg, [perm[:cut], perm[cut:]]))
    return best


def _random_case(rng, k):
    starts = [rng.uniform(-60.0, 60.0, 2), rng.uniform(-60.0, 60.0, 2)]
    tg = [rng.uniform(-250.0, 250.0, 2) for _ in range(k)]
    return starts, tg


# ======================================================================
# 1. 精确解确实是最优解
# ======================================================================
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_exact_matches_brute_force(k):
    """随机算例上与暴力枚举逐位一致 —— 这是"精确"二字的唯一凭据。"""
    rng = np.random.default_rng(1000 + k)
    for _ in range(12):
        starts, tg = _random_case(rng, k)
        routes, used = solve_minmax_vrp_from(starts, tg, solver="exact")
        assert used == "exact"
        got = _objective(starts, tg, routes)
        opt = _brute_force_optimum(starts, tg)
        assert got == pytest.approx(opt, rel=1e-9), \
            f"k={k}: 精确解 {got:.6f} 不等于暴力枚举最优 {opt:.6f}"


def test_exact_visits_every_target_exactly_once():
    """分工必须是一个划分:不重不漏。"""
    rng = np.random.default_rng(7)
    for k in (1, 3, 7, 11):
        starts, tg = _random_case(rng, k)
        routes, _ = solve_minmax_vrp_from(starts, tg, solver="exact")
        flat = sorted(j for r in routes for j in r)
        assert flat == list(range(k)), f"k={k}: 目标未被恰好访问一次 → {flat}"


def test_exact_beats_ortools_on_the_recorded_failure_shape():
    """ortools 会输的那类构型:两台车分处两侧、目标明显该二分。

    首解 `PATH_CHEAPEST_ARC` 把全部目标塞给一台车,而退化后的 GLS 爬不出来。
    这里不断言 ortools 一定输(它有随机性),只断言**精确解不比它差**,
    且精确解达到暴力枚举最优。
    """
    starts = [np.array([0.0, -60.0]), np.array([0.0, 60.0])]
    tg = [np.array([20.0, -80.0]), np.array([40.0, -70.0]), np.array([60.0, -85.0]),
          np.array([20.0, 80.0]), np.array([40.0, 70.0]), np.array([60.0, 85.0])]
    ex_routes, _ = solve_minmax_vrp_from(starts, tg, solver="exact")
    ex = _objective(starts, tg, ex_routes)
    assert ex == pytest.approx(_brute_force_optimum(starts, tg), rel=1e-9)

    or_routes, used = solve_minmax_vrp_from(starts, tg, solver="ortools",
                                            time_limit_s=1.0)
    if used == "ortools":                       # 无 ortools 环境时跳过比较
        assert ex <= _objective(starts, tg, or_routes) * (1 + 1e-9)


# ======================================================================
# 2. 确定性 —— 这是换精确解的第二个理由(ortools routing 没有随机种子可固定,
#    它的不可复现来自"预算内能跑多少次迭代取决于机器负载")
# ======================================================================
def test_exact_is_bit_identical_across_calls():
    rng = np.random.default_rng(99)
    for k in (5, 9, 13):
        starts, tg = _random_case(rng, k)
        a, _ = solve_minmax_vrp_from(starts, tg, solver="exact")
        b, _ = solve_minmax_vrp_from(starts, tg, solver="exact")
        assert a == b, f"k={k}: 同输入两次调用结果不同,精确解必须是构造性确定的"


# ======================================================================
# 3. 求解器口径:什么时候精确、什么时候退 ortools,且退化可审计
# ======================================================================
def test_exact_falls_back_above_threshold_and_says_so():
    """超过 `vrp_exact_max_targets` 必须退回 ortools,并**如实**报出 solver 名。

    不静默混用 —— `Assignment.solver` / `PlanRound.solver` 会把这个名字记进
    结果文件,事后审计得出来哪一轮用的是什么。
    """
    rng = np.random.default_rng(5)
    cap = 6
    starts, tg = _random_case(rng, cap)
    _, used = solve_minmax_vrp_from(starts, tg, solver="exact", exact_max_targets=cap)
    assert used == "exact", "恰好等于阈值时应当仍走精确解"

    starts, tg = _random_case(rng, cap + 1)
    _, used = solve_minmax_vrp_from(starts, tg, solver="exact", exact_max_targets=cap,
                                    time_limit_s=0.1)
    assert used != "exact", "超过阈值必须退化"
    assert used in ("ortools", "greedy"), f"退化后的 solver 名不合法: {used}"


def test_exact_falls_back_when_not_two_vehicles():
    """外层枚举是 2^k 的二分工,只对 2 台车成立;其它机数必须退化,不许假装精确。"""
    rng = np.random.default_rng(11)
    _, tg = _random_case(rng, 4)
    for n_veh in (1, 3):
        starts = [rng.uniform(-60.0, 60.0, 2) for _ in range(n_veh)]
        routes, used = solve_minmax_vrp_from(starts, tg, solver="exact",
                                            time_limit_s=0.1)
        assert used != "exact", f"{n_veh} 台车时不许报 exact"
        assert len(routes) == n_veh
        assert sorted(j for r in routes for j in r) == list(range(len(tg)))


def test_empty_pool_is_trivial_not_exact():
    starts = [np.zeros(2), np.ones(2)]
    routes, used = solve_minmax_vrp_from(starts, [], solver="exact")
    assert used == "trivial"
    assert routes == [[], []]


def test_exact_threshold_default_comes_from_the_contract():
    """阈值的唯一真相源是契约,不许在 planner 里另写一个数字。"""
    cfg = MissionConfig()
    assert cfg.vrp_exact_max_targets == 13
    assert cfg.solver == "exact", "契约默认求解器应为 exact(见 D21)"


def test_exact_stays_within_the_planning_cycle_at_the_threshold():
    """阈值处的最坏单轮耗时必须远小于规划周期 —— 这是阈值取 13 的依据。

    实测 k=13 约 0.4 s;这里放宽到 3 s 只为挡住"实现退化成指数灾难"的回归,
    不做性能基准(CI 机器慢,别把它当 benchmark)。
    """
    rng = np.random.default_rng(3)
    cap = MissionConfig().vrp_exact_max_targets
    starts, tg = _random_case(rng, cap)
    t0 = time.perf_counter()
    _, used = solve_minmax_vrp_from(starts, tg, solver="exact")
    dt = time.perf_counter() - t0
    assert used == "exact"
    assert dt < 3.0, f"k={cap} 精确求解耗时 {dt:.2f} s,超出规划周期可接受范围"


# ======================================================================
# 4. 预算口径:两个 VRP 方法共用同一个数字(结构保证,不是人工同步)
# ======================================================================
def test_solver_budget_is_shared_by_both_vrp_methods():
    from vrpsim.contracts.twophase import TwoPhaseConfig
    base = MissionConfig(vrp_time_limit_s=7.0)
    assert TwoPhaseConfig(base=base).vrp_time_limit_s == 7.0


def test_default_budget_is_thirty_seconds():
    """二段式全局VRP(n=46..66)实测 1 s 远未收敛:最长单机航程虚高 13~34%。

    人工裁决(2026-08-25):直接给 30 s —— 相对几千秒的任务时长可忽略。
    """
    assert MissionConfig().vrp_time_limit_s == 30.0
