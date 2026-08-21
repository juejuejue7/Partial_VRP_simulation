"""[A1 验收测试 · #8 A6 基线] ortools_vrp / exact_milp / greedy —— 措辞与解的性质。

判据来源:baselines/{ortools_vrp,exact_milp,greedy}.py 骨架签名 + docstring、
harness §2(A6:ortools=近似最优非 ground truth;exact=小实例精确解=ground truth;
greedy=朴素贪心)、CLAUDE.md(不 overclaim)。

契约:solve_*(cfg, targets) -> List[List[Waypoint]],返回 N=cfg.n_followers 条 Follower 路径。
本套断言结构性不变量(路径条数=N、所有 target 被分配、不重复分配)与
**exact 作 ground truth 的最优性**(min-max 路程 ≤ 近似/贪心),不臆测内部求解细节。
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from msim.contracts.config import SimConfig
from msim.baselines.greedy import solve_greedy
from msim.baselines.ortools_vrp import solve_minmax_vrp
from msim.baselines.exact_milp import solve_exact_minmax


def _small_targets() -> list:
    # 6 个目标,N=2 可手感均衡分;坐标可被精确求解器在合理时间内处理
    return [np.array(p, dtype=np.float64) for p in
            [(10.0, 10.0), (12.0, 12.0), (14.0, 10.0),
             (90.0, 80.0), (88.0, 82.0), (92.0, 80.0)]]


def _cfg() -> SimConfig:
    return SimConfig(n_followers=2)


def _path_length(path: list) -> float:
    if len(path) < 2:
        return 0.0
    pts = [np.asarray(p, dtype=np.float64) for p in path]
    return float(sum(np.hypot(*(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)))


def _assigned_targets(routes: list) -> list:
    out = []
    for r in routes:
        for p in r:
            out.append(tuple(np.round(np.asarray(p, dtype=np.float64), 6)))
    return out


def _check_partition(routes: list, targets: list, n_followers: int) -> None:
    assert len(routes) == n_followers, f"应返回 {n_followers} 条路径,得 {len(routes)}"
    tgt_set = {tuple(np.round(np.asarray(t, dtype=np.float64), 6)) for t in targets}
    assigned = _assigned_targets(routes)
    assigned_set = set(assigned)
    # 所有 target 被覆盖
    assert tgt_set.issubset(assigned_set), f"未覆盖全部 target,漏:{tgt_set - assigned_set}"
    # 不重复分配(每个 target 恰被一条路径分到一次)——只在分配集合==target集合时校验
    target_assignments = [a for a in assigned if a in tgt_set]
    assert len(target_assignments) == len(set(target_assignments)), "存在 target 被重复分配"


# ======================================================================
# greedy —— 纯 numpy,无外部求解器,基本结构正确
# ======================================================================
def test_greedy_partition_valid() -> None:
    cfg = _cfg()
    routes = solve_greedy(cfg, _small_targets())
    _check_partition(routes, _small_targets(), cfg.n_followers)


# ======================================================================
# ortools —— 近似最优(措辞:非 ground truth),结构有效
# ======================================================================
def test_ortools_partition_valid() -> None:
    pytest.importorskip("ortools")
    cfg = _cfg()
    routes = solve_minmax_vrp(cfg, _small_targets())
    _check_partition(routes, _small_targets(), cfg.n_followers)


def test_ortools_docstring_says_approximate_not_ground_truth() -> None:
    """不 overclaim:ortools 措辞须为'近似最优',不得自称精确/ground truth。"""
    doc = (solve_minmax_vrp.__doc__ or "")
    assert "近似" in doc, "ortools VRP 须标注'近似最优'(min-max VRP NP-hard)"
    # 不得正向自称 ground truth;但允许免责声明(如"非 ground truth")。
    # 判据:若出现"ground truth",其前必须紧跟否定词("非")。
    low = doc.lower()
    if "ground truth" in low:
        assert "非 ground truth" in low, "ortools 不得自称 ground truth(仅允许'非 ground truth'免责声明)"


# ======================================================================
# exact_milp —— 小实例精确解 = ground truth(最优性)
# ======================================================================
def test_exact_partition_valid() -> None:
    pytest.importorskip("pulp")
    cfg = _cfg()
    routes = solve_exact_minmax(cfg, _small_targets())
    _check_partition(routes, _small_targets(), cfg.n_followers)


def test_exact_is_ground_truth_optimal_vs_baselines() -> None:
    """exact 的 min-max 路程 ≤ ortools 与 greedy(精确解是下界 → 可作 ground truth)。"""
    pytest.importorskip("pulp")
    pytest.importorskip("ortools")
    cfg = _cfg()
    targets = _small_targets()

    exact = solve_exact_minmax(cfg, targets)
    ortools = solve_minmax_vrp(cfg, targets)
    greedy = solve_greedy(cfg, targets)

    def minmax(routes):
        return max(_path_length(r) for r in routes)

    mm_exact = minmax(exact)
    # 精确解的 min-max 不应劣于任何近似/启发式(容许数值/路程口径微小松弛)
    assert mm_exact <= minmax(ortools) + 1e-6, "exact 应不劣于 ortools(近似)"
    assert mm_exact <= minmax(greedy) + 1e-6, "exact 应不劣于 greedy(朴素)"


def test_exact_docstring_claims_ground_truth() -> None:
    doc = (solve_exact_minmax.__doc__ or "")
    assert "精确" in doc, "exact_milp 须标注'精确解'(ground truth)"
