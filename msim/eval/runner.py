"""[第5层][A6] 消融 harness:balance on/off 消融 + 多基线对照。不含 N 扫描(锁 N=2)。

balance on/off 的实现纪律(CLAUDE.md 硬纪律3/§4.2):
  消融 = RewardConfig.balance_weight 取 0(off)/ 非 0(on)的**连续权重**,**不是 if 分支**。
  统一加权目标:  J = total_energy + balance_weight · max_energy
  - balance_weight=0  → J 退化为纯总能耗(效率优先,均衡项被权重消去)。
  - balance_weight>0  → max_energy(min-max 项)进入目标,偏好均衡分配。
  同一份求解结果在两种权重下评估,权重连续作用于目标,无 if master/doctor 式分支。

求解器(三档强度):
  - greedy      朴素贪心(性能地板,既非近似最优也非 ground truth)
  - ortools_vrp min-max VRP 近似最优(NP-hard,非 ground truth)
  - exact_milp  小实例 MILP 精确解(ground truth,仅小实例)

⚠ ortools/pulp 由各基线模块延迟 import;exact_milp 对大实例自动跳过。

默认目标场口径(§v1.0.3 耦合 ground-truth 实例,人类裁定 2026-06-12):
  目标点**不再全场均匀随机**,而是消费 env_static 的耦合实例:
  真实目标点(簇点过程)= 绝对 ground truth,先验场由其高斯 KDE 派生,二者共享单一真值。
  baseline VRP/评估用 `instance.targets`(ground-truth 真值目标点,簇结构,非均匀)。
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from msim.contracts.config import SimConfig, resolve_obs_candidate_params
from msim.contracts.env_static import GroundTruthInstance
from msim.contracts.types import Waypoint
from msim.eval.metrics import (
    max_energy,
    routes_to_energies,
    summarize,
    total_energy,
)


# ======================================================================
# 默认目标场:消费 env_static 的耦合 ground-truth 实例(§v1.0.3)
# 目标点为簇结构真值(非均匀随机),先验场由同一组目标点 KDE 派生 → 单一真值。
# seed 由 SimConfig.seed 锁定;同 seed+参数 → 实例逐位复现。
# ======================================================================


def _default_cluster_params(cfg: SimConfig, n_targets: int) -> Dict[str, float]:
    """把目标总数分解成簇结构。

    簇结构(每簇点数、簇内散布)由 cfg.target_field 取,**独立于相机 footprint**(§v1.0.8)。
    - targets_per_cluster 取 cfg.target_field.targets_per_cluster(默认 3,底栖聚集典型);
      n_clusters = ceil(n_targets / targets_per_cluster)。
    - intra_spread_m 取 cfg.target_field.intra_spread_m(默认 12.0m);
      不再由 sensor footprint 派生,改相机不再动目标场几何。
    """
    per = cfg.target_field.targets_per_cluster
    n_clusters = max(1, math.ceil(n_targets / per))
    intra_spread_m = cfg.target_field.intra_spread_m
    return {
        "n_clusters": n_clusters,
        "targets_per_cluster": per,
        "intra_spread_m": float(intra_spread_m),
    }


def default_instance(cfg: SimConfig, n_targets: int = 6) -> GroundTruthInstance:
    """构造默认耦合 ground-truth 实例(targets + 由其派生的先验 field),供评估/可视化共享真值。

    消费 env_static 的簇点过程生成器(§v1.0.3):**先**撒 ground-truth 目标点(簇结构,
    绝对参考),**再**由其高斯 KDE 派生先验场。seed 由 cfg.seed 锁定 → 逐位复现。

    n_targets 经 `_default_cluster_params` 分解为簇参数;真实目标数 = n_clusters×targets_per_cluster,
    当 n_targets 非 targets_per_cluster 整数倍时向上取整(故实际目标数可能略多于 n_targets)。
    返回 GroundTruthInstance,其 `.targets` 为 ground-truth 真值目标点、`.field` 为派生先验场。

    先验 KDE 带宽 σ 取 cfg.target_field.bandwidth_m(§v1.0.8,独立于相机 footprint):
    默认 3.0m,改相机 footprint 不再影响带宽。3m 使先验呈"每目标清晰光晕峰"(>0.5 占场 ~7%),
    熵降价值非平凡;不沿用契约 None→intra_spread 的通用默认(那对 A1 直测契约函数适用,此处口径不同)。
    """
    from msim.env_static.field import make_clustered_instance

    rng = np.random.default_rng(cfg.seed)
    params = _default_cluster_params(cfg, n_targets)
    bandwidth_m = cfg.target_field.bandwidth_m  # §v1.0.8:独立于相机 footprint
    return make_clustered_instance(
        cfg.world,
        rng,
        n_clusters=params["n_clusters"],
        targets_per_cluster=params["targets_per_cluster"],
        intra_spread_m=params["intra_spread_m"],
        bandwidth_m=bandwidth_m,
    )


def default_targets(cfg: SimConfig, n_targets: int = 6) -> List[Waypoint]:
    """旁路无 env 时的默认目标点 = 耦合 ground-truth 实例的真值目标点(簇结构,§v1.0.3)。

    **不再全场均匀随机**(违 §13 已退场):返回 `default_instance(cfg, n_targets).targets`,
    即簇点过程生成的 ground-truth 目标点(绝对参考),与先验场共享单一真值。
    seed 由 cfg.seed 锁定 → 逐位复现。返回 List[shape (2,)]。

    注:实际目标数 = n_clusters×targets_per_cluster(向上取整),可能略多于 n_targets。
    需共享派生 field(评估/可视化)时改用 `default_instance`。
    """
    return list(default_instance(cfg, n_targets).targets)


def default_detections(
    cfg: SimConfig,
    instance: Optional[GroundTruthInstance] = None,
    *,
    threshold_rel: Optional[float] = None,
    fill_threshold_rel: Optional[float] = None,
    peak_min_separation_m: Optional[float] = None,
    fill_radius_m: Optional[float] = None,
    min_separation_m: Optional[float] = None,
) -> List[Waypoint]:
    """从 Leader 扫描的先验概率图提取离散【候选观测点】(估计/A′ 混合,非 ground truth)。

    这是**现实基线**的目标输入源:VRP/评估不再读真值目标点,而是消费 Leader
    扫描派生的先验场,经契约原语 `observation_candidates_from_field` 抽取候选观测
    航点。返回的点是**估计**(峰值锚点 + 高概率区覆盖补点的 A′ 混合,由概率图阈值/
    非极大抑制 + 覆盖补点得到),与真值目标点(GT)有偏差,用于度量"现实可得信息下"
    的基线性能(对照 oracle_gt 上界)。

    §v1.0.5 由纯峰值 `detections_from_field` 改为 A′ 混合 `observation_candidates_from_field`:
    纯峰值会把 σ 模糊糊成一块的多近邻目标当成单点(吞掉簇内目标数);A′ 在保留峰值
    锚点(逐点原样)前提下,对高概率区按覆盖半径补点修复该缺陷。点数通常多于纯峰值。

    instance 缺省 → `default_instance(cfg)`(同一耦合实例,共享单一真值派生的场)。

    §v1.0.5/§v1.0.6 追补:四参数统一经 config 入口 `resolve_obs_candidate_params(cfg)` 取默认值
    (config 是全项目超参单一真相源)。形参显式传入的值优先,未传(None)则用 config 解析值:
      - threshold_rel:峰值锚点相对阈值,缺省 → cfg.obs_candidate.threshold_rel(=0.3)。
      - fill_threshold_rel:补点高概率区相对阈值(§v1.0.6 与峰值阈值拆开),缺省 →
        cfg.obs_candidate.fill_threshold_rel(=0.5,更严)。提高它使补点区收缩 → 补点数单调不增。
      - peak_min_separation_m:峰值锚点 NMS 间隔,缺省 → config 解析值
        (cfg.obs_candidate.peak_min_separation_m,None 时回退 0.5×footprint_lateral=3m)。
      - fill_radius_m:覆盖补点半径 **R**,缺省 → config 解析值
        (cfg.obs_candidate.fill_radius_m,None 时回退 0.9×footprint_lateral=5.4m)。
    **R 默认经 `cfg.obs_candidate.fill_radius_m`(=5.4m,可调试旋钮)**:把它改成别的值,
    本函数会自动跟随(单一真相源),无需改本函数。R 调小 → 候选更密、点数变多。
    **补点阈值默认经 `cfg.obs_candidate.fill_threshold_rel`(=0.5)**:同为单一真相源旋钮。

    向后兼容(§v1.0.5):旧关键字 `min_separation_m=` 保留,语义=**峰值锚点间隔**,映射到
    底层 `peak_min_separation_m`;仅在新参 `peak_min_separation_m` 未传时生效。
    """
    from msim.env_static.field import observation_candidates_from_field

    if instance is None:
        instance = default_instance(cfg)

    # config 入口取默认值(单一真相源);显式形参优先,未传则回退 config 解析值。
    rt, rft, rps, rfr = resolve_obs_candidate_params(cfg)
    thr = threshold_rel if threshold_rel is not None else rt
    fthr = fill_threshold_rel if fill_threshold_rel is not None else rft
    psep = (
        peak_min_separation_m
        if peak_min_separation_m is not None
        else (min_separation_m if min_separation_m is not None else rps)
    )
    fr = fill_radius_m if fill_radius_m is not None else rfr

    return list(
        observation_candidates_from_field(
            instance.field,
            cfg.world,
            threshold_rel=thr,
            fill_threshold_rel=fthr,
            peak_min_separation_m=psep,
            fill_radius_m=fr,
        )
    )


# ======================================================================
# 加权目标(连续权重,非 if 分支)
# ======================================================================
def weighted_objective(per_follower_energy_J, balance_weight: float) -> float:
    """统一加权目标 J = total + balance_weight · max。

    balance_weight=0 → 纯 total(效率);>0 → 引入 max(min-max 均衡)。
    连续权重,无分支:balance on/off 消融的唯一切换旋钮。
    """
    return total_energy(per_follower_energy_J) + float(balance_weight) * max_energy(
        per_follower_energy_J
    )


# ======================================================================
# 单个求解器在给定目标场上的评估
# ======================================================================
def _start_poses_for_eval(cfg: SimConfig) -> List[np.ndarray]:
    """与基线一致的 Follower 起点位姿(复用 greedy 的约定)。"""
    from msim.baselines.greedy import _start_poses

    return _start_poses(cfg)


def evaluate_solver(
    name: str,
    solver: Callable[[SimConfig, List[Waypoint]], List[List[Waypoint]]],
    cfg: SimConfig,
    targets: List[Waypoint],
    *,
    fairness: str = "jain",
) -> Optional[Dict]:
    """跑一个求解器并汇总 §4.2 指标(含推理时间)。

    求解器不可用(ImportError,如缺 ortools/pulp)或拒解(ValueError,如 MILP 实例过大)
    → 返回 None(由 runner 跳过该档,不让整次消融崩溃)。
    """
    try:
        t0 = time.perf_counter()
        routes = solver(cfg, targets)
        infer_s = time.perf_counter() - t0
    except (ImportError, ValueError) as exc:  # noqa: PERF203
        return {"solver": name, "skipped": True, "reason": repr(exc)}

    starts = _start_poses_for_eval(cfg)
    energies = routes_to_energies(routes, starts, cfg.rollout)
    summary = summarize(energies, infer_s, fairness=fairness)
    summary["solver"] = name
    summary["skipped"] = False
    summary["objective"] = weighted_objective(energies, cfg.reward.balance_weight)
    summary["n_targets"] = len(targets)
    return summary


# ======================================================================
# balance on/off 消融(主入口)
# ======================================================================
def _with_balance_weight(cfg: SimConfig, w: float) -> SimConfig:
    """返回 balance_weight 替换为 w 的新 SimConfig(frozen dataclass,replace)。

    这是 balance on/off 的连续切换:只改一个权重标量,不引入任何 if 分支。
    """
    from dataclasses import replace

    new_reward = replace(cfg.reward, balance_weight=w)
    return replace(cfg, reward=new_reward)


def run_ablation(
    cfg: SimConfig,
    targets: Optional[List[Waypoint]] = None,
    *,
    fairness: str = "jain",
    solvers: Optional[List[str]] = None,
    balance_on_weight: Optional[float] = None,
    target_source: str = "scanned_map",
) -> dict:
    """跑 balance on/off 消融,对每个求解器汇总 §4.2 指标。

    balance off = balance_weight 0;balance on = balance_weight = balance_on_weight
    (默认取 cfg.reward.balance_weight,若其为 0 则回退 1.0 以保证 on≠off)。
    两支共用同一目标场与求解结果评估口径,差异仅在加权目标的权重(连续,非 if 分支)。

    target_source: 目标输入**源选择器**(仅当调用方未显式传 targets 时生效),
        取值 {"scanned_map","oracle_gt"}:
          - "scanned_map"(默认,**现实基线**)→ targets = default_detections(cfg),
            即从 Leader 扫描的先验概率图抽取的离散候选检测点(估计,非 GT)。
          - "oracle_gt"(GT 上界参照)→ targets = default_targets(cfg),真值目标点。
        二者皆修士基线(评估输入选择,非 if master/doctor 分支)。
        若调用方显式传了 targets,沿用之,target_source 不覆盖。

    solvers: 选 {"greedy","ortools_vrp","exact_milp"} 子集;默认全跑。
    返回:
        {
          "n_targets": int,
          "target_source": str,                     # 实际使用的输入源标记
          "balance_off": {solver: summary, ...},   # balance_weight=0
          "balance_on":  {solver: summary, ...},   # balance_weight>0
          "balance_on_weight": float,
        }
    每个 summary 含 max/total/fairness_value/inference_time_s/objective/skipped。

    注:显式传 targets 时,target_source 仍以传入值原样记录在返回 dict(标注口径用),
        但 targets 本身不被其覆盖。
    """
    if targets is None:
        if target_source == "scanned_map":
            targets = default_detections(cfg)
        elif target_source == "oracle_gt":
            targets = default_targets(cfg)
        else:
            raise ValueError(
                f"未知 target_source {target_source!r};取值 {{'scanned_map','oracle_gt'}}"
            )

    if solvers is None:
        solvers = ["greedy", "ortools_vrp", "exact_milp"]

    # 求解器名 → 可调用(延迟 import 各基线,避免在缺求解器时炸 import)。
    def _get_solver(sname: str) -> Callable:
        if sname == "greedy":
            from msim.baselines.greedy import solve_greedy

            return solve_greedy
        if sname == "ortools_vrp":
            from msim.baselines.ortools_vrp import solve_minmax_vrp

            return solve_minmax_vrp
        if sname == "exact_milp":
            from msim.baselines.exact_milp import solve_exact_minmax

            return solve_exact_minmax
        raise ValueError(f"未知求解器 {sname!r}")

    if balance_on_weight is None:
        balance_on_weight = cfg.reward.balance_weight or 1.0

    cfg_off = _with_balance_weight(cfg, 0.0)
    cfg_on = _with_balance_weight(cfg, balance_on_weight)

    result: dict = {
        "n_targets": len(targets),
        "target_source": target_source,
        "balance_on_weight": float(balance_on_weight),
        "balance_off": {},
        "balance_on": {},
    }
    for sname in solvers:
        solver = _get_solver(sname)
        result["balance_off"][sname] = evaluate_solver(
            sname, solver, cfg_off, targets, fairness=fairness
        )
        result["balance_on"][sname] = evaluate_solver(
            sname, solver, cfg_on, targets, fairness=fairness
        )
    return result
