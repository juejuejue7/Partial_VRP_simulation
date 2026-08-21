"""[第3层] 二段式全局VRP 基线(D19)。

    阶段1  Leader 走完整条测线,探明全部目标        [0, t_survey)
           Follower 在起点待机(STATUS_IDLE)
    阶段2  用完整目标表解**一次**全局 min-max VRP  [t_survey, t_finish)
           两台 Follower 各自开环执行,不重规划、不通信

与 `mission.py` 的唯一差别就是**探査与観測串行**。运动学、场景、速度、停留
全部复用同一份 `MissionConfig`,不新写任何运动学 —— Leader / Follower 直接用
`agents.py`,VRP 直接用 `planner.solve_minmax_vrp_from`。

⚠ 阶段1 走完时全部目标必然已被探明:观测窗口宽 100 m == 场地 East 跨度,
  目标 North 最大 442.1 < 测线终点 500 ⇒ 每个目标都进过窗口。
  这不是假设,是 `test_survey_phase_detects_every_target` 钉住的事实。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from msim.contracts.types import FLOAT

from .agents import Follower, Leader, QueueItem
from .contracts.mission import STATUS_DWELL, STATUS_IDLE, STATUS_TRANSIT
from .contracts.twophase import TwoPhaseConfig
from .planner import solve_minmax_vrp_from
from .world import MothraWorld, build_mothra_world

__all__ = ["TwoPhaseResult", "run_twophase", "save_twophase_result",
           "load_twophase_result"]


# ======================================================================
# 结果
# ======================================================================
@dataclass
class TwoPhaseResult:
    """一次二段式基线仿真的产物。字段命名与 `MissionResult` / `LawnmowerResult` 平行。"""
    cfg: TwoPhaseConfig
    t_s: np.ndarray                      # (T,)
    leader_north_m: np.ndarray           # (T,)
    vehicle_pos: np.ndarray              # (T,V,2) [x_N, y_E]
    vehicle_status: np.ndarray           # (T,V) int
    vehicle_distance_m: np.ndarray       # (T,V)
    visited_count: np.ndarray            # (T,)
    visit_time_s: np.ndarray             # (n_targets,) 完成観測的时刻;未访问为 nan
    visit_by: np.ndarray                 # (n_targets,) 由哪台车観測;未访问 -1
    waypoint_ids: np.ndarray             # (n_targets,)
    routes: List[List[int]]              # 阶段2 全局 VRP 的解(每台一条,targets 下标)
    t_survey_s: float                    # 阶段1 结束时刻 = Leader 走完测线
    n_vrp_solves: int                    # 全程解了几次 VRP —— 二段式必须恰为 1
    meta: Dict[str, Any] = field(default_factory=dict)

    # --- 与另外两个场景平行的口径 ----------------------------------------
    @property
    def n_targets(self) -> int:
        return int(self.waypoint_ids.size)

    @property
    def visited_mask(self) -> np.ndarray:
        return ~np.isnan(self.visit_time_s)

    @property
    def coverage(self) -> float:
        return float(self.visited_mask.mean()) if self.n_targets else 0.0

    @property
    def missed_wp_ids(self) -> List[int]:
        return [int(w) for w in self.waypoint_ids[~self.visited_mask]]

    @property
    def per_vehicle_distance_m(self) -> np.ndarray:
        return self.vehicle_distance_m[-1]

    @property
    def per_vehicle_visits(self) -> np.ndarray:
        return np.asarray([int((self.visit_by == i).sum())
                           for i in range(self.cfg.n_vehicles)], dtype=int)

    @property
    def t_finish_s(self) -> float:
        """**実測**終了時刻 = max(阶段1 结束, 最后一个目标完成観測)。不检查覆盖率。

        串行 ⇒ 正常情况下必然是后者。口径与 `MissionResult.t_finish_s` 一致。
        """
        if self.visited_mask.any():
            return float(max(self.t_survey_s, np.nanmax(self.visit_time_s)))
        return float(self.t_survey_s)

    @property
    def t_complete_s(self) -> float:
        """任务**完成**时刻 = 全覆盖前提下的 `t_finish_s`;未全覆盖为 nan。"""
        if self.coverage < 1.0:
            return float("nan")
        return self.t_finish_s

    @property
    def t_observe_s(self) -> float:
        """阶段2 的净时长 = 収工 − 阶段1 结束。**由串行结构解析确定的量**,非估计。"""
        return self.t_finish_s - self.t_survey_s

    @property
    def active_mask(self) -> np.ndarray:
        """稼働率一类指标的**有効区間**:t <= t_finish_s(与另外两个场景同一口径)。"""
        tf = self.t_finish_s
        if not np.isfinite(tf):
            return np.ones_like(self.t_s, dtype=bool)
        return self.t_s <= tf + 1e-9

    def duty_frac(self, status: int) -> float:
        st = self.vehicle_status[self.active_mask]
        return float((st == int(status)).mean()) if st.size else 0.0

    @property
    def busy_time_s(self) -> np.ndarray:
        """各机稼働時間 (V,) = 非 IDLE 的拍数 x dt。⑥ 負荷均衡族的输入。"""
        dt = float(self.meta.get("dt_s", self.cfg.dt_s))
        return (self.vehicle_status != STATUS_IDLE).sum(axis=0).astype(FLOAT) * dt

    def summary(self) -> Dict[str, Any]:
        from .metrics_util import balance_metrics       # 局部 import,避免环依赖

        d = self.per_vehicle_distance_m
        vt = self.visit_time_s[self.visited_mask]
        n_vis = int(self.visited_mask.sum())
        idle = self.duty_frac(STATUS_IDLE)
        out: Dict[str, Any] = {
            "scenario": "twophase",
            "t_complete_s": self.t_complete_s,
            "t_finish_s": self.t_finish_s,
            "duration_s": float(self.t_s[-1]),
            "t_last_observation_s": (float(np.nanmax(self.visit_time_s))
                                     if self.visited_mask.any() else float("nan")),
            "coverage": self.coverage,
            "visited": n_vis,
            "n_targets": self.n_targets,
            "missed_wp_ids": self.missed_wp_ids,
            # --- 二段式特有的阶段拆分(本场景的诊断,不进跨方法表) ---------
            "t_survey_s": self.t_survey_s,
            "t_observe_s": self.t_observe_s,
            "n_vrp_solves": int(self.n_vrp_solves),
            "route_counts": [len(r) for r in self.routes],
            "n_vehicles": int(self.cfg.n_vehicles),
            # --- ① 時間効率(跨方法) --------------------------------------
            "t_observation_mean_s": float(vt.mean()) if vt.size else float("nan"),
            "t_observation_median_s": float(np.median(vt)) if vt.size else float("nan"),
            "t_observation_p90_s": (float(np.percentile(vt, 90)) if vt.size
                                    else float("nan")),
            "t_per_target_s": (self.t_finish_s / n_vis if n_vis else float("nan")),
            # --- ② 稼働率 -------------------------------------------------
            "duty_transit_frac": self.duty_frac(STATUS_TRANSIT),
            "duty_dwell_frac": self.duty_frac(STATUS_DWELL),
            "duty_idle_frac": idle,
            "duty_productive_frac": 1.0 - idle,
            "leader_hold_frac": 0.0,     # Leader 不停船等队友(没有队友要等)
            # --- ③ 移動効率 -----------------------------------------------
            "per_follower_distance_m": [float(x) for x in d],
            "per_vehicle_visits": [int(x) for x in self.per_vehicle_visits],
            "max_distance_m": float(d.max()) if d.size else 0.0,
            "total_distance_m": float(d.sum()),
            "leader_distance_m": float(self.leader_north_m[-1]),
            "fleet_distance_m": float(self.leader_north_m[-1]) + float(d.sum()),
            "shadow_error_max_m": 0.0,   # 没有影子模型(不需要:全局路线离线算好)
            "timed_out": bool(self.meta.get("timed_out", False)),
            # --- ④ 通信コスト:**真的是 0,不是"不适用"** --------------------
            # 全局路线在下水前/母船上算好一次性装载,任务期间不需要任何声学报文。
            # 这是二段式**相对本文提案的优势**,记 0 而不是 —— 对照组占的便宜
            # 要如实记,否则表格会把提案的 430 条报文说成"别人也有的开销"。
            "n_comm_messages": 0,
            "channel_duty_frac": 0.0,
            "messages_per_target": 0.0,
            # --- ⑤ 計画品質:全局只规划一次 ⇒ 这一族对它是**最好的成绩** -----
            # 每个目标恰好被下发一次、从不改派 ⇒ 序列利用率 100%。
            # 同样是对照组的优势,不许藏进 —— 里。
            "n_plan_rounds": int(self.n_vrp_solves),
            "pool_size_mean": float(self.n_targets),   # 一次面对全部目标
            "wp_issue_total": int(sum(len(r) for r in self.routes)),
            "reassignment_count": 0,
            "reassigned_wp_count": 0,
            "solve_wall_mean_s": float(self.meta.get("solve_wall_s", 0.0)),
        }
        issued = out["wp_issue_total"]
        out["wp_issues_per_target"] = (issued / self.n_targets
                                       if self.n_targets else 0.0)
        out["sequence_utilisation"] = (self.n_targets / issued if issued
                                       else float("nan"))
        out["distance_per_target_m"] = (out["fleet_distance_m"] / n_vis if n_vis
                                        else float("nan"))
        # --- ⑥ 負荷均衡(D19) ---------------------------------------------
        out.update(balance_metrics(d, self.busy_time_s))
        return out


# ======================================================================
# 主循环
# ======================================================================
def run_twophase(cfg: Optional[TwoPhaseConfig] = None,
                 world: Optional[MothraWorld] = None,
                 *, verbose: bool = False) -> TwoPhaseResult:
    """跑一次二段式全局VRP 基线。纯确定性(除 ortools 元启发式的时间预算外)。"""
    cfg = cfg or TwoPhaseConfig()
    mw = world or build_mothra_world(cfg.sim)
    ds = mw.dataset
    n_t = ds.n
    dt = cfg.dt_s
    base = cfg.base

    leader = Leader(base, track_end_north_m=mw.world.x_max_m)
    vehicles = [Follower(i, s, cfg.vehicle_cfg)
                for i, s in enumerate(cfg.vehicle_starts_ned)]

    # ---------- 阶段2 的全局 VRP:整个仿真里**只解这一次** ----------------
    # 起点 = 各机在阶段1 全程待机的位置 = 它们的起点(口径 T1,见契约)。
    # 解在这里(而非阶段1 结束那一拍)是因为串行结构下起点已知且不会变,
    # 结果逐位相同;`n_vrp_solves` 会把"只解一次"这件事记下来供测试断言。
    starts = [np.asarray(s, dtype=FLOAT).reshape(2) for s in cfg.vehicle_starts_ned]
    _t0 = time.perf_counter()
    routes, solver_used = solve_minmax_vrp_from(
        starts, list(ds.targets_ned),
        time_limit_s=cfg.vrp_time_limit_s,
        span_coefficient=base.vrp_span_coefficient,
        solver=cfg.solver)
    # ⚠ 挂钟量,宿主机相关,**不进入仿真时间轴**(与 D15 对 solve_wall_s 的定性一致)
    solve_wall_s = time.perf_counter() - _t0
    n_vrp_solves = 1

    t_survey = mw.world.x_max_m / cfg.leader_speed_mps

    visit_time = np.full(n_t, np.nan, dtype=FLOAT)
    visit_by = np.full(n_t, -1, dtype=np.int32)
    visited = [False] * n_t
    id_to_idx = {int(w): j for j, w in enumerate(ds.waypoint_id)}

    rec_t: List[float] = []
    rec_ln: List[float] = []
    rec_pos: List[np.ndarray] = []
    rec_st: List[List[int]] = []
    rec_dist: List[List[float]] = []
    rec_vis: List[int] = []

    dispatched = False
    timed_out = False
    t = 0.0
    while t <= cfg.max_mission_time_s + 1e-9:
        # --- 阶段切换:Leader 走完测线的那一拍,把全局路线一次性下发 -------
        # ⚠ 严格串行的落点就在这里:在此之前 vehicles 的队列全空 ⇒ STATUS_IDLE,
        #   不可能有任何一拍是"Leader 在走 且 某台在観測"。
        if not dispatched and leader.finished:
            for i, r in enumerate(routes):
                vehicles[i].assign([
                    QueueItem(wp_id=int(ds.waypoint_id[j]),
                              point=np.asarray(ds.targets_ned[j], dtype=FLOAT).copy(),
                              target_idx=int(j))
                    for j in r])
            dispatched = True
            if verbose:
                print(f"[twophase] t={t:.1f}s 阶段1 结束(探査 {t:.1f}s),"
                      f"下发全局路线 {[len(r) for r in routes]} 点 (solver={solver_used})")

        # --- 记录 ---------------------------------------------------------
        rec_t.append(t)
        rec_ln.append(leader.north)
        rec_pos.append(np.stack([v.pos.copy() for v in vehicles]))
        rec_st.append([int(v.status) for v in vehicles])
        rec_dist.append([float(v.distance_m) for v in vehicles])
        rec_vis.append(int(np.sum(visited)))

        # --- 推进 ---------------------------------------------------------
        leader.step(dt)                       # 阶段1 匀速;走完后 step 自动空转
        for v in vehicles:
            done = v.step(dt)
            for wp in done:
                j = id_to_idx.get(int(wp))
                if j is not None and not visited[j]:
                    visited[j] = True
                    visit_time[j] = t + dt
                    visit_by[j] = v.id

        t = round(t + dt, 9)

        # --- 终止:阶段1 已过 且 全部队列跑空 且 过了收尾余量 --------------
        if (dispatched and all(not v.queue for v in vehicles)
                and t > t_survey + base.settle_time_s):
            break
    else:
        timed_out = True

    if not dispatched:
        timed_out = True      # Leader 连测线都没走完 ⇒ 阶段2 从未开始

    return TwoPhaseResult(
        cfg=cfg,
        t_s=np.asarray(rec_t, dtype=FLOAT),
        leader_north_m=np.asarray(rec_ln, dtype=FLOAT),
        vehicle_pos=np.asarray(rec_pos, dtype=FLOAT),
        vehicle_status=np.asarray(rec_st, dtype=np.int8),
        vehicle_distance_m=np.asarray(rec_dist, dtype=FLOAT),
        visited_count=np.asarray(rec_vis, dtype=np.int32),
        visit_time_s=visit_time, visit_by=visit_by,
        waypoint_ids=np.asarray(ds.waypoint_id, dtype=np.int32),
        routes=[[int(j) for j in r] for r in routes],
        t_survey_s=float(t_survey),
        n_vrp_solves=n_vrp_solves,
        meta={"dt_s": dt, "solver": solver_used, "timed_out": timed_out,
              "solve_wall_s": float(solve_wall_s),
              "survey_line_length_m": float(mw.world.x_max_m),
              "route_lengths_m": [
                  float(sum(np.linalg.norm(ds.targets_ned[r[k]]
                                           - (starts[i] if k == 0
                                              else ds.targets_ned[r[k - 1]]))
                            for k in range(len(r))))
                  for i, r in enumerate(routes)]},
    )


# ======================================================================
# 落盘 / 回读
# ======================================================================
def save_twophase_result(res: TwoPhaseResult, npz_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez(npz_path,
             t_s=res.t_s, leader_north_m=res.leader_north_m,
             vehicle_pos=res.vehicle_pos, vehicle_status=res.vehicle_status,
             vehicle_distance_m=res.vehicle_distance_m,
             visited_count=res.visited_count, visit_time_s=res.visit_time_s,
             visit_by=res.visit_by, waypoint_ids=res.waypoint_ids,
             meta_json=np.array(json.dumps(res.meta, ensure_ascii=False)))
    base = os.path.splitext(npz_path)[0]
    with open(base + "_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": res.summary(),
                   "routes": res.routes,
                   "t_survey_s": res.t_survey_s,
                   "t_observe_s": res.t_observe_s}, f,
                  ensure_ascii=False, indent=2)


def load_twophase_result(npz_path: str) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files if k != "meta_json"}
        out["meta"] = json.loads(str(d["meta_json"]))
    return out
