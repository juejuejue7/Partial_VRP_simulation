"""[第3层] Lawnmower 全覆盖基线的主循环 —— 3 台 Follower 分区往复扫描。

规格见 `contracts/lawnmower.py` 的模块 docstring。本文件把 world(第1层)与
agents(第2层)串成时间线,与 `mission.py` **共用同一个 `Follower` 状态机**、
同一套 dt 网格、同一套访问记账口径 —— 两个场景的时间才可比。

对照关系(逐条对齐 `mission.py`):
    Leader / 观测窗口 / 请求 / VRP 规划 / 投影补点   → 本基线**全部没有**
    Follower 匀速趋近 + 到点停 `dwell_time_s`        → 完全相同(同一个类)
    `visit_time[j] = t + dt`(拍末时刻,dt 量化)     → 完全相同
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from msim.contracts.types import FLOAT

from .agents import Follower, QueueItem
from .contracts.lawnmower import LawnmowerConfig
from .contracts.mission import STATUS_NAMES, WP_PROJECTION
from .world import MothraWorld, build_mothra_world

__all__ = ["LawnmowerResult", "run_lawnmower", "strip_bounds", "lane_easts",
           "build_vehicle_plans", "save_lawn_result", "load_lawn_result"]


# ======================================================================
# 测线几何(纯确定量,可单独测)
# ======================================================================
def strip_bounds(east_max_m: float, n_vehicles: int) -> List[Tuple[float, float]]:
    """把 East 跨度均分成 n 条互不重叠的条带,每台车一条。"""
    w = float(east_max_m) / int(n_vehicles)
    return [(i * w, (i + 1) * w) for i in range(int(n_vehicles))]


def lane_easts(strip_lo: float, strip_hi: float, swath_width_m: float) -> np.ndarray:
    """带内测线的 East 坐标(均匀铺开,间距 ≤ 幅宽 ⇒ 带内无缝隙)。

    条数 = ceil(带宽 / 幅宽);间距 = 带宽 / 条数;测线落在各自子段的中线上。
    这样最外侧测线的幅宽只溢出带外 (幅宽-间距)/2,不会大片侵占邻带。
    """
    width = float(strip_hi) - float(strip_lo)
    n = max(1, int(math.ceil(width / float(swath_width_m) - 1e-9)))
    spacing = width / n
    return float(strip_lo) + (np.arange(n, dtype=FLOAT) + 0.5) * spacing


def _segments(lanes: Sequence[float], north_lo: float, north_hi: float,
              boustrophedon: bool) -> List[Tuple[np.ndarray, np.ndarray, bool]]:
    """航迹分段:(起点, 终点, 是否为扫描段)。非扫描段 = 换线横移 / 空回航。"""
    segs: List[Tuple[np.ndarray, np.ndarray, bool]] = []

    def P(n: float, e: float) -> np.ndarray:
        return np.array([float(n), float(e)], dtype=FLOAT)

    for i, e in enumerate(lanes):
        if boustrophedon and i % 2 == 1:
            a, b = north_hi, north_lo
        else:
            a, b = north_lo, north_hi
        segs.append((P(a, e), P(b, e), True))
        if not boustrophedon:
            segs.append((P(b, e), P(a, e), False))       # 空回航(不扫描)
            b = a
        if i + 1 < len(lanes):
            segs.append((P(b, e), P(b, lanes[i + 1]), False))   # 换线横移
    return segs


def build_vehicle_plans(mw: MothraWorld, cfg: LawnmowerConfig
                        ) -> Tuple[List[List[QueueItem]], List[np.ndarray],
                                   List[np.ndarray], List[int]]:
    """为每台车生成 waypoint 队列。

    Returns:
        queues       : 每台的 `QueueItem` 序列(真实目标点 + 标为 WP_PROJECTION 的拐点)
        lanes        : 每台的测线 East 坐标
        starts       : 每台的起点 (2,) = 第一条测线的南端
        owner        : 每个目标归属哪台车(按 East 落在哪条带,长度 n_targets)

    不变量(违反即抛):每个目标恰好被安排**一次**停留 —— 既不会漏(带内无缝隙),
    也不会因相邻测线幅宽搭接而被排两次。
    """
    ds = mw.dataset
    e_max = float(mw.world.y_max_m)
    n_lo, n_hi = 0.0, float(mw.world.x_max_m)
    half = 0.5 * float(cfg.swath_width_m)

    bounds = strip_bounds(e_max, cfg.n_vehicles)
    tgt_n = ds.targets_ned[:, 0].astype(FLOAT)
    tgt_e = ds.targets_ned[:, 1].astype(FLOAT)

    # 目标 → 车:按 East 落在哪条带。归属唯一,不靠先到先得。
    strip_w = e_max / cfg.n_vehicles
    owner = np.clip((tgt_e / strip_w).astype(int), 0, cfg.n_vehicles - 1).tolist()

    queues: List[List[QueueItem]] = []
    lanes_all: List[np.ndarray] = []
    starts: List[np.ndarray] = []
    scheduled: List[int] = []

    for v in range(cfg.n_vehicles):
        lanes = lane_easts(bounds[v][0], bounds[v][1], cfg.swath_width_m)
        lanes_all.append(lanes)
        starts.append(np.array([n_lo, float(lanes[0])], dtype=FLOAT))

        mine = [j for j in range(ds.n) if owner[j] == v]
        done: set = set()
        q: List[QueueItem] = []
        for p0, p1, scanning in _segments(lanes, n_lo, n_hi, cfg.boustrophedon):
            if scanning:
                e_lane = float(p0[1])
                lo_n, hi_n = (p0[0], p1[0]) if p0[0] <= p1[0] else (p1[0], p0[0])
                hits = [j for j in mine
                        if j not in done
                        and abs(tgt_e[j] - e_lane) <= half + 1e-9
                        and lo_n - 1e-9 <= tgt_n[j] <= hi_n + 1e-9]
                # 沿行进方向排序 —— 车不会为了目标掉头
                hits.sort(key=lambda j: abs(float(tgt_n[j]) - float(p0[0])))
                for j in hits:
                    done.add(j)
                    # ⚠ 停留点在**测线上**,North 对齐目标,East 保持 lane ——
                    #   车不离开测线去目标正上方(见契约"覆盖判据")。
                    q.append(QueueItem(wp_id=int(ds.waypoint_id[j]),
                                       point=np.array([tgt_n[j], e_lane], dtype=FLOAT),
                                       target_idx=j))
            q.append(QueueItem(wp_id=WP_PROJECTION, point=p1.copy(), target_idx=-1))

        missed = sorted(set(mine) - done)
        if missed:
            raise ValueError(
                f"车 {v} 的带内有目标没被任何测线幅宽覆盖: "
                f"wp {[int(ds.waypoint_id[j]) for j in missed]} —— "
                f"测线间距 {float(lanes[1] - lanes[0]) if lanes.size > 1 else 0.0:.3f} m "
                f"与幅宽 {cfg.swath_width_m} m 不自洽")
        scheduled.extend(done)
        queues.append(q)

    if sorted(scheduled) != list(range(ds.n)):
        raise ValueError(f"目标安排不是恰好一次: 安排了 {len(scheduled)} 项 / "
                         f"目标 {ds.n} 个")
    return queues, lanes_all, starts, owner


# ======================================================================
# 结果
# ======================================================================
@dataclass
class LawnmowerResult:
    """一次 lawnmower 基线仿真的产物。字段命名与 `MissionResult` 平行。"""
    cfg: LawnmowerConfig
    t_s: np.ndarray                      # (T,)
    vehicle_pos: np.ndarray              # (T,V,2) [x_N, y_E]
    vehicle_status: np.ndarray           # (T,V) int
    vehicle_distance_m: np.ndarray       # (T,V)
    visited_count: np.ndarray            # (T,)
    visit_time_s: np.ndarray             # (n_targets,) 完成観測的时刻;未访问为 nan
    visit_by: np.ndarray                 # (n_targets,) 由哪台车観測;未访问 -1
    waypoint_ids: np.ndarray             # (n_targets,)
    finish_time_s: np.ndarray            # (V,) 各车走完自己全部测线的时刻
    meta: Dict[str, Any] = field(default_factory=dict)

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
    def t_complete_s(self) -> float:
        """任务完成时刻 = 最后一台车走完自己最后一条测线的时刻(契约"完成时刻的口径")。

        ⚠ 不是"最后一个目标被観測的时刻":全覆盖扫描要**走完**才能宣称扫完,
          哪怕最后一段测线上一个目标都没有。
        """
        return float(np.nanmax(self.finish_time_s))

    @property
    def busy_time_s(self) -> np.ndarray:
        """各车稼働時間 (V,) = 非 IDLE 的拍数 x dt。⑥ 負荷均衡族的输入(D19)。"""
        from .contracts.mission import STATUS_IDLE
        dt = float(self.meta.get("dt_s", self.cfg.dt_s))
        return (self.vehicle_status != STATUS_IDLE).sum(axis=0).astype(float) * dt

    def summary(self) -> Dict[str, Any]:
        from .metrics_util import balance_metrics

        d = self.per_vehicle_distance_m
        out = {
            "scenario": "lawnmower",
            "t_complete_s": self.t_complete_s,
            "duration_s": float(self.t_s[-1]),
            "t_last_observation_s": (float(np.nanmax(self.visit_time_s))
                                     if self.visited_mask.any() else float("nan")),
            "coverage": self.coverage,
            "visited": int(self.visited_mask.sum()),
            "n_targets": self.n_targets,
            "missed_wp_ids": self.missed_wp_ids,
            "n_vehicles": int(self.cfg.n_vehicles),
            "swath_width_m": float(self.cfg.swath_width_m),
            "n_lanes_per_vehicle": [len(x) for x in self.meta["lane_easts"]],
            "lane_spacing_m": float(self.meta["lane_spacing_m"]),
            "survey_line_length_m": float(self.meta["survey_line_length_m"]),
            "per_vehicle_distance_m": [float(x) for x in d],
            "per_vehicle_visits": [int(x) for x in self.per_vehicle_visits],
            "finish_time_s": [float(x) for x in self.finish_time_s],
            "max_distance_m": float(d.max()) if d.size else 0.0,
            "total_distance_m": float(d.sum()),
        }
        # ⑥ 負荷均衡(D19):与另外两个场景共用同一份实现,口径逐位一致。
        out.update(balance_metrics(d, self.busy_time_s))
        return out


# ======================================================================
# 主循环
# ======================================================================
def run_lawnmower(cfg: Optional[LawnmowerConfig] = None,
                  world: Optional[MothraWorld] = None,
                  *, verbose: bool = False) -> LawnmowerResult:
    """跑一次 lawnmower 全覆盖基线。纯确定性,无 RNG、无求解器。"""
    cfg = cfg or LawnmowerConfig()
    mw = world or build_mothra_world(cfg.sim)
    ds = mw.dataset
    n_t = ds.n
    V = cfg.n_vehicles
    dt = cfg.dt_s

    queues, lanes_all, starts, owner = build_vehicle_plans(mw, cfg)
    vcfg = cfg.vehicle_cfg                      # 运动学 = VRP 场景那一份
    vehicles = [Follower(i, tuple(starts[i]), vcfg) for i in range(V)]
    for i, v in enumerate(vehicles):
        v.assign(queues[i])

    visited = [False] * n_t
    visit_time = np.full(n_t, np.nan, dtype=FLOAT)
    visit_by = np.full(n_t, -1, dtype=np.int16)
    id_to_idx = {int(w): j for j, w in enumerate(ds.waypoint_id)}
    finish_t = np.full(V, np.nan, dtype=FLOAT)

    rec_t, rec_pos, rec_st, rec_dist, rec_vis = [], [], [], [], []

    t = 0.0
    while True:
        # --- 记录(与 mission.py 同口径:先记当前状态,再推进) -----------
        rec_t.append(t)
        rec_pos.append([v.pos.copy() for v in vehicles])
        rec_st.append([v.status for v in vehicles])
        rec_dist.append([v.distance_m for v in vehicles])
        rec_vis.append(int(np.sum(visited)))

        for i, v in enumerate(vehicles):
            if not v.queue and np.isnan(finish_t[i]):
                finish_t[i] = t                 # 队列刚空的这一拍即该车完工时刻

        if all(not v.queue for v in vehicles):
            break
        if t > cfg.max_mission_time_s:
            raise RuntimeError(
                f"lawnmower 超过 max_mission_time_s={cfg.max_mission_time_s} s 仍未跑完;"
                f" 已観測 {int(np.sum(visited))}/{n_t}。把 base.max_mission_time_s 调大。")

        # --- 推进 --------------------------------------------------------
        for i, v in enumerate(vehicles):
            for wid in v.step(dt):
                j = id_to_idx[int(wid)]
                if not visited[j]:
                    visited[j] = True
                    visit_time[j] = t + dt      # ← 与 mission.py 逐字相同
                    visit_by[j] = i
                    if verbose:
                        print(f"  t={t + dt:7.1f}s  V{i} 観測 wp{wid}"
                              f"  ({int(np.sum(visited))}/{n_t})")
        t += dt

    spacing = (float(lanes_all[0][1] - lanes_all[0][0]) if lanes_all[0].size > 1
               else float(mw.world.y_max_m) / cfg.n_vehicles)
    meta: Dict[str, Any] = {
        "world": mw.meta,
        "dt_s": dt,
        "n_vehicles": V,
        "swath_width_m": float(cfg.swath_width_m),
        "boustrophedon": bool(cfg.boustrophedon),
        "lane_easts": [[float(x) for x in a] for a in lanes_all],
        "lane_spacing_m": spacing,
        "survey_line_length_m": float(mw.world.x_max_m),
        "strips_east_m": [list(b) for b in strip_bounds(mw.world.y_max_m, V)],
        "starts_ned": [[float(p[0]), float(p[1])] for p in starts],
        "target_owner": [int(o) for o in owner],
        "follower_speed_mps": float(cfg.follower_speed_mps),
        "dwell_time_s": float(cfg.dwell_time_s),
    }
    return LawnmowerResult(
        cfg=cfg,
        t_s=np.asarray(rec_t, dtype=FLOAT),
        vehicle_pos=np.asarray(rec_pos, dtype=FLOAT),
        vehicle_status=np.asarray(rec_st, dtype=np.int8),
        vehicle_distance_m=np.asarray(rec_dist, dtype=FLOAT),
        visited_count=np.asarray(rec_vis, dtype=np.int32),
        visit_time_s=visit_time, visit_by=visit_by,
        waypoint_ids=ds.waypoint_id.copy(),
        finish_time_s=finish_t, meta=meta)


# ======================================================================
# 落盘 / 回读(与 mission.py 同形式)
# ======================================================================
def save_lawn_result(res: LawnmowerResult, npz_path: str, *,
                     also_csv: bool = True) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez(npz_path,
             t_s=res.t_s, vehicle_pos=res.vehicle_pos,
             vehicle_status=res.vehicle_status,
             vehicle_distance_m=res.vehicle_distance_m,
             visited_count=res.visited_count, visit_time_s=res.visit_time_s,
             visit_by=res.visit_by, waypoint_ids=res.waypoint_ids,
             finish_time_s=res.finish_time_s,
             meta_json=np.array(json.dumps(res.meta, ensure_ascii=False)))

    base = os.path.splitext(npz_path)[0]
    with open(base + "_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": res.summary(), "meta": res.meta},
                  f, ensure_ascii=False, indent=2)

    if also_csv:
        _write_timeline_csv(res, base + "_timeline.csv")


def _write_timeline_csv(res: LawnmowerResult, path: str) -> None:
    V = res.vehicle_pos.shape[1]
    cols = ["t_s", "visited_count"]
    for i in range(V):
        cols += [f"v{i}_north_m", f"v{i}_east_m", f"v{i}_status", f"v{i}_dist_m"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for k in range(len(res.t_s)):
            row = [f"{res.t_s[k]:.2f}", int(res.visited_count[k])]
            for i in range(V):
                row += [f"{res.vehicle_pos[k, i, 0]:.3f}",
                        f"{res.vehicle_pos[k, i, 1]:.3f}",
                        STATUS_NAMES[int(res.vehicle_status[k, i])],
                        f"{res.vehicle_distance_m[k, i]:.2f}"]
            w.writerow(row)


def load_lawn_result(npz_path: str) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files if k != "meta_json"}
        out["meta"] = json.loads(str(d["meta_json"]))
    return out
