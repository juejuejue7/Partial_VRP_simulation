"""[第3层] 任务仿真主循环 —— Leader 扫描 + 双 Follower 声学握手 + 联合分区域 VRP。

规格见 `contracts/mission.py` 的模块 docstring(§1–§7)。本文件负责把
world(第1层)/ planner(第2层)/ agents(第2层)串成一条时间线,并按规格 §7
**逐时刻**记录 waypoint 序列与各 Follower 位置。

时间推进:定步长 dt(默认 0.5 s)。声学报文建模成**在途事件**:每条报文在发出时刻
入队、飞 `acoustic_hop_s` 秒后在到达时刻被处理。一轮联合规划 = 2N+1 条报文,
见 `contracts/mission.py` §4 的时序图(D15)。

⚠ 本文件里"广播落地"是整个模型的对齐点:真机 `Follower.assign()` 与影子
  `LeaderTracker.on_assignment()` **必须在同一拍**调用。否则影子会提前一跳沿新序列
  行进,`shadow_error` 凭空变大,"承诺队列解析计算是确定量"这条结论就不再成立
  (CLAUDE.md 硬纪律 4)。D15 之前的版本在这里是错的。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from msim.contracts.types import FLOAT

from .agents import Follower, Leader, QueueItem
from .contracts.config import mothra_window_config
from msim.geometry.window import window_contains_world

from .contracts.mission import (HOLD_ENDANGERED, HOLD_LAGGING, HOLD_NONE,
                                HOLD_TURNAROUND,
                                MSG_BROADCAST, MSG_POLL_REP, MSG_POLL_REQ,
                                NODE_ALL, NODE_LEADER, STATUS_DWELL, STATUS_IDLE,
                                STATUS_NAMES, STATUS_TRANSIT, WP_NONE,
                                WP_PROJECTION, Assignment, FollowerRequest,
                                MissionConfig, PlanRound)
from .metrics_util import balance_metrics
from .planner import available_pool, plan_round
from .tracking import LeaderTracker
from .windows import along_back_m, leader_track, window_north_span
from .world import MothraWorld, build_mothra_world

__all__ = ["MissionResult", "run_mission", "save_result", "load_result",
           "feasibility_estimate"]


# ======================================================================
# 事前可行性判据 —— 覆盖率上不去时,先看是不是根本做不到
# ======================================================================
def feasibility_estimate(world: MothraWorld, cfg: MissionConfig) -> Dict[str, Any]:
    """任务时间的**参照量**,以及"Leader 必须停船等队友吗"。

    两个量:
      `t_leader_min`  = 测线长 / Leader 速度。**不停船**时 Leader 走完测线的时刻。
      `t_route_ref`   = 忽略窗口约束、对全部目标解一次全局 min-max VRP,取
          最长单机路线 / Follower 速度 + 该机的停留时间。

    ⚠ **"参照"不是"下界"(D19 更名的理由,CLAUDE.md 硬纪律 4:不 overclaim)。**
      两个方向的偏差同时存在:
        偏乐观 —— 忽略了窗口约束、下发截断、声学时延、往返,这些只会让实际更慢;
        偏悲观 —— `solve_minmax_vrp_from` 均衡的是**最长距离**,而这里却按
                  `距离/v + 点数×dwell` 取最大值。实测切成 [18 点 452.1 m,
                  4 点 451.2 m]:距离几乎相等而时间差 141.8 s(13.1%),
                  说明按**时间**重切能更小 ⇒ 本量可能比真最优**高**。
      故它是**距离均衡最优解对应的时间**,是参照基准,不是可证明的下界。
      这个分叉本身已被 ⑥ 負荷均衡族的 `load_imbalance_{distance,time}_frac`
      量化进对比表(D19),不再只是注释里的一句话。

    ⚠ 它**只对本文提案(局部VRP)有意义**:分子含 `测线长/v_leader`,
      lawnmower 与二段式没有 Leader 测线。跨方法比时间请用
      `t_per_target_s`(见 contracts/metrics.py 的 applies_to)。

    ⚠ "测线长"取 `leader_track` 生成的**实际折线总长**(多车道往复 + 换道横移),
      不是 `world.x_max_m`(North 跨度)—— 场地 East 跨度超过声呐幅宽
      (`cfg.window_width_m`)时会有多条车道,折线总长远大于单纯的 North 跨度;
      两者只在单车道(East 跨度 <= 声呐幅宽,如 Mothra)时相等。

    **D16 之前**(Leader 匀速):`t_route_lb > t_leader_min` ⇒ Leader 走完后窗口冻结在
    最北 `look_back` 米,更南的目标永久错过 ⇒ **无论怎么调节拍都不可能全覆盖**。

    **D16 之后**(Leader 会停船等队友):这条不再是死刑判决。Leader 用自己的 DR 推算
    发现队友跟不上就原地停,窗口不会提前冻结 —— 于是
        **「覆盖率约束」被换成了「时间约束」**:
        任务时间下界 = max(t_leader_min, t_route_lb)
    `wait_required = t_route_lb > t_leader_min` 表示这套参数下 Leader **必然要等**,
    此时实际完成时刻 = 测线长/v_leader + 累计停船时长,`t_leader_min` 只是它的下界。

    仍然是**必要条件**,不是充分条件。
    """
    from .planner import solve_minmax_vrp_from

    ds = world.dataset
    starts = [np.asarray(s, dtype=FLOAT).reshape(2) for s in cfg.follower_starts_ned]
    routes, solver = solve_minmax_vrp_from(starts, list(ds.targets_ned),
                                           time_limit_s=cfg.vrp_time_limit_s,
                                           span_coefficient=cfg.vrp_span_coefficient,
                                           exact_max_targets=cfg.vrp_exact_max_targets,
                                           gls_lambda=cfg.vrp_gls_lambda,
                                           solver=cfg.solver)
    lengths, counts = [], []
    for i, r in enumerate(routes):
        p = starts[i]
        s = 0.0
        for j in r:
            s += float(np.linalg.norm(ds.targets_ned[j] - p))
            p = ds.targets_ned[j]
        lengths.append(s)
        counts.append(len(r))

    k = int(np.argmax([l / cfg.follower_speed_mps + c * cfg.dwell_time_s
                       for l, c in zip(lengths, counts)]))
    t_route_lb = lengths[k] / cfg.follower_speed_mps + counts[k] * cfg.dwell_time_s
    path = leader_track(world.world, cfg.window_width_m)
    track_len_m = float(sum(np.linalg.norm(path[i + 1] - path[i])
                            for i in range(len(path) - 1)))
    t_leader = track_len_m / cfg.leader_speed_mps

    return {
        "t_leader_finish_s": float(t_leader),      # 不停船时的 Leader 完成时刻
        "t_route_reference_s": float(t_route_lb),
        # 等待型 Leader 下的任务时间下界(D16)
        "t_mission_reference_s": float(max(t_leader, t_route_lb)),
        "route_lengths_m": [float(x) for x in lengths],
        "route_counts": [int(c) for c in counts],
        # D16:Leader 会等 ⇒ 覆盖不再受速度比封死;这里报的是"必须等吗"。
        "wait_required": bool(t_route_lb > t_leader),
        "leader_waits": bool(cfg.leader_waits),
        # 匀速 Leader(关掉等待)时仍能全覆盖的最快 Leader 速度
        "max_leader_speed_mps": float(track_len_m / t_route_lb),
        # 只有关掉等待策略时,"速度比"才重新变成硬约束
        "feasible": bool(cfg.leader_waits or t_route_lb <= t_leader),
        # 第一条序列真正落到 Follower 手里的时刻(D15)。下界本身与调度无关,不含它;
        # 但它是这个乐观下界之外**必然**要付的开销,单独列出来免得判据看着偏乐观。
        "t_first_broadcast_s": float(cfg.first_plan_s + cfg.plan_latency_s),
        "plan_latency_s": float(cfg.plan_latency_s),
        "solver": solver,
    }


@dataclass
class MissionResult:
    """一次任务仿真的全部产物。

    时间线数组第 0 维都是时刻(长度 T),第 1 维是 follower(长度 F)。
    """
    cfg: MissionConfig
    t_s: np.ndarray                      # (T,)
    leader_north_m: np.ndarray           # (T,)
    window_north: np.ndarray             # (T,2) [后沿, 前沿]
    follower_pos: np.ndarray             # (T,F,2) [x_N, y_E]  ← 规格 §7
    follower_status: np.ndarray          # (T,F) int
    follower_occupied: np.ndarray        # (T,F) waypoint_id / WP_NONE
    follower_queue: np.ndarray           # (T,F,K) waypoint_id / WP_PROJECTION / WP_NONE ← 规格 §7
    follower_queue_xy: np.ndarray        # (T,F,K,2) 对应点坐标(投影点会随重解移动)
    follower_distance_m: np.ndarray      # (T,F)
    visited_count: np.ndarray            # (T,)
    assignments: List[Assignment]
    requests: List[FollowerRequest]
    # 规划时刻 Leader 对**每台** Follower 位置的推算误差 (t_ref, follower_id, 误差 m)。
    # nav_drift=0 时应当恒为 0 —— 那正是"推算是确定量而非估计"的证据。
    shadow_error: np.ndarray             # (R,3) float
    visit_time_s: np.ndarray             # (n_targets,) 完成观测的时刻;未访问为 nan
    visit_by: np.ndarray                 # (n_targets,) 由哪台 follower 观测;未访问 -1
    waypoint_ids: np.ndarray             # (n_targets,)
    # --- Leader 停船(D16) ----------------------------------------------
    leader_holding: np.ndarray = field(                # (T,) bool
        default_factory=lambda: np.zeros(0, dtype=bool))
    leader_hold_reason: np.ndarray = field(            # (T,) HOLD_* 位标志
        default_factory=lambda: np.zeros(0, dtype=np.int8))
    # 多车道(boustrophedon)起 Leader 的 East/航向不再是常量, 必须单独记录才能
    # 正确还原窗口的旋转几何(见 `vrpsim/viz.py::sweep_times` 如何用它们重建
    # 与仿真主循环逐位一致的 `window_contains_world` 判定)。
    leader_east_m: np.ndarray = field(                 # (T,)
        default_factory=lambda: np.zeros(0, dtype=FLOAT))
    leader_psi: np.ndarray = field(                    # (T,) rad, 从 North 起向 East 为正
        default_factory=lambda: np.zeros(0, dtype=FLOAT))
    # --- 声学通信与规划轮次(D15) --------------------------------------
    plan_rounds: List[PlanRound] = field(default_factory=list)
    # 每条声学报文一行:[t_send_s, t_recv_s, kind, src, dst]。
    # kind ∈ {MSG_POLL_REQ, MSG_POLL_REP, MSG_BROADCAST};src/dst 用 NODE_* 或 follower_id。
    comm_events: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 5), dtype=FLOAT))
    # 每轮规划一行(与 plan_rounds 同序,给 npz 用的规整数值版):
    # [round_idx, t_start, t_plan, t_deliver, pool_size, n_assigned_total,
    #  n_projection, solve_wall_s, forced_final]
    plan_stats: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 9), dtype=FLOAT))
    meta: Dict[str, Any] = field(default_factory=dict)

    # --- 摘要 -----------------------------------------------------------
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
    def per_follower_distance_m(self) -> np.ndarray:
        return self.follower_distance_m[-1]

    @property
    def leader_distance_m(self) -> float:
        """Leader 沿测线走过的**实际路程**。停船不产生位移,故与等待时长无关。

        ⚠ 多车道(boustrophedon)起不能再用 `north[-1]-north[0]` 反推 —— 折线
          往复 + 换道横移的总路程恒 >= North 净位移,二者只在单车道时相等。
          取 `meta['leader_total_distance_m']`(仿真主循环里 `Leader.distance_m`
          的实测累计值);旧存档没有这个键时回退到净位移(与改动前行为一致,
          仅影响单车道场景,数值不变)。
        """
        v = self.meta.get("leader_total_distance_m")
        if v is not None:
            return float(v)
        ln = self.leader_north_m
        return float(ln[-1] - ln[0]) if ln.size else 0.0

    @property
    def fleet_distance_m(self) -> float:
        """**全部 AUV 的航程之和,含 Leader** —— 跨场景比能耗的正确口径。

        ⚠ `total_distance_m` 只统计 Follower。lawnmower 场景没有 Leader,3 台车的和
          就是艦隊総航程;協調探査若只报 Follower 的和,等于白送掉 Leader 的一整条
          测线,对比就不公平了。
        """
        return self.leader_distance_m + float(self.per_follower_distance_m.sum())

    @property
    def per_follower_visits(self) -> np.ndarray:
        f = self.cfg.n_followers
        return np.asarray([int((self.visit_by == i).sum()) for i in range(f)], dtype=int)

    @property
    def t_leader_finish_s(self) -> float:
        """Leader 走完**全部车道**折线的时刻(全域被声呐扫遍的时刻)。

        ⚠ **必须从记录取,不许反推**:D16 起 Leader 会停船等队友,"测线长/v" 已经
          不等于完成时刻;多车道(boustrophedon)起 `leader_north_m` 还会在换道时
          往复,`argmax(ln>=ln.max())` 只会碰到某条车道的**局部**最北点,不是"走完
          全部车道"的时刻。取仿真主循环里 `leader.finished` 首次为真那一拍记的
          `meta['leader_finish_s']`;旧存档没有这个键时回退到旧口径(仅影响
          单车道场景,数值不变)。
        """
        v = self.meta.get("leader_finish_s")
        if v is not None:
            return float(v)
        ln = self.leader_north_m
        return float(self.t_s[int(np.argmax(ln >= float(ln.max()) - 1e-9))])

    @property
    def t_finish_s(self) -> float:
        """**実測**終了時刻 = max(全域扫遍, 最后一个目标完成観測)。**不检查覆盖率**(D18)。

        与 `t_complete_s` 是同一把尺子,唯一区别是漏了目标也照样给数 —— 它回答的是
        「这趟实际什么时候收的工」,而不是「任务是否达成」。

        ⚠ **漏点组的这个数天生偏小**:少做几个目标当然早收工。单独拿它横比会得出
          「漏点方案更快」的假结论,必须与 `visited` / `coverage` 同读,
          或改看对漏点中性的 `t_per_target_s`(见 contracts/metrics.py)。
        ⚠ 一个目标都没观测到时退回 `t_leader_finish_s`。
        """
        if self.visited_mask.any():
            return float(max(self.t_leader_finish_s, np.nanmax(self.visit_time_s)))
        return float(self.t_leader_finish_s)

    @property
    def t_complete_s(self) -> float:
        """任务**完成**时刻 = 全覆盖前提下的 `t_finish_s`。**跨场景比时间用这个。**

        口径与 `contracts/lawnmower.py` 的"完成时刻的口径"一致,两个场景同一把尺子。

        ⚠ 与 `duration_s` 不是一回事:`duration_s` = 时间线末端,含
          `settle_time_s`(默认 60 s)的收尾余量 —— 拿它比时间会凭空多算 60 s。
        ⚠ 未全覆盖时任务根本没完成,返回 nan(别用一个"漏了目标的完成时刻"去比)。
          要那个数请取 `t_finish_s`,并连同覆盖率一起读。
        """
        if self.coverage < 1.0:
            return float("nan")
        return self.t_finish_s

    @property
    def leader_hold_total_s(self) -> float:
        """Leader 累计停船时长(D16)。等待队友的代价直接就是这个数。"""
        dt = float(self.meta.get("dt_s", 0.0))
        return float(self.leader_holding.sum()) * dt

    def leader_hold_s_by(self, flag: int) -> float:
        """按停船原因拆分的时长(HOLD_LAGGING / HOLD_ENDANGERED)。

        ⚠ 两条判据可以同时成立,所以分项之和 >= 总时长,不是相加关系。
        """
        dt = float(self.meta.get("dt_s", 0.0))
        return float((self.leader_hold_reason & int(flag) != 0).sum()) * dt

    @property
    def fix_age_max_s(self) -> float:
        """规划时刻 USBL 定位陈旧度的最大值。由 `usbl_period_s` 决定,与规划周期无关。"""
        ages = [r.max_fix_age_s for r in self.plan_rounds]
        return max(ages) if ages else 0.0

    # --- 評価指標(定义见 contracts/metrics.py) --------------------------
    @property
    def active_mask(self) -> np.ndarray:
        """稼働率一类指标的**有効区間**:t <= t_finish_s 的那一段。

        为什么不用整条时间线:尾部有 `settle_time_s`(默认 60 s)的收尾余量,那段全是
        IDLE,算进去会凭空抬高"空转率",而且余量长度与被比较的参数无关。

        ⚠ D18 起分界取 `t_finish_s` 而非 `t_complete_s`:后者在未全覆盖时是 nan,
          会让漏点组退回整条时间线、被 settle 尾巴污染成"空转率高得离谱",
          与全覆盖组不在同一把尺子上。全覆盖时两者恒等,既有各组的数一个不变。
        """
        tf = self.t_finish_s
        if not np.isfinite(tf):
            return np.ones_like(self.t_s, dtype=bool)
        return self.t_s <= tf + 1e-9

    def duty_frac(self, status: int) -> float:
        """全 Follower 处于某状态的时间占比(有効区間内)。"""
        st = self.follower_status[self.active_mask]
        return float((st == int(status)).mean()) if st.size else 0.0

    @property
    def churn(self) -> Dict[str, int]:
        """计划反复程度。定义见 `contracts/metrics.py`:

        `wp_issue_total`      Σ 每条 Assignment 里真实 waypoint 的个数(重复算多次);
        `reassignment_count`  下发时该点的上一个归属是**另一台**的次数 —— 真正有代价的
                              那种变更(A 已经朝它走了一段,改派给 B 就白走了);
        `reassigned_wp_count` 至少被改派过一次的 waypoint 个数。
        """
        owner: Dict[int, int] = {}
        total = 0
        swaps = 0
        moved = set()
        for a in self.assignments:
            for w in a.wp_ids:
                total += 1
                w = int(w)
                if w in owner and owner[w] != a.follower_id:
                    swaps += 1
                    moved.add(w)
                owner[w] = a.follower_id
        return {"wp_issue_total": total, "reassignment_count": swaps,
                "reassigned_wp_count": len(moved)}

    def time_efficiency(self, lower_bound_s: float) -> float:
        """离理论最优还差多少 = 下界 / 实测**完成**时刻 ∈ (0, 1]。**严格口径**。

        `lower_bound_s` 取 `feasibility_estimate()["t_mission_reference_s"]`。
        它需要解一次全局 VRP,故不放进 `summary()`(那会让每次 run_mission 都慢 2 s),
        由 `08_sweep.py` 每组算一次。

        ⚠ 未全覆盖 ⇒ nan(任务没完成,谈不上"效率")。要漏点组也出数请用
          `time_efficiency_obs`,但先读那个方法的警告。
        """
        tc = self.t_complete_s
        return float(lower_bound_s / tc) if np.isfinite(tc) and tc > 0 else float("nan")

    def time_efficiency_obs(self, lower_bound_s: float) -> float:
        """**実測**時間効率 = 下界 / `t_finish_s`。漏点组也有数(D18)。

        ⚠ **这个数会被漏点抬高,不是可以直接横比的效率。** 分子(下界)是按 22 个目标
          全做完算的,分母却只是"做完手上那几个"的收工时刻 —— 漏得越多分母越小,
          数字越好看。极端情形:一个目标都不做,分母 = 走完测线,它反而接近 1.0。
          它只回答一件事:**在这趟实际收工的时间里,相对全任务下界处在什么位置**。
        ⚠ 要一个能横比的单列,用 `time_efficiency_cov`(乘覆盖率折扣)或
          `t_per_target_s`(对漏点中性)。
        """
        tf = self.t_finish_s
        return float(lower_bound_s / tf) if np.isfinite(tf) and tf > 0 else float("nan")

    def time_efficiency_cov(self, lower_bound_s: float) -> float:
        """覆盖折扣時間効率 = `coverage` x `time_efficiency_obs` —— **可横比的单列**。

        等价写法 `lower_bound_s / (t_finish_s / coverage)`:分母是"按本组实测的
        単位観測あたり時間、外推到把 22 个目标全做完"所需的时间。漏点不再被奖励 ——
        漏一半就等于把分母翻一倍。

        ⚠ 这是**线性外推**,假设剩下的目标与已做的一样费时。漏掉的往往是最难够到的
          那些(waitoff 组漏的正是 Leader 甩开后落在窗口外的点),所以这个数仍偏乐观,
          是**上界**不是预测。全覆盖时它与 `time_efficiency` 恒等。
        """
        return float(self.coverage * self.time_efficiency_obs(lower_bound_s))

    @property
    def solve_wall_s(self) -> np.ndarray:
        """每轮 VRP 求解的**实测挂钟**耗时(s)。

        ⚠ 宿主机性能 + `cfg.vrp_time_limit_s` 决定,**不进入仿真时间轴、不进入任何
          数值结论**。仿真侧给求解留的余量是 `cfg.plan_solve_s`(默认 0),两者别混。
        ⚠ 因为是挂钟量,同一份输入两次运行的 npz **不会逐字节相同**。
        """
        return np.asarray([r.solve_wall_s for r in self.plan_rounds], dtype=FLOAT)

    @property
    def pool_sizes(self) -> np.ndarray:
        """每轮可规划池的大小(= 该轮 VRP 的问题规模)。"""
        return np.asarray([r.pool_size for r in self.plan_rounds], dtype=int)

    def summary(self) -> Dict[str, Any]:
        d = self.per_follower_distance_m
        v = self.per_follower_visits
        wall = self.solve_wall_s
        pool = self.pool_sizes
        assigned = np.asarray([r.n_assigned_total for r in self.plan_rounds], dtype=int)
        # --- 評価指標(口径见 contracts/metrics.py) -----------------------
        vt = self.visit_time_s[self.visited_mask]
        n_vis = int(self.visited_mask.sum())
        idle = self.duty_frac(STATUS_IDLE)
        n_msg = int(self.comm_events.shape[0])
        dur = float(self.t_s[-1]) if self.t_s.size else 0.0
        ch = self.churn
        out = {
            "scenario": "vrp",
            "t_complete_s": self.t_complete_s,
            "t_finish_s": self.t_finish_s,
            "t_leader_finish_s": self.t_leader_finish_s,
            "t_last_observation_s": (float(np.nanmax(self.visit_time_s))
                                     if self.visited_mask.any() else float("nan")),
            "duration_s": float(self.t_s[-1]),
            "coverage": self.coverage,
            "visited": int(self.visited_mask.sum()),
            "n_targets": self.n_targets,
            "missed_wp_ids": self.missed_wp_ids,
            "n_requests": len(self.requests),
            "n_assignments": len(self.assignments),
            "n_assignments_with_projection": sum(1 for a in self.assignments
                                                 if a.has_projection),
            "per_follower_distance_m": [float(x) for x in d],
            "per_follower_visits": [int(x) for x in v],
            "max_distance_m": float(d.max()) if d.size else 0.0,
            "total_distance_m": float(d.sum()),
            # ⑥ 負荷均衡(D19)四列在下方 out.update(balance_metrics(...)) 里统一给,
            # 三个场景共用 metrics_util 的同一份实现,免得口径抄三份漂移。
            # Leader 对队友位置的推算误差(nav_drift=0 时应恒为 0)
            "shadow_error_max_m": (float(self.shadow_error[:, 2].max())
                                   if self.shadow_error.size else 0.0),
            "shadow_error_mean_m": (float(self.shadow_error[:, 2].mean())
                                    if self.shadow_error.size else 0.0),
            # --- Leader 停船(D16) --------------------------------------
            "leader_hold_total_s": self.leader_hold_total_s,
            "leader_hold_lagging_s": self.leader_hold_s_by(HOLD_LAGGING),
            "leader_hold_endangered_s": self.leader_hold_s_by(HOLD_ENDANGERED),
            "leader_hold_turnaround_s": self.leader_hold_s_by(HOLD_TURNAROUND),
            "leader_hold_frac": (float(self.leader_holding.mean())
                                 if self.leader_holding.size else 0.0),
            "timed_out": bool(self.meta.get("timed_out", False)),
            # --- 声学通信与规划轮次(D15 / D16) --------------------------
            "acoustic_hop_s": float(self.cfg.acoustic_hop_s),
            "usbl_period_s": float(self.cfg.usbl_period_s),
            "usbl_cycle_s": float(self.cfg.usbl_cycle_s),
            "plan_period_s": float(self.cfg.plan_period_s),
            "plan_latency_s": float(self.cfg.plan_latency_s),
            "fix_age_max_s": self.fix_age_max_s,
            "n_plan_rounds": len(self.plan_rounds),
            "n_comm_messages": int(self.comm_events.shape[0]),
            "pool_size_mean": float(pool.mean()) if pool.size else 0.0,
            "pool_size_max": int(pool.max()) if pool.size else 0,
            "assigned_per_round_mean": float(assigned.mean()) if assigned.size else 0.0,
            # ⚠ 挂钟量,只做诊断;别拿它做任何跨机器的结论。
            "solve_wall_total_s": float(wall.sum()) if wall.size else 0.0,
            "solve_wall_max_s": float(wall.max()) if wall.size else 0.0,
            "solve_wall_mean_s": float(wall.mean()) if wall.size else 0.0,

            # ===== 評価指標 —— 定义、公式、方向全在 contracts/metrics.py =====
            # ① 時間効率(`time_efficiency` 需要全局 VRP 下界,由 08_sweep 算)
            "t_observation_mean_s": float(vt.mean()) if vt.size else float("nan"),
            "t_observation_median_s": float(np.median(vt)) if vt.size else float("nan"),
            "t_observation_p90_s": (float(np.percentile(vt, 90)) if vt.size
                                    else float("nan")),
            # 分子用 t_finish_s(不是 t_complete_s):这样漏点组也有数,而且分子分母
            # 同时缩小 ⇒ 对漏点中性,不奖励"少做几个点提前收工"(D18)。
            "t_per_target_s": (self.t_finish_s / n_vis if n_vis
                               else float("nan")),
            # ② 稼働率(分母 = 有効区間 t <= t_finish_s,见 active_mask)
            "duty_transit_frac": self.duty_frac(STATUS_TRANSIT),
            "duty_dwell_frac": self.duty_frac(STATUS_DWELL),
            "duty_idle_frac": idle,
            "duty_productive_frac": 1.0 - idle,
            # ④ 通信コスト
            "channel_duty_frac": (n_msg * float(self.cfg.acoustic_hop_s) / dur
                                  if dur > 0 else 0.0),
            "messages_per_target": (n_msg / n_vis) if n_vis else float("nan"),
            # ③ 移動効率(`total_distance_m` / `max_distance_m` / `jain_fairness`
            #   已在上面;这里补含 Leader 的口径)
            "leader_distance_m": self.leader_distance_m,
            "fleet_distance_m": self.fleet_distance_m,
            "distance_per_target_m": (self.fleet_distance_m / n_vis if n_vis
                                      else float("nan")),
            # ⑤ 計画品質
            "wp_issue_total": ch["wp_issue_total"],
            "wp_issues_per_target": (ch["wp_issue_total"] / self.n_targets
                                     if self.n_targets else 0.0),
            "sequence_utilisation": (self.n_targets / ch["wp_issue_total"]
                                     if ch["wp_issue_total"] else float("nan")),
            "reassignment_count": ch["reassignment_count"],
            "reassigned_wp_count": ch["reassigned_wp_count"],
        }
        # ⑥ 負荷均衡(D19):距离与时间分开量 —— min-max VRP 均衡的是距离,
        # 而完成时刻由最慢那台的**时间**决定,停留占比高时两者会分叉。
        out.update(balance_metrics(d, self.busy_time_s))
        return out

    @property
    def busy_time_s(self) -> np.ndarray:
        """各 Follower 稼働時間 (N,) = 非 IDLE 的拍数 x dt。⑥ 負荷均衡族的输入。

        ⚠ 用的是"真正在干活(移動 + 停留)的时长",不是任务总时长 ——
          用总时长的话所有机都一样,量不出任何东西。
        """
        dt = float(self.meta.get("dt_s", self.cfg.dt_s))
        return (self.follower_status != STATUS_IDLE).sum(axis=0).astype(FLOAT) * dt


# ======================================================================
# 主循环
# ======================================================================
def run_mission(cfg: Optional[MissionConfig] = None,
                world: Optional[MothraWorld] = None,
                *, verbose: bool = False) -> MissionResult:
    """跑一次完整任务。纯确定性(除 ortools 元启发式的时间预算外)。"""
    cfg = cfg or MissionConfig()
    mw = world or build_mothra_world(cfg.sim)
    ds = mw.dataset
    n_t = ds.n
    n_f = cfg.n_followers
    K = cfg.max_sequence_len + 1                       # +1 给投影点留位

    win_cfg = mothra_window_config(look_back_m=cfg.window_look_back_m,
                                   width_m=cfg.window_width_m)
    # 车道间距 = 声呐幅宽(window_width_m)⇒ 相邻车道边到边覆盖、不留空档,
    # 与 `01_build_world.py` 诊断用的 `leader_track` 同一惯例。场地 East 跨度
    # <= 声呐幅宽时退化为单车道直线,path 只有 2 个点,与旧版单测线行为逐位相同。
    path = leader_track(mw.world, cfg.window_width_m)
    leader = Leader(cfg, path=path)
    followers = [Follower(i, s, cfg) for i, s in enumerate(cfg.follower_starts_ned)]

    visited = [False] * n_t
    visit_time = np.full(n_t, np.nan, dtype=FLOAT)
    visit_by = np.full(n_t, -1, dtype=np.int16)
    id_to_idx = {int(w): j for j, w in enumerate(ds.waypoint_id)}

    tracker = LeaderTracker(cfg)                       # Leader 侧的队友认知,唯一入口
    assignments: List[Assignment] = []
    requests: List[FollowerRequest] = []
    plan_rounds: List[PlanRound] = []
    comm: List[Tuple[float, float, int, int, int]] = []   # 声学报文流水
    events: List[Tuple[float, str, Any]] = []          # 在途报文 (到达时刻, 类型, 载荷)
    shadow_err: List[Tuple[float, int, float]] = []    # (t_ref, follower_id, 推算误差 m)

    rec_t, rec_ln, rec_le, rec_psi, rec_win = [], [], [], [], []
    rec_pos, rec_st, rec_occ = [], [], []
    rec_q, rec_qxy, rec_dist, rec_vis = [], [], [], []

    t = 0.0
    dt = cfg.dt_s
    tau = float(cfg.acoustic_hop_s)
    next_usbl_t = cfg.first_usbl_s        # USBL 定位的时钟
    next_plan_t = cfg.first_plan_s        # 路径规划的时钟(D16:两者互不相干)
    usbl_idx = 0
    round_idx = 0
    plan_end_t = -1.0             # 当前在途那一轮的广播落地时刻(强插末轮时用来避让)
    forced_start_t: Optional[float] = None   # 强插末轮的起点;None = 还没安排过
    leader_was_finished = False
    leader_finish_t = float("inf")
    held_lagging = False                  # 判据 A 的迟滞状态
    rec_hold, rec_hold_reason = [], []
    timed_out = True                      # 正常终止会把它置 False

    while t <= cfg.max_mission_time_s + 1e-9:
        # 影子与仿真时钟**逐拍同步**:Leader 的停船判据要用它,不能超前也不能滞后。
        tracker.advance_to(t)
        region = leader.window(win_cfg)
        # 窗口矩形在 North 上的真实跨度(仅供记录/出图;判后沿一律走 `along_back_m`,
        # 见 `_hold_reason`)。南行时后沿在 Leader 北侧、横移段则由窗口**宽度**定,
        # 都由 `window_north_span` 按 psi 统一算出。
        win_lo, win_hi = window_north_span(region)
        win_lo = max(0.0, win_lo)

        # --- 0. Leader 走完测线 → 强插最后一轮(不等周期,D15) ----------
        # 窗口就此冻结在最北 look_back 米,落在这段里、此前没被派出去的目标是最后
        # 的机会。周期性轮次最坏要再等一个 plan_period_s,而终止条件可能先生效。
        if leader.finished and not leader_was_finished:
            leader_was_finished = True
            leader_finish_t = t
            forced_start_t = max(t, plan_end_t)
            next_plan_t = forced_start_t

        # --- 1. USBL 定位循环起点:向第 0 台发询问 ----------------------
        if t >= next_usbl_t - 1e-9:
            comm.append((t, t + tau, MSG_POLL_REQ, NODE_LEADER, 0))
            events.append((t + tau, "POLL_REQ", {"fid": 0, "cycle": usbl_idx}))
            usbl_idx += 1
            next_usbl_t = t + cfg.usbl_period_s

        # --- 2. 规划轮:取最近一次定位 → 推算 → 解 VRP → 广播 ------------
        # ⚠ Leader 走完测线**且**全部目标已观测之后就不再开新轮:此后池恒为空,
        #   再规划只会给每台又下发一个投影点。投影点不是活儿,但它让队列非空 ⇒
        #   终止条件 `all(not f.queue ...)` 永远不成立 ⇒ 一直跑到 max_mission_time_s。
        #   实测(mothra, greedy):plan_period=15 s + 投影间隔 20 m 时,目标 1107.5 s
        #   就全做完了,却因为每 15 s 又追一个新投影点而空跑到 8000 s,航程与稼働時間
        #   被灌成任务本身的 6 倍。停止开新轮之后,**已下发**的最后一条投影腿仍照常
        #   走完并计入航程 —— `09_compare_methods.py` 明文要求把它作为"開ループ下発の
        #   実コスト"如实计上,这里只掐掉"无限续杯",不改那条口径。
        #   判据要用"池里还有没有能派的点",**不能**用 `all(visited)`:一旦有目标被
        #   永久错过(漏点),`all(visited)` 永远为假 ⇒ 又变回无限续杯。实测
        #   high_rise / dense_1 的提案组各漏 2~4 个点,就这样跑满了 300000 s。
        #   Leader 走完后窗口冻结,窗口外的点再也进不了池,故"池空"即"此后不会再有
        #   任何可派的活",是这里唯一正确的判据。
        no_work_left = (leader.finished
                        and available_pool(ds.targets_ned, region, visited=visited,
                                           occupied=[WP_NONE] * n_t,
                                           cfg=cfg).size == 0)
        if t >= next_plan_t - 1e-9 and not no_work_left:
            is_forced = (forced_start_t is not None
                         and abs(t - forced_start_t) < 1e-9)
            if is_forced:
                forced_start_t = None      # 只标记被强插的那一轮,之后恢复周期节拍
            t_deliver = t + cfg.plan_latency_s

            for i in range(n_f):
                shadow_err.append((t, i, float(np.linalg.norm(
                    tracker.shadows[i].position - followers[i].pos))))
            fix_ages = tuple(tracker.fix_ages_s)
            # VRP 起点 = 最近一次 USBL 定位沿承诺队列**解析推算**到此刻的位置。
            # 陈旧度由 usbl_period_s 决定,与 plan_period_s 无关 —— 这正是解耦的意义。
            positions = [np.asarray(p, dtype=FLOAT).reshape(2).copy()
                         for p in tracker.positions]
            # 占用/已访问用 **t_deliver** 口径(只读探测,不推进影子本体):那一跳的
            # 飞行时间里队列还在推进,按落地时刻冻结才与 `Follower.assign()` 保留的
            # 队首精确对齐,否则被释放的"下一个点"可能同时派给两台。
            occupied, seen = tracker.peek(n_t, id_to_idx, t_deliver)

            asgs, wall = plan_round(
                t_plan_s=t, t_deliver_s=t_deliver,
                follower_positions=positions,
                targets_ned=ds.targets_ned, waypoint_ids=ds.waypoint_id,
                region=region, visited=seen, occupied=occupied,
                world_east_max_m=mw.world.y_max_m, cfg=cfg,
                world_north_max_m=mw.world.x_max_m)
            assignments.extend(asgs)
            plan_rounds.append(PlanRound(
                round_idx=round_idx, t_start_s=float(t),
                t_plan_s=float(t), t_deliver_s=float(t_deliver),
                leader_north_m=float(region.leader_pose[0]),
                pool_size=len(asgs[0].pool_ids) if asgs else 0,
                n_assigned=tuple(a.n_real for a in asgs),
                n_projection=sum(1 for a in asgs if a.has_projection),
                solver=asgs[0].solver if asgs else "trivial",
                solve_wall_s=float(wall), forced_final=bool(is_forced),
                fix_age_s=fix_ages))
            comm.append((t, t_deliver, MSG_BROADCAST, NODE_LEADER, NODE_ALL))
            events.append((t_deliver, "BROADCAST", {"asgs": asgs}))
            if verbose:
                pr = plan_rounds[-1]
                print(f"  轮{pr.round_idx:3d} 规划@{t:7.1f}s 落地@{t_deliver:7.1f}s "
                      f"定位陈旧{pr.max_fix_age_s:5.1f}s "
                      f"池{pr.pool_size:3d} 下发{list(pr.n_assigned)} "
                      f"投影{pr.n_projection} "
                      f"求解{pr.solve_wall_s * 1e3:6.1f}ms [{pr.solver}]"
                      f"{' ←末轮' if pr.forced_final else ''}")
            plan_end_t = t_deliver
            round_idx += 1
            # 用 `t +` 而不是 `+=`:强插末轮会把 next_plan_t 拨到任意时刻,
            # 累加会算出一个过去的时刻从而在下一拍立刻再触发一轮。
            next_plan_t = t + cfg.plan_period_s

        # --- 3. 本拍到期的在途报文 --------------------------------------
        # 用 while 而不是一次性取:acoustic_hop_s = 0 时整个定位循环会塌进同一拍,
        # 后续报文必须在本拍继续处理,否则每跳白白拖一个 dt。
        while True:
            due_now = [e for e in events if e[0] <= t + 1e-9]
            if not due_now:
                break
            events = [e for e in events if e[0] > t + 1e-9]
            for _due, kind, payload in due_now:
                if kind == "POLL_REQ":
                    # 定位询问送到 F_i:它**此刻**采样自己的位置与占用点并回复。
                    fid = int(payload["fid"])
                    tracker.mark_report_sample(t, fid, followers[fid].occupied_wp)
                    req = FollowerRequest(
                        t_s=t, follower_id=fid,
                        position_ned=followers[fid].pos.copy(),
                        occupied_wp=followers[fid].occupied_wp,
                        t_recv_s=t + tau)
                    requests.append(req)
                    comm.append((t, t + tau, MSG_POLL_REP, fid, NODE_LEADER))
                    events.append((t + tau, "POLL_REP", {"req": req,
                                                         "cycle": payload["cycle"]}))

                elif kind == "POLL_REP":
                    # 回复送到 Leader:校正该台影子的坐标认知,再询问下一台。
                    req = payload["req"]
                    fid = int(req.follower_id)
                    tracker.apply_report(t, fid, req.position_ned)
                    if fid + 1 < n_f:
                        comm.append((t, t + tau, MSG_POLL_REQ, NODE_LEADER, fid + 1))
                        events.append((t + tau, "POLL_REQ",
                                       {"fid": fid + 1, "cycle": payload["cycle"]}))

                elif kind == "BROADCAST":
                    # 广播落地。真机与影子在**同一拍**装上同一条序列 ——
                    # 这是整个模型的对齐点,拆开就等于让 Leader 提前一跳知道未来。
                    for asg in payload["asgs"]:
                        followers[asg.follower_id].assign(_to_items(asg, id_to_idx))
                        tracker.on_assignment(asg, id_to_idx)

        # --- 3.5 Leader 停船判据(D16),**只用影子推算,不读仿真真值** -----
        hold_reason = _hold_reason(cfg, tracker, ds, id_to_idx, n_t, region,
                                   held_lagging,
                                   dist_to_turn_m=leader.dist_to_segment_end_m,
                                   turn_ahead=leader.has_next_segment)
        held_lagging = bool(hold_reason & HOLD_LAGGING)
        hold = hold_reason != HOLD_NONE

        # --- 4. 记录(规格 §7:逐时刻的序列与位置) ----------------------
        rec_t.append(t)
        rec_ln.append(leader.north)
        rec_le.append(leader.east)
        rec_psi.append(leader.psi)
        rec_win.append([win_lo, win_hi])
        rec_hold.append(hold)
        rec_hold_reason.append(hold_reason)
        rec_pos.append([f.pos.copy() for f in followers])
        rec_st.append([f.status for f in followers])
        rec_occ.append([f.occupied_wp for f in followers])
        rec_dist.append([f.distance_m for f in followers])
        rec_vis.append(int(np.sum(visited)))
        q = np.full((n_f, K), WP_NONE, dtype=np.int16)
        qxy = np.full((n_f, K, 2), np.nan, dtype=np.float32)
        for i, f in enumerate(followers):
            for k, it in enumerate(f.queue[:K]):
                q[i, k] = it.wp_id
                qxy[i, k] = it.point
        rec_q.append(q)
        rec_qxy.append(qxy)

        # --- 5. 推进 ----------------------------------------------------
        leader.step(dt, hold=hold)
        for i, f in enumerate(followers):
            for wid in f.step(dt):
                j = id_to_idx[int(wid)]
                if not visited[j]:
                    visited[j] = True
                    visit_time[j] = t + dt
                    visit_by[j] = i
        t += dt

        # --- 6. 终止:Leader 走完 + 全部空闲 + 信道无在途 + 窗口里没活了 ---
        # ⚠ 收尾余量从 **Leader 实际走完的时刻**起算,不是 `测线长/v_leader` ——
        #   D16 起 Leader 会停船等队友,那个公式不再是它的完成时刻。
        if (leader.finished and not events
                and all(not f.queue for f in followers)
                and t > leader_finish_t + cfg.settle_time_s):
            # 窗口里还有没被访问的点就不许退出(D15):此刻队列全空 ⇒ 无人占用,
            # 下一轮规划一定会把它们派出去。窗口**外**的点本就进不了池,不阻塞终止,
            # 所以这个条件必然收敛;`max_mission_time_s` 仍是硬兜底。
            left = available_pool(ds.targets_ned, region, visited=visited,
                                  occupied=[WP_NONE] * n_t, cfg=cfg)
            if left.size == 0:
                timed_out = False
                break

    result = MissionResult(
        cfg=cfg,
        t_s=np.asarray(rec_t, dtype=FLOAT),
        leader_north_m=np.asarray(rec_ln, dtype=FLOAT),
        window_north=np.asarray(rec_win, dtype=FLOAT),
        follower_pos=np.asarray(rec_pos, dtype=FLOAT),
        follower_status=np.asarray(rec_st, dtype=np.int8),
        follower_occupied=np.asarray(rec_occ, dtype=np.int16),
        follower_queue=np.asarray(rec_q, dtype=np.int16),
        follower_queue_xy=np.asarray(rec_qxy, dtype=np.float32),
        follower_distance_m=np.asarray(rec_dist, dtype=FLOAT),
        visited_count=np.asarray(rec_vis, dtype=np.int32),
        assignments=assignments, requests=requests,
        shadow_error=(np.asarray(shadow_err, dtype=FLOAT).reshape(-1, 3)
                      if shadow_err else np.zeros((0, 3), dtype=FLOAT)),
        visit_time_s=visit_time, visit_by=visit_by,
        waypoint_ids=ds.waypoint_id.copy(),
        leader_holding=np.asarray(rec_hold, dtype=bool),
        leader_hold_reason=np.asarray(rec_hold_reason, dtype=np.int8),
        leader_east_m=np.asarray(rec_le, dtype=FLOAT),
        leader_psi=np.asarray(rec_psi, dtype=FLOAT),
        plan_rounds=plan_rounds,
        comm_events=(np.asarray(comm, dtype=FLOAT).reshape(-1, 5) if comm
                     else np.zeros((0, 5), dtype=FLOAT)),
        plan_stats=(np.asarray(
            [[r.round_idx, r.t_start_s, r.t_plan_s, r.t_deliver_s, r.pool_size,
              r.n_assigned_total, r.n_projection, r.solve_wall_s,
              float(r.forced_final)] for r in plan_rounds],
            dtype=FLOAT).reshape(-1, 9) if plan_rounds
            else np.zeros((0, 9), dtype=FLOAT)),
        meta={"world": mw.meta, "window": {"look_back_m": win_cfg.look_back_m,
                                           "width_m": win_cfg.width_m},
              "dt_s": dt, "n_followers": n_f, "max_sequence_len": cfg.max_sequence_len,
              "acoustic_hop_s": tau, "plan_period_s": float(cfg.plan_period_s),
              "usbl_period_s": float(cfg.usbl_period_s),
              "usbl_cycle_s": float(cfg.usbl_cycle_s),
              "plan_latency_s": float(cfg.plan_latency_s),
              "leader_finish_s": float(leader_finish_t),
              # 折线走过的**实际路程**(含换道横移与往复), 与 window_width_m
              # 的多车道推广一起加入 —— north[-1]-north[0] 不再等于总路程,见
              # `MissionResult.leader_distance_m`。
              "leader_total_distance_m": float(leader.distance_m),
              "leader_lane_spacing_m": float(win_cfg.width_m),
              "n_leader_lanes": int(len(path) // 2),
              "timed_out": bool(timed_out),
              "report_mismatch_count": tracker.report_mismatch_count},
    )
    return result


def _unassigned_in_window(tracker: LeaderTracker, ds, id_to_idx, n_targets: int,
                          region) -> List[int]:
    """窗口内**尚未观测且无人前往**的目标下标。判据 C / D 共用的扫描。"""
    occ = tracker.occupancy(n_targets)
    seen = tracker.visited_mask(n_targets, id_to_idx)
    tg = ds.targets_ned
    return [j for j in range(n_targets)
            if not seen[j] and occ[j] == WP_NONE
            and window_contains_world(region, tg[j])]


def _hold_reason(cfg: MissionConfig, tracker: LeaderTracker, ds, id_to_idx,
                 n_targets: int, region, held_lagging: bool,
                 dist_to_turn_m: float = float("inf"),
                 turn_ahead: bool = False) -> int:
    """Leader 这一拍该不该停船,以及为什么(D16)。

    ⚠ **只吃影子模型**(`vrpsim/tracking.py`)与目标坐标 —— 目标坐标是 Leader 自己
      声呐扫出来的,合法可知;队友位置则一律走推算,绝不读仿真真值。这个函数如果哪天
      多了一个 `followers` 形参,信息边界就破了。

    判据 A(`leader_wait_on_lagging_follower`):某台的推算位置落到窗口后沿之后。
        带迟滞:已经在停了的话,要重新进到后沿以内 `leader_wait_release_margin_m`
        米才算解除 —— 两者同速时不留裕度会逐拍停走振荡。
    判据 C(`leader_wait_on_endangered_target`):窗口内还有**尚未被分配**的目标,
        再走 `wait_lookahead_s` 就要掉出后沿。这正是覆盖率掉下去的直接原因。

    ⚠ 两条判据都按 `along_back_m`(沿 Leader **后方**的局部距离 s)判,**不用**
      North 坐标。2026-08-24 之前这里拿 `back_north = leader.north - look_back`
      当后沿,只有北行才对;多车道 boustrophedon 的南行测线上后沿在 Leader 北侧,
      该写法下两条判据在南行全程**一次都不触发**(sparse_2 实测南行占 46% 时长、
      触发 0 次,4 个目标因此无人接单就滑出窗口)。s 的定义与
      `window_contains_world` 同源,故"还在窗口内"与"离后沿多远"必然自洽。
    """
    reason = HOLD_NONE
    look_back = float(region.look_back_m)

    if cfg.leader_wait_on_lagging_follower:
        # 落后 = 沿后方距离已超过后沿(s > look_back)。迟滞:已在停时把门槛提前
        # release_margin 米,要真正回到窗口里才解除。
        thr = look_back - (cfg.leader_wait_release_margin_m if held_lagging else 0.0)
        if any(along_back_m(region, s.position) > thr + 1e-9 for s in tracker.shadows):
            reason |= HOLD_LAGGING

    pending = None      # 窗口内未观测且无人前往的点;C 与 D 共用,只扫一次
    if cfg.leader_wait_on_endangered_target or (
            cfg.leader_wait_on_turnaround and turn_ahead):
        pending = _unassigned_in_window(tracker, ds, id_to_idx, n_targets, region)

    if cfg.leader_wait_on_endangered_target and pending:
        # 再走一个前瞻期就要越过后沿的点 = 濒危。
        edge = look_back - cfg.leader_speed_mps * cfg.wait_lookahead_s
        tg = ds.targets_ned
        if any(along_back_m(region, tg[j]) > edge + 1e-9 for j in pending):
            reason |= HOLD_ENDANGERED

    if cfg.leader_wait_on_turnaround and turn_ahead and pending:
        # 折返在即:窗口**整体**要转向,窗内所有未分配的点都会一次性出局 ——
        # 不问它们离后沿多远(判据 C 那把尺子在这里失效),只问"还有没有没派出去的"。
        if dist_to_turn_m <= cfg.leader_speed_mps * cfg.wait_lookahead_s + 1e-9:
            reason |= HOLD_TURNAROUND
    return reason


def _to_items(asg: Assignment, id_to_idx: Dict[int, int]) -> List[QueueItem]:
    """Assignment → Follower 队列项。末尾投影点标为 WP_PROJECTION(不占用)。"""
    items: List[QueueItem] = []
    for k in range(asg.n_points):
        if k < asg.n_real:
            wid = int(asg.wp_ids[k])
            items.append(QueueItem(wp_id=wid, point=asg.points_ned[k].copy(),
                                   target_idx=id_to_idx[wid]))
        else:
            items.append(QueueItem(wp_id=WP_PROJECTION,
                                   point=asg.points_ned[k].copy(), target_idx=-1))
    return items


# 占用表的计算已移到 `LeaderTracker.occupancy()` —— 它只用影子模型,不碰仿真真值。
# 原来那份 `_occupancy(followers, ...)` 直接读了另一台"此刻正在前往哪个点",
# 属于真值泄露,已删除。


# ======================================================================
# 落盘 / 回读
# ======================================================================
def save_result(res: MissionResult, npz_path: str, *, also_csv: bool = True) -> None:
    """时间线存 .npz;分配事件另存一份易读的 JSON(变长,不塞进 npz)。"""
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez(npz_path,
             t_s=res.t_s, leader_north_m=res.leader_north_m,
             leader_east_m=res.leader_east_m, leader_psi=res.leader_psi,
             window_north=res.window_north, follower_pos=res.follower_pos,
             follower_status=res.follower_status, follower_occupied=res.follower_occupied,
             follower_queue=res.follower_queue, follower_queue_xy=res.follower_queue_xy,
             follower_distance_m=res.follower_distance_m,
             visited_count=res.visited_count, visit_time_s=res.visit_time_s,
             visit_by=res.visit_by, waypoint_ids=res.waypoint_ids,
             shadow_error=res.shadow_error,
             leader_holding=res.leader_holding,
             leader_hold_reason=res.leader_hold_reason,
             comm_events=res.comm_events, plan_stats=res.plan_stats,
             meta_json=np.array(json.dumps(res.meta, ensure_ascii=False)))

    base = os.path.splitext(npz_path)[0]
    with open(base + "_assignments.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": res.summary(),
            # 每轮一条:握手时刻 + 问题规模 + 实测求解耗时(D15)
            "plan_rounds": [{
                "round_idx": r.round_idx, "t_start_s": r.t_start_s,
                "t_plan_s": r.t_plan_s, "t_deliver_s": r.t_deliver_s,
                "leader_north_m": round(r.leader_north_m, 3),
                "pool_size": r.pool_size, "n_assigned": list(r.n_assigned),
                "n_assigned_total": r.n_assigned_total,
                "n_projection": r.n_projection, "solver": r.solver,
                # ⚠ 挂钟量,宿主机相关,不进入仿真时间轴
                "solve_wall_s": round(r.solve_wall_s, 6),
                "forced_final": r.forced_final,
                "fix_age_s": [round(x, 3) for x in r.fix_age_s],
            } for r in res.plan_rounds],
            "assignments": [{
                "t_s": a.t_s, "t_deliver_s": a.t_deliver_s,
                "follower": a.follower_id, "leader_north_m": a.leader_north_m,
                "wp_ids": list(a.wp_ids), "has_projection": a.has_projection,
                "points_ned": a.points_ned.round(3).tolist(),
                "pool_ids": list(a.pool_ids), "solver": a.solver,
            } for a in res.assignments],
        }, f, ensure_ascii=False, indent=2)

    if also_csv:
        _write_timeline_csv(res, base + "_timeline.csv")


def _write_timeline_csv(res: MissionResult, path: str) -> None:
    """逐时刻一行的宽表(规格 §7 的人读版本)。"""
    import csv

    n_f = res.follower_pos.shape[1]
    K = res.follower_queue.shape[2]
    cols = ["t_s", "leader_north_m", "leader_east_m", "win_back_m", "win_front_m",
            "visited_count"]
    for i in range(n_f):
        cols += [f"f{i}_north_m", f"f{i}_east_m", f"f{i}_status", f"f{i}_occupied_wp",
                 f"f{i}_dist_m", f"f{i}_queue"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for k in range(len(res.t_s)):
            row = [f"{res.t_s[k]:.2f}", f"{res.leader_north_m[k]:.3f}",
                   f"{res.leader_east_m[k]:.3f}",
                   f"{res.window_north[k, 0]:.3f}", f"{res.window_north[k, 1]:.3f}",
                   int(res.visited_count[k])]
            for i in range(n_f):
                q = [int(v) for v in res.follower_queue[k, i] if v != WP_NONE]
                row += [f"{res.follower_pos[k, i, 0]:.3f}",
                        f"{res.follower_pos[k, i, 1]:.3f}",
                        STATUS_NAMES[int(res.follower_status[k, i])],
                        int(res.follower_occupied[k, i]),
                        f"{res.follower_distance_m[k, i]:.2f}",
                        "|".join("proj" if v == WP_PROJECTION else str(v) for v in q)]
            w.writerow(row)


def load_result(npz_path: str) -> Dict[str, Any]:
    """回读时间线(诊断/出图用;不还原 Assignment 对象,那份在同名 JSON 里)。"""
    with np.load(npz_path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files if k != "meta_json"}
        out["meta"] = json.loads(str(d["meta_json"]))
    return out
