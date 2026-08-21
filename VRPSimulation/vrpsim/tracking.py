"""[第2层] Leader 侧对 Follower 的**承诺队列解析推算**(影子模型)。

为什么需要它
--------------------------------------------------------------------------------
规划时 Leader 要用两台 Follower 的位置,但它拿到的每份上报都是**采样时刻**的快照,
而回复本身还要在水里飞一跳声学(D15)。到了规划时刻,先上报那台已经陈旧 3 跳、
后上报那台陈旧 1 跳 —— 这段时间里它们在哪、走到队列第几个了,Leader 并没有收到过。
直接去读仿真器里那台 Follower 的真实状态就是**真值泄露**,仿真会显得比实际更"聪明"。

本模块把 Leader 对每台 Follower 的认知显式建模成一个影子对象,它**只**吃三样
Leader 合法拥有的信息:

    1. 该 Follower 上次请求时上报的位置与占用点;
    2. Leader 自己发给它的 waypoint 序列(Leader 是作者,当然知道);
    3. 已知的运动学常数(巡航速度、到点停留时间)。

然后按同一套运动学把执行过程前推到当前时刻。这正是 `project_summary.md` §10 说的
"由承诺队列解析计算",不是"估计"。

⚠ 术语纪律(CLAUDE.md 硬纪律 4):`nav_drift_frac_of_distance == 0` 时,推算结果与真值
**逐位相同**(因为 Follower 就是严格照 Leader 发的序列、以已知速度执行的),
此时它是**确定量**,论文里不许写成"估计"。只有开了漂移旋钮,它才真的是估计。
`tests/test_mission.py::test_shadow_matches_truth_without_drift` 把这条钉住。

⚠ 与 `belief` 无关:本项目里 belief 是修士/博士信息回写开关的锁定术语
(`architecture_blueprint.md` §4.2),别把这里叫 belief。

漂移模型的口径
--------------------------------------------------------------------------------
建模的是 **Leader 对队友位置的认知误差**,不是 Follower 自己的执行误差 —— 两者是
两回事,混在一起就错了。故影子的**队列进度是精确的**(哪个点走完了、正在去哪个,
Leader 靠"自己派的活 + 已知速度"能算准),只有**坐标**带误差。
误差量级按 DVL/INS 惯例取"随行进距离线性增长"(0.5% 即 5 m/km),方向由 seed 决定,
保证可复现。
"""
from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

import numpy as np
from msim.contracts.types import FLOAT

from .agents import Follower, QueueItem
from .contracts.mission import WP_NONE, Assignment, MissionConfig

__all__ = ["FollowerShadow", "LeaderTracker"]


class FollowerShadow:
    """Leader 对**一台** Follower 的推算状态。"""

    def __init__(self, follower_id: int, start_ned, cfg: MissionConfig,
                 rng: Optional[np.random.Generator] = None):
        self.id = int(follower_id)
        self.cfg = cfg
        # 复用**同一套** Follower 运动学:这样影子与真值的差别只可能来自输入信息,
        # 不可能来自"两套运动学写得不一样"。
        self._sim = Follower(follower_id, start_ned, cfg)
        self.t = 0.0
        self.last_sync_t = 0.0
        self._dist_at_sync = 0.0
        self._rng = rng if rng is not None else np.random.default_rng(follower_id)
        self._drift = np.zeros(2, dtype=FLOAT)
        self._drift_dir = self._new_drift_dir()
        # 采样那一拍的快照(见 mark_report_sample);回复送达时用来把上报坐标搬到当前时刻。
        self._pos_at_sample = self._sim.pos.copy()
        self._dist_at_sample = 0.0
        self.reported_occupied_wp: int = WP_NONE
        # 上报的占用点与影子推出来的对不上的次数。两者本应逐次相同(队列进度是精确的),
        # 不为零就说明影子被喂错了信息。这里只计数不纠正,由 mission/脚本层暴露出来。
        self.report_mismatch_count: int = 0

    # --- 内部 -----------------------------------------------------------
    def _new_drift_dir(self) -> np.ndarray:
        a = float(self._rng.uniform(0.0, 2.0 * np.pi))
        return np.array([np.cos(a), np.sin(a)], dtype=FLOAT)

    def _refresh_drift(self) -> None:
        """漂移随"自上次上报以来行进的距离"线性增长,上报时清零。"""
        rate = float(self.cfg.nav_drift_frac_of_distance)
        if rate <= 0.0:
            self._drift = np.zeros(2, dtype=FLOAT)
            return
        travelled = float(self._sim.distance_m - self._dist_at_sync)
        self._drift = self._drift_dir * (rate * travelled)

    # --- Leader 合法可用的三样输入 --------------------------------------
    # 上报分两拍(D15):Follower 在 t_sample **采样并发出**,回复还要飞一跳声学,
    # Leader 到 t_recv 才拿到手。故拆成 mark_report_sample / apply_report。
    def mark_report_sample(self, t_s: float, occupied_wp: int) -> None:
        """采样那一拍调用:记下此刻影子的位置与里程,作为坐标误差的重累积原点。"""
        self.advance_to(t_s)
        self._pos_at_sample = self._sim.pos.copy()
        self._dist_at_sample = float(self._sim.distance_m)

        # 上报的占用点应当就是影子推出来的那个(队列进度是精确的);不一致说明影子
        # 被喂错了信息。不静默纠正 —— 只计数,让 mission/脚本层把它暴露出来。
        self.reported_occupied_wp = int(occupied_wp)
        if int(occupied_wp) != int(self.occupied_wp):
            self.report_mismatch_count += 1

    def apply_report(self, t_now_s: float, position_ned) -> None:
        """回复送达那一拍调用。

        Leader 学到的是**采样时刻**的坐标。要把它搬到当前时刻,加上"采样→现在"这段
        由承诺队列**解析推出**的位移即可(硬纪律 4:这是确定量,不是外插估计)。
        坐标误差的累积原点随之回退到采样时刻 —— 陈旧的正是那一跳的飞行时间。

        ⚠ `nav_drift_frac_of_distance == 0` 时本方法是恒等变换:影子与真值本就逐位
          相同,上报值等于 `_pos_at_sample`,位移抵消。
        """
        self.advance_to(t_now_s)
        reported = np.asarray(position_ned, dtype=FLOAT).reshape(2)
        self._sim.pos = reported + (self._sim.pos - self._pos_at_sample)
        self.last_sync_t = float(t_now_s)
        self._dist_at_sync = float(self._dist_at_sample)
        self._drift_dir = self._new_drift_dir()
        self._refresh_drift()

    def on_assignment(self, asg: Assignment, id_to_idx) -> None:
        """Leader 下发序列 → 影子照同样的规则更新队列(它知道自己发了什么)。"""
        items: List[QueueItem] = []
        from .contracts.mission import WP_PROJECTION
        for k in range(asg.n_points):
            if k < asg.n_real:
                wid = int(asg.wp_ids[k])
                items.append(QueueItem(wid, asg.points_ned[k].copy(), id_to_idx[wid]))
            else:
                items.append(QueueItem(WP_PROJECTION, asg.points_ned[k].copy(), -1))
        self._sim.assign(items)

    def advance_to(self, t_s: float) -> None:
        """把影子前推到 t_s(按已知运动学解析执行承诺队列)。"""
        dt = float(t_s) - self.t
        if dt > 1e-12:
            self._sim.step(dt)
            self.t = float(t_s)
        self._refresh_drift()

    def peek_at(self, t_s: float):
        """看一眼 t_s 时刻的队列进度,返回 `(occupied_idx, visited_wp_ids)`。

        **不动本体的时钟** —— 这是"看一眼未来",不是"走到未来"。规划要按广播落地
        时刻冻结占用(见 LeaderTracker.peek),但影子本体必须与仿真时钟逐拍同步,
        否则停船判据会拿到一个超前的位置。
        """
        sim = copy.deepcopy(self._sim)
        dt = float(t_s) - self.t
        if dt > 1e-12:
            sim.step(dt)
        return sim.occupied_idx, list(sim.visited_wp_ids)

    # --- Leader 眼中的该 Follower ---------------------------------------
    @property
    def position(self) -> np.ndarray:
        """推算位置(含漂移)。漂移为 0 时 == 真值。"""
        return self._sim.pos + self._drift

    @property
    def true_model_position(self) -> np.ndarray:
        """不含漂移的解析推算位置(诊断用:与真值比对应当逐位相同)。"""
        return self._sim.pos.copy()

    @property
    def drift_m(self) -> float:
        return float(np.linalg.norm(self._drift))

    @property
    def occupied_idx(self) -> int:
        return self._sim.occupied_idx

    @property
    def occupied_wp(self) -> int:
        return self._sim.occupied_wp

    @property
    def pending_idx(self) -> List[int]:
        return self._sim.pending_idx

    @property
    def visited_wp_ids(self) -> List[int]:
        return list(self._sim.visited_wp_ids)


class LeaderTracker:
    """Leader 对**全部** Follower 的认知。规划器只许从这里取队友状态。"""

    def __init__(self, cfg: MissionConfig):
        self.cfg = cfg
        base = int(cfg.seed)
        self.shadows: List[FollowerShadow] = [
            FollowerShadow(i, s, cfg, np.random.default_rng(base + 1000 * (i + 1)))
            for i, s in enumerate(cfg.follower_starts_ned)]

    def advance_to(self, t_s: float) -> None:
        for s in self.shadows:
            s.advance_to(t_s)

    def mark_report_sample(self, t_s: float, follower_id: int,
                           occupied_wp: int) -> None:
        self.shadows[follower_id].mark_report_sample(t_s, occupied_wp)

    def apply_report(self, t_now_s: float, follower_id: int, position_ned) -> None:
        self.shadows[follower_id].apply_report(t_now_s, position_ned)

    def on_assignment(self, asg: Assignment, id_to_idx) -> None:
        self.shadows[asg.follower_id].on_assignment(asg, id_to_idx)

    @property
    def positions(self) -> List[np.ndarray]:
        return [s.position for s in self.shadows]

    @property
    def report_mismatch_count(self) -> int:
        return int(sum(s.report_mismatch_count for s in self.shadows))

    @property
    def fix_ages_s(self) -> List[float]:
        """每台"最近一次 USBL 定位到现在"的陈旧度(s)。"""
        return [float(s.t - s.last_sync_t) for s in self.shadows]

    def peek(self, n_targets: int, id_to_idx, t_s: float) -> Tuple[List[int], List[bool]]:
        """看一眼 t_s 时刻的 `(占用表, 已观测掩码)`,**不推进影子本体的时钟**。

        规划要按**广播落地时刻**冻结占用(理由见 `occupancy` 的注释),但影子本体
        必须与仿真时钟逐拍同步 —— 否则 Leader 的停船判据会拿到一个超前几秒的位置,
        等于偷看未来。两个需求靠这个只读探测分开。
        """
        occ = [WP_NONE] * n_targets
        vis = [False] * n_targets
        for s in self.shadows:
            j, wids = s.peek_at(t_s)
            if j >= 0:
                occ[j] = s.id
            for w in wids:
                vis[id_to_idx[int(w)]] = True
        return occ, vis

    def occupancy(self, n_targets: int) -> List[int]:
        """占用表,**全部由影子推出**,不碰真值。

        口径(D15 收口,关闭待裁决 6):**只有每台"当前正在前往"的那一个点被冻结**,
        已下发未走到的队列尾全部释放回池。

        为什么不再有 `reserve_other_queue` 那个开关:旧时序里两台交替请求,一次只
        重规划一台,另一台的队列尾必须预留否则会被派去同一批点。新时序下**每轮两台
        一起上报、一起重规划**,min-max VRP 一次解出的两条路线天然互斥,预留反而是
        白白锁住自由度。

        ⚠ 调用时机很关键:必须把影子推到**广播落地时刻**再取,不能在规划时刻取。
          那一跳的飞行时间里某台可能走完当前点、自动前进到队列下一个;按规划时刻
          冻结的话,被释放的"下一个"可能同时被派给另一台,而 `Follower.assign()` 的
          D8 保头规则又会把它留在队首 ⇒ 两台去同一点。见 `vrpsim/mission.py`。
        """
        occ = [WP_NONE] * n_targets
        for s in self.shadows:
            j = s.occupied_idx
            if j >= 0:
                occ[j] = s.id
        return occ

    def visited_mask(self, n_targets: int, id_to_idx) -> List[bool]:
        """Leader 认为已经观测完的目标。

        同样只用影子:Follower 不会逐点回报"我拍完了",但 Leader 派了活、又知道
        速度与停留时间,能算出对方走完了哪几个。
        """
        vis = [False] * n_targets
        for s in self.shadows:
            for wid in s.visited_wp_ids:
                vis[id_to_idx[int(wid)]] = True
        return vis
