"""[第2层] Leader / Follower 的运动学状态机(定步长可控,供任务层逐拍推进)。

为什么不用 `msim.physics.rollout`
--------------------------------------------------------------------------------
`RolloutEngine.execute(start_pose, waypoints)` 是**批处理**接口:一次吃完整条
waypoint 序列,返回 (轨迹, 能耗, 用时)。本任务要在序列执行**中途**接受新序列
(D8:新结果覆盖旧结果),还要逐时刻记录位置(规格 §7),批处理接口对不上。
故这里实现最小的定步长推进器:匀速直线趋近 + 到点停留。

能耗不在本步范围(规格没要),只记路程;要能耗时可把整条实际轨迹交给
`msim.eval.metrics.path_cost` 事后折算,口径不变。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from msim.contracts.geometry import WindowRegion
from msim.contracts.types import FLOAT
from msim.geometry.window import get_window

from .contracts.mission import (STATUS_DWELL, STATUS_IDLE, STATUS_TRANSIT, WP_NONE,
                                WP_PROJECTION, MissionConfig)

__all__ = ["Leader", "Follower", "QueueItem"]


@dataclass
class QueueItem:
    """待执行的一个航点。`wp_id` 为 WP_PROJECTION 时是投影点(不占用、不计覆盖)。"""
    wp_id: int
    point: np.ndarray            # (2,) [x_N, y_E]
    target_idx: int = -1         # 在 dataset 里的下标;投影点为 -1

    @property
    def is_projection(self) -> bool:
        return self.wp_id == WP_PROJECTION


class Leader:
    """沿测线前进的广域扫描 AUV。窗口前边界恒与其位置重合。

    ⚠ D16 起**不是匀速**:任务层每拍传 `hold=True` 就原地停船(速度置 0)。
      停不停由任务层按 Leader 自己的 DR 推算判(见 `contracts/mission.py` §8),
      本类只负责执行,不做判断 —— 免得运动学层去碰它不该知道的队友状态。
    """

    def __init__(self, cfg: MissionConfig, track_end_north_m: float):
        self.cfg = cfg
        self.north = float(cfg.leader_start_ned[0])
        self.east = float(cfg.leader_start_ned[1])
        self.psi = 0.0                      # 指 North
        self.track_end = float(track_end_north_m)
        self.distance_m = 0.0
        self.hold_time_s = 0.0              # 累计停船时长(诊断用)

    @property
    def pose(self) -> np.ndarray:
        return np.array([self.north, self.east, self.psi], dtype=FLOAT)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.north, self.east], dtype=FLOAT)

    @property
    def finished(self) -> bool:
        return self.north >= self.track_end - 1e-9

    def step(self, dt_s: float, *, hold: bool = False) -> None:
        """推进一拍。`hold=True` 时原地停船:不前进、不计路程,只累计停船时长。"""
        if self.finished:
            return
        if hold:
            self.hold_time_s += float(dt_s)
            return
        adv = min(self.cfg.leader_speed_mps * dt_s, self.track_end - self.north)
        self.north += adv
        self.distance_m += adv

    def window(self, window_cfg) -> WindowRegion:
        """当前观测窗口。用 msim 的 `get_window` —— 它的语义就是"窗口在 Leader 正后方",
        即前边界与 Leader 位置重合(规格 §3)。"""
        return get_window(self.pose, window_cfg)


class Follower:
    """执行 waypoint 序列的观测 AUV:匀速趋近 → 到达 → 停留 → 下一个。"""

    def __init__(self, follower_id: int, start_ned: Tuple[float, float],
                 cfg: MissionConfig):
        self.id = int(follower_id)
        self.cfg = cfg
        self.pos = np.asarray(start_ned, dtype=FLOAT).reshape(2).copy()
        self.queue: List[QueueItem] = []
        self.status: int = STATUS_IDLE
        self.dwell_left_s: float = 0.0
        self.distance_m: float = 0.0
        self.visited_wp_ids: List[int] = []
        self.n_projection_legs: int = 0

    # --- 占用(规格 §4/§5:一次只占当前那一个) -------------------------
    @property
    def active(self) -> Optional[QueueItem]:
        return self.queue[0] if self.queue else None

    @property
    def occupied_wp(self) -> int:
        """当前占用的 waypoint_id;投影点与空队列均为 WP_NONE(投影点不占用)。"""
        a = self.active
        if a is None or a.is_projection:
            return WP_NONE
        return a.wp_id

    @property
    def occupied_idx(self) -> int:
        a = self.active
        return -1 if (a is None or a.is_projection) else a.target_idx

    @property
    def pending_idx(self) -> List[int]:
        """已下发但**还没走到**的队列尾在 dataset 里的下标(不含当前占用的那个)。"""
        return [it.target_idx for it in self.queue[1:]
                if not it.is_projection and it.target_idx >= 0]

    # --- 接收新序列(D8:覆盖旧序列) -----------------------------------
    def assign(self, items: List[QueueItem]) -> None:
        """新序列覆盖旧序列;**正在前往的那个点保留在队首**。

        理由:该点已被占用、规划器把它排除在池外(D8"重解只涉及未被占用的点")。
        若丢弃它,这个点既不在新序列里、也不再有人去 → 永远没人访问。
        """
        keep = [self.active] if (self.active is not None
                                 and not self.active.is_projection) else []
        self.queue = keep + list(items)
        if self.status == STATUS_IDLE and self.queue:
            self.status = STATUS_TRANSIT

    # --- 一拍推进 --------------------------------------------------------
    def step(self, dt_s: float) -> List[int]:
        """推进 dt。返回本拍**完成观测**的 waypoint_id 列表(投影点不计)。"""
        done: List[int] = []
        remaining = float(dt_s)

        while remaining > 1e-12:
            if not self.queue:
                self.status = STATUS_IDLE
                break

            item = self.queue[0]
            if self.status == STATUS_DWELL:
                spent = min(remaining, self.dwell_left_s)
                self.dwell_left_s -= spent
                remaining -= spent
                if self.dwell_left_s > 1e-12:
                    break
                if not item.is_projection:
                    done.append(item.wp_id)
                    self.visited_wp_ids.append(item.wp_id)
                else:
                    self.n_projection_legs += 1
                self.queue.pop(0)
                self.status = STATUS_TRANSIT if self.queue else STATUS_IDLE
                continue

            # STATUS_TRANSIT / 刚拿到队列
            self.status = STATUS_TRANSIT
            delta = item.point - self.pos
            dist = float(np.linalg.norm(delta))
            if dist <= max(self.cfg.arrive_radius_m, 1e-9):
                # 已在容差内 ⇒ 判定到达。arrive_radius 默认 0,所以这里通常是
                # 上一轮 travel=dist 走满后的收尾,dist≈0,不存在"瞬移"。
                self.pos = item.point.copy()
                self.distance_m += dist
                dwell = (self.cfg.dwell_time_s
                         if (not item.is_projection or self.cfg.dwell_at_projection)
                         else 0.0)
                self.status = STATUS_DWELL
                self.dwell_left_s = dwell
                continue

            travel = min(self.cfg.follower_speed_mps * remaining, dist)
            self.pos = self.pos + delta / dist * travel
            self.distance_m += travel
            remaining -= travel / self.cfg.follower_speed_mps

        return done

    @property
    def queue_wp_ids(self) -> List[int]:
        return [it.wp_id for it in self.queue]
