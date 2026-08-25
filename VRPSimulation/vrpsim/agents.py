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
    """沿多车道 lawnmower 测线前进的广域扫描 AUV。窗口前边界恒与其位置重合。

    路径 `path`:(M,2) `[x_N, y_E]` 折线端点,M>=2,由
    `vrpsim/windows.py::leader_track` 生成(boustrophedon 往复,车道间距 =
    声呐幅宽 `window_cfg.width_m`,边到边覆盖、不留空档)。Leader 沿折线逐段
    前进,到达一段终点即接着走下一段 —— 车道换道时航向 `psi` 跟着转向东西向,
    这不是特例分支,是同一段推进逻辑的自然结果。

    单车道场景(场地 East 跨度 <= 车道间距)的 `path` 退化为 2 个点的直线,
    与旧版单测线实现逐位相同(Mothra 500x100 + 100m 窗口宽即此例)。

    ⚠ D16 起**不是匀速**:任务层每拍传 `hold=True` 就原地停船(速度置 0)。
      停不停由任务层按 Leader 自己的 DR 推算判(见 `contracts/mission.py` §8),
      本类只负责执行,不做判断 —— 免得运动学层去碰它不该知道的队友状态。
    """

    def __init__(self, cfg: MissionConfig, path: np.ndarray):
        self.cfg = cfg
        p = np.asarray(path, dtype=FLOAT).reshape(-1, 2)
        if p.shape[0] < 2:
            raise ValueError(f"Leader 路径至少需要 2 个点(1 条测线),收到 {p.shape[0]} 个")
        self.path = p
        self._seg = 0                       # 当前所在折线段 [path[_seg], path[_seg+1]]
        self._seg_progress_m = 0.0          # 本段已走过的距离
        self.distance_m = 0.0
        self.hold_time_s = 0.0              # 累计停船时长(诊断用)

    def _seg_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.path[self._seg], self.path[self._seg + 1]

    def _seg_len(self) -> float:
        a, b = self._seg_endpoints()
        return float(np.hypot(*(b - a)))

    @property
    def north(self) -> float:
        a, b = self._seg_endpoints()
        L = self._seg_len()
        frac = 0.0 if L <= 1e-12 else self._seg_progress_m / L
        return float(a[0] + frac * (b[0] - a[0]))

    @property
    def east(self) -> float:
        a, b = self._seg_endpoints()
        L = self._seg_len()
        frac = 0.0 if L <= 1e-12 else self._seg_progress_m / L
        return float(a[1] + frac * (b[1] - a[1]))

    @property
    def psi(self) -> float:
        """当前航向:从 North(+x)起、向 East(+y)为正(与 `msim/contracts/types.py` 同口径)。"""
        a, b = self._seg_endpoints()
        d = b - a
        if np.hypot(*d) <= 1e-12:
            return 0.0
        return float(np.arctan2(d[1], d[0]))

    @property
    def pose(self) -> np.ndarray:
        return np.array([self.north, self.east, self.psi], dtype=FLOAT)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.north, self.east], dtype=FLOAT)

    @property
    def finished(self) -> bool:
        """走完折线最后一段。"""
        return (self._seg >= len(self.path) - 2
                and self._seg_progress_m >= self._seg_len() - 1e-9)

    @property
    def dist_to_segment_end_m(self) -> float:
        """距当前折线段终点还有多少米。多车道时 = 距下一次**折返/换道**还有多远。

        判据 D(折返丢窗)要用它:折返瞬间窗口整体转向,原本在 Leader 身后的目标
        会一拍之内变成"在身前",直接出局 —— 那不是从后沿滑出去的,判据 C 感知不到。
        """
        return max(0.0, self._seg_len() - self._seg_progress_m)

    @property
    def has_next_segment(self) -> bool:
        """后面还有段可走(即这次到段尾是**折返**, 不是任务结束)。"""
        return self._seg < len(self.path) - 2

    def step(self, dt_s: float, *, hold: bool = False) -> None:
        """推进一拍。`hold=True` 时原地停船:不前进、不计路程,只累计停船时长。

        一拍内可能走完当前段并接着走下一段(车道很短、dt 较大时);用 while 而不是
        单次 min() 裁剪,否则换道瞬间会白白损失这一拍剩下的行程。
        """
        if self.finished:
            return
        if hold:
            self.hold_time_s += float(dt_s)
            return
        remaining = self.cfg.leader_speed_mps * dt_s
        while remaining > 1e-12 and not self.finished:
            seg_remaining = self._seg_len() - self._seg_progress_m
            adv = min(remaining, seg_remaining)
            self._seg_progress_m += adv
            self.distance_m += adv
            remaining -= adv
            if self._seg_progress_m >= self._seg_len() - 1e-9 and self._seg < len(self.path) - 2:
                self._seg += 1
                self._seg_progress_m = 0.0

    def window(self, window_cfg) -> WindowRegion:
        """当前观测窗口。用 msim 的 `get_window` —— 它的语义就是"窗口在 Leader 正后方",
        即前边界与 Leader 位置重合(规格 §3);换道横移时窗口随瞬时航向 psi 转向,
        是 `get_window`/`_local_coords` 既有的旋转语义(msim 已验收实现),不是本类特例。"""
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
