"""[第1层] Leader 测线 + 后方滑动观测窗口(D8)。

窗口语义(人工裁决 2026-08-09):
    **沿测线滑动**,不是固定分块。Leader 沿 lawnmower 前进,窗口是它正后方
    look_back x width 的矩形;窗口每推进 `advance_threshold_m` 触发一次**重解**。
    重解只涉及**尚未被 Follower occupied 的路径点**;新序列下发后覆盖旧序列。

    ⚠ 本模块只提供**几何**:测线、窗口位姿序列、每个窗口里有哪些目标。
      "哪些点已被 occupied / 怎么重解 / 怎么下发覆盖"属任务调度层,下一步实现,
      不在这里。这样几何可以脱离调度单独验证(architecture_blueprint §1 原则一)。

窗口几何全部复用 msim 已验收实现(`msim.geometry.window`),本模块不重写 ——
只是把"沿单条 North 测线前进"这个特例包装成便于枚举的形式。

D9(人工裁决):窗口几何**严格规则**,不按目标分布调边界。Leader 测线与概率场解耦
(project_summary §3),所以窗口从 North=0 起按固定步长推进,哪怕这会把某个复合体
切到两次重解里去(实测 N=400 附近的 Faulty Towers 就会被切开)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from msim.contracts.config import WindowConfig, WorldConfig
from msim.contracts.geometry import WindowRegion
from msim.contracts.types import FLOAT
from msim.env_static.leader_path import lawnmower_waypoints
from msim.geometry.window import get_window, window_cells, window_contains_world

__all__ = ["WindowSnapshot", "leader_track", "window_anchors", "enumerate_windows",
           "targets_in_window", "window_occupancy"]

# Leader 沿 +North 前进时的航向(psi 从 North 起、向 East 为正)
_PSI_NORTH = 0.0


@dataclass(frozen=True)
class WindowSnapshot:
    """一次重解时刻的窗口快照(纯几何,不含任务状态)。"""
    index: int                      # 第几次触发,从 0 起
    leader_pose: np.ndarray         # (3,) [x_N, y_E, psi]
    region: WindowRegion
    target_idx: np.ndarray          # 落在本窗口内的目标在 dataset 里的下标

    @property
    def leader_north_m(self) -> float:
        return float(self.leader_pose[0])

    @property
    def north_span(self) -> Tuple[float, float]:
        """窗口在 North 上的区间 [后沿, 前沿](已按场地下界 0 截断)。"""
        front = self.leader_north_m
        return (max(0.0, front - self.region.look_back_m), front)


def leader_track(world: WorldConfig, lane_spacing_m: float) -> List[np.ndarray]:
    """Leader lawnmower 测线端点(复用 msim 实现,不重写)。

    lane_spacing == 窗口宽 == 场地 East 跨度时,退化为单条测线(East = spacing/2),
    这正是 D7 裁切后的情形:500x100 的场地 + 100 m 窗口 ⇒ 1 条 East=50 的测线。
    """
    return [np.asarray(p, dtype=FLOAT) for p in lawnmower_waypoints(world, lane_spacing_m)]


def window_anchors(world: WorldConfig, window: WindowConfig,
                   *, include_start: bool = False) -> np.ndarray:
    """重解触发时刻的 Leader North 位置序列。

    从 North=advance_threshold 起每 advance_threshold 触发一次,直到走完测线;
    末端若不整除,补一个恰好在场地北界的触发点(保证最后一段不被漏掉)。

    `include_start` 默认 False:North=0 处窗口面积为零(还什么都没扫到),
    是个退化窗口。任务层若需要"出发即规划"的空拍,自己在 t=0 加一次空操作即可。
    """
    step = float(window.advance_threshold_m)
    if step <= 0.0:
        raise ValueError(f"advance_threshold_m 须 >0,收到 {step}")

    end = float(world.x_max_m)
    anchors = [0.0] if include_start else []
    n = step
    while n < end - 1e-9:
        anchors.append(n)
        n += step
    anchors.append(end)
    return np.asarray(anchors, dtype=FLOAT)


def enumerate_windows(world: WorldConfig, window: WindowConfig,
                      targets_ned: Optional[np.ndarray] = None,
                      *, east_m: Optional[float] = None,
                      include_start: bool = False) -> List[WindowSnapshot]:
    """枚举整条测线上的所有窗口快照(单条 North 测线的特例)。

    east_m 默认取场地 East 中线(= 单测线的 lane 位置)。窗口宽 == East 跨度时,
    窗口横向恰好铺满全场,East 具体取值不影响隶属判定 —— 但仍显式给出,免得
    将来改成多车道时静默出错。
    """
    if east_m is None:
        east_m = float(world.y_max_m) / 2.0

    snaps: List[WindowSnapshot] = []
    for i, n in enumerate(window_anchors(world, window, include_start=include_start)):
        pose = np.array([float(n), float(east_m), _PSI_NORTH], dtype=FLOAT)
        region = get_window(pose, window)
        idx = (targets_in_window(region, targets_ned)
               if targets_ned is not None else np.zeros(0, dtype=int))
        snaps.append(WindowSnapshot(index=i, leader_pose=pose, region=region,
                                    target_idx=idx))
    return snaps


def targets_in_window(region: WindowRegion, targets_ned: np.ndarray) -> np.ndarray:
    """落在窗口内的目标下标(用 msim 的隶属判定,口径与栅格覆盖一致)。"""
    tg = np.asarray(targets_ned, dtype=FLOAT).reshape(-1, 2)
    hit = [i for i, p in enumerate(tg) if window_contains_world(region, p)]
    return np.asarray(hit, dtype=int)


def window_occupancy(snaps: Sequence[WindowSnapshot]) -> np.ndarray:
    """每个窗口里的目标数,(len(snaps),) int。诊断用。"""
    return np.asarray([len(s.target_idx) for s in snaps], dtype=int)


def first_seen_window(snaps: Sequence[WindowSnapshot], n_targets: int) -> np.ndarray:
    """每个目标**第一次**进入窗口的快照序号;从未进入者为 -1。

    这是"永久错过"判定与重解调度的几何前提:目标只有进过窗口才可能被分配。
    (project_summary §7:"永久错过"仅针对滑出窗口前从未被分配的高概率区域。)
    """
    first = np.full(n_targets, -1, dtype=int)
    for s in snaps:
        for i in s.target_idx:
            if first[i] < 0:
                first[i] = s.index
    return first
