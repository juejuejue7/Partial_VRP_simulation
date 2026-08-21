"""[第0层][A0 冻结 D19] 二段式全局VRP 基线的配置契约。

==================================================================
这个基线是干什么的
==================================================================
给"Leader 広域探査 + Follower 逐次 VRP 観測"(`contracts/mission.py`,本文提案)
做**第二个对照组** —— 与 lawnmower 那个"取消异种分工"的对照不同,
这里**保留完全相同的异种分工与硬件**,只把**探査与観測从并行改成串行**:

    阶段1  Leader 走完整条测线,探明全部目标           [0, t_survey)
           Follower 在起点待机(STATUS_IDLE)
    阶段2  用**完整**目标表解**一次**全局 min-max VRP  [t_survey, t_finish)
           两台 Follower 各自开环执行,不重规划、不通信

⇒ 局部VRP 与本基线之间**唯一的变量就是「并行 vs 串行」**。这是一个干净的
  受控对比,也正是本文提案的卖点所在。

==================================================================
台数:三个场景都是 3 台,不存在"吃亏一台"
==================================================================
    局部VRP    1 Leader + 2 Follower = 3 台   探査与観測**并行**
    二段式     1 Leader + 2 Follower = 3 台   探査与観測**串行**
    lawnmower  3 台(全部相机机)                探査即観測

D19 人工裁决里的"执行全局VRP 用 2 台"指的是**阶段2 执行観測的机数**
(与本文提案的 Follower 数对齐),阶段1 仍然需要 Leader ——
声呐才能做目標探査,相机做不到 100 m 幅宽(`contracts/lawnmower.py` 待裁决 L3)。

==================================================================
"配置与另外两个场景相同"是**结构保证**,不是人工同步的数字
==================================================================
与 `LawnmowerConfig` 同一套做法:本配置**内嵌**一份 `MissionConfig`(字段 `base`),
运动学(速度、停留、到达容差、时钟步长)与场景(裁切、waypoint、概率场)全部从它取,
本文件一个都不重新声明。改 `MissionConfig` 的速度,三个场景同时改。

⚠ 特别地**不另立 `n_vehicles`**:阶段2 的机数与起点都由
`base.follower_starts_ned` 唯一决定。另立字段就会出现第二个真相源,
迟早与 `MissionConfig` 对不上(D15 在 `06_run_lawnmower.py --dwell` 上踩过一次)。

==================================================================
两条有意选定的口径(不是疏漏,写进 D19)
==================================================================
【T1】**Follower 在阶段1 全程待机于起点**,不做预置前移。
      一个"聪明的"二段式会让 Follower 边等边前移到场地中部,但那已经开始向
      本文提案靠拢,会模糊掉被测的那个变量。本基线按**教科书串行**实现。
      ⚠ 这对二段式最不利,须在论文里写明。实测预置前移最多省下
        Follower 从 North=0 走到第一个目标的时间(约 300 s),结论不翻转。

【T2】**阶段2 不需要声学通信**:全局路线在下水前/母船上算好,一次性装载。
      这是二段式**相对本文提案的优势**,通信族记 0 而不是抹掉 ——
      对照组占的便宜要如实记。

==================================================================
完成时刻的口径(与另外两个场景同一把尺子)
==================================================================
    t_complete = 全域被扫遍 **且** 全部目标完成観測 的时刻
               = max(阶段1 结束, 最后一个目标完成観測)
               = t_survey + t_observe   (串行 ⇒ 后者必然更晚)

与 `mission.MissionResult.t_complete_s` / `lawnmower.LawnmowerResult.t_complete_s`
口径一致。三边都**不是** `duration_s`(那含 `settle_time_s` 的收尾余量)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from .mission import MissionConfig

__all__ = ["TwoPhaseConfig"]


@dataclass(frozen=True)
class TwoPhaseConfig:
    """二段式全局VRP 基线的全部输入。"""

    # --- 与另外两个场景共用的同一份参数(场景 / 运动学 / 时钟) -------------
    # ⚠ 不要在本类里另加速度、停留时长、机数等字段 —— 那会让三个场景的参数产生
    #   第二个真相源,"配置相同"就退化成靠人工同步,迟早对不上。
    base: MissionConfig = field(default_factory=MissionConfig)

    # --- 二段式特有 -------------------------------------------------------
    # 恒 False。"全局只规划一次"正是本基线的定义 —— 允许重规划它就变成本文提案了。
    # 保留成字段而非硬编码,是为了让 `test_no_replanning_in_phase_two` 有东西可断言。
    replan: bool = False

    def __post_init__(self) -> None:
        if self.replan:
            raise ValueError(
                "二段式基线不允许重规划(replan 必须为 False):"
                "一旦允许,它就退化成本文提案的局部VRP,对比失去意义。")

    # --- 从 base 派生(单一真相源) ---------------------------------------
    @property
    def sim(self):
        """场景 = 另外两个场景同一份 `MothraSimConfig`(同样的 D7 裁切与 22 个目标)。"""
        return self.base.sim

    @property
    def n_vehicles(self) -> int:
        """阶段2 执行観測的机数 = Follower 数(D19 裁决 = 2)。**由起点表派生,不另立字段。**"""
        return self.base.n_followers

    @property
    def vehicle_starts_ned(self) -> Tuple[Tuple[float, float], ...]:
        return self.base.follower_starts_ned

    @property
    def leader_speed_mps(self) -> float:
        return self.base.leader_speed_mps

    @property
    def follower_speed_mps(self) -> float:
        return self.base.follower_speed_mps

    @property
    def dwell_time_s(self) -> float:
        return self.base.dwell_time_s

    @property
    def dt_s(self) -> float:
        return self.base.dt_s

    @property
    def max_mission_time_s(self) -> float:
        return self.base.max_mission_time_s

    @property
    def solver(self) -> str:
        return self.base.solver

    @property
    def vrp_time_limit_s(self) -> float:
        return self.base.vrp_time_limit_s

    @property
    def vehicle_cfg(self) -> MissionConfig:
        """喂给 `agents.Follower` 的运动学配置。

        直接就是 `base` —— 阶段2 的序列里**没有投影点**(全局 VRP 一次派完全部
        真实目标),所以 `dwell_at_projection` 无从生效,不需要像 lawnmower 那样
        翻任何开关。速度 / 停留 / 到达容差与另外两个场景**逐位相同**。
        """
        return self.base
