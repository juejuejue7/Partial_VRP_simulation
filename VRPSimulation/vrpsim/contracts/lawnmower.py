"""[第0层][A0 冻结草案 D14] Lawnmower 全覆盖基线的配置契约。

==================================================================
这个基线是干什么的
==================================================================
用来给"Leader 广域探査 + Follower 请求式 VRP 観測"(`contracts/mission.py`)做
**对照组**:同一片场地、同一批目标、同样 3 台 AUV,但取消异种分工 ——
3 台全部是 Follower(相机机),把场地按测线切成 3 份各做 lawnmower 全覆盖扫描。

对比的量是**任务完成时刻**(见下方"完成时刻的口径")。

==================================================================
"配置与 VRP 场景相同"是**结构保证**,不是人工同步的数字
==================================================================
本配置**内嵌**一份 `MissionConfig`(字段 `base`),运动学参数(速度、停留、
到达容差、时钟步长)与场景(`sim`:裁切、waypoint、概率场)全部从它取,
本文件一个都不重新声明。改 `MissionConfig` 的速度,两个场景同时改。

唯一**故意不同**的一项是 `dwell_at_turn`:VRP 场景里 `dwell_at_projection=True`
(规格 §5 字面:下发序列里的点到了都停 5 s);lawnmower 的测线拐点根本不是
下发的观测点,只是折线几何,所以不停。见 `vehicle_cfg`。

==================================================================
覆盖判据(确定量,不是感知模型)
==================================================================
车辆沿测线走,横向 ±`swath_width_m`/2 之内的目标算被相机幅宽扫到。这是**纯几何**
的确定量,与 `mission.py` 一样**没有**检出概率 / 漏检 / 误分类模型(CLAUDE.md 硬纪律 4:
命名与注释须写明是确定量,不许在论文里写成"检出")。

扫到之后在测线上、该目标 North 处**原地**停 `dwell_time_s` 秒模拟観測 ——
车辆**不离开测线**去目标正上方。这样航程 = 纯测线长度 + 换线横移,不因目标分布绕路,
是对 lawnmower **最有利**的记法。

==================================================================
测线布置
==================================================================
场地 East 跨度按 `n_vehicles` 均分成互不重叠的条带,每台一条带:

    strip_k = [E_max·k/n,  E_max·(k+1)/n)

带内测线条数 `n_lanes = ceil(strip_width / swath_width_m)`,测线在带内**均匀**
铺开(间距 = strip_width / n_lanes ≤ swath_width_m):

    lane_east[i] = strip_lo + (i + 0.5)·spacing,   i = 0 .. n_lanes-1

⇒ 相邻幅宽必然搭接、带内无缝隙,且最外侧测线的幅宽只溢出带外 (swath-spacing)/2。
   每个目标按其 East 落在哪条带**唯一**归属一台车(不靠先到先得),故不会重复观测。
   `lawnmower.build_vehicle_plans` 会断言"每个目标恰好被安排一次",违反即当场炸。

测线沿 **North**(长边)方向,牛耕式(boustrophedon)往复,不含空回航。
起点 = 各自第一条测线的南端 `(0, lane_east[0])`。

==================================================================
完成时刻的口径(跨场景可比的唯一定义)
==================================================================
    t_complete = 全域被扫遍 **且** 全部目标完成観測 的时刻

- lawnmower:= 最后一台车走完自己最后一条测线的时刻。
  注意**不是**"最后一个目标被観測的时刻" —— 全覆盖扫描要走完才能宣称扫完,
  哪怕最后一段测线上一个目标都没有。
- VRP 场景:= max(Leader 走完测线的时刻, 最后一个目标完成観測的时刻)。
  见 `mission.MissionResult.t_complete_s`。

两边都**不是** `duration_s`:`MissionResult.duration_s` 含 `settle_time_s=60 s`
的收尾余量,拿它比时间会凭空给 VRP 场景加 60 s 的账。

==================================================================
待裁决项(A0 提交人工确认,未确认前按默认值跑)
==================================================================
【L1】`swath_width_m = 6.0` 取自 msim `SensorConfig.footprint_lateral_m`(横向 footprint)。
      msim 另有"走廊宽 = 0.9 × footprint = 5.4 m"的口径(为覆盖率留搭接余量)。
      取 6.0 是对 lawnmower **最有利**的一档(测线更少、更快);取 5.4 会让每台从
      6 条测线变 7 条,时间再涨约 1/6。默认取 6.0,理由是"宁可让对照组占便宜"。
【L2】不建模转弯耗时(与 VRP 场景的 Follower 一致:纯匀速直线趋近,无转弯罚时)。
      lawnmower 的转弯次数远多于 VRP 路线,不罚转弯同样是让对照组占便宜。
【L3】3 台全部是相机机 ⇒ 幅宽只能是相机的 6 m,不能用 Leader 声呐的 100 m 窗口。
      这正是被对比的东西(异种混成 vs 同构全覆盖),不是给对照组穿小鞋 ——
      但论文里必须写清:100 m 是**声呐探査**幅宽,6 m 是**相机観測**幅宽,两者
      不可互换(声呐不能做目標観測,相机做不到 100 m 幅宽)。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .mission import MissionConfig

__all__ = ["LawnmowerConfig", "SWATH_FROM_MSIM_FOOTPRINT_M"]

# msim `SensorConfig.footprint_lateral_m` 的值。写成常数是为了让"这个数不是自造的"
# 一眼可见;它与 msim 的一致性由 tests/test_lawnmower.py 钉住。
SWATH_FROM_MSIM_FOOTPRINT_M: float = 6.0


@dataclass(frozen=True)
class LawnmowerConfig:
    """3 台 Follower 分区 lawnmower 全覆盖基线的全部输入。"""

    # --- 与 VRP 场景共用的同一份参数(场景 / 运动学 / 时钟) ---------------
    # ⚠ 不要在本类里另加速度、停留时长等字段 —— 那会让两个场景的参数产生第二个
    #   真相源,"配置相同"就退化成靠人工同步,迟早对不上。
    base: MissionConfig = field(default_factory=MissionConfig)

    # --- lawnmower 特有 ---------------------------------------------------
    n_vehicles: int = 3
    swath_width_m: float = SWATH_FROM_MSIM_FOOTPRINT_M   # 【待裁决 L1】
    boustrophedon: bool = True           # True = 牛耕式往复;False = 每条测线都从南端起(含空回航)
    dwell_at_turn: bool = False          # 测线拐点不是観測点,不停留(与 VRP 的唯一故意差异)

    def __post_init__(self) -> None:
        if self.n_vehicles < 1:
            raise ValueError(f"n_vehicles 必须 ≥ 1,收到 {self.n_vehicles}")
        if self.swath_width_m <= 0.0:
            raise ValueError(f"swath_width_m 必须 > 0,收到 {self.swath_width_m}")

    # --- 从 base 派生(单一真相源) ---------------------------------------
    @property
    def sim(self):
        """场景 = VRP 场景同一份 `MothraSimConfig`(同样的 D7 裁切与 22 个目标)。"""
        return self.base.sim

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
    def vehicle_cfg(self) -> MissionConfig:
        """喂给 `agents.Follower` 的运动学配置。

        由 `base` 直接派生,只翻 `dwell_at_projection` —— lawnmower 用
        `WP_PROJECTION` 标记测线拐点(不占用、不计覆盖),拐点不该停留。
        其余每一项(速度 / 停留 / 到达容差)都与 VRP 场景**逐位相同**。
        """
        return replace(self.base, dwell_at_projection=self.dwell_at_turn)
