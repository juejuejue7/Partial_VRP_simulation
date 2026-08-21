"""
第2a层物理契约:rollout 窄接口、Leader actor、相机传感器模型。

时间尺度隔离(architecture §4.1):2b/RL 只通过 rollout 向 2a 索要"起点+waypoint 序列
→ (能耗, 用时, 末位姿)",不碰 dt 级积分细节。

两种 rollout 实现(非两套代码,而是同一 Protocol 的两实现):
- Level 1(修士训练默认):路径长度 + 转弯惩罚(能耗)、长度/cruise + 转弯时间近似 + 停留(用时)。
  纯几何、确定、快;不调用 AUV 类。state 的 t_avail / 事件调度全用此口径。
- Level 2(验证旁路,Q2):用根目录 AUV 类 step_motion 细步长积分,能耗/时间均物理真实(含转弯)。
  仅用于高保真分析 / 第②层真机对接,不进训练 reward。

Q3 落地:已撤销"转弯不计时间"。Level 1 用时含转弯近似项;Level 2 积分天然含转弯。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import RolloutConfig, SensorConfig
from .types import Pose, Waypoint


# ======================================================================
# rollout 返回结构(Q1)
# ======================================================================
@dataclass
class RolloutResult:
    """一段 waypoint 序列执行后的代价汇总。"""
    energy_J: float                       # 整段总能耗(进 reward 能耗项 / 均衡)
    duration_s: float                     # 整段总用时(state t_avail / 事件调度口径)
    end_pose: Pose                        # 末端 [x_N, y_E, psi],供下一段接续
    per_wp_energy_J: List[float] = field(default_factory=list)   # 每 waypoint 段能耗
    per_wp_duration_s: List[float] = field(default_factory=list)  # 每段用时(价值"到达那刻"结算时刻)
    trajectory: Optional[np.ndarray] = None  # (T,3) 细采样,默认 None;可视化时开启(Q1)


@runtime_checkable
class RolloutEngine(Protocol):
    """窄接口:waypoint 序列 → 代价。Level1RolloutEngine / Level2RolloutEngine 两实现。"""

    def execute(self, start_pose: Pose, waypoints: Sequence[Waypoint]) -> RolloutResult:
        """从 start_pose 依次经过 waypoints,返回 RolloutResult。

        实现持有所需配置(Level 1:RolloutConfig;Level 2:AUV 工厂 + dt),
        故本签名对两实现统一。trajectory 是否填充由实现的开关决定(默认 None)。

        转角口径(Level 1,v1.0.1 澄清;见 DECISIONS §v1.0.1):
        设位置序列 p=start_pose.position, w_0..w_{n-1},段方向
        s_0=dir(p→w_0), s_k=dir(w_{k-1}→w_k);**只累计队列内相邻段间转角**
        Σ|Δψ| = Σ_{k=1}^{n-1} |wrap(s_k − s_{k-1})|(n−1 个;n=1 时为 0)。
        **不计** start_pose.psi→s_0 的首段转入角(Level 1 用时/能耗的转角项均不含它)。
        理由:与 resolve_available_time(只拿 Position、无朝向)严格同口径。
        start_pose.psi 仅供 Level 2 积分初值与 end_pose 朝向输出,不进 Level 1。
        """
        ...


# ======================================================================
# 相机传感器模型(Q15;被 task.corridor 与 task.value 依赖,早冻结)
# ======================================================================
@dataclass(frozen=True)
class CameraSensorModel:
    """Follower 相机:检出率/虚警率/footprint 几何。"""
    p_d: float = 0.95
    p_fa: float = 0.05
    footprint_along_m: float = 4.0
    footprint_lateral_m: float = 6.0

    @property
    def corridor_width_m(self) -> float:
        """走廊宽度 = footprint 横向宽度(×1.0,与 grid cell 严格对齐;§v1.0.7)。

        §v1.0.7(人类裁定):原 ×0.9 保守扩带(DECISIONS §v1.0.2)取消。相机探测范围按
        "实际探测精度"建模为以 Follower 为中心、与 grid cell 等宽的方形 footprint
        ⇒ 走廊宽 = footprint_lateral(取 footprint=cell=2m 时走廊宽=2.0m=1 个 cell)。
        """
        return self.footprint_lateral_m


# ======================================================================
# Leader actor(Q16:用 AUV 类 + lawnmower 动力学推进;能耗不进 reward)
# ======================================================================
@runtime_checkable
class LeaderActorProtocol(Protocol):
    """Leader = 用 HeterogeneousAUV("leader") 实例沿 lawnmower waypoint 动力学推进。

    env 维护连续全局时钟:每次推进到事件时刻,调 advance(dt) 把 Leader 积分到该时刻;
    window 由 Leader 的**实际动力学位姿**生成。能耗累计但**不进 reward**(Q16)。
    """

    @property
    def pose(self) -> Pose:
        """Leader 当前动力学位姿 [x_N, y_E, psi]。"""
        ...

    @property
    def energy_J_cumulative(self) -> float:
        """累计能耗(旁路统计,不进 reward)。"""
        ...

    @property
    def finished(self) -> bool:
        """是否已走完整条 lawnmower 测线(episode 终止判据之一)。"""
        ...

    def reset(self) -> None:
        ...

    def advance(self, dt: float) -> None:
        """沿 lawnmower 用 AUV 动力学推进 dt 秒(内部按 sim_dt 细分积分)。"""
        ...


# ======================================================================
# Follower 物理封装(承诺队列住 mdp 层,不在此)
# ======================================================================
@runtime_checkable
class FollowerProtocol(Protocol):
    """Follower 物理状态 + 传感器。仅暴露物理量;"已分配未走完 waypoint" 住 mdp.commitment。"""

    sensor: CameraSensorModel

    @property
    def pose(self) -> Pose:
        ...

    @property
    def energy_norm(self) -> float:
        """剩余电量归一化 ∈ [0,1](AUV 类 energy_norm)。"""
        ...

    def reset(self, pose: Pose) -> None:
        ...
