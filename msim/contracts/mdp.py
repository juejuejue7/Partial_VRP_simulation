"""
第3层 MDP 契约:承诺队列(+解析量)、obs 组装、动作解码、Gymnasium env 协议。

事件驱动 ↔ 标准 Gym 适配:step(action) 的"一步" = 为当前空闲 Follower 做一次决策。
env 内部维护连续全局时钟 + 事件队列(谁先到点/窗口推进谁先触发):
- Follower 用 Level 1 解析时间(rollout.duration_s)确定到点事件时刻;
- Leader(Q16)用 sim_dt 积分推进到事件时刻,window 由其实际动力学位姿生成;
- step 推进到下一个需决策事件,组装该 Follower 的 obs,对外返回标准五元组。
忙着执行的 Follower 当拍不参与决策(天然处理"暂时不可用");一次只决策一台。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Tuple, runtime_checkable

import numpy as np

from .config import RolloutConfig, SimConfig
from .geometry import GridProtocol, WindowRegion
from .types import CoarseMap, Position, Waypoint

if TYPE_CHECKING:  # 不在运行期硬依赖 gymnasium(骨架 import 不需要它)
    import gymnasium as gym


# ======================================================================
# 承诺队列(Q14;住 mdp 层,不在 AUV 类)+ 解析确定量(纪律4)
# ======================================================================
@dataclass
class FollowerCommitment:
    """某台 Follower 已分配未走完的 waypoint 序列 + 执行游标。"""
    queue: List[Waypoint] = field(default_factory=list)  # 世界坐标 waypoint 序列
    cursor: int = 0                                       # 已执行到队列第几个(< cursor 的已走完)

    @property
    def is_busy(self) -> bool:
        """队列中仍有未走完 waypoint ⇒ 忙(当拍不参与决策)。纯数据派生。"""
        return self.cursor < len(self.queue)


def resolve_remaining_distance(current_pos: Position,
                               commitment: FollowerCommitment) -> float:
    """剩余执行距离 d_i_rem(承诺队列**解析**计算的确定量,非估计):

        d = ‖pos − w_cursor‖ + Σ_k ‖w_k − w_{k+1}‖   (队列内相邻点)
    """
    raise NotImplementedError  # [A5 波次2]


def resolve_available_time(current_pos: Position, commitment: FollowerCommitment,
                           cfg: RolloutConfig) -> float:
    """预计可用时间 t_i_avail(解析确定量,非估计):

        t = d_rem / cruise_speed
          + Σ |Δψ_k| / nominal_yaw_rate         (转弯时间近似;Q3:转弯计入时间)
          + dwell_time × 剩余 waypoint 数
    与 Level 1 rollout.duration_s 口径一致。

    Σ|Δψ_k| 定义(v1.0.1 澄清;见 DECISIONS §v1.0.1):设 p=current_pos、
    队列剩余点 w_cursor..w_{last},段方向 s_0=dir(p→w_cursor)、
    s_k=dir(w_{k-1}→w_k);**只累计相邻段间转角** Σ_{k≥1} |wrap(s_k − s_{k-1})|。
    因本函数只有 Position、无当前朝向,故**不含**"当前朝向→s_0"的首段转入角;
    Level 1 execute 同口径忽略 start_pose.psi 的首段转入角,二者逐项一致。
    """
    raise NotImplementedError  # [A5 波次2]


# ======================================================================
# state —— obs 组装(Q4:Dict 结构)
# ======================================================================
@dataclass
class FollowerObsState:
    """单台 Follower 喂入 obs 的归一化状态。"""
    energy_norm: float          # 剩余电量 ∈ [0,1]
    pos_norm: np.ndarray        # [x_N, y_E] / 窗口尺度,∈ [0,1]^2
    remaining_distance_norm: float   # d_rem / 窗口尺度
    available_time_norm: float       # t_avail / max_episode_time_s


def assemble_obs(coarse_map: CoarseMap,
                 self_state: FollowerObsState,
                 teammate_state: FollowerObsState) -> Dict[str, np.ndarray]:
    """组装 Dict 观测(Q4)。one-at-a-time 下"为哪台决策"由 self/teammate 槽位隐式编码(N=2)。

    返回:
        {
          "map": float32 (Hc, Wc),   # 窗口**残余**先验概率场降采样到粗网格(§v1.1.6:已扣除 planned)
          "vec": float32 (10,),       # [self: e,dx,dy,d_rem,t_avail | teammate: 同 5 项]
        }
    全部归一化。具体 vec 布局以本 docstring 为冻结定义。

    §v1.1.6:map 单元 = `field[center]·(1−planned[center])`(粗格中心采样;planned 硬置 0)——
    即"残余"先验(走廊扫过即衰减到 0),**与 reward 的 dense_mass 同口径**(reward 亦只计未 planned)。
    目的:消观测混叠——让策略"看得见"哪条带已被扫,从而两台分带、单台推进到下一带。
    planned 是布尔覆盖掩码(修士也更新,见 Q6 planned_coverage 与 belief 分离),**非** belief 回写;
    field 逐位不变、NullUpdater 仍开环;obs 仍不含 GT。map ∈ [0,1]、形状不变,不破坏 obs/action space。
    """
    raise NotImplementedError  # [A5 波次2]


# ======================================================================
# action —— 粗网格标量索引 → 世界 waypoint(纯几何)
# ======================================================================
def decode_action(action_idx: int, window: WindowRegion,
                  grid: GridProtocol, cfg: SimConfig) -> Waypoint:
    """离散动作 action_idx ∈ [0, Hc*Wc) → 窗口内该粗格中心的世界 waypoint。

    纯几何映射:不做可达性检查、避障、吸附(architecture §4.5)。
    内部委托 geometry.coarse_index_to_world。
    """
    raise NotImplementedError  # [A5 波次2]


# ======================================================================
# coop_survey_env —— 标准 Gymnasium 接口
# ======================================================================
@runtime_checkable
class CoopSurveyEnvProtocol(Protocol):
    """事件驱动协同探查 env。实现须继承 gymnasium.Env(coop_survey_env.py,A5)。

    - observation_space: gym.spaces.Dict({"map": Box, "vec": Box})
    - action_space:      gym.spaces.Discrete(Hc*Wc)
    验收(harness):通过 Gymnasium env_checker;忙着的 Follower 不被决策;一次只决策一台。
    """

    observation_space: "Any"
    action_space: "Any"

    def reset(self, *, seed: int | None = None,
              options: Dict[str, Any] | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """组装首个空闲 Follower 的 obs。返回 (obs, info)。"""
        ...

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """为当前空闲 Follower 解码动作→入承诺队列→推进事件时钟→结算到达段价值
        →组装下一个空闲 Follower 的 obs。返回标准五元组 (obs, reward, terminated, truncated, info)。

        reward(§v1.1.4:先验概率质量覆盖 + 真检出 − 均衡 − 软冗余,见 RewardConfig 公式):
            dense_mass = Σ_{走廊∩窗口∩未planned} field[cell]          # 已解析先验概率质量(GT-free)
            reward = value_weight·(dense_mass/value_norm_ref)         # dense:缩放不截断
                   + detect_weight·n_new_detections                  # sparse:本步新覆盖 GT 目标格数(去重)
                   − balance_weight·(balance_penalty(per_follower_energy)/balance_norm_ref)
                   − redundancy_weight·n_redundant。                  # 软冗余:与其它 Follower 覆盖重叠格数
        能耗项只基于 per_follower_energy(Follower 台,不含领航);GT 仅入 reward(训练信号),策略 obs 不含 GT。
        terminated = Leader 走完 lawnmower 后 drain 至工作完成/无进展;
        truncated  = 达 max_episode_time_s 或任一 Follower 电量耗尽。
        """
        ...
