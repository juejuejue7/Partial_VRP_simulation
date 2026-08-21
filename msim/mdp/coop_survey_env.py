"""[第3层][A5] 事件驱动协同探查 Gymnasium env。实现 contracts.mdp.CoopSurveyEnvProtocol。

事件驱动 ↔ 标准 Gym 适配(architecture §4.3):
- step(action) 的"一步" = 为**当前空闲** Follower 做**一次**决策(忙者不参与、一次一台)。
- env 内部维护**连续全局时钟** + 隐式事件队列:每台 Follower 持一个"空闲时刻"
  free_time_i(其承诺段按 Level 1 解析用时算出的到点时刻);step 末把全局时钟推进到
  下一个 Follower 空闲时刻 → 该台成为下一决策者。Leader(Q16)用 sim_dt 动力学积分推进
  到该时刻。
- §v1.1.5:窗口为**离散固定块**(非 Leader 连续滑窗)。沿 North 每 look_back 一块,所有决策
  用当前活动块的固定窗口;Leader 连续位姿只门控"哪些块已扫"。详见类 docstring。
- 对外严格返回 Gymnasium 五元组 (obs, reward, terminated, truncated, info)。

层间依赖(对契约编程;上游实现按 §v1.0.2 路径 import):
  geometry: Grid / get_window / window_cells / coarse_grid_shape / coarse_index_to_world
  env_static: default_instance(生成耦合 (targets, field) 实例供 reset)
  physics: LeaderActor(动力学推进)/ Follower(物理量)/ Level1RolloutEngine(解析用时/能耗)
  task: corridor_cells / mark_planned / balance_penalty / NullUpdater
  mdp: decode_action / assemble_obs / FollowerObsState / FollowerCommitment / resolve_*

reward(§v1.1.4:先验概率质量覆盖 + 真检出 − 均衡 − 软冗余;见 contracts.config.RewardConfig):
    dense_mass     = Σ_{走廊∩窗口∩未planned} field[cell]          # 已解析先验概率质量(GT-free)
    n_new_detect   = 本步首次确定性覆盖的 GT 目标格数(detected 去重)
    n_redundant    = 本步覆盖格 ∩ 其它 Follower 已覆盖格 的数量(软冗余,不终止)
    dense_term     = value_weight · (dense_mass / value_norm_ref)  # 缩放**不截断**(点2)
    detect_term    = detect_weight · n_new_detect
    balance_term   = balance_weight · (balance_pen / balance_norm_ref)
    redundancy_term= redundancy_weight · n_redundant
    reward = dense_term + detect_term − balance_term − redundancy_term
所有权重/基准从 cfg.reward.* 读(占位标定值,不硬编码;改值不改公式结构)。

开环(修士):注入 NullUpdater,field 逐位不变;planned 掩码仍更新(dense 衰减掩码:走过即衰减
到 0,不重复计数),是 **reward 侧记账**,不回写策略 belief(obs 仍静态先验 coarse map)。GT 目标格
**仅入 reward**(训练信号),**绝不进 obs**。任何位置无 if master/doctor 分支(belief 注入点为对象切换)。

Leader 对齐(v1.1.1,人类裁定):
- reset() 内给 lawnmower 加 lead-in/out over_run 外扩(over_run = leader arrive_radius + 2.0),
  与 examples/mission_windows_survey.py 完全一致;起点 x ≈ -3.5m(lead-in),终点 x ≈ 62m(lead-out)。
- 去掉原先 advance(look_back/cruise) 预推进(该操作把 Leader 推到 x≈14,领先 Follower 整个窗口)。
- 终止改为"工作完成"判据:Leader.finished 后**不立刻 terminated**,而是进入 drain 阶段让
  Follower 继续决策直到全局高概率区全 planned 或 drain"无进展即止"或 drain 步数上限耗尽再 terminated。
  保留既有 truncation 安全阀(max_episode_time_s / 电量耗尽)。

窗口出界修复(v1.1.4 点4,方案 A):reset 末把 Leader 从 lead-in x≈-3.5 用小步 advance 前扫到
look_back 窗口完全入场(pose[0] >= look_back_m)再开始决策,使 Follower 首航点解码落在场内。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from msim.contracts.config import (
    SensorConfig,
    SimConfig,
    TargetFieldConfig,
    WindowConfig,
    WorldConfig,
)
from msim.contracts.types import FLOAT, OBS_FLOAT, CoverageMask, Field, Pose
# --- 上游实现(对契约编程,从 msim.<层>.<模块> import 实现,§v1.0.2)---
from msim.geometry.grid import Grid
from msim.geometry.window import (
    coarse_grid_shape,
    coarse_index_to_world,
    get_window,
    window_cells,
)
from msim.env_static.leader_path import lawnmower_waypoints
from msim.physics.leader import LeaderActor
from msim.physics.follower import Follower
from msim.physics.rollout import Level1RolloutEngine
from msim.task.corridor import corridor_cells
from msim.task.value import mark_planned  # §v1.1.4:dense 改累加 field 概率质量,不再用 expected_entropy_reduction
from msim.task.energy_balance import balance_penalty
from msim.task.belief import NullUpdater
from msim.mdp.action import decode_action
from msim.mdp.state import FollowerObsState, assemble_obs
from msim.mdp.commitment import (
    FollowerCommitment,
    resolve_available_time,
    resolve_remaining_distance,
)

__all__ = ["CoopSurveyEnv", "small_cfg", "windowed_survey_cfg"]


# ======================================================================
# 小尺度 cfg 工厂(便利函数,**非契约改动**):用于 env_checker / smoke / 训练
# ----------------------------------------------------------------------
# 默认 SimConfig 世界 200×100m@res0.2 → 细网格 1000×500,corridor_cells(全网格扫描,
# A4 实现 O(nx·ny))单段 ~0.8s,且 Leader 走完 lawnmower 是很长 episode。波次2 DoD 只
# 要求"能过 checker / 能训能跑完",故训练/校验用小尺度世界让 episode 短、栅格小。
# env 本体对任意 SimConfig 通用(大尺度只是慢),小尺度仅是缺省运行配置。
# ======================================================================
def small_cfg(seed: int = 0) -> SimConfig:
    """小尺度训练/校验配置:40×20m 世界(res 2m → 20×10 细网格)、8×20m 观测窗口(单测线、全 East)、
    4×4m footprint(粗网格 Hc=ceil(8/4)=2 × Wc=ceil(20/4)=5 → Discrete(10))。lawnmower 短、
    episode 几十~几百步即结束。

    §v1.1.5(A0 裁决):window.width_m = y_max(=20)使**单测线、块=全 East 的 North 条带**,与离散块
    的单测线前提自洽(原 width=8<y_max 会让 Leader 多测线、块窗口却只覆盖 East 中心条带 → 语义不
    自洽)。look_back 保持 8(块仍是 8m North 条带)。该 cfg 为 small 自身 smoke 配置、不绑任何
    checkpoint,Discrete(4)→(10) 无碍;A1 按 coarse_grid_shape 动态算 action_space,不受影响。
    """
    return SimConfig(
        world=WorldConfig(x_max_m=40.0, y_max_m=20.0, res_m=2.0),
        # §v1.1.5:width_m=y_max(单测线、全 East),满足离散块"块=全 East 的 North 条带"前提。
        window=WindowConfig(look_back_m=8.0, width_m=20.0, advance_threshold_m=4.0),
        sensor=SensorConfig(footprint_along_m=4.0, footprint_lateral_m=4.0),
        # §v1.1.3:小尺度世界塞少量目标(1 簇×3=3),保持 episode 短、env_checker 快。
        target_field=TargetFieldConfig(n_targets=3),
        n_followers=2,
        max_episode_time_s=600.0,
        seed=seed,
    )


# ======================================================================
# windowed 演示配置工厂:与 examples/mission_windows_survey.py 逐字一致
# ----------------------------------------------------------------------
# **训练与演示的单一配置真相源**:A0 演示(examples/mission_learned_survey.py)复用
# mission_windows_survey 的滑动观测窗口模式并加载训练 checkpoint;策略动作空间
# Discrete(Hc×Wc) 由 coarse_grid_shape(window, sensor) 决定,**只有训练 config 的
# window/sensor 与演示一致,checkpoint 才装得进去**(否则触发 train_ppo 的 n_actions
# 不匹配报错)。故此工厂的 world/sensor/window 必须与 mission_windows_survey 完全相同:
#   world  = WorldConfig(x_max_m=60.0, y_max_m=20.0, res_m=2.0)
#   sensor = SensorConfig(footprint_along_m=2.0, footprint_lateral_m=2.0)
#   window = 默认 WindowConfig(look_back_m=20、width_m=20)
# → coarse_grid_shape = (Hc=ceil(20/2), Wc=ceil(20/2)) = (10, 10) → Discrete(100)。
# 演示侧应 `from msim.mdp.coop_survey_env import windowed_survey_cfg` 导入,别处不得重复定义。
# ======================================================================
def windowed_survey_cfg(seed: int = 0) -> SimConfig:
    """windowed 演示/训练配置:60×20m 世界(res 2m)、默认 20×20 观测窗口、2×2m footprint
    (粗网格 10×10 → Discrete(100)),n_followers=2。与 examples/mission_windows_survey.py
    的 world/sensor/window **逐字一致**,使训练 checkpoint 能被该演示原样加载。

    §v1.1.3:target_field.n_targets=9(3 簇×3)对齐 examples/mission_learned_survey.py 的
    演示场景分布——训练分布 = 演示分布(避免在"1 簇 3 目标"上训、却在"3 簇 9 目标"上演示的
    外推);env.reset 经 self._cfg.target_field.n_targets 读此值作 default_instance 的目标数。
    """
    return SimConfig(
        world=WorldConfig(x_max_m=60.0, y_max_m=20.0, res_m=2.0),
        sensor=SensorConfig(footprint_along_m=2.0, footprint_lateral_m=2.0),
        # window 用默认 WindowConfig(look_back_m=20、width_m=20)——与演示一致。
        # §v1.1.3:目标数 9(3 簇×3)= 演示场景分布(单一真相源:训练实例与演示共用)。
        target_field=TargetFieldConfig(n_targets=9),
        n_followers=2,
        seed=seed,
    )


# ======================================================================
# 单台 Follower 的运行时调度状态(物理量住 Follower,调度量住此)
# ======================================================================
class _FollowerRuntime:
    """Follower 调度状态:物理对象 + 承诺队列 + 空闲时刻 + 累计能耗。"""

    __slots__ = ("phys", "commitment", "free_time", "energy_J", "pose")

    def __init__(self, phys: Follower, start_pose: Pose) -> None:
        self.phys: Follower = phys
        self.commitment: FollowerCommitment = FollowerCommitment(queue=[], cursor=0)
        self.free_time: float = 0.0                      # 何时空闲(可接受下一决策)
        self.energy_J: float = 0.0                       # 累计能耗(进均衡惩罚)
        self.pose: np.ndarray = np.asarray(start_pose, dtype=FLOAT).copy()


class CoopSurveyEnv(gym.Env):
    """事件驱动协同探查 env(标准 Gymnasium 五元组对外)。满足 CoopSurveyEnvProtocol。

    one-at-a-time:每 step 为一台**空闲** Follower 决策;step 末推进时钟到下一空闲时刻、
    Leader 动力学推进。N=2 用 self/teammate 槽位隐式编码"为哪台决策"。

    §v1.1.5 离散固定块窗口(人类选定,对齐 mission_windows_survey 演示 + 便于实装按块分配):
    - 窗口从"Leader 位姿连续滑窗"改成"**固定块、按块切换**"。块边界 edges = [k*look_back],
      windowed→[0,20,40,60](3 块)、small→[0,8,16,24,32,40](5 块)。
    - 所有决策用**当前活动块** _block_k 的固定窗口 [edges[k],edges[k+1]]×[East 中带 width],
      由锚点位姿 [edges[k+1], y_max/2, 0] 经 get_window 构造(复用 coarse_*/decode_action 全不变)。
    - Leader 仍随事件时钟前进,其连续位姿**只门控"哪些块已扫"**(块 k 可派 ⟺ leader.pose[0]>=edges[k+1]),
      不再定义窗口。活动块 = 最低的"已扫且未完成"块;两台 Follower 一次一台顺序覆盖该块(块0→1→2)。
    - 块"完成":当前块连续 2*n_followers 次 dense_mass<ε,或该块内高概率区(field>=0.5)全 planned。
      done 后活动块推进到下一未完成块。**全部块 done → terminated**(取代旧 leader-finished+drain)。

    Leader 对齐(v1.1.1):lawnmower 含 lead-in/out over_run 外扩(≡ window 演示),无预推进。
    """

    metadata = {"render_modes": []}

    # §v1.1.5 块"完成"判据参数(per-block,泛化自旧 drain 的"无进展即止"):
    # - 块完成主判据 = "无进展即止":当前块连续 2*n_followers 次决策 dense_mass<ε → 该块榨干 → done。
    # - 安全兜底:单块决策数超上限(防退化死循环,正常应由无进展提前 done)。
    _BLOCK_STEPS_PER_FOLLOWER: int = 60        # 单块安全兜底步数(× n_followers);正常不触达
    _DRAIN_NOPROGRESS_EPS: float = 1e-3        # "无进展"单步 dense_mass 阈值(概率质量;ε~1e-3)

    # ------------------------------------------------------------------
    def __init__(self, cfg: Optional[SimConfig] = None) -> None:
        super().__init__()
        self._cfg: SimConfig = cfg if cfg is not None else small_cfg()
        self._grid = Grid(self._cfg.world)
        self._rollout = Level1RolloutEngine(self._cfg.rollout)
        self._belief = NullUpdater()  # 修士开环唯一注入点(对象切换,非 if 分支)

        # 粗网格形状 → 动作/观测空间。
        Hc, Wc = coarse_grid_shape(self._cfg.window, self._cfg.sensor)
        self._Hc, self._Wc = int(Hc), int(Wc)
        n_actions = self._Hc * self._Wc

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(
                    low=0.0, high=1.0, shape=(self._Hc, self._Wc), dtype=OBS_FLOAT
                ),
                "vec": spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=OBS_FLOAT),
            }
        )

        # lawnmower lane 间隔挂窗口宽度(窗口横向覆盖一条 lane),Leader 沿其推进。
        self._lane_spacing_m: float = float(self._cfg.window.width_m)

        # §v1.1.5 离散固定块:沿 North 每 look_back 一块;edges=[0,look_back,2*look_back,...,x_max]。
        # n_blocks = round(x_max/look_back);末块锚到 x_max(非整除时仅末块短一截)。
        look_back = float(self._cfg.window.look_back_m)
        x_max = float(self._cfg.world.x_max_m)
        n_blocks = max(1, int(round(x_max / look_back)))
        edges = [min(k * look_back, x_max) for k in range(n_blocks + 1)]
        edges[-1] = x_max  # 末边界锚到场边(整除时即 n_blocks*look_back==x_max,无副作用)
        self._n_blocks: int = n_blocks
        self._edges: List[float] = edges
        self._block_anchor_y: float = 0.5 * float(self._cfg.world.y_max_m)  # 块窗口 East 锚 = y_max/2

        # 归一化尺度:位置/距离用窗口对角尺度,时间用 max_episode_time_s。
        wb = look_back
        ww = float(self._cfg.window.width_m)
        self._window_scale: float = float(np.hypot(wb, ww))  # 窗口对角(>0)
        self._time_scale: float = float(self._cfg.max_episode_time_s)

        # 运行时状态(reset 填充)。
        self._followers: List[_FollowerRuntime] = []
        self._leader: Optional[LeaderActor] = None
        self._field: Optional[Field] = None
        self._planned: Optional[CoverageMask] = None
        self._clock: float = 0.0
        self._current: int = 0          # 当前决策者 Follower 索引
        self._n_decisions: int = 0      # 已决策次数(防御性截断兜底)
        # §v1.1.4 reward 侧状态(reset 填充):GT 目标格(仅入 reward,不进 obs)+ 检出去重 + 各台覆盖。
        self._target_cells: set = set()       # GT 目标所在 cell 集合(sparse 真检出用)
        self._detected: set = set()           # 已检出 GT 目标格(去重,每目标一次)
        self._cov_by_foll: List[set] = []     # 每台 Follower 累计覆盖 cell(软冗余:仅他台重叠才罚)
        # §v1.1.5 块调度状态(reset 填充):活动块索引 + 各块 done 标记 + 当前块无进展计数。
        self._block_k: int = 0                # 活动块索引(最低的"已扫且未完成"块)
        self._block_done: List[bool] = []     # 每块是否已完成(榨干/全 planned)
        self._block_noprogress_run: int = 0   # 当前活动块连续低进展(dense_mass<ε)决策计数
        self._block_noprogress_need: int = 0  # 触发块 done 所需的连续低进展次数(2*n_followers)
        self._block_budget: int = 0           # 单块安全兜底步数上限(纯防死循环)
        self._block_k_started_at: int = 0     # 当前活动块开始的 _n_decisions(算单块决策数)

    # ------------------------------------------------------------------
    # 起点位姿:与基线 greedy._start_poses 同约定(East 均布,x=0,航向指 North)
    # ------------------------------------------------------------------
    def _start_poses(self) -> List[np.ndarray]:
        n = self._cfg.n_followers
        y = self._cfg.world.y_max_m
        return [
            np.array([0.0, (i + 0.5) / n * y, 0.0], dtype=FLOAT) for i in range(n)
        ]

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None,
              options: Dict[str, Any] | None = None
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """生成耦合实例 → 放置 Followers / Leader → 组装首决策者 obs。

        Leader 对齐(v1.1.1):lawnmower 加 lead-in/out over_run 外扩(over_run = leader
        arrive_radius + 2.0),与 examples/mission_windows_survey.py 完全一致;不做预推进。
        首个窗口可能退化(Leader 在 x<0,look_back 窗口落在场外)→ coarse_map 接近全零。
        obs 仍在 Box[0,1] 内、window_cells 可能为空,env_checker 不崩。
        """
        super().reset(seed=seed)

        # seed 注入:用 gymnasium 标准 self.np_random,并据其派生确定性实例 seed。
        eff_seed = int(self.np_random.integers(0, 2**31 - 1))
        inst_cfg = replace(self._cfg, seed=eff_seed)

        # 耦合 ground-truth 实例(targets + 派生先验场);RL 只用 field(开环逐步揭示)。
        # §v1.1.3:目标数从配置单一真相源 target_field.n_targets 读(训练分布 = 演示分布)。
        from msim.eval.runner import default_instance  # 延迟 import(避免顶层环路)
        n_targets = int(self._cfg.target_field.n_targets)
        instance = default_instance(inst_cfg, n_targets=n_targets)
        self._field = np.asarray(instance.field, dtype=FLOAT).copy()
        self._planned = np.zeros((self._grid.nx, self._grid.ny), dtype=bool)

        # §v1.1.4 点2(sparse 真检出):存 GT 目标格(仅入 reward,**绝不进 obs**);检出集合去重。
        self._target_cells = {
            self._grid.world_to_cell(np.asarray(t, dtype=FLOAT)) for t in instance.targets
        }
        self._detected: set = set()
        # §v1.1.4 点3(软冗余):每台 Follower 的累计覆盖 cell 集合(仅他台重叠才罚)。
        self._cov_by_foll: List[set] = [set() for _ in range(self._cfg.n_followers)]

        # Leader:沿 lawnmower 动力学推进。
        # v1.1.1:加 lead-in/out over_run 外扩,与 mission_windows_survey.py main() 顶部完全一致。
        # over_run = leader arrive_radius_m + 2.0(由 Leader 实际动力学参数解析的确定量,非估计)。
        lawn = [np.asarray(w, float) for w in lawnmower_waypoints(self._cfg.world, self._lane_spacing_m)]
        arrive_r = float(self._cfg.leader_overrides.get("arrive_radius_m", 1.5))
        over_run_m = arrive_r + 2.0
        for _k in range(0, len(lawn) - 1, 2):
            _a, _b = lawn[_k], lawn[_k + 1]
            _d = _b - _a
            _n = float(np.hypot(_d[0], _d[1]))
            if _n > 0.0:
                _u = _d / _n
                lawn[_k] = _a - _u * over_run_m
                lawn[_k + 1] = _b + _u * over_run_m
        self._leader = LeaderActor(self._cfg, lawn)
        self._leader.reset()
        # v1.1.1:不做任何预推进(原先 advance(look_back/cruise) 抢跑已去掉)。
        # Leader 从 x≈-3.5(lead-in)起步,与 window 演示一致。

        # Followers:物理 + 调度状态。
        starts = self._start_poses()
        self._followers = []
        for i in range(self._cfg.n_followers):
            phys = Follower(self._cfg)
            phys.reset(starts[i])
            self._followers.append(_FollowerRuntime(phys, starts[i]))

        self._clock = 0.0
        self._n_decisions = 0
        # §v1.1.5 块调度初始化:活动块 0、各块未完成、当前块无进展计数清零。
        self._block_k = 0
        self._block_done = [False] * self._n_blocks
        self._block_noprogress_run = 0
        self._block_noprogress_need = 2 * self._cfg.n_followers
        self._block_budget = self._cfg.n_followers * self._BLOCK_STEPS_PER_FOLLOWER
        self._block_k_started_at = 0

        # §v1.1.5 / v1.1.4 点4:reset 预扫——Leader 从 lead-in x≈-3.5 用小步 advance 前扫到
        # block 0 已扫(leader.pose[0] >= edges[1] = look_back),使块0 可派、首窗口完全入场。
        # Leader 轨迹起点仍是 -3.5;clock 不动(部署期 Leader 单独前扫,Follower 在 t=0 待命,
        # 与演示"扫完 W0 才派 Follower"一致)。
        self._advance_leader_until(self._edges[1])

        self._current = self._pick_decider()

        obs = self._assemble_current_obs()
        info = self._make_info(reward=0.0)
        return obs, info

    # ------------------------------------------------------------------
    # Leader 小步前扫(§v1.1.5):推进 Leader 到 pose[0] >= target_x(或 Leader 走完)
    # ------------------------------------------------------------------
    def _advance_leader_until(self, target_x: float) -> None:
        """用与 step 同源的小步 advance 推进 Leader,直到 leader.pose[0] >= target_x 或 Leader 走完。

        步长用 sim_dt_s(单 sim_dt 的 cruise 位移 ~0.15m)→ 停在**刚好 ≥ target_x**(超调 ≤ 一个
        sim_dt 位移)。用于:(a) reset 预扫到 edges[1](块0 可派);(b) step 开头把活动块扫入场。
        Leader.advance 内部按 sim_dt 细分积分(同 step 推进口径)。max_iters 硬上限防退化测线死循环。
        """
        assert self._leader is not None
        step_dt = self._cfg.rollout.sim_dt_s
        max_iters = 10000  # 安全兜底(防退化测线/零速死循环)
        it = 0
        while (not self._leader.finished
               and float(self._leader.pose[0]) < float(target_x)
               and it < max_iters):
            self._leader.advance(step_dt)
            it += 1

    # ------------------------------------------------------------------
    # §v1.1.5 离散块窗口 / 块门控辅助
    # ------------------------------------------------------------------
    def _block_window(self, k: int):
        """块 k 的固定窗口 [edges[k],edges[k+1]]×[y_max/2 ± width/2]。

        构造法:锚点位姿 = [edges[k+1], y_max/2, psi=0](朝 North)→ get_window 沿后方延伸
        look_back → North 范围 [edges[k+1]-look_back, edges[k+1]];整除块即 [edges[k],edges[k+1]]。
        复用 get_window 使 coarse_grid_shape/coarse_index_to_world/decode_action 全不变。

        ⚠ 离散块**前提**(§v1.1.5,A0 裁决):**单测线**,即 window.width_m == world.y_max_m
        → 块 = 全 East 的 North 条带(East 范围 [y_max/2±width/2] = [0, y_max])。windowed 满足
        (width=20=y_max);small_cfg 已对齐(width=20=y_max)。**若 width_m < y_max**(多测线),块窗口
        只覆盖 East 中心条带 [y_max/2±width/2],East 两侧不入任何块 → 语义不自洽,勿如此配置。
        """
        anchor = np.array(
            [float(self._edges[k + 1]), self._block_anchor_y, 0.0], dtype=FLOAT
        )
        return get_window(anchor, self._cfg.window)

    def _current_window(self):
        """当前活动块 _block_k 的固定窗口(替代旧 get_window(leader.pose) 连续滑窗)。
        全块 done 时 _block_k 可能越界(=n_blocks)→ 钳到末块,保证 obs/decode 仍有合法窗口
        (此情形 episode 已 terminated,仅防御 env_checker 在 terminated 后多 step 一次)。"""
        k = min(self._block_k, self._n_blocks - 1)
        return self._block_window(k)

    def _block_scanned(self, k: int) -> bool:
        """块 k 是否已被 Leader 扫到(可派):leader.pose[0] >= edges[k+1]。"""
        assert self._leader is not None
        return float(self._leader.pose[0]) >= float(self._edges[k + 1]) - 1e-9

    def _block_highprob_all_planned(self, k: int) -> bool:
        """块 k 内高概率区(field>=阈值)是否全部已 planned(无高概率区视为 True)。"""
        assert self._field is not None and self._planned is not None
        cells = window_cells(self._block_window(k), self._grid)
        hp = [(cx, cy) for (cx, cy) in cells
              if self._field[cx, cy] >= self._HIGHPROB_THRESHOLD]
        if not hp:
            return True
        return all(self._planned[cx, cy] for (cx, cy) in hp)

    def _activate_next_block(self) -> None:
        """把活动块 _block_k 推进到下一个"未完成"块;并把活动块的状态计数复位。

        若当前活动块尚未扫到(leader 未走过其右边界),用小步 advance 把 Leader 扫入该块
        (或 Leader 走完)。**仅当活动块真的切换时**才清零无进展计数 / 记录起始决策序号
        (否则 step 开头每次无条件复位会清掉上一步累计的无进展计数 → 块永远 done 不了)。
        """
        assert self._leader is not None
        prev_k = self._block_k
        # 跳过已 done 的块,定位最低未完成块。
        while self._block_k < self._n_blocks and self._block_done[self._block_k]:
            self._block_k += 1
        if self._block_k >= self._n_blocks:
            return  # 全块 done(终止由 _is_terminated 处理)
        # 确保活动块已扫(被 Leader 走过右边界);否则小步前扫到它。
        if not self._block_scanned(self._block_k):
            self._advance_leader_until(self._edges[self._block_k + 1])
        # 仅活动块切换时复位计数(进入新块重新累计无进展);未切换则保留累计。
        if self._block_k != prev_k:
            self._block_noprogress_run = 0
            self._block_k_started_at = self._n_decisions

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: int
             ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """为当前空闲 Follower 解码动作→入承诺队列→Level1 解析用时/能耗→结算到达段价值(在
        **当前活动块**的固定窗口内)→推进全局时钟到下一空闲时刻(Leader 动力学推进、块门控)→
        组装下一决策者 obs。§v1.1.5:窗口=离散固定块,非连续滑窗。"""
        assert self._leader is not None and self._field is not None
        assert self._planned is not None

        # §v1.1.5 step 开头:确保活动块有效且已扫(跳过已 done 块、必要时小步前扫到活动块)。
        self._activate_next_block()

        i = self._current
        fr = self._followers[i]

        # --- 解码动作:当前**活动块固定窗口**内的粗格中心 waypoint(纯几何;非 leader 滑窗)---
        window = self._current_window()
        waypoint = decode_action(int(action), window, self._grid, self._cfg)
        # 离散块下 waypoint 天然在 [0,x_max]×East 带内;clip 留作安全无害兜底(浮点边界)。
        waypoint = np.clip(
            np.asarray(waypoint, dtype=FLOAT),
            [0.0, 0.0],
            [self._cfg.world.x_max_m, self._cfg.world.y_max_m],
        )

        old_pose = fr.pose.copy()

        # --- Level 1 解析:该段用时/能耗(承诺队列解析确定量)---
        res = self._rollout.execute(old_pose, [waypoint])
        seg_duration = float(res.duration_s)
        seg_energy = float(res.energy_J)

        # --- 价值结算:走廊覆盖 ∩ 当前窗口 = 本步可观测格(只在已揭示窗口水域内结算)---
        corridor = corridor_cells(
            old_pose[:2], waypoint, fr.phys.sensor.corridor_width_m, self._grid
        )
        win_cells = window_cells(window, self._grid)
        observable = corridor & win_cells

        # §v1.1.4 点1(dense):已解析先验概率质量 = Σ 走廊∩窗口∩未planned 的 field[cell]。
        #   GT-free、部署可算;复用布尔 planned 掩码(走过即衰减到 0,不重复计数)。这是
        #   **reward 侧记账**(衰减掉的先验概率),不回写策略 belief(obs 仍静态先验)。
        dense_mass = 0.0
        for (cx, cy) in observable:
            if not self._planned[cx, cy]:
                dense_mass += float(self._field[cx, cy])

        # §B 高信噪比"瞄准"信号:所选粗格中心(=动作落点)的**残余先验**(planned 置 0),与 greedy
        #   挑 obs map argmax 同口径、去转移距离污染。在 mark_planned **之前**取值(否则被清零)。
        #   GT-free、部署可算;默认 aim_weight=0 时不进 reward(向后兼容)。
        aim_ix, aim_iy = self._grid.world_to_cell(np.asarray(waypoint, dtype=FLOAT))
        aim_residual = 0.0 if self._planned[aim_ix, aim_iy] else float(self._field[aim_ix, aim_iy])

        # §v1.1.4 点2(sparse):本步**首次确定性覆盖**的 GT 目标格(detected 集合去重,每目标一次)。
        #   GT 仅入 reward;策略 obs 不含 GT(不违开环,无 master/doctor 分支)。
        new_det = {c for c in observable if c in self._target_cells and c not in self._detected}
        self._detected |= new_det
        n_new_detect = len(new_det)

        # §v1.1.4 点3(软冗余):本步覆盖格 ∩ 其它 Follower 已覆盖格的数量(仅他台重叠才罚;自重叠不罚)。
        other_cov: set = set()
        for j in range(len(self._followers)):
            if j != i:
                other_cov |= self._cov_by_foll[j]
        n_redundant = len(observable & other_cov)
        self._cov_by_foll[i] |= observable

        mark_planned(self._planned, observable)  # 不重复计数(dense 衰减掩码;修士也更新)
        # 开环 belief 回写(NullUpdater:field 逐位不变)。
        self._belief.update(self._field, observable, fr.phys.sensor, None)

        # --- 更新该 Follower 调度状态:承诺 + 空闲时刻 + 累计能耗 + 末位姿 ---
        fr.commitment.queue.append(np.asarray(waypoint, dtype=FLOAT)[:2].copy())
        fr.commitment.cursor = len(fr.commitment.queue)  # 解析模型:本段记账即视为走完
        fr.free_time = fr.free_time + seg_duration
        fr.energy_J += seg_energy
        fr.pose = np.asarray(res.end_pose, dtype=FLOAT).copy()
        fr.phys.reset(fr.pose)  # 物理对象同步到段末位姿(energy_norm 复满,均衡用解析 energy_J)

        self._n_decisions += 1

        # --- 推进全局时钟到下一空闲时刻;Leader 动力学推进(块门控:其位姿决定哪些块已扫)---
        next_clock = min(f.free_time for f in self._followers)
        dt = max(0.0, next_clock - self._clock)
        if dt > 0.0:
            self._leader.advance(dt)
        self._clock = next_clock

        # --- §v1.1.5 当前活动块"完成"判据(per-block,泛化自旧 drain 无进展即止)---
        # 进展信号 = 本步 dense_mass(对**当前活动块**的衰减概率质量;observable 已限定在块窗口内)。
        # 连续 2*n_followers 次 < ε,或块内高概率区全 planned,或单块步数超兜底 → 标记该块 done。
        # (守卫 _block_k<n_blocks:全块 done 后越界则跳过;此情形 episode 已 terminated。)
        if dense_mass < self._DRAIN_NOPROGRESS_EPS:
            self._block_noprogress_run += 1
        else:
            self._block_noprogress_run = 0
        if self._block_k < self._n_blocks:
            block_steps = self._n_decisions - self._block_k_started_at
            if (self._block_noprogress_run >= self._block_noprogress_need
                    or self._block_highprob_all_planned(self._block_k)
                    or block_steps >= self._block_budget):
                self._block_done[self._block_k] = True
                # 立即推进到下一未完成块(并在必要时小步前扫扫入),使下一决策落在下一块。
                self._activate_next_block()

        # --- reward(§v1.1.4:dense 概率质量 + sparse 真检出 − 均衡 − 软冗余)---
        # dense 缩放**不截断**(点2);detect 直接计数;balance 降权(点3);redundancy 软惩罚(点3)。
        rc = self._cfg.reward
        dense_term = rc.value_weight * (dense_mass / rc.value_norm_ref)
        detect_term = rc.detect_weight * float(n_new_detect)
        aim_term = rc.aim_weight * aim_residual  # §B 默认 aim_weight=0 → 不改变现有公式/A1 验收
        balance_pen = balance_penalty([f.energy_J for f in self._followers])
        balance_term = rc.balance_weight * (balance_pen / rc.balance_norm_ref)
        redundancy_term = rc.redundancy_weight * float(n_redundant)
        # §B 微调:每步移动(段能耗)惩罚,治单台自重叠绕路;默认 length_weight=0 → 不改变公式/A1。
        length_term = rc.length_weight * (seg_energy / rc.length_norm_ref)
        reward = float(dense_term + detect_term + aim_term
                       - balance_term - redundancy_term - length_term)

        # --- 终止/截断判定 ---
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # --- 下一决策者 obs(忙者不参与:_pick_decider 取最早空闲台)---
        if not (terminated or truncated):
            self._current = self._pick_decider()
        obs = self._assemble_current_obs()

        info = self._make_info(
            reward=reward,
            dense_mass=dense_mass, dense_term=dense_term,
            n_new_detections=n_new_detect, detect_term=detect_term,
            balance_pen=balance_pen, balance_term=balance_term,
            n_redundant=n_redundant, redundancy_term=redundancy_term,
            aim_residual=aim_residual, aim_term=aim_term,
            seg_energy=seg_energy, length_term=length_term,
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # 决策者选取:一次一台(取最早空闲;同时空闲取低索引)
    # ------------------------------------------------------------------
    def _pick_decider(self) -> int:
        return int(min(range(len(self._followers)),
                       key=lambda i: (self._followers[i].free_time, i)))

    # ------------------------------------------------------------------
    # obs 组装:self=当前决策者,teammate=另一台(N=2 隐式编码"为哪台决策")
    # ------------------------------------------------------------------
    def _coarse_map(self) -> np.ndarray:
        """当前**活动块固定窗口**概率场降采样到 (Hc,Wc):粗格中心 → 细 cell → 读 field。
        §v1.1.5:coarse map 铺在当前活动块上(非 leader 连续滑窗)。
        §v1.1.6:map 单元 = field[center]·(1−planned[center])(残余先验,与 reward
        dense_mass 同口径,消观测混叠);planned 是布尔覆盖掩码、非 belief 回写,
        field 逐位不变、NullUpdater 仍开环。"""
        assert self._leader is not None and self._field is not None
        window = self._current_window()
        cmap = np.zeros((self._Hc, self._Wc), dtype=OBS_FLOAT)
        for row in range(self._Hc):
            for col in range(self._Wc):
                idx = row * self._Wc + col
                center = coarse_index_to_world(
                    idx, window, self._cfg.window, self._cfg.sensor
                )
                ix, iy = self._grid.world_to_cell(center)
                planned = self._planned is not None and bool(self._planned[ix, iy])
                cmap[row, col] = OBS_FLOAT(0.0 if planned else self._field[ix, iy])
        return cmap

    def _obs_state_for(self, idx: int) -> FollowerObsState:
        fr = self._followers[idx]
        d_rem = resolve_remaining_distance(fr.pose[:2], fr.commitment)
        t_avail = resolve_available_time(fr.pose[:2], fr.commitment, self._cfg.rollout)
        # 位置归一化:场内坐标 / 场尺度(落 [0,1]);距离用窗口尺度;时间用 max_episode_time。
        pos = np.array(
            [
                fr.pose[0] / max(self._cfg.world.x_max_m, 1e-9),
                fr.pose[1] / max(self._cfg.world.y_max_m, 1e-9),
            ],
            dtype=FLOAT,
        )
        return FollowerObsState(
            energy_norm=float(fr.phys.energy_norm),
            pos_norm=pos,
            remaining_distance_norm=float(d_rem / self._window_scale),
            available_time_norm=float(t_avail / self._time_scale),
        )

    def _assemble_current_obs(self) -> Dict[str, np.ndarray]:
        cmap = self._coarse_map()
        i = self._current
        teammate = (i + 1) % len(self._followers)
        return assemble_obs(
            cmap, self._obs_state_for(i), self._obs_state_for(teammate)
        )

    # ------------------------------------------------------------------
    # 终止/截断
    # ------------------------------------------------------------------
    # 高概率区阈值(field 归一化 [0,1];与候选提取的"高概率核心"口径同量级)。
    _HIGHPROB_THRESHOLD: float = 0.5

    def _is_terminated(self) -> bool:
        """terminated = **全部块 done**(§v1.1.5 块语义;取代旧 leader-finished+drain)。

        每块由 step 内"完成"判据(连续 2*n_followers 次 dense_mass<ε / 块内高概率区全 planned /
        单块步数超兜底)逐块标记 done;Leader 走完被块扫描门控自然蕴含(末块需 Leader 越过其右
        边界才可派、才能 done)。全部块 done ⟺ 全场已逐块榨干 → terminated。
        保留既有 truncation 安全阀(max_episode_time_s / 电量耗尽)兜底。
        """
        assert self._leader is not None and self._field is not None
        assert self._planned is not None
        return all(self._block_done)

    def _is_truncated(self) -> bool:
        """truncated = 达 max_episode_time_s 或任一 Follower 电量耗尽。"""
        if self._clock >= self._cfg.max_episode_time_s:
            return True
        for fr in self._followers:
            if fr.phys.energy_norm <= 0.0:
                return True
        return False

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------
    def _make_info(self, *, reward: float,
                   dense_mass: float = 0.0, dense_term: float = 0.0,
                   n_new_detections: int = 0, detect_term: float = 0.0,
                   balance_pen: float = 0.0, balance_term: float = 0.0,
                   n_redundant: int = 0, redundancy_term: float = 0.0,
                   aim_residual: float = 0.0, aim_term: float = 0.0,
                   seg_energy: float = 0.0, length_term: float = 0.0) -> Dict[str, Any]:
        """组装 info(§v1.1.4 契约:reward + 分解项 + 调度/能耗元信息;§B 增 aim_*/length_* 项)。

        保证 reward == dense+detect+aim − balance − redundancy − length(容差 1e-9)。
        aim_term/length_term 默认 0(aim_weight/length_weight=0)→ 退化为原 §v1.1.4 公式,A1 验收不变。
        """
        assert self._leader is not None
        return {
            "clock_s": float(self._clock),
            "current_follower": int(self._current),
            "n_decisions": int(self._n_decisions),
            "leader_finished": bool(self._leader.finished),
            "per_follower_energy_J": [float(f.energy_J) for f in self._followers],
            # §v1.1.4 reward 分解项(供 A1 核验公式)。
            "dense_mass": float(dense_mass),
            "dense_term": float(dense_term),
            "n_new_detections": int(n_new_detections),
            "detect_term": float(detect_term),
            "balance_pen": float(balance_pen),
            "balance_term": float(balance_term),
            "n_redundant": int(n_redundant),
            "redundancy_term": float(redundancy_term),
            "aim_residual": float(aim_residual),
            "aim_term": float(aim_term),
            "seg_energy": float(seg_energy),
            "length_term": float(length_term),
            "reward": float(reward),
            # §v1.1.5 块调度元信息(供观测/自验;不属 reward 契约,加键不影响 obs space)。
            "block_k": int(self._block_k),
            "n_blocks": int(self._n_blocks),
            "n_blocks_done": int(sum(self._block_done)) if self._block_done else 0,
        }
