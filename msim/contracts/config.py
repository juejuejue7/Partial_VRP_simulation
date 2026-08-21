"""
SimConfig 及子配置 —— 全项目超参的单一真相源(第0层,纯 Python dataclass,不引 YAML)。

设计要点:
- 纯 dataclass,可类型检查、可复现、无外部解析层。
- 异构(Leader/Follower)物理差异经 `leader_overrides` / `follower_overrides`(纯 dict)注入;
  config 层不 import AUVConfig(那是第2a层),避免反向依赖。physics 层的 AUV 工厂消费这些 dict。
- balance on/off 消融 = RewardConfig.balance_weight 置 0 / 非 0(不是 if 分支)。
- 修士/博士差异不在此处:由注入的 BeliefUpdater 决定(见 contracts/task.py)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class WorldConfig:
    """世界尺度与栅格分辨率(NED;x=North 长边,y=East 短边)。"""
    x_max_m: float = 200.0          # North 跨度(主测线方向)
    y_max_m: float = 100.0          # East 跨度
    res_m: float = 0.2              # 细网格分辨率

    @property
    def nx(self) -> int:            # 沿 North 的细网格列数
        return int(round(self.x_max_m / self.res_m))   # 1000

    @property
    def ny(self) -> int:            # 沿 East 的细网格行数
        return int(round(self.y_max_m / self.res_m))   # 500


@dataclass(frozen=True)
class WindowConfig:
    """Leader 后方观测窗口(滑动 mask）。窗口编码"观测资格/物理约束",非信息可得性。"""
    look_back_m: float = 20.0       # 沿测线向后回看长度
    width_m: float = 20.0           # 横向宽度
    advance_threshold_m: float = 5.0  # 窗口推进触发一次决策事件的阈值


@dataclass(frozen=True)
class SensorConfig:
    """相机传感器模型("接近完美但不完美",为博士 false-positive 修正留空间)。"""
    p_d: float = 0.95               # 检出率(高)
    p_fa: float = 0.05              # 虚警率(小)
    footprint_along_m: float = 4.0  # 沿测线 footprint(决定粗网格沿测线格距)
    footprint_lateral_m: float = 6.0  # 横向 footprint(走廊宽 = ×0.9;决定粗网格横向格距)


@dataclass(frozen=True)
class RewardConfig:
    """价值 = 已解析先验概率质量(dense)+ 真检出(sparse);减 能耗均衡 与 软冗余 惩罚。

    §v1.1.4(人类裁决,EIG 熵降 → 先验概率质量;Q6 更新,见 DECISIONS §v1.1.4):
    二元熵在 p=0.5 最大、p=1 为零 → 旧 EIG 奖励"光晕"(p≈0.5)、躲"目标中心"(场峰 p≈1,熵=0)→
    策略擦肩目标 1-2m。改为 **已解析的先验概率质量**:Follower 走廊扫过一格 → 该格剩余先验概率
    衰减到 0(复用 planned 掩码),reward 取**衰减掉的概率值 = field[cell]**。概率质量在目标中心
    (field=1)最大 → 奖励对准目标;且 **GT-free、部署可算、无需熵/无需先验校准**。再叠加 **稀疏检出**
    (用 GT)直接奖励找到真目标。**点2**:value 项**缩放不截断**。**点3**:降 balance 权重 + 软冗余惩罚。

        dense_mass = Σ_{走廊∩窗口∩未planned} field[cell]            # 已解析先验概率质量(GT-free)
        reward = value_weight · (dense_mass / value_norm_ref)        # dense:缩放(不截断)
               + detect_weight · n_new_detections                    # sparse:本步新覆盖 GT 目标格数(detected 去重)
               − balance_weight · (balance_pen / balance_norm_ref)   # 跨 Follower 能耗均衡惩罚(基于 per_follower_energy)
               − redundancy_weight · n_redundant                     # 软冗余:本步与其它 Follower 已覆盖重叠的格数(不终止)
    所有 *_weight / *_norm_ref 为**占位标定值**(评估阶段调;改值不改公式结构)。
    开环纪律:dense 的"剩余先验衰减"是 **reward 侧记账**,不回写策略 belief(obs 仍静态先验,
    NullUpdater 不变);GT **仅入 reward**(训练信号),策略 obs 不含 GT,master/doctor 不分支。
    """
    value_weight: float = 1.0       # dense 覆盖权重(作用于缩放后的已解析先验概率质量)
    value_norm_ref: float = 5.0     # dense 缩放分母(占位;**缩放不截断**,§v1.1.4 点2;待标定)
    detect_weight: float = 1.0      # sparse 每个新检出 GT 目标格的奖励(占位;待标定)
    aim_weight: float = 0.0         # §B:高信噪比"瞄准"奖励——所选粗格残余先验(GT-free,与 greedy
                                    #   同口径、去转移距离污染)。**默认 0 → 不改变现有 reward 公式与 A1 验收**;
                                    #   仅训练实验 cfg 置 >0 以增强 actor 的 per-action 信用分配信号。
    balance_weight: float = 0.2     # 跨 Follower 能耗均衡惩罚权重(§v1.1.4 点3 从 1.0 降;=0 即 balance-off 消融)
    balance_norm_ref: float = 100.0  # 均衡归一化分母(能耗代理同量纲;占位,待标定)
    redundancy_weight: float = 0.1  # 软冗余惩罚权重(每个与他台覆盖重叠的格;§v1.1.4 点3;占位;**不终止 episode**)
    length_weight: float = 0.0      # §B 微调:每步移动(段能耗)惩罚——治"单台重走自己扫过区"的绕路
                                    #   (n_redundant 只罚跨台重叠,单台自重叠免费)。**默认 0 → 不改变现有公式/A1**;
                                    #   仅微调实验 cfg 置 >0,把 min-max 路程往 greedy/VRP 压。
    length_norm_ref: float = 100.0  # 段能耗惩罚归一化分母(与 balance 同量纲;占位,待标定)
    entropy_unit: str = "bit"       # 保留(历史兼容;§v1.1.4 后 dense 已不用熵)


@dataclass(frozen=True)
class RolloutConfig:
    """Level 1(训练,解析确定量)与 Level 2 / Leader(AUV 积分)共用的运动/能耗参数。"""
    cruise_speed_mps: float = 1.5         # 巡航速度(Level 1 用时口径)
    nominal_yaw_rate_rps: float = 0.5236  # 名义转弯角速度 ~30deg/s;Level 1 转弯时间近似用
    dwell_time_s: float = 5.0             # 每 waypoint 停留拍照时间
    length_energy_coeff: float = 1.0      # Level 1:单位路径长度能耗
    turn_energy_coeff: float = 1.0        # Level 1:单位转角能耗(转弯进能耗)
    sim_dt_s: float = 0.1                 # Level 2 / Leader 的 AUV 积分步长


@dataclass(frozen=True)
class ObsCandidateConfig:
    """A′ 混合候选观测航点提取参数(§v1.0.5;**可调试旋钮**,从 Leader 扫描概率图取候选点)。

    半径默认挂相机 footprint(无 magic number);为 None 时由 resolve_obs_candidate_params 回退到物理默认。
    - threshold_rel:**峰值锚点**相对阈值(field >= 该比例×max 的格才算峰值候选;默认 0.5,§v1.0.10 定稿)。
    - fill_threshold_rel:**补点**高概率区相对阈值(§v1.0.6 与峰值阈值拆开、§v1.0.10 调严;默认 0.7 →
      补点只落在更确信的密集区/合并簇核心,不在弱晕区撒点;约定 ≥ threshold_rel)。
    - peak_min_separation_m:峰值锚点 NMS 间隔;None → 0.5*footprint_lateral_m(=先验模糊 σ=3m)。
    - fill_radius_m:覆盖补点半径 R;None → 0.9*footprint_lateral_m(=相机走廊宽=5.4m,默认)。
      **调试提示**:R 调小(如 2.7=走廊半宽)→ 候选更密、Follower 走廊更能扫遍高概率区(端到端覆盖↑),
      但候选点数与路线/能耗显著增;R 调大 → 候选更少更省、但端到端覆盖受走廊半宽限制。
    """
    threshold_rel: float = 0.3                        # 峰值锚点阈值(§v1.0.10 定稿,原 0.3)
    fill_threshold_rel: float = 0.5                   # 补点高概率区阈值(§v1.0.6 拆开 / §v1.0.10 调严,原 0.5)
    peak_min_separation_m: Optional[float] = None     # None → 0.5*footprint_lateral_m(σ)
    fill_radius_m: Optional[float] = None             # None → 0.9*footprint_lateral_m(走廊宽=5.4m)
    # §v1.0.12:候选【绝对阈值】(归一化先验值口径,**不锚定任何 max**)——通用鲁棒模式的单一真相源。
    # 动机:相对阈值(threshold_rel×field.max())仅在「全局归一化场(max=1)」时等价于绝对值;一旦输入是
    #   **子窗口/局部场**(如 windowed 模式按 20×20 观测窗口分块规划),局部 max<1 会把空/弱窗口里 ≈0.005 的
    #   KDE 拖尾相对放大成"满量程"→ 生成伪候选、Follower 被派去白扫整窗,鲁棒性差;若改用全局 field.max()
    #   又等于提前使用**未揭示区**信息(违反修士开环逐步揭示,纪律4)。固定绝对阈值对所有窗口一致:远高于
    #   拖尾、低于真实目标峰高 → 空窗口零候选(鲁棒)。消费者在「几何生成(相对阈值)」后,按这两个绝对
    #   阈值对候选点处的先验值再做一道筛选(见 examples/mission_*_survey.py 的统一口径)。
    # 为 None → 退化为纯相对阈值(向后兼容,现有全局口径与 A1 验收不变);全局场设 = 相对阈值即数值等价。
    peak_abs_threshold: Optional[float] = None        # 峰值锚点绝对阈值(先验∈[0,1]);None→纯相对
    fill_abs_threshold: Optional[float] = None         # 覆盖补点绝对阈值;None→纯相对(约定 ≥ peak_abs_threshold)


@dataclass(frozen=True)
class TargetFieldConfig:
    """Ground-truth 目标场尺度(海底环境/地质属性,**独立于相机 footprint**;§v1.0.8)。

    目标簇结构(多大/多密)与先验图模糊度是环境属性,与"用多大的相机观测"无关。
    §v1.0.8 前 default_instance 借 sensor.footprint 当尺度基准(§v1.0.3 便利选择),
    导致改相机 footprint 连带重画整个 ground-truth 场景(目标点坐标 + 先验图)。现解耦
    为独立旋钮:改 cfg.sensor.footprint 不再动目标场;场尺度单独经本配置调。
    默认值 = 原 footprint=6 时的值 → 默认场景逐位不变(向后兼容)。
    """
    bandwidth_m: float = 3.0       # 先验 KDE 带宽 σ(光晕宽窄);原 = 0.5×footprint_lateral = 3m
    intra_spread_m: float = 12.0   # 簇内目标散布尺度;原 = 2×max(footprint) = 12m
    targets_per_cluster: int = 3   # 每簇目标数(底栖聚集典型);原硬编码在 runner._default_cluster_params
    n_targets: int = 6             # 场景目标总数(§v1.1.3 单一真相源:env 训练实例 与 演示场景共用)。
                                   #   实际目标数 = ceil(n_targets/targets_per_cluster)×targets_per_cluster
                                   #   (向上取整成整簇);default_instance 据此分解簇结构。训练/演示对齐:
                                   #   windowed_survey_cfg 设 9(3 簇×3),使 RL 训练分布 = 演示场景分布。


@dataclass(frozen=True)
class SimConfig:
    """顶层仿真配置。env / 基线 / 训练统一从此构造。"""
    world: WorldConfig = field(default_factory=WorldConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    obs_candidate: ObsCandidateConfig = field(default_factory=ObsCandidateConfig)
    target_field: TargetFieldConfig = field(default_factory=TargetFieldConfig)  # §v1.0.8:目标场尺度,独立于相机 footprint

    n_followers: int = 1                  # 修士锁 N=2;保留修改接口
    max_episode_time_s: float = 3600.0    # truncated 阈值 + t_avail 归一化分母
    seed: int = 0

    # 异构物理 override(纯 dict;physics 层 AUV 工厂消费,config 不 import AUVConfig)
    leader_overrides: Dict[str, float] = field(
        default_factory=lambda: {
            "cruise_speed": 1.5,
            "battery_capacity_J": 1.0e7,
            "sensor_static_power_W": 25.0,   # 多波束声呐高功耗
            "arrive_radius_m": 1.5,
        }
    )
    follower_overrides: Dict[str, float] = field(
        default_factory=lambda: {
            "cruise_speed": 1.5,
            "battery_capacity_J": 5.0e6,
            "sensor_static_power_W": 3.0,    # 光学低功耗
            "arrive_radius_m": 1.5,
        }
    )


def apply_overrides(base: object, overrides: Dict[str, float]) -> object:
    """把 overrides dict 合并进一个 AUVConfig 实例,返回新配置。

    住 config 层(纯 dict 操作);physics 层调用本函数把 SimConfig.*_overrides
    注入 AUVConfig。**异构差异的唯一注入路径**(AUV 类本体只保留通用物理常数)。
    """
    raise NotImplementedError  # [A2 波次1]


def resolve_obs_candidate_params(cfg: "SimConfig") -> Tuple[float, float, float, float]:
    """解析 A′ 候选观测航点提取参数(§v1.0.5/§v1.0.6):None 的半径回退到挂 footprint 的物理默认。

    返回 (threshold_rel, fill_threshold_rel, peak_min_separation_m, fill_radius_m)。
    消费者(eval.runner.default_detections / 验收脚本)统一经本函数取值,使 `cfg.obs_candidate.*`
    成为调试入口:`fill_radius_m`(R,默认 5.4m)与 `fill_threshold_rel`(补点阈值,默认 0.5)等。
    """
    oc = cfg.obs_candidate
    fl = cfg.sensor.footprint_lateral_m
    peak_sep = oc.peak_min_separation_m if oc.peak_min_separation_m is not None else 0.5 * fl
    fill_r = oc.fill_radius_m if oc.fill_radius_m is not None else 0.9 * fl
    return oc.threshold_rel, oc.fill_threshold_rel, float(peak_sep), float(fill_r)


def resolve_obs_candidate_abs_thresholds(cfg: "SimConfig") -> Tuple[Optional[float], Optional[float]]:
    """解析候选【绝对阈值】(§v1.0.12;通用鲁棒模式单一真相源):返回 (peak_abs, fill_abs)。

    取自 `cfg.obs_candidate.peak_abs_threshold / fill_abs_threshold`(归一化先验值口径,不锚定 max)。
    任一为 None → 该项不做绝对筛选(退化纯相对,向后兼容)。消费者(baseline / windowed / 将来 RL
    或 A6 default_detections)统一经本函数取阈值,在「几何生成(相对阈值)」后按其筛选候选点的先验值,
    使空/弱/子窗口不被局部 max 放大出伪候选(详见 ObsCandidateConfig 的字段说明)。
    """
    oc = cfg.obs_candidate
    return oc.peak_abs_threshold, oc.fill_abs_threshold
