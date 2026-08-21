"""第1层 env_static 契约:ground-truth 耦合实例生成(真实目标点 + 由其派生的先验场)。

冻结决策(DECISIONS §v1.0.3,人类裁定 2026-06-12):
- **真实目标点 = 绝对 ground truth**(所有验证与下游数据的参考),由**簇点过程**生成
  (簇中心 + 簇内散布),贴合底栖聚集统计(project_summary §13:避免均匀随机)。
- **先验概率场 = 对目标点做各向同性高斯 KDE 后归一化到 [0,1]**(模糊信念,带宽 σ≈簇内散布
  → 先验∈(0,1),熵降价值非平凡)。**场严格由目标点派生,二者不再是独立流。**
- RL 价值用 `field`;baseline VRP/评估用**同一组** `targets` → 完全耦合、单一真值。
- 修士开环:实例在 episode 内静态不变(belief=NullUpdater);save/load 可逐位复现(Q19)。

归属:实现住 msim/env_static/(A2);消费者(A6 eval / A5 env / 验收脚本)从 msim.env_static.*
import 实现、从本契约 import 签名。轴序 field[ix,iy](第0轴=North=x),与 types.py 一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import List, Optional

import numpy as np

from .config import WorldConfig
from .types import Field, Waypoint


# ======================================================================
# 耦合实例(目标点为绝对参考,场由其派生)
# ======================================================================
@dataclass(frozen=True)
class GroundTruthInstance:
    """一个可复现的合成实例:真实目标点 + 由其派生的先验概率场。

    不变量(A1 验收):
    - field 是 targets 的归一化高斯 KDE → field 峰值与目标簇**共位**(场跟随目标)。
    - targets 处的平均 field 值 **显著高于** 全场随机非目标 cell 的平均(耦合性证明)。
    - 同 seed + 同参数 → (targets, field) 逐位复现。
    """
    targets: List[Waypoint]                              # ground-truth 目标点 [x_N,y_E];绝对参考
    field: Field                                         # 先验场 (nx,ny) float64 ∈[0,1];= targets 的归一化高斯KDE
    cluster_centers: List[Waypoint] = _field(default_factory=list)  # 簇中心(诊断/可视化,可空)


def field_from_targets(targets: List[Waypoint], world: WorldConfig,
                       *, bandwidth_m: float) -> Field:
    """先验场 = Σ_target 各向同性高斯核(中心=target,带宽=bandwidth_m)叠加后归一化到 [0,1]。

    纯几何确定:同 targets + 同 bandwidth → 同 field。轴序 field[ix,iy]。
    bandwidth_m 即"先验不确定性"(σ):越大越模糊;约定 σ≈簇内散布(模糊信念)。
    targets 为空 → 返回全零场。
    """
    raise NotImplementedError  # [A2 波次1·§v1.0.3]


def make_clustered_instance(world: WorldConfig, rng: np.random.Generator, *,
                            n_clusters: int, targets_per_cluster: int,
                            intra_spread_m: float,
                            bandwidth_m: Optional[float] = None) -> GroundTruthInstance:
    """生成耦合实例:**先**按簇点过程撒 ground-truth 目标点,**再**由其派生先验场。

    步骤:
      1. 撒 n_clusters 个簇中心(场内均匀);每个簇中心周围按高斯散布(std=intra_spread_m)
         采样 targets_per_cluster 个目标点 → 共 K=n_clusters×targets_per_cluster 个 ground-truth 目标点。
         目标点 clip 到场内。这些点是**绝对参考**。
      2. field = field_from_targets(targets, world, bandwidth_m)。
         bandwidth_m 默认 = intra_spread_m(模糊信念,§v1.0.3 裁定)。

    返回 GroundTruthInstance(targets, field, cluster_centers)。
    确定性:同 rng 状态 + 同参数 → 同实例(供复现验证)。
    """
    raise NotImplementedError  # [A2 波次1·§v1.0.3]


def detections_from_field(field: Field, world: WorldConfig, *,
                          threshold_rel: float = 0.3,
                          min_separation_m: float) -> List[Waypoint]:
    """从概率图(Leader 扫描所得「目標分布確率マップ」)提取离散【候选检测点】。

    = field_from_targets 的**有损逆过程**。算法(确定性、无 RNG):
      1. 候选格 = field >= threshold_rel * field.max() 且为 8 邻域局部极大(>= 全部邻居);
      2. 按 field 值降序贪心 NMS:与已接受点的世界距离 >= min_separation_m 才接受,否则丢弃;
      3. 接受格 → cell 中心世界坐标 ((ix+0.5)*res, (iy+0.5)*res);按 field 值降序稳定排序返回。

    ⚠ 返回点是【估计/检测】,**非 ground truth**:因离散化 + KDE 模糊而偏离真值,可能合并近邻目标、
    漏掉阈下弱峰、或在簇叠加处产生伪峰。下游 baseline 在这些**估计点**上规划(模拟真实探查里
    Follower 规划器只看 Leader 扫描图);评估(TP/FN/FP)始终以 GroundTruthInstance.targets 为**绝对参考**。

    参数标定(无 magic number):min_separation_m ≈ 先验模糊 σ(= 0.5*footprint_lateral_m);
    比 σ 更近的两峰本就不可分辨。threshold_rel 滤掉 max 的该比例以下的晕开尾。
    field 全零/空 → 返回 []。同 field + 同参数 → 逐位复现(开环可复现)。

    归属:实现住 msim/env_static/field.py(A2),与 field_from_targets 配对(纯 numpy,勿引 scipy)。
    """
    raise NotImplementedError  # [A2 波次1·§v1.0.4]


def observation_candidates_from_field(field: Field, world: WorldConfig, *,
                                      threshold_rel: float = 0.3,
                                      fill_threshold_rel: float = 0.5,
                                      peak_min_separation_m: float,
                                      fill_radius_m: float) -> List[Waypoint]:
    """从概率图提取【候选观测航点】(估计,非 GT)= A′ 混合:峰值锚点 + 高概率区覆盖补点。

    §v1.0.4 的纯峰值 detections_from_field 有本质缺陷:σ 模糊把多个近邻目标糊成一**整块**
    高概率区,而单峰 blob 只有一个局部极大 → 一整块被当成**单点**、簇内目标数被吞掉。
    本函数在**保留峰值思想**的前提下修这个缺陷:峰值当锚点(逐点原样保留),再保证整块
    高概率区被「覆盖半径 fill_radius_m」覆盖。

    算法(确定性、无 RNG):
      1. peaks = detections_from_field(field, world, threshold_rel,
                 min_separation_m=peak_min_separation_m);accepted = peaks(逐点保留)。
         (threshold_rel = 峰值/锚点相对阈值,默认 0.3,**不变**。)
      2. 补点高概率区 M = {field >= fill_threshold_rel*field.max()}(§v1.0.6:**与峰值阈值拆开**,
         默认 0.5,比峰值阈值更严 → 补点只落在更确信的密集区/合并簇核心,不在弱晕区撒点)。
         在世界上铺间隔≈fill_radius_m 的栅格,取落在 M 内的格点为补点候选;按 field 值降序
         (同值按 (ix,iy) 升序)依次考察,接受当且仅当与**所有已 accepted 点**(峰值 + 已补点)
         的世界距离 >= fill_radius_m。
      3. 返回 peaks ++ 已接受补点(峰值在前;逐位可复现)。

    ⚠ 返回点是【估计/候选观测点,NOT ground truth】:下游 baseline 在其上规划;评估(TP/FN/FP)
    始终以 GroundTruthInstance.targets 为绝对参考。**不利用生成核 σ 反推**(区别于去模糊,无作弊嫌疑)。

    自适应:孤立小目标(blob 半径 < fill_radius_m)→ 无 > R 空隙 → **不补点**,退化回纯峰值;
    多目标合并的大块 → 按面积补点铺满。field 空/全零 → 返回 []。

    参数(挂物理量、无 magic number):
      - threshold_rel:峰值锚点相对阈值(默认 0.3);fill_threshold_rel:补点高概率区相对阈值
        (默认 0.5,约定 ≥ threshold_rel 更保守;§v1.0.6 与峰值阈值拆开,可独立调)。
      - peak_min_separation_m ≈ 先验模糊 σ(= 0.5*footprint_lateral_m);
      - fill_radius_m = 相机走廊宽(= 0.9*footprint_lateral_m):点距 ≤ R 时相机走廊即可扫遍该区。
    归属:实现住 msim/env_static/field.py(A2),以 detections_from_field 为子步(纯 numpy)。
    """
    raise NotImplementedError  # [A2 波次1·§v1.0.6]


def save_instance(instance: GroundTruthInstance, path: str,
                  metadata: Optional[dict] = None) -> None:
    """落盘 ground-truth 实例(targets + field + cluster_centers + metadata),供复现/对比(Q19)。

    用 .npz;load_instance 须逐位复现。metadata 任意可序列化 dict(seed/簇参数/bandwidth 等)。
    """
    raise NotImplementedError  # [A2 波次1·§v1.0.3]


def load_instance(path: str) -> GroundTruthInstance:
    """加载 save_instance 落盘的 ground-truth 实例(逐位复现 targets 与 field)。"""
    raise NotImplementedError  # [A2 波次1·§v1.0.3]
