"""[第1层][A2] 目標分布確率マップ:ground-truth 耦合实例(真实目标点 + 由其派生的先验场)。

设计基准:contracts/env_static.py(§v1.0.3,人类裁定 2026-06-12):
- **真实目标点 = 绝对 ground truth**,由簇点过程生成(簇中心 + 簇内高斯散布),
  贴合底栖聚集统计(project_summary §13:避免均匀随机)。
- **先验概率场 = 对目标点做各向同性高斯 KDE 后线性归一化到 [0,1]**(模糊信念,带宽 σ≈簇内散布)。
  **场严格由目标点派生,二者不再是独立流。**
- 修士开环:实例在 episode 内静态不变;save/load 可逐位复现(Q19)。

轴序:field[ix, iy](第0轴=North=x,第1轴=East=y),与 contracts/types.py 一致。

兼容:保留旧 API make_clustered_field / save_field / load_field(现有 tests/test_env_static.py 依赖)。
make_clustered_field 现为 field_from_targets 之上的便捷封装(内部生成耦合实例后返回 .field)。
"""
from __future__ import annotations

import json
from typing import List, Optional

import numpy as np

from msim.contracts.config import WorldConfig
from msim.contracts.env_static import GroundTruthInstance
from msim.contracts.types import FLOAT, Field, Waypoint


# ======================================================================
# 先验场:严格由目标点派生(各向同性高斯 KDE → 归一化 [0,1])
# ======================================================================
def field_from_targets(targets: List[Waypoint], world: WorldConfig,
                       *, bandwidth_m: float) -> Field:
    """先验场 = Σ_target 各向同性高斯核(中心=target,带宽=bandwidth_m)叠加后归一化到 [0,1]。

    纯几何确定:同 targets + 同 bandwidth → 逐位同 field。轴序 field[ix, iy]。
    bandwidth_m 即"先验不确定性"(σ):越大越模糊;约定 σ≈簇内散布(模糊信念)。
    targets 为空 → 返回全零场。
    """
    if bandwidth_m <= 0.0:
        raise ValueError(f"bandwidth_m 须 >0,收到 {bandwidth_m}")

    nx, ny = world.nx, world.ny
    res = world.res_m

    field = np.zeros((nx, ny), dtype=FLOAT)
    if len(targets) == 0:
        return field  # 无目标 → 全零场(契约约定)

    # cell 中心的世界坐标网格(NED:axis0=North=x, axis1=East=y)。
    xs = (np.arange(nx, dtype=FLOAT) + 0.5) * res          # (nx,)
    ys = (np.arange(ny, dtype=FLOAT) + 0.5) * res          # (ny,)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")            # (nx, ny) each

    inv_two_sigma2 = 1.0 / (2.0 * bandwidth_m ** 2)
    for t in targets:
        tx = float(t[0])
        ty = float(t[1])
        d2 = (gx - tx) ** 2 + (gy - ty) ** 2
        field += np.exp(-d2 * inv_two_sigma2)

    # 线性归一化到 [0,1];兜底全零(targets 非空时 fmax>0)。
    fmax = float(field.max())
    if fmax > 0.0:
        field /= fmax
    return field.astype(FLOAT)


# ======================================================================
# 先验场的有损逆过程:从概率图提取离散【估计/检测点】(estimate, NOT ground truth)
# ======================================================================
def detections_from_field(field: Field, world: WorldConfig, *,
                          threshold_rel: float = 0.3,
                          min_separation_m: float) -> List[Waypoint]:
    """从概率图(Leader 扫描所得「目標分布確率マップ」)提取离散【候选检测点】。

    = field_from_targets 的**有损逆过程**(确定性、无 RNG)。

    ⚠ 返回点是【估计/检测点(estimate, NOT ground truth)】:因离散化 + KDE 模糊而偏离
    真值,可能合并近邻目标、漏掉阈下弱峰、或在簇叠加处产生伪峰。下游 baseline 在这些
    **估计点**上规划;评估(TP/FN/FP)始终以 GroundTruthInstance.targets 为**绝对参考**。

    算法:
      1. field 空 / field.max() <= 0 → 返回 []。
      2. thr = threshold_rel * field.max()。
      3. 候选格 = field >= thr 且为 8 邻域局部极大(>= 全部存在的邻居;边界格仅比存在邻居)。
      4. 贪心 NMS:候选按 field 值降序(同值再按 (ix,iy) 升序)排序;依次接受,当且仅当
         与所有已接受点的世界距离 >= min_separation_m;否则丢弃。
      5. 接受格 → cell 中心世界坐标 ((ix+0.5)*res, (iy+0.5)*res),与 field_from_targets 同约定。
      6. 返回 List[np.ndarray (2,) float],按 field 值降序稳定排序(同值再按 (ix,iy) 升序),逐位可复现。

    轴序 field[ix, iy](第0轴=North=x,第1轴=East=y)。同 field + 同参数 → 逐位相同输出。
    纯 numpy(勿引 scipy/skimage)。
    """
    field = np.asarray(field)
    # 1. 空场 / 退化场 → 无检测点。
    if field.size == 0:
        return []
    fmax = float(field.max())
    if fmax <= 0.0:
        return []

    res = world.res_m
    nx, ny = field.shape[0], field.shape[1]

    # 2. 相对阈值。
    thr = threshold_rel * fmax

    # 3. 8 邻域局部极大(纯 numpy 移位比较)。
    #    is_peak[ix,iy] = field[ix,iy] >= 所有存在的 8 邻居。边界处缺失的邻居不参与比较
    #    (即仅与存在的邻居比),通过逐方向移位 + 仅在重叠区域施加约束实现。
    is_peak = np.ones((nx, ny), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            # 源切片 = 中心格;目标切片 = 该方向的邻居格。仅在两者都存在的重叠区比较。
            src_x = slice(max(0, -dx), nx - max(0, dx))
            src_y = slice(max(0, -dy), ny - max(0, dy))
            nbr_x = slice(max(0, dx), nx - max(0, -dx))
            nbr_y = slice(max(0, dy), ny - max(0, -dy))
            ge = field[src_x, src_y] >= field[nbr_x, nbr_y]
            is_peak[src_x, src_y] &= ge

    candidate_mask = is_peak & (field >= thr)

    cand_ix, cand_iy = np.nonzero(candidate_mask)
    if cand_ix.size == 0:
        return []
    cand_vals = field[cand_ix, cand_iy]

    # 候选排序:field 值降序;同值按 (ix,iy) 升序(逐位可复现的确定序)。
    # lexsort 末键为主键:主 = -value(升序 → value 降序),次 = ix,末 = iy。
    order = np.lexsort((cand_iy, cand_ix, -cand_vals))
    cand_ix = cand_ix[order]
    cand_iy = cand_iy[order]

    # 4. 贪心 NMS:按上面顺序逐个接受,世界距离 >= min_separation_m 才接受。
    min_sep2 = float(min_separation_m) ** 2
    accepted_x: List[float] = []   # 已接受点世界 x(=North)
    accepted_y: List[float] = []   # 已接受点世界 y(=East)
    detections: List[Waypoint] = []
    for ix, iy in zip(cand_ix.tolist(), cand_iy.tolist()):
        # 5. cell 中心世界坐标(与 field_from_targets 构造网格同约定)。
        wx = (ix + 0.5) * res
        wy = (iy + 0.5) * res
        keep = True
        for ax, ay in zip(accepted_x, accepted_y):
            if (wx - ax) ** 2 + (wy - ay) ** 2 < min_sep2:
                keep = False
                break
        if keep:
            accepted_x.append(wx)
            accepted_y.append(wy)
            # 6. 估计/检测点(estimate, NOT ground truth)。
            detections.append(np.array([wx, wy], dtype=FLOAT))

    # detections 已按 (field 降序, (ix,iy) 升序) 的接受顺序排列 → 即所需返回序。
    return detections


# ======================================================================
# A′ 混合:峰值锚点 + 高概率区覆盖补点(估计/候选观测点,NOT ground truth)
# ======================================================================
def observation_candidates_from_field(field: Field, world: WorldConfig, *,
                                      threshold_rel: float = 0.3,
                                      fill_threshold_rel: float = 0.5,
                                      peak_min_separation_m: float,
                                      fill_radius_m: float) -> List[Waypoint]:
    """从概率图提取【候选观测航点】(估计,NOT ground truth)= A′ 混合:峰值锚点 + 高概率区覆盖补点。

    §v1.0.4 的纯峰值 detections_from_field 有本质缺陷:σ 模糊把多个近邻目标糊成一**整块**
    高概率区,而单峰 blob 只有一个 8 邻域局部极大 → 整块被当成**单点**、簇内目标数被吞掉。
    本函数在**保留峰值思想**的前提下修这个缺陷:峰值当锚点(逐点原样保留、顺序在前),
    再在「覆盖半径 fill_radius_m」下确保整块高概率区被点覆盖(贪心铺点)。

    §v1.0.6:**补点高概率区阈值与峰值阈值拆开**——峰值锚点仍用 threshold_rel(默认 0.3),
    补点区改用独立的 fill_threshold_rel(默认 0.5,更严)。二者可独立调:提高 fill_threshold_rel
    使高概率区收缩 → 补点数单调不增,且峰值集合**只随 threshold_rel 变、与 fill_threshold_rel 无关**。

    算法(确定性、**无 RNG**;复用 detections_from_field 为子步):
      1. 峰值锚点:peaks = detections_from_field(field, world, threshold_rel,
         min_separation_m=peak_min_separation_m);accepted = list(peaks)(逐点原样保留,顺序在前)。
         (峰值阈值用 threshold_rel,与 fill_threshold_rel 无关。)
      2. 补点高概率区 M = {field >= fill_threshold_rel * field.max()}(§v1.0.6:与峰值阈值拆开)。
      3. 栅格补点候选:step = max(1, round(fill_radius_m / res));在 ix∈range(step//2, nx, step)、
         iy∈range(step//2, ny, step) 的格点上,若 M[ix,iy] 为真,记 (field[ix,iy], ix, iy)。
      4. 补点候选按 field 值**降序**(同值按 (ix,iy) 升序)排序——与 detections_from_field
         同口径(lexsort),逐位可复现。
      5. 贪心 dedup:依次考察补点候选,算其 cell 中心世界坐标 ((ix+0.5)*res,(iy+0.5)*res),
         **当且仅当**与**所有已 accepted 点**(峰值 + 已接受补点)的世界距离 >= fill_radius_m
         时接受并加入 accepted。
      6. 返回 accepted(峰值在前、补点在后;均为 np.ndarray shape (2,) float)。

    ⚠ 返回点是【估计/候选观测点,NOT ground truth】:因离散化 + KDE 模糊而偏离真值;下游 baseline
    在其上规划(模拟 Follower 规划器只看 Leader 扫描图),评估(TP/FN/FP)始终以
    GroundTruthInstance.targets 为**绝对参考**。**不利用生成核 σ 反推**(区别于去模糊,无作弊嫌疑)。

    不变量:
      - 峰值保留:返回 ⊇ detections_from_field 同参输出(每个峰值点逐位都在结果里、且在最前面)。
      - 自适应:孤立小目标(blob 半径 < fill_radius_m)→ 无 >R 空隙 → 退化回纯峰值(不补点);
        多目标合并大块 → 返回点数 > 峰值点数。
      - field 空 / 全零 → 返回 []。同 field + 同参数 → 逐位相同输出(无 RNG)。

    轴序 field[ix, iy](第0轴=North=x,第1轴=East=y)。纯 numpy(勿引 scipy/skimage)。
    """
    field = np.asarray(field)
    # 空场 / 退化场 → 无候选(与 detections_from_field 同口径)。
    if field.size == 0:
        return []
    fmax = float(field.max())
    if fmax <= 0.0:
        return []

    res = world.res_m
    nx, ny = field.shape[0], field.shape[1]

    # 1. 峰值锚点(逐点原样保留,顺序在前)。复用已实现的 detections_from_field 为子步。
    peaks = detections_from_field(field, world,
                                  threshold_rel=threshold_rel,
                                  min_separation_m=peak_min_separation_m)
    accepted: List[Waypoint] = list(peaks)
    # 已 accepted 点的世界坐标缓存(峰值 + 后续补点),用于 dedup 距离判定。
    accepted_x: List[float] = [float(p[0]) for p in accepted]
    accepted_y: List[float] = [float(p[1]) for p in accepted]

    # 2. 补点高概率区 M = {field >= fill_threshold_rel * field.max()}(§v1.0.6:与峰值阈值拆开,
    #    默认 0.5 更严 → 补点只落在更确信的密集区/合并簇核心,不在弱晕区撒点)。
    thr = fill_threshold_rel * fmax
    M = field >= thr

    # 3. 栅格补点候选:间隔 step(≈ fill_radius_m / res),从 step//2 起步铺格点;落在 M 内的为候选。
    step = max(1, int(round(fill_radius_m / res)))
    fill_ix = np.arange(step // 2, nx, step)
    fill_iy = np.arange(step // 2, ny, step)
    if fill_ix.size == 0 or fill_iy.size == 0:
        return accepted  # 场比一个 step 还小 → 无栅格候选,退化回纯峰值。

    gx, gy = np.meshgrid(fill_ix, fill_iy, indexing="ij")   # (len_ix, len_iy)
    cand_ix = gx.ravel()
    cand_iy = gy.ravel()
    in_M = M[cand_ix, cand_iy]
    cand_ix = cand_ix[in_M]
    cand_iy = cand_iy[in_M]
    if cand_ix.size == 0:
        return accepted  # 栅格格点无一落在高概率区 → 退化回纯峰值。
    cand_vals = field[cand_ix, cand_iy]

    # 4. 排序:field 值降序;同值按 (ix,iy) 升序(与 detections_from_field 同口径,逐位可复现)。
    #    lexsort 末键为主键:主 = -value(value 降序),次 = ix,末 = iy。
    order = np.lexsort((cand_iy, cand_ix, -cand_vals))
    cand_ix = cand_ix[order]
    cand_iy = cand_iy[order]

    # 5. 贪心 dedup:与所有已 accepted 点(峰值 + 已补点)世界距离 >= fill_radius_m 才接受。
    fill_r2 = float(fill_radius_m) ** 2
    for ix, iy in zip(cand_ix.tolist(), cand_iy.tolist()):
        wx = (ix + 0.5) * res
        wy = (iy + 0.5) * res
        keep = True
        for ax, ay in zip(accepted_x, accepted_y):
            if (wx - ax) ** 2 + (wy - ay) ** 2 < fill_r2:
                keep = False
                break
        if keep:
            accepted_x.append(wx)
            accepted_y.append(wy)
            # 估计/候选观测点(estimate, NOT ground truth)。
            accepted.append(np.array([wx, wy], dtype=FLOAT))

    # 6. 峰值在前、补点在后,逐位可复现。
    return accepted


# ======================================================================
# 耦合实例:先撒 ground-truth 目标点(簇点过程),再由其派生先验场
# ======================================================================
def make_clustered_instance(world: WorldConfig, rng: np.random.Generator, *,
                            n_clusters: int, targets_per_cluster: int,
                            intra_spread_m: float,
                            bandwidth_m: Optional[float] = None) -> GroundTruthInstance:
    """生成耦合实例:**先**按簇点过程撒 ground-truth 目标点,**再**由其派生先验场。

    步骤:
      1. 撒 n_clusters 个簇中心(场内**带 margin 内区**均匀,margin≈intra_spread,避免靠边)。
      2. 每个簇中心周围按 N(0, intra_spread_m) 高斯散布采样 targets_per_cluster 个目标点;
         **不 clip**:落在场外的点**拒绝重采样**(带上限 + 兜底),且与已采目标**两两不重合**
         → 共 K=n_clusters×targets_per_cluster 个 distinct、严格在场内的 ground-truth 目标点(绝对参考)。
      3. field = field_from_targets(targets, world, bandwidth_m or intra_spread_m)。
         bandwidth_m 默认 = intra_spread_m(模糊信念,§v1.0.3 裁定)。

    不变量:① 所有目标严格在场内(0<x<x_max 且 0<y<y_max,不贴边堆叠);
    ② K 个目标两两不重合(distinct)。
    返回 GroundTruthInstance(targets, field, cluster_centers)。
    确定性:同 rng 状态 + 同参数 → 逐位同实例(供复现验证)。
    """
    if n_clusters < 1:
        raise ValueError(f"n_clusters 须 >=1,收到 {n_clusters}")
    if targets_per_cluster < 1:
        raise ValueError(f"targets_per_cluster 须 >=1,收到 {targets_per_cluster}")
    if intra_spread_m <= 0.0:
        raise ValueError(f"intra_spread_m 须 >0,收到 {intra_spread_m}")

    x_max = world.x_max_m
    y_max = world.y_max_m

    # 严格内部边距:留出极小 eps 使 0<x<x_max(而非贴边 0/x_max);非 clip。
    eps = 1e-9

    def _in_field(px: float, py: float) -> bool:
        return eps < px < x_max - eps and eps < py < y_max - eps

    # ① 簇中心:放在带 margin 的内区(margin≈intra_spread,且不超过半场,防小场退化)。
    mx = min(intra_spread_m, 0.49 * x_max)
    my = min(intra_spread_m, 0.49 * y_max)
    cluster_centers: List[Waypoint] = []
    for _ in range(n_clusters):
        cx = float(rng.uniform(mx, x_max - mx))
        cy = float(rng.uniform(my, y_max - my))
        cluster_centers.append(np.array([cx, cy], dtype=FLOAT))

    # ② 簇内高斯散布采样 → 严格在场内、两两不重合的 ground-truth 目标点。
    #    出界点拒绝重采样;重合点(与任一已采目标完全相同)亦重采样。带上限 + 兜底。
    max_tries = 10000
    seen: set = set()                      # 已采目标的 (x,y) 精确键,用于 distinct 判定
    targets: List[Waypoint] = []
    for center in cluster_centers:
        for _ in range(targets_per_cluster):
            tx = ty = 0.0
            ok = False
            for _try in range(max_tries):
                tx = float(center[0] + rng.normal(0.0, intra_spread_m))
                ty = float(center[1] + rng.normal(0.0, intra_spread_m))
                if _in_field(tx, ty) and (tx, ty) not in seen:
                    ok = True
                    break
            if not ok:
                # 兜底(极端参数下重采样上限耗尽):在内区均匀补一个 distinct、场内点。
                for _try in range(max_tries):
                    tx = float(rng.uniform(eps, x_max - eps))
                    ty = float(rng.uniform(eps, y_max - eps))
                    if (tx, ty) not in seen:
                        break
            seen.add((tx, ty))
            targets.append(np.array([tx, ty], dtype=FLOAT))

    # ③ 先验场严格由目标点派生(默认 bandwidth=intra_spread,模糊信念)。
    bw = bandwidth_m if bandwidth_m is not None else intra_spread_m
    field = field_from_targets(targets, world, bandwidth_m=bw)

    return GroundTruthInstance(targets=targets, field=field,
                               cluster_centers=cluster_centers)


# ======================================================================
# 落盘 / 加载(逐位复现 targets + field;Q19)
# ======================================================================
def save_instance(instance: GroundTruthInstance, path: str,
                  metadata: Optional[dict] = None) -> None:
    """落盘 ground-truth 实例(targets + field + cluster_centers + metadata),供复现/对比(Q19)。

    用 .npz;load_instance 须逐位复现。metadata 任意可序列化 dict(seed/簇参数/bandwidth 等)。
    空 targets / cluster_centers 也安全(存为 shape (0,2) 数组)。
    """
    meta_json = json.dumps(metadata if metadata is not None else {})
    targets_arr = (np.asarray(instance.targets, dtype=FLOAT)
                   if len(instance.targets) > 0
                   else np.zeros((0, 2), dtype=FLOAT))
    centers_arr = (np.asarray(instance.cluster_centers, dtype=FLOAT)
                   if len(instance.cluster_centers) > 0
                   else np.zeros((0, 2), dtype=FLOAT))
    # 显式写入文件句柄,避免 np.savez 在无 .npz 后缀时自动追加导致 save/load 路径错位。
    with open(path, "wb") as fh:
        np.savez(fh,
                 targets=targets_arr,
                 field=np.asarray(instance.field, dtype=FLOAT),
                 cluster_centers=centers_arr,
                 metadata=meta_json)


def load_instance(path: str) -> GroundTruthInstance:
    """加载 save_instance 落盘的 ground-truth 实例(逐位复现 targets 与 field)。"""
    with open(path, "rb") as fh:
        with np.load(fh, allow_pickle=False) as data:
            targets_arr = data["targets"].astype(FLOAT)
            field = data["field"].astype(FLOAT)
            centers_arr = data["cluster_centers"].astype(FLOAT)
    targets: List[Waypoint] = [targets_arr[i] for i in range(targets_arr.shape[0])]
    cluster_centers: List[Waypoint] = [centers_arr[i] for i in range(centers_arr.shape[0])]
    return GroundTruthInstance(targets=targets, field=field,
                               cluster_centers=cluster_centers)


# ======================================================================
# 兼容旧 API(现有 tests/test_env_static.py 依赖;勿破坏)
# ======================================================================
def make_clustered_field(world: WorldConfig, rng: np.random.Generator,
                         n_clusters: int, intra_density: float) -> Field:
    """[兼容封装] 生成带簇结构的静态概率场 field[ix, iy] ∈ [0,1](NED 轴序)。

    现为 field_from_targets 之上的便捷封装:内部按簇点过程生成耦合实例后返回 .field
    (§v1.0.3:场严格由目标点派生)。intra_density 越大簇越尖锐 → 映射为越小的簇内散布/带宽。

    确定性:同 rng 状态 + 同参数 → 逐位同 field。
    """
    if n_clusters < 1:
        raise ValueError(f"n_clusters 须 >=1,收到 {n_clusters}")
    if intra_density <= 0.0:
        raise ValueError(f"intra_density 须 >0,收到 {intra_density}")

    # 簇尺度:intra_density 越大 → spread/带宽 越小(更紧致)。以短边为参考尺度,
    # 不小于一个 cell;每簇取单点目标(旧语义只需 graded 簇隆起)。
    base_scale = min(world.x_max_m, world.y_max_m)
    spread = max(base_scale / (intra_density * 10.0), world.res_m)
    instance = make_clustered_instance(
        world, rng,
        n_clusters=n_clusters, targets_per_cluster=1,
        intra_spread_m=spread, bandwidth_m=spread,
    )
    return instance.field


def save_field(field: Field, path: str, metadata: Optional[dict] = None) -> None:
    """[兼容] 落盘 groundtruth 概率场 + metadata(Q19)。

    用 .npz 存:field 数组 + 一个 metadata json 字符串(任意可序列化 dict)。
    """
    meta_json = json.dumps(metadata if metadata is not None else {})
    # 显式写入文件句柄,避免 np.savez 在无 .npz 后缀时自动追加导致 save/load 路径错位。
    with open(path, "wb") as fh:
        np.savez(fh, field=np.asarray(field, dtype=FLOAT), metadata=meta_json)


def load_field(path: str) -> Field:
    """[兼容] 加载已保存的 groundtruth 概率场(逐位复现 save 的数组)。"""
    with open(path, "rb") as fh:
        with np.load(fh, allow_pickle=False) as data:
            return data["field"].astype(FLOAT)
