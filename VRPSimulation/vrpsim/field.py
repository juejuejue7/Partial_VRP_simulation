"""[第1层] 目標分布確率マップ p(x,y) 的构建。

口径与 msim 合成场**逐位同源**:默认路径直接调用
`msim.env_static.field.field_from_targets`(各向同性高斯 KDE + max 归一化到 [0,1],
`msim/env_static/field.py:44-64`),不另写一套。真实场与合成场只有"目标点从哪来"
的差别,概率语义完全一致 —— 这样两者上的消融结果才可比。

轴序 field[ix, iy](0 轴 = North = x,1 轴 = East = y),同 `msim/contracts/types.py`。
"""
from __future__ import annotations

import numpy as np
from msim.contracts.config import WorldConfig
from msim.contracts.types import FLOAT, Field
from msim.env_static.field import field_from_targets

from .contracts.config import FieldBuildConfig
from .contracts.dataset import MothraDataset

__all__ = ["build_probability_field"]


def _weighted_kde(targets: np.ndarray, weights: np.ndarray,
                  world: WorldConfig, bandwidth_m: float) -> Field:
    """带权高斯 KDE。除权重外与 `field_from_targets` 逐字同构(含归一化口径)。"""
    nx, ny = world.nx, world.ny
    res = world.res_m
    xs = (np.arange(nx, dtype=FLOAT) + 0.5) * res
    ys = (np.arange(ny, dtype=FLOAT) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    inv_two_sigma2 = 1.0 / (2.0 * bandwidth_m ** 2)
    field = np.zeros((nx, ny), dtype=FLOAT)
    for (tx, ty), w in zip(targets, weights):
        d2 = (gx - float(tx)) ** 2 + (gy - float(ty)) ** 2
        field += float(w) * np.exp(-d2 * inv_two_sigma2)

    fmax = float(field.max())
    if fmax > 0.0:
        field /= fmax
    return field.astype(FLOAT)


def build_probability_field(dataset: MothraDataset, world: WorldConfig,
                            cfg: FieldBuildConfig) -> Field:
    """28 个真实热液口 → 目標分布確率マップ,(nx, ny) float64 ∈ [0,1]。

    纯确定性:同 dataset + 同 cfg + 同 world → 逐位同 field(无 RNG)。

    ⚠ 该场是"声学初查后的模糊信念",不是检测掩膜(project_summary §3:graded 非二值)。
      峰值 1.0 落在目标最密处,不一定落在单个目标上。
    """
    if cfg.bandwidth_m <= 0.0:
        raise ValueError(f"bandwidth_m 须 >0,收到 {cfg.bandwidth_m}")

    if cfg.weight_mode == "uniform":
        # 与合成场同一条代码路径,保证语义逐位同源
        return field_from_targets(dataset.as_waypoint_list(), world,
                                  bandwidth_m=cfg.bandwidth_m)

    if cfg.weight_mode == "height":
        # 【待裁决项 2】未经人工裁决不要设为默认。
        h = dataset.height_m.astype(FLOAT)
        if not np.all(h > 0.0):
            raise ValueError("height 加权要求所有 height_m > 0")
        return _weighted_kde(dataset.targets_ned, h / h.max(), world, cfg.bandwidth_m)

    raise ValueError(f"未知 weight_mode {cfg.weight_mode!r},合法值 'uniform' / 'height'")
