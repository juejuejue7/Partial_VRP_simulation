"""[第0层][A0 冻结] 本仿真的配置契约。

世界几何直接复用 `msim.contracts.config.WorldConfig`(同一套 NED 语义,不另造),
本文件只加真实数据特有的部分:数据路径、裁切范围、窗口几何、概率场构建口径、落盘位置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from msim.contracts.config import SensorConfig, WindowConfig, WorldConfig

from .frames import (GRID_1M_RES_M, RASTER_EAST_EXTENT_M, RASTER_NORTH_EXTENT_M,
                     WINDOW_LOOK_BACK_M, WINDOW_WIDTH_M, WORLD_EAST_EXTENT_M,
                     WORLD_NORTH_EXTENT_M)

# 仓库根(= Master_research/),用于定位 msim 与上游数据
REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
SIM_ROOT: str = os.path.join(REPO_ROOT, "VRPSimulation")

DEFAULT_WAYPOINTS_CSV: str = os.path.join(SIM_ROOT, "waypoints", "mothra_waypoints.csv")
DEFAULT_DATA_DIR: str = os.path.join(SIM_ROOT, "data")
DEFAULT_FIGURE_DIR: str = os.path.join(SIM_ROOT, "figures")

# 上游水深数据(D3:**不是**环境层,仅 viz 底图可选读取)
UPSTREAM_BUNDLE_NPZ: str = os.path.join(
    REPO_ROOT, "Bethmetory_data_process", "outputs", "mothra_bundle.npz")


@dataclass(frozen=True)
class CropConfig:
    """分析域裁切(D7)。原点**不变**,只缩小外框。

    默认 North [0,500) x East [0,100):丢掉 6 个目标(wp23 孤点 + wp24-28 Cauldron 区),
    留下 22 个 = 14 chimney + 8 mound,范围 North 76.8-442.1 / East 6.7-94.8。

    为什么裁:
    - Cauldron 一簇(wp24-28)在 North 610-632,与主群相隔 80 m 空白,拖长测线 130 m。
    - wp23 是孤点(North 529.8 / East 143.7,离最近目标 85.9 m),它一个点就把 East
      从 1 条车道撑到 2 条、North 从 5 段撑到 6 段 —— Leader 要为它多飞一整条 600 m 测线。
    裁完后 East 跨度 100 m == 窗口宽度 ⇒ Leader 单条测线全覆盖,任务结构最干净。

    ⚠ 裁切是**分析域**的事,不改原始数据:`load_waypoints` 照旧读全部 28 个点并按
      原始栅格外框校验,裁切在 `MothraDataset.crop()` 里显式发生,被丢的点可追溯。
    """
    north_m: Tuple[float, float] = (0.0, WORLD_NORTH_EXTENT_M)
    east_m: Tuple[float, float] = (0.0, WORLD_EAST_EXTENT_M)

    @property
    def extent(self) -> Tuple[float, float]:
        """(North 跨度, East 跨度)。"""
        return (self.north_m[1] - self.north_m[0], self.east_m[1] - self.east_m[0])

    def contains(self, x_north_m, y_east_m):
        """半开区间 [lo, hi):落在上边界正好被排除,避免同一点归两个域。"""
        return ((x_north_m >= self.north_m[0]) & (x_north_m < self.north_m[1])
                & (y_east_m >= self.east_m[0]) & (y_east_m < self.east_m[1]))


def full_raster_crop() -> CropConfig:
    """不裁切(全幅 653x294),供溯源与对照实验。"""
    return CropConfig(north_m=(0.0, RASTER_NORTH_EXTENT_M),
                      east_m=(0.0, RASTER_EAST_EXTENT_M))


def mothra_world_config(res_m: float = GRID_1M_RES_M,
                        crop: Optional[CropConfig] = None) -> WorldConfig:
    """Mothra 分析域的 WorldConfig。

    res_m 默认 1.0:与 1 m 水深栅格逐格同构,将来加地形层可 bit-exact 落位、
    无需重采样。msim 合成场默认 0.2 m —— 真实场用 0.2 m 会到百万级 cell,不划算。

    ⚠ 原点固定在栅格西南角(D2),所以 crop 的下界目前必须是 0 —— 非零下界会引入
      平移项,让 NED 坐标与栅格索引脱钩。需要时提交 A0 裁决再放开。
    """
    crop = crop or CropConfig()
    if crop.north_m[0] != 0.0 or crop.east_m[0] != 0.0:
        raise ValueError(
            f"crop 下界必须为 0(原点锁在栅格西南角,D2),收到 {crop.north_m=} {crop.east_m=}")
    n_ext, e_ext = crop.extent
    return WorldConfig(x_max_m=n_ext, y_max_m=e_ext, res_m=res_m)


def mothra_window_config(advance_threshold_m: float = 50.0,
                         look_back_m: float = WINDOW_LOOK_BACK_M,
                         width_m: float = WINDOW_WIDTH_M) -> WindowConfig:
    """Leader 后方**滑动**观测窗口(D8)。窗口前边界恒与 Leader 位置重合。

    look_back = width = 100 m(默认)。width == 场地 East 跨度 ⇒ 窗口是
    "全 East 的 North 条带",Leader 走单条测线(East=50)即可让窗口扫过全场。
    长宽都是可调参数(任务规格 §3)。

    advance_threshold_m:窗口推进多少米触发一次重解(几何枚举用)。
        默认 50 m = 半个窗口 ⇒ 相邻两次重解的窗口重叠一半,全程 500/50 = 10 次。
        ⚠ 该值属【待裁决项 5】。任务仿真里改用**时间周期**触发(见 mission.py),
          这个空间阈值只用于 `windows.enumerate_windows` 的静态诊断。
    """
    return WindowConfig(look_back_m=look_back_m, width_m=width_m,
                        advance_threshold_m=advance_threshold_m)


WeightMode = Literal["uniform", "height"]


@dataclass(frozen=True)
class FieldBuildConfig:
    """目標分布確率マップ p(x,y) 的构建口径。

    bandwidth_m:
        各向同性高斯 KDE 的 σ。沿用 msim `TargetFieldConfig.bandwidth_m` 的 3.0 m
        (= 0.5 x footprint_lateral)。本场地实测最近邻中位距 6.6 m ≈ 2.2σ,
        簇结构不会糊成一片;最近的一对相距 2.7 m ≈ 0.9σ,会合并成单峰(符合预期:
        它们本就在同一个复合体内)。
    weight_mode:
        "uniform" —— 每个目标核权重为 1,与 msim `field_from_targets` **逐位同源**,
                     真实场与合成场的概率语义一致,消融对比才成立。**默认**。
        "height"  —— 按 height_m 归一化后加权(4~18 m,差 4.5 倍)。
                     ⚠ 是否该按烟囱高度加权属【待裁决项 2】,未经裁决不要改默认值。
    """
    bandwidth_m: float = 3.0
    weight_mode: WeightMode = "uniform"


@dataclass(frozen=True)
class MothraSimConfig:
    """构建一个静态基础世界所需的全部输入。"""
    waypoints_csv: str = DEFAULT_WAYPOINTS_CSV
    res_m: float = GRID_1M_RES_M
    crop: CropConfig = field(default_factory=CropConfig)
    field_build: FieldBuildConfig = field(default_factory=FieldBuildConfig)
    window_advance_threshold_m: float = 50.0
    lane_spacing_m: float = WINDOW_WIDTH_M     # == 窗口宽 ⇒ 单条测线全覆盖
    strict_dataset: bool = True

    @property
    def world(self) -> WorldConfig:
        return mothra_world_config(self.res_m, self.crop)

    @property
    def window(self) -> WindowConfig:
        return mothra_window_config(self.window_advance_threshold_m)
