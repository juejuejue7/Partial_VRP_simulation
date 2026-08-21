"""[第0层][A0 冻结] 真实目标数据集契约 —— Mothra 28 个热液口的 NED 表示。

数据来源链路(只读,本仿真不改上游):
    Bethy_data/TABLE_SI.xlsx (Supplemental Table S1, 572 点)
      → Bethmetory_data_process/scripts/06_extract_vents.py   (按 Mothra bbox 筛出 28 点)
      → Bethmetory_data_process/scripts/07_make_waypoints.py  (附局部米制坐标)
      → VRPSimulation/waypoints/mothra_waypoints.csv          (本仿真的唯一输入)

==================================================================
⚠ 轴序陷阱(全项目最高风险点,A0 在此明文钉死)
==================================================================
CSV 的列名与本项目 NED 的轴**正好相反**:

    CSV  x_local_m  =  East    ←→  NED  y_E
    CSV  y_local_m  =  North   ←→  NED  x_N

装载时必须交换。写反了场地会整体转置,而且**不会有任何报错** ——
`MothraDataset` 的构造器与 `tests/test_dataset_ned.py` 各设一道断言拦这件事。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# 目标形态学分类(源表 morphology 列的全部取值)
TARGET_TYPES: Tuple[str, ...] = ("chimney", "mound")

# 源 CSV 的固定规模(源自 Mothra 裁剪 bbox,变了就说明输入被换过)
N_TARGETS_EXPECTED: int = 28
N_CHIMNEY_EXPECTED: int = 20
N_MOUND_EXPECTED: int = 8

# 裁切(D7,North[0,500) x East[0,100))之后的规模。丢掉的 6 个全是 chimney:
#   wp23  N529.8 E143.7  h=12  (无名)      —— 孤点,离最近目标 85.9 m
#   wp24  N610.2 E194.0  h=11  (无名)      \
#   wp25  N611.9 E168.8  h= 4  (无名)       |
#   wp26  N612.9 E186.2  h=11  Cauldron     |—— Cauldron 区,与主群隔 80 m 空白
#   wp27  N621.7 E184.2  h= 5  (无名)       |
#   wp28  N632.5 E195.2  h= 4  (无名)      /
N_TARGETS_CROPPED: int = 22
N_CHIMNEY_CROPPED: int = 14
N_MOUND_CROPPED: int = 8
DROPPED_WAYPOINT_IDS: Tuple[int, ...] = (23, 24, 25, 26, 27, 28)

# CSV 必须具备的列(缺一即拒绝加载)
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "waypoint_id", "sequence_id", "lon", "lat", "type", "height_m",
    "x_local_m", "y_local_m", "seafloor_depth_m",
    "utm9n_easting", "utm9n_northing", "name",
)

# 自算 NED 与 CSV 既有列的交叉核对容差。
# CSV 的米制列只存 3 位小数 ⇒ 舍入下限就是 0.5 mm,这里留 2 倍余量。
CROSS_CHECK_TOL_M: float = 1e-3


@dataclass(frozen=True)
class MothraDataset:
    """28 个真实热液口在 NED 帧下的目标表(不可变;episode 内静态)。

    不变量(A1 验收):
    - `targets_ned[:, 0]` 是 North、`targets_ned[:, 1]` 是 East(**已相对 CSV 换轴**)。
    - 所有点落在 [0, NORTH_EXTENT_M] x [0, EAST_EXTENT_M] 内。
    - `depth_D_m` 向下为正(= -seafloor_depth_m),量级 ~2.25e3 m。
    - 各数组第 0 维长度一致,顺序 = CSV 行序 = `waypoint_id` 升序。
    """
    targets_ned: np.ndarray      # (N, 2) float64 = [x_N, y_E]
    depth_D_m: np.ndarray        # (N,)   float64,向下为正
    height_m: np.ndarray         # (N,)   float64,烟囱/丘体高出海底的高度(≠ 水深)
    type_: np.ndarray            # (N,)   <U7,   chimney / mound
    name: np.ndarray             # (N,)   <U32,  文献命名,无名者为空串
    waypoint_id: np.ndarray      # (N,)   int32, 1..N 稳定标签,**非访问顺序**
    sequence_id: np.ndarray      # (N,)   int32, 源表 Sequence ID,供回溯
    lon: np.ndarray              # (N,)   float64,溯源用
    lat: np.ndarray              # (N,)   float64,溯源用

    @property
    def n(self) -> int:
        return int(self.targets_ned.shape[0])

    def select(self, mask: np.ndarray) -> "MothraDataset":
        """按布尔掩膜取子集,所有字段同步切片(顺序不变)。

        裁切(D7)就是它的一个应用:`ds.select(crop.contains(N, E))`。
        NED 坐标**逐位不变** —— 裁切只缩小外框,不平移原点。
        """
        m = np.asarray(mask, dtype=bool)
        if m.shape != (self.n,):
            raise ValueError(f"mask 形状应为 ({self.n},),收到 {m.shape}")
        return MothraDataset(
            targets_ned=self.targets_ned[m], depth_D_m=self.depth_D_m[m],
            height_m=self.height_m[m], type_=self.type_[m], name=self.name[m],
            waypoint_id=self.waypoint_id[m], sequence_id=self.sequence_id[m],
            lon=self.lon[m], lat=self.lat[m])

    def as_waypoint_list(self) -> List[np.ndarray]:
        """转成 msim 契约的 `List[Waypoint]`(每个 (2,) float64 = [x_N, y_E])。

        `msim.env_static.field.field_from_targets` 与 `msim.baselines.*` 都吃这个形状。
        """
        return [np.asarray(p, dtype=np.float64) for p in self.targets_ned]

    def mask_of_type(self, t: str) -> np.ndarray:
        if t not in TARGET_TYPES:
            raise ValueError(f"未知目标类型 {t!r},合法值 {TARGET_TYPES}")
        return self.type_ == t


def load_waypoints(csv_path: str, *, strict: bool = True) -> MothraDataset:
    """读 `mothra_waypoints.csv` → NED 目标表。

    实现须满足(见 vrpsim/dataset.py):
    1. 以 `utf-8-sig` 解码(源文件带 BOM),换行 CRLF。
    2. NED 由 `lon/lat` 经 `geodesy.wgs84_to_ned` **重新算出**,而不是直接搬 CSV 的
       `x_local_m/y_local_m` —— 再与之交叉核对(容差 CROSS_CHECK_TOL_M)。
       这样 CSV 若被上游改动/错列,会当场炸而不是静默带病运行。
    3. 换轴:x_N ← y_local_m,y_E ← x_local_m。
    4. `strict=True` 时额外断言规模为 28 = 20 chimney + 8 mound。

    Raises:
        ValueError: 缺列 / 交叉核对超差 / 点越界 / strict 下规模不符。
    """
    raise NotImplementedError  # 实现在 vrpsim/dataset.py
