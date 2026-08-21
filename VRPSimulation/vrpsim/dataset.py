"""[第0层] `mothra_waypoints.csv` → NED 目标表。

契约见 `vrpsim/contracts/dataset.py`。本文件只做加载与校验,不做任何几何规划。

装载策略:**不信列值,重算再对拍。**
CSV 里已经有 `x_local_m/y_local_m`,直接搬过来最省事 —— 但那样上游一旦改了原点或
错位一列,仿真会静默地在错误坐标上跑完全程。这里改为由 `lon/lat` 重新算 NED,
再与 CSV 的米制列逐点核对(容差 1 mm);不一致当场抛。
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np

from .contracts.dataset import (
    CROSS_CHECK_TOL_M,
    N_CHIMNEY_EXPECTED,
    N_MOUND_EXPECTED,
    N_TARGETS_EXPECTED,
    REQUIRED_COLUMNS,
    TARGET_TYPES,
    MothraDataset,
)
from .contracts.frames import RASTER_EAST_EXTENT_M, RASTER_NORTH_EXTENT_M
from .geodesy import ned_to_utm9n, wgs84_to_ned

__all__ = ["load_waypoints"]


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"waypoint 表不存在:{csv_path}")
    # utf-8-sig:源文件由 07_make_waypoints.py 写出,带 BOM。
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"waypoint 表为空:{csv_path}")

    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"waypoint 表缺列 {missing};实际列 {list(rows[0].keys())}")
    return rows


def load_waypoints(csv_path: str, *, strict: bool = True) -> MothraDataset:
    """读 waypoint CSV → `MothraDataset`(NED,已换轴)。

    见 `vrpsim/contracts/dataset.py::load_waypoints` 的契约说明。
    """
    rows = _read_rows(csv_path)

    lon = np.array([float(r["lon"]) for r in rows], dtype=np.float64)
    lat = np.array([float(r["lat"]) for r in rows], dtype=np.float64)
    z_up = np.array([float(r["seafloor_depth_m"]) for r in rows], dtype=np.float64)

    # --- 授权换算:经纬度 → NED(D2) ---------------------------------
    x_north, y_east, z_down = wgs84_to_ned(lon, lat, z_up)

    # --- 交叉核对:CSV 的米制列(x=East, y=North)必须与自算值一致 ----
    # ⚠ 轴序陷阱就在这两行:CSV 的 x 对我们的 y,CSV 的 y 对我们的 x。
    csv_east = np.array([float(r["x_local_m"]) for r in rows], dtype=np.float64)
    csv_north = np.array([float(r["y_local_m"]) for r in rows], dtype=np.float64)
    d_north = float(np.abs(x_north - csv_north).max())
    d_east = float(np.abs(y_east - csv_east).max())
    if d_north > CROSS_CHECK_TOL_M or d_east > CROSS_CHECK_TOL_M:
        raise ValueError(
            "自算 NED 与 CSV 米制列不一致(疑似上游原点变更或列错位):"
            f"max|dNorth|={d_north:.6f} m, max|dEast|={d_east:.6f} m, "
            f"容差 {CROSS_CHECK_TOL_M} m")

    # UTM 列也顺手对一次,把上游三条口径(lon/lat、UTM、local)全钉在一起
    csv_e = np.array([float(r["utm9n_easting"]) for r in rows], dtype=np.float64)
    csv_n = np.array([float(r["utm9n_northing"]) for r in rows], dtype=np.float64)
    own_e, own_n, _ = ned_to_utm9n(x_north, y_east, z_down)
    d_utm = max(float(np.abs(own_e - csv_e).max()),
                float(np.abs(own_n - csv_n).max()))
    if d_utm > CROSS_CHECK_TOL_M:
        raise ValueError(f"自算 UTM 与 CSV utm9n_* 列不一致:max diff {d_utm:.6f} m")

    # --- 越界检查 -----------------------------------------------------
    # 这里比的是**原始栅格**外框,不是裁切后的分析域 —— 装载阶段读全部 28 个点,
    # 裁切(D7)是后面 `MothraDataset.select()` 里显式发生的一步,被丢的点可追溯。
    out = ((x_north < 0.0) | (x_north > RASTER_NORTH_EXTENT_M) |
           (y_east < 0.0) | (y_east > RASTER_EAST_EXTENT_M))
    if bool(out.any()):
        bad = np.where(out)[0]
        raise ValueError(
            f"{len(bad)} 个目标落在原始栅格外(索引 {bad.tolist()});"
            f"栅格 North [0,{RASTER_NORTH_EXTENT_M}] x East [0,{RASTER_EAST_EXTENT_M}]")

    type_ = np.array([r["type"].strip() for r in rows], dtype="<U7")
    unknown = sorted(set(type_.tolist()) - set(TARGET_TYPES))
    if unknown:
        raise ValueError(f"未知目标类型 {unknown},合法值 {TARGET_TYPES}")

    ds = MothraDataset(
        targets_ned=np.column_stack([x_north, y_east]).astype(np.float64),
        depth_D_m=z_down.astype(np.float64),
        height_m=np.array([float(r["height_m"]) for r in rows], dtype=np.float64),
        type_=type_,
        name=np.array([r["name"].strip() for r in rows], dtype="<U32"),
        waypoint_id=np.array([int(r["waypoint_id"]) for r in rows], dtype=np.int32),
        sequence_id=np.array([int(r["sequence_id"]) for r in rows], dtype=np.int32),
        lon=lon,
        lat=lat,
    )

    if strict:
        n_ch = int((ds.type_ == "chimney").sum())
        n_mo = int((ds.type_ == "mound").sum())
        if (ds.n, n_ch, n_mo) != (N_TARGETS_EXPECTED, N_CHIMNEY_EXPECTED, N_MOUND_EXPECTED):
            raise ValueError(
                f"数据集规模与冻结值不符:得到 n={ds.n}(chimney {n_ch} / mound {n_mo}),"
                f"期望 n={N_TARGETS_EXPECTED}({N_CHIMNEY_EXPECTED}/{N_MOUND_EXPECTED})。"
                "若确系换了输入数据,请以 strict=False 加载并更新 contracts/dataset.py。")
    return ds
