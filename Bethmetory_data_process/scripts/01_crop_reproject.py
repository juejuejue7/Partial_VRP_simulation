#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01 — Mothra 热液场子区裁剪 + 重投影为各向同性 1 m 栅格
================================================================================
输入 : Bethy_data/EndeavourAUVSouthCentral1.asc
       AAIGrid / 单波段 float32 / nodata = -99999 / 无内嵌 CRS
       cellsize = 1.338111292e-05 deg  ->  @lat 47.923: E-W 1.000 m, N-S 1.488 m

输出 (outputs/):
       mothra_wgs84.tif        裁剪结果, EPSG:4326, 原生各向异性像元
       mothra_wgs84.asc/.prj   同上, AAIGrid 格式
       mothra_utm9n_1m.tif     重投影 EPSG:32609, 正方形 1 m 像元  <- 分析用主产品
       mothra_utm9n_1m.asc/.prj 同上, AAIGrid 格式
       mothra_utm9n_1m.npz     纯 numpy 打包, 供无 rasterio 的环境出图
       endeavour_context.npz   全测区降采样版, 供出"Mothra 在哪"的上下文底图
       mothra_meta.json        元数据侧车文件

处理要点:
  1. nodata(-99999) 先掩膜为 NaN, 再做任何插值 —— 否则 -99999 会被重采样核
     拖进邻域, 污染后续滤波与形态学。
  2. 源无 CRS, 按数据来源手动指定 EPSG:4326。
  3. 裁剪时向外多取 BUFFER_CELLS 圈像元, 重投影后再裁回精确 bbox, 使内部区域
     不会因重采样核触边而产生 nodata 缺口。
  4. 重采样默认 bilinear: 不产生新的极值。cubic 会在陡坎处 overshoot, 制造出
     虚假局部极大, 而形态学 top-hat / opening 恰恰会把它误判成热液丘状体。

运行环境: 需要 rasterio + pyproj (本机为 conda base)
    python scripts/01_crop_reproject.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

# ------------------------------------------------------------------ 配置 ----
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Bethy_data" / "EndeavourAUVSouthCentral1.asc"
OUTDIR = ROOT / "outputs"

SRC_CRS = CRS.from_epsg(4326)      # 源文件无内嵌 CRS, 手动指定
DST_CRS = CRS.from_epsg(32609)     # UTM zone 9N, 中央经线 -129deg
DST_RES = 1.0                      # 目标像元 1 m x 1 m

# Mothra 热液场裁剪范围 (WGS84 十进制度)
LON_MIN, LON_MAX = -129.10974, -129.10583
LAT_MIN, LAT_MAX = 47.92026, 47.92612

BUFFER_CELLS = 16                  # 裁剪缓冲圈数, 仅用于喂重采样核, 最终会裁掉
CONTEXT_DECIMATE = 3               # 全测区上下文底图的降采样倍数

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
}


# ------------------------------------------------------------------ 工具 ----
def write_aaigrid(path: Path, arr: np.ndarray, transform, crs: CRS,
                  nodata: float = -99999.0, decimals: int = 3) -> None:
    """写 AAIGrid。GDAL 的 AAIGrid 驱动只支持 CreateCopy, 故先建内存 GeoTIFF。"""
    out = np.where(np.isnan(arr), nodata, arr).astype("float32")
    profile = dict(driver="GTiff", height=out.shape[0], width=out.shape[1],
                   count=1, dtype="float32", crs=crs, transform=transform,
                   nodata=nodata)
    with MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            tmp.write(out, 1)
        with mem.open() as tmp:
            rasterio.shutil.copy(
                tmp, str(path), driver="AAIGrid",
                DECIMAL_PRECISION=str(decimals), FORCE_CELLSIZE="YES",
            )


def write_gtiff(path: Path, arr: np.ndarray, transform, crs: CRS,
                nodata: float = -99999.0) -> None:
    out = np.where(np.isnan(arr), nodata, arr).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=out.shape[0], width=out.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=nodata,
        compress="deflate", predictor=3, tiled=True,
    ) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "seafloor depth (m, negative down)")


def describe(tag: str, a: np.ndarray) -> dict:
    finite = np.isfinite(a)
    n_nan = int((~finite).sum())
    d = {
        "shape": [int(a.shape[0]), int(a.shape[1])],
        "n_cells": int(a.size),
        "n_nodata": n_nan,
        "nodata_pct": round(100.0 * n_nan / a.size, 6),
        "z_min": float(np.nanmin(a)), "z_max": float(np.nanmax(a)),
        "z_mean": float(np.nanmean(a)), "z_std": float(np.nanstd(a)),
        "relief_m": float(np.nanmax(a) - np.nanmin(a)),
    }
    print(f"  [{tag}] {d['shape'][0]}x{d['shape'][1]}  "
          f"nodata={d['n_nodata']} ({d['nodata_pct']:.4f}%)  "
          f"z=[{d['z_min']:.2f}, {d['z_max']:.2f}] m  relief={d['relief_m']:.2f} m")
    return d


# ------------------------------------------------------------------ 主流程 --
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resampling", default="bilinear", choices=sorted(RESAMPLING),
                    help="重投影重采样核 (默认 bilinear, 不引入虚假极值)")
    ap.add_argument("--res", type=float, default=DST_RES, help="目标像元边长 (m)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta: dict = {}

    # ---------------------------------------------------------- 1. 读+裁剪 --
    print(f"[1/4] 读取并裁剪  {SRC.name}")
    with rasterio.open(SRC) as ds:
        assert ds.count == 1 and ds.dtypes[0] == "float32"
        src_nodata = ds.nodata if ds.nodata is not None else -99999.0
        meta["source"] = {
            "path": str(SRC.relative_to(ROOT)).replace("\\", "/"),
            "driver": ds.driver, "dtype": ds.dtypes[0],
            "shape": [ds.height, ds.width],
            "crs_embedded": None, "crs_assigned": "EPSG:4326",
            "nodata": float(src_nodata),
            "cellsize_deg": float(ds.transform.a),
            "bounds": [float(v) for v in ds.bounds],
        }
        print(f"  源: {ds.height}x{ds.width}, CRS={ds.crs} -> 手动指定 EPSG:4326, "
              f"nodata={src_nodata}")

        # 精确 bbox 窗口: offset 向下取整 / 右下边界向上取整, 保证完整覆盖请求范围。
        # 注意: rasterio >= 1.4 把 Window.round_offsets/round_lengths 的签名改成了
        # **kwds, op= 参数被静默忽略并一律按四舍五入处理 —— 写 op="ceil" 也拿不到
        # ceil (292.20 会变成 292, 东边少半个像元)。这里显式算, 不依赖那两个方法。
        raw = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, ds.transform)
        c0, r0 = int(np.floor(raw.col_off)), int(np.floor(raw.row_off))
        win = Window(c0, r0,
                     int(np.ceil(raw.col_off + raw.width - c0)),
                     int(np.ceil(raw.row_off + raw.height - r0)))
        # 带缓冲窗口, 且裁到栅格边界内
        buf = Window(win.col_off - BUFFER_CELLS, win.row_off - BUFFER_CELLS,
                     win.width + 2 * BUFFER_CELLS, win.height + 2 * BUFFER_CELLS)
        buf = buf.intersection(Window(0, 0, ds.width, ds.height))

        z_exact = ds.read(1, window=win).astype("float64")
        z_buf = ds.read(1, window=buf).astype("float64")
        tf_exact = ds.window_transform(win)
        tf_buf = ds.window_transform(buf)

        # 全测区降采样底图: 只用于上下文示意, 不参与任何定量分析。
        # 先掩膜再块平均, 避免 -99999 被平均进来。
        z_full = ds.read(1).astype("float64")
        z_full[z_full == src_nodata] = np.nan
        full_bounds = [float(v) for v in ds.bounds]

    k = CONTEXT_DECIMATE
    hh, ww = (z_full.shape[0] // k) * k, (z_full.shape[1] // k) * k
    with warnings.catch_warnings():          # 全 NaN 块 -> NaN, 是预期行为
        warnings.simplefilter("ignore", RuntimeWarning)
        z_ctx = np.nanmean(
            z_full[:hh, :ww].reshape(hh // k, k, ww // k, k).transpose(0, 2, 1, 3)
            .reshape(hh // k, ww // k, k * k), axis=2)
    del z_full

    # 关键: 先掩膜再做任何插值
    z_exact[z_exact == src_nodata] = np.nan
    z_buf[z_buf == src_nodata] = np.nan

    print(f"  精确窗口 {win}")
    meta["crop_wgs84"] = describe("WGS84 crop", z_exact)
    n_buf_nan = int((~np.isfinite(z_buf)).sum())
    print(f"  缓冲窗口 {z_buf.shape} (+{BUFFER_CELLS} 圈), nodata={n_buf_nan}")

    b = rasterio.transform.array_bounds(z_exact.shape[0], z_exact.shape[1], tf_exact)
    meta["crop_wgs84"].update({
        "crs": "EPSG:4326",
        "bounds_lon_lat": [float(v) for v in b],   # left, bottom, right, top
        "cellsize_deg": float(tf_exact.a),
        "requested_bbox": {"lon_min": LON_MIN, "lon_max": LON_MAX,
                           "lat_min": LAT_MIN, "lat_max": LAT_MAX},
    })

    # ------------------------------------------------------- 2. 目标 UTM 格网 --
    print(f"[2/4] 构建 UTM 9N 目标格网 ({args.res} m 正方形像元)")
    ux0, uy0, ux1, uy1 = transform_bounds(SRC_CRS, DST_CRS, b[0], b[1], b[2], b[3],
                                          densify_pts=64)
    r = args.res
    left = np.floor(ux0 / r) * r
    bottom = np.floor(uy0 / r) * r
    right = np.ceil(ux1 / r) * r
    top = np.ceil(uy1 / r) * r
    width = int(round((right - left) / r))
    height = int(round((top - bottom) / r))
    dst_tf = from_origin(left, top, r, r)
    print(f"  UTM bbox: E {left:.1f}..{right:.1f}  N {bottom:.1f}..{top:.1f}")
    print(f"  目标栅格: {height} x {width}")

    # --------------------------------------------------------- 3. 重投影 ----
    print(f"[3/4] 重投影 EPSG:4326 -> EPSG:32609  (resampling={args.resampling})")
    dst = np.full((height, width), np.nan, dtype="float64")
    reproject(
        source=z_buf, destination=dst,
        src_transform=tf_buf, src_crs=SRC_CRS, src_nodata=np.nan,
        dst_transform=dst_tf, dst_crs=DST_CRS, dst_nodata=np.nan,
        resampling=RESAMPLING[args.resampling],
    )
    meta["utm9n_1m"] = describe("UTM9N 1m", dst)
    meta["utm9n_1m"].update({
        "crs": "EPSG:32609",
        "resolution_m": [r, r],
        "bounds_utm": [float(left), float(bottom), float(right), float(top)],
        "resampling": args.resampling,
        "transform": [float(v) for v in dst_tf.to_gdal()],
    })

    nod = meta["utm9n_1m"]["n_nodata"]
    if nod:
        print(f"  ! 重投影后仍有 {nod} 个 nodata 像元 "
              f"({meta['utm9n_1m']['nodata_pct']:.4f}%) —— 检查缓冲是否够大")
    else:
        print("  重投影后 nodata = 0, 内部无缺口")

    # ---------------------------------------------------------- 4. 写出 ----
    print("[4/4] 写出文件")
    write_gtiff(OUTDIR / "mothra_wgs84.tif", z_exact, tf_exact, SRC_CRS)
    write_aaigrid(OUTDIR / "mothra_wgs84.asc", z_exact, tf_exact, SRC_CRS, decimals=8)
    write_gtiff(OUTDIR / "mothra_utm9n_1m.tif", dst, dst_tf, DST_CRS)
    write_aaigrid(OUTDIR / "mothra_utm9n_1m.asc", dst, dst_tf, DST_CRS, decimals=3)

    # numpy 打包: 供没有 rasterio 的环境 (如 auv_py310) 直接出图
    np.savez_compressed(
        OUTDIR / "mothra_utm9n_1m.npz",
        z=dst.astype("float32"),
        transform=np.array(dst_tf.to_gdal(), dtype="float64"),
        bounds_utm=np.array([left, bottom, right, top], dtype="float64"),
        bounds_wgs84=np.array(b, dtype="float64"),
        res=np.array([r, r], dtype="float64"),
        epsg=np.int32(32609),
        z_wgs84=z_exact.astype("float32"),
        transform_wgs84=np.array(tf_exact.to_gdal(), dtype="float64"),
    )

    # 全测区上下文底图 (仅示意用)
    np.savez_compressed(
        OUTDIR / "endeavour_context.npz",
        z=z_ctx.astype("float32"),
        bounds_wgs84=np.array(full_bounds, dtype="float64"),
        decimate=np.int32(CONTEXT_DECIMATE),
        mothra_bbox=np.array([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX], dtype="float64"),
    )
    meta["context_grid"] = {
        "shape": [int(z_ctx.shape[0]), int(z_ctx.shape[1])],
        "decimate": CONTEXT_DECIMATE,
        "bounds_lon_lat": full_bounds,
        "note": "全测区降采样底图, 仅供上下文示意, 不用于定量分析",
    }
    print(f"  [context] {z_ctx.shape[0]}x{z_ctx.shape[1]} (1/{CONTEXT_DECIMATE} 降采样)")

    meta["outputs"] = sorted(p.name for p in OUTDIR.iterdir() if p.is_file())
    (OUTDIR / "mothra_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for p in sorted(OUTDIR.iterdir()):
        if p.is_file():
            print(f"  {p.name:28s} {p.stat().st_size / 1024:9.1f} KB")
    print("\n完成。分析用主产品: outputs/mothra_utm9n_1m.tif")


if __name__ == "__main__":
    main()
