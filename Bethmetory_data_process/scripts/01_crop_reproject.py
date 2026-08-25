#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01 — 热液场子区裁剪 + 重投影为各向同性 1 m 栅格(registry 驱动, 多场景)
================================================================================
输入 : scenarios.json 登记的场景 bbox + 源瓦片(.asc 或 .grd, 单波段 float32,
       无内嵌 CRS, cellsize ~1.338e-05 deg)。
       --scenario mothra(默认)读 Bethy_data/EndeavourAUVSouthCentral1.asc,
       nodata=-99999;其余场景读 ../Bethmetory_data/MGDS_Download/.../*.grd,
       nodata=NaN —— 两种源文件在读入后统一转成 NaN 掩膜, 下游处理逻辑不分叉。

输出 (统一落在 outputs/scenarios/<id>/, <id> 含 mothra 本身, 不再有例外路径):
           <id>_wgs84.tif/.asc/.prj, <id>_utm9n_1m.tif/.asc/.prj/.npz, <id>_meta.json
       Mothra 额外产出 endeavour_context.npz(全测区降采样底图, 只在 Mothra 场景下
       生成一次, 其余场景不重复生成 ——"全段在哪" 已由 00b_plot_scenario_overview.py
       的总览图覆盖)。

处理要点:
  1. nodata 先掩膜为 NaN, 再做任何插值 —— 否则源哨兵值(-99999)会被重采样核
     拖进邻域, 污染后续滤波与形态学。
  2. 源无 CRS, 按数据来源手动指定 EPSG:4326。
  3. 裁剪时向外多取 BUFFER_CELLS 圈像元, 重投影后再裁回精确 bbox, 使内部区域
     不会因重采样核触边而产生 nodata 缺口。
  4. 重采样默认 bilinear: 不产生新的极值。cubic 会在陡坎处 overshoot, 制造出
     虚假局部极大, 而形态学 top-hat / opening 恰恰会把它误判成热液丘状体。

运行环境: 需要 rasterio + pyproj (本机为 conda base)
    python scripts/01_crop_reproject.py                      # 默认 --scenario mothra
    python scripts/01_crop_reproject.py --scenario mef
    python scripts/01_crop_reproject.py --all                # registry 里全部场景
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
REPO_ROOT = ROOT.parent
OUTDIR = ROOT / "outputs"
REGISTRY = ROOT / "scenarios.json"

SRC_CRS = CRS.from_epsg(4326)      # 源文件无内嵌 CRS, 手动指定
DST_CRS = CRS.from_epsg(32609)     # UTM zone 9N, 中央经线 -129deg
DST_RES = 1.0                      # 目标像元 1 m x 1 m

BUFFER_CELLS = 16                  # 裁剪缓冲圈数, 仅用于喂重采样核, 最终会裁掉
CONTEXT_DECIMATE = 3               # 全测区上下文底图的降采样倍数(仅 Mothra 生成)

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
}


def load_registry() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def find_scenario(scenario_id: str) -> dict:
    for s in load_registry():
        if s["id"] == scenario_id:
            return s
    raise KeyError(f"scenarios.json 里没有 id={scenario_id!r};"
                   f"先在 registry 里冻结, 不要在脚本里临时加 bbox")


def to_nan(arr: np.ndarray, nodata) -> np.ndarray:
    """把源 nodata 统一转成 NaN。两种源约定都处理:
    .asc 是哨兵值(-99999, 需要显式比较替换); .grd(netCDF) 本身就在无效处存 NaN,
    此时 nodata==NaN, 比较 `arr == nodata` 恒假(NaN 不等于自身), 直接跳过即可,
    不能对它做等值替换(那是 no-op 但容易误判成"没生效")。"""
    if nodata is None:
        return arr
    if isinstance(nodata, float) and np.isnan(nodata):
        return arr
    out = arr.copy()
    out[out == nodata] = np.nan
    return out


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
def process_scenario(scenario_id: str, args) -> None:
    sc = find_scenario(scenario_id)
    src = REPO_ROOT / sc["source_tile"]
    lon_min, lon_max = sc["bbox_wgs84"]["lon_min"], sc["bbox_wgs84"]["lon_max"]
    lat_min, lat_max = sc["bbox_wgs84"]["lat_min"], sc["bbox_wgs84"]["lat_max"]
    is_mothra = scenario_id == "mothra"
    outdir = OUTDIR / "scenarios" / scenario_id
    prefix = scenario_id
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== 场景 {scenario_id} ({sc['label']}) =====")
    if not src.is_file():
        raise FileNotFoundError(
            f"源瓦片不存在: {src}\n"
            f"见 Bethmetory_data/README.md 的获取说明(MGDS 数据集 21403, "
            f"CC BY-NC-SA 3.0), 下载后放到该路径, 若是 .grd.gz 需先解压。")

    meta: dict = {"scenario": {"id": scenario_id, "label": sc["label"],
                               "kind": sc["kind"], "provenance": sc["provenance"]}}

    # ---------------------------------------------------------- 1. 读+裁剪 --
    print(f"[1/4] 读取并裁剪  {src.name}")
    with rasterio.open(src) as ds:
        assert ds.count == 1 and ds.dtypes[0] == "float32"
        src_nodata = ds.nodata
        meta["source"] = {
            "path": str(src.relative_to(REPO_ROOT)).replace("\\", "/"),
            "driver": ds.driver, "dtype": ds.dtypes[0],
            "shape": [ds.height, ds.width],
            "crs_embedded": None, "crs_assigned": "EPSG:4326",
            "nodata": None if src_nodata is None else float(src_nodata),
            "cellsize_deg": float(ds.transform.a),
            "bounds": [float(v) for v in ds.bounds],
        }
        print(f"  源: {ds.height}x{ds.width}, CRS={ds.crs} -> 手动指定 EPSG:4326, "
              f"nodata={src_nodata}")

        # 精确 bbox 窗口: offset 向下取整 / 右下边界向上取整, 保证完整覆盖请求范围。
        # 注意: rasterio >= 1.4 把 Window.round_offsets/round_lengths 的签名改成了
        # **kwds, op= 参数被静默忽略并一律按四舍五入处理 —— 写 op="ceil" 也拿不到
        # ceil (292.20 会变成 292, 东边少半个像元)。这里显式算, 不依赖那两个方法。
        raw = from_bounds(lon_min, lat_min, lon_max, lat_max, ds.transform)
        c0, r0 = int(np.floor(raw.col_off)), int(np.floor(raw.row_off))
        win = Window(c0, r0,
                     int(np.ceil(raw.col_off + raw.width - c0)),
                     int(np.ceil(raw.row_off + raw.height - r0)))
        # 带缓冲窗口, 且裁到栅格边界内
        buf = Window(win.col_off - BUFFER_CELLS, win.row_off - BUFFER_CELLS,
                     win.width + 2 * BUFFER_CELLS, win.height + 2 * BUFFER_CELLS)
        buf = buf.intersection(Window(0, 0, ds.width, ds.height))
        if win.col_off < 0 or win.row_off < 0 or \
           win.col_off + win.width > ds.width or win.row_off + win.height > ds.height:
            raise ValueError(f"场景 {scenario_id} 的 bbox 超出源瓦片 {src.name} 范围, "
                             f"registry 里的 source_tile 配错了瓦片")

        z_exact = ds.read(1, window=win).astype("float64")
        z_buf = ds.read(1, window=buf).astype("float64")
        tf_exact = ds.window_transform(win)
        tf_buf = ds.window_transform(buf)
        full_bounds = [float(v) for v in ds.bounds]

        z_full = None
        z_ctx = None
        if is_mothra:
            # 全测区降采样底图: 只用于上下文示意, 不参与任何定量分析(仅 Mothra 生成,
            # 保持与既有产物逐字节一致; 全段总览已由 00b_plot_scenario_overview.py 覆盖)。
            z_full = to_nan(ds.read(1).astype("float64"), src_nodata)

    if z_full is not None:
        k = CONTEXT_DECIMATE
        hh, ww = (z_full.shape[0] // k) * k, (z_full.shape[1] // k) * k
        with warnings.catch_warnings():          # 全 NaN 块 -> NaN, 是预期行为
            warnings.simplefilter("ignore", RuntimeWarning)
            z_ctx = np.nanmean(
                z_full[:hh, :ww].reshape(hh // k, k, ww // k, k).transpose(0, 2, 1, 3)
                .reshape(hh // k, ww // k, k * k), axis=2)
        del z_full

    # 关键: 先掩膜再做任何插值
    z_exact = to_nan(z_exact, src_nodata)
    z_buf = to_nan(z_buf, src_nodata)

    print(f"  精确窗口 {win}")
    meta["crop_wgs84"] = describe("WGS84 crop", z_exact)
    n_buf_nan = int((~np.isfinite(z_buf)).sum())
    print(f"  缓冲窗口 {z_buf.shape} (+{BUFFER_CELLS} 圈), nodata={n_buf_nan}")

    b = rasterio.transform.array_bounds(z_exact.shape[0], z_exact.shape[1], tf_exact)
    meta["crop_wgs84"].update({
        "crs": "EPSG:4326",
        "bounds_lon_lat": [float(v) for v in b],   # left, bottom, right, top
        "cellsize_deg": float(tf_exact.a),
        "requested_bbox": {"lon_min": lon_min, "lon_max": lon_max,
                           "lat_min": lat_min, "lat_max": lat_max},
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
        if not args.allow_nodata:
            raise ValueError(
                f"场景 {scenario_id} 重投影后仍有 nodata, 拒绝写出(加 --allow-nodata "
                f"可强制放行, 但要先确认不是 bbox/缓冲配置的问题)")
    else:
        print("  重投影后 nodata = 0, 内部无缺口")

    # ---------------------------------------------------------- 4. 写出 ----
    print("[4/4] 写出文件")
    write_gtiff(outdir / f"{prefix}_wgs84.tif", z_exact, tf_exact, SRC_CRS)
    write_aaigrid(outdir / f"{prefix}_wgs84.asc", z_exact, tf_exact, SRC_CRS, decimals=8)
    write_gtiff(outdir / f"{prefix}_utm9n_1m.tif", dst, dst_tf, DST_CRS)
    write_aaigrid(outdir / f"{prefix}_utm9n_1m.asc", dst, dst_tf, DST_CRS, decimals=3)

    # numpy 打包: 供没有 rasterio 的环境 (如 auv_py310) 直接出图
    np.savez_compressed(
        outdir / f"{prefix}_utm9n_1m.npz",
        z=dst.astype("float32"),
        transform=np.array(dst_tf.to_gdal(), dtype="float64"),
        bounds_utm=np.array([left, bottom, right, top], dtype="float64"),
        bounds_wgs84=np.array(b, dtype="float64"),
        res=np.array([r, r], dtype="float64"),
        epsg=np.int32(32609),
        z_wgs84=z_exact.astype("float32"),
        transform_wgs84=np.array(tf_exact.to_gdal(), dtype="float64"),
    )

    if is_mothra:
        # 全测区上下文底图 (仅示意用, 仅 Mothra 生成一次, 与既有产物逐字节一致)
        np.savez_compressed(
            outdir / "endeavour_context.npz",
            z=z_ctx.astype("float32"),
            bounds_wgs84=np.array(full_bounds, dtype="float64"),
            decimate=np.int32(CONTEXT_DECIMATE),
            mothra_bbox=np.array([lon_min, lat_min, lon_max, lat_max], dtype="float64"),
        )
        meta["context_grid"] = {
            "shape": [int(z_ctx.shape[0]), int(z_ctx.shape[1])],
            "decimate": CONTEXT_DECIMATE,
            "bounds_lon_lat": full_bounds,
            "note": "全测区降采样底图, 仅供上下文示意, 不用于定量分析",
        }
        print(f"  [context] {z_ctx.shape[0]}x{z_ctx.shape[1]} (1/{CONTEXT_DECIMATE} 降采样)")

    meta["outputs"] = sorted(p.name for p in outdir.iterdir() if p.is_file())
    (outdir / f"{prefix}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for p in sorted(outdir.iterdir()):
        if p.is_file():
            print(f"  {p.name:28s} {p.stat().st_size / 1024:9.1f} KB")
    print(f"完成。分析用主产品: {(outdir / f'{prefix}_utm9n_1m.tif').relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mothra",
                    help="scenarios.json 里的场景 id (默认 mothra)")
    ap.add_argument("--all", action="store_true", help="跑 registry 里的全部场景")
    ap.add_argument("--resampling", default="bilinear", choices=sorted(RESAMPLING),
                    help="重投影重采样核 (默认 bilinear, 不引入虚假极值)")
    ap.add_argument("--res", type=float, default=DST_RES, help="目标像元边长 (m)")
    ap.add_argument("--allow-nodata", action="store_true",
                    help="允许重投影后仍有 nodata 残留(默认拒绝写出)")
    args = ap.parse_args()

    ids = [s["id"] for s in load_registry()] if args.all else [args.scenario]
    for sid in ids:
        process_scenario(sid, args)


if __name__ == "__main__":
    main()
