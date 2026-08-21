#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06 — Mothra 范围内热液口坐标提取 + 与水深数据打包
================================================================================
输入 : Bethy_data/TABLE_SI.xlsx   (Supplemental Table S1. Chimneys determined
                                   from AUV map; 单表 forPublication)
       outputs/mothra_wgs84.tif   (裁剪后的水深栅格, 用来取各热液口处海底水深)

输出 (outputs/):
       mothra_vents.csv       28 个热液口的完整属性表 (UTF-8-BOM, Excel 可直接开)
       mothra_vents.geojson    同上, EPSG:4326, 供 GIS
       mothra_bundle.npz      水深栅格 + 热液口一起打包的单文件

表结构说明:
  * 表里夹着 "Segment AV1/AV2a/AV2b" 这类分节行和一行脚注, 经纬度列非数值,
    按"经纬度必须同时是数值"来筛掉, 不按行号硬切。
  * `Height (m)` 是烟囱自身高出海底的高度, 与本脚本另采的 `seafloor_depth_m`
    (我们栅格在该点的海底水深) 是两回事, 分列存放, 不合并。

筛选口径:
  按几何落点筛 —— 经纬度落在我们 Mothra 裁剪 bbox 内。
  另用表中 `Vent Field Name == 'Mothra'` 做交叉核对; 两者若不一致会打印出来。

运行环境: 需要 rasterio + pyproj + openpyxl (本机为 conda base)
    python scripts/06_extract_vents.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import openpyxl
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
SRC_XLSX = ROOT / "Bethy_data" / "TABLE_SI.xlsx"
OUTDIR = ROOT / "outputs"

FIELDS = ["orig_order", "sequence_id", "lon", "lat", "height_m", "morphology",
          "vent_field", "chimney_name", "dist_from_axis_m",
          "utm9n_easting", "utm9n_northing", "seafloor_depth_m",
          "col_wgs84", "row_wgs84", "col_utm1m", "row_utm1m"]


def main() -> None:
    meta = json.loads((OUTDIR / "mothra_meta.json").read_text(encoding="utf-8"))
    L, B, R, T = meta["crop_wgs84"]["bounds_lon_lat"]
    print(f"[1/4] 读表  {SRC_XLSX.name}")
    ws = openpyxl.load_workbook(SRC_XLSX, data_only=True)["forPublication"]
    rows = list(ws.iter_rows(min_row=3, values_only=True))
    data = [r for r in rows
            if isinstance(r[2], (int, float)) and isinstance(r[3], (int, float))]
    skipped = [r for r in rows if r not in data and any(v not in (None, "") for v in r)]
    print(f"  数据行 {len(data)}, 跳过的分节/脚注行 {len(skipped)}: "
          + "; ".join(str([v for v in r if v not in (None, '')][0]) for r in skipped))

    print(f"[2/4] 按 bbox 筛选  lon {L:.6f}..{R:.6f}  lat {B:.6f}..{T:.6f}")
    sel = [r for r in data if L <= r[2] <= R and B <= r[3] <= T]
    labeled = {id(r) for r in data if (r[6] or "").strip().lower() == "mothra"}
    only_box = [r for r in sel if id(r) not in labeled]
    only_lab = [r for r in data if id(r) in labeled and r not in sel]
    print(f"  落在 bbox 内 {len(sel)} 个;  表中标 'Mothra' 的 {len(labeled)} 个")
    if only_box or only_lab:
        print(f"  ! 口径不一致: 仅 bbox 内 {len(only_box)} 个, 仅标签 {len(only_lab)} 个")
        for r in only_box + only_lab:
            print(f"    seq{r[1]} {r[2]:.6f} {r[3]:.6f} field={r[6]!r}")
    else:
        print("  两种口径完全一致 (bbox 内 = 标 'Mothra'), 无歧义")

    print("[3/4] 采样海底水深 + 换算 UTM / 像元下标")
    tr = Transformer.from_crs(4326, 32609, always_xy=True)
    with rasterio.open(OUTDIR / "mothra_wgs84.tif") as ds_g, \
         rasterio.open(OUTDIR / "mothra_utm9n_1m.tif") as ds_u:
        lonlat = [(r[2], r[3]) for r in sel]
        depths = [float(v[0]) for v in ds_g.sample(lonlat)]
        recs = []
        for r, dep in zip(sel, depths):
            x, y = tr.transform(r[2], r[3])
            rg, cg = ds_g.index(r[2], r[3])
            ru, cu = ds_u.index(x, y)
            recs.append({
                "orig_order": r[0], "sequence_id": r[1],
                "lon": round(float(r[2]), 9), "lat": round(float(r[3]), 9),
                "height_m": r[4], "morphology": r[5],
                "vent_field": r[6] or "", "chimney_name": (r[7] or "").rstrip(":"),
                "dist_from_axis_m": (round(float(r[8]), 4)
                                     if isinstance(r[8], (int, float)) else ""),
                "utm9n_easting": round(x, 3), "utm9n_northing": round(y, 3),
                "seafloor_depth_m": round(dep, 3),
                "col_wgs84": cg, "row_wgs84": rg,
                "col_utm1m": cu, "row_utm1m": ru,
            })
    recs.sort(key=lambda d: d["sequence_id"])
    bad = [d for d in recs if d["seafloor_depth_m"] < -99998]
    print(f"  采样 {len(recs)} 点, 落在 nodata 上的 {len(bad)} 个")
    dd = [d["seafloor_depth_m"] for d in recs]
    print(f"  热液口处海底水深 {-max(dd):.1f}..{-min(dd):.1f} m "
          f"(全区 {-meta['crop_wgs84']['z_max']:.1f}..{-meta['crop_wgs84']['z_min']:.1f} m)")

    print("[4/4] 写出")
    csv_p = OUTDIR / "mothra_vents.csv"
    with csv_p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(recs)

    gj = {
        "type": "FeatureCollection",
        "name": "mothra_vents",
        "crs": {"type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [d["lon"], d["lat"]]},
             "properties": {k: v for k, v in d.items() if k not in ("lon", "lat")}}
            for d in recs],
    }
    (OUTDIR / "mothra_vents.geojson").write_text(
        json.dumps(gj, indent=1, ensure_ascii=False), encoding="utf-8")

    # 水深 + 热液口 打包成单个自洽文件
    grid = np.load(OUTDIR / "mothra_utm9n_1m.npz")
    np.savez_compressed(
        OUTDIR / "mothra_bundle.npz",
        # --- 水深 ---
        z_utm1m=grid["z"], transform_utm1m=grid["transform"],
        bounds_utm=grid["bounds_utm"], res_utm=grid["res"], epsg_utm=np.int32(32609),
        z_wgs84=grid["z_wgs84"], transform_wgs84=grid["transform_wgs84"],
        bounds_wgs84=grid["bounds_wgs84"], epsg_wgs84=np.int32(4326),
        # --- 热液口 ---
        vent_lon=np.array([d["lon"] for d in recs]),
        vent_lat=np.array([d["lat"] for d in recs]),
        vent_easting=np.array([d["utm9n_easting"] for d in recs]),
        vent_northing=np.array([d["utm9n_northing"] for d in recs]),
        vent_height_m=np.array([d["height_m"] for d in recs], dtype="float32"),
        vent_seafloor_depth_m=np.array([d["seafloor_depth_m"] for d in recs],
                                       dtype="float32"),
        vent_morphology=np.array([d["morphology"] for d in recs]),
        vent_name=np.array([d["chimney_name"] for d in recs]),
        vent_sequence_id=np.array([d["sequence_id"] for d in recs], dtype="int32"),
        vent_col_utm1m=np.array([d["col_utm1m"] for d in recs], dtype="int32"),
        vent_row_utm1m=np.array([d["row_utm1m"] for d in recs], dtype="int32"),
    )

    meta["vents"] = {
        "source": "Bethy_data/TABLE_SI.xlsx (Supplemental Table S1)",
        "selection": "经纬度落在 crop_wgs84.bounds_lon_lat 内",
        "cross_check": "与表中 Vent Field Name == 'Mothra' 完全一致"
        if not (only_box or only_lab) else "与 Vent Field Name 标签不一致, 见日志",
        "n_total_in_table": len(data),
        "n_selected": len(recs),
        "by_morphology": {m: sum(1 for d in recs if d["morphology"] == m)
                          for m in sorted({d["morphology"] for d in recs})},
        "height_m_range": [min(d["height_m"] for d in recs),
                           max(d["height_m"] for d in recs)],
        "named": sorted({d["chimney_name"] for d in recs if d["chimney_name"]}),
        "note": "height_m = 烟囱高出海底的高度(表中给定); "
                "seafloor_depth_m = 本项目栅格在该点的海底水深(负向下), 两者含义不同",
    }
    (OUTDIR / "mothra_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for n in ("mothra_vents.csv", "mothra_vents.geojson", "mothra_bundle.npz"):
        p = OUTDIR / n
        print(f"  {p.name:26s} {p.stat().st_size / 1024:9.1f} KB")


if __name__ == "__main__":
    main()
