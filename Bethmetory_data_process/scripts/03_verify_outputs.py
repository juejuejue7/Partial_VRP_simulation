#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03 — 产物校验
================================================================================
不信自报, 回读每个输出文件实际比对:
  * .asc / .tif / .npz 三种载体的像元值是否一致
  * CRS / transform / 分辨率是否为声明的值
  * nodata 是否已正确掩膜, 是否残留 -99999
  * UTM 产物的像元是否真的是正方形 1 m
  * 用独立路径 (pyproj 正算) 复核 UTM 角点坐标, 而不是复用写文件时那次变换

运行环境: 需要 rasterio + pyproj (本机为 conda base)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

fails: list[str] = []
checks = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(label)


print("[1] WGS84 裁剪产物")
with rasterio.open(OUT / "mothra_wgs84.tif") as a, \
     rasterio.open(OUT / "mothra_wgs84.asc") as b:
    za, zb = a.read(1, masked=True), b.read(1, masked=True)
    check(a.crs.to_epsg() == 4326, "GeoTIFF CRS = EPSG:4326", str(a.crs))
    check((a.height, a.width) == (438, 293), "尺寸 438x293", f"{a.height}x{a.width}")
    check(za.count() == za.size, "无 nodata", f"{za.size - za.count()} 个缺失")
    check(float(za.min()) > -99998, "无 -99999 残留", f"min={float(za.min()):.3f}")
    check(np.allclose(za.filled(0), zb.filled(0), atol=1e-3),
          ".asc 与 .tif 像元值一致",
          f"最大差 {float(np.abs(za - zb).max()):.2e} m")
    check(abs(a.transform.a - 1.338111292e-05) < 1e-12, "cellsize 保持源分辨率",
          f"{a.transform.a:.12g} deg")
    prj = (OUT / "mothra_wgs84.prj").read_text().strip()
    check("GCS" in prj or "GEOGCS" in prj, ".prj 为地理坐标系", prj[:46] + "...")

print("\n[2] UTM 9N 1 m 产物")
with rasterio.open(OUT / "mothra_utm9n_1m.tif") as a, \
     rasterio.open(OUT / "mothra_utm9n_1m.asc") as b:
    za, zb = a.read(1, masked=True), b.read(1, masked=True)
    check(a.crs.to_epsg() == 32609, "CRS = EPSG:32609", str(a.crs))
    check((a.height, a.width) == (653, 294), "尺寸 653x294", f"{a.height}x{a.width}")
    check(abs(a.transform.a - 1.0) < 1e-9 and abs(a.transform.e + 1.0) < 1e-9,
          "像元为正方形 1 m x 1 m",
          f"dx={a.transform.a:.6f}, dy={a.transform.e:.6f}")
    check(abs(a.transform.b) < 1e-12 and abs(a.transform.d) < 1e-12,
          "无旋转项 (轴对齐)")
    check(za.count() == za.size, "无 nodata", f"{za.size - za.count()} 个缺失")
    check(float(za.min()) > -99998, "无 -99999 残留", f"min={float(za.min()):.3f}")
    check(np.allclose(za.filled(0), zb.filled(0), atol=5e-3),
          ".asc 与 .tif 像元值一致 (asc 精度 3 位小数)",
          f"最大差 {float(np.abs(za - zb).max()):.2e} m")
    utm_bounds = a.bounds
    z_utm = za.filled(np.nan)

print("\n[3] 独立复核 UTM 角点 (pyproj 正算, 不复用写文件那次变换)")
LON_MIN, LON_MAX = -129.10974, -129.10583
LAT_MIN, LAT_MAX = 47.92026, 47.92612
tr = Transformer.from_crs(4326, 32609, always_xy=True)
xs, ys = [], []
for lo in (LON_MIN, LON_MAX):
    for la in (LAT_MIN, LAT_MAX):
        x, y = tr.transform(lo, la)
        xs.append(x)
        ys.append(y)
# 栅格是请求 bbox 向外对齐到整米后的外接矩形, 故应完整包住它, 且余量 < 1 个像元
dl, db = min(xs) - utm_bounds.left, min(ys) - utm_bounds.bottom
dr, dt = utm_bounds.right - max(xs), utm_bounds.top - max(ys)
check(all(v >= 0 for v in (dl, db, dr, dt)), "栅格完整包住请求 bbox",
      f"余量 L{dl:.2f} B{db:.2f} R{dr:.2f} T{dt:.2f} m")
check(all(v < 1.0 for v in (dl, db, dr, dt)), "对齐余量 < 1 像元 (无多余外扩)")

print("\n[4] npz 与栅格一致")
d = np.load(OUT / "mothra_utm9n_1m.npz")
check(np.allclose(d["z"], z_utm, atol=1e-3, equal_nan=True),
      "npz z 与 GeoTIFF 一致", f"最大差 {float(np.nanmax(np.abs(d['z'] - z_utm))):.2e} m")
check(int(d["epsg"]) == 32609, "npz 记录 EPSG:32609")
check(np.allclose(d["bounds_utm"], [utm_bounds.left, utm_bounds.bottom,
                                    utm_bounds.right, utm_bounds.top]),
      "npz bounds 与 GeoTIFF 一致")

print("\n[5] 与源文件逐像元比对 (WGS84 裁剪 = 源窗口的精确子集)")
with rasterio.open(ROOT / "Bethy_data" / "EndeavourAUVSouthCentral1.asc") as src:
    ref = src.read(1, window=rasterio.windows.Window(2336, 664, 293, 438))
with rasterio.open(OUT / "mothra_wgs84.tif") as a:
    got = a.read(1)
check(np.array_equal(ref, got), "裁剪未改动任何像元值 (逐位相等)")

print("\n[6] 热液口提取与打包")
import csv as _csv
import openpyxl

rows = list(_csv.DictReader(
    (OUT / "mothra_vents.csv").read_text(encoding="utf-8-sig").splitlines()))
gj = json.loads((OUT / "mothra_vents.geojson").read_text(encoding="utf-8"))
bd = np.load(OUT / "mothra_bundle.npz", allow_pickle=False)

check(len(rows) == 28, "CSV 28 条记录", f"{len(rows)} 条")
check(len(gj["features"]) == len(rows), "GeoJSON 条数与 CSV 一致")
check(len(bd["vent_lon"]) == len(rows), "bundle 热液口数与 CSV 一致")

vlon = np.array([float(r["lon"]) for r in rows])
vlat = np.array([float(r["lat"]) for r in rows])
check(np.allclose(np.sort(bd["vent_lon"]), np.sort(vlon)) and
      np.allclose(np.sort(bd["vent_lat"]), np.sort(vlat)),
      "bundle 坐标与 CSV 一致")

with rasterio.open(OUT / "mothra_wgs84.tif") as a:
    L, Bo, R, Tp = a.bounds
check(bool(np.all((vlon >= L) & (vlon <= R) & (vlat >= Bo) & (vlat <= Tp))),
      "全部 28 点落在裁剪 bbox 内")

# 与源表独立重算一次, 不复用 06 的筛选结果
_ws = openpyxl.load_workbook(ROOT / "Bethy_data" / "TABLE_SI.xlsx",
                             data_only=True)["forPublication"]
_all = [r for r in _ws.iter_rows(min_row=3, values_only=True)
        if isinstance(r[2], (int, float)) and isinstance(r[3], (int, float))]
_box = [r for r in _all if L <= r[2] <= R and Bo <= r[3] <= Tp]
_lab = [r for r in _all if (r[6] or "").strip().lower() == "mothra"]
check(len(_box) == 28, "独立重算 bbox 内点数 = 28", f"{len(_box)}")
check(len(_lab) == 28 and {(r[2], r[3]) for r in _box} == {(r[2], r[3]) for r in _lab},
      "bbox 口径与表中 'Mothra' 标签完全一致")
check(np.allclose(np.sort(np.array([r[2] for r in _box])), np.sort(vlon)),
      "CSV 经度与源表逐点相等")

# 采样水深应与栅格该像元逐位相等
with rasterio.open(OUT / "mothra_wgs84.tif") as a:
    grid = a.read(1)
    ok_d = all(
        abs(float(r["seafloor_depth_m"]) - float(grid[int(r["row_wgs84"]),
                                                      int(r["col_wgs84"])])) < 1e-3
        for r in rows)
check(ok_d, "seafloor_depth_m 与栅格对应像元一致")
check(all(float(r["seafloor_depth_m"]) > -99998 for r in rows), "无热液口落在 nodata 上")
check(all(r["morphology"] in ("chimney", "mound") for r in rows),
      "morphology 取值合法",
      f"chimney {sum(r['morphology'] == 'chimney' for r in rows)}, "
      f"mound {sum(r['morphology'] == 'mound' for r in rows)}")
check(all(r["vent_field"] == "Mothra" for r in rows), "vent_field 全部为 Mothra")
check(np.array_equal(bd["z_wgs84"], np.load(OUT / "mothra_utm9n_1m.npz")["z_wgs84"]),
      "bundle 里的水深与独立栅格文件一致")

print("\n[7] 路径规划 waypoint 表")
wp = list(_csv.DictReader(
    (OUT / "mothra_waypoints.csv").read_text(encoding="utf-8-sig").splitlines()))
wmeta = json.loads((OUT / "mothra_waypoints_meta.json").read_text(encoding="utf-8"))

check(len(wp) == 28, "28 行 waypoint", f"{len(wp)} 行")
check([int(r["waypoint_id"]) for r in wp] == list(range(1, 29)),
      "waypoint_id 为 1..28 连续无缺")
check(all(r["type"] in ("chimney", "mound") for r in wp), "type 取值合法",
      f"chimney {sum(r['type'] == 'chimney' for r in wp)}, "
      f"mound {sum(r['type'] == 'mound' for r in wp)}")
check(all(float(r["height_m"]) > 0 for r in wp), "height_m 全为正",
      f"{min(float(r['height_m']) for r in wp):.0f}.."
      f"{max(float(r['height_m']) for r in wp):.0f} m")

# 与 vents 表逐点对齐 (waypoint 是它的重排/精简, 不能有漂移)
vseq = {int(r["sequence_id"]): r for r in rows}
check(all(int(r["sequence_id"]) in vseq for r in wp), "sequence_id 全部能回溯到 vents 表")
check(all(abs(float(r["lon"]) - float(vseq[int(r["sequence_id"])]["lon"])) < 1e-9 and
          abs(float(r["lat"]) - float(vseq[int(r["sequence_id"])]["lat"])) < 1e-9
          for r in wp), "经纬度与 vents 表逐点相等")
check(all(r["type"] == vseq[int(r["sequence_id"])]["morphology"] for r in wp),
      "type 与 vents 表 morphology 一致")

# 局部米制坐标: 独立重算, 不复用 07 的结果
ox, oy = wmeta["local_frame"]["origin_utm9n"]
check(all(abs(float(r["x_local_m"]) - (float(r["utm9n_easting"]) - ox)) < 1e-3 and
          abs(float(r["y_local_m"]) - (float(r["utm9n_northing"]) - oy)) < 1e-3
          for r in wp), "x/y_local_m = UTM 坐标 - 原点")

with rasterio.open(OUT / "mothra_utm9n_1m.tif") as a:
    gu = a.read(1)
nr = gu.shape[0]
_d = [abs(float(gu[(nr - 1) - int(float(r["y_local_m"])), int(float(r["x_local_m"]))])
          - float(r["seafloor_depth_m"])) for r in wp]
check(max(_d) < 1e-3,
      "按 meta 索引公式取栅格, 水深与 seafloor_depth_m 逐位相等",
      f"最大差 {max(_d):.2e} m")

# waypoint 的水深来自 UTM 栅格, vents 表的来自 WGS84 原生栅格; 两者应当接近但不等
_dv = [abs(float(r["seafloor_depth_m"])
           - float(vseq[int(r["sequence_id"])]["seafloor_depth_m"])) for r in wp]
check(max(_dv) < 3.0,
      "与 vents 表原生采样值之差在重采样合理范围内",
      f"中位 {sorted(_dv)[len(_dv) // 2]:.3f} m, 最大 {max(_dv):.3f} m")

# 米制坐标的方向性: 经度越大 -> x 越大; 纬度越大 -> y 越大
lon_o = np.argsort([float(r["lon"]) for r in wp])
x_o = np.argsort([float(r["x_local_m"]) for r in wp])
lat_o = np.argsort([float(r["lat"]) for r in wp])
y_o = np.argsort([float(r["y_local_m"]) for r in wp])
check(np.array_equal(lon_o, x_o) and np.array_equal(lat_o, y_o),
      "x 随经度单调、y 随纬度单调 (未出现轴向翻转)")

print(f"\n{'=' * 62}")
print(f"{checks} 项检查, {len(fails)} 项失败" + (f": {fails}" if fails else " —— 全部通过"))
sys.exit(1 if fails else 0)
