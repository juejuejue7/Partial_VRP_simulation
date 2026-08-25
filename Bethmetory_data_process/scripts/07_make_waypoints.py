#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07 — 热液口 -> 路径规划用 waypoint 表(registry 驱动, 多场景)
================================================================================
输入 : <outdir>/<prefix>_bundle.npz   (水深 + 热液口, 由 06 生成)
输出 : <outdir>/<prefix>_waypoints.csv        N 行, 纯表 (无注释行, 可直接 read_csv)
       <outdir>/<prefix>_waypoints_meta.json  局部坐标系定义等元数据
       (outdir 统一为 outputs/scenarios/<id>/, <id> 含 mothra 本身)

列:
    waypoint_id        1..28 的稳定编号。**不是访问顺序**, 只是标签;
                       顺序沿用源表 sequence_id, 便于回溯到原始文献表。
    sequence_id        源表 (TABLE_SI.xlsx) 的 Sequence ID, 用于 join 回原表
    lon, lat           WGS84 经纬度 (度)
    type               chimney / mound
    height_m           烟囱/丘体高出海底的高度 (源表给定)
    x_local_m,         局部平面直角坐标 (米), 原点 = UTM 栅格西南角,
    y_local_m          x 向东 / y 向北。见下方"为什么要给米制坐标"
    seafloor_depth_m   UTM 1 m 栅格在该点所在格子的海底水深 (负向下)
    utm9n_easting,     EPSG:32609 坐标 (米), 供 GIS 直接用
    utm9n_northing
    name               文献中的命名 (无名者为空)

为什么除了经纬度还要给米制坐标
--------------------------------------------------------------------------------
在本区纬度 (47.92N), 1 度经度 ≈ 74.6 km 而 1 度纬度 ≈ 111.2 km —— 相差 1.49 倍。
直接拿经纬度当平面坐标做距离计算/路径规划, 东西向距离会被高估约 49%。
故这里预先换算好局部米制坐标, 规划器直接用 x_local_m / y_local_m 即可,
不必自己再处理各向异性 (本项目早期正是在这一点上踩过坑)。

局部坐标与 1 m 栅格严格对齐, 可直接索引 z_utm1m:
    col = int(x_local_m)
    row = (nrows - 1) - int(y_local_m)
校验脚本会逐点验证这个关系。

关于 seafloor_depth_m 的取值来源
--------------------------------------------------------------------------------
本表的 seafloor_depth_m **取自 UTM 1 m 栅格上述格子的值**, 而不是沿用
mothra_vents.csv 里那一列 —— 后者采自 WGS84 原生栅格。两者在陡坡处可差到
约 1.2 m (中位差 0.04 m), 属重采样正常差异, 不是错误。
这里之所以改用 UTM 栅格: 规划器按 x/y_local_m 索引 z_utm1m 时, 取到的值必须与
本表逐位相同, 否则同一个点会有两个"水深"。要溯源到原生采样值请查
mothra_vents.csv 的同名列。

运行环境: 只需 numpy (base 或 auv_py310 均可)
    python scripts/07_make_waypoints.py                # 默认 --scenario mothra
    python scripts/07_make_waypoints.py --scenario mef
    python scripts/07_make_waypoints.py --all           # registry 里全部场景
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
REGISTRY = ROOT / "scenarios.json"

FIELDS = ["waypoint_id", "sequence_id", "lon", "lat", "type", "height_m",
          "x_local_m", "y_local_m", "seafloor_depth_m",
          "utm9n_easting", "utm9n_northing", "name"]


def load_registry() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def find_scenario(scenario_id: str) -> dict:
    for s in load_registry():
        if s["id"] == scenario_id:
            return s
    raise KeyError(f"scenarios.json 里没有 id={scenario_id!r}")


def process_scenario(scenario_id: str) -> None:
    sc = find_scenario(scenario_id)
    outdir = OUTDIR / "scenarios" / scenario_id
    prefix = scenario_id

    print(f"\n===== 场景 {scenario_id} ({sc['label']}) =====")
    bundle_p = outdir / f"{prefix}_bundle.npz"
    if not bundle_p.is_file():
        raise FileNotFoundError(f"{bundle_p} 不存在, 先跑 06_extract_vents.py "
                                f"--scenario {scenario_id}")
    d = np.load(bundle_p, allow_pickle=False)
    n = len(d["vent_lon"])
    nrows, ncols = d["z_utm1m"].shape
    x0, y0, x1, y1 = [float(v) for v in d["bounds_utm"]]
    print(f"[1/3] 载入 {n} 个热液口; UTM 栅格 {nrows}x{ncols}, "
          f"原点(西南角) E{x0:.1f} N{y0:.1f}")

    east = d["vent_easting"].astype("float64")
    north = d["vent_northing"].astype("float64")
    xl = east - x0                      # 向东为正, 单位米
    yl = north - y0                     # 向北为正, 单位米
    morph = d["vent_morphology"].astype(str)
    names = d["vent_name"].astype(str)

    print("[2/3] 校验局部坐标与 1 m 栅格对齐")
    col = np.floor(xl).astype(int)
    row = (nrows - 1) - np.floor(yl).astype(int)
    ok_col = int(np.sum(col == d["vent_col_utm1m"]))
    ok_row = int(np.sum(row == d["vent_row_utm1m"]))
    print(f"  col 对齐 {ok_col}/{n}, row 对齐 {ok_row}/{n}")
    if ok_col != n or ok_row != n:
        bad = np.where((col != d["vent_col_utm1m"]) | (row != d["vent_row_utm1m"]))[0]
        for i in bad:
            print(f"  ! wp{i + 1}: 算得 ({row[i]},{col[i]}) "
                  f"vs bundle ({d['vent_row_utm1m'][i]},{d['vent_col_utm1m'][i]})")
    inside = bool(np.all((xl >= 0) & (xl <= x1 - x0) &
                         (yl >= 0) & (yl <= y1 - y0)))
    print(f"  全部落在栅格范围内: {inside}")

    # 水深取自 UTM 栅格上局部坐标所指的那个格子, 保证与规划器索引结果逐位一致
    zu = d["z_utm1m"]
    depth = zu[row, col].astype("float64")
    ref = d["vent_seafloor_depth_m"].astype("float64")   # WGS84 原生采样值
    dif = np.abs(depth - ref)
    print(f"  与 WGS84 原生采样值之差: 中位 {np.median(dif):.3f} m, "
          f"最大 {dif.max():.3f} m (重采样正常差异)")

    recs = []
    for i in range(n):
        recs.append({
            "waypoint_id": i + 1,
            "sequence_id": int(d["vent_sequence_id"][i]),
            "lon": f"{d['vent_lon'][i]:.9f}",
            "lat": f"{d['vent_lat'][i]:.9f}",
            "type": morph[i],
            "height_m": f"{d['vent_height_m'][i]:.0f}",
            "x_local_m": f"{xl[i]:.3f}",
            "y_local_m": f"{yl[i]:.3f}",
            "seafloor_depth_m": f"{depth[i]:.3f}",
            "utm9n_easting": f"{east[i]:.3f}",
            "utm9n_northing": f"{north[i]:.3f}",
            "name": names[i],
        })

    print("[3/3] 写出")
    p = outdir / f"{prefix}_waypoints.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(recs)

    bundle_rel = str(bundle_p.relative_to(ROOT)).replace("\\", "/")
    meta = {
        "scenario_id": scenario_id,
        "n_waypoints": n,
        "source": f"{bundle_rel} (由 06_extract_vents.py 从 "
                  f"Bethy_data/TABLE_SI.xlsx 提取, 已按 {sc['label']} 裁剪 bbox 筛选)",
        "waypoint_id_note": "1..N 的稳定标签, 沿用源表 sequence_id 顺序; "
                            "不代表访问顺序, 路径规划器自行决定次序",
        "local_frame": {
            "definition": "原点 = UTM 9N 栅格西南角; x 向东, y 向北, 单位米",
            "origin_utm9n": [x0, y0],
            "epsg": 32609,
            "extent_m": [x1 - x0, y1 - y0],
            "grid_shape_rows_cols": [int(nrows), int(ncols)],
            "index_from_local": "col = int(x_local_m); row = (nrows-1) - int(y_local_m)",
            "why": "本区 1 度经度约 74.6 km 而 1 度纬度约 111.2 km (差 1.49 倍); "
                   "直接用经纬度做平面距离会把东西向高估约 49%",
        },
        "columns": {
            "height_m": "烟囱/丘体高出海底的高度, 源表给定",
            "seafloor_depth_m": "UTM 1 m 栅格在该点格子的海底水深, 负向下; "
                                "与 height_m 含义不同, 不要相加或混用。"
                                f"溯源用的 WGS84 原生采样值见 {prefix}_vents.csv "
                                "同名列, 两者在陡坡处可差约 1.2 m",
            "type": "chimney / mound",
        },
        "type_counts": {t: int(np.sum(morph == t)) for t in sorted(set(morph.tolist()))},
        "height_m_range": [float(d["vent_height_m"].min()),
                           float(d["vent_height_m"].max())],
        "local_extent_used_m": {"x": [float(xl.min()), float(xl.max())],
                                "y": [float(yl.min()), float(yl.max())]},
    }
    (outdir / f"{prefix}_waypoints_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  {p.name:32s} {p.stat().st_size / 1024:7.1f} KB  ({n} 行)")
    q = outdir / f"{prefix}_waypoints_meta.json"
    print(f"  {q.name:32s} {q.stat().st_size / 1024:7.1f} KB")
    print(f"  局部坐标跨度 x {xl.min():.1f}..{xl.max():.1f} m, "
          f"y {yl.min():.1f}..{yl.max():.1f} m")
    print(f"  类型 {meta['type_counts']}, 高度 "
          f"{meta['height_m_range'][0]:.0f}..{meta['height_m_range'][1]:.0f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mothra")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    ids = [s["id"] for s in load_registry()] if args.all else [args.scenario]
    for sid in ids:
        process_scenario(sid)


if __name__ == "__main__":
    main()
