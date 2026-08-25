#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
00 — 全段场景候选搜索(命名热液场 + 滑窗补空档)
================================================================================
输入 : Bethy_data/EndeavourAUVSouthCentral1.asc                    (SouthCentral, 现有)
       Bethy_data/TABLE_SI.xlsx                                    (572 点热液口全表)
       ../Bethmetory_data/MGDS_Download/JdF_Endeavour_Bathymetry/
           EndeavourAUVTopoSouth1mArc.grd / TopoCentral1mArc.grd / TopoNorth1mArc.grd
       (三块新瓦片来自同一 MGDS 数据集 21403, 已解压; 需要的话见
        Bethmetory_data/README.md 的获取说明)

输出 (outputs/):
       scenario_overview.npz     10 m 粗格网底图(全段拼接)+ 全部候选框定义,
                                  供 00b_plot_scenario_overview.py 出图
       scenario_candidates.csv   候选框平铺表(命名场 + 三个 preset 的搜索结果),
                                  供人工挑选, 不代表已冻结

本脚本只探路, 不写 scenarios.json —— 冻结动作是后续人工从候选里选定之后,
由 A0 手动写入 registry 的一步, 不在这里自动发生。

口径:
  * 目标筛选一律按几何落点(经纬度落在 bbox 内), 不按 Vent Field Name 标签
    —— 与 Mothra 既定口径一致(见 vrpsim/contracts/dataset.py 的注释)。
  * 四块瓦片 cellsize 各不相同(1.3374~1.3398e-5 deg), 不在同一原生格网上,
    搜索统一走 10 m 公共粗格网(积分图), 真正裁剪时每个场景只从它所属的那
    一块瓦片裁, 不做跨瓦片拼接; 跨瓦片边界的候选窗口会被剔除。
  * Mothra 的 bbox 是已冻结的生产值(01_crop_reproject.py), 直接 import 复用,
    不重新用"点集+缓冲"规则推导 —— 那条规则只用于本脚本新构造的命名场。

运行环境: 需要 rasterio + pyproj + openpyxl (本机为 conda base)
    python scripts/00_find_scenarios.py
"""
from __future__ import annotations

import math
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import openpyxl
import rasterio

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
OUTDIR = ROOT / "outputs"
SRC_XLSX = ROOT / "Bethy_data" / "TABLE_SI.xlsx"

sys.path.insert(0, str(Path(__file__).resolve().parent))
_m01 = import_module("01_crop_reproject")

TILES = {
    # name -> (path, "先南后北" 排布顺序仅用于打印, 与搜索逻辑无关)
    "South": ROOT.parent / "Bethmetory_data" / "MGDS_Download" /
             "JdF_Endeavour_Bathymetry" / "EndeavourAUVTopoSouth1mArc.grd",
    "SouthCentral": ROOT / "Bethy_data" / "EndeavourAUVSouthCentral1.asc",
    "Central": ROOT.parent / "Bethmetory_data" / "MGDS_Download" /
               "JdF_Endeavour_Bathymetry" / "EndeavourAUVTopoCentral1mArc.grd",
    "North": ROOT.parent / "Bethmetory_data" / "MGDS_Download" /
             "JdF_Endeavour_Bathymetry" / "EndeavourAUVTopoNorth1mArc.grd",
}

LAT0_REF = 47.95                       # 米制换算参考纬度(全段跨度小, 单一参考足够)
C = 10.0                                # 粗格边长 (m), 背景底图与搜索共用同一格网
NMS_SEP_M = 300.0                       # 同一 preset 内候选框中心的最小间距(强制真实地理多样性)
TOPK_PER_PRESET = 3

# 三个搜索 preset: (密度下限, 密度上限, AR 上限, AR 下限, 最小面积 ha, 最小目标数, 打分口径)
# crossaxis 的 ar_hi 收紧到 0.6(而非仅 <0.8): 打分口径若单纯按 n 排序, 会偏向"面积
# 大到刚好卡在 AR 上限"的近方形窗口, 選不出真正東西向拉长的窄条 —— 收紧上限把形状
# 逼扁, 再在扁窗口里挑目标数最多的。
PRESETS = {
    "sparse": dict(dens_lo=0.25, dens_hi=1.00, ar_lo=None, ar_hi=None,
                   min_area_ha=10.0, min_n=4, score="area"),
    "crossaxis": dict(dens_lo=None, dens_hi=None, ar_lo=None, ar_hi=0.6,
                      min_area_ha=5.0, min_n=8, score="n"),
    "dense": dict(dens_lo=4.0, dens_hi=None, ar_lo=None, ar_hi=None,
                 min_area_ha=12.0, min_n=None, score="dens"),
}
WIN_SIZES_CELLS = [(w, h) for w in range(15, 71, 5) for h in range(15, 71, 5)]
STRIDE_CELLS = 2


def m_per_deg(lat: float) -> tuple[float, float]:
    return _m01.np.cos(_m01.np.radians(lat)) * 111320.0, 110570.0


def load_vents() -> dict:
    ws = openpyxl.load_workbook(SRC_XLSX, data_only=True)["forPublication"]
    rows = [r for r in ws.iter_rows(min_row=3, values_only=True)
            if isinstance(r[2], (int, float)) and isinstance(r[3], (int, float))]
    lon = np.array([r[2] for r in rows], dtype="float64")
    lat = np.array([r[3] for r in rows], dtype="float64")
    field = np.array([(r[6] or "").strip().replace("HIgh", "High") for r in rows])
    return dict(lon=lon, lat=lat, field=field, n=len(rows))


def named_field_bbox(lon: np.ndarray, lat: np.ndarray, buffer_m: float = 30.0):
    mlon, mlat = m_per_deg(float(np.mean(lat)))
    blon, blat = buffer_m / mlon, buffer_m / mlat
    return (float(lon.min() - blon), float(lon.max() + blon),
            float(lat.min() - blat), float(lat.max() + blat))


def which_tile(lat_c: float) -> str | None:
    for name, path in TILES.items():
        with rasterio.open(path) as ds:
            b = ds.bounds
            if b.bottom <= lat_c <= b.top:
                return name
    return None


def count_in_bbox(vents: dict, L, R, B, T) -> int:
    m = ((vents["lon"] >= L) & (vents["lon"] <= R)
         & (vents["lat"] >= B) & (vents["lat"] <= T))
    return int(m.sum())


def build_named_fields(vents: dict) -> list[dict]:
    out = []
    # Mothra: 冻结生产值, 不重新推导
    L, R = _m01.LON_MIN, _m01.LON_MAX
    B, T = _m01.LAT_MIN, _m01.LAT_MAX
    out.append(dict(id="mothra", label="Mothra", lon_min=L, lon_max=R,
                    lat_min=B, lat_max=T, source="frozen(01_crop_reproject.py)"))
    for field in ["MEF", "High Rise", "Salty Dawg", "Sasquatch"]:
        m = vents["field"] == field
        L, R, B, T = named_field_bbox(vents["lon"][m], vents["lat"][m])
        out.append(dict(id=field.lower().replace(" ", "_"), label=field,
                        lon_min=L, lon_max=R, lat_min=B, lat_max=T,
                        source="points+30m buffer"))
    return out


def build_mosaic_and_search_grid(vents: dict):
    lon0, lon1 = -129.141, -129.0560
    lat0, lat1 = 47.8820, 48.0050
    mlon, mlat = m_per_deg(LAT0_REF)
    dlon, dlat = C / mlon, C / mlat
    nc = int(np.ceil((lon1 - lon0) / dlon))
    nr = int(np.ceil((lat1 - lat0) / dlat))
    print(f"[1/4] 粗格网 {nr} x {nc} ({C:.0f} m 格, 全段拼接)")

    tot = np.zeros(nr * nc, dtype=np.int64)
    val = np.zeros(nr * nc, dtype=np.int64)
    zsum = np.zeros(nr * nc, dtype=np.float64)
    for name, p in TILES.items():
        with rasterio.open(p) as ds:
            z = ds.read(1)
            nod = ds.nodata
            ok = np.isfinite(z)
            if nod is not None and not np.isnan(nod):
                ok &= (z != nod)
            tf = ds.transform
            lats = tf.f + (np.arange(ds.height) + 0.5) * tf.e
            lons = tf.c + (np.arange(ds.width) + 0.5) * tf.a
            ri_full = np.floor((lats - lat0) / dlat).astype(np.int64)
            ci_full = np.floor((lons - lon0) / dlon).astype(np.int64)
            m_r = (ri_full >= 0) & (ri_full < nr)
            m_c = (ci_full >= 0) & (ci_full < nc)
            ri = ri_full[m_r]
            ci = ci_full[m_c]
            okc = ok[np.ix_(m_r, m_c)]
            zc = np.where(okc, z[np.ix_(m_r, m_c)], 0.0)
            idx = (ri[:, None] * nc + ci[None, :]).ravel()
            tot += np.bincount(idx, minlength=nr * nc)
            val += np.bincount(idx, weights=okc.ravel().astype(np.float64),
                               minlength=nr * nc).astype(np.int64)
            zsum += np.bincount(idx, weights=zc.ravel(), minlength=nr * nc)
            del z, ok, okc, zc, idx
        print(f"  吃进 {name}: {p.name}")

    tot = tot.reshape(nr, nc)
    val = val.reshape(nr, nc)
    zsum = zsum.reshape(nr, nc)
    zmean = np.full((nr, nc), np.nan)
    has = val > 0
    zmean[has] = zsum[has] / val[has]
    good = (tot > 0) & (val == tot)          # 粗格 100% 有效(供搜索用的硬条件)
    print(f"  全有效粗格 {int(good.sum())}/{int((tot > 0).sum())} 有数据格")

    tgt = np.zeros((nr, nc), dtype=np.int64)
    vr = np.floor((vents["lat"] - lat0) / dlat).astype(int)
    vc = np.floor((vents["lon"] - lon0) / dlon).astype(int)
    mm = (vr >= 0) & (vr < nr) & (vc >= 0) & (vc < nc)
    np.add.at(tgt, (vr[mm], vc[mm]), 1)
    print(f"  目标点入格 {int(mm.sum())}/{vents['n']}")

    grid = dict(nr=nr, nc=nc, lon0=lon0, lat0=lat0, dlon=dlon, dlat=dlat,
               zmean=zmean, good=good, tgt=tgt, tile_id=_tile_id_grid(nr, nc, lat0, dlat))
    return grid


def _tile_id_grid(nr, nc, lat0, dlat):
    """每行落在哪块瓦片(用于剔除跨瓦片候选窗口)。以行中心纬度判定, 唯一命中记 id, 否则 -1。"""
    row_lat = lat0 + (np.arange(nr) + 0.5) * dlat
    tid = np.full(nr, -1, dtype=np.int8)
    for i, (name, p) in enumerate(TILES.items()):
        with rasterio.open(p) as ds:
            b = ds.bounds
            m = (row_lat >= b.bottom) & (row_lat <= b.top)
            tid[m & (tid == -1)] = i
            tid[m & (tid != -1) & (tid != i)] = -2   # 落在多块瓦片重叠带, 排除
    return tid


def _integral(a):
    return np.pad(np.cumsum(np.cumsum(a.astype(np.int64), 0), 1), ((1, 0), (1, 0)))


def _boxsum(I, r0, c0, h, w):
    return I[r0 + h, c0 + w] - I[r0, c0 + w] - I[r0 + h, c0] + I[r0, c0]


def search_preset(grid: dict, name: str, cfg: dict) -> list[dict]:
    nr, nc = grid["nr"], grid["nc"]
    I_bad = _integral(~grid["good"])
    I_tgt = _integral(grid["tgt"])
    mlon, mlat = m_per_deg(LAT0_REF)
    cands = []
    for w, h in WIN_SIZES_CELLS:
        r0 = np.arange(0, nr - h, STRIDE_CELLS)
        c0 = np.arange(0, nc - w, STRIDE_CELLS)
        R, Cc = np.meshgrid(r0, c0, indexing="ij")
        bad = _boxsum(I_bad, R, Cc, h, w)
        n = _boxsum(I_tgt, R, Cc, h, w)
        ok = (bad == 0) & (n >= (cfg["min_n"] or 0))
        if not ok.any():
            continue
        Wm = w * C
        Hm = h * C
        area_ha = Wm * Hm / 1e4
        ar = Hm / Wm
        dens = np.where(area_ha > 0, n / area_ha, 0.0)
        m2 = ok & (area_ha >= cfg["min_area_ha"])
        if cfg["dens_lo"] is not None:
            m2 &= dens >= cfg["dens_lo"]
        if cfg["dens_hi"] is not None:
            m2 &= dens <= cfg["dens_hi"]
        if cfg["ar_lo"] is not None:
            m2 &= ar >= cfg["ar_lo"]
        if cfg["ar_hi"] is not None:
            m2 &= ar <= cfg["ar_hi"]
        # 同一行内瓦片必须一致: 用行首/行尾的 tile_id 相等且非负判定
        tid_top = grid["tile_id"][R]
        tid_bot = grid["tile_id"][np.minimum(R + h - 1, nr - 1)]
        m2 &= (tid_top == tid_bot) & (tid_top >= 0)
        if not m2.any():
            continue
        for r, c, nn, dd in zip(R[m2], Cc[m2], n[m2], dens[m2]):
            lon_c = grid["lon0"] + (c + w / 2) * grid["dlon"]
            lat_c = grid["lat0"] + (r + h / 2) * grid["dlat"]
            cands.append(dict(r=int(r), c=int(c), h=h, w=w, n=int(nn), dens=float(dd),
                              area_ha=float(area_ha), ar=float(ar), lon_c=lon_c, lat_c=lat_c,
                              W_m=Wm, H_m=Hm,
                              lon_min=grid["lon0"] + c * grid["dlon"],
                              lon_max=grid["lon0"] + (c + w) * grid["dlon"],
                              lat_min=grid["lat0"] + r * grid["dlat"],
                              lat_max=grid["lat0"] + (r + h) * grid["dlat"]))
    print(f"  [{name}] 满足硬条件的候选窗口 {len(cands)} 个")
    if not cands:
        return []

    score_key = cfg["score"]
    scores = np.array([c[{"area": "area_ha", "n": "n", "dens": "dens"}[score_key]]
                       for c in cands])
    order = np.argsort(-scores)
    picked: list[dict] = []
    picked_xy = []
    for i in order:
        cd = cands[i]
        x = (cd["lon_c"] - grid["lon0"]) * mlon
        y = (cd["lat_c"] - grid["lat0"]) * mlat
        if all(math.hypot(x - px, y - py) >= NMS_SEP_M for px, py in picked_xy):
            picked.append(cd)
            picked_xy.append((x, y))
        if len(picked) >= TOPK_PER_PRESET:
            break
    for cd in picked:
        cd["tile"] = which_tile(cd["lat_c"])
    return picked


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] 读取热液口全表")
    vents = load_vents()
    print(f"  共 {vents['n']} 点")

    print("[2/4] 构建命名场 bbox")
    named = build_named_fields(vents)
    for f in named:
        f["n_targets"] = count_in_bbox(vents, f["lon_min"], f["lon_max"],
                                       f["lat_min"], f["lat_max"])
        mlon, mlat = m_per_deg(0.5 * (f["lat_min"] + f["lat_max"]))
        f["W_m"] = (f["lon_max"] - f["lon_min"]) * mlon
        f["H_m"] = (f["lat_max"] - f["lat_min"]) * mlat
        f["area_ha"] = f["W_m"] * f["H_m"] / 1e4
        f["ar"] = f["H_m"] / f["W_m"]
        f["dens"] = f["n_targets"] / f["area_ha"]
        f["tile"] = which_tile(0.5 * (f["lat_min"] + f["lat_max"]))
        print(f"  {f['id']:12s} tile={f['tile']:13s} N={f['n_targets']:3d} "
              f"{f['W_m']:5.0f}x{f['H_m']:5.0f}m AR={f['ar']:.2f} "
              f"{f['area_ha']:5.1f}ha {f['dens']:.2f}/ha")

    print("[3/4] 拼接全段 10 m 粗格网 + 三个 preset 滑窗搜索")
    grid = build_mosaic_and_search_grid(vents)
    searched = []
    for name, cfg in PRESETS.items():
        picks = search_preset(grid, name, cfg)
        for i, cd in enumerate(picks, start=1):
            cd["id"] = f"{name}_{i}"
            cd["label"] = f"{name}#{i}"
            cd["preset"] = name
            searched.append(cd)

    print("[4/4] 写出")
    rows = []
    for f in named:
        rows.append(dict(id=f["id"], kind="named", preset="", tile=f["tile"],
                         lon_min=f["lon_min"], lon_max=f["lon_max"],
                         lat_min=f["lat_min"], lat_max=f["lat_max"],
                         W_m=f["W_m"], H_m=f["H_m"], AR=f["ar"],
                         area_ha=f["area_ha"], n_targets=f["n_targets"],
                         density_per_ha=f["dens"]))
    for cd in searched:
        rows.append(dict(id=cd["id"], kind="searched", preset=cd["preset"],
                         tile=cd["tile"], lon_min=cd["lon_min"], lon_max=cd["lon_max"],
                         lat_min=cd["lat_min"], lat_max=cd["lat_max"],
                         W_m=cd["W_m"], H_m=cd["H_m"], AR=cd["ar"],
                         area_ha=cd["area_ha"], n_targets=cd["n"],
                         density_per_ha=cd["dens"]))

    import csv
    csv_p = OUTDIR / "scenario_candidates.csv"
    with csv_p.open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {csv_p.name} ({len(rows)} 行)")

    np.savez_compressed(
        OUTDIR / "scenario_overview.npz",
        zmean=grid["zmean"].astype("float32"),
        lon0=grid["lon0"], lat0=grid["lat0"], dlon=grid["dlon"], dlat=grid["dlat"],
        nr=grid["nr"], nc=grid["nc"],
        vent_lon=vents["lon"], vent_lat=vents["lat"], vent_field=vents["field"],
        named_ids=np.array([f["id"] for f in named]),
        named_labels=np.array([f["label"] for f in named]),
        named_boxes=np.array([[f["lon_min"], f["lon_max"], f["lat_min"], f["lat_max"]]
                              for f in named]),
        named_stats=np.array([[f["n_targets"], f["dens"], f["area_ha"], f["ar"]]
                              for f in named]),
        searched_ids=np.array([c["id"] for c in searched]),
        searched_preset=np.array([c["preset"] for c in searched]),
        searched_boxes=np.array([[c["lon_min"], c["lon_max"], c["lat_min"], c["lat_max"]]
                                 for c in searched]) if searched else np.zeros((0, 4)),
        searched_stats=np.array([[c["n"], c["dens"], c["area_ha"], c["ar"]]
                                 for c in searched]) if searched else np.zeros((0, 4)),
    )
    print(f"  scenario_overview.npz")
    print("\n完成。下一步: D:/nixingxing/Anaconda/envs/auv_py310/python.exe "
          "scripts/00b_plot_scenario_overview.py")


if __name__ == "__main__":
    main()
