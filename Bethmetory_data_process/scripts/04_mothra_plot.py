#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04 — Mothra 最简渲染图 (正式渲染方案)
================================================================================
输入 : outputs/scenarios/mothra/mothra_bundle.npz   (水深 + 28 个热液口; 无则退回只有水深的那份)
输出 : figures/mothra_plot.png

只画地形本身: 经纬度坐标轴 + 比例尺 + 热液口标记。
无标题/无说明文字/无 colorbar/无三维图。

渲染方案:
  * haxby 论文风格测深色带 (8 步, 深蓝->青->绿->黄->橙)
  * 排名直方图均衡: 按水深排名映射到 [0,1], 让拥挤的深水段摊开
  * 不过曝晕渲: 乘性 0.45 + 0.55*hs, 只压暗不提亮, 浅水区不会被光照冲白

已知取舍: haxby 亮度非单调 (升到 #eeea79 L=0.917 后回落到 #ff8c3a L=0.753),
灰度打印时若干深度会撞车, 最突出的是 #32c8ff(约 2272 m) 与 #ff8c3a(约 2197 m)
相差 75 m 却几乎同亮度 (ΔL=0.028)。彩色显示无碍; 要出黑白版需另换色阶。
直方图均衡另有一层代价: 颜色与水深不再等比, 且映射依赖当前窗口的数据分布,
换一块区域重出图, 同一颜色对应的水深会变 —— 跨图比较颜色是不成立的。

本脚本取代了早期的 viridis + 线性归一化版本 (其成图仍留在
figures/mothra_plot_viridis.png 备查)。渲染核心 render_rgb / make_cmap /
normalize 供 05_shade_compare.py 与 08_context_plot.py 复用。

运行环境: 需要 matplotlib (本机为 conda env auv_py310)
    D:/nixingxing/Anaconda/envs/auv_py310/python.exe scripts/04_mothra_plot.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
FIGDIR = ROOT / "figures"

# --- 渲染参数 ---
AZDEG = 315.0              # 光源方位: 西北高照, 制图惯例
ALTDEG = 45.0              # 光源高度角: 越高 -> 阴影越浅越短
VERT_EXAG = 2.0            # 垂直夸张
SHADE_STRENGTH = 1.0       # 阴影强度: 0 = 不压暗, 1 = 全量 (乘性下限 0.45)
CMAP_LO, CMAP_HI = 0.0, 1.0    # 色带截断区间
NORM = "equalize"          # equalize = 排名直方图均衡; linear = 与水深等比
TICK_STEP = 0.001          # 经纬度刻度间隔 (度)

# 论文风格测深色带 (haxby)
HAXBY = ["#0a2a8c", "#1e6ffb", "#32c8ff", "#6aebd6",
         "#b6ee9e", "#eeea79", "#ffbd57", "#ff8c3a"]
SHADE_FLOOR = 0.45         # 晕渲乘性下限: 最暗处也只压到原色的 45%, 不至于死黑
HS_LO, HS_SPAN = 0.15, 0.8  # 光强先线性拉伸再 clip, 避免高光过曝

VENT_EDGE = "#ffffff"
VENT_SIZE = 46.0
VENT_STYLE = {                      
    "chimney": ("o", "#e34948"),    
    "mound": ("^", "#eda100"),      
}
VENT_FALLBACK = ("s", "#e87ba4")    
SCALE_BAR_M = 50.0
DPI = 300

INK = "#0b0b0b"
MUTED = "#898781"
AXIS = "#c3c2b7"


def m_per_deg(lat: float) -> tuple[float, float]:
    """给定纬度下 1 度经度 / 1 度纬度各是多少米。"""
    r = np.radians(lat)
    lon_m = 111412.84 * np.cos(r) - 93.5 * np.cos(3 * r)
    lat_m = 111132.92 - 559.82 * np.cos(2 * r) + 1.175 * np.cos(4 * r)
    return float(lon_m), float(lat_m)


def make_cmap(lo: float = CMAP_LO, hi: float = CMAP_HI, n: int = 256):
    """haxby 色带, 可截取其中一段。"""
    base = LinearSegmentedColormap.from_list("bathy", HAXBY, N=n)
    cm = (base if (lo <= 0.0 and hi >= 1.0) else
          LinearSegmentedColormap.from_list(
              f"bathy_{lo:g}_{hi:g}", base(np.linspace(lo, hi, n)), N=n))
    cm.set_bad("white")
    return cm


def normalize(z: np.ndarray, mask: np.ndarray, mode: str = NORM) -> np.ndarray:
    """把水深映射到 [0,1]; 无效格留 NaN。

    equalize 按排名映射: 深水段在本区占 83% 面积却只占线性色阶的 40%, 均衡后
      主体层次拉得开。代价是颜色与水深不再等比, 且映射依赖当前窗口的数据分布。
    linear   与水深等比, 可跨图比较颜色, 但主体会挤在色带的一小段里。
    """
    out = np.full(z.shape, np.nan)
    valid = z[~mask]
    if valid.size == 0:
        return out
    if mode == "equalize":
        order = np.argsort(valid)
        ranks = np.empty(valid.size, float)
        ranks[order] = np.linspace(0.0, 1.0, valid.size)
        out[~mask] = ranks
    else:
        lo, hi = float(valid.min()), float(valid.max())
        out[~mask] = (valid - lo) / (hi - lo) if hi > lo else 0.5
    return out


def render_rgb(z: np.ndarray, dx_m: float, dy_m: float, *,
               strength: float = SHADE_STRENGTH, vert_exag: float = VERT_EXAG,
               altdeg: float = ALTDEG, azdeg: float = AZDEG,
               cmap_lo: float = CMAP_LO, cmap_hi: float = CMAP_HI,
               norm_mode: str = NORM) -> np.ndarray:
    """haxby 填色 + 不过曝晕渲。返回 (H, W, 3) 的 RGB。"""
    mask = np.isnan(z)
    # 算晕渲前先把无效格填掉, 否则 NaN 会沿梯度算子扩散出一圈伪影
    filled = np.where(mask, np.nanmean(z), z)

    rgb = make_cmap(cmap_lo, cmap_hi)(normalize(z, mask, norm_mode))[..., :3]

    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    hs = np.clip((ls.hillshade(filled, vert_exag=vert_exag, dx=dx_m, dy=dy_m)
                  - HS_LO) / HS_SPAN, 0.0, 1.0)
    # 乘性且只压暗: hs=1 时保持原色, hs=0 时压到 SHADE_FLOOR。
    # strength 在"不压暗"和"全量压暗"之间线性插值。
    mult = 1.0 - strength * (1.0 - (SHADE_FLOOR + (1.0 - SHADE_FLOOR) * hs))
    out = rgb * mult[..., None]
    out[mask] = 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shade", type=float, default=SHADE_STRENGTH,
                    help="阴影混合权重 0..1 (0 = 无阴影)")
    ap.add_argument("--vert-exag", type=float, default=VERT_EXAG,
                    help="垂直夸张, 越小光照越柔")
    ap.add_argument("--sun-alt", type=float, default=ALTDEG,
                    help="光源高度角(度), 越大阴影越浅")
    ap.add_argument("--no-hillshade", action="store_true",
                    help="不叠山体阴影, 只出纯色带填色")
    ap.add_argument("--cmap-lo", type=float, default=CMAP_LO,
                    help="haxby 色带起点 0..1 (截深蓝端)")
    ap.add_argument("--cmap-hi", type=float, default=CMAP_HI,
                    help="haxby 色带终点 0..1 (截橙端)")
    ap.add_argument("--norm", default=NORM, choices=["equalize", "linear"],
                    help="equalize=排名直方图均衡(默认); "
                         "linear=与水深等比, 可跨图比较颜色")
    ap.add_argument("--vents", default="morphology",
                    choices=["morphology", "uniform", "none"],
                    help="热液口标记: 按 chimney-mound 分符号 / 统一符号 / 不画")
    ap.add_argument("--vent-labels", action="store_true",
                    help="给有命名的热液口加名称标注")
    ap.add_argument("--no-legend", action="store_true",
                    help="不画图例 (分类标记下会让符号无从解读, 慎用)")
    ap.add_argument("--name", default="mothra_plot")
    args = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    
    mothra_dir = OUTDIR / "scenarios" / "mothra"
    bundle = mothra_dir / "mothra_bundle.npz"
    d = np.load(bundle if bundle.exists() else mothra_dir / "mothra_utm9n_1m.npz",
                allow_pickle=False)
    z = d["z_wgs84"].astype("float64")
    lon0, lat0, lon1, lat1 = [float(v) for v in d["bounds_wgs84"]]
    h, w = z.shape
    has_vents = "vent_lon" in d.files

    latm = 0.5 * (lat0 + lat1)
    mlon, mlat = m_per_deg(latm)
    dx_m = (lon1 - lon0) / w * mlon        # 列间距 1.000 m
    dy_m = (lat1 - lat0) / h * mlat        # 行间距 1.488 m

    strength = 0.0 if args.no_hillshade else args.shade
    rgb = render_rgb(z, dx_m, dy_m, strength=strength,
                     vert_exag=args.vert_exag, altdeg=args.sun_alt,
                     cmap_lo=args.cmap_lo, cmap_hi=args.cmap_hi,
                     norm_mode=args.norm)

    yscale = mlat / mlon                   # 屏幕纵横比校正
    map_w_in = 4.6
    map_h_in = map_w_in * (lat1 - lat0) * yscale / (lon1 - lon0)
    ml, mr, mt, mb = 0.92, 0.22, 0.16, 0.62
    fw, fh = map_w_in + ml + mr, map_h_in + mt + mb

    fig = plt.figure(figsize=(fw, fh), facecolor="white")
    ax = fig.add_axes([ml / fw, mb / fh, map_w_in / fw, map_h_in / fh])

    ax.imshow(rgb, extent=[lon0, lon1, lat0, lat1], origin="upper",
              interpolation="bilinear", zorder=1)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(yscale)

    ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_STEP))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.3f}°W"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}°N"))
    ax.tick_params(colors=MUTED, labelsize=8, width=0.6, length=3)
    for s in ax.spines.values():
        s.set_color(AXIS)
        s.set_linewidth(0.6)

    # 热液口标记 
    n_vent = 0
    if has_vents and args.vents != "none":
        vlon, vlat = d["vent_lon"], d["vent_lat"]
        morph = d["vent_morphology"].astype(str)
        names = d["vent_name"].astype(str)
        n_vent = len(vlon)
        if args.vents == "uniform":
            groups = [("热液口", np.ones(n_vent, bool), *VENT_STYLE["chimney"])]
        else:
            groups = [(m, morph == m, *VENT_STYLE.get(m, VENT_FALLBACK))
                      for m in sorted(set(morph.tolist()))]
        handles = []
        for lab, mask_vent, marker, fc in groups:
            if mask_vent.any():
                handles.append(ax.scatter(
                    vlon[mask_vent], vlat[mask_vent], s=VENT_SIZE, marker=marker,
                    facecolor=fc, edgecolor=VENT_EDGE, linewidth=1.0, zorder=5,
                    label=f"{lab} ({int(mask_vent.sum())})"))
        if len(handles) >= 2 and not args.no_legend:
            leg = ax.legend(handles=handles, loc="upper right", fontsize=8,
                            frameon=True, framealpha=0.92, borderpad=0.55,
                            handletextpad=0.5, labelspacing=0.45,
                            facecolor="white", edgecolor=AXIS)
            leg.get_frame().set_linewidth(0.6)
            leg.set_zorder(7)
            for t in leg.get_texts():
                t.set_color(INK)
        if args.vent_labels:
            for lo_, la_, nm in zip(vlon, vlat, names):
                if nm:
                    ax.annotate(nm, xy=(lo_, la_), xytext=(6, 4),
                                textcoords="offset points", fontsize=6.5,
                                color="white", zorder=6,
                                path_effects=[pe.withStroke(linewidth=2.0,
                                                            foreground=INK)])

    # 比例尺
    bar_deg = SCALE_BAR_M / mlon
    bx = lon0 + 0.055 * (lon1 - lon0)
    by = lat0 + 0.030 * (lat1 - lat0)
    bh = 0.0055 * (lat1 - lat0)
    ax.add_patch(Rectangle((bx, by), bar_deg, bh, facecolor=INK,
                           edgecolor="white", linewidth=0.8, zorder=6))
    ax.add_patch(Rectangle((bx, by), bar_deg / 2, bh, facecolor="white",
                           edgecolor="white", linewidth=0.8, zorder=7))
    ax.text(bx + bar_deg / 2, by + 2.4 * bh, f"{SCALE_BAR_M:g} m",
            ha="center", va="bottom", fontsize=8.5, color="white", zorder=8,
            path_effects=[pe.withStroke(linewidth=2.2, foreground=INK)])

    p = FIGDIR / f"{args.name}.png"
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)

    print(f"  阴影 strength={strength:g}  vert_exag={args.vert_exag:g}  "
          f"sun_alt={args.sun_alt:g}deg  az={AZDEG:g}deg  "
          f"(乘性下限 {SHADE_FLOOR:g}, 只压暗不提亮)")
    print(f"  色带 haxby[{args.cmap_lo:g}, {args.cmap_hi:g}]  norm={args.norm}"
          + ("  (颜色与水深不等比, 且依赖本窗口分布, 跨图不可比)"
             if args.norm == "equalize" else "  (颜色与水深等比)"))
    print(f"  {z.shape[0]}x{z.shape[1]}  格距 {dx_m:.3f} x {dy_m:.3f} m")
    print(f"  经度 {abs(lon0):.5f}°W..{abs(lon1):.5f}°W   "
          f"纬度 {lat0:.5f}°N..{lat1:.5f}°N")
    print(f"  水深 {-np.nanmax(z):.1f}..{-np.nanmin(z):.1f} m")
    print(f"  热液口 {n_vent} 个 (样式 {args.vents}"
          + (", 带名称标注" if args.vent_labels else "") + ")")
    print(f"  {p.name}  {p.stat().st_size / 1024:.1f} KB  "
          f"({int(fw * DPI)}x{int(fh * DPI)} px)")


if __name__ == "__main__":
    main()