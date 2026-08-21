#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08 — 测区总览图 (Mothra 位置示意)
================================================================================
输入 : outputs/endeavour_context.npz  (全测区 1/3 降采样底图 + Mothra bbox)
输出 : figures/fig02_context_endeavour.png

只画: 地形 + Mothra 范围框 + 经纬度坐标轴。
无 colorbar / 无标题 / 无说明文字。

配色沿用 04_mothra_plot.py 的正式方案 (haxby + 排名直方图均衡 + 不过曝晕渲),
直接 import 其 render_rgb, 保证与主图同一套渲染, 不会各调各的。

两点与主图不同, 都是这张图特有的:
  1. 全测区只有 28.93% 是有效测深条带, 其余为 NaN -> 渲成白色; 出图前先裁到
     有效数据的外接范围, 否则右侧近半幅是空白。
  2. 底图是 1/3 降采样, 故晕渲格距按 3 倍给 (经向 3.0 m / 纬向 4.46 m)。
     这张图只作位置示意, 不用于任何定量分析。

Mothra 框用"白色外壳 + 深色芯"的双线画法: haxby 从深蓝跨到橙, 任何单色描边都会
在某一段底色上糊掉, 双线则压在哪一段都读得出来。

运行环境: 需要 matplotlib (本机为 conda env auv_py310)
    D:/nixingxing/Anaconda/envs/auv_py310/python.exe scripts/08_context_plot.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_m = import_module("04_mothra_plot")
render_rgb, m_per_deg = _m.render_rgb, _m.m_per_deg

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
FIGDIR = ROOT / "figures"

TICK_STEP = 0.005          # 经纬度刻度间隔 (度) —— 测区跨度比 Mothra 大得多
PAD_CELLS = 6              # 裁到有效数据外接范围时向外留的格数
BOX_CORE = "#d81f42"       # Mothra 框芯色
BOX_CASING = "#ffffff"     # 框外壳
DPI = 300

MUTED = "#898781"
AXIS = "#c3c2b7"
INK = "#0b0b0b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true",
                    help="在框旁标注 'Mothra' (默认不标, 保持无文字)")
    ap.add_argument("--shade", type=float, default=1.0, help="阴影强度 0..1")
    ap.add_argument("--vert-exag", type=float, default=2.0, help="垂直夸张")
    ap.add_argument("--sun-alt", type=float, default=45.0, help="光源高度角(度)")
    ap.add_argument("--norm", default="equalize", choices=["equalize", "linear"])
    ap.add_argument("--name", default="fig02_context_endeavour")
    args = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    c = np.load(OUTDIR / "endeavour_context.npz", allow_pickle=False)
    z = c["z"].astype("float64")
    lon0, lat0, lon1, lat1 = [float(v) for v in c["bounds_wgs84"]]
    mlon0, mlat0, mlon1, mlat1 = [float(v) for v in c["mothra_bbox"]]
    k = int(c["decimate"])
    h, w = z.shape

    latm = 0.5 * (lat0 + lat1)
    mlon, mlat = m_per_deg(latm)
    dx_m = (lon1 - lon0) / w * mlon          # 降采样后的实际格距
    dy_m = (lat1 - lat0) / h * mlat
    print(f"[1/3] 底图 {h}x{w} (1/{k} 降采样), 格距 {dx_m:.2f} x {dy_m:.2f} m")
    print(f"  有效格 {np.isfinite(z).sum()}/{z.size} "
          f"({100 * np.isfinite(z).mean():.2f}%)")

    rgb = render_rgb(z, dx_m, dy_m, strength=args.shade,
                     vert_exag=args.vert_exag, altdeg=args.sun_alt,
                     norm_mode=args.norm)

    # 裁到有效数据外接范围, 免得右侧近半幅是空白
    fin = np.isfinite(z)
    cc, rr = np.where(fin.any(axis=0))[0], np.where(fin.any(axis=1))[0]
    dlon, dlat = (lon1 - lon0) / w, (lat1 - lat0) / h
    vlon0 = lon0 + max(cc.min() - PAD_CELLS, 0) * dlon
    vlon1 = lon0 + min(cc.max() + 1 + PAD_CELLS, w) * dlon
    vlat1 = lat1 - max(rr.min() - PAD_CELLS, 0) * dlat
    vlat0 = lat1 - min(rr.max() + 1 + PAD_CELLS, h) * dlat
    print(f"[2/3] 裁到有效范围 lon {abs(vlon0):.4f}..{abs(vlon1):.4f}°W  "
          f"lat {vlat0:.4f}..{vlat1:.4f}°N")

    yscale = mlat / mlon
    map_w_in = 7.2
    map_h_in = map_w_in * (vlat1 - vlat0) * yscale / (vlon1 - vlon0)
    ml, mr, mt, mb = 0.92, 0.22, 0.16, 0.60
    fw, fh = map_w_in + ml + mr, map_h_in + mt + mb

    fig = plt.figure(figsize=(fw, fh), facecolor="white")
    ax = fig.add_axes([ml / fw, mb / fh, map_w_in / fw, map_h_in / fh])

    ax.imshow(rgb, extent=[lon0, lon1, lat0, lat1], origin="upper",
              interpolation="bilinear", zorder=1)
    ax.set_xlim(vlon0, vlon1)
    ax.set_ylim(vlat0, vlat1)
    ax.set_aspect(yscale)

    # Mothra 范围框: 白壳 + 深芯双线, 压在色带任何一段都读得出来
    for lw, col, zo in ((3.4, BOX_CASING, 4), (1.5, BOX_CORE, 5)):
        ax.add_patch(Rectangle((mlon0, mlat0), mlon1 - mlon0, mlat1 - mlat0,
                               facecolor="none", edgecolor=col,
                               linewidth=lw, zorder=zo))
    if args.label:
        ax.annotate("Mothra", xy=(mlon1, 0.5 * (mlat0 + mlat1)),
                    xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, color="white",
                    zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground=INK)])

    ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_STEP))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.3f}°W"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}°N"))
    ax.tick_params(colors=MUTED, labelsize=8, width=0.6, length=3)
    for s in ax.spines.values():
        s.set_color(AXIS)
        s.set_linewidth(0.6)

    p = FIGDIR / f"{args.name}.png"
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)

    print(f"[3/3] 水深 {-np.nanmax(z):.1f}..{-np.nanmin(z):.1f} m  "
          f"norm={args.norm}")
    print(f"  {p.name}  {p.stat().st_size / 1024:.1f} KB  "
          f"({int(fw * DPI)}x{int(fh * DPI)} px)")


if __name__ == "__main__":
    main()
