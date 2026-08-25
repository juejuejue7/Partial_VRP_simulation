#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02 — Mothra 热液场地形渲染
================================================================================
输入 : outputs/scenarios/mothra/mothra_utm9n_1m.npz  (653x293, EPSG:32609, 1 m 正方形像元)
       outputs/scenarios/mothra/endeavour_context.npz (全测区降采样底图)

输出 (figures/):
       fig01_mothra_relief_light.png    主图: 晕渲地形 + 等深线   <- 展示用
       fig01_mothra_relief_dark.png     同上, 暗色版 (投影片用)
       fig02_context_endeavour.png      上下文: 全测区 + Mothra 框
       fig03_mothra_3d.png              三维透视

配色遵循 dataviz skill:
  * 水深是"量级", 用 sequential = 单色相蓝色阶 (palette.md step 100->700),
    浅 = 浅蓝, 深 = 深蓝。不用 jet/terrain 这类彩虹色阶。
  * 出图前用 _ramp_check 实测亮度单调性, 不靠眼睛判断。
  * 等深线带标注 = 水深的非颜色读法, 避免"连续量只靠颜色编码"。
  * 暗色版是自己重新取步 (去掉最深的 700 步), 不是把亮色版直接反相。
  * 网格/坐标轴为一档之差的实线细发丝线, 不用虚线。

运行环境: 需要 matplotlib (本机为 conda env auv_py310)
    D:/nixingxing/Anaconda/envs/auv_py310/python.exe scripts/02_plot_mothra.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from _palette import (ACCENT, AXIS, GRIDLINE, INK_MUTED, INK_PRIMARY,
                      INK_SECONDARY, PAGE, SEQ_BLUE, SEQ_BLUE_DARK, SURFACE)
from _ramp_check import check_ramp

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
FIGDIR = ROOT / "figures"

AZDEG, ALTDEG = 315.0, 45.0     # 晕渲光源: 西北高照, 制图惯例
VERT_EXAG = 2.0                 # 晕渲垂直夸张
CONTOUR_STEP = 10.0             # 等深线间隔 (m)
CONTOUR_LABEL_EVERY = 2         # 每隔几条标注一次
DPI = 200


# ------------------------------------------------------------------ 文本 ----
ZH = {
    "title": "Mothra 热液场 海底地形",
    "sub": "Endeavour 段, Juan de Fuca 洋脊 — AUV 多波束测深, 1 m 网格",
    "depth": "水深 (m)",
    "east": "东向距离 (m)", "north": "北向距离 (m)",
    "lon": "经度", "lat": "纬度",
    "ctx_title": "测区总览与 Mothra 裁剪范围",
    "ctx_sub": "EndeavourAUVSouthCentral1 全测区 (仅有效测深条带)",
    "d3_title": "Mothra 热液场 三维地形",
    "mothra": "Mothra",
    "relief": "起伏", "mean": "平均水深", "cell": "网格", "area": "范围",
    "bbox": "地理范围", "src": "源",
    "note3d": "垂直夸张 ×{ve:g}   ·   光源方位 {az:.0f}° / 高度角 {alt:.0f}°",
}
EN = {
    "title": "Mothra Hydrothermal Field — Seafloor Bathymetry",
    "sub": "Endeavour Segment, Juan de Fuca Ridge — AUV multibeam, 1 m grid",
    "depth": "Depth (m)",
    "east": "Easting (m)", "north": "Northing (m)",
    "lon": "Longitude", "lat": "Latitude",
    "ctx_title": "Survey Overview and Mothra Crop Extent",
    "ctx_sub": "EndeavourAUVSouthCentral1 full survey (valid swath only)",
    "d3_title": "Mothra Hydrothermal Field — 3D Terrain",
    "mothra": "Mothra",
    "relief": "Relief", "mean": "Mean depth", "cell": "Cell", "area": "Extent",
    "bbox": "Geographic extent", "src": "Source",
    "note3d": "vertical exaggeration ×{ve:g}   ·   illumination {az:.0f}° az / {alt:.0f}° alt",
}


def setup_fonts(lang: str) -> dict:
    """挑一个装得上的 CJK 字体; 没有就退回英文标签, 不出豆腐块。"""
    have = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                            "Source Han Sans SC", "MS Gothic") if n in have), None)
    if lang == "zh" and cjk is None:
        print("  ! 未找到中文字体, 标签退回英文")
        lang = "en"
    stack = ([cjk] if cjk else []) + ["DejaVu Sans", "sans-serif"]
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": stack,
        "axes.unicode_minus": False,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        # 水深 z 为负值, matplotlib 默认把负等值线画成虚线。虚线会被读成
        # "推测/阈值线", 这里全部强制为实线。
        "contour.negative_linestyle": "solid",
    })
    print(f"  字体: {stack[0]}   语言: {lang}")
    return ZH if lang == "zh" else EN


def theme(mode: str) -> dict:
    steps = SEQ_BLUE if mode == "light" else SEQ_BLUE_DARK
    check_ramp(steps, mode=mode, surface=SURFACE[mode],
               name=f"depth ramp ({mode})", verbose=True)
    return {
        "cmap": LinearSegmentedColormap.from_list(f"depth_{mode}", steps[::-1], N=256),
        "surface": SURFACE[mode], "page": PAGE[mode],
        "ink": INK_PRIMARY[mode], "ink2": INK_SECONDARY[mode],
        "muted": INK_MUTED[mode], "grid": GRIDLINE[mode],
        "axis": AXIS[mode], "accent": ACCENT[mode],
    }


# ------------------------------------------------------------------ 工具 ----
def shade(z: np.ndarray, cmap, norm, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """晕渲: 单色相水深底色 x 山体阴影。"""
    ls = LightSource(azdeg=AZDEG, altdeg=ALTDEG)
    rgb = cmap(norm(z))[..., :3]
    return ls.shade_rgb(rgb, z, vert_exag=VERT_EXAG, dx=dx, dy=dy, blend_mode="soft")


def style_axes(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for s in ax.spines.values():
        s.set_color(t["axis"])
        s.set_linewidth(0.6)
    ax.tick_params(colors=t["muted"], labelsize=8, width=0.6, length=3)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def scale_bar(ax, length: float, t: dict, label: str | None = None,
              x0: float = 0.06, y0: float = 0.045) -> None:
    """比例尺: 数据坐标即米, 直接量。"""
    xa, xb = ax.get_xlim()
    ya, yb = ax.get_ylim()
    px = xa + x0 * (xb - xa)
    py = ya + y0 * (yb - ya)
    h = 0.006 * (yb - ya)
    ax.add_patch(Rectangle((px, py), length, h, facecolor=t["ink"],
                           edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((px, py), length / 2, h, facecolor=t["surface"],
                           edgecolor=t["ink"], linewidth=0.5, zorder=7))
    ax.text(px + length / 2, py + 2.4 * h, label or f"{length:g} m",
            ha="center", va="bottom", fontsize=7.5, color=t["ink"], zorder=7)


def north_arrow(ax, t: dict, x: float = 0.93, y: float = 0.955) -> None:
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.055),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=9, color=t["ink"],
                arrowprops=dict(arrowstyle="-|>", color=t["ink"], linewidth=1.1,
                                shrinkA=0, shrinkB=1), zorder=7)


def depth_fmt(v, _pos=None) -> str:
    """z 是负向下; 刻度按海洋学惯例显示为正水深。"""
    return f"{-v:.0f}"


# ---------------------------------------------------------- 图 1: 主晕渲图 --
def fig_relief(d: dict, T: dict, mode: str) -> Path:
    t = theme(mode)
    z = d["z"].astype("float64")
    h, w = z.shape
    res = d["res"]
    extent = [0.0, w * res[1], 0.0, h * res[0]]      # 局部米制, 原点在西南角
    lon0, lat0, lon1, lat1 = d["bounds_wgs84"]

    norm = Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
    rgb = shade(z, t["cmap"], norm, dx=res[1], dy=res[0])

    # 按英寸显式排版: 地图是 equal aspect, 让画布去迁就它, 免得两侧留空缝
    map_w_in = 4.4
    map_h_in = map_w_in * (h * res[0]) / (w * res[1])
    # 左/右(含 colorbar)/上/下 留白; 下边距要同时容下刻度+轴标题+两行页脚
    ml, mr, mt, mb = 0.82, 1.30, 1.30, 1.12
    fw, fh = map_w_in + ml + mr, map_h_in + mt + mb
    fig = plt.figure(figsize=(fw, fh), facecolor=t["page"])
    ax = fig.add_axes([ml / fw, mb / fh, map_w_in / fw, map_h_in / fh])
    cax = fig.add_axes([(ml + map_w_in + 0.16) / fw, mb / fh,
                        0.17 / fw, map_h_in / fh])
    style_axes(ax, t)

    ax.imshow(rgb, extent=extent, origin="upper", interpolation="bilinear", zorder=1)

    # 等深线: 水深的非颜色读法
    lo = np.floor(np.nanmin(z) / CONTOUR_STEP) * CONTOUR_STEP
    hi = np.ceil(np.nanmax(z) / CONTOUR_STEP) * CONTOUR_STEP
    levels = np.arange(lo, hi + CONTOUR_STEP, CONTOUR_STEP)
    yy, xx = np.mgrid[0:h, 0:w]
    cs = ax.contour(xx * res[1] + res[1] / 2,
                    extent[3] - (yy * res[0] + res[0] / 2), z,
                    levels=levels, colors=t["ink"], linewidths=0.45,
                    linestyles="solid", alpha=0.42, zorder=3)
    ax.clabel(cs, levels[::CONTOUR_LABEL_EVERY], fmt=depth_fmt, fontsize=6.5,
              colors=t["ink"], inline=True, inline_spacing=3)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel(T["east"], fontsize=9)
    ax.set_ylabel(T["north"], fontsize=9)
    ax.set_xticks(np.arange(0, extent[1] + 1, 50))
    ax.set_yticks(np.arange(0, extent[3] + 1, 100))
    scale_bar(ax, 50.0, t)
    north_arrow(ax, t)

    sm = plt.cm.ScalarMappable(cmap=t["cmap"], norm=norm)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(T["depth"], fontsize=9, color=t["ink2"])
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(depth_fmt))
    cb.ax.tick_params(colors=t["muted"], labelsize=8, width=0.6, length=3)
    cb.outline.set_edgecolor(t["axis"])
    cb.outline.set_linewidth(0.6)

    xl = ml / fw
    fig.text(xl, 1 - 0.46 / fh, T["title"], fontsize=15, color=t["ink"],
             ha="left", va="top")
    fig.text(xl, 1 - 0.76 / fh, T["sub"], fontsize=8.5, color=t["ink2"],
             ha="left", va="top")
    stats = (f"{T['area']} {w * res[1]:.0f} × {h * res[0]:.0f} m   ·   "
             f"{T['cell']} {res[1]:g} × {res[0]:g} m (EPSG:32609)   ·   "
             f"{T['relief']} {np.nanmax(z) - np.nanmin(z):.1f} m   ·   "
             f"{T['mean']} {-np.nanmean(z):.1f} m")
    geo = (f"{T['bbox']} {abs(lon0):.5f}°W–{abs(lon1):.5f}°W, "
           f"{lat0:.5f}°N–{lat1:.5f}°N   ·   {T['src']} EndeavourAUVSouthCentral1.asc")
    fig.text(xl, 0.345 / fh, stats, fontsize=7.5, color=t["muted"],
             ha="left", va="bottom")
    fig.text(xl, 0.175 / fh, geo, fontsize=7.0, color=t["muted"],
             ha="left", va="bottom")

    p = FIGDIR / f"fig01_mothra_relief_{mode}.png"
    fig.savefig(p, facecolor=t["page"])
    plt.close(fig)
    return p


# ------------------------------------------------------- 图 2: 测区上下文 --
def fig_context(c: dict, T: dict, mode: str = "light") -> Path:
    t = theme(mode)
    z = c["z"].astype("float64")
    lon0, lat0, lon1, lat1 = c["bounds_wgs84"]
    mlon0, mlat0, mlon1, mlat1 = c["mothra_bbox"]

    norm = Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
    rgb = shade(np.nan_to_num(z, nan=np.nanmean(z)), t["cmap"], norm,
                dx=3.0, dy=4.46)
    rgba = np.dstack([rgb, np.where(np.isfinite(z), 1.0, 0.0)])

    # 全测区里只有 29% 是有效测深条带; 裁到有效范围, 否则右侧近半幅是空白
    finite = np.isfinite(z)
    cc = np.where(finite.any(axis=0))[0]
    rr = np.where(finite.any(axis=1))[0]
    dlon = (lon1 - lon0) / z.shape[1]
    dlat = (lat1 - lat0) / z.shape[0]
    pad = 6
    vlon0 = lon0 + max(cc.min() - pad, 0) * dlon
    vlon1 = lon0 + min(cc.max() + 1 + pad, z.shape[1]) * dlon
    vlat1 = lat1 - max(rr.min() - pad, 0) * dlat
    vlat0 = lat1 - min(rr.max() + 1 + pad, z.shape[0]) * dlat

    # 显式按英寸排版: 先按等比投影算出地图该有的宽高比, 再让画布迁就它
    yscale = 1.0 / np.cos(np.radians(0.5 * (lat0 + lat1)))
    map_w_in = 6.7
    map_h_in = map_w_in * (vlat1 - vlat0) * yscale / (vlon1 - vlon0)
    ml, mr, mt, mb = 0.95, 1.45, 1.32, 0.74
    fw, fh = map_w_in + ml + mr, map_h_in + mt + mb
    fig = plt.figure(figsize=(fw, fh), facecolor=t["page"])
    ax = fig.add_axes([ml / fw, mb / fh, map_w_in / fw, map_h_in / fh])
    cax = fig.add_axes([(ml + map_w_in + 0.20) / fw, mb / fh,
                        0.16 / fw, map_h_in / fh])
    style_axes(ax, t)

    ax.imshow(rgba, extent=[lon0, lon1, lat0, lat1], origin="upper",
              interpolation="bilinear", zorder=1)
    ax.add_patch(Rectangle((mlon0, mlat0), mlon1 - mlon0, mlat1 - mlat0,
                           facecolor="none", edgecolor=t["accent"],
                           linewidth=1.8, zorder=4))
    ax.annotate(T["mothra"], xy=(mlon1, 0.5 * (mlat0 + mlat1)),
                xytext=(9, 0), textcoords="offset points", ha="left",
                va="center", fontsize=11, color=t["accent"], zorder=5,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=t["surface"],
                          edgecolor="none", alpha=0.82))

    ax.set_xlim(vlon0, vlon1)
    ax.set_ylim(vlat0, vlat1)
    ax.set_aspect(yscale)
    ax.set_xlabel(T["lon"], fontsize=9)
    ax.set_ylabel(T["lat"], fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.3f}°W"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}°N"))
    # 经纬网压在栅格上是地图惯例, 但要退到最不抢眼的一档
    ax.grid(True, color=t["grid"], linewidth=0.4, linestyle="-", alpha=0.32, zorder=2)
    ax.set_axisbelow(False)

    sm = plt.cm.ScalarMappable(cmap=t["cmap"], norm=norm)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(T["depth"], fontsize=9, color=t["ink2"])
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(depth_fmt))
    cb.ax.tick_params(colors=t["muted"], labelsize=8, width=0.6, length=3)
    cb.outline.set_edgecolor(t["axis"])
    cb.outline.set_linewidth(0.6)

    xl = ml / fw
    fig.text(xl, 1 - 0.46 / fh, T["ctx_title"], fontsize=14, color=t["ink"],
             ha="left", va="top")
    fig.text(xl, 1 - 0.78 / fh, T["ctx_sub"], fontsize=8.5, color=t["ink2"],
             ha="left", va="top")
    p = FIGDIR / "fig02_context_endeavour.png"
    fig.savefig(p, facecolor=t["page"])
    plt.close(fig)
    return p


# ----------------------------------------------------------- 图 3: 三维图 --
def fig_3d(d: dict, T: dict, mode: str = "light", stride: int = 2) -> Path:
    t = theme(mode)
    z = d["z"].astype("float64")[::stride, ::stride]
    res = d["res"]
    h, w = z.shape
    x = np.arange(w) * res[1] * stride
    y = np.arange(h) * res[0] * stride
    X, Y = np.meshgrid(x, y[::-1])

    norm = Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
    # 降采样后梯度被摊薄, 晕渲要加重才看得出微地形
    ls = LightSource(azdeg=AZDEG, altdeg=ALTDEG)
    rgb = ls.shade_rgb(t["cmap"](norm(z))[..., :3], z, vert_exag=VERT_EXAG * 2.5,
                       dx=res[1] * stride, dy=res[0] * stride, blend_mode="soft")

    fig = plt.figure(figsize=(10.4, 6.6), facecolor=t["page"])
    # 3D 轴默认在画布里留一大圈空白, 显式给一个超出边界的 rect 把它撑开
    ax = fig.add_axes([-0.085, -0.045, 1.10, 1.06], projection="3d",
                      facecolor=t["page"])
    ax.plot_surface(X, Y, z, facecolors=rgb, rcount=h, ccount=w,
                    linewidth=0, antialiased=False, shade=False)

    # box_aspect 各轴单位 = 米, 于是 z 轴给的盒长 / z 实际跨度 就是垂直夸张倍数
    x_m, y_m = w * res[1] * stride, h * res[0] * stride
    z_m = float(np.nanmax(z) - np.nanmin(z))
    z_box = 3.0 * z_m                      # 垂直夸张 x3
    ve = z_box / z_m
    ax.set_box_aspect((x_m, y_m, z_box))
    ax.view_init(elev=34, azim=-142)     # 长轴(南北)斜贯画面, 不把地形挤到一角
    ax.set_xlabel(T["east"], fontsize=8.5, color=t["ink2"], labelpad=4)
    ax.set_ylabel(T["north"], fontsize=8.5, color=t["ink2"], labelpad=10)
    ax.set_zlabel(T["depth"], fontsize=8.5, color=t["ink2"], labelpad=6)
    ax.zaxis.set_major_formatter(plt.FuncFormatter(depth_fmt))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0))
        a._axinfo["grid"].update(color=t["grid"], linewidth=0.5, linestyle="-")
    ax.tick_params(colors=t["muted"], labelsize=7)

    sm = plt.cm.ScalarMappable(cmap=t["cmap"], norm=norm)
    cax = fig.add_axes([0.905, 0.20, 0.012, 0.5])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(T["depth"], fontsize=9, color=t["ink2"])
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(depth_fmt))
    cb.ax.tick_params(colors=t["muted"], labelsize=8, width=0.6, length=3)
    cb.outline.set_edgecolor(t["axis"])
    cb.outline.set_linewidth(0.6)

    fig.text(0.042, 0.945, T["d3_title"], fontsize=14, color=t["ink"])
    fig.text(0.042, 0.898, T["note3d"].format(ve=ve, az=AZDEG, alt=ALTDEG),
             fontsize=8.5, color=t["ink2"])
    p = FIGDIR / "fig03_mothra_3d.png"
    fig.savefig(p, facecolor=t["page"])
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 主流程 --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--only", default=None, help="只出某张图: relief/context/3d")
    args = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    T = setup_fonts(args.lang)

    d = dict(np.load(OUTDIR / "scenarios" / "mothra" / "mothra_utm9n_1m.npz"))
    c = dict(np.load(OUTDIR / "scenarios" / "mothra" / "endeavour_context.npz"))
    print(f"  载入 Mothra {d['z'].shape}, 上下文 {c['z'].shape}")

    made: list[Path] = []
    if args.only in (None, "relief"):
        for mode in ("light", "dark"):
            made.append(fig_relief(d, T, mode))
    if args.only in (None, "context"):
        made.append(fig_context(c, T))
    if args.only in (None, "3d"):
        made.append(fig_3d(d, T))

    print()
    for p in made:
        print(f"  {p.name:34s} {p.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
