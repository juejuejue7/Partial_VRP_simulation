#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
00b — 全段场景候选总览图(供人工挑选, 非最终产物)
================================================================================
输入 : outputs/scenario_overview.npz   (由 00_find_scenarios.py 生成)
输出 : figures/scenario_overview.png

上排: 全段(四瓦片拼接)10 m 粗格网地形 + 572 个热液口 + 全部候选框(编号)
      + 右侧逐框统计表(尺寸/长宽比/面积/目标数/密度), 与编号一一对应。
下排: 三处候选框密集重叠区域的放大插图(North / Central-upper / Central-lower)
      —— 全段尺度下这些框彼此嵌套, 编号会挤在一起, 需要放大才看得清具体边界。

命名场用实线 + 橙色(项目既有 ACCENT 色); 三个搜索 preset 各用一种虚线样式 +
一种类别色, 线型与颜色双重编码(不单靠色觉区分, 沿用本项目热液口标记的一贯做法)。
候选框只是"待选", 不代表已冻结 —— 图上不写死为最终场景。

渲染沿用 04_mothra_plot.py 的 render_rgb(haxby + 排名直方图均衡 + 不过曝晕渲),
保证与项目其余地形图同一套配色语言。

运行环境: 需要 matplotlib (本机为 conda env auv_py310)
    D:/nixingxing/Anaconda/envs/auv_py310/python.exe scripts/00b_plot_scenario_overview.py
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["font.monospace"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans Mono"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
_m04 = import_module("04_mothra_plot")
render_rgb, m_per_deg = _m04.render_rgb, _m04.m_per_deg

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
FIGDIR = ROOT / "figures"
DPI = 220

INK = "#0b0b0b"
MUTED = "#898781"
AXIS = "#c3c2b7"

NAMED_STYLE = dict(color="#eb6834", ls="-", lw=1.6, casing="#ffffff")
PRESET_STYLE = {
    "sparse":    dict(color="#1c9099", ls=(0, (6, 3)), lw=1.5, casing="#ffffff", tag="S"),
    "crossaxis": dict(color="#8856a7", ls=(0, (2, 2)), lw=1.5, casing="#ffffff", tag="X"),
    "dense":     dict(color="#d81f42", ls=(0, (1, 1.6)), lw=1.6, casing="#ffffff", tag="D"),
}
PRESET_ORDER = ["sparse", "crossaxis", "dense"]
# 编号圆标放在各自框的角上(而非统一放顶边中点): 命名场与搜索候选的框经常彼此
# 嵌套或紧邻, 同放一个角会互相完全遮住(先画的圆被后画的盖掉)。四个角各自的
# 坐标随框的具体范围而不同, 天然错开。
CORNER = {"named": "tl", "sparse": "tr", "crossaxis": "bl", "dense": "br"}

# 三处候选框密集重叠区域的放大插图范围(WGS84 度, 已含约 40-60 m 边距)
INSETS = [
    ("North",          -129.0762, -129.0620, 47.9922, 48.0018,
     "Sasquatch(5) + sparse S1-S3"),
    ("Central-upper",  -129.0950, -129.0792, 47.9640, 47.9714,
     "High Rise(3) + crossaxis X1-X2 + dense D2"),
    ("Central-lower",  -129.1014, -129.0923, 47.9436, 47.9534,
     "MEF(2) + crossaxis X3 + dense D1,D3"),
]


def draw_box(ax, L, R, B, T, style, num_label, corner, zorder=5):
    w, h = R - L, T - B
    ax.add_patch(Rectangle((L, B), w, h, fill=False, ec=style["casing"],
                            lw=style["lw"] + 1.6, zorder=zorder))
    ax.add_patch(Rectangle((L, B), w, h, fill=False, ec=style["color"],
                            ls=style["ls"], lw=style["lw"], zorder=zorder + 1))
    cx = L if corner[1] == "l" else R
    cy = T if corner[0] == "t" else B
    ax.plot([cx], [cy], marker="o", ms=13, mfc=style["color"], mec="white",
            mew=1.3, zorder=zorder + 2)
    ax.text(cx, cy, num_label, ha="center", va="center", fontsize=7.5,
           color="white", weight="bold", zorder=zorder + 3)


def crop_rgb(rgb, lon0, lat0, dlon, dlat, nr, nc, L, R, B, T):
    c0 = max(0, int((L - lon0) / dlon) - 1)
    c1 = min(nc, int(np.ceil((R - lon0) / dlon)) + 1)
    r0 = max(0, int((B - lat0) / dlat) - 1)
    r1 = min(nr, int(np.ceil((T - lat0) / dlat)) + 1)
    sub = rgb[r0:r1, c0:c1]
    ext = [lon0 + c0 * dlon, lon0 + c1 * dlon, lat0 + r0 * dlat, lat0 + r1 * dlat]
    return sub, ext


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d = np.load(OUTDIR / "scenario_overview.npz", allow_pickle=False)
    zmean = d["zmean"].astype("float64")
    lon0, lat0 = float(d["lon0"]), float(d["lat0"])
    dlon, dlat = float(d["dlon"]), float(d["dlat"])
    nr, nc = int(d["nr"]), int(d["nc"])
    lat_ref = lat0 + 0.5 * nr * dlat
    mlon, mlat = m_per_deg(lat_ref)
    dx_m, dy_m = dlon * mlon, dlat * mlat
    print(f"[1/5] 底图 {nr}x{nc} (10 m 格), 格距 {dx_m:.2f} x {dy_m:.2f} m")

    named_ids = [str(x) for x in d["named_ids"]]
    named_labels = [str(x) for x in d["named_labels"]]
    named_boxes = d["named_boxes"]
    named_stats = d["named_stats"]
    searched_ids = [str(x) for x in d["searched_ids"]]
    searched_preset = [str(x) for x in d["searched_preset"]]
    searched_boxes = d["searched_boxes"]
    searched_stats = d["searched_stats"]
    vent_lon, vent_lat = d["vent_lon"], d["vent_lat"]

    print("[2/5] 渲染背景(haxby + 晕渲)")
    rgb = render_rgb(zmean, dx_m, dy_m, strength=0.6, vert_exag=2.5, norm_mode="equalize")
    extent = [lon0, lon0 + nc * dlon, lat0, lat0 + nr * dlat]

    # --- 汇总全部候选框为统一记录, 主图与插图共用 ----------------------------
    print("[3/5] 汇总候选框记录")
    boxes = []
    idx = 1
    for i in range(len(named_ids)):
        L, R, B, T = named_boxes[i]
        n, dens, area_ha, ar = named_stats[i]
        mlon_i, mlat_i = m_per_deg(0.5 * (B + T))
        boxes.append(dict(num=str(idx), kind="named", style=NAMED_STYLE,
                          corner=CORNER["named"], L=L, R=R, B=B, T=T,
                          label=named_labels[i], sid=named_ids[i],
                          w_m=(R - L) * mlon_i, h_m=(T - B) * mlat_i,
                          ar=float(ar), area_ha=float(area_ha), n=int(n), dens=float(dens)))
        idx += 1
    preset_counter = {p: 0 for p in PRESET_ORDER}
    for i in range(len(searched_ids)):
        preset = searched_preset[i]
        style = PRESET_STYLE[preset]
        preset_counter[preset] += 1
        num = f"{style['tag']}{preset_counter[preset]}"
        L, R, B, T = searched_boxes[i]
        n, dens, area_ha, ar = searched_stats[i]
        mlon_i, mlat_i = m_per_deg(0.5 * (B + T))
        boxes.append(dict(num=num, kind=preset, style=style, corner=CORNER[preset],
                          L=L, R=R, B=B, T=T, label="", sid=searched_ids[i],
                          w_m=(R - L) * mlon_i, h_m=(T - B) * mlat_i,
                          ar=float(ar), area_ha=float(area_ha), n=int(n), dens=float(dens)))

    print("[4/5] 画主图 + 三处放大插图 + 统计表")
    fig = plt.figure(figsize=(16.0, 17.2), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[3.05, 1.0], hspace=0.10)
    top = outer[0].subgridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.04)
    bot = outer[1].subgridspec(1, 3, wspace=0.32)

    ax = fig.add_subplot(top[0, 0])
    axt = fig.add_subplot(top[0, 1])
    axt.axis("off")

    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")
    ax.scatter(vent_lon, vent_lat, s=2.0, c=INK, alpha=0.55, linewidths=0, zorder=3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / (mlon / mlat))
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.set_xlabel("Longitude (deg)", color=MUTED, fontsize=9)
    ax.set_ylabel("Latitude (deg)", color=MUTED, fontsize=9)

    for bx in boxes:
        draw_box(ax, bx["L"], bx["R"], bx["B"], bx["T"], bx["style"], bx["num"], bx["corner"])

    # 在主图上标出三处插图的取景框(浅灰虚线), 让人看得出插图对应哪块
    for name, L, R, B, T, _ in INSETS:
        ax.add_patch(Rectangle((L, B), R - L, T - B, fill=False, ec=MUTED,
                                lw=0.9, ls=(0, (4, 3)), zorder=4))

    legend_handles = [Line2D([0], [0], color=NAMED_STYLE["color"], lw=2, label="命名热液场 (1-5)")]
    for p in PRESET_ORDER:
        s = PRESET_STYLE[p]
        legend_handles.append(Line2D([0], [0], color=s["color"], lw=2, ls=s["ls"],
                                     label=f"搜索: {p} ({s['tag']}1-{s['tag']}3)"))
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.9, edgecolor=AXIS)

    # --- 三处放大插图 --------------------------------------------------------
    for (name, L, R, B, T, sub_desc), gsij in zip(INSETS, [bot[0, 0], bot[0, 1], bot[0, 2]]):
        axi = fig.add_subplot(gsij)
        sub_rgb, sub_ext = crop_rgb(rgb, lon0, lat0, dlon, dlat, nr, nc, L, R, B, T)
        axi.imshow(sub_rgb, origin="lower", extent=sub_ext, interpolation="nearest")
        vm = ((vent_lon >= sub_ext[0]) & (vent_lon <= sub_ext[1])
             & (vent_lat >= sub_ext[2]) & (vent_lat <= sub_ext[3]))
        axi.scatter(vent_lon[vm], vent_lat[vm], s=6.0, c=INK, alpha=0.65, linewidths=0, zorder=3)
        for bx in boxes:
            if bx["R"] < L or bx["L"] > R or bx["T"] < B or bx["B"] > T:
                continue
            draw_box(axi, bx["L"], bx["R"], bx["B"], bx["T"], bx["style"], bx["num"], bx["corner"])
        axi.set_xlim(sub_ext[0], sub_ext[1])
        axi.set_ylim(sub_ext[2], sub_ext[3])
        mlon_i, mlat_i = m_per_deg(0.5 * (B + T))
        axi.set_aspect(1.0 / (mlon_i / mlat_i))
        axi.tick_params(colors=MUTED, labelsize=6.5, rotation=25)
        axi.xaxis.set_major_locator(MultipleLocator(0.002))
        axi.yaxis.set_major_locator(MultipleLocator(0.002))
        # 默认 ScalarFormatter 会在范围很窄时把公共前缀拆成右下角的 "+/-1.29e2"
        # 偏移量, 反而更难读; 直接定宽定点格式化, 刻度自成一行完整经纬度。
        axi.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
        axi.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
        for s in axi.spines.values():
            s.set_color(AXIS)
        axi.set_title(f"{name}\n{sub_desc}", fontsize=8.5, color=INK, pad=4)

    # --- 右侧统计表 ----------------------------------------------------------
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    header = f"{'#':>4}  {'label/id':<13} {'W×H(m)':>11} {'AR':>5} {'ha':>6} {'N':>3} {'/ha':>5}"
    lines = [("全段候选场景一览", 15, "bold", INK),
            ("背景: 10 m 粗格网水深, 灰点 = 572 个热液口全表", 8.5, "normal", MUTED),
            ("", 4, "normal", MUTED),
            (header, 9, "bold", INK)]
    for bx in boxes:
        tag = bx["label"] if bx["kind"] == "named" else bx["sid"]
        line = (f"{bx['num']:>4}  {tag:<13} {bx['w_m']:4.0f}x{bx['h_m']:<5.0f} "
               f"{bx['ar']:5.2f} {bx['area_ha']:6.1f} {bx['n']:3d} {bx['dens']:5.2f}")
        lines.append((line, 8.5, "normal", INK))
        note = (f"        ({bx['label']}, 几何落点计数, 非标签口径)" if bx["kind"] == "named"
               else f"        (preset={bx['kind']})")
        lines.append((note, 7.3, "normal", MUTED))

    lines.append(("", 4, "normal", MUTED))
    lines.append(("对照: Mothra(#1) 现为生产基准", 8, "italic", MUTED))
    lines.append(("候选框均为待选, 未冻结; 冻结需人工写入", 8, "italic", MUTED))
    lines.append(("registry (scenarios.json)", 8, "italic", MUTED))

    y = 0.985
    for text, size, weight, color in lines:
        fw = "bold" if weight == "bold" else "normal"
        style_kw = dict(style="italic") if weight == "italic" else {}
        axt.text(0.0, y, text, fontsize=size, family="monospace", color=color,
                 fontweight=fw, va="top", ha="left", transform=axt.transAxes, **style_kw)
        y -= size / 560.0 + 0.006

    fig.suptitle("Endeavour Segment — 全段场景候选总览 (决策支持图, 非最终产物)",
                 fontsize=13, color=INK, y=0.998)
    out = FIGDIR / "scenario_overview.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"[5/5] 写出 {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
