#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02 — 出静态基础环境图。

输入 : VRPSimulation/data/mothra_world.npz    (由 01 生成)
       VRPSimulation/data/mothra_basemap.npz  (可选底色)
输出 : VRPSimulation/figures/mothra_world_ned.png

人工检查项:场地必须是 **653 m 高 x 294 m 宽的竖长条**,28 个标记落在概率场亮斑上。
若出来是横长条 —— 轴序写反了(CSV 的 x 是 East,本项目 x 是 North)。

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/02_plot_world.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vrpsim.contracts.config import DEFAULT_DATA_DIR, DEFAULT_FIGURE_DIR  # noqa: E402
from vrpsim.contracts.style import dump_style, load_style  # noqa: E402
from vrpsim.viz import load_basemap, plot_world, use_cjk_font  # noqa: E402
from vrpsim.world import load_world  # noqa: E402


def _clamp(lo: float, hi: float, floor: float, ceil: float):
    """把 [lo,hi] 平移进 [floor,ceil],保持长度(装不下就退化为整段)。"""
    span = min(hi - lo, ceil - floor)
    lo = min(max(lo, floor), ceil - span)
    return lo, lo + span


def _densest_center(mw, *, half_m: float):
    """目标最密的 (2*half_m)^2 窗口中心 —— 用来选放大位置,不写死坐标。"""
    import numpy as np

    tg = mw.dataset.targets_ned
    d = np.hypot(tg[:, None, 0] - tg[None, :, 0], tg[:, None, 1] - tg[None, :, 1])
    counts = (d <= half_m).sum(axis=1)
    near = d[int(np.argmax(counts))] <= half_m
    return tg[near].mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=os.path.join(DEFAULT_DATA_DIR, "mothra_world.npz"))
    ap.add_argument("--basemap", default=os.path.join(DEFAULT_DATA_DIR, "mothra_basemap.npz"))
    ap.add_argument("--no-basemap", action="store_true", help="不铺水深晕渲底色")
    ap.add_argument("--names", action="store_true", help="标注已命名的热液口")
    ap.add_argument("--out", default=os.path.join(DEFAULT_FIGURE_DIR, "mothra_world_ned.png"))
    ap.add_argument("--style", default=None, metavar="JSON",
                    help="样式文件(见 vrpsim/contracts/style.py);不传即默认样式")
    ap.add_argument("--dump-style", default=None, metavar="JSON",
                    help="把当前样式写成 JSON 模板后退出")
    ap.add_argument("--font-scale", type=float, default=1.0, help="所有字号统一乘该系数")
    ap.add_argument("--dpi", type=int, default=None, help="覆盖样式里的 dpi(默认 160)")
    args = ap.parse_args()

    style = load_style(args.style, font_scale=args.font_scale, dpi=args.dpi)
    if args.dump_style:
        print(f"样式模板已写出: {dump_style(args.dump_style, style)}")
        return

    if not os.path.isfile(args.world):
        raise SystemExit(f"找不到 {args.world};先跑 01_build_world.py")

    font = use_cjk_font(style.font_family)
    style.apply_rcparams()
    print(f"[1/3] 字体 {font or '(无 CJK 字体,标题可能出豆腐块)'}")

    mw = load_world(args.world)
    base = None if args.no_basemap else load_basemap(args.basemap)
    print(f"[2/3] 世界 {mw.field.shape}  底图 {'有' if base is not None else '无'}")

    # 三面板:概率场在全场尺度上极为峰化(>0.5 的 cell 只占 0.1%),
    # 单看总览等于看不见,故配一张"场本体"和一张簇内放大。
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 9.2),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.25],
                                          "wspace": 0.45})

    plot_world(mw, ax=axes[0], basemap=base, show_names=args.names,
               show_field=base is None, colorbar=False, style=style,
               title=f"(a) 地形 + 熱水噴出口 {mw.dataset.n}"
                     f"  ({mw.world.x_max_m:.0f}x{mw.world.y_max_m:.0f} m)")
    axes[0].legend(loc="lower right", framealpha=0.9, fontsize=style.font_size_legend)

    plot_world(mw, ax=axes[1], basemap=None, show_field=True, colorbar=True,
               style=style, title="(b) 目標分布確率マップ p(x,y)")

    # 放大窗:取目标最密的 80 m 见方(Faulty Towers / Twin Peaks 复合体)。
    # 边界夹到场地内,免得画出一块空白的"场外"。
    half = 40.0
    zc = _densest_center(mw, half_m=half)
    n0, n1 = _clamp(zc[0] - half, zc[0] + half, 0.0, mw.world.x_max_m)
    e0, e1 = _clamp(zc[1] - half, zc[1] + half, 0.0, mw.world.y_max_m)
    plot_world(mw, ax=axes[2], basemap=None, show_field=True, colorbar=False,
               marker_scale=2.2, show_names=True, bounds=(n0, n1, e0, e1), style=style,
               title=f"(c) 拡大 {n1 - n0:.0f}x{e1 - e0:.0f} m @ N{zc[0]:.0f} E{zc[1]:.0f}")

    for a in axes[1:]:
        a.set_ylabel("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=style.dpi, bbox_inches="tight", facecolor=style.facecolor)
    plt.close(fig)
    print(f"[3/3] {args.out}  ({os.path.getsize(args.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
