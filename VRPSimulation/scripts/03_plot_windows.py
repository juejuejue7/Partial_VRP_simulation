#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""03 — 窗口尺寸诊断图:把 100x100 m 滑动窗口摊在真实目标分布上看。

输入 : VRPSimulation/data/mothra_world.npz  (由 01 生成,已裁切)
       VRPSimulation/data/mothra_basemap.npz(可选底色)
输出 : VRPSimulation/figures/mothra_windows.png

三面板:
  (a) 裁切后的世界 + Leader 测线 + 几个代表性窗口位置
  (b) 每次重解时窗口内的目标数(判断负载是否均衡)
  (c) 目标进入/离开窗口的时序甘特(判断有没有目标只在一次重解里露过面)

人工检查项:(c) 里每个目标至少有一段;若某目标只覆盖 1 次重解,说明它一进窗口就
必须被立刻分配,否则永久错过 —— 那是推进阈值取太大的信号。

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/03_plot_windows.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vrpsim.contracts.config import (DEFAULT_DATA_DIR, DEFAULT_FIGURE_DIR,  # noqa: E402
                                     MothraSimConfig)
from vrpsim.contracts.style import dump_style, load_style  # noqa: E402
from vrpsim.viz import (load_basemap, plot_leader_track, plot_window,  # noqa: E402
                        plot_world, use_cjk_font)
from vrpsim.windows import (enumerate_windows, first_seen_window, leader_track,  # noqa: E402
                            window_occupancy)
from vrpsim.world import load_world  # noqa: E402

_WIN_COLORS = ["#2aa198", "#3987e5", "#b48ead", "#eb6834"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=os.path.join(DEFAULT_DATA_DIR, "mothra_world.npz"))
    ap.add_argument("--basemap", default=os.path.join(DEFAULT_DATA_DIR, "mothra_basemap.npz"))
    ap.add_argument("--advance-m", type=float, default=50.0)
    ap.add_argument("--show-windows", type=int, default=4,
                    help="面板 (a) 里画几个代表性窗口")
    ap.add_argument("--out", default=os.path.join(DEFAULT_FIGURE_DIR, "mothra_windows.png"))
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

    use_cjk_font(style.font_family)
    style.apply_rcparams()
    mw = load_world(args.world)
    ds = mw.dataset
    cfg = MothraSimConfig(window_advance_threshold_m=args.advance_m)
    track = leader_track(mw.world, cfg.lane_spacing_m)
    snaps = enumerate_windows(mw.world, cfg.window, ds.targets_ned)
    occ = window_occupancy(snaps)
    print(f"[1/3] 世界 {mw.field.shape}  重解 {len(snaps)} 次  窗口内目标 "
          f"min {occ.min()} / max {occ.max()}")

    fig = plt.figure(figsize=(15.0, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], hspace=0.32, wspace=0.28)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_occ = fig.add_subplot(gs[0, 1])
    ax_gantt = fig.add_subplot(gs[1, 1])

    # --- (a) 地图 + 测线 + 代表性窗口 --------------------------------
    base = load_basemap(args.basemap)
    plot_world(mw, ax=ax_map, basemap=base, show_field=base is None, colorbar=False,
               style=style,
               title=f"(a) 裁切後 {mw.world.x_max_m:.0f}x{mw.world.y_max_m:.0f} m"
                     f" + Leader 測線 + 窗口 "
                     f"{cfg.window.look_back_m:.0f}x{cfg.window.width_m:.0f} m")
    plot_leader_track(ax_map, track, linewidth=style.overview_track_linewidth)
    pick = np.linspace(0, len(snaps) - 1, args.show_windows).round().astype(int)
    for j, i in enumerate(pick):
        s = snaps[int(i)]
        a, b = s.north_span
        plot_window(ax_map, s, color=_WIN_COLORS[j % len(_WIN_COLORS)],
                    linewidth=style.overview_window_linewidth,
                    alpha=style.window_face_alpha,
                    label=f"#{s.index} N[{a:.0f},{b:.0f}] ({len(s.target_idx)})")
    ax_map.legend(loc="lower right", framealpha=0.9, fontsize=style.font_size_legend)

    # --- (b) 每次重解的窗口负载 --------------------------------------
    x = np.arange(len(snaps))
    bars = ax_occ.bar(x, occ, color="#3987e5", edgecolor="#1a1a19", linewidth=0.6)
    for xi, o in zip(x, occ):
        if o == 0:
            bars[xi].set_color("#c3c2b7")
    ax_occ.axhline(float(occ.mean()), color="#eb6834", linestyle="--", linewidth=1.2,
                   label=f"平均 {occ.mean():.1f}")
    ax_occ.set_xticks(x)
    ax_occ.set_xticklabels([f"#{s.index}\nN{s.leader_north_m:.0f}" for s in snaps],
                           fontsize=style.font_size_tick_snapshot)
    ax_occ.set_ylabel("窗口内目标数")
    ax_occ.set_title(f"(b) 各次重解的窗口负载(推进阈值 {cfg.window.advance_threshold_m:.0f} m)")
    ax_occ.legend(fontsize=style.font_size_legend)
    ax_occ.grid(axis="y", alpha=0.3)

    # --- (c) 目标 x 重解 的可见性甘特 --------------------------------
    vis = np.zeros((ds.n, len(snaps)), dtype=bool)
    for s in snaps:
        vis[s.target_idx, s.index] = True
    ax_gantt.imshow(vis, aspect="auto", origin="lower", cmap="Blues",
                    interpolation="nearest", vmin=0, vmax=1.4)
    n_windows_seen = vis.sum(axis=1)
    ax_gantt.set_yticks(np.arange(ds.n))
    ax_gantt.set_yticklabels([f"wp{int(w)}" for w in ds.waypoint_id],
                             fontsize=style.font_size_small)
    ax_gantt.set_xticks(x)
    ax_gantt.set_xticklabels([f"#{i}" for i in x], fontsize=style.font_size_tick_snapshot)
    ax_gantt.set_xlabel("重解序号")
    ax_gantt.set_title(f"(c) 目标在窗口内的可见跨度(最少 {n_windows_seen.min()} 次 / "
                       f"最多 {n_windows_seen.max()} 次)")
    for i in range(ds.n):
        ax_gantt.text(len(snaps) - 0.4, i, str(int(n_windows_seen[i])),
                      va="center", fontsize=style.font_size_small, color="#52514e")

    first = first_seen_window(snaps, ds.n)
    never = ds.waypoint_id[first < 0].tolist()
    print(f"[2/3] 每个目标可见 {n_windows_seen.min()}~{n_windows_seen.max()} 次"
          + (f"  ⚠ 从未进窗 {never}" if never else "  (无遗漏)"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=style.dpi, bbox_inches="tight", facecolor=style.facecolor)
    plt.close(fig)
    print(f"[3/3] {args.out}  ({os.path.getsize(args.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
