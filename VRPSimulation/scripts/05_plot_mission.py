#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05 — 任务仿真结果出图:时序快照 (a) + North-时间 (b)。

输入(统一在 VRPSimulation/data/scenarios/<id>/, <id> 含 mothra 本身):
       mission.npz(由 04 生成)、world.npz / basemap.npz(由 01 生成)
输出 : VRPSimulation/figures/scenarios/<id>/mission_overview.png

版式(人工规格 2026-08-09):
  上排 = 按固定时间间隔(默认 100 s)的一组快照,展示任务如何推进;
  下排 = North-时间图,**x 轴与上排逐帧对齐**(第 i 帧的水平中心对准 t_i)。

  **只画图像和坐标轴**:无图例、无标题、无轴标签、无 colorbar、无文字标注。

快照里的目标点三态(见 vrpsim/viz.py::plot_mission_snapshot):
  未被 Leader 窗口扫过 → 不画;已扫未观测 → 空心;已观测 → 实心。

样式(字号/线宽/配色/版面)全部来自 `vrpsim/contracts/style.py`,不写死在本文件:
  --dump-style my_style.json    先导出一份带全部字段的模板
  (手改 JSON 里的数字)
  --style my_style.json         用它重新出图(不需要重跑 04)
  --font-scale 1.4              懒人档:所有字号乘 1.4

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/05_plot_mission.py
      D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/05_plot_mission.py --scenario sparse_2
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vrpsim.contracts.config import DEFAULT_DATA_DIR, DEFAULT_FIGURE_DIR  # noqa: E402
from vrpsim.contracts.style import dump_style, load_style  # noqa: E402
from vrpsim.mission import load_result  # noqa: E402
from vrpsim.viz import (load_basemap, plot_mission_snapshot,  # noqa: E402
                        sweep_times, use_cjk_font)
from vrpsim.world import load_world  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mothra",
                    help="Bethmetory_data_process/scenarios.json 里的场景 id "
                         "(默认 mothra); 决定 --mission/--world/--basemap/--out 的默认路径")
    ap.add_argument("--mission", default=None)
    ap.add_argument("--world", default=None)
    ap.add_argument("--basemap", default=None)
    ap.add_argument("--snapshot-interval", type=float, default=100.0,
                    help="上排快照的时间间隔 [s]")
    ap.add_argument("--no-time-guides", action="store_true",
                    help="不在 (b) 上画各快照时刻的竖直参考线")
    ap.add_argument("--out", default=None)
    # ---- 样式 ----
    ap.add_argument("--style", default=None, metavar="JSON",
                    help="样式文件;不传则用 contracts/style.py 的默认样式")
    ap.add_argument("--dump-style", default=None, metavar="JSON",
                    help="把当前样式写成 JSON 模板后退出(改完再 --style 传回)")
    ap.add_argument("--font-scale", type=float, default=3.0,
                    help="所有字号统一乘这个系数(不改 JSON 时的快捷档)")
    ap.add_argument("--dpi", type=int, default=None,
                    help="覆盖样式里的 dpi(默认 160)")
    ap.add_argument("--fig-width", type=float, default=None,
                    help="覆盖图宽 [in](图高由快照的等比例需求自动反推)")
    args = ap.parse_args()

    scen_data_dir = os.path.join(DEFAULT_DATA_DIR, "scenarios", args.scenario)
    mission_p = args.mission or os.path.join(scen_data_dir, "mission.npz")
    world_p = args.world or os.path.join(scen_data_dir, "world.npz")
    basemap_p = args.basemap or os.path.join(scen_data_dir, "basemap.npz")
    out_p = args.out or os.path.join(DEFAULT_FIGURE_DIR, "scenarios", args.scenario,
                                     "mission_overview.png")

    style = load_style(args.style, font_scale=args.font_scale,
                       dpi=args.dpi, fig_width_in=args.fig_width)
    if args.dump_style:
        print(f"样式模板已写出: {dump_style(args.dump_style, style)}")
        print("改完其中的数值后用  --style " + args.dump_style + "  重新出图。")
        return

    for p in (mission_p, world_p):
        if not os.path.isfile(p):
            raise SystemExit(f"找不到 {p};先跑 01_build_world.py 与 04_run_mission.py"
                             f"(都带 --scenario {args.scenario})")

    use_cjk_font(style.font_family)
    style.apply_rcparams()
    d = load_result(mission_p)
    mw = load_world(world_p)
    base = load_basemap(basemap_p)

    t = np.asarray(d["t_s"], dtype=float)
    t_end = float(t[-1])
    dt_snap = float(args.snapshot_interval)
    snap_t = np.arange(0.0, t_end + 1e-9, dt_snap)
    # snap_t = [100,200,400,500]
    n = len(snap_t)

    swept = sweep_times(mw, d)
    visit = np.asarray(d["visit_time_s"], dtype=float)
    print(f"[1/3] 时间线 {len(t)} 拍 / {t_end:.0f} s,快照 {n} 帧 @ {dt_snap:.0f} s"
          f"   样式 {args.style or '(默认)'}"
          + (f" x{args.font_scale:g} 字号" if args.font_scale != 1.0 else ""))
    for ts in snap_t:
        print(f"      t={ts:6.0f} s  已揭示 {int(np.nansum(swept <= ts)):2d}"
              f"  已观测 {int(np.nansum(visit <= ts)):2d} / {len(visit)}")

    # ---- 时间轴 → 图幅 x 的线性映射(对齐的全部依据) -----------------
    # xlim 取 [t_0 - Δ/2, max(t_end, t_last + Δ/2)] ⇒ 第 0 帧左边缘恰好压在绘图区左端。
    t_lo = snap_t[0] - dt_snap / 2.0
    t_hi = max(t_end, snap_t[-1] + dt_snap / 2.0)
    W = style.margin_right - style.margin_left
    sw = W * dt_snap / (t_hi - t_lo)                    # 快照宽(图幅分数)

    def t_to_figx(tv: float) -> float:
        return style.margin_left + W * (float(tv) - t_lo) / (t_hi - t_lo)

    # 相邻帧中心正好相隔 sw ⇒ 直接用 sw 当宽度会让六帧边缘相接、看着像一张连续长图。
    # 按 frame_fill 收窄留缝,**中心不动**,所以对齐不受影响。
    sw_draw = sw * style.frame_fill

    # 快照高度:让 axes 框本身就是等比例,这样 set_aspect("equal") 不会在框内再缩放
    # (一缩放水平中心就偏,对齐即失效)。
    aspect = mw.world.x_max_m / mw.world.y_max_m        # North/East = 5
    sw_in = sw_draw * style.fig_width_in
    sh_in = sw_in * aspect
    fig_h_in = (style.margin_bottom_in + style.panel_b_height_in + style.row_gap_in
                + sh_in + style.margin_top_in)

    fig = plt.figure(figsize=(style.fig_width_in, fig_h_in))
    b_bottom = style.margin_bottom_in / fig_h_in
    b_height = style.panel_b_height_in / fig_h_in
    s_bottom = (style.margin_bottom_in + style.panel_b_height_in
                + style.row_gap_in) / fig_h_in
    s_height = sh_in / fig_h_in

    # # ---- 上排:快照 ---------------------------------------------------
    for i, ts in enumerate(snap_t):
        cx = t_to_figx(ts)
        ax = fig.add_axes([cx - sw_draw / 2.0, s_bottom, sw_draw, s_height])
        plot_mission_snapshot(ax, mw, d, float(ts), basemap=base, swept=swept,
                              style=style)
        # 六帧共用同一套坐标范围,刻度**线**都保留(坐标轴要在),
        # 但刻度**数字**只在最左一帧标一次 —— 每帧都标会互相撞成一片。
        # 三个刻度按该场景自己的 East 跨度均分(Mothra East=100 时退化为 [0,50,100])。
        ax.set_xticks(np.linspace(0.0, mw.world.y_max_m, 3))
        if i > 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.tick_params(labelsize=style.font_size_tick_snapshot)

    # ---- 下排:North-时间,x 轴与上排同一线性映射 ---------------------
    axb = fig.add_axes([style.margin_left, b_bottom, W, b_height])
    # axb = fig.add_axes([style.margin_left, s_bottom, W, s_height])
    axb.fill_between(t, d["window_north"][:, 0], d["window_north"][:, 1],
                     color=style.leader_color, alpha=style.panel_b_window_alpha,
                     linewidth=0)
    axb.plot(t, d["leader_north_m"], "-", color=style.leader_color,
             linewidth=style.panel_b_leader_linewidth)
    pos = np.asarray(d["follower_pos"], dtype=float)
    for i in range(pos.shape[1]):
        axb.plot(t, pos[:, i, 0], "-", color=style.follower_color(i),
                 linewidth=style.panel_b_follower_linewidth)
    ok = ~np.isnan(visit)
    axb.scatter(visit[ok], mw.dataset.targets_ned[ok, 0],
                s=style.panel_b_visit_marker_size, facecolor=style.visit_color,
                edgecolor=style.marker_edge_color,
                linewidth=style.panel_b_visit_edge_width, zorder=6)
    if not args.no_time_guides:
        for ts in snap_t:
            axb.axvline(float(ts), color=style.time_guide_color,
                        linewidth=style.time_guide_linewidth,
                        alpha=style.time_guide_alpha, zorder=1)
    axb.set_xlim(t_lo, t_hi)
    axb.set_ylim(0.0, mw.world.x_max_m)
    axb.set_xticks(snap_t)
    axb.tick_params(labelsize=style.font_size_tick_panel_b)
    axb.grid(alpha=style.panel_b_grid_alpha, linewidth=style.panel_b_grid_linewidth)

    print("[2/3] 绘制完成(无图例 / 无标题 / 无轴标签)")
    os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
    fig.savefig(out_p, dpi=style.dpi, facecolor=style.facecolor)
    plt.close(fig)
    print(f"[3/3] {out_p}  ({os.path.getsize(out_p) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
