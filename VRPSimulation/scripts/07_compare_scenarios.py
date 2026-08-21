#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""07 — 两个场景的对比:協調探査(VRP) vs Lawnmower 全覆盖基线(D14)。

输入 : VRPSimulation/data/mission.npz + mission_assignments.json   (04 产出)
       VRPSimulation/data/lawnmower.npz + lawnmower_summary.json   (06 产出)
输出 : VRPSimulation/figures/scenario_compare.png   覆盖進捗曲線 + 航程对比
       VRPSimulation/figures/lawnmower_paths.png    基线的 3 条 lawnmower 航迹

⚠ 比的是**任务完成时刻** `t_complete_s`,不是 `duration_s`:
    VRP       = max(Leader 走完测线, 最后一个目标完成観測)
    lawnmower = 最后一台车走完自己全部测线
  两个定义都在 `contracts/lawnmower.py` 的"完成时刻的口径"里,本脚本**不重算**,
  直接读各自 summary 里由 `MissionResult` / `LawnmowerResult` 算好的值 ——
  免得同一个量在两处各有一份实现,哪天改了口径只改一处。

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/07_compare_scenarios.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vrpsim.contracts.config import DEFAULT_DATA_DIR, DEFAULT_FIGURE_DIR  # noqa: E402
from vrpsim.contracts.style import dump_style, load_style  # noqa: E402
from vrpsim.viz import load_basemap, plot_world, use_cjk_font  # noqa: E402
from vrpsim.world import load_world  # noqa: E402

VRP_LABEL = "協調探査 (1 Leader + 2 Follower, VRP)"
LAWN_LABEL = "全覆盖基线 (3 Follower, lawnmower)"
VRP_COLOR = "#2aa198"
LAWN_COLOR = "#c1440e"


def _load_summary(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)["summary"]


def _coverage_curve(visit_time_s: np.ndarray, t_end: float) -> tuple:
    """観測完成时刻 → 阶梯曲线 (t, 累计観測数)。未観測的点不进曲线。

    末端补一段到 `t_end`(= 任务完成时刻)的水平线:lawnmower 在最后一个目标
    被観測之后还要继续扫完剩余测线,那段"什么也没找到但仍在花时间"必须画出来,
    否则曲线会看起来比实际早结束。
    """
    v = np.sort(visit_time_s[~np.isnan(visit_time_s)])
    t = np.concatenate([[0.0], v, [max(float(t_end), float(v[-1]) if v.size else 0.0)]])
    c = np.concatenate([np.arange(v.size + 1), [v.size]])
    return t, c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default=os.path.join(DEFAULT_DATA_DIR, "mission.npz"))
    ap.add_argument("--lawnmower", default=os.path.join(DEFAULT_DATA_DIR, "lawnmower.npz"))
    ap.add_argument("--world", default=os.path.join(DEFAULT_DATA_DIR, "mothra_world.npz"))
    ap.add_argument("--basemap", default=os.path.join(DEFAULT_DATA_DIR, "mothra_basemap.npz"))
    ap.add_argument("--out", default=os.path.join(DEFAULT_FIGURE_DIR, "scenario_compare.png"))
    ap.add_argument("--out-paths", default=os.path.join(DEFAULT_FIGURE_DIR,
                                                        "lawnmower_paths.png"))
    ap.add_argument("--no-paths", action="store_true", help="不出 lawnmower 航迹图")
    ap.add_argument("--style", default=None, metavar="JSON")
    ap.add_argument("--dump-style", default=None, metavar="JSON")
    ap.add_argument("--font-scale", type=float, default=1.0)
    ap.add_argument("--dpi", type=int, default=None)
    args = ap.parse_args()

    style = load_style(args.style, font_scale=args.font_scale, dpi=args.dpi)
    if args.dump_style:
        print(f"样式模板已写出: {dump_style(args.dump_style, style)}")
        return

    m_json = os.path.splitext(args.mission)[0] + "_assignments.json"
    l_json = os.path.splitext(args.lawnmower)[0] + "_summary.json"
    for p in (args.mission, m_json, args.lawnmower, l_json):
        if not os.path.isfile(p):
            raise SystemExit(f"找不到 {p};先跑 04_run_mission.py 与 06_run_lawnmower.py")

    use_cjk_font(style.font_family)
    style.apply_rcparams()

    m_sum, l_sum = _load_summary(m_json), _load_summary(l_json)
    with np.load(args.mission, allow_pickle=False) as d:
        m_visit = np.asarray(d["visit_time_s"], dtype=float)
        m_dist = np.asarray(d["follower_distance_m"], dtype=float)[-1]
    lawn = np.load(args.lawnmower, allow_pickle=False)
    l_visit = np.asarray(lawn["visit_time_s"], dtype=float)
    l_dist = np.asarray(lawn["vehicle_distance_m"], dtype=float)[-1]
    l_pos = np.asarray(lawn["vehicle_pos"], dtype=float)
    l_meta = json.loads(str(lawn["meta_json"]))
    lawn.close()

    # ---- 文字对比 ----------------------------------------------------
    tv, tl = m_sum["t_complete_s"], l_sum["t_complete_s"]
    print("=" * 74)
    print(f"{'':30s}{'協調探査 (VRP)':>20s}{'全覆盖 (lawnmower)':>22s}")
    print("-" * 74)
    rows = [
        ("AUV 台数", f"1 Leader + 2 Follower", f"{l_sum['n_vehicles']} Follower"),
        ("観測幅宽 [m]", "声呐窗口 100 / 相机点観測",
         f"相机 {l_sum['swath_width_m']:.1f}"),
        ("**任务完成时刻 [s]**", f"{tv:.1f}", f"{tl:.1f}"),
        ("  全域扫遍于 [s]", f"{m_sum['t_leader_finish_s']:.1f}",
         f"{tl:.1f}"),
        ("  最后一个目标観測于 [s]", f"{m_sum['t_last_observation_s']:.1f}",
         f"{l_sum['t_last_observation_s']:.1f}"),
        ("覆盖", f"{m_sum['visited']}/{m_sum['n_targets']}",
         f"{l_sum['visited']}/{l_sum['n_targets']}"),
        ("最长单机航程 [m]", f"{m_sum['max_distance_m']:.1f}",
         f"{l_sum['max_distance_m']:.1f}"),
        ("总航程 [m]", f"{m_sum['total_distance_m']:.1f}",
         f"{l_sum['total_distance_m']:.1f}"),
        ("Jain 均衡度", f"{m_sum['jain_fairness']:.3f}", f"{l_sum['jain_fairness']:.3f}"),
    ]
    for k, a, b in rows:
        print(f"{k:30s}{a:>20s}{b:>22s}")
    print("-" * 74)
    # `t_complete_s` 在未全覆盖时按定义是 nan(漏了目标的"完成时刻"没有意义)。
    # 不要拿 nan 去算比值或喂 set_xlim —— 那会抛 "Axis limits cannot be NaN"。
    comparable = bool(np.isfinite(tv) and np.isfinite(tl))
    if comparable:
        print(f"完成时刻比  lawnmower / VRP = {tl / tv:.2f} x"
              f"     缩短 {100 * (1 - tv / tl):.1f} %")
    else:
        who = [n for n, s in (("協調探査(VRP)", m_sum), ("全覆盖(lawnmower)", l_sum))
               if s["visited"] < s["n_targets"]]
        print(f"⚠ {' 与 '.join(who)} 未达成全覆盖 ⇒ 任务未完成,"
              f"跨场景时间**不可比**(t_complete_s = nan)。")
        print("  时间面板跳过;先把覆盖率修到 100% 再来比时间。")
    print(f"总航程比    lawnmower / VRP = "
          f"{l_sum['total_distance_m'] / m_sum['total_distance_m']:.2f} x")
    print("=" * 74)

    # ---- 图 1:覆盖進捗 + 航程 ----------------------------------------
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                  gridspec_kw={"width_ratios": [2.15, 1.0]})

    n_t = int(m_sum["n_targets"])
    # 未全覆盖时没有"完成时刻",曲线末端退回到各自最后一次観測的时刻。
    tv_end = tv if np.isfinite(tv) else float(np.nanmax(m_visit))
    tl_end = tl if np.isfinite(tl) else float(np.nanmax(l_visit))
    for (t, c), col, lab in ((_coverage_curve(m_visit, tv_end), VRP_COLOR, VRP_LABEL),
                             (_coverage_curve(l_visit, tl_end), LAWN_COLOR, LAWN_LABEL)):
        ax.step(t, c, where="post", color=col, linewidth=2.2, label=lab, zorder=3)
    if comparable:
        for tc, col, txt, ha in ((tv, VRP_COLOR, f"完了 {tv:.0f} s", "left"),
                                 (tl, LAWN_COLOR, f"完了 {tl:.0f} s", "right")):
            ax.axvline(tc, color=col, linewidth=1.2, linestyle="--", alpha=0.9, zorder=2)
            ax.annotate(txt, (tc, n_t + 0.35),
                        xytext=(4 if ha == "left" else -4, 0),
                        textcoords="offset points", color=col,
                        fontsize=style.font_size_annotation, ha=ha, va="bottom")
    else:
        ax.annotate("未全覆盖 ⇒ 完成时刻未定义", (0.5, n_t + 0.35),
                    xycoords=("axes fraction", "data"), color="#c1440e",
                    fontsize=style.font_size_annotation, ha="center", va="bottom")
    ax.axhline(n_t, color="#898781", linewidth=0.7, alpha=0.6, zorder=1)
    ax.set_xlim(0.0, max(tv_end, tl_end) * 1.06)
    ax.set_ylim(0.0, n_t + 2.6)
    ax.set_xlabel("時間 t [s]", fontsize=style.font_size_label)
    ax.set_ylabel("観測完了 目標数", fontsize=style.font_size_label)
    ax.tick_params(labelsize=style.font_size_tick)
    ax.grid(alpha=style.panel_b_grid_alpha, linewidth=style.panel_b_grid_linewidth)
    ax.legend(fontsize=style.font_size_legend, loc="lower right", framealpha=0.92)
    ax.set_title(f"(a) 観測進捗と任務完了時刻  —  {tl / tv:.2f}x",
                 fontsize=style.font_size_title)

    # 航程:两组各自按机体排开
    xs_m = np.arange(len(m_dist))
    xs_l = np.arange(len(l_dist)) + len(m_dist) + 0.8
    axb.bar(xs_m, m_dist, color=VRP_COLOR, width=0.72,
            edgecolor=style.marker_edge_color, linewidth=0.6)
    axb.bar(xs_l, l_dist, color=LAWN_COLOR, width=0.72,
            edgecolor=style.marker_edge_color, linewidth=0.6)
    axb.set_xticks(list(xs_m) + list(xs_l))
    axb.set_xticklabels([f"F{i}" for i in range(len(m_dist))]
                        + [f"V{i}" for i in range(len(l_dist))],
                        fontsize=style.font_size_tick_panel_b)
    axb.set_ylabel("航程 [m]", fontsize=style.font_size_label)
    axb.tick_params(axis="y", labelsize=style.font_size_tick)
    axb.grid(axis="y", alpha=style.panel_b_grid_alpha,
             linewidth=style.panel_b_grid_linewidth)
    axb.set_title("(b) 機体別 航程", fontsize=style.font_size_title)
    for x, v in zip(list(xs_m) + list(xs_l), list(m_dist) + list(l_dist)):
        axb.annotate(f"{v:.0f}", (x, v), xytext=(0, 2), textcoords="offset points",
                     ha="center", fontsize=style.font_size_small)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=style.dpi, facecolor=style.facecolor)
    plt.close(fig)
    print(f"  {args.out}  ({os.path.getsize(args.out) / 1024:.1f} KB)")

    # ---- 图 2:lawnmower 航迹 -----------------------------------------
    if args.no_paths:
        return
    if not os.path.isfile(args.world):
        print("  (跳过航迹图:找不到 mothra_world.npz)")
        return
    mw = load_world(args.world)
    base = load_basemap(args.basemap)
    h = 8.5
    fig2, ax2 = plt.subplots(figsize=(max(3.4, h * mw.world.y_max_m / mw.world.x_max_m
                                          + 1.4), h))
    plot_world(mw, ax=ax2, basemap=base, show_field=False, colorbar=False,
               title="", style=style)
    from matplotlib.patheffects import withStroke
    halo = [withStroke(linewidth=style.traj_linewidth + style.traj_halo_linewidth,
                       foreground=style.traj_halo_color)]
    for i in range(l_pos.shape[1]):
        ax2.plot(l_pos[:, i, 1], l_pos[:, i, 0], "-",
                 color=style.follower_color(i), linewidth=style.traj_linewidth,
                 zorder=4, path_effects=halo,
                 label=f"V{i}  {len(l_meta['lane_easts'][i])} 測線 / "
                       f"{l_dist[i]:.0f} m")
    for lo, hi in l_meta["strips_east_m"][1:]:
        ax2.axvline(lo, color="#ffffff", linewidth=0.8, linestyle=":", alpha=0.8, zorder=3)
    ax2.legend(fontsize=style.font_size_legend, loc="lower left", framealpha=0.92)
    ax2.set_title(f"Lawnmower 全覆盖基线\n幅宽 {l_meta['swath_width_m']:.0f} m / "
                  f"測線間隔 {l_meta['lane_spacing_m']:.2f} m",
                  fontsize=style.font_size_title)
    fig2.tight_layout()
    fig2.savefig(args.out_paths, dpi=style.dpi, facecolor=style.facecolor)
    plt.close(fig2)
    print(f"  {args.out_paths}  ({os.path.getsize(args.out_paths) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
