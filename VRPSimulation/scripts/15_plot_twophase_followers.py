#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""15 — 二段式全局VRP 基线 (D19) 的 **Follower 轨迹图**,七场景各出一张。

输入 : VRPSimulation/data/scenarios/<id>/twophase.npz + twophase_summary.json
                                        (由 09_compare_methods.py 生成)
       VRPSimulation/data/scenarios/<id>/{world,basemap}.npz  (由 01 生成)
输出 : VRPSimulation/figures/scenarios/<id>/twophase_followers.png

为什么快照只覆盖阶段2,而不是整条任务时间轴
================================================================================
二段式是**串行**的:阶段1 Leader 独自走完整条测线,两台 Follower 在起点原地待机
(STATUS_IDLE),一米都不走。阶段1 占整条任务时间轴的比重实测是

    sparse_3 84% / sparse_2 80% / mef 66% / high_rise 56% / dense_2 54%
    / dense_1 53% / mothra 48%

⇒ 若按整条时间轴等分取帧,最多有 84% 的帧画的是「两台车停在起点」的同一张图。
所以本图的取帧区间是 **[t_survey_s, t_finish_s]**,即阶段2 的净时长:
第 0 帧 = 路线刚下发、一个目标都没确认;最后一帧 = 全部确认完毕。
`t_survey_s` 取自 `twophase_summary.json`(二段式的 npz 里没有这一列)。

⚠ 取帧区间的下界不能再往前挪:阶段2 之前目标尚未被 Leader 探明,画出来就等于
  泄露 ground truth(口径见 `viz.plot_twophase_snapshot` 的第 2 条)。

版式
--------------------------------------------------------------------------------
只有一排快照 —— 人工指定「现在只需要 Follower 的轨迹图」,不带 05 的
North-时间 面板。帧宽的自适应算法与 12_plot_all_scenarios.py 同一套(先定图高
上限、再反推帧宽,与帧宽上限取小者),所以本图的单帧与提案手法的
`mission_overview.png` 尺度接近,两张图并排看不会一大一小。

时刻标签是本图**唯一**的文字(ASCII,不含中日文字 —— 会议图口径)。没有下排
时间面板就没有别的时间参照物了,故默认打开;`--no-time-labels` 可关。

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe \
      VRPSimulation/scripts/15_plot_twophase_followers.py
      (只出某几个场景) ... 15_plot_twophase_followers.py --scenarios mothra dense_1
      (不垫开环预定路线) ... --no-planned-route
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
from matplotlib.patheffects import withStroke  # noqa: E402

from vrpsim.contracts.config import DEFAULT_DATA_DIR, DEFAULT_FIGURE_DIR  # noqa: E402
from vrpsim.contracts.style import load_style  # noqa: E402
from vrpsim.twophase import load_twophase_result  # noqa: E402
from vrpsim.viz import load_basemap, plot_twophase_snapshot, use_cjk_font  # noqa: E402
from vrpsim.world import load_world  # noqa: E402

SIDS = ("mothra", "mef", "high_rise", "sparse_2", "sparse_3", "dense_1", "dense_2")

# 与 12_plot_all_scenarios.py 同一基准版式(目视确认 2026-08-25):
# fig_width=22 in 下 font_scale=3.0 观感合适,图宽一变字号按同比例跟随。
BASE_FIG_W_IN, BASE_FONT_SCALE = 22.0, 3.0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=list(SIDS),
                    help=f"要出图的场景 id(默认全部 {len(SIDS)} 个)")
    ap.add_argument("--frames", type=int, default=6,
                    help="每张图的快照帧数;帧数越多每帧越窄")
    ap.add_argument("--frame-width-max", type=float, default=4.2,
                    help="单帧宽度上限 [in]。再宽收益递减,文件却继续涨")
    ap.add_argument("--fig-height-max", type=float, default=20.2,
                    help="整图高度上限 [in]。长宽比大的场景(mothra 5:1)会顶到这里")
    ap.add_argument("--dpi", type=int, default=230)
    ap.add_argument("--no-planned-route", action="store_true",
                    help="不垫阶段2 开环预定路线的虚线")
    ap.add_argument("--no-time-labels", action="store_true",
                    help="不在每帧左上角标该帧时刻")
    ap.add_argument("--style", default=None, metavar="JSON")
    ap.add_argument("--dry-run", action="store_true", help="只打印版式,不出图")
    return ap


def plot_one(sid: str, args, *, fig_w: float, font: float) -> str:
    d_dir = os.path.join(DEFAULT_DATA_DIR, "scenarios", sid)
    out_p = os.path.join(DEFAULT_FIGURE_DIR, "scenarios", sid,
                         "twophase_followers.png")

    d = load_twophase_result(os.path.join(d_dir, "twophase.npz"))
    with open(os.path.join(d_dir, "twophase_summary.json"), encoding="utf-8") as f:
        J = json.load(f)
    routes = None if args.no_planned_route else J.get("routes")
    mw = load_world(os.path.join(d_dir, "world.npz"))
    base = load_basemap(os.path.join(d_dir, "basemap.npz"))

    style = load_style(args.style, font_scale=font, dpi=args.dpi, fig_width_in=fig_w)
    use_cjk_font(style.font_family)
    style.apply_rcparams()

    t_survey = float(J["t_survey_s"])
    t_fin = float(J["summary"]["t_finish_s"])
    n = max(1, int(args.frames))
    snap_t = np.linspace(t_survey, t_fin, n)

    # 一排快照等分绘图区;帧宽按 frame_fill 收窄留缝(中心不动)
    W = style.margin_right - style.margin_left
    sw = W / n
    sw_draw = sw * style.frame_fill
    aspect = mw.world.x_max_m / mw.world.y_max_m
    sh_in = sw_draw * style.fig_width_in * aspect
    fig_h_in = style.margin_bottom_in + sh_in + style.margin_top_in

    fig = plt.figure(figsize=(style.fig_width_in, fig_h_in))
    s_bottom = style.margin_bottom_in / fig_h_in
    s_height = sh_in / fig_h_in

    for i, ts in enumerate(snap_t):
        cx = style.margin_left + sw * (i + 0.5)
        ax = fig.add_axes([cx - sw_draw / 2.0, s_bottom, sw_draw, s_height])
        plot_twophase_snapshot(ax, mw, d, float(ts), basemap=base, style=style,
                               routes=routes)
        # 刻度**线**每帧都留(坐标轴要在),刻度**数字**只标最左一帧 ——
        # 每帧都标会互相撞成一片。三个刻度按该场景自己的 East 跨度均分。
        ax.set_xticks(np.linspace(0.0, mw.world.y_max_m, 3))
        if i > 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.tick_params(labelsize=style.font_size_tick_snapshot)
        if not args.no_time_labels:
            ax.text(0.03, 0.985, f"t = {ts:.0f} s", transform=ax.transAxes,
                    ha="left", va="top", color="#ffffff",
                    fontsize=style.font_size_tick_snapshot * 1.15, zorder=9,
                    path_effects=[withStroke(linewidth=2.2, foreground="#1a1a19")])

    os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
    fig.savefig(out_p, dpi=style.dpi, facecolor=style.facecolor)
    plt.close(fig)
    return out_p


def main() -> None:
    args = build_parser().parse_args()

    s = load_style(args.style)
    frame_frac = (s.margin_right - s.margin_left) / max(1, args.frames) * s.frame_fill
    fixed_h = s.margin_bottom_in + s.margin_top_in

    print(f"{'场景':>10} {'长宽比':>7} {'阶段1(s)':>9} {'阶段2(s)':>9} "
          f"{'图宽(in)':>9} {'图高(in)':>9} {'帧宽(px)':>9} {'字号x':>6}")
    jobs = []
    for sid in args.scenarios:
        d_dir = os.path.join(DEFAULT_DATA_DIR, "scenarios", sid)
        need = [os.path.join(d_dir, "twophase.npz"),
                os.path.join(d_dir, "twophase_summary.json"),
                os.path.join(d_dir, "world.npz")]
        for p in need:
            if not os.path.isfile(p):
                raise SystemExit(f"找不到 {p};先跑 01_build_world.py 与 "
                                 f"09_compare_methods.py --scenario {sid}")
        with open(need[1], encoding="utf-8") as f:
            J = json.load(f)
        mw = load_world(need[2])
        aspect = mw.world.x_max_m / mw.world.y_max_m

        # 两个约束取小者:帧宽上限、以及「图高不超过上限」反推出的帧宽
        frame_w = min(args.frame_width_max, (args.fig_height_max - fixed_h) / aspect)
        fig_w = frame_w / frame_frac
        fig_h = fixed_h + frame_w * aspect
        font = BASE_FONT_SCALE * fig_w / BASE_FIG_W_IN
        t_sv = float(J["t_survey_s"])
        t_fin = float(J["summary"]["t_finish_s"])

        print(f"{sid:>10} {aspect:7.2f} {t_sv:9.0f} {t_fin - t_sv:9.0f} "
              f"{fig_w:9.2f} {fig_h:9.2f} {frame_w * args.dpi:9.0f} {font:6.2f}",
              flush=True)
        jobs.append((sid, fig_w, font))

    if args.dry_run:
        return

    for sid, fig_w, font in jobs:
        out_p = plot_one(sid, args, fig_w=fig_w, font=font)
        print(f"  {out_p}  ({os.path.getsize(out_p) / 1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
