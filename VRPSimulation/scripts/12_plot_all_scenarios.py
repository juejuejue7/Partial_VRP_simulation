#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""12 — 为全部场景批量出轨迹图(逐场景调 05_plot_mission.py,版式按长宽比自适应)。

输入 : VRPSimulation/data/scenarios/<id>/{mission,world,basemap}.npz
输出 : VRPSimulation/figures/scenarios/<id>/mission_overview.png

为什么需要自适应,而不是所有场景用同一个 --fig-width
================================================================================
05 的版式里,上排 N 帧快照沿时间轴等分整个绘图区,**帧高由帧宽和场地长宽比决定**:

    帧宽(in) = fig_width x (margin_right - margin_left) / N x frame_fill
    图高(in) = margin_bottom + panel_b + row_gap + 帧宽 x 长宽比 + margin_top

七个场景的长宽比从 1.00(sparse_2/3,方场)到 5.00(mothra,窄长条)差 5 倍。
固定 fig_width 的后果是两头都不讨好:

    mothra   长宽比 5.0 → 帧一放大就顶到天花板,图高失控
    sparse_2 长宽比 1.0 → 帧高只有帧宽那么点,可用高度白白浪费,帧被迫画小

所以这里反过来:**先定图高上限,再反推每个场景能用多宽的帧**,并给帧宽一个上限
(再宽收益递减、文件却继续涨)。两个约束取小者。

字号跟着图宽等比例放大(`--font-scale`),否则图一变宽、字就显得越来越小,
版面比例会跟目视确认过的基准版式漂开。

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe \
      VRPSimulation/scripts/12_plot_all_scenarios.py
      (只出某几个场景) ... 12_plot_all_scenarios.py --scenarios mothra dense_1
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import subprocess
import sys

import numpy as np

from vrpsim.contracts.config import DEFAULT_DATA_DIR  # noqa: E402
from vrpsim.contracts.style import load_style  # noqa: E402
from vrpsim.mission import load_result  # noqa: E402
from vrpsim.world import load_world  # noqa: E402

SIDS = ("mothra", "mef", "high_rise", "sparse_2", "sparse_3", "dense_1", "dense_2")

# 目视确认过的基准版式(2026-08-25):fig_width=22 / font_scale=3.0 下版面比例合适。
# 图宽一变,字号按同一比例跟随,版面观感才不漂。
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
    ap.add_argument("--dry-run", action="store_true", help="只打印版式,不出图")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    plot_py = os.path.join(here, "05_plot_mission.py")

    s = load_style(None)
    # 一帧占绘图区的宽度比例(与 05 的版式算法逐字一致 —— 那边改了这里必须跟)
    frame_frac = (s.margin_right - s.margin_left) / args.frames * s.frame_fill
    fixed_h = (s.margin_bottom_in + s.panel_b_height_in
               + s.row_gap_in + s.margin_top_in)

    print(f"{'场景':>10} {'长宽比':>7} {'快照间隔(s)':>11} {'图宽(in)':>9} "
          f"{'图高(in)':>9} {'帧宽(px)':>9} {'字号x':>6}")
    jobs = []
    for sid in args.scenarios:
        d_dir = os.path.join(DEFAULT_DATA_DIR, "scenarios", sid)
        mission_p = os.path.join(d_dir, "mission.npz")
        world_p = os.path.join(d_dir, "world.npz")
        for p in (mission_p, world_p):
            if not os.path.isfile(p):
                raise SystemExit(f"找不到 {p};先跑 01_build_world.py 与 "
                                 f"04_run_mission.py(或 09_compare_methods.py)"
                                 f" --scenario {sid}")
        d = load_result(mission_p)
        mw = load_world(world_p)
        t_end = float(np.asarray(d["t_s"], dtype=float)[-1])
        aspect = mw.world.x_max_m / mw.world.y_max_m

        # 两个约束取小者:帧宽上限、以及"图高不超过上限"反推出的帧宽
        frame_w = min(args.frame_width_max,
                      (args.fig_height_max - fixed_h) / aspect)
        fig_w = frame_w / frame_frac
        fig_h = fixed_h + frame_w * aspect
        font = BASE_FONT_SCALE * fig_w / BASE_FIG_W_IN
        interval = t_end / max(1, args.frames - 1)

        print(f"{sid:>10} {aspect:7.2f} {interval:11.1f} {fig_w:9.2f} "
              f"{fig_h:9.2f} {frame_w * args.dpi:9.0f} {font:6.2f}", flush=True)
        jobs.append((sid, interval, fig_w, font))

    if args.dry_run:
        return

    for sid, interval, fig_w, font in jobs:
        r = subprocess.run([sys.executable, plot_py, "--scenario", sid,
                            "--snapshot-interval", f"{interval:.4f}",
                            "--fig-width", f"{fig_w:.4f}",
                            "--font-scale", f"{font:.4f}",
                            "--dpi", str(args.dpi)],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-2000:], file=sys.stderr)
            print(r.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"{sid} 出图失败(退出码 {r.returncode})")
        last = [ln for ln in r.stdout.splitlines() if ln.strip()]
        print("  " + (last[-1].strip() if last else f"{sid} 完成"), flush=True)


if __name__ == "__main__":
    main()
