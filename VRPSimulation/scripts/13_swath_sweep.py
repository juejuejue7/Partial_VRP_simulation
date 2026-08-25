#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""13 — 声呐幅宽单变量实验:验证「Follower 航程由測線長决定,而非目标数」。

要检验的假设(2026-08-25,由七场景汇总数据提出)
================================================================================
提案の Follower は滑动窗口の中に居続けなければならない ⇒ **Leader の測線を丸ごと
一度なぞる**のが構造的な下限になる。目標訪問はその上に乗る増分でしかない。

七场景の実測はこの形をしている:

    提案 Follower 総航程 ÷ (2 x 測線長) = 0.99 ~ 1.23   (2 = Follower 台数)

しかし七场景では**車道数と目標密度が共線**(車道が多い場ほど疎)なので、
どちらが効いているのか判別できない。そこで本スクリプトは:

    **場景(目標分布)を完全に固定し、声呐幅宽だけを変える。**

幅宽を変えると車道数 → 測線長が変わる。目標の数も配置も一切動かない。
よって「Follower 航程 ~ 測線長」が成り立てば、効いているのは測線長であって
目標数ではない、と言い切れる。

反証可能な予測:どの幅宽でも  Follower 総航程 ≈ 2 x 測線長 (比 ≈ 1.0~1.3)。

車道数の刻み方
--------------------------------------------------------------------------------
`lawnmower_waypoints` は y = spacing/2 から spacing 刻みで y_max まで測線を敷く
⇒ 車道数 = floor(y_max/spacing - 0.5) + 1。よって **spacing = East/k で丁度 k 条**。

⚠ 幅宽は「車道間隔」と「観測窓の横幅」を**同時に**動かす(物理的に同じ量 ——
  声呐の幅そのもの)。よって本実験の独立変数は「車道数」ではなく「声呐幅宽」で
  あり、窓が広がれば一度に見える目標も増える。この交絡は消せないし、消すべきでも
  ない(実機では同じ一つの量)。結論を書くときはそう書くこと。

⚠ 本実験の幅宽は **感度解析の値**であって運用値ではない。運用値は契約既定の
  100 m(`MissionConfig.window_width_m`)。

出力 : VRPSimulation/logs/swath_sweep.csv

運行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe \
      VRPSimulation/scripts/13_swath_sweep.py
      ... 13_swath_sweep.py --scenarios sparse_2 dense_1 --lanes 1 2 3 4
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import os
import time
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from vrpsim.contracts.config import DEFAULT_DATA_DIR  # noqa: E402
from vrpsim.contracts.mission import MissionConfig  # noqa: E402
from vrpsim.mission import run_mission  # noqa: E402
from vrpsim.windows import leader_track  # noqa: E402
from vrpsim.world import build_mothra_world, build_world_for_scenario  # noqa: E402

LOGS_DIR = os.path.join(os.path.dirname(DEFAULT_DATA_DIR), "logs")

COLS = ("scenario", "n_targets", "east_m", "north_m", "swath_m", "lanes",
        "track_len_m", "leader_distance_m", "follower_total_m",
        "follower_per_track", "t_complete_s", "t_finish_s", "visited",
        "coverage", "pool_max", "solver_mix", "timed_out", "wall_s")


def track_length(path) -> float:
    p = np.asarray(path, dtype=float).reshape(-1, 2)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=["sparse_2"],
                    help="固定不动的场景(目标分布完全不变,只改幅宽)")
    ap.add_argument("--lanes", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 7],
                    help="要取到的车道数;幅宽由 East/k 反推")
    ap.add_argument("--plan-period", type=float, default=15.0)
    ap.add_argument("--max-time", type=float, default=300000.0)
    ap.add_argument("--out", default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    out_p = args.out or os.path.join(LOGS_DIR, "swath_sweep.csv")
    os.makedirs(LOGS_DIR, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    print(f"{'场景':>9} {'车道':>4} {'幅宽(m)':>8} {'測線長':>8} {'Follower総':>10} "
          f"{'÷(2×測線長)':>11} {'総耗時':>9} {'覆盖':>8} {'池max':>6} {'挂钟':>7}")
    for sid in args.scenarios:
        mw = (build_mothra_world() if sid == "mothra"
              else build_world_for_scenario(sid))
        east = float(mw.world.y_max_m)
        for k in args.lanes:
            swath = east / float(k)
            path = leader_track(mw.world, swath)
            n_lane = len(path) // 2
            if n_lane != k:            # East/k 应当恰好给 k 条;不合就明说,不静默
                print(f"  ⚠ {sid} 幅宽 {swath:.2f} m 得到 {n_lane} 条车道(期望 {k}),跳过")
                continue
            n0, e0 = float(path[0][0]), float(path[0][1])
            half = 0.5 * swath
            cfg = MissionConfig(window_width_m=swath,
                                follower_starts_ned=((n0, e0 - half), (n0, e0 + half)),
                                plan_period_s=args.plan_period,
                                max_mission_time_s=args.max_time)
            t0 = time.perf_counter()
            res = run_mission(cfg, mw, verbose=False)
            wall = time.perf_counter() - t0
            s = res.summary()

            tl = track_length(path)
            fol = float(s["total_distance_m"])
            act = [r for r in res.plan_rounds if r.pool_size > 0]
            mix = Counter(r.solver for r in act)
            row = dict(
                scenario=sid, n_targets=int(s["n_targets"]), east_m=east,
                north_m=float(mw.world.x_max_m), swath_m=swath, lanes=n_lane,
                track_len_m=tl, leader_distance_m=float(s["leader_distance_m"]),
                follower_total_m=fol, follower_per_track=fol / (2.0 * tl),
                t_complete_s=float(s["t_complete_s"]),
                t_finish_s=float(s["t_finish_s"]), visited=int(s["visited"]),
                coverage=float(s["coverage"]),
                pool_max=max((r.pool_size for r in act), default=0),
                solver_mix="+".join(f"{a}:{b}" for a, b in mix.most_common()),
                timed_out=bool(s["timed_out"]), wall_s=wall)
            rows.append(row)
            print(f"{sid:>9} {n_lane:4d} {swath:8.1f} {tl:8.0f} {fol:10.0f} "
                  f"{row['follower_per_track']:11.2f} {row['t_complete_s']:9.1f} "
                  f"{int(s['visited']):3d}/{int(s['n_targets']):<4d} "
                  f"{row['pool_max']:6d} {wall:6.0f}s", flush=True)

    with open(out_p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{out_p}  ({os.path.getsize(out_p) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
