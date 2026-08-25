#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""14 — 本文の主図:提案手法の航程コストは「目標数」ではなく「測線長」で決まる。

入力(いずれも既存の産物を**読むだけ**、指标は一切再計算しない)
================================================================================
    logs/scenarios/all_scenarios_summary.json   09 → 11 が出した七場景 x 三方法
    logs/swath_sweep_*.csv                      13 が出した幅宽単変量実験

図の構成
--------------------------------------------------------------------------------
(a) **機構** — Follower 総航程 対 2 x 測線長。
    提案の Follower は滑動窓の中に居続けねばならず、Leader の測線を**丸ごと一度
    なぞる**のが構造的な下限になる。y = x 線がその下限。七場景(運用幅宽 100 m)
    と幅宽単変量実験(場景固定・幅宽だけ変更)を重ねて描く —— 後者は目標分布を
    一切動かしていないので、線に乗ることが「効いているのは測線長であって目標数
    ではない」の直接証拠になる。

(b) **帰結** — 提案 / 二段式の比 対 車道数。
    時間比(<1 = 提案が速い)と航程比(>1 = 提案が遠回り)が車道数に対して
    **どちらも単調**。車道が増えるほど提案の旨みは減り代償は増える ⇒ 適用境界。

⚠ 図には中日文字を一切載せない(契約 metrics.py の口径)。方法名は
  `SCENARIO_NAMES_EN` から取る —— 出図スクリプトに英訳を書かない。

出力 : VRPSimulation/figures/core_findings.png

運行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe \
      VRPSimulation/scripts/14_plot_core_findings.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vrpsim.contracts.config import DEFAULT_DATA_DIR, DEFAULT_FIGURE_DIR  # noqa: E402
from vrpsim.contracts.metrics import (SCENARIO_NAMES_EN, SCENARIO_TWOPHASE,  # noqa: E402
                                      SCENARIO_VRP)
from vrpsim.contracts.style import load_style  # noqa: E402
from vrpsim.viz import use_cjk_font  # noqa: E402

LOGS_DIR = os.path.join(os.path.dirname(DEFAULT_DATA_DIR), "logs")


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_scenarios() -> Dict[str, Any]:
    p = os.path.join(LOGS_DIR, "scenarios", "all_scenarios_summary.json")
    if not os.path.isfile(p):
        raise SystemExit(f"找不到 {p};先跑 11_compare_all_scenarios.py")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_sweeps() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(os.path.join(LOGS_DIR, "swath_sweep*.csv"))):
        with open(p, encoding="utf-8-sig") as f:
            rows.extend(list(csv.DictReader(f)))
    if not rows:
        raise SystemExit("找不到 logs/swath_sweep*.csv;先跑 13_swath_sweep.py")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--font-scale", type=float, default=1.0)
    args = ap.parse_args()
    out_p = args.out or os.path.join(DEFAULT_FIGURE_DIR, "core_findings.png")

    style = load_style(None, font_scale=args.font_scale, dpi=args.dpi)
    use_cjk_font(style.font_family)
    style.apply_rcparams()

    J = load_scenarios()
    data, prof = J["data"], J["profile"]
    sweeps = load_sweeps()
    sids = list(data.keys())

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # ---------- (a) 機構:Follower 総航程 対 2 x 測線長 -------------------
    x_all: List[float] = []
    y_all: List[float] = []

    by_scen: Dict[str, List] = {}
    for r in sweeps:
        by_scen.setdefault(r["scenario"], []).append(r)
    sweep_markers = ("o", "s", "^", "D")
    for k, (sid, rs) in enumerate(sorted(by_scen.items())):
        rs = sorted(rs, key=lambda r: int(r["lanes"]))
        x = [2.0 * _f(r["track_len_m"]) for r in rs]
        y = [_f(r["follower_total_m"]) for r in rs]
        x_all += x
        y_all += y
        axa.plot(x, y, marker=sweep_markers[k % len(sweep_markers)], ms=6.0,
                 lw=1.4, alpha=0.9,
                 label=f"swath sweep: {sid} (1-{max(int(r['lanes']) for r in rs)} lanes)")

    xs = [2.0 * _f(data[s][SCENARIO_VRP]["leader_distance_m"]) for s in sids]
    ys = [_f(data[s][SCENARIO_VRP]["total_distance_m"]) for s in sids]
    x_all += xs
    y_all += ys
    axa.scatter(xs, ys, s=70, marker="*", zorder=5, color="#d33682",
                edgecolor="white", linewidth=0.8,
                label="7 scenarios @ operating swath 100 m")

    hi = max(x_all + y_all) * 1.06
    axa.plot([0, hi], [0, hi], "--", lw=1.2, color="#586e75", zorder=1,
             label="shadowing floor  y = x")
    axa.set_xlim(0, hi)
    axa.set_ylim(0, hi)
    axa.set_xlabel("2 x survey track length [m]")
    axa.set_ylabel("Follower total distance [m]  (proposed)")
    axa.set_title("(a) Follower distance is set by track length")
    axa.grid(alpha=0.25)
    axa.legend(fontsize=8.0, loc="upper left", framealpha=0.9)

    # ---------- (b) 帰結:提案/二段式 の比 対 車道数 ---------------------
    lanes = np.array([prof[s]["lanes"] for s in sids], dtype=float)
    t_ratio = np.array([_f(data[s][SCENARIO_VRP]["t_complete_s"])
                        / _f(data[s][SCENARIO_TWOPHASE]["t_complete_s"])
                        for s in sids])
    d_ratio = np.array([_f(data[s][SCENARIO_VRP]["fleet_distance_m"])
                        / _f(data[s][SCENARIO_TWOPHASE]["fleet_distance_m"])
                        for s in sids])
    o = np.argsort(lanes)

    c_t, c_d = "#268bd2", "#cb4b16"
    axb.plot(lanes[o], t_ratio[o], "o-", color=c_t, lw=1.8, ms=7,
             label="time  $t_{complete}$ ratio")
    axb.axhline(1.0, ls=":", lw=1.2, color="#586e75")
    axb.set_xlabel("Leader survey lanes  (field width / sonar swath)")
    axb.set_ylabel("proposed / two-phase   (time)", color=c_t)
    axb.tick_params(axis="y", labelcolor=c_t)
    axb.set_ylim(0.0, 1.15)

    ax2 = axb.twinx()
    ax2.plot(lanes[o], d_ratio[o], "s--", color=c_d, lw=1.8, ms=7,
             label="distance  fleet ratio")
    ax2.set_ylabel("proposed / two-phase   (distance)", color=c_d)
    ax2.tick_params(axis="y", labelcolor=c_d)
    ax2.set_ylim(0.0, 2.6)

    # 同じ車道数の場景が複数ある(2 車道が 3 つ、7 車道が 2 つ)ので、
    # ラベルは x ごとに段をずらす —— 固定オフセットだと重なって読めない。
    seen: Dict[float, int] = {}
    for s, lx, ty in sorted(zip(sids, lanes, t_ratio), key=lambda z: (z[1], -z[2])):
        k = seen.get(lx, 0)
        seen[lx] = k + 1
        axb.annotate(s, (lx, ty), textcoords="offset points",
                     xytext=(8, -4 - 11 * k), ha="left", fontsize=7.5,
                     color="#586e75")
    axb.set_title("(b) Both advantage and cost are monotone in lane count")
    axb.grid(alpha=0.25)
    h1, l1 = axb.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axb.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="center right", framealpha=0.9)

    fig.suptitle(f"{SCENARIO_NAMES_EN[SCENARIO_VRP]} vs "
                 f"{SCENARIO_NAMES_EN[SCENARIO_TWOPHASE]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    fig.savefig(out_p, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"{out_p}  ({os.path.getsize(out_p) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
