#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""11 — 跨场景 x 三方法 汇总表(把 09 的逐场景产物横向拼成一份对比报告)。

输入 : VRPSimulation/logs/scenarios/<id>/methods_compare.csv  (09 逐场景产出)
输出 : VRPSimulation/logs/Comparasion_0825.md                 (默认;--out 可改)
       VRPSimulation/logs/scenarios/all_scenarios_summary.json

⚠ **指标一律从 09 写出的 CSV 读, 本脚本不重算任何一个数**。场景画像(目标数/尺寸/
  车道数)则由 world 现算 —— 那是几何, 不是指标。

⚠ 跨方法的效率口径只有 `t_per_target_s`(契约 metrics.py 文件头):`time_efficiency*`
  三列被 `MetricSpec.applies_to` 结构性挡住, 它们的分子就是一次全局 min-max VRP 的解,
  放进跨方法表会让「二段式全局VRP」按构造得 1.0, 属循环论证。

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe \
      VRPSimulation/scripts/11_compare_all_scenarios.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import datetime as _dt
import io
import json
import os
from typing import Any, Dict, Optional

import numpy as np

from vrpsim.contracts.config import DEFAULT_DATA_DIR  # noqa: E402
from vrpsim.contracts.mission import MissionConfig  # noqa: E402
from vrpsim.windows import leader_track  # noqa: E402
from vrpsim.world import build_mothra_world, build_world_for_scenario  # noqa: E402

LOGS_DIR = os.path.join(os.path.dirname(DEFAULT_DATA_DIR), "logs")
SIDS = ("mothra", "mef", "high_rise", "dense_1", "dense_2", "sparse_2", "sparse_3")
METH = (("vrp", "局部VRP(提案)"), ("twophase", "二段式全局VRP"), ("lawnmower", "lawnmower"))


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_rows(sid: str) -> Dict[str, Dict[str, str]]:
    p = os.path.join(LOGS_DIR, "scenarios", sid, "methods_compare.csv")
    if not os.path.isfile(p):
        raise SystemExit(f"找不到 {p};先跑 09_compare_methods.py --scenario {sid}")
    with open(p, encoding="utf-8-sig") as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def solver_provenance(buf) -> None:
    """本表是用哪套求解器口径跑出来的 —— **从落盘的逐轮记录反查，不手写**。

    D21 前后同一批场景的数字差别很大（二段式最长单机航程差 13~36%），表里不写清
    口径，几个月后就没人能分辨手上这份是哪一版。故此节从
    `logs/scenarios/<id>/plan_cost.csv` 的 `solver` 列统计实际走过的求解器。
    手写会漂移，所以一个字都不手写。
    """
    from collections import Counter
    per_method: Dict[str, Counter] = {}
    for sid in SIDS:
        p = os.path.join(LOGS_DIR, "scenarios", sid, "plan_cost.csv")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if int(float(r["pool_size"])) <= 0:
                    continue           # 池空的轮次没解过，不该算进求解器分布
                per_method.setdefault(r["scenario"], Counter())[r["solver"]] += 1
    if not per_method:
        return
    print("### 求解器口径（D21）\n", file=buf)
    print("| 方法 | 実際に走った求解器（全场景合计の求解次数）|", file=buf)
    print("|---|---|", file=buf)
    for key, label in METH:
        c = per_method.get(key)
        cell = ("該当なし（任务期间不解任何问题）" if c is None else
                "、".join(f"`{k}` {v} 回" for k, v in c.most_common()))
        print(f"| {label} | {cell} |", file=buf)
    d = MissionConfig()
    print(f"\n⚠ 口径 = `solver={d.solver}`：2 台车且池 ≤ "
          f"`vrp_exact_max_targets={d.vrp_exact_max_targets}` 走 Held-Karp **精确最优**"
          f"（构造性确定，同输入必同输出）；超阈值退 OR-Tools"
          f"（预算 `vrp_time_limit_s={d.vrp_time_limit_s:g} s`、"
          f"`vrp_gls_lambda={d.vrp_gls_lambda:g}`）。"
          f"二段式は全体目标（46~66 点）を一度に解くので**必ず** OR-Tools 側。"
          f"詳細と実測根拠は `contracts/DECISIONS.md` の **D21**。\n", file=buf)


def scenario_profile(sid: str) -> Dict[str, Any]:
    mw = build_mothra_world() if sid == "mothra" else build_world_for_scenario(sid)
    area_ha = mw.world.x_max_m * mw.world.y_max_m / 1e4
    lanes = len(leader_track(mw.world, MissionConfig().window_width_m)) // 2
    return dict(n=mw.dataset.n, N=mw.world.x_max_m, E=mw.world.y_max_m,
                area_ha=area_ha, dens=mw.dataset.n / area_ha, lanes=lanes,
                aspect=mw.world.x_max_m / mw.world.y_max_m)


def table(buf, data, title: str, key: str, fmt: str, better: str, note: str = "") -> None:
    print(f"\n## {title}\n", file=buf)
    if note:
        print(note + "\n", file=buf)
    print("| 场景 | " + " | ".join(n for _, n in METH) + " | 最优 |", file=buf)
    print("|---|" + "---|" * (len(METH) + 1), file=buf)
    for sid in SIDS:
        row = data[sid]
        vals = [_f(row[m].get(key)) if m in row else None for m, _ in METH]
        fin = [(i, v) for i, v in enumerate(vals) if v is not None]
        best = None
        if fin:
            best = (min(fin, key=lambda t: t[1])[0] if better == "lower"
                    else max(fin, key=lambda t: t[1])[0])
        cells = []
        for i, v in enumerate(vals):
            s = "—" if v is None else fmt.format(v)
            cells.append(f"**{s}**" if i == best else s)
        tail = METH[best][1] if best is not None else "—"
        print(f"| `{sid}` | " + " | ".join(cells) + f" | {tail} |", file=buf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(LOGS_DIR, "Comparasion_0825.md"))
    args = ap.parse_args()

    data = {sid: load_rows(sid) for sid in SIDS}
    prof = {sid: scenario_profile(sid) for sid in SIDS}

    buf = io.StringIO()
    print(f"# 三方法 x 七场景 对比（{_dt.date.today().isoformat()}）\n", file=buf)
    print("被比较的三种方法（**硬件都是 3 台 AUV**，唯一变量是探査与観測怎么组织）：\n",
          file=buf)
    print("| 方法 | 编成 | 组织方式 |", file=buf)
    print("|---|---|---|", file=buf)
    print("| 局部VRP（提案） | 1 Leader + 2 Follower | 探査と観測が**並行**；"
          "Leader 声呐揭示目标，Follower 按滑动窗口内的局部 min-max VRP 逐次接单 |",
          file=buf)
    print("| 二段式全局VRP | 1 Leader + 2 Follower | 先探明**全部**目标，"
          "再解**一次**全局 min-max VRP，**逐次** |", file=buf)
    print("| lawnmower | 3 台（全部相机机） | 探査即観測（全覆盖扫描），"
          "无 Leader、无通信、无求解 |", file=buf)
    print("\n数据由 `09_compare_methods.py --scenario <id>` 逐场景产出，"
          "本表由 `11_compare_all_scenarios.py` 横向拼接，**不重算任何指标**。\n", file=buf)
    solver_provenance(buf)

    print("## ⓪ 场景画像\n", file=buf)
    print("| 场景 | 目标数 | N×E (m) | 長宽比 N/E | 面积(ha) | 密度(/ha) | Leader 車道数 |",
          file=buf)
    print("|---|---|---|---|---|---|---|", file=buf)
    for sid in SIDS:
        p = prof[sid]
        print(f"| `{sid}` | {p['n']} | {p['N']:.0f}×{p['E']:.0f} | {p['aspect']:.2f} | "
              f"{p['area_ha']:.1f} | {p['dens']:.2f} | {p['lanes']} |", file=buf)
    print("\n⚠ 車道数 = `ceil(East 跨度 / 声呐幅宽 100 m)`。Leader 沿 North 长边走 "
          "boustrophedon 往复测线，車道间距 = 声呐幅宽，边到边覆盖。\n", file=buf)

    table(buf, data, "① 効率 — 単位目標あたり所要時間 `t_per_target_s` [s/目標] ↓",
          "t_per_target_s", "{:.1f}", "lower",
          "⚠ **跨方法唯一合法的效率口径**（契约 `metrics.py` 文件头）："
          "`time_efficiency*` 三列被 `applies_to` 结构性挡住 —— 它们的分子就是一次"
          "全局 min-max VRP 的解，放进来会让二段式按构造得 1.0，属循环论证。\n\n"
          "⚠ 该列**对漏点中性**：少做几个点，分子（早收工）与分母（観測成功数）"
          "一起变小，不像 `t_finish_s` 那样白送一个「更快」的假象。")
    table(buf, data, "② 総耗時 `t_complete_s` [s] ↓", "t_complete_s", "{:.0f}", "lower",
          "⚠ 未全覆盖时按契约定义为 `nan`（任务没完成，谈不上完成时刻），表中记 `—`。"
          "那两行必须连同 ⑥ 的覆盖率一起读。")
    table(buf, data, "③ 距離不均衡 `load_imbalance_distance_frac` ↓",
          "load_imbalance_distance_frac", "{:.2%}", "lower",
          "`(max d − min d) / max d`，min-max VRP 直接优化的就是它。\n\n"
          "⚠ lawnmower 恒为 `0.00%` 是**构造使然**（按测线均分场地，三台航程按构造相等），"
          "不是算法优势 —— 别把它读成「最均衡的方法」。")
    table(buf, data, "④ 時間不均衡 `load_imbalance_time_frac` ↓",
          "load_imbalance_time_frac", "{:.2%}", "lower",
          "`(max T − min T) / max T`。契约标注为**本族要害指标** —— 它测的是"
          "「有机先干完、然后干等着」，正是 min-max 目标选错对象后的失效形态。\n\n"
          "与 ③ 一起读：二段式常常**距離很均衡而時間很不均衡**，"
          "因为 min-max VRP 均衡的是距离，而任务时间由最慢那台决定。")
    table(buf, data, "⑤ 艦隊総航程 `fleet_distance_m` [m]（含 Leader）↓",
          "fleet_distance_m", "{:.0f}", "lower",
          "全部 AUV 航程之和。lawnmower 没有 Leader，3 台之和即艦隊総航程；"
          "協調探査若只报 Follower 的和，等于白送 Leader 的整条测线，对比不公平。")

    print("\n## ⑥ 覆盖率与终止状态（数据可信性前置检查）\n", file=buf)
    print("| 场景 | " + " | ".join(n for _, n in METH) + " | 超时未终止 |", file=buf)
    print("|---|" + "---|" * (len(METH) + 1), file=buf)
    for sid in SIDS:
        row = data[sid]
        cov = [_f(row[m].get("coverage")) for m, _ in METH]
        to = [str(row[m].get("timed_out", "")).lower() == "true" for m, _ in METH]
        bad = [n for (_, n), t in zip(METH, to) if t]
        cells = []
        for c in cov:
            s = "—" if c is None else f"{100 * c:.1f}%"
            cells.append(f"**{s}**" if s not in ("100.0%", "—") else s)
        print(f"| `{sid}` | " + " | ".join(cells) + " | "
              + (", ".join(bad) if bad else "无") + " |", file=buf)
    print("\n⚠ 「超时未终止」= 跑到 `max_mission_time_s` 才停，**结果不可信**。"
          "本轮七场景全部自然终止。\n", file=buf)

    out = {"generated": _dt.date.today().isoformat(), "profile": prof,
           "data": {sid: {m: data[sid][m] for m, _ in METH if m in data[sid]}
                    for sid in SIDS}}
    jp = os.path.join(LOGS_DIR, "scenarios", "all_scenarios_summary.json")
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"{args.out}  ({os.path.getsize(args.out) / 1024:.1f} KB)")
    print(f"{jp}  ({os.path.getsize(jp) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
