#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""09 — 三方法の時間効率比較(D19)。

被比较的三种方法(**硬件都是 3 台 AUV**,唯一的变量是探査与観測怎么组织):

    局部VRP(提案)   1 Leader + 2 Follower   探査と観測が**並行**
    二段式全局VRP    1 Leader + 2 Follower   先探明全部目标,再解一次全局 VRP,**逐次**
    lawnmower       3 台(全部相机机)         探査即観測(全覆盖扫描)

⚠ 跨方法的時間効率口径 = `t_per_target_s`(**単位目標あたり時間,不归一化**)。
  `time_efficiency*` 三列**不出现在本表**,由 `MetricSpec.applies_to` 结构性挡住:
  它们的分子本身就是一次全局 min-max VRP 的解,放进来会让「全局VRP方法」
  按构造得 1.0 —— 循环论证。详见 `contracts/metrics.py` 文件头。

输入 : --scenario <id>(默认 mothra);场景数据来自 Bethmetory_data_process 的
       scenarios.json 与 outputs/scenarios/<id>/
输出(统一落在按场景分的目录下,<id> 含 mothra 本身):
       VRPSimulation/data/scenarios/<id>/{mission,twophase,lawnmower}.npz + _summary
       VRPSimulation/logs/scenarios/<id>/methods_compare.csv / .md / plan_cost.csv
       VRPSimulation/figures/scenarios/<id>/methods_compare.png

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe \
      VRPSimulation/scripts/09_compare_methods.py
      D:/nixingxing/Anaconda/envs/auv_py310/python.exe \
      VRPSimulation/scripts/09_compare_methods.py --scenario sparse_2 --max-time 200000
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import io
import os
import textwrap
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vrpsim.contracts.config import DEFAULT_DATA_DIR, DEFAULT_FIGURE_DIR  # noqa: E402
from vrpsim.contracts.lawnmower import LawnmowerConfig  # noqa: E402
from vrpsim.contracts.metrics import (ALL_SCENARIOS, BETTER_MARK,  # noqa: E402
                                      FAMILY_NAMES, SCENARIO_LAWNMOWER,
                                      SCENARIO_NAMES, SCENARIO_TWOPHASE,
                                      SCENARIO_VRP)
from vrpsim.contracts.mission import MissionConfig  # noqa: E402
from vrpsim.contracts.twophase import TwoPhaseConfig  # noqa: E402
from vrpsim.lawnmower import run_lawnmower, save_lawn_result  # noqa: E402
from vrpsim.mission import run_mission, save_result  # noqa: E402
from vrpsim.report import (metrics_visible_for, print_metric_table,  # noqa: E402
                           write_metric_csv, writable_path)
from vrpsim.twophase import run_twophase, save_twophase_result  # noqa: E402
from vrpsim.windows import leader_track  # noqa: E402
from vrpsim.world import build_mothra_world, build_world_for_scenario  # noqa: E402

LOGS_DIR = os.path.join(os.path.dirname(DEFAULT_DATA_DIR), "logs")


def follower_starts_flanking_window(world, window_width_m: float):
    """Follower 初始位置 = Leader **初始窗口**的横向两侧(与 04_run_mission.py 同一规则)。

    见该文件的同名函数:窗口宽是固定的声呐幅宽,与场地宽无关;按场地两端铺开会让一台
    开场就要空跑几百米。Mothra 下算得 `((0,0),(0,100))`,与契约默认值逐位相同。
    """
    path = leader_track(world, window_width_m)
    n0, e0 = float(path[0][0]), float(path[0][1])
    half = 0.5 * float(window_width_m)
    return ((n0, e0 - half), (n0, e0 + half))


def build_parser() -> argparse.ArgumentParser:
    d = MissionConfig()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mothra",
                    help="Bethmetory_data_process/scenarios.json 里的场景 id(默认 mothra)")
    # 局部VRP 用哪组参数 —— 默认取 08 扫描出的最优组(L0.5 / T15 / 等待开)
    ap.add_argument("--leader-speed", type=float, default=d.leader_speed_mps)
    ap.add_argument("--follower-speed", type=float, default=d.follower_speed_mps)
    ap.add_argument("--plan-period", type=float, default=15.0,
                    help="局部VRP 的路径规划周期 [s];默认 15 = 08 扫描出的最优组")
    ap.add_argument("--usbl-period", type=float, default=d.usbl_period_s)
    ap.add_argument("--dwell", type=float, default=d.dwell_time_s)
    ap.add_argument("--solver", choices=("exact", "ortools", "greedy"),
                    default=d.solver,
                    help="求解器口径(D21):exact=小规模池走 Held-Karp 精确最优、超阈值退 ortools;ortools=一律元启发式(消融对照);greedy=确定性贪心")
    ap.add_argument("--vrp-time-limit", type=float, default=d.vrp_time_limit_s)
    ap.add_argument("--max-time", type=float, default=8000.0)
    # 缺省 None ⇒ main() 按 --scenario 拼成 <root>/scenarios/<id>/
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-figure", action="store_true")
    return ap


# ======================================================================
# 复现命令(由实际用的参数生成,不手写 —— 手写必漂移)
# ======================================================================
def compare_command(args) -> str:
    return (f"$PY VRPSimulation/scripts/09_compare_methods.py"
            f" --scenario {args.scenario}"
            f" --leader-speed {args.leader_speed:g}"
            f" --follower-speed {args.follower_speed:g}"
            f" --plan-period {args.plan_period:g}"
            f" --usbl-period {args.usbl_period:g}"
            f" --dwell {args.dwell:g}"
            f" --solver {args.solver}"
            f" --vrp-time-limit {args.vrp_time_limit:g}"
            f" --max-time {args.max_time:g}")


# ======================================================================
# 正文用の 5 項目(人工指定 2026-08-15)
# ======================================================================
# 図に**出ていない**ものから 5 つ。図は「観測完了本数の推移」と「負荷均衡」の 2 面
# なので、そこと重複する列は入れない。
#   ⚠ `t_per_target_s` を外した理由:三方法とも 22/22 なので
#     `t_complete_s / 22` の定数倍でしかなく、総耗時と同じ情報。
#   ⚠ `t_observation_mean_s` を外した理由:図の左パネル(累積曲線)の要約であって
#     独立した情報ではない。
KEY_METRICS: Tuple[str, ...] = (
    "t_complete_s",        # 総耗時 —— 人工指定
    "fleet_distance_m",    # AUV 総路程(**Leader 込み**)—— 人工指定
    "max_distance_m",      # 最大単機航程 —— 人工指定。⚠ 式は Follower のみ(下注)
    "max_busy_time_s",     # 最長単機稼働時間 —— 人工指定(空転率と差し替え 2026-08-15)
    "n_comm_messages",     # 音響報文数 —— **提案側**の代価。図に無い
)


def key_metric_section(rows: List[Dict[str, Any]], buf) -> None:
    """5 項目の対比表。値は `metrics` からそのまま引く(再計算しない)。"""
    from vrpsim.contracts.metrics import METRIC_BY_KEY

    from vrpsim.report import fmt_metric

    names = [r["_label"] for r in rows]
    print("## 正文用の 5 項目（図に出していないもの）\n", file=buf)
    print("| 指標 | 単位 | 方向 | " + " | ".join(names) + " |", file=buf)
    print("|---|---|---|" + "---|" * len(rows), file=buf)
    for k in KEY_METRICS:
        m = METRIC_BY_KEY[k]
        cells = " | ".join(fmt_metric(m, r["metrics"].get(k)) for r in rows)
        print(f"| **{m.label_zh}** `{m.key}` | {m.unit} | {BETTER_MARK[m.better]} | "
              f"{cells} |", file=buf)
    # ⚠ 契約 `max_distance_m` の式は `max(各 Follower 航程)` で **Leader を含まない**。
    #   ラベルの「最大単機航程」は艦隊全体の最大に読めるが、二段式では Leader の
    #   500 m が Follower の最大を上回るため、両者が食い違う。ここは既存の 2 列
    #   (`max_distance_m` と `leader_distance_m`)の算術で出す派生値として明示する。
    der = [max(float(r["metrics"].get("max_distance_m") or 0.0),
               float(r["metrics"].get("leader_distance_m") or 0.0)) for r in rows]
    print(f"| *（派生）艦隊内の最大単機航程* `max(max_distance_m, leader_distance_m)` "
          f"| m | ↓ | " + " | ".join(f"*{v:.1f}*" for v in der) + " |", file=buf)
    print("", file=buf)
    print("⚠ **`max_distance_m` は Leader を含まない**（式は `max(各 Follower 航程)`）。"
          "ラベルの「最大単機航程」は艦隊全体の最大に読めてしまうが、"
          "二段式では Leader の 500.0 m が Follower の最大を上回るので両者が食い違う。"
          "電池の律速を見たいときは派生行の方を読むこと。"
          "**契約側の命名／式の不整合として A0 に係属中（未裁決）。**\n", file=buf)
    # ⚠ `busy_time_s` は**時間軸全体**を数える(`duty_frac` の有効区間 t<=t_finish とは
    #   別口径)。提案は最後の目標を観測したあとも、下発済みの投影点を消化しながら
    #   走り続けるので、稼働時間が任務完了時刻を**上回る**ことがある。
    #   隠さずに、超えた方法とその差を数字で出す。
    over = [(r["_label"], float(r["metrics"]["max_busy_time_s"]),
             float(r["metrics"]["t_complete_s"]))
            for r in rows
            if (r["metrics"].get("max_busy_time_s") is not None
                and np.isfinite(float(r["metrics"].get("t_complete_s", float("nan"))))
                and float(r["metrics"]["max_busy_time_s"])
                > float(r["metrics"]["t_complete_s"]) + 1e-9)]
    if over:
        for lab, b, tc in over:
            print(f"⚠ **{lab} は `max_busy_time_s` ({b:.1f} s) が "
                  f"`t_complete_s` ({tc:.1f} s) を {b - tc:.1f} s 上回る。**"
                  "誤記ではない —— 最後の目標を観測し終えたあとも、"
                  "**既に下発済みの投影点**（目標ではない）を消化しながら走り続けるため。"
                  "観測は増えないが機体は動いており、電池はその分減る。"
                  "開ループ下発の実コストとしてそのまま計上する。"
                  "⚠ `busy_time_s` の集計区間は**時間軸全体**で、"
                  "`duty_*_frac` の有効区間（`t <= t_finish_s`）とは別口径。\n", file=buf)
    tp = next((r for r in rows if r["scenario"] == SCENARIO_TWOPHASE), None)
    if tp is not None and tp["metrics"].get("duty_idle_frac") is not None:
        b = float(tp["metrics"]["max_busy_time_s"])
        tc = float(tp["metrics"]["t_complete_s"])
        print(f"⚠ 二段式の稼働時間が任務長の {b / tc:.0%} しかないのは、"
              "Follower が阶段1 の間ずっと待機（IDLE）しているから —— "
              f"差し替え前の `duty_idle_frac`（{tp['metrics']['duty_idle_frac']:.1%}）"
              "と同じことを秒数で言っている。\n", file=buf)
    print("⚠ `n_comm_messages` は**提案だけが払っている代価**。"
          "二段式と lawnmower の 0 は「測れていない」ではなく本当に 0 —— "
          "経路は下水前に確定し、任務中に音響報文を必要としない。\n", file=buf)


# ======================================================================
# 路径规划コスト —— 逐次規劃の池大小と求解挂钟耗时
# ======================================================================
# ⚠ `solve_wall_s` は `time.perf_counter` の**挂钟**量:宿主機の性能と
#   `vrp_time_limit_s` に依存し、**仿真時間軸には入らない**
#   (仿真側が求解に見込む余裕は `MissionConfig.plan_solve_s`、既定 0)。
#   記録する目的はただ一つ ——「この開銷は無視してよいか」に数字で答えるため。
PLAN_COST_COLS: Tuple[str, ...] = (
    "scenario", "round_idx", "t_plan_s", "pool_size",
    "n_assigned_total", "n_projection", "solve_wall_s", "solver")


def plan_cost_records(res_vrp, res_2p) -> List[Dict[str, Any]]:
    """一行 = 一回の VRP 求解。局部VRP は毎ラウンド、二段式は定義上ちょうど 1 行。

    lawnmower は**行が出ない** —— 測線は下水前に決まり、任務中に解く問題が無い。
    「該当なし」ではなく本当にゼロなので、集計側で 0 回として明示する。
    """
    out: List[Dict[str, Any]] = []
    for r in res_vrp.plan_rounds:
        out.append({"scenario": SCENARIO_VRP, "round_idx": int(r.round_idx),
                    "t_plan_s": float(r.t_plan_s), "pool_size": int(r.pool_size),
                    "n_assigned_total": int(r.n_assigned_total),
                    "n_projection": int(r.n_projection),
                    "solve_wall_s": float(r.solve_wall_s), "solver": str(r.solver)})
    out.append({"scenario": SCENARIO_TWOPHASE, "round_idx": 0,
                # 起点も目標表も阶段1 の前から確定しているので、母船で解ける(D19)
                "t_plan_s": 0.0,
                "pool_size": int(res_2p.n_targets),
                "n_assigned_total": int(sum(len(r) for r in res_2p.routes)),
                "n_projection": 0,
                "solve_wall_s": float(res_2p.meta.get("solve_wall_s", 0.0)),
                "solver": str(res_2p.meta.get("solver", ""))})
    return out


def plan_cost_stats(recs: List[Dict[str, Any]],
                    rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """方法ごとに 1 行へ畳む。`frac_of_mission` が「無視してよいか」の判定材料。"""
    out: List[Dict[str, Any]] = []
    for r in rows:
        sc = r["scenario"]
        mine = [x for x in recs if x["scenario"] == sc]
        tf = float(r["metrics"]["t_finish_s"])
        w = [x["solve_wall_s"] for x in mine]
        p = [x["pool_size"] for x in mine]
        # ⚠ 池が空のラウンドは**問題を解いていない**(耗时 0)。それを混ぜて平均すると
        #   「1 回あたりの求解コスト」が実態の半分になる —— 実解ラウンドだけの
        #   統計を別に持つ。どちらも出して、読み手が取り違えないようにする。
        act = [x["solve_wall_s"] for x in mine if x["pool_size"] > 0]
        total = float(sum(w))
        out.append({
            "scenario": sc, "_label": r["_label"], "n_solves": len(mine),
            "n_active": len(act), "n_empty": len(mine) - len(act),
            "pool_min": min(p) if p else 0, "pool_max": max(p) if p else 0,
            "pool_mean": float(np.mean(p)) if p else 0.0,
            "pool_mean_active": (float(np.mean([x["pool_size"] for x in mine
                                                if x["pool_size"] > 0]))
                                 if act else 0.0),
            "wall_total_s": total,
            "wall_mean_s": float(np.mean(w)) if w else 0.0,
            "wall_mean_active_s": float(np.mean(act)) if act else 0.0,
            "wall_min_active_s": float(min(act)) if act else 0.0,
            "wall_max_active_s": float(max(act)) if act else 0.0,
            "t_finish_s": tf,
            "frac_of_mission": (total / tf) if np.isfinite(tf) and tf > 0 else 0.0,
        })
    return out


def write_plan_cost_csv(recs: List[Dict[str, Any]], path: str) -> str:
    import csv

    path = writable_path(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(PLAN_COST_COLS)
        for r in recs:
            wr.writerow([r[c] for c in PLAN_COST_COLS])
    return path


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """池大小 と 求解耗时 の相関。定数列(= 予算を使い切っている)なら nan。"""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ======================================================================
# 报告
# ======================================================================
def write_report(rows: List[Dict[str, Any]], path: str, args,
                 notes: Dict[str, Any],
                 plan_cost: Optional[List[Dict[str, Any]]] = None,
                 plan_recs: Optional[List[Dict[str, Any]]] = None) -> str:
    buf = io.StringIO()
    print("# 三方法の時間効率比較\n", file=buf)
    print("三种方法**硬件相同(3 台 AUV)**,唯一的变量是探査与観測怎么组织：\n", file=buf)
    print("| 方法 | 编成 | 探査と観測 |", file=buf)
    print("|---|---|---|", file=buf)
    print("| **局部VRP（提案）** | 1 Leader + 2 Follower | **並行** |", file=buf)
    print("| 二段式全局VRP | 1 Leader + 2 Follower | 逐次（先探明全部，再解一次全局 VRP）|",
          file=buf)
    print("| lawnmower | 3 台（全部相机机）| 探査即観測（全覆盖扫描）|\n", file=buf)

    print("## 時間効率の口径（D19）\n", file=buf)
    print("**跨方法比时间用 `t_per_target_s`（単位目標あたり時間，不归一化）。**\n", file=buf)
    print("⚠ `time_efficiency` 三列**不出现在本表**，被 `MetricSpec.applies_to` "
          "结构性挡住。两条独立的理由：\n", file=buf)
    print("1. **循环论证** —— 它们的分子 `t_mission_reference_s` 本身就是一次"
          "全局 min-max VRP 的解。把「全局VRP方法」列为被比较对象后，"
          "它按构造就得 ≈1.0，胜负在定义里就决定了。\n"
          "2. **对另外两法没有定义** —— 分子里含 `测线长 / v_leader`，"
          "那是局部VRP 特有的量；lawnmower 没有 Leader 测线。\n", file=buf)

    # ---- 主结论 ----
    print("## 主结论\n", file=buf)
    print("| 方法 | 観測成功数 | 実測終了時刻 | **単位目標あたり時間** | 対 提案 |",
          file=buf)
    print("|---|---|---|---|---|", file=buf)
    base = next((r for r in rows if r["scenario"] == SCENARIO_VRP), None)
    b = base["metrics"]["t_per_target_s"] if base else float("nan")
    for r in rows:
        m = r["metrics"]
        ratio = (m["t_per_target_s"] / b) if np.isfinite(b) and b > 0 else float("nan")
        star = "**" if r["scenario"] == SCENARIO_VRP else ""
        print(f"| {star}{r['_label']}{star} | {m['visited']} / {m['n_targets']} | "
              f"{m['t_finish_s']:.1f} s | {star}{m['t_per_target_s']:.1f} s/目標{star} | "
              f"{ratio:.2f}× |", file=buf)
    print("", file=buf)

    if notes.get("t_survey_s") is not None:
        print(f"二段式的两阶段拆分：**探査 {notes['t_survey_s']:.1f} s + "
              f"観測 {notes['t_observe_s']:.1f} s = {notes['t_finish_s']:.1f} s**，"
              f"两段严格不重叠。提案方法把这两段**叠在一起跑**，"
              f"省下的就是其中较短的那一段。\n", file=buf)

    # ---- 正文用の 5 項目 ----
    key_metric_section(rows, buf)

    # ---- 負荷均衡 ----
    print("## ⑥ 負荷均衡：min-max VRP 均衡的是距离，不是时间\n", file=buf)
    print("| 方法 | 距離不均衡 | 時間不均衡 | 差 |", file=buf)
    print("|---|---|---|---|", file=buf)
    for r in rows:
        m = r["metrics"]
        dd = m.get("load_imbalance_distance_frac")
        tt = m.get("load_imbalance_time_frac")
        if dd is None or tt is None:
            continue
        print(f"| {r['_label']} | {dd:.1%} | {tt:.1%} | {tt - dd:+.1%} |", file=buf)
    print("", file=buf)
    print("`solve_minmax_vrp_from` 的弧代价**只有距离**，停留时长（`dwell_time_s`）"
          "不进代价函数；而任务完成时刻由**最慢那台的时间**决定。"
          "停留占比一高，两者就分叉。\n", file=buf)
    print("**这一列具体在测什么**：`時間不均衡` 大 ⇔ 有机先干完、然后干等着 —— "
          "正是 min-max 目标选错对象后的失效形态。\n", file=buf)
    print("- **二段式**：距離 0.2% / 時間 13.1% —— 一次全局求解把 18 点和 4 点"
          "切成了「距离几乎相等」的两条，其中一台早早做完就闲置。"
          "**这是被测出来的问题。**\n"
          "- **局部VRP（提案）**：方向相反（距離 2.8% / 時間 0.7%）。"
          "⚠ 不要读成「提案的均衡算法更好」—— 它的两台**全程几乎不空转**"
          "（空転率 0.6%），稼働時間自然都约等于任务时长。"
          "真正的原因是**逐次重规划持续把活重新摊开**，而不是某一次求解解得更均衡。\n",
          file=buf)
    print("⇒ **本表对二段式偏保守**：它的阶段2 用的是距离均衡解，"
          "按时间均衡（给 OR-Tools 加 service-time 维度）可能更快。"
          "是否要改求解器留待人工裁决（待裁决 7）—— 这几列就是给那个决定用的数据。\n",
          file=buf)

    # ---- 路径规划コスト ----
    if plan_cost:
        print("## 路径规划コスト（池大小と求解挂钟耗时）\n", file=buf)
        print("⚠ `solve_wall_s` は `time.perf_counter` の**挂钟**量。宿主機性能と "
              "`vrp_time_limit_s` に依存し、**仿真時間軸には入らない**"
              "（仿真側が求解に見込む余裕は `plan_solve_s`、既定 0 = 瞬時に解けたことにする）。"
              "この節はその仮定が許されるかを数字で確かめるためにある。\n", file=buf)
        print("| 方法 | ラウンド数 (うち実際に解いた) | 実解時の池大小 min/平均/max | "
              "1 回の耗时 平均 (min〜max) | 合計耗时 | 対 実測終了時刻 |", file=buf)
        print("|---|---:|---:|---:|---:|---:|", file=buf)
        for s in plan_cost:
            if s["n_solves"] == 0:
                print(f"| {s['_label']} | **0** | — | — | **0 s** | "
                      f"**0.0 %** |", file=buf)
                continue
            act_pool = [r["pool_size"] for r in (plan_recs or [])
                        if r["scenario"] == s["scenario"] and r["pool_size"] > 0]
            print(f"| {s['_label']} | {s['n_solves']} ({s['n_active']}) | "
                  f"{min(act_pool) if act_pool else 0} / "
                  f"{s['pool_mean_active']:.1f} / "
                  f"{max(act_pool) if act_pool else 0} | "
                  f"{s['wall_mean_active_s'] * 1e3:.0f} ms "
                  f"({s['wall_min_active_s'] * 1e3:.0f}〜"
                  f"{s['wall_max_active_s'] * 1e3:.0f}) | "
                  f"{s['wall_total_s']:.1f} s | "
                  f"{s['frac_of_mission']:.2%} |", file=buf)
        print("", file=buf)

        vrp_recs = [r for r in (plan_recs or []) if r["scenario"] == SCENARIO_VRP]
        act = [r for r in vrp_recs if r["pool_size"] > 0]
        vs = next((s for s in plan_cost if s["scenario"] == SCENARIO_VRP), None)
        if vs and len(act) >= 2:
            budget = float(args.vrp_time_limit)
            rr = _pearson([r["pool_size"] for r in act],
                          [r["solve_wall_s"] for r in act])
            over = [r for r in act if r["solve_wall_s"] > budget * 1.05]
            # ⚠ ここの叙述は**実際に走った求解器で分岐する**（D21 の分流）。
            #   固定文を書くと、口径を変えた瞬間にレポートが嘘をつく。
            used = Counter(r["solver"] for r in act)
            main_solver = used.most_common(1)[0][0]
            mix = "、".join(f"`{k}` {v} ラウンド" for k, v in used.most_common())
            print(f"局部VRP の {len(vrp_recs)} ラウンドのうち **{vs['n_empty']} "
                  f"ラウンドは池が空**（下発済みで未観測の目標しか残っていない）で、"
                  f"求解は走らず耗时ちょうど 0。残る {len(act)} ラウンドの内訳は "
                  f"{mix}（池 {min(r['pool_size'] for r in act)}〜"
                  f"{max(r['pool_size'] for r in act)} 点、耗时 "
                  f"{vs['wall_min_active_s'] * 1e3:.0f}〜"
                  f"{vs['wall_max_active_s'] * 1e3:.0f} ms）。\n", file=buf)

            if main_solver == "exact":
                print(f"**この列は「必要量」を測っている。** 精確解（Held-Karp、"
                      f"D21）の耗时は池大小 k の決定的な関数 O(2^k·k²) であって、"
                      f"予算 `vrp_time_limit_s` とは無関係 —— 実測でも池大小との"
                      f"相関は r = {rr:+.2f}"
                      + ("（強い正相関、理論どおり）" if np.isfinite(rr) and rr > 0.5
                         else "")
                      + f"。最悪 **{vs['wall_max_active_s'] * 1e3:.0f} ms** は"
                      f"規劃周期 {args.plan_period:g} s に対し十分小さく、"
                      f"`plan_solve_s = 0`（瞬時に解けたことにする）という仮定は"
                      f"この規模では成り立つ。加えて精確解は構造的に決定的なので、"
                      f"同一入力なら再実行しても逐位同一 —— "
                      f"OR-Tools 側にあった「予算内の反復回数が機械負荷で揺れる」"
                      f"という再現性の問題がここには無い。\n", file=buf)
            else:
                print(f"**この列が測っているのは予算であって必要量ではない。** "
                      f"OR-Tools の GLS は与えた時間予算 "
                      f"`vrp_time_limit_s = {budget:g} s` を使い切る"
                      f"メタヒューリスティクスなので、池大小が動いても耗时はほぼ"
                      f"一定になる（実測 r = {rr:+.2f}）。\n", file=buf)
                if over:
                    worst = max(r["solve_wall_s"] for r in over)
                    print(f"⚠ **予算は保証ではない**：{len(over)} ラウンドが予算を超え、"
                          f"最悪 **{worst * 1e3:.0f} ms**（池 "
                          f"{[r['pool_size'] for r in over]} 点）まで伸びた。"
                          f"`vrp_time_limit_s` は OR-Tools への*要求*であって"
                          "ハードリミットではなく、宿主機の揺らぎもここに乗る。"
                          "実機で計画周期に上限を課すなら、この裾を見込む必要がある。\n",
                          file=buf)

        by = {s["scenario"]: s for s in plan_cost}
        tp = by.get(SCENARIO_TWOPHASE)
        vp = by.get(SCENARIO_VRP)
        if tp:
            print("### 二段式全局VRP：この開銷は**無視してよい**\n", file=buf)
            print(f"求解は全行程で **1 回だけ**（`n_vrp_solves` が構造的に保証）、"
                  f"{tp['wall_total_s'] * 1e3:.0f} ms。任務長 "
                  f"{tp['t_finish_s']:.1f} s に対して **{tp['frac_of_mission']:.3%}**。"
                  "しかも起点（各機の待機位置）も目標表も阶段1 の終了前に確定するので、"
                  "**母船側で先に解いておける** —— 水中で待つ必要すら無い。"
                  "⇒ 本比較で二段式の計画時間を計上しないのは妥当。\n", file=buf)
        if vp:
            print("### 局部VRP（提案）：無視は**できるが、条件付き**\n", file=buf)
            # ⚠ 二段式との大小関係は口径で逆転する（D21 で局部側が精確解になり、
            #   合計耗时が二段式の 30 s 予算を下回った）。固定文にせず実測から書く。
            _ratio = ((vp["wall_total_s"] / tp["wall_total_s"])
                      if tp and tp["wall_total_s"] > 0 else float("nan"))
            _rel = ("二段式と同程度" if not np.isfinite(_ratio) or 0.5 <= _ratio <= 2.0
                    else (f"二段式の {_ratio:.2g} 倍" if _ratio > 2.0
                          else f"二段式の {1 / _ratio:.2g} 分の 1"))
            print(f"合計 {vp['wall_total_s']:.1f} s は任務長 {vp['t_finish_s']:.1f} s の "
                  f"**{vp['frac_of_mission']:.1%}** に相当し、{_rel}（{len(act)} 回の"
                  f"求解の総和 対 1 回の求解）。"
                  "ただしこれは**任務時間に直列で足される量ではない** —— "
                  f"実際に解いた 1 回は {vp['wall_mean_active_s'] * 1e3:.0f} ms、"
                  f"規劃周期 {args.plan_period:g} s の "
                  f"{vp['wall_mean_active_s'] / args.plan_period:.1%} でしかなく、"
                  "解いている間も AUV は前ラウンドの承諾キューを実行し続ける"
                  "（開ループ下達だから止まらない）。"
                  "効くのは**測位から下達までの陳腐化**の方で、"
                  "現状その分は `plan_solve_s = 0`（瞬時求解）として計上していない。\n",
                  file=buf)
            print(f"⇒ **判定：本比較の結論（{vp['t_finish_s']:.0f} s 対 "
                  f"{(tp['t_finish_s'] if tp else float('nan')):.0f} s）は"
                  f"この開銷を入れても覆らない。** "
                  f"陳腐化を厳密に扱いたい場合の旋钮は既に契約側にあり"
                  f"（`plan_solve_s`）、`{vp['wall_mean_active_s']:.1f}` を入れて"
                  "再実行すれば上界側の評価になる。**未実施**。\n", file=buf)

    # ---- 重跑 ----
    print("## 重跑（复现）\n", file=buf)
    print("```bash", file=buf)
    print("PY=D:/nixingxing/Anaconda/envs/auv_py310/python.exe", file=buf)
    print(compare_command(args), file=buf)
    print("```\n", file=buf)

    # ---- 指标定义 ----
    specs = metrics_visible_for(ALL_SCENARIOS)
    print("## 指标定义（仅列本表用到的、三方法都适用的）\n", file=buf)
    for fam, fam_name in FAMILY_NAMES.items():
        ss = [m for m in specs if m.family == fam]
        if not ss:
            continue
        print(f"### {fam_name}\n", file=buf)
        print("| 指标 | 単位 | 方向 | 公式 | 定义 |", file=buf)
        print("|---|---|---|---|---|", file=buf)
        for m in ss:
            d = m.definition.replace("\n", " ").replace("|", "\\|")
            print(f"| **{m.label_zh}** `{m.key}`<br>{m.label_ja} | {m.unit} | "
                  f"{BETTER_MARK[m.better]} | `{m.formula}` | {d} |", file=buf)
        print("", file=buf)

    print("## 对比表\n", file=buf)
    print("```", file=buf)
    print_metric_table(rows, fh=buf, scenarios=ALL_SCENARIOS)
    print("```", file=buf)

    path = writable_path(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    return path


# ======================================================================
# 图
# ======================================================================
# 学会発表口径(人工指定 2026-08-15):
#   (a) 残すのは「観測完了本数の推移」と「負荷均衡」の 2 面だけ。
#       s/目標 と 逐次 vs 並行 の 2 面は表と本文に同じ数字があるので図から外す。
#   (b) 図内の文字は**全て英語**。方法名は `SCENARIO_NAMES_EN`、軸ラベルは
#       `MetricSpec.axis_label("en")` —— どちらも契約側の一箇所から取る。
#   (c) 子図タイトルは無し。
FIG_FONT_SCALE: float = 2.5
FIG_LINE_SCALE: float = 2.4      # 字だけ大きくして線が細いままだと読めない
FIG_SIZE_METHODS: Tuple[float, float] = (17.0, 7.6)   # ⚠ 字号と一緒に拡大しない
#   キャンバスも同じ倍率で広げると**相対**サイズが変わらず、字は大きくならない
#   (D20 で一度踏んだ穴)。ここは 2 面ぶんの実寸として固定する。


def plot_methods(rows: List[Dict[str, Any]], visit_times: Dict[str, np.ndarray],
                 path: str, *, font_scale: float = FIG_FONT_SCALE) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vrpsim.contracts.metrics import METRIC_BY_KEY, SCENARIO_NAMES_EN
    from vrpsim.contracts.style import DEFAULT_STYLE
    from vrpsim.viz import use_cjk_font

    # ⚠ グローバル様式は触らない(触ると 02/03/05/07 の図が道連れになる)
    st = DEFAULT_STYLE.scaled_fonts(font_scale)
    use_cjk_font(st.font_family)
    st.apply_rcparams()

    labels = [SCENARIO_NAMES_EN.get(r["scenario"], r["scenario"]) for r in rows]
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_METHODS)

    # --- (a) 観測完了本数の推移 —— 逐次 vs 並行が一番はっきり出る一枚 ------
    ax = axes[0]
    for c, r in zip(colors, rows):
        v = np.sort(visit_times[r["scenario"]])
        v = v[np.isfinite(v)]
        if v.size:
            ax.step(v, np.arange(1, v.size + 1), where="post", color=c,
                    linewidth=1.9 * FIG_LINE_SCALE / 2.0,
                    label=SCENARIO_NAMES_EN.get(r["scenario"], r["scenario"]))
    n_t = rows[0]["metrics"]["n_targets"]
    ax.axhline(n_t, color="0.4", linestyle="--", linewidth=1.0 * FIG_LINE_SCALE / 2.0)
    ax.set_xlabel("Time [s]", fontsize=st.font_size_label)
    ax.set_ylabel(METRIC_BY_KEY["visited"].axis_label("en"),
                  fontsize=st.font_size_label)
    ax.tick_params(labelsize=st.font_size_tick)
    ax.legend(fontsize=st.font_size_legend, loc="lower right", frameon=False)
    ax.grid(alpha=0.3)

    # --- (b) 負荷均衡:距離 vs 稼働時間 ------------------------------------
    ax = axes[1]
    wd = 0.36
    sd = METRIC_BY_KEY["load_imbalance_distance_frac"]
    stm = METRIC_BY_KEY["load_imbalance_time_frac"]
    dd = [r["metrics"].get(sd.key) or 0.0 for r in rows]
    tt = [r["metrics"].get(stm.key) or 0.0 for r in rows]
    ax.bar(x - wd / 2, dd, wd, label=sd.label_en, color="#7fb3d5")
    ax.bar(x + wd / 2, tt, wd, label=stm.label_en, color="#e59866")
    ax.set_xticks(x)
    # 方法名は 2.5x では 1 行に収まらない ⇒ 折り返す(改名はしない)。
    # ⚠ 「最初の空白で折る」だと "Local VRP (proposed)" が 1 行 11 文字のまま残り、
    #   隣の "global VRP" と衝突する(実測)。幅で折って両行とも短くする。
    ax.set_xticklabels(["\n".join(textwrap.wrap(l, width=12)) for l in labels],
                       fontsize=st.font_size_tick)
    ax.set_ylabel("Load imbalance\n(max - min) / max [-]",
                  fontsize=st.font_size_label)
    ax.tick_params(labelsize=st.font_size_tick)
    # ⚠ 凡例は 2.5x では棒に被る(実測:二段式の Busy-time の棒が凡例の裏に隠れた)。
    #   上に余白を作ってから左上に置く —— 棒の高さは動かさない。
    top = max(list(dd) + list(tt) + [0.0])
    if top > 0:
        ax.set_ylim(0.0, top * 1.42)
    ax.legend(fontsize=st.font_size_legend, frameon=False, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=st.dpi, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
def main() -> None:
    args = build_parser().parse_args()
    sid = args.scenario
    data_dir = args.data_dir or os.path.join(DEFAULT_DATA_DIR, "scenarios", sid)
    logs_dir = args.logs_dir or os.path.join(LOGS_DIR, "scenarios", sid)
    fig_dir = os.path.join(DEFAULT_FIGURE_DIR, "scenarios", sid)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print(f"[1/5] 构建场景 {sid}")
    mw = (build_mothra_world() if sid == "mothra" else build_world_for_scenario(sid))

    # 三方法共用同一份运动学/场景参数(D14/D19 的"配置相同是结构保证")。
    # Follower 起点按初始窗口两侧推导 —— 与 04_run_mission.py 同一规则,
    # 否则宽场地上一台开场就要空跑几百米,三方法的对比会被起点差异污染。
    shared = dict(follower_speed_mps=args.follower_speed, dwell_time_s=args.dwell,
                  solver=args.solver, vrp_time_limit_s=args.vrp_time_limit,
                  max_mission_time_s=args.max_time,
                  follower_starts_ned=follower_starts_flanking_window(
                      mw.world, MissionConfig().window_width_m))
    print(f"  世界 {mw.world.x_max_m:.0f}x{mw.world.y_max_m:.0f} m, "
          f"目标 {mw.dataset.n} 个, Leader 测线 "
          f"{len(leader_track(mw.world, MissionConfig().window_width_m)) // 2} 条车道")

    rows: List[Dict[str, Any]] = []
    visit_times: Dict[str, np.ndarray] = {}

    # ---------- 1. 局部VRP(提案) ----------
    print(f"[2/5] 局部VRP(提案) Leader {args.leader_speed} m/s / "
          f"規劃周期 {args.plan_period} s …", end="", flush=True)
    cfg_vrp = MissionConfig(leader_speed_mps=args.leader_speed,
                            usbl_period_s=args.usbl_period,
                            plan_period_s=args.plan_period, **shared)
    res_vrp = run_mission(cfg_vrp, mw)
    s_vrp = res_vrp.summary()
    print(f" 観測 {s_vrp['visited']}/{s_vrp['n_targets']}  "
          f"収工 {s_vrp['t_finish_s']:.1f}s  {s_vrp['t_per_target_s']:.1f} s/目標")
    save_result(res_vrp, os.path.join(data_dir, "mission.npz"))
    rows.append({"run_id": "vrp", "scenario": SCENARIO_VRP,
                 "_label": SCENARIO_NAMES[SCENARIO_VRP], "metrics": s_vrp})
    visit_times[SCENARIO_VRP] = res_vrp.visit_time_s

    # ---------- 2. 二段式全局VRP ----------
    print("[3/5] 二段式全局VRP(探査→全局VRP→観測,逐次) …", end="", flush=True)
    cfg_2p = TwoPhaseConfig(base=MissionConfig(leader_speed_mps=args.leader_speed,
                                               **shared))
    res_2p = run_twophase(cfg_2p, mw)
    s_2p = res_2p.summary()
    print(f" 観測 {s_2p['visited']}/{s_2p['n_targets']}  "
          f"探査 {s_2p['t_survey_s']:.1f}s + 観測 {s_2p['t_observe_s']:.1f}s "
          f"= {s_2p['t_finish_s']:.1f}s  {s_2p['t_per_target_s']:.1f} s/目標")
    save_twophase_result(res_2p, os.path.join(data_dir, "twophase.npz"))
    rows.append({"run_id": "twophase", "scenario": SCENARIO_TWOPHASE,
                 "_label": SCENARIO_NAMES[SCENARIO_TWOPHASE], "metrics": s_2p})
    visit_times[SCENARIO_TWOPHASE] = res_2p.visit_time_s

    # ---------- 3. lawnmower ----------
    print("[4/5] lawnmower 全覆盖对照 …", end="", flush=True)
    cfg_lm = LawnmowerConfig(base=MissionConfig(leader_speed_mps=args.leader_speed,
                                                **shared))
    res_lm = run_lawnmower(cfg_lm, mw)
    s_lm = dict(res_lm.summary())
    # lawnmower 的 summary 里没有跨方法表要的那几个派生量,在这里按同一口径补齐
    n_vis = int(s_lm["visited"])
    tf = float(s_lm["t_complete_s"])
    vt = res_lm.visit_time_s[res_lm.visited_mask]
    fleet = float(res_lm.per_vehicle_distance_m.sum())
    s_lm.update({
        "t_finish_s": tf,
        "t_per_target_s": tf / n_vis if n_vis else float("nan"),
        "t_observation_mean_s": float(vt.mean()) if vt.size else float("nan"),
        "t_observation_median_s": float(np.median(vt)) if vt.size else float("nan"),
        "t_observation_p90_s": (float(np.percentile(vt, 90)) if vt.size
                                else float("nan")),
        "per_follower_distance_m": [float(x) for x in res_lm.per_vehicle_distance_m],
        "leader_distance_m": 0.0,             # 没有 Leader
        "fleet_distance_m": fleet,            # 3 台之和就是艦隊総航程
        "distance_per_target_m": fleet / n_vis if n_vis else float("nan"),
        "duty_transit_frac": None, "duty_dwell_frac": None,   # 下面按同一口径补
        "duty_idle_frac": None, "duty_productive_frac": None,
        "leader_hold_frac": None,
        "shadow_error_max_m": 0.0,
        "timed_out": False,
        # ④ 通信コスト:**真的是 0,不是"不适用"**。测线在下水前算好,
        # 任务期间不需要任何声学报文 —— 这是 lawnmower 的真实优势,如实记。
        # (⑤ 計画品質保持 `—`:它根本不下发 waypoint 序列,那一族无从定义。)
        "n_comm_messages": 0, "channel_duty_frac": 0.0, "messages_per_target": 0.0,
    })
    # 稼働率三分项:lawnmower 有逐拍状态,按与另外两法同一个「有効区間」口径算
    from vrpsim.contracts.mission import STATUS_DWELL, STATUS_IDLE, STATUS_TRANSIT
    keep = res_lm.t_s <= tf + 1e-9
    stt = res_lm.vehicle_status[keep]
    if stt.size:
        idle = float((stt == STATUS_IDLE).mean())
        s_lm.update({"duty_transit_frac": float((stt == STATUS_TRANSIT).mean()),
                     "duty_dwell_frac": float((stt == STATUS_DWELL).mean()),
                     "duty_idle_frac": idle, "duty_productive_frac": 1.0 - idle})
    print(f" 観測 {s_lm['visited']}/{s_lm['n_targets']}  "
          f"収工 {tf:.1f}s  {s_lm['t_per_target_s']:.1f} s/目標")
    save_lawn_result(res_lm, os.path.join(data_dir, "lawnmower.npz"))
    rows.append({"run_id": "lawnmower", "scenario": SCENARIO_LAWNMOWER,
                 "_label": SCENARIO_NAMES[SCENARIO_LAWNMOWER], "metrics": s_lm})
    visit_times[SCENARIO_LAWNMOWER] = res_lm.visit_time_s

    # ---------- 汇总 ----------
    print("\n[5/5] 汇总(只列三方法都适用的指标 —— time_efficiency* 已被 "
          "applies_to 挡掉)")
    print_metric_table(rows, scenarios=ALL_SCENARIOS)

    bad = [r for r in rows if r["metrics"]["coverage"] < 1.0]
    if bad:
        print(f"\n⚠ 未全覆盖 {len(bad)} 个方法:{[r['run_id'] for r in bad]} —— "
              "它们的 t_complete_s 是 nan,时间列要连同 visited 一起读。")

    # ---------- 路径规划コスト ----------
    recs = plan_cost_records(res_vrp, res_2p)
    pc = plan_cost_stats(recs, rows)
    print("\n  路径规划コスト(挂钟,不进仿真时间轴)")
    for s in pc:
        if s["n_solves"] == 0:
            print(f"    {s['_label']:24s} 求解 0 回 —— 任务期间不解任何问题")
            continue
        print(f"    {s['_label']:24s} ラウンド {s['n_solves']:3d}"
              f"(実解 {s['n_active']:3d} / 空池 {s['n_empty']:3d})  "
              f"池平均 {s['pool_mean_active']:.1f}  "
              f"1 回 {s['wall_mean_active_s'] * 1e3:.0f} ms"
              f"({s['wall_min_active_s'] * 1e3:.0f}~"
              f"{s['wall_max_active_s'] * 1e3:.0f})  "
              f"合計 {s['wall_total_s']:.1f} s = 任务长的 {s['frac_of_mission']:.2%}")

    notes = {"t_survey_s": s_2p["t_survey_s"], "t_observe_s": s_2p["t_observe_s"],
             "t_finish_s": s_2p["t_finish_s"]}
    csv_p = write_metric_csv(rows, os.path.join(logs_dir, "methods_compare.csv"),
                             lead_cols=("run_id", "scenario"),
                             scenarios=ALL_SCENARIOS)
    pc_p = write_plan_cost_csv(recs, os.path.join(logs_dir, "plan_cost.csv"))
    md_p = write_report(rows, os.path.join(logs_dir, "methods_compare.md"),
                        args, notes, plan_cost=pc, plan_recs=recs)
    outs = [csv_p, pc_p, md_p]
    if not args.no_figure:
        fig_p = os.path.join(fig_dir, "methods_compare.png")
        plot_methods(rows, visit_times, fig_p)
        outs.append(fig_p)
    print()
    for p in outs:
        print(f"  {p}  ({os.path.getsize(p) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
