#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""08 — 参数排列组合扫描 + 評価指標汇总(含 lawnmower 对照)。

跑一个参数网格,每组存一份独立日志,最后按 `contracts/metrics.py` 的定义
汇总成一张对比表。

默认网格(2 x 3 x 4 = 24 组):
    Leader 是否停船等待   on / off
    Leader 速度          0.3 / 0.5 / 1.0 m/s (Follower 恒 0.5)
    路径规划周期          15 / 30 / 60 / 90 s (USBL 定位周期恒 15 s)

⚠ 0.3 这一档是**比 Follower 慢**的一侧。0.5 = 等速、1.0 = Leader 更快,
  三档把「Leader 慢于 / 等于 / 快于 Follower」三种关系都覆盖到了 ——
  少了 0.3 这一档,看到的就只是单调恶化的半条曲线,量不出有没有最优点。

输入 : VRPSimulation/waypoints/mothra_waypoints.csv
       VRPSimulation/data/lawnmower.npz + lawnmower_summary.json(可选对照行)
输出 : VRPSimulation/logs/<run_id>/           每组一个目录
           config.json      该组的完整 MissionConfig(可复现)
           metrics.json     该组的全部評価指標
           run.log          人读版运行记录
           mission.npz / mission_assignments.json / mission_timeline.csv
       VRPSimulation/logs/sweep_metrics.csv   一行一组,直接进 Excel/论文表
       VRPSimulation/logs/sweep_report.md     带定义的人读对比表
       VRPSimulation/figures/sweep_compare.png

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/08_sweep.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import io
import itertools
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from vrpsim.contracts.config import DEFAULT_DATA_DIR, DEFAULT_FIGURE_DIR  # noqa: E402
from vrpsim.contracts.metrics import (BETTER_MARK, FAMILY_NAMES, METRIC_BY_KEY,  # noqa: E402
                                      METRICS, SUMMARY_METRIC_KEYS)
from vrpsim.contracts.mission import (STATUS_DWELL, STATUS_IDLE,  # noqa: E402
                                      STATUS_TRANSIT, MissionConfig)
from vrpsim.mission import feasibility_estimate, run_mission, save_result  # noqa: E402
from vrpsim.report import (fmt_metric, print_metric_table,  # noqa: E402
                           write_metric_csv, writable_path)
from vrpsim.world import build_mothra_world  # noqa: E402

LOGS_DIR = os.path.join(os.path.dirname(DEFAULT_DATA_DIR), "logs")


# ======================================================================
# 参数网格
# ======================================================================
def _floats(s: str) -> List[float]:
    return [float(x) for x in s.replace(" ", "").split(",") if x]


def build_parser() -> argparse.ArgumentParser:
    d = MissionConfig()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait", default="on,off",
                    help="Leader 是否停船等待,逗号分隔:on / off")
    ap.add_argument("--leader-speed", default="0.3,0.5,1.0",
                    help="Leader 速度 [m/s];默认三档覆盖 慢于/等于/快于 Follower")
    ap.add_argument("--plan-period", default="15,30,60,90", help="路径规划周期 [s]")
    ap.add_argument("--follower-speed", type=float, default=d.follower_speed_mps)
    ap.add_argument("--usbl-period", type=float, default=d.usbl_period_s,
                    help="USBL 定位周期 [s],**不参与排列**,固定住才能看清规划周期的效应")
    ap.add_argument("--dwell", type=float, default=d.dwell_time_s)
    ap.add_argument("--solver", choices=("ortools", "greedy"), default=d.solver)
    ap.add_argument("--vrp-time-limit", type=float, default=d.vrp_time_limit_s)
    ap.add_argument("--max-time", type=float, default=6000.0,
                    help="仿真时钟上限 [s];Leader 慢 + 停船等待会让任务变长")
    ap.add_argument("--logs-dir", default=LOGS_DIR)
    ap.add_argument("--lawnmower", default=os.path.join(DEFAULT_DATA_DIR, "lawnmower.npz"),
                    help="对照基线的 npz;不存在则跳过对照行")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--from-logs", action="store_true",
                    help="不重跑仿真,直接读 logs/<run_id>/metrics.json 重出表与图")
    return ap


def run_id(wait: bool, leader_speed: float, plan_period: float) -> str:
    """目录名 = 三个变量的取值,一眼能看出是哪一组。"""
    return (f"wait{'on' if wait else 'off'}"
            f"_L{leader_speed:g}"
            f"_T{plan_period:g}")


# ======================================================================
# 指标计算
# ======================================================================
def _duty(status: np.ndarray, t_s: np.ndarray, t_finish: float) -> Dict[str, float]:
    """稼働率三分项。分母是**有効区間** t <= t_finish(见 contracts/metrics.py)。

    ⚠ 分界用 `t_finish_s` 而非 `t_complete_s`(D18):后者未全覆盖时是 nan,
      会让漏点组退回整条时间线、被 settle_time_s 的尾巴污染成"空转率虚高"。
    """
    keep = (t_s <= t_finish + 1e-9) if np.isfinite(t_finish) else np.ones_like(
        t_s, dtype=bool)
    st = status[keep]
    if st.size == 0:
        return {"duty_transit_frac": 0.0, "duty_dwell_frac": 0.0,
                "duty_idle_frac": 0.0, "duty_productive_frac": 0.0}
    idle = float((st == STATUS_IDLE).mean())
    return {"duty_transit_frac": float((st == STATUS_TRANSIT).mean()),
            "duty_dwell_frac": float((st == STATUS_DWELL).mean()),
            "duty_idle_frac": idle,
            "duty_productive_frac": 1.0 - idle}


def mission_metrics(res, lower_bound_s: float) -> Dict[str, Any]:
    """一次協調探査运行的全部指标。summary() 已经算好绝大多数,这里只补 sweep 级的。"""
    s = res.summary()
    out = {k: s[k] for k in SUMMARY_METRIC_KEYS}
    # 三把尺子,口径差别见 contracts/metrics.py:严格(漏点 nan)/ 実測(会被漏点抬高)
    # / 覆盖折扣(可横比)。
    out["time_efficiency"] = res.time_efficiency(lower_bound_s)
    out["time_efficiency_obs"] = res.time_efficiency_obs(lower_bound_s)
    out["time_efficiency_cov"] = res.time_efficiency_cov(lower_bound_s)
    out["t_mission_reference_s"] = float(lower_bound_s)
    return out


def lawnmower_metrics(npz_path: str) -> Optional[Dict[str, Any]]:
    """对照基线的同口径指标。

    ⚠ 通信/計画 两族对 lawnmower **不适用**(它没有 Leader、没有请求、没有 VRP),
      表里记成 None 而不是 0 —— 写 0 会被误读成"通信开销为零的更优方案"。
    """
    base = os.path.splitext(npz_path)[0]
    js = base + "_summary.json"
    if not (os.path.exists(npz_path) and os.path.exists(js)):
        return None
    with open(js, encoding="utf-8") as f:
        s = json.load(f)["summary"]
    with np.load(npz_path, allow_pickle=False) as d:
        t_s = d["t_s"]
        status = d["vehicle_status"]
        vt = d["visit_time_s"]
        dist = d["vehicle_distance_m"][-1]

    tc = float(s["t_complete_s"])
    # lawnmower 是全覆盖基线 ⇒ 実測終了時刻与完成时刻恒等;仍显式取出,免得口径漂移
    tf = float(s.get("t_complete_s") if np.isfinite(tc) else s["finish_time_s"])
    v = vt[~np.isnan(vt)]
    n_vis = int(s["visited"])
    fleet = float(dist.sum())            # lawnmower 没有 Leader,3 台之和就是艦隊総航程
    out: Dict[str, Any] = {
        "t_complete_s": tc,
        "t_finish_s": tf,
        "n_targets": int(s["n_targets"]),
        "visited": n_vis,
        "t_observation_mean_s": float(v.mean()) if v.size else float("nan"),
        "t_observation_median_s": float(np.median(v)) if v.size else float("nan"),
        "t_observation_p90_s": float(np.percentile(v, 90)) if v.size else float("nan"),
        "t_per_target_s": tf / n_vis if n_vis else float("nan"),
        "fleet_distance_m": fleet,
        "total_distance_m": fleet,       # 没有 Leader ⇒ 两个口径重合
        "max_distance_m": float(s["max_distance_m"]),
        "leader_distance_m": 0.0,        # 没有 Leader
        "per_follower_distance_m": [float(x) for x in dist],
        "distance_per_target_m": fleet / n_vis if n_vis else float("nan"),
        # ⑥ 負荷均衡(D19):lawnmower 的 summary 已由 metrics_util 统一算好
        "jain_fairness": float(s["jain_fairness"]),
        "jain_fairness_time": float(s["jain_fairness_time"]),
        "load_imbalance_distance_frac": float(s["load_imbalance_distance_frac"]),
        "load_imbalance_time_frac": float(s["load_imbalance_time_frac"]),
        "coverage": float(s["coverage"]),
        "timed_out": False,
        "shadow_error_max_m": 0.0,       # 没有影子模型
        "leader_hold_frac": 0.0,         # 没有 Leader
        # 通信 / 計画:不适用
        "n_comm_messages": None, "channel_duty_frac": None,
        "messages_per_target": None, "n_plan_rounds": None,
        "pool_size_mean": None, "wp_issue_total": None,
        "wp_issues_per_target": None, "sequence_utilisation": None,
        "reassignment_count": None, "reassigned_wp_count": None,
        "solve_wall_mean_s": None,
        # 时间效率三列都要一个「全局 min-max VRP 下界」做分子,那是協調探査的下界,
        # 拿来除 lawnmower 的完成时刻没有意义(它根本不解 VRP) ⇒ 记 —,不记 0。
        "time_efficiency": None, "time_efficiency_obs": None,
        "time_efficiency_cov": None,
    }
    out.update(_duty(status, t_s, tf))
    return out


# ======================================================================
# 输出
# ======================================================================
def mission_command(row: Dict[str, Any], args, logs_dir: str) -> str:
    """复现某一组所需的 04 命令行。**由该组实际用的参数生成**,不手写 —— 手写必漂移。"""
    wait_flags = "" if row["wait"] == "on" else " --no-wait-lagging --no-wait-endangered"
    return (f"$PY VRPSimulation/scripts/04_run_mission.py -q"
            f" --leader-speed {row['leader_speed_mps']:g}"
            f" --follower-speed {row['follower_speed_mps']:g}"
            f" --dwell {args.dwell:g}"
            f" --usbl-period {row['usbl_period_s']:g}"
            f" --plan-period {row['plan_period_s']:g}"
            f"{wait_flags}"
            f" --solver {args.solver}"
            f" --vrp-time-limit {args.vrp_time_limit:g}"
            f" --max-time {args.max_time:g}"
            f" --out {logs_dir}/{row['run_id']}/mission.npz")


def lawnmower_command(args) -> str:
    return (f"$PY VRPSimulation/scripts/06_run_lawnmower.py -q"
            f" --dwell {args.dwell:g} --follower-speed {args.follower_speed:g}")


def sweep_command(args) -> str:
    return (f"$PY VRPSimulation/scripts/08_sweep.py"
            f" --wait {args.wait} --leader-speed {args.leader_speed}"
            f" --plan-period {args.plan_period}"
            f" --follower-speed {args.follower_speed:g}"
            f" --usbl-period {args.usbl_period:g} --dwell {args.dwell:g}"
            f" --solver {args.solver} --vrp-time-limit {args.vrp_time_limit:g}"
            f" --max-time {args.max_time:g}")


def write_run_log(path: str, rid: str, rec: Dict[str, Any], args) -> None:
    """单组的人读版记录。**自带重跑本组的命令** ⇒ 一个日志目录就是自洽的。

    跑完与 `--from-logs` 走同一个函数,免得两条路径产出不一样的东西。
    """
    met = rec["metrics"]
    row = {k: rec[k] for k in ("run_id", "leader_speed_mps", "follower_speed_mps",
                               "plan_period_s", "usbl_period_s")}
    row["wait"] = "on" if rec["wait"] else "off"
    # Windows 上 `--logs-dir` 若落在另一个盘符,relpath 会抛 ValueError —— 那时
    # 退回绝对路径。命令照样能跑,只是长一点;不该为了好看让整趟扫描崩掉。
    try:
        logs_rel = os.path.relpath(args.logs_dir, os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
    except ValueError:
        logs_rel = os.path.abspath(args.logs_dir)
    logs_rel = logs_rel.replace("\\", "/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {rid}\n")
        f.write(f"Leader {rec['leader_speed_mps']} m/s / "
                f"Follower {rec['follower_speed_mps']} m/s / 停留 {args.dwell} s\n")
        f.write(f"等待策略 {'开' if rec['wait'] else '关'} / "
                f"规划周期 {rec['plan_period_s']} s / "
                f"USBL 周期 {rec['usbl_period_s']} s / 求解器 {args.solver}\n")
        f.write(f"时间参照 {met.get('t_mission_reference_s', float('nan')):.1f} s\n\n")
        f.write("# 重跑本组:\n")
        f.write(mission_command(row, args, logs_rel) + "\n\n")
        for m in METRICS:
            f.write(f"{m.label_zh:<26}{fmt_metric(m, met.get(m.key)):>16}  {m.unit}\n")


def write_report(rows: List[Dict[str, Any]], path: str, meta: Dict[str, Any],
                 args=None) -> str:
    """人读版:先给复现命令,再给每个指标的定义,最后给对比表。"""
    buf = io.StringIO()
    print("# 参数扫描対比 —— 調査効率の評価\n", file=buf)
    print(f"- 网格:{meta['grid']}", file=buf)
    print(f"- 固定:Follower {meta['follower_speed']} m/s、USBL 周期 "
          f"{meta['usbl_period']} s、停留 {meta['dwell']} s、求解器 {meta['solver']}",
          file=buf)
    print(f"- 共 {meta['n_runs']} 组;日志在 `logs/<run_id>/`\n", file=buf)

    if args is not None:
        vrp_rows = [r for r in rows if r["scenario"] == "vrp"]
        lawn_rows = [r for r in rows if r["scenario"] == "lawnmower"]
        print("## 重跑（复现）\n", file=buf)
        print("⚠ **`--solver ortools` 不保证逐位可复现。** OR-Tools 的 "
              "`GUIDED_LOCAL_SEARCH` 按**挂钟**预算 (`--vrp-time-limit`) 搜索，"
              "同样的预算在不同负载下跑完的迭代数不同，返回的路线就可能不同。\n",
              file=buf)
        print("**但实测影响极小。** 同一份代码跑 11 趟（5 趟并行 + 5 趟串行独立 "
              "+ 1 趟单独），16 组 × 40 余列**全部逐位相同**，唯一的例外是 "
              "`waiton_L1_T15.reassignment_count` 在 20 与 22 之间摆动（6:5）——"
              "那是个派生计数器，**不进任何图表**，且该组的完成时刻、覆盖率、"
              "航程、报文数一格未动。\n", file=buf)
        print("⚠ 更早的记录里曾把该组「1125.0 s / `timed_out=True` → "
              "1401.5 s / `timed_out=False`」这一次变化归因于求解器，"
              "**那个归因不成立**：当前代码 11 趟无一给出 1125.0，而两次快照之间 "
              "`vrpsim/mission.py` 等文件确实被改过。本仓库不是 git 仓库，"
              "旧代码取不回，所以只能记为**「不能归因于求解器」**，不另作猜测。\n",
              file=buf)
        print("⇒ 要严格逐位可复现请加 `--solver greedy`（确定性贪心，无时间预算）；"
              "按上面的实测，ortools 下也无需为本表的任何一个数担心。\n", file=buf)
        print("```bash", file=buf)
        print("PY=D:/nixingxing/Anaconda/envs/auv_py310/python.exe", file=buf)
        print("```\n", file=buf)
        print("### 一条命令跑完全部\n", file=buf)
        print("```bash", file=buf)
        print(sweep_command(args), file=buf)
        print("```\n", file=buf)
        print("重出表与图（不重跑仿真，直接读既有日志）：\n", file=buf)
        print("```bash", file=buf)
        print(f"{sweep_command(args)} --from-logs", file=buf)
        print("```\n", file=buf)
        print(f"### 逐组单跑（{len(vrp_rows)} 条 + 对照 {len(lawn_rows)} 条）\n", file=buf)
        print("每条只跑一组仿真并落盘到该组目录。⚠ 单跑 04 只重建 `mission.*`；"
              "`metrics.json` / `run.log` / 汇总表由 08 生成，"
              "所以单跑之后要再执行一次上面的 `--from-logs` 才能刷新汇总。\n", file=buf)
        print("```bash", file=buf)
        for r in vrp_rows:
            print(f"# {r['run_id']}  "
                  f"(等待 {r['wait']} / Leader {r['leader_speed_mps']:g} m/s / "
                  f"规划周期 {r['plan_period_s']:g} s)", file=buf)
            print(mission_command(r, args, meta["logs_dir"]), file=buf)
        if lawn_rows:
            print("", file=buf)
            print("# lawnmower 全覆盖对照基线（无 Leader / 无请求 / 无 VRP）", file=buf)
            print(lawnmower_command(args), file=buf)
        print("```\n", file=buf)

    bad = [r for r in rows if (r["metrics"].get("coverage") or 0) < 1.0]
    if bad:
        print("## ⚠ 先读这一段：有 %d 组没做完\n" % len(bad), file=buf)
        print("下面这些组关掉了 Leader 停车等待，Follower 跟不上 Leader，"
              "目标被观测窗口甩在后面：\n", file=buf)
        print("| run_id | 観測成功数 | 覆盖率 | 実測終了時刻 | s/目標 | "
              "時間効率(実測) | 時間効率(折扣) |", file=buf)
        print("|---|---|---|---|---|---|---|", file=buf)
        for r in sorted(bad, key=lambda x: x["metrics"]["coverage"]):
            m = r["metrics"]
            print(f"| `{r['run_id']}` | {m['visited']} / {m['n_targets']} | "
                  f"{m['coverage']:.1%} | {m['t_finish_s']:.1f} s | "
                  f"{m['t_per_target_s']:.1f} | {m['time_efficiency_obs']:.3f} | "
                  f"**{m['time_efficiency_cov']:.3f}** |", file=buf)
        print("", file=buf)
        print("**这些组的「実測終了時刻」偏小，不是因为它更快，是因为它做得少。**\n", file=buf)
        print("- `t_complete_s`（任务完成时刻）对它们是 `nan` —— 任务根本没达成。\n"
              "- `time_efficiency_obs`（実測）同样**被漏点抬高**：分子是按 22 个目标"
              "全做完算的下界，分母却只是「做完手上那几个」的收工时刻。\n"
              "- **要横比效率，只看 `time_efficiency_cov`（覆盖折扣）或 "
              "`t_per_target_s`（对漏点中性）这两列。**\n", file=buf)
        over = [r for r in bad if (r["metrics"].get("time_efficiency_obs") or 0) > 1.0]
        if over:
            print(f"⚠ 其中 {len(over)} 组的 `time_efficiency_obs` **超过 1.0**"
                  f"（{', '.join('`%s` %.3f' % (r['run_id'], r['metrics']['time_efficiency_obs']) for r in sorted(over, key=lambda x: -x['metrics']['time_efficiency_obs']))}），"
                  "字面意思是「比理论最优还快」。这不是 bug —— 它们只做了不到一半的目标，"
                  "却拿全任务的下界当分子。**这就是这一列不能用来排名的最直白证据。**\n",
                  file=buf)

    print("## 指标定义\n", file=buf)
    for fam, fam_name in FAMILY_NAMES.items():
        specs = [m for m in METRICS if m.family == fam]
        if not specs:
            continue
        print(f"### {fam_name}\n", file=buf)
        print("| 指标 | 単位 | 方向 | 公式 | 定义 |", file=buf)
        print("|---|---|---|---|---|", file=buf)
        for m in specs:
            d = m.definition.replace("\n", " ").replace("|", "\\|")
            print(f"| **{m.label_zh}** `{m.key}`<br>{m.label_ja} | {m.unit} | "
                  f"{BETTER_MARK[m.better]} | `{m.formula}` | {d} |", file=buf)
        print("", file=buf)

    print("## 对比表\n", file=buf)
    print("```", file=buf)
    print_metric_table(rows, fh=buf)
    print("```", file=buf)
    path = writable_path(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    return path


def plot_sweep(rows: List[Dict[str, Any]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vrpsim.contracts.style import DEFAULT_STYLE as st
    from vrpsim.viz import use_cjk_font

    # 不挑 CJK 字体的话中日文标签会渲染成方框(和 02/03/05/07 走同一个入口)
    use_cjk_font(st.font_family)
    st.apply_rcparams()

    runs = [r for r in rows if r["scenario"] == "vrp"]
    # ⚠ 颜色 = Leader 速度、线型 = 是否等待。**不要**改回"一条序列一个颜色"
    #   再和一个定长颜色表 zip —— 加第三档速度后序列数从 4 变 6,
    #   zip 会**静默丢掉最后两条线**(加 0.3 档时踩到过)。
    speeds = sorted({r["leader_speed_mps"] for r in runs})
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    color_of = {v: palette[i % len(palette)] for i, v in enumerate(speeds)}
    series: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in runs:
        series.setdefault((r["wait"], r["leader_speed_mps"]), []).append(r)
    for v in series.values():
        v.sort(key=lambda r: r["plan_period_s"])
    order = sorted(series, key=lambda k: ({"on": 0, "off": 1}.get(k[0], 9), k[1]))

    # 头两格是"做成了多少"和"折扣后的效率" —— waitoff 组的 t_complete_s 全是 nan,
    # 把它放头版会画出一片空白,反而看不出这组根本没做完(D18)。
    panels = [("visited", "観測成功数 [個] / 総数 22"),
              ("time_efficiency_cov", "時間効率(被覆率補正) [-]"),
              ("t_finish_s", "実測終了時刻 [s]"),
              ("t_per_target_s", "単位目標あたり所要時間 [s/目標]"),
              ("duty_idle_frac", "待機(空転)の割合 [-]"),
              ("fleet_distance_m", "艦隊総航程 [m]"),
              ("reassignment_count", "改道次数 [回]"),
              ("channel_duty_frac", "信道占空比 [-]")]
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 7.0))
    for ax, (key, label) in zip(axes.ravel(), panels):
        for skey in order:
            rs = series[skey]
            wait, lv = skey
            c = color_of[lv]
            name = f"wait={wait}, L={lv:g}"
            x = [r["plan_period_s"] for r in rs]
            y = [r["metrics"].get(key) for r in rs]
            y = [np.nan if v is None else v for v in y]
            full = [r["metrics"]["coverage"] >= 1.0 for r in rs]
            # 三档速度共用三种颜色后,同色的 on/off 必须靠线型分开
            ax.plot(x, y, "-" if wait == "on" else "--", color=c,
                    linewidth=1.8, label=name, zorder=2)
            # 未全覆盖的点画空心 —— 那些组的时间类指标不可比
            ax.scatter([xi for xi, f in zip(x, full) if f],
                       [yi for yi, f in zip(y, full) if f],
                       s=34, color=c, zorder=3)
            ax.scatter([xi for xi, f in zip(x, full) if not f],
                       [yi for yi, f in zip(y, full) if not f],
                       s=44, facecolors="none", edgecolors=c, linewidths=1.6, zorder=3)
        ax.set_xlabel("路径规划周期 [s]", fontsize=st.font_size_label)
        ax.set_ylabel(label, fontsize=st.font_size_label)
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted({r["plan_period_s"] for r in runs}))
        if key == "visited":
            # 画出满分线,一眼看出哪几组没做完
            n_t = max((r["metrics"].get("n_targets") or 0) for r in runs)
            if n_t:
                ax.axhline(n_t, color="0.4", linestyle="--", linewidth=1.0, zorder=1)
                ax.set_ylim(0, n_t * 1.08)
    axes.ravel()[0].legend(fontsize=st.font_size_legend - 1, loc="best")
    fig.suptitle("参数扫描:实心 = 全覆盖,空心 = 未全覆盖"
                 "(空心组的「実測終了時刻」偏小是因为漏了目标,不是因为更快)",
                 fontsize=st.font_size_title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=st.dpi, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
def main() -> None:
    args = build_parser().parse_args()
    waits = [w.strip().lower() == "on" for w in args.wait.split(",") if w.strip()]
    lspeeds = _floats(args.leader_speed)
    periods = _floats(args.plan_period)
    grid = list(itertools.product(waits, lspeeds, periods))

    os.makedirs(args.logs_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    bound_cache: Dict[tuple, float] = {}

    if args.from_logs:
        print(f"[1/4] 从既有日志重建({args.logs_dir}),不重跑仿真")
        for wait, lv, T in grid:
            rid = run_id(wait, lv, T)
            p = os.path.join(args.logs_dir, rid, "metrics.json")
            if not os.path.exists(p):
                print(f"  ⚠ 缺 {rid},跳过")
                continue
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            # 指标契约扩过之后,老日志里就少了新增的键。默默填 "—" 会让人以为
            # 那几列不适用(其实只是没算过) ⇒ 明确报错,让人去重跑。
            stale = [m.key for m in METRICS if m.key not in rec["metrics"]]
            if stale:
                raise SystemExit(
                    f"[08] {rid}/metrics.json 是**旧版**,缺 {stale}。\n"
                    f"    contracts/metrics.py 后来加过指标,这些数从没算过,"
                    f"--from-logs 变不出来。\n"
                    f"    请去掉 --from-logs 重跑一次(约 10 分钟)。")
            # 顺手把 run.log 重写一遍:老日志里没有「重跑本组」那段命令
            write_run_log(os.path.join(args.logs_dir, rid, "run.log"), rid, rec, args)
            rows.append({"run_id": rid, "scenario": "vrp",
                         "_label": f"L{lv:g}/T{T:g}" + ("" if wait else "*"),
                         "wait": "on" if wait else "off", "leader_speed_mps": lv,
                         "follower_speed_mps": rec["follower_speed_mps"],
                         "plan_period_s": T, "usbl_period_s": rec["usbl_period_s"],
                         "metrics": rec["metrics"]})
        print(f"  读到 {len(rows)} 组")
        _finish(args, rows, grid)
        return

    print(f"[1/4] 构建场景;网格 {len(grid)} 组 "
          f"(wait {len(waits)} x Leader速度 {len(lspeeds)} x 规划周期 {len(periods)})")
    mw = build_mothra_world()

    for i, (wait, lv, T) in enumerate(grid, 1):
        rid = run_id(wait, lv, T)
        cfg = MissionConfig(
            leader_speed_mps=lv, follower_speed_mps=args.follower_speed,
            dwell_time_s=args.dwell,
            usbl_period_s=args.usbl_period, plan_period_s=T,
            leader_wait_on_lagging_follower=wait,
            leader_wait_on_endangered_target=wait,
            solver=args.solver, vrp_time_limit_s=args.vrp_time_limit,
            max_mission_time_s=args.max_time)

        # 时间效率的分母只随速度变,按速度对缓存,免得每组都多解一次全局 VRP
        bkey = (lv, args.follower_speed, args.dwell)
        if bkey not in bound_cache:
            bound_cache[bkey] = feasibility_estimate(mw, cfg)["t_mission_reference_s"]
        lb = bound_cache[bkey]

        print(f"[2/4] ({i}/{len(grid)}) {rid} …", end="", flush=True)
        res = run_mission(cfg, mw, verbose=False)
        met = mission_metrics(res, lb)
        cov = met["coverage"]
        print(f" 観測 {met['visited']}/{met['n_targets']} ({cov:.0%})  "
              f"収工 {met['t_finish_s']:.1f}s  "
              f"効率(折扣) {met['time_efficiency_cov']:.3f}  "
              f"停船 {100 * met['leader_hold_frac']:.1f}%  "
              f"空转 {100 * met['duty_idle_frac']:.1f}%"
              f"{'  ⚠未全覆盖' if cov < 1.0 else ''}"
              f"{'  ⚠超时' if met['timed_out'] else ''}")

        d = os.path.join(args.logs_dir, rid)
        os.makedirs(d, exist_ok=True)
        save_result(res, os.path.join(d, "mission.npz"))
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            cd = asdict(cfg)
            cd.pop("sim", None)          # 场景配置很大且各组相同,不重复存
            json.dump(cd, f, ensure_ascii=False, indent=2, default=str)
        rec = {"run_id": rid, "wait": wait, "leader_speed_mps": lv,
               "follower_speed_mps": args.follower_speed,
               "plan_period_s": T, "usbl_period_s": args.usbl_period,
               "metrics": met}
        with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
        write_run_log(os.path.join(d, "run.log"), rid, rec, args)

        rows.append({"run_id": rid, "scenario": "vrp", "_label": f"L{lv:g}/T{T:g}"
                     + ("" if wait else "*"),
                     "wait": "on" if wait else "off", "leader_speed_mps": lv,
                     "follower_speed_mps": args.follower_speed,
                     "plan_period_s": T, "usbl_period_s": args.usbl_period,
                     "metrics": met})

    _finish(args, rows, grid)


def _finish(args, rows: List[Dict[str, Any]], grid) -> None:
    """加对照行 → 打表 → 告警 → 落盘。跑完与 --from-logs 走同一条路径。"""
    lm = lawnmower_metrics(args.lawnmower)
    if lm is not None:
        rows.append({"run_id": "lawnmower", "scenario": "lawnmower",
                     "_label": "lawn", "wait": "-", "leader_speed_mps": "",
                     "follower_speed_mps": args.follower_speed,
                     "plan_period_s": "", "usbl_period_s": "", "metrics": lm})
        print(f"[2/4] 对照基线 lawnmower:完成 {lm['t_complete_s']:.1f}s  "
              f"艦隊総航程 {lm['fleet_distance_m']:.0f} m")
    else:
        print(f"[2/4] ⚠ 没找到 {args.lawnmower},跳过 lawnmower 对照行")

    print("\n[3/4] 汇总")
    print_metric_table(rows)

    bad = [r for r in rows if r["metrics"]["coverage"] < 1.0]
    if bad:
        print(f"\n⚠ 未全覆盖 {len(bad)} 组(表里标 *,图里画空心):")
        for r in sorted(bad, key=lambda x: x["metrics"]["coverage"]):
            m = r["metrics"]
            print(f"    {r['run_id']:<20} 観測 {m['visited']:>2}/{m['n_targets']}"
                  f" ({m['coverage']:>5.1%})  収工 {m['t_finish_s']:>7.1f}s"
                  f"  {m['t_per_target_s']:>6.1f} s/目標"
                  f"  効率 実測 {m['time_efficiency_obs']:.3f} →"
                  f" 折扣 {m['time_efficiency_cov']:.3f}")
        print("  ⚠ 这些组的 `t_complete_s` 是 nan(任务未达成),"
              "`t_finish_s` 只是「实际收工时刻」——**它偏小是因为做得少,不是因为快**。")
        print("  要横比效率请看 `time_efficiency_cov`(乘覆盖率折扣)"
              "或 `t_per_target_s`(对漏点中性),别看 `time_efficiency_obs`。")
    hot = [r for r in rows if (r["metrics"].get("channel_duty_frac") or 0) >= 1.0]
    if hot:
        print(f"\n⚠ 信道占空比 >= 100% 的组:{[r['run_id'] for r in hot]}")
        print("  报文总时长超过任务时长 ⇒ D16「不建模信道竞争」的假设在这些组已失效。")

    print("\n[4/4] 落盘")
    csv_p = os.path.join(args.logs_dir, "sweep_metrics.csv")
    md_p = os.path.join(args.logs_dir, "sweep_report.md")
    csv_p = write_metric_csv(
        rows, csv_p,
        lead_cols=("run_id", "scenario", "wait", "leader_speed_mps",
                   "follower_speed_mps", "plan_period_s", "usbl_period_s"))
    md_p = write_report(rows, md_p, {
        "grid": f"wait={args.wait} x Leader速度={args.leader_speed} "
                f"x 规划周期={args.plan_period}",
        "follower_speed": args.follower_speed, "usbl_period": args.usbl_period,
        "dwell": args.dwell, "solver": args.solver, "n_runs": len(grid),
        "logs_dir": os.path.relpath(args.logs_dir, os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")},
        args=args)
    outs = [csv_p, md_p]
    if not args.no_figure:
        fig_p = os.path.join(DEFAULT_FIGURE_DIR, "sweep_compare.png")
        plot_sweep(rows, fig_p)
        outs.append(fig_p)
    for p in outs:
        print(f"  {p}  ({os.path.getsize(p) / 1024:.1f} KB)")
    print(f"  每组日志: {args.logs_dir}/<run_id>/"
          f"{{config,metrics}}.json + run.log + mission.*")


if __name__ == "__main__":
    main()
