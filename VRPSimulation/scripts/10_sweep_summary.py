#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""10 — 参数扫描的**论文用**汇总:多次重复 → 均值 ± 标准差 → 正文图 + 正文表。

与 08_sweep.py 的分工
================================================================================
`08_sweep.py`  跑**一趟** 24 组,出全 40 余列的工作用表与 2x4 八面板诊断图。
`10_sweep_summary.py`(本文件)  **不跑仿真**,只读若干趟的 `sweep_metrics.csv`,
                聚合成「均值 ± 标准差」,出**收窄到 4 个面板**的正文图与两块正文表。

⚠ 重复批必须**串行独立**跑,不能并行
================================================================================
ortools 的元启发式在 `vrp_time_limit_s` 这个**挂钟**预算内跑多少轮迭代,
取决于跑的时候机器有多忙。5 个 rep 并行 ⇒ 它们共享同一负载 ⇒ 迭代数相同 ⇒
解逐位相同,std 恒为 0。**那不是「可复现」,那是样本相关。**
实测:5 个并行 rep 与单独跑的那趟在 `waiton_L1_T15.reassignment_count` 上
差了一格(20 vs 22),其余全部逐位相同 —— 方差确实存在,只是并行把它压掉了。
⇒ 生成重复批请用 `logs/reps/run_seq.sh`(一趟接一趟),别用 `&` 并发。

输入 : 若干个 `<reps-dir>/<prefix>K/sweep_metrics.csv`(由 08_sweep.py 产出)
输出 : VRPSimulation/logs/sweep_summary.md    正文表(按 wait 拆两块)+ 定义 + 限制
       VRPSimulation/logs/sweep_summary.csv   一行一组,每指标 mean/std 两列
       VRPSimulation/figures/sweep_summary.png 正文图(2x2,均值线 + ±std 带)
       VRPSimulation/figures/sweep_summary_leader_speed.png 補足図(横轴 = Leader 速度)

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/10_sweep_summary.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import io
import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vrpsim.contracts.config import DEFAULT_DATA_DIR, DEFAULT_FIGURE_DIR  # noqa: E402
from vrpsim.contracts.metrics import METRIC_BY_KEY  # noqa: E402
from vrpsim.report import writable_path  # noqa: E402

LOGS_DIR = os.path.join(os.path.dirname(DEFAULT_DATA_DIR), "logs")

# ======================================================================
# 实验目的 —— 论文正文那一句。放在这里而不是散在 md 里,是为了让
# 「图/表/正文」三处引用同一个字符串,改一次就全改。
# ======================================================================
PURPOSE_ZH = (
    "系统扫描 Leader 停船等待有无 x Leader 巡航速度 x 路径重规划周期三因子"
    "(2x3x4 = 24 组;Leader 速度三档覆盖**慢于 / 等于 / 快于** Follower 三种关系),"
    "厘清提案的局部VRP 協調探査方式中,**観測完遂性(被覆率)与調査時間効率"
    "分别由哪一个运用参数支配**,并给出各参数的作用机理、代价与最优点。")
PURPOSE_JA = (
    "Leader の停船待機の有無・Leader 巡航速度・経路再計画周期の 3 因子"
    "(2x3x4 = 24 条件。Leader 速度は Follower より**遅い / 等しい / 速い**の"
    "3 水準)を系統的に振り、提案する局所 VRP 協調探査方式において"
    "**目標観測の完遂性(被覆率)と調査時間効率がどの運用パラメータに支配されるか**を、"
    "その作用機序・代償・最適点とともに明らかにする。")

# ======================================================================
# 指标选取 —— 原则是「每个被扫的因子,配一个能解释它的指标」,
# 不是「每族凑一个」。理由逐条写在 SELECTION_NOTES 里,会进 md。
# ======================================================================
FIGURE_METRICS: Tuple[str, ...] = (
    "visited",                        # wait 因子的全部效应都在这一列
    "t_per_target_s",                 # 24 组唯一都有定义、且对漏点中性的时间列
    "leader_hold_frac",               # Leader 停船率:leader_speed 因子的代价
    "load_imbalance_distance_frac",   # Follower 移動路径距离的均衡
)
# 正文表比图多几列:t_complete_s(达成判定)与解释 plan_period 的三列。
TABLE_METRICS: Tuple[str, ...] = (
    "t_complete_s", "visited", "t_per_target_s", "duty_idle_frac",
    "sequence_utilisation", "leader_hold_frac", "channel_duty_frac",
    "load_imbalance_distance_frac",
)

SELECTION_NOTES: Tuple[Tuple[str, str], ...] = (
    ("visited",
     "wait 因子的效应几乎全在这一列:L>=0.5 时 on 恒 22/22、off 掉到 36.4%~95.5%;"
     "但 L=0.3(比 Follower 慢)时 on/off 都是 22/22 —— **wait 在那一档失效**。"
     "不放它,waitoff 组「早收工」会被误读成更快。"),
    ("t_per_target_s",
     "24 组里唯一**都有定义**、且对漏点中性的时间列 —— t_complete_s 有 8 组是 nan。"),
    ("duty_idle_frac",
     "plan_period 的**直接代价**。⚠ **只在 L>=0.5 的 4 条序列上单调递增**;"
     "L=0.3 的两条是 12.2/8.3/17.6/22.6,**非单调** —— 那一档的空转主要来自"
     "「等 Leader 把下一批目标扫出来」而不是「等下一个规划轮」,"
     "T=30 反而比 T=15 更能凑成整批。"),
    ("sequence_utilisation",
     "plan_period 的**作用机理**,**六条序列全部单调递增**:T 小则序列被反复覆盖,"
     "T=15 时 62%~91% 的下发点次白费(L=1.0 最浪费,只有 8.8% 转化成観測)。"),
    ("leader_hold_frac",
     "leader_speed 因子的**代价**,★图:L=0.3 恒 **0.0%**(慢于 Follower,永远追不开,"
     "等待逻辑一次都不触发)、L=0.5 为 4.6%~18.3%、L=1.0 为 **44.0%~62.1%** —— "
     "提速后一半以上时间在原地干等,编队速度由 Follower 封顶。"
     "wait=off 的 12 组恒为 0(不等待就没有停船),所以这一列只在 wait=on 内部有区分度。"),
    ("load_imbalance_distance_frac",
     "**Follower 移動路径距离的均衡**,★图。⚠ 同一件事有两种写法,这里用的是"
     "**灵敏的那种**:`jain_fairness`(距離均衡度)在这 24 组里落在 0.9957~1.0000,"
     "画出来是一条平线,n=2 时 Jain 太钝;`(max-min)/max` 是 0.04%~12.4%,"
     "才看得出差别。**它越小越均衡** —— 与「均衡度」方向相反,读图时注意。"),
    ("channel_duty_frac",
     "④ 通信族里唯一干净的一列:实测它几乎**只是 plan_period 的函数**"
     "(0.67/0.60/0.57/0.56,**六条序列**在三位小数上都落进同一组值),"
     "且全组 56%~67%,逼近「不建模信道竞争」这条假设的边界。"
     "n_comm_messages 被任务时长与规划轮数两个因素混叠、不单调,故不选。"),
)

# 挂钟量:进不了仿真时间轴,趟趟不同是正常的,不能算"非确定性"。
WALLCLOCK_COLS: Tuple[str, ...] = ("solve_wall_mean_s",)

# 图的字号相对 contracts/style.py 默认值放大多少倍(学术汇报口径)。
# 轴标签单独一档:标签本来就最长,和标题刻度一起放到 3x 会顶满整幅画。
# 走 FigureStyle.scaled_fonts() + with_overrides(),**不改全局样式** ——
# 改了 02/03/05/07 的图会跟着变。
FONT_SCALE: float = 3.0        # 标题 / 刻度 / 图例 / 图内标注
LABEL_FONT_SCALE: float = 3.0  # 轴标签(与上面同档;两档分离的机制保留备用)

# 画布尺寸(英寸)。⚠ **不随字号缩放**:早先按 font_scale 同比放大画布,
# 结果字与图的**相对**大小纹丝不动 —— 等于没放大。要让字真的变大,画布必须固定。
FIG_SIZE_MAIN: Tuple[float, float] = (19.0, 15.5)
FIG_SIZE_SPEED: Tuple[float, float] = (19.0, 8.6)
LINE_SCALE: float = 2.2        # 线宽 / 标记随字号一起加粗,免得字大线细

EXCLUDED_NOTES: Tuple[Tuple[str, str], ...] = (
    ("③ 移動効率(整族)",
     "24 组 fleet_distance_m 落在 1559~1901 m,几乎恒定(加 L=0.3 档后区间未变)。"
     "**「参数只改变时间与通信,不改变能耗」本身就是一句结论**,一行字写掉即可,"
     "不值得占一个面板。"),
    ("time_efficiency / _obs",
     "前者 8 组 nan;后者被漏点抬到 1.02~1.04(「比理论最优还快」),"
     "contracts/metrics.py 里已写明不能用于排名。"),
    ("⑥ 負荷均衡(整族)",
     "本扫描内 Jain 落在 0.996~1.000,无区分度。它是给三方法比较(09)用的。"),
)


# ======================================================================
# 聚合
# ======================================================================
@dataclass(frozen=True)
class Stat:
    """一个指标在 n 趟重复上的分布。

    `n_nan` 单独记:t_complete_s 在未全覆盖组按定义是 nan,那是**正常**的;
    但「有的趟 nan、有的趟不 nan」意味着覆盖率在重复之间翻越了 100%,
    是必须报警的情况(见 `check_consistency`)。
    """
    mean: float
    std: float          # 样本标准差(ddof=1);n<2 时为 0.0
    lo: float
    hi: float
    n: int              # 参与统计的**非 nan** 样本数
    n_nan: int

    @property
    def is_flat(self) -> bool:
        """n 趟逐位相同 —— 表里就不必挂一个 ±0.0 的尾巴。"""
        return self.n >= 1 and self.std == 0.0 and self.lo == self.hi


def _stat(values: Sequence[float]) -> Stat:
    good = [v for v in values if v is not None and not math.isnan(v)]
    n_nan = len(values) - len(good)
    if not good:
        return Stat(float("nan"), 0.0, float("nan"), float("nan"), 0, n_nan)
    std = statistics.stdev(good) if len(good) >= 2 else 0.0
    return Stat(statistics.fmean(good), std, min(good), max(good), len(good), n_nan)


def _num(s: str) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_reps(rep_dirs: Sequence[str]) -> Dict[str, List[Dict[str, str]]]:
    """读若干趟的 sweep_metrics.csv,按 run_id 归拢。只保留 scenario == vrp 的行。

    ⚠ `utf-8-sig`:write_metric_csv 写的是带 BOM 的 CSV(Excel 要),
      用 `utf-8` 读会让第一列的键变成 `\\ufeffrun_id`。
    """
    out: Dict[str, List[Dict[str, str]]] = {}
    for d in rep_dirs:
        p = os.path.join(d, "sweep_metrics.csv")
        if not os.path.exists(p):
            raise SystemExit(
                f"缺 {p}\n"
                f"  重复批由 08_sweep.py 产出:\n"
                f"    08_sweep.py --logs-dir {d} --no-figure\n"
                f"  ⚠ 多趟必须**串行**跑(见本文件头),并行会让 std 假性归零。")
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("scenario") != "vrp":
                    continue
                out.setdefault(r["run_id"], []).append(r)
    return out


def check_consistency(by_run: Dict[str, List[Dict[str, str]]],
                      n_reps: int) -> List[str]:
    """聚合前的守卫。返回告警行(空 = 干净)。

    不静默平均:重复趟数不齐、或覆盖率在趟与趟之间翻越 100%,
    都会让「均值 ± 标准差」这句话本身失去意义。
    """
    warns: List[str] = []
    for rid, rows in sorted(by_run.items()):
        if len(rows) != n_reps:
            warns.append(f"{rid}: 只有 {len(rows)}/{n_reps} 趟,均值口径不齐")
        cov = [_num(r.get("coverage")) for r in rows]
        full = [c is not None and c >= 1.0 for c in cov]
        if any(full) and not all(full):
            warns.append(
                f"{rid}: 覆盖率在重复之间翻越 100%({sum(full)}/{len(full)} 趟全覆盖)"
                f" ⇒ t_complete_s 部分 nan,该组的时间均值不可用")
        if any((r.get("timed_out") or "").lower() == "true" for r in rows):
            k = sum((r.get("timed_out") or "").lower() == "true" for r in rows)
            warns.append(f"{rid}: {k}/{len(rows)} 趟 timed_out ⇒ 该组结果不可信")
    return warns


def scan_all_columns(by_run: Dict[str, List[Dict[str, str]]]
                     ) -> List[Tuple[str, str, List[str]]]:
    """**全列**扫一遍,找出在重复之间取值不同的格子。返回 (run_id, 列名, 取值集)。

    为什么不只扫进表的那 7 列:非确定性不一定落在你选中的指标上。
    实测里全部 24 组 x 40 余列**只有** `waiton_L1_T15.reassignment_count`
    在 20 与 22 之间摆动 —— 那一列既不在表里也不在图里。只扫选中列的话,
    会得出"完全确定"的结论,把这条真实存在(只是无害)的性质漏掉。
    """
    moved: List[Tuple[str, str, List[str]]] = []
    for rid, rows in sorted(by_run.items()):
        for col in rows[0]:
            if col in WALLCLOCK_COLS:      # 挂钟量必然趟趟不同,不是结论
                continue
            vals = {r.get(col) for r in rows}
            if len(vals) > 1:
                moved.append((rid, col, sorted(str(v) for v in vals)))
    return moved


def aggregate(by_run: Dict[str, List[Dict[str, str]]],
              keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """{run_id: {"wait":…, "L":…, "T":…, "stats": {key: Stat}}}"""
    out: Dict[str, Dict[str, Any]] = {}
    for rid, rows in by_run.items():
        head = rows[0]
        for col in ("wait", "leader_speed_mps", "plan_period_s"):
            vals = {r[col] for r in rows}
            if len(vals) > 1:                       # 同名组的参数居然不一致
                raise SystemExit(f"{rid}: 各趟的 {col} 不一致 {sorted(vals)}")
        out[rid] = {
            "run_id": rid,
            "wait": head["wait"],
            "leader_speed_mps": float(head["leader_speed_mps"]),
            "plan_period_s": float(head["plan_period_s"]),
            "n_reps": len(rows),
            "stats": {k: _stat([_num(r.get(k)) for r in rows]) for k in keys},
        }
    return out


# ======================================================================
# 显示
# ======================================================================
def _fmt_one(spec, v: float) -> str:
    """按 MetricSpec.fmt 格式化一个数。

    ⚠ 整数格指标(`{:d}`,如 観測成功数)取了均值就成了浮点,直接 format 会抛
      ValueError。整数均值仍按整数显示(`22` 而不是 `22.0`),只有真的出现小数
      —— 即重复之间做成的目标数不同 —— 才显示一位小数,那正是要让人看见的事。
    """
    try:
        return spec.fmt.format(v)
    except (ValueError, TypeError):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.1f}"


def fmt_stat(key: str, s: Stat) -> str:
    """`mean±std`,格式随 MetricSpec.fmt 走。

    逐位相同的格子**不挂 ±0.0** —— 24 组 x 7 列里绝大多数是那种,
    挂上去只会把真正有波动的那几格淹掉。是否逐位相同在表注里统一说明。
    """
    spec = METRIC_BY_KEY[key]
    if s.n == 0:
        return "nan"
    m = _fmt_one(spec, s.mean)
    if s.is_flat:
        return m
    return f"{m}±{_fmt_one(spec, s.std).lstrip()}"


def _blocks(agg: Dict[str, Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """按 wait 拆两块。

    为什么必须拆:未全覆盖的行没有 t_complete_s(按定义 nan),必须与 visited 同读。
    混在一张表里横比,读者一定会拿「早收工」当「更快」。

    ⚠ 表头**由数据算出**,不写死。早先写死成「wait=off 那 8 组全部漏点」,
      加了 L=0.3 档(比 Follower 慢)之后立刻成了假话 —— 那一档 Leader 根本
      不需要等,off 也照样 22/22。写死的描述会比数据更晚被发现是错的。
    """
    order = {"on": 0, "off": 1}
    blocks: Dict[str, List[Dict[str, Any]]] = {}
    for rec in agg.values():
        blocks.setdefault(rec["wait"], []).append(rec)
    out = []
    for w in sorted(blocks, key=lambda x: order.get(x, 9)):
        rows = sorted(blocks[w], key=lambda r: (r["leader_speed_mps"],
                                                r["plan_period_s"]))
        n = len(rows)
        full = sum(1 for r in rows if r["stats"]["t_complete_s"].n > 0)
        head = ("wait=on(Leader 停船等待)" if w == "on" else "wait=off(Leader 不等)")
        if full == n:
            title = f"{head} — {n} 组全部全覆盖"
        elif full == 0:
            title = f"{head} — {n} 组全部漏点,t_complete_s 按定义是 nan"
        else:
            title = (f"{head} — {n} 组中 {full} 组全覆盖 / {n - full} 组漏点"
                     f"(漏点组的 t_complete_s 按定义是 nan)")
        out.append((title, rows))
    return out


def wait_noop_speeds(agg: Dict[str, Dict[str, Any]]) -> List[float]:
    """找出「开不开 Leader 等待都一样」的那些 Leader 速度。

    Leader 比 Follower 慢时它根本追不开队友,等待逻辑永远不触发 ⇒ on/off 两行
    **逐位相同**。这不是巧合,是 wait 这个因子在该速度下**失效**的直接证据,
    值得单独报一行,而不是让读者自己去比对两张表。
    """
    by: Dict[Tuple[float, float], Dict[str, Dict[str, Any]]] = {}
    for r in agg.values():
        by.setdefault((r["leader_speed_mps"], r["plan_period_s"]),
                      {})[r["wait"]] = r
    same: Dict[float, List[bool]] = {}
    for (lv, _T), pair in by.items():
        if {"on", "off"} - set(pair):
            continue
        ok = all(_same_stat(pair["on"]["stats"][k], pair["off"]["stats"][k])
                 for k in TABLE_METRICS)
        same.setdefault(lv, []).append(ok)
    return sorted(lv for lv, oks in same.items() if oks and all(oks))


def wrapped_axis_label(spec, *, percent: bool = False) -> str:
    """`Time per target\\n[s/target]` —— 把单位折到第二行。

    ⚠ 不是排版洁癖:纵轴标签是**旋转 90 度**的,它的可用长度是**面板高度**。
      3x 字号下「Time per target [s/target]」一行需要约 375 pt,而面板只有约
      324 pt 高 ⇒ 直接溢出压到隔壁面板上(实测发生过)。折行后每行都放得下。
      文字来源仍是契约的 `MetricSpec.axis_label()`,这里只改断行。
    """
    text = spec.axis_label("en")
    if percent:                                   # 百分比刻度下单位写 %
        text = spec.label_en + " [%]"
    head, sep, unit = text.rpartition(" [")
    return f"{head}\n[{unit}" if sep else text


def _legend_band(fig, st, *, rows: int = 2) -> float:
    """图底部要留给 figure 级图例的高度,占图幅的比例。

    按**字号与画布高度**算,不写死一个 0.08 —— 字号一改(本项目从 2x 调到 3x
    就发生过)写死的值立刻不够,图例会压在下排的轴标签上。
    """
    line_in = st.font_size_legend * 1.9 / 72.0        # 一行图例约多高(英寸)
    return min(0.35, rows * line_in / fig.get_figheight())


def figure_style(font_scale: float = FONT_SCALE,
                 label_scale: float = LABEL_FONT_SCALE):
    """本脚本两张图共用的样式:全部字号 x font_scale,**轴标签单独** x label_scale。

    ⚠ 只是一份局部副本 —— `DEFAULT_STYLE` 本身不动,否则 02/03/05/07 的图会跟着变。
    """
    from vrpsim.contracts.style import DEFAULT_STYLE

    return DEFAULT_STYLE.scaled_fonts(font_scale).with_overrides(
        font_size_label=DEFAULT_STYLE.font_size_label * label_scale)


def drawn_series(agg: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
    """图里**实际要画**的 (wait, Leader 速度) 序列,已排好绘制顺序。

    抽成函数是为了能被测试直接断言:`wait` 失效的速度上必须只剩一条线
    (两条完全重合,画两条只是占图例)。
    """
    keys = {(r["wait"], r["leader_speed_mps"]) for r in agg.values()}
    noop = set(wait_noop_speeds(agg))
    keys = {k for k in keys if not (k[1] in noop and k[0] == "off")}
    return sorted(keys, key=lambda k: ({"on": 0, "off": 1}.get(k[0], 9), k[1]))


def _same_stat(a: Stat, b: Stat) -> bool:
    """两个 Stat 是否同一个数。nan 与 nan 视为相同(都表示「未全覆盖」)。"""
    if a.n == 0 or b.n == 0:
        return a.n == b.n
    return (a.mean == b.mean) and (a.std == b.std)


def write_summary_md(agg: Dict[str, Dict[str, Any]], path: str, *,
                     rep_dirs: Sequence[str], warns: Sequence[str],
                     moved: Sequence[Tuple[str, str, List[str]]] = ()) -> str:
    buf = io.StringIO()
    w = lambda s="": print(s, file=buf)   # noqa: E731
    n_reps = max(r["n_reps"] for r in agg.values())

    w("# 参数扫描汇总 — 提案(局部VRP)方法")
    w()
    w("## 実験目的")
    w()
    w(f"**中文** {PURPOSE_ZH}")
    w()
    w(f"**日本語** {PURPOSE_JA}")
    w()
    # 网格描述由数据算出 —— 写死的话,加一档速度就成了假话(加 0.3 档时踩到过)
    waits = sorted({r["wait"] for r in agg.values()}, key=lambda x: x != "on")
    lvs = sorted({r["leader_speed_mps"] for r in agg.values()})
    Ts = sorted({r["plan_period_s"] for r in agg.values()})
    w(f"扫描因子:`wait` ({'/'.join(waits)}) x "
      f"`leader_speed_mps` ({'/'.join(f'{v:g}' for v in lvs)}) x "
      f"`plan_period_s` ({'/'.join(f'{v:g}' for v in Ts)}) = "
      f"**{len(agg)} 组**,各组独立重复 **{n_reps} 趟**。")
    w("`usbl_period_s` 固定 15 s、`follower_speed_mps` 固定 0.5 m/s —— "
      "固定住才能看清规划周期与 Leader 速度各自的效应。")
    w()

    if warns:
        w("## ⚠ 聚合守卫告警")
        w()
        for x in warns:
            w(f"- {x}")
        w()

    w("## 正文表")
    w()
    w(f"数值 = {n_reps} 趟重复的**均值**;`±` 后为样本标准差(ddof=1)。"
      f"**未挂 `±` 的格子表示 {n_reps} 趟逐位相同。**")
    w()
    noop = wait_noop_speeds(agg)
    if noop:
        w("⚠ **Leader 速度 " + " / ".join(f"{v:g}" for v in noop) +
          " m/s 时,wait=on 与 wait=off 两块的对应行逐位相同** —— "
          "Leader 比 Follower 慢就永远追不开队友,等待逻辑一次都不触发。"
          "**wait 这个因子在这一档上是失效的**,不是「效果小」。")
        w()
    for title, rows in _blocks(agg):
        w(f"### {title}")
        w()
        head = ["L [m/s]", "T [s]"] + [
            f"{METRIC_BY_KEY[k].label_zh} [{METRIC_BY_KEY[k].unit}]"
            for k in TABLE_METRICS]
        w("| " + " | ".join(head) + " |")
        w("|" + "|".join(["---"] * len(head)) + "|")
        for r in rows:
            cells = [f"{r['leader_speed_mps']:.1f}", f"{r['plan_period_s']:g}"]
            cells += [fmt_stat(k, r["stats"][k]) for k in TABLE_METRICS]
            w("| " + " | ".join(cells) + " |")
        w()

    w("## 指标为什么选这几个")
    w()
    w("原则:**每个被扫的因子,配一个能解释它的指标**,不是每族凑一个。")
    w()
    for k, why in SELECTION_NOTES:
        spec = METRIC_BY_KEY[k]
        star = " ★图" if k in FIGURE_METRICS else ""
        w(f"- **{spec.label_zh}** `{k}` [{spec.unit}]{star} — {why}")
        w(f"  - 公式:`{spec.formula}`")
    w()
    w("### 主动排除的(不是漏掉)")
    w()
    for name, why in EXCLUDED_NOTES:
        w(f"- **{name}** — {why}")
    w()

    w("## 限制")
    w()
    w(f"- 各条件 n = {n_reps},重复批**串行独立**跑。ortools 的元启发式在 "
      "`vrp_time_limit_s` 这个**挂钟**预算内跑多少轮迭代取决于机器负载,"
      "所以并行跑出来的重复是**相关样本**,std 会假性归零 —— 本汇总的重复批"
      "由 `logs/reps/run_seq.sh` 一趟接一趟产出。")
    w("- `solve_wall_mean_s` 是挂钟量,**不进入仿真时间轴**,不构成任何数值结论;"
      "本表未收录。")
    w()
    w("### 全列非确定性扫描")
    w()
    w("只扫进表的那几列会漏掉别处的波动,所以这一步扫**全部**列(挂钟列除外):")
    w()
    if moved:
        w("| 组 | 列 | 各趟取值 |")
        w("|---|---|---|")
        for rid, col, vals in moved:
            w(f"| `{rid}` | `{col}` | {' / '.join(vals)} |")
        w()
        keys = {c for _, c, _ in moved}
        if not (keys & set(TABLE_METRICS)):
            w("⇒ 摆动**全部落在进表进图之外的列上** —— 本汇总的每一个数在 "
              f"{n_reps} 趟里逐位相同。")
    else:
        w(f"{n_reps} 趟**全列逐位相同**,一格未动。")
    w("- wait=off 的 8 组未全覆盖 ⇒ `t_complete_s` 按定义是 nan。"
      "那 8 行**只能**通过 `t_per_target_s` 与 `visited` 与 wait=on 组比较。")
    w()

    w("## 重跑(复现)")
    w()
    w("```bash")
    w("PY=D:/nixingxing/Anaconda/envs/auv_py310/python.exe")
    w("# 1) 串行产出重复批(⚠ 不要用 & 并发,会让 std 假性归零)")
    w("sh VRPSimulation/logs/reps/run_seq.sh")
    w("# 2) 聚合出正文图表")
    w("$PY VRPSimulation/scripts/10_sweep_summary.py")
    w("```")
    w()
    w("本次使用的重复批:")
    for d in rep_dirs:
        w(f"- `{d}`")

    path = writable_path(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    return path


def write_summary_csv(agg: Dict[str, Dict[str, Any]], path: str) -> str:
    cols = ["run_id", "wait", "leader_speed_mps", "plan_period_s", "n_reps"]
    for k in TABLE_METRICS:
        cols += [f"{k}_mean", f"{k}_std", f"{k}_min", f"{k}_max"]
    path = writable_path(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for _, rows in _blocks(agg):
            for r in rows:
                row: List[Any] = [r["run_id"], r["wait"], r["leader_speed_mps"],
                                  r["plan_period_s"], r["n_reps"]]
                for k in TABLE_METRICS:
                    s = r["stats"][k]
                    row += [s.mean, s.std, s.lo, s.hi]
                wr.writerow(row)
    return path


def plot_summary(agg: Dict[str, Dict[str, Any]], path: str, *,
                 font_scale: float = FONT_SCALE,
                 label_scale: float = LABEL_FONT_SCALE) -> str:
    """正文图。**全英文标注**(学术汇报口径),字号由 `figure_style()` 统一放大。

    ⚠ 在 `wait` 失效的速度上只画一条线:那两条完全重合,画两条纯属占图例。
      失效速度由 `wait_noop_speeds()` **从数据里认**,不写死 0.3。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vrpsim.viz import use_cjk_font

    st = figure_style(font_scale, label_scale)
    use_cjk_font(st.font_family)
    st.apply_rcparams()

    from matplotlib.ticker import PercentFormatter

    # 两个因子映射到两种视觉通道,而不是"一条线一个颜色":
    #   颜色 = Leader 速度,线型 = 是否停船等待。
    # 这样黑白打印后 wait 因子仍然分得开,而 wait 恰恰是本图最重要的那个因子。
    # ⚠ 颜色表按**实际出现的速度**动态生成,不写死 {0.5, 1.0} ——
    #   加 0.3 档时写死的表会让新那条线掉进 fallback 色,与别的线撞色。
    PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    STYLE = {"on": "-", "off": "--"}
    speeds = sorted({r["leader_speed_mps"] for r in agg.values()})
    COLOR = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(speeds)}
    series: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in agg.values():
        series.setdefault((r["wait"], r["leader_speed_mps"]), []).append(r)
    for v in series.values():
        v.sort(key=lambda r: r["plan_period_s"])

    noop = set(wait_noop_speeds(agg))
    n_reps = max(r["n_reps"] for r in agg.values())
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_MAIN)
    # ⚠ 线宽按绘制顺序递减:観測成功数那一格里**三条 wait=on 线全部压在 22 上**,
    #   等宽的话只看得见最后画的那条,会被读成"只有那一档做满了"。
    #   递减宽度让它们像同心带一样都露出来 —— 不平移数据,不造假。
    drawn = drawn_series(agg)
    LW = {k: 1.9 + 1.1 * (len(drawn) - 1 - i) / max(1, len(drawn) - 1)
          for i, k in enumerate(drawn)}
    for ax, key in zip(axes.ravel(), FIGURE_METRICS):
        spec = METRIC_BY_KEY[key]
        pct = spec.fmt.rstrip("}").endswith("%")
        for k in drawn:
            wait, lv = k
            rs = series[k]
            c, ls = COLOR[lv], STYLE.get(wait, ":")
            x = [r["plan_period_s"] for r in rs]
            mu = [r["stats"][key].mean for r in rs]
            sd = [r["stats"][key].std for r in rs]
            label = (f"Leader {lv:g} m/s (wait on = off)" if lv in noop
                     else f"Leader {lv:g} m/s, wait {wait}")
            ax.plot(x, mu, ls, color=c, linewidth=LW[k] * LINE_SCALE,
                    marker="o" if wait == "on" else "s",
                    markersize=5 * LINE_SCALE,
                    markerfacecolor=c if wait == "on" else "white",
                    label=label, zorder=3)
            # ±std 带。n 趟逐位相同时带宽为 0,画出来就是一条线 —— 那本身
            # 就是要传达的信息(该条件下重复无波动),所以不做「零宽就不画」的特判。
            ax.fill_between(x, [m - s for m, s in zip(mu, sd)],
                            [m + s for m, s in zip(mu, sd)],
                            color=c, alpha=0.18, linewidth=0, zorder=2)
        ax.set_xlabel("Replanning period T [s]", fontsize=st.font_size_label)
        # 轴标签从契约取(MetricSpec.axis_label),不在脚本里手写英文
        if pct:                       # 契约里是百分比格式,纵轴就别再显示 0.05
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel(wrapped_axis_label(spec, percent=pct),
                      fontsize=st.font_size_label)
        ax.tick_params(labelsize=st.font_size_tick)
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted({r["plan_period_s"] for r in agg.values()}))
        if key == "visited":
            n_t = max(r["stats"]["visited"].hi for r in agg.values())
            # 満点線 —— 一眼看出哪几条线根本没做完
            ax.axhline(n_t, color="0.4", linestyle=":",
                       linewidth=1.2 * LINE_SCALE, zorder=1)
            ax.set_ylim(0, n_t * 1.34)     # 顶上留出放注记的空
            # 用轴分数定位,不跟着 n_t 走 —— 早先按数据坐标放,注记正好压在満点線上
            ax.annotate(f"all {n_t:g} targets observed", xy=(0.03, 0.97),
                        xycoords="axes fraction", ha="left", va="top",
                        fontsize=st.font_size_legend, color="0.35")
        if key == "leader_hold_frac":
            # 4 条线压在 0 上不是画漏了:不等待就没有停船,慢 Leader 也追不开。
            # 放在两条数据线之间的空白带(轴分数 ~0.5),不要放顶上 —— 顶上是紫线。
            ax.annotate("wait=off (all speeds)\nand Leader 0.3 m/s: 0%",
                        xy=(0.97, 0.52), xycoords="axes fraction",
                        ha="right", va="center",
                        fontsize=st.font_size_legend, color="0.35")
    # ⚠ 图例放**图底部**而不是某个面板里:3x 字号下的图例框会盖住半张子图
    #   (实测把 wait=off / Leader 1 m/s 那条整个挡掉了)。
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=st.font_size_legend,
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    # 标题在 3x 字号下一行放不下 —— 显式折行,别让 tight_layout 把它裁掉
    fig.suptitle(f"Parameter sweep: line = mean of {n_reps} repetitions\n"
                 f"band = +/-1 SD (zero width = bit-identical across all "
                 f"{n_reps} runs)", fontsize=st.font_size_title)
    # ⚠ tight_layout 不认 figure 级图例 —— 必须用 rect 手动把底部让出来,
    #   否则图例会直接压在下排的 "Replanning period T [s]" 上。
    fig.tight_layout(rect=(0, _legend_band(fig, st, rows=2), 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=st.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_leader_speed(agg: Dict[str, Dict[str, Any]], path: str, *,
                      font_scale: float = FONT_SCALE,
                      label_scale: float = LABEL_FONT_SCALE) -> str:
    """横轴 = Leader 速度的补充图 —— 主图里速度只是颜色,看不出有没有最优点。

    每个速度上，线 = 该 wait 档在 4 个规划周期里的**最好值**(能做到多好)，
    阴影 = 4 个周期的最好~最差(周期这个参数在这一档上还剩多少影响力)。
    带宽塌成一条线,就说明该速度下**规划周期已经不起作用**了。

    ⚠ `wait` 失效的速度上不画 off 那条(与 on 逐位相同,画上去是重影),
      改用一个空心圈标出"两档在此重合",免得看成缺数据。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vrpsim.viz import use_cjk_font

    st = figure_style(font_scale, label_scale)
    use_cjk_font(st.font_family)
    st.apply_rcparams()

    STYLE = {"on": ("-", "o", "#1b9e77"), "off": ("--", "s", "#d95f02")}
    speeds = sorted({r["leader_speed_mps"] for r in agg.values()})
    noop = set(wait_noop_speeds(agg))
    v_follower = 0.5
    panels = [("t_per_target_s", True), ("visited", False)]

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_SPEED)
    for ax, (key, lower_better) in zip(axes, panels):
        spec = METRIC_BY_KEY[key]
        for wait in ("on", "off"):
            ls, mk, c = STYLE[wait]
            xs, best, lo, hi = [], [], [], []
            for lv in speeds:
                if lv in noop and wait == "off":
                    continue          # 与 on 重合,不画重影
                vals = [r["stats"][key].mean for r in agg.values()
                        if r["leader_speed_mps"] == lv and r["wait"] == wait
                        and r["stats"][key].n]
                if not vals:
                    continue
                xs.append(lv)
                best.append(min(vals) if lower_better else max(vals))
                lo.append(min(vals))
                hi.append(max(vals))
            if not xs:
                continue
            ax.plot(xs, best, ls, color=c, marker=mk, markersize=6 * LINE_SCALE,
                    markerfacecolor=c if wait == "on" else "white",
                    linewidth=1.9 * LINE_SCALE,
                    label=f"wait {wait} (best of 4 periods)", zorder=3)
            ax.fill_between(xs, lo, hi, color=c, alpha=0.16, linewidth=0,
                            zorder=2, label=f"wait {wait}: range over periods")
        # wait 失效的速度上标一个空心圈:那里 on/off 重合,不是漏画
        for lv in sorted(noop):
            vals = [r["stats"][key].mean for r in agg.values()
                    if r["leader_speed_mps"] == lv and r["stats"][key].n]
            if vals:
                y = min(vals) if lower_better else max(vals)
                ax.plot([lv], [y], "o", markersize=13 * LINE_SCALE,
                        markerfacecolor="none", markeredgecolor="0.35",
                        markeredgewidth=1.4 * LINE_SCALE, zorder=4,
                        label="wait has no effect here" if key == "visited" else None)
        # Follower 速度 = 分界线。左边 Leader 更慢,右边 Leader 更快。
        ax.axvline(v_follower, color="0.45", linestyle=":",
                   linewidth=1.2 * LINE_SCALE, zorder=1)
        # ⚠ 用 xaxis_transform(x 取数据坐标、y 取轴分数)钉在画面顶端。
        #   早先按 `ax.get_ylim()[1]` 定位,右图的数据线正好在 22/25 处,
        #   标注直接压在线上。轴分数与数据无关,换数据也不会再撞。
        ax.annotate(f"Follower speed\n({v_follower:g} m/s)",
                    xy=(v_follower, 0.985), xycoords=ax.get_xaxis_transform(),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="top",
                    fontsize=st.font_size_legend, color="0.35")
        ax.set_xlabel("Leader cruise speed [m/s]", fontsize=st.font_size_label)
        ax.set_ylabel(wrapped_axis_label(spec), fontsize=st.font_size_label)
        ax.tick_params(labelsize=st.font_size_tick)
        ax.set_xticks(speeds)
        ax.grid(alpha=0.3)
        if key == "visited":
            n_t = max(r["stats"]["visited"].hi for r in agg.values())
            ax.axhline(n_t, color="0.4", linestyle=":",
                       linewidth=1.2 * LINE_SCALE, zorder=1)
            # 顶上留出放「Follower speed」标注的空间
            ax.set_ylim(0, n_t * 1.30)
    # 图例同样放图底部:3x 字号下的图例框会盖住整整半张子图
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=st.font_size_legend,
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Optimal Leader speed: too slow and the survey line is the\n"
                 "bottleneck; too fast and the Followers cannot keep up",
                 fontsize=st.font_size_title)
    fig.tight_layout(rect=(0, _legend_band(fig, st, rows=2), 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=st.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps-dir", default=os.path.join(LOGS_DIR, "reps"),
                    help="重复批的父目录")
    ap.add_argument("--prefix", default="seq",
                    help="重复批子目录前缀(串行批 = seq)")
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--reps", default="",
                    help="直接指定重复批目录,逗号分隔;给了就忽略上面三个")
    ap.add_argument("--out-dir", default=LOGS_DIR)
    ap.add_argument("--figure", default=os.path.join(DEFAULT_FIGURE_DIR,
                                                     "sweep_summary.png"))
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--font-scale", type=float, default=FONT_SCALE,
                    help="标题/刻度/图例/标注的字号相对 contracts/style.py "
                         "默认值的倍数(默认 3.0)")
    ap.add_argument("--label-font-scale", type=float, default=LABEL_FONT_SCALE,
                    help="**轴标签**单独一档的倍数(默认 2.0):标签最长,"
                         "跟着标题一起到 3x 会顶满整幅画")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.reps.strip():
        rep_dirs = [d.strip() for d in args.reps.split(",") if d.strip()]
    else:
        rep_dirs = [os.path.join(args.reps_dir, f"{args.prefix}{k}")
                    for k in range(args.n_reps)]

    print(f"[1/3] 读 {len(rep_dirs)} 趟重复")
    by_run = load_reps(rep_dirs)
    warns = check_consistency(by_run, len(rep_dirs))
    for x in warns:
        print(f"  ⚠ {x}")
    agg = aggregate(by_run, TABLE_METRICS)
    moved_cols = scan_all_columns(by_run)
    print(f"  {len(agg)} 组 x {len(TABLE_METRICS)} 指标;"
          f"全列非确定性扫描:{len(moved_cols)} 格在重复之间摆动")
    for rid, col, vals in moved_cols:
        print(f"    {rid:<18} {col:<24} {' / '.join(vals)}")

    print("[2/3] 出表")
    md_p = write_summary_md(agg, os.path.join(args.out_dir, "sweep_summary.md"),
                            rep_dirs=rep_dirs, warns=warns, moved=moved_cols)
    csv_p = write_summary_csv(agg, os.path.join(args.out_dir, "sweep_summary.csv"))
    print(f"  {md_p}")
    print(f"  {csv_p}")

    if not args.no_figure:
        print(f"[3/3] 出图(英文标注,字号 x{args.font_scale:g}"
              f",轴标签 x{args.label_font_scale:g})")
        kw = dict(font_scale=args.font_scale, label_scale=args.label_font_scale)
        print(f"  {plot_summary(agg, args.figure, **kw)}")
        base, ext = os.path.splitext(args.figure)
        print(f"  {plot_leader_speed(agg, f'{base}_leader_speed{ext}', **kw)}")

    # 有波动的格子挑出来单独报 —— 论文里凡是要写「A 比 B 快」的地方,
    # 都得先确认那两个格子的 ±std 不重叠。
    moved = [(rid, k, r["stats"][k]) for rid, r in sorted(agg.items())
             for k in TABLE_METRICS if not r["stats"][k].is_flat and r["stats"][k].n]
    print(f"\n重复之间有波动的格子:{len(moved)} / "
          f"{len(agg) * len(TABLE_METRICS)}")
    for rid, k, s in moved:
        spec = METRIC_BY_KEY[k]
        print(f"  {rid:<18} {spec.label_zh:<16} "
              f"{spec.fmt.format(s.mean)} ± {spec.fmt.format(s.std)} "
              f"[{spec.fmt.format(s.lo)}, {spec.fmt.format(s.hi)}]")


if __name__ == "__main__":
    main()
