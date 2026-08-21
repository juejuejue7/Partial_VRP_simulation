"""[第2层] 按 `contracts/metrics.py` 出表的通用件 —— 参数扫描与三方法对比共用。

抽出来的理由:`08_sweep.py`(16 组参数扫描)与 `09_compare_methods.py`(三方法对比)
要的是同一套"行 = 指标、列 = 被比较对象"的表。抄两份必然漂移 ——
指标族一变、格式一改,两张表就对不上了。

`rows` 的形状(两个脚本共用的唯一约定):

    {"_label": "L0.5/T15",        # 列头
     "scenario": "vrp",           # 用于按 MetricSpec.applies_to 过滤(D19)
     "metrics": {key: value, …},  # 值为 None ⇒ 该格记 "—"(不适用)
     …其余键随调用方自由附加(run_id / leader_speed_mps / …)}
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .contracts.metrics import (BETTER_MARK, FAMILY_NAMES, METRICS, MetricSpec)

__all__ = ["fmt_metric", "metrics_visible_for", "print_metric_table",
           "write_metric_csv", "writable_path"]


def fmt_metric(spec: MetricSpec, val: Any) -> str:
    """一个格子的显示形式。

    `None` ⇒ `—`(**不适用**),不是 `0` —— 写 0 会被误读成
    "这项开销为零的更优方案"(lawnmower 的通信族就是这个坑)。
    """
    if val is None:
        return "—"
    if isinstance(val, float) and not np.isfinite(val):
        return "nan"
    if spec.vector:
        return " / ".join(spec.fmt.format(x) for x in val)
    try:
        return spec.fmt.format(val)
    except (ValueError, TypeError):
        return str(val)


def metrics_visible_for(scenarios: Optional[Iterable[str]]) -> List[MetricSpec]:
    """在这批被比较对象上**都适用**的指标(D19)。

    `scenarios=None` ⇒ 不过滤(参数扫描内部用:那 16 组全是 `vrp`)。
    三方法对比传 `{"vrp", "twophase", "lawnmower"}`,`time_efficiency*`
    会被自动挡掉 —— 那三列的分子本身就是一次全局 VRP 的解,
    放进三方法表就是循环论证。
    """
    if scenarios is None:
        return list(METRICS)
    ss = set(scenarios)
    return [m for m in METRICS if ss <= set(m.applies_to)]


def print_metric_table(rows: Sequence[Dict[str, Any]], *, fh=None,
                       scenarios: Optional[Iterable[str]] = None) -> None:
    """按指标族分块打印。行 = 指标,列 = 被比较对象 —— 指标名长,横着排更好读。"""
    def w(s):
        print(s, file=fh) if fh else print(s)

    specs_all = metrics_visible_for(scenarios)
    names = [r["_label"] for r in rows]
    # 逐列取实际内容的最大宽度 —— 向量指标(各AUV航程)会明显宽于表头,
    # 用固定列宽会让相邻两列糊在一起。
    widths = []
    for j, r in enumerate(rows):
        longest = max([len(names[j])]
                      + [len(fmt_metric(m, r["metrics"].get(m.key))) for m in specs_all])
        widths.append(max(12, longest) + 2)
    head = (f"{'指标':<26}{'单位':<10}{'向':<3}"
            + "".join(f"{n:>{c}}" for n, c in zip(names, widths)))
    for fam, fam_name in FAMILY_NAMES.items():
        specs = [m for m in specs_all if m.family == fam]
        if not specs:
            continue
        w("")
        w(f"===== {fam_name} " + "=" * max(0, len(head) - len(fam_name) - 8))
        w(head)
        w("-" * len(head))
        for m in specs:
            cells = "".join(f"{fmt_metric(m, r['metrics'].get(m.key)):>{c}}"
                            for r, c in zip(rows, widths))
            w(f"{m.label_zh:<26}{m.unit:<10}{BETTER_MARK[m.better]:<3}{cells}")


def writable_path(path: str) -> str:
    """目标文件被 Excel 之类占着时改写到 `<名>.new.<后缀>`,而不是把整趟扫描扔掉。

    这个函数存在的理由:16 组要跑 10 分钟,最后一步因为「表格还开着」抛
    PermissionError 会让那 10 分钟白费 —— 仿真结果其实早就落盘了。
    """
    try:
        with open(path, "a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt = f"{base}.new{ext}"
        print(f"  ⚠ {os.path.basename(path)} 被占用(通常是 Excel 开着),"
              f"改写到 {os.path.basename(alt)};关掉后重跑即可覆盖原文件")
        return alt


def write_metric_csv(rows: Sequence[Dict[str, Any]], path: str, *,
                     lead_cols: Sequence[str] = (),
                     scenarios: Optional[Iterable[str]] = None) -> str:
    """一行一个被比较对象。向量指标展开成多列。返回**实际**写入的路径。

    `lead_cols` = 指标之前的那几列(run_id / wait / leader_speed_mps / …),
    由调用方决定,因为参数扫描与三方法对比要标的东西不一样。
    """
    specs = metrics_visible_for(scenarios)
    n_vec = max((len(r["metrics"].get("per_follower_distance_m") or [])
                 for r in rows), default=0)
    cols = list(lead_cols)
    for m in specs:
        if m.vector:
            cols += [f"{m.key}_{i}" for i in range(n_vec)]
        else:
            cols.append(m.key)
    path = writable_path(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for r in rows:
            row: List[Any] = [r.get(c, "") for c in lead_cols]
            for m in specs:
                v = r["metrics"].get(m.key)
                if m.vector:
                    v = list(v or [])
                    row += [v[i] if i < len(v) else "" for i in range(n_vec)]
                else:
                    row.append("" if v is None else v)
            wr.writerow(row)
    return path
