"""出表机器(`vrpsim/report.py`)与两个报告脚本的验收。

为什么要有这个文件
================================================================================
08 / 09 两个脚本此前**一行测试都没有**。实测代价:把 `_fmt` 抽到 `report.py`
并改名为 `fmt_metric` 时,`write_run_log` 里漏改了一处 —— 整个测试套件全绿,
却在真跑 16 组扫描时于第 1 组抛 `NameError`,前面的仿真白跑。

这类"只在写报告那一步才炸"的 bug,单靠指标测试永远抓不到:
它们在 `summary()` 之后、在断言之外。所以这里**真的调用**那几个写文件的函数。
"""
from __future__ import annotations

import csv
import io
import os

import pytest

from vrpsim.contracts.metrics import (ALL_SCENARIOS, METRICS, SCENARIO_LAWNMOWER,
                                      SCENARIO_TWOPHASE, SCENARIO_VRP,
                                      METRIC_BY_KEY)
from vrpsim.report import (fmt_metric, metrics_visible_for, print_metric_table,
                           writable_path, write_metric_csv)


def _load_script(name: str):
    """按路径加载 scripts/ 下的脚本(文件名以数字开头,不能当模块 import)。"""
    import importlib.util
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sd = os.path.join(root, "scripts")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), os.path.join(sd, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(scenario: str, label: str, **over):
    """一行合成数据:每个指标都给个值,好让格式化路径整条走一遍。"""
    met = {}
    for m in METRICS:
        if m.vector:
            met[m.key] = [1.0, 2.0]
        elif m.key == "timed_out":
            met[m.key] = False
        elif m.fmt == "{:d}":
            met[m.key] = 22
        else:
            met[m.key] = 1.5
    met.update(over)
    return {"run_id": label, "scenario": scenario, "_label": label, "metrics": met,
            "wait": "on", "leader_speed_mps": 0.5, "follower_speed_mps": 0.5,
            "plan_period_s": 15.0, "usbl_period_s": 15.0}


# ======================================================================
# report.py 本身
# ======================================================================
def test_none_renders_as_dash_not_zero():
    """`—` = 不适用;写 0 会被误读成"这项开销为零的更优方案"。"""
    spec = METRIC_BY_KEY["n_comm_messages"]
    assert fmt_metric(spec, None) == "—"
    assert fmt_metric(spec, 0) == "0"      # 真的是 0 ⇒ 就显示 0
    assert fmt_metric(spec, float("nan")) == "nan"


def test_cross_method_view_hides_the_circular_metrics():
    """三方法视图必须挡掉 `time_efficiency*`(循环论证),留下不归一化那把尺子。"""
    keys = {m.key for m in metrics_visible_for(ALL_SCENARIOS)}
    assert not (keys & {"time_efficiency", "time_efficiency_obs",
                        "time_efficiency_cov"})
    assert {"t_per_target_s", "t_finish_s", "visited", "n_targets"} <= keys
    # 单场景视图(参数扫描)不过滤
    assert "time_efficiency" in {m.key for m in metrics_visible_for(None)}
    assert "time_efficiency" in {m.key for m in metrics_visible_for([SCENARIO_VRP])}


def test_table_renders_for_all_three_methods():
    rows = [_row(SCENARIO_VRP, "提案"), _row(SCENARIO_TWOPHASE, "二段式"),
            _row(SCENARIO_LAWNMOWER, "lawn")]
    buf = io.StringIO()
    print_metric_table(rows, fh=buf, scenarios=ALL_SCENARIOS)
    out = buf.getvalue()
    assert "① 時間効率" in out and "⑥ 負荷均衡" in out
    assert "単位目標あたり所要时间" in out
    assert "时间效率(严格)" not in out, "循环论证那三列漏进三方法表了"


def test_csv_round_trips_with_vector_columns(tmp_path):
    rows = [_row(SCENARIO_VRP, "a"), _row(SCENARIO_TWOPHASE, "b")]
    p = str(tmp_path / "m.csv")
    got = write_metric_csv(rows, p, lead_cols=("run_id", "scenario"),
                           scenarios=ALL_SCENARIOS)
    assert got == p
    with io.open(p, encoding="utf-8-sig") as fh:
        table = list(csv.reader(fh))
    head, body = table[0], table[1:]
    assert head[:2] == ["run_id", "scenario"]
    assert "per_follower_distance_m_0" in head and "per_follower_distance_m_1" in head
    assert "time_efficiency" not in head
    assert len(body) == 2


def test_writable_path_falls_back_when_locked(tmp_path, monkeypatch):
    """被 Excel 占着时改写 `.new.csv`,而不是让整趟扫描白跑。"""
    p = str(tmp_path / "x.csv")
    real_open = io.open

    def fake_open(path, *a, **k):
        if str(path) == p:
            raise PermissionError("locked")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    alt = writable_path(p)
    assert alt.endswith(".new.csv") and alt != p


# ======================================================================
# 脚本级:真的把报告写出来 —— 这才是上面那个 NameError 的捕手
# ======================================================================
def test_sweep_run_log_actually_writes(tmp_path):
    """`write_run_log` 必须能真的跑完。**这条就是为漏改 `_fmt` 那次准备的。**"""
    sweep = _load_script("08_sweep.py")
    args = sweep.build_parser().parse_args([])
    args.logs_dir = str(tmp_path)
    rec = {"run_id": "waiton_L0.5_T15", "wait": True, "leader_speed_mps": 0.5,
           "follower_speed_mps": 0.5, "plan_period_s": 15.0, "usbl_period_s": 15.0,
           "metrics": _row(SCENARIO_VRP, "x")["metrics"]}
    p = str(tmp_path / "run.log")
    sweep.write_run_log(p, "waiton_L0.5_T15", rec, args)
    text = io.open(p, encoding="utf-8").read()
    assert "# 重跑本组:" in text
    assert "04_run_mission.py" in text
    # 每个指标都得落到日志里
    for m in METRICS:
        assert m.label_zh in text, f"{m.key} 没进 run.log"


def test_sweep_report_actually_writes(tmp_path):
    sweep = _load_script("08_sweep.py")
    args = sweep.build_parser().parse_args([])
    args.logs_dir = str(tmp_path)
    rows = [_row(SCENARIO_VRP, "L0.5/T15")]
    p = str(tmp_path / "r.md")
    got = sweep.write_report(rows, p, {
        "grid": "g", "follower_speed": 0.5, "usbl_period": 15.0, "dwell": 10.0,
        "solver": "greedy", "n_runs": 1, "logs_dir": "logs"}, args=args)
    text = io.open(got, encoding="utf-8").read()
    assert "## 指标定义" in text and "## 对比表" in text
    assert "重跑（复现）" in text


def test_methods_report_actually_writes(tmp_path):
    """09 的报告同样真写一遍 —— 它有更多手写的表格拼接,更容易出这类错。"""
    cmp_ = _load_script("09_compare_methods.py")
    args = cmp_.build_parser().parse_args([])
    args.logs_dir = str(tmp_path)
    rows = [_row(SCENARIO_VRP, "局部VRP(提案)"),
            _row(SCENARIO_TWOPHASE, "二段式全局VRP"),
            _row(SCENARIO_LAWNMOWER, "lawnmower")]
    p = str(tmp_path / "m.md")
    got = cmp_.write_report(rows, p, args,
                            {"t_survey_s": 1000.0, "t_observe_s": 1084.5,
                             "t_finish_s": 2084.5})
    text = io.open(got, encoding="utf-8").read()
    assert "三方法" in text
    assert "単位目標あたり時間" in text
    assert "負荷均衡" in text
    assert "循环论证" in text, "为什么不用 time_efficiency 的理由必须写进报告"


def test_both_scripts_share_kinematic_defaults():
    """08 与 09 的共享运动学默认值都必须来自契约(D15 踩过的坑)。"""
    from vrpsim.contracts.mission import MissionConfig
    d = MissionConfig()
    for name in ("08_sweep.py", "09_compare_methods.py"):
        ap = _load_script(name).build_parser()
        assert ap.get_default("follower_speed") == d.follower_speed_mps
        assert ap.get_default("dwell") == d.dwell_time_s
        assert ap.get_default("usbl_period") == d.usbl_period_s
        assert ap.get_default("solver") == d.solver
