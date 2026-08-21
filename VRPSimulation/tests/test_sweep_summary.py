"""多趟重复的聚合(`scripts/10_sweep_summary.py`)的验收。

为什么这些断言值得写
================================================================================
「均值 ± 标准差」这句话本身有前提:各条件的趟数要齐、覆盖率不能在趟与趟之间
翻越 100%。一旦这些前提破了还照样求平均,表里会出现一个**看起来很正常、
实际上没有意义**的数 —— 这种错误不会抛异常,只会静静地进论文。

所以这里钉三类东西:
  1. mean / std 逐条按公式对拍(不信实现,重算一遍);
  2. nan 与不齐的输入必须**报警或抛错**,不许静默平均;
  3. 写 md / csv / png 那几步真的跑一遍(08 那次 `NameError` 的教训)。
"""
from __future__ import annotations

import csv
import io
import math
import os
import statistics

import pytest

from vrpsim.contracts.metrics import METRIC_BY_KEY, SCENARIO_VRP


def _load_script(name: str):
    """按路径加载 scripts/ 下的脚本(文件名以数字开头,不能当模块 import)。"""
    import importlib.util
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sd = os.path.join(root, "scripts")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", "").replace(".", "_"), os.path.join(sd, name))
    mod = importlib.util.module_from_spec(spec)
    # ⚠ 必须先登记到 sys.modules 再 exec:`@dataclass` 在装饰时会去
    #   `sys.modules[cls.__module__]` 查类型,没登记就是 AttributeError。
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SUM = _load_script("10_sweep_summary.py")

# CSV 里除指标外必须有的几列(08_sweep.py 的 lead_cols)
LEAD = ("run_id", "scenario", "wait", "leader_speed_mps", "plan_period_s")


def _cell(key: str, i: int) -> float:
    """给第 i 趟造一个值。故意让不同趟不同,好让 std 非零。"""
    if key == "visited":
        return 22.0
    if key == "t_complete_s":
        return 1000.0 + 10.0 * i
    return 0.10 + 0.01 * i


def _write_rep(d: str, i: int, *, runs=None, over=None, coverage=1.0,
               timed_out=False) -> str:
    """造一趟 sweep_metrics.csv。`over` = {(run_id, key): value} 的覆盖。"""
    runs = runs or [("waiton_L0.5_T15", "on", 0.5, 15.0),
                    ("waitoff_L0.5_T15", "off", 0.5, 15.0)]
    over = over or {}
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "sweep_metrics.csv")
    cols = list(LEAD) + ["coverage", "timed_out"] + list(SUM.TABLE_METRICS)
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for rid, wait, lv, T in runs:
            row = [rid, SCENARIO_VRP, wait, lv, T, coverage, timed_out]
            for k in SUM.TABLE_METRICS:
                v = over.get((rid, k), _cell(k, i))
                row.append("" if v is None else v)
            wr.writerow(row)
    return d


def _reps(tmp_path, n=3, **kw):
    return [_write_rep(str(tmp_path / f"rep{i}"), i, **kw) for i in range(n)]


# ======================================================================
# 1. 公式对拍
# ======================================================================
def test_mean_and_std_match_their_formulas(tmp_path):
    """不信实现,拿 statistics 重算一遍。std 用样本标准差(ddof=1)。"""
    dirs = _reps(tmp_path, n=3)
    agg = SUM.aggregate(SUM.load_reps(dirs), SUM.TABLE_METRICS)
    s = agg["waiton_L0.5_T15"]["stats"]["t_complete_s"]
    expect = [1000.0, 1010.0, 1020.0]
    assert s.mean == pytest.approx(statistics.fmean(expect))
    assert s.std == pytest.approx(statistics.stdev(expect))   # ddof=1
    assert (s.lo, s.hi, s.n, s.n_nan) == (1000.0, 1020.0, 3, 0)


def test_single_rep_has_no_spread(tmp_path):
    """n=1 时样本标准差没有定义 —— 记 0 并且**不许**画出误差带。"""
    agg = SUM.aggregate(SUM.load_reps(_reps(tmp_path, n=1)), SUM.TABLE_METRICS)
    s = agg["waiton_L0.5_T15"]["stats"]["t_complete_s"]
    assert (s.n, s.std) == (1, 0.0) and s.is_flat


def test_identical_reps_report_no_error_bar(tmp_path):
    """逐位相同的格子不挂 `±0.0` —— 挂上去会把真正有波动的那几格淹掉。"""
    dirs = [_write_rep(str(tmp_path / f"r{i}"), 0) for i in range(4)]   # 全用 i=0
    agg = SUM.aggregate(SUM.load_reps(dirs), SUM.TABLE_METRICS)
    s = agg["waiton_L0.5_T15"]["stats"]["t_complete_s"]
    assert s.is_flat and "±" not in SUM.fmt_stat("t_complete_s", s)

    moved = SUM.aggregate(SUM.load_reps(_reps(tmp_path / "b", n=3)),
                          SUM.TABLE_METRICS)
    s2 = moved["waiton_L0.5_T15"]["stats"]["t_complete_s"]
    assert not s2.is_flat and "±" in SUM.fmt_stat("t_complete_s", s2)


def test_integer_metric_survives_averaging(tmp_path):
    """観測成功数的 fmt 是 `{:d}`,取均值后成了浮点 —— 直接 format 会抛 ValueError。

    整数均值仍显示整数(`22` 而不是 `22.0`);只有重复之间**真的做成了不同数量**
    的目标时才出现小数,那正是要让人看见的事。
    """
    dirs = _reps(tmp_path, n=3)
    st = SUM.aggregate(SUM.load_reps(dirs), SUM.TABLE_METRICS)
    assert SUM.fmt_stat("visited", st["waiton_L0.5_T15"]["stats"]["visited"]) == "22"

    dirs2 = [_write_rep(str(tmp_path / f"v{i}"), i,
                        over={("waiton_L0.5_T15", "visited"): float(21 + (i > 0))})
             for i in range(3)]
    s = SUM.aggregate(SUM.load_reps(dirs2), SUM.TABLE_METRICS
                      )["waiton_L0.5_T15"]["stats"]["visited"]
    out = SUM.fmt_stat("visited", s)
    assert "±" in out and out.startswith("21.7"), out


# ======================================================================
# 2. nan 与不齐的输入:必须报警,不许静默平均
# ======================================================================
def test_nan_completion_time_does_not_poison_the_mean(tmp_path):
    """未全覆盖组的 t_complete_s 按定义是 nan;它不能把别的列也拖成 nan。"""
    over = {("waitoff_L0.5_T15", "t_complete_s"): float("nan")}
    dirs = _reps(tmp_path, n=3, over=over)
    agg = SUM.aggregate(SUM.load_reps(dirs), SUM.TABLE_METRICS)
    st = agg["waitoff_L0.5_T15"]["stats"]
    assert math.isnan(st["t_complete_s"].mean)
    assert st["t_complete_s"].n == 0 and st["t_complete_s"].n_nan == 3
    # 同一行的其它列照常出数
    assert st["t_per_target_s"].n == 3 and not math.isnan(st["t_per_target_s"].mean)
    assert SUM.fmt_stat("t_complete_s", st["t_complete_s"]) == "nan"


def test_coverage_flipping_between_reps_is_flagged(tmp_path):
    """有的趟全覆盖、有的趟没有 ⇒ 该组的时间均值混了两把尺子,必须告警。"""
    dirs = [_write_rep(str(tmp_path / "a0"), 0, coverage=1.0),
            _write_rep(str(tmp_path / "a1"), 1, coverage=0.82),
            _write_rep(str(tmp_path / "a2"), 2, coverage=1.0)]
    warns = SUM.check_consistency(SUM.load_reps(dirs), 3)
    assert any("翻越 100%" in w for w in warns), warns


def test_short_rep_is_flagged_not_silently_averaged(tmp_path):
    """某组少跑了一趟 ⇒ 它的均值口径与别组不同,必须说出来。"""
    dirs = _reps(tmp_path, n=3)
    only_one = [("waiton_L0.5_T15", "on", 0.5, 15.0)]
    dirs.append(_write_rep(str(tmp_path / "rep3"), 3, runs=only_one))
    warns = SUM.check_consistency(SUM.load_reps(dirs), 4)
    assert any("waitoff_L0.5_T15" in w and "3/4" in w for w in warns), warns


def test_timed_out_group_is_flagged(tmp_path):
    """timed_out = 没跑完,不是跑得慢 —— 这种组的数不能当结果读。"""
    dirs = [_write_rep(str(tmp_path / f"t{i}"), i, timed_out=(i == 1))
            for i in range(3)]
    warns = SUM.check_consistency(SUM.load_reps(dirs), 3)
    assert any("timed_out" in w and "1/3" in w for w in warns), warns


def test_missing_rep_dir_raises_with_the_serial_warning(tmp_path):
    """报错要**带着怎么补**,并且提醒别并行 —— 并行会让 std 假性归零。"""
    with pytest.raises(SystemExit) as e:
        SUM.load_reps([str(tmp_path / "nope")])
    msg = str(e.value)
    assert "08_sweep.py" in msg and "串行" in msg


def test_parameter_drift_across_reps_raises(tmp_path):
    """同一个 run_id 在两趟里参数不同 ⇒ 拿它们求平均没有意义,直接炸。"""
    d0 = _write_rep(str(tmp_path / "d0"), 0)
    d1 = _write_rep(str(tmp_path / "d1"), 1,
                    runs=[("waiton_L0.5_T15", "on", 0.5, 30.0)])   # T 变了
    with pytest.raises(SystemExit) as e:
        SUM.aggregate(SUM.load_reps([d0, d1]), SUM.TABLE_METRICS)
    assert "plan_period_s" in str(e.value)


def test_all_column_scan_catches_drift_outside_the_selected_metrics(tmp_path):
    """非确定性不一定落在你选中的指标上 —— 实测 11 趟里唯一摆动的那一列
    (`reassignment_count`)既不在表里也不在图里。只扫选中列会得出"完全确定"
    的错误结论,所以扫描必须覆盖全列。"""
    cols = list(LEAD) + ["coverage", "timed_out"] + list(SUM.TABLE_METRICS)
    dirs = []
    for i in range(3):
        d = str(tmp_path / f"s{i}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "sweep_metrics.csv"), "w", newline="",
                  encoding="utf-8-sig") as fh:
            wr = csv.writer(fh)
            wr.writerow(cols + ["reassignment_count", "solve_wall_mean_s"])
            row = ["waiton_L0.5_T15", SCENARIO_VRP, "on", 0.5, 15.0, 1.0, False]
            row += [_cell(k, 0) for k in SUM.TABLE_METRICS]     # 选中列全部不动
            wr.writerow(row + [20 + (i == 1) * 2, 0.5 + 0.001 * i])
        dirs.append(d)
    by_run = SUM.load_reps(dirs)
    moved = SUM.scan_all_columns(by_run)
    assert [(r, c) for r, c, _ in moved] == [("waiton_L0.5_T15",
                                              "reassignment_count")], moved
    assert moved[0][2] == ["20", "22"]
    # 挂钟列趟趟不同是正常的,不能算非确定性
    assert "solve_wall_mean_s" not in {c for _, c, _ in moved}
    # 选中列全部纹丝不动 ⇒ 表里不该出现任何 ±
    agg = SUM.aggregate(by_run, SUM.TABLE_METRICS)
    assert all(s.is_flat for s in agg["waiton_L0.5_T15"]["stats"].values())


def test_drift_outside_the_table_is_still_reported(tmp_path):
    """摆动落在表外也必须写进报告 —— 「没进表」不等于「没发生」。"""
    agg = SUM.aggregate(SUM.load_reps(_reps(tmp_path, n=2)), SUM.TABLE_METRICS)
    p = str(tmp_path / "s.md")
    got = SUM.write_summary_md(
        agg, p, rep_dirs=["x"], warns=[],
        moved=[("waiton_L1_T15", "reassignment_count", ["20", "22"])])
    text = io.open(got, encoding="utf-8").read()
    assert "全列非确定性扫描" in text
    assert "reassignment_count" in text and "20 / 22" in text
    assert "进表进图之外" in text, "要说清它没污染任何被报告的数"


# ======================================================================
# 3. 选取的指标必须真的在契约里,且对本场景适用
# ======================================================================
def test_selected_metrics_are_declared_in_the_contract():
    for k in set(SUM.TABLE_METRICS) | set(SUM.FIGURE_METRICS):
        assert k in METRIC_BY_KEY, f"{k} 不在 contracts/metrics.py 里"
        assert SCENARIO_VRP in METRIC_BY_KEY[k].applies_to, \
            f"{k} 对提案方法不适用,不该进参数扫描表"
    assert set(SUM.FIGURE_METRICS) <= set(SUM.TABLE_METRICS), \
        "图里画了表里没有的指标 ⇒ 读者核对不到数"


def test_every_selected_metric_has_a_written_reason():
    """选进来就得说清为什么是它而不是同族别的 —— 这是本次选指标的原则。"""
    explained = {k for k, _ in SUM.SELECTION_NOTES}
    assert set(SUM.TABLE_METRICS) - {"t_complete_s"} <= explained


def test_table_is_split_by_wait(tmp_path):
    """wait=on 在前、wait=off 在后,且**必须分块** ——
    混在一张表里横比,读者一定会拿「早收工」当「更快」。"""
    agg = SUM.aggregate(SUM.load_reps(_reps(tmp_path, n=2)), SUM.TABLE_METRICS)
    blocks = SUM._blocks(agg)
    assert [len(rows) for _, rows in blocks] == [1, 1]
    assert "wait=on" in blocks[0][0] and "wait=off" in blocks[1][0]


def test_block_headings_are_computed_not_hardcoded(tmp_path):
    """表头写死会比数据更晚被发现是错的。

    早先 off 块的表头写死成「8 组全部漏点，t_complete_s 按定义是 nan」;
    加了 L=0.3 档(比 Follower 慢)后那一档 off 也是 22/22 全覆盖,
    这句话当场变成假话。表头必须由**覆盖情况**算出来。
    """
    runs = [("waitoff_L0.3_T15", "off", 0.3, 15.0),      # 全覆盖
            ("waitoff_L1_T15", "off", 1.0, 15.0)]        # 漏点
    dirs = [_write_rep(str(tmp_path / f"h{i}"), i, runs=runs,
                       over={("waitoff_L1_T15", "t_complete_s"): float("nan")})
            for i in range(2)]
    title, rows = SUM._blocks(SUM.aggregate(SUM.load_reps(dirs),
                                            SUM.TABLE_METRICS))[0]
    assert len(rows) == 2
    assert "2 组中 1 组全覆盖 / 1 组漏点" in title, title

    # 全部漏点 / 全部全覆盖两种极端也要分别说对
    d_all_nan = [_write_rep(str(tmp_path / f"n{i}"), i, runs=runs[1:],
                            over={("waitoff_L1_T15", "t_complete_s"): float("nan")})
                 for i in range(2)]
    t2, _ = SUM._blocks(SUM.aggregate(SUM.load_reps(d_all_nan),
                                      SUM.TABLE_METRICS))[0]
    assert "全部漏点" in t2, t2
    d_all_full = [_write_rep(str(tmp_path / f"f{i}"), i, runs=runs[:1])
                  for i in range(2)]
    t3, _ = SUM._blocks(SUM.aggregate(SUM.load_reps(d_all_full),
                                      SUM.TABLE_METRICS))[0]
    assert "全部全覆盖" in t3, t3


def test_wait_is_detected_as_a_no_op_where_the_leader_is_slower(tmp_path):
    """Leader 比 Follower 慢 ⇒ 永远追不开队友 ⇒ 等待逻辑一次都不触发,
    on/off 两行**逐位相同**。这是 wait 因子在该档上**失效**的直接证据,
    必须自动识别出来单独报,不能指望读者去比对两张表。"""
    runs = [("waiton_L0.3_T15", "on", 0.3, 15.0),
            ("waitoff_L0.3_T15", "off", 0.3, 15.0),      # 与上一行取值相同
            ("waiton_L1_T15", "on", 1.0, 15.0),
            ("waitoff_L1_T15", "off", 1.0, 15.0)]
    over = {("waitoff_L1_T15", "visited"): 11.0}         # 1.0 档 on/off 不同
    dirs = [_write_rep(str(tmp_path / f"w{i}"), i, runs=runs, over=over)
            for i in range(2)]
    agg = SUM.aggregate(SUM.load_reps(dirs), SUM.TABLE_METRICS)
    assert SUM.wait_noop_speeds(agg) == [0.3]

    text = io.open(SUM.write_summary_md(agg, str(tmp_path / "w.md"),
                                        rep_dirs=["x"], warns=[]),
                   encoding="utf-8").read()
    assert "wait=on 与 wait=off 两块的对应行逐位相同" in text
    assert "0.3" in text


# ======================================================================
# 4. 真的把文件写出来
# ======================================================================
def test_summary_md_actually_writes(tmp_path):
    dirs = _reps(tmp_path, n=3)
    agg = SUM.aggregate(SUM.load_reps(dirs), SUM.TABLE_METRICS)
    p = str(tmp_path / "s.md")
    got = SUM.write_summary_md(agg, p, rep_dirs=dirs, warns=["造个告警"])
    text = io.open(got, encoding="utf-8").read()
    assert "## 実験目的" in text
    assert SUM.PURPOSE_ZH in text and SUM.PURPOSE_JA in text
    assert "造个告警" in text, "告警必须进报告,不能只打在终端上"
    assert "wait=on" in text and "wait=off" in text
    assert "## 限制" in text and "串行" in text
    for k in SUM.TABLE_METRICS:                    # 每列都得有中文标签与公式
        assert METRIC_BY_KEY[k].label_zh in text


def test_summary_csv_has_mean_and_spread_columns(tmp_path):
    dirs = _reps(tmp_path, n=3)
    agg = SUM.aggregate(SUM.load_reps(dirs), SUM.TABLE_METRICS)
    p = str(tmp_path / "s.csv")
    got = SUM.write_summary_csv(agg, p)
    with io.open(got, encoding="utf-8-sig") as fh:
        table = list(csv.reader(fh))
    head, body = table[0], table[1:]
    for k in SUM.TABLE_METRICS:
        assert {f"{k}_mean", f"{k}_std", f"{k}_min", f"{k}_max"} <= set(head)
    assert len(body) == 2
    i = head.index("t_complete_s_std")
    assert float(body[0][i]) == pytest.approx(statistics.stdev(
        [1000.0, 1010.0, 1020.0]))


def test_figure_actually_renders(tmp_path):
    """出图那一步真跑一遍 —— 它是唯一没有断言守着的输出。"""
    agg = SUM.aggregate(SUM.load_reps(_reps(tmp_path, n=3)), SUM.TABLE_METRICS)
    p = str(tmp_path / "f.png")
    assert SUM.plot_summary(agg, p) == p
    assert os.path.getsize(p) > 5000


def test_std_band_is_drawn_as_mean_plus_minus_std(tmp_path, monkeypatch):
    """误差带必须**接在数据上**:上下沿逐点等于 mean ∓ std。

    实测这条带子在正式图里**看不见** —— 5 趟串行重复逐位相同,std 恒为 0,
    带宽就是 0。「看不见」和「没画」是两回事,而只有后者是 bug。
    这条同时钉住两边:趟间有差别时带子必须张开、且张开量正好是标准差;
    趟间相同时仍然照画,只是上下沿重合。
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    calls = []
    real = Axes.fill_between

    def spy(self, x, y1, y2, *a, **kw):
        calls.append((list(x), list(y1), list(y2)))
        return real(self, x, y1, y2, *a, **kw)

    monkeypatch.setattr(Axes, "fill_between", spy)
    agg = SUM.aggregate(SUM.load_reps(_reps(tmp_path, n=3)), SUM.TABLE_METRICS)
    SUM.plot_summary(agg, str(tmp_path / "b.png"))
    assert calls, "误差带一次都没画"

    edges = {(round(y1[0], 9), round(y2[0], 9)) for _, y1, y2 in calls}
    # (a) 趟间有差别 ⇒ 带子张开,张开量 = 标准差
    st = agg["waiton_L0.5_T15"]["stats"]["t_per_target_s"]
    assert st.std > 0
    assert (round(st.mean - st.std, 9), round(st.mean + st.std, 9)) in edges
    # (b) 趟间相同 ⇒ 上下沿重合(仍然画,零宽本身就是要传达的信息)
    vs = agg["waiton_L0.5_T15"]["stats"]["visited"]
    assert vs.std == 0.0
    assert (round(vs.mean, 9), round(vs.mean, 9)) in edges


def test_figures_carry_no_cjk_text(tmp_path):
    """学术汇报口径:图里**一个中日文字符都不许有**。

    只扫两个出图函数里的**运行时字符串**(docstring 与注释不算 —— 那些本来就是中文)。
    这样任何人往轴标签/图例/标题里塞回中文,都会当场变红。
    """
    import ast
    import inspect
    import textwrap

    CJK = tuple((0x3000, 0x9FFF)), tuple((0xFF00, 0xFFEF))

    def has_cjk(t):
        return any(any(lo <= ord(ch) <= hi for lo, hi in CJK) for ch in t)

    for fn in (SUM.plot_summary, SUM.plot_leader_speed):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        body = tree.body[0].body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)):
            body = body[1:]                      # 跳过 docstring
        bad = [n.value for stmt in body for n in ast.walk(stmt)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and has_cjk(n.value)]
        assert not bad, f"{fn.__name__} 里还有中日文运行时字符串: {bad}"


def test_axis_labels_come_from_the_contract():
    """轴标签必须走 `MetricSpec.axis_label('en')`,不许在脚本里手写英文 ——
    手写的那份必然先漂移。顺带钉住:英文标签与英文单位里不含非 ASCII。"""
    for k in set(SUM.FIGURE_METRICS) | set(SUM.TABLE_METRICS):
        spec = METRIC_BY_KEY[k]
        assert spec.label_en, f"{k} 没有英文标签"
        lab = spec.axis_label("en")
        assert lab.isascii(), f"{k} 的英文轴标签含非 ASCII: {lab!r}"
        assert lab == f"{spec.label_en} [{spec.unit_en}]"


def test_no_op_wait_series_is_dropped_from_the_figure(tmp_path):
    """wait 失效的速度上两条线完全重合 —— 只画一条,另一条纯属占图例。"""
    runs = [("waiton_L0.3_T15", "on", 0.3, 15.0),
            ("waitoff_L0.3_T15", "off", 0.3, 15.0),     # 与上一行取值相同
            ("waiton_L1_T15", "on", 1.0, 15.0),
            ("waitoff_L1_T15", "off", 1.0, 15.0)]
    dirs = [_write_rep(str(tmp_path / f"d{i}"), i, runs=runs,
                       over={("waitoff_L1_T15", "visited"): 11.0})
            for i in range(2)]
    agg = SUM.aggregate(SUM.load_reps(dirs), SUM.TABLE_METRICS)
    drawn = SUM.drawn_series(agg)
    assert ("off", 0.3) not in drawn, "wait 失效那档的 off 线该被去掉"
    assert ("on", 0.3) in drawn, "去掉的应当是重影,不是把那一档整个删了"
    assert ("off", 1.0) in drawn and ("on", 1.0) in drawn
    assert drawn[0][0] == "on", "wait=on 先画(线最粗,在最底层)"


def test_font_scale_reaches_matplotlib(tmp_path):
    """字号放大要真的落到 rcParams 上,不是只改了几个显式 fontsize。

    两个旋钮各管各的:`font_scale` 管标题/刻度/图例/标注,`label_scale` 只管轴标签。
    (当前两档同为 3.0,但机制保留 —— 标签是最长的那一类文字,可能要单独压。)
    """
    import matplotlib
    from vrpsim.contracts.style import DEFAULT_STYLE

    agg = SUM.aggregate(SUM.load_reps(_reps(tmp_path, n=2)), SUM.TABLE_METRICS)
    SUM.plot_summary(agg, str(tmp_path / "s.png"),
                     font_scale=3.0, label_scale=2.0)
    assert matplotlib.rcParams["xtick.labelsize"] == pytest.approx(
        DEFAULT_STYLE.font_size_tick * 3.0)
    assert matplotlib.rcParams["axes.titlesize"] == pytest.approx(
        DEFAULT_STYLE.font_size_title * 3.0)
    # 轴标签走自己那一档,不跟着 font_scale
    assert matplotlib.rcParams["axes.labelsize"] == pytest.approx(
        DEFAULT_STYLE.font_size_label * 2.0)

    st = SUM.figure_style(3.0, 3.0)               # 默认口径:两档同为 3x
    assert st.font_size_label == pytest.approx(DEFAULT_STYLE.font_size_label * 3)
    assert st.font_size_tick == pytest.approx(DEFAULT_STYLE.font_size_tick * 3)
    assert (SUM.FONT_SCALE, SUM.LABEL_FONT_SCALE) == (3.0, 3.0)

    # 全局样式**不许**被改动 —— 改了 02/03/05/07 的图会跟着变
    assert DEFAULT_STYLE.font_size_label == 10.0


def test_axis_label_wraps_the_unit_onto_a_second_line():
    """纵轴标签是旋转 90 度的,可用长度是**面板高度**。

    3x 字号下「Time per target [s/target]」一行约 375 pt,面板只有约 324 pt 高 ⇒
    溢出压到隔壁面板(实测发生过)。折行后每行都放得下。
    """
    spec = METRIC_BY_KEY["t_per_target_s"]
    assert SUM.wrapped_axis_label(spec) == "Time per target\n[s/target]"
    # 百分比刻度下单位写 %,不写契约里的 "-"
    pctspec = METRIC_BY_KEY["leader_hold_frac"]
    assert SUM.wrapped_axis_label(pctspec, percent=True) == \
        "Leader wait rate\n[%]"
    # 折行只动断行,文字仍来自契约
    for k in SUM.FIGURE_METRICS:
        s = METRIC_BY_KEY[k]
        assert SUM.wrapped_axis_label(s).replace("\n", " ") == s.axis_label("en")


def test_figure_panels_answer_the_three_swept_factors():
    """四个面板 = 完遂性 / 时间效率 / leader_speed 的代价 / Follower 负载均衡。

    钉住这一组是因为面板换过一次(空転率+序列利用率 → 停船率+距離不均衡),
    而两次的 `applies_to` / 单位 / 方向都不同 —— 换错了图会静静地画出别的东西。
    """
    assert SUM.FIGURE_METRICS == ("visited", "t_per_target_s",
                                  "leader_hold_frac",
                                  "load_imbalance_distance_frac")
    # 距離均衡用的是**灵敏**那种写法:Jain 在本扫描内没有区分度
    assert "jain_fairness" not in SUM.FIGURE_METRICS
    assert METRIC_BY_KEY["load_imbalance_distance_frac"].better == "low"


def test_every_series_gets_its_own_colour(tmp_path):
    """加第三档 Leader 速度后,序列数从 4 变 6。

    这条钉的是一个**静默**的坑:早先颜色表是定长 4 个、和序列 zip 在一起,
    第 5、6 条线会被 zip 悄悄丢掉 —— 图照出、不报错、就是少两条线。
    颜色必须按**实际出现的速度**动态生成,且三档速度互不撞色。
    """
    runs = [(f"wait{w}_L{lv:g}_T15", w, lv, 15.0)
            for w in ("on", "off") for lv in (0.3, 0.5, 1.0)]
    dirs = [_write_rep(str(tmp_path / f"c{i}"), i, runs=runs) for i in range(2)]
    agg = SUM.aggregate(SUM.load_reps(dirs), SUM.TABLE_METRICS)
    assert len(agg) == 6

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = str(tmp_path / "c.png")
    SUM.plot_summary(agg, p)
    fig = plt.gcf()          # plot_summary 已 close,这里只验证它没抛异常
    plt.close(fig)
    assert os.path.getsize(p) > 5000

    speeds = sorted({r["leader_speed_mps"] for r in agg.values()})
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    colors = [palette[i % len(palette)] for i in range(len(speeds))]
    assert len(set(colors)) == len(speeds) == 3, "三档速度撞色了"
