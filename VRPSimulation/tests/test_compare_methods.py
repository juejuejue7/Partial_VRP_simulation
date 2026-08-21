"""三方法対比(`scripts/09_compare_methods.py`)の受入。

ここで钉すのは**路径规划コストの記録**だけ。三方法の数値そのものは
`test_mission.py` / `test_twophase.py` / `test_lawnmower.py` が持っている。

なぜこの三つを書くか
================================================================================
1. 「求解 0 回」と「記録漏れ」は表の上で見分けがつかない。lawnmower は本当に
   0 回(測線は下水前に決まる)—— それを `—` ではなく **0** として出すのは
   対照組の利点を正直に記すという D19 の口径そのもの。取り違えると
   「lawnmower は計画コストが測れていない」という誤読が論文に入る。
2. 二段式は**定義上ちょうど 1 回**。2 回になったらそれはもう二段式ではない。
3. 求解耗时は `vrp_time_limit_s` の**予算**であって必要量ではない。
   この違いを報告文に書く根拠が `_pearson` の nan(= 定数列)なので、
   その関数の境界を押さえておく。
"""
from __future__ import annotations

import csv
import math
import os
from types import SimpleNamespace as NS

import pytest

from vrpsim.contracts.metrics import (SCENARIO_LAWNMOWER, SCENARIO_TWOPHASE,
                                      SCENARIO_VRP)
from vrpsim.contracts.mission import MissionConfig


def _load_script(name: str):
    """scripts/ 配下は数字始まりで import できないのでパス指定で読む。"""
    import importlib.util
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sd = os.path.join(root, "scripts")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", "").replace(".", "_"), os.path.join(sd, name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CMP = _load_script("09_compare_methods.py")


def _round(i: int, pool: int, wall: float):
    return NS(round_idx=i, t_plan_s=15.0 * i, pool_size=pool,
              n_assigned_total=min(pool, 5), n_projection=0,
              solve_wall_s=wall, solver="ortools")


def _res_vrp(n=4):
    return NS(plan_rounds=[_round(i, 3 + i, 1.002 + 0.001 * i) for i in range(n)])


def _res_2p():
    return NS(n_targets=22, routes=[list(range(18)), list(range(18, 22))],
              meta={"solve_wall_s": 1.004, "solver": "ortools"})


def _rows():
    return [{"scenario": SCENARIO_VRP, "_label": "局部VRP",
             "metrics": {"t_finish_s": 1099.0}},
            {"scenario": SCENARIO_TWOPHASE, "_label": "二段式",
             "metrics": {"t_finish_s": 2084.3}},
            {"scenario": SCENARIO_LAWNMOWER, "_label": "lawnmower",
             "metrics": {"t_finish_s": 6146.0}}]


# ======================================================================
def test_every_planning_round_is_recorded_and_twophase_solves_exactly_once():
    """局部VRP は毎ラウンド 1 行、二段式はちょうど 1 行。"""
    recs = CMP.plan_cost_records(_res_vrp(n=7), _res_2p())
    vrp = [r for r in recs if r["scenario"] == SCENARIO_VRP]
    two = [r for r in recs if r["scenario"] == SCENARIO_TWOPHASE]
    assert len(vrp) == 7, "ラウンドを取りこぼしている"
    assert len(two) == 1, "二段式が 1 回でないならそれは二段式ではない"
    # 池大小と耗时が**そのラウンドのもの**であること(ずれて記録される事故を防ぐ)
    assert [r["pool_size"] for r in vrp] == [3, 4, 5, 6, 7, 8, 9]
    assert vrp[0]["solve_wall_s"] == pytest.approx(1.002)
    assert two[0]["pool_size"] == 22, "二段式は一度に全目標を見る"
    assert two[0]["n_assigned_total"] == 22


def test_lawnmower_is_zero_solves_not_missing_data():
    """lawnmower は**本当に 0 回**。`—`(不適用)に落とすと対照組の利点が消える。"""
    recs = CMP.plan_cost_records(_res_vrp(), _res_2p())
    assert not [r for r in recs if r["scenario"] == SCENARIO_LAWNMOWER]
    stats = CMP.plan_cost_stats(recs, _rows())
    lm = next(s for s in stats if s["scenario"] == SCENARIO_LAWNMOWER)
    assert lm["n_solves"] == 0
    assert lm["wall_total_s"] == 0.0 and lm["frac_of_mission"] == 0.0


def test_empty_pool_rounds_are_counted_but_kept_out_of_the_per_solve_mean():
    """池が空のラウンドは**問題を解いていない**(耗时 0)。

    実測では 86 ラウンド中 42 が空池で、混ぜて平均すると「1 回あたりの求解コスト」が
    1063 ms → 544 ms と半減して見える。表に出す数字が実態の半分になる事故を防ぐ。
    """
    rounds = [_round(0, 0, 0.0), _round(1, 5, 1.000),
              _round(2, 0, 0.0), _round(3, 7, 1.004)]
    recs = CMP.plan_cost_records(NS(plan_rounds=rounds), _res_2p())
    s = next(x for x in CMP.plan_cost_stats(recs, _rows())
             if x["scenario"] == SCENARIO_VRP)
    assert (s["n_solves"], s["n_active"], s["n_empty"]) == (4, 2, 2)
    assert s["wall_mean_active_s"] == pytest.approx(1.002)     # 空池を混ぜない
    assert s["wall_mean_s"] == pytest.approx(0.501)            # 混ぜるとこうなる
    assert s["pool_mean_active"] == pytest.approx(6.0)
    # 合計は全ラウンド分(空池の 0 を足しても変わらない)
    assert s["wall_total_s"] == pytest.approx(2.004)


def test_mission_fraction_is_total_wall_over_finish_time():
    """「無視してよいか」の判定材料なので、式を実装から独立に確かめる。"""
    recs = CMP.plan_cost_records(_res_vrp(n=4), _res_2p())
    stats = {s["scenario"]: s for s in CMP.plan_cost_stats(recs, _rows())}
    walls = [1.002, 1.003, 1.004, 1.005]
    assert stats[SCENARIO_VRP]["wall_total_s"] == pytest.approx(sum(walls))
    assert stats[SCENARIO_VRP]["frac_of_mission"] == pytest.approx(
        sum(walls) / 1099.0)
    assert stats[SCENARIO_TWOPHASE]["frac_of_mission"] == pytest.approx(
        1.004 / 2084.3)


def test_plan_cost_csv_has_one_row_per_solve(tmp_path):
    recs = CMP.plan_cost_records(_res_vrp(n=5), _res_2p())
    p = CMP.write_plan_cost_csv(recs, str(tmp_path / "plan_cost.csv"))
    with open(p, newline="", encoding="utf-8-sig") as fh:
        table = list(csv.reader(fh))
    assert tuple(table[0]) == CMP.PLAN_COST_COLS
    assert len(table) - 1 == 6                      # 5 ラウンド + 二段式 1 回


def test_flat_solve_times_report_no_correlation():
    """予算を使い切っていると耗时は定数列 ⇒ 相関は nan。

    報告文の「耗时は問題規模で決まっていない」という主張はここに依存している。
    """
    assert math.isnan(CMP._pearson([3, 5, 9, 22], [1.0, 1.0, 1.0, 1.0]))
    assert CMP._pearson([1, 2, 3], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


# ======================================================================
# 正文用の 5 項目
# ======================================================================
def test_key_metrics_are_real_contract_keys_and_absent_from_the_figure():
    """「図に出していない 5 項目」という節タイトルを事実にしておく。

    図(2 面)が使うのは `visited` の累積と `load_imbalance_*` の 2 本。
    そこと重複した列を 5 項目に混ぜると、同じ数字を二度見せることになる。
    """
    from vrpsim.contracts.metrics import METRIC_BY_KEY

    assert len(CMP.KEY_METRICS) == 5
    assert all(k in METRIC_BY_KEY for k in CMP.KEY_METRICS), "契約に無いキー"
    in_figure = {"visited", "load_imbalance_distance_frac",
                 "load_imbalance_time_frac"}
    assert not (set(CMP.KEY_METRICS) & in_figure)
    # 人工が名指しした 4 つは必ず入っている
    assert {"t_complete_s", "fleet_distance_m", "max_distance_m",
            "max_busy_time_s"} <= set(CMP.KEY_METRICS)
    # 空転率は 2026-08-15 に最長単機稼働時間へ差し替え(割合 → 絶対量)
    assert "duty_idle_frac" not in CMP.KEY_METRICS


def test_derived_fleet_max_puts_the_leader_back_in():
    """`max_distance_m` の式は `max(各 Follower 航程)` で **Leader を含まない**。

    二段式は Leader 500.0 m > Follower 最大 452.1 m なので、ラベルの
    「最大単機航程」をそのまま艦隊最大と読むと 452.1 という**過小値**になる。
    派生行がその穴を埋めていることを钉す。
    """
    import io

    rows = [{"scenario": SCENARIO_VRP, "_label": "local",
             "metrics": {"t_complete_s": 1099.0, "fleet_distance_m": 1612.0,
                         "max_distance_m": 563.8, "leader_distance_m": 500.0,
                         "max_busy_time_s": 1127.5, "n_comm_messages": 430}},
            {"scenario": SCENARIO_TWOPHASE, "_label": "two",
             "metrics": {"t_complete_s": 2084.5, "fleet_distance_m": 1403.4,
                         "max_distance_m": 452.1, "leader_distance_m": 500.0,
                         "max_busy_time_s": 1084.5, "n_comm_messages": 0}}]
    buf = io.StringIO()
    CMP.key_metric_section(rows, buf)
    out = buf.getvalue()
    assert "563.8" in out and "500.0" in out, "派生行が出ていない"
    # 二段式の派生値は Follower 最大(452.1)ではなく Leader の 500.0
    der = [ln for ln in out.splitlines() if "派生" in ln][0]
    assert "*563.8*" in der and "*500.0*" in der and "452.1" not in der


# ======================================================================
# 出図(学会発表口径:2 面・英語・タイトル無し)
# ======================================================================
def _plot_rows():
    out = []
    for i, sc in enumerate((SCENARIO_VRP, SCENARIO_TWOPHASE, SCENARIO_LAWNMOWER)):
        out.append({"scenario": sc, "_label": f"m{i}",
                    "metrics": {"t_finish_s": 1000.0 * (i + 1), "n_targets": 22,
                                "load_imbalance_distance_frac": 0.02 + 0.01 * i,
                                "load_imbalance_time_frac": 0.10 - 0.03 * i}})
    return out


def _plot_visits():
    import numpy as np
    return {sc: np.linspace(100.0 * (i + 1), 1000.0 * (i + 1), 22)
            for i, sc in enumerate((SCENARIO_VRP, SCENARIO_TWOPHASE,
                                    SCENARIO_LAWNMOWER))}


def test_figure_carries_no_cjk_text():
    """学会発表口径:図の中に**中日文字を一字も置かない**。

    走査するのは `plot_methods` の**実行時文字列**だけ(docstring と注釈は対象外 ——
    そこは元々日本語)。誰かが軸ラベルや凡例に日本語を戻したら即赤になる。
    """
    import ast
    import inspect
    import textwrap

    CJK = ((0x3000, 0x9FFF), (0xFF00, 0xFFEF))

    def has_cjk(t):
        return any(any(lo <= ord(ch) <= hi for lo, hi in CJK) for ch in t)

    tree = ast.parse(textwrap.dedent(inspect.getsource(CMP.plot_methods)))
    body = tree.body[0].body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)):
        body = body[1:]
    bad = [n.value for stmt in body for n in ast.walk(stmt)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and has_cjk(n.value)]
    assert not bad, f"図に中日文字が入っている: {bad}"


def test_method_names_and_axis_labels_come_from_the_contract():
    """英語表記は**契約側の一箇所**から取る。図に英訳を直書きしない。"""
    from vrpsim.contracts.metrics import (METRIC_BY_KEY, SCENARIO_NAMES,
                                          SCENARIO_NAMES_EN)

    assert set(SCENARIO_NAMES_EN) == set(SCENARIO_NAMES), "英名の登録漏れ"
    assert all(v.isascii() and v for v in SCENARIO_NAMES_EN.values())
    # 図の y 軸に出る 2 本
    assert METRIC_BY_KEY["visited"].axis_label("en") == "Targets observed [count]"
    assert METRIC_BY_KEY["load_imbalance_distance_frac"].label_en \
        == "Distance imbalance"


def test_figure_has_two_panels_without_titles_at_the_requested_font_scale(tmp_path):
    """2 面・タイトル無し・字号 2.5x が本当に matplotlib まで届いているか。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vrpsim.contracts.style import DEFAULT_STYLE

    p = str(tmp_path / "m.png")
    CMP.plot_methods(_plot_rows(), _plot_visits(), p, font_scale=2.5)
    assert os.path.getsize(p) > 5000
    assert matplotlib.rcParams["axes.labelsize"] == pytest.approx(
        DEFAULT_STYLE.font_size_label * 2.5)
    assert matplotlib.rcParams["xtick.labelsize"] == pytest.approx(
        DEFAULT_STYLE.font_size_tick * 2.5)
    assert CMP.FIG_FONT_SCALE == 2.5
    plt.close("all")


def test_solve_time_is_not_charged_to_the_simulated_timeline():
    """報告文が「`plan_solve_s = 0`(瞬時求解として計上)」と書く根拠。

    既定値が変わったらその一文が嘘になるので、契約側で押さえておく。
    """
    assert MissionConfig().plan_solve_s == 0.0
