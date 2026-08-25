"""評価指標の验收 —— 定义(contracts/metrics.py)与实现(summary())不许漂移。

这个文件存在的唯一理由:指标是要写进论文的。一旦"文档里写的定义"和"代码里算的东西"
对不上,论文里的数就成了错的,而且**不会有任何别的测试变红**。
"""
from __future__ import annotations

import numpy as np
import pytest

from vrpsim.contracts.metrics import (ALL_SCENARIOS, FAMILY_NAMES, HIGHER_BETTER,
                                      LOWER_BETTER, METRIC_BY_KEY, METRICS,
                                      SCENARIO_VRP, SHOULD_BE_FLAT,
                                      SUMMARY_METRIC_KEYS, metrics_for, metrics_of)
from vrpsim.contracts.mission import (STATUS_DWELL, STATUS_IDLE, STATUS_TRANSIT,
                                      MissionConfig)
from vrpsim.contracts.twophase import TwoPhaseConfig
from vrpsim.mission import feasibility_estimate, run_mission
from vrpsim.twophase import run_twophase
from vrpsim.world import build_mothra_world

FAST = dict(solver="greedy", vrp_time_limit_s=0.1)


@pytest.fixture(scope="module")
def mw():
    return build_mothra_world()


@pytest.fixture(scope="module")
def res(mw):
    return run_mission(MissionConfig(**FAST), mw)


# ======================================================================
# 定义与实现的一致性
# ======================================================================
def test_every_declared_metric_exists_in_summary(res):
    """声明了却没算 → 红。这是"论文里的定义"落到代码上的那一步。"""
    s = res.summary()
    missing = [k for k in SUMMARY_METRIC_KEYS if k not in s]
    assert not missing, f"contracts/metrics.py 声明了但 summary() 没算: {missing}"


def test_metric_specs_are_complete_and_unique():
    """每个指标都得有公式、白话定义、单位、方向 —— 缺一条就不是"清晰易懂"。"""
    keys = [m.key for m in METRICS]
    assert len(keys) == len(set(keys)), "指标 key 重复"
    for m in METRICS:
        assert m.formula.strip(), f"{m.key} 没有公式"
        assert len(m.definition.strip()) >= 20, f"{m.key} 的白话定义太短,说不清"
        assert m.unit.strip(), f"{m.key} 没有单位"
        assert m.better in (LOWER_BETTER, HIGHER_BETTER, SHOULD_BE_FLAT)
        assert m.family in FAMILY_NAMES, f"{m.key} 的指标族未登记"
        assert m.label_zh.strip() and m.label_ja.strip(), f"{m.key} 缺标签"
        assert m.source in ("summary", "sweep")
    assert METRIC_BY_KEY[keys[0]] is METRICS[0]


def test_every_family_has_metrics():
    for fam in FAMILY_NAMES:
        assert metrics_of(fam), f"指标族 {fam} 是空的"


def test_summary_metrics_are_finite_numbers(res):
    """指标值得能直接进表格:不许是 None、不许意外的 nan。"""
    s = res.summary()
    for k in SUMMARY_METRIC_KEYS:
        v = s[k]
        spec = METRIC_BY_KEY[k]
        if k == "timed_out":
            assert isinstance(v, bool)
            continue
        if spec.vector:
            assert isinstance(v, list) and len(v) == res.cfg.n_followers
            assert all(np.isfinite(x) for x in v), f"{k} 有非有限值"
            continue
        assert isinstance(v, (int, float)) and not isinstance(v, bool), f"{k} 不是数"
        assert np.isfinite(v), f"{k} = {v},全覆盖的正常运行里不该出现"


def test_fleet_distance_includes_the_leader(res):
    """跨场景比能耗必须用含 Leader 的口径,否则白送掉一整条测线。"""
    s = res.summary()
    assert s["leader_distance_m"] == pytest.approx(500.0), \
        "Leader 航程应恒等于测线长;停船不产生位移"
    assert s["fleet_distance_m"] == pytest.approx(
        s["leader_distance_m"] + sum(s["per_follower_distance_m"]))
    assert s["fleet_distance_m"] > s["total_distance_m"], \
        "fleet 必须严格大于只含 Follower 的 total —— 否则 Leader 被漏掉了"
    assert s["distance_per_target_m"] == pytest.approx(
        s["fleet_distance_m"] / s["visited"])
    assert s["max_distance_m"] == pytest.approx(max(s["per_follower_distance_m"]))


def test_leader_distance_is_independent_of_waiting(mw):
    """Leader 停船不产生位移 ⇒ 等多久航程都一样,这条把它钉死。"""
    on = run_mission(MissionConfig(**FAST), mw).summary()
    off = run_mission(MissionConfig(leader_wait_on_lagging_follower=False,
                                    leader_wait_on_endangered_target=False,
                                    leader_wait_on_turnaround=False,
                                    **FAST), mw).summary()
    assert on["leader_hold_frac"] > 0 and off["leader_hold_frac"] == 0.0
    assert on["leader_distance_m"] == pytest.approx(off["leader_distance_m"])


# ======================================================================
# 各指标的口径确实按公式算
# ======================================================================
def test_duty_fractions_use_the_active_window_and_sum_to_one(res):
    """三个稼働率分项必须恰好加和为 1,且分母是"有効区間"而非整条时间线。"""
    s = res.summary()
    total = (s["duty_transit_frac"] + s["duty_dwell_frac"] + s["duty_idle_frac"])
    assert total == pytest.approx(1.0), "TRANSIT/DWELL/IDLE 之外还有第四种状态?"
    assert s["duty_productive_frac"] == pytest.approx(1.0 - s["duty_idle_frac"])

    # 有効区間确实比整条时间线短(尾部有 settle_time_s 的收尾余量)
    assert res.active_mask.sum() < res.t_s.size
    # 用整条时间线算会高估空转 —— 正是不这么做的理由
    st_all = res.follower_status
    idle_all = float((st_all == STATUS_IDLE).mean())
    assert idle_all > s["duty_idle_frac"]


def test_duty_dwell_matches_the_observation_budget(res):
    """DWELL 时间总量 = 目标数 x dwell_time_s(+ 投影点停留),是一条免费的自检。"""
    cfg = res.cfg
    st = res.follower_status
    dwell_s = float((st == STATUS_DWELL).sum()) * cfg.dt_s
    # 真实目标的停留必然在内;投影点也停(dwell_at_projection 默认 True)所以只能给下界
    assert dwell_s >= res.n_targets * cfg.dwell_time_s - cfg.dt_s
    # 关掉投影点停留后,总量应当恰好收敛到"目标数 x 停留时长"
    r2 = run_mission(MissionConfig(dwell_at_projection=False, **FAST), res.cfg.sim
                     and build_mothra_world(res.cfg.sim))
    st2 = r2.follower_status
    dwell2 = float((st2 == STATUS_DWELL).sum()) * r2.cfg.dt_s
    assert dwell2 == pytest.approx(r2.n_targets * r2.cfg.dwell_time_s, abs=r2.cfg.dt_s)


def test_observation_time_quantiles_are_ordered(res):
    """mean / median / p90 必须自洽,且都落在 (0, t_complete]。"""
    s = res.summary()
    assert 0 < s["t_observation_median_s"] <= s["t_observation_p90_s"]
    assert s["t_observation_p90_s"] <= s["t_complete_s"] + 1e-9
    vt = res.visit_time_s[res.visited_mask]
    assert s["t_observation_mean_s"] == pytest.approx(float(vt.mean()))
    assert s["t_per_target_s"] == pytest.approx(s["t_complete_s"] / s["visited"])


def test_channel_duty_and_messages_per_target(res):
    s = res.summary()
    want = s["n_comm_messages"] * res.cfg.acoustic_hop_s / s["duration_s"]
    assert s["channel_duty_frac"] == pytest.approx(want)
    assert s["messages_per_target"] == pytest.approx(
        s["n_comm_messages"] / s["visited"])
    # 超过 1.0 就说明"不建模信道竞争"这个假设失效了 —— 默认参数下不该发生
    assert s["channel_duty_frac"] < 1.0


def test_churn_counts_match_the_assignment_log(res):
    """改道次数必须能从 assignments 逐条数出来,不是估的。"""
    s = res.summary()
    owner, total, swaps, moved = {}, 0, 0, set()
    for a in res.assignments:
        for w in a.wp_ids:
            total += 1
            if w in owner and owner[w] != a.follower_id:
                swaps += 1
                moved.add(w)
            owner[w] = a.follower_id
    assert s["wp_issue_total"] == total
    assert s["reassignment_count"] == swaps
    assert s["reassigned_wp_count"] == len(moved)
    assert s["wp_issues_per_target"] == pytest.approx(total / res.n_targets)
    assert s["sequence_utilisation"] == pytest.approx(res.n_targets / total)
    # 每个目标至少被下发过一次(否则它不可能被观测到)
    assert total >= res.n_targets
    assert s["reassigned_wp_count"] <= s["reassignment_count"]


def test_time_efficiency_is_bounded_by_one(res, mw):
    """时间效率 = 下界/实测 ∈ (0,1]:实测不可能快过理论下界。"""
    lb = feasibility_estimate(mw, res.cfg)["t_mission_reference_s"]
    eff = res.time_efficiency(lb)
    assert 0.0 < eff <= 1.0 + 1e-9, f"实测比理论下界还快({eff:.3f}) ⇒ 下界算错了"
    assert eff > 0.5, "离下界差一倍以上,不像正常运行"


# ======================================================================
# 未全覆盖的组也要能进对比表(D18)
# ======================================================================
@pytest.fixture(scope="module")
def res_missed(mw):
    """关掉等待策略 + Leader 提速 ⇒ Follower 跟不上,大量漏点。

    这就是 `contracts/metrics.py::time_efficiency_obs` 定义里引用的 `waitoff_L1_*`
    那组(L=1.0 m/s、等待全关),实测 10/22、実測効率 1.019 —— 正是"这一列可以
    超过 1.0"的现场。

    ⚠ 不要退回 L=0.5:那组只漏 1/22,收工时刻与基准**几乎相等**(2026-08-24 把投影点
      横向间隔定到 20 m 后实测两者都是 1139.0 s),"漏点组更早收工"这个前提直接不成立,
      三条陷阱断言全部落空。要钉的是**指标的性质**,就得用一个margin 稳的配置,
      而不是卡在 1 个目标的刀刃上。
    """
    r = run_mission(MissionConfig(leader_speed_mps=1.0,
                                  leader_wait_on_lagging_follower=False,
                                  leader_wait_on_endangered_target=False,
                                  leader_wait_on_turnaround=False,
                                  **FAST), mw)
    assert r.coverage < 1.0, "这个 fixture 的前提就是漏点;不漏就没什么好测的了"
    return r


def test_finish_time_equals_complete_time_under_full_coverage(res):
    """全覆盖时两把尺子必须逐位相同 —— 否则 D18 就是偷偷改了既有各组的数。"""
    assert res.t_finish_s == pytest.approx(res.t_complete_s)
    s = res.summary()
    assert s["t_finish_s"] == pytest.approx(s["t_complete_s"])
    assert s["t_per_target_s"] == pytest.approx(s["t_finish_s"] / s["visited"])


def test_finish_time_survives_missing_targets(res_missed):
    """漏点组:t_complete_s 仍是 nan(任务没达成),但 t_finish_s 必须给出实际收工时刻。"""
    s = res_missed.summary()
    assert np.isnan(s["t_complete_s"]), "漏了目标就不该有「完成时刻」"
    assert np.isfinite(s["t_finish_s"]) and s["t_finish_s"] > 0
    # 收工时刻 = max(走完测线, 最后一次観測) —— 与全覆盖时同一个公式
    assert s["t_finish_s"] == pytest.approx(
        max(s["t_leader_finish_s"], s["t_last_observation_s"]))
    assert np.isfinite(s["t_per_target_s"]), "对漏点中性的那一列必须有值"


def test_target_counts_are_in_the_summary(res, res_missed):
    """总目标数与観測成功数是主指标,不是可选项(用户要求;时间列必须与它们同读)。"""
    for r in (res, res_missed):
        s = r.summary()
        assert s["n_targets"] == 22, "本场地目标总数应恒为 22"
        assert 0 < s["visited"] <= s["n_targets"]
        assert s["coverage"] == pytest.approx(s["visited"] / s["n_targets"])
    assert res_missed.summary()["visited"] < res.summary()["visited"], \
        "关掉等待反而没少做?那 D16 的等待策略就是白加的"


def test_observed_efficiency_is_flattered_by_missing_targets(res, res_missed, mw):
    """**反向钉住这个陷阱**:漏点会把「実測」時間効率抬高。

    这条测试存在的理由:有人一定会直接拿 time_efficiency_obs 排序,然后得出
    「waitoff 更高效」的结论。这里把「它确实会给出那个假象」写死成断言,
    以后谁改动这几列,注释和数会一起被拽回来。
    """
    lb = feasibility_estimate(mw, res.cfg)["t_mission_reference_s"]
    lb_m = feasibility_estimate(mw, res_missed.cfg)["t_mission_reference_s"]

    # 漏点组更早收工 ⇒ 実測効率**更好看**,尽管它只做了一部分目标
    assert res_missed.t_finish_s < res.t_finish_s
    assert res_missed.time_efficiency_obs(lb_m) > res.time_efficiency_obs(lb), \
        "漏点组的実測効率没有虚高?那这条陷阱注释需要更新"


def test_coverage_discount_narrows_the_flattery_but_need_not_reverse_it(
        res, res_missed, mw):
    """覆盖折扣把漏点带来的虚高**大幅压回去**;是否翻转排序**看漏得多狠**,不保证。

    实测(greedy,`waitoff_L1` 组:L=1.0 + 等待全关,10/22):
        実測効率 1.019 vs 0.909(虚高 +0.110,**而且超过了 1.0** ——
                                 拿全任务的下界去除半任务的耗时,荒谬之处一目了然)
        折扣后   0.463 vs 0.909(不但压没了虚高,还翻转成 -0.446)
    折扣是**线性外推**:假设没做的目标与已做的一样费时。实际漏掉的是被 Leader 甩出
    窗口、只有停船等待才够得着的点,所以它偏乐观,是**上界**不是预测
    (见 contracts/metrics.py 的 time_efficiency_cov 定义)。

    ⚠ 漏得少时折扣**未必**翻转:同样 greedy、L=0.5 只漏 1/22 那组,虚高 +0.046
      压到 +0.005 仍未翻转。本用例只断言"压小",不断言"翻转" —— 名字里的
      `need_not_reverse` 就是这个意思。

    ⇒ **効率列本身分不出 wait on/off 的高下,能分出的是 `visited`。**
      这正是把「目標総数 / 観測成功数」放进对比表的理由(D18)。
    """
    lb = feasibility_estimate(mw, res.cfg)["t_mission_reference_s"]
    lb_m = feasibility_estimate(mw, res_missed.cfg)["t_mission_reference_s"]

    gap_obs = res_missed.time_efficiency_obs(lb_m) - res.time_efficiency_obs(lb)
    gap_cov = res_missed.time_efficiency_cov(lb_m) - res.time_efficiency_cov(lb)
    assert gap_obs > 0, "前提:実測列确实虚高"
    assert gap_cov < gap_obs, "折扣没有把虚高压小 ⇒ 这一列失去存在意义"
    assert gap_cov < 0.5 * gap_obs, f"折扣只压掉不到一半({gap_cov:.4f}/{gap_obs:.4f})"

    # 真正能分出高下的那一列:
    assert res_missed.summary()["visited"] < res.summary()["visited"]


def test_coverage_discounted_efficiency_matches_its_formula(res, res_missed, mw):
    for r in (res, res_missed):
        lb = feasibility_estimate(mw, r.cfg)["t_mission_reference_s"]
        assert r.time_efficiency_cov(lb) == pytest.approx(
            r.coverage * r.time_efficiency_obs(lb))
        # 折扣列永远不高于実測列;漏点时严格更低
        assert r.time_efficiency_cov(lb) <= r.time_efficiency_obs(lb) + 1e-12
        if r.coverage < 1.0:
            assert r.time_efficiency_cov(lb) < r.time_efficiency_obs(lb)
    # 全覆盖时三把尺子合一
    lb = feasibility_estimate(mw, res.cfg)["t_mission_reference_s"]
    assert res.time_efficiency_cov(lb) == pytest.approx(res.time_efficiency(lb))
    assert res.time_efficiency_obs(lb) == pytest.approx(res.time_efficiency(lb))


def test_efficiency_numerator_changes_regime_across_leader_speeds(mw):
    """効率列的分子是个 `max()`,会随 Leader 速度**换主导项** —— 所以
    `time_efficiency*` **不能跨 Leader 速度比**(D20)。

    这条钉的是契约里那句规则的**事实依据**:早先文件头写着「各组共用同一个分子
    常数,纯粹是 1/t_finish 的统一缩放」,那只在 L ∈ {0.5, 1.0} 时成立。
    加了 0.3 档(**比 Follower 慢**的一侧)后测线项反超 VRP 项,分子跳了 54%。
    谁要是日后改了 `feasibility_estimate` 的取值方式,这条会把他拽回来。
    """
    num = {lv: feasibility_estimate(
        mw, MissionConfig(leader_speed_mps=lv, **FAST))["t_mission_reference_s"]
        for lv in (0.3, 0.5, 1.0)}

    # L >= 0.5:VRP 路线主导 ⇒ 分子与 Leader 速度无关,同一把尺子
    assert num[0.5] == pytest.approx(num[1.0])
    # L = 0.3:测线 500/0.3 = 1666.7 s 反超 ⇒ 换了主导项
    assert num[0.3] == pytest.approx(500.0 / 0.3, rel=1e-6)
    assert num[0.3] > num[0.5] * 1.5, "0.3 档的分子应当明显更大(换了主导项)"

    # ⇒ 同一个 t_finish 在 0.3 档上会得到虚高的效率;这不是配置更好,
    #   只是「快不过 Leader 自己走完测线」。
    t_finish = 1800.0
    assert num[0.3] / t_finish > num[0.5] / t_finish * 1.5


def test_active_window_is_truncated_at_finish_even_without_full_coverage(res_missed):
    """漏点组的有効区間必须仍按 t_finish_s 截断,不许退回整条时间线(D18)。

    D18 之前分界取 `t_complete_s`,漏点时是 nan ⇒ 整条时间线都算进去,
    连收工之后那段都被计入稼働率,与全覆盖组根本不在一把尺子上。

    ⚠ 不断言「空转率因此下降」:实测尾段 91.7% 是 TRANSIT(还在赶往够不着的点)、
      只有 0.4% 是 IDLE,截掉它反而让空转率略升。方向不是重点,**同一把尺子**才是。
    """
    assert not np.isfinite(res_missed.t_complete_s), "这个 fixture 的前提是漏点"
    tf = res_missed.t_finish_s
    assert np.isfinite(tf)
    # 分界确实是 t_finish_s,而不是整条时间线
    assert np.array_equal(res_missed.active_mask, res_missed.t_s <= tf + 1e-9)
    assert res_missed.active_mask.sum() < res_missed.t_s.size

    s = res_missed.summary()
    assert (s["duty_transit_frac"] + s["duty_dwell_frac"]
            + s["duty_idle_frac"]) == pytest.approx(1.0)
    # 截断确实改变了结果(否则这个机制形同虚设)
    st_all = res_missed.follower_status
    assert s["duty_idle_frac"] != pytest.approx(
        float((st_all == STATUS_IDLE).mean()), abs=1e-6)


# ======================================================================
# 区分度 —— 指标得真的能把参数组分开,否则放进表里只是凑数
# ======================================================================
def test_planning_period_metrics_actually_discriminate(mw):
    """拉长规划周期时,計画品質一族必须单调变化。

    这是这些指标存在的理由:`coverage` 之类的终点量在各组之间恒为 1.0,
    分辨不出任何东西;真正有区分度的是过程量。
    """
    runs = {T: run_mission(MissionConfig(plan_period_s=T, **FAST), mw)
            for T in (30.0, 60.0, 90.0)}
    for r in runs.values():
        assert r.coverage == 1.0, "各组都得完成任务,否则不可比"

    issues = [runs[T].summary()["wp_issues_per_target"] for T in (30.0, 60.0, 90.0)]
    assert issues[0] > issues[1] > issues[2], \
        f"规划越频繁,每目标下发次数就该越多,实测 {issues}"
    util = [runs[T].summary()["sequence_utilisation"] for T in (30.0, 60.0, 90.0)]
    assert util[0] < util[1] < util[2], f"序列利用率应当反向单调,实测 {util}"
    swaps = [runs[T].summary()["reassignment_count"] for T in (30.0, 60.0, 90.0)]
    assert swaps[0] > swaps[-1], f"规划越频繁改道越多,实测 {swaps}"


# ======================================================================
# 三方法比较的口径(D19)
# ======================================================================
def test_normalised_efficiency_is_scoped_to_vrp():
    """**反向钉住循环论证**:`time_efficiency*` 三列只许用于局部VRP。

    它们的分子 `t_mission_reference_s` 本身就是一次全局 min-max VRP 的解 ——
    放进三方法表会让「全局VRP方法」按构造得 ≈1.0,胜负在定义里就决定了。
    另外分子里还含 `测线长/v_leader`,对 lawnmower 根本没有定义。
    """
    for key in ("time_efficiency", "time_efficiency_obs", "time_efficiency_cov"):
        assert METRIC_BY_KEY[key].applies_to == (SCENARIO_VRP,), \
            f"{key} 被放开到三方法表了 —— 那会恢复循环论证"
    visible = {m.key for m in metrics_for(*ALL_SCENARIOS)}
    assert not (visible & {"time_efficiency", "time_efficiency_obs",
                           "time_efficiency_cov"})
    # 跨方法的時間効率口径必须在表里
    assert "t_per_target_s" in visible, "不归一化的那把尺子被挡掉了?"
    assert "t_observation_mean_s" in visible


def test_cross_method_metrics_are_defined_for_all_three(mw):
    """`applies_to` 含三方法的指标,三个场景的 summary 里都得真的有。"""
    a = run_mission(MissionConfig(**FAST), mw).summary()
    b = run_twophase(TwoPhaseConfig(base=MissionConfig(**FAST)), mw).summary()
    shared = [m for m in metrics_for(*ALL_SCENARIOS) if m.source == "summary"]
    # lawnmower 的部分派生量由 09 脚本按同口径补(它的 summary 更精简),
    # 这里检 vrp 与 twophase 两个直接产出 summary 的场景。
    for name, s in (("vrp", a), ("twophase", b)):
        missing = [m.key for m in shared if m.key not in s]
        assert not missing, f"{name} 的 summary 缺三方法通用指标: {missing}"


# ======================================================================
# ⑥ 負荷均衡(D19)
# ======================================================================
def test_balance_metrics_match_their_formulas(res):
    """四列均衡指标逐条按公式对拍 —— 从原始数组重算一遍,不信 summary 自报。"""
    s = res.summary()
    d = np.asarray(s["per_follower_distance_m"], dtype=float)
    t = np.asarray(res.busy_time_s, dtype=float)

    assert s["jain_fairness"] == pytest.approx(d.sum() ** 2 / (d.size * (d ** 2).sum()))
    assert s["jain_fairness_time"] == pytest.approx(
        t.sum() ** 2 / (t.size * (t ** 2).sum()))
    assert s["load_imbalance_distance_frac"] == pytest.approx(
        (d.max() - d.min()) / d.max())
    assert s["load_imbalance_time_frac"] == pytest.approx(
        (t.max() - t.min()) / t.max())

    # 稼働時間用的是"非 IDLE 的时长",不是任务总时长(那样各台会一样,量不出东西)
    dt = res.cfg.dt_s
    want = (res.follower_status != STATUS_IDLE).sum(axis=0) * dt
    assert np.allclose(t, want)
    assert (t < res.t_s[-1]).all(), "稼働時間不该等于整条时间线"


def test_max_busy_time_is_the_longest_working_vehicle_not_the_mission_length(res):
    """② 稼働率の**絶対量**(2026-08-15 人工指定で `duty_idle_frac` と差し替え)。

    钉すのは三つ:
      1. 式どおり `max(busy_time_s)` であること(summary の自報を信じない);
      2. **Leader を含まない** —— 対象は観測を担う機体だけ;
      3. **任務長そのものではない** —— IDLE を差し引いた実働時間なので必ず短い。
         ここが崩れると「電池の律速」という読み方が成立しなくなる。
    """
    s = res.summary()
    t = np.asarray(res.busy_time_s, dtype=float)
    assert s["max_busy_time_s"] == pytest.approx(t.max())
    assert len(t) == res.follower_pos.shape[1], "Leader が混ざっている"
    assert s["max_busy_time_s"] < res.t_s[-1], "任務長と同じなら IDLE を引けていない"


def test_distance_balance_and_time_balance_can_disagree(mw):
    """**本族存在的理由**:min-max VRP 均衡的是距离,而完成时刻由时间决定。

    二段式实测切成 [18 点 452.1 m, 4 点 451.2 m] —— 距离几乎相等(不均衡 0.2%)
    但时间差 141.8 s(不均衡 13.1%)。这条把这个分叉钉成事实,
    后人若改动求解器(给 OR-Tools 加 service-time 维度)会被它拽回来。
    """
    s = run_twophase(TwoPhaseConfig(base=MissionConfig())).summary()
    dd = s["load_imbalance_distance_frac"]
    tt = s["load_imbalance_time_frac"]
    assert dd < 0.02, f"距離不均衡应当很小(min-max VRP 优化的就是它),实测 {dd:.3f}"
    assert tt > 0.10, f"時間不均衡应当明显,实测 {tt:.3f}"
    assert tt > 5 * dd, "两者没有分叉 ⇒ 这一族就没有存在的必要了"
    # 点数差得很多,而距离几乎一样 —— 这就是分叉的来源
    assert max(s["route_counts"]) >= 3 * min(s["route_counts"])


def test_jain_is_too_blunt_to_see_it(mw):
    """反向钉住:同一组数据 Jain 看不出来 —— 这正是要额外加 max-min 那两列的理由。"""
    s = run_twophase(TwoPhaseConfig(base=MissionConfig())).summary()
    assert s["jain_fairness_time"] > 0.99, \
        "Jain 居然分辨出来了?那 load_imbalance_* 的必要性说明需要更新"
    assert s["load_imbalance_time_frac"] > 0.10, "而 max-min 口径看得很清楚"


def test_terminal_metrics_do_not_discriminate_across_planning_period(mw):
    """反向钉住:**只改规划周期**时终点类指标是常数 —— 那个维度上别拿它们比效率。

    ⚠ 换成 wait on/off 这个维度就完全不是这样了:覆盖率会从 100% 掉到 36%,
      `visited` 是那里最重要的一列。见 test_target_counts_are_in_the_summary。
    """
    a = run_mission(MissionConfig(plan_period_s=30.0, **FAST), mw).summary()
    b = run_mission(MissionConfig(plan_period_s=90.0, **FAST), mw).summary()
    for k in ("coverage", "visited", "n_targets"):
        assert a[k] == b[k], f"{k} 居然区分开了,那本条注释需要更新"