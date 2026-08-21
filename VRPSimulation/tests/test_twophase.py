"""二段式全局VRP 基线の验收(D19)。

这个基线存在的唯一理由:与本文提案**只差一个变量** —— 探査与観測是并行还是串行。
所以本文件最重要的两条是:
  1. `test_phases_are_strictly_serial`   —— 串行这件事必须真的成立
  2. `test_kinematics_come_from_the_same_mission_config` —— 其余一切必须相同
少了任何一条,这个对照组就不能证明任何东西。
"""
from __future__ import annotations

import numpy as np
import pytest

from vrpsim.contracts.mission import (STATUS_DWELL, STATUS_IDLE, STATUS_TRANSIT,
                                      MissionConfig)
from vrpsim.contracts.twophase import TwoPhaseConfig
from vrpsim.mission import run_mission
from vrpsim.twophase import run_twophase
from vrpsim.world import build_mothra_world

FAST = dict(solver="greedy", vrp_time_limit_s=0.1)


@pytest.fixture(scope="module")
def mw():
    return build_mothra_world()


@pytest.fixture(scope="module")
def res(mw):
    return run_twophase(TwoPhaseConfig(base=MissionConfig(**FAST)), mw)


# ======================================================================
# 阶段结构 —— 这个基线的定义本身
# ======================================================================
def test_survey_phase_detects_every_target(res):
    """阶段1 走完 ⇒ 目标表完整。窗口宽 100 m == 场地 East 跨度,不该漏任何一个。"""
    assert res.coverage == 1.0, f"漏了 {res.missed_wp_ids}"
    assert res.summary()["visited"] == res.n_targets == 22
    # 阶段1 的时长是解析量:测线长 / Leader 速度
    assert res.t_survey_s == pytest.approx(500.0 / res.cfg.leader_speed_mps)


def test_phases_are_strictly_serial(res):
    """**本基线的定义**:没有任何一拍是「Leader 还在走 且 某台正在観測」。

    这条一旦红,二段式就退化成了本文提案的某种变体,整个对比失去意义。
    """
    moving = res.leader_north_m < 500.0 - 1e-9        # Leader 还没走完测线
    busy = (res.vehicle_status != STATUS_IDLE).any(axis=1)   # 至少一台在干活
    overlap = moving & busy
    assert not overlap.any(), (
        f"有 {int(overlap.sum())} 拍探査与観測重叠了 —— 那就不是串行基线了;"
        f"首次重叠于 t={res.t_s[np.argmax(overlap)]:.1f}s")

    # 反过来:阶段2 里 Leader 必须已经停在终点
    after = res.t_s > res.t_survey_s + 1e-9
    assert np.allclose(res.leader_north_m[after], 500.0)


def test_no_replanning_in_phase_two(res):
    """全局只规划一次 —— 允许重规划它就变成本文提案了。"""
    assert res.n_vrp_solves == 1
    assert res.summary()["n_plan_rounds"] == 1
    # 每个目标恰好被下发一次 ⇒ 从不改派,序列利用率 100%
    issued = sum(len(r) for r in res.routes)
    assert issued == res.n_targets
    assert res.summary()["sequence_utilisation"] == pytest.approx(1.0)
    assert res.summary()["reassignment_count"] == 0
    # 两条路线互斥且并起来正好是全部目标
    flat = [j for r in res.routes for j in r]
    assert sorted(flat) == list(range(res.n_targets))


def test_replan_switch_is_refused():
    """`replan=True` 必须当场炸 —— 它会让这个基线名不副实。"""
    with pytest.raises(ValueError, match="重规划"):
        TwoPhaseConfig(replan=True)


def test_no_acoustic_communication_is_needed(res):
    """阶段2 开环执行 ⇒ 声学报文恰为 0。**这是对照组的优势,如实记 0 不记 —。**"""
    s = res.summary()
    assert s["n_comm_messages"] == 0
    assert s["channel_duty_frac"] == 0.0


# ======================================================================
# 三个场景必须共用同一份参数
# ======================================================================
def test_kinematics_come_from_the_same_mission_config():
    """速度 / 停留 / 步长只有一个真相源;台数也由起点表派生,不另立字段。

    照抄 `test_lawnmower.py` 的同名测试 —— D15 在 `06_run_lawnmower.py --dwell`
    上踩过一次「手抄参数」的坑,三个场景之后只会更容易漂移。
    """
    base = MissionConfig(follower_speed_mps=0.37, dwell_time_s=12.5,
                         leader_speed_mps=0.81)
    cfg = TwoPhaseConfig(base=base)
    assert cfg.follower_speed_mps == 0.37
    assert cfg.dwell_time_s == 12.5
    assert cfg.leader_speed_mps == 0.81
    assert cfg.dt_s == base.dt_s
    assert cfg.sim is base.sim
    # 台数与起点都由 base 派生 —— 没有第二个真相源
    assert cfg.n_vehicles == base.n_followers == 2
    assert cfg.vehicle_starts_ned == base.follower_starts_ned
    # 喂给 Follower 的运动学配置就是 base 本身
    assert cfg.vehicle_cfg is base


def test_the_only_difference_from_the_proposal_is_serialisation(mw):
    """与本文提案跑同一份 `MissionConfig` ⇒ 差异只可能来自串行。"""
    base = MissionConfig(**FAST)
    a = run_mission(base, mw).summary()
    b = run_twophase(TwoPhaseConfig(base=base), mw).summary()
    for k in ("n_targets", "coverage", "visited"):
        assert a[k] == b[k], f"{k} 不同 ⇒ 两个场景跑的不是同一个任务"
    assert a["leader_distance_m"] == pytest.approx(b["leader_distance_m"]), \
        "Leader 都要走完同一条测线"


# ======================================================================
# 时间 —— 本对照组要证明的那件事
# ======================================================================
def test_two_phase_is_slower_than_concurrent(mw):
    """串行必然慢于并行,且慢出来的量**约等于被浪费掉的探査时长**。"""
    base = MissionConfig(**FAST)
    a = run_mission(base, mw)
    b = run_twophase(TwoPhaseConfig(base=base), mw)
    assert b.t_finish_s > a.t_finish_s, "串行居然不比并行慢?那这个对照组说明不了什么"
    # 二段式 = 探査 + 観測,两段严格相加
    assert b.t_finish_s == pytest.approx(b.t_survey_s + b.t_observe_s)
    # 提案省下的接近整个探査段(不完全等于:提案自己也要等 Leader 推进窗口)
    assert (b.t_finish_s - a.t_finish_s) > 0.5 * b.t_survey_s


def test_followers_idle_through_the_survey_phase(res):
    """口径 T1:Follower 在阶段1 全程待机 ⇒ 空转率必然很高。这是串行的真实代价。"""
    before = res.t_s < res.t_survey_s - 1e-9
    assert (res.vehicle_status[before] == STATUS_IDLE).all(), \
        "阶段1 里 Follower 动了 ⇒ 口径 T1(不预置前移)被破坏了"
    # 而提案方法几乎不空转 —— 对比表里这一列的差距就是串行的代价
    assert res.summary()["duty_idle_frac"] > 0.3


def test_per_target_time_is_the_cross_method_yardstick(res):
    """跨方法時間効率的口径(D19):単位目標あたり時間,不归一化。"""
    s = res.summary()
    assert s["t_per_target_s"] == pytest.approx(s["t_finish_s"] / s["visited"])
    assert np.isfinite(s["t_per_target_s"])


def test_phase_split_is_recorded(res):
    s = res.summary()
    assert s["t_survey_s"] > 0 and s["t_observe_s"] > 0
    assert s["t_survey_s"] + s["t_observe_s"] == pytest.approx(s["t_finish_s"])
    assert s["route_counts"] == [len(r) for r in res.routes]
    assert sum(s["route_counts"]) == res.n_targets
