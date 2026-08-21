"""[A1 验收测试 · L3 commitment] 承诺队列解析确定量(d_rem / t_avail)。

判据来源:contracts/mdp.py(resolve_remaining_distance / resolve_available_time docstring;
纪律4:解析**确定量**非估计)、DECISIONS §v1.0.1(只累计相邻段间转角、忽略首段转入角)。

**单一真相源交叉验证**(独立于被测实现内部算法,避免共因 bug):
在**同一组 (start_pose / current_pos, waypoints)** 上,
- d_rem ↔ 折线相邻段长之和(QA 独立累加,不复用 commitment 内部);
- t_avail ↔ `Level1RolloutEngine.execute(...).duration_s`(同口径单一真相源)。
二者数值一致(容差 ~1e-12)。

口径对齐说明(§v1.0.1):resolve_* 只有 Position 无朝向 → 不含首段转入角;
Level1 execute 同口径忽略 start_pose.psi 的首段转入角 → 对 execute 传入的
start_pose.psi 任取(此处取 0)不影响 duration_s,二者逐项一致。
另:cursor>0 时 resolve_* 用 (current_pos → 队列内 cursor 起剩余点),
对应 Level1 的 execute(start=current_pos, waypoints=剩余点)。
"""
from __future__ import annotations

import numpy as np
import pytest

from msim.contracts.config import RolloutConfig
from msim.mdp.commitment import (
    FollowerCommitment,
    resolve_available_time,
    resolve_remaining_distance,
)
from msim.physics.rollout import Level1RolloutEngine

_TOL = 1e-12


@pytest.fixture
def rcfg() -> RolloutConfig:
    return RolloutConfig()


# ----------------------------------------------------------------------
# QA 独立解析参考(不复用被测实现:防共因 bug)
# ----------------------------------------------------------------------
def _ref_remaining_distance(current_pos, remaining_pts) -> float:
    """折线 current_pos → 剩余点 的相邻段长之和(QA 独立累加)。"""
    pts = [np.asarray(current_pos, dtype=float)[:2]] + [
        np.asarray(w, dtype=float)[:2] for w in remaining_pts
    ]
    return float(sum(np.hypot(*(pts[k] - pts[k - 1])) for k in range(1, len(pts))))


def _remaining_pts(queue, cursor):
    return [queue[k] for k in range(cursor, len(queue))]


# ======================================================================
# d_rem —— 解析确定量,与折线段长和逐数值一致(随机多组)
# ======================================================================
def test_d_rem_matches_polyline_length_random() -> None:
    rng = np.random.default_rng(20240622)
    for _ in range(50):
        current = rng.uniform(-50, 50, size=2)
        n = int(rng.integers(1, 6))
        queue = [rng.uniform(-50, 50, size=2) for _ in range(n)]
        com = FollowerCommitment(queue=[q.copy() for q in queue], cursor=0)

        d = resolve_remaining_distance(current, com)
        ref = _ref_remaining_distance(current, queue)
        assert d == pytest.approx(ref, abs=_TOL, rel=0.0)


# ======================================================================
# t_avail —— 与 Level1 execute(同 start/waypoints) duration_s 逐数值一致
# ======================================================================
def test_t_avail_matches_level1_duration_random(rcfg: RolloutConfig) -> None:
    rng = np.random.default_rng(7)
    eng = Level1RolloutEngine(rcfg)
    for _ in range(50):
        current = rng.uniform(-40, 40, size=2)
        n = int(rng.integers(1, 6))
        queue = [rng.uniform(-40, 40, size=2) for _ in range(n)]
        com = FollowerCommitment(queue=[q.copy() for q in queue], cursor=0)

        t = resolve_available_time(current, com, rcfg)
        # start_pose.psi 任取(§v1.0.1:不含首段转入角 → psi 不影响 duration_s)
        start_pose = np.array([current[0], current[1], rng.uniform(-np.pi, np.pi)])
        res = eng.execute(start_pose, queue)
        assert t == pytest.approx(res.duration_s, abs=1e-9, rel=0.0)


def test_d_rem_and_t_avail_consistent_with_level1_jointly(rcfg: RolloutConfig) -> None:
    """同一组 (current, queue) 上 d_rem 与 t_avail 同时对齐 Level1(联合单一真相源)。"""
    rng = np.random.default_rng(99)
    eng = Level1RolloutEngine(rcfg)
    for _ in range(30):
        current = rng.uniform(-30, 30, size=2)
        n = int(rng.integers(1, 5))
        queue = [rng.uniform(-30, 30, size=2) for _ in range(n)]
        com = FollowerCommitment(queue=[q.copy() for q in queue], cursor=0)

        d = resolve_remaining_distance(current, com)
        t = resolve_available_time(current, com, rcfg)
        res = eng.execute(np.array([current[0], current[1], 0.3]), queue)

        ref_d = _ref_remaining_distance(current, queue)
        assert d == pytest.approx(ref_d, abs=_TOL)
        assert t == pytest.approx(res.duration_s, abs=1e-9)


# ======================================================================
# cursor>0 —— 只计 cursor 起剩余点;对齐 Level1(start=current, wps=剩余点)
# ======================================================================
def test_cursor_offset_only_counts_remaining(rcfg: RolloutConfig) -> None:
    eng = Level1RolloutEngine(rcfg)
    queue = [
        np.array([0.0, 0.0]),
        np.array([5.0, 0.0]),
        np.array([5.0, 5.0]),
        np.array([0.0, 5.0]),
    ]
    current = np.array([3.0, 1.0])
    for cursor in range(len(queue) + 1):
        com = FollowerCommitment(queue=[q.copy() for q in queue], cursor=cursor)
        rem = _remaining_pts(queue, cursor)

        d = resolve_remaining_distance(current, com)
        t = resolve_available_time(current, com, rcfg)
        ref_d = _ref_remaining_distance(current, rem)
        res = eng.execute(np.array([current[0], current[1], 0.0]), rem)

        assert d == pytest.approx(ref_d, abs=_TOL), f"cursor={cursor}"
        assert t == pytest.approx(res.duration_s, abs=1e-9), f"cursor={cursor}"


# ======================================================================
# 边界:空队列 / 已走完队列 → 0.0;单点 → 直线距离
# ======================================================================
def test_empty_queue_zero(rcfg: RolloutConfig) -> None:
    com = FollowerCommitment(queue=[], cursor=0)
    pos = np.array([3.0, 4.0])
    assert resolve_remaining_distance(pos, com) == 0.0
    assert resolve_available_time(pos, com, rcfg) == 0.0


def test_fully_consumed_queue_zero(rcfg: RolloutConfig) -> None:
    queue = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    com = FollowerCommitment(queue=queue, cursor=len(queue))
    pos = np.array([0.0, 0.0])
    assert resolve_remaining_distance(pos, com) == 0.0
    assert resolve_available_time(pos, com, rcfg) == 0.0


def test_single_point_distance(rcfg: RolloutConfig) -> None:
    com = FollowerCommitment(queue=[np.array([3.0, 4.0])], cursor=0)
    d = resolve_remaining_distance(np.array([0.0, 0.0]), com)
    assert d == pytest.approx(5.0, abs=_TOL)  # 3-4-5 直角三角形
    # 单点:无相邻段间转角 → t = d/cruise + dwell×1
    t = resolve_available_time(np.array([0.0, 0.0]), com, rcfg)
    exp = 5.0 / rcfg.cruise_speed_mps + rcfg.dwell_time_s
    assert t == pytest.approx(exp, abs=1e-9)


# ======================================================================
# 解析确定性:同输入逐位复现(纪律4:确定量,非随机估计)
# ======================================================================
def test_resolve_is_deterministic(rcfg: RolloutConfig) -> None:
    queue = [np.array([2.0, 3.0]), np.array([9.0, 3.0]), np.array([9.0, 11.0])]
    com = FollowerCommitment(queue=queue, cursor=0)
    pos = np.array([1.0, 1.0])
    d1 = resolve_remaining_distance(pos, com)
    d2 = resolve_remaining_distance(pos, com)
    t1 = resolve_available_time(pos, com, rcfg)
    t2 = resolve_available_time(pos, com, rcfg)
    assert d1 == d2 and t1 == t2


# ======================================================================
# is_busy 派生语义(纯数据):cursor<len ⇒ 忙
# ======================================================================
def test_is_busy_derives_from_cursor() -> None:
    q = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    assert FollowerCommitment(queue=q, cursor=0).is_busy is True
    assert FollowerCommitment(queue=q, cursor=1).is_busy is True
    assert FollowerCommitment(queue=q, cursor=2).is_busy is False
    assert FollowerCommitment(queue=[], cursor=0).is_busy is False
