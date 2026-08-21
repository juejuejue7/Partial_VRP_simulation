"""窗口验收(D8):滑动窗口的几何、推进、覆盖完整性。

⚠ 本文件只测**几何**。"哪些点已被 occupied / 怎么重解 / 新序列怎么覆盖旧序列"
  属任务调度层,下一步实现,不在这里测。
"""
from __future__ import annotations

import numpy as np
import pytest

from msim.contracts.config import WindowConfig
from msim.geometry.window import window_contains_world

from vrpsim.contracts.config import MothraSimConfig, mothra_window_config
from vrpsim.windows import (enumerate_windows, first_seen_window, leader_track,
                            targets_in_window, window_anchors, window_occupancy)
from vrpsim.world import build_mothra_world


@pytest.fixture(scope="module")
def mw():
    return build_mothra_world(MothraSimConfig())


@pytest.fixture(scope="module")
def cfg():
    return MothraSimConfig()


# ======================================================================
# 窗口几何
# ======================================================================
def test_window_is_100x100(cfg):
    w = cfg.window
    assert isinstance(w, WindowConfig)
    assert (w.look_back_m, w.width_m) == (100.0, 100.0)


def test_single_leader_track(mw, cfg):
    """裁切后 East 跨度 == 窗口宽 ⇒ 单条测线,总航程 = North 跨度。"""
    track = leader_track(mw.world, cfg.lane_spacing_m)
    assert len(track) == 2
    total = float(np.linalg.norm(track[1] - track[0]))
    assert total == pytest.approx(mw.world.x_max_m)
    assert track[0][1] == track[1][1] == mw.world.y_max_m / 2.0


def test_window_is_behind_leader(mw, cfg):
    """窗口恒在 Leader 后方:窗口内任意点的 North <= Leader 的 North。"""
    for s in enumerate_windows(mw.world, cfg.window, mw.dataset.targets_ned):
        a, b = s.north_span
        assert b == pytest.approx(s.leader_north_m)
        assert a >= 0.0 and a <= b
        for i in s.target_idx:
            assert mw.dataset.targets_ned[i, 0] <= s.leader_north_m + 1e-6


def test_window_advances_monotonically(mw, cfg):
    snaps = enumerate_windows(mw.world, cfg.window, mw.dataset.targets_ned)
    north = [s.leader_north_m for s in snaps]
    assert all(b > a for a, b in zip(north[:-1], north[1:]))
    assert north[-1] == pytest.approx(mw.world.x_max_m), "最后一次重解必须走到北界"


def test_anchors_cover_track_end(mw):
    """推进阈值不整除时,末端要补一个触发点,否则最后一段永远不被规划。"""
    a = window_anchors(mw.world, mothra_window_config(advance_threshold_m=70.0))
    assert a[-1] == pytest.approx(mw.world.x_max_m)
    assert np.all(np.diff(a) > 0)
    assert np.all(np.diff(a) <= 70.0 + 1e-9)


def test_anchor_step_rejects_nonpositive(mw):
    with pytest.raises(ValueError):
        window_anchors(mw.world, mothra_window_config(advance_threshold_m=0.0))


def test_no_degenerate_start_window(mw, cfg):
    """默认不含 North=0 的零面积退化窗口。"""
    snaps = enumerate_windows(mw.world, cfg.window, mw.dataset.targets_ned)
    assert all(s.north_span[1] > s.north_span[0] for s in snaps)
    with_start = enumerate_windows(mw.world, cfg.window, mw.dataset.targets_ned,
                                   include_start=True)
    assert len(with_start) == len(snaps) + 1


# ======================================================================
# 覆盖完整性 —— 这是选窗口尺寸/推进阈值的硬判据
# ======================================================================
def test_every_target_enters_a_window(mw, cfg):
    """任何目标都必须至少进过一次窗口,否则它在几何上就不可能被观测。"""
    snaps = enumerate_windows(mw.world, cfg.window, mw.dataset.targets_ned)
    first = first_seen_window(snaps, mw.dataset.n)
    never = mw.dataset.waypoint_id[first < 0].tolist()
    assert not never, f"这些目标从未进入窗口:{never}"


def test_half_window_advance_gives_two_chances(mw):
    """推进阈值 = 半窗 ⇒ 每个目标恰好在 2 次连续重解里可见。

    这是"能不能救回来"的定量判据:目标只有 2 次被分配的机会,错过即永久错过
    (project_summary §7)。若这里从 2 掉到 1,说明推进阈值被调得过大。
    """
    snaps = enumerate_windows(mw.world, mothra_window_config(advance_threshold_m=50.0),
                              mw.dataset.targets_ned)
    vis = np.zeros((mw.dataset.n, len(snaps)), dtype=bool)
    for s in snaps:
        vis[s.target_idx, s.index] = True
    seen = vis.sum(axis=1)
    assert seen.min() == 2 and seen.max() == 2
    # 且必须是**连续**的两次(窗口滑动,不会一进一出再进)
    for row in vis:
        idx = np.where(row)[0]
        assert idx[-1] - idx[0] == len(idx) - 1


def test_smaller_advance_gives_more_chances(mw):
    """推进阈值越小 → 重解越密 → 每个目标的可见次数越多(单调性)。"""
    counts = []
    for step in (100.0, 50.0, 25.0):
        snaps = enumerate_windows(mw.world, mothra_window_config(advance_threshold_m=step),
                                  mw.dataset.targets_ned)
        vis = np.zeros((mw.dataset.n, len(snaps)), dtype=bool)
        for s in snaps:
            vis[s.target_idx, s.index] = True
        counts.append(int(vis.sum(axis=1).min()))
    assert counts[0] < counts[1] < counts[2]


def test_occupancy_matches_manual_count(mw, cfg):
    """窗口占用数必须与逐点隶属判定一致(不许有第二套口径)。"""
    snaps = enumerate_windows(mw.world, cfg.window, mw.dataset.targets_ned)
    occ = window_occupancy(snaps)
    for s, o in zip(snaps, occ):
        manual = sum(1 for p in mw.dataset.targets_ned
                     if window_contains_world(s.region, p))
        assert o == manual == len(s.target_idx)
    assert occ.max() == 10, "最密的一次重解应有 10 个目标(Crab Basin + Faulty Towers)"


def test_targets_in_window_empty_ok(mw, cfg):
    snaps = enumerate_windows(mw.world, cfg.window)
    assert all(len(s.target_idx) == 0 for s in snaps)
    assert targets_in_window(snaps[0].region, np.zeros((0, 2))).size == 0


def test_faulty_towers_is_split_by_geometry(mw, cfg):
    """D9 的既成事实:严格规则几何会把 Faulty Towers 复合体切到不同重解里。

    这不是 bug,是"测线与概率场解耦"的代价,已人工裁决接受。钉在这里是为了
    将来有人"顺手优化"窗口边界时,能看到它改变了已裁决的行为。
    """
    names = mw.dataset.name
    ft = np.where(np.char.startswith(names.astype(str), "Faulty Towers"))[0]
    assert ft.size >= 2

    snaps = enumerate_windows(mw.world, cfg.window, mw.dataset.targets_ned)
    first = first_seen_window(snaps, mw.dataset.n)
    assert len(set(first[ft].tolist())) > 1, "该复合体应被窗口边界切开(D9 已接受)"
