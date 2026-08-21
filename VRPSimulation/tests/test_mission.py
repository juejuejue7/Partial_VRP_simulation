"""任务层验收 —— 逐条对着规格 §1–§7 验。

多数用例用 `solver="greedy"`(确定性、瞬时);ortools 只做一次冒烟,
免得 37 次元启发式求解把测试拖成分钟级。
"""
from __future__ import annotations

import numpy as np
import pytest
from msim.geometry.window import window_contains_world

from vrpsim.agents import Follower, Leader, QueueItem
from vrpsim.contracts.config import mothra_window_config
from vrpsim.contracts.mission import (MSG_BROADCAST, MSG_POLL_REP, MSG_POLL_REQ,
                                      NODE_ALL, NODE_LEADER, STATUS_DWELL,
                                      STATUS_TRANSIT, WP_NONE, WP_PROJECTION,
                                      MissionConfig)
from vrpsim.mission import (feasibility_estimate, load_result, run_mission,
                            save_result)
from vrpsim.planner import (available_pool, plan_round, project_to_window_front,
                            solve_minmax_vrp_from)
from vrpsim.world import build_mothra_world

FAST = dict(solver="greedy", vrp_time_limit_s=0.1)


@pytest.fixture(scope="module")
def mw():
    return build_mothra_world()


@pytest.fixture(scope="module")
def res(mw):
    return run_mission(MissionConfig(**FAST), mw)


# ======================================================================
# §1 场景 / §2 部署
# ======================================================================
def test_scene_is_cropped_world(res, mw):
    """§1:用 D7 裁切后的地图与 waypoint 配置。"""
    assert (mw.world.x_max_m, mw.world.y_max_m) == (500.0, 100.0)
    assert res.n_targets == 22
    assert res.meta["world"]["crop"]["dropped_waypoint_ids"] == [23, 24, 25, 26, 27, 28]


def test_deployment_positions(res):
    """§2:Leader (0,50);Follower (0,0) 与 (0,100)。"""
    cfg = res.cfg
    assert cfg.leader_start_ned == (0.0, 50.0)
    assert cfg.follower_starts_ned == ((0.0, 0.0), (0.0, 100.0))
    assert cfg.n_followers == 2
    assert np.allclose(res.follower_pos[0, 0], [0.0, 0.0])
    assert np.allclose(res.follower_pos[0, 1], [0.0, 100.0])


def test_leader_track_along_long_edge(res, mw):
    """§2:Leader 沿长边(North)走,East 恒为 50。"""
    ln = res.leader_north_m
    assert ln[0] == 0.0
    assert np.all(np.diff(ln) >= -1e-9), "Leader 只前进不后退"
    assert ln[-1] == pytest.approx(mw.world.x_max_m)


# ======================================================================
# §3 窗口:前边界与 Leader 位置重合;长宽可调
# ======================================================================
def test_window_front_edge_tracks_leader(res):
    """§3:窗口前边界恒 == Leader 位置;后沿 = 前沿 - look_back(截到 0)。"""
    front = res.window_north[:, 1]
    back = res.window_north[:, 0]
    look = res.cfg.window_look_back_m
    assert np.allclose(front, res.leader_north_m)
    assert np.allclose(back, np.maximum(0.0, front - look))


def test_window_size_is_configurable(mw):
    """§3:长宽是可调参数,改了要真的生效。"""
    r = run_mission(MissionConfig(window_look_back_m=60.0, window_width_m=40.0, **FAST), mw)
    assert r.meta["window"] == {"look_back_m": 60.0, "width_m": 40.0}
    assert np.allclose(r.window_north[:, 0],
                       np.maximum(0.0, r.window_north[:, 1] - 60.0))
    # 窗口变窄 ⇒ 池里只可能出现横向落在窗口内的点
    for a in r.assignments:
        for wid in a.wp_ids:
            j = int(np.where(mw.dataset.waypoint_id == wid)[0][0])
            assert abs(mw.dataset.targets_ned[j, 1] - 50.0) <= 20.0 + 1e-6


# ======================================================================
# §4 请求 / 规划 / 截断 / 投影
# ======================================================================
def test_usbl_cycles_are_periodic_and_poll_every_follower(res):
    """§4 / D16:USBL 每个定位循环按 id 顺序轮询全部 Follower,循环起点等间隔。"""
    cfg = res.cfg
    n_f = cfg.n_followers
    assert len(res.requests) % n_f == 0, "有定位循环没走完"
    ids = [r.follower_id for r in res.requests]
    assert ids == [i % n_f for i in range(len(ids))], "每循环按 0,1,... 顺序轮询"

    # 每个循环第 0 台的采样时刻 = 循环起点 + 1 跳
    firsts = np.asarray([r.t_s for r in res.requests[::n_f]])
    assert firsts[0] == pytest.approx(cfg.first_usbl_s + cfg.acoustic_hop_s)
    assert np.allclose(np.diff(firsts), cfg.usbl_period_s)


def test_plan_rounds_are_periodic(res):
    """§4 / D16:规划轮起点等间隔。

    Leader 走完测线后会强插一轮(不等周期),故周期性只在那之前成立。
    """
    cfg = res.cfg
    starts = np.asarray([r.t_plan_s for r in res.plan_rounds])
    forced = [i for i, r in enumerate(res.plan_rounds) if r.forced_final]
    cut = forced[0] if forced else len(starts)
    assert starts[0] == pytest.approx(cfg.first_plan_s)
    assert np.allclose(np.diff(starts[:cut]), cfg.plan_period_s)


def test_usbl_and_planning_clocks_are_independent(mw):
    """D16 的核心:拉长规划周期**不影响**定位刷新率,反之亦然。

    D16 之前两者绑死(一轮握手里先轮询再规划),拉长规划周期就等于让位置变陈旧。
    """
    base = dict(usbl_period_s=15.0, **FAST)
    slow_plan = run_mission(MissionConfig(plan_period_s=60.0, **base), mw)
    fast_plan = run_mission(MissionConfig(plan_period_s=30.0, **base), mw)
    # 规划周期翻倍 ⇒ 规划轮数大致减半,但定位循环数与陈旧度上限不变
    assert len(slow_plan.plan_rounds) < len(fast_plan.plan_rounds)
    for r in (slow_plan, fast_plan):
        assert r.fix_age_max_s <= r.cfg.usbl_period_s + 1e-9
    # 反向:拉长定位周期只影响陈旧度,不改规划轮的节拍
    stale = run_mission(MissionConfig(plan_period_s=30.0, usbl_period_s=30.0, **FAST), mw)
    assert stale.fix_age_max_s > fast_plan.fix_age_max_s
    assert np.allclose(
        np.diff([r.t_plan_s for r in stale.plan_rounds
                 if not r.forced_final][:5]), 30.0)


def test_every_message_flies_exactly_one_hop(res):
    """D15:任何一条声学报文的飞行时间都恰是 acoustic_hop_s。"""
    tau = res.cfg.acoustic_hop_s
    assert res.comm_events.shape[1] == 5
    assert np.allclose(res.comm_events[:, 1] - res.comm_events[:, 0], tau)


def test_usbl_cycle_is_2n_hops(res):
    """D16:一个 USBL 定位循环 = 逐台「询问 → 回复」= 2N 跳,不含广播。"""
    cfg = res.cfg
    usbl = res.comm_events[res.comm_events[:, 2] != MSG_BROADCAST]
    n_hops = 2 * cfg.n_followers
    assert usbl.shape[0] % n_hops == 0, "有定位循环没走完"
    assert cfg.usbl_cycle_s == pytest.approx(n_hops * cfg.acoustic_hop_s)
    for row in usbl[:, 2].reshape(-1, n_hops):
        assert list(row) == [MSG_POLL_REQ, MSG_POLL_REP] * cfg.n_followers
    # 一个循环从第一条询问发出到最后一条回复送达,恰 usbl_cycle_s
    cyc = usbl.reshape(-1, n_hops, 5)
    assert np.allclose(cyc[:, -1, 1] - cyc[:, 0, 0], cfg.usbl_cycle_s)


def test_plan_round_is_one_broadcast_hop(res):
    """D16:规划轮不再自己轮询 —— 只剩求解 + 广播 1 跳。"""
    cfg = res.cfg
    bc = res.comm_events[res.comm_events[:, 2] == MSG_BROADCAST]
    assert bc.shape[0] == len(res.plan_rounds), "每轮恰一条广播"
    # 广播的收方是"全体",不是某一台
    assert np.all(bc[:, 3] == NODE_LEADER) and np.all(bc[:, 4] == NODE_ALL)
    for r in res.plan_rounds:
        assert r.t_start_s == pytest.approx(r.t_plan_s), "规划轮没有轮询段了"
        assert r.t_deliver_s - r.t_plan_s == pytest.approx(cfg.plan_latency_s)


def test_plan_rounds_never_nest(res):
    """上一轮的序列还没落地就开下一轮 ⇒ 语义无从定义,不许发生。"""
    for a, b in zip(res.plan_rounds, res.plan_rounds[1:]):
        assert b.t_plan_s >= a.t_deliver_s - 1e-9, \
            f"轮{a.round_idx} 的序列还在水里就开了轮{b.round_idx}"


def test_overlapping_period_config_is_rejected():
    """周期比它自己的时长还短 ⇒ 物理不可能,构造配置时就该炸。"""
    with pytest.raises(ValueError, match="重叠"):
        MissionConfig(acoustic_hop_s=5.0, usbl_period_s=15.0)   # 4x5 = 20 > 15
    MissionConfig(acoustic_hop_s=3.75, usbl_period_s=15.0)      # 4x3.75 = 15,刚好允许
    with pytest.raises(ValueError, match="下发时延"):
        MissionConfig(acoustic_hop_s=2.0, plan_period_s=1.0)


def test_report_is_sampled_one_hop_before_leader_receives_it(res):
    """D15:上报的 `t_s` 是**采样时刻**,Leader 到 `t_recv_s` 才拿到。"""
    cfg = res.cfg
    tau = cfg.acoustic_hop_s
    n_f = cfg.n_followers
    for r in res.requests:
        assert r.t_recv_s == pytest.approx(r.t_s + tau)
    # 一个定位循环内 F0 先于 F1 采样,相邻两台差 2 跳
    # (前一台的回复飞一跳 + 给后一台的询问飞一跳)
    for k in range(len(res.requests) // n_f):
        cyc = res.requests[k * n_f:(k + 1) * n_f]
        t0 = cyc[0].t_s - tau                       # 本循环起点
        for i, req in enumerate(cyc):
            assert req.t_s == pytest.approx(t0 + (2 * i + 1) * tau)


def test_planner_uses_dead_reckoned_positions_not_the_stale_fix(res):
    """D16:规划用的是推算到规划时刻的位置,不是上一次 USBL 定位那个陈旧坐标。

    定位陈旧度由 `usbl_period_s` 封顶(与规划周期无关 —— 这正是解耦的意义)。
    只要那段时间里 Follower 在动,规划所用位置就必须与定位值不同。
    """
    cfg = res.cfg
    for r in res.plan_rounds:
        assert len(r.fix_age_s) == cfg.n_followers
        assert 0.0 <= r.max_fix_age_s <= cfg.usbl_period_s + 1e-9

    moved = 0
    for r in res.plan_rounds:
        # 规划时刻之前、每台最近的一次定位上报
        for i in range(cfg.n_followers):
            prior = [q for q in res.requests
                     if q.follower_id == i and q.t_recv_s <= r.t_plan_s + 1e-9]
            if not prior:
                continue
            fix = prior[-1]
            kp = int(np.argmin(np.abs(res.t_s - r.t_plan_s)))
            gap = float(np.linalg.norm(res.follower_pos[kp, i] - fix.position_ned))
            if gap > 1e-6:
                moved += 1
            # 陈旧度封顶 ⇒ 位移不可能超过这段时间全速直线跑的距离
            age = r.t_plan_s - fix.t_s
            assert gap <= age * cfg.follower_speed_mps + 1e-6
    assert moved > 0, "整条时间线里没有一台在两次定位之间动过,这条用例失去意义"


def test_request_carries_occupied_waypoint(res):
    """§4:请求里带着自己当前占用的 waypoint,且位置与当拍记录一致。

    ⚠ 不能直接比 `follower_occupied[k]`:请求发生在**下发之前**,而记录发生在
      下发之后,同一拍里 occupied 会被新序列改掉。这里比的是"请求时刻的语义":
      占用的要么是空,要么是一个当时还没被观测掉的真实 waypoint。
    """
    valid = set(int(w) for w in res.waypoint_ids)
    for r in res.requests:
        k = int(np.argmin(np.abs(res.t_s - r.t_s)))
        assert np.allclose(r.position_ned, res.follower_pos[k, r.follower_id])
        assert r.occupied_wp == WP_NONE or r.occupied_wp in valid
        if r.occupied_wp != WP_NONE:
            j = int(np.where(res.waypoint_ids == r.occupied_wp)[0][0])
            assert np.isnan(res.visit_time_s[j]) or res.visit_time_s[j] >= r.t_s


def _round_assignments(res, rnd):
    """取某一轮下发的全部 Assignment(同一轮里两台的 t_s 都等于该轮的 t_plan_s)。"""
    return [a for a in res.assignments if a.t_s == rnd.t_plan_s]


def test_occupied_waypoint_is_frozen_by_planner(res):
    """§4 + D8:**广播落地时刻**各台正在前往的那个点被冻结,不进池、不被下发。

    口径是 t_deliver 而不是规划时刻(D15):那一跳的飞行时间里队列还在推进,按落地
    时刻冻结才与 `Follower.assign()` 保留的队首对齐。故这里读**落地前一拍**的记录
    (落地那一拍已经装上新序列了)。
    """
    for rnd in res.plan_rounds:
        k = int(np.argmin(np.abs(res.t_s - rnd.t_deliver_s)))
        k_before = max(0, k - 1)
        asgs = _round_assignments(res, rnd)
        for i in range(res.cfg.n_followers):
            held = int(res.follower_occupied[k_before, i])
            if held == WP_NONE:
                continue
            for a in asgs:
                assert held not in a.wp_ids, f"轮{rnd.round_idx}: wp{held} 被冻结却仍下发"
                assert held not in a.pool_ids, f"轮{rnd.round_idx}: wp{held} 被冻结却进了池"


def test_no_waypoint_assigned_to_two_followers_in_one_round(res):
    """D15:一次联合 VRP 解出的路线天然互斥,同一轮不可能把一个点派给两台。"""
    for rnd in res.plan_rounds:
        flat = [w for a in _round_assignments(res, rnd) for w in a.wp_ids]
        assert len(flat) == len(set(flat)), f"轮{rnd.round_idx} 重复下发 {flat}"


def test_every_round_assigns_all_followers_at_once(res):
    """D15:一轮广播同时给全部 Follower —— 不再是"只给请求方"。"""
    for rnd in res.plan_rounds:
        asgs = _round_assignments(res, rnd)
        assert len(asgs) == res.cfg.n_followers
        assert sorted(a.follower_id for a in asgs) == list(range(res.cfg.n_followers))
        assert all(a.t_deliver_s == rnd.t_deliver_s for a in asgs)
        # 同一轮两台看到的是同一个池
        assert len({a.pool_ids for a in asgs}) == 1


def test_sequence_length_capped(res):
    """§4:一次至多 5 个真实 waypoint。"""
    for a in res.assignments:
        assert a.n_real <= res.cfg.max_sequence_len
        assert a.n_points <= res.cfg.max_sequence_len + 1


def test_projection_only_when_short(res):
    """§4:不足 5 个才补投影点,且投影点在**最后**。"""
    for a in res.assignments:
        assert a.has_projection == (a.n_real < res.cfg.max_sequence_len)
        if a.has_projection:
            assert a.n_points == a.n_real + 1


def test_projection_point_lies_on_window_front(res, mw):
    """§4:投影 = 把 Follower 位置抬到窗口前边界(North = Leader),East 不变。"""
    for a in res.assignments:
        if not a.has_projection:
            continue
        p = a.points_ned[-1]
        assert p[0] == pytest.approx(a.leader_north_m)
        assert 0.0 <= p[1] <= mw.world.y_max_m


def test_projection_is_never_occupied(res):
    """§4:投影点不占用 —— 记录里 occupied 永远不会是 WP_PROJECTION。"""
    assert not np.any(res.follower_occupied == WP_PROJECTION)
    # 队列里可以出现投影点,但它不进 occupied
    assert np.any(res.follower_queue == WP_PROJECTION)


def test_planner_uses_both_follower_positions(mw):
    """§4:规划要用**两台**的当前位置(否则 min-max 无从谈起)。

    把另一台挪到目标群正中,请求方的路线应当变短(活被分走了)。
    """
    ds = mw.dataset
    region = mothra_window_config()
    from msim.geometry.window import get_window
    reg = get_window(np.array([450.0, 50.0, 0.0]), mothra_window_config())
    cfg = MissionConfig(**FAST)
    occ = [WP_NONE] * ds.n
    vis = [False] * ds.n

    def _plan(peer_pos):
        asgs, wall = plan_round(
            t_plan_s=0.0, t_deliver_s=2.0,
            follower_positions=[np.array([350.0, 50.0]), np.asarray(peer_pos)],
            targets_ned=ds.targets_ned, waypoint_ids=ds.waypoint_id,
            region=reg, visited=vis, occupied=occ,
            world_east_max_m=mw.world.y_max_m, cfg=cfg)
        assert wall >= 0.0
        return asgs

    far = _plan([0.0, 0.0])
    near = _plan([420.0, 85.0])
    assert near[0].n_real < far[0].n_real, "队友近了却没分走活,说明没用上它的位置"
    # 队友那条路线现在也真的下发了(D15:不再算完就扔)
    assert near[1].n_real > far[1].n_real


def test_pool_excludes_visited_and_occupied(mw):
    from msim.geometry.window import get_window

    ds = mw.dataset
    reg = get_window(np.array([400.0, 50.0, 0.0]), mothra_window_config())
    cfg = MissionConfig(**FAST)
    vis = [False] * ds.n
    occ = [WP_NONE] * ds.n
    base = available_pool(ds.targets_ned, reg, visited=vis, occupied=occ,
                          cfg=cfg)
    assert base.size > 2

    vis[int(base[0])] = True
    occ[int(base[1])] = 1                      # 被另一台占着
    after = available_pool(ds.targets_ned, reg, visited=vis, occupied=occ,
                           cfg=cfg)
    assert int(base[0]) not in after.tolist()
    assert int(base[1]) not in after.tolist()
    assert after.size == base.size - 2


def test_pool_is_inside_window(mw):
    from msim.geometry.window import get_window

    ds = mw.dataset
    reg = get_window(np.array([300.0, 50.0, 0.0]), mothra_window_config())
    cfg = MissionConfig(**FAST)
    pool = available_pool(ds.targets_ned, reg, visited=[False] * ds.n,
                          occupied=[WP_NONE] * ds.n, cfg=cfg)
    for j in pool:
        assert window_contains_world(reg, ds.targets_ned[j])


def test_project_to_window_front_clamps_east(mw):
    from msim.geometry.window import get_window

    reg = get_window(np.array([200.0, 50.0, 0.0]), mothra_window_config(width_m=40.0))
    p = project_to_window_front(np.array([100.0, 95.0]), reg, mw.world.y_max_m)
    assert p[0] == pytest.approx(200.0)
    assert p[1] == pytest.approx(70.0), "East 要夹进窗口横向范围"


# ======================================================================
# §5 Follower 执行:逐个占用 / 到点停 5 s
# ======================================================================
def test_follower_occupies_one_at_a_time(res):
    """§5:任一时刻每台至多占用一个 waypoint,且两台不撞车。"""
    occ = res.follower_occupied
    for k in range(occ.shape[0]):
        row = [int(v) for v in occ[k] if v != WP_NONE]
        assert len(row) == len(set(row)), f"t={res.t_s[k]} 两台占用同一点 {row}"


@pytest.mark.parametrize("dwell", [5.0, 10.0])
def test_dwell_lasts_configured_time(dwell):
    """§5:到达后原地停 `dwell_time_s`。

    ⚠ 期望值从 cfg 算,不写死数字 —— 停留时长是人工裁决的旋钮(D15 把它从 5 改到 10),
      写死会让下次调参又红一次。
    """
    cfg = MissionConfig(dwell_time_s=dwell, dt_s=0.5, follower_speed_mps=1.5, **FAST)
    f = Follower(0, (0.0, 0.0), cfg)
    f.assign([QueueItem(wp_id=1, point=np.array([3.0, 0.0]), target_idx=0)])
    t = 0.0
    done = []
    while not done and t < 120.0:
        done = f.step(cfg.dt_s)
        t += cfg.dt_s
    expect = 3.0 / cfg.follower_speed_mps + cfg.dwell_time_s
    assert expect - cfg.dt_s <= t <= expect + cfg.dt_s
    assert done == [1]
    assert np.allclose(f.pos, [3.0, 0.0])


def test_default_dwell_is_the_adjudicated_ten_seconds(res):
    """D15 人工裁决:目標観測的定点停留 = 10 s(原 5 s)。

    实测:每个目标的 DWELL 连续拍数 x dt 恰等于 dwell_time_s。
    """
    cfg = res.cfg
    assert cfg.dwell_time_s == 10.0
    st = res.follower_status
    runs = []
    for i in range(cfg.n_followers):
        n = 0
        for k in range(st.shape[0]):
            if st[k, i] == STATUS_DWELL:
                n += 1
            elif n:
                runs.append(n * cfg.dt_s)
                n = 0
        if n:
            runs.append(n * cfg.dt_s)
    assert runs, "整条时间线里没有一次停留"
    assert all(abs(x - cfg.dwell_time_s) <= cfg.dt_s for x in runs), \
        f"停留时长不都等于 {cfg.dwell_time_s} s: {sorted(set(runs))}"


def test_each_target_observed_at_most_once(res):
    ids = res.waypoint_ids[res.visited_mask]
    assert len(set(ids.tolist())) == len(ids)


def test_new_sequence_overwrites_but_keeps_active():
    """D8:新序列覆盖旧序列,但**正在前往**的那个点保留在队首。"""
    cfg = MissionConfig(**FAST)
    f = Follower(0, (0.0, 0.0), cfg)
    f.assign([QueueItem(1, np.array([50.0, 0.0]), 0), QueueItem(2, np.array([60.0, 0.0]), 1)])
    f.step(1.0)                                    # 开始前往 wp1
    assert f.occupied_wp == 1
    f.assign([QueueItem(7, np.array([80.0, 0.0]), 6)])
    assert f.queue_wp_ids == [1, 7], "wp1 该保留,wp2 该被覆盖掉"


# ======================================================================
# §6 VRP 求解器
# ======================================================================
def test_assignments_come_from_a_solver(res):
    assert {a.solver for a in res.assignments} <= {"greedy", "ortools", "trivial"}
    assert any(a.solver != "trivial" for a in res.assignments)


def test_vrp_visits_every_target_exactly_once():
    starts = [np.array([0.0, 0.0]), np.array([0.0, 100.0])]
    tg = [np.array([float(10 * i), 50.0]) for i in range(8)]
    routes, _ = solve_minmax_vrp_from(starts, tg, solver="greedy")
    flat = sorted(j for r in routes for j in r)
    assert flat == list(range(8))


def test_vrp_balances_two_vehicles():
    """min-max:两簇分别贴着一台起点 ⇒ 各拿一簇,而不是一台全包。"""
    starts = [np.array([0.0, 0.0]), np.array([400.0, 100.0])]
    tg = ([np.array([5.0, 5.0 + i]) for i in range(4)]
          + [np.array([395.0, 95.0 - i]) for i in range(4)])
    routes, _ = solve_minmax_vrp_from(starts, tg, solver="greedy")
    assert len(routes[0]) == len(routes[1]) == 4


def test_greedy_is_deterministic():
    starts = [np.array([0.0, 0.0]), np.array([0.0, 100.0])]
    tg = [np.array([float(7 * i % 37), float(3 * i % 53)]) for i in range(12)]
    a, _ = solve_minmax_vrp_from(starts, tg, solver="greedy")
    b, _ = solve_minmax_vrp_from(starts, tg, solver="greedy")
    assert a == b


@pytest.mark.parametrize("solver", ["greedy", "ortools"])
def test_mission_runs_with_either_solver(mw, solver):
    r = run_mission(MissionConfig(solver=solver, vrp_time_limit_s=0.1), mw)
    assert r.coverage > 0.5


# ======================================================================
# §7 记录
# ======================================================================
def test_timeline_shapes_are_consistent(res):
    T = len(res.t_s)
    F = res.cfg.n_followers
    K = res.cfg.max_sequence_len + 1
    assert res.leader_north_m.shape == (T,)
    assert res.window_north.shape == (T, 2)
    assert res.follower_pos.shape == (T, F, 2)          # ← 各 Follower 位置
    assert res.follower_queue.shape == (T, F, K)        # ← waypoint 序列
    assert res.follower_queue_xy.shape == (T, F, K, 2)
    assert res.follower_status.shape == (T, F)
    assert res.follower_occupied.shape == (T, F)
    assert np.all(np.diff(res.t_s) > 0)


def test_queue_record_matches_assignments(res):
    """**广播落地**那一刻的队列记录里,必须能看到刚发出去的那些 waypoint。

    ⚠ 断言点是 `t_deliver_s` 而不是规划时刻 `t_s`(D15):序列在水里还要飞一跳,
      规划那一拍 Follower 手里还是旧队列。
    """
    for a in res.assignments:
        if a.n_real == 0:
            continue
        k = int(np.argmin(np.abs(res.t_s - a.t_deliver_s)))
        # 落地在记录之前发生(同一拍),故当拍队列应含全部新点
        q = [int(v) for v in res.follower_queue[k, a.follower_id]]
        for wid in a.wp_ids:
            assert wid in q, f"落地 t={a.t_deliver_s} 的 wp{wid} 没出现在当拍队列记录里"

def test_nothing_is_assigned_before_the_first_broadcast_lands(res):
    """第一条序列要飞满一整轮握手才落地 —— 在那之前谁的队列都必须是空的。

    这是"序列在水里"最不含歧义的判据:队首在别处也可能因为 Follower 自己走完上一个
    点而变,但**任务开局**队列非空只可能来自下发。
    """
    cfg = res.cfg
    t_first = cfg.first_plan_s + cfg.plan_latency_s
    assert res.plan_rounds[0].t_deliver_s == pytest.approx(t_first)
    before = res.t_s < t_first - 1e-9
    assert before.any(), "第一轮瞬间落地,这条用例失去意义"
    assert np.all(res.follower_queue[before] == WP_NONE), \
        f"t < {t_first} s 就有 Follower 拿到了序列,声学延迟被短路"
    k = int(np.argmin(np.abs(res.t_s - t_first)))
    assert np.any(res.follower_queue[k] != WP_NONE), "落地那一拍没装上"


def test_shadow_and_truth_get_the_sequence_on_the_same_tick(mw):
    """影子与真机必须在**同一拍**装上同一条序列 —— D15 的对齐点。

    D15 之前的实现在规划时刻就更新影子、真机却要等延迟,影子会提前一跳沿新序列行进。
    把单跳拉大到 10 s 放大这个效应:一旦两边不同拍,推算误差会涨到米级,
    "承诺队列解析计算是确定量"这条结论随即失效(CLAUDE.md 硬纪律 4)。
    """
    r = run_mission(MissionConfig(acoustic_hop_s=10.0, usbl_period_s=60.0,
                                  plan_period_s=60.0, **FAST), mw)
    assert r.cfg.plan_latency_s == 10.0 and r.cfg.usbl_cycle_s == 40.0
    assert float(np.abs(r.shadow_error[:, 2]).max()) < 1e-9, \
        "单跳拉大后推算误差冒头 ⇒ 影子与真机不在同一拍收到序列"


def test_visit_bookkeeping_consistent(res):
    assert res.visited_mask.sum() == res.visited_count[-1]
    for j, ok in enumerate(res.visited_mask):
        assert (res.visit_by[j] >= 0) == ok
        if ok:
            assert 0 < res.visit_time_s[j] <= res.t_s[-1]


def test_save_load_roundtrip(res, tmp_path):
    p = str(tmp_path / "m.npz")
    save_result(res, p)
    back = load_result(p)
    assert np.array_equal(back["follower_pos"], res.follower_pos)
    assert np.array_equal(back["follower_queue"], res.follower_queue)
    assert back["meta"]["n_followers"] == res.cfg.n_followers
    import json
    import os
    with open(os.path.splitext(p)[0] + "_assignments.json", encoding="utf-8") as fh:
        j = json.load(fh)
    assert len(j["assignments"]) == len(res.assignments)
    assert j["summary"]["n_targets"] == res.n_targets


# ======================================================================
# Leader 侧的队友认知(信息边界)—— 见 vrpsim/tracking.py
# ======================================================================
def test_shadow_matches_truth_without_drift(res):
    """漂移关掉时,承诺队列解析推算与真值**逐位相同**。

    这条是术语纪律的凭据(CLAUDE.md 硬纪律 4):此时它是"由承诺队列解析计算的确定量",
    论文里不许写成"估计"。一旦有人让影子拿不到该拿的信息(或运动学写岔),这里会红。
    """
    assert res.cfg.nav_drift_frac_of_distance == 0.0
    # 每轮规划记一行 x 每台(D16:定位与规划解耦后,误差在**规划时刻**取样)
    assert res.shadow_error.shape[0] == res.cfg.n_followers * len(res.plan_rounds)
    assert float(np.abs(res.shadow_error[:, 2]).max()) < 1e-9


def test_drift_knob_actually_degrades_knowledge(mw):
    """开了漂移旋钮才真的变成"估计":误差随漂移率单调增大。"""
    errs = []
    for rate in (0.0, 0.005, 0.02):
        r = run_mission(MissionConfig(nav_drift_frac_of_distance=rate, seed=7, **FAST), mw)
        errs.append(r.summary()["shadow_error_max_m"])
    assert errs[0] < 1e-9                       # 浮点残差,非真实误差
    assert errs[0] < errs[1] < errs[2]


def test_drift_direction_depends_on_seed(mw):
    """漂移**量级**由行进距离定(与 seed 无关),**方向**由 seed 定且可复现。

    所以别拿误差范数去测 seed 敏感性 —— 那个量按构造就是 seed 无关的。
    """
    from vrpsim.tracking import FollowerShadow

    cfg = MissionConfig(nav_drift_frac_of_distance=0.02, **FAST)

    def drift_vec(seed):
        sh = FollowerShadow(0, (0.0, 0.0), cfg, np.random.default_rng(seed))
        sh.on_assignment(_fake_assignment(mw), {int(w): j for j, w in
                                                enumerate(mw.dataset.waypoint_id)})
        sh.advance_to(60.0)
        return sh.position - sh.true_model_position

    a, b, c = drift_vec(3), drift_vec(3), drift_vec(99)
    assert np.allclose(a, b), "同 seed 必须完全复现"
    assert not np.allclose(a, c), "不同 seed 的漂移方向应当不同"
    assert np.isclose(np.linalg.norm(a), np.linalg.norm(c)), "量级只由行进距离定"


def _fake_assignment(mw):
    """构造一条最简单的下发序列,供影子单测驱动。"""
    from vrpsim.contracts.mission import Assignment

    return Assignment(t_s=0.0, follower_id=0, wp_ids=(int(mw.dataset.waypoint_id[0]),),
                      points_ned=mw.dataset.targets_ned[:1].copy(),
                      has_projection=False, pool_ids=(), solver="test",
                      leader_north_m=100.0)


def test_planner_inputs_come_from_shadow_not_truth(mw, monkeypatch):
    """信息边界最硬的判据:开漂移后,喂进规划器的**队友**位置必须偏离真值。

    若规划器还在偷看仿真真值,这个偏差会恒为 0。
    (不能用"篡改真值再比结果"的做法 —— 那会连带改掉真实动力学,测的就不是信息流了。)
    """
    import vrpsim.mission as M

    seen = []
    orig = M.plan_round

    def spy(**kw):
        seen.append((kw["t_plan_s"],
                     [np.asarray(p).copy() for p in kw["follower_positions"]]))
        return orig(**kw)

    monkeypatch.setattr(M, "plan_round", spy)
    res = run_mission(MissionConfig(nav_drift_frac_of_distance=0.02, seed=5, **FAST), mw)

    assert seen, "规划器没被调用"
    n_f = res.cfg.n_followers
    # shadow_error 每轮 n_f 行,与 seen 每轮 1 条一一对应
    assert res.shadow_error.shape[0] == n_f * len(seen)
    offs = []
    for r, (t, pos_in) in enumerate(seen):
        k = int(np.argmin(np.abs(res.t_s - t)))
        for i in range(n_f):
            d = float(np.linalg.norm(pos_in[i] - res.follower_pos[k, i]))
            offs.append(d)
            assert np.isclose(d, res.shadow_error[r * n_f + i, 2], atol=1e-9), \
                "记录的推算误差与实际喂进规划器的位置对不上"

    # 每台的上报都要飞一跳才到 Leader,规划时刻它们全部是推算值 ⇒ 都得偏离真值。
    # 若规划器还在偷看仿真真值,这个偏差会恒为 0。
    assert max(offs) > 0.05, f"喂进规划器的位置与真值零偏差 {max(offs)} —— 疑似仍读真值"


# ======================================================================
# D15:Leader 走完测线后强插一轮 + 池空才终止
# ======================================================================
def test_final_round_fires_right_after_leader_finishes(res):
    """Leader 走完 ⇒ 窗口冻结,必须立刻补一轮,不等周期。"""
    forced = [r for r in res.plan_rounds if r.forced_final]
    assert len(forced) == 1, f"强插末轮应当恰好一次,实际 {len(forced)}"
    f = forced[0]
    assert f.t_start_s >= res.t_leader_finish_s - 1e-9
    # 与前一轮的间隔严格小于周期 —— 证明它是被强插的,不是等到了周期
    prev = [r for r in res.plan_rounds if r.round_idx == f.round_idx - 1][0]
    assert f.t_start_s - prev.t_start_s < res.cfg.plan_period_s - 1e-9
    # 强插只此一次;之后恢复周期节拍
    later = [r for r in res.plan_rounds if r.round_idx > f.round_idx]
    assert all(not r.forced_final for r in later)


def test_never_terminates_with_assignable_targets_left(res, mw):
    """D15 终止条件:窗口里还有未观测的点就不许退出。"""
    from msim.geometry.window import get_window

    reg = get_window(np.array([res.leader_north_m[-1], 50.0, 0.0]),
                     mothra_window_config(look_back_m=res.cfg.window_look_back_m,
                                          width_m=res.cfg.window_width_m))
    left = available_pool(mw.dataset.targets_ned, reg,
                          visited=list(res.visited_mask),
                          occupied=[WP_NONE] * mw.dataset.n, cfg=res.cfg)
    assert left.size == 0, f"末拍窗口里还剩 {left.size} 个可规划点却退出了"
    # 末拍队列里不许还有**真实** waypoint 没走完。
    # (投影点可以剩:它不是观测目标,且记录发生在推进之前,末尾那一步会把它走掉。)
    assert not np.any(res.follower_queue[-1] >= 0), "末拍还有真实 waypoint 没走完"


# ======================================================================
# D15:求解耗时与问题规模的记录
# ======================================================================
def test_solve_wall_time_and_pool_size_are_recorded(res):
    """每轮都记下**实测**求解耗时与该轮的问题规模。"""
    assert len(res.plan_rounds) > 0
    assert res.plan_stats.shape == (len(res.plan_rounds), 9)
    for r in res.plan_rounds:
        assert r.solve_wall_s >= 0.0
        assert r.pool_size >= 0
        assert len(r.n_assigned) == res.cfg.n_followers
        assert r.n_assigned_total == sum(r.n_assigned) <= r.pool_size
        assert 0 <= r.n_projection <= res.cfg.n_followers
    s = res.summary()
    assert s["n_plan_rounds"] == len(res.plan_rounds)
    assert s["solve_wall_total_s"] == pytest.approx(float(res.solve_wall_s.sum()))
    assert s["pool_size_mean"] == pytest.approx(float(res.pool_sizes.mean()))
    # 池非空的轮里求解确实跑了
    assert any(r.solve_wall_s > 0.0 for r in res.plan_rounds if r.pool_size)


def test_solve_wall_time_stays_out_of_the_numeric_path(res):
    """挂钟耗时是诊断量,绝不许渗进仿真时间轴。

    仿真侧给求解留的余量是 `plan_solve_s`(默认 0);若挂钟量被误用,
    握手时长就会变成 5τ + 一个随机数,时序断言随即失效。
    """
    cfg = res.cfg
    assert cfg.plan_solve_s == 0.0
    for r in res.plan_rounds:
        assert r.t_deliver_s - r.t_plan_s == pytest.approx(cfg.plan_latency_s)
    # 时间轴是 dt 的整数倍,不含任何挂钟成分
    assert np.allclose(np.diff(res.t_s), cfg.dt_s)
    wall = res.solve_wall_s
    assert wall.size and float(wall.max()) > 0.0, "求解耗时全为 0,这条用例失去意义"
    for v in (float(res.t_complete_s), float(res.t_leader_finish_s)):
        assert abs(v / cfg.dt_s - round(v / cfg.dt_s)) < 1e-9


def test_tracker_occupancy_comes_from_shadow(mw):
    """占用表由影子推出;没派活时无人占用。"""
    from vrpsim.tracking import LeaderTracker

    cfg = MissionConfig(**FAST)
    tr = LeaderTracker(cfg)
    assert len(tr.shadows) == cfg.n_followers
    occ = tr.occupancy(mw.dataset.n)
    assert occ == [WP_NONE] * mw.dataset.n, "还没派任何活时不该有人占用"
    assert tr.visited_mask(mw.dataset.n, {int(w): j for j, w in
                                          enumerate(mw.dataset.waypoint_id)}) == \
        [False] * mw.dataset.n


def test_request_position_is_the_message_not_truth(res):
    """请求方的位置来自请求消息本身,且与该时刻真值一致(它确实合法可知)。"""
    for r in res.requests:
        k = int(np.argmin(np.abs(res.t_s - r.t_s)))
        assert np.allclose(r.position_ned, res.follower_pos[k, r.follower_id])


# ======================================================================
# 可行性判据
# ======================================================================
def test_feasibility_says_when_the_leader_must_wait(mw):
    """D16:判据从"能不能全覆盖"变成"必须停船等吗 + 时间下界多少"。

    默认 0.5/0.5 下 `t_route_lb`(1084 s)> `t_leader`(1000 s) ⇒ Leader 必然要等;
    等待策略开着时这不再是死刑,而是把覆盖率约束换成了时间约束。
    """
    fe = feasibility_estimate(mw, MissionConfig(**FAST))
    assert fe["wait_required"], "0.5/0.5 下 Leader 本就该等,判据没报出来"
    assert fe["leader_waits"] and fe["feasible"], "等待策略开着就不该判不可行"
    assert fe["t_mission_reference_s"] == pytest.approx(
        max(fe["t_leader_finish_s"], fe["t_route_reference_s"]))

    # 关掉等待策略 ⇒ 速度比重新变成硬约束,同一套参数就判不可行
    off = feasibility_estimate(mw, MissionConfig(
        leader_wait_on_lagging_follower=False,
        leader_wait_on_endangered_target=False, **FAST))
    assert not off["leader_waits"] and not off["feasible"]
    # 匀速 Leader 想全覆盖就得慢到这个速度以下
    assert off["max_leader_speed_mps"] < MissionConfig().leader_speed_mps

    # Leader 够慢时根本不必等
    ok = feasibility_estimate(mw, MissionConfig(leader_speed_mps=0.4, **FAST))
    assert not ok["wait_required"] and ok["feasible"]


def test_leader_waiting_is_what_makes_full_coverage_possible(mw):
    """D16 的核心论据:关掉等待策略,同一套参数就掉覆盖。

    判据不是纸上谈兵 —— 0.5/0.5 下匀速 Leader 会先走完、窗口冻结,南边的点永久错过。
    """
    off = run_mission(MissionConfig(leader_wait_on_lagging_follower=False,
                                    leader_wait_on_endangered_target=False,
                                    **FAST), mw)
    assert off.coverage < 1.0, "关掉等待却仍然全覆盖 ⇒ 这个机制没在起作用"
    assert np.isnan(off.t_complete_s)
    assert off.leader_hold_total_s == 0.0
    # 打开等待 ⇒ 同一套参数拿回全覆盖
    on = run_mission(MissionConfig(**FAST), mw)
    assert on.coverage == 1.0
    assert on.leader_hold_total_s > 0.0


@pytest.mark.parametrize("kw,flag", [
    (dict(leader_wait_on_endangered_target=False), "leader_hold_lagging_s"),
    (dict(leader_wait_on_lagging_follower=False), "leader_hold_endangered_s"),
])
def test_each_wait_criterion_works_on_its_own(mw, kw, flag):
    """两条判据各自单独打开都够用,且停船时长只记在自己那一栏(可做消融)。"""
    r = run_mission(MissionConfig(**kw, **FAST), mw)
    s = r.summary()
    assert r.coverage == 1.0, f"只开一条判据就掉覆盖: {kw}"
    assert s[flag] > 0.0, "开着的那条判据从没触发过"
    other = ("leader_hold_endangered_s" if flag == "leader_hold_lagging_s"
             else "leader_hold_lagging_s")
    assert s[other] == 0.0, "关掉的判据居然记了停船时长"


def test_hold_decision_cannot_see_the_truth(mw):
    """停船判据的信息边界:它只许吃影子模型与目标坐标,拿不到仿真真值。

    用签名钉死 —— 哪天有人给 `_hold_reason` 加一个 `followers` 形参,Leader 就成了
    全知全能的,整套"Leader 只知道自己推算出来的东西"就不成立了(D11 / D16)。
    """
    import inspect

    import vrpsim.mission as M

    params = set(inspect.signature(M._hold_reason).parameters)
    assert "followers" not in params and "truth" not in params, \
        f"停船判据的入参出现了真值通道: {sorted(params)}"
    assert {"tracker", "region"} <= params

    # 行为上也验一遍:开了漂移(Leader 的认知带误差)后停船时长会变 ——
    # 若判据偷看真值,漂移对它就毫无影响。
    base = run_mission(MissionConfig(nav_drift_frac_of_distance=0.0, **FAST), mw)
    drift = run_mission(MissionConfig(nav_drift_frac_of_distance=0.05, seed=3,
                                      **FAST), mw)
    assert base.leader_hold_total_s != drift.leader_hold_total_s, \
        "认知误差完全不影响停船决策 —— 疑似判据在读真值"


def test_leader_hold_bookkeeping_is_consistent(res):
    """停船时长 = 记录里 holding 的拍数 x dt;且停船时 Leader 真的没动。"""
    dt = res.cfg.dt_s
    assert res.leader_holding.shape == res.t_s.shape
    assert res.leader_hold_total_s == pytest.approx(float(res.leader_holding.sum()) * dt)
    # 标着停船的那些拍,North 不许增加
    moved = np.diff(res.leader_north_m) > 1e-12
    assert not np.any(moved & res.leader_holding[:-1]), "标着停船却仍在前进"
    # 完成时刻 = 匀速用时 + 累计停船
    want = float(res.leader_north_m.max()) / res.cfg.leader_speed_mps \
        + res.leader_hold_total_s
    assert abs(res.t_leader_finish_s - want) <= dt + 1e-6
    # 停船原因位标志与 holding 布尔一致
    assert np.array_equal(res.leader_holding, res.leader_hold_reason != 0)


def test_default_config_achieves_full_coverage(res):
    """默认参数必须能把 22 个目标全观测到 —— 否则默认值就不该是这个。"""
    assert res.coverage == 1.0, f"漏了 {res.missed_wp_ids}"
