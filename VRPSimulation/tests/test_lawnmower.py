"""Lawnmower 全覆盖基线(D14)验收。

这个基线的唯一用途是给协同方案当**对照组**,所以测的重点有两条:
  1. 几何真的做到了全覆盖(不是"跑完发现刚好都扫到了"),且每个目标只観測一次;
  2. 与 VRP 场景**共用同一份参数**这件事是结构保证的 —— 有人往
     `LawnmowerConfig` 里另加一个速度字段,这里必须红。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from vrpsim.contracts.lawnmower import (SWATH_FROM_MSIM_FOOTPRINT_M, LawnmowerConfig)
from vrpsim.contracts.mission import STATUS_IDLE, WP_PROJECTION, MissionConfig
from vrpsim.lawnmower import (build_vehicle_plans, lane_easts, load_lawn_result,
                              run_lawnmower, save_lawn_result, strip_bounds)
from vrpsim.mission import run_mission
from vrpsim.world import build_mothra_world

FAST_VRP = dict(solver="greedy", vrp_time_limit_s=0.1)


@pytest.fixture(scope="module")
def mw():
    return build_mothra_world()


@pytest.fixture(scope="module")
def cfg():
    return LawnmowerConfig(base=MissionConfig(max_mission_time_s=20000.0))


@pytest.fixture(scope="module")
def res(mw, cfg):
    return run_lawnmower(cfg, mw)


# ======================================================================
# 参数同源:与 VRP 场景"配置相同"必须是结构保证,不是人工同步
# ======================================================================
def test_kinematics_come_from_the_same_mission_config():
    """运动学全部从内嵌的 `base` 派生;`vehicle_cfg` 只翻 dwell_at_projection。"""
    base = MissionConfig(follower_speed_mps=2.25, dwell_time_s=9.0,
                         arrive_radius_m=0.0, dt_s=0.25)
    c = LawnmowerConfig(base=base)
    assert c.follower_speed_mps == 2.25 and c.dwell_time_s == 9.0 and c.dt_s == 0.25
    v = c.vehicle_cfg
    for f in ("follower_speed_mps", "dwell_time_s", "arrive_radius_m", "dt_s", "sim"):
        assert getattr(v, f) == getattr(base, f), f"{f} 与 VRP 场景不同源"
    assert v.dwell_at_projection is False, "测线拐点不该停留"
    assert base.dwell_at_projection is True, "不许反向改动 VRP 场景那一份"


def test_no_second_source_of_truth_for_shared_params():
    """`LawnmowerConfig` 自己不许再声明速度/停留/时钟 —— 那会变成第二个真相源。"""
    from dataclasses import fields

    own = {f.name for f in fields(LawnmowerConfig)}
    forbidden = {"follower_speed_mps", "dwell_time_s", "dt_s", "arrive_radius_m",
                 "sim", "max_mission_time_s"}
    assert own & forbidden == set(), f"这些字段应从 base 派生: {own & forbidden}"


def test_swath_default_tracks_msim_sensor_config():
    """默认幅宽 = msim `SensorConfig.footprint_lateral_m`,不是自造的数(【待裁决 L1】)。"""
    from msim.contracts.config import SensorConfig

    assert SWATH_FROM_MSIM_FOOTPRINT_M == SensorConfig().footprint_lateral_m
    assert LawnmowerConfig().swath_width_m == SensorConfig().footprint_lateral_m


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        LawnmowerConfig(n_vehicles=0)
    with pytest.raises(ValueError):
        LawnmowerConfig(swath_width_m=0.0)


# ======================================================================
# 测线几何
# ======================================================================
def test_strips_partition_east_exactly():
    b = strip_bounds(100.0, 3)
    assert len(b) == 3
    assert b[0][0] == 0.0 and b[-1][1] == pytest.approx(100.0)
    for a, c in zip(b, b[1:]):
        assert a[1] == pytest.approx(c[0]), "条带之间不许有缝或重叠"


@pytest.mark.parametrize("width,swath,n_want", [(33.0, 6.0, 6), (36.0, 6.0, 6),
                                                (30.0, 6.0, 5), (5.0, 6.0, 1)])
def test_lane_count_and_spacing(width, swath, n_want):
    """条数 = ceil(带宽/幅宽);间距 ≤ 幅宽 ⇒ 带内无缝隙。"""
    lanes = lane_easts(10.0, 10.0 + width, swath)
    assert lanes.size == n_want
    if lanes.size > 1:
        sp = float(lanes[1] - lanes[0])
        assert sp <= swath + 1e-9
        assert np.allclose(np.diff(lanes), sp), "测线必须等距"
    assert lanes[0] >= 10.0 and lanes[-1] <= 10.0 + width, "测线不许跑到带外"


def test_lanes_cover_the_whole_strip(mw, cfg):
    """每条带内的任意 East 都落在某条测线的幅宽里 —— 全覆盖是几何保证的。"""
    half = 0.5 * cfg.swath_width_m
    for lo, hi in strip_bounds(mw.world.y_max_m, cfg.n_vehicles):
        lanes = lane_easts(lo, hi, cfg.swath_width_m)
        probe = np.linspace(lo, hi - 1e-9, 400)
        d = np.abs(probe[:, None] - lanes[None, :]).min(axis=1)
        assert d.max() <= half + 1e-9, f"条带 [{lo},{hi}) 有缝隙"


def test_every_target_scheduled_exactly_once(mw, cfg):
    queues, lanes, starts, owner = build_vehicle_plans(mw, cfg)
    real = [it.wp_id for q in queues for it in q if not it.is_projection]
    assert sorted(real) == sorted(int(w) for w in mw.dataset.waypoint_id)
    assert len(real) == len(set(real)), "同一目标被排了两次停留(相邻测线幅宽搭接)"


def test_dwell_point_stays_on_the_lane(mw, cfg):
    """停留点在测线上、North 对齐目标 —— 车不离开测线去目标正上方。"""
    queues, lanes, _, owner = build_vehicle_plans(mw, cfg)
    ds = mw.dataset
    id_to_idx = {int(w): j for j, w in enumerate(ds.waypoint_id)}
    for v, q in enumerate(queues):
        lane_set = set(np.round(lanes[v], 9).tolist())
        for it in q:
            if it.is_projection:
                continue
            j = id_to_idx[it.wp_id]
            assert it.point[0] == pytest.approx(ds.targets_ned[j, 0])
            assert round(float(it.point[1]), 9) in lane_set
            assert abs(float(it.point[1]) - ds.targets_ned[j, 1]) <= \
                0.5 * cfg.swath_width_m + 1e-9, "目标不在该测线幅宽内"


def test_infeasible_swath_is_loud(mw):
    """幅宽小到覆盖不住带内目标时必须当场炸,而不是静默漏観測。

    (`lane_easts` 会按幅宽加密测线,所以这里直接绕过它,构造一个不自洽的几何。)
    """
    from vrpsim import lawnmower as L

    cfg = LawnmowerConfig(swath_width_m=6.0)
    orig = L.lane_easts
    try:
        L.lane_easts = lambda lo, hi, sw: np.array([0.5 * (lo + hi)])   # 每带只 1 条
        with pytest.raises(ValueError, match="没被任何测线幅宽覆盖"):
            L.build_vehicle_plans(mw, cfg)
    finally:
        L.lane_easts = orig


# ======================================================================
# 仿真结果
# ======================================================================
def test_full_coverage_and_no_repeat(res):
    assert res.coverage == 1.0, f"全覆盖扫描却漏了 {res.missed_wp_ids}"
    ids = res.waypoint_ids[res.visited_mask].tolist()
    assert len(set(ids)) == len(ids)
    assert set(res.visit_by.tolist()) <= set(range(res.cfg.n_vehicles))


def test_distance_matches_lane_geometry(res):
    """航程 = 测线长 x 条数 + 换线横移 x (条数-1);停留不产生位移。"""
    L = res.meta["survey_line_length_m"]
    sp = res.meta["lane_spacing_m"]
    for i, lanes in enumerate(res.meta["lane_easts"]):
        n = len(lanes)
        assert float(res.per_vehicle_distance_m[i]) == \
            pytest.approx(n * L + (n - 1) * sp, abs=1e-6)


def test_vehicles_never_leave_their_lanes(res):
    """East 只能取测线值或换线横移途中的中间值,且始终在场地内。"""
    e = res.vehicle_pos[:, :, 1]
    n = res.vehicle_pos[:, :, 0]
    assert e.min() >= 0.0 and e.max() <= 100.0
    assert n.min() >= -1e-9 and n.max() <= 500.0 + 1e-9
    for i, lanes in enumerate(res.meta["lane_easts"]):
        assert float(np.abs(e[:, i] - lanes[0]).min()) < 1e-9
        assert e[0, i] == pytest.approx(lanes[0]), "起点应在自己第一条测线上"
        assert n[0, i] == pytest.approx(0.0)


def test_completion_time_is_end_of_survey_not_last_observation(res):
    """完成时刻 = 走完全部测线,**晚于**最后一个目标被観測的时刻。

    这条是口径本身:全覆盖扫描不走完就不能宣称扫完。谁把 `t_complete_s` 改成
    `nanmax(visit_time_s)`,这里会红 —— 那等于白送 lawnmower 几百秒。
    """
    t_last = float(np.nanmax(res.visit_time_s))
    assert res.t_complete_s > t_last
    assert res.t_complete_s == pytest.approx(float(res.t_s[-1]))
    assert res.t_complete_s == pytest.approx(float(np.nanmax(res.finish_time_s)))


def test_all_queues_empty_at_the_end(res):
    assert np.all(res.vehicle_status[-1] == STATUS_IDLE)
    assert not np.any(np.isnan(res.finish_time_s))


def test_dwell_total_matches_target_count(res):
    """总用时 = 航程/速度 + 観測数 x 停留(拐点不停) —— 拆得开才说明没多花时间。"""
    for i in range(res.cfg.n_vehicles):
        travel = float(res.per_vehicle_distance_m[i]) / res.cfg.follower_speed_mps
        n_obs = int((res.visit_by == i).sum())
        want = travel + n_obs * res.cfg.dwell_time_s
        # finish 落在 dt 网格上,故容差取一个 dt
        assert abs(float(res.finish_time_s[i]) - want) <= res.cfg.dt_s + 1e-9


def test_turn_points_never_count_as_observation(res, mw, cfg):
    """拐点标为 WP_PROJECTION,既不停留也不计覆盖。"""
    queues, _, _, _ = build_vehicle_plans(mw, cfg)
    assert any(it.wp_id == WP_PROJECTION for q in queues for it in q)
    assert WP_PROJECTION not in res.waypoint_ids.tolist()


def test_timeline_shapes(res):
    T, V = len(res.t_s), res.cfg.n_vehicles
    assert res.vehicle_pos.shape == (T, V, 2)
    assert res.vehicle_status.shape == (T, V)
    assert res.vehicle_distance_m.shape == (T, V)
    assert res.visited_count.shape == (T,)
    assert res.finish_time_s.shape == (V,)
    assert np.all(np.diff(res.t_s) > 0)
    assert res.visited_count[-1] == res.visited_mask.sum()


def test_deterministic(mw, cfg):
    a, b = run_lawnmower(cfg, mw), run_lawnmower(cfg, mw)
    assert np.array_equal(a.vehicle_pos, b.vehicle_pos)
    assert np.array_equal(np.nan_to_num(a.visit_time_s, nan=-1.0),
                          np.nan_to_num(b.visit_time_s, nan=-1.0))


def test_save_load_roundtrip(res, tmp_path):
    import json
    import os

    p = str(tmp_path / "lawn.npz")
    save_lawn_result(res, p)
    back = load_lawn_result(p)
    assert np.array_equal(back["vehicle_pos"], res.vehicle_pos)
    assert back["meta"]["n_vehicles"] == res.cfg.n_vehicles
    with open(os.path.splitext(p)[0] + "_summary.json", encoding="utf-8") as fh:
        s = json.load(fh)["summary"]
    assert s["t_complete_s"] == pytest.approx(res.t_complete_s)
    assert s["coverage"] == 1.0


# ======================================================================
# 旋钮真的起作用(改了参数结果必须变)
# ======================================================================
def test_narrower_swath_costs_more_time(mw):
    """幅宽变窄 ⇒ 测线更多 ⇒ 更慢。这条钉住【待裁决 L1】的方向性。"""
    base = MissionConfig(max_mission_time_s=30000.0)
    wide = run_lawnmower(LawnmowerConfig(base=base, swath_width_m=6.0), mw)
    narrow = run_lawnmower(LawnmowerConfig(base=base, swath_width_m=5.4), mw)
    assert len(narrow.meta["lane_easts"][0]) > len(wide.meta["lane_easts"][0])
    assert narrow.t_complete_s > wide.t_complete_s


def test_more_vehicles_is_faster(mw):
    base = MissionConfig(max_mission_time_s=30000.0)
    three = run_lawnmower(LawnmowerConfig(base=base, n_vehicles=3), mw)
    six = run_lawnmower(LawnmowerConfig(base=base, n_vehicles=6), mw)
    assert six.t_complete_s < three.t_complete_s
    assert six.coverage == three.coverage == 1.0


def test_return_legs_cost_more_than_boustrophedon(mw):
    base = MissionConfig(max_mission_time_s=30000.0)
    bou = run_lawnmower(LawnmowerConfig(base=base, boustrophedon=True), mw)
    one_way = run_lawnmower(LawnmowerConfig(base=base, boustrophedon=False), mw)
    assert one_way.t_complete_s > bou.t_complete_s
    assert float(one_way.per_vehicle_distance_m.max()) > \
        float(bou.per_vehicle_distance_m.max()) * 1.9      # 多出整整一趟空回航


# ======================================================================
# 跨场景对比(本基线存在的理由)
# ======================================================================
def test_completion_time_definition_is_the_same_ruler(mw):
    """两个场景的 `t_complete_s` 必须是同一把尺子:都 ≥ 各自最后一次観測时刻,
    且都**不是** `duration_s`(VRP 的 duration 含 settle 余量)。"""
    vrp = run_mission(MissionConfig(**FAST_VRP), mw)
    lawn = run_lawnmower(LawnmowerConfig(base=MissionConfig(max_mission_time_s=20000.0)),
                         mw)
    for r in (vrp, lawn):
        assert r.t_complete_s >= float(np.nanmax(r.visit_time_s))
    assert vrp.t_complete_s < vrp.summary()["duration_s"], \
        "VRP 的 duration 含 settle_time_s 余量,拿它比时间会多算"
    assert vrp.summary()["scenario"] == "vrp"
    assert lawn.summary()["scenario"] == "lawnmower"


def test_vrp_t_complete_is_nan_without_full_coverage(mw):
    """没全覆盖就没有"完成时刻" —— 返回 nan,不许拿一个漏了目标的数去比。

    ⚠ 制造"漏点"的方法随 D16 变了:0.5/0.5 下 Leader 会停船等队友,单纯调快 Leader
      已经掉不了覆盖了(那正是 D16 的成果)。要掉覆盖得把等待策略关掉。
    """
    bad = run_mission(MissionConfig(leader_wait_on_lagging_follower=False,
                                    leader_wait_on_endangered_target=False,
                                    leader_wait_on_turnaround=False,
                                    **FAST_VRP), mw)
    assert bad.coverage < 1.0
    assert np.isnan(bad.t_complete_s)


def test_cooperative_beats_lawnmower(mw):
    """本基线存在的理由:同样 3 台 AUV、同样 100% 覆盖,協調探査明显更快。"""
    vrp = run_mission(MissionConfig(**FAST_VRP), mw)
    lawn = run_lawnmower(LawnmowerConfig(base=MissionConfig(max_mission_time_s=20000.0)),
                         mw)
    assert vrp.coverage == lawn.coverage == 1.0
    assert vrp.cfg.n_followers + 1 == lawn.cfg.n_vehicles, "台数要一样才可比"
    assert lawn.t_complete_s > 3.0 * vrp.t_complete_s


# ======================================================================
# 脚本 CLI 默认值不许成为第二个真值源
# ======================================================================
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


@pytest.mark.parametrize("script", ["04_run_mission.py", "06_run_lawnmower.py"])
def test_script_defaults_come_from_the_contract(script):
    """两个脚本的**共享运动学**默认值必须逐个等于 `MissionConfig` 的默认值。

    为什么要有这条:D15 把 `dwell_time_s` 从 5 改到 10 时,契约、测试、VRP 场景全都
    跟着变了,唯独 `06_run_lawnmower.py` 的 `--dwell` 里手抄着一个 5.0 ——
    于是对照基线仍按 5 s 跑,两个场景的"配置相同"被悄悄破坏,而且**没有任何测试变红**。
    在这里把 CLI 默认值也钉到契约上,同类漏网不会再发生。
    """
    d = MissionConfig()
    p = _load_script(script).build_parser()
    shared = {"dwell": d.dwell_time_s,
              "follower_speed": d.follower_speed_mps,
              "dt": d.dt_s}
    for dest, want in shared.items():
        got = p.get_default(dest)
        assert got == want, (
            f"{script} 的 --{dest.replace('_', '-')} 默认值 {got} != 契约的 {want};"
            f"共享参数不许在脚本里写字面量")


def test_both_scenarios_agree_on_kinematics_via_cli_defaults():
    """走一遍两个脚本的默认 CLI → 构出的配置在共享参数上必须完全一致。"""
    m = _load_script("04_run_mission.py").build_parser().parse_args([])
    lw = _load_script("06_run_lawnmower.py").build_parser().parse_args([])
    assert (m.dwell, m.follower_speed, m.dt) == (lw.dwell, lw.follower_speed, lw.dt)
