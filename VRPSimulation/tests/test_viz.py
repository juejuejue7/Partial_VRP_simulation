"""viz 冒烟测试:出图不能炸,朝向不能反,回放接口要能吃 route/coverage。"""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vrpsim.contracts.config import MothraSimConfig  # noqa: E402
from vrpsim.viz import (FOLLOWER_COLORS, TYPE_STYLE, export_basemap,  # noqa: E402
                        load_basemap, plot_coverage, plot_leader_track,
                        plot_mission_snapshot, plot_routes, plot_window, plot_world,
                        sweep_times, use_cjk_font)
from vrpsim.world import build_mothra_world  # noqa: E402


@pytest.fixture(scope="module")
def mw():
    return build_mothra_world(MothraSimConfig())


def test_plot_world_runs(mw):
    import matplotlib.pyplot as plt

    ax = plot_world(mw, show_names=True)
    assert ax.get_xlim() == (0.0, mw.world.y_max_m), "横轴必须是 East"
    assert ax.get_ylim() == (0.0, mw.world.x_max_m), "纵轴必须是 North"
    assert ax.get_xlabel().startswith("East") and ax.get_ylabel().startswith("North")
    plt.close(ax.figure)


def test_plot_world_extent_is_portrait(mw):
    """场地 500(N) x 100(E) ⇒ 图必须是竖长条。横过来 = 轴序写反了。"""
    import matplotlib.pyplot as plt

    ax = plot_world(mw)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert (y1 - y0) > (x1 - x0) * 2.0
    plt.close(ax.figure)


def test_window_and_track_overlay(mw):
    """窗口矩形与 Leader 测线能叠上去,且窗口画在场地范围内。"""
    import matplotlib.pyplot as plt

    from vrpsim.contracts.config import MothraSimConfig
    from vrpsim.windows import enumerate_windows, leader_track

    cfg = MothraSimConfig()
    ax = plot_world(mw, show_field=False)
    plot_leader_track(ax, leader_track(mw.world, cfg.lane_spacing_m))
    snaps = enumerate_windows(mw.world, cfg.window, mw.dataset.targets_ned)
    rect = plot_window(ax, snaps[-1], label="last")
    x, y = rect.get_xy()
    assert -2.0 <= x and x + rect.get_width() <= mw.world.y_max_m + 2.0
    assert 0.0 <= y and y + rect.get_height() <= mw.world.x_max_m + 2.0
    plt.close(ax.figure)


def test_marker_style_is_shape_plus_color():
    """形状 + 颜色双重编码(色觉可达性),不许退化成只靠颜色。"""
    markers = {m for m, _ in TYPE_STYLE.values()}
    colors = {c for _, c in TYPE_STYLE.values()}
    assert len(markers) == len(TYPE_STYLE) and len(colors) == len(TYPE_STYLE)


def test_routes_and_coverage_overlay(mw):
    import matplotlib.pyplot as plt

    ax = plot_world(mw, show_field=False)
    routes = [mw.dataset.targets_ned[:5], mw.dataset.targets_ned[5:9]]
    plot_routes(ax, routes, starts=[np.array([0.0, 0.0]), np.array([0.0, 50.0])])
    cov = np.zeros((mw.world.nx, mw.world.ny), dtype=bool)
    cov[100:150, 20:60] = True
    plot_coverage(ax, cov, mw.world)
    plt.close(ax.figure)


def test_basemap_roundtrip_is_north_ascending(tmp_path, mw):
    """底图行序必须翻成 North 升序,与 field 同朝向。"""
    dst = str(tmp_path / "base.npz")
    got = export_basemap(dst)
    if got is None:
        pytest.skip("上游 mothra_bundle.npz 不在,底图是可选项")

    z = load_basemap(dst)
    assert z is not None and z.shape == (mw.world.nx, mw.world.ny)

    # 与上游原始栅格逐格对照:翻行序后再按分析域切,z[ix, iy] == z_utm1m[652-ix, iy]
    from vrpsim.contracts.config import UPSTREAM_BUNDLE_NPZ
    with np.load(UPSTREAM_BUNDLE_NPZ, allow_pickle=False) as d:
        raw = np.asarray(d["z_utm1m"], dtype=np.float32)
    assert np.array_equal(z, raw[::-1, :][: mw.world.nx, : mw.world.ny])

    # 每个目标点按 NED 索引取到的水深,应与其表列 depth 一致(±1.5 m,考虑格心 vs 点位)
    for (x_n, y_e), d_down in zip(mw.dataset.targets_ned, mw.dataset.depth_D_m):
        assert abs(-float(z[int(x_n), int(y_e)]) - d_down) < 1.5


def test_load_basemap_missing_returns_none(tmp_path):
    assert load_basemap(str(tmp_path / "nope.npz")) is None


def test_show_targets_false_draws_no_markers(mw):
    """快照要自己按状态画目标,所以 plot_world 必须能只铺底图。"""
    import matplotlib.pyplot as plt

    ax = plot_world(mw, show_field=False, show_targets=False, colorbar=False, title="")
    assert len(ax.collections) == 0
    plt.close(ax.figure)


# ======================================================================
# 任务时序快照
# ======================================================================
@pytest.fixture(scope="module")
def timeline(mw):
    """跑一次短任务,转成 load_result() 那种 dict(快照函数吃的就是它)。"""
    from vrpsim.contracts.mission import MissionConfig
    from vrpsim.mission import run_mission

    res = run_mission(MissionConfig(solver="greedy", vrp_time_limit_s=0.1), mw)
    return {
        "t_s": res.t_s, "leader_north_m": res.leader_north_m,
        "window_north": res.window_north, "follower_pos": res.follower_pos,
        "visit_time_s": res.visit_time_s, "meta": res.meta,
    }


def test_sweep_times_all_revealed_and_ordered(mw, timeline):
    """North 越靠北的目标,越晚被窗口扫到;且全部都会被扫到。"""
    sw = sweep_times(mw, timeline)
    assert not np.any(np.isnan(sw)), "有目标从未进过窗口"
    order = np.argsort(mw.dataset.targets_ned[:, 0])
    assert np.all(np.diff(sw[order]) >= -1e-9), "首次进窗时刻应随 North 单调不减"


def test_sweep_precedes_observation(mw, timeline):
    """不可能没被扫到就先被观测 —— 揭示必须早于(或等于)观测。"""
    sw = sweep_times(mw, timeline)
    vis = timeline["visit_time_s"]
    ok = ~np.isnan(vis)
    assert np.all(sw[ok] <= vis[ok] + 1e-9)


def test_snapshot_hides_unswept_targets(mw, timeline):
    """核心要求:t=0 一个目标都不画;末时刻全部画出。"""
    import matplotlib.pyplot as plt

    def n_markers(t):
        _, ax = plt.subplots()
        plot_mission_snapshot(ax, mw, timeline, t)
        n = int(sum(c.get_offsets().shape[0] for c in ax.collections))
        plt.close(ax.figure)
        return n

    assert n_markers(0.0) == 0, "t=0 时窗口面积为零,不该画出任何目标"
    assert n_markers(float(timeline["t_s"][-1])) == mw.dataset.n
    # 中途单调不减
    counts = [n_markers(t) for t in (0.0, 100.0, 200.0, 300.0, 400.0, 500.0)]
    assert all(b >= a for a, b in zip(counts[:-1], counts[1:])), counts


def _hollow_collections(ax):
    """返回「空心」scatter(面色全透明)的 collection 列表。"""
    out = []
    for c in ax.collections:
        fc = c.get_facecolors()
        if len(fc) == 0 or float(np.asarray(fc)[:, 3].max()) < 1e-6:
            out.append(c)
    return out


def test_unconfirmed_targets_are_type_agnostic(mw, timeline):
    """未被 Follower 确认的目标**不分类型**:任何时刻空心标记只有一个 collection。

    依据锁定术语:目標探査(Leader 声呐)只知道"这里有东西",分不出 chimney/mound;
    要到目標観測(Follower 相机)之后才分得清。若有人把它拆回按类型画,这里会红。
    """
    import matplotlib.pyplot as plt

    sw = sweep_times(mw, timeline)
    vis = timeline["visit_time_s"]
    saw_mixed = False
    for t in (100.0, 200.0, 300.0, 400.0, 450.0, 500.0):
        todo = (~np.isnan(sw) & (sw <= t)) & ~(~np.isnan(vis) & (vis <= t))
        _, ax = plt.subplots()
        plot_mission_snapshot(ax, mw, timeline, t)
        hollow = _hollow_collections(ax)
        assert len(hollow) <= 1, f"t={t}: 空心标记被拆成了 {len(hollow)} 组"
        if todo.any():
            assert hollow and hollow[0].get_offsets().shape[0] == int(todo.sum())
            if len(set(mw.dataset.type_[todo].tolist())) > 1:
                saw_mixed = True     # 确实出现过"两种类型同时未确认"
        plt.close(ax.figure)
    assert saw_mixed, "样本里没出现两种类型同时未确认的时刻,该断言没被真正检验到"


def test_confirmed_targets_keep_type_distinction(mw, timeline):
    """已确认的目标必须按 chimney / mound 分开画(实心,两组)。"""
    import matplotlib.pyplot as plt

    t = float(timeline["t_s"][-1])
    _, ax = plt.subplots()
    plot_mission_snapshot(ax, mw, timeline, t)
    solid = [c for c in ax.collections if c not in _hollow_collections(ax)]
    assert len(solid) == len(set(mw.dataset.type_.tolist())) == 2
    assert sum(c.get_offsets().shape[0] for c in solid) == mw.dataset.n
    plt.close(ax.figure)


def test_snapshot_has_no_text_or_legend(mw, timeline):
    """规格:只画图像和坐标轴 —— 无图例、无标题、无轴标签。"""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    plot_mission_snapshot(ax, mw, timeline, 300.0)
    assert ax.get_legend() is None
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "" and ax.get_ylabel() == ""
    assert not [t for t in ax.texts if t.get_text().strip()]
    plt.close(ax.figure)


def test_snapshot_trajectory_is_truncated_at_t(mw, timeline):
    """轨迹只画到当前时刻 —— 这是"时序"的关键,画多了就成了剧透。"""
    import matplotlib.pyplot as plt

    t = 250.0
    k = int(np.searchsorted(timeline["t_s"], t + 1e-9, side="right") - 1)
    _, ax = plt.subplots()
    plot_mission_snapshot(ax, mw, timeline, t)
    traj = [ln for ln in ax.lines if len(ln.get_xdata()) > 2]
    assert traj, "没找到轨迹线"
    for ln in traj:
        assert len(ln.get_xdata()) == k + 1
    plt.close(ax.figure)


def test_snapshot_keeps_portrait_extent(mw, timeline):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    plot_mission_snapshot(ax, mw, timeline, 300.0)
    assert ax.get_xlim() == (0.0, mw.world.y_max_m)
    assert ax.get_ylim() == (0.0, mw.world.x_max_m)
    plt.close(ax.figure)


def test_follower_colors_distinct_from_target_colors():
    """无图例的图靠颜色认人:Follower 配色不能和目标点撞。"""
    from matplotlib.colors import to_rgb

    tgt = [to_rgb(c) for _, c in TYPE_STYLE.values()]
    for fc in FOLLOWER_COLORS:
        for tc in tgt:
            d = sum((a - b) ** 2 for a, b in zip(to_rgb(fc), tc)) ** 0.5
            assert d > 0.35, f"{fc} 与目标色 {tc} 太接近"


def test_cjk_font_available():
    """图里有中日文;本机无 CJK 字体会出豆腐块 —— 至少要能选到一个。"""
    assert use_cjk_font() is not None
