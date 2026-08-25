"""样式契约测试:JSON 往返、字号缩放、**改了样式画面必须真的变**。

最后一条最重要 —— 一个"传了 style 但画的时候没用上"的 bug 不会报错,
只会让用户改完 JSON 出一张一模一样的图,而且看不出来。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vrpsim.contracts.config import MothraSimConfig  # noqa: E402
from vrpsim.contracts.style import (DEFAULT_STYLE, FONT_FIELDS,  # noqa: E402
                                    FigureStyle, dump_style, load_style)
from vrpsim.viz import plot_mission_snapshot, plot_world  # noqa: E402
from vrpsim.world import build_mothra_world  # noqa: E402


@pytest.fixture(scope="module")
def mw():
    return build_mothra_world(MothraSimConfig())


@pytest.fixture(scope="module")
def timeline(mw):
    from vrpsim.contracts.mission import MissionConfig
    from vrpsim.mission import run_mission

    res = run_mission(MissionConfig(solver="greedy", vrp_time_limit_s=0.1), mw)
    return {"t_s": res.t_s, "leader_north_m": res.leader_north_m,
            "window_north": res.window_north, "follower_pos": res.follower_pos,
            "visit_time_s": res.visit_time_s, "meta": res.meta}


# ======================================================================
# 契约本身
# ======================================================================
def test_json_roundtrip_is_exact(tmp_path):
    """写出去再读回来必须**逐字段相等**(tuple 不许退化成 list)。"""
    p = str(tmp_path / "s.json")
    dump_style(p)
    back = FigureStyle.from_json(p)
    assert back == DEFAULT_STYLE
    assert isinstance(back.follower_colors, tuple)


def test_dumped_json_has_every_field(tmp_path):
    """模板必须是全字段的 —— 用户拿到半份模板会以为剩下的不能改。"""
    from dataclasses import fields

    p = str(tmp_path / "s.json")
    dump_style(p)
    with open(p, encoding="utf-8") as fp:
        d = json.load(fp)
    assert set(d) == {f.name for f in fields(FigureStyle)}


def test_unknown_key_raises_with_suggestion():
    """手改 JSON 打错字必须当场报错,还要给出最接近的正确字段名。"""
    with pytest.raises(ValueError, match="font_size_tick"):
        FigureStyle.from_dict({"font_size_tik": 12.0})
    with pytest.raises(ValueError):
        DEFAULT_STYLE.with_overrides(nonexistent_field=1.0)


def test_font_scale_hits_every_font_field_and_nothing_else():
    st = DEFAULT_STYLE.scaled_fonts(1.5)
    assert FONT_FIELDS, "字号字段一个都没认出来 —— 前缀约定被破坏了"
    for name in FONT_FIELDS:
        assert getattr(st, name) == pytest.approx(getattr(DEFAULT_STYLE, name) * 1.5)
    # 线宽/配色/版面不许被字号缩放误伤
    assert st.traj_linewidth == DEFAULT_STYLE.traj_linewidth
    assert st.fig_width_in == DEFAULT_STYLE.fig_width_in
    assert st.follower_colors == DEFAULT_STYLE.follower_colors


def test_font_scale_must_be_positive():
    with pytest.raises(ValueError):
        DEFAULT_STYLE.scaled_fonts(0.0)


def test_load_style_missing_file_is_loud(tmp_path):
    """路径给了却读不到 ⇒ 报错。静默回落到默认样式会让人以为改生效了。"""
    with pytest.raises(SystemExit):
        load_style(str(tmp_path / "nope.json"))
    assert load_style(None) == DEFAULT_STYLE          # 不传 = 默认,不报错


def test_load_style_override_order(tmp_path):
    """文件 → 逐字段覆盖 → 字号缩放,三段的先后顺序。"""
    p = str(tmp_path / "s.json")
    DEFAULT_STYLE.with_overrides(dpi=300, font_size_tick=10.0).to_json(p)
    st = load_style(p, font_scale=2.0, dpi=None)       # None 覆盖 = 不覆盖
    assert st.dpi == 300
    assert st.font_size_tick == pytest.approx(20.0)    # 缩放作用在覆盖之后


def test_default_font_sizes_match_matplotlib_defaults():
    """base/title/label/tick 的默认值必须等于 matplotlib 的默认值。

    02/03 在接管前用的就是 matplotlib 默认字号 —— 这条钉住"引入样式契约不改变
    任何一张既有图"。谁想让图更紧凑,请改自己的 style JSON,别改契约默认值。
    """
    from matplotlib.font_manager import font_scalings

    rcd = matplotlib.rcParamsDefault
    base = float(rcd["font.size"])

    def pt(key):
        """matplotlib 的默认字号是 'large'/'medium' 这类**相对**值,先折算成 pt。"""
        v = rcd[key]
        return float(v) if isinstance(v, (int, float)) else base * font_scalings[v]

    assert DEFAULT_STYLE.font_size_base == base
    assert DEFAULT_STYLE.font_size_title == pytest.approx(pt("axes.titlesize"))
    assert DEFAULT_STYLE.font_size_label == pytest.approx(pt("axes.labelsize"))
    assert DEFAULT_STYLE.font_size_tick == pytest.approx(pt("xtick.labelsize"))
    assert DEFAULT_STYLE.font_size_tick == pytest.approx(pt("ytick.labelsize"))


def test_apply_rcparams_leaves_figure_dpi_alone():
    """figure.dpi 会扰动 bbox_inches='tight' 的出图尺寸,不许在这里动。"""
    before = matplotlib.rcParams["figure.dpi"]
    try:
        DEFAULT_STYLE.with_overrides(dpi=333).apply_rcparams()
        assert matplotlib.rcParams["figure.dpi"] == before
        assert matplotlib.rcParams["savefig.dpi"] == 333
    finally:
        DEFAULT_STYLE.apply_rcparams()


def test_apply_rcparams_writes_font_sizes():
    st = DEFAULT_STYLE.scaled_fonts(2.0)
    try:
        st.apply_rcparams()
        assert matplotlib.rcParams["font.size"] == pytest.approx(st.font_size_base)
        assert matplotlib.rcParams["legend.fontsize"] == pytest.approx(st.font_size_legend)
        assert matplotlib.rcParams["axes.unicode_minus"] is False
    finally:
        DEFAULT_STYLE.apply_rcparams()                 # 全局状态,用完还原


def test_viz_module_constants_track_the_contract():
    """viz.py 的模块常数是**派生**的,不是第二处配置源。"""
    from vrpsim import viz

    assert viz.FOLLOWER_COLORS == DEFAULT_STYLE.follower_colors
    assert viz.LEADER_COLOR == DEFAULT_STYLE.leader_color
    assert viz.TYPE_STYLE == DEFAULT_STYLE.type_style
    assert viz._MARKER_SIZE == DEFAULT_STYLE.target_marker_size


# ======================================================================
# 样式 → 画面(核心:改了必须真的变)
# ======================================================================
def test_snapshot_honors_marker_size_and_linewidth(mw, timeline):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    st = DEFAULT_STYLE.with_overrides(target_marker_size=180.0, traj_linewidth=5.5,
                                      follower_marker_size=21.0)
    _, ax = plt.subplots()
    plot_mission_snapshot(ax, mw, timeline, 300.0, style=st)

    assert ax.collections, "该时刻应有目标被画出"
    for c in ax.collections:
        assert float(np.asarray(c.get_sizes()).max()) == pytest.approx(180.0)
    # >2 个点的线里, Leader 自己走过的测线(多车道起也是多点折线)用的是
    # track_linewidth,不是这里在测的 traj_linewidth —— 按颜色排除掉它,
    # 只留 Follower 轨迹。
    traj = [ln for ln in ax.lines if len(ln.get_xdata()) > 2
           and to_rgb(ln.get_color()) != to_rgb(st.leader_color)]
    assert traj and all(ln.get_linewidth() == pytest.approx(5.5) for ln in traj)
    marks = [ln for ln in ax.lines if len(ln.get_xdata()) == 1]
    assert any(ln.get_markersize() == pytest.approx(21.0) for ln in marks)
    plt.close(ax.figure)


def test_snapshot_honors_colors(mw, timeline):
    """换配色必须落到画面上(轨迹线 + 已确认目标的面色)。"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    st = DEFAULT_STYLE.with_overrides(follower_colors=("#000080", "#008000"),
                                      chimney_color="#123456")
    _, ax = plt.subplots()
    plot_mission_snapshot(ax, mw, timeline, float(timeline["t_s"][-1]), style=st)

    traj_colors = {to_rgb(ln.get_color()) for ln in ax.lines
                   if len(ln.get_xdata()) > 2}
    assert to_rgb("#000080") in traj_colors and to_rgb("#008000") in traj_colors

    face = [tuple(np.asarray(c.get_facecolors())[0][:3]) for c in ax.collections
            if len(c.get_facecolors())]
    assert any(np.allclose(f, to_rgb("#123456")) for f in face)
    plt.close(ax.figure)


def test_style_default_is_pixel_identical_to_no_style(mw, timeline):
    """显式传默认样式,与不传 style,画出的图必须完全一样 ——
    否则说明重构时偷偷改了某个默认值,历史图就复现不了了。"""
    import matplotlib.pyplot as plt

    def render(**kw):
        fig, ax = plt.subplots(figsize=(3, 5), dpi=50)
        plot_mission_snapshot(ax, mw, timeline, 300.0, **kw)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        return buf

    assert np.array_equal(render(), render(style=DEFAULT_STYLE))


def test_plot_world_honors_style(mw):
    import matplotlib.pyplot as plt

    st = DEFAULT_STYLE.with_overrides(target_marker_size=200.0, font_size_label=17.0)
    ax = plot_world(mw, show_field=False, style=st)
    assert ax.get_xlabel() and ax.xaxis.label.get_size() == pytest.approx(17.0)
    assert all(float(np.asarray(c.get_sizes()).max()) == pytest.approx(200.0)
               for c in ax.collections)
    plt.close(ax.figure)


def test_trajectory_halo_follows_style(mw, timeline):
    """彩色底图上轨迹要有深色描边;设 0 必须真的关掉(灰度底图下用)。"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    def traj_lines(style):
        _, ax = plt.subplots()
        plot_mission_snapshot(ax, mw, timeline, 300.0, style=style)
        # Leader 自己的测线(多点折线)不描边, 排除掉, 只看 Follower 轨迹。
        out = [ln for ln in ax.lines if len(ln.get_xdata()) > 2
              and to_rgb(ln.get_color()) != to_rgb(style.leader_color)]
        plt.close(ax.figure)
        return out

    assert DEFAULT_STYLE.traj_halo_linewidth > 0, "默认应带描边(默认底图是彩色测深图)"
    assert all(ln.get_path_effects() for ln in traj_lines(DEFAULT_STYLE))
    off = DEFAULT_STYLE.with_overrides(traj_halo_linewidth=0.0)
    assert not any(ln.get_path_effects() for ln in traj_lines(off))


def test_scale_knobs_still_override_style(mw, timeline):
    """临时试参数用的单点覆盖仍然优先于 style。"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    st = DEFAULT_STYLE.with_overrides(traj_linewidth=1.0)
    _, ax = plt.subplots()
    plot_mission_snapshot(ax, mw, timeline, 300.0, style=st, traj_linewidth=7.0)
    traj = [ln for ln in ax.lines if len(ln.get_xdata()) > 2
           and to_rgb(ln.get_color()) != to_rgb(st.leader_color)]
    assert traj and all(ln.get_linewidth() == pytest.approx(7.0) for ln in traj)
    plt.close(ax.figure)
