"""地形底图渲染:必须与 Bethmetory_data_process 的正式方案**逐元素一致**。

核心是 `test_matches_upstream_elementwise` —— 直接把上游脚本 import 进来对拍。
复刻一段渲染代码而不锁死一致性,过几个月两边就会悄悄分叉,而分叉的症状只是
"图看着有点不一样",没人会当成 bug。
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vrpsim.contracts.config import REPO_ROOT  # noqa: E402
from vrpsim.contracts.style import DEFAULT_STYLE, FigureStyle  # noqa: E402
from vrpsim.viz import (GRAY_RAMP, HAXBY, make_bathy_cmap,  # noqa: E402
                        normalize_bathy, render_terrain_rgb)

_UPSTREAM = os.path.join(REPO_ROOT, "Bethmetory_data_process", "scripts",
                         "04_mothra_plot.py")


@pytest.fixture(scope="module")
def upstream():
    """按路径 import 上游脚本(它不是包,文件名还以数字开头)。"""
    if not os.path.isfile(_UPSTREAM):
        pytest.skip("上游 04_mothra_plot.py 不在,渲染对拍跳过")
    spec = importlib.util.spec_from_file_location("upstream_mothra_plot", _UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dem():
    """一小块真实地形(拿不到就用带噪声的合成面,统计性质接近即可)。"""
    from vrpsim.contracts.config import UPSTREAM_BUNDLE_NPZ

    if os.path.isfile(UPSTREAM_BUNDLE_NPZ):
        with np.load(UPSTREAM_BUNDLE_NPZ, allow_pickle=False) as d:
            return np.asarray(d["z_utm1m"], dtype=np.float64)[:120, :90]
    rng = np.random.default_rng(0)
    ix, iy = np.mgrid[0:120, 0:90]
    return (-2270.0 + 0.06 * ix - 0.04 * iy
            + 3.0 * np.sin(ix / 9.0) * np.cos(iy / 7.0)
            + rng.normal(0.0, 0.25, (120, 90)))


# ======================================================================
# 与上游对拍
# ======================================================================
def test_palette_is_copied_verbatim(upstream):
    assert list(HAXBY) == list(upstream.HAXBY)


def test_shading_constants_match_upstream(upstream):
    from vrpsim import viz

    assert viz._SHADE_FLOOR == upstream.SHADE_FLOOR
    assert (viz._HS_LO, viz._HS_SPAN) == (upstream.HS_LO, upstream.HS_SPAN)
    assert DEFAULT_STYLE.basemap_sun_azimuth_deg == upstream.AZDEG
    assert DEFAULT_STYLE.basemap_sun_altitude_deg == upstream.ALTDEG
    assert DEFAULT_STYLE.basemap_vert_exag == upstream.VERT_EXAG
    assert DEFAULT_STYLE.basemap_shade_strength == upstream.SHADE_STRENGTH
    assert DEFAULT_STYLE.basemap_norm == upstream.NORM


@pytest.mark.parametrize("norm", ["equalize", "linear"])
def test_normalize_matches_upstream(upstream, dem, norm):
    mask = np.isnan(dem)
    assert np.array_equal(normalize_bathy(dem, mask, norm),
                          upstream.normalize(dem, mask, norm), equal_nan=True)


def test_cmap_matches_upstream(upstream):
    x = np.linspace(0.0, 1.0, 256)
    for lo, hi in ((0.0, 1.0), (0.1, 0.8)):
        assert np.allclose(make_bathy_cmap(lo, hi)(x), upstream.make_cmap(lo, hi)(x))


@pytest.mark.parametrize("norm", ["equalize", "linear"])
@pytest.mark.parametrize("strength,vert_exag,altdeg", [(1.0, 2.0, 45.0),
                                                       (0.6, 1.2, 60.0)])
def test_matches_upstream_elementwise(upstream, dem, norm, strength, vert_exag, altdeg):
    """同一份 row0=北 的数组、同一组参数 ⇒ 输出必须逐元素相同。"""
    ours = render_terrain_rgb(dem, dx_m=1.0, dy_m=1.0, rows_north_up=True,
                              strength=strength, vert_exag=vert_exag, altdeg=altdeg,
                              norm_mode=norm)
    theirs = upstream.render_rgb(dem, 1.0, 1.0, strength=strength,
                                 vert_exag=vert_exag, altdeg=altdeg, norm_mode=norm)
    assert np.array_equal(ours, theirs)


def test_nan_becomes_white(upstream, dem):
    z = dem.copy()
    z[10:20, 5:15] = np.nan
    ours = render_terrain_rgb(z, rows_north_up=True)
    assert np.all(ours[10:20, 5:15] == 1.0)
    assert np.array_equal(ours, upstream.render_rgb(z, 1.0, 1.0))


# ======================================================================
# 本项目特有:数组是 North 升序,必须先翻转再算晕渲
# ======================================================================
def _mean_brightness(z, **kw):
    return float(render_terrain_rgb(z, **kw).mean())


def test_north_ascending_flip_is_applied():
    """`rows_north_up=False`(本项目 z_ned)必须等价于"翻过来算再翻回去"。"""
    rng = np.random.default_rng(1)
    z = rng.normal(-2270.0, 2.0, (60, 50))
    a = render_terrain_rgb(z, rows_north_up=False)
    b = render_terrain_rgb(z[::-1], rows_north_up=True)[::-1]
    assert np.array_equal(a, b)


def test_west_facing_slope_is_lit_not_shaded():
    """光源在西北(315°) ⇒ **西向坡迎光**、东向坡背光。

    这条钉死之前那版自实现 Horn 晕渲的东西向翻转 bug ——
    它把西向坡压暗、东向坡提亮,山脊看着像沟,正是底图"发怪"的根因之一。
    """
    iy = np.ones((40, 1)) * np.arange(40)
    west_facing = _mean_brightness(-2270.0 + 0.05 * iy)   # 东高西低 ⇒ 坡面朝西
    east_facing = _mean_brightness(-2270.0 - 0.05 * iy)   # 西高东低 ⇒ 坡面朝东
    assert west_facing > east_facing


def test_north_facing_slope_is_lit_not_shaded():
    """同理,北向坡(南高北低)迎光。ix 是 North 升序,所以"南高"= 前几行大。"""
    ix = np.arange(40)[:, None] * np.ones((1, 40))
    north_facing = _mean_brightness(-2270.0 - 0.05 * ix)  # 南高北低 ⇒ 坡面朝北
    south_facing = _mean_brightness(-2270.0 + 0.05 * ix)
    assert north_facing > south_facing


# ======================================================================
# 只压暗不提亮 / 色带方向
# ======================================================================
def test_shading_only_darkens(dem):
    """晕渲是乘性且 ≤1:任何像素都不会比纯填色更亮 —— 这是不过曝的定义。"""
    flat = render_terrain_rgb(dem, rows_north_up=True, strength=0.0)
    shaded = render_terrain_rgb(dem, rows_north_up=True, strength=1.0)
    assert np.all(shaded <= flat + 1e-12)
    assert np.all(shaded >= flat * 0.45 - 1e-12)      # 下限 _SHADE_FLOOR


def test_deep_is_blue_shallow_is_warm():
    """linear 归一化下:最深处偏蓝(B>R),最浅处偏暖(R>B)。"""
    z = np.linspace(-2300.0, -2200.0, 64)[:, None] * np.ones((1, 8))
    rgb = render_terrain_rgb(z, rows_north_up=True, strength=0.0, norm_mode="linear")
    deep, shallow = rgb[0].mean(axis=0), rgb[-1].mean(axis=0)
    assert deep[2] > deep[0], "最深处应偏蓝"
    assert shallow[0] > shallow[2], "最浅处应偏暖"


def test_cmap_hi_truncates_the_orange_end():
    """`basemap_cmap_hi<1` 要真的截掉暖端(用于避开 mound 琥珀标记的撞色)。"""
    z = np.linspace(-2300.0, -2200.0, 64)[:, None] * np.ones((1, 8))
    full = render_terrain_rgb(z, rows_north_up=True, strength=0.0,
                              norm_mode="linear")[-1].mean(axis=0)
    cut = render_terrain_rgb(z, rows_north_up=True, strength=0.0, norm_mode="linear",
                             cmap_hi=0.6)[-1].mean(axis=0)
    assert cut[0] < full[0], "截掉暖端后最浅处应当没那么红"


def test_lighten_zero_is_upstream_and_positive_brightens(upstream, dem):
    """`basemap_lighten` 是本项目加的一档,默认 0 必须**逐位**等于上游。"""
    base = render_terrain_rgb(dem, rows_north_up=True, lighten=0.0)
    assert np.array_equal(base, upstream.render_rgb(dem, 1.0, 1.0))

    lit = render_terrain_rgb(dem, rows_north_up=True, lighten=0.4)
    assert np.all(lit >= base - 1e-12) and lit.mean() > base.mean()
    assert lit.max() <= 1.0 + 1e-12
    assert np.allclose(render_terrain_rgb(dem, rows_north_up=True, lighten=1.0), 1.0)


def test_gray_mode_is_achromatic():
    """gray 模式必须真的无彩:R≈G≈B。"""
    z = np.linspace(-2300.0, -2200.0, 32)[:, None] * np.ones((1, 8))
    rgb = render_terrain_rgb(z, rows_north_up=True, strength=0.0,
                             norm_mode="linear", colors=GRAY_RAMP)
    assert np.allclose(rgb[..., 0], rgb[..., 1], atol=0.02)
    assert np.allclose(rgb[..., 1], rgb[..., 2], atol=0.02)


# ======================================================================
# 样式契约
# ======================================================================
def test_enum_fields_reject_typos():
    for bad in ({"basemap_mode": "heqby"}, {"basemap_norm": "equalise"},
                {"basemap_interpolation": "bilinar"}):
        with pytest.raises(ValueError):
            FigureStyle(**bad)


def test_style_switches_basemap_palette(dem):
    """style 里换 basemap_mode,画面必须真的从彩色变灰。"""
    import matplotlib.pyplot as plt

    from vrpsim.contracts.config import MothraSimConfig
    from vrpsim.viz import plot_world
    from vrpsim.world import build_mothra_world

    mw = build_mothra_world(MothraSimConfig())
    base = np.asarray(dem[:1, :1], dtype=np.float32)   # 占位,下面用真尺寸
    base = np.resize(dem, (mw.world.nx, mw.world.ny)).astype(np.float32)

    def chroma(style):
        ax = plot_world(mw, basemap=base, show_field=False, colorbar=False,
                        title="", style=style)
        img = ax.images[0].get_array()
        rgb = np.asarray(img)[..., :3]
        plt.close(ax.figure)
        return float(np.abs(rgb.max(axis=-1) - rgb.min(axis=-1)).mean())

    assert chroma(DEFAULT_STYLE) > 0.05, "haxby 模式应当是彩色的"
    assert chroma(DEFAULT_STYLE.with_overrides(basemap_mode="gray")) < 0.02
