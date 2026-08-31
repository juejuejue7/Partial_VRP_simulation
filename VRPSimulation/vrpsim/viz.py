"""[第1层] 基础环境的可视化与回放。

==================================================================
出图朝向(唯一约定,别处不许再转轴)
==================================================================
本项目数组是 `field[ix, iy]`(0 轴 = North,1 轴 = East),而地图惯例是
**East 放横轴、North 放纵轴、北朝上**。二者恰好对上:

    imshow(field, origin="lower", extent=[0, y_max_E, 0, x_max_N])
      → row = ix = North(origin="lower" ⇒ 0 在下,北朝上)✅
      → col = iy = East (0 在左)✅

所以这里**不转置**。`msim/contracts/types.py` 里"可视化时自行转置"的提醒针对的是
"想把 x(North)放横轴"的画法 —— 本项目不采用那种画法,因为出来是躺倒的地图。
`tests/test_world_field.py::test_plot_orientation_matches_ned` 把朝向钉成回归测试。

上游水深底图(可选)只用于**出图**,不是环境层(D3);没有它照样能画。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from .contracts.config import UPSTREAM_BUNDLE_NPZ
from .contracts.style import DEFAULT_STYLE, FigureStyle
from .world import MothraWorld

__all__ = ["load_basemap", "export_basemap", "plot_world", "plot_routes",
           "plot_coverage", "plot_leader_track", "plot_window", "use_cjk_font",
           "sweep_times", "plot_mission_snapshot", "plot_twophase_snapshot",
           "render_terrain_rgb", "make_bathy_cmap", "normalize_bathy",
           "HAXBY", "GRAY_RAMP",
           "TYPE_STYLE", "FOLLOWER_COLORS", "LEADER_COLOR",
           "UNCONFIRMED_MARKER", "UNCONFIRMED_COLOR"]

# ⚠ 以下模块常数全部**派生自 `contracts/style.py` 的默认样式**,是给不传 style=
# 的老调用点用的兜底,不是第二处配置源。要改配色/大小请改契约或传 `style=`,
# 别在这里改数字 —— 在这里改会和 `--style` 出来的图不一致。
FOLLOWER_COLORS = DEFAULT_STYLE.follower_colors
LEADER_COLOR = DEFAULT_STYLE.leader_color
_FOLLOWER_COLORS = FOLLOWER_COLORS
_LEADER_COLOR = LEADER_COLOR

# 图里**同时**有简体中文与日文假名(锁定术语「目標分布確率マップ」),默认 DejaVu Sans
# 会出豆腐块。注意没有哪一款单独够用:Yu Gothic 缺"标/负/载/阈"等简体字,
# SimHei 假名不全 —— 所以把**全部**可用的 CJK 字体排成一列,靠 matplotlib 的字体回退
# 逐个补齐。Microsoft YaHei 排首位(简体 + 假名覆盖最广)。
_CJK_CANDIDATES = ("Microsoft YaHei", "Yu Gothic", "Meiryo", "MS Gothic",
                   "SimHei", "SimSun", "Malgun Gothic")


def use_cjk_font(prefer: Sequence[str] = ()) -> Optional[str]:
    """把 matplotlib 字体链切到本机可用的中日文字体(全部串上,靠回退补字)。

    `prefer` 里的字体排到最前(来自 `FigureStyle.font_family`);本机装不上的会被
    静默跳过 —— 换台机器出图不该因为字体名不同就跑不起来。

    返回排在首位的字体名;一个都没有时返回 None。
    """
    import matplotlib
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    chain = [n for n in tuple(prefer) + _CJK_CANDIDATES if n in available]
    chain = list(dict.fromkeys(chain))          # 去重且保序
    if not chain:
        return None
    matplotlib.rcParams["font.sans-serif"] = chain + [
        n for n in matplotlib.rcParams.get("font.sans-serif", []) if n not in chain]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chain[0]

# 目标标记样式:形状 + 颜色**双重编码**,识别不单靠颜色(详见 FigureStyle.type_style)。
TYPE_STYLE: Dict[str, tuple] = DEFAULT_STYLE.type_style
_TYPE_FALLBACK = ("s", "#e87ba4")
_MARKER_EDGE = DEFAULT_STYLE.marker_edge_color
_MARKER_SIZE = DEFAULT_STYLE.target_marker_size

# 未被 Follower 相机确认过的目标:**不分类型**,统一一种空心标记。
# 依据锁定术语:目標探査(Leader 声呐初查)只知道"这里有东西",
# 分不出 chimney / mound;要到目標観測(Follower 相机)之后才分得清。
# 画成"白圈 + 深色描边"是因为底图是灰度晕渲 —— 纯白在亮地形上、纯黑在暗地形上
# 都会消失,双层描边才能两头都跳得出来。
UNCONFIRMED_MARKER = "o"
UNCONFIRMED_COLOR = DEFAULT_STYLE.unconfirmed_color
UNCONFIRMED_HALO = DEFAULT_STYLE.unconfirmed_halo_color


# ======================================================================
# 可选底图(仅出图用,D3:不是环境层)
# ======================================================================
def export_basemap(dst_npz: str, src_bundle: str = UPSTREAM_BUNDLE_NPZ,
                   crop: Optional[object] = None) -> Optional[str]:
    """把上游 1 m 水深栅格转成 NED 朝向、裁到分析域后存一份,供出图当底。

    行序翻转:上游 `z_utm1m` 的 row 0 是**最北**一行(transform 的 dy = -1),
    而本项目 `field` 的 ix=0 是 North=0(最南)。故 `z_ned = z_utm1m[::-1, :]`,
    与 `NEDFrame.grid_index_1m` 的 `row = 652 - int(x_N)` 是同一件事。

    裁切(D7)因为原点不动,就是对翻转后的数组切 `[0:N, 0:E]`,无偏移项。

    源文件不存在时返回 None(不报错 —— 底图是可选的)。
    """
    if not os.path.isfile(src_bundle):
        return None
    from .contracts.config import CropConfig

    crop = crop or CropConfig()
    with np.load(src_bundle, allow_pickle=False) as d:
        z_ned = np.asarray(d["z_utm1m"], dtype=np.float32)[::-1, :]
    n_ext, e_ext = crop.extent
    z_ned = z_ned[: int(round(n_ext)), : int(round(e_ext))]

    os.makedirs(os.path.dirname(os.path.abspath(dst_npz)), exist_ok=True)
    np.savez(dst_npz, z_ned=z_ned, res_m=np.float64(1.0),
             crop_extent=np.array([n_ext, e_ext], dtype=np.float64),
             note=np.array("仅供出图底色,非环境层(D3);row=ix=North 升序, col=iy=East"))
    return dst_npz


def load_basemap(path: str) -> Optional[np.ndarray]:
    """读回 `export_basemap` 存的水深(NED 朝向,(653,294) float32)。缺文件返回 None。"""
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=False) as d:
        return np.asarray(d["z_ned"], dtype=np.float32)


def _text_halo():
    """白字加深色描边,压在亮暗都跳得出来。"""
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=2.0, foreground="#1a1a19")]


def _line_halo():
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=3.6, foreground="#1a1a19")]


# ======================================================================
# 地形底图渲染 —— 与 Bethmetory_data_process/scripts/04_mothra_plot.py 同一方案
# ======================================================================
# 上游那套是"haxby 填色 + 排名直方图均衡 + 不过曝乘性晕渲",本节逐式照搬,
# `tests/test_basemap_render.py` 直接 import 上游脚本做逐元素对拍。
#
# 为什么不 import 上游脚本来用:它是 script 不是包,import 即触发
# `matplotlib.use("Agg")`,会把后端强加给所有调用方。故在此复刻并用测试锁死一致性。
#
# ⚠ 与上游的**唯一**差异是格距:上游渲的是 WGS84 栅格(dx 1.000 / dy 1.488 m),
#   本项目渲的是 UTM 1 m 栅格(dx = dy = 1.0)。晕渲依赖格距,所以两张图的
#   阴影不会逐像素相同 —— 这是投影不同的必然结果,不是实现偏差。

#: haxby 论文风格测深色带(8 步,深蓝→青→绿→黄→橙)
HAXBY = ("#0a2a8c", "#1e6ffb", "#32c8ff", "#6aebd6",
         "#b6ee9e", "#eeea79", "#ffbd57", "#ff8c3a")
#: 中性灰色带(basemap_mode="gray"):底色不抢戏,让 AUV 与目标标记跳出来
GRAY_RAMP = ("#6f6f6b", "#f6f5f1")

_SHADE_FLOOR = 0.45          # 晕渲乘性下限:最暗处也只压到原色的 45%,不至于死黑
_HS_LO, _HS_SPAN = 0.15, 0.8  # 光强先线性拉伸再 clip,避免高光过曝


def make_bathy_cmap(lo: float = 0.0, hi: float = 1.0, n: int = 256,
                    colors: Sequence[str] = HAXBY):
    """测深色带,可截取其中一段(`hi<1` 截掉最暖的橙端,避免与目标标记撞色)。"""
    from matplotlib.colors import LinearSegmentedColormap

    base = LinearSegmentedColormap.from_list("bathy", list(colors), N=n)
    cm = (base if (lo <= 0.0 and hi >= 1.0) else
          LinearSegmentedColormap.from_list(
              f"bathy_{lo:g}_{hi:g}", base(np.linspace(lo, hi, n)), N=n))
    cm.set_bad("white")
    return cm


def normalize_bathy(z: np.ndarray, mask: Optional[np.ndarray] = None,
                    mode: str = "equalize") -> np.ndarray:
    """把水深映射到 [0,1];无效格留 NaN。

    equalize 按**排名**映射:深水段在本区占大半面积却只占线性色阶的一小段,
      均衡后主体层次拉得开。代价是颜色与水深不再等比,且映射依赖当前窗口的数据分布
      —— **换一块裁切范围重出图,同一颜色对应的水深会变,跨图比颜色不成立**。
    linear   与水深等比,可跨图比较颜色,但主体会挤在色带的一小段里。
    """
    z = np.asarray(z, dtype=np.float64)
    mask = np.isnan(z) if mask is None else np.asarray(mask, dtype=bool)
    out = np.full(z.shape, np.nan)
    valid = z[~mask]
    if valid.size == 0:
        return out
    if mode == "equalize":
        order = np.argsort(valid)
        ranks = np.empty(valid.size, float)
        ranks[order] = np.linspace(0.0, 1.0, valid.size)
        out[~mask] = ranks
    elif mode == "linear":
        lo, hi = float(valid.min()), float(valid.max())
        out[~mask] = (valid - lo) / (hi - lo) if hi > lo else 0.5
    else:
        raise ValueError(f"norm 只能是 'equalize' / 'linear',收到 {mode!r}")
    return out


def render_terrain_rgb(z: np.ndarray, *, dx_m: float = 1.0, dy_m: float = 1.0,
                       rows_north_up: bool = False, strength: float = 1.0,
                       vert_exag: float = 2.0, azdeg: float = 315.0,
                       altdeg: float = 45.0, cmap_lo: float = 0.0,
                       cmap_hi: float = 1.0, norm_mode: str = "equalize",
                       lighten: float = 0.0,
                       colors: Sequence[str] = HAXBY) -> np.ndarray:
    """色带填色 + 不过曝晕渲,返回 (H, W, 3) RGB。

    `rows_north_up`:**本项目的默认是 False** —— `z_ned` 的 row 0 是 North=0(最南),
      而 matplotlib 的 `LightSource` 按制图惯例假定 row 0 是最北(内部把 dy 取负)。
      不翻转就会让东西向的光照左右颠倒,山脊看着像沟(实测:西向坡本该迎光却被压暗)。
      故这里先翻成 row0=北算晕渲,再翻回来。
    """
    from matplotlib.colors import LightSource

    z = np.asarray(z, dtype=np.float64)
    flip = not rows_north_up
    zz = z[::-1] if flip else z

    mask = np.isnan(zz)
    # 算晕渲前先把无效格填掉,否则 NaN 会沿梯度算子扩散出一圈伪影
    filled = np.where(mask, np.nanmean(zz) if (~mask).any() else 0.0, zz)

    rgb = make_bathy_cmap(cmap_lo, cmap_hi, colors=colors)(
        normalize_bathy(zz, mask, norm_mode))[..., :3]

    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    hs = np.clip((ls.hillshade(filled, vert_exag=vert_exag, dx=dx_m, dy=dy_m)
                  - _HS_LO) / _HS_SPAN, 0.0, 1.0)
    # 乘性且**只压暗**:hs=1 保持原色,hs=0 压到 _SHADE_FLOOR。
    # 只压暗才不会把浅水区冲白 —— 这正是原灰度方案过曝发花的根因。
    mult = 1.0 - strength * (1.0 - (_SHADE_FLOOR + (1.0 - _SHADE_FLOOR) * hs))
    out = rgb * mult[..., None]
    # 朝白色线性提亮。lighten=0 时**逐位**等于上游输出(对拍测试钉住),
    # 调大则底图整体退向背景,让轨迹与标记跳出来。上游没有这一档 —— 它不叠图层。
    if lighten:
        out = out + (1.0 - out) * float(np.clip(lighten, 0.0, 1.0))
    out[mask] = 1.0
    return out[::-1] if flip else out


# ======================================================================
# 主图
# ======================================================================
def plot_world(mw: MothraWorld, *, ax=None, basemap: Optional[np.ndarray] = None,
               show_names: bool = False, show_field: bool = True,
               cmap: str = "magma", field_gamma: float = 0.45,
               title: Optional[str] = None, colorbar: bool = True,
               bounds: Optional[tuple] = None, marker_scale: float = 1.0,
               show_targets: bool = True, style: Optional[FigureStyle] = None):
    """画静态基础世界:概率场 + 28 个目标 + NED 轴。返回 Axes。

    `style`:外观参数(见 `contracts/style.py`);None = 默认样式。

    ⚠ 概率场在本场地上极为峰化(sigma=3 m 对 653x294 m,>0.5 的 cell 只占 0.1%)。
      叠在晕渲底图上时用**逐像素 alpha = p^field_gamma**,而不是整层统一半透明 ——
      后者会让 p≈0 的绝大多数格子(在 magma 里接近黑)把底图整片压暗,
      场反而看不见。alpha 只影响可见性,colorbar 读到的仍是 p 的真值。
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    st = style or DEFAULT_STYLE
    if ax is None:
        h = 9.0
        w = max(3.6, h * mw.world.y_max_m / mw.world.x_max_m + 2.2)
        _, ax = plt.subplots(figsize=(w, h))

    extent = (0.0, mw.world.y_max_m, 0.0, mw.world.x_max_m)  # (E_left,E_right,N_bot,N_top)

    if basemap is not None:
        bm = np.asarray(basemap, dtype=np.float64)
        # 格距从 extent/shape 反推,不写死 1 m —— 换分辨率的底图也不会算错晕渲。
        rgb = render_terrain_rgb(
            bm, dy_m=mw.world.x_max_m / bm.shape[0],
            dx_m=mw.world.y_max_m / bm.shape[1], rows_north_up=False,
            strength=st.basemap_shade_strength, vert_exag=st.basemap_vert_exag,
            azdeg=st.basemap_sun_azimuth_deg, altdeg=st.basemap_sun_altitude_deg,
            cmap_lo=st.basemap_cmap_lo, cmap_hi=st.basemap_cmap_hi,
            norm_mode=st.basemap_norm, lighten=st.basemap_lighten,
            colors=(GRAY_RAMP if st.basemap_mode == "gray" else HAXBY))
        ax.imshow(rgb, origin="lower", extent=extent,
                  interpolation=st.basemap_interpolation, zorder=0)

    if show_field:
        mappable = cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap)
        if basemap is None:
            ax.imshow(mw.field, origin="lower", extent=extent, cmap=cmap,
                      vmin=0.0, vmax=1.0, interpolation="nearest", zorder=1)
        else:
            rgba = mappable.to_rgba(mw.field)
            rgba[..., 3] = np.clip(mw.field, 0.0, 1.0) ** field_gamma
            ax.imshow(rgba, origin="lower", extent=extent,
                      interpolation="nearest", zorder=1)
        if colorbar:
            cb = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("目標分布確率 p(x,y)", fontsize=st.font_size_label)
            cb.ax.tick_params(labelsize=st.font_size_tick)

    tg = mw.dataset.targets_ned
    if show_targets:
        type_style = st.type_style
        size = st.target_marker_size * st.target_marker_scale * marker_scale
        for t in sorted(set(mw.dataset.type_.tolist())):
            m = mw.dataset.type_ == t
            marker, color = type_style.get(t, _TYPE_FALLBACK)
            ax.scatter(tg[m, 1], tg[m, 0], s=size, marker=marker,
                       facecolor=color, edgecolor=st.marker_edge_color, linewidth=1.0,
                       zorder=5, label=f"{t} ({int(m.sum())})")

    if show_names and show_targets:
        # 只标"复合体"级别的名字(冒号前),每个复合体标一次 ——
        # 28 个点里有 6 个复合体,逐点标注会糊成一团。
        seen = set()
        for (x_n, y_e), nm in zip(tg, mw.dataset.name):
            if not nm:
                continue
            group = nm.split(":")[0]
            if group in seen:
                continue
            seen.add(group)
            ax.annotate(group, (y_e, x_n), textcoords="offset points",
                        xytext=(9, 4), fontsize=st.font_size_annotation,
                        color="#ffffff", zorder=6, path_effects=_text_halo())

    if bounds is None:
        ax.set_xlim(0.0, mw.world.y_max_m)
        ax.set_ylim(0.0, mw.world.x_max_m)
    else:
        n0, n1, e0, e1 = bounds
        ax.set_xlim(e0, e1)
        ax.set_ylim(n0, n1)
    ax.set_aspect("equal")
    ax.set_xlabel("East  y_E [m]", fontsize=st.font_size_label)
    ax.set_ylabel("North  x_N [m]", fontsize=st.font_size_label)
    if title != "":
        ax.set_title(title or "Mothra 熱水噴出域 — 静的基礎環境 (NED)",
                     fontsize=st.font_size_title)
    return ax


def plot_routes(ax, routes: Iterable[Sequence], *, starts: Optional[Sequence] = None,
                colors: Optional[Sequence[str]] = None, labels: Optional[Sequence[str]] = None):
    """叠加多 AUV 航线。routes[i] = [[x_N, y_E], ...](NED,与目标同帧)。

    给下一步放入 multi-AUV 后的结果出图与回放留的位;本步不产生 route。
    """
    default = ["#3987e5", "#eb6834", "#2aa198", "#b48ead"]
    for i, r in enumerate(routes):
        pts = np.asarray(list(r), dtype=np.float64).reshape(-1, 2)
        if starts is not None:
            pts = np.vstack([np.asarray(starts[i], dtype=np.float64).reshape(1, 2), pts])
        c = (colors[i] if colors else default[i % len(default)])
        ax.plot(pts[:, 1], pts[:, 0], "-", color=c, linewidth=1.6, zorder=4,
                label=(labels[i] if labels else f"AUV{i}"))
        ax.plot(pts[0, 1], pts[0, 0], "*", color=c, markersize=13,
                markeredgecolor=_MARKER_EDGE, zorder=6)
    return ax


def plot_leader_track(ax, track: Sequence, *, color: str = "#3987e5",
                      label: str = "Leader lawnmower", linewidth: float = 2.0,
                      markersize: float = 7.0):
    """叠加 Leader 测线(端点序列 [[x_N, y_E], ...])。"""
    pts = np.asarray(list(track), dtype=np.float64).reshape(-1, 2)
    ax.plot(pts[:, 1], pts[:, 0], "-", color=color, linewidth=linewidth, zorder=4,
            label=label, path_effects=_line_halo())
    ax.plot(pts[0, 1], pts[0, 0], "o", color=color, markersize=markersize,
            markeredgecolor=_MARKER_EDGE, zorder=6)
    return ax


def plot_window(ax, snapshot, *, color: str = "#2aa198", alpha: float = 0.10,
                label: Optional[str] = None, inset_m: float = 1.5,
                linewidth: float = 1.8):
    """叠加一个窗口快照的矩形(沿 North 的条带,已按场地下界截断)。

    以**虚线描边 + 极淡填色**画:窗口宽 == 场地宽时,几个窗口铺满整图,
    若用实心半透明填色会被误读成底图自身的色带。`inset_m` 让相邻窗口的边不重合。
    """
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Rectangle

    n0, n1 = snapshot.north_span
    e_c = float(snapshot.leader_pose[1])
    half = snapshot.region.width_m / 2.0
    # 不传 alpha= —— artist 级 alpha 会盖掉颜色自带的 alpha,面与边就没法分别控制。
    rect = Rectangle((e_c - half + inset_m, n0 + inset_m),
                     snapshot.region.width_m - 2 * inset_m,
                     max(0.0, (n1 - n0) - 2 * inset_m),
                     facecolor=(*to_rgb(color), alpha), edgecolor=color,
                     linewidth=linewidth, linestyle="--", zorder=3, label=label)
    ax.add_patch(rect)
    return rect


# ======================================================================
# 任务时序快照 —— "Leader 窗口揭示地图"这条语义的可视化
# ======================================================================
def sweep_times(mw: MothraWorld, timeline: Mapping[str, Any]) -> np.ndarray:
    """每个目标**首次进入 Leader 窗口**的时刻;从未进过为 nan。

    为什么需要它:`project_summary.md` §3 里 Leader 是**地图揭示器** ——
    没被窗口扫到的目标,系统还不知道它存在。出图时把它画出来等于泄露 ground truth,
    所以快照只画已揭示的。

    揭示是**累积**的:窗口滑过去之后目标仍算已揭示,故下游用 `swept <= t` 判定,
    而不是"此刻是否在窗口内"。

    判定公式与仿真主循环用来过滤 VRP 目标池的
    `msim.geometry.window._local_coords` / `window_contains_world` **逐式相同**
    (这里按 (T,J) 向量化,避免出图脚本另撑一套可能悄悄跑偏的几何):
    `s = d·back, t = d·lat, back=(-cosψ,-sinψ), lat=(-sinψ,cosψ)`,
    `inside ⟺ 0<=s<=look_back_m 且 |t|<=width_m/2`。

    Leader 的 East/航向按 `timeline["leader_east_m"/"leader_psi"]` 逐拍取 ——
    多车道(boustrophedon)起两者都不是常量;旧存档没有这两列时退回"单车道、
    East 恒为场地中线"的旧近似(仅影响用旧 npz 重出图,数值不变)。

    纯粹从已有记录推导,不需要重跑任务(04)。
    """
    tg = np.asarray(mw.dataset.targets_ned, dtype=np.float64)         # (J,2)
    t_s = np.asarray(timeline["t_s"], dtype=np.float64)               # (T,)
    win_n = np.asarray(timeline["window_north"], dtype=np.float64)    # (T,2) [后沿,前沿]
    win_meta = timeline.get("meta", {}).get("window", {})
    width = float(win_meta.get("width_m", mw.world.y_max_m))
    # 直接读契约值,不要从 window_north 的首拍差反推——Leader 起点在场地边界上时
    # 窗口后沿会被截断到 0(`back_north = max(0, north - look_back_m)`),首拍差
    # 会是 0,不是真正的 look_back_m。
    look_back = float(win_meta.get("look_back_m", width))

    if "leader_east_m" in timeline and "leader_psi" in timeline:
        east = np.asarray(timeline["leader_east_m"], dtype=np.float64)     # (T,)
        psi = np.asarray(timeline["leader_psi"], dtype=np.float64)         # (T,)
        north = np.asarray(timeline["leader_north_m"], dtype=np.float64)   # (T,)
        dN = tg[None, :, 0] - north[:, None]        # (T,J)
        dE = tg[None, :, 1] - east[:, None]
        c, s_ = np.cos(psi)[:, None], np.sin(psi)[:, None]
        s = -(dN * c + dE * s_)                     # d·back, back=(-c,-s)
        t = -dN * s_ + dE * c                        # d·lat,  lat =(-s, c)
        inside = (s >= -1e-9) & (s <= look_back + 1e-9) & (np.abs(t) <= width / 2.0 + 1e-9)
    else:
        east_c = mw.world.y_max_m / 2.0              # 旧存档兜底:单车道、East 恒定
        in_east = np.abs(tg[:, 1] - east_c) <= width / 2.0 + 1e-9       # (J,)
        in_north = ((tg[None, :, 0] >= win_n[:, 0:1] - 1e-9)
                    & (tg[None, :, 0] <= win_n[:, 1:2] + 1e-9))
        inside = in_north & in_east[None, :]

    out = np.full(tg.shape[0], np.nan, dtype=np.float64)
    ever = inside.any(axis=0)
    first_k = np.argmax(inside, axis=0)              # 首个 True 的下标
    out[ever] = t_s[first_k[ever]]
    return out


def plot_mission_snapshot(ax, mw: MothraWorld, timeline: Mapping[str, Any], t_s: float,
                          *, basemap: Optional[np.ndarray] = None,
                          swept: Optional[np.ndarray] = None,
                          style: Optional[FigureStyle] = None,
                          follower_colors: Optional[Sequence[str]] = None,
                          marker_scale: Optional[float] = None,
                          traj_linewidth: Optional[float] = None,
                          auv_marker_scale: Optional[float] = None):
    """任务在 t_s 时刻的单帧快照。**不画图例、标题、轴标签、任何文字**。

    目标点的三态处理(人工裁决 2026-08-09):
        未被窗口扫过          → **一个都不画**
        已扫过、尚未被确认    → **统一空心标记,不分类型**(声呐只知道有东西)
        已被 Follower 确认    → 按 chimney / mound 分实心标记(相机才分得清)

    Follower 轨迹只画到 t_s(`follower_pos[:k+1]`),这是"时序"的关键。

    样式:统一由 `style`(见 `contracts/style.py`)给;末尾几个 `*_scale` /
    `*_linewidth` 是**单点覆盖**,只在临时试参数时用 —— 传 None 即完全听 style 的。
    """
    from matplotlib.patches import Rectangle
    from matplotlib.patheffects import withStroke

    st = style or DEFAULT_STYLE
    colors = list(follower_colors or st.follower_colors)
    m_scale = st.target_marker_scale if marker_scale is None else float(marker_scale)
    traj_lw = st.traj_linewidth if traj_linewidth is None else float(traj_linewidth)
    auv_scale = st.auv_marker_scale if auv_marker_scale is None else float(auv_marker_scale)
    times = np.asarray(timeline["t_s"], dtype=np.float64)
    k = int(np.searchsorted(times, t_s + 1e-9, side="right") - 1)
    k = int(np.clip(k, 0, len(times) - 1))

    pos = np.asarray(timeline["follower_pos"], dtype=np.float64)     # (T,F,2)
    leader_north_all = np.asarray(timeline["leader_north_m"], dtype=np.float64)
    leader_n = float(leader_north_all[k])
    has_multilane = "leader_east_m" in timeline and "leader_psi" in timeline
    leader_east_all = (np.asarray(timeline["leader_east_m"], dtype=np.float64)
                       if has_multilane
                       else np.full_like(leader_north_all, mw.world.y_max_m / 2.0))
    leader_e = float(leader_east_all[k])
    leader_psi_k = float(np.asarray(timeline["leader_psi"])[k]) if has_multilane else 0.0
    visit = np.asarray(timeline["visit_time_s"], dtype=np.float64)
    if swept is None:
        swept = sweep_times(mw, timeline)

    # --- 底图(只铺地形,目标自己画) --------------------------------
    plot_world(mw, ax=ax, basemap=basemap, show_field=False, show_targets=False,
               colorbar=False, title="", style=st)
    ax.set_xlabel("")
    ax.set_ylabel("")

    win_meta = timeline.get("meta", {}).get("window", {})
    width = float(win_meta.get("width_m", mw.world.y_max_m))
    look_back = float(win_meta.get("look_back_m", width))

    # --- Leader 已走过的测线(多车道往复;单车道退化为一条竖直线,与旧版
    #     画法一致)+ 当前窗口 + Leader 位置 -------------------------------
    ax.plot(leader_east_all[:k + 1], leader_north_all[:k + 1], "-",
           color=st.leader_color, linewidth=st.track_linewidth, alpha=0.85, zorder=3)

    # 窗口矩形:按 Leader 瞬时航向 psi 用 get_window 同款局部系(s=沿后方, t=横向)
    # 反算世界坐标下的 4 个角。lawnmower 折线的 psi 恒为基本方向(0/±90/180度),
    # 故 back/lat 恰好落在世界坐标轴上, 4 角的外接框就是精确窗口, 不是近似。
    c_, s_ = round(np.cos(leader_psi_k)), round(np.sin(leader_psi_k))
    back = (-c_, -s_)
    lat = (-s_, c_)
    corners = [(leader_n + s * back[0] + t * lat[0], leader_e + s * back[1] + t * lat[1])
              for s in (0.0, look_back) for t in (-width / 2.0, width / 2.0)]
    ns = [p[0] for p in corners]
    es = [p[1] for p in corners]
    n0, n1 = max(0.0, min(ns)), min(float(mw.world.x_max_m), max(ns))
    e0, e1 = min(es), max(es)
    if n1 > n0:
        from matplotlib.colors import to_rgb
        ax.add_patch(Rectangle((e0, n0), e1 - e0, n1 - n0,
                               facecolor=(*to_rgb(st.leader_color), st.window_face_alpha),
                               edgecolor=st.leader_color, linewidth=st.window_linewidth,
                               linestyle="--", zorder=2))
    ax.plot([leader_e], [leader_n], marker="v", color=st.leader_color,
            markersize=st.leader_marker_size * auv_scale,
            markeredgecolor=st.marker_edge_color,
            markeredgewidth=st.auv_marker_edge_width, zorder=7)

    # --- Follower 轨迹(只到 t)+ 当前位置 ----------------------------
    # 彩色测深底图上,蓝色 Follower 会淹没在深蓝的轴部裂谷里 ⇒ 给轨迹加深色描边。
    halo = ([withStroke(linewidth=traj_lw + st.traj_halo_linewidth,
                        foreground=st.traj_halo_color)]
            if st.traj_halo_linewidth > 0 else None)
    for i in range(pos.shape[1]):
        c = colors[i % len(colors)]
        ax.plot(pos[:k + 1, i, 1], pos[:k + 1, i, 0], "-", color=c,
                linewidth=traj_lw, alpha=0.95, zorder=5,
                solid_joinstyle="round", solid_capstyle="round",
                path_effects=halo)
        ax.plot([pos[k, i, 1]], [pos[k, i, 0]], marker="o", color=c,
                markersize=st.follower_marker_size * auv_scale,
                markeredgecolor=st.marker_edge_color,
                markeredgewidth=st.auv_marker_edge_width, zorder=8)

    # --- 目标:未揭示的不画 -------------------------------------------
    tg = mw.dataset.targets_ned
    revealed = ~np.isnan(swept) & (swept <= t_s + 1e-9)
    observed = ~np.isnan(visit) & (visit <= t_s + 1e-9)

    size = st.target_marker_size * m_scale

    # 未确认:**一个 collection 画完,不分类型** —— 声呐初查分不出 chimney/mound
    todo = revealed & ~observed
    if todo.any():
        ax.scatter(tg[todo, 1], tg[todo, 0], s=size,
                   marker=UNCONFIRMED_MARKER, facecolor="none",
                   edgecolor=st.unconfirmed_color, linewidth=st.unconfirmed_linewidth,
                   zorder=6,
                   path_effects=[withStroke(linewidth=st.unconfirmed_halo_linewidth,
                                            foreground=st.unconfirmed_halo_color)])

    # 已确认:相机看清了,才按类型分实心标记
    type_style = st.type_style
    for t_name in sorted(set(mw.dataset.type_.tolist())):
        done = (mw.dataset.type_ == t_name) & revealed & observed
        if not done.any():
            continue
        marker, color = type_style.get(t_name, _TYPE_FALLBACK)
        ax.scatter(tg[done, 1], tg[done, 0], s=size,
                   marker=marker, facecolor=color, edgecolor=st.marker_edge_color,
                   linewidth=st.confirmed_edge_linewidth, zorder=7)
    return ax


def plot_coverage(ax, coverage: np.ndarray, world_cfg, *, color: str = "#ffffff",
                  alpha: float = 0.28):
    """叠加覆盖掩膜 (nx, ny) bool,朝向同 `plot_world`。"""
    cov = np.asarray(coverage, dtype=bool)
    rgba = np.zeros(cov.shape + (4,), dtype=np.float64)
    from matplotlib.colors import to_rgb
    rgba[..., :3] = np.array(to_rgb(color))
    rgba[..., 3] = np.where(cov, alpha, 0.0)
    ax.imshow(rgba, origin="lower",
              extent=(0.0, world_cfg.y_max_m, 0.0, world_cfg.x_max_m),
              interpolation="nearest", zorder=3)
    return ax


def plot_twophase_snapshot(ax, mw: MothraWorld, timeline: Mapping[str, Any],
                           t_s: float, *,
                           basemap: Optional[np.ndarray] = None,
                           style: Optional[FigureStyle] = None,
                           routes: Optional[Sequence[Sequence[int]]] = None,
                           vehicle_colors: Optional[Sequence[str]] = None,
                           marker_scale: Optional[float] = None,
                           traj_linewidth: Optional[float] = None,
                           auv_marker_scale: Optional[float] = None,
                           show_planned_route: bool = True):
    """二段式基线 (D19) 在 t_s 时刻的单帧快照 —— **只画执行机(Follower)的轨迹**。

    与 `plot_mission_snapshot` 的三处差异全部来自二段式的**串行结构**,不是画风偏好:

    1. **不画 Leader、不画观测窗口。** 二段式的 Leader 在阶段1 就走完了整条测线,
       阶段2 一步也不动;滑动窗口这个机构在二段式里根本不存在(阶段2 一次性对
       全局目标表求解)。所以本图从头到尾只有两台 Follower。

    2. **目标没有「未揭示」这个状态。** 进入阶段2 的那一刻全部目标已探明 ——
       这正是 `twophase.py` 模块文档写明的前提(Mothra 由
       `test_survey_phase_detects_every_target` 钉成事实,其余场景靠把场地裁到
       「最后车道 + 幅宽/2」在几何上保证)。因此这里**不调用** `sweep_times`
       (二段式的记录里本来也没有 `window_north` 这一列),目标只有两态:
       未确认 = 空心,已确认 = 按 chimney / mound 分实心。
       ⚠ 调用方必须保证 `t_s >= t_survey_s`;传更早的时刻会把 ground truth
       画出来(那时 Leader 还没扫到),口径就错了。

    3. `routes` 传进来时,把**开环预先定死的路线**用细虚线垫在底下。
       这是与提案手法最核心的一处对照:二段式头一次解出来的路线一直执行到底,
       轨迹与预定路线逐点重合;提案则是随探査推进不断重解。默认开,
       `show_planned_route=False` 可关掉。

    `routes` 是 `TwoPhaseResult.routes` 的原样(每台一条,元素为
    `mw.dataset.targets_ned` 的下标),起点取 `vehicle_pos[0]`(阶段1 全程待机
    ⇒ 首拍位置就是 VRP 的起点,与 `twophase.run_twophase` 的口径逐字一致)。
    """
    from matplotlib.patheffects import withStroke

    st = style or DEFAULT_STYLE
    colors = list(vehicle_colors or st.follower_colors)
    m_scale = st.target_marker_scale if marker_scale is None else float(marker_scale)
    traj_lw = st.traj_linewidth if traj_linewidth is None else float(traj_linewidth)
    auv_scale = st.auv_marker_scale if auv_marker_scale is None else float(auv_marker_scale)

    times = np.asarray(timeline["t_s"], dtype=np.float64)
    k = int(np.searchsorted(times, t_s + 1e-9, side="right") - 1)
    k = int(np.clip(k, 0, len(times) - 1))

    pos = np.asarray(timeline["vehicle_pos"], dtype=np.float64)      # (T,V,2)
    visit = np.asarray(timeline["visit_time_s"], dtype=np.float64)
    tg = np.asarray(mw.dataset.targets_ned, dtype=np.float64)

    # --- 底图(只铺地形,目标自己画) --------------------------------
    plot_world(mw, ax=ax, basemap=basemap, show_field=False, show_targets=False,
               colorbar=False, title="", style=st)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # --- 阶段2 开环预定路线(虚线垫底) --------------------------------
    if show_planned_route and routes is not None:
        for i, r in enumerate(routes):
            idx = [int(j) for j in r]
            if not idx:
                continue
            c = colors[i % len(colors)]
            xy = np.vstack([pos[0, i][None, :], tg[idx]])            # (1+m,2)
            ax.plot(xy[:, 1], xy[:, 0], "--", color=c, linewidth=traj_lw * 0.55,
                    alpha=0.55, zorder=4)

    # --- 执行机轨迹(只到 t)+ 当前位置 --------------------------------
    # 彩色测深底图上,蓝色轨迹会淹没在深蓝的轴部裂谷里 ⇒ 给轨迹加深色描边
    # (与 `plot_mission_snapshot` 同一处理,两张图并排看时线宽观感才一致)。
    halo = ([withStroke(linewidth=traj_lw + st.traj_halo_linewidth,
                        foreground=st.traj_halo_color)]
            if st.traj_halo_linewidth > 0 else None)
    for i in range(pos.shape[1]):
        c = colors[i % len(colors)]
        ax.plot(pos[:k + 1, i, 1], pos[:k + 1, i, 0], "-", color=c,
                linewidth=traj_lw, alpha=0.95, zorder=5,
                solid_joinstyle="round", solid_capstyle="round",
                path_effects=halo)
        ax.plot([pos[k, i, 1]], [pos[k, i, 0]], marker="o", color=c,
                markersize=st.follower_marker_size * auv_scale,
                markeredgecolor=st.marker_edge_color,
                markeredgewidth=st.auv_marker_edge_width, zorder=8)

    # --- 目标:阶段2 全部已揭示,只分「未确认 / 已确认」两态 -------------
    observed = ~np.isnan(visit) & (visit <= t_s + 1e-9)
    size = st.target_marker_size * m_scale

    if (~observed).any():
        ax.scatter(tg[~observed, 1], tg[~observed, 0], s=size,
                   marker=UNCONFIRMED_MARKER, facecolor="none",
                   edgecolor=st.unconfirmed_color, linewidth=st.unconfirmed_linewidth,
                   zorder=6,
                   path_effects=[withStroke(linewidth=st.unconfirmed_halo_linewidth,
                                            foreground=st.unconfirmed_halo_color)])

    type_style = st.type_style
    for t_name in sorted(set(mw.dataset.type_.tolist())):
        done = (mw.dataset.type_ == t_name) & observed
        if not done.any():
            continue
        marker, color = type_style.get(t_name, _TYPE_FALLBACK)
        ax.scatter(tg[done, 1], tg[done, 0], s=size,
                   marker=marker, facecolor=color, edgecolor=st.marker_edge_color,
                   linewidth=st.confirmed_edge_linewidth, zorder=7)
    return ax
