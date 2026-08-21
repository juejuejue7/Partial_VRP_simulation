"""[第0层][A0 冻结] 出图样式契约。

为什么值得单开一个契约:字号 / 线宽 / 标记大小 / 配色原本散在 `vrpsim/viz.py` 的模块
常数与 `scripts/05_plot_mission.py` 的版面常数里 —— 想把字调大一号要同时动两个文件,
而且改完没人说得清"论文里那张图当时用的是什么参数"。收敛成一个**可序列化的
dataclass** 之后:

  - 改样式 = 改一个 JSON,不碰代码(`--style my_style.json`);
  - 每张图用的样式可以随图落盘,可复现、可追溯;
  - 字段的增删改走 A0(硬纪律 2),别处不许再各调各的。

样式**只影响画面**,不影响任何数值结果 —— 换样式重出图不需要重跑 04。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

__all__ = ["FigureStyle", "DEFAULT_STYLE", "load_style", "dump_style", "FONT_FIELDS"]


@dataclass(frozen=True)
class FigureStyle:
    """一张图的全部外观参数。字段全为标量或字符串元组 ⇒ 可直接 JSON 往返。

    数值单位:`*_in` 是英寸,`*_size`/`*_linewidth` 是 matplotlib 的点(pt),
    `margin_left/right` 与 `frame_fill` 是**图幅分数**(0~1),`*_alpha` 是不透明度。

    ⚠ `target_marker_size` 用的是 `scatter(s=...)` 的口径(**面积** pt²),
      而 `leader_marker_size`/`follower_marker_size` 用的是 `plot(markersize=...)`
      的口径(**直径** pt)。两者不同源是 matplotlib 的历史包袱,不是笔误 ——
      想让点变大一倍:前者乘 4,后者乘 2。
    """

    # ================= 字体 =================
    # 名字以 `font_size_` 开头的字段会被 `scaled_fonts()` 统一缩放,加新字号请沿用该前缀。
    #
    # ⚠ 默认值**照抄接管前各处的实际取值**(base/title/label/tick 就是 matplotlib 的默认
    #   10/12/10/10),所以引入本契约不改变任何一张既有图 ——
    #   `tests/test_style.py` 与 02/03/05 的输出哈希都钉着这一点。想让图更紧凑请调这里,
    #   别去改调用点。
    font_size_base: float = 10.0         # rcParams["font.size"],未显式指定处的兜底
    font_size_tick: float = 10.0         # 一般刻度数字(02/03 的各轴、colorbar)
    font_size_tick_panel_b: float = 8.0  # 05 下排 North-时间 面板的刻度
    font_size_tick_snapshot: float = 7.0  # 05 上排快照的刻度(六格并排,要更小)
    font_size_small: float = 6.0         # 挤在格子里的小号标签(03 甘特图的 wp 名与计数)
    font_size_label: float = 10.0        # 轴标签(任务总览图按规格不画,给 02/03 用)
    font_size_title: float = 12.0
    font_size_legend: float = 8.0
    font_size_annotation: float = 7.5    # 图内文字标注(热液口名等)
    font_family: Tuple[str, ...] = ()    # 空 = 由 `use_cjk_font()` 自动挑本机可用的 CJK 字体

    # ================= 版面(仅任务总览图 05) =================
    dpi: int = 160
    fig_width_in: float = 12.0           # 图宽固定,图高由快照的等比例需求反推
    panel_b_height_in: float = 2.1       # 下排 North-时间 面板的高度
    margin_left: float = 0.055           # 绘图区左边界(图幅分数)
    margin_right: float = 0.995
    margin_bottom_in: float = 0.55
    margin_top_in: float = 0.15
    row_gap_in: float = 0.40             # 上下两排之间的间隙
    frame_fill: float = 0.88             # 每帧快照画多宽(占其时间槽的比例);
                                         # <1 才能留缝分隔相邻帧,**中心不动 ⇒ 不影响时间轴对齐**
    facecolor: str = "white"

    # ================= 配色 =================
    # 无图例的图**只能**靠颜色 + 形状认人,所以集中定义。
    # ⚠ Follower1 是紫色不是橙色:目标点已占了红(chimney)与琥珀(mound),
    #   再上橙色会在 Twin Peaks 那一簇里和 mound 撞;
    #   `tests/test_viz.py::test_follower_colors_distinct_from_target_colors` 钉着这条。
    # ⚠ 第 3 个色(绿)是给 lawnmower 对照基线的 3 台车用的(D14)。**只许往后加,
    #   前两个索引的取值不许动** —— 动了 VRP 场景的历史图就复现不了。
    leader_color: str = "#2aa198"
    follower_colors: Tuple[str, ...] = ("#3987e5", "#8b5fd6", "#4c9f38")
    chimney_color: str = "#e34948"
    mound_color: str = "#eda100"
    unconfirmed_color: str = "#ffffff"       # 已扫到未确认:白圈
    unconfirmed_halo_color: str = "#1a1a19"  # + 深色描边,亮暗底图上都跳得出来
    marker_edge_color: str = "#ffffff"
    visit_color: str = "#e34948"             # 面板 (b) 里"观测成立"的散点

    # ================= 快照(上排)的元素尺寸 =================
    target_marker_size: float = 46.0     # scatter 面积 pt²
    target_marker_scale: float = 1.0     # 目标点整体缩放(只想调目标点大小时改这个)
    unconfirmed_linewidth: float = 1.8
    unconfirmed_halo_linewidth: float = 3.8
    confirmed_edge_linewidth: float = 0.9
    leader_marker_size: float = 11.0     # markersize 直径 pt
    follower_marker_size: float = 9.0
    auv_marker_scale: float = 1.0        # 三台 AUV 标记整体缩放
    auv_marker_edge_width: float = 1.1
    traj_linewidth: float = 2.0          # Follower 轨迹
    # 轨迹描边:彩色测深底图上,蓝色 Follower 正好压在深蓝的轴部裂谷里会看不见。
    # 描边宽度 = traj_linewidth + 本值;设 0 关闭(灰度底图下可以关)。
    traj_halo_linewidth: float = 1.8
    traj_halo_color: str = "#1a1a19"
    track_linewidth: float = 1.2         # Leader 测线(快照里六帧并排,必须比单图细)
    window_linewidth: float = 1.2        # 观测窗口虚线框(同上)
    window_face_alpha: float = 0.10

    # ================= 地形底图渲染 =================
    # 与 Bethmetory_data_process/scripts/04_mothra_plot.py 同一方案(见 viz.render_terrain_rgb)。
    # 默认值逐个照抄上游的正式渲染参数。
    basemap_mode: str = "haxby"            # "haxby" = 上游测深色带 | "gray" = 中性灰(底色不抢戏)
    basemap_norm: str = "equalize"         # "equalize" = 排名直方图均衡 | "linear" = 与水深等比
    basemap_shade_strength: float = 1.0    # 0 = 不压暗,1 = 全量(乘性下限 0.45)
    basemap_vert_exag: float = 2.0         # 垂直夸张,越小光照越柔
    basemap_sun_azimuth_deg: float = 315.0  # 西北高照,制图惯例
    basemap_sun_altitude_deg: float = 45.0  # 越大阴影越浅
    basemap_cmap_lo: float = 0.0           # 色带截断:调高截掉深蓝端
    basemap_cmap_hi: float = 1.0           # 调低截掉最暖的橙端(避免与 mound 琥珀标记撞色)
    basemap_interpolation: str = "bilinear"
    basemap_lighten: float = 0.0           # 朝白色提亮:0 = 上游原样,0.3~0.5 = 底图退居背景
                                           # 彩色底图会抢戏,这是保留地形色又让上层跳出来的旋钮

    # ================= 单张大图(02 世界图 / 03 窗口诊断图)专用 =================
    # 与上面的快照版**不共用**:六帧并排的细线放到单图里会看不见,反之亦然。
    overview_track_linewidth: float = 2.0    # Leader 测线
    overview_window_linewidth: float = 1.8   # 窗口矩形描边

    # ================= 面板 (b) 的元素尺寸 =================
    panel_b_leader_linewidth: float = 2.4
    panel_b_follower_linewidth: float = 1.9
    panel_b_window_alpha: float = 0.16
    panel_b_visit_marker_size: float = 26.0
    panel_b_visit_edge_width: float = 0.6
    panel_b_grid_alpha: float = 0.25
    panel_b_grid_linewidth: float = 0.6
    time_guide_linewidth: float = 0.7    # (b) 上标记各快照时刻的竖直参考线
    time_guide_alpha: float = 0.55
    time_guide_color: str = "#898781"

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """枚举型字段只认白名单里的取值。

        数值字段写错还看得出来(图明显不对),字符串字段写错(`"heqby"`)会静默
        走到 matplotlib 深处才报一句莫名其妙的错 —— 在入口拦掉。
        """
        for name, allowed in _ENUM_FIELDS.items():
            v = getattr(self, name)
            if v not in allowed:
                raise ValueError(
                    f"{name} 只能取 {sorted(allowed)},收到 {v!r}")

    # ------------------------------------------------------------------
    # 派生
    # ------------------------------------------------------------------
    @property
    def type_style(self) -> Dict[str, Tuple[str, str]]:
        """目标类型 → (marker, color)。形状 + 颜色**双重编码**,识别不单靠颜色。

        取自 Bethmetory_data_process/scripts/04_mothra_plot.py 的 VENT_STYLE ——
        那组配色经 Machado-Oliveira-Fernandes 色觉模拟实测过
        (红 + 黄 ΔE 20.8 / deutan 15.3,PASS)。
        """
        return {"chimney": ("o", self.chimney_color),
                "mound": ("^", self.mound_color)}

    def follower_color(self, i: int) -> str:
        return self.follower_colors[i % len(self.follower_colors)]

    # ------------------------------------------------------------------
    # 变换
    # ------------------------------------------------------------------
    def scaled_fonts(self, k: float) -> "FigureStyle":
        """所有字号乘 k,其余不动。`--font-scale 1.4` 走的就是这里。"""
        if k <= 0:
            raise ValueError(f"font_scale 必须为正,收到 {k}")
        return replace(self, **{n: getattr(self, n) * float(k) for n in FONT_FIELDS})

    def with_overrides(self, **kw: Any) -> "FigureStyle":
        """逐字段覆盖(未知字段直接报错,不静默吞掉)。"""
        _check_keys(kw)
        return replace(self, **_coerce(kw))

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FigureStyle":
        """从 dict 构造。**未知字段报错** —— 手改 JSON 打错字必须当场知道,
        否则会出一张"改了但没生效"的图,而且看不出来。
        """
        _check_keys(d)
        return cls(**_coerce(d))

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, indent=2, ensure_ascii=False)
            fp.write("\n")
        return path

    @classmethod
    def from_json(cls, path: str) -> "FigureStyle":
        with open(path, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        if not isinstance(d, dict):
            raise ValueError(f"{path} 顶层必须是一个 JSON 对象,读到 {type(d).__name__}")
        return cls.from_dict(d)

    # ------------------------------------------------------------------
    # 落到 matplotlib
    # ------------------------------------------------------------------
    def apply_rcparams(self) -> None:
        """把字号等写进 matplotlib 全局 rcParams。

        覆盖面:没有显式传 `fontsize=` 的地方(02/03 的标题、图例、轴标签、刻度)
        全部跟着走。05 的两排刻度另有显式字号(六帧并排必须更小),由本契约的
        `font_size_tick_snapshot` / `font_size_tick_panel_b` 单独控制,不走这里。

        ⚠ 故意**不设** `figure.dpi`:02/03 用 `bbox_inches="tight"`,而 tight 包围盒是
          按像素量文字再折回英寸的,动了 figure.dpi 会让出图尺寸有亚像素级抖动。
          输出分辨率由 `savefig(dpi=...)` 显式给。
        """
        import matplotlib

        rc = matplotlib.rcParams
        rc["font.size"] = self.font_size_base
        rc["axes.titlesize"] = self.font_size_title
        rc["axes.labelsize"] = self.font_size_label
        rc["xtick.labelsize"] = self.font_size_tick
        rc["ytick.labelsize"] = self.font_size_tick
        rc["legend.fontsize"] = self.font_size_legend
        rc["savefig.dpi"] = self.dpi
        rc["savefig.facecolor"] = self.facecolor
        rc["axes.unicode_minus"] = False
        if self.font_family:
            chain = list(self.font_family)
            rc["font.sans-serif"] = chain + [n for n in rc.get("font.sans-serif", [])
                                             if n not in chain]


#: 枚举型字段的白名单(由 `__post_init__` 强制)
_ENUM_FIELDS: Dict[str, frozenset] = {
    "basemap_mode": frozenset({"haxby", "gray"}),
    "basemap_norm": frozenset({"equalize", "linear"}),
    "basemap_interpolation": frozenset({"nearest", "bilinear", "bicubic",
                                        "antialiased", "none"}),
}

DEFAULT_STYLE = FigureStyle()

#: 会被 `scaled_fonts()` 统一缩放的字段(靠前缀自动认,加新字号无需改这里)
FONT_FIELDS: Tuple[str, ...] = tuple(f.name for f in fields(FigureStyle)
                                     if f.name.startswith("font_size_"))

#: 需要从 JSON 的 list 还原成 tuple 的字段(靠默认值类型自动认)
_TUPLE_FIELDS: Tuple[str, ...] = tuple(f.name for f in fields(FigureStyle)
                                       if isinstance(f.default, tuple))
_FIELD_NAMES = frozenset(f.name for f in fields(FigureStyle))


def _check_keys(d: Mapping[str, Any]) -> None:
    bad = sorted(set(d) - _FIELD_NAMES)
    if not bad:
        return
    import difflib
    hint = []
    for k in bad:
        near = difflib.get_close_matches(k, sorted(_FIELD_NAMES), n=1)
        hint.append(f"{k}" + (f"(是不是想写 {near[0]}?)" if near else ""))
    raise ValueError("FigureStyle 没有这些字段: " + ", ".join(hint)
                     + "\n可用字段见 vrpsim/contracts/style.py;"
                       "新增字段须经 A0 裁决(硬纪律 2)。")


def _coerce(d: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    for k in _TUPLE_FIELDS:
        if k in out and out[k] is not None:
            out[k] = tuple(out[k])
    return out


def load_style(path: Optional[str] = None, *, font_scale: float = 1.0,
               **overrides: Any) -> FigureStyle:
    """取样式:JSON 文件(可选)→ 逐字段覆盖 → 字号统一缩放。

    `path` 为 None / 空 / 文件不存在时用默认样式(不报错 —— 样式是可选项);
    但**路径给了却读不到**会报错,免得手误写错文件名后拿到一张默认样式的图还以为生效了。
    """
    if path:
        if not os.path.isfile(path):
            raise SystemExit(f"样式文件不存在: {path}(不传 --style 即用默认样式)")
        st = FigureStyle.from_json(path)
    else:
        st = DEFAULT_STYLE
    if overrides:
        st = st.with_overrides(**{k: v for k, v in overrides.items() if v is not None})
    if font_scale != 1.0:
        st = st.scaled_fonts(font_scale)
    return st


def dump_style(path: str, style: Optional[FigureStyle] = None) -> str:
    """把一份样式写成 JSON 模板,供人手改后用 `--style` 传回。"""
    return (style or DEFAULT_STYLE).to_json(path)
