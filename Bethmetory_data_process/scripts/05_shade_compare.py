#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05 — 渲染参数对比表
================================================================================
并排渲染不同参数, 供挑选数值。两种模式:

  --what shade   光照强度对比。取一块有陡坎也有微地形的方形子区, 免得整幅太瘦。
                 -> figures/mothra_shade_compare.png
  --what color   色阶截断 / 归一化对比。用整幅, 因为问题出在西北角最浅处。
                 -> figures/mothra_color_compare.png

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe scripts/05_shade_compare.py --what color
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_m = import_module("04_mothra_plot")
render_rgb = _m.render_rgb

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "outputs"
FIGDIR = ROOT / "figures"

# (标题, strength, vert_exag, sun_alt)
VARIANTS = [
    ("无阴影\nshade 0",                       0.00, 2.0, 45.0),
    ("很柔\nshade 0.35 · ve 0.7 · alt 60",    0.35, 0.7, 60.0),
    ("柔\nshade 0.55 · ve 1.0 · alt 55",      0.55, 1.0, 55.0),
    ("中\nshade 0.75 · ve 1.5 · alt 50",      0.75, 1.5, 50.0),
    ("当前默认\nshade 1.0 · ve 2.0 · alt 45", 1.00, 2.0, 45.0),
]

# (标题, cmap_lo, cmap_hi, norm)
COLOR_VARIANTS = [
    ("当前默认\nhaxby[0, 1] · 均衡",        0.0, 1.00, "equalize"),
    ("截橙端\nhaxby[0, 0.85] · 均衡",       0.0, 0.85, "equalize"),
    ("多截一点\nhaxby[0, 0.72] · 均衡",     0.0, 0.72, "equalize"),
    ("线性归一化\nhaxby[0, 1] · 线性",      0.0, 1.00, "linear"),
    ("截橙端 + 线性\nhaxby[0, 0.85] · 线性", 0.0, 0.85, "linear"),
]

# 子区: 取中北部一块含陡坎与微地形的方形窗 (行, 列 起点与边长, 单位=像元)
R0, C0, NR, NC = 120, 40, 200, 200


def setup_font() -> None:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
                if n in have), None)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ([cjk] if cjk else []) + ["DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,
    })


def sheet(z: np.ndarray, panels: list, title: str, footer: str, out: Path,
          panel_w: float, dx_m: float = 1.0, dy_m: float = 1.0) -> Path:
    """并排渲染一行面板。panels = [(标题, rgb 生成函数), ...]"""
    n = len(panels)
    aspect = z.shape[0] / z.shape[1]
    panel_h = panel_w * aspect
    ml, mr, mt, mb, gap = 0.12, 0.12, 1.00, 0.20, 0.14
    fw = ml + n * panel_w + (n - 1) * gap + mr
    fh = mt + panel_h + mb
    fig = plt.figure(figsize=(fw, fh), facecolor="white")

    for i, (label, make) in enumerate(panels):
        left = (ml + i * (panel_w + gap)) / fw
        ax = fig.add_axes([left, mb / fh, panel_w / fw, panel_h / fh])
        ax.imshow(make(), origin="upper", interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
            sp.set_linewidth(0.6)
        ax.set_title(label, fontsize=8.5, color="#0b0b0b", pad=6, linespacing=1.5)

    fig.text(ml / fw, 1 - 0.28 / fh, title, fontsize=12, color="#0b0b0b",
             ha="left", va="top")
    fig.text(ml / fw, 0.06 / fh, footer, fontsize=7.5, color="#898781",
             ha="left", va="bottom")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="shade", choices=["shade", "color"])
    args = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    setup_font()
    z_full = np.load(OUTDIR / "mothra_utm9n_1m.npz")["z"].astype("float64")

    if args.what == "shade":
        z = z_full[R0:R0 + NR, C0:C0 + NC]
        panels = [
            (lab, (lambda s=s, ve=ve, alt=alt: render_rgb(
                z, 1.0, 1.0, strength=s, vert_exag=ve, altdeg=alt)))
            for lab, s, ve, alt in VARIANTS
        ]
        p = sheet(z, panels,
                  f"山体阴影参数对比 — Mothra {NR}×{NC} m 子区 (UTM 1 m 格网)",
                  "shade = 阴影强度 · ve = 垂直夸张 · alt = 光源高度角(度) · "
                  "方位角固定 315° · 晕渲为乘性, 只压暗不提亮",
                  FIGDIR / "mothra_shade_compare.png", panel_w=2.5)
    else:
        z = z_full
        panels = [
            (lab, (lambda lo=lo, hi=hi, nm=nm: render_rgb(
                z, 1.0, 1.0, strength=1.0, vert_exag=2.0, altdeg=45.0,
                cmap_lo=lo, cmap_hi=hi, norm_mode=nm)))
            for lab, lo, hi, nm in COLOR_VARIANTS
        ]
        p = sheet(z, panels,
                  "色带截断 / 归一化对比 — Mothra 全区 (UTM 1 m 格网)",
                  "线性归一化下 82.78% 的面积挤在色带最深的 40% 里; 排名均衡把这段摊开, "
                  "代价是颜色与水深不再等比、且跨图不可比",
                  FIGDIR / "mothra_color_compare.png", panel_w=1.55)

    print(f"  z = {np.nanmin(z):.1f} .. {np.nanmax(z):.1f} m "
          f"(起伏 {np.nanmax(z) - np.nanmin(z):.1f} m)")
    print(f"  {p.name}  {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
