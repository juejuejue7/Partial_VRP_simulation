#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
序列色阶校验 —— dataviz skill 的 `validate_palette.js --ordinal` 的 Python 实现。

本机没有 node, 但"配色要算不要靠眼睛"这条规矩不能省, 所以在这里用 numpy 复现:
  * sRGB -> 线性 RGB -> OKLab/OKLCH (Björn Ottosson 变换)
  * WCAG 相对亮度对比度

对 sequential (连续量级, 带 colorbar) 色阶, skill 规定的判据是
"跨色阶的亮度单调性, 而不是相邻 CVD" —— 用类别型六项检查去测序列色阶必然假失败。
故这里只测:
  1. OKLCH L 严格单调
  2. 单色相 (色相角跨度 <= HUE_TOL)
  3. 亮度跨度足够 (深浅两端可分)
对 ordinal (离散有序标记) 另加: 相邻 ΔL >= 0.06, 浅端对比度 >= 2.0:1。
"""
from __future__ import annotations

import numpy as np

HUE_TOL = 25.0        # 单色相容差 (度)
MIN_L_SPAN = 0.35     # 序列色阶最小亮度跨度
MIN_DL_ORDINAL = 0.06
MIN_CONTRAST_ORDINAL = 2.0


# ---------------------------------------------------------------- 色彩空间 --
def hex_to_srgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return m2 @ np.cbrt(m1 @ rgb)


def hex_to_oklch(h: str) -> tuple[float, float, float]:
    lab = linear_to_oklab(srgb_to_linear(hex_to_srgb(h)))
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    return L, float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) % 360.0)


def relative_luminance(h: str) -> float:
    lin = srgb_to_linear(hex_to_srgb(h))
    return float(np.dot([0.2126, 0.7152, 0.0722], lin))


def contrast(a: str, b: str) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _hue_span(hues: list[float]) -> float:
    """考虑 0/360 环绕的色相角跨度。"""
    h = np.radians(np.asarray(hues))
    mean = np.arctan2(np.sin(h).mean(), np.cos(h).mean())
    d = np.degrees(np.abs(np.angle(np.exp(1j * (h - mean)))))
    return float(d.max() - d.min()) if len(d) > 1 else 0.0


# ------------------------------------------------------------------ 检查 ----
def check_ramp(hexes: list[str], *, mode: str = "light", surface: str = "#fcfcfb",
               ordinal: bool = False, name: str = "ramp", verbose: bool = True) -> bool:
    """返回 True = 无硬失败。"""
    lch = [hex_to_oklch(h) for h in hexes]
    Ls = [v[0] for v in lch]
    Cs = [v[1] for v in lch]
    Hs = [v[2] for v in lch]
    rows: list[tuple[str, str, str]] = []

    inc = all(b > a for a, b in zip(Ls, Ls[1:]))
    dec = all(b < a for a, b in zip(Ls, Ls[1:]))
    rows.append(("L 单调" if not ordinal else "L monotone",
                 "PASS" if (inc or dec) else "FAIL",
                 f"{'递增' if inc else '递减' if dec else '非单调'}  "
                 f"L {Ls[0]:.3f} -> {Ls[-1]:.3f}"))

    span = max(Ls) - min(Ls)
    rows.append(("L 跨度", "PASS" if span >= MIN_L_SPAN else "WARN",
                 f"{span:.3f} (>= {MIN_L_SPAN})"))

    hs = _hue_span(Hs)
    rows.append(("单色相", "PASS" if hs <= HUE_TOL else "FAIL",
                 f"色相跨度 {hs:.1f}deg (<= {HUE_TOL}), C {min(Cs):.3f}..{max(Cs):.3f}"))

    if ordinal:
        dl = [abs(b - a) for a, b in zip(Ls, Ls[1:])]
        rows.append(("相邻 ΔL", "PASS" if min(dl) >= MIN_DL_ORDINAL else "FAIL",
                     f"最小 {min(dl):.3f} (>= {MIN_DL_ORDINAL})"))
        light_end = hexes[int(np.argmax(Ls))]
        cr = contrast(light_end, surface)
        rows.append(("浅端对比度", "PASS" if cr >= MIN_CONTRAST_ORDINAL else "FAIL",
                     f"{light_end} vs {surface} = {cr:.2f}:1 (>= {MIN_CONTRAST_ORDINAL})"))

    ok = not any(r[1] == "FAIL" for r in rows)
    if verbose:
        kind = "ordinal" if ordinal else "sequential"
        print(f"  [ramp-check] {name}  ({kind}, mode={mode}, {len(hexes)} steps)")
        for label, verdict, detail in rows:
            print(f"    {verdict:4s}  {label:<12s} {detail}")
        print(f"    => {'无硬失败' if ok else '存在 FAIL, 必须修正'}")
    return ok


# --------------------------------------------------------- 类别型 CVD 检查 --
# Machado, Oliveira & Fernandes (2009), severity 1.0, 作用于线性 RGB。
# dataviz skill 把这个模型本身算作判据的一部分, 故照抄系数, 不换别的模型。
CVD_MATRICES = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
}
CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR = 8.0, 6.0, 15.0


def _oklab_of(rgb_lin: np.ndarray) -> np.ndarray:
    return linear_to_oklab(np.clip(rgb_lin, 0.0, 1.0))


def delta_e(a: str, b: str, cvd: str | None = None) -> float:
    """OKLab 欧氏距离 x100; cvd 为 None 时是常视觉。"""
    la, lb = srgb_to_linear(hex_to_srgb(a)), srgb_to_linear(hex_to_srgb(b))
    if cvd:
        m = CVD_MATRICES[cvd]
        la, lb = m @ la, m @ lb
    return float(np.linalg.norm(_oklab_of(la) - _oklab_of(lb)) * 100.0)


def check_categorical(hexes: list[str], labels: list[str], *,
                      backgrounds: list[tuple[str, str]] | None = None,
                      secondary_encoding: bool = False,
                      name: str = "palette", verbose: bool = True) -> bool:
    """两两检查类别色: 常视觉下限 15 (硬门), CVD 目标 8 / 下限 6
    (6-8 仅在有次级编码时合法)。backgrounds = [(说明, hex), ...] 另测对比度。"""
    ok = True
    if verbose:
        print(f"  [categorical] {name}  ({len(hexes)} 色, 全配对)")
    for i in range(len(hexes)):
        for j in range(i + 1, len(hexes)):
            a, b = hexes[i], hexes[j]
            dn = delta_e(a, b)
            dp = delta_e(a, b, "protanopia")
            dd = delta_e(a, b, "deuteranopia")
            worst = min(dp, dd)
            if dn < NORMAL_FLOOR:
                v, ok = "FAIL", False
            elif worst < CVD_FLOOR:
                v, ok = "FAIL", False
            elif worst < CVD_TARGET:
                v = "WARN" if secondary_encoding else "FAIL"
                ok = ok and secondary_encoding
            else:
                v = "PASS"
            if verbose:
                print(f"    {v:4s}  {labels[i]} vs {labels[j]}   "
                      f"常视觉 ΔE={dn:.1f} (>={NORMAL_FLOOR:g})  "
                      f"protan={dp:.1f} deutan={dd:.1f} (>={CVD_TARGET:g})")
    for k, h in enumerate(hexes):
        for bg_name, bg in (backgrounds or []):
            cr = contrast(h, bg)
            if verbose:
                print(f"    {'PASS' if cr >= 3.0 else 'WARN':4s}  "
                      f"{labels[k]} on {bg_name} {bg} = {cr:.2f}:1 (>=3.0)")
    if verbose:
        print(f"    => {'无硬失败' if ok else '存在 FAIL, 必须换色'}"
              + ("  (WARN 依赖形状等次级编码)" if secondary_encoding else ""))
    return ok


def check_text_contrast(pairs: list[tuple[str, str, str]], surface_name: str = "") -> None:
    """WCAG 文本对比度 (正文 4.5:1 / 大字 3:1)。"""
    print(f"  [text-contrast] {surface_name}")
    for label, fg, bg in pairs:
        cr = contrast(fg, bg)
        v = "PASS" if cr >= 4.5 else ("LARGE-OK" if cr >= 3.0 else "WARN")
        print(f"    {v:9s} {label:<18s} {fg} on {bg} = {cr:.2f}:1")


if __name__ == "__main__":
    from palette import SEQ_BLUE, SURFACE  # noqa: F401  (仅手动运行时)
    check_ramp(SEQ_BLUE, name="blue sequential")
