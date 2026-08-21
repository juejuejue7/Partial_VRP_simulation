#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配色参数 —— 全部取自 dataviz skill 的 references/palette.md, 不自行调色。"""

# 蓝色单色相序列色阶 (step 100 -> 700, 浅 -> 深)
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# 第二个序列语境用橙色 (skill: 同屏两个序列时, 第二个取下一个类别槽的色相)
SEQ_ORANGE_ANCHOR = "#eb6834"

# 暗色模式: 同一色阶自己的取步, 去掉最深的 700, 免得最深处并进暗底
SEQ_BLUE_DARK = SEQ_BLUE[:-1]

SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
PAGE = {"light": "#f9f9f7", "dark": "#0d0d0d"}
INK_PRIMARY = {"light": "#0b0b0b", "dark": "#ffffff"}
INK_SECONDARY = {"light": "#52514e", "dark": "#c3c2b7"}
INK_MUTED = {"light": "#898781", "dark": "#898781"}
GRIDLINE = {"light": "#e1e0d9", "dark": "#2c2c2a"}
AXIS = {"light": "#c3c2b7", "dark": "#383835"}

# 强调 (Mothra 框线): 类别槽 2 橙色 —— 与蓝色底图色相相反, 不会被误读成水深
ACCENT = {"light": "#eb6834", "dark": "#d95926"}
