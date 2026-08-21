"""[第2b层][A4] 走廊覆盖:线段按 footprint×0.9 扩带 → 细网格覆盖格集。

实现 contracts.task.corridor_cells。验收:扩带宽度精确;覆盖格集与解析解一致(不漏/不多)。

覆盖判据(A0 冻结 DECISIONS §v1.0.2,A4-Q1):
- **中心法**:cell 中心 ∈ 走廊矩形(以线段为轴、横向 ±width_m/2 的旋转矩形)即判覆盖,
  与 window_contains_cell 中心法统一 → 全项目单一覆盖判据。
- **butt cap(平头,无端帽)**:沿线段方向的投影参数 s∈[0,L] 才算带内;
  端部不加半径为 width/2 的圆帽(契约只传横向 width_m,不含端延伸)。

退化线段(零长 L==0):以端点为圆心、半径 width/2 的圆邻域(此时无方向,butt 退化为点邻域)。
"""
from __future__ import annotations

import numpy as np

from msim.contracts.geometry import GridProtocol
from msim.contracts.types import CellSet, Waypoint

__all__ = ["corridor_cells"]


def corridor_cells(seg_start: Waypoint, seg_end: Waypoint,
                   width_m: float, grid: GridProtocol) -> CellSet:
    """把线段 [seg_start → seg_end] 按 width_m 扩成矩形带(butt cap),落栅格,返回覆盖的 cell 集合。

    判据(中心法 + butt cap):cell 中心 c 满足
      s = (c − a)·d / L  ∈ [0, L]            (沿轴投影落在线段内,无端帽)
      |横向偏移| = |(c − a) − s·d| ≤ width/2  (横向半宽内)
    其中 d = (b − a)/L 为单位方向。L==0 时退化为以 a 为圆心、半径 width/2 的圆邻域。

    width_m 由调用方传 CameraSensorModel.corridor_width_m。
    """
    a = np.asarray(seg_start, dtype=np.float64)
    b = np.asarray(seg_end, dtype=np.float64)
    half = width_m / 2.0
    eps = 1e-9  # 边界含等号容差,与 _ref 解析解一致(<= half + eps)

    seg = b - a
    seg_len2 = float(seg @ seg)

    out: CellSet = set()
    for ix in range(grid.nx):
        for iy in range(grid.ny):
            c = np.asarray(grid.cell_to_world((ix, iy)), dtype=np.float64)
            if seg_len2 == 0.0:
                # 退化线段:点邻域(半径 half)
                dist = float(np.hypot(*(c - a)))
                covered = dist <= half + eps
            else:
                # 中心法 + butt cap:投影参数 t∈[0,1] 才算带内,端部不加圆帽
                t = float((c - a) @ seg / seg_len2)
                if t < 0.0 or t > 1.0:
                    covered = False
                else:
                    proj = a + t * seg
                    dist = float(np.hypot(*(c - proj)))
                    covered = dist <= half + eps
            if covered:
                out.add((ix, iy))
    return out
