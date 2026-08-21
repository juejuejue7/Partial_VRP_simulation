"""[第1层][A2] Leader lawnmower(割草机往复测线)轨迹生成。

完全事前定义、不随现场调整、与概率场解耦(测线布局不依据概率场优化)。
输出 waypoint 序列供 LeaderActor(physics)用 AUV 动力学逐点推进(Q16)。

解耦保证:本模块仅依赖 WorldConfig 几何 + lane_spacing,**不 import env_static.field**,
测线布局与目标概率分布无关(harness §4.2:lawnmower 与概率场解耦)。
"""
from __future__ import annotations

from typing import List

import numpy as np

from msim.contracts.config import WorldConfig
from msim.contracts.types import FLOAT, Waypoint


def lawnmower_waypoints(world: WorldConfig, lane_spacing_m: float) -> List[Waypoint]:
    """生成 lawnmower 测线 waypoint 序列(沿 North 长边往复,East 步进 lane_spacing_m)。

    布局(boustrophedon 蛇形往复):
    - 测线沿 North(x)长边铺设,逐条沿 East(y)步进 lane_spacing_m。
    - 第一条测线 y = lane_spacing/2,之后 y += lane_spacing,直到覆盖到 y_max。
    - 奇偶测线方向交替(x:0→x_max 与 x_max→0),相邻测线端点相连 → 连续覆盖。
    - 每条测线产出 2 个端点 waypoint [x_N, y_E]。

    返回世界坐标 waypoint(np.ndarray shape (2,))列表。与概率场无关。
    """
    if lane_spacing_m <= 0.0:
        raise ValueError(f"lane_spacing_m 须 >0,收到 {lane_spacing_m}")

    x_lo, x_hi = 0.0, float(world.x_max_m)
    waypoints: List[Waypoint] = []

    y = lane_spacing_m / 2.0
    lane = 0
    while y <= world.y_max_m:
        if lane % 2 == 0:
            start_x, end_x = x_lo, x_hi          # 偶数测线:South→North
        else:
            start_x, end_x = x_hi, x_lo          # 奇数测线:North→South
        waypoints.append(np.array([start_x, y], dtype=FLOAT))
        waypoints.append(np.array([end_x, y], dtype=FLOAT))
        y += lane_spacing_m
        lane += 1

    return waypoints
