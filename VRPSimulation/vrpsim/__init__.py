"""VRPSimulation — 真实 Mothra 热液场上的多 AUV 協調探査 simulation。

分层(依赖单向自底向上,同 architecture_blueprint.md):
    contracts/   [A0 独占] 冻结的坐标系 / 数据集 / 配置契约
    geodesy.py   [第0层] 经纬度 <-> UTM 9N <-> NED(纯 numpy)
    dataset.py   [第0层] waypoint CSV -> NED 目标表
    field.py     [第1层] 目標分布確率マップ p(x,y)
    world.py     [第1层] MothraWorld 静态基础世界(含 D7 裁切)
    windows.py   [第1层] Leader 测线 + 后方滑动观测窗口(D8,静态几何)
    planner.py   [第2层] Leader 侧规划:选池 + min-max VRP + 截断 + 投影补点
    agents.py    [第2层] Leader / Follower 运动学状态机
    mission.py   [第3层] 任务主循环 + 逐时刻记录(D10)
    lawnmower.py [第3层] 对照基线:3 台 Follower 分区全覆盖扫描(D14)
    viz.py       [第1层] 可视化与回放(样式取自 contracts/style.py,D12)

依赖方向:VRPSimulation → msim(单向)。msim 永不 import 本包。
可复用的 msim 符号白名单见 `contracts/REUSE_FROM_MSIM.md`。
"""
from __future__ import annotations

__version__ = "0.1.0"

from .contracts.config import (CropConfig, FieldBuildConfig, MothraSimConfig,
                               full_raster_crop, mothra_window_config,
                               mothra_world_config)
from .contracts.dataset import MothraDataset
from .contracts.frames import NEDFrame
from .dataset import load_waypoints
from .field import build_probability_field
from .contracts.lawnmower import LawnmowerConfig
from .contracts.mission import (Assignment, FollowerRequest, MissionConfig,
                                PlanRound)
from .contracts.style import DEFAULT_STYLE, FigureStyle, dump_style, load_style
from .lawnmower import (LawnmowerResult, build_vehicle_plans, lane_easts,
                        load_lawn_result, run_lawnmower, save_lawn_result,
                        strip_bounds)
from .mission import (MissionResult, feasibility_estimate, load_result, run_mission,
                      save_result)
from .windows import (WindowSnapshot, enumerate_windows, first_seen_window,
                      leader_track, window_anchors, window_occupancy)
from .world import MothraWorld, build_mothra_world, load_world, save_world

__all__ = [
    "MothraSimConfig", "FieldBuildConfig", "CropConfig", "full_raster_crop",
    "mothra_world_config", "mothra_window_config",
    "MothraDataset", "NEDFrame",
    "load_waypoints", "build_probability_field",
    "MothraWorld", "build_mothra_world", "save_world", "load_world",
    "WindowSnapshot", "leader_track", "window_anchors", "enumerate_windows",
    "window_occupancy", "first_seen_window",
    "MissionConfig", "Assignment", "FollowerRequest", "PlanRound", "MissionResult",
    "run_mission", "feasibility_estimate", "save_result", "load_result",
    "LawnmowerConfig", "LawnmowerResult", "run_lawnmower",
    "save_lawn_result", "load_lawn_result",
    "strip_bounds", "lane_easts", "build_vehicle_plans",
    "FigureStyle", "DEFAULT_STYLE", "load_style", "dump_style",
]
