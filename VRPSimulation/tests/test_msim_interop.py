"""互操作验收:真实世界必须能直接喂进 msim 既有链路(D1 的复用前提)。

只测"接得上",不测 msim 内部逻辑 —— 那是 msim 自己的验收范围。
"""
from __future__ import annotations

import numpy as np
import pytest

from msim.contracts.config import SensorConfig, WindowConfig, WorldConfig
from msim.contracts.env_static import GroundTruthInstance
from msim.env_static.field import field_from_targets
from msim.geometry.grid import Grid

from vrpsim.contracts.config import MothraSimConfig, mothra_world_config
from vrpsim.world import build_mothra_world


@pytest.fixture(scope="module")
def mw():
    return build_mothra_world(MothraSimConfig())


def test_world_config_is_msim_type(mw):
    assert isinstance(mw.world, WorldConfig)
    assert isinstance(mw.grid, Grid)
    assert (mw.world.x_max_m, mw.world.y_max_m) == (500.0, 100.0)
    assert mw.world.x_max_m > mw.world.y_max_m, "North 必须是长边(msim 的默认场地约定)"


def test_to_gt_instance(mw):
    inst = mw.to_gt_instance()
    assert isinstance(inst, GroundTruthInstance)
    assert len(inst.targets) == mw.dataset.n
    assert inst.field.shape == (mw.world.nx, mw.world.ny)
    assert all(t.shape == (2,) for t in inst.targets)


def test_default_field_path_is_msim_verbatim(mw):
    """默认 weight_mode='uniform' 必须与直接调 msim 的实现逐位相同(REUSE 白名单的前提)。"""
    ref = field_from_targets(mw.dataset.as_waypoint_list(), mw.world, bandwidth_m=3.0)
    assert np.array_equal(ref, mw.field)


def test_targets_map_into_grid(mw):
    """每个目标都能被 msim 的 Grid 量化进界内,且 cell 中心与原点距离 < 半个格。"""
    half = mw.world.res_m * 0.5 * np.sqrt(2.0)
    for t in mw.dataset.as_waypoint_list():
        ix, iy = mw.grid.world_to_cell(t)
        assert 0 <= ix < mw.world.nx and 0 <= iy < mw.world.ny
        assert float(np.linalg.norm(mw.grid.cell_to_world((ix, iy)) - t)) <= half


def test_detections_from_field_runs(mw):
    """msim 的候选提取能在真实场上跑通,且提出来的点落在界内。

    ⚠ 只验"接得上"。候选点的口径(阈值/NMS)是 msim DECISIONS 的事,
      且 §v1.0.10 的 threshold 取值本身就在【待裁决项 4】里。
    """
    from msim.env_static.field import detections_from_field

    pts = detections_from_field(mw.field, mw.world,
                                threshold_rel=0.5, min_separation_m=3.0)
    assert len(pts) > 0
    arr = np.asarray(pts, dtype=np.float64)
    assert arr[:, 0].min() >= 0.0 and arr[:, 0].max() <= mw.world.x_max_m
    assert arr[:, 1].min() >= 0.0 and arr[:, 1].max() <= mw.world.y_max_m


def test_lawnmower_covers_field(mw):
    """Leader 测线沿 North 长边铺、沿 East 步进,且全部落在真实场地内。"""
    from msim.env_static.leader_path import lawnmower_waypoints

    wps = np.asarray(lawnmower_waypoints(mw.world, lane_spacing_m=20.0), dtype=np.float64)
    assert len(wps) >= 4
    assert wps[:, 0].min() >= 0.0 and wps[:, 0].max() <= mw.world.x_max_m
    assert wps[:, 1].min() >= 0.0 and wps[:, 1].max() <= mw.world.y_max_m
    # 每条测线是一对同 East 的端点,North 跨全场
    assert abs(float(np.abs(wps[1, 0] - wps[0, 0])) - mw.world.x_max_m) < 1.0


def test_lane_spacing_equal_east_gives_single_track(mw):
    """lane_spacing == East 跨度 ⇒ 恰好 1 条测线(D7 裁切的直接收益)。"""
    from msim.env_static.leader_path import lawnmower_waypoints

    wps = np.asarray(lawnmower_waypoints(mw.world, lane_spacing_m=100.0), dtype=np.float64)
    assert len(wps) == 2, f"应为单条测线的两个端点,得到 {len(wps)} 个点"
    assert wps[0, 1] == wps[1, 1] == 50.0            # East 中线
    assert {wps[0, 0], wps[1, 0]} == {0.0, 500.0}    # North 跨全场


def test_window_width_equals_east_extent(mw):
    """D7+D8 的自洽条件:窗口宽 == 场地 East 跨度。

    这让窗口成为"全 East 的 North 条带",Leader 单条测线就能让窗口扫过全场。
    裁到 East=100 m 正是为了配上 100 m 的窗口宽 —— 两者必须同步改,
    所以在这里钉死。(它同时也满足 msim §v1.1.5 对窗口宽的要求,虽然那条是
    为 RL 动作空间定的、本仿真不做 RL。)
    """
    cfg = MothraSimConfig()
    assert cfg.window.width_m == mw.world.y_max_m == 100.0
    assert cfg.window.look_back_m == 100.0
    assert cfg.lane_spacing_m == cfg.window.width_m, "lane_spacing 必须等于窗口宽,否则会漏扫"


def test_msim_does_not_import_vrpsim():
    """依赖方向单向(D1):msim 里不许出现对本包的 import。"""
    import os
    import re

    msim_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "msim")
    pat = re.compile(r"^\s*(from|import)\s+vrpsim\b", re.MULTILINE)
    offenders = []
    for dirpath, _, files in os.walk(msim_root):
        for fn in files:
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    if pat.search(f.read()):
                        offenders.append(p)
    assert not offenders, f"msim 反向依赖了 VRPSimulation:{offenders}"


def test_res_knob_still_builds():
    """res_m 是旋钮(待裁决项 3):换成 0.5 也要能建起来,形状随之变。"""
    w = mothra_world_config(res_m=0.5)
    assert (w.nx, w.ny) == (1000, 200)
    mw2 = build_mothra_world(MothraSimConfig(res_m=0.5))
    assert mw2.field.shape == (1000, 200)
    assert np.isclose(mw2.field.max(), 1.0)
