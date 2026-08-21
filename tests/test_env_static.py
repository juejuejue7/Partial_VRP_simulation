"""[A1 验收测试 · #5 A2 地基] field(概率场 save/load 复现) + leader_path(lawnmower)。

判据来源:env_static/field.py、env_static/leader_path.py 骨架签名 + project_summary
(graded 概率场、lawnmower 与概率场解耦)+ CLAUDE.md(leader_path 与概率场解耦)。

- field:make_clustered_field 形状/范围;save→load 逐位复现(Q19 可复现 groundtruth)。
- leader_path:lawnmower 覆盖整个 North 长边、East 按 lane_spacing 步进;**不依赖概率场**
  (签名里压根没有 field 入参 → 结构性解耦,此处显式断言其确定性、与 seed/场无关)。
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

from msim.contracts.config import WorldConfig
from msim.env_static.field import load_field, make_clustered_field, save_field
from msim.env_static.leader_path import lawnmower_waypoints


@pytest.fixture
def world() -> WorldConfig:
    # 用小场地降低测试体量(契约属性按 res 推导,仍走真实公式)
    return WorldConfig(x_max_m=20.0, y_max_m=10.0, res_m=0.5)  # nx=40, ny=20


# ======================================================================
# field —— graded 概率场
# ======================================================================
def test_field_shape_and_range(world: WorldConfig) -> None:
    rng = np.random.default_rng(123)
    f = make_clustered_field(world, rng, n_clusters=3, intra_density=0.8)
    assert f.shape == (world.nx, world.ny), "field 轴序须为 (nx, ny)=(North,East)"
    assert f.dtype == np.float64
    assert np.all(f >= 0.0) and np.all(f <= 1.0), "概率场须 ∈ [0,1]"


def test_field_reproducible_same_seed(world: WorldConfig) -> None:
    f1 = make_clustered_field(world, np.random.default_rng(7), n_clusters=4, intra_density=0.5)
    f2 = make_clustered_field(world, np.random.default_rng(7), n_clusters=4, intra_density=0.5)
    np.testing.assert_array_equal(f1, f2, "同 seed 同参数须逐位复现")


def test_field_save_load_roundtrip(world: WorldConfig, tmp_path) -> None:
    f = make_clustered_field(world, np.random.default_rng(11), n_clusters=2, intra_density=0.6)
    path = os.path.join(str(tmp_path), "gt_field.npz")
    save_field(f, path, metadata={"seed": 11, "n_clusters": 2})
    loaded = load_field(path)
    assert loaded.shape == f.shape
    np.testing.assert_array_equal(loaded, f, "save→load 须逐位复现 groundtruth(Q19)")


def test_field_clusters_not_uniform(world: WorldConfig) -> None:
    # 带簇结构:概率场不应是常数(否则退化为均匀,违背"贴合聚集统计")
    f = make_clustered_field(world, np.random.default_rng(3), n_clusters=3, intra_density=0.9)
    assert float(f.std()) > 0.0, "graded 概率场不应为常数场"


# ======================================================================
# leader_path —— lawnmower 测线
# ======================================================================
def test_lawnmower_returns_waypoints(world: WorldConfig) -> None:
    wps = lawnmower_waypoints(world, lane_spacing_m=5.0)
    assert len(wps) >= 2
    for wp in wps:
        arr = np.asarray(wp, dtype=np.float64)
        assert arr.shape == (2,)


def test_lawnmower_covers_north_span(world: WorldConfig) -> None:
    # 往复测线沿 North 长边:应同时出现接近 0 和接近 x_max 的 North 端点
    wps = np.asarray([np.asarray(w, dtype=np.float64) for w in
                      lawnmower_waypoints(world, lane_spacing_m=5.0)])
    north = wps[:, 0]
    assert north.min() <= world.x_max_m * 0.1, "应触及 North 起始端"
    assert north.max() >= world.x_max_m * 0.9, "应触及 North 末端(覆盖长边)"


def test_lawnmower_east_steps_by_spacing(world: WorldConfig) -> None:
    # East 方向按 lane_spacing 步进,横跨 East 短边
    spacing = 5.0
    wps = np.asarray([np.asarray(w, dtype=np.float64) for w in
                      lawnmower_waypoints(world, lane_spacing_m=spacing)])
    east = wps[:, 1]
    east_unique = np.unique(np.round(east, 6))
    assert east_unique.size >= 2, "至少两条测线(East 步进)"
    assert east.max() - east.min() >= world.y_max_m * 0.5, "East 方向应跨越短边主体"


def test_lawnmower_decoupled_from_field(world: WorldConfig) -> None:
    """与概率场解耦(CLAUDE.md / project_summary):
    1) 签名无 field 入参(结构性解耦);2) 完全确定,与任何概率场/seed 无关。
    """
    sig = inspect.signature(lawnmower_waypoints)
    assert "field" not in sig.parameters, "lawnmower 不应接收概率场入参(须解耦)"
    # 同入参两次调用结果一致(纯几何、确定)
    a = lawnmower_waypoints(world, lane_spacing_m=4.0)
    b = lawnmower_waypoints(world, lane_spacing_m=4.0)
    aa = np.asarray([np.asarray(w) for w in a])
    bb = np.asarray([np.asarray(w) for w in b])
    np.testing.assert_array_equal(aa, bb, "lawnmower 须为确定性几何输出")
