"""[A1 验收测试 · §v1.0.3 耦合实例] env_static —— 目标点=绝对 ground truth,先验场由其 KDE 派生。

判据来源:contracts/env_static.py(GroundTruthInstance / field_from_targets /
make_clustered_instance / save_instance / load_instance)、DECISIONS §v1.0.3
(人类裁定 2026-06-12)、project_summary §13(簇结构,避免均匀随机)、CLAUDE.md
(不 overclaim:目标点是确定的真值,场严格由其派生)。

核心契约(§v1.0.3 冻结):
- **真实目标点 = 绝对 ground truth**,由簇点过程生成(簇中心 + 簇内高斯散布)。
- **先验场 = 目标点各向同性高斯 KDE 归一化到 [0,1]**;场严格由目标点派生(非独立流)。
- ★耦合不变量★ mean(field@target cells) ≫ mean(field@随机非目标 cells)——本轮核心判据。
- save/load 逐位复现(Q19)。

实现一律从 msim.env_static.field import,不从 contracts import(契约层是 NotImplementedError)。
轴序 field[ix,iy](第0轴=North=x),与 contracts/types.py 一致。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from msim.contracts.config import WorldConfig
from msim.contracts.env_static import GroundTruthInstance
from msim.env_static.field import (
    field_from_targets,
    load_instance,
    make_clustered_instance,
    save_instance,
)
from msim.geometry.grid import Grid


@pytest.fixture
def world() -> WorldConfig:
    # 足够大的场地,让簇可分、KDE 峰值集中在 target 处而非弥散到全场。
    return WorldConfig(x_max_m=100.0, y_max_m=60.0, res_m=1.0)  # nx=100, ny=60


def _target_cells(grid: Grid, targets) -> set:
    return {grid.world_to_cell(np.asarray(t, dtype=np.float64)) for t in targets}


def _field_at_cells(field: np.ndarray, cells: set) -> np.ndarray:
    return np.array([field[ix, iy] for (ix, iy) in cells], dtype=np.float64)


# ======================================================================
# field_from_targets —— 归一化、峰值共位、空→全零、复现
# ======================================================================
def test_field_from_targets_normalized_to_unit(world: WorldConfig) -> None:
    targets = [np.array([30.0, 20.0]), np.array([70.0, 40.0])]
    f = field_from_targets(targets, world, bandwidth_m=5.0)
    assert f.shape == (world.nx, world.ny), "field 轴序须为 (nx, ny)=(North,East)"
    assert f.dtype == np.float64
    assert np.all(f >= 0.0) and np.all(f <= 1.0), "先验场须归一化到 [0,1]"
    assert float(f.max()) == pytest.approx(1.0, abs=1e-9), "归一化后峰值须为 1.0"


def test_field_from_targets_peak_at_target(world: WorldConfig) -> None:
    """峰值与目标共位:每个 target 所在 cell 的 field 值显著高于全场均值。"""
    grid = Grid(world)
    targets = [np.array([25.0, 15.0]), np.array([75.0, 45.0])]
    bw = 4.0
    f = field_from_targets(targets, world, bandwidth_m=bw)
    mean_all = float(f.mean())
    for t in targets:
        ix, iy = grid.world_to_cell(t)
        assert f[ix, iy] > mean_all * 3.0, (
            f"target {t} 所在 cell 的 field 值 {f[ix, iy]:.4f} 未显著高于场均值 {mean_all:.4f}"
        )


def test_field_from_targets_empty_is_all_zero(world: WorldConfig) -> None:
    f = field_from_targets([], world, bandwidth_m=5.0)
    assert f.shape == (world.nx, world.ny)
    assert f.dtype == np.float64
    assert np.all(f == 0.0), "空 targets 须返回全零场"


def test_field_from_targets_reproducible(world: WorldConfig) -> None:
    """纯几何确定:同 targets + 同 bandwidth → 逐位相同 field。"""
    targets = [np.array([40.0, 30.0]), np.array([60.0, 20.0]), np.array([20.0, 50.0])]
    f1 = field_from_targets(targets, world, bandwidth_m=6.0)
    f2 = field_from_targets(targets, world, bandwidth_m=6.0)
    np.testing.assert_array_equal(f1, f2, "同输入须逐位复现")


def test_field_from_targets_wider_bandwidth_more_diffuse(world: WorldConfig) -> None:
    """带宽即先验不确定性:更大 σ → 场更模糊(峰值处相对集中度下降,即 std 占比降低)。

    判据不依赖绝对值:窄带宽的场应比宽带宽的场更"尖"(归一化后非零质量更集中)。
    用过半幅阈值覆盖面积衡量:窄带宽覆盖面积 ≤ 宽带宽覆盖面积。
    """
    targets = [np.array([50.0, 30.0])]
    f_narrow = field_from_targets(targets, world, bandwidth_m=2.0)
    f_wide = field_from_targets(targets, world, bandwidth_m=10.0)
    area_narrow = int(np.count_nonzero(f_narrow >= 0.5))
    area_wide = int(np.count_nonzero(f_wide >= 0.5))
    assert area_narrow <= area_wide, "更大带宽应让先验更弥散(过半幅覆盖面积不减)"


# ======================================================================
# make_clustered_instance —— 计数、簇结构、★耦合不变量★、复现
# ======================================================================
def test_instance_target_count(world: WorldConfig) -> None:
    inst = make_clustered_instance(
        world, np.random.default_rng(1),
        n_clusters=3, targets_per_cluster=4, intra_spread_m=3.0,
    )
    assert isinstance(inst, GroundTruthInstance)
    assert len(inst.targets) == 3 * 4, "目标点数须 = n_clusters × targets_per_cluster"
    assert inst.field.shape == (world.nx, world.ny)
    assert np.all(inst.field >= 0.0) and np.all(inst.field <= 1.0)


def test_instance_targets_are_clustered_not_uniform(world: WorldConfig) -> None:
    """簇结构:同簇内点间距 ≪ 跨簇点间距(非均匀随机散布)。"""
    n_clusters, tpc, spread = 3, 5, 3.0
    inst = make_clustered_instance(
        world, np.random.default_rng(2),
        n_clusters=n_clusters, targets_per_cluster=tpc, intra_spread_m=spread,
    )
    pts = np.asarray([np.asarray(t, dtype=np.float64) for t in inst.targets])
    # 按生成顺序切片成簇(契约:逐簇连续采样 targets_per_cluster 个)。
    clusters = [pts[i * tpc:(i + 1) * tpc] for i in range(n_clusters)]

    def mean_pairwise(a: np.ndarray) -> float:
        ds = [float(np.hypot(*(a[i] - a[j])))
              for i in range(len(a)) for j in range(i + 1, len(a))]
        return float(np.mean(ds)) if ds else 0.0

    intra = float(np.mean([mean_pairwise(c) for c in clusters]))
    centers = np.asarray([c.mean(axis=0) for c in clusters])
    inter = mean_pairwise(centers)
    assert intra < inter, f"簇内平均距 {intra:.2f} 应远小于簇间中心距 {inter:.2f}(呈簇结构)"


def test_instance_field_couples_to_targets(world: WorldConfig) -> None:
    """★本轮核心耦合不变量★:mean(field@target cells) ≫ mean(field@随机非目标 cells)。

    证明先验场严格由目标点派生(场跟随目标),而非独立/均匀流。
    """
    grid = Grid(world)
    rng = np.random.default_rng(42)
    inst = make_clustered_instance(
        world, rng, n_clusters=4, targets_per_cluster=4, intra_spread_m=3.0,
    )
    tgt_cells = _target_cells(grid, inst.targets)
    mean_at_targets = float(_field_at_cells(inst.field, tgt_cells).mean())

    # 采样一批随机非目标 cell 作对照。
    sampler = np.random.default_rng(999)
    non_target = []
    while len(non_target) < 200:
        ix = int(sampler.integers(0, world.nx))
        iy = int(sampler.integers(0, world.ny))
        if (ix, iy) not in tgt_cells:
            non_target.append((ix, iy))
    mean_at_random = float(_field_at_cells(inst.field, set(non_target)).mean())

    assert mean_at_targets > 5.0 * (mean_at_random + 1e-12), (
        f"耦合不变量被破坏:target cells 均值 {mean_at_targets:.4f} 未远高于"
        f"随机非目标 cells 均值 {mean_at_random:.4f}"
    )


def test_instance_field_is_kde_of_targets(world: WorldConfig) -> None:
    """场严格由目标派生:inst.field 应等于对 inst.targets 做 field_from_targets 的结果。

    bandwidth 默认 = intra_spread_m(§v1.0.3 裁定)。
    """
    spread = 3.0
    inst = make_clustered_instance(
        world, np.random.default_rng(7),
        n_clusters=3, targets_per_cluster=4, intra_spread_m=spread,
    )
    f_derived = field_from_targets(inst.targets, world, bandwidth_m=spread)
    np.testing.assert_array_equal(
        inst.field, f_derived,
        "inst.field 须 = field_from_targets(targets, bandwidth=intra_spread)(场由目标派生)",
    )


def test_instance_reproducible_same_seed_and_params(world: WorldConfig) -> None:
    """同 seed + 同参数 → (targets, field) 逐位复现。"""
    kw = dict(n_clusters=3, targets_per_cluster=4, intra_spread_m=3.0)
    a = make_clustered_instance(world, np.random.default_rng(13), **kw)
    b = make_clustered_instance(world, np.random.default_rng(13), **kw)
    ta = np.asarray([np.asarray(t, dtype=np.float64) for t in a.targets])
    tb = np.asarray([np.asarray(t, dtype=np.float64) for t in b.targets])
    np.testing.assert_array_equal(ta, tb, "同 seed+参数 targets 须逐位复现")
    np.testing.assert_array_equal(a.field, b.field, "同 seed+参数 field 须逐位复现")


def test_instance_targets_in_bounds(world: WorldConfig) -> None:
    """目标点 clip 到场内(契约步骤1)。"""
    inst = make_clustered_instance(
        world, np.random.default_rng(5),
        n_clusters=4, targets_per_cluster=4, intra_spread_m=8.0,
    )
    for t in inst.targets:
        x, y = float(t[0]), float(t[1])
        assert 0.0 <= x <= world.x_max_m, f"目标 North={x} 越界"
        assert 0.0 <= y <= world.y_max_m, f"目标 East={y} 越界"


# ----------------------------------------------------------------------
# 回归(§v1.0.3 集成发现):边界 clip 曾把溢出点压成重合/贴边点。
# 易触发参数:簇少(n=2)、每簇点多、intra_spread 大(8m)且场地小(20×10)→
# 高斯尾部大量溢出边界。生成器须拒绝重采样(distinct + 严格内部),不得简单 clip。
# 扫多 seed 以稳健逼出该场景(单 seed 可能偶然不触发)。
# ----------------------------------------------------------------------
@pytest.fixture
def small_world() -> WorldConfig:
    # 小场地放大边界溢出概率,逼出曾经的 clip 重合 bug。
    return WorldConfig(x_max_m=20.0, y_max_m=10.0, res_m=1.0)


def test_instance_targets_are_distinct(small_world: WorldConfig) -> None:
    """① 两两不重合:clip 不得把多个溢出点压到同一坐标(须拒绝重采样)。"""
    n_clusters, tpc, spread = 2, 6, 8.0
    expected = n_clusters * tpc
    for seed in range(20):  # 多 seed 稳健触发
        inst = make_clustered_instance(
            small_world, np.random.default_rng(seed),
            n_clusters=n_clusters, targets_per_cluster=tpc, intra_spread_m=spread,
        )
        keys = {tuple(np.round(np.asarray(t, dtype=np.float64), 6)) for t in inst.targets}
        assert len(keys) == expected, (
            f"seed={seed}:存在重合目标点(去重后 {len(keys)} < 期望 {expected});"
            f"边界 clip 不得产生重合点,须拒绝重采样"
        )


def test_instance_targets_strictly_interior(small_world: WorldConfig) -> None:
    """② 严格在场内:0 < x < x_max 且 0 < y < y_max(不贴边 0/边界)。"""
    n_clusters, tpc, spread = 2, 6, 8.0
    for seed in range(20):
        inst = make_clustered_instance(
            small_world, np.random.default_rng(seed),
            n_clusters=n_clusters, targets_per_cluster=tpc, intra_spread_m=spread,
        )
        for t in inst.targets:
            x, y = float(t[0]), float(t[1])
            assert 0.0 < x < small_world.x_max_m, (
                f"seed={seed}:目标 North={x} 贴边/越界(须严格内部,非 clip 到边界)"
            )
            assert 0.0 < y < small_world.y_max_m, (
                f"seed={seed}:目标 East={y} 贴边/越界(须严格内部,非 clip 到边界)"
            )


# ======================================================================
# save_instance / load_instance —— targets 与 field 逐位复现(Q19)
# ======================================================================
def test_save_load_roundtrip(world: WorldConfig, tmp_path) -> None:
    inst = make_clustered_instance(
        world, np.random.default_rng(21),
        n_clusters=3, targets_per_cluster=4, intra_spread_m=3.0,
    )
    path = os.path.join(str(tmp_path), "gt_instance.npz")
    save_instance(inst, path, metadata={"seed": 21, "n_clusters": 3})
    loaded = load_instance(path)

    assert isinstance(loaded, GroundTruthInstance)
    np.testing.assert_array_equal(loaded.field, inst.field, "load 后 field 须逐位复现")

    src = np.asarray([np.asarray(t, dtype=np.float64) for t in inst.targets])
    dst = np.asarray([np.asarray(t, dtype=np.float64) for t in loaded.targets])
    assert dst.shape == src.shape, "load 后目标点数须一致"
    np.testing.assert_array_equal(dst, src, "load 后 targets 须逐位复现(绝对 ground truth)")
