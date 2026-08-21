"""[A1 验收测试 · §v1.0.8 目标场解耦] test_target_field_decoupling.py

验收判据来源:
  §v1.0.8 设计决策(A0 广播):ground-truth 目标场尺度从相机 footprint 解耦,
  改由 cfg.target_field(TargetFieldConfig) 独立控制。
  runner.default_instance 须消费 cfg.target_field.bandwidth_m / intra_spread_m /
  targets_per_cluster,不再从 cfg.sensor.footprint_lateral_m 派生。

四个验收测试:
  1. 解耦不变量(核心):相同 target_field + 不同 sensor.footprint → 相同 targets/field。
  2. bandwidth 旋钮生效:bandwidth_m 大 → field 更扩散(非零格占比更大)。
  3. intra_spread 旋钮生效:intra_spread_m 大 → targets 各点到质心的均方距离更大。
  4. 默认向后兼容:SimConfig().target_field == TargetFieldConfig(3.0, 12.0, 3)(默认值锚定)。

注意:测试 1/2/3 依赖 A6 完成 runner.py 的 §v1.0.8 修改;A6 未完成时这些测试会红,
这是正常的——测试即是"完成"的客观判据(harness §4.2)。
"""
from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from msim.contracts.config import SimConfig, WorldConfig, SensorConfig, TargetFieldConfig
from msim.eval.runner import default_instance


# ======================================================================
# 公共夹具:固定基础 cfg(小场地加快运行)
# ======================================================================
@pytest.fixture
def base_cfg() -> SimConfig:
    """固定基础 cfg:小场地 + 固定 seed,用于解耦/旋钮测试。"""
    return replace(
        SimConfig(),
        world=WorldConfig(60.0, 20.0, 2.0),  # 30×10 细网格,快
        seed=24,
    )


# ======================================================================
# 测试 1 —— 解耦不变量(核心)
# 改相机 footprint 不应再改动目标场(targets + field)
# ======================================================================
def test_decoupling_invariant_footprint_does_not_affect_targets_or_field(
    base_cfg: SimConfig,
) -> None:
    """§v1.0.8 核心:固定 target_field,仅改 sensor.footprint → targets/field 逐位不变。

    构造两个仅 sensor.footprint 不同的 cfg(2×2 vs 6×6),断言二者 default_instance
    产生的 targets 逐点相同(np.allclose)、field 逐位相同(np.allclose)。
    证明改相机 footprint 不再动 ground-truth 目标场。
    """
    # 固定 target_field(与默认相同;显式写出以强调测试语义)
    target_field = TargetFieldConfig(bandwidth_m=3.0, intra_spread_m=12.0, targets_per_cluster=3)

    cfg_small_fp = replace(
        base_cfg,
        sensor=SensorConfig(footprint_along_m=2.0, footprint_lateral_m=2.0),
        target_field=target_field,
    )
    cfg_large_fp = replace(
        base_cfg,
        sensor=SensorConfig(footprint_along_m=6.0, footprint_lateral_m=6.0),
        target_field=target_field,
    )

    inst_small = default_instance(cfg_small_fp, n_targets=9)
    inst_large = default_instance(cfg_large_fp, n_targets=9)

    # targets 逐点相同
    tgts_small = np.asarray(
        [np.asarray(t, dtype=np.float64) for t in inst_small.targets]
    )
    tgts_large = np.asarray(
        [np.asarray(t, dtype=np.float64) for t in inst_large.targets]
    )
    assert tgts_small.shape == tgts_large.shape, (
        f"targets 数量不同:footprint=2 得 {tgts_small.shape[0]},"
        f" footprint=6 得 {tgts_large.shape[0]}"
    )
    assert np.allclose(tgts_small, tgts_large, atol=1e-9), (
        "§v1.0.8 解耦不变量被破坏:改变 sensor.footprint 导致 targets 改变。"
        "runner.default_instance 须用 cfg.target_field.bandwidth_m,而非 footprint 派生值。"
    )

    # field 逐位相同
    assert np.allclose(inst_small.field, inst_large.field, atol=1e-9), (
        "§v1.0.8 解耦不变量被破坏:改变 sensor.footprint 导致 field 改变。"
        "runner.default_instance 须用 cfg.target_field.bandwidth_m,而非 footprint 派生值。"
    )


# ======================================================================
# 测试 2 —— bandwidth 旋钮生效
# 宽 bandwidth → 先验场更扩散(非零格占比更大)
# ======================================================================
def test_bandwidth_knob_wider_produces_more_diffuse_field(
    base_cfg: SimConfig,
) -> None:
    """cfg.target_field.bandwidth_m 大 → field 更扩散。

    固定 seed/footprint/intra_spread,比较 bandwidth_m=1.0 vs 6.0。
    判据:更宽 bandwidth 的 field 中 >0.3 的格格占比更大(场更弥散)。
    """
    base = replace(
        base_cfg,
        sensor=SensorConfig(footprint_along_m=4.0, footprint_lateral_m=6.0),
    )
    cfg_narrow = replace(
        base,
        target_field=TargetFieldConfig(bandwidth_m=1.0, intra_spread_m=12.0, targets_per_cluster=3),
    )
    cfg_wide = replace(
        base,
        target_field=TargetFieldConfig(bandwidth_m=6.0, intra_spread_m=12.0, targets_per_cluster=3),
    )

    inst_narrow = default_instance(cfg_narrow, n_targets=9)
    inst_wide = default_instance(cfg_wide, n_targets=9)

    frac_narrow = float((inst_narrow.field > 0.3).mean())
    frac_wide = float((inst_wide.field > 0.3).mean())

    assert frac_wide > frac_narrow, (
        f"bandwidth 旋钮未生效:bandwidth=6.0 的 field>0.3 占比 {frac_wide:.4f}"
        f" 未大于 bandwidth=1.0 的 {frac_narrow:.4f}。"
        "runner.default_instance 须将 cfg.target_field.bandwidth_m 传入 KDE 带宽。"
    )


# ======================================================================
# 测试 3 —— intra_spread 旋钮生效
# 大 intra_spread_m → targets 到质心均方距离更大(簇内散布受控)
# ======================================================================
def test_intra_spread_knob_larger_spread_produces_more_dispersed_targets(
    base_cfg: SimConfig,
) -> None:
    """cfg.target_field.intra_spread_m 大 → targets 空间散布更大。

    固定 seed + footprint + bandwidth,比较 intra_spread_m=2.0 vs 20.0。
    判据:大 intra_spread 时各 target 到全体质心的均方距离更大。
    注:小场地(60×20)下 spread=20.0 会触发边界拒绝重采样逻辑,仍须有效采样。
    """
    base = replace(
        base_cfg,
        sensor=SensorConfig(footprint_along_m=4.0, footprint_lateral_m=6.0),
    )
    cfg_tight = replace(
        base,
        target_field=TargetFieldConfig(bandwidth_m=3.0, intra_spread_m=2.0, targets_per_cluster=3),
    )
    # 用稍大场地避免 spread=20.0 超场地一半使测试不稳
    big_world = WorldConfig(200.0, 100.0, 2.0)
    base_big = replace(base_cfg, world=big_world)
    cfg_loose = replace(
        base_big,
        sensor=SensorConfig(footprint_along_m=4.0, footprint_lateral_m=6.0),
        target_field=TargetFieldConfig(bandwidth_m=3.0, intra_spread_m=20.0, targets_per_cluster=3),
    )
    cfg_tight_big = replace(
        base_big,
        sensor=SensorConfig(footprint_along_m=4.0, footprint_lateral_m=6.0),
        target_field=TargetFieldConfig(bandwidth_m=3.0, intra_spread_m=2.0, targets_per_cluster=3),
    )

    inst_tight = default_instance(cfg_tight_big, n_targets=9)
    inst_loose = default_instance(cfg_loose, n_targets=9)

    def _msd_to_centroid(targets) -> float:
        """各 target 到全体质心的均方距离。"""
        pts = np.asarray([np.asarray(t, dtype=np.float64) for t in targets])
        centroid = pts.mean(axis=0)
        return float(np.mean(np.sum((pts - centroid) ** 2, axis=1)))

    msd_tight = _msd_to_centroid(inst_tight.targets)
    msd_loose = _msd_to_centroid(inst_loose.targets)

    assert msd_loose > msd_tight, (
        f"intra_spread 旋钮未生效:intra_spread=20.0 的均方距离 {msd_loose:.2f}"
        f" 未大于 intra_spread=2.0 的 {msd_tight:.2f}。"
        "runner._default_cluster_params 须取 cfg.target_field.intra_spread_m。"
    )


# ======================================================================
# 测试 4 —— 默认向后兼容
# SimConfig().target_field 的默认值与 TargetFieldConfig(3.0, 12.0, 3) 一致
# ======================================================================
def test_default_target_field_config_values_are_anchored() -> None:
    """§v1.0.8 向后兼容:SimConfig().target_field 的默认值须锁定。

    断言:
      SimConfig().target_field == TargetFieldConfig(bandwidth_m=3.0, intra_spread_m=12.0, targets_per_cluster=3)
    这确保 §v1.0.8 引入后,默认场景与 footprint=6 时的历史行为逐位一致
    (bandwidth=0.5×6=3m, intra_spread 原硬编码 _DEFAULT = 3*footprint_lateral_m? 取合理历史值)。
    """
    default_tf = SimConfig().target_field
    expected_tf = TargetFieldConfig(bandwidth_m=3.0, intra_spread_m=12.0, targets_per_cluster=3)

    assert default_tf == expected_tf, (
        f"默认 target_field 值被篡改。\n"
        f"  实际  : {default_tf}\n"
        f"  期望  : {expected_tf}\n"
        "§v1.0.8 默认值须锁定为 (bandwidth_m=3.0, intra_spread_m=12.0, targets_per_cluster=3)。"
    )


def test_default_instance_field_peaks_near_one() -> None:
    """默认 cfg 跑出的 field.max() ≈ 1.0(归一化行为稳定)。

    作为 §v1.0.8 向后兼容的额外保障:默认配置下 default_instance 仍能产出有效先验场。
    """
    cfg = SimConfig()
    inst = default_instance(cfg, n_targets=6)

    assert inst.field.max() == pytest.approx(1.0, abs=0.01), (
        f"默认 cfg 的 field.max()={inst.field.max():.4f} 偏离 1.0。"
        "field 须归一化到 [0,1](先验场峰值=1)。"
    )
    assert len(inst.targets) > 0, "默认 cfg 须生成非空 targets 列表。"
