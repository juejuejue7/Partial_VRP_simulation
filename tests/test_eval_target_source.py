"""[A1 验收测试 · 波次1.x §v1.0.4] 评估管线 target_source —— default_detections / run_ablation。

判据来源:eval/runner.py 的 default_detections / default_targets / run_ablation 签名 + docstring、
contracts/env_static.py(detections=估计,GT=targets)、本波次补丁单(必测 8–9)。

被测对象(对契约编程,只断言结构/确定性,不断言 scanned_map 与 oracle_gt 数值相等):
    default_detections(cfg, instance=None, *, threshold_rel=0.3, min_separation_m=None) -> List[Waypoint]
    run_ablation(cfg, targets=None, *, target_source="scanned_map", ...) -> dict
        target_source ∈ {"scanned_map","oracle_gt"};非法值 raise ValueError。

口径:cfg 用 WorldConfig(60,30,0.5) 装入 SimConfig,seed 固定(0)→ 实例逐位复现。
为不依赖可选求解器(ortools/pulp)且保证键集稳定,run_ablation 仅跑 solvers=["greedy"]。
注:scanned_map 喂检测点(估计)、oracle_gt 喂 GT;只断言**结构**(键集)一致,不断言数值相等。
"""
from __future__ import annotations

import numpy as np
import pytest

from msim.contracts.config import SimConfig, WorldConfig
from msim.eval.runner import default_detections, run_ablation


def _cfg() -> SimConfig:
    # 小场降体量(nx=120, ny=60);seed 固定 → 耦合实例逐位复现。
    return SimConfig(world=WorldConfig(x_max_m=60.0, y_max_m=30.0, res_m=0.5), seed=0)


# ======================================================================
# 必测 8:default_detections —— 非空 List、每点 shape (2,) 且在场内、两次调用确定性一致
# ======================================================================
def test_default_detections_shape_bounds_nonempty() -> None:
    cfg = _cfg()
    dets = default_detections(cfg)
    assert isinstance(dets, list), "default_detections 须返回 List"
    assert len(dets) > 0, "默认耦合实例应抽出非空检测点"
    for p in dets:
        arr = np.asarray(p, dtype=np.float64)
        assert arr.shape == (2,), f"每个检测点 shape 须为 (2,),得 {arr.shape}"
        assert 0.0 <= float(arr[0]) <= cfg.world.x_max_m, f"检测点 North 越界:{arr}"
        assert 0.0 <= float(arr[1]) <= cfg.world.y_max_m, f"检测点 East 越界:{arr}"


def test_default_detections_deterministic() -> None:
    cfg = _cfg()
    d1 = default_detections(cfg)
    d2 = default_detections(cfg)
    assert len(d1) == len(d2), "同 cfg(seed 固定)两次调用检测点数须相同"
    for a, b in zip(d1, d2):
        assert np.array_equal(np.asarray(a), np.asarray(b)), "同 cfg 两次调用须逐位相同(确定性)"


# ======================================================================
# 必测 9:run_ablation 两种 target_source 键集一致;非法值 raise ValueError
# ======================================================================
def test_run_ablation_scanned_map_and_oracle_gt_same_keys() -> None:
    cfg = _cfg()
    # 仅跑 greedy:无可选求解器依赖,键集稳定(键集由 solvers 决定,与 target_source 无关)。
    r_scan = run_ablation(cfg, target_source="scanned_map", solvers=["greedy"])
    r_oracle = run_ablation(cfg, target_source="oracle_gt", solvers=["greedy"])

    assert isinstance(r_scan, dict) and isinstance(r_oracle, dict), "run_ablation 须返回 dict"
    # 顶层键集一致(结构一致,不断言数值相等)。
    assert set(r_scan.keys()) == set(r_oracle.keys()), \
        f"两种 target_source 顶层键集须一致:{set(r_scan.keys())} vs {set(r_oracle.keys())}"
    # balance_off / balance_on 子 dict 的求解器键集一致。
    for branch in ("balance_off", "balance_on"):
        assert set(r_scan[branch].keys()) == set(r_oracle[branch].keys()), \
            f"{branch} 求解器键集须一致"
    # 标记位如实记录各自的 source(口径标注)。
    assert r_scan["target_source"] == "scanned_map"
    assert r_oracle["target_source"] == "oracle_gt"


def test_run_ablation_rejects_unknown_target_source() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError):
        run_ablation(cfg, target_source="bogus", solvers=["greedy"])
