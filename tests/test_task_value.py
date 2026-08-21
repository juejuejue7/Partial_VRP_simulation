"""[A1 验收测试 · #7 A4 任务/价值 · 加倍] value —— 不重复计数 + 期望熵降公式。

判据来源:contracts/task.py(expected_entropy_reduction / mark_planned)、
harness §4.2(value 行,加倍测试)、project_summary(开环、先验期望熵降)。

>>> 这是学术可信度的命门:value 的 bug 不会让程序崩,只会让结论悄悄变好看。<<<

期望熵降(单格,bit)= 互信息 I(target; z),手算参考实现见 _ref_cell_eig:
  P(z=1) = p·p_d + (1-p)·p_fa
  p_post(z=1) = p·p_d / P(z=1);  p_post(z=0) = p·(1-p_d) / P(z=0)
  EIG = H(p) - [P(z=1)·H(p_post1) + P(z=0)·H(p_post0)]，H 为二元熵(log2)。
不重复计数:对 planned_coverage==True 的格不再累加(同一格被两 Follower 重复经过只计一次)。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from msim.contracts.physics import CameraSensorModel
from msim.task.value import expected_entropy_reduction, mark_planned


# ----------------------------------------------------------------------
# 手算参考实现(独立于被测实现,作为 ground truth)
# ----------------------------------------------------------------------
def _h_bit(q: float) -> float:
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return -q * math.log2(q) - (1.0 - q) * math.log2(1.0 - q)


def _ref_cell_eig(p: float, p_d: float, p_fa: float) -> float:
    pz1 = p * p_d + (1.0 - p) * p_fa
    pz0 = 1.0 - pz1
    eig = _h_bit(p)
    if pz1 > 0.0:
        post1 = p * p_d / pz1
        eig -= pz1 * _h_bit(post1)
    if pz0 > 0.0:
        post0 = p * (1.0 - p_d) / pz0
        eig -= pz0 * _h_bit(post0)
    return eig


@pytest.fixture
def sensor() -> CameraSensorModel:
    return CameraSensorModel(p_d=0.95, p_fa=0.05)


def _fresh_mask(nx: int, ny: int) -> np.ndarray:
    return np.zeros((nx, ny), dtype=bool)


# ======================================================================
# 公式正确性:单格 EIG 与手算参考一致
# ======================================================================
def test_single_cell_eig_matches_reference(sensor: CameraSensorModel) -> None:
    nx, ny = 4, 4
    field = np.full((nx, ny), 0.3, dtype=np.float64)
    field[1, 1] = 0.5    # 最大不确定 → 最大熵降
    planned = _fresh_mask(nx, ny)

    for cell in [(1, 1), (0, 0), (2, 3)]:
        p = float(field[cell])
        got = expected_entropy_reduction({cell}, planned, field, sensor, unit="bit")
        exp = _ref_cell_eig(p, sensor.p_d, sensor.p_fa)
        assert got == pytest.approx(exp, rel=1e-9, abs=1e-12), \
            f"cell {cell} (p={p}) EIG={got} != ref {exp}"


def test_multi_cell_eig_is_sum(sensor: CameraSensorModel) -> None:
    nx, ny = 4, 4
    field = np.array([[0.1, 0.5, 0.9, 0.2],
                      [0.5, 0.5, 0.3, 0.7],
                      [0.4, 0.6, 0.5, 0.8],
                      [0.2, 0.5, 0.1, 0.5]], dtype=np.float64)
    planned = _fresh_mask(nx, ny)
    cells = {(0, 1), (1, 0), (2, 2), (3, 3)}
    got = expected_entropy_reduction(cells, planned, field, sensor, unit="bit")
    exp = sum(_ref_cell_eig(float(field[c]), sensor.p_d, sensor.p_fa) for c in cells)
    assert got == pytest.approx(exp, rel=1e-9, abs=1e-12)


# ======================================================================
# p=0.5 → 最大熵降;p∈{0,1}(确定格)→ 熵降 0
# ======================================================================
def test_certain_cells_zero_eig(sensor: CameraSensorModel) -> None:
    nx, ny = 3, 3
    field = np.zeros((nx, ny), dtype=np.float64)
    field[0, 0] = 0.0    # 确定无目标
    field[1, 1] = 1.0    # 确定有目标
    planned = _fresh_mask(nx, ny)
    assert expected_entropy_reduction({(0, 0)}, planned, field, sensor) == pytest.approx(0.0, abs=1e-12)
    assert expected_entropy_reduction({(1, 1)}, planned, field, sensor) == pytest.approx(0.0, abs=1e-12)


def test_p_half_is_max_eig(sensor: CameraSensorModel) -> None:
    nx, ny = 3, 3
    field = np.full((nx, ny), 0.5, dtype=np.float64)
    planned = _fresh_mask(nx, ny)
    eig_half = expected_entropy_reduction({(0, 0)}, planned, field, sensor)
    field2 = field.copy(); field2[0, 0] = 0.2
    eig_other = expected_entropy_reduction({(0, 0)}, planned, field2, sensor)
    assert eig_half >= eig_other - 1e-12, "p=0.5 应给出不小于偏离 0.5 的熵降"


# ======================================================================
# >>> 不重复计数(命门)<<<:planned_coverage 已 True 的格不再贡献熵降
# ======================================================================
def test_planned_cells_excluded(sensor: CameraSensorModel) -> None:
    nx, ny = 4, 4
    field = np.full((nx, ny), 0.5, dtype=np.float64)
    planned = _fresh_mask(nx, ny)
    cell = (2, 2)

    first = expected_entropy_reduction({cell}, planned, field, sensor)
    assert first > 0.0

    # 标记为已计划覆盖后,同一格熵降应为 0(不重复计数)
    mark_planned(planned, {cell})
    second = expected_entropy_reduction({cell}, planned, field, sensor)
    assert second == pytest.approx(0.0, abs=1e-12), \
        "已计划覆盖的格须被排除(不重复计数)"


def test_two_followers_same_cell_counts_once(sensor: CameraSensorModel) -> None:
    """同一格被两台 Follower 走廊重复经过,熵降只计一次(harness §4.2 核心判据)。"""
    nx, ny = 5, 5
    field = np.full((nx, ny), 0.4, dtype=np.float64)
    planned = _fresh_mask(nx, ny)

    follower_A = {(1, 1), (1, 2), (2, 1)}
    follower_B = {(2, 1), (2, 2), (1, 2)}    # 与 A 在 (1,2),(2,1) 重叠
    union = follower_A | follower_B

    # 顺序结算:A 先走、标记,再算 B(只新增 B 独有格)
    eig_A = expected_entropy_reduction(follower_A, planned, field, sensor)
    mark_planned(planned, follower_A)
    eig_B = expected_entropy_reduction(follower_B, planned, field, sensor)
    mark_planned(planned, follower_B)
    total_sequential = eig_A + eig_B

    # 一次性按并集结算(每格恰一次)
    eig_union = expected_entropy_reduction(union, _fresh_mask(nx, ny), field, sensor)

    assert total_sequential == pytest.approx(eig_union, rel=1e-9, abs=1e-12), \
        "重叠格被重复计数:顺序结算总熵降 != 并集一次性熵降"


def test_repeated_pass_adds_nothing(sensor: CameraSensorModel) -> None:
    """同一台 Follower 再走一遍已覆盖区,熵降增量为 0。"""
    nx, ny = 4, 4
    field = np.full((nx, ny), 0.6, dtype=np.float64)
    planned = _fresh_mask(nx, ny)
    cells = {(0, 0), (0, 1), (1, 0)}
    e1 = expected_entropy_reduction(cells, planned, field, sensor)
    mark_planned(planned, cells)
    e2 = expected_entropy_reduction(cells, planned, field, sensor)  # 再走一遍
    assert e1 > 0.0
    assert e2 == pytest.approx(0.0, abs=1e-12), "重复经过不得再贡献熵降"


# ======================================================================
# 只读契约:expected_entropy_reduction 不得改写 field / planned
# ======================================================================
def test_eig_is_read_only(sensor: CameraSensorModel) -> None:
    nx, ny = 4, 4
    field = np.full((nx, ny), 0.5, dtype=np.float64)
    planned = _fresh_mask(nx, ny)
    field_before = field.copy()
    planned_before = planned.copy()
    _ = expected_entropy_reduction({(1, 1), (2, 2)}, planned, field, sensor)
    np.testing.assert_array_equal(field, field_before, "EIG 不得改写 field")
    np.testing.assert_array_equal(planned, planned_before, "EIG 不得改写 planned_coverage")


# ======================================================================
# mark_planned —— in-place 置 True,只动指定格
# ======================================================================
def test_mark_planned_in_place(sensor: CameraSensorModel) -> None:
    nx, ny = 4, 4
    planned = _fresh_mask(nx, ny)
    mark_planned(planned, {(1, 1), (3, 2)})
    assert planned[1, 1] and planned[3, 2]
    # 其它格不变
    assert planned.sum() == 2, "mark_planned 只应置位指定格"
