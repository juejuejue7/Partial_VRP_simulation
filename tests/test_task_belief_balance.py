"""[A1 验收测试 · #7 A4 任务/价值 · 加倍] belief(开环性) + energy_balance(min-max)。

判据来源:contracts/task.py(BeliefUpdater / balance_penalty)、harness §4.2
(belief 行:NullUpdater 逐位不变;value/belief 加倍)、CLAUDE.md(修士/博士唯一注入点)。

>>> belief 是开环底线的守卫:NullUpdater(修士)调用后 field 必须逐位不变,
    否则开环被静默破坏、实验结论失真。<<<
"""
from __future__ import annotations

import numpy as np
import pytest

from msim.contracts.physics import CameraSensorModel
from msim.task.belief import NullUpdater


@pytest.fixture
def sensor() -> CameraSensorModel:
    return CameraSensorModel()


# ======================================================================
# >>> 开环性证明:NullUpdater.update 后 field 逐位不变 <<<
# ======================================================================
def test_null_updater_leaves_field_bitwise_unchanged(sensor: CameraSensorModel) -> None:
    field = np.array([[0.1, 0.9, 0.5],
                      [0.3, 0.7, 0.42],
                      [0.0, 1.0, 0.5]], dtype=np.float64)
    before = field.copy()
    observed = {(0, 0), (1, 1), (2, 2), (0, 2)}
    NullUpdater().update(field, observed, sensor)
    assert np.array_equal(field, before), "NullUpdater 破坏开环性:field 被改写"


def test_null_updater_random_field_unchanged(sensor: CameraSensorModel) -> None:
    rng = np.random.default_rng(2024)
    field = rng.random((16, 9)).astype(np.float64)
    before = field.copy()
    # 覆盖全图所有格也不应改变(空操作)
    all_cells = {(i, j) for i in range(16) for j in range(9)}
    NullUpdater().update(field, all_cells, sensor)
    np.testing.assert_array_equal(field, before, "NullUpdater 对全图观测仍须逐位不变")


def test_null_updater_returns_none(sensor: CameraSensorModel) -> None:
    field = np.full((3, 3), 0.5, dtype=np.float64)
    ret = NullUpdater().update(field, {(0, 0)}, sensor)
    assert ret is None, "update 契约返回 None(in-place 语义)"


def test_null_updater_idempotent_repeated_calls(sensor: CameraSensorModel) -> None:
    field = np.full((4, 4), 0.5, dtype=np.float64)
    before = field.copy()
    for _ in range(5):
        NullUpdater().update(field, {(1, 1), (2, 2)}, sensor)
    np.testing.assert_array_equal(field, before, "多次调用 NullUpdater 仍须逐位不变")


def test_null_updater_measurements_ignored(sensor: CameraSensorModel) -> None:
    """measurements 修士恒为 None;即便误传也不得破坏开环。"""
    field = np.full((3, 3), 0.7, dtype=np.float64)
    before = field.copy()
    NullUpdater().update(field, {(0, 0)}, sensor, measurements=None)
    np.testing.assert_array_equal(field, before)


# ======================================================================
# energy_balance —— min-max 主形式:返回 max_i E_i
# ======================================================================
def test_balance_penalty_is_max() -> None:
    from msim.task.energy_balance import balance_penalty
    assert balance_penalty([10.0, 30.0]) == pytest.approx(30.0)
    assert balance_penalty([5.0, 5.0, 5.0]) == pytest.approx(5.0)
    assert balance_penalty([100.0]) == pytest.approx(100.0)


def test_balance_penalty_prefers_balanced_allocation() -> None:
    """均衡与效率分离:同样总能耗下,均衡分配的 min-max 罚更小。"""
    from msim.task.energy_balance import balance_penalty
    balanced = balance_penalty([50.0, 50.0])    # max=50
    skewed = balance_penalty([20.0, 80.0])      # 同总量 100,max=80
    assert balanced < skewed, "min-max 须偏好更均衡(max 更小)的分配"


def test_balance_penalty_nonnegative() -> None:
    from msim.task.energy_balance import balance_penalty
    assert balance_penalty([0.0, 0.0]) == pytest.approx(0.0)
    assert balance_penalty([1.0, 2.0, 3.0]) >= 0.0
