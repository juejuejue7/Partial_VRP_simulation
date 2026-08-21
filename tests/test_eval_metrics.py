"""[A1 验收测试 · #8 A6 评估] metrics —— max/(CV 或 Jain)/总能耗。

判据来源:eval/metrics.py 骨架(max_energy/coefficient_of_variation/jain_index/
total_energy)、harness §2(A6 指标)、CLAUDE.md(CV 与 Jain 单调对应 J=1/(1+CV²))。

手算参考:
  [50,50]: max=50, total=100, CV=0,    Jain=1.0
  [20,80]: max=80, total=100, CV=0.6,  Jain=(100²)/(2·6800)=0.73529...
"""
from __future__ import annotations

import math

import pytest

from msim.eval.metrics import (
    coefficient_of_variation,
    jain_index,
    max_energy,
    total_energy,
)


def test_max_energy() -> None:
    assert max_energy([10.0, 30.0, 20.0]) == pytest.approx(30.0)
    assert max_energy([7.0]) == pytest.approx(7.0)


def test_total_energy() -> None:
    assert total_energy([10.0, 30.0, 20.0]) == pytest.approx(60.0)
    assert total_energy([50.0, 50.0]) == pytest.approx(100.0)


def test_cv_zero_when_equal() -> None:
    assert coefficient_of_variation([50.0, 50.0]) == pytest.approx(0.0, abs=1e-12)
    assert coefficient_of_variation([7.0, 7.0, 7.0]) == pytest.approx(0.0, abs=1e-12)


def test_cv_value_handcalc() -> None:
    # [20,80]: mean=50, std(population)=30 → CV=0.6
    cv = coefficient_of_variation([20.0, 80.0])
    assert cv == pytest.approx(0.6, rel=1e-9), f"CV={cv}"


def test_jain_perfect_when_equal() -> None:
    assert jain_index([50.0, 50.0]) == pytest.approx(1.0, rel=1e-9)
    assert jain_index([3.0, 3.0, 3.0, 3.0]) == pytest.approx(1.0, rel=1e-9)


def test_jain_value_handcalc() -> None:
    # [20,80]: (100²)/(2·(400+6400)) = 10000/13600
    j = jain_index([20.0, 80.0])
    assert j == pytest.approx(10000.0 / 13600.0, rel=1e-9), f"Jain={j}"


def test_jain_cv_monotone_relation() -> None:
    """CV 与 Jain 单调对应:J = 1/(1+CV²)(CLAUDE.md/metrics docstring)。"""
    for vals in ([20.0, 80.0], [10.0, 40.0, 70.0], [5.0, 5.0, 50.0]):
        cv = coefficient_of_variation(vals)
        j = jain_index(vals)
        assert j == pytest.approx(1.0 / (1.0 + cv * cv), rel=1e-9), \
            f"J 与 CV 的单调关系不成立:vals={vals}"


def test_jain_in_unit_interval() -> None:
    for vals in ([1.0, 99.0], [50.0, 50.0], [1.0, 2.0, 3.0, 4.0]):
        j = jain_index(vals)
        assert 0.0 < j <= 1.0 + 1e-12


def test_balance_metric_orders_allocations() -> None:
    """更均衡分配:max 更小、CV 更小、Jain 更大(同总量)。"""
    bal = [50.0, 50.0]
    skew = [20.0, 80.0]
    assert max_energy(bal) < max_energy(skew)
    assert coefficient_of_variation(bal) < coefficient_of_variation(skew)
    assert jain_index(bal) > jain_index(skew)
