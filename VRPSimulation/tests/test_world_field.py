"""world / field 验收:概率场口径、栅格对齐、落盘复现、出图朝向。"""
from __future__ import annotations

import numpy as np
import pytest

from vrpsim.contracts.config import (CropConfig, FieldBuildConfig, MothraSimConfig,
                                     full_raster_crop, mothra_world_config)
from vrpsim.contracts.dataset import (DROPPED_WAYPOINT_IDS, N_CHIMNEY_CROPPED,
                                      N_MOUND_CROPPED, N_TARGETS_CROPPED)
from vrpsim.contracts.frames import (GRID_1M_COLS, GRID_1M_ROWS, NEDFrame,
                                     WORLD_EAST_EXTENT_M, WORLD_NORTH_EXTENT_M)
from vrpsim.field import build_probability_field
from vrpsim.world import build_mothra_world, load_world, save_world


@pytest.fixture(scope="module")
def mw():
    return build_mothra_world(MothraSimConfig())


def test_cropped_world_shape(mw):
    """D7:分析域 500(N) x 100(E),res_m=1.0 ⇒ 场 (500, 100)。"""
    assert (mw.world.x_max_m, mw.world.y_max_m) == (WORLD_NORTH_EXTENT_M, WORLD_EAST_EXTENT_M)
    assert mw.field.shape == (500, 100)
    assert mw.field.dtype == np.float64
    assert mw.world.x_max_m > mw.world.y_max_m, "North 必须是长边"


def test_crop_keeps_expected_targets(mw):
    """裁切后 22 个 = 14 chimney + 8 mound;丢的 6 个全是 chimney,id 与契约一致。"""
    assert mw.dataset.n == N_TARGETS_CROPPED == 22
    assert int(mw.dataset.mask_of_type("chimney").sum()) == N_CHIMNEY_CROPPED
    assert int(mw.dataset.mask_of_type("mound").sum()) == N_MOUND_CROPPED
    assert tuple(mw.meta["crop"]["dropped_waypoint_ids"]) == DROPPED_WAYPOINT_IDS
    assert mw.meta["crop"]["n_loaded"] == 28


def test_crop_does_not_move_coordinates():
    """裁切只缩外框、不平移原点 ⇒ 保留点的 NED 坐标与全幅世界逐位相同。"""
    full = build_mothra_world(MothraSimConfig(crop=full_raster_crop()))
    crop = build_mothra_world(MothraSimConfig())
    keep = np.isin(full.dataset.waypoint_id, crop.dataset.waypoint_id)
    assert np.array_equal(full.dataset.targets_ned[keep], crop.dataset.targets_ned)


def test_full_raster_crop_still_available():
    """全幅世界(653x294 / 28 点)仍可构建,供溯源与对照。"""
    full = build_mothra_world(MothraSimConfig(crop=full_raster_crop()))
    assert full.world.nx == GRID_1M_ROWS == 653
    assert full.world.ny == GRID_1M_COLS == 294
    assert full.dataset.n == 28


def test_crop_lower_bound_must_be_zero():
    """非零下界会引入平移项、让 NED 与栅格索引脱钩 —— 必须拒绝。"""
    with pytest.raises(ValueError, match="下界必须为 0"):
        mothra_world_config(1.0, CropConfig(north_m=(50.0, 500.0)))


def test_field_is_graded_probability(mw):
    """graded(分级连续),不是二值掩膜(project_summary §3)。"""
    f = mw.field
    assert f.min() >= 0.0 and f.max() <= 1.0
    assert np.isclose(f.max(), 1.0)
    frac_mid = float(((f > 0.05) & (f < 0.95)).mean())
    assert frac_mid > 0.001, "几乎没有中间值 —— 场退化成二值了"


def test_field_peaks_colocate_with_targets(mw):
    """目标处的场值必须显著高于全场(耦合性,同 msim GroundTruthInstance 的不变量)。"""
    at_t = np.array([mw.field_at(x, y) for x, y in mw.dataset.targets_ned])
    assert at_t.min() > 10.0 * float(np.median(mw.field))
    assert float(np.median(at_t)) > 0.2


def test_field_deterministic(mw):
    """纯几何确定:重建一次必须逐位相同(无 RNG)。"""
    again = build_mothra_world(MothraSimConfig())
    assert np.array_equal(again.field, mw.field)
    assert np.array_equal(again.dataset.targets_ned, mw.dataset.targets_ned)


def test_bandwidth_widens_field(mw):
    """带宽变大 → 场更弥散(>0.5 的面积变大)。防止 bandwidth 被接错/忽略。"""
    wide = build_probability_field(mw.dataset, mw.world, FieldBuildConfig(bandwidth_m=9.0))
    assert float((wide > 0.5).mean()) > float((mw.field > 0.5).mean())


def test_height_weight_mode_changes_field(mw):
    """height 加权是【待裁决项 2】:必须可用、但绝不能是默认。"""
    assert MothraSimConfig().field_build.weight_mode == "uniform"
    weighted = build_probability_field(mw.dataset, mw.world,
                                       FieldBuildConfig(weight_mode="height"))
    assert weighted.shape == mw.field.shape
    assert not np.array_equal(weighted, mw.field)


def test_rejects_bad_field_config(mw):
    with pytest.raises(ValueError):
        build_probability_field(mw.dataset, mw.world, FieldBuildConfig(bandwidth_m=0.0))
    with pytest.raises(ValueError):
        build_probability_field(mw.dataset, mw.world,
                                FieldBuildConfig(weight_mode="nope"))  # type: ignore[arg-type]


# ======================================================================
# 栅格索引口径
# ======================================================================
def test_grid_index_1m_matches_upstream_formula(mw):
    """NEDFrame.grid_index_1m 必须与上游 meta 的 index_from_local 等价(已换轴)。

    ⚠ 这是**原始栅格**索引,与裁切无关 —— 原点不动,所以裁切前后逐位相同。
    """
    frame = NEDFrame()
    for (x_n, y_e) in mw.dataset.targets_ned:
        row, col = frame.grid_index_1m(x_n, y_e)
        assert col == int(y_e)                       # 上游: col = int(x_local) = int(East)
        assert row == (653 - 1) - int(x_n)           # 上游: row = (nrows-1) - int(y_local)
        assert 0 <= row < 653 and 0 <= col < 294


def test_crop_index_matches_grid(mw):
    """裁切后数组索引 = msim Grid 的 world_to_cell,同口径(floor,无偏移)。"""
    for t in mw.dataset.as_waypoint_list():
        assert NEDFrame.crop_index(t[0], t[1], mw.world.res_m) == mw.grid.world_to_cell(t)


def test_grid_roundtrip_invariant(mw):
    """msim 的 world_to_cell(cell_to_world(c)) == c 不变量在本场地上也成立。"""
    rng = np.random.default_rng(0)
    for ix, iy in zip(rng.integers(0, mw.world.nx, 200), rng.integers(0, mw.world.ny, 200)):
        c = (int(ix), int(iy))
        assert mw.grid.world_to_cell(mw.grid.cell_to_world(c)) == c


# ======================================================================
# 落盘复现
# ======================================================================
def test_save_load_bitwise(mw, tmp_path):
    p = str(tmp_path / "w.npz")
    save_world(mw, p)
    back = load_world(p)
    assert np.array_equal(back.field, mw.field)
    assert np.array_equal(back.dataset.targets_ned, mw.dataset.targets_ned)
    assert np.array_equal(back.dataset.type_, mw.dataset.type_)
    assert np.array_equal(back.dataset.name, mw.dataset.name)
    assert back.world == mw.world
    assert back.meta["n_targets"] == mw.meta["n_targets"]
    assert back.meta["field_build"] == mw.meta["field_build"]


# ======================================================================
# 出图朝向(D6)
# ======================================================================
def test_plot_orientation_matches_ned():
    """单目标场:imshow(field, origin='lower') 下,该目标应落在 (row=North, col=East)。

    这条是 D6 的看门狗 —— 若哪天有人在 viz 里加了转置,North/East 会互换,这里会红。
    """
    from vrpsim.contracts.dataset import MothraDataset

    world = mothra_world_config(res_m=1.0)
    one = MothraDataset(
        targets_ned=np.array([[400.0, 50.0]]),      # 靠北、居中
        depth_D_m=np.array([2280.0]), height_m=np.array([10.0]),
        type_=np.array(["chimney"], dtype="<U7"), name=np.array([""], dtype="<U32"),
        waypoint_id=np.array([1], dtype=np.int32), sequence_id=np.array([1], dtype=np.int32),
        lon=np.array([0.0]), lat=np.array([0.0]))
    f = build_probability_field(one, world, FieldBuildConfig(bandwidth_m=3.0))
    row, col = np.unravel_index(int(np.argmax(f)), f.shape)
    assert abs(row - 400) <= 1, "0 轴应是 North"
    assert abs(col - 50) <= 1, "1 轴应是 East"
    # origin='lower' ⇒ row 大 = 图上偏高 = 偏北;row 400 在 500 行里靠顶部
    assert row > f.shape[0] * 0.7
