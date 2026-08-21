"""dataset 验收:28 个真实目标点的装载、换轴、越界与防错机制。"""
from __future__ import annotations

import os

import numpy as np
import pytest

from vrpsim.contracts.config import DEFAULT_WAYPOINTS_CSV
from vrpsim.contracts.dataset import (N_CHIMNEY_EXPECTED, N_MOUND_EXPECTED,
                                      N_TARGETS_EXPECTED, MothraDataset)
from vrpsim.contracts.frames import RASTER_EAST_EXTENT_M, RASTER_NORTH_EXTENT_M
from vrpsim.dataset import load_waypoints


@pytest.fixture(scope="module")
def ds() -> MothraDataset:
    return load_waypoints(DEFAULT_WAYPOINTS_CSV)


def test_source_file_present_and_bom():
    assert os.path.isfile(DEFAULT_WAYPOINTS_CSV)
    with open(DEFAULT_WAYPOINTS_CSV, "rb") as f:
        head = f.read(16)
    assert head.startswith(b"\xef\xbb\xbf"), "源文件应为 UTF-8-BOM(07_make_waypoints.py 的口径)"


def test_size_and_types(ds):
    assert ds.n == N_TARGETS_EXPECTED
    assert int(ds.mask_of_type("chimney").sum()) == N_CHIMNEY_EXPECTED
    assert int(ds.mask_of_type("mound").sum()) == N_MOUND_EXPECTED
    for arr in (ds.depth_D_m, ds.height_m, ds.type_, ds.name,
                ds.waypoint_id, ds.sequence_id, ds.lon, ds.lat):
        assert arr.shape == (ds.n,)
    assert ds.targets_ned.shape == (ds.n, 2)
    assert ds.targets_ned.dtype == np.float64


def test_axis_swap_is_correct(ds):
    """North 是长边方向(跨度 556 m),East 是短边(跨度 188 m)。写反了这里会红。"""
    span_north = float(np.ptp(ds.targets_ned[:, 0]))
    span_east = float(np.ptp(ds.targets_ned[:, 1]))
    assert span_north > span_east, "North 跨度应远大于 East —— 疑似轴序写反"
    assert 550.0 < span_north < 560.0
    assert 185.0 < span_east < 192.0


def test_all_targets_in_raster_bounds(ds):
    """装载阶段比的是**原始栅格**外框(653x294),裁切是后面的事(D7)。"""
    x_n, y_e = ds.targets_ned[:, 0], ds.targets_ned[:, 1]
    assert x_n.min() >= 0.0 and x_n.max() <= RASTER_NORTH_EXTENT_M
    assert y_e.min() >= 0.0 and y_e.max() <= RASTER_EAST_EXTENT_M


def test_select_preserves_ned_and_fields(ds):
    """裁切只是取子集:保留点的 NED 坐标逐位不变(原点不动),各字段同步切片。"""
    from vrpsim.contracts.config import CropConfig

    keep = CropConfig().contains(ds.targets_ned[:, 0], ds.targets_ned[:, 1])
    sub = ds.select(keep)
    assert sub.n == int(keep.sum())
    assert np.array_equal(sub.targets_ned, ds.targets_ned[keep])
    assert np.array_equal(sub.waypoint_id, ds.waypoint_id[keep])
    assert np.array_equal(sub.name, ds.name[keep])
    with pytest.raises(ValueError):
        ds.select(np.ones(ds.n + 1, dtype=bool))


def test_depth_is_down_positive(ds):
    """D 向下为正:海底在海面下 ~2.25 km ⇒ 全为大正数。"""
    assert ds.depth_D_m.min() > 2250.0
    assert ds.depth_D_m.max() < 2290.0


def test_height_is_not_depth(ds):
    """height_m 是烟囱高出海底的高度(4~18 m),与 depth_D_m 不是一回事,不许混用。"""
    assert ds.height_m.min() >= 4.0 and ds.height_m.max() <= 18.0
    assert not np.any(np.isclose(ds.height_m, ds.depth_D_m))


def test_ids_stable_and_ordered(ds):
    assert np.array_equal(ds.waypoint_id, np.arange(1, ds.n + 1, dtype=np.int32))
    assert np.all(np.diff(ds.sequence_id) > 0), "sequence_id 应严格递增(沿用源表顺序)"


def test_as_waypoint_list_shape(ds):
    wps = ds.as_waypoint_list()
    assert len(wps) == ds.n
    assert all(w.shape == (2,) and w.dtype == np.float64 for w in wps)
    assert np.allclose(np.array(wps), ds.targets_ned)


def test_named_sites_present(ds):
    names = {n for n in ds.name.tolist() if n}
    assert "Stonehenge" in names and any(n.startswith("Faulty Towers") for n in names)


# ======================================================================
# 防错机制:改坏输入必须炸,不许静默带病运行
# ======================================================================
def _write_variant(tmp_path, mutate) -> str:
    with open(DEFAULT_WAYPOINTS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()
    lines = mutate(lines)
    p = tmp_path / "variant.csv"
    p.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8-sig")
    return str(p)


def test_rejects_missing_column(tmp_path):
    path = _write_variant(tmp_path, lambda ls: [ls[0].replace("height_m,", "")] + ls[1:])
    with pytest.raises(ValueError, match="缺列"):
        load_waypoints(path)


def test_rejects_tampered_local_columns(tmp_path):
    """把某行的 x_local_m 挪 5 m —— 与 lon/lat 重算值对不上,必须炸。"""
    def mutate(ls):
        cols = ls[1].split(",")
        cols[6] = f"{float(cols[6]) + 5.0:.3f}"
        ls[1] = ",".join(cols)
        return ls

    with pytest.raises(ValueError, match="不一致"):
        load_waypoints(_write_variant(tmp_path, mutate))


def test_rejects_swapped_axes(tmp_path):
    """把 x_local_m / y_local_m 两列对调 —— 正是轴序陷阱,必须炸。"""
    def mutate(ls):
        out = [ls[0]]
        for line in ls[1:]:
            c = line.split(",")
            c[6], c[7] = c[7], c[6]
            out.append(",".join(c))
        return out

    with pytest.raises(ValueError, match="不一致"):
        load_waypoints(_write_variant(tmp_path, mutate))


def test_strict_flag_guards_dataset_size(tmp_path):
    path = _write_variant(tmp_path, lambda ls: ls[:-1])          # 删掉最后一个目标
    with pytest.raises(ValueError, match="规模与冻结值不符"):
        load_waypoints(path, strict=True)
    assert load_waypoints(path, strict=False).n == N_TARGETS_EXPECTED - 1
