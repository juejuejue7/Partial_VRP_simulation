"""[第1层] `MothraWorld` —— 真实 Mothra 热液场的静态基础世界。

包含(D3 冻结的本步交付面):
    - NED 帧 + 栅格(`msim.geometry.grid.Grid`)
    - 28 个真实目标点(`MothraDataset`)
    - 目標分布確率マップ p(x,y)

**不含**水深地形栅格与坡度/可航行性层(D3 明确排除)。地形只以两种形式露头:
每个目标自带的 `depth_D_m`,以及 `NEDFrame.grid_index_1m` 冻结的索引口径 ——
将来要加地形层,从这两处接上即可,无需改坐标系。

修士开环:世界在 episode 内**永不变化**。save/load 逐位复现(对齐 msim 的 Q19 口径)。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from msim.contracts.config import WorldConfig
from msim.contracts.env_static import GroundTruthInstance
from msim.contracts.types import Field
from msim.geometry.grid import Grid

from .contracts.config import FieldBuildConfig, MothraSimConfig
from .contracts.dataset import MothraDataset
from .contracts.frames import NEDFrame
from .dataset import load_waypoints
from .field import build_probability_field

__all__ = ["MothraWorld", "build_mothra_world", "save_world", "load_world"]

_WORLD_NPZ_VERSION = 1


@dataclass(frozen=True)
class MothraWorld:
    """静态基础世界(不可变)。"""
    world: WorldConfig
    grid: Grid
    frame: NEDFrame
    dataset: MothraDataset
    field: Field                    # (nx, ny) float64 ∈ [0,1];field[ix, iy]
    meta: Dict[str, Any]

    # --- 便捷视图 ------------------------------------------------------
    @property
    def shape(self) -> tuple:
        return self.field.shape

    def to_gt_instance(self) -> GroundTruthInstance:
        """转成 msim 的耦合实例,直接喂给既有 window / rollout / VRP baseline 链路。

        真实场景下 28 个热液口就是绝对参考(它们由多波束+文献人工判读确定),
        语义上对应 msim 的 `targets` 位。
        """
        return GroundTruthInstance(targets=self.dataset.as_waypoint_list(),
                                   field=self.field,
                                   cluster_centers=[])

    def field_at(self, x_north_m: float, y_east_m: float) -> float:
        """世界坐标处的概率值(经 Grid 量化,越界 clip —— 同 msim 口径)。"""
        ix, iy = self.grid.world_to_cell(np.array([x_north_m, y_east_m], dtype=np.float64))
        return float(self.field[ix, iy])


# ======================================================================
# 构建
# ======================================================================
def build_mothra_world(cfg: Optional[MothraSimConfig] = None) -> MothraWorld:
    """从 waypoint CSV 构建静态基础世界。纯确定性,无 RNG。

    步骤:装载全部 28 个点 → 按 `cfg.crop` 裁到分析域(D7)→ 由保留的点派生概率场。
    裁切在这里**显式**发生,被丢的 waypoint_id 记进 meta,可追溯。
    """
    cfg = cfg or MothraSimConfig()
    world = cfg.world
    full = load_waypoints(cfg.waypoints_csv, strict=cfg.strict_dataset)

    keep = cfg.crop.contains(full.targets_ned[:, 0], full.targets_ned[:, 1])
    dataset = full.select(keep)
    dropped = full.waypoint_id[~keep].tolist()
    if dataset.n == 0:
        raise ValueError(f"裁切后一个目标都不剩:crop={cfg.crop}")

    field = build_probability_field(dataset, world, cfg.field_build)

    meta: Dict[str, Any] = {
        "version": _WORLD_NPZ_VERSION,
        "source_csv": os.path.basename(cfg.waypoints_csv),
        "crop": {"north_m": list(cfg.crop.north_m), "east_m": list(cfg.crop.east_m),
                 "n_loaded": full.n, "n_kept": dataset.n,
                 "dropped_waypoint_ids": [int(i) for i in dropped]},
        "window": {"look_back_m": cfg.window.look_back_m,
                   "width_m": cfg.window.width_m,
                   "advance_threshold_m": cfg.window.advance_threshold_m,
                   "lane_spacing_m": cfg.lane_spacing_m,
                   "mode": "sliding (D8)"},
        "n_targets": dataset.n,
        "n_chimney": int((dataset.type_ == "chimney").sum()),
        "n_mound": int((dataset.type_ == "mound").sum()),
        "world": {"x_max_m": world.x_max_m, "y_max_m": world.y_max_m,
                  "res_m": world.res_m, "nx": world.nx, "ny": world.ny},
        "frame": {"origin_utm9n": [NEDFrame().origin_easting_m, NEDFrame().origin_northing_m],
                  "origin_llh": [NEDFrame().origin_lon_deg, NEDFrame().origin_lat_deg, 0.0],
                  "epsg_utm": NEDFrame().epsg_utm,
                  "axes": "x=North, y=East, z=Down (msim/contracts/types.py)"},
        "field_build": {"bandwidth_m": cfg.field_build.bandwidth_m,
                        "weight_mode": cfg.field_build.weight_mode},
        "ned_extent_used_m": {
            "north": [float(dataset.targets_ned[:, 0].min()),
                      float(dataset.targets_ned[:, 0].max())],
            "east": [float(dataset.targets_ned[:, 1].min()),
                     float(dataset.targets_ned[:, 1].max())]},
        "depth_D_m_range": [float(dataset.depth_D_m.min()), float(dataset.depth_D_m.max())],
    }
    return MothraWorld(world=world, grid=Grid(world), frame=NEDFrame(),
                       dataset=dataset, field=field, meta=meta)


# ======================================================================
# 落盘 / 加载(逐位复现)
# ======================================================================
def save_world(mw: MothraWorld, path: str) -> None:
    """存为 .npz(allow_pickle=False 可读)。键与 dtype 固定。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path,
        field=mw.field,
        targets_ned=mw.dataset.targets_ned,
        depth_D_m=mw.dataset.depth_D_m,
        height_m=mw.dataset.height_m,
        type_=mw.dataset.type_,
        name=mw.dataset.name,
        waypoint_id=mw.dataset.waypoint_id,
        sequence_id=mw.dataset.sequence_id,
        lon=mw.dataset.lon,
        lat=mw.dataset.lat,
        world_xyres=np.array([mw.world.x_max_m, mw.world.y_max_m, mw.world.res_m],
                             dtype=np.float64),
        meta_json=np.array(json.dumps(mw.meta, ensure_ascii=False)),
    )


def load_world(path: str) -> MothraWorld:
    """`save_world` 的逆。不读 CSV,直接还原 —— 用于复现与跨进程传递。"""
    with np.load(path, allow_pickle=False) as d:
        x_max, y_max, res = (float(v) for v in d["world_xyres"])
        world = WorldConfig(x_max_m=x_max, y_max_m=y_max, res_m=res)
        dataset = MothraDataset(
            targets_ned=d["targets_ned"], depth_D_m=d["depth_D_m"],
            height_m=d["height_m"], type_=d["type_"], name=d["name"],
            waypoint_id=d["waypoint_id"], sequence_id=d["sequence_id"],
            lon=d["lon"], lat=d["lat"])
        meta = json.loads(str(d["meta_json"]))
        field = d["field"]
    return MothraWorld(world=world, grid=Grid(world), frame=NEDFrame(),
                       dataset=dataset, field=field, meta=meta)
