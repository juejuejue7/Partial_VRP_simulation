"""[第1层] `MothraWorld` —— 真实 Mothra 热液场的静态基础世界。

包含(D3 冻结的本步交付面):
    - NED 帧 + 栅格(`msim.geometry.grid.Grid`)
    - 28 个真实目标点(`MothraDataset`)
    - 目標分布確率マップ p(x,y)

**不含**水深地形栅格与坡度/可航行性层(D3 明确排除)。地形只以两种形式露头:
每个目标自带的 `depth_D_m`,以及 `NEDFrame.grid_index_1m` 冻结的索引口径 ——
将来要加地形层,从这两处接上即可,无需改坐标系。

修士开环:世界在 episode 内**永不变化**。save/load 逐位复现(对齐 msim 的 Q19 口径)。

本文件还提供 `build_world_for_scenario`(供 Mothra 以外的场景使用, 见其
docstring)—— 新增函数, 不改 `build_mothra_world` 的任何行为。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from msim.contracts.config import WorldConfig
from msim.contracts.env_static import GroundTruthInstance
from msim.contracts.types import Field
from msim.geometry.grid import Grid

from .contracts.config import FieldBuildConfig, MothraSimConfig, mothra_window_config
from .contracts.dataset import MothraDataset
from .contracts.frames import GRID_1M_RES_M, NEDFrame
from .dataset import load_scenario_waypoints, load_waypoints
from .field import build_probability_field

__all__ = ["MothraWorld", "build_mothra_world", "build_world_for_scenario",
          "save_world", "load_world"]

_WORLD_NPZ_VERSION = 1

# 数据层 registry 与产物根 —— 与 Bethmetory_data_process/scenarios.json /
# scripts/07_make_waypoints.py 的输出布局保持同一份真相源, 这里不复制 bbox 等
# 字段, 只读它写好的 <id>_waypoints.csv / <id>_waypoints_meta.json。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_PROC_ROOT = _REPO_ROOT / "Bethmetory_data_process"
_DATA_REGISTRY = _DATA_PROC_ROOT / "scenarios.json"
_DATA_OUTPUTS = _DATA_PROC_ROOT / "outputs"


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


def _scenario_paths(scenario_id: str) -> tuple[dict, Path, str]:
    registry = json.loads(_DATA_REGISTRY.read_text(encoding="utf-8"))
    sc = next((s for s in registry if s["id"] == scenario_id), None)
    if sc is None:
        raise KeyError(f"{_DATA_REGISTRY} 里没有 id={scenario_id!r};"
                       f"先在数据层 registry 里冻结场景, 不要在这里临时传 bbox")
    outdir = _DATA_OUTPUTS / "scenarios" / scenario_id
    prefix = scenario_id
    return sc, outdir, prefix


def build_world_for_scenario(scenario_id: str, *,
                             res_m: float = GRID_1M_RES_M,
                             field_build: Optional[FieldBuildConfig] = None,
                             window_advance_threshold_m: float = 50.0,
                             strict: bool = True) -> MothraWorld:
    """从 `Bethmetory_data_process` 的场景产物构建静态基础世界(Mothra 以外的场景)。

    与 `build_mothra_world` 的区别:
    - 目标表来自 `Bethmetory_data_process/outputs/{,scenarios/<id>/}<id>_waypoints.csv`,
      不是 `VRPSimulation/waypoints/mothra_waypoints.csv`;NED 原点/栅格跨度取该
      场景自己的 `<id>_waypoints_meta.json`(`load_scenario_waypoints`), 不是
      `frames.py` 冻结的 Mothra 常量。
    - **不做 D7 式裁切** —— 场景整幅都用(裁切是 Mothra 特有的、针对孤点/空档簇的
      人工裁决, 新场景没有这个已知问题, 默认不裁, 需要裁再另提 A0 裁决)。
    - 窗口宽度默认取该场景自己的 East 跨度(`WINDOW_WIDTH_M == 场地 East 跨度`
      的既定惯例的推广, 见 `contracts/frames.py` 顶部说明), 不是 Mothra 的 100 m。
    - 规模校验用数据层 registry 里的 `expected.n_targets`, 不是 Mothra 的 28。

    Mothra 场景请继续用 `build_mothra_world`; 这个函数只服务其它场景。
    """
    if scenario_id == "mothra":
        raise ValueError("Mothra 场景请用 build_mothra_world, 不要用这个函数"
                         "(两者刻意不共用同一条构建路径)")
    sc, outdir, prefix = _scenario_paths(scenario_id)
    csv_path = outdir / f"{prefix}_waypoints.csv"
    meta_path = outdir / f"{prefix}_waypoints_meta.json"
    if not csv_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} 或 {meta_path} 不存在;先在 Bethmetory_data_process 跑 "
            f"01/06/07_*.py --scenario {scenario_id}")

    dataset, frame = load_scenario_waypoints(
        str(csv_path), str(meta_path),
        expected_n=sc["expected"]["n_targets"] if strict else None)

    world = WorldConfig(x_max_m=frame.north_extent_m, y_max_m=frame.east_extent_m,
                        res_m=res_m)
    fb = field_build or FieldBuildConfig()
    field = build_probability_field(dataset, world, fb)
    window = mothra_window_config(window_advance_threshold_m,
                                  look_back_m=frame.east_extent_m,
                                  width_m=frame.east_extent_m)

    meta: Dict[str, Any] = {
        "version": _WORLD_NPZ_VERSION,
        "scenario_id": scenario_id,
        "scenario_label": sc["label"],
        "source_csv": csv_path.name,
        "crop": {"north_m": [0.0, frame.north_extent_m],
                 "east_m": [0.0, frame.east_extent_m],
                 "n_loaded": dataset.n, "n_kept": dataset.n,
                 "dropped_waypoint_ids": [],
                 "note": "本场景不做 D7 式裁切, 整幅使用"},
        "window": {"look_back_m": window.look_back_m, "width_m": window.width_m,
                   "advance_threshold_m": window.advance_threshold_m,
                   "lane_spacing_m": frame.east_extent_m,
                   "mode": "sliding (D8 惯例推广: 窗宽 = 场地 East 跨度)"},
        "n_targets": dataset.n,
        "n_chimney": int((dataset.type_ == "chimney").sum()),
        "n_mound": int((dataset.type_ == "mound").sum()),
        "world": {"x_max_m": world.x_max_m, "y_max_m": world.y_max_m,
                  "res_m": world.res_m, "nx": world.nx, "ny": world.ny},
        "frame": {"origin_utm9n": [frame.origin_easting_m, frame.origin_northing_m],
                  "origin_llh": [frame.origin_lon_deg, frame.origin_lat_deg, 0.0],
                  "epsg_utm": frame.epsg_utm,
                  "axes": "x=North, y=East, z=Down (msim/contracts/types.py)"},
        "field_build": {"bandwidth_m": fb.bandwidth_m, "weight_mode": fb.weight_mode},
        "ned_extent_used_m": {
            "north": [float(dataset.targets_ned[:, 0].min()),
                      float(dataset.targets_ned[:, 0].max())],
            "east": [float(dataset.targets_ned[:, 1].min()),
                     float(dataset.targets_ned[:, 1].max())]},
        "depth_D_m_range": [float(dataset.depth_D_m.min()), float(dataset.depth_D_m.max())],
    }
    return MothraWorld(world=world, grid=Grid(world), frame=frame,
                       dataset=dataset, field=field, meta=meta)


# ======================================================================
# 落盘 / 加载(逐位复现)
# ======================================================================
def save_world(mw: MothraWorld, path: str) -> None:
    """存为 .npz(allow_pickle=False 可读)。键与 dtype 固定。

    额外落盘 `mw.frame` 的标量字段(新增键, 向后兼容: 旧存档没有这些键,
    `load_world` 读不到就回退 Mothra 默认帧)—— 这样非 Mothra 场景的世界也能
    存盘复现, 不必每次都重新跑数据层链路。
    """
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
        frame_origin_e=np.float64(mw.frame.origin_easting_m),
        frame_origin_n=np.float64(mw.frame.origin_northing_m),
        frame_origin_lon=np.float64(mw.frame.origin_lon_deg),
        frame_origin_lat=np.float64(mw.frame.origin_lat_deg),
        frame_north_extent=np.float64(mw.frame.north_extent_m),
        frame_east_extent=np.float64(mw.frame.east_extent_m),
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
        if "frame_origin_e" in d.files:
            frame = NEDFrame(origin_easting_m=float(d["frame_origin_e"]),
                             origin_northing_m=float(d["frame_origin_n"]),
                             origin_lon_deg=float(d["frame_origin_lon"]),
                             origin_lat_deg=float(d["frame_origin_lat"]),
                             north_extent_m=float(d["frame_north_extent"]),
                             east_extent_m=float(d["frame_east_extent"]))
        else:
            # 旧存档(先于本次多场景改动生成)没有落盘 frame 字段 —— 那些存档
            # 一律是 Mothra, 回退默认帧(=Mothra 冻结值), 行为与改动前逐位相同。
            frame = NEDFrame()
    return MothraWorld(world=world, grid=Grid(world), frame=frame,
                       dataset=dataset, field=field, meta=meta)
