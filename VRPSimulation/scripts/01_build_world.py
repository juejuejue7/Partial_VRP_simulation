#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01 — 构建静态基础世界并落盘 + 打印校验摘要。

输入 : --scenario mothra(默认): VRPSimulation/waypoints/mothra_waypoints.csv
       --scenario <id>(其它场景): Bethmetory_data_process 的场景产物, 整幅不裁切
输出(统一落在 VRPSimulation/data/scenarios/<id>/, <id> 含 mothra 本身):
       world.npz     静态世界(目标点 + 概率场 + meta)
       basemap.npz   出图底色(可选,上游 bundle 存在时才生成)

--scenario mothra 走完整诊断(D2 椭球残差交叉校验 + Leader 测线/窗口枚举,均为
Mothra 契约冻结值的专属校验)。其它场景没有对应的契约阈值可比, 走精简流程:
只建世界、存盘、打印基本规模, 不跑那两段 Mothra 专属诊断。

运行:  D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/01_build_world.py
       D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/01_build_world.py --scenario mef
       只需 numpy(matplotlib 在 02 里才用到)。
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path 引导,必须最先 import)

import argparse
import os

import numpy as np

from vrpsim.contracts.config import (DEFAULT_DATA_DIR, UPSTREAM_BUNDLE_NPZ, CropConfig,
                                     FieldBuildConfig, MothraSimConfig, full_raster_crop)
from vrpsim.contracts.frames import (MERIDIAN_CONVERGENCE_DEG,
                                     NET_DISTANCE_ERROR_AT_SEAFLOOR_PPM, ORIGIN_LLH,
                                     ORIGIN_UTM9N, REFERENCE_SEAFLOOR_DEPTH_M,
                                     UTM_VS_ELLIPSOID_DISTANCE_PPM,
                                     UTM_VS_ELLIPSOID_MAX_RESIDUAL_M)
from vrpsim.geodesy import wgs84_to_ned_ellipsoidal
from vrpsim.viz import export_basemap
from vrpsim.windows import (enumerate_windows, first_seen_window, leader_track,
                            window_occupancy)
from vrpsim.world import build_mothra_world, build_world_for_scenario, load_world, save_world


def _pairwise(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.hypot(u[:, None] - u[None, :], v[:, None] - v[None, :])


def _build_and_save_generic(scenario_id: str, outdir: str, args) -> None:
    """非 Mothra 场景的精简流程: 建世界 → 存盘, 跳过 Mothra 契约专属诊断。"""
    from vrpsim.contracts.dataset import TARGET_TYPES  # noqa: F401 (仅用于说明, 保留可读性)

    print(f"[1/3] 构建场景 {scenario_id} 的静态世界(整幅, 不做 D7 裁切)")
    mw = build_world_for_scenario(
        scenario_id, res_m=args.res_m,
        field_build=FieldBuildConfig(bandwidth_m=args.bandwidth_m, weight_mode=args.weight_mode),
        window_advance_threshold_m=args.advance_m)
    ds = mw.dataset
    print(f"  世界 {mw.world.x_max_m:.0f}x{mw.world.y_max_m:.0f} m, 目标 {ds.n} 个:"
          f"chimney {mw.meta['n_chimney']} / mound {mw.meta['n_mound']}")
    print(f"  原点 UTM9N E{mw.frame.origin_easting_m:.1f} N{mw.frame.origin_northing_m:.1f}"
          f"  = lon {mw.frame.origin_lon_deg:.9f} / lat {mw.frame.origin_lat_deg:.9f}")
    print(f"  D(向下为正) {ds.depth_D_m.min():.2f} .. {ds.depth_D_m.max():.2f} m")
    f = mw.field
    print(f"  概率场 形状 {f.shape} 值域 {f.min():.4f} .. {f.max():.4f} 均值 {f.mean():.4f}")

    print("[2/3] 落盘")
    os.makedirs(outdir, exist_ok=True)
    world_npz = os.path.join(outdir, "world.npz")
    save_world(mw, world_npz)
    back = load_world(world_npz)
    same = (np.array_equal(back.field, mw.field)
            and np.array_equal(back.dataset.targets_ned, ds.targets_ned)
            and np.array_equal(back.dataset.type_, ds.type_))
    print(f"  {os.path.basename(world_npz):16s} {os.path.getsize(world_npz) / 1024:8.1f} KB"
          f"   round-trip 逐位相等: {same}")
    if not same:
        raise SystemExit("save/load 未逐位复现 —— 中止")

    # 场景自己的上游 bundle(Bethmetory_data_process 产出), 不是 Mothra 那份
    from vrpsim.world import _scenario_paths
    sc, data_outdir, prefix = _scenario_paths(scenario_id)
    src_bundle = str(data_outdir / f"{prefix}_bundle.npz")
    crop = CropConfig(north_m=(0.0, mw.frame.north_extent_m),
                      east_m=(0.0, mw.frame.east_extent_m))
    base_npz = os.path.join(outdir, "basemap.npz")
    got = export_basemap(base_npz, src_bundle=src_bundle, crop=crop)
    if got:
        print(f"  {os.path.basename(base_npz):16s} {os.path.getsize(base_npz) / 1024:8.1f} KB"
              f"   (仅供出图,非环境层)")
    else:
        print(f"  底图跳过:上游 {src_bundle} 不存在(可选项,不影响仿真)")

    print("[3/3] 完成。下一步:python VRPSimulation/scripts/05_plot_mission.py"
          f" --scenario {scenario_id}(先跑 04_run_mission.py --scenario {scenario_id})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mothra",
                    help="Bethmetory_data_process/scenarios.json 里的场景 id "
                         "(默认 mothra, 走完整 D2 诊断流程)")
    ap.add_argument("--res-m", type=float, default=1.0, help="栅格分辨率(默认 1.0,与水深栅格同构)")
    ap.add_argument("--bandwidth-m", type=float, default=3.0, help="KDE 带宽 sigma")
    ap.add_argument("--weight-mode", choices=("uniform", "height"), default="uniform",
                    help="概率场核权重(height 属待裁决项 2,勿设为默认)")
    ap.add_argument("--advance-m", type=float, default=50.0,
                    help="窗口推进多少米触发一次重解(默认 50 = 半窗;待裁决项 5)")
    ap.add_argument("--no-crop", action="store_true",
                    help="不裁切,建全幅 653x294 世界(仅 mothra 场景适用;溯源/对照用)")
    ap.add_argument("--outdir", default=None,
                    help="默认写 VRPSimulation/data/scenarios/<id>/")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(DEFAULT_DATA_DIR, "scenarios", args.scenario)

    if args.scenario != "mothra":
        _build_and_save_generic(args.scenario, outdir, args)
        return

    cfg = MothraSimConfig(res_m=args.res_m,
                          crop=full_raster_crop() if args.no_crop else CropConfig(),
                          window_advance_threshold_m=args.advance_m,
                          field_build=FieldBuildConfig(bandwidth_m=args.bandwidth_m,
                                                       weight_mode=args.weight_mode))

    print("[1/6] 加载 waypoint 并换算 NED")
    mw = build_mothra_world(cfg)
    ds = mw.dataset
    print(f"  目标 {ds.n} 个:chimney {mw.meta['n_chimney']} / mound {mw.meta['n_mound']}")
    print(f"  原点 UTM9N E{ORIGIN_UTM9N[0]:.1f} N{ORIGIN_UTM9N[1]:.1f}"
          f"  = lon {ORIGIN_LLH[0]:.9f} / lat {ORIGIN_LLH[1]:.9f}")
    print(f"  NED 范围  North {ds.targets_ned[:, 0].min():7.2f} .. {ds.targets_ned[:, 0].max():7.2f} m"
          f"   East {ds.targets_ned[:, 1].min():7.2f} .. {ds.targets_ned[:, 1].max():7.2f} m")
    print(f"  D(向下为正) {ds.depth_D_m.min():.2f} .. {ds.depth_D_m.max():.2f} m"
          f"   height {ds.height_m.min():.0f} .. {ds.height_m.max():.0f} m")
    cr = mw.meta["crop"]
    print(f"  裁切(D7) North[{cr['north_m'][0]:.0f},{cr['north_m'][1]:.0f}) x "
          f"East[{cr['east_m'][0]:.0f},{cr['east_m'][1]:.0f}):"
          f" 载入 {cr['n_loaded']} → 保留 {cr['n_kept']},丢弃 wp {cr['dropped_waypoint_ids']}")

    print("[2/6] 交叉校验:UTM 派生 NED vs 椭球局部切平面(D2 的代价)")
    d_utm = _pairwise(ds.targets_ned[:, 0], ds.targets_ned[:, 1])
    m = d_utm > 0.0

    def _ppm(h_m):
        n_h, e_h, _ = wgs84_to_ned_ellipsoidal(ds.lon, ds.lat, h_m)
        d = _pairwise(n_h, e_h)
        return float(np.abs(d[m] / d_utm[m] - 1.0).max() * 1e6), n_h, e_h

    # 契约口径:与 h=0 的椭球面比(纯投影性质,与水深无关)
    ppm0, n_e, e_e = _ppm(0.0)
    res_n = float(np.abs(n_e - ds.targets_ned[:, 0]).max())
    res_e = float(np.abs(e_e - ds.targets_ned[:, 1]).max())
    print(f"  位置残差 max|dN| {res_n:.3f} m / max|dE| {res_e:.3f} m"
          f"   (契约上界 {UTM_VS_ELLIPSOID_MAX_RESIDUAL_M} m)")
    print(f"  成对距离偏差 @h=0  {ppm0:.1f} ppm (契约值 {UTM_VS_ELLIPSOID_DISTANCE_PPM} ppm)"
          f"；网格北偏角 {MERIDIAN_CONVERGENCE_DEG} deg")

    # 实用口径:AUV 走在海底,半径 R-|h| 处地面距离本就短 |h|/R,与 UTM 误差反向抵消
    ppm_sea, _, _ = _ppm(-REFERENCE_SEAFLOOR_DEPTH_M)
    print(f"  成对距离偏差 @海底 h=-{REFERENCE_SEAFLOOR_DEPTH_M:.0f} m  {ppm_sea:.1f} ppm"
          f" (契约值 {NET_DISTANCE_ERROR_AT_SEAFLOOR_PPM} ppm)"
          f" = {ppm_sea * 0.1:.1f} cm/km ⇒ 全场最长边 653 m 上误差 "
          f"{ppm_sea * 1e-6 * 653 * 100:.1f} cm")

    print("[3/6] 目標分布確率マップ")
    f = mw.field
    at_t = np.array([mw.field_at(x, y) for x, y in ds.targets_ned])
    print(f"  形状 {f.shape} dtype {f.dtype}  值域 {f.min():.4f} .. {f.max():.4f}"
          f"  均值 {f.mean():.4f}")
    print(f"  bandwidth {cfg.field_build.bandwidth_m} m / weight_mode {cfg.field_build.weight_mode}")
    print(f"  目标处场值 中位 {np.median(at_t):.3f} / 最小 {at_t.min():.3f}")
    print(f"  稀疏度:>0.5 的 cell {int((f > 0.5).sum())} 个({100 * (f > 0.5).mean():.2f}%),"
          f" >0.05 的 {int((f > 0.05).sum())} 个({100 * (f > 0.05).mean():.2f}%),"
          f" 概率质量 sum={f.sum():.0f}")
    print(f"  ⚠ sigma={cfg.field_build.bandwidth_m:.0f} m 在 "
          f"{mw.world.x_max_m:.0f}x{mw.world.y_max_m:.0f} m 场上很峰化 —— "
          "若价值结算需要更宽的先验,调 --bandwidth-m(当前沿用 msim 默认)")

    print("[4/6] Leader 测线 + 滑动窗口(D8)")
    track = leader_track(mw.world, cfg.lane_spacing_m)
    total = sum(float(np.linalg.norm(track[i + 1] - track[i])) for i in range(len(track) - 1))
    print(f"  测线 {len(track) // 2} 条(lane_spacing {cfg.lane_spacing_m:.0f} m,"
          f" East={track[0][1]:.0f} m),总航程 {total:.0f} m")
    snaps = enumerate_windows(mw.world, cfg.window, ds.targets_ned)
    occ = window_occupancy(snaps)
    print(f"  窗口 {cfg.window.look_back_m:.0f}x{cfg.window.width_m:.0f} m,"
          f" 推进阈值 {cfg.window.advance_threshold_m:.0f} m ⇒ 重解 {len(snaps)} 次")
    for s, o in zip(snaps, occ):
        a, b = s.north_span
        ids = sorted(int(i) for i in ds.waypoint_id[s.target_idx])
        print(f"    #{s.index:<2d} leader N={s.leader_north_m:5.0f}  "
              f"窗口 N[{a:5.0f},{b:5.0f}]  目标 {o:2d}  {ids}")
    first = first_seen_window(snaps, ds.n)
    never = ds.waypoint_id[first < 0].tolist()
    print(f"  每个目标都至少进过一次窗口: {not never}"
          + (f"  ⚠ 漏掉 {never}" if never else ""))
    print(f"  窗口内目标数 min {occ.min()} / 中位 {int(np.median(occ))} / max {occ.max()}"
          f"；空窗 {int((occ == 0).sum())}/{len(occ)}")

    print("[5/6] 落盘")
    os.makedirs(outdir, exist_ok=True)
    world_npz = os.path.join(outdir, "world.npz")
    save_world(mw, world_npz)
    back = load_world(world_npz)
    same = (np.array_equal(back.field, mw.field)
            and np.array_equal(back.dataset.targets_ned, ds.targets_ned)
            and np.array_equal(back.dataset.type_, ds.type_))
    print(f"  {os.path.basename(world_npz):28s} {os.path.getsize(world_npz) / 1024:8.1f} KB"
          f"   round-trip 逐位相等: {same}")
    if not same:
        raise SystemExit("save/load 未逐位复现 —— 中止")

    base_npz = os.path.join(outdir, "basemap.npz")
    got = export_basemap(base_npz, crop=cfg.crop)
    if got:
        print(f"  {os.path.basename(base_npz):28s} {os.path.getsize(base_npz) / 1024:8.1f} KB"
              f"   (仅供出图,非环境层)")
    else:
        print(f"  底图跳过:上游 {UPSTREAM_BUNDLE_NPZ} 不存在(可选项,不影响仿真)")

    print("[6/6] 完成。下一步:python VRPSimulation/scripts/02_plot_world.py"
          " 与 03_plot_windows.py")


if __name__ == "__main__":
    main()
