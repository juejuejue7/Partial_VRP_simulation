#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""06 — 跑对照基线:3 台 Follower 分区 lawnmower 全覆盖扫描(D14)。

与 04 的关系:场地、目标、Follower 速度、停留时长、时钟步长**全部相同**
(配置由 `LawnmowerConfig.base` 内嵌的同一份 `MissionConfig` 派生,不是人工同步)。
唯一的差别是取消异种分工:没有 Leader、没有观测窗口、没有请求与 VRP 规划,
3 台相机机各扫 1/3 场地。

输入 : --scenario mothra(默认): VRPSimulation/waypoints/mothra_waypoints.csv
       (经 D7 裁切 → 500x100 m / 22 点)
       --scenario <id>(其它场景): Bethmetory_data_process 的场景产物, 整幅不裁切
输出(统一落在 VRPSimulation/data/scenarios/<id>/, <id> 含 mothra 本身):
           lawnmower.npz / lawnmower_summary.json / lawnmower_timeline.csv

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/06_run_lawnmower.py
      D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/06_run_lawnmower.py --scenario mef
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os

import numpy as np

from vrpsim.contracts.config import DEFAULT_DATA_DIR  # noqa: E402
from vrpsim.contracts.lawnmower import LawnmowerConfig  # noqa: E402
from vrpsim.contracts.mission import MissionConfig  # noqa: E402
from vrpsim.lawnmower import run_lawnmower, save_lawn_result  # noqa: E402
from vrpsim.world import build_mothra_world, build_world_for_scenario  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """CLI 定义。单独抽出来是为了让测试能直接读默认值 ——
    所有共享参数的默认值必须来自 `MissionConfig`,不许在这里写字面量。
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mothra",
                    help="Bethmetory_data_process/scenarios.json 里的场景 id "
                         "(默认 mothra, 走既有的 D7 裁切世界)")
    ap.add_argument("--n-vehicles", type=int, default=3, help="全覆盖扫描的 AUV 台数")
    ap.add_argument("--swath", type=float, default=6.0,
                    help="相机横向幅宽 [m];6.0 = msim SensorConfig.footprint_lateral_m"
                         "(取 5.4 = 0.9x footprint 的走廊宽会更慢,见契约【待裁决 L1】)")
    # ⚠ 这三个默认值**从契约取**,不写字面量。两个场景必须共用同一份运动学
    #   (D14 的"配置相同是结构保证"),在这里手抄一份数字就等于开了第二个真值源 ——
    #   D15 把停留从 5 改到 10 时,这里正是漏掉的那处。
    _MC = MissionConfig()
    ap.add_argument("--dwell", type=float, default=_MC.dwell_time_s,
                    help=f"扫到目标后原地停留 [s];默认 {_MC.dwell_time_s:.0f} = 与 VRP 场景同值")
    ap.add_argument("--follower-speed", type=float, default=_MC.follower_speed_mps)
    ap.add_argument("--dt", type=float, default=_MC.dt_s)
    ap.add_argument("--no-boustrophedon", action="store_true",
                    help="不做牛耕式往复,每条测线都从南端起(含空回航,更慢)")
    ap.add_argument("--max-time", type=float, default=20000.0,
                    help="仿真时钟上限 [s];lawnmower 比 VRP 慢数倍,默认放宽")
    ap.add_argument("--out", default=None,
                    help="默认写 data/scenarios/<id>/lawnmower.npz")
    ap.add_argument("-q", "--quiet", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    # base 里只动"两个场景本就共有"的参数;lawnmower 特有的参数在外层。
    base = MissionConfig(dwell_time_s=args.dwell,
                         follower_speed_mps=args.follower_speed,
                         dt_s=args.dt,
                         max_mission_time_s=args.max_time)
    cfg = LawnmowerConfig(base=base,
                          n_vehicles=args.n_vehicles,
                          swath_width_m=args.swath,
                          boustrophedon=not args.no_boustrophedon)

    out = args.out
    if out is None:
        out = os.path.join(DEFAULT_DATA_DIR, "scenarios", args.scenario, "lawnmower.npz")

    print(f"[1/3] 构建场景 {args.scenario}"
          + ("(与 04 同一份世界)" if args.scenario == "mothra" else "(整幅, 不做 D7 裁切)"))
    mw = (build_mothra_world(cfg.sim) if args.scenario == "mothra"
         else build_world_for_scenario(args.scenario))
    print(f"  世界 {mw.world.x_max_m:.0f}x{mw.world.y_max_m:.0f} m,"
          f" 目标 {mw.dataset.n} 个(丢弃 wp {mw.meta['crop']['dropped_waypoint_ids']})")
    print(f"  {cfg.n_vehicles} 台 Follower,速度 {cfg.follower_speed_mps} m/s,"
          f" 幅宽 {cfg.swath_width_m} m,停留 {cfg.dwell_time_s:.0f} s"
          f"{'(牛耕式往复)' if cfg.boustrophedon else '(单向 + 空回航)'}")

    print("[2/3] 仿真")
    res = run_lawnmower(cfg, mw, verbose=not args.quiet)

    s = res.summary()
    strips = res.meta["strips_east_m"]
    print("\n  ---- 测线布置 ----")
    for i, lanes in enumerate(res.meta["lane_easts"]):
        print(f"  V{i}  条带 East [{strips[i][0]:6.2f}, {strips[i][1]:6.2f})  "
              f"{len(lanes):2d} 条测线  East = "
              f"{', '.join(f'{x:.2f}' for x in lanes)}")
    print(f"  测线间距 {s['lane_spacing_m']:.3f} m ≤ 幅宽 {s['swath_width_m']:.1f} m"
          f" ⇒ 带内无缝隙")

    print("\n  ---- 结果 ----")
    print(f"  **任务完成时刻 {s['t_complete_s']:.1f} s**"
          f"(= 最后一台走完全部测线;各台 "
          f"{[round(x, 1) for x in s['finish_time_s']]} s)")
    print(f"  最后一个目标被観測于 {s['t_last_observation_s']:.1f} s"
          f"  —— 之后仍须走完剩余测线才能宣称全覆盖")
    print(f"  覆盖 {s['visited']}/{s['n_targets']} = {100 * s['coverage']:.1f}%")
    if s["missed_wp_ids"]:
        print(f"  ⚠ 未観測 wp {s['missed_wp_ids']}")
    print(f"  各车航程 {[round(x, 1) for x in s['per_vehicle_distance_m']]} m"
          f"   観測数 {s['per_vehicle_visits']}")
    print(f"  min-max(最长单机航程) {s['max_distance_m']:.1f} m   "
          f"总航程 {s['total_distance_m']:.1f} m   Jain {s['jain_fairness']:.3f}")

    ok = _check_invariants(res)
    print(f"  不变量自检: {'全部通过' if ok else '**有违反,见上**'}")

    print("[3/3] 落盘")
    save_lawn_result(res, out)
    base_p = os.path.splitext(out)[0]
    for p in (out, base_p + "_summary.json", base_p + "_timeline.csv"):
        print(f"  {os.path.basename(p):32s} {os.path.getsize(p) / 1024:8.1f} KB")


def _check_invariants(res) -> bool:
    """跑完就地自检:全覆盖 / 无重复観測 / 航程与理论测线长度自洽。"""
    ok = True
    if res.coverage < 1.0:
        print(f"  ! 全覆盖扫描却漏了 wp {res.missed_wp_ids} —— 测线几何有问题")
        ok = False
    ids = res.waypoint_ids[res.visited_mask]
    if len(set(ids.tolist())) != len(ids):
        print("  ! 有目标被観測多次")
        ok = False
    # 航程 = 测线长 x 条数 + 换线横移 x (条数-1);停留不产生位移
    L = res.meta["survey_line_length_m"]
    sp = res.meta["lane_spacing_m"]
    for i, lanes in enumerate(res.meta["lane_easts"]):
        n = len(lanes)
        want = n * L + (n - 1) * sp
        if not res.cfg.boustrophedon:
            want += n * L                       # 空回航
        got = float(res.per_vehicle_distance_m[i])
        if abs(got - want) > 1e-6:
            print(f"  ! V{i} 航程 {got:.3f} m 与测线几何 {want:.3f} m 不符")
            ok = False
    return ok


if __name__ == "__main__":
    main()
