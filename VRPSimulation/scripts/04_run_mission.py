#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""04 — 跑一次协同探査任务仿真(Leader 广域扫描 + 双 Follower 请求式分区域 VRP)。

输入 : VRPSimulation/waypoints/mothra_waypoints.csv(经 D7 裁切 → 500x100 m / 22 点)
输出 : VRPSimulation/data/mission.npz              逐时刻时间线(规格 §7)
       VRPSimulation/data/mission_assignments.json 每次规划的下发序列 + 摘要
       VRPSimulation/data/mission_timeline.csv     人读版宽表

运行: D:/nixingxing/Anaconda/envs/auv_py310/python.exe VRPSimulation/scripts/04_run_mission.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os

import numpy as np

from vrpsim.contracts.config import DEFAULT_DATA_DIR  # noqa: E402
from vrpsim.contracts.mission import MissionConfig  # noqa: E402
from vrpsim.mission import feasibility_estimate, run_mission, save_result  # noqa: E402
from vrpsim.world import build_mothra_world  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """CLI 定义。单独抽出来是为了让测试能直接读默认值 ——
    所有共享参数的默认值必须来自 `MissionConfig`,不许在这里写字面量。
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ⚠ 默认值一律**从契约取**,不写字面量:`contracts/mission.py::MissionConfig` 是
    #   唯一真值源,在这里手抄数字等于开第二个源,改契约时必漏。
    d = MissionConfig()
    ap.add_argument("--window-look-back", type=float, default=d.window_look_back_m,
                    help="窗口沿测线长度 [m]")
    ap.add_argument("--window-width", type=float, default=d.window_width_m,
                    help="窗口横向宽度 [m]")
    ap.add_argument("--acoustic-hop", type=float, default=d.acoustic_hop_s,
                    help="单跳声学传输耗时 [s](传播+包时长+保护间隔)")
    ap.add_argument("--usbl-period", type=float, default=d.usbl_period_s,
                    help="USBL 定位周期 [s]，一个循环 = 2N 跳。**独立于规划周期**")
    ap.add_argument("--plan-period", type=float, default=d.plan_period_s,
                    help="路径规划周期 [s]。独立于 USBL，拉长它做对比实验(30/45/60…)")
    ap.add_argument("--first-plan-at", type=float, default=d.first_plan_at_s,
                    help="第一轮规划的起点 [s];默认 0")
    ap.add_argument("--no-wait-lagging", action="store_true",
                    help="关掉判据 A(队友落到窗口后沿之后就停船)")
    ap.add_argument("--no-wait-endangered", action="store_true",
                    help="关掉判据 C(窗口里还有未分配目标要掉出后沿就停船)")
    ap.add_argument("--wait-release-margin", type=float,
                    default=d.leader_wait_release_margin_m,
                    help="判据 A 的迟滞裕度 [m]，防逐拍停走振荡")
    ap.add_argument("--max-seq", type=int, default=d.max_sequence_len,
                    help="一次下发至多几个 waypoint")
    ap.add_argument("--dwell", type=float, default=d.dwell_time_s,
                    help="到点定点観測停留 [s]")
    ap.add_argument("--leader-speed", type=float, default=d.leader_speed_mps,
                    help="Leader 测线速度 [m/s]。⚠ D16 起 Leader 会停船等队友,这只是"
                         "**不停船时**的速度;实际完成时刻 = 测线长/它 + 累计停船")
    ap.add_argument("--follower-speed", type=float, default=d.follower_speed_mps)
    ap.add_argument("--dt", type=float, default=d.dt_s)
    ap.add_argument("--solver", choices=("ortools", "greedy"), default=d.solver)
    ap.add_argument("--nav-drift", type=float, default=d.nav_drift_frac_of_distance,
                    help="Leader 对队友位置的认知误差,按行进距离比例(0.005=0.5%%);"
                         "默认 0 = 解析确定量,非估计")
    ap.add_argument("--seed", type=int, default=d.seed, help="漂移方向的随机种子")
    ap.add_argument("--vrp-time-limit", type=float, default=d.vrp_time_limit_s,
                    help="ortools 每次求解的时间预算 [s]。⚠ 实测求解耗时几乎总会吃满"
                         "这个预算,想看问题难度的真实差异要把它调小")
    ap.add_argument("--max-time", type=float, default=d.max_mission_time_s,
                    help="仿真时钟上限 [s];跑到这里还没正常终止会明确告警")
    ap.add_argument("--no-dwell-at-projection", action="store_true",
                    help="投影点不停留(它不是观测目标)")
    ap.add_argument("--out", default=os.path.join(DEFAULT_DATA_DIR, "mission.npz"))
    ap.add_argument("-q", "--quiet", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    cfg = MissionConfig(
        window_look_back_m=args.window_look_back,
        window_width_m=args.window_width,
        acoustic_hop_s=args.acoustic_hop,
        usbl_period_s=args.usbl_period,
        plan_period_s=args.plan_period,
        first_plan_at_s=args.first_plan_at,
        leader_wait_on_lagging_follower=not args.no_wait_lagging,
        leader_wait_on_endangered_target=not args.no_wait_endangered,
        leader_wait_release_margin_m=args.wait_release_margin,
        max_sequence_len=args.max_seq,
        dwell_time_s=args.dwell,
        leader_speed_mps=args.leader_speed,
        follower_speed_mps=args.follower_speed,
        dt_s=args.dt,
        solver=args.solver,
        vrp_time_limit_s=args.vrp_time_limit,
        max_mission_time_s=args.max_time,
        nav_drift_frac_of_distance=args.nav_drift,
        seed=args.seed,
        dwell_at_projection=not args.no_dwell_at_projection,
    )

    print("[1/4] 构建场景")
    mw = build_mothra_world(cfg.sim)
    print(f"  世界 {mw.world.x_max_m:.0f}x{mw.world.y_max_m:.0f} m,"
          f" 目标 {mw.dataset.n} 个(丢弃 wp {mw.meta['crop']['dropped_waypoint_ids']})")
    print(f"  Leader 起点 {cfg.leader_start_ned}  速度 {cfg.leader_speed_mps} m/s")
    for i, s in enumerate(cfg.follower_starts_ned):
        print(f"  Follower{i} 起点 {s}  速度 {cfg.follower_speed_mps} m/s")
    print(f"  窗口 {cfg.window_look_back_m:.0f}x{cfg.window_width_m:.0f} m(前边界贴 Leader)")
    print(f"  声学单跳 {cfg.acoustic_hop_s:.1f} s;USBL 定位周期 "
          f"{cfg.usbl_period_s:.0f} s(循环 {cfg.usbl_cycle_s:.0f} s = "
          f"{2 * cfg.n_followers} 跳 + 静默 {cfg.usbl_period_s - cfg.usbl_cycle_s:.0f} s)")
    print(f"  规划周期 {cfg.plan_period_s:.0f} s(**独立于定位**),下发时延 "
          f"{cfg.plan_latency_s:.0f} s;一次至多 {cfg.max_sequence_len} 点;"
          f"定点観測停留 {cfg.dwell_time_s:.0f} s")
    waits = [n for n, on in (("A 落后队友", cfg.leader_wait_on_lagging_follower),
                             ("C 濒危目标", cfg.leader_wait_on_endangered_target)) if on]
    print(f"  Leader 等待策略: {' + '.join(waits) if waits else '关(匀速 Leader)'}"
          f"{f'  前瞻 {cfg.wait_lookahead_s:.0f} s / 迟滞 {cfg.leader_wait_release_margin_m:.0f} m' if waits else ''}")

    print("[2/4] 事前可行性判据(必要条件,乐观下界)")
    fe = feasibility_estimate(mw, cfg)
    print(f"  全局 min-max VRP 最长单机路线 {max(fe['route_lengths_m']):.0f} m"
          f"(分配 {fe['route_counts']} 个点)")
    print(f"  完成全部观测至少需 {fe['t_route_reference_s']:.0f} s;"
          f" Leader 不停船走完测线要 {fe['t_leader_finish_s']:.0f} s")
    print(f"  ⇒ 任务时间下界 = max(两者) = {fe['t_mission_reference_s']:.0f} s"
          f"  (第一条序列 {fe['t_first_broadcast_s']:.0f} s 才落地)")
    if not fe["wait_required"]:
        print("  Leader 不必等队友:走完测线时观测本就能完成。")
    elif cfg.leader_waits:
        print(f"  ⚠ **Leader 必然要停船等队友**:观测下界比 Leader 走完测线晚 "
              f"{fe['t_route_reference_s'] - fe['t_leader_finish_s']:.0f} s。")
        print("     等待策略开着 ⇒ 窗口不会提前冻结,覆盖率约束被换成时间约束。")
    else:
        print(f"  ⇒ ⚠ **原理上不可能全覆盖**:等待策略被关掉,Leader 走完后窗口冻结在"
              f"最北 {cfg.window_look_back_m:.0f} m,更南的目标永久错过。")
        print(f"     要么打开等待策略,要么把 Leader 降到 "
              f"≤{fe['max_leader_speed_mps']:.2f} m/s,要么加长窗口/提 Follower 速度。")

    print("[3/4] 仿真")
    res = run_mission(cfg, mw, verbose=not args.quiet)

    s = res.summary()
    print("\n  ---- 结果 ----")
    print(f"  完成时刻 {s['t_complete_s']:.1f} s "
          f"(全域扫遍 {s['t_leader_finish_s']:.1f} s / 末次観測 "
          f"{s['t_last_observation_s']:.1f} s);时间线长 {s['duration_s']:.0f} s")
    print(f"  Leader 停船 {s['leader_hold_total_s']:.1f} s"
          f"({100 * s['leader_hold_frac']:.1f}% 的时间)"
          f"  —— A 落后队友 {s['leader_hold_lagging_s']:.1f} s / "
          f"C 濒危目标 {s['leader_hold_endangered_s']:.1f} s(两条可同时成立,不相加)")
    print(f"  规划 {s['n_plan_rounds']} 轮(周期 {s['plan_period_s']:.0f} s) / "
          f"USBL 定位 {s['n_requests'] // max(cfg.n_followers, 1)} 循环"
          f"(周期 {s['usbl_period_s']:.0f} s,规划时刻定位陈旧度最大 "
          f"{s['fix_age_max_s']:.1f} s) / 声学报文 {s['n_comm_messages']} 条")
    print(f"  平均求解耗时 {1e3 * s['solve_wall_mean_s']:.0f} ms"
          f"  ⚠ 挂钟量,不进仿真时间轴")
    print(f"  覆盖 {s['visited']}/{s['n_targets']} = {100 * s['coverage']:.1f}%")
    if s["timed_out"]:
        print(f"  ⚠ **跑到 max_mission_time_s = {cfg.max_mission_time_s:.0f} s 才停**"
              f" —— 不是正常终止,结果不可信")
    if s["missed_wp_ids"]:
        print(f"  ⚠ 未观测 wp {s['missed_wp_ids']}")
    print(f"  各 Follower 航程 {[round(x, 1) for x in s['per_follower_distance_m']]} m"
          f"   观测数 {s['per_follower_visits']}")
    print(f"  min-max(最长单机航程) {s['max_distance_m']:.1f} m   "
          f"总航程 {s['total_distance_m']:.1f} m   Jain {s['jain_fairness']:.3f}")
    if cfg.nav_drift_frac_of_distance > 0:
        print(f"  Leader 对队友位置的**估计**误差 max {s['shadow_error_max_m']:.2f} m / "
              f"mean {s['shadow_error_mean_m']:.2f} m"
              f"  (漂移 {100 * cfg.nav_drift_frac_of_distance:.1f}%/行进距离)")
    else:
        print(f"  Leader 对队友位置为**解析确定量**(承诺队列推算),"
              f"与真值残差 {s['shadow_error_max_m']:.1e} m —— 仅浮点噪声")

    _print_solve_table(res)

    ok = _check_invariants(res)
    print(f"  不变量自检: {'全部通过' if ok else '**有违反,见上**'}")

    print("[4/4] 落盘")
    save_result(res, args.out)
    base = os.path.splitext(args.out)[0]
    for p in (args.out, base + "_assignments.json", base + "_timeline.csv"):
        print(f"  {os.path.basename(p):32s} {os.path.getsize(p) / 1024:8.1f} KB")


def _print_solve_table(res) -> None:
    """逐次求解:问题规模 vs 实测耗时。

    ⚠ 耗时是**挂钟量**,取决于宿主机与 `--vrp-time-limit`,不进入仿真时间轴。
      ortools 的元启发式几乎总会吃满时间预算 ⇒ 这一列主要反映的是**预算**;
      真正反映难度的是 `池` 那一列。想看难度差异请把 `--vrp-time-limit` 调小。
    """
    rounds = [r for r in res.plan_rounds if r.pool_size > 0]
    if not rounds:
        return
    print("\n  ---- 逐次求解:池大小 vs 实测耗时 ----")
    print(f"  {'轮':>4} {'规划时刻[s]':>11} {'池':>4} {'下发':>6} {'耗时[ms]':>10}  求解器")
    for r in rounds:
        print(f"  {r.round_idx:>4} {r.t_plan_s:>11.1f} {r.pool_size:>4} "
              f"{r.n_assigned_total:>6} {1e3 * r.solve_wall_s:>10.3f}  {r.solver}")
    # 按池大小汇总,看规模与耗时的关系
    by_size = {}
    for r in rounds:
        by_size.setdefault(r.pool_size, []).append(r.solve_wall_s)
    print(f"  {'-' * 46}")
    print(f"  {'池大小':>6} {'轮数':>6} {'平均耗时[ms]':>13}")
    for k in sorted(by_size):
        v = by_size[k]
        print(f"  {k:>6} {len(v):>6} {1e3 * sum(v) / len(v):>13.3f}")
    allw = [r.solve_wall_s for r in rounds]
    print(f"  {'全部':>6} {len(allw):>6} {1e3 * sum(allw) / len(allw):>13.3f}"
          f"   ← 平均求解耗时")


def _check_invariants(res) -> bool:
    """跑完就地自检:重复观测 / 同点撞车 / 序列超长 / 握手时序。"""
    ok = True
    cfg = res.cfg
    tau = cfg.acoustic_hop_s

    # --- 声学时序:USBL 与规划两条独立时钟(D15 / D16) ------------------
    from vrpsim.contracts.mission import MSG_BROADCAST

    if res.comm_events.size:
        flight = res.comm_events[:, 1] - res.comm_events[:, 0]
        if not np.allclose(flight, tau, atol=1e-9):
            print(f"  ! 有报文的飞行时间不等于单跳 {tau} s"); ok = False
        n_bc = int((res.comm_events[:, 2] == MSG_BROADCAST).sum())
        if n_bc != len(res.plan_rounds):
            print(f"  ! 广播 {n_bc} 条 != 规划 {len(res.plan_rounds)} 轮"); ok = False
        n_usbl = res.comm_events.shape[0] - n_bc
        if n_usbl % (2 * cfg.n_followers):
            print(f"  ! USBL 报文 {n_usbl} 条不是 {2 * cfg.n_followers} 的整数倍"
                  f"(有循环没走完)"); ok = False
    for r in res.plan_rounds:
        if not np.isclose(r.t_deliver_s - r.t_plan_s, cfg.plan_latency_s, atol=1e-9):
            print(f"  ! 轮{r.round_idx}: 下发时延不是 {cfg.plan_latency_s} s"); ok = False
        if r.fix_age_s and max(r.fix_age_s) > cfg.usbl_period_s + 1e-9:
            print(f"  ! 轮{r.round_idx}: 定位陈旧度 {max(r.fix_age_s):.1f} s "
                  f"超过 USBL 周期 {cfg.usbl_period_s} s"); ok = False
    # 相邻两轮规划不许套娃
    for a, b in zip(res.plan_rounds, res.plan_rounds[1:]):
        if b.t_plan_s < a.t_deliver_s - 1e-9:
            print(f"  ! 轮{a.round_idx} 的序列还没落地就开了轮{b.round_idx}"); ok = False

    # --- Leader 停船(D16) ----------------------------------------------
    dt = float(res.meta["dt_s"])
    moved = np.diff(res.leader_north_m) > 1e-12
    if np.any(moved & res.leader_holding[:-1]):
        print("  ! Leader 标着停船却仍在前进"); ok = False
    # 走完测线的时刻应当 = 匀速用时 + 累计停船(浮点余量给一拍)
    track = float(res.leader_north_m.max())
    want = track / cfg.leader_speed_mps + res.leader_hold_total_s
    if abs(res.t_leader_finish_s - want) > dt + 1e-6:
        print(f"  ! Leader 完成时刻 {res.t_leader_finish_s:.1f} s 与"
              f" 匀速{want - res.leader_hold_total_s:.0f}+停船"
              f"{res.leader_hold_total_s:.0f} 对不上"); ok = False
    # 同一轮内不许把同一个 waypoint 派给两台
    for r in res.plan_rounds:
        same = [a for a in res.assignments if a.t_s == r.t_plan_s]
        flat = [w for a in same for w in a.wp_ids]
        if len(flat) != len(set(flat)):
            print(f"  ! 轮{r.round_idx}: 同一 waypoint 派给了两台 {flat}"); ok = False
    # 上报的占用点与影子推算不一致(队列进度本应是精确的)
    mm = res.meta.get("report_mismatch_count", 0)
    if mm:
        print(f"  ! 上报占用点与影子推算不一致 {mm} 次"); ok = False

    for a in res.assignments:
        if a.n_real > res.cfg.max_sequence_len:
            print(f"  ! t={a.t_s}: 下发 {a.n_real} 个,超过上限"); ok = False
        if a.has_projection and a.n_real >= res.cfg.max_sequence_len:
            print(f"  ! t={a.t_s}: 已满 {a.n_real} 个还补了投影点"); ok = False
    # 每个目标至多被观测一次
    ids, counts = np.unique(res.waypoint_ids[res.visited_mask], return_counts=True)
    if counts.size and counts.max() > 1:
        print(f"  ! 有目标被观测多次: {ids[counts > 1].tolist()}"); ok = False
    # 两台不会同时占用同一个 waypoint
    occ = res.follower_occupied
    for k in range(occ.shape[0]):
        row = [v for v in occ[k] if v >= 0]
        if len(row) != len(set(row)):
            print(f"  ! t={res.t_s[k]}: 两台占用同一 waypoint {row}"); ok = False
            break
    return ok


if __name__ == "__main__":
    main()
