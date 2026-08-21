"""[A0 集成测试用例] 一次完整的 **滑动观测窗口(windowed)** 协同探查任务(端到端,联合全模块)。

本文件 = `mission_baseline_survey.py` 的窗口化变体:**基本任务流/模块调用保持不变**,只把
"Leader 先揭示整张全局先验图 → 再一次性全局规划"改成"按 20×20 观测窗口分块、边扫边规划边观测"。

与 baseline 的差异(仅四点,逐条对应人类需求):
  1) **分块揭示**:全局地图 60×20,观测窗口 = cfg.window(look_back_m=20、width_m=20=全 East)。
     Leader 沿 North 走单条测线;**每扫完一个 20m 块**(leader 的 North 位置越过该块右边界 → 其
     后方 20m 观测窗口恰好覆盖该块)即"揭示"当前窗口内容,**窗口不回退、其它块不更新**,Leader
     继续前进模拟继续扫描下一块。⚠ 仍是修士开环:先验值是静态真值,只"揭示/巡查",不在线重估。
  2) **窗口内规划**:路径规划输入范围由"全局"改为"**当前 20×20 观测窗口**"——峰值检测、概率
     补点(A′ 混合)、VRP 给 waypoint 全部只在当前窗口块内完成(用块掩码把窗口外置 0)。
     ⚠ 候选去留用**固定绝对阈值**(模块顶部 PEAK/FILL_ABS_THRESHOLD 旋钮),**不锚定任何 max**:
       既不偷看全局 field.max()(=用未揭示区信息,违反开环),也不用各窗口局部 max(否则空/弱
       窗口会把 ≈0.005 的 KDE 拖尾放大成伪候选,Follower 白扫整窗,鲁棒性差)→ 空窗口零候选。
  3) **即时派遣**:给出当前窗口的 waypoint 后,Follower **立即出发**前往观测该窗口。
  4) **Leader/Follower 并行**:Leader 给出第 k 窗口后继续向前扫描;扫完第 k+1 个 20m 块即把观测窗口
     推进到下一块,重复 2)/3),直至 60×20 全局调查完成。
     ⚠ 实现说明:先验为静态、开环(观测不回写规划器),故"并行时间线"对最终覆盖/TP/FN/FP 等量
       与"按窗口顺序处理"等价 → 本脚本按窗口顺序逐块处理;Follower **跨窗口连续行动**(每个窗口
       从上一窗口的终端位姿接续出发),如实体现"边扫边观测、持续推进"的并行语义。
     ⚠ 纪律4:窗口推进触发点(leader 越过块边界)是"由 Leader 实际动力学位姿 + 窗口几何解析判定
       的确定量",非估计。

运行:D:\\nixingxing\\Anaconda\\envs\\auv_py310\\python.exe examples/mission_windows_survey.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataclasses import replace

from msim.contracts.config import (SimConfig, WorldConfig, SensorConfig, ObsCandidateConfig,
                                   resolve_obs_candidate_params, resolve_obs_candidate_abs_thresholds)
from msim.geometry.grid import Grid
from msim.geometry.window import get_window, window_cells
from msim.env_static.leader_path import lawnmower_waypoints
from msim.env_static.field import detections_from_field, observation_candidates_from_field
from msim.physics.leader import LeaderActor
from msim.physics.follower import Follower
from msim.physics.rollout import Level2RolloutEngine
from msim.task.corridor import corridor_cells
from msim.baselines.greedy import _start_poses
from msim.baselines.ortools_vrp import solve_minmax_vrp
from msim.eval.metrics import max_energy, routes_to_energies, total_energy
from msim.eval.runner import default_instance

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def _route_corridor_union(start_xy, route, width_m, grid) -> set:
    """一台 Follower 沿 [start → route...] 的走廊覆盖 cell 并集。"""
    pts = [np.asarray(start_xy, float)[:2]] + [np.asarray(p, float) for p in route]
    cov: set = set()
    for a, b in zip(pts[:-1], pts[1:]):
        cov |= corridor_cells(a, b, width_m, grid)
    return cov


def _point_seg_dist(p, a, b) -> float:
    """点 p 到线段 a-b 的垂直距离(投影裁到段内)。"""
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0.0:
        return float(np.hypot(*(p - a)))
    t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.hypot(*(p - (a + t * ab))))


def _douglas_peucker(pts, tol):
    """Douglas-Peucker 折线简化:保留偏离首尾连线 > tol 的转折点,移除其余;返回保留点(含首尾)。"""
    if len(pts) < 3:
        return list(pts)
    a, b = pts[0], pts[-1]
    dmax, idx = -1.0, 0
    for i in range(1, len(pts) - 1):
        d = _point_seg_dist(pts[i], a, b)
        if d > dmax:
            dmax, idx = d, i
    if dmax > tol:
        return _douglas_peucker(pts[:idx + 1], tol)[:-1] + _douglas_peucker(pts[idx:], tol)
    return [a, b]


def _simplify_route(start_xy, route, tol):
    """§v1.0.11 去冗余导航点:移除落在前后连线 tol(=走廊半宽)内、AUV 自然会经过的中间点,只留
    转折点 → 导航点变疏、避免密集点(点距<转弯半径2.86m)处绕圈(弹簧轨迹)。被移除点偏离简化连线
    < tol < 走廊半宽 → 仍被实际走廊覆盖,覆盖不丢。返回简化导航序列(不含 start)。"""
    if not route:
        return []
    pts = [np.asarray(start_xy, float)[:2]] + [np.asarray(p, float) for p in route]
    return _douglas_peucker(pts, tol)[1:]


def _route_len(sp, r):
    pts = [np.asarray(sp, float)[:2]] + [np.asarray(p, float) for p in r]
    return float(sum(np.hypot(*(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)))


def main() -> int:
    print("=" * 78)
    print("完整 windowed(滑动观测窗口)协同探查任务(端到端集成测试)")
    print("=" * 78)

    # ---- 配置:与 baseline 完全一致(同 seed/同场景),保证可对照 ----
    _foll_ov = dict(SimConfig().follower_overrides)
    _foll_ov["arrive_radius_m"] = 0.5
    cfg = replace(SimConfig(),
                  world=WorldConfig(x_max_m=60.0, y_max_m=20.0, res_m=2.0),
                  n_followers=1, seed=42,
                  sensor=SensorConfig(footprint_along_m=2.0, footprint_lateral_m=2.0),
                  follower_overrides=_foll_ov,
                  # §v1.0.12:候选【固定绝对阈值】(单一真相源 cfg.obs_candidate;不锚定任何 max)。初值
                  #   0.5/0.7 = baseline 全局等效口径;空窗口零候选(拖尾≈0.005<0.5),弱窗口峰高保留
                  #   (seed=9 实测 W0=0.618/W1=1.000/W2=0.925)。识别算法接入后在此标定契合值。
                  obs_candidate=ObsCandidateConfig(fill_radius_m=1.0,
                                                   peak_abs_threshold=0.3, fill_abs_threshold=0.5))
    world, grid = cfg.world, Grid(cfg.world)
    N = cfg.n_followers

    # ===== 真值 + 先验图(ground truth;Leader 巡查将分块揭示)=====
    inst = default_instance(cfg, n_targets=9)         # 9 个真实目标点(3 簇),先验=其 KDE(σ=3m)
    targets = inst.targets
    field = inst.field
    K = len(targets)

    # ===== 观测窗口分块:沿 North 每 look_back_m 一块(60/20 = 3 块)=====
    look = float(cfg.window.look_back_m)               # = 20m,窗口沿测线回看长度
    n_windows = int(round(world.x_max_m / look))       # = 3 个 20×20 观测窗口块
    edges = [k * look for k in range(n_windows + 1)]   # 块边界 [0,20,40,60]

    def _block_mask(k: int) -> np.ndarray:
        """块 k 的细网格掩码:cell 中心 North 坐标落在 [edges[k], edges[k+1]) 内为 True。"""
        m = np.zeros((grid.nx, grid.ny), dtype=bool)
        xc = (np.arange(grid.nx) + 0.5) * world.res_m
        sel = (xc >= edges[k]) & (xc < edges[k + 1])
        m[sel, :] = True
        return m
    block_masks = [_block_mask(k) for k in range(n_windows)]

    # ===== Phase 0:部署(同区,空白地图)=====
    # 单条 lawnmower 测线(lane_spacing=20 → y_max=20 仅一条,y=10),Leader 沿 North 直走全程。
    lawn = [np.asarray(w, float) for w in lawnmower_waypoints(world, lane_spacing_m=20.0)]
    # lead-in/lead-out 外扩(同 baseline):每条测线端点沿测线方向外伸 over_run_m,补满边缘揭示。
    arrive_r = float(cfg.leader_overrides.get("arrive_radius_m", 0.5))
    over_run_m = arrive_r + 2.0
    for _k in range(0, len(lawn) - 1, 2):
        _a, _b = lawn[_k], lawn[_k + 1]
        _d = _b - _a
        _n = float(np.hypot(_d[0], _d[1]))
        if _n > 0.0:
            _u = _d / _n
            lawn[_k] = _a - _u * over_run_m
            lawn[_k + 1] = _b + _u * over_run_m
    leader = LeaderActor(cfg, lawn)
    foll_starts = _start_poses(cfg)                   # N 台 Follower 起始位姿(x≈0 起始边)
    print(f"\n[Phase 0] 部署:Leader@{np.round(leader.pose[:2],1)} + "
          f"{N} Followers@{[np.round(s[:2],1).tolist() for s in foll_starts]}"
          f"  —— 同区(起始边 x≈0);地图初始无信息(已巡查=0)。")
    print(f"          真值:{K} 个目标点(3 簇);先验图 = 目标 KDE(σ=3m),按窗口分 {n_windows} 块逐块揭示。")
    print(f"          观测窗口 = {look:.0f}×{cfg.window.width_m:.0f} m;块边界(North)= {edges}。")

    # ===== A′ 候选提取参数(单一真相源:cfg.obs_candidate)=====
    thr_rel, fill_thr, peak_sep, fill_R = resolve_obs_candidate_params(cfg)
    peak_abs, fill_abs = resolve_obs_candidate_abs_thresholds(cfg)  # §v1.0.12 固定绝对阈值(鲁棒)

    # ===== 执行器:Follower 跨窗口连续行动(各自维护当前位姿,链式接续)=====
    foll = Follower(cfg)
    cam = foll.sensor                                 # CameraSensorModel(p_d, p_fa, corridor_width)
    width = cam.corridor_width_m
    eng2 = Level2RolloutEngine(cfg, auv_type="follower", record_trajectory=True)

    foll_pose = [np.asarray(s, float).copy() for s in foll_starts]   # 各 Follower 当前位姿(链式)
    coverage: set = set()                              # 累积走廊覆盖(跨所有窗口)
    revealed = np.zeros((grid.nx, grid.ny), dtype=bool)       # 已揭示细网格(累积)
    reveal_window = np.full((grid.nx, grid.ny), -1, dtype=int)  # 每 cell 由哪个窗口揭示(可视化用)
    leader_track = [leader.pose[:2].copy()]
    windows = []                                       # 每窗口的规划/执行记录

    def process_window(k: int) -> None:
        """块 k 已被 Leader 扫完 → 在当前 20×20 观测窗口内规划并派遣 Follower 观测。"""
        bmask = block_masks[k]
        # 仅当前窗口块内、已揭示的先验值进入规划(窗口外 = 0,不更新其它块)。
        belief_k = np.where(revealed & bmask, field, 0.0)
        # 候选几何生成:峰值锚点(局部极大) + A′ 覆盖补点(高概率区栅格)。
        peaks0 = detections_from_field(belief_k, world, threshold_rel=thr_rel, min_separation_m=peak_sep)
        dets0 = observation_candidates_from_field(belief_k, world, threshold_rel=thr_rel,
                                                  fill_threshold_rel=fill_thr,
                                                  peak_min_separation_m=peak_sep, fill_radius_m=fill_R)
        fills0 = dets0[len(peaks0):]
        # ⚠ 鲁棒性:用【固定绝对阈值】(单一真相源 cfg.obs_candidate,不锚定任何 max)筛掉低于阈值的候选。
        #   空窗口拖尾(≈0.005)被全部滤除 → 不产生伪候选、不白扫整窗;对所有窗口口径一致。None→不筛(纯相对)。
        def _val(p) -> float:
            return float(field[grid.world_to_cell(np.asarray(p, float))])

        peaks = [p for p in peaks0 if peak_abs is None or _val(p) >= peak_abs - 1e-12]
        fills = [f for f in fills0 if fill_abs is None or _val(f) >= fill_abs - 1e-12]
        dets = list(peaks) + list(fills)
        peak_arr = np.asarray([np.asarray(p, float) for p in peaks]) if peaks else np.empty((0, 2))
        fill_arr = np.asarray([np.asarray(f, float) for f in fills]) if fills else np.empty((0, 2))
        det_arr = np.asarray([np.asarray(d, float) for d in dets]) if dets else np.empty((0, 2))

        # 窗口内 baseline 规划(min-max VRP);⚠ solve_minmax_vrp 内部按 x≈0 起点分配/排序
        #   (合同函数不接受自定义起点),故"哪台 Follower 访问哪些点/访问顺序"按起始边布阵求得;
        #   实际执行从各 Follower 的**当前链式位姿**接续出发(首段为转移段),per-window 均衡。
        routes = solve_minmax_vrp(cfg, dets)

        start_poses_k = [p.copy() for p in foll_pose]  # 本窗口各 Follower 出发位姿(执行前快照)
        navs, res_list = [], []
        for i in range(N):
            nav = _simplify_route(foll_pose[i], routes[i], tol=width / 2.0)
            navs.append(nav)
            res = eng2.execute(foll_pose[i], [np.asarray(p, float) for p in nav]) if nav else None
            res_list.append(res)
            if res is not None and getattr(res, "trajectory", None) is not None and len(res.trajectory) > 1:
                tr = [np.asarray(p, float) for p in res.trajectory[:, :2]]
                coverage.update(_route_corridor_union(tr[0], tr[1:], width, grid))
                foll_pose[i] = np.asarray(res.end_pose, float).copy()      # 链式接续:更新当前位姿
            elif nav:
                coverage.update(_route_corridor_union(foll_pose[i], nav, width, grid))
                last = np.asarray(nav[-1], float)
                foll_pose[i] = np.array([last[0], last[1], float(foll_pose[i][2])])

        # 窗口内召回(块内 GT 是否被块内某候选点 fill_R 内覆盖)。
        gt_block = np.asarray([t for t in targets if edges[k] <= float(t[0]) < edges[k + 1]], float)
        if len(gt_block) and len(det_arr):
            hit_b = [min(np.hypot(*(g - d)) for d in det_arr) <= fill_R for g in gt_block]
        else:
            hit_b = [False] * len(gt_block)
        windows.append(dict(
            k=k, leader_x=float(leader.pose[0]),
            peak_arr=peak_arr, fill_arr=fill_arr, det_arr=det_arr,
            routes=routes, navs=navs, res_list=res_list,
            start_poses=start_poses_k, end_poses=[p.copy() for p in foll_pose],
            n_gt=len(gt_block), n_hit=int(sum(hit_b)),
        ))
        rlens = [_route_len(start_poses_k[i], routes[i]) for i in range(N)]
        print(f"\n[窗口 W{k}] Leader 扫完块 North∈[{edges[k]:.0f},{edges[k+1]:.0f}]m(位置 x={leader.pose[0]:.1f}m)→ 揭示并规划。")
        print(f"          候选:峰值 {len(peak_arr)} + 补点 {len(fill_arr)} = {len(det_arr)} 个【估计,非 GT】;"
              f"块内 GT {len(gt_block)} 个,窗口召回 {windows[-1]['n_hit']}/{len(gt_block)}。")
        for i in range(N):
            print(f"          F{i}: {len(routes[i])} 目标, 本窗路程={rlens[i]:.1f}m"
                  f"(出发 x={start_poses_k[i][0]:.1f} → 终端 x={foll_pose[i][0]:.1f})")

    # ===== 主循环:Leader 推进 → 每扫完一个 20m 块即推进观测窗口并规划/派遣 =====
    next_k = 0
    n_sim = 0
    while not leader.finished and n_sim < 6000:
        leader.advance(1.0)
        for (ix, iy) in window_cells(get_window(leader.pose, cfg.window), grid):
            revealed[ix, iy] = True
        leader_track.append(leader.pose[:2].copy())
        n_sim += 1
        # Leader 的 North 位置越过块 k 的右边界 → 其后方 20m 窗口恰覆盖块 k(块扫完)→ 推进窗口。
        while next_k < n_windows and float(leader.pose[0]) >= edges[next_k + 1] - 1e-9:
            sel = block_masks[next_k] & revealed & (reveal_window < 0)
            reveal_window[sel] = next_k
            process_window(next_k)
            next_k += 1
    # 兜底:若 Leader 提前走完(理论上 lead-out 保证越过 60),处理剩余窗口。
    while next_k < n_windows:
        sel = block_masks[next_k] & revealed & (reveal_window < 0)
        reveal_window[sel] = next_k
        process_window(next_k)
        next_k += 1
    leader_track = np.asarray(leader_track)
    surveyed_pct = 100.0 * revealed.sum() / (grid.nx * grid.ny)
    print(f"\n[Leader] 走完 lawnmower({n_sim} 仿真秒,能耗旁路={leader.energy_J_cumulative:.0f} J);"
          f"分块揭示完成,已巡查覆盖 = {surveyed_pct:.1f}%。")

    # ===== 全局规划汇总(跨窗口)=====
    all_dets = [d for w in windows for d in w["det_arr"]]
    all_peaks = np.vstack([w["peak_arr"] for w in windows]) if any(len(w["peak_arr"]) for w in windows) else np.empty((0, 2))
    all_fills = np.vstack([w["fill_arr"] for w in windows]) if any(len(w["fill_arr"]) for w in windows) else np.empty((0, 2))
    gt_arr = np.asarray([np.asarray(t, float) for t in targets])
    hit = ([min(np.hypot(*(g - np.asarray(d))) for d in all_dets) <= fill_R for g in gt_arr]
           if all_dets else [False] * K)
    det_recall = 100.0 * sum(hit) / K
    # 各 Follower 跨窗口累计路程/能耗(从每窗口出发位姿对该窗口 route 计代价后求和)。
    cum_len = [0.0] * N
    cum_eng = [0.0] * N
    for w in windows:
        eng_w = routes_to_energies(w["routes"], w["start_poses"], cfg.rollout)
        for i in range(N):
            cum_len[i] += _route_len(w["start_poses"][i], w["routes"][i])
            cum_eng[i] += eng_w[i]
    print(f"\n[规划汇总] 跨 {n_windows} 窗口共生成候选 {len(all_dets)} 个(峰值 {len(all_peaks)} + 补点 {len(all_fills)});"
          f"全局覆盖召回(GT 在某候选 {fill_R:.1f}m 内)= {sum(hit)}/{K}({det_recall:.0f}%)。")
    for i in range(N):
        print(f"          F{i} 跨窗累计:路程={cum_len[i]:.1f}m, 能耗={cum_eng[i]:.1f}J")
    print(f"          min-max 路程={max(cum_len):.1f}m | max 能耗={max_energy(cum_eng):.1f}J | total={total_energy(cum_eng):.1f}J")

    # ===== 实际观测(评估侧,不回写规划器;同 baseline 口径,基于累积覆盖)=====
    obs_rng = np.random.default_rng(cfg.seed + 100)
    target_cells = [grid.world_to_cell(t) for t in targets]
    tp, fn_cov, fn_sensor = [], [], []
    for t, tc in zip(targets, target_cells):
        if tc in coverage:
            if obs_rng.random() < cam.p_d:
                tp.append(t)
            else:
                fn_sensor.append(t)
        else:
            fn_cov.append(t)
    step = max(cam.footprint_along_m, cam.footprint_lateral_m)
    fp = []
    tgt_cell_set = set(target_cells)
    for gx in np.arange(step / 2, world.x_max_m, step):
        for gy in np.arange(step / 2, world.y_max_m, step):
            c = grid.world_to_cell(np.array([gx, gy]))
            if c in coverage and c not in tgt_cell_set and obs_rng.random() < cam.p_fa:
                fp.append(np.array([gx, gy]))

    det_rate = len(tp) / K * 100.0
    cov_of_targets = (K - len(fn_cov)) / K * 100.0
    print(f"\n[观测] Follower 沿窗口路径观测(相机 p_d={cam.p_d}, p_fa={cam.p_fa}, 走廊宽={width:.1f}m):")
    print(f"          覆盖到的目标 = {K-len(fn_cov)}/{K}({cov_of_targets:.0f}%);其中检出(TP)={len(tp)}、p_d 漏检={len(fn_sensor)}")
    print(f"          未被任何 Follower 覆盖(漏)={len(fn_cov)} | 虚警 FP={len(fp)} | 总检出率={det_rate:.0f}%")
    print(f"          ⚠ 观测结果为评估侧产物,**未回写规划器**(belief=NullUpdater)→ 开环。")

    # ===== 出图 =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"\n(无 matplotlib,跳过出图:{e!r})")
        return 0

    ext = [0, world.x_max_m, 0, world.y_max_m]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    win_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def _draw_dividers(ax, color="white"):
        """画观测窗口块分界线 + W0/W1/W2 标注。"""
        for x in edges[1:-1]:
            ax.axvline(x, color=color, ls=":", lw=1.3, alpha=0.85, zorder=5)
        for k in range(n_windows):
            xc = 0.5 * (edges[k] + edges[k + 1])
            ax.text(xc, world.y_max_m * 0.96, f"W{k}", color=color, fontsize=9,
                    ha="center", va="top", fontweight="bold", zorder=6)

    def _scatter(ax, pts, **kw):
        if len(pts):
            a = np.asarray([np.asarray(p, float) for p in pts])
            ax.scatter(a[:, 0], a[:, 1], **kw)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # A: 部署 + 计划测线 + 窗口分块(空白地图)
    ax = axes[0, 0]
    ax.imshow(np.zeros_like(field).T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    lawn_arr = np.asarray(lawn)
    ax.plot(lawn_arr[:, 0], lawn_arr[:, 1], "--", color="gray", lw=1, label="planned lawnmower")
    ax.plot(leader_track[0, 0], leader_track[0, 1], "ks", ms=8, label="Leader")
    ax.text(leader_track[0, 0], leader_track[0, 1], "  Leader Start", color="black", fontsize=8, va="bottom")
    for i, sp in enumerate(foll_starts):
        ax.plot(sp[0], sp[1], "^", color=colors[i % 4], ms=11, mec="white")
        ax.text(sp[0], sp[1], f"  F{i} Start", color="white", fontweight="bold", fontsize=8, va="center")
    _draw_dividers(ax)
    ax.set_title(f"Phase 0: deploy together, EMPTY map; split into {n_windows} obs-windows "
                 f"({look:.0f}x{cfg.window.width_m:.0f} m)", fontsize=8)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # B: 分块揭示(按窗口着色的揭示先验图)
    ax = axes[0, 1]
    shown = np.where(revealed, field, np.nan)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("lightgray")
    im = ax.imshow(shown.T, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prior prob")
    # 每个窗口块用半透明色块标出"是哪个窗口揭示的"。
    for k in range(n_windows):
        ax.axvspan(edges[k], edges[k + 1], color=win_colors[k % len(win_colors)], alpha=0.10, zorder=1)
    ax.plot(leader_track[:, 0], leader_track[:, 1], "-", color="black", lw=1.0, label="Leader path")
    ax.plot(leader_track[0, 0], leader_track[0, 1], "go", ms=6)
    _draw_dividers(ax, color="black")
    ax.set_title(f"Windowed reveal: Leader scans block-by-block (surveyed {surveyed_pct:.0f}%)\n"
                 f"each {look:.0f}m block revealed when Leader passes its edge; OTHER blocks not updated", fontsize=8)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # C: 窗口内规划(各窗口候选 + Follower 连续路径)
    ax = axes[1, 0]
    im = ax.imshow(field.T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prior prob")
    ax.scatter(gt_arr[:, 0], gt_arr[:, 1], s=55, facecolors="none", edgecolors="red", linewidths=1.4,
               marker="s", label="GT targets (reference only)")
    if len(all_peaks):
        ax.scatter(all_peaks[:, 0], all_peaks[:, 1], s=64, facecolors="gold", edgecolors="black",
                   linewidths=0.8, marker="D", label=f"peak anchors ({len(all_peaks)})", zorder=6)
    if len(all_fills):
        ax.scatter(all_fills[:, 0], all_fills[:, 1], s=46, facecolors="cyan", edgecolors="black",
                   linewidths=0.7, marker="o", label=f"coverage fill ({len(all_fills)})", zorder=6)
    # Follower 连续路径:逐窗口把 [出发位姿 → 该窗口 route] 连起来(同色),体现跨窗口接续。
    for i in range(N):
        seg_x, seg_y = [foll_starts[i][0]], [foll_starts[i][1]]
        for w in windows:
            for p in w["routes"][i]:
                seg_x.append(float(p[0])); seg_y.append(float(p[1]))
        ax.plot(seg_x, seg_y, "-o", color=colors[i % 4], ms=4, mec="white", label=f"F{i} route (all windows)")
        ax.plot(foll_starts[i][0], foll_starts[i][1], "^", color=colors[i % 4], ms=11, mec="white")
    _draw_dividers(ax)
    ax.set_title(f"Per-window planning on A' candidates (peak+fill, estimate), NOT GT\n"
                 f"({len(all_dets)} cand [{len(all_peaks)} peak +{len(all_fills)} fill] vs {K} GT, "
                 f"coverage recall {det_recall:.0f}%; min-max {max(cum_len):.0f} m)", fontsize=8)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # D: 实际观测到目标的分布图(TP/FN/FP + 覆盖)
    ax = axes[1, 1]
    cov_mask = np.zeros((grid.nx, grid.ny), dtype=float)
    for (ix, iy) in coverage:
        cov_mask[ix, iy] = 1.0
    ax.imshow(np.where(cov_mask > 0, 0.25, np.nan).T, origin="lower", extent=ext, aspect="auto",
              cmap="Greys", vmin=0, vmax=1, alpha=0.5)
    for i in range(N):
        seg_x, seg_y = [foll_starts[i][0]], [foll_starts[i][1]]
        for w in windows:
            for p in w["routes"][i]:
                seg_x.append(float(p[0])); seg_y.append(float(p[1]))
        ax.plot(seg_x, seg_y, "-", color=colors[i % 4], lw=0.8, alpha=0.6)
    _scatter(ax, tp, s=70, marker="o", facecolors="lime", edgecolors="black", linewidths=0.8, label=f"TP detected ({len(tp)})", zorder=6)
    _scatter(ax, fn_sensor, s=70, marker="x", color="red", linewidths=2, label=f"FN sensor-miss ({len(fn_sensor)})", zorder=6)
    _scatter(ax, fn_cov, s=70, marker="x", color="darkred", linewidths=2, label=f"FN not-covered ({len(fn_cov)})", zorder=6)
    _scatter(ax, fp, s=60, marker="^", facecolors="orange", edgecolors="black", linewidths=0.6, label=f"FP false-alarm ({len(fp)})", zorder=6)
    _draw_dividers(ax, color="black")
    ax.set_xlim(0, world.x_max_m); ax.set_ylim(0, world.y_max_m)
    ax.set_title(f"ACTUAL observed-target distribution (detect rate {det_rate:.0f}%)\n"
                 f"camera p_d={cam.p_d}/p_fa={cam.p_fa}; OPEN-LOOP: NOT fed back to planner", fontsize=8)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("End-to-end WINDOWED cooperative survey: Leader reveals block-by-block, Followers observe window-by-window",
                 fontsize=11)
    p = os.path.join(OUT, "mission_windows_survey.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(p, dpi=115); plt.close(fig)
    print(f"\n图已存(四面板窗口任务流): {p}")

    # ---- 另外一张:实际观测到目标的分布图(独立大图)----
    fig2, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(field.T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1, alpha=0.45)
    fig2.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prior prob (for reference)")
    cov_xy = np.array([grid.cell_to_world(c) for c in coverage]) if coverage else np.empty((0, 2))
    if len(cov_xy):
        ax.scatter(cov_xy[:, 0], cov_xy[:, 1], s=2, color="0.6", alpha=0.25, marker="s", zorder=2)
    # 各 Follower 跨窗口实际移动轨迹(Level2 积分,逐窗口段连续同色)。
    for i in range(N):
        c = colors[i % 4]
        labeled = False
        for w in windows:
            res = w["res_list"][i]
            if res is not None and getattr(res, "trajectory", None) is not None and len(res.trajectory) > 1:
                tr = np.asarray(res.trajectory)[:, :2]
                ax.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=1.5, alpha=0.9, zorder=3,
                        label=(f"F{i} actual path (Level2)" if not labeled else None))
                labeled = True
        sp_xy = np.asarray(foll_starts[i], float)[:2]
        ax.plot(sp_xy[0], sp_xy[1], "^", color=c, ms=13, mec="white", zorder=7)
        ax.text(sp_xy[0], sp_xy[1], f"  F{i} Start", color="white", fontweight="bold", fontsize=9, va="center", zorder=7)
    if len(all_peaks):
        ax.scatter(all_peaks[:, 0], all_peaks[:, 1], s=130, marker="D", facecolors="none",
                   edgecolors="gold", linewidths=2.0, label=f"peak anchors (estimate) ({len(all_peaks)})", zorder=6)
    if len(all_fills):
        ax.scatter(all_fills[:, 0], all_fills[:, 1], s=95, marker="o", facecolors="none",
                   edgecolors="deepskyblue", linewidths=1.8, label=f"coverage fill (estimate) ({len(all_fills)})", zorder=6)
    _scatter(ax, tp, s=110, marker="o", facecolors="lime", edgecolors="black", linewidths=1.0,
             label=f"detected target TP ({len(tp)})", zorder=8)
    _scatter(ax, fn_sensor, s=110, marker="x", color="red", linewidths=2.5,
             label=f"missed (sensor p_d) FN ({len(fn_sensor)})", zorder=8)
    _scatter(ax, fn_cov, s=110, marker="x", color="darkred", linewidths=2.5,
             label=f"missed (not covered) FN ({len(fn_cov)})", zorder=8)
    _scatter(ax, fp, s=95, marker="^", facecolors="orange", edgecolors="black", linewidths=0.8,
             label=f"false alarm FP ({len(fp)})", zorder=8)
    _draw_dividers(ax, color="black")
    ax.set_xlim(0, world.x_max_m); ax.set_ylim(0, world.y_max_m)
    ax.set_xlabel("North x (m)"); ax.set_ylabel("East y (m)")
    ax.set_title(f"WINDOWED ACTUAL observed-target distribution along Follower paths  "
                 f"(detect rate {det_rate:.0f}%, {len(tp)}/{K} detected)\n"
                 f"obs-window {look:.0f}x{cfg.window.width_m:.0f}m (W0|W1|W2); gold=peak + blue=fill (estimate; recall {det_recall:.0f}%), NOT GT; "
                 f"camera p_d={cam.p_d}/p_fa={cam.p_fa}; gray=coverage; OPEN-LOOP (master)", fontsize=8)
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    p2 = os.path.join(OUT, "mission_windows_observed_targets.png")
    fig2.tight_layout(); fig2.savefig(p2, dpi=120); plt.close(fig2)
    print(f"图已存(窗口模式实际观测到目标分布,独立): {p2}")
    print("  实线=各AUV实际移动轨迹(Level2积分,跨窗口连续) | ▲=各AUV起点 | 虚线=窗口分界 | 金菱=峰值锚点 | 青圈=覆盖补点(估计) | 绿圈=检出 TP | 红叉=漏检 | 橙三角=虚警 FP | 灰=相机覆盖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
