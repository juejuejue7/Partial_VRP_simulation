"""[A0 集成测试用例] 一次完整的 baseline 协同探查任务(端到端,联合全模块)。

任务流(对应修士设定):
  0) 部署:Leader + N Followers 初始同区部署(起始边 x≈0);地图初始无信息(已巡查=0)。
  1) Leader 探查:走完整条 lawnmower,用其后方观测窗口的累积覆盖**把整张静态先验概率图揭示出来**
     (覆盖 0→~100%)。⚠ 修士开环:先验**值**是静态真值,Leader 不在线重估,只"揭示/巡查"。
  1.5) 候选提取(A′ 混合,§v1.0.5):从 Leader 扫描概率图(belief=已揭示先验)取【候选观测航点】=
      峰值锚点 + 高概率区覆盖补点(修复纯峰值把"一整块合并簇"当单点的缺陷)。点是估计、非 GT;
      模拟真实探查里 Follower 规划器只看 Leader 图、不预知真实目标位置。
  2) Baseline 规划:对**候选观测航点**(估计,**非 ground truth**)跑 min-max VRP 近似最优,生成 N 条 Follower 观测路径。
  3) Follower 观测执行:每台 Follower 依次前往其路径点;沿走廊用相机(p_d/p_fa)产生**实际观测**。
  4) 产出"实际观测到目标的分布图":TP(检出)/FN(漏检:未覆盖或 p_d 漏)/FP(虚警),叠加 Follower 覆盖。
     ⚠ 该观测结果是**评估侧产物,不回写规划器**(belief=NullUpdater)——体现"观测但不闭环"的修士边界。
     (相机观测模拟本属波次2 env;此处由 A0 集成脚本承担,不引入 builder 模块。)

运行:D:\\nixingxing\\Anaconda\\envs\\auv_py310\\python.exe examples/mission_baseline_survey.py
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


def main() -> int:
    print("=" * 78)
    print("完整 baseline 协同探查任务(端到端集成测试)")
    print("=" * 78)

    # ---- 配置:小场(快、可视),N=2,固定 seed 复现 ----
    # ⚙ 调试旋钮:A′ 候选补点半径 R 经 cfg.obs_candidate.fill_radius_m 调(默认 None→0.9*footprint=5.4m=走廊宽)。
    #   seed=24 是最难场景(多目标挤成一整块高概率区);A′ 把候选从纯峰值 5 点补到 9 点(≈9 真值)。
    #   想让 Follower 走廊真正扫遍簇、端到端覆盖↑:换 ObsCandidateConfig(fill_radius_m=2.7)(代价=路线/能耗增)。
    # §v1.0.9:精细场景(2m 点距)到点容差——Follower arrive_radius 从大场景默认 1.5m 调小到 0.5m
    # (<0.25×点距),AUV 更精确压点、实际轨迹更贴合 waypoint(避免 1.5m 容差在密集点处"切角")。
    # 仅 mission 局部覆盖(大场景 acceptance/测试仍用默认 1.5m)。
    _foll_ov = dict(SimConfig().follower_overrides)
    _foll_ov["arrive_radius_m"] = 0.5
    cfg = replace(SimConfig(),
                  world=WorldConfig(x_max_m=60.0, y_max_m=20.0, res_m=2.0),
                  n_followers=1, seed=42,
                  # 相机探测范围 = 以 Follower 为重心的 2m×2m 方形(§v1.0.7):footprint=2×2,
                  # 走廊宽 = footprint_lateral × 1.0 = 2.0m = 1 个 grid cell(res_m=2.0),严格对齐。
                  sensor=SensorConfig(footprint_along_m=2.0, footprint_lateral_m=2.0),
                  # 目标场尺度(先验 σ / 簇散布)§v1.0.8 起独立于相机 footprint:不设即用默认
                  # cfg.target_field(σ=3m、簇散布=12m),改相机 footprint 不再动 ground-truth 场景;
                  # 想调目标分布紧凑度/先验模糊度 → 设 target_field=TargetFieldConfig(bandwidth_m=…, intra_spread_m=…)。
                  follower_overrides=_foll_ov,
                  # §v1.0.12:候选【固定绝对阈值】(单一真相源 cfg.obs_candidate,与 windowed 模式统一口径)。
                  #   全局揭示场 max=1.0 → 绝对 0.5/0.7 与相对阈值数值等价,baseline 候选不变,仅口径统一、可对照。
                  obs_candidate=ObsCandidateConfig(fill_radius_m=1.0,
                                                   peak_abs_threshold=0.3, fill_abs_threshold=0.5))
    world, grid = cfg.world, Grid(cfg.world)

    # ===== 真值 + 先验图(ground truth;Leader 巡查将把它揭示出来)=====
    inst = default_instance(cfg, n_targets=9)         # 9 个真实目标点(3 簇),先验=其 KDE(σ=3m)
    targets = inst.targets
    field = inst.field
    K = len(targets)

    # ===== Phase 0:部署(同区,空白地图)=====
    lawn = [np.asarray(w, float) for w in lawnmower_waypoints(world, lane_spacing_m=20.0)]
    # lead-in/lead-out 外扩(真实测绘标准做法):Leader 因 arrive_radius 提前停 + 声呐窗口仅覆盖
    # 后方,会在每条测线端点(尤其整条路径最终终点,Leader 停后不再移动)留下约 arrive_radius 宽
    # 的未揭示带 → Phase 1/2 图上端列发灰、surveyed<100%。让每条测线端点沿测线方向各外伸
    # over_run_m(越过测区边界);Leader 越过原端点后窗口即扫过边界,而 window_cells 的 in_bounds
    # 自动把界外 cell 裁回 [0,x_max] → 边缘揭示补满、不污染场外。Leader 能耗为旁路(不进 reward)。
    arrive_r = float(cfg.leader_overrides.get("arrive_radius_m", 0.5))
    over_run_m = arrive_r + 2.0                        # ≥ arrive_radius 方能真正越过原端点;+2m 余量稳妥落入后方窗口
    for _k in range(0, len(lawn) - 1, 2):             # lawnmower 每条测线一对端点 (a,b)
        _a, _b = lawn[_k], lawn[_k + 1]
        _d = _b - _a
        _n = float(np.hypot(_d[0], _d[1]))
        if _n > 0.0:
            _u = _d / _n
            lawn[_k] = _a - _u * over_run_m           # 起点端外伸(lead-in)
            lawn[_k + 1] = _b + _u * over_run_m       # 终点端外伸(lead-out)
    leader = LeaderActor(cfg, lawn)
    foll_starts = _start_poses(cfg)                   # N 台 Follower 起始位姿(x≈0 起始边)
    deploy_xy = np.array([0.0, world.y_max_m / 2.0])  # 部署区中心(起始边)
    print(f"\n[Phase 0] 部署:Leader@{np.round(leader.pose[:2],1)} + "
          f"{cfg.n_followers} Followers@{[np.round(s[:2],1).tolist() for s in foll_starts]}"
          f"  —— 同区(起始边 x≈0);地图初始无信息(已巡查=0)。")
    print(f"          真值:{K} 个目标点(3 簇);先验图 = 目标 KDE(σ=3m),Leader 巡查后才被揭示。")

    # ===== Phase 1:Leader 探查(走完 lawnmower → 揭示整张先验图)=====
    revealed = np.zeros((grid.nx, grid.ny), dtype=bool)
    leader_track = [leader.pose[:2].copy()]
    n = 0
    while not leader.finished and n < 4000:
        leader.advance(1.0)
        for (ix, iy) in window_cells(get_window(leader.pose, cfg.window), grid):
            revealed[ix, iy] = True
        leader_track.append(leader.pose[:2].copy())
        n += 1
    leader_track = np.asarray(leader_track)
    surveyed_pct = 100.0 * revealed.sum() / (grid.nx * grid.ny)
    print(f"\n[Phase 1] Leader 走完 lawnmower({n} 仿真秒,能耗旁路={leader.energy_J_cumulative:.0f} J)。")
    print(f"          已巡查覆盖 = {surveyed_pct:.1f}% → 整张先验概率图生成完毕(开环:仅揭示静态真值,未重估)。")

    # ===== Phase 1.5:从 Leader 扫描图提取候选观测航点(A′ 混合:峰值锚点+覆盖补点;估计,非 GT)=====
    belief = np.where(revealed, field, 0.0)           # Leader 实际扫到的概率图(已揭示先验;未扫到处=0)
    thr_rel, fill_thr, peak_sep, fill_R = resolve_obs_candidate_params(cfg)  # 经 cfg.obs_candidate 调(R 默认 5.4m;补点阈值默认 0.5)
    peak_abs, fill_abs = resolve_obs_candidate_abs_thresholds(cfg)  # §v1.0.12 固定绝对阈值(与 windowed 统一口径)
    peaks0 = detections_from_field(belief, world, threshold_rel=thr_rel, min_separation_m=peak_sep)  # §v1.0.4 纯峰值(对照)
    dets0 = observation_candidates_from_field(belief, world, threshold_rel=thr_rel,
                                              fill_threshold_rel=fill_thr,
                                              peak_min_separation_m=peak_sep, fill_radius_m=fill_R)
    fills0 = dets0[len(peaks0):]
    # §v1.0.12 统一鲁棒口径:几何生成后按【固定绝对阈值】筛选候选(单一真相源 cfg.obs_candidate)。
    #   全局揭示场 max=1.0 → 绝对 0.5/0.7 与相对阈值数值等价 → baseline 候选不变,仅与 windowed 模式口径统一;
    #   None→不筛(纯相对,向后兼容)。
    def _val(p) -> float:
        return float(field[grid.world_to_cell(np.asarray(p, float))])
    peaks = [p for p in peaks0 if peak_abs is None or _val(p) >= peak_abs - 1e-12]
    detections = list(peaks) + [f for f in fills0 if fill_abs is None or _val(f) >= fill_abs - 1e-12]
    peak_arr = np.asarray([np.asarray(p, float) for p in peaks]) if peaks else np.empty((0, 2))
    det_arr = np.asarray([np.asarray(d, float) for d in detections]) if detections else np.empty((0, 2))
    fill_arr = det_arr[len(peak_arr):] if len(det_arr) > len(peak_arr) else np.empty((0, 2))  # 补点=峰值后缀
    gt_arr = np.asarray([np.asarray(t, float) for t in targets])
    hit = ([min(np.hypot(*(g - d)) for d in det_arr) <= fill_R for g in gt_arr]
           if len(det_arr) else [False] * K)
    hit_pk = ([min(np.hypot(*(g - p)) for p in peak_arr) <= fill_R for g in gt_arr]
              if len(peak_arr) else [False] * K)
    det_recall = 100.0 * sum(hit) / K
    peak_recall = 100.0 * sum(hit_pk) / K
    print(f"\n[Phase 1.5] A′ 混合候选观测航点(峰值阈值={thr_rel} / 补点阈值={fill_thr},σ={peak_sep:.0f}m / R={fill_R:.1f}m):"
          f"峰值 {len(peaks)} + 补点 {len(detections) - len(peaks)} = {len(detections)} 个【估计,非 GT】。")
    print(f"          覆盖召回(GT 在某候选点 {fill_R:.1f}m 内)= {sum(hit)}/{K}({det_recall:.0f}%);"
          f"对照纯峰值 {sum(hit_pk)}/{K}({peak_recall:.0f}%) → 覆盖补点修复了"
          f"\"一整块合并簇被当单点\"丢目标的缺陷。")

    # ===== Phase 2:Baseline 规划(min-max VRP;输入=A′ 候选观测航点估计,**不看 GT**)=====
    routes = solve_minmax_vrp(cfg, detections)        # 基线只看 Leader 图的候选航点(估计),非 ground truth
    energies = routes_to_energies(routes, foll_starts, cfg.rollout)
    def _route_len(sp, r):
        pts = [np.asarray(sp, float)[:2]] + [np.asarray(p, float) for p in r]
        return float(sum(np.hypot(*(pts[i+1]-pts[i])) for i in range(len(pts)-1)))
    minmax = max(_route_len(sp, r) for sp, r in zip(foll_starts, routes))
    print(f"\n[Phase 2] Baseline=ortools min-max VRP(输入={len(detections)} 个 A′ 候选航点[峰值+补点,估计非 GT],近似最优):")
    for i, r in enumerate(routes):
        print(f"          Follower{i}: {len(r)} 目标, 路程={_route_len(foll_starts[i], r):.1f}m, 能耗={energies[i]:.1f}J")
    print(f"          min-max 路程={minmax:.1f}m | max 能耗={max_energy(energies):.1f}J | total={total_energy(energies):.1f}J")

    # ===== Phase 3:Follower 观测执行(rollout + 走廊覆盖 + 相机 p_d/p_fa)=====
    foll = Follower(cfg)
    cam = foll.sensor                                 # CameraSensorModel(p_d, p_fa, corridor_width)
    width = cam.corridor_width_m
    eng2 = Level2RolloutEngine(cfg, auv_type="follower", record_trajectory=True)  # 记录 Level2 积分轨迹 → fig2 画各 AUV 实际移动路径

    coverage: set = set()
    foll_tracks = []
    nav_routes = []  # §v1.0.11:去冗余后的实际导航点序列(执行 + fig2 用)
    for i, r in enumerate(routes):
        # §v1.0.11:导航前先去冗余(DP 简化,容差=走廊半宽)——移除密集冗余点,避免点距<转弯半径处绕圈。
        nav = _simplify_route(foll_starts[i], r, tol=width / 2.0)
        nav_routes.append(nav)
        res = eng2.execute(foll_starts[i], [np.asarray(p, float) for p in nav]) if nav else None
        foll_tracks.append((foll_starts[i], nav, res))
        # §v1.0.9:覆盖判定用 AUV **实际**走过的轨迹走廊(非规划 route 折线)——res.trajectory 是
        # Level2 动力学真实走线,更贴近"相机真正扫过哪里";无轨迹时回退简化导航。
        if res is not None and getattr(res, "trajectory", None) is not None and len(res.trajectory) > 1:
            tr = [np.asarray(p, float) for p in res.trajectory[:, :2]]
            coverage |= _route_corridor_union(tr[0], tr[1:], width, grid)
        elif nav:
            coverage |= _route_corridor_union(foll_starts[i], nav, width, grid)
    print(f"          [§v1.0.11] 导航去冗余:Follower 点数 "
          f"{[len(r) for r in routes]} → {[len(n) for n in nav_routes]}(DP 简化,容差={width/2:.1f}m)。")

    # 相机观测模拟(评估侧;不回写规划器)。固定 seed 复现。
    obs_rng = np.random.default_rng(cfg.seed + 100)
    target_cells = [grid.world_to_cell(t) for t in targets]
    tp, fn_cov, fn_sensor = [], [], []                # 检出 / 漏检(未覆盖)/ 漏检(p_d 漏)
    for t, tc in zip(targets, target_cells):
        if tc in coverage:
            if obs_rng.random() < cam.p_d:
                tp.append(t)
            else:
                fn_sensor.append(t)
        else:
            fn_cov.append(t)
    # 虚警 FP:在覆盖区按 footprint 尺度独立"看一眼",非目标处以 p_fa 误报。
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
    print(f"\n[Phase 3] Follower 沿路径观测(相机 p_d={cam.p_d}, p_fa={cam.p_fa}, 走廊宽={width:.1f}m):")
    print(f"          覆盖到的目标 = {K-len(fn_cov)}/{K}({cov_of_targets:.0f}%);其中检出(TP)={len(tp)}、p_d 漏检={len(fn_sensor)}")
    print(f"          未被任何 Follower 覆盖(漏)={len(fn_cov)} | 虚警 FP={len(fp)} | 总检出率={det_rate:.0f}%")
    print(f"          ⚠ 该观测结果为评估侧产物,**未回写规划器**(belief=NullUpdater)→ 开环。")

    # ===== Phase 4:出图 =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"\n(无 matplotlib,跳过出图:{e!r})")
        return 0

    ext = [0, world.x_max_m, 0, world.y_max_m]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # A: 部署(空白地图)
    ax = axes[0, 0]
    ax.imshow(np.zeros_like(field).T, origin="lower", extent=ext, aspect="auto",
              cmap="viridis", vmin=0, vmax=1)
    ax.plot(lawn[:, 0] if isinstance(lawn, np.ndarray) else [w[0] for w in lawn],
            [w[1] for w in lawn], "--", color="gray", lw=1, label="planned lawnmower")
    ax.plot(leader.pose[0] * 0 + leader_track[0, 0], leader_track[0, 1], "ks", ms=8, label="Leader")
    ax.text(leader_track[0, 0], leader_track[0, 1], "  Leader Start", color="black", fontsize=8, va="bottom")
    for i, sp in enumerate(foll_starts):
        ax.plot(sp[0], sp[1], "^", color=colors[i % 4], ms=11, mec="white")
        ax.text(sp[0], sp[1], f"  F{i} Start", color="white", fontweight="bold", fontsize=8, va="center")
    ax.set_title("Phase 0: deploy together, EMPTY map (no info yet)", fontsize=9)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # B: Leader 巡查后 → 完整先验图
    ax = axes[0, 1]
    shown = np.where(revealed, field, np.nan)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("lightgray")
    im = ax.imshow(shown.T, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prior prob")
    ax.plot(leader_track[:, 0], leader_track[:, 1], "-", color="black", lw=1.0, label="Leader path")
    ax.plot(leader_track[0, 0], leader_track[0, 1], "go", ms=6)
    ax.text(leader_track[0, 0], leader_track[0, 1], "  Start", color="green", fontweight="bold", fontsize=8,     va="bottom")
    ax.set_title(f"Phase 1: Leader survey done -> full prior map (surveyed {surveyed_pct:.0f}%)\n"
                 f"(OPEN-LOOP: static-prior VALUES, only revealed by coverage)", fontsize=8)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # C: Baseline Follower 路径
    ax = axes[1, 0]
    im = ax.imshow(field.T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prior prob")
    tg = np.asarray([np.asarray(t, float) for t in targets])
    ax.scatter(tg[:, 0], tg[:, 1], s=55, facecolors="none", edgecolors="red", linewidths=1.4, marker="s", label="GT targets (reference only)")
    if len(peak_arr):
        ax.scatter(peak_arr[:, 0], peak_arr[:, 1], s=64, facecolors="gold", edgecolors="black",
                   linewidths=0.8, marker="D", label=f"peak anchors ({len(peak_arr)})", zorder=6)
    if len(fill_arr):
        ax.scatter(fill_arr[:, 0], fill_arr[:, 1], s=46, facecolors="cyan", edgecolors="black",
                   linewidths=0.7, marker="o", label=f"coverage fill ({len(fill_arr)})", zorder=6)
    for i, r in enumerate(routes):
        if not r:
            continue
        pts = np.asarray([foll_starts[i][:2]] + [np.asarray(p, float) for p in r])
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=colors[i % 4], ms=4, mec="white", label=f"F{i} route")
        ax.plot(foll_starts[i][0], foll_starts[i][1], "^", color=colors[i % 4], ms=11, mec="white")
        ax.text(foll_starts[i][0], foll_starts[i][1], f"  F{i} Start", color="white", fontweight="bold", fontsize=8, va="center")
    ax.set_title(f"Phase 2: baseline plans on A' candidates (peak+fill, estimate), NOT GT\n"
                 f"({len(detections)} cand [{len(peak_arr)} peak +{len(fill_arr)} fill] vs {K} GT, "
                 f"coverage recall {det_recall:.0f}%; min-max {minmax:.0f} m)", fontsize=8)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # D: 实际观测到目标的分布图(TP/FN/FP + 覆盖)
    ax = axes[1, 1]
    cov_mask = np.zeros((grid.nx, grid.ny), dtype=float)
    for (ix, iy) in coverage:
        cov_mask[ix, iy] = 1.0
    ax.imshow(np.where(cov_mask > 0, 0.25, np.nan).T, origin="lower", extent=ext, aspect="auto",
              cmap="Greys", vmin=0, vmax=1, alpha=0.5)  # 覆盖区淡灰
    for i, r in enumerate(routes):
        if not r:
            continue
        pts = np.asarray([foll_starts[i][:2]] + [np.asarray(p, float) for p in r])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=colors[i % 4], lw=0.8, alpha=0.6)
    def _scatter(pts, **kw):
        if pts:
            a = np.asarray([np.asarray(p, float) for p in pts])
            ax.scatter(a[:, 0], a[:, 1], **kw)
    _scatter(tp, s=70, marker="o", facecolors="lime", edgecolors="black", linewidths=0.8, label=f"TP detected ({len(tp)})", zorder=6)
    _scatter(fn_sensor, s=70, marker="x", color="red", linewidths=2, label=f"FN sensor-miss ({len(fn_sensor)})", zorder=6)
    _scatter(fn_cov, s=70, marker="x", color="darkred", linewidths=2, label=f"FN not-covered ({len(fn_cov)})", zorder=6)
    _scatter(fp, s=60, marker="^", facecolors="orange", edgecolors="black", linewidths=0.6, label=f"FP false-alarm ({len(fp)})", zorder=6)
    ax.set_xlim(0, world.x_max_m); ax.set_ylim(0, world.y_max_m)
    ax.set_title(f"Phase 4: ACTUAL observed-target distribution (detect rate {det_rate:.0f}%)\n"
                 f"camera p_d={cam.p_d}/p_fa={cam.p_fa}; OPEN-LOOP: NOT fed back to planner", fontsize=8)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("End-to-end baseline cooperative survey: Leader builds prior, Followers observe via baseline routes",
                 fontsize=11)
    p = os.path.join(OUT, "mission_baseline_survey.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(p, dpi=115); plt.close(fig)
    print(f"\n图已存(四面板任务流): {p}")

    # ---- 另外一张:实际观测到目标的分布图(独立大图)----
    fig2, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(field.T, origin="lower", extent=ext, aspect="auto",
                   cmap="viridis", vmin=0, vmax=1, alpha=0.45)              # 先验淡背景作对照
    fig2.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prior prob (for reference)")
    # Follower 走廊覆盖(相机实际看过的区域)
    cov_xy = np.array([grid.cell_to_world(c) for c in coverage]) if coverage else np.empty((0, 2))
    if len(cov_xy):
        ax.scatter(cov_xy[:, 0], cov_xy[:, 1], s=2, color="0.6", alpha=0.25, marker="s", zorder=2)
    for i, (sp, r, res) in enumerate(foll_tracks):
        c = colors[i % 4]
        sp_xy = np.asarray(sp, float)[:2]
        # 实际移动路径 = Level2 动力学积分轨迹(含转弯曲率的真实走线);无轨迹时回退规划 route 折线
        if res is not None and getattr(res, "trajectory", None) is not None and len(res.trajectory) > 1:
            tr = np.asarray(res.trajectory)[:, :2]
            ax.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=1.5, alpha=0.9, zorder=3,
                    label=f"F{i} actual path (Level2)")
        elif r:
            pts = np.asarray([sp_xy] + [np.asarray(p, float) for p in r])
            ax.plot(pts[:, 0], pts[:, 1], "--", color=c, lw=1.0, alpha=0.6, zorder=3,
                    label=f"F{i} planned route (no traj)")
        # start point(各 AUV 起点,强调)
        ax.plot(sp_xy[0], sp_xy[1], "^", color=c, ms=13, mec="white", zorder=7)
        ax.text(sp_xy[0], sp_xy[1], f"  F{i} Start", color="white",
                fontweight="bold", fontsize=9, va="center", zorder=7)
    if len(peak_arr):
        ax.scatter(peak_arr[:, 0], peak_arr[:, 1], s=130, marker="D", facecolors="none",
                   edgecolors="gold", linewidths=2.0,
                   label=f"peak anchors (estimate) ({len(peak_arr)})", zorder=6)
    if len(fill_arr):
        ax.scatter(fill_arr[:, 0], fill_arr[:, 1], s=95, marker="o", facecolors="none",
                   edgecolors="deepskyblue", linewidths=1.8,
                   label=f"coverage fill (estimate) ({len(fill_arr)})", zorder=6)
    _scatter(tp, s=110, marker="o", facecolors="lime", edgecolors="black", linewidths=1.0,
             label=f"detected target TP ({len(tp)})", zorder=8)
    _scatter(fn_sensor, s=110, marker="x", color="red", linewidths=2.5,
             label=f"missed (sensor p_d) FN ({len(fn_sensor)})", zorder=8)
    _scatter(fn_cov, s=110, marker="x", color="darkred", linewidths=2.5,
             label=f"missed (not covered) FN ({len(fn_cov)})", zorder=8)
    _scatter(fp, s=95, marker="^", facecolors="orange", edgecolors="black", linewidths=0.8,
             label=f"false alarm FP ({len(fp)})", zorder=8)
    ax.set_xlim(0, world.x_max_m); ax.set_ylim(0, world.y_max_m)
    ax.set_xlabel("North x (m)"); ax.set_ylabel("East y (m)")
    ax.set_title(f"ACTUAL observed-target distribution along Follower paths  "
                 f"(detect rate {det_rate:.0f}%, {len(tp)}/{K} detected)\n"
                 f"baseline aims at A' candidates: gold=peak anchors + blue=coverage fill (estimate; coverage recall {det_recall:.0f}%), NOT GT; "
                 f"camera p_d={cam.p_d}/p_fa={cam.p_fa}; gray=coverage; "
                 f"OPEN-LOOP: observations NOT fed back to planner (master)", fontsize=8)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    p2 = os.path.join(OUT, "mission_observed_targets.png")
    fig2.tight_layout(); fig2.savefig(p2, dpi=120); plt.close(fig2)
    print(f"图已存(实际观测到目标分布,独立): {p2}")
    print("  实线=各AUV实际移动轨迹(Level2积分) | ▲=各AUV起点 | 金菱=峰值锚点 | 青圈=覆盖补点(估计) | 绿圈=检出 TP | 红叉=漏检(p_d/未覆盖) | 橙三角=虚警 FP | 灰=相机覆盖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
