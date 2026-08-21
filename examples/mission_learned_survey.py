"""[A0 集成演示] 一次完整的 **learned policy(强化学习策略)** 协同探查任务(端到端)。

本文件 = `mission_windows_survey.py` 的 **learned 变体**:**场景/环境/调查模式完全不变**
(同 seed=9、同 60×20 世界、同 20×20 滑动观测窗口、同 9 目标 3 簇、同 Follower 起点/相机),
**仅把"路径规划方法"由 per-window min-max VRP 换成训练好的 PPO 策略**(§v1.1.0 DoD④:
learned policy 复用 windows_survey 相同环境与调查模式,仅替换 path-plan 方法)。

与 window 演示的差异(仅"谁来给航点"这一处):
  - window 演示:每个 20m 块内提取候选 → solve_minmax_vrp 批量给 waypoint。
  - 本演示:把决策权交给 `CoopSurveyEnv`(事件驱动 MDP)里的训练策略——env 每"一步"为
    **当前空闲** Follower 在**当前 Leader 位姿窗口**的粗网格(Discrete(Hc×Wc)=100)里选一个
    格中心作为下一航点(忙者不决策、一次一台),策略 = checkpoint 加载的 actor-critic 取 argmax。
  其余完全一致:Leader 沿单测线分块揭示;Follower 把策略给出的承诺航点序列用 **Level2** 实际
  积分执行 → 走廊覆盖 → 同口径 TP/FN/FP 观测评估;开环(NullUpdater),观测不回写规划器。

为什么策略能直接跑这个 9 目标场景(虽训练时是 seed=0 的 2 目标实例):
  策略的观测/动作空间(粗窗口图 + Follower 状态向量 → Discrete(100))**与目标个数无关**,
  目标数只改变先验场 field 的形状与终止条件。故同一策略可在任意同 (world/window/sensor) 场景
  推理;**它在 9 目标场景下表现好不好,正是本演示要给人看、由人评估的东西**(§v1.1.0 验收③:
  机制能跑通即过闸,效果由人类裁定)。

实例 pin(公平对照):env.reset 默认会自行生成 2 目标实例;本演示在 reset 后把 env 的先验场
  **替换为与 `mission_windows_survey.py` 逐字相同的实例**(`default_instance(cfg, n_targets=9)`,
  seed=9)→ 策略与 window/baseline 看的是**同一张图**,产物可直接并排对照。此注入只动 env 的
  运行时数据(`_field`/`_planned`),不改 A5 的 env 文件、不改任何契约。

运行:D:\\nixingxing\\Anaconda\\envs\\auv_py310\\python.exe examples/mission_learned_survey.py
  (需先有 checkpoint:runs/policy_latest.pt;可用 `--checkpoint <path>` 指定其它。)
"""
from __future__ import annotations

import argparse
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

from msim.contracts.config import SimConfig
from msim.geometry.grid import Grid
from msim.geometry.window import get_window, window_cells
from msim.env_static.leader_path import lawnmower_waypoints
from msim.physics.follower import Follower
from msim.physics.rollout import Level2RolloutEngine
from msim.task.corridor import corridor_cells
from msim.eval.metrics import max_energy, routes_to_energies, total_energy
from msim.eval.runner import default_instance
from msim.mdp.coop_survey_env import CoopSurveyEnv, windowed_survey_cfg
from msim.policy.network import build_actor_critic
from msim.policy.train_ppo import load_checkpoint

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# DEFAULT_CKPT = os.path.join(ROOT, "runs", "policy_latest.pt")  # §B 收尾:已提为 B 微调版(==windowed_B_ft.pt)
DEFAULT_CKPT = os.path.join(ROOT, "runs", "policy_latest_1.pt")  # §B 收尾:已提为 B 微调版(==windowed_B_ft.pt)
# DEFAULT_CKPT = os.path.join(ROOT, "runs", "windowed_B_ft.pt")  # §B 收尾:已提为 B 微调版(==windowed_B_ft.pt)
DECISION_CAP = 600          # 防御性:策略 rollout 决策次数上限(正常应被 env 终止/截断收住)


# ======================================================================
# 小工具(与 window 演示同口径:走廊覆盖并集 / 折线长度)
# ======================================================================
def _route_corridor_union(start_xy, route, width_m, grid) -> set:
    """一台 Follower 沿 [start → route...] 的走廊覆盖 cell 并集(同 window 演示)。"""
    pts = [np.asarray(start_xy, float)[:2]] + [np.asarray(p, float) for p in route]
    cov: set = set()
    for a, b in zip(pts[:-1], pts[1:]):
        cov |= corridor_cells(a, b, width_m, grid)
    return cov


def _route_len(sp, r) -> float:
    pts = [np.asarray(sp, float)[:2]] + [np.asarray(p, float) for p in r]
    return float(sum(np.hypot(*(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)))


# ======================================================================
# 策略加载 + 贪心推理
# ======================================================================
def _load_policy(cfg: SimConfig, ckpt_path: str, n_actions_expected: int):
    """加载 checkpoint → 重建网络 → 载入权重 → eval 模式。

    动作空间不匹配(checkpoint n_actions ≠ 当前 env 的 Discrete n)直接报错:粗网格形状由
    cfg.window/cfg.sensor 决定,只有训练 config 与本演示一致 checkpoint 才装得进。
    """
    import torch

    payload = load_checkpoint(ckpt_path, map_location="cpu")
    ckpt_n = int(payload.get("n_actions", -1))
    if ckpt_n != int(n_actions_expected):
        raise ValueError(
            f"checkpoint 动作空间不匹配:checkpoint n_actions={ckpt_n} ≠ 本演示 env n_actions="
            f"{n_actions_expected}。checkpoint 的粗网格(coarse_grid_shape,由其 SimConfig 的 "
            f"window/sensor 推导)与 windowed_survey_cfg 不一致 → 该 checkpoint 不是用 "
            f"`--world windowed` 训练的。请用 windowed 配置重训后再演示。"
        )
    net = build_actor_critic(cfg, ckpt_n)
    net.load_state_dict(payload["model_state"])
    net.eval()
    return net, payload


def _greedy_action(net, obs) -> int:
    """对单条 Dict obs 取策略 argmax 动作(确定性演示,可复现)。"""
    import torch

    with torch.no_grad():
        t_obs = {
            "map": torch.as_tensor(np.asarray(obs["map"], dtype=np.float32)),
            "vec": torch.as_tensor(np.asarray(obs["vec"], dtype=np.float32)),
        }
        logits, _ = net(t_obs)                       # (1, n_actions)
        return int(torch.argmax(logits, dim=-1).item())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="learned policy 协同探查演示(与 windows_survey 同场景)。")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                        help="训练好的 policy checkpoint(默认 runs/policy_latest.pt)。")
    parser.add_argument("--seed", type=int, default=42,
                        help="场景 seed(默认 9,与 mission_windows_survey.py 同场景)。")
    args = parser.parse_args(argv)

    print("=" * 82)
    print("完整 LEARNED POLICY 协同探查任务(端到端;同 windows 场景,仅换路径规划方法)")
    print("=" * 82)

    if not os.path.isfile(args.checkpoint):
        print(f"\n[错误] 找不到 checkpoint: {args.checkpoint}")
        print("       请先训练:python -m msim.policy.train_ppo --world windowed --wandb --timesteps N")
        print("       训练中按 ESC 存盘 → 生成 runs/policy_latest.pt,再运行本演示。")
        return 2

    # ---- 配置:windowed_survey_cfg(与 mission_windows_survey 同 world/window/sensor;
    #      seed=9 锁同场景)+ Follower arrive_radius=0.5(仅影响 Level2 执行,不影响策略/动作空间)----
    base = windowed_survey_cfg(seed=args.seed)
    _foll_ov = dict(base.follower_overrides)
    _foll_ov["arrive_radius_m"] = 0.5
    cfg = replace(base, follower_overrides=_foll_ov)
    world, grid = cfg.world, Grid(cfg.world)
    N = cfg.n_followers

    # ===== 真值 + 先验图:与 window/baseline 演示**逐字相同**的实例(seed=9,9 目标 3 簇)=====
    # §v1.1.3:目标数读 cfg.target_field.n_targets(单一真相源),与 RL 训练实例同分布。
    inst = default_instance(cfg, n_targets=cfg.target_field.n_targets)
    targets = inst.targets
    field = np.asarray(inst.field, dtype=float)
    K = len(targets)

    # ===== 观测窗口分块(仅用于出图与 window 演示对齐:60/20 = 3 块)=====
    look = float(cfg.window.look_back_m)               # = 20m
    n_windows = int(round(world.x_max_m / look))       # = 3
    edges = [k * look for k in range(n_windows + 1)]    # [0,20,40,60]

    # ===== 加载策略 =====
    env = CoopSurveyEnv(cfg)
    n_actions = int(env.action_space.n)
    net, payload = _load_policy(cfg, args.checkpoint, n_actions)
    Hc, Wc = env._Hc, env._Wc
    print(f"\n[策略] 已加载 checkpoint: {args.checkpoint}")
    print(f"       动作空间 = Discrete({n_actions})(粗网格 {Hc}×{Wc};与训练 config 对齐)。")
    print(f"       训练总步数 = {payload.get('ppo_config', {}).get('total_timesteps', '?')}。")

    # ===== Phase 0:部署(同区,空白地图)=====
    obs, info = env.reset(seed=args.seed)
    # —— 实例 pin:把 env 自生成的实例替换为与 window 演示相同的 9 目标实例(只动运行时数据)——
    env._field = field.copy()
    env._planned[:] = False
    env._current = env._pick_decider()
    obs = env._assemble_current_obs()

    foll_starts = env._start_poses()                   # N 台 Follower 起始位姿(x=0;同基线约定)
    lawn = [np.asarray(w, float) for w in lawnmower_waypoints(world, cfg.window.width_m)]
    foll = Follower(cfg)
    cam = foll.sensor
    width = cam.corridor_width_m
    print(f"\n[Phase 0] 部署:{N} Followers@{[np.round(s[:2],1).tolist() for s in foll_starts]}"
          f"  —— 同区(起始边 x≈0);地图初始无信息。")
    print(f"          真值:{K} 个目标点(3 簇);先验图 = 目标 KDE(σ=3m),按窗口分 {n_windows} 块逐块揭示。")
    print(f"          观测窗口 = {look:.0f}×{cfg.window.width_m:.0f} m;块边界(North)= {edges}。")

    # ===== 策略 rollout(贪心):env 事件驱动决策 = learned 路径规划 =====
    leader_poses = [env._leader.pose.copy()]
    records = []                                        # 逐决策记录(供轨迹/出图/复盘)
    steps = 0
    done = False
    sum_reward = 0.0
    sum_dense_mass = 0.0
    sum_detect = 0
    while not done and steps < DECISION_CAP:
        i = int(env._current)                          # 本步决策者(最早空闲台)
        leader_x_before = float(env._leader.pose[0])
        a = _greedy_action(net, obs)
        obs, reward, terminated, truncated, info = env.step(a)
        wp = np.asarray(env._followers[i].commitment.queue[-1], float).copy()  # env 实际承诺的航点
        records.append(dict(step=steps, follower=i, action=a, waypoint=wp,
                            leader_x=leader_x_before, reward=float(reward),
                            dense_mass=float(info["dense_mass"]),
                            n_new_detections=int(info["n_new_detections"])))
        leader_poses.append(env._leader.pose.copy())
        sum_reward += float(reward)
        sum_dense_mass += float(info["dense_mass"])
        sum_detect += int(info["n_new_detections"])
        done = bool(terminated or truncated)
        steps += 1

    end_tag = "terminated(任务完成/Leader 走完)" if terminated else (
        "truncated(超时/电量)" if truncated else "达决策上限")
    print(f"\n[策略 rollout] {steps} 次决策后结束:{end_tag}。")
    print(f"          累计 reward={sum_reward:.3f};累计 dense 先验质量={sum_dense_mass:.3f}"
          f"(承诺队列解析的确定量,非估计);累计新检出 GT 目标格={sum_detect}。")

    # ===== 每台 Follower 的承诺航点序列 = learned 规划路径 =====
    routes = [[r["waypoint"] for r in records if r["follower"] == i] for i in range(N)]
    for i in range(N):
        n_dec_i = len(routes[i])
        print(f"          F{i}: {n_dec_i} 次决策(承诺航点);出发 x={foll_starts[i][0]:.1f}, y={foll_starts[i][1]:.1f}")

    # ===== Level2 实际执行(同 window 演示口径):积分轨迹 + 走廊覆盖 =====
    eng2 = Level2RolloutEngine(cfg, auv_type="follower", record_trajectory=True)
    trajectories = []
    coverage: set = set()
    for i in range(N):
        if not routes[i]:
            trajectories.append(None)
            continue
        res = eng2.execute(foll_starts[i], [np.asarray(p, float) for p in routes[i]])
        trajectories.append(res)
        if res is not None and getattr(res, "trajectory", None) is not None and len(res.trajectory) > 1:
            tr = [np.asarray(p, float) for p in res.trajectory[:, :2]]
            coverage.update(_route_corridor_union(tr[0], tr[1:], width, grid))
        else:
            coverage.update(_route_corridor_union(foll_starts[i], routes[i], width, grid))

    # ===== 跨步累计路程/能耗(同 window 演示:Level1 解析能耗口径)=====
    energies = routes_to_energies(routes, foll_starts, cfg.rollout)
    lengths = [_route_len(foll_starts[i], routes[i]) for i in range(N)]
    print(f"\n[规划汇总] learned 策略给出航点 → Level2 执行:")
    for i in range(N):
        print(f"          F{i} 累计:路程={lengths[i]:.1f}m, 能耗={energies[i]:.1f}J")
    print(f"          min-max 路程={max(lengths):.1f}m | max 能耗={max_energy(energies):.1f}J | "
          f"total={total_energy(energies):.1f}J")

    # ===== 已揭示掩码(用决策时刻的 Leader 位姿窗口重建,供出图与 window 演示对齐)=====
    revealed = np.zeros((grid.nx, grid.ny), dtype=bool)
    for pose in leader_poses:
        for (ix, iy) in window_cells(get_window(pose, cfg.window), grid):
            revealed[ix, iy] = True
    surveyed_pct = 100.0 * revealed.sum() / (grid.nx * grid.ny)
    leader_track = np.asarray([p[:2] for p in leader_poses])

    # ===== 覆盖召回(GT 是否被某 Follower 走廊实际扫过;确定量,几何判定)=====
    target_cells = [grid.world_to_cell(t) for t in targets]
    covered = [tc in coverage for tc in target_cells]
    n_cov = int(sum(covered))
    cov_recall = 100.0 * n_cov / K

    # ===== lead-lag 缺口(协作诊断量):Follower 走廊覆盖的 North 末端 vs Leader 末端 =====
    cov_world = np.array([grid.cell_to_world(c) for c in coverage]) if coverage else np.empty((0, 2))
    cov_x_max = float(cov_world[:, 0].max()) if len(cov_world) else 0.0
    leader_end_x = float(leader_track[-1, 0])
    tail_gap = max(0.0, leader_end_x - cov_x_max)

    # ===== 实际观测(评估侧,不回写规划器;与 window 演示同口径)=====
    obs_rng = np.random.default_rng(cfg.seed + 100)
    tp, fn_cov, fn_sensor = [], [], []
    for t, tc in zip(targets, target_cells):
        if tc in coverage:
            if obs_rng.random() < cam.p_d:
                tp.append(t)
            else:
                fn_sensor.append(t)
        else:
            fn_cov.append(t)
    step_sz = max(cam.footprint_along_m, cam.footprint_lateral_m)
    fp = []
    tgt_cell_set = set(target_cells)
    for gx in np.arange(step_sz / 2, world.x_max_m, step_sz):
        for gy in np.arange(step_sz / 2, world.y_max_m, step_sz):
            c = grid.world_to_cell(np.array([gx, gy]))
            if c in coverage and c not in tgt_cell_set and obs_rng.random() < cam.p_fa:
                fp.append(np.array([gx, gy]))
    det_rate = len(tp) / K * 100.0

    print(f"\n[观测] Follower 沿 learned 路径观测(相机 p_d={cam.p_d}, p_fa={cam.p_fa}, 走廊宽={width:.1f}m):")
    print(f"          走廊扫到的目标 = {n_cov}/{K}({cov_recall:.0f}%);其中检出(TP)={len(tp)}、p_d 漏检={len(fn_sensor)}")
    print(f"          未被任何 Follower 覆盖(漏)={len(fn_cov)} | 虚警 FP={len(fp)} | 总检出率={det_rate:.0f}%")
    print(f"          ⚠ 观测为评估侧产物,**未回写规划器**(belief=NullUpdater)→ 开环(修士)。")
    print(f"          ⚠ lead-lag 缺口:Follower 走廊覆盖只到 x={cov_x_max:.1f}m,Leader 末端 x={leader_end_x:.1f}m → "
          f"尾部 ~{tail_gap:.0f}m(含最后窗口大半)被揭示但无任何 Follower 航点(协作机制问题,见报告)。")

    # ===== 与 window/baseline 演示的对照说明(同场景 seed=9,可并排看图)=====
    print(f"\n[对照] 本 learned 结果与以下同场景(seed={args.seed})sibling 演示**可直接并排对照**:")
    print(f"          window 方法 → examples/out/mission_windows_survey.png / *_observed_targets.png")
    print(f"          baseline 方法 → examples/out/mission_baseline_survey.png / mission_observed_targets.png")
    print(f"          三者环境/场景/调查模式相同,**仅路径规划方法不同**;效果优劣由人类评估。")

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
    gt_arr = np.asarray([np.asarray(t, float) for t in targets])

    def _draw_dividers(ax, color="white"):
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

    def _learned_route_xy(i):
        """F i 的 learned 路径折线(起点 + 承诺航点序列)。"""
        xs, ys = [foll_starts[i][0]], [foll_starts[i][1]]
        for p in routes[i]:
            xs.append(float(p[0])); ys.append(float(p[1]))
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # A: 部署 + 计划测线 + 窗口分块(空白地图)
    ax = axes[0, 0]
    ax.imshow(np.zeros_like(field).T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    lawn_arr = np.asarray(lawn)
    ax.plot(lawn_arr[:, 0], lawn_arr[:, 1], "--", color="gray", lw=1, label="planned lawnmower")
    # v1.1.2:env 已对齐 window 演示——Leader 从 lead-in 起点(x≈-3.5)起步;标记方式与 window 演示一致。
    ax.plot(leader_track[0, 0], leader_track[0, 1], "ks", ms=8, label="Leader")
    ax.text(leader_track[0, 0], leader_track[0, 1], "  Leader Start", color="black", fontsize=8, va="bottom")
    for i, sp in enumerate(foll_starts):
        ax.plot(sp[0], sp[1], "^", color=colors[i % 4], ms=11, mec="white")
        ax.text(sp[0], sp[1], f"  F{i} Start", color="white", fontweight="bold", fontsize=8, va="center")
    _draw_dividers(ax)
    ax.set_title(f"Phase 0: deploy together, EMPTY map; {n_windows} obs-windows "
                 f"({look:.0f}x{cfg.window.width_m:.0f} m)", fontsize=8)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # B: 分块揭示 + Leader 路径(learned 策略在每个窗口内决策)
    ax = axes[0, 1]
    shown = np.where(revealed, field, np.nan)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("lightgray")
    im = ax.imshow(shown.T, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prior prob")
    ax.plot(leader_track[:, 0], leader_track[:, 1], "-", color="black", lw=1.0, label="Leader path")
    ax.plot(leader_track[0, 0], leader_track[0, 1], "go", ms=6)
    _draw_dividers(ax, color="black")
    ax.set_title(f"Windowed reveal: Leader scans block-by-block (surveyed {surveyed_pct:.0f}%)\n"
                 f"learned policy decides next waypoint inside each sliding window", fontsize=8)
    ax.set_xlabel("North x (m)", fontsize=8); ax.set_ylabel("East y (m)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

    # C: learned 规划(策略承诺航点 + Level2 实际轨迹)
    ax = axes[1, 0]
    im = ax.imshow(field.T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prior prob")
    ax.scatter(gt_arr[:, 0], gt_arr[:, 1], s=55, facecolors="none", edgecolors="red", linewidths=1.4,
               marker="s", label="GT targets (reference only)")
    for i in range(N):
        xs, ys = _learned_route_xy(i)
        ax.plot(xs, ys, "-o", color=colors[i % 4], ms=4, mec="white", label=f"F{i} policy waypoints")
        ax.plot(foll_starts[i][0], foll_starts[i][1], "^", color=colors[i % 4], ms=11, mec="white")
    # lead-lag 缺口:Leader 揭示了但 Follower 没覆盖到的尾部(协作机制问题,直观标出)。
    if tail_gap > 1.0:
        ax.axvspan(cov_x_max, world.x_max_m, color="red", alpha=0.12, zorder=1)
        ax.text(0.5 * (cov_x_max + world.x_max_m), world.y_max_m * 0.5,
                f"unworked tail\n~{tail_gap:.0f} m\n(Leader done,\nFollowers lagged)",
                color="darkred", fontsize=7, ha="center", va="center", fontweight="bold", zorder=7)
    _draw_dividers(ax)
    ax.set_title(f"LEARNED planning: PPO policy picks coarse-grid waypoints (Discrete({n_actions}))\n"
                 f"covered {n_cov}/{K} GT ({cov_recall:.0f}%); min-max {max(lengths):.0f} m, "
                 f"max-energy {max_energy(energies):.0f} J", fontsize=8)
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
        res = trajectories[i]
        if res is not None and getattr(res, "trajectory", None) is not None and len(res.trajectory) > 1:
            tr = np.asarray(res.trajectory)[:, :2]
            ax.plot(tr[:, 0], tr[:, 1], "-", color=colors[i % 4], lw=0.9, alpha=0.7)
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

    fig.suptitle("End-to-end LEARNED cooperative survey: SAME scenario as windows_survey, "
                 "path-planner replaced by trained PPO policy", fontsize=11)
    p = os.path.join(OUT, "mission_learned_survey.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(p, dpi=115); plt.close(fig)
    print(f"\n图已存(四面板 learned 任务流): {p}")

    # ---- 另外一张:实际观测到目标的分布图(独立大图,与 window 演示同款)----
    fig2, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(field.T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1, alpha=0.45)
    fig2.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="prior prob (for reference)")
    cov_xy = np.array([grid.cell_to_world(c) for c in coverage]) if coverage else np.empty((0, 2))
    if len(cov_xy):
        ax.scatter(cov_xy[:, 0], cov_xy[:, 1], s=2, color="0.6", alpha=0.25, marker="s", zorder=2)
    for i in range(N):
        c = colors[i % 4]
        res = trajectories[i]
        if res is not None and getattr(res, "trajectory", None) is not None and len(res.trajectory) > 1:
            tr = np.asarray(res.trajectory)[:, :2]
            ax.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=1.5, alpha=0.9, zorder=3, label=f"F{i} actual path (Level2)")
        sp_xy = np.asarray(foll_starts[i], float)[:2]
        ax.plot(sp_xy[0], sp_xy[1], "^", color=c, ms=13, mec="white", zorder=7)
        ax.text(sp_xy[0], sp_xy[1], f"  F{i} Start", color="white", fontweight="bold", fontsize=9, va="center", zorder=7)
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
    ax.set_title(f"LEARNED ACTUAL observed-target distribution along Follower paths  "
                 f"(detect rate {det_rate:.0f}%, {len(tp)}/{K} detected)\n"
                 f"obs-window {look:.0f}x{cfg.window.width_m:.0f}m (W0|W1|W2); PPO policy waypoints; "
                 f"camera p_d={cam.p_d}/p_fa={cam.p_fa}; gray=coverage; OPEN-LOOP (master)", fontsize=8)
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    p2 = os.path.join(OUT, "mission_learned_observed_targets.png")
    fig2.tight_layout(); fig2.savefig(p2, dpi=120); plt.close(fig2)
    print(f"图已存(learned 模式实际观测到目标分布,独立): {p2}")
    print("  实线=各AUV实际移动轨迹(Level2积分) | ▲=各AUV起点 | 虚线=窗口分界 | 绿圈=检出 TP | 红叉=漏检 | 橙三角=虚警 FP | 灰=相机覆盖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
