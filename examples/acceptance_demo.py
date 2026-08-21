"""[A0 集成/人工验收 harness] 联合波次1 全模块的端到端验收脚本。

本脚本不属于任何 builder 的独占模块(A0 的集成交付物),只 import 各层**已验收实现**,
给人类做"看得见、可判定"的人工验收。三段对应三项需求:

  Part 1  地图初始化创建是否正常  —— config / grid / field / leader_path / window
  Part 2  AUV 实例创建与在地图中运动是否正常 —— Follower / Level1&Level2 rollout / LeaderActor
          (附 2.5 任务层联合:corridor 覆盖 + value 不重复计数 + belief 开环,joins 多模块)
  Part 3  Baseline 算法能否产生合理结果 —— greedy / ortools_vrp / exact_milp / metrics / runner

每个检查打印 [✓]/[✗] 与实测值;脚本结尾汇总通过率。绘图可选(装了 matplotlib 才出 PNG,
否则纯文本,不影响判定)。

运行(务必用项目环境的全路径解释器):
  D:\\nixingxing\\Anaconda\\envs\\auv_py310\\python.exe examples/acceptance_demo.py
  D:\\nixingxing\\Anaconda\\envs\\auv_py310\\python.exe examples/acceptance_demo.py --part 1
  D:\\nixingxing\\Anaconda\\envs\\auv_py310\\python.exe examples/acceptance_demo.py --no-plot
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# --- Windows 控制台编码兜底:非当前码页字符降级为 '?' 而非崩溃(中文在 GBK 仍正常)---
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

# --- 项目无安装,注入根目录到 sys.path(与 tests/conftest.py 同法)---------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataclasses import replace

from msim.contracts.config import SimConfig, WorldConfig
from msim.geometry.grid import Grid
from msim.geometry.window import coarse_grid_shape, get_window, window_cells
from msim.env_static.field import load_field, make_clustered_field, save_field
from msim.env_static.leader_path import lawnmower_waypoints
from msim.physics.follower import Follower
from msim.physics.leader import LeaderActor
from msim.physics.rollout import Level1RolloutEngine, Level2RolloutEngine
from msim.task.belief import NullUpdater
from msim.task.corridor import corridor_cells
from msim.task.value import expected_entropy_reduction, mark_planned
from msim.baselines.greedy import _start_poses, solve_greedy
from msim.eval.metrics import jain_index, max_energy, routes_to_energies, total_energy
from msim.eval.runner import default_instance, default_targets, run_ablation

# ----------------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
_RESULTS: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    """打印一条判据结果并登记,返回 ok。"""
    _RESULTS.append((label, ok))
    mark = "[OK]" if ok else "[XX]"
    line = f"  {mark} {label}"
    if detail:
        line += f"   —— {detail}"
    print(line)
    return ok


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _try_plot():
    """返回 (plt or None)。无 matplotlib 时返回 None(纯文本降级)。"""
    try:
        import matplotlib

        matplotlib.use("Agg")  # 无显示环境,直接存 PNG
        import matplotlib.pyplot as plt

        os.makedirs(OUT_DIR, exist_ok=True)
        return plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (未启用绘图:{exc!r};纯文本验收不受影响)")
        return None


def _window_corners(pose, look_back_m: float, width_m: float) -> np.ndarray:
    """当前观测窗口矩形的 4 角(世界坐标)。几何严格对齐 msim.geometry.window:
    锚=Leader 位姿,沿 back=-fwd 延伸 [0,look_back],横向 lat=(-sinψ,cosψ) 取 ±width/2。"""
    x, y, psi = float(pose[0]), float(pose[1]), float(pose[2])
    back = np.array([-np.cos(psi), -np.sin(psi)])
    lat = np.array([-np.sin(psi), np.cos(psi)])
    o = np.array([x, y])
    hw = width_m / 2.0
    return np.array([
        o - hw * lat,                        # s=0,         t=-w/2
        o + hw * lat,                        # s=0,         t=+w/2
        o + look_back_m * back + hw * lat,   # s=look_back, t=+w/2
        o + look_back_m * back - hw * lat,   # s=look_back, t=-w/2
    ])


def demo_cfg() -> SimConfig:
    """验收用小场地(快、可视),其余按默认。真实训练用 SimConfig() 默认 200×100。"""
    world = WorldConfig(x_max_m=60.0, y_max_m=30.0, res_m=0.5)  # nx=120, ny=60
    return replace(SimConfig(), world=world, n_followers=2)


# ==================================================================================
# Part 1 —— 地图初始化创建
# ==================================================================================
def part1_map(cfg: SimConfig, plot) -> None:
    section("Part 1 · 地图初始化创建(config / grid / field / leader_path / window)")
    world = cfg.world
    grid = Grid(world)

    print(f"世界尺度: North x_max={world.x_max_m}m, East y_max={world.y_max_m}m, res={world.res_m}m")
    print(f"细网格形状(轴序 field[ix,iy]=North×East): nx={grid.nx}, ny={grid.ny}")
    check("grid 形状由 world 推导一致", grid.nx == world.nx and grid.ny == world.ny,
          f"nx={grid.nx}=={world.nx}, ny={grid.ny}=={world.ny}")

    # 栅格↔世界↔栅格往返无损(全项目静默 bug 高发区)
    sample = [(0, 0), (grid.nx - 1, grid.ny - 1), (grid.nx // 2, grid.ny // 3)]
    roundtrip_ok = all(grid.world_to_cell(grid.cell_to_world(c)) == c for c in sample)
    check("栅格→世界→栅格往返误差为 0", roundtrip_ok, f"抽样 cells={sample}")

    # 概率场:graded、∈[0,1]、可复现、可 save/load
    rng = np.random.default_rng(cfg.seed)
    field = make_clustered_field(world, rng, n_clusters=4, intra_density=0.8)
    check("概率场形状=(nx,ny) 且 dtype=float64",
          field.shape == (grid.nx, grid.ny) and field.dtype == np.float64,
          f"shape={field.shape}")
    check("概率场取值 ∈ [0,1]", bool(field.min() >= 0.0 and field.max() <= 1.0),
          f"min={field.min():.3f}, max={field.max():.3f}")
    check("概率场非常数(有簇结构)", float(field.std()) > 0.0, f"std={field.std():.4f}")

    field2 = make_clustered_field(world, np.random.default_rng(cfg.seed),
                                  n_clusters=4, intra_density=0.8)
    check("同 seed 同参数逐位复现", bool(np.array_equal(field, field2)))

    os.makedirs(OUT_DIR, exist_ok=True)
    npz = os.path.join(OUT_DIR, "gt_field.npz")
    save_field(field, npz, metadata={"seed": cfg.seed, "n_clusters": 4})
    loaded = load_field(npz)
    check("groundtruth save→load 逐位复现", bool(np.array_equal(loaded, field)),
          f"已存 {npz}")

    # lawnmower 测线:覆盖 North 长边、与概率场解耦
    lawn = [np.asarray(w, float) for w in lawnmower_waypoints(world, lane_spacing_m=10.0)]
    lawn_arr = np.asarray(lawn)
    check("lawnmower 触及 North 两端(覆盖长边)",
          bool(lawn_arr[:, 0].min() <= world.x_max_m * 0.1
               and lawn_arr[:, 0].max() >= world.x_max_m * 0.9),
          f"North∈[{lawn_arr[:,0].min():.1f},{lawn_arr[:,0].max():.1f}], {len(lawn)} 个 waypoint")
    check("lawnmower East 多测线步进",
          np.unique(np.round(lawn_arr[:, 1], 6)).size >= 2,
          f"East 测线数={np.unique(np.round(lawn_arr[:,1],6)).size}")

    # 观测窗口(Leader 后方)+ 粗网格形状
    leader_pose = np.array([world.x_max_m * 0.5, world.y_max_m * 0.5, 0.0])  # 朝 North
    region = get_window(leader_pose, cfg.window)
    wcells = window_cells(region, grid)
    Hc, Wc = coarse_grid_shape(cfg.window, cfg.sensor)
    check("窗口覆盖非空 cell 集", len(wcells) > 0, f"window_cells 数={len(wcells)}")
    check("粗网格形状由 window÷footprint 推导(Hc,Wc>0)", Hc > 0 and Wc > 0,
          f"(Hc,Wc)=({Hc},{Wc}) → action_space=Discrete({Hc*Wc})")

    if plot is not None:
        fig, ax = plot.subplots(figsize=(8, 4))
        # field[ix,iy] 轴序与世界轴序对齐;imshow 习惯 row=y,故转置 + origin lower
        ax.imshow(field.T, origin="lower",
                  extent=[0, world.x_max_m, 0, world.y_max_m],
                  aspect="auto", cmap="viridis")
        ax.plot(lawn_arr[:, 0], lawn_arr[:, 1], "-o", color="white", ms=2, lw=1,
                label="Leader lawnmower")
        ax.set_xlabel("North x (m)")
        ax.set_ylabel("East y (m)")
        ax.set_title("Part1: probability field + Leader lawnmower track")
        ax.legend(loc="upper right", fontsize=8)
        p = os.path.join(OUT_DIR, "part1_map.png")
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plot.close(fig)
        print(f"  图已存: {p}")


# ==================================================================================
# Part 2 —— AUV 实例创建与在地图中运动
# ==================================================================================
def part2_auv(cfg: SimConfig, plot) -> None:
    section("Part 2 · AUV 实例创建与运动(Follower / rollout / LeaderActor)")
    world = cfg.world

    # --- 2a Follower 实例 + 两级 rollout ------------------------------------------
    print("[2a] Follower 实例 + Level1(解析)/Level2(动力学积分)rollout")
    foll = Follower(cfg)
    start = np.array([0.0, world.y_max_m * 0.5, 0.0])  # 起点,朝 North
    foll.reset(start)
    check("Follower 创建并 reset,初始满电 energy_norm=1",
          abs(foll.energy_norm - 1.0) < 1e-9, f"energy_norm={foll.energy_norm:.4f}")
    check("Follower 暴露相机模型(corridor_width=footprint横向×1.0;§v1.0.7)",
          abs(foll.sensor.corridor_width_m - foll.sensor.footprint_lateral_m) < 1e-9,
          f"corridor_width={foll.sensor.corridor_width_m:.3f}m")

    # 一段含一个直角转弯的 waypoint 序列
    wps = [np.array([15.0, world.y_max_m * 0.5]),
           np.array([15.0, world.y_max_m * 0.5 + 8.0])]
    eng1 = Level1RolloutEngine(cfg.rollout)
    eng2 = Level2RolloutEngine(cfg, auv_type="follower")
    r1 = eng1.execute(start, wps)
    r2 = eng2.execute(start, wps)
    print(f"    Level1(训练口径,解析): energy={r1.energy_J:.3f} J, duration={r1.duration_s:.3f} s, "
          f"end={np.round(np.asarray(r1.end_pose,float),2)}")
    print(f"    Level2(验证口径,积分): energy={r2.energy_J:.3f} J, duration={r2.duration_s:.3f} s, "
          f"end={np.round(np.asarray(r2.end_pose,float),2)}")
    check("Level1 用时/能耗为正且有限", r1.duration_s > 0 and r1.energy_J > 0)
    check("Level2 物理积分用时/能耗为正", r2.duration_s > 0 and r2.energy_J > 0)
    check("Level1 末位置抵达最后 waypoint",
          float(np.linalg.norm(np.asarray(r1.end_pose, float)[:2] - wps[-1])) < 1e-6)
    check("Level2 末位姿抵达最后 waypoint 邻域(arrive_radius 量级)",
          float(np.linalg.norm(np.asarray(r2.end_pose, float)[:2] - wps[-1])) < 5.0)

    # --- 2b LeaderActor 沿 lawnmower 动力学推进 -----------------------------------
    print("[2b] LeaderActor 沿 lawnmower 用 AUV 动力学推进(能耗旁路,不进 reward)")
    lawn = [np.asarray(w, float) for w in lawnmower_waypoints(world, lane_spacing_m=10.0)]
    leader = LeaderActor(cfg, lawn)
    p0 = leader.pose.copy()
    track = [p0.copy()]
    dt_chunk = 1.0           # 每外层步推进 1 仿真秒(内部按 sim_dt 细分)
    max_chunks = 4000
    n = 0
    while not leader.finished and n < max_chunks:
        leader.advance(dt_chunk)
        track.append(leader.pose.copy())
        n += 1
    track = np.asarray(track)
    moved = float(np.linalg.norm(track[-1][:2] - p0[:2]))
    check("Leader 创建并 reset(起点=测线首点)",
          float(np.linalg.norm(p0[:2] - lawn[0])) < 2.0,
          f"起点 pose={np.round(p0,2)}")
    check("Leader 沿测线实际移动(动力学位姿前进)", moved > world.x_max_m * 0.5,
          f"净位移={moved:.1f}m, 推进 {n} 仿真秒")
    check("Leader 在步数上限内走完整条 lawnmower(finished)", bool(leader.finished),
          f"finished={leader.finished}, 累计能耗(旁路)={leader.energy_J_cumulative:.1f} J")

    if plot is not None:
        fig, ax = plot.subplots(figsize=(8, 4))
        lawn_arr = np.asarray(lawn)
        ax.plot(lawn_arr[:, 0], lawn_arr[:, 1], "--", color="gray", lw=1, label="lawnmower target track")
        ax.plot(track[:, 0], track[:, 1], "-", color="tab:blue", lw=1.2, label="Leader actual dynamics path")
        ax.plot(track[0, 0], track[0, 1], "go", label="start")
        ax.plot(track[-1, 0], track[-1, 1], "r*", ms=12, label="end")
        ax.text(track[0, 0], track[0, 1], "  Start", color="green",
                fontweight="bold", va="bottom", ha="left", fontsize=9)
        ax.set_xlabel("North x (m)"); ax.set_ylabel("East y (m)")
        ax.set_title("Part2: Leader dynamics trajectory along lawnmower")
        ax.legend(loc="upper right", fontsize=8)
        p = os.path.join(OUT_DIR, "part2_leader_track.png")
        fig.tight_layout(); fig.savefig(p, dpi=110); plot.close(fig)
        print(f"  图已存: {p}")

    # --- 2.5 任务层联合:corridor 覆盖 + value 不重复计数 + belief 开环 ----------
    print("[2.5] 任务层联合(corridor 覆盖 / value 不重复计数 / belief 开环)")
    grid = Grid(world)
    rng = np.random.default_rng(cfg.seed)
    field = make_clustered_field(world, rng, n_clusters=4, intra_density=0.8)
    width = foll.sensor.corridor_width_m

    yc = world.y_max_m * 0.5
    segA = (np.array([10.0, yc]), np.array([30.0, yc]))
    segB = (np.array([20.0, yc]), np.array([40.0, yc]))   # 与 A 在 [20,30] 重叠
    cellsA = corridor_cells(segA[0], segA[1], width, grid)
    cellsB = corridor_cells(segB[0], segB[1], width, grid)
    overlap = cellsA & cellsB
    check("corridor 覆盖格集非空、两段有重叠", len(cellsA) > 0 and len(overlap) > 0,
          f"|A|={len(cellsA)}, |B|={len(cellsB)}, 重叠={len(overlap)} 格")

    planned = np.zeros((grid.nx, grid.ny), dtype=bool)
    eigA = expected_entropy_reduction(cellsA, planned, field, foll.sensor)
    mark_planned(planned, cellsA)                 # env 在结算后显式调用
    eigB_after = expected_entropy_reduction(cellsB, planned, field, foll.sensor)  # 跳过已计划格

    planned_union = np.zeros((grid.nx, grid.ny), dtype=bool)
    eig_union = expected_entropy_reduction(cellsA | cellsB, planned_union, field, foll.sensor)
    check("不重复计数:r(A)+r(B∖A) == r(A∪B) 一次性结算",
          abs((eigA + eigB_after) - eig_union) < 1e-9,
          f"r(A)={eigA:.4f} + r(B∖A)={eigB_after:.4f} = {eigA+eigB_after:.4f} vs r(A∪B)={eig_union:.4f} bit")
    check("期望熵降非负(互信息 I(H;Z)≥0)", eigA >= 0 and eig_union >= 0)

    before = field.copy()
    NullUpdater().update(field, cellsA, foll.sensor)
    check("belief 开环:NullUpdater 调用后概率场逐位不变", bool(np.array_equal(field, before)))


# ==================================================================================
# Part 2.6 —— Leader 凭自身运动逐步揭示概率图(修士开环:值不变,仅覆盖扩展)
# ==================================================================================
def part2_leader_reveal(cfg: SimConfig, plot) -> None:
    """呈现"Leader 意识里的概率分布图逐步产生的过程"。

    ⚠ 设计真相(修士开环,CLAUDE.md/project_summary):概率图是**静态先验**,belief=NullUpdater
    空操作 → 概率**值永不被观测改写**。Leader 凭运动逐步获得的是**已巡查覆盖**:其后方观测
    窗口(Follower 观测资格区)随前进**累积并集**,逐步把先验图对应区域纳入"已观测/已揭示"。
    故图中:灰=尚未巡查;着色=已巡查(显示**静态先验值**)。这**不是**贝叶斯在线建图
    (那是博士 BayesianUpdater,当前仅占位)。
    """
    section("Part 2.6 · Leader 运动逐步揭示概率图(修士开环:先验值不变,仅覆盖扩展)")
    world = cfg.world
    grid = Grid(world)
    field = make_clustered_field(world, np.random.default_rng(cfg.seed),
                                 n_clusters=4, intra_density=0.8)
    field_before = field.copy()
    lawn = [np.asarray(w, float) for w in lawnmower_waypoints(world, lane_spacing_m=10.0)]
    leader = LeaderActor(cfg, lawn)

    revealed = np.zeros((grid.nx, grid.ny), dtype=bool)
    masks: list[np.ndarray] = []
    areas: list[int] = []
    win_areas: list[int] = []                 # 当前(瞬时)窗口覆盖格数
    init_xy = leader.pose[:2].copy()
    poses: list[np.ndarray] = []              # 每步完整位姿 [x,y,psi](画窗口需朝向)
    dt_chunk = 2.0
    n = 0
    while not leader.finished and n < 4000:
        leader.advance(dt_chunk)
        region = get_window(leader.pose, cfg.window)     # Leader 后方观测窗口(随实际位姿滑动)
        wc = window_cells(region, grid)                  # 当前窗口内 cell
        for (ix, iy) in wc:                              # 累积并集 = 已巡查/已揭示区域
            revealed[ix, iy] = True
        masks.append(revealed.copy())
        areas.append(int(revealed.sum()))
        win_areas.append(len(wc))
        poses.append(leader.pose.copy())
        n += 1
    poses = np.asarray(poses)

    total = grid.nx * grid.ny
    final_pct = 100.0 * areas[-1] / total if areas else 0.0
    monotone = all(areas[i] <= areas[i + 1] for i in range(len(areas) - 1))
    check("已巡查覆盖随 Leader 运动单调扩展(逐步获得)", monotone,
          f"覆盖 {areas[0] if areas else 0}→{areas[-1] if areas else 0} cells ({final_pct:.0f}% of map)")
    check("修士开环:整个揭示过程概率值逐位不变(只揭示,不改写)",
          bool(np.array_equal(field, field_before)),
          "field 始终==初始先验 → 非贝叶斯在线建图(博士 BayesianUpdater 才改值)")
    check("Leader 揭示地图主体(覆盖 > 40%)", final_pct > 40.0, f"覆盖={final_pct:.0f}%")
    win_typ = int(np.median(win_areas)) if win_areas else 0
    check("当前观测窗口为有界滑动区(瞬时覆盖 << 累积覆盖)",
          0 < win_typ < areas[-1] if areas else False,
          f"瞬时窗口≈{win_typ} cells vs 累积={areas[-1] if areas else 0} cells")

    if plot is not None and masks:
        k = min(6, len(masks))
        idx = [int(round(i * (len(masks) - 1) / (k - 1))) for i in range(k)] if k > 1 else [0]
        fig, axes = plot.subplots(2, 3, figsize=(13, 7))
        cmap = plot.cm.viridis.copy()
        cmap.set_bad("lightgray")     # 未巡查 = 灰(Leader 意识中尚未知)
        for ax, j in zip(axes.ravel(), idx):
            shown = np.where(masks[j], field, np.nan)     # 已揭示=先验值;未揭示=NaN(灰)
            ax.imshow(shown.T, origin="lower",
                      extent=[0, world.x_max_m, 0, world.y_max_m],
                      aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
            line = np.vstack([init_xy, poses[:j + 1, :2]])
            ax.plot(line[:, 0], line[:, 1], "-", color="white", lw=1.0)
            # 当前观测窗口(红框,Leader 后方滑动矩形 = 本步的"观察窗口内的部分")
            corners = _window_corners(poses[j], cfg.window.look_back_m, cfg.window.width_m)
            cc = np.vstack([corners, corners[0]])
            ax.plot(cc[:, 0], cc[:, 1], "-", color="red", lw=1.6)
            ax.fill(corners[:, 0], corners[:, 1], color="red", alpha=0.18)
            ax.plot(poses[j, 0], poses[j, 1], "r*", ms=10)   # Leader 当前位置
            ax.plot(init_xy[0], init_xy[1], "go", ms=6)
            ax.text(init_xy[0], init_xy[1], "  Start", color="green",
                    fontweight="bold", va="bottom", ha="left", fontsize=8)
            pct = 100.0 * int(masks[j].sum()) / total
            ax.set_title(f"step {j+1}/{len(masks)}   surveyed {pct:.0f}%   (red=window)", fontsize=9)
            ax.set_xlabel("North x (m)", fontsize=8)
            ax.set_ylabel("East y (m)", fontsize=8)
        # 多余子图(k<6 时)隐藏
        for ax in axes.ravel()[k:]:
            ax.axis("off")
        fig.suptitle("Map in Leader's awareness, revealed progressively by its own motion\n"
                     "(master OPEN-LOOP: static-prior values FIXED; only surveyed coverage grows; "
                     "gray = not yet surveyed; RED box = current sliding observation window)", fontsize=10)
        p = os.path.join(OUT_DIR, "part2_leader_belief_reveal.png")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(p, dpi=110)
        plot.close(fig)
        print(f"  图已存: {p}  (灰=未巡查;着色=已巡查的静态先验;红框=当前观测窗口)")


# ==================================================================================
# Part 3 —— Baseline 算法合理性
# ==================================================================================
def _minmax_len(routes, starts) -> float:
    """各 Follower 路径长度(含起点→首目标)的 max,用于比较 min-max 优劣。"""
    def route_len(sp, route):
        pts = [np.asarray(sp, float)[:2]] + [np.asarray(p, float) for p in route]
        return sum(float(np.hypot(*(pts[i + 1] - pts[i]))) for i in range(len(pts) - 1))
    return max((route_len(sp, r) for sp, r in zip(starts, routes)), default=0.0)


def _partition_ok(routes, targets, n) -> bool:
    if len(routes) != n:
        return False
    tgt = {tuple(np.round(np.asarray(t, float), 6)) for t in targets}
    assigned = [tuple(np.round(np.asarray(p, float), 6)) for r in routes for p in r]
    tgt_hits = [a for a in assigned if a in tgt]
    return tgt.issubset(set(assigned)) and len(tgt_hits) == len(set(tgt_hits))


def part3_baselines(cfg: SimConfig, plot) -> None:
    section("Part 3 · Baseline 合理性(greedy / ortools_vrp / exact_milp / metrics)")
    cfg = replace(cfg, n_followers=2)
    inst = default_instance(cfg, n_targets=6)        # §v1.0.3 耦合实例:目标点(真值)+ 派生先验场
    targets = list(inst.targets)
    starts = _start_poses(cfg)
    print(f"目标数={len(targets)}(ground-truth 簇结构), Follower 数={cfg.n_followers}, "
          f"起点={[np.round(s[:2],1).tolist() for s in starts]}")

    from msim.baselines.ortools_vrp import solve_minmax_vrp
    from msim.baselines.exact_milp import solve_exact_minmax

    solvers = {"greedy": solve_greedy, "ortools_vrp": solve_minmax_vrp,
               "exact_milp": solve_exact_minmax}
    mm = {}
    for name, fn in solvers.items():
        try:
            routes = fn(cfg, targets)
        except (ImportError, ValueError) as exc:  # 求解器缺失/拒解 → 跳过,不崩
            check(f"{name}: 可用", False, f"跳过:{exc!r}")
            continue
        ok_part = _partition_ok(routes, targets, cfg.n_followers)
        energies = routes_to_energies(routes, starts, cfg.rollout)
        mm[name] = _minmax_len(routes, starts)
        check(f"{name}: 合法划分(N 条路径、每目标恰分一次)", ok_part)
        print(f"      → min-max 路程={mm[name]:.2f}m | per-follower 能耗={[round(e,1) for e in energies]} "
              f"| max={max_energy(energies):.1f} total={total_energy(energies):.1f} Jain={jain_index(energies):.3f}")

    # 合理性:精确解(ground truth)的 min-max 不劣于近似/贪心
    if {"exact_milp", "ortools_vrp", "greedy"} <= set(mm):
        check("exact(ground truth)min-max ≤ ortools(近似)",
              mm["exact_milp"] <= mm["ortools_vrp"] + 1e-6,
              f"{mm['exact_milp']:.2f} ≤ {mm['ortools_vrp']:.2f}")
        check("exact(ground truth)min-max ≤ greedy(朴素)",
              mm["exact_milp"] <= mm["greedy"] + 1e-6,
              f"{mm['exact_milp']:.2f} ≤ {mm['greedy']:.2f}")

    # balance on/off 消融(连续权重,非 if 分支)
    print("[3.5] balance on/off 消融(runner.run_ablation;weight 0 vs >0)")
    ab = run_ablation(cfg, targets, fairness="jain")
    for name in ["greedy", "ortools_vrp", "exact_milp"]:
        off = ab["balance_off"].get(name, {})
        on = ab["balance_on"].get(name, {})
        if off.get("skipped") or on.get("skipped") or "objective" not in off:
            print(f"      {name}: 跳过")
            continue
        print(f"      {name}: off J={off['objective']:.1f}(=total {off['total']:.1f}) | "
              f"on J={on['objective']:.1f}(=total+{ab['balance_on_weight']:.0f}·max)")
        check(f"{name}: balance off 目标==纯总能耗(max 项被权重消去)",
              abs(off["objective"] - off["total"]) < 1e-6)
        check(f"{name}: balance on 目标=total+w·max(引入 min-max 项)",
              abs(on["objective"] - (on["total"] + ab["balance_on_weight"] * on["max"])) < 1e-6)

    if plot is not None and "greedy" in mm:
        world = cfg.world
        # §v1.0.3:目标点=绝对 ground truth;先验场 = 对这些目标点的高斯KDE(同一实例,完全耦合)。
        field = inst.field
        tg = np.asarray([np.asarray(t, float) for t in targets])
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        ext = [0, world.x_max_m, 0, world.y_max_m]

        # --- 图A:ground-truth 目标点 叠加在 由其派生的先验场 上(耦合)---
        figA, axA = plot.subplots(figsize=(8.5, 4.2))
        imA = axA.imshow(field.T, origin="lower", extent=ext, aspect="auto",
                         cmap="viridis", vmin=0.0, vmax=1.0)
        figA.colorbar(imA, ax=axA, label="prior = Gaussian KDE of targets")
        axA.scatter(tg[:, 0], tg[:, 1], s=95, facecolors="red", edgecolors="white",
                    linewidths=1.3, marker="o", zorder=5, label="ground-truth targets")
        for i, sp in enumerate(starts):
            axA.plot(sp[0], sp[1], "^", color=colors[i % 4], ms=11, mec="white", zorder=6)
            axA.text(sp[0], sp[1], f"  Start F{i}", color="white",
                     fontweight="bold", va="center", ha="left", fontsize=8)
        axA.set_xlabel("North x (m)"); axA.set_ylabel("East y (m)")
        axA.set_title("Part3: ground-truth targets over their DERIVED probability field (COUPLED, v1.0.3)\n"
                      "(targets = absolute ground truth; prior field = normalized Gaussian KDE over these targets)",
                      fontsize=9)
        axA.legend(loc="upper right", fontsize=8)
        pA = os.path.join(OUT_DIR, "part3_targets_on_field.png")
        figA.tight_layout(); figA.savefig(pA, dpi=110); plot.close(figA)
        print(f"  图已存: {pA}  (先验场 + 其 ground-truth 目标点;已耦合,目标落在高概率团上)")

        # --- 图B:greedy 路径 叠加在概率场上(看 Follower 相对先验的走向)---
        figB, axB = plot.subplots(figsize=(8.5, 4.2))
        imB = axB.imshow(field.T, origin="lower", extent=ext, aspect="auto",
                         cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.85)
        figB.colorbar(imB, ax=axB, label="static-prior target probability")
        axB.scatter(tg[:, 0], tg[:, 1], s=70, facecolors="none", edgecolors="red",
                    linewidths=1.5, marker="s", zorder=5, label="targets")
        routes = solve_greedy(cfg, targets)
        for i, (sp, r) in enumerate(zip(starts, routes)):
            if not r:
                continue
            pts = np.asarray([sp[:2]] + [np.asarray(p, float) for p in r])
            axB.plot(pts[:, 0], pts[:, 1], "-o", color=colors[i % 4], ms=4, mec="white",
                     zorder=4, label=f"Follower{i} route (greedy)")
            axB.plot(sp[0], sp[1], "^", color=colors[i % 4], ms=11, mec="white", zorder=6)
            axB.text(sp[0], sp[1], f"  Start F{i}", color="white",
                     fontweight="bold", va="center", ha="left", fontsize=8)
        axB.set_xlabel("North x (m)"); axB.set_ylabel("East y (m)")
        axB.set_title("Part3: greedy baseline routes over probability field")
        axB.legend(loc="best", fontsize=7)
        pB = os.path.join(OUT_DIR, "part3_baseline_routes.png")
        figB.tight_layout(); figB.savefig(pB, dpi=110); plot.close(figB)
        print(f"  图已存: {pB}  (概率场 + 路径分配)")


# ==================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="波次1 全模块人工验收 harness")
    ap.add_argument("--part", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--no-plot", action="store_true", help="禁用绘图(纯文本验收)")
    args = ap.parse_args()

    print("多AUV協調探査 · 波次1 人工验收(A0 集成 harness)")
    cfg = demo_cfg()
    plot = None if args.no_plot else _try_plot()

    if args.part in ("1", "all"):
        part1_map(cfg, plot)
    if args.part in ("2", "all"):
        part2_auv(cfg, plot)
        part2_leader_reveal(cfg, plot)
    if args.part in ("3", "all"):
        part3_baselines(cfg, plot)

    section("验收汇总")
    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    for label, ok in _RESULTS:
        if not ok:
            print(f"  [XX] {label}")
    print(f"\n  通过 {passed}/{total} 项判据。" + ("  全部通过 [ALL PASS]" if passed == total else "  有未通过项 [FAIL]"))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
