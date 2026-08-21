"""[第5层][A6] 评估指标 + 路径代价口径(纯几何,自包含,不依赖 RL/A3 实现)。

两组职责:
1. path 代价口径(path_cost):一条 waypoint 路径 → (能耗, 用时)。
   口径与 contracts/config.py 的 RolloutConfig 注释严格一致(Level 1 解析口径):
     - 能耗 = Σ段长 × length_energy_coeff + Σ|Δψ| × turn_energy_coeff
     - 用时 = Σ段长 / cruise_speed + Σ|Δψ| / nominal_yaw_rate + dwell × (waypoint 数)
   纯几何、确定、快;只读 RolloutConfig 标量,**不 import A3 实现**(旁路不被其进度阻塞)。
   待 A3 的 Level1RolloutEngine 落地后,可由 A0 决定是否统一切到它。

2. 评估指标(§4.2):max / (CV 或 Jain 二选一) / 总能耗 / 推理时间。
   CV 与 Jain 互为单调变换(J = 1/(1+CV²)),报告时二选一不并列。

⚠ 措辞:本文件只计算确定量,不产生 ground truth 标签;ground truth 仅指 exact_milp 小实例精确解。
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from msim.contracts.config import RolloutConfig
from msim.contracts.types import wrap_to_pi


# ======================================================================
# path 代价口径(Level 1 解析,自包含)
# ======================================================================
def _heading(seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """从 seg_start 指向 seg_end 的航向(NED:从 North(+x) 起、向 East(+y) 为正)。

    与 types.py 坐标约定一致:psi = atan2(dy_East, dx_North)。零长度段返回 0。
    """
    d = np.asarray(seg_end, dtype=float) - np.asarray(seg_start, dtype=float)
    if float(np.hypot(d[0], d[1])) < 1e-12:
        return 0.0
    return math.atan2(float(d[1]), float(d[0]))  # atan2(East, North)


def path_cost(
    start_pose: np.ndarray,
    waypoints: Sequence[np.ndarray],
    cfg: RolloutConfig,
) -> Tuple[float, float]:
    """一条路径(从 start_pose 依次经过 waypoints)的解析 (能耗_J, 用时_s)。

    ⚠ baselines 本地镜像 RolloutConfig 口径,与 contracts §v1.0.1 一致(单一真相源):
        段长   L_k = ‖w_k − w_{k-1}‖(w_0 = start_pose 的位置)
        段向   s_k = dir(w_{k-1} → w_k)              # 仅对非零长段定义
        转角   Σ|Δψ| = Σ_{k=1}^{n-1} |wrap(s_k − s_{k-1})|
               **只累计路径内相邻段间转角,忽略首段"转入角"(start_pose[2] 不参与转角)。**
               这样基线代价与 A3 的 Level1RolloutEngine 同源(§v1.0.1)。
        能耗 = ΣL_k · length_energy_coeff + Σ|Δψ| · turn_energy_coeff
        用时 = ΣL_k / cruise_speed_mps + Σ|Δψ| / nominal_yaw_rate_rps
               + dwell_time_s · len(waypoints)

    start_pose: shape (3,) [x_N, y_E, psi];psi 仅用于(未来)接续,**不计入本段转角**。
    waypoints: 每个 shape (2,) [x_N, y_E]。空 waypoints → (0, 0)。
    **只读 cfg 标量,不修改输入。**集成门会断言本函数与 A3 引擎在同 routes 上数值一致。
    """
    start_pose = np.asarray(start_pose, dtype=float)
    if len(waypoints) == 0:
        return 0.0, 0.0

    prev_pos = start_pose[:2].copy()
    prev_heading = None  # 首个有效段方向作基准;首段不计转入角(§v1.0.1)

    total_len = 0.0
    total_turn = 0.0
    for wp in waypoints:
        wp = np.asarray(wp, dtype=float)
        seg = wp - prev_pos
        seg_len = float(np.hypot(seg[0], seg[1]))
        total_len += seg_len
        if seg_len >= 1e-12:
            h = math.atan2(float(seg[1]), float(seg[0]))  # atan2(East, North)
            if prev_heading is not None:  # 跳过首段:只累计相邻段间转角
                total_turn += abs(wrap_to_pi(h - prev_heading))
            prev_heading = h
        prev_pos = wp

    energy_J = (
        total_len * cfg.length_energy_coeff + total_turn * cfg.turn_energy_coeff
    )
    duration_s = (
        total_len / cfg.cruise_speed_mps
        + total_turn / cfg.nominal_yaw_rate_rps
        + cfg.dwell_time_s * len(waypoints)
    )
    return energy_J, duration_s


def path_length(start_pose: np.ndarray, waypoints: Sequence[np.ndarray]) -> float:
    """纯路径长度(不含转弯/停留),供求解器内部建图(距离矩阵)用。"""
    start_pose = np.asarray(start_pose, dtype=float)
    prev = start_pose[:2].copy()
    total = 0.0
    for wp in waypoints:
        wp = np.asarray(wp, dtype=float)
        total += float(np.hypot(wp[0] - prev[0], wp[1] - prev[1]))
        prev = wp
    return total


def routes_to_energies(
    routes: List[List[np.ndarray]],
    start_poses: Sequence[np.ndarray],
    cfg: RolloutConfig,
) -> List[float]:
    """N 条 Follower 路径 → per-follower 能耗 list(供 metrics / 均衡评估)。

    routes[i] 是第 i 台 Follower 的 waypoint 序列;start_poses[i] 是其出发位姿。
    路径为空 → 该 Follower 能耗 0。
    """
    if len(start_poses) != len(routes):
        raise ValueError(
            f"routes({len(routes)}) 与 start_poses({len(start_poses)}) 数量不一致"
        )
    out: List[float] = []
    for route, sp in zip(routes, start_poses):
        e, _ = path_cost(sp, route, cfg)
        out.append(e)
    return out


# ======================================================================
# 评估指标(§4.2)
# ======================================================================
def max_energy(per_follower_energy_J: Sequence[float]) -> float:
    """min-max 目标本身:max_i E_i。空输入返回 0。"""
    if len(per_follower_energy_J) == 0:
        return 0.0
    return float(max(per_follower_energy_J))


def coefficient_of_variation(per_follower_energy_J: Sequence[float]) -> float:
    """能耗离散度 CV = std/mean(总体标准差,ddof=0)。

    mean=0(全空)→ 0;只有一台 Follower → 0(无离散)。
    """
    arr = np.asarray(per_follower_energy_J, dtype=float)
    if arr.size == 0:
        return 0.0
    mean = float(arr.mean())
    if mean <= 0.0:
        return 0.0
    return float(arr.std(ddof=0) / mean)


def jain_index(per_follower_energy_J: Sequence[float]) -> float:
    """Jain 公平指数 ∈ (0,1]:J = (Σx)² / (n·Σx²)。

    完全均衡 = 1;与 CV 单调对应(J = 1/(1+CV²),报告时二选一)。
    全零 → 1(退化为完全公平)。
    """
    arr = np.asarray(per_follower_energy_J, dtype=float)
    n = arr.size
    if n == 0:
        return 1.0
    s = float(arr.sum())
    sq = float((arr * arr).sum())
    if sq <= 0.0:
        return 1.0
    return float((s * s) / (n * sq))


def total_energy(per_follower_energy_J: Sequence[float]) -> float:
    """总能耗 Σ_i E_i(揭示均衡 vs 效率 trade-off)。"""
    if len(per_follower_energy_J) == 0:
        return 0.0
    return float(sum(per_follower_energy_J))


def summarize(
    per_follower_energy_J: Sequence[float],
    inference_time_s: float,
    *,
    fairness: str = "jain",
) -> dict:
    """把一次求解结果汇总为 §4.2 四类指标。

    fairness: "jain" | "cv"(二选一,不并列报告)。
    返回 dict:{max, total, fairness_metric, fairness_value, inference_time_s, per_follower_energy_J}。
    """
    if fairness == "jain":
        fval = jain_index(per_follower_energy_J)
    elif fairness == "cv":
        fval = coefficient_of_variation(per_follower_energy_J)
    else:
        raise ValueError(f"fairness 须为 'jain' 或 'cv',得到 {fairness!r}")
    return {
        "max": max_energy(per_follower_energy_J),
        "total": total_energy(per_follower_energy_J),
        "fairness_metric": fairness,
        "fairness_value": fval,
        "inference_time_s": float(inference_time_s),
        "per_follower_energy_J": [float(e) for e in per_follower_energy_J],
    }
