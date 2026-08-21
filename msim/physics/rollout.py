"""[第2a层][A3] rollout 窄接口的两种实现。实现 contracts.physics.RolloutEngine。

- Level1RolloutEngine(修士训练默认):路径长度+转弯惩罚(能耗)、长度/cruise+转弯时间近似+停留(用时);
  纯几何、确定、快;不调用 AUV 类。state t_avail / 事件调度全用此口径。
- Level2RolloutEngine(验证旁路,Q2):用 AUV 类 step_motion 细步长积分,能耗/时间物理真实(含转弯);
  不进训练 reward,仅高保真分析 / 第②层真机对接。

时间尺度隔离(architecture §4.1):2b/RL 只经本窄接口向 2a 索要"起点+waypoint 序列
→ (能耗, 用时, 末位姿)",不碰 dt 级积分细节。Level 1/2 是同一 Protocol 的两实现,非两套代码。

==== Level 1 解析口径(DECISIONS.md §v1.0.1;A0 冻结:相邻段间转角,不计首段转入角)====
对 start_pose=[x0,y0,psi0] 与 waypoints=[w0,...,w_{n-1}](位置序列 p=start[:2], w_0..w_{n-1}):
- 段长 L_k = ‖w_k − w_{k-1}‖,其中 w_{-1} := p;总长 L = Σ_{k=0}^{n-1} L_k。
- 段方向 s_0 = dir(p→w_0),s_k = dir(w_{k-1}→w_k);转角 Σ|Δψ| = Σ_{k=1}^{n-1} |wrap(s_k − s_{k-1})|
  (共 n−1 项;n=1 时为 0)。**不含** start_pose.psi→s_0 的首段转入角:start_pose.psi 仅供
  Level 2 积分初值与 end_pose 朝向输出,不进 Level 1 用时/能耗。这是承诺队列**解析确定量**,
  与 mdp.resolve_available_time(只有 current_pos、无朝向)严格同口径。
- 用时 = L/cruise_speed + Σ|Δψ|/nominal_yaw_rate + dwell_time × n(每 waypoint 停留)。
- 能耗 = length_energy_coeff × L + turn_energy_coeff × Σ|Δψ|。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from msim.contracts.config import RolloutConfig, SimConfig
from msim.contracts.physics import RolloutEngine, RolloutResult  # re-export
from msim.contracts.types import Pose, Waypoint, wrap_to_pi
from msim.physics.auv_dynamics import AUVConfig, HeterogeneousAUV

__all__ = ["RolloutResult", "Level1RolloutEngine", "Level2RolloutEngine"]


# ======================================================================
# Level 1 解析几何核(纯函数,无 AUV 类;口径单一真相源)
# ======================================================================
def _segment_geometry(
    start_pose: Pose, waypoints: Sequence[Waypoint]
) -> Tuple[List[float], List[float]]:
    """把 start_pose + waypoints 拆成 (每段长度, 每段转入转角绝对值)。

    返回 (leg_lengths, turn_abs):
      leg_lengths[k] = ‖w_k − w_{k-1}‖(w_{-1}=start)。长度 n。
      turn_abs[k]    = |wrap(h_k − h_{k-1})|(k≥1),turn_abs[0]=0(第一段无前序方向)。长度 n。
    两者均"解析确定量",与 mdp.resolve_available_time 的相邻段间转角口径一致(A0 裁决)。
    """
    start_xy = np.asarray(start_pose, dtype=np.float64)[:2]
    pts = [start_xy] + [np.asarray(w, dtype=np.float64)[:2] for w in waypoints]

    leg_lengths: List[float] = []
    headings: List[float] = []
    for k in range(1, len(pts)):
        d = pts[k] - pts[k - 1]
        leg_lengths.append(float(np.hypot(d[0], d[1])))
        # 零长段方向不确定,沿用前一段方向(不引入虚假转角);首段零长则朝向取 0
        if leg_lengths[-1] > 0.0:
            headings.append(float(np.arctan2(d[1], d[0])))
        else:
            headings.append(headings[-1] if headings else 0.0)

    turn_abs: List[float] = []
    for k in range(len(leg_lengths)):
        if k == 0:
            turn_abs.append(0.0)  # 第一段无前序方向 → 不计转入角(口径 B)
        else:
            turn_abs.append(abs(wrap_to_pi(headings[k] - headings[k - 1])))
    return leg_lengths, turn_abs


class Level1RolloutEngine:
    """长度+转弯的解析 rollout(修士训练)。满足 RolloutEngine。

    全程纯几何确定量,不构造 AUV 类。per_wp_* 给每段(到达每个 waypoint 那刻)的代价,
    供价值"到达那刻"结算与能耗均衡逐段记账。
    """

    def __init__(self, cfg: RolloutConfig) -> None:
        self._cfg = cfg

    def execute(self, start_pose: Pose, waypoints: Sequence[Waypoint]) -> RolloutResult:
        cfg = self._cfg
        start = np.asarray(start_pose, dtype=np.float64).copy()

        if len(waypoints) == 0:
            # 空序列:零代价,末位姿即起点(原样返回,含朝向)。
            return RolloutResult(
                energy_J=0.0,
                duration_s=0.0,
                end_pose=start.copy(),
                per_wp_energy_J=[],
                per_wp_duration_s=[],
                trajectory=None,
            )

        leg_lengths, turn_abs = _segment_geometry(start, waypoints)

        per_wp_energy: List[float] = []
        per_wp_duration: List[float] = []
        for k in range(len(leg_lengths)):
            seg_energy = (
                cfg.length_energy_coeff * leg_lengths[k]
                + cfg.turn_energy_coeff * turn_abs[k]
            )
            seg_duration = (
                leg_lengths[k] / cfg.cruise_speed_mps
                + turn_abs[k] / cfg.nominal_yaw_rate_rps
                + cfg.dwell_time_s  # 每到达一个 waypoint 停留拍照
            )
            per_wp_energy.append(seg_energy)
            per_wp_duration.append(seg_duration)

        # 末位姿:位置=最后一个 waypoint;朝向=最后一段方向(零长段沿用前序)。
        last_wp = np.asarray(waypoints[-1], dtype=np.float64)[:2]
        end_psi = _final_heading(start, waypoints, fallback_psi=float(start[2]))
        end_pose = np.array([last_wp[0], last_wp[1], end_psi], dtype=np.float64)

        return RolloutResult(
            energy_J=float(sum(per_wp_energy)),
            duration_s=float(sum(per_wp_duration)),
            end_pose=end_pose,
            per_wp_energy_J=per_wp_energy,
            per_wp_duration_s=per_wp_duration,
            trajectory=None,
        )


def _final_heading(start_pose: Pose, waypoints: Sequence[Waypoint], fallback_psi: float) -> float:
    """最后一段的航向(end_pose.psi)。所有段皆零长时回退到 fallback_psi。"""
    pts = [np.asarray(start_pose, dtype=np.float64)[:2]] + [
        np.asarray(w, dtype=np.float64)[:2] for w in waypoints
    ]
    for k in range(len(pts) - 1, 0, -1):
        d = pts[k] - pts[k - 1]
        if float(np.hypot(d[0], d[1])) > 0.0:
            return float(np.arctan2(d[1], d[0]))
    return fallback_psi


# ======================================================================
# Level 2 高保真 rollout(用 AUV 类 step_motion 细步长积分,验证旁路 Q2)
# ======================================================================
class Level2RolloutEngine:
    """AUV 积分 rollout(高保真验证旁路)。满足 RolloutEngine。

    用 HeterogeneousAUV 沿 waypoint 序列以 sim_dt_s 细步长积分,能耗/用时均物理真实(含转弯、
    含静功耗与传感器功耗)。仅用于高保真分析 / 第②层真机对接,**不进训练 reward**(Q2)。

    与 Level 1 的对应:到点判定用 AUV 类 arrive_radius_m;dwell 停留以静功耗形式计入能耗与时间,
    口径与 Level 1 的 dwell_time_s 一致(per waypoint)。
    """

    # 单条 waypoint 的最大积分步数保护(防 LOS 制导极端情形不收敛而死循环)。
    _MAX_STEPS_PER_WP = 200_000

    def __init__(
        self,
        cfg: SimConfig,
        auv_type: str = "follower",
        record_trajectory: bool = False,
    ) -> None:
        self._cfg = cfg
        self._auv_type = auv_type
        self._record_trajectory = record_trajectory

    def _build_auv(self, start_pose: Pose) -> HeterogeneousAUV:
        """按异构 override 构造 AUV(异构唯一注入路径:config.apply_overrides)。"""
        from msim.config.config import apply_overrides  # 运行期依赖 A2,延迟 import

        overrides = (
            self._cfg.leader_overrides
            if self._auv_type == "leader"
            else self._cfg.follower_overrides
        )
        auv_cfg = apply_overrides(AUVConfig(), overrides)
        return HeterogeneousAUV(self._auv_type, np.asarray(start_pose, dtype=np.float64), cfg=auv_cfg)

    def execute(self, start_pose: Pose, waypoints: Sequence[Waypoint]) -> RolloutResult:
        cfg = self._cfg
        dt = cfg.rollout.sim_dt_s
        dwell = cfg.rollout.dwell_time_s
        start = np.asarray(start_pose, dtype=np.float64).copy()

        if len(waypoints) == 0:
            return RolloutResult(
                energy_J=0.0,
                duration_s=0.0,
                end_pose=start.copy(),
                per_wp_energy_J=[],
                per_wp_duration_s=[],
                trajectory=None,
            )

        auv = self._build_auv(start)
        arrive_r = auv.cfg.arrive_radius_m

        per_wp_energy: List[float] = []
        per_wp_duration: List[float] = []
        traj: List[np.ndarray] = []
        if self._record_trajectory:
            traj.append(auv.eta.copy())

        for w in waypoints:
            wp = np.asarray(w, dtype=np.float64)[:2]
            seg_energy = 0.0
            seg_duration = 0.0
            # --- 航行段:积分到 arrive_radius 内 ---
            steps = 0
            while np.linalg.norm(auv.pos - wp) >= arrive_r:
                seg_energy += auv.step_motion(wp, dt)
                seg_duration += dt
                steps += 1
                if self._record_trajectory:
                    traj.append(auv.eta.copy())
                if steps >= self._MAX_STEPS_PER_WP:
                    break  # 保护:不无限积分(极端 LOS 情形)
            # --- 停留段:原地 dwell,以静功耗形式计入能耗与时间 ---
            n_dwell = int(round(dwell / dt))
            for _ in range(n_dwell):
                # 目标设为当前位姿 → 速度趋零,只产生静态+传感器功耗
                seg_energy += auv.step_motion(auv.pos, dt)
                seg_duration += dt
                if self._record_trajectory:
                    traj.append(auv.eta.copy())

            per_wp_energy.append(seg_energy)
            per_wp_duration.append(seg_duration)

        trajectory = np.asarray(traj, dtype=np.float64) if self._record_trajectory else None
        return RolloutResult(
            energy_J=float(sum(per_wp_energy)),
            duration_s=float(sum(per_wp_duration)),
            end_pose=auv.eta.copy(),
            per_wp_energy_J=per_wp_energy,
            per_wp_duration_s=per_wp_duration,
            trajectory=trajectory,
        )
