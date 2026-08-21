"""
dynamics.py
============
异构 AUV 的 3-DOF 水平面动力学（Fossen 方程的简化版）。

Fossen 方程 (3-DOF)：
    M * dν/dt + C(ν) * ν + D(ν) * ν = τ
其中：
    η = [x, y, ψ]^T 是世界坐标系位姿
    ν = [u, v, r]^T 是机体坐标系速度 (surge, sway, yaw-rate)
    M = diag(m_u, m_v, I_z) 是刚体 + 附加质量
    D(ν) = D_l + D_q(ν)，分别为线性与二次阻尼 (对角近似)
    C(ν) Coriolis 项在小速度下忽略

位姿更新 (body → world)：
    dx/dt = u cosψ - v sinψ
    dy/dt = u sinψ + v cosψ
    dψ/dt = r
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------
@dataclass
class AUVConfig:
    """异构 AUV 物理与能耗参数。"""

    # 质量 / 惯量 (kg, kg*m^2)
    m_u: float = 358.0
    m_v: float = 647.0
    I_z: float = 40.1
    # 线性阻尼
    X_u: float = 11.0
    Y_v: float = 4.0
    N_r: float = 36.4
    # 二次阻尼
    X_uu_fwd: float = 114.0
    X_uu_rev: float = 684.0  # 后退阻力显著大于前进
    Y_vv: float = 1242.0
    N_rr: float = 8.4
    # 控制器增益
    Kp_u: float = 400.0
    Kp_v: float = 800.0
    Kp_r: float = 150.0
    # 推力限幅 (N / N·m)
    tau_u_max: float = 300.0
    tau_v_max: float = 250.0
    tau_r_max: float = 50.0
    # 巡航速度 (m/s)
    cruise_speed: float = 1.5
    # 电池容量 (J)
    battery_capacity_J: float = 5.0e6
    # 传感器静功耗 (W)：Leader 多波束声呐远高于 Follower 光学
    sensor_static_power_W: float = 25.0
    # 能耗系数 (推力 → 电功率，P ∝ |F|^1.5)
    c_fwd: float = 0.5
    c_rev: float = 1.0
    c_sway: float = 0.1
    c_yaw: float = 0.75
    p_static_W: float = 1.0
    # 到点判定
    arrive_radius_m: float = 1.5


# ----------------------------------------------------------------------
# 异构 AUV
# 注：Leader / Follower 的异构差异项 (cruise_speed、battery_capacity_J、
#     sensor_static_power_W、arrive_radius_m) 由 configs/default.yaml 提供，
#     通过 envs/config.apply_overrides 注入。本文件只保留通用物理常数。
# ----------------------------------------------------------------------
class HeterogeneousAUV:
    """可配置为 Leader 或 Follower 的 3-DOF AUV。"""

    def __init__(
        self,
        auv_type: str,
        init_pose: np.ndarray,
        init_vel: Optional[np.ndarray] = None,
        cfg: Optional[AUVConfig] = None,
    ) -> None:
        """
        :param auv_type: "leader" 或 "follower"
        :param init_pose: [x(m), y(m), yaw(rad)]
        :param init_vel: [u(m/s), v(m/s), r(rad/s)]，默认 0。
        :param cfg: 物理参数，若为 None 按类型选默认。
        """
        if auv_type not in ("leader", "follower"):
            raise ValueError(f"auv_type must be leader/follower, got {auv_type}")
        self.auv_type = auv_type
        # 默认使用通用 AUVConfig；Leader/Follower 差异靠外部 YAML 注入
        self.cfg: AUVConfig = cfg if cfg is not None else AUVConfig()

        self.eta: np.ndarray = np.array(init_pose, dtype=np.float64).copy()
        self.nu: np.ndarray = (
            np.zeros(3, dtype=np.float64) if init_vel is None else np.array(init_vel, dtype=np.float64).copy()
        )
        self.energy_norm: float = 1.0
        self.step_energy_J: float = 0.0
        self.is_busy: bool = False
        self.target_wp: Optional[np.ndarray] = None
        self.last_tau: np.ndarray = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # 便捷只读属性
    # ------------------------------------------------------------------
    @property
    def pos(self) -> np.ndarray:
        return self.eta[:2].copy()

    @property
    def yaw(self) -> float:
        return float(self.eta[2])

    # ------------------------------------------------------------------
    # 复位
    # ------------------------------------------------------------------
    def reset(self, pose: np.ndarray, vel: Optional[np.ndarray] = None) -> None:
        self.eta = np.array(pose, dtype=np.float64).copy()
        self.nu = np.zeros(3) if vel is None else np.array(vel, dtype=np.float64).copy()
        self.energy_norm = 1.0
        self.step_energy_J = 0.0
        self.is_busy = False
        self.target_wp = None
        self.last_tau = np.zeros(3)

    # ------------------------------------------------------------------
    # LOS 制导：航点 → 目标速度
    # ------------------------------------------------------------------
    def waypoint_to_target_vel(self, wp: np.ndarray) -> np.ndarray:
        """Line-Of-Sight 制导：
            ψ_des = atan2(Δy, Δx)
            r_target = Kψ * wrap(ψ_des - ψ)
            u_target = cruise_speed, 接近终点时线性减速
            v_target = 0 (期望零横移)
        """
        dx = wp[0] - self.eta[0]
        dy = wp[1] - self.eta[1]
        dist = float(np.hypot(dx, dy))
        psi_des = np.arctan2(dy, dx)
        psi_err = _wrap_to_pi(psi_des - self.eta[2])

        # 接近终点时减速：距离 < 2*arrive_radius 线性减速
        slow_r = max(1.0, 2.0 * self.cfg.arrive_radius_m)
        u_target = self.cfg.cruise_speed * min(1.0, dist / slow_r)
        
        # 朝向偏差太大时先转向再前进，避免盘旋
        if abs(psi_err) > np.deg2rad(5.0):
            # u_target *= max(0.0, np.cos(psi_err))
            u_target = 0
        v_target = 0.0
        # r_target 给一个舒适的转弯角速度上限 (rad/s)
        r_target = float(np.clip(1.5 * psi_err, -np.deg2rad(30.0), np.deg2rad(30.0)))
        return np.array([u_target, v_target, r_target], dtype=np.float64)

    # ------------------------------------------------------------------
    # 底层 P 控制器
    # ------------------------------------------------------------------
    def low_level_controller(self, target_vel: np.ndarray) -> np.ndarray:
        """三路独立 P 控制，输出机体系广义力 τ = [τ_u, τ_v, τ_r]，并做推力限幅。"""
        u_cur, v_cur, r_cur = self.nu
        u_tgt, v_tgt, r_tgt = target_vel
        tau_u = np.clip(self.cfg.Kp_u * (u_tgt - u_cur), -self.cfg.tau_u_max, self.cfg.tau_u_max)
        tau_v = np.clip(self.cfg.Kp_v * (v_tgt - v_cur), -self.cfg.tau_v_max, self.cfg.tau_v_max)
        tau_r = np.clip(self.cfg.Kp_r * (r_tgt - r_cur), -self.cfg.tau_r_max, self.cfg.tau_r_max)
        return np.array([tau_u, tau_v, tau_r], dtype=np.float64)

    # ------------------------------------------------------------------
    # 动力学积分
    # ------------------------------------------------------------------
    def dynamics_step(self, tau: np.ndarray, dt: float) -> None:
        """按 Fossen 方程的简化形式积分。

        牛顿第二定律（对角近似）：
            a_u = (τ_u - X_u*u - X_uu(u)*|u|*u) / m_u
            a_v = (τ_v - Y_v*v - Y_vv*|v|*v)   / m_v
            a_r = (τ_r - N_r*r - N_rr*|r|*r)   / I_z
        surge 方向正反向阻尼不对称：后退阻力远大于前进。
        """
        u, v, r = self.nu
        cfg = self.cfg
        # surge 阻尼（非对称）
        if u >= 0:
            drag_u = cfg.X_u * u + cfg.X_uu_fwd * abs(u) * u
        else:
            drag_u = cfg.X_u * u + cfg.X_uu_rev * abs(u) * u
        drag_v = cfg.Y_v * v + cfg.Y_vv * abs(v) * v
        drag_r = cfg.N_r * r + cfg.N_rr * abs(r) * r

        a_u = (tau[0] - drag_u) / cfg.m_u
        a_v = (tau[1] - drag_v) / cfg.m_v
        a_r = (tau[2] - drag_r) / cfg.I_z

        # 速度积分
        self.nu[0] = u + a_u * dt
        self.nu[1] = v + a_v * dt
        self.nu[2] = r + a_r * dt

        # 位姿积分 (body → world)
        psi = self.eta[2]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        self.eta[0] += (self.nu[0] * cos_psi - self.nu[1] * sin_psi) * dt
        self.eta[1] += (self.nu[0] * sin_psi + self.nu[1] * cos_psi) * dt
        self.eta[2] = _wrap_to_pi(psi + self.nu[2] * dt)

    # ------------------------------------------------------------------
    # 能耗映射
    # ------------------------------------------------------------------
    def calculate_energy(self, tau: np.ndarray, dt: float) -> float:
        """电功率经验模型：P_elec_i = c_i * |F_i|^{1.5} (电机+螺旋桨近似非线性)。

        surge 方向后退比前进效率低；额外加静态功耗与传感器功耗。
        返回本步消耗能量 (J)。
        """
        tau_u, tau_v, tau_r = tau
        cfg = self.cfg
        p_u = cfg.c_fwd * abs(tau_u) ** 1.5 if tau_u >= 0 else cfg.c_rev * abs(tau_u) ** 1.5
        p_v = cfg.c_sway * abs(tau_v) ** 1.5
        p_r = cfg.c_yaw * abs(tau_r) ** 1.5
        p_total = p_u + p_v + p_r + cfg.p_static_W + cfg.sensor_static_power_W
        step_J = float(p_total * dt)
        self.step_energy_J = step_J
        self.energy_norm = max(0.0, self.energy_norm - step_J / cfg.battery_capacity_J)
        return step_J

    # ------------------------------------------------------------------
    # 顶层一步
    # ------------------------------------------------------------------
    def step_motion(self, target_wp: np.ndarray, dt: float) -> float:
        """朝 target_wp 推进一小步。返回本步能耗 (J)。"""
        self.target_wp = np.asarray(target_wp, dtype=np.float64).copy()
        tgt_vel = self.waypoint_to_target_vel(self.target_wp)
        tau = self.low_level_controller(tgt_vel)
        self.last_tau = tau
        self.dynamics_step(tau, dt)
        step_J = self.calculate_energy(tau, dt)
        # 到点判定
        if np.linalg.norm(self.pos - self.target_wp) < self.cfg.arrive_radius_m:
            self.is_busy = False
        return step_J


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _wrap_to_pi(angle: float) -> float:
    """把角度规范到 [-π, π)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
