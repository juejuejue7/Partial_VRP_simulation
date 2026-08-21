"""[第2a层][A3] Leader 特化:用 HeterogeneousAUV("leader") 沿 lawnmower 动力学推进。

Q16:Leader 经 AUV 动力学维护真实位姿(window 由其实际位姿生成);能耗累计但**不进 reward**。
实现 contracts.physics.LeaderActorProtocol。

时间口径:env 维护连续全局时钟,每次推进到事件时刻调 advance(dt);本类内部按 sim_dt_s
细分积分到该时刻,沿 lawnmower waypoint 逐点用 AUV 类 step_motion 推进。dt 不必是 sim_dt 整数倍,
末尾以余量步长补齐(保证积分到精确 dt)。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from msim.config.config import apply_overrides
from msim.contracts.config import SimConfig
from msim.contracts.types import Pose, Waypoint
from msim.physics.auv_dynamics import AUVConfig, HeterogeneousAUV


class LeaderActor:
    """满足 LeaderActorProtocol。

    Leader 沿外生预设 lawnmower 测线(env_static.leader_path 产出)用 AUV 动力学逐点推进,
    维护真实动力学位姿;累计能耗为旁路统计,**不进 reward**(Q16)。
    """

    def __init__(self, cfg: SimConfig, lawnmower: List[Waypoint]) -> None:
        self._cfg = cfg
        # 防御性拷贝,定为 (M,2) 数组,顺序即推进顺序。
        self._lawnmower: List[np.ndarray] = [
            np.asarray(w, dtype=np.float64)[:2].copy() for w in lawnmower
        ]
        self._auv_cfg: AUVConfig = apply_overrides(AUVConfig(), cfg.leader_overrides)
        self._sim_dt: float = cfg.rollout.sim_dt_s

        self._auv: Optional[HeterogeneousAUV] = None
        self._wp_cursor: int = 0           # 当前正驶向 lawnmower[_wp_cursor]
        self._energy_J: float = 0.0        # 累计能耗(旁路统计,不进 reward)
        self.reset()

    # ------------------------------------------------------------------
    # 只读属性(LeaderActorProtocol)
    # ------------------------------------------------------------------
    @property
    def pose(self) -> Pose:
        """Leader 当前动力学位姿 [x_N, y_E, psi]。"""
        assert self._auv is not None
        return self._auv.eta.copy()

    @property
    def energy_J_cumulative(self) -> float:
        """累计能耗(旁路统计,**不进 reward**)。"""
        return float(self._energy_J)

    @property
    def finished(self) -> bool:
        """是否已走完整条 lawnmower 测线(episode 终止判据之一)。

        游标越过最后一个 waypoint(已抵达终点)即 finished。空测线视为立即 finished。
        """
        return self._wp_cursor >= len(self._lawnmower)

    # ------------------------------------------------------------------
    # 复位
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """复位:回到测线起点(首个 waypoint),电量满,能耗清零,游标指向首点。

        起点位姿:位置=lawnmower[0];朝向=朝向 lawnmower[1] 的方向(若存在),否则 0。
        起点即首 waypoint,故出发即驶向第二个点(_wp_cursor=1)。空/单点测线特判。
        """
        if len(self._lawnmower) == 0:
            # 无测线:构造在原点的占位 AUV,立即 finished。
            self._auv = HeterogeneousAUV("leader", np.zeros(3, dtype=np.float64), cfg=self._auv_cfg)
            self._wp_cursor = 0
            self._energy_J = 0.0
            return

        start_xy = self._lawnmower[0]
        if len(self._lawnmower) >= 2:
            d = self._lawnmower[1] - start_xy
            start_psi = float(np.arctan2(d[1], d[0])) if np.hypot(d[0], d[1]) > 0.0 else 0.0
        else:
            start_psi = 0.0
        init_pose = np.array([start_xy[0], start_xy[1], start_psi], dtype=np.float64)
        self._auv = HeterogeneousAUV("leader", init_pose, cfg=self._auv_cfg)
        # 起点即首 waypoint,直接驶向下一个;单点测线则已在终点 → finished。
        self._wp_cursor = 1
        self._energy_J = 0.0

    # ------------------------------------------------------------------
    # 推进
    # ------------------------------------------------------------------
    def advance(self, dt: float) -> None:
        """沿 lawnmower 用 AUV 动力学推进 dt 秒(内部按 sim_dt_s 细分积分)。

        每个 sim_dt 子步朝当前目标 waypoint step_motion;抵达 arrive_radius 内则游标推进到
        下一点。走完最后一点后剩余子步原地保持(能耗仍累计静功耗,位姿不再前进)。
        dt 非 sim_dt 整数倍时,末尾以余量子步补齐到精确 dt。
        """
        assert self._auv is not None
        if dt <= 0.0:
            return

        remaining = float(dt)
        eps = 1e-12
        while remaining > eps:
            sub_dt = self._sim_dt if remaining >= self._sim_dt else remaining
            remaining -= sub_dt
            self._step_once(sub_dt)

    def _step_once(self, sub_dt: float) -> None:
        """单个子步:朝当前目标 waypoint 推进 sub_dt,并处理到点切换。"""
        assert self._auv is not None
        if self.finished:
            # 已走完测线:原地保持(目标=当前位置),仅累计静功耗,位姿基本不变。
            self._energy_J += self._auv.step_motion(self._auv.pos, sub_dt)
            return

        target = self._lawnmower[self._wp_cursor]
        self._energy_J += self._auv.step_motion(target, sub_dt)
        # 到点判定:进入 arrive_radius 即切换下一目标。
        if np.linalg.norm(self._auv.pos - target) < self._auv.cfg.arrive_radius_m:
            self._wp_cursor += 1
