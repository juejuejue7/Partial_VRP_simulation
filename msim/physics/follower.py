"""[第2a层][A3] Follower 特化:HeterogeneousAUV("follower") + 相机传感器模型。

承诺队列不在此(住 mdp.commitment)。实现 contracts.physics.FollowerProtocol。
仅暴露物理量(pose / energy_norm)+ 相机模型(sensor);"已分配未走完 waypoint" 不在此。
"""
from __future__ import annotations

import numpy as np

from msim.config.config import apply_overrides
from msim.contracts.config import SimConfig
from msim.contracts.physics import CameraSensorModel  # re-export 数据契约
from msim.contracts.types import Pose
from msim.physics.auv_dynamics import AUVConfig, HeterogeneousAUV

__all__ = ["CameraSensorModel", "Follower"]


class Follower:
    """满足 FollowerProtocol。物理状态(异构 follower AUV)+ 相机模型。

    相机模型从 SimConfig.sensor 派生;corridor_width 走 CameraSensorModel.corridor_width_m
    派生属性(= footprint 横向 × 1.0;§v1.0.7,与 grid cell 对齐),不在此重复计算(单一真相源)。
    """

    def __init__(self, cfg: SimConfig) -> None:
        self._cfg = cfg
        self.sensor = CameraSensorModel(
            p_d=cfg.sensor.p_d,
            p_fa=cfg.sensor.p_fa,
            footprint_along_m=cfg.sensor.footprint_along_m,
            footprint_lateral_m=cfg.sensor.footprint_lateral_m,
        )
        # 异构唯一注入路径:把 follower_overrides 合并进 AUVConfig(config 层纯 dict 操作)。
        self._auv_cfg: AUVConfig = apply_overrides(AUVConfig(), cfg.follower_overrides)
        # 初始位姿待 reset 注入;此处构造在原点,reset 后才有效。
        self._auv: HeterogeneousAUV = HeterogeneousAUV(
            "follower", np.zeros(3, dtype=np.float64), cfg=self._auv_cfg
        )

    @property
    def pose(self) -> Pose:
        """Follower 当前动力学位姿 [x_N, y_E, psi]。"""
        return self._auv.eta.copy()

    @property
    def energy_norm(self) -> float:
        """剩余电量归一化 ∈ [0,1](直取 AUV 类 energy_norm)。"""
        return float(self._auv.energy_norm)

    def reset(self, pose: Pose) -> None:
        """复位到给定位姿,电量满。"""
        self._auv.reset(np.asarray(pose, dtype=np.float64))
