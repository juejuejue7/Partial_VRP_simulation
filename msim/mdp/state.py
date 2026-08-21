"""[第3层][A5] obs 组装(Dict 结构,Q4)。实现 contracts.mdp.assemble_obs。

obs = {"map": float32 (Hc,Wc), "vec": float32 (10,)};**全部归一化到 [0,1]**;
one-at-a-time 下"为哪台决策"由 self/teammate 槽位隐式编码(N=2)。

vec 布局(契约冻结定义,见 contracts.mdp.assemble_obs docstring):
    [ self.e, self.dx, self.dy, self.d_rem, self.t_avail,
      teammate.e, teammate.dx, teammate.dy, teammate.d_rem, teammate.t_avail ]
其中 dx,dy = pos_norm(位置 / 窗口尺度)、d_rem = remaining_distance_norm(d / 窗口尺度)、
t_avail = available_time_norm(t / max_episode_time_s)。各项在上游已归一化,本函数再 clip 到
[0,1] 保证落在 observation_space.Box 内(防浮点越界 / 电量等量纲安全)。

本模块重定义 contracts.mdp.assemble_obs 的**实现**(覆盖 NIE 骨架);FollowerObsState 为
纯数据 dataclass,从 contracts 复用。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from msim.contracts.mdp import FollowerObsState  # 纯数据 dataclass,复用
from msim.contracts.types import OBS_FLOAT, CoarseMap

__all__ = ["FollowerObsState", "assemble_obs"]


def _vec_for(state: FollowerObsState) -> list:
    """单台 Follower 的 5 项归一化向量 [e, dx, dy, d_rem, t_avail]。"""
    pos = np.asarray(state.pos_norm, dtype=np.float64).reshape(-1)
    return [
        float(state.energy_norm),
        float(pos[0]),
        float(pos[1]),
        float(state.remaining_distance_norm),
        float(state.available_time_norm),
    ]


def assemble_obs(coarse_map: CoarseMap,
                 self_state: FollowerObsState,
                 teammate_state: FollowerObsState) -> Dict[str, np.ndarray]:
    """组装 Dict 观测(Q4)。

    返回:
        {
          "map": float32 (Hc, Wc),   # 窗口概率场降采样到粗网格,clip [0,1]
          "vec": float32 (10,),       # [self 5 项 | teammate 5 项],clip [0,1]
        }
    全部归一化。vec 布局以 contracts docstring 为冻结定义。
    """
    map_arr = np.asarray(coarse_map, dtype=OBS_FLOAT)
    map_arr = np.clip(map_arr, 0.0, 1.0).astype(OBS_FLOAT, copy=False)

    vec = np.asarray(_vec_for(self_state) + _vec_for(teammate_state), dtype=OBS_FLOAT)
    vec = np.clip(vec, 0.0, 1.0).astype(OBS_FLOAT, copy=False)

    return {"map": map_arr, "vec": vec}
