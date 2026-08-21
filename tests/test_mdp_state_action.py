"""[A1 验收测试 · L3 state / action] obs 组装(Dict 归一化)+ 动作纯几何解码。

判据来源:
- state: contracts/mdp.py assemble_obs docstring(Q4)——返回 {"map":float32(Hc,Wc),
  "vec":float32(10,)};vec 布局 [self: e,dx,dy,d_rem,t_avail | teammate 同 5 项];全 ∈[0,1]
  (喂越界值会被 clip)。
- action: contracts/mdp.py decode_action docstring(纯几何)——结果逐位 ==
  geometry.window.coarse_index_to_world(idx, window, cfg.window, cfg.sensor);
  grid 参数不影响结果;action_idx 遍历 [0,Hc*Wc) 落在窗口内(§v1.0.2 A2-Q2:
  QA **独立**解析反推 (row,col),不复用 A2 反函数)。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from msim.contracts.config import SensorConfig, SimConfig, WindowConfig, WorldConfig
from msim.contracts.mdp import FollowerObsState
from msim.geometry.grid import Grid
from msim.geometry.window import (
    coarse_grid_shape,
    coarse_index_to_world,
    get_window,
    window_contains_world,
)
from msim.mdp.action import decode_action
from msim.mdp.coop_survey_env import small_cfg
from msim.mdp.state import assemble_obs


# ======================================================================
# state —— assemble_obs:结构 / dtype / 布局 / 归一化 clip
# ======================================================================
def _obs_state(e, dx, dy, d_rem, t_avail) -> FollowerObsState:
    return FollowerObsState(
        energy_norm=e,
        pos_norm=np.array([dx, dy], dtype=np.float64),
        remaining_distance_norm=d_rem,
        available_time_norm=t_avail,
    )


def test_assemble_obs_structure_and_dtype() -> None:
    Hc, Wc = 2, 3
    cmap = np.full((Hc, Wc), 0.5, dtype=np.float64)
    self_s = _obs_state(0.9, 0.1, 0.2, 0.3, 0.4)
    mate_s = _obs_state(0.8, 0.5, 0.6, 0.7, 0.05)
    obs = assemble_obs(cmap, self_s, mate_s)

    assert set(obs.keys()) == {"map", "vec"}
    assert obs["map"].shape == (Hc, Wc)
    assert obs["map"].dtype == np.float32
    assert obs["vec"].shape == (10,)
    assert obs["vec"].dtype == np.float32


def test_assemble_obs_vec_layout() -> None:
    """vec 布局逐项 == [self: e,dx,dy,d_rem,t_avail | teammate: 同 5 项](契约冻结)。"""
    cmap = np.zeros((1, 1), dtype=np.float64)
    self_s = _obs_state(0.10, 0.11, 0.12, 0.13, 0.14)
    mate_s = _obs_state(0.20, 0.21, 0.22, 0.23, 0.24)
    obs = assemble_obs(cmap, self_s, mate_s)
    expected = np.array(
        [0.10, 0.11, 0.12, 0.13, 0.14, 0.20, 0.21, 0.22, 0.23, 0.24],
        dtype=np.float32,
    )
    np.testing.assert_allclose(obs["vec"], expected, atol=1e-6)


def test_assemble_obs_clips_out_of_range() -> None:
    """喂越界值(负 / >1):map 与 vec 均被 clip 到 [0,1](落 observation_space.Box)。"""
    cmap = np.array([[-0.5, 2.0], [0.3, 1.5]], dtype=np.float64)
    self_s = _obs_state(2.0, -1.0, 0.5, 5.0, -0.3)   # 含越界
    mate_s = _obs_state(-0.2, 1.7, 0.4, -2.0, 0.6)
    obs = assemble_obs(cmap, self_s, mate_s)

    assert obs["map"].min() >= 0.0 and obs["map"].max() <= 1.0
    assert obs["vec"].min() >= 0.0 and obs["vec"].max() <= 1.0
    # clip 具体值:负→0、>1→1、范围内不变
    np.testing.assert_allclose(
        obs["map"], np.array([[0.0, 1.0], [0.3, 1.0]], dtype=np.float32), atol=1e-6
    )
    # self.e=2.0→1, self.dx=-1.0→0, self.d_rem=5.0→1, self.t_avail=-0.3→0
    np.testing.assert_allclose(obs["vec"][0], 1.0, atol=1e-6)
    np.testing.assert_allclose(obs["vec"][1], 0.0, atol=1e-6)
    np.testing.assert_allclose(obs["vec"][3], 1.0, atol=1e-6)
    np.testing.assert_allclose(obs["vec"][4], 0.0, atol=1e-6)


def test_assemble_obs_in_observation_space() -> None:
    """组装出的 obs ∈ env.observation_space(粗网格形状取自 env)。"""
    from msim.mdp.coop_survey_env import CoopSurveyEnv

    cfg = small_cfg()
    env = CoopSurveyEnv(cfg)
    Hc, Wc = coarse_grid_shape(cfg.window, cfg.sensor)
    cmap = np.full((Hc, Wc), 0.5, dtype=np.float64)
    s = _obs_state(0.5, 0.5, 0.5, 0.5, 0.5)
    obs = assemble_obs(cmap, s, s)
    assert env.observation_space.contains(obs)


# ======================================================================
# action —— decode_action 纯几何:逐位 == coarse_index_to_world
# ======================================================================
def _coarse_centers_window():
    cfg = small_cfg()
    Hc, Wc = coarse_grid_shape(cfg.window, cfg.sensor)
    window = get_window(np.array([20.0, 10.0, 0.37]), cfg.window)
    grid = Grid(cfg.world)
    return cfg, Hc, Wc, window, grid


def test_decode_action_equals_coarse_index_to_world() -> None:
    cfg, Hc, Wc, window, grid = _coarse_centers_window()
    for idx in range(Hc * Wc):
        wp = decode_action(idx, window, grid, cfg)
        ref = coarse_index_to_world(idx, window, cfg.window, cfg.sensor)
        np.testing.assert_allclose(np.asarray(wp, dtype=float), np.asarray(ref, dtype=float), atol=1e-12)


def test_decode_action_grid_independent() -> None:
    """grid 参数不影响结果:传不同 grid、同 idx → 同 waypoint(纯几何,不依赖细网格)。"""
    cfg, Hc, Wc, window, _ = _coarse_centers_window()
    grid_a = Grid(cfg.world)
    grid_b = Grid(WorldConfig(x_max_m=123.0, y_max_m=77.0, res_m=0.5))  # 完全不同细网格
    for idx in range(Hc * Wc):
        wa = decode_action(idx, window, grid_a, cfg)
        wb = decode_action(idx, window, grid_b, cfg)
        np.testing.assert_allclose(np.asarray(wa, dtype=float), np.asarray(wb, dtype=float), atol=1e-12)


def test_decode_action_all_indices_inside_window() -> None:
    """action_idx 遍历 [0,Hc*Wc) → 解码 waypoint 落在窗口内(粗格中心必在窗口)。"""
    cfg, Hc, Wc, window, grid = _coarse_centers_window()
    for idx in range(Hc * Wc):
        wp = decode_action(idx, window, grid, cfg)
        assert window_contains_world(window, wp), f"idx {idx} 解码点不在窗口内"


def test_decode_action_independent_rowcol_recovery() -> None:
    """QA **独立**解析反推 (row,col)(§v1.0.2 A2-Q2:不复用 A2 反函数,防共因 bug):
    对窗口位姿,把解码世界点投回窗口局部坐标 (s 沿后方, t 横向),由格距独立解出
    (row=t 索引, col=s 索引),再 row*Wc+col 须 recover 原 idx。
    """
    cfg, Hc, Wc, window, grid = _coarse_centers_window()
    lx, ly, psi = float(window.leader_pose[0]), float(window.leader_pose[1]), float(window.leader_pose[2])
    # QA 独立构造的后方/横向单位向量(不 import A2 的 _frame)
    back = np.array([-math.cos(psi), -math.sin(psi)])
    lat = np.array([-math.sin(psi), math.cos(psi)])
    cell_along = window.look_back_m / Wc
    cell_lateral = window.width_m / Hc

    for idx in range(Hc * Wc):
        wp = np.asarray(decode_action(idx, window, grid, cfg), dtype=float)
        d = wp - np.array([lx, ly])
        s_c = float(d @ back)                       # 沿后方距离
        t_c = float(d @ lat)                        # 横向偏移
        # 反解索引:s_c=(col+0.5)*cell_along, t_c=-width/2+(row+0.5)*cell_lateral
        col = int(round(s_c / cell_along - 0.5))
        row = int(round((t_c + window.width_m / 2.0) / cell_lateral - 0.5))
        assert 0 <= col < Wc and 0 <= row < Hc, (idx, row, col)
        assert row * Wc + col == idx, f"idx {idx} 反推得 row={row} col={col}"


def test_decode_action_distinct_centers() -> None:
    """不同 idx → 不同粗格中心(单射;粗↔细可逆的必要条件)。"""
    cfg, Hc, Wc, window, grid = _coarse_centers_window()
    pts = [tuple(np.round(np.asarray(decode_action(i, window, grid, cfg), float), 9))
           for i in range(Hc * Wc)]
    assert len(set(pts)) == Hc * Wc, "存在重复粗格中心"
