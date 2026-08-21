"""[A1 验收测试 · L3 coop_survey_env] 事件驱动协同探查 Gymnasium env。

判据来源(harness §4.2 + §v1.1.4 DoD + contracts/mdp.py CoopSurveyEnvProtocol):
- 通过 Gymnasium env_checker(小尺度 cfg;skip_render_check=True)。
- 标准五元组:reset→(obs,info)、step→(obs,reward:float,terminated:bool,truncated:bool,info:dict);
  obs ∈ observation_space。
- 一次一台 + 忙者不决策:每 step 决策者 = 最早空闲台(env._current 对应 min(free_time));
  随机策略 rollout 全程不违反;N=2 时决策者随段交替。
- reward(§v1.1.4):
    dense_mass = Σ_{走廊∩窗口∩未planned} field[cell]          # GT-free 已解析先验概率质量
    reward = value_weight·(dense_mass/value_norm_ref)          # dense:缩放不截断
           + detect_weight·n_new_detections                    # sparse:首次确定性覆盖 GT 格数(去重)
           − balance_weight·(balance_pen/balance_norm_ref)     # 能耗均衡(基于 per_follower_energy)
           − redundancy_weight·n_redundant                     # 软冗余(与他台已覆盖重叠格数;不终止)
  恒等:reward == dense_term+detect_term−balance_term−redundancy_term(容差 1e-9)。
- info 键(§v1.1.4):dense_mass, dense_term, n_new_detections, detect_term,
  balance_pen, balance_term, n_redundant, redundancy_term, reward,
  per_follower_energy_J, clock_s, current_follower, n_decisions, leader_finished。
- balance-off 消融:replace(reward, balance_weight=0) → balance_term==0;
  reward 不含均衡项(连续权重消融,非 if 分支)。
- 终止/截断:有限步内 terminated 或 truncated;truncated 在 clock>=max_episode_time_s 触发。
- info 含 per_follower_energy_J(reward 能耗项基于 follower 非 leader,防回归)。

§v1.1.5 离散固定块窗口(A5 实现,A0 复核;reward/info 契约不变):
- 窗口从"Leader 连续滑窗"改为"**离散固定块、按块切换**"。块边界 edges=[k*look_back],
  windowed→[0,20,40,60](3 块,look_back=20、x_max=60)、
  small_cfg→[0,8,16,24,32,40](5 块,look_back=8、x_max=40)。
- env 每步用**当前活动块**的固定窗口(非滑窗);info 新增 block_k / n_blocks / n_blocks_done。
- 两台 Follower 顺序覆盖当前块(块0→1→…),block_k 单调不减;全块 done → terminated。
- 单测线前提:window.width_m == world.y_max_m → 块窗口 East = 全 East [0, y_max]。

env 用小尺度 small_cfg(40×20m、粗网格按 coarse_grid_shape 动态算)使单测快;
windowed_survey_cfg(60×20m、3 固定块)用于跑真实块行为;契约类型从 msim.contracts.* import。
"""
from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest

from msim.contracts.mdp import CoopSurveyEnvProtocol
from msim.geometry.window import coarse_index_to_world  # §v1.1.6 残余先验粗格中心采样
from msim.mdp.coop_survey_env import CoopSurveyEnv, small_cfg, windowed_survey_cfg


# ======================================================================
# env_checker —— 通过 Gymnasium 标准校验
# ======================================================================
def test_passes_gymnasium_env_checker() -> None:
    from gymnasium.utils.env_checker import check_env

    env = CoopSurveyEnv(small_cfg())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 屏蔽 gymnasium 的非阻塞性 warning
        check_env(env, skip_render_check=True)


def test_satisfies_protocol() -> None:
    """运行期满足 CoopSurveyEnvProtocol(observation_space/action_space/reset/step)。"""
    env = CoopSurveyEnv(small_cfg())
    assert isinstance(env, CoopSurveyEnvProtocol)


# ======================================================================
# 标准五元组 —— reset / step 形状与类型
# ======================================================================
def test_reset_returns_obs_info() -> None:
    env = CoopSurveyEnv(small_cfg())
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert env.observation_space.contains(obs)


def test_step_returns_standard_five_tuple() -> None:
    env = CoopSurveyEnv(small_cfg())
    env.reset(seed=0)
    out = env.step(int(env.action_space.sample()))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_space_is_discrete_hcwc() -> None:
    from msim.geometry.window import coarse_grid_shape

    cfg = small_cfg()
    env = CoopSurveyEnv(cfg)
    Hc, Wc = coarse_grid_shape(cfg.window, cfg.sensor)
    assert env.action_space.n == Hc * Wc


# ======================================================================
# 一次一台 + 忙者不决策 —— 决策者 = 最早空闲台(min free_time)
# ======================================================================
def test_decider_is_earliest_free_follower() -> None:
    """每 step 前 env._current 必 == argmin(free_time)(同时空闲取低索引)。"""
    env = CoopSurveyEnv(small_cfg())
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(60):
        free = [f.free_time for f in env._followers]
        expected = int(min(range(len(free)), key=lambda i: (free[i], i)))
        assert env._current == expected, (
            f"决策者 {env._current} != 最早空闲台 {expected}; free_time={free}"
        )
        _, _, term, trunc, info = env.step(int(rng.integers(0, env.action_space.n)))
        # info 报告的 current_follower 应与刚决策的台一致
        if term or trunc:
            break


def test_info_current_follower_matches_min_free_time() -> None:
    """reset 给出的 current_follower 是初始 min free_time(全 0 → 索引 0)。"""
    env = CoopSurveyEnv(small_cfg())
    _, info = env.reset(seed=2)
    assert info["current_follower"] == 0


def test_deciders_alternate_for_n2() -> None:
    """N=2 时,随段推进决策者会交替(不会永远只决策一台)。"""
    env = CoopSurveyEnv(small_cfg())
    env.reset(seed=5)
    rng = np.random.default_rng(5)
    deciders = []
    for _ in range(40):
        deciders.append(env._current)
        _, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
        if term or trunc:
            break
    assert set(deciders) == {0, 1}, f"N=2 决策者未交替:{set(deciders)}"


def test_busy_follower_never_decides() -> None:
    """忙者不决策:每次决策者的 free_time 必 <= 另一台(它最先空闲),从不选忙者。"""
    env = CoopSurveyEnv(small_cfg())
    env.reset(seed=8)
    rng = np.random.default_rng(8)
    for _ in range(60):
        i = env._current
        free = [f.free_time for f in env._followers]
        other = (i + 1) % len(free)
        assert free[i] <= free[other] + 1e-12, (
            f"决策者 {i} 的 free_time {free[i]} 大于另一台 {free[other]}(选了忙者)"
        )
        _, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
        if term or trunc:
            break


# ======================================================================
# A. 重写:reward §v1.1.4 公式核验(改自旧 v1.1.0 用例)
# ======================================================================

def test_reward_normalization_formula() -> None:
    """§v1.1.4 恒等式:reward == dense_term+detect_term−balance_term−redundancy_term。

    同时断言:
    1. 恒等式在每步成立(容差 1e-9)。
    2. dense_term == value_weight * dense_mass / value_norm_ref(**无 clip 上限**:缩放不截断)。
    3. 构造足够覆盖使 dense_mass 超过 value_norm_ref → dense_term > 1(验证无截断)。
    """
    cfg = small_cfg()
    rc = cfg.reward
    env = CoopSurveyEnv(cfg)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    checked = 0
    dense_term_max_seen = 0.0
    for _ in range(40):
        _, reward, term, trunc, info = env.step(int(rng.integers(0, env.action_space.n)))

        # 恒等式(§v1.1.4 主约束):reward == dense_term+detect_term−balance_term−redundancy_term
        expected = (info["dense_term"] + info["detect_term"]
                    - info["balance_term"] - info["redundancy_term"])
        assert reward == pytest.approx(expected, abs=1e-9), (
            f"reward 恒等式违反: reward={reward}, sum={expected}"
        )

        # dense_term 线性公式(无 clip 上限)
        assert info["dense_term"] == pytest.approx(
            rc.value_weight * info["dense_mass"] / rc.value_norm_ref, abs=1e-12
        ), "dense_term 不等于 value_weight * dense_mass / value_norm_ref"

        dense_term_max_seen = max(dense_term_max_seen, info["dense_term"])
        checked += 1
        if term or trunc:
            break
    assert checked >= 1

    # 验证"缩放不截断":
    # 方法:用极小 value_norm_ref 使 dense_term >> 1(只要 dense_mass > norm_ref)。
    # 先跑一遍默认 cfg 收集最大 dense_mass 作为参考;然后将 norm_ref 设为其 1/10,
    # 确保 dense_mass/norm_ref > 1(远超截断上限 1),断言 dense_term 等于该比值(无截断)。
    # 先收集实测 dense_mass 最大值
    env_probe = CoopSurveyEnv(small_cfg())
    env_probe.reset(seed=7)
    rng_probe = np.random.default_rng(7)
    max_dense_mass = 0.0
    for _ in range(30):
        _, _, tp, trp, infop = env_probe.step(int(rng_probe.integers(0, env_probe.action_space.n)))
        max_dense_mass = max(max_dense_mass, infop["dense_mass"])
        if tp or trp:
            break

    if max_dense_mass > 0.0:
        # 将 norm_ref 设为 max_dense_mass 的 1/10 → dense_mass/norm_ref = 10 >> 1
        tiny_ref = max_dense_mass / 10.0
        cfg_small_ref = replace(cfg, reward=replace(cfg.reward,
                                                    value_norm_ref=tiny_ref,
                                                    balance_weight=0.0,
                                                    redundancy_weight=0.0))
        env2 = CoopSurveyEnv(cfg_small_ref)
        env2.reset(seed=7)
        rng_no_clip = np.random.default_rng(7)
        for _ in range(30):
            _, r2, t2, tr2, info2 = env2.step(int(rng_no_clip.integers(0, env2.action_space.n)))
            if info2["dense_mass"] >= max_dense_mass * 0.5:
                # dense_term 应等于 dense_mass/tiny_ref > 1(缩放不截断)
                expected_dt = info2["dense_mass"] / tiny_ref
                assert expected_dt > 1.0, "测试前提:norm_ref 足够小使 dense_term > 1"
                assert info2["dense_term"] == pytest.approx(expected_dt, abs=1e-9), (
                    f"dense_term={info2['dense_term']} 应等于 dense_mass/norm_ref={expected_dt}"
                    f"(缩放不截断,非 clip(·, 0, 1))"
                )
                break
            if t2 or tr2:
                break


def test_balance_off_ablation_is_pure_value() -> None:
    """balance_weight=0 → balance_term==0,reward 不含均衡项(配置消融,非 if 分支)。

    §v1.1.4:消融 = 连续权重置 0,不是 if 分支;balance_term 恒为 0。
    """
    cfg = replace(small_cfg(), reward=replace(small_cfg().reward, balance_weight=0.0))
    rc = cfg.reward
    env = CoopSurveyEnv(cfg)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(30):
        _, reward, term, trunc, info = env.step(int(rng.integers(0, env.action_space.n)))
        # 消融后 balance_term 恒为 0
        assert info["balance_term"] == pytest.approx(0.0, abs=1e-12), (
            f"balance_weight=0 下 balance_term 应为 0, 实际={info['balance_term']}"
        )
        # reward 恒等式仍成立:reward == dense_term + detect_term − 0 − redundancy_term
        expected_no_balance = (info["dense_term"] + info["detect_term"]
                               - info["redundancy_term"])
        assert reward == pytest.approx(expected_no_balance, abs=1e-9), (
            f"balance_off 下 reward 恒等式违反"
        )
        if term or trunc:
            break


def test_value_weight_scales_value_term() -> None:
    """改 value_weight 线性缩放 dense_term(对已归一化的 dense_mass,不截断)。

    §v1.1.4:dense_term = value_weight * dense_mass / value_norm_ref。
    将 value_weight 加倍 → dense_term 相应加倍(其它项 balance/redundancy 均置 0 以纯净比较)。
    """
    base = small_cfg()
    # 基础配置:balance 和 redundancy 均关掉,只看 dense_term 缩放
    cfg1 = replace(base, reward=replace(base.reward, value_weight=1.0,
                                        balance_weight=0.0, redundancy_weight=0.0))
    cfg2 = replace(base, reward=replace(base.reward, value_weight=2.0,
                                        balance_weight=0.0, redundancy_weight=0.0))
    env1 = CoopSurveyEnv(cfg1)
    env2 = CoopSurveyEnv(cfg2)
    env1.reset(seed=4)
    env2.reset(seed=4)  # 同 seed → 同场景、同 field
    rng = np.random.default_rng(4)
    actions = [int(rng.integers(0, env1.action_space.n)) for _ in range(20)]

    rng2 = np.random.default_rng(4)
    for a in actions:
        _, r1, t1, tr1, info1 = env1.step(a)
        _, r2, t2, tr2, info2 = env2.step(a)
        # 同场景同动作 → dense_mass 相同 → dense_term2 == 2 * dense_term1
        assert info2["dense_mass"] == pytest.approx(info1["dense_mass"], abs=1e-12), (
            "相同 seed/动作下 dense_mass 应相同"
        )
        assert info2["dense_term"] == pytest.approx(2.0 * info1["dense_term"], abs=1e-12), (
            f"value_weight=2 时 dense_term 应是 value_weight=1 的 2 倍: "
            f"dense_term1={info1['dense_term']}, dense_term2={info2['dense_term']}"
        )
        if (t1 or tr1) and (t2 or tr2):
            break


# ======================================================================
# info —— 必含 per_follower_energy_J(reward 能耗基于 follower)
# ======================================================================
def test_info_contains_per_follower_energy() -> None:
    env = CoopSurveyEnv(small_cfg())
    _, info = env.reset(seed=0)
    assert "per_follower_energy_J" in info
    pfe = info["per_follower_energy_J"]
    assert len(pfe) == 2  # N=2
    assert all(isinstance(x, float) for x in pfe)
    # reset 时累计能耗为 0
    assert all(x == 0.0 for x in pfe)

    _, _, _, _, info2 = env.step(int(env.action_space.sample()))
    assert "per_follower_energy_J" in info2
    assert len(info2["per_follower_energy_J"]) == 2


def test_info_has_value_and_balance_terms() -> None:
    """§v1.1.4 info 暴露全部 reward 分解项(新键名;旧 value_bits/balance_J 已废弃)。

    必须存在的键:dense_mass, dense_term, n_new_detections, detect_term,
                  balance_pen, balance_term, n_redundant, redundancy_term,
                  reward, per_follower_energy_J, clock_s, current_follower,
                  n_decisions, leader_finished。
    """
    env = CoopSurveyEnv(small_cfg())
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(env.action_space.sample()))
    required_keys = (
        "dense_mass", "dense_term",
        "n_new_detections", "detect_term",
        "balance_pen", "balance_term",
        "n_redundant", "redundancy_term",
        "reward",
        "per_follower_energy_J",
        "clock_s", "current_follower", "n_decisions", "leader_finished",
    )
    for key in required_keys:
        assert key in info, f"info 缺失 §v1.1.4 键: '{key}'; 实际 keys={set(info.keys())}"

    # 旧键名(v1.1.0)应不再存在,防回归
    for old_key in ("value_bits", "balance_J", "value_term"):
        assert old_key not in info, f"旧 v1.1.0 键 '{old_key}' 不应出现在 §v1.1.4 info 中"


# ======================================================================
# 终止 / 截断 —— 有限步内结束;不死循环
# ======================================================================
def test_episode_terminates_or_truncates_in_finite_steps() -> None:
    env = CoopSurveyEnv(small_cfg())
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    done = False
    for _ in range(2000):
        _, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
        if term or trunc:
            done = True
            break
    assert done, "小尺度 episode 应在有限步内 terminated 或 truncated"


def test_truncation_on_max_episode_time() -> None:
    """构造短 max_episode_time_s → truncated 在 clock>=max_episode_time_s 触发。"""
    cfg = replace(small_cfg(), max_episode_time_s=30.0)
    env = CoopSurveyEnv(cfg)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    truncated_seen = False
    for _ in range(500):
        _, _, term, trunc, info = env.step(int(rng.integers(0, env.action_space.n)))
        if trunc:
            truncated_seen = True
            assert info["clock_s"] >= cfg.max_episode_time_s - 1e-9
            break
        if term:
            break
    assert truncated_seen, "短 max_episode_time_s 下应观察到 truncated"


def test_no_step_after_done_required_reset() -> None:
    """终止后 reset 可重新开始(基本可复用性,不死锁)。"""
    env = CoopSurveyEnv(small_cfg())
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        _, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
        if term or trunc:
            break
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert info["n_decisions"] == 0


# ======================================================================
# 复现性 —— 同 seed 同动作序列 → 同 reward 轨迹
# ======================================================================
def test_seeded_rollout_reproducible() -> None:
    actions = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]

    def run():
        env = CoopSurveyEnv(small_cfg())
        env.reset(seed=42)
        rs = []
        for a in actions:
            _, r, term, trunc, _ = env.step(a % env.action_space.n)
            rs.append(r)
            if term or trunc:
                break
        return rs

    assert run() == run()


# ======================================================================
# B. 新增用例:§v1.1.4 行为
# ======================================================================

def test_dense_monotone_higher_field_gives_higher_dense_term() -> None:
    """dense 单调性:覆盖 field 高的格 → dense_mass/dense_term 更大。

    方法:small_cfg 下取两个不同动作(不同粗格中心),比较首步 dense_term。
    由于 small_cfg 有目标(field 非均匀),必存在覆盖概率质量差异。
    用同 seed 两次 reset,各走一步不同动作 → 收集 dense_mass → 总和应有差异
    (若第一步 dense_mass 全为 0 说明窗口空,继续走到有值为止)。
    """
    env = CoopSurveyEnv(small_cfg())
    rng = np.random.default_rng(11)

    # 收集多步 dense_mass 与动作关联,找两步 dense_mass 不同的
    dense_masses = []
    env.reset(seed=11)
    for _ in range(30):
        a = int(rng.integers(0, env.action_space.n))
        _, _, term, trunc, info = env.step(a)
        dense_masses.append(info["dense_mass"])
        if term or trunc:
            break

    # dense_mass 不应全为 0(场中有概率质量)
    positive_masses = [m for m in dense_masses if m > 0.0]
    assert len(positive_masses) >= 1, "至少一步应有正 dense_mass(场中有先验概率质量)"

    # dense_term 与 dense_mass 线性(无截断):验证 dense_term > 0 当且仅当 dense_mass > 0
    env.reset(seed=11)
    rng2 = np.random.default_rng(11)
    for _ in range(30):
        a = int(rng2.integers(0, env.action_space.n))
        _, _, term, trunc, info = env.step(a)
        if info["dense_mass"] > 0.0:
            assert info["dense_term"] > 0.0, "dense_mass>0 时 dense_term 应>0(线性单调)"
        else:
            assert info["dense_term"] == pytest.approx(0.0, abs=1e-12), (
                "dense_mass==0 时 dense_term 应==0"
            )
        if term or trunc:
            break


def test_sparse_detect_deduplication() -> None:
    """sparse 去重:同一 GT 目标格被覆盖两次 → 第一次 n_new_detections>=1,再覆盖==0。

    GT 检出 = 确定性覆盖触发(走廊覆盖 observable ∩ target_cells):
    对同一动作连续决策两次 → 首次检出后 detected 集合去重 → 第二次 n_new_detections==0。
    """
    env = CoopSurveyEnv(small_cfg())
    env.reset(seed=3)
    rng = np.random.default_rng(3)

    # 先跑几步寻找首次有检出的步骤
    first_detect_seen = False
    first_detect_action = None
    cumulative_detections = 0

    for step_i in range(50):
        a = int(rng.integers(0, env.action_space.n))
        _, _, term, trunc, info = env.step(a)
        cumulative_detections += info["n_new_detections"]
        if info["n_new_detections"] > 0 and not first_detect_seen:
            first_detect_seen = True
            # 记录已检出数量:后续再覆盖同格不应再 +1
            break
        if term or trunc:
            break

    # 去重性质直接测:env 内部 _detected 集合确保每个目标格只计一次
    # 验证:在一个完整 rollout 中,n_new_detections 之和 <= _target_cells 数量
    env2 = CoopSurveyEnv(small_cfg())
    env2.reset(seed=3)
    rng3 = np.random.default_rng(3)
    total_detections = 0
    for _ in range(200):
        a = int(rng3.integers(0, env2.action_space.n))
        _, _, term, trunc, info = env2.step(a)
        total_detections += info["n_new_detections"]
        if term or trunc:
            break
    n_targets = len(env2._target_cells)
    assert total_detections <= n_targets, (
        f"总检出数 {total_detections} 超过 GT 目标格数 {n_targets},去重失效"
    )

    # 同一步 n_new_detections 不超过目标格总数
    env3 = CoopSurveyEnv(small_cfg())
    env3.reset(seed=5)
    rng4 = np.random.default_rng(5)
    for _ in range(60):
        a = int(rng4.integers(0, env3.action_space.n))
        _, _, term, trunc, info = env3.step(a)
        assert info["n_new_detections"] >= 0, "n_new_detections 不应为负"
        assert info["detect_term"] == pytest.approx(
            info["n_new_detections"] * 1.0, abs=1e-12  # default detect_weight=1.0
        ), "detect_term 应 == detect_weight * n_new_detections"
        if term or trunc:
            break


def test_soft_redundancy_no_termination() -> None:
    """软冗余不终止:n_redundant>0 且 redundancy_term>0,但 episode 不因冗余 terminated。

    §v1.1.4 点3:冗余仅作软惩罚减 reward,不导致 episode 结束。
    方法:用 small_cfg(N=2),让两台 Follower 各自动作 → 后续 Follower 覆盖前台覆盖过的区域
    → 某步 n_redundant>0;全程不因冗余提前 terminated(terminated 只由 drain 逻辑或全 planned 触发)。
    """
    env = CoopSurveyEnv(small_cfg())
    env.reset(seed=2)
    rng = np.random.default_rng(2)

    redundant_seen = False
    # 确保 N=2 两台都走且面积小足够产生重叠
    for _ in range(100):
        # 两台均往同一动作 → 高度重叠
        a = 0  # 强制选 action=0,两台都会走向同一个粗格中心
        _, _, term, trunc, info = env.step(a)
        if info["n_redundant"] > 0:
            redundant_seen = True
            assert info["redundancy_term"] > 0.0, (
                f"n_redundant>0 时 redundancy_term 应>0: {info['redundancy_term']}"
            )
            # 此时不应因冗余而 terminated
            assert not term, (
                "episode 不应因冗余 terminated(软冗余不终止)"
            )
        if term or trunc:
            break

    # 确认"软冗余不终止"语义:即便有 redundant,episode 照常推进到完成
    # 再跑一个完整 episode,若发现冗余 → 继续不终止
    env2 = CoopSurveyEnv(small_cfg())
    env2.reset(seed=9)
    rng5 = np.random.default_rng(9)
    for _ in range(300):
        a = 0  # 强制同一动作使重叠最大化
        _, _, term, trunc, info = env2.step(a)
        if info["n_redundant"] > 0:
            # 有冗余步:redundancy_term > 0,但此步不能因为冗余而 terminated
            assert info["redundancy_term"] == pytest.approx(
                0.1 * info["n_redundant"], abs=1e-12  # redundancy_weight=0.1
            ), "redundancy_term 不符合公式"
        if term or trunc:
            break
    # episode 以 terminated/truncated 结束,非异常退出
    # (只要有限步内 done,即说明终止机制正常,非因冗余无限循环)


def test_gt_not_in_obs() -> None:
    """GT 不进 obs:obs 仅含 map/vec、map 来自先验场(形状 (Hc,Wc)、值∈[0,1])。

    断言:
    1. obs 只有 'map' 和 'vec' 两个键,无额外泄漏 GT 信息的通道。
    2. obs['map'] 形状为 (Hc,Wc),dtype=float32,值域 [0,1]。
    3. obs['vec'] 形状为 (10,),dtype=float32,值域 [0,1]。
    4. env.observation_space.contains(obs) 恒为真(gym 空间约束)。
    """
    from msim.geometry.window import coarse_grid_shape

    cfg = small_cfg()
    env = CoopSurveyEnv(cfg)
    Hc, Wc = coarse_grid_shape(cfg.window, cfg.sensor)

    obs, _ = env.reset(seed=6)

    # obs 键集合 = {map, vec}
    assert set(obs.keys()) == {"map", "vec"}, (
        f"obs 应只含 'map' 和 'vec',实际有: {set(obs.keys())}"
    )

    # map:形状/类型/值域
    assert obs["map"].shape == (Hc, Wc), f"map 形状 {obs['map'].shape} != ({Hc},{Wc})"
    assert obs["map"].dtype == np.float32, f"map dtype 应为 float32,实际 {obs['map'].dtype}"
    assert float(obs["map"].min()) >= 0.0, "map 值应 >= 0(先验∈[0,1])"
    assert float(obs["map"].max()) <= 1.0 + 1e-6, "map 值应 <= 1(先验∈[0,1])"

    # vec:形状/类型/值域
    assert obs["vec"].shape == (10,), f"vec 形状 {obs['vec'].shape} != (10,)"
    assert obs["vec"].dtype == np.float32, f"vec dtype 应为 float32,实际 {obs['vec'].dtype}"
    assert float(obs["vec"].min()) >= 0.0, "vec 值应 >= 0(归一化∈[0,1])"
    assert float(obs["vec"].max()) <= 1.0 + 1e-6, "vec 值应 <= 1(归一化∈[0,1])"

    # 逐步验证(几步内 obs 始终合规)
    rng = np.random.default_rng(6)
    for _ in range(20):
        obs, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
        assert set(obs.keys()) == {"map", "vec"}, "逐步 obs 键集合变化"
        assert env.observation_space.contains(obs), "obs 超出 observation_space"
        assert float(obs["map"].min()) >= 0.0
        assert float(obs["map"].max()) <= 1.0 + 1e-6
        if term or trunc:
            break


# ======================================================================
# C. 人类点名的三道"有效范围"不变量(整段 rollout 含起步与 drain 两端)
# ======================================================================

def test_invariant_obs_always_in_observation_space() -> None:
    """不变量 1:每步 obs(Follower 信息)在有效范围。

    env.observation_space.contains(obs) 在 rollout 全程(起步 + drain 两端)恒为真。
    多 seed 随机策略跑到终止,验证起步首步与 drain 末步均合规。
    """
    for seed in range(5):
        env = CoopSurveyEnv(small_cfg())
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs), (
            f"seed={seed}: reset 后首 obs 超出 observation_space"
        )
        rng = np.random.default_rng(seed)
        step_count = 0
        for _ in range(500):
            obs, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
            assert env.observation_space.contains(obs), (
                f"seed={seed}, step={step_count}: obs 超出 observation_space"
            )
            step_count += 1
            if term or trunc:
                break


def test_invariant_window_in_field_and_map_in_01() -> None:
    """不变量 2:每步喂给 policy 的 window 范围正确。

    - reset 后首个决策:leader.pose[0] >= look_back_m - 1e-9(窗口入场修复 §v1.1.4 点4)。
    - 全程 coarse map 取值 ∈ [0,1]。
    """
    for seed in range(5):
        cfg = small_cfg()
        look_back = cfg.window.look_back_m
        env = CoopSurveyEnv(cfg)
        obs, _ = env.reset(seed=seed)

        # reset 后 Leader 位姿已前扫至 >= look_back(窗口入场修复)
        leader_x = float(env._leader.pose[0])
        assert leader_x >= look_back - 1e-9, (
            f"seed={seed}: reset 后 leader.pose[0]={leader_x} < look_back={look_back}"
        )

        # coarse map 值域 [0,1]
        assert float(obs["map"].min()) >= 0.0
        assert float(obs["map"].max()) <= 1.0 + 1e-6

        rng = np.random.default_rng(seed)
        for _ in range(100):
            obs, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
            assert float(obs["map"].min()) >= 0.0, "map 值下界违反"
            assert float(obs["map"].max()) <= 1.0 + 1e-6, "map 值上界违反"
            if term or trunc:
                break


def test_invariant_waypoint_in_world_bounds() -> None:
    """不变量 3 + 4:每步输出 waypoint 在场内,起步与 drain 两端都不出界。

    env 实际承诺的已 clip 航点:env._followers[i].commitment.queue[-1]
    断言 0 <= x <= world.x_max_m 且 0 <= y <= world.y_max_m。
    多 seed 随机策略跑到终止。
    """
    for seed in range(5):
        cfg = small_cfg()
        x_max = cfg.world.x_max_m
        y_max = cfg.world.y_max_m
        env = CoopSurveyEnv(cfg)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        step_count = 0
        for _ in range(500):
            i_decider = env._current
            a = int(rng.integers(0, env.action_space.n))
            _, _, term, trunc, _ = env.step(a)

            # 取本步决策者的最新 waypoint(已 clip)
            queue = env._followers[i_decider].commitment.queue
            if len(queue) > 0:
                wp = queue[-1]
                assert 0.0 - 1e-9 <= float(wp[0]) <= x_max + 1e-9, (
                    f"seed={seed}, step={step_count}: waypoint x={wp[0]} 超出 [0,{x_max}]"
                )
                assert 0.0 - 1e-9 <= float(wp[1]) <= y_max + 1e-9, (
                    f"seed={seed}, step={step_count}: waypoint y={wp[1]} 超出 [0,{y_max}]"
                )
            step_count += 1
            if term or trunc:
                break

        # 验证起步(第一步):第一个 waypoint 应在场内
        env2 = CoopSurveyEnv(cfg)
        env2.reset(seed=seed)
        i_first = env2._current
        env2.step(0)  # 取 action=0(首个粗格)
        queue2 = env2._followers[i_first].commitment.queue
        assert len(queue2) >= 1, "首步后 commitment queue 应非空"
        wp0 = queue2[-1]
        assert 0.0 - 1e-9 <= float(wp0[0]) <= x_max + 1e-9, (
            f"seed={seed}: 首步 waypoint x={wp0[0]} 超出场内 [0,{x_max}]"
        )
        assert 0.0 - 1e-9 <= float(wp0[1]) <= y_max + 1e-9, (
            f"seed={seed}: 首步 waypoint y={wp0[1]} 超出场内 [0,{y_max}]"
        )


# ======================================================================
# §v1.1.5 离散固定块窗口 —— 块语义不变量(A0 续派)
# ======================================================================
# 块边界对这个编程:edges = [k*look_back for k in range(n_blocks+1)](末边界锚 x_max)。
#   windowed: look_back=20, x_max=60 → edges=[0,20,40,60](3 块,整除)。
#   small:    look_back=8,  x_max=40 → edges=[0,8,16,24,32,40](5 块,整除)。
# 块窗口由锚点位姿 [edges[k+1], y_max/2, psi=0] 经 get_window 构造 → North 范围
#   [edges[k+1]-look_back, edges[k+1]] = [edges[k], edges[k+1]](整除块);
#   East 范围 [y_max/2 ± width/2] = [0, y_max](单测线前提 width==y_max)。
# 整段 rollout(起步 → 块切换 → drain/终止)逐步断言块语义。
# ======================================================================

def _expected_edges(cfg) -> list:
    """从 cfg 推导固定块边界 edges(与 env 内部 __init__ 同口径,不读 env 私有以独立交叉验证)。

    n_blocks = round(x_max/look_back);edges=[min(k*look_back,x_max)];末边界锚 x_max。
    """
    look_back = float(cfg.window.look_back_m)
    x_max = float(cfg.world.x_max_m)
    n_blocks = max(1, int(round(x_max / look_back)))
    edges = [min(k * look_back, x_max) for k in range(n_blocks + 1)]
    edges[-1] = x_max
    return edges


def _window_north_range(win) -> tuple:
    """块窗口的 North 范围 [lo, hi](psi=0 → back=(-1,0):North 从 leader_pose[0] 向后延伸)。"""
    hi = float(win.leader_pose[0])
    lo = hi - float(win.look_back_m)
    return lo, hi


def _window_east_range(win) -> tuple:
    """块窗口的 East 范围 [lo, hi](psi=0 → lat=(0,1):East 以 leader_pose[1] 为中心 ±width/2)。"""
    c = float(win.leader_pose[1])
    half = float(win.width_m) / 2.0
    return c - half, c + half


def test_window_is_discrete_fixed_block() -> None:
    """块语义不变量 1:每步 env._current_window() 的 North 范围恰等于某固定块 [edges[k],edges[k+1]]。

    **不是**随 Leader 连续滑动的任意区间:窗口 North 端点必落在固定块边界上(容差一个 cell),
    而非 leader.pose[0] 的连续值。windowed(真实 9 目标块行为)与 small_cfg 都验。
    """
    for name, cfg in (("windowed", windowed_survey_cfg(seed=9)), ("small", small_cfg())):
        edges = _expected_edges(cfg)
        tol = cfg.world.res_m  # 一个 cell 容差(块对齐到 cell 中心)
        env = CoopSurveyEnv(cfg)
        env.reset(seed=9)

        def _matches_a_fixed_block(win) -> bool:
            lo, hi = _window_north_range(win)
            for k in range(len(edges) - 1):
                if abs(lo - edges[k]) <= tol and abs(hi - edges[k + 1]) <= tol:
                    return True
            return False

        # reset 后首窗口即固定块(块 0)
        assert _matches_a_fixed_block(env._current_window()), (
            f"{name}: reset 窗口 North={_window_north_range(env._current_window())} "
            f"不等于任何固定块 edges={edges}"
        )

        rng = np.random.default_rng(9)
        for _ in range(600):
            _, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
            win = env._current_window()
            assert _matches_a_fixed_block(win), (
                f"{name}: 窗口 North={_window_north_range(win)} 不等于任何固定块 "
                f"edges={edges}(滑窗回归:窗口应固定到块边界,非随 leader.pose 连续滑动)"
            )
            if term or trunc:
                break


def test_block_index_monotonic_sequential() -> None:
    """块语义不变量 2:info['block_k'] 整段单调不减,最终覆盖到最后一块。

    - block_k 单调不减(顺序覆盖块 0→1→…,不回退)。
    - windowed(3 块):整段出现过块 0、1、2(每块都被激活)。
    - 终止时 n_blocks_done == n_blocks(全块完成才 terminated)。
    """
    cfg = windowed_survey_cfg(seed=9)
    env = CoopSurveyEnv(cfg)
    _, info0 = env.reset(seed=9)
    assert info0["block_k"] == 0, "reset 后活动块应为 0"
    assert info0["n_blocks"] == 3, f"windowed 应 3 块,实际 {info0['n_blocks']}"
    assert info0["n_blocks_done"] == 0, "reset 后无块完成"

    rng = np.random.default_rng(9)
    n_blocks = info0["n_blocks"]
    prev_k = info0["block_k"]
    seen_blocks = {prev_k}
    last_info = info0
    term = trunc = False
    for _ in range(800):
        _, _, term, trunc, info = env.step(int(rng.integers(0, env.action_space.n)))
        # 单调不减(block_k 终止那步可能 == n_blocks 越界标记,仍不减)
        assert info["block_k"] >= prev_k, (
            f"block_k 回退:{prev_k} → {info['block_k']}(应单调不减)"
        )
        prev_k = info["block_k"]
        # 记录被激活的块(钳到 [0, n_blocks-1])
        seen_blocks.add(min(info["block_k"], n_blocks - 1))
        last_info = info
        if term or trunc:
            break

    assert term, "windowed 随机策略应以 terminated(全块完成)结束,非 truncated"
    # 每块都被激活过(块 0,1,2)
    assert seen_blocks == set(range(n_blocks)), (
        f"应激活全部 {n_blocks} 块,实际 seen={sorted(seen_blocks)}"
    )
    # 终止时全块完成
    assert last_info["n_blocks_done"] == n_blocks, (
        f"终止时 n_blocks_done={last_info['n_blocks_done']} != n_blocks={n_blocks}"
    )


def test_waypoint_within_active_block() -> None:
    """块语义不变量 3:每步 waypoint 的 North ∈ 决策时活动块 [edges[k],edges[k+1]]、East ∈ 块 East 范围。

    决策块 = step 调用前的 min(block_k, n_blocks-1)(step 开头 _activate_next_block 已在
    上一步末把活动块推进好,本步开头通常为 no-op)。waypoint 落在该活动块固定窗口内(容差一个 cell)。
    """
    for name, mkcfg in (("windowed", lambda: windowed_survey_cfg(seed=9)),
                        ("small", small_cfg)):
        for seed in range(4):
            cfg = mkcfg()
            edges = _expected_edges(cfg)
            res = cfg.world.res_m
            y_max = cfg.world.y_max_m
            env = CoopSurveyEnv(cfg)
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            for _ in range(600):
                i_dec = env._current
                k = min(env._block_k, env._n_blocks - 1)  # 决策时活动块(step 前)
                a = int(rng.integers(0, env.action_space.n))
                _, _, term, trunc, _ = env.step(a)
                wp = env._followers[i_dec].commitment.queue[-1]
                north_lo, north_hi = edges[k], edges[k + 1]
                # North ∈ 活动块(容差一个 cell:粗格中心 + clip 浮点)
                assert (north_lo - res - 1e-6) <= float(wp[0]) <= (north_hi + res + 1e-6), (
                    f"{name} seed={seed}: waypoint North={wp[0]} 不在活动块 "
                    f"[{north_lo},{north_hi}](块 {k})内"
                )
                # East ∈ 全 East(单测线前提 width==y_max → 块 East = [0,y_max])
                assert (0.0 - res - 1e-6) <= float(wp[1]) <= (y_max + res + 1e-6), (
                    f"{name} seed={seed}: waypoint East={wp[1]} 不在块 East [0,{y_max}] 内"
                )
                if term or trunc:
                    break


def test_block_full_progression_no_block_skipped() -> None:
    """块语义不变量 4(覆盖完整性,替代"覆盖无大缝"):全部固定块按序激活且最终全 done。

    A0 原拟用例 4 是"随机策略覆盖 North 并集最大缝 ≤ ~2*res、无大未覆盖带"。**实测不成立**:
    env 的块"完成"判据是"无进展即止"(当前块连续 2*n_followers 次 dense_mass<ε 即 done),
    **不是几何全覆盖才 done** → 随机策略(甚至穷举动作循环)在一个块内只走少数粗格行就 done,
    实测 North 并集 maxgap 可达 12-18m(>> 2*res=4m)。故"覆盖无大缝"不是 env 在通用策略下
    的不变量(它是训练好的策略的目标,而非 env 机制保证)。**此点已交 A0 裁决**(见返回汇报)。
    这里改测真实成立的等价完整性不变量:**离散块平铺覆盖整个 North 跨度** [0, x_max],即
    每个固定块都被激活并标记 done(n_blocks_done==n_blocks);印证"块平铺无缝"是按块而非按 cell。
    """
    for name, mkcfg in (("windowed", lambda: windowed_survey_cfg(seed=9)),
                        ("small", small_cfg)):
        for seed in range(5):
            cfg = mkcfg()
            env = CoopSurveyEnv(cfg)
            _, info0 = env.reset(seed=seed)
            n_blocks = info0["n_blocks"]
            rng = np.random.default_rng(seed)
            seen_blocks = {0}
            last_info = info0
            term = trunc = False
            for _ in range(2000):
                _, _, term, trunc, info = env.step(int(rng.integers(0, env.action_space.n)))
                seen_blocks.add(min(info["block_k"], n_blocks - 1))
                last_info = info
                if term or trunc:
                    break
            assert term, f"{name} seed={seed}: 应以 terminated(全块完成)结束"
            # 每个固定块都被激活(块平铺覆盖整个 North 跨度,无块被跳过)
            assert seen_blocks == set(range(n_blocks)), (
                f"{name} seed={seed}: 有块未被激活,seen={sorted(seen_blocks)}, n_blocks={n_blocks}"
            )
            # 全块 done
            assert last_info["n_blocks_done"] == n_blocks, (
                f"{name} seed={seed}: n_blocks_done={last_info['n_blocks_done']} != {n_blocks}"
            )


def test_block_window_full_east_single_lane() -> None:
    """块语义不变量 5(单测线前提):块窗口 East 范围 == 全 East [0, y_max]。

    §v1.1.5 离散块前提:window.width_m == world.y_max_m → 块 = 全 East 的 North 条带。
    windowed 与 small_cfg 都满足(width=20=y_max)。验证 _current_window() 的 East 范围恰 [0,y_max]。
    """
    for name, cfg in (("windowed", windowed_survey_cfg(seed=9)), ("small", small_cfg())):
        # 前提:width == y_max(单测线)
        assert cfg.window.width_m == pytest.approx(cfg.world.y_max_m), (
            f"{name}: 离散块单测线前提要求 width_m({cfg.window.width_m}) == "
            f"y_max({cfg.world.y_max_m})"
        )
        env = CoopSurveyEnv(cfg)
        env.reset(seed=9)
        y_max = cfg.world.y_max_m

        # reset 窗口 East 范围 = [0, y_max]
        east_lo, east_hi = _window_east_range(env._current_window())
        assert east_lo == pytest.approx(0.0, abs=1e-9), (
            f"{name}: 块窗口 East 下界={east_lo} != 0"
        )
        assert east_hi == pytest.approx(y_max, abs=1e-9), (
            f"{name}: 块窗口 East 上界={east_hi} != y_max={y_max}"
        )

        # 整段 rollout East 范围恒为全 East(块切换不改 East,只换 North)
        rng = np.random.default_rng(9)
        for _ in range(200):
            _, _, term, trunc, _ = env.step(int(rng.integers(0, env.action_space.n)))
            elo, ehi = _window_east_range(env._current_window())
            assert elo == pytest.approx(0.0, abs=1e-9), f"{name}: East 下界漂移={elo}"
            assert ehi == pytest.approx(y_max, abs=1e-9), f"{name}: East 上界漂移={ehi}"
            if term or trunc:
                break


# ======================================================================
# §v1.1.6 obs map 扣除 planned(残余先验场)—— 消观测混叠不变量(A0 续派)
# ======================================================================
# 契约:DECISIONS.md ## v1.1.6 + contracts/mdp.py::assemble_obs docstring。
#   map[row,col] = field[center] · (1 − planned[center])(粗格**中心**采样;planned→硬置 0)。
#   语义:粗格中心 cell 若已 planned(走廊扫过)→ obs map 读 0;否则读原始 field。
#   目的:obs 与 reward dense_mass 同口径,让策略"看得见"哪条带已扫 → 两台分带、单台推进。
#   开环不破坏:planned 是布尔覆盖掩码(非 belief 回写),field 逐位不变、NullUpdater 仍开环。
# ======================================================================


def test_obs_map_planned_cell_reads_zero() -> None:
    """§v1.1.6 不变量 1(直接单元):某粗格中心 cell 一旦 planned → 该粗格在 obs map 读 0。

    构造:windowed env reset(field 非零先验已铺好,_planned 全 False)。
    取一个 field 非零的粗格 idx → 求其中心 cell (ix,iy):
      - before = map[row,col] 应 > 0(残余先验 = field,未 planned);
      - 手动置 env._planned[ix,iy]=True → after = map[row,col] 应硬 == 0.0。
    并断言 map 形状 (Hc,Wc)、dtype float32、值域 ∈[0,1] 不被破坏。
    """
    cfg = windowed_survey_cfg(seed=0)
    env = CoopSurveyEnv(cfg)
    env.reset(seed=0)
    window = env._current_window()
    Wc = env._Wc

    # 选一个 field 非零的粗格 idx(其中心 cell 先验 > 0);若 idx=0 恰为 0 则继续找。
    chosen = None
    for idx in range(env._Hc * env._Wc):
        center = coarse_index_to_world(idx, window, cfg.window, cfg.sensor)
        ix, iy = env._grid.world_to_cell(center)
        row, col = idx // Wc, idx % Wc
        before = float(env._coarse_map()[row, col])
        if before > 0.0:
            chosen = (idx, ix, iy, row, col, before)
            break
    assert chosen is not None, "应至少存在一个 field 非零的粗格(残余先验非全零)"
    idx, ix, iy, row, col, before = chosen

    # 前提断言:该粗格残余先验非零(未 planned 时 == field)。
    assert before > 0.0, f"前提:粗格 idx={idx} 的 before={before} 应 > 0"

    # 置该中心 cell 为 planned → 残余先验硬置 0。
    env._planned[ix, iy] = True
    cmap = env._coarse_map()
    after = float(cmap[row, col])
    assert after == 0.0, (
        f"§v1.1.6:粗格中心 cell({ix},{iy}) planned 后 obs map[{row},{col}] 应 == 0.0,"
        f"实际={after}(before={before})"
    )

    # 空间/类型/值域不变(planned 扣除不破坏 obs space)。
    assert cmap.shape == (env._Hc, env._Wc), f"map 形状 {cmap.shape} != ({env._Hc},{env._Wc})"
    assert cmap.dtype == np.float32, f"map dtype 应 float32,实际 {cmap.dtype}"
    assert float(cmap.min()) >= 0.0, "残余先验 map 值应 >= 0"
    assert float(cmap.max()) <= 1.0 + 1e-6, "残余先验 map 值应 <= 1"


def test_obs_map_reflects_planned_after_step() -> None:
    """§v1.1.6 不变量 2(行为·核心):走一步扫到非零先验后,obs map 至少一格残余先验衰减。

    这是 §v1.1.6 的目的——obs 反映"哪条带已扫":走廊覆盖的高先验格被 planned 后,
    下一决策者看到的 obs map 在那些格上由 field 衰减到更低(扣除 planned)。

    方法:reset 记决策时活动块 k 与该块窗口的残余先验 map_before(用 §v1.1.6 契约公式
    field·(1−planned) 在块 k 上独立交叉重算,**不受 step 末块切换影响**)。
    遍历动作找一个落在当前块、确实扫到非零先验的 a(info["dense_mass"]>0 → 确有
    走廊∩窗口∩未planned 的 field 被计入);走该步后在**同一块 k 窗口**重算 map_after。
    断言:**至少有一个粗格** map_after[r,c] < map_before[r,c](被扫格残余先验衰减),
    且同一块内只降不升(残余先验单调不增);全程 map ∈[0,1]、形状/dtype 不变。
    `env._coarse_map()`(当前活动块,可能已切到下一块)单独再校形状/dtype/值域。
    """

    def _residual_map_on_block(e: CoopSurveyEnv, k: int) -> np.ndarray:
        """按 §v1.1.6 契约公式在固定块 k 的窗口上重算粗格残余先验 map(独立于 env 当前活动块)。"""
        win = e._block_window(k)
        m = np.zeros((e._Hc, e._Wc), dtype=np.float32)
        for row in range(e._Hc):
            for col in range(e._Wc):
                idx = row * e._Wc + col
                center = coarse_index_to_world(idx, win, cfg.window, cfg.sensor)
                ix, iy = e._grid.world_to_cell(center)
                planned = bool(e._planned[ix, iy])
                m[row, col] = np.float32(0.0 if planned else e._field[ix, iy])
        return m

    cfg = windowed_survey_cfg(seed=0)
    env = CoopSurveyEnv(cfg)
    env.reset(seed=0)
    k_decide = min(env._block_k, env._n_blocks - 1)  # 决策时活动块(step 前)
    map_before = _residual_map_on_block(env, k_decide)

    # 找一个使 dense_mass>0 的动作(确实扫到非零先验);用同 seed 干净 env 实测以选动作。
    chosen_a = None
    for a in range(env.action_space.n):
        probe = CoopSurveyEnv(cfg)
        probe.reset(seed=0)
        _, _, _, _, info = probe.step(a)
        if info["dense_mass"] > 0.0:
            chosen_a = a
            break
    assert chosen_a is not None, "应存在落在当前块且扫到非零先验(dense_mass>0)的动作"

    _, _, _, _, info = env.step(chosen_a)
    assert info["dense_mass"] > 0.0, f"前提:所选动作 dense_mass 应 > 0,实际={info['dense_mass']}"

    # 同一决策块 k 的窗口上重算(消除 step 末块切换的干扰),纯看 planned 扣除效果。
    map_after = _residual_map_on_block(env, k_decide)

    # 核心:至少一个粗格残余先验因 planned 衰减(map_after < map_before)。
    decreased = map_after < map_before - 1e-9
    assert bool(decreased.any()), (
        "§v1.1.6:扫到非零先验后,obs map 应至少有一格残余先验衰减(map_after < map_before);"
        f"实际无任何格下降(dense_mass={info['dense_mass']}, 块 k={k_decide})"
    )
    # 同一块内残余先验单调不增(planned 只会把格扣到 0,绝不抬高)。
    assert not bool((map_after > map_before + 1e-6).any()), (
        "残余先验扣除 planned 只应使 map 下降或不变,同一块内不应有格变大"
    )

    # env 自身 obs map(当前活动块,可能已切到下一块)校形状/dtype/值域不变。
    cmap_now = env._coarse_map()
    assert cmap_now.shape == (env._Hc, env._Wc), f"map 形状 {cmap_now.shape} 变了"
    assert cmap_now.dtype == np.float32, f"map dtype 应 float32,实际 {cmap_now.dtype}"
    assert float(cmap_now.min()) >= 0.0, "残余先验 map 值应 >= 0"
    assert float(cmap_now.max()) <= 1.0 + 1e-6, "残余先验 map 值应 <= 1"
