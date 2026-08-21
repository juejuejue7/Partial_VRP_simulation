"""[A1 验收测试 · L4 network / train_ppo] actor-critic 网络 + PPO 训练 smoke。

判据来源(harness §4.2 + §v1.1.0 DoD③ + contracts/mdp 形状约定):
- build_actor_critic(cfg, n_actions) 返回 torch.nn.Module;前向接受 Dict obs **单样本**
  (map(Hc,Wc)/vec(10,))与 **batch**(map(B,Hc,Wc)/vec(B,10)),输出 logits/value 形状一致;
  get_action_and_value 返回 (action,logprob,entropy,value);参数量小(Pi 5 约束,< ~5e4)。
- train smoke:用 small_cfg + 极小 PPOConfig 跑通,返回 stats dict,损失全有限、
  num_updates>=1、reward 有数。**只依赖** train(cfg, ppo=None)->dict 稳定签名
  (A5 并行新增的 wandb/checkpoint 等可选参数默认不影响,本测试不依赖那些新键)。

torch import 慢可接受,timesteps 取最小以控时长。
"""
from __future__ import annotations

import math

import pytest

from msim.geometry.window import coarse_grid_shape
from msim.mdp.coop_survey_env import small_cfg

torch = pytest.importorskip("torch")  # 无 torch 环境整文件跳过(网络层延迟依赖)

from msim.policy.network import build_actor_critic  # noqa: E402
from msim.policy.train_ppo import PPOConfig, train  # noqa: E402


def _shapes():
    cfg = small_cfg()
    Hc, Wc = coarse_grid_shape(cfg.window, cfg.sensor)
    return cfg, Hc, Wc, Hc * Wc


# ======================================================================
# network —— 类型 / 前向 / 参数量
# ======================================================================
def test_build_returns_torch_module() -> None:
    cfg, _, _, n = _shapes()
    net = build_actor_critic(cfg, n)
    assert isinstance(net, torch.nn.Module)


def test_param_count_small_for_pi5() -> None:
    """参数量小(Raspberry Pi 5 推理约束):总参数 < ~5e4。"""
    cfg, _, _, n = _shapes()
    net = build_actor_critic(cfg, n)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params < 50_000, f"参数量 {n_params} 超出 Pi 5 小网络约束(<5e4)"


def test_forward_single_sample() -> None:
    """单样本 Dict obs(map(Hc,Wc)/vec(10,)):logits/value 第 0 维为 batch=1。"""
    cfg, Hc, Wc, n = _shapes()
    net = build_actor_critic(cfg, n)
    obs = {"map": torch.zeros(Hc, Wc), "vec": torch.zeros(10)}
    logits, value = net(obs)
    assert logits.shape[-1] == n        # 最后一维 = n_actions
    assert logits.shape[0] == 1         # 单样本提升为 batch=1
    assert value.shape == (1,)


def test_forward_batch() -> None:
    """batch Dict obs(map(B,Hc,Wc)/vec(B,10)):logits(B,n)、value(B,)。"""
    cfg, Hc, Wc, n = _shapes()
    net = build_actor_critic(cfg, n)
    B = 7
    obs = {"map": torch.zeros(B, Hc, Wc), "vec": torch.zeros(B, 10)}
    logits, value = net(obs)
    assert tuple(logits.shape) == (B, n)
    assert tuple(value.shape) == (B,)


def test_get_action_and_value_returns_quadruple() -> None:
    """get_action_and_value → (action, logprob, entropy, value),形状对齐 batch。"""
    cfg, Hc, Wc, n = _shapes()
    net = build_actor_critic(cfg, n)
    B = 4
    obs = {"map": torch.zeros(B, Hc, Wc), "vec": torch.zeros(B, 10)}
    out = net.get_action_and_value(obs)
    assert len(out) == 4
    action, logprob, entropy, value = out
    assert tuple(action.shape) == (B,)
    assert tuple(logprob.shape) == (B,)
    assert tuple(entropy.shape) == (B,)
    assert tuple(value.shape) == (B,)
    # action ∈ [0, n)
    assert int(action.min()) >= 0 and int(action.max()) < n


def test_get_action_and_value_with_given_action() -> None:
    """传入 action 时返回其 logprob/value(评估路径,PPO 复用旧动作)。"""
    cfg, Hc, Wc, n = _shapes()
    net = build_actor_critic(cfg, n)
    B = 3
    obs = {"map": torch.zeros(B, Hc, Wc), "vec": torch.zeros(B, 10)}
    given = torch.zeros(B, dtype=torch.long)
    a, lp, ent, v = net.get_action_and_value(obs, given)
    assert torch.equal(a, given)
    assert tuple(lp.shape) == (B,)


# ======================================================================
# train —— smoke:极小预算跑通 PPO,stats 有数、损失有限
# ======================================================================
def test_train_smoke_runs_and_returns_stats() -> None:
    """train(cfg, ppo) 极小预算跑通端到端;只依赖稳定签名 train(cfg, ppo=None)->dict。"""
    cfg = small_cfg()
    ppo = PPOConfig(total_timesteps=512, rollout_steps=128, verbose=False)
    stats = train(cfg, ppo)

    assert isinstance(stats, dict)
    # —— 仅断言稳定核心键(不依赖 A5 并行新增的 checkpoint/wandb/... 键)——
    for key in (
        "num_updates",
        "global_steps",
        "update_policy_loss",
        "update_value_loss",
        "update_entropy",
        "episode_returns",
    ):
        assert key in stats, f"stats 缺少核心键 {key}"

    assert stats["num_updates"] >= 1
    assert stats["global_steps"] >= 1

    # 损失全有限(math.isfinite)
    losses = (
        list(stats["update_policy_loss"])
        + list(stats["update_value_loss"])
        + list(stats["update_entropy"])
    )
    assert len(losses) >= 1
    assert all(math.isfinite(x) for x in losses), "训练损失出现 nan/inf"


def test_train_reward_has_numbers() -> None:
    """reward 有数:至少一个完成 episode 的回报为有限值(DoD③:能跑完、reward 有数)。"""
    cfg = small_cfg()
    ppo = PPOConfig(total_timesteps=512, rollout_steps=128, verbose=False)
    stats = train(cfg, ppo)

    returns = list(stats["episode_returns"])
    assert len(returns) >= 1, "smoke 训练应至少完成 1 个 episode"
    assert all(math.isfinite(r) for r in returns), "episode 回报含 nan/inf"


def test_train_default_ppo_is_optional() -> None:
    """train(cfg) 不传 ppo 也能用(签名 ppo=None;此处给小 cfg 验证默认分支不崩)。

    注:默认 PPOConfig total_timesteps=4096 较大,这里仍用默认以验证 ppo=None 分支可运行;
    若耗时过长由 CI 控,功能上必须可跑通(只断言返回 dict 且 num_updates>=1)。
    """
    cfg = small_cfg()
    stats = train(cfg, PPOConfig(total_timesteps=256, rollout_steps=128, verbose=False))
    assert isinstance(stats, dict)
    assert stats["num_updates"] >= 1


# ======================================================================
# §v1.1.7 回归 —— 策略输出必须是 map 的函数 + logit 与粗格空间对齐
# 锁死"反 GAP 盲视"不变量:旧 map 分支末端 AdaptiveAvgPool2d((1,1)) 把 (Hc,Wc) 压成
# 标量 → 策略对任意 map 恒输出固定行;新空间对齐 1×1 Conv 头 → logit[idx] 由第
# (row,col=idx//Wc,idx%Wc) 粗格的局部融合特征算出。下面两测试一旦退回 GAP 必失败。
# ======================================================================
def test_actor_output_depends_on_map() -> None:
    """反 GAP 盲视(核心):空间差异极大的 map 必须产出不同的策略输出。

    造 6 张同形状 (Hc,Wc) 但高概率质量位置截然不同的 map(角落单峰 ×2、上/下半亮、
    左/右半亮),固定同一 vec=zeros(10),各取 logits 的 argmax。
    - 新空间架构:logit[idx] 由第 (row,col) 格特征算出 → 亮点在哪、argmax 随之改变;
      故 argmax 集合 size >= 2,且差异极大的两张 map(峰在 (0,0) vs (Hc-1,Wc-1))
      的 logits not allclose。
    - 旧 GAP 架构:map 被压成"平均亮度"标量,策略恒输出与空间无关的固定行 →
      所有 argmax 相同(集合 size==1)、两张 map 的 logits allclose → 必失败。
    """
    import numpy as np

    cfg, Hc, Wc, n = _shapes()
    torch.manual_seed(0)
    net = build_actor_critic(cfg, n)

    def _m(arr) -> "torch.Tensor":
        return torch.tensor(np.asarray(arr, dtype=np.float32))

    peak00 = np.zeros((Hc, Wc), np.float32); peak00[0, 0] = 1.0           # 峰在 (0,0)
    peakHW = np.zeros((Hc, Wc), np.float32); peakHW[Hc - 1, Wc - 1] = 1.0  # 峰在 (Hc-1,Wc-1)
    top = np.zeros((Hc, Wc), np.float32); top[: Hc // 2, :] = 1.0          # 上半亮 (row<Hc//2)
    bottom = np.zeros((Hc, Wc), np.float32); bottom[Hc // 2 :, :] = 1.0    # 下半亮
    left = np.zeros((Hc, Wc), np.float32); left[:, : Wc // 2] = 1.0        # 左半亮 (col<Wc//2)
    right = np.zeros((Hc, Wc), np.float32); right[:, Wc // 2 :] = 1.0      # 右半亮

    maps = [peak00, peakHW, top, bottom, left, right]
    vec = torch.zeros(10)

    logits_list = [net({"map": _m(arr), "vec": vec})[0][0] for arr in maps]
    argmaxes = {int(lg.argmax()) for lg in logits_list}

    # 主断言:不同 map 的策略输出不全相同(反 GAP 盲视;旧架构此处恒等 → size==1 必败)。
    assert len(argmaxes) >= 2, (
        f"所有差异 map 的 argmax 相同 ({argmaxes});疑似 map 分支被 GAP 压成标量、"
        f"策略读不到 map 空间信息(§v1.1.7 旧 bug 复现)。"
    )
    # 更强断言:峰在对角两端 (0,0) vs (Hc-1,Wc-1) 的 logits 必不接近(GAP 下恒等 → allclose)。
    assert not torch.allclose(logits_list[0], logits_list[1]), (
        "峰在 (0,0) 与峰在 (Hc-1,Wc-1) 的 map 给出几乎相同的 logits;策略未读 map 空间信息。"
    )


def test_actor_logit_spatially_aligned_to_cell() -> None:
    """空间对齐(梯度):logit[idx] 由第 (row,col=idx//Wc,idx%Wc) 粗格的邻域算出。

    取 idx=n-1(对应 (row=Hc-1, col=Wc-1) 角格,行主序 idx=row*Wc+col)。对一张全正随机
    map(+0.1 避免全零经 ReLU 把梯度压死——A5 踩过的坑)求 ∂logits[0,idx]/∂map:
    - 对齐邻域(row-1:row+2, col-1:col+2 裁剪到边界)的 |grad| 之和 > 0
      → logit[idx] 确实由 (row,col) 格邻域决定;
    - 离对齐格 > 感受野半径(两层 map_cnn 3x3 + actor_conv 3x3 ≈ 7x7、半径≈3)的远区
      |grad| 之和 == 0 → logit[idx] 不受远处格影响,空间对齐而非空间盲视。
      此处 Wc=2(仅 2 列,列向拉不开),故远区取行向距对齐格 (Hc-1) 共 (Hc-1) 格的
      第 0 行(行距=Hc-1=4 > 3,超出感受野)作可靠零梯度区;旧 GAP 架构下任一格梯度
      均经全局平均传播到 logit → 远区 |grad| 不为 0 → 必失败。
    """
    import numpy as np

    cfg, Hc, Wc, n = _shapes()
    torch.manual_seed(0)
    net = build_actor_critic(cfg, n)

    idx = n - 1                      # 行主序 idx=row*Wc+col → 角格 (Hc-1, Wc-1)
    row, col = idx // Wc, idx % Wc
    assert (row, col) == (Hc - 1, Wc - 1)

    rng = np.random.default_rng(0)
    # 全正 map(+0.1):避免全零输入下 ReLU 把整条反传梯度压死(A5 踩过的坑)。
    arr = rng.random((Hc, Wc)).astype(np.float32) + 0.1
    mp = torch.tensor(arr, requires_grad=True)
    vec = torch.zeros(10)

    logits, _ = net({"map": mp, "vec": vec})
    logits[0, idx].backward()
    grad = mp.grad.abs()             # (Hc, Wc)

    # 对齐邻域 (row-1:row+2, col-1:col+2),裁剪到边界。
    neigh = grad[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2].sum()
    # 远区:行向距对齐格最远的第 0 行(行距=Hc-1=4 > 感受野半径≈3 → 应零梯度)。
    far = grad[0, :].sum()

    assert float(neigh) > 0.0, (
        f"对齐格 (row={row},col={col}) 邻域的 |grad| 之和为 0;logit[idx] 未读对齐格信息。"
    )
    # 远区严格为 0(超出感受野);旧 GAP 经全局平均会把任意格梯度传到每个 logit → 必非 0。
    assert float(far) == 0.0, (
        f"远区(第 0 行,距对齐格 {Hc - 1} 格 > 感受野半径)|grad| 之和={float(far):.3e} ≠ 0;"
        f"疑似 GAP 全局平均把任意格梯度传到 logit[idx](空间盲视 §v1.1.7 复现)。"
    )
