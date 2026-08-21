"""[第4层][A5] actor-critic 网络(Dict obs,**空间对齐卷积 actor 头**)。

§v1.1.7 裁决:旧版 map 分支末端做 AdaptiveAvgPool2d((1,1))(全局平均池化 GAP),把
(Hc,Wc) 粗网格压成单标量 → 丢掉"高概率在哪个格"的空间信息;而动作空间是空间的
(挑粗格 idx=row*Wc+col)→ 策略恒输出固定行、读不到 map。本版改为空间对齐卷积头:

- map 分支:2 层小 CNN(padding=1 保形),**不做 GAP**,保持 (B,cnn_ch,Hc,Wc);
- vec 分支:小 MLP → (B,vec_hidden),再 view(B,vec_hidden,1,1).expand 广播到 (Hc,Wc),
  与 map 特征在通道维 concat → (B, cnn_ch+vec_hidden, Hc, Wc);
- actor 头:Conv3x3(保形)→ReLU → 1×1 Conv(→1) → (B,1,Hc,Wc) → reshape(B,Hc*Wc)
  = logits。**logit[r,c] 由第 (r,c) 格的局部融合特征(含其先验概率)算出** → 策略能
  "指向"高概率格。
- critic 头:对融合空间特征 GAP→flatten→小 MLP→标量 V(价值是标量,池化无妨)。

⚠ correctness 关键(idx 顺序):logits = (B,1,Hc,Wc).reshape(B, Hc*Wc)。torch 默认行主序
(C 序):对 (Hc,Wc) 先遍历 Hc(行=横向)再 Wc(列=沿测线),展平后 idx = row*Wc + col,
**与 coarse_index_to_world / decode_action / env._coarse_map(cmap[row,col]) 完全一致**。
故 logits[idx] 对应的格 == decode_action(idx) 对应的格,动作语义对齐。

形状:Hc,Wc 不由 env 传入,而由 cfg.window/cfg.sensor 经 coarse_grid_shape 推出,在
__init__ 算出 self.Hc/self.Wc,并断言 Hc*Wc == n_actions(与 env.action_space.n 同源)。

⚠ torch 延迟 import(波次0/契约层不依赖 torch)。规模受 Raspberry Pi 5 推理约束 → **小网络**
(cnn_ch=16、vec_hidden=32、fuse_hidden=64;1×1 conv 头很廉价)。

输入:obs = {"map": (B,Hc,Wc) 或 (Hc,Wc), "vec": (B,10) 或 (10,)} float32。
输出:(logits[B,n_actions], value[B])。延迟 import 使本模块在无 torch 环境仍可被 import
(函数体内才触发 torch 依赖)。
"""
from __future__ import annotations

from typing import Any

from msim.contracts.config import SimConfig
from msim.geometry.window import coarse_grid_shape

__all__ = ["build_actor_critic"]


def build_actor_critic(cfg: SimConfig, n_actions: int) -> Any:
    """构造 actor-critic(返回 torch.nn.Module)。torch 在此函数内延迟 import。

    cfg 用于读取与 obs 结构相关的形状约定:map 单通道、vec 10 维,且粗网格形状
    (Hc,Wc) = coarse_grid_shape(cfg.window, cfg.sensor)(actor 头按此输出每格一个 logit)。
    n_actions = Discrete(Hc*Wc) 的 n,由 env.action_space.n 传入;须满足 Hc*Wc == n_actions。
    """
    import torch
    import torch.nn as nn

    # —— 小规模超参(Pi 5 推理约束:通道/隐藏维都取小)——
    cnn_ch = 16          # map CNN 通道
    vec_hidden = 32      # vec MLP 隐藏维(广播进空间)
    fuse_hidden = 64     # 融合卷积 / critic MLP 隐藏维
    vec_dim = 10         # obs["vec"] 维度(self 5 + teammate 5,契约冻结)

    # 粗网格形状(actor 头空间分辨率)。与 env._coarse_map / action_space 同源。
    Hc, Wc = coarse_grid_shape(cfg.window, cfg.sensor)
    Hc, Wc = int(Hc), int(Wc)
    if Hc * Wc != n_actions:
        raise ValueError(
            f"coarse_grid_shape 推出 Hc*Wc={Hc * Wc} 与 n_actions={n_actions} 不一致"
            f"(Hc={Hc}, Wc={Wc});actor 头每格一个 logit 需二者相等。"
        )

    class ActorCritic(nn.Module):
        """Dict obs actor-critic。map→CNN(保形)、vec→MLP 广播注入空间、融合→空间 actor + GAP critic。"""

        def __init__(self) -> None:
            super().__init__()
            self.Hc, self.Wc = Hc, Wc
            self.n_actions = n_actions

            # map 分支:单通道粗网格 → 小 CNN(padding=1 保形,兼容极小 Hc/Wc)。**不做 GAP**。
            self.map_cnn = nn.Sequential(
                nn.Conv2d(1, cnn_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cnn_ch, cnn_ch, kernel_size=3, padding=1),
                nn.ReLU(),
            )  # → (B, cnn_ch, Hc, Wc)
            # vec 分支:小 MLP → (B, vec_hidden),后续广播成 (B, vec_hidden, Hc, Wc)。
            self.vec_mlp = nn.Sequential(
                nn.Linear(vec_dim, vec_hidden),
                nn.Tanh(),
                nn.Linear(vec_hidden, vec_hidden),
                nn.Tanh(),
            )
            fused_ch = cnn_ch + vec_hidden  # 融合后通道
            # actor 头:保形卷积融合 → 1×1 Conv 每格一个 logit。
            self.actor_conv = nn.Sequential(
                nn.Conv2d(fused_ch, fuse_hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            )  # → (B, fuse_hidden, Hc, Wc)
            self.actor_head = nn.Conv2d(fuse_hidden, 1, kernel_size=1)  # → (B, 1, Hc, Wc)
            # critic 头:融合空间特征 GAP → 小 MLP → 标量 V。
            self.critic_mlp = nn.Sequential(
                nn.Linear(fused_ch, fuse_hidden),
                nn.Tanh(),
            )
            self.critic_head = nn.Linear(fuse_hidden, 1)
            self._init_weights()

        def _init_weights(self) -> None:
            # 正交初始化(PPO 稳定常用);actor 最后 1×1 Conv 头用小增益(≈0.01)使初始策略接近均匀。
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)  # 1×1 Conv,小增益→近均匀初始策略
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

        def _fused_spatial(self, obs: dict) -> "torch.Tensor":
            """obs → 融合空间特征 (B, cnn_ch+vec_hidden, Hc, Wc)。

            兼容无 batch obs(map (Hc,Wc) / vec (10,))与有 batch obs(map (B,Hc,Wc) / vec (B,10))。
            """
            m = obs["map"]
            v = obs["vec"]
            if m.dim() == 2:          # (Hc,Wc) → (1,1,Hc,Wc)
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.dim() == 3:        # (B,Hc,Wc) → (B,1,Hc,Wc)
                m = m.unsqueeze(1)
            if v.dim() == 1:          # (10,) → (1,10)
                v = v.unsqueeze(0)
            B, _, Hc_, Wc_ = m.shape
            m_feat = self.map_cnn(m)                          # (B, cnn_ch, Hc, Wc)
            v_feat = self.vec_mlp(v)                          # (B, vec_hidden)
            # 广播注入空间:每个粗格都拼上同一份 vec 上下文。
            v_spatial = v_feat.view(B, -1, 1, 1).expand(B, -1, Hc_, Wc_)  # (B, vec_hidden, Hc, Wc)
            return torch.cat([m_feat, v_spatial], dim=1)      # (B, cnn_ch+vec_hidden, Hc, Wc)

        def forward(self, obs: dict):
            """返回 (logits[B,n_actions], value[B])。

            actor:融合空间特征 → actor_conv → 1×1 Conv → (B,1,Hc,Wc) → reshape(B,Hc*Wc)。
            **reshape 行主序 ⇒ logits[idx] 对应粗格 idx=row*Wc+col**(与 decode_action 一致)。
            critic:融合空间特征 GAP → MLP → 标量 V。
            """
            fused = self._fused_spatial(obs)                  # (B, C, Hc, Wc)
            # —— actor:空间 1×1 conv 头 → 每格一个 logit ——
            a = self.actor_conv(fused)                        # (B, fuse_hidden, Hc, Wc)
            a = self.actor_head(a)                            # (B, 1, Hc, Wc)
            # 行主序展平:(B,1,Hc,Wc) → (B, Hc*Wc),idx = row*Wc + col(与几何契约一致)。
            logits = a.reshape(a.shape[0], -1)                # (B, Hc*Wc) == (B, n_actions)
            # —— critic:GAP → MLP → 标量 ——
            c = fused.mean(dim=(2, 3))                         # GAP: (B, C)
            value = self.critic_head(self.critic_mlp(c)).squeeze(-1)  # (B,)
            return logits, value

        # —— PPO 便捷接口 ——
        def get_value(self, obs: dict) -> "torch.Tensor":
            return self.forward(obs)[1]

        def get_action_and_value(self, obs: dict, action=None):
            """采样/评估动作 + 价值。返回 (action, log_prob, entropy, value)。"""
            logits, value = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value

    return ActorCritic()
