"""[第4层][A5] CleanRL 单文件式 PPO 训练循环(gymnasium 1.2.x API)+ 训练体验增强。

⚠ torch / gymnasium 延迟 import(波次0/契约层不依赖)。按 §v1.1.0:
- 目标环境 = auv_py310(gymnasium 1.2.3 + torch 2.5.1);**不**引入 cleanrl / gymnasium 0.28 依赖。
- gymnasium 1.2.x API:reset(seed=)→(obs,info)、step→五元组 (obs,reward,terminated,truncated,info)。
- 通信延迟/丢包训练时关闭:CoopSurveyEnv 为开环、reward 干净(无通信模型),天然满足。

实现要点:
- 单环境(非向量化)逐 episode 收集 rollout;Dict obs(map+vec)按键打包成张量。
- 标准 PPO:GAE-λ 优势、clip 目标、value clip、entropy bonus、梯度裁剪、多 epoch minibatch。
- 验收边界(§v1.1.0 DoD③):PPO 能连续训练若干 episode 不崩、reward 有数;**不要求**收敛。

训练体验增强(人类追加,2026-06-22):
1. **wandb 接入(默认关,gated)**:use_wandb=True 才延迟 import wandb + init/log/finish;关时零依赖。
2. **checkpoint save/load**:存 net+optimizer+PPOConfig+n_actions+SimConfig;resume_from 续训 / None 重训。
   动作空间(粗网格)不匹配时拒绝加载并报清晰错误。
3. **ESC 键盘监听(Windows/msvcrt,非阻塞)**:训练中按 ESC → 完成当前 update → 存盘 → 优雅退出。
   全程 try/except 保护;停止判定抽成可编程触发的 StopController(便于非交互自验)。
4. **CLI 入口**:main() + argparse,支持人类 `python -m msim.policy.train_ppo --wandb ...` 交互式跑。

**向后兼容**:train(cfg, ppo=None)->dict 基础签名不变;所有新行为经 PPOConfig 带默认值字段驱动
(默认:无 wandb、无 resume、ESC 监听开但非交互时静默跳过)。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from msim.contracts.config import SimConfig

__all__ = ["PPOConfig", "StopController", "train", "save_checkpoint", "load_checkpoint", "main"]


# ======================================================================
# 超参
# ======================================================================
@dataclass
class PPOConfig:
    """PPO 超参(小规模默认:DoD 只需能训能跑完,非收敛)+ 训练体验旋钮。

    新增字段全部带默认值(向后兼容):默认无 wandb、无 resume、ESC 监听开但非交互时静默跳过。
    """
    total_timesteps: int = 4096        # 总环境步数(可调小快速跑通)
    rollout_steps: int = 256           # 每次更新收集的步数(buffer 大小)
    num_epochs: int = 4                # 每批数据复用 epoch 数
    num_minibatches: int = 4           # minibatch 数
    gamma: float = 0.99                # 折扣
    gae_lambda: float = 0.95           # GAE λ
    clip_coef: float = 0.2             # PPO clip 系数
    ent_coef: float = 0.01             # 熵奖励系数
    vf_coef: float = 0.5               # value loss 系数
    max_grad_norm: float = 0.5         # 梯度裁剪
    learning_rate: float = 3.0e-4
    seed: int = 0
    device: str = "cpu"                # 小网络 CPU 即可;Pi 5 推理一致
    log_interval_updates: int = 1      # 每多少次更新打印一次
    verbose: bool = True

    # —— wandb 接入(默认关,gated;延迟 import)——
    use_wandb: bool = False
    wandb_project: str = "auv-coop-survey"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"         # online | offline | disabled

    # —— checkpoint save/load ——
    checkpoint_path: str = "runs/policy_latest.pt"  # ESC 保存与训练结束保存都写这里
    resume_from: Optional[str] = None  # 非 None → 加载续训;None → 从头训(重训)

    # —— ESC 键盘监听(Windows/msvcrt 非阻塞)——
    enable_keyboard_stop: bool = True  # 训练中按 ESC → 优雅存盘退出;非交互时静默跳过

    # —— §B 行为克隆暖启动(greedy-oracle 示范;默认关,向后兼容)——
    bc_demo_steps: int = 0   # 收集多少 greedy (obs,action) 示范步;0 = 不做 BC
    bc_epochs: int = 0       # BC(交叉熵模仿)训练 epoch 数;0 = 不做 BC
    bc_lr: float = 1.0e-3    # BC 独立优化器学习率(不污染 PPO Adam 动量)


# ======================================================================
# 停止控制器(ESC 监听抽象;可编程触发,便于非交互自验)
# ======================================================================
class StopController:
    """停止控制器

    交互式:训练循环每步轮询 poll(),内部用 msvcrt 非阻塞检测 ESC;读到 ESC → 置 stop 标志。
    非交互式(子 agent / 无 tty / 非 Windows):监听静默禁用,poll() 始终不置位;
    自验时直接 request_stop() **编程触发**,验证存盘 + 优雅退出,无需真键盘。

    全程 try/except 保护:msvcrt 缺失 / stdin 非交互 / 任何异常都不致崩,只静默退化为"不监听"。
    """

    ESC = b"\x1b"

    def __init__(self, enable: bool = True) -> None:
        self._stop = False
        self._reason = ""
        self._msvcrt = None
        self._listening = False
        if enable:
            self._try_enable_msvcrt()

    def _try_enable_msvcrt(self) -> None:
        """尝试启用 msvcrt 非阻塞键盘监听;任何不满足条件即静默跳过。"""
        try:
            import sys
            import msvcrt  # Windows 原生;非 Windows 直接 ImportError

            # 仅当 stdin 是交互式 tty 才监听(子 agent / 管道 / 重定向 → 跳过,避免误读)。
            if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
                self._listening = False
                return
            self._msvcrt = msvcrt
            self._listening = True
        except Exception:
            # 非 Windows / 无 msvcrt / 任何异常 → 静默退化为不监听。
            self._msvcrt = None
            self._listening = False

    @property
    def listening(self) -> bool:
        """是否真正启用了键盘监听(交互式 Windows tty)。"""
        return self._listening

    @property
    def stop_requested(self) -> bool:
        return self._stop

    @property
    def reason(self) -> str:
        return self._reason

    def request_stop(self, reason: str = "programmatic") -> None:
        """编程触发停止(自验 / 外部信号用):置位停止标志。"""
        self._stop = True
        if not self._reason:
            self._reason = reason

    def poll(self) -> bool:
        """轮询键盘(非阻塞)。读到 ESC → 置停止标志。返回当前 stop_requested。

        try/except 全包:监听过程中任何异常都不致崩,退化为不监听。
        """
        if self._stop:
            return True
        if not self._listening or self._msvcrt is None:
            return self._stop
        try:
            # 把缓冲里所有按键都吃掉,任一为 ESC 即停。
            while self._msvcrt.kbhit():
                ch = self._msvcrt.getch()
                if ch == self.ESC:
                    self.request_stop("ESC pressed")
                    break
        except Exception:
            # 监听异常 → 关闭监听,绝不影响训练。
            self._listening = False
        return self._stop


# ======================================================================
# checkpoint save / load
# ======================================================================
def save_checkpoint(path: str, net, optimizer, ppo: "PPOConfig",
                    n_actions: int, cfg: SimConfig) -> None:
    """保存 checkpoint(net + optimizer + PPOConfig + n_actions + SimConfig,够 resume)。

    torch 延迟 import;自动建目录。SimConfig 存为可序列化 dict(asdict)。
    """
    import torch

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "format_version": 1,
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "ppo_config": asdict(ppo),
        "n_actions": int(n_actions),
        "sim_config": asdict(cfg),
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """加载 checkpoint payload(不直接重建对象,交由 train 据当前 cfg 重建后载入 state)。

    torch 延迟 import。文件不存在 → FileNotFoundError(清晰报错)。
    """
    import torch

    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    # weights_only=False:payload 含 PPOConfig/SimConfig 的 python 容器(dict/list),非纯权重。
    return torch.load(path, map_location=map_location, weights_only=False)


def _restore_into(net, optimizer, payload: Dict[str, Any], n_actions: int,
                  torch) -> None:
    """把 checkpoint payload 的 net/optimizer 状态载入当前对象。

    **动作空间(粗网格形状)不匹配** → 拒绝加载并报清晰错误:checkpoint 的 n_actions 由其
    SimConfig 的 window/sensor 推导,若与当前 cfg 算出的 n_actions 不同,则网络 actor 头维度
    不一致,续训无意义且会 shape mismatch。
    """
    ckpt_n = int(payload.get("n_actions", -1))
    if ckpt_n != int(n_actions):
        raise ValueError(
            f"checkpoint 动作空间不匹配: checkpoint n_actions={ckpt_n} ≠ 当前 cfg n_actions={n_actions}。"
            f" 粗网格形状(coarse_grid_shape,由 cfg.window/cfg.sensor 推导)不一致,无法续训;"
            f" 请用与 checkpoint 相同的 SimConfig(window/sensor/footprint),或从头训练(resume_from=None)。"
        )
    net.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])


# ======================================================================
# obs 打包
# ======================================================================
def _obs_to_tensors(obs_list: List[Dict[str, np.ndarray]], torch, device):
    """把一批 Dict obs([{"map":..,"vec":..}, ...])打包成 {"map":(B,..),"vec":(B,..)} 张量。"""
    maps = np.stack([o["map"] for o in obs_list]).astype(np.float32)
    vecs = np.stack([o["vec"] for o in obs_list]).astype(np.float32)
    return {
        "map": torch.as_tensor(maps, device=device),
        "vec": torch.as_tensor(vecs, device=device),
    }


def _single_obs_tensor(obs: Dict[str, np.ndarray], torch, device):
    return {
        "map": torch.as_tensor(obs["map"][None], device=device),
        "vec": torch.as_tensor(obs["vec"][None], device=device),
    }


# ======================================================================
# wandb 接入(gated;延迟 import)
# ======================================================================
def _init_wandb(ppo: "PPOConfig", cfg: SimConfig, n_actions: int):
    """use_wandb 时延迟 import 并 init;返回 run(或 None)。关时零依赖。"""
    if not ppo.use_wandb:
        return None
    import wandb  # 仅 use_wandb 时 import(关时零依赖)

    config = dict(asdict(ppo))
    # 附带 SimConfig 关键项(便于面板筛选/复现),避免塞整棵嵌套树。
    config.update(
        {
            "n_actions": int(n_actions),
            "world_x_max_m": cfg.world.x_max_m,
            "world_y_max_m": cfg.world.y_max_m,
            "world_res_m": cfg.world.res_m,
            "window_look_back_m": cfg.window.look_back_m,
            "window_width_m": cfg.window.width_m,
            "footprint_along_m": cfg.sensor.footprint_along_m,
            "footprint_lateral_m": cfg.sensor.footprint_lateral_m,
            "n_followers": cfg.n_followers,
            "max_episode_time_s": cfg.max_episode_time_s,
            "value_weight": cfg.reward.value_weight,
            "balance_weight": cfg.reward.balance_weight,
            "value_norm_ref": cfg.reward.value_norm_ref,
            "balance_norm_ref": cfg.reward.balance_norm_ref,
        }
    )
    return wandb.init(
        project=ppo.wandb_project,
        name=ppo.wandb_run_name,
        mode=ppo.wandb_mode,
        config=config,
    )


# ======================================================================
# §B 行为克隆暖启动(用 greedy-oracle 示范把 actor 从均匀随机拉到"会瞄准")
# ======================================================================
def _collect_greedy_demos(cfg: SimConfig, demo_steps: int):
    """跑 greedy-oracle(每步选 obs map 残余最高粗格)收集 (map, vec, action) 示范。

    greedy 在本 env 机制下已实测 100% 覆盖(诊断 C 组),是完美示范器;动作 = argmax(obs['map'])。
    """
    from msim.mdp.coop_survey_env import CoopSurveyEnv

    env = CoopSurveyEnv(cfg)
    obs, _ = env.reset()
    maps: List[np.ndarray] = []
    vecs: List[np.ndarray] = []
    acts: List[int] = []
    for _ in range(int(demo_steps)):
        a = int(np.argmax(np.asarray(obs["map"], dtype=np.float32).reshape(-1)))
        maps.append(np.asarray(obs["map"], dtype=np.float32))
        vecs.append(np.asarray(obs["vec"], dtype=np.float32))
        acts.append(a)
        obs, _r, terminated, truncated, _i = env.step(a)
        if terminated or truncated:
            obs, _ = env.reset()
    return maps, vecs, acts


def _behavior_clone(net, cfg: SimConfig, ppo: "PPOConfig", torch, nn, device) -> Dict[str, Any]:
    """BC 暖启动:交叉熵模仿 greedy 示范动作。用**独立优化器**(不污染 PPO Adam 动量)。

    只优化 actor(+共享主干);critic 头不进 CE 损失,留待 PPO 学。默认 bc_*=0 → 跳过。
    """
    if ppo.bc_demo_steps <= 0 or ppo.bc_epochs <= 0:
        return {"bc_done": False}
    import torch.optim as optim

    maps, vecs, acts = _collect_greedy_demos(cfg, ppo.bc_demo_steps)
    n = len(acts)
    if n == 0:
        return {"bc_done": False}
    t_map = torch.as_tensor(np.stack(maps), device=device)
    t_vec = torch.as_tensor(np.stack(vecs), device=device)
    t_act = torch.as_tensor(np.asarray(acts), dtype=torch.long, device=device)

    bc_opt = optim.Adam(net.parameters(), lr=ppo.bc_lr, eps=1e-5)
    ce = nn.CrossEntropyLoss()
    mb = max(1, n // 4)
    idx = np.arange(n)
    last = 0.0
    for ep in range(ppo.bc_epochs):
        np.random.shuffle(idx)
        tot, nb = 0.0, 0
        for s in range(0, n, mb):
            j = idx[s:s + mb]
            logits, _ = net.forward({"map": t_map[j], "vec": t_vec[j]})
            loss = ce(logits, t_act[j])
            bc_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), ppo.max_grad_norm)
            bc_opt.step()
            tot += float(loss.item())
            nb += 1
        last = tot / max(1, nb)
        if ppo.verbose:
            # 训练集上的模仿准确率(argmax==greedy 动作的比例),粗看是否拉离均匀。
            with torch.no_grad():
                pred = torch.argmax(net.forward({"map": t_map, "vec": t_vec})[0], dim=-1)
                acc = float((pred == t_act).float().mean().item())
            print(f"[BC] epoch {ep + 1}/{ppo.bc_epochs} ce_loss {last:.4f} imitate_acc {acc:.3f} "
                  f"(demos={n})")
    return {"bc_done": True, "bc_demos": n, "bc_ce_loss": last}


# ======================================================================
# 训练主入口
# ======================================================================
def train(cfg: SimConfig, ppo: Optional[PPOConfig] = None,
          stop_controller: Optional["StopController"] = None) -> Dict[str, Any]:
    """PPO 训练入口(torch/gymnasium 在函数内延迟 import)。

    cfg: SimConfig(env 构造 + reward 归一化基准来源)。
    ppo: 可选 PPO 超参;缺省用小规模 PPOConfig(便于快速跑通端到端)。
    stop_controller: 可选优雅停止控制器(自验编程触发用);缺省按 ppo.enable_keyboard_stop
        新建一个(交互式 Windows tty 才真正监听 ESC,否则静默禁用)。

    续训 or 重训(参数驱动):ppo.resume_from 非 None → 加载该 checkpoint 续训(net+optimizer);
    为 None → 全新空白模型从头训。

    返回训练统计 dict(更新次数、各更新平均回报/损失、是否被 ESC 提前停止、checkpoint 路径等)。
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from msim.mdp.coop_survey_env import CoopSurveyEnv
    from msim.policy.network import build_actor_critic

    if ppo is None:
        ppo = PPOConfig()
    if stop_controller is None:
        stop_controller = StopController(enable=ppo.enable_keyboard_stop)

    # —— 复现性 ——
    np.random.seed(ppo.seed)
    torch.manual_seed(ppo.seed)
    device = torch.device(ppo.device)

    # —— env + 网络 + 优化器 ——
    env = CoopSurveyEnv(cfg)
    n_actions = int(env.action_space.n)
    net = build_actor_critic(cfg, n_actions).to(device)
    optimizer = optim.Adam(net.parameters(), lr=ppo.learning_rate, eps=1e-5)

    # —— 续训 / 重训选择(参数驱动)——
    resumed = False
    if ppo.resume_from is not None:
        payload = load_checkpoint(ppo.resume_from, map_location=ppo.device)
        _restore_into(net, optimizer, payload, n_actions, torch)  # 不匹配会 raise
        resumed = True
        if ppo.verbose:
            print(f"[PPO] resumed from checkpoint: {ppo.resume_from} (n_actions={n_actions})")
    elif ppo.verbose:
        print("[PPO] fresh model (no resume).")

    # —— §B 行为克隆暖启动(greedy 示范;默认 bc_*=0 时跳过)——
    bc_stats = _behavior_clone(net, cfg, ppo, torch, nn, device)

    # —— wandb(gated)——
    wandb_run = _init_wandb(ppo, cfg, n_actions)

    batch_size = ppo.rollout_steps
    minibatch_size = max(1, batch_size // ppo.num_minibatches)
    num_updates = max(1, ppo.total_timesteps // batch_size)

    if ppo.verbose:
        if stop_controller.listening:
            print("[PPO] 训练中按 ESC 保存当前模型并结束。")
        else:
            print("[PPO] (非交互环境:ESC 键盘监听已静默禁用;可编程触发 StopController 停止。)")

    # —— rollout buffer(单环境;Dict obs 分键存)——
    obs, _ = env.reset(seed=ppo.seed)
    global_step = 0
    ep_return = 0.0           # 当前 episode 累计 reward
    ep_len = 0
    completed_returns: List[float] = []  # 已完成 episode 回报

    stats: Dict[str, Any] = {
        "num_updates": 0,
        "global_steps": 0,
        "update_mean_return": [],   # 每更新窗口内完成 episode 的平均回报
        "update_policy_loss": [],
        "update_value_loss": [],
        "update_entropy": [],
        "episode_returns": completed_returns,
        "resumed": resumed,
        "stopped_early": False,
        "stop_reason": "",
        "checkpoint_path": ppo.checkpoint_path,
        "wandb": bool(wandb_run is not None),
        "bc": bc_stats,
    }

    try:
        for update in range(num_updates):
            b_obs: List[Dict[str, np.ndarray]] = []
            b_actions: List[int] = []
            b_logprobs: List[float] = []
            b_rewards: List[float] = []
            b_dones: List[float] = []      # terminated or truncated
            b_values: List[float] = []
            window_returns: List[float] = []

            # ---------- 收集 rollout ----------
            for _ in range(ppo.rollout_steps):
                global_step += 1
                stop_controller.poll()  # 非阻塞轮询 ESC(交互时);非交互静默
                with torch.no_grad():
                    t_obs = _single_obs_tensor(obs, torch, device)
                    action, logprob, _, value = net.get_action_and_value(t_obs)
                a = int(action.item())

                next_obs, reward, terminated, truncated, _info = env.step(a)
                done = bool(terminated or truncated)

                b_obs.append(obs)
                b_actions.append(a)
                b_logprobs.append(float(logprob.item()))
                b_rewards.append(float(reward))
                b_dones.append(1.0 if done else 0.0)
                b_values.append(float(value.item()))

                ep_return += float(reward)
                ep_len += 1
                obs = next_obs

                if done:
                    completed_returns.append(ep_return)
                    window_returns.append(ep_return)
                    ep_return = 0.0
                    ep_len = 0
                    obs, _ = env.reset()

            # ---------- bootstrap value(最后一步之后)----------
            with torch.no_grad():
                t_obs = _single_obs_tensor(obs, torch, device)
                next_value = float(net.get_value(t_obs).item())

            # ---------- GAE-λ 优势 ----------
            rewards = np.asarray(b_rewards, dtype=np.float32)
            values = np.asarray(b_values, dtype=np.float32)
            dones = np.asarray(b_dones, dtype=np.float32)
            advantages = np.zeros(ppo.rollout_steps, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(ppo.rollout_steps)):
                next_nonterminal = 1.0 - dones[t]
                next_val = next_value if t == ppo.rollout_steps - 1 else values[t + 1]
                delta = rewards[t] + ppo.gamma * next_val * next_nonterminal - values[t]
                last_gae = delta + ppo.gamma * ppo.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values

            # ---------- 打包成张量 ----------
            bt_obs = _obs_to_tensors(b_obs, torch, device)
            bt_actions = torch.as_tensor(np.asarray(b_actions), dtype=torch.long, device=device)
            bt_logprobs = torch.as_tensor(np.asarray(b_logprobs, dtype=np.float32), device=device)
            bt_advantages = torch.as_tensor(advantages, device=device)
            bt_returns = torch.as_tensor(returns, device=device)
            bt_values = torch.as_tensor(values, device=device)

            # ---------- PPO 多 epoch minibatch 更新 ----------
            idx = np.arange(batch_size)
            last_pg = last_vf = last_ent = 0.0
            for _epoch in range(ppo.num_epochs):
                np.random.shuffle(idx)
                for start in range(0, batch_size, minibatch_size):
                    mb = idx[start:start + minibatch_size]
                    mb_obs = {
                        "map": bt_obs["map"][mb],
                        "vec": bt_obs["vec"][mb],
                    }
                    _, new_logprob, entropy, new_value = net.get_action_and_value(
                        mb_obs, bt_actions[mb]
                    )
                    logratio = new_logprob - bt_logprobs[mb]
                    ratio = logratio.exp()

                    mb_adv = bt_advantages[mb]
                    # 优势标准化(minibatch 内;数值稳定)。
                    if mb_adv.numel() > 1:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    # 策略 clip 损失。
                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - ppo.clip_coef, 1 + ppo.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    # value clip 损失。
                    new_value = new_value.view(-1)
                    v_unclipped = (new_value - bt_returns[mb]) ** 2
                    v_clipped = bt_values[mb] + torch.clamp(
                        new_value - bt_values[mb], -ppo.clip_coef, ppo.clip_coef
                    )
                    v_clipped = (v_clipped - bt_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                    ent_loss = entropy.mean()
                    loss = pg_loss - ppo.ent_coef * ent_loss + ppo.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), ppo.max_grad_norm)
                    optimizer.step()

                    last_pg, last_vf, last_ent = (
                        float(pg_loss.item()),
                        float(v_loss.item()),
                        float(ent_loss.item()),
                    )

            mean_ret = float(np.mean(window_returns)) if window_returns else float("nan")
            stats["num_updates"] = update + 1
            stats["global_steps"] = global_step
            stats["update_mean_return"].append(mean_ret)
            stats["update_policy_loss"].append(last_pg)
            stats["update_value_loss"].append(last_vf)
            stats["update_entropy"].append(last_ent)

            n_done = len(window_returns)
            if ppo.verbose and (update % ppo.log_interval_updates == 0):
                print(
                    f"[PPO] update {update + 1}/{num_updates} "
                    f"step {global_step} episodes_done {n_done} "
                    f"mean_return {mean_ret:.4f} "
                    f"pg_loss {last_pg:.4f} v_loss {last_vf:.4f} entropy {last_ent:.4f}"
                )

            # —— wandb log(gated)——
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "mean_return": mean_ret,
                        "pg_loss": last_pg,
                        "v_loss": last_vf,
                        "entropy": last_ent,
                        "episodes_done": n_done,
                        "update": update + 1,
                    },
                    step=global_step,
                )

            # —— ESC 优雅停止:完成当前 update 后存盘并 break ——
            if stop_controller.poll():
                stats["stopped_early"] = True
                stats["stop_reason"] = stop_controller.reason or "stop_requested"
                if ppo.verbose:
                    print(
                        f"[PPO] 收到停止请求({stats['stop_reason']}):完成当前 update,保存 checkpoint 并优雅退出。"
                    )
                break

        # —— 训练结束 / ESC 停止:统一保存 checkpoint ——
        save_checkpoint(ppo.checkpoint_path, net, optimizer, ppo, n_actions, cfg)
        stats["checkpoint_saved"] = True
        if ppo.verbose:
            print(f"[PPO] checkpoint 已保存: {ppo.checkpoint_path}")

    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if ppo.verbose:
        tag = " (ESC 提前停止)" if stats["stopped_early"] else ""
        print(
            f"[PPO] done{tag}: {stats['num_updates']} updates, "
            f"{stats['global_steps']} steps, "
            f"{len(completed_returns)} episodes completed."
        )
    return stats


# ======================================================================
# CLI 入口(人类交互式跑真实训练:看 wandb、按 ESC 存盘)
# ======================================================================
def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """命令行入口。示例:

        # 小尺度从头训 + offline wandb(免登录)
        python -m msim.policy.train_ppo --timesteps 8192 --wandb --wandb-mode offline

        # online wandb 实时看板(需先 `wandb login` 或设 WANDB_API_KEY)
        python -m msim.policy.train_ppo --timesteps 20000 --wandb

        # 从 checkpoint 续训
        python -m msim.policy.train_ppo --resume runs/policy_latest.pt --timesteps 8192
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="CoopSurvey PPO 训练(gymnasium 1.2.x;ESC 存盘退出;可选 wandb)。"
    )
    parser.add_argument("--timesteps", type=int, default=PPOConfig.total_timesteps,
                        help="总环境步数。")
    parser.add_argument("--rollout-steps", type=int, default=PPOConfig.rollout_steps,
                        help="每次更新收集的步数。")
    parser.add_argument("--seed", type=int, default=PPOConfig.seed, help="随机种子。")
    parser.add_argument("--lr", type=float, default=PPOConfig.learning_rate, help="学习率。")
    parser.add_argument("--device", type=str, default=PPOConfig.device,
                        help="cpu | cuda。")
    parser.add_argument("--checkpoint", type=str, default=PPOConfig.checkpoint_path,
                        help="checkpoint 保存路径(ESC 与训练结束都写此)。")
    parser.add_argument("--resume", type=str, default=None,
                        help="从该 checkpoint 续训(默认从头训)。")
    parser.add_argument("--no-keyboard", action="store_true",
                        help="禁用 ESC 键盘监听。")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb。")
    parser.add_argument("--wandb-project", type=str, default=PPOConfig.wandb_project,
                        help="wandb project 名。")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run 名。")
    parser.add_argument("--wandb-mode", type=str, default=PPOConfig.wandb_mode,
                        choices=["online", "offline", "disabled"], help="wandb 模式。")
    parser.add_argument("--world", type=str, default="small",
                        choices=["small", "default", "windowed"],
                        help="env 尺度:small(快速训练/校验)| default(全尺度 SimConfig)| "
                             "windowed(与 examples/mission_windows_survey 一致,checkpoint 可被该演示加载)。")
    args = parser.parse_args(argv)

    # env 配置:small_cfg(训练默认,episode 短)/ 全尺度 SimConfig / windowed_survey_cfg(演示复用)。
    from msim.mdp.coop_survey_env import small_cfg, windowed_survey_cfg

    if args.world == "small":
        cfg = small_cfg(args.seed)
    elif args.world == "windowed":
        cfg = windowed_survey_cfg(args.seed)
    else:
        cfg = SimConfig()

    ppo = PPOConfig(
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume,
        enable_keyboard_stop=not args.no_keyboard,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    return train(cfg, ppo)


if __name__ == "__main__":
    main()
