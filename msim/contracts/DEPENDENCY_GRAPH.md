# 依赖图(DEPENDENCY GRAPH)

> 单向自底向上。**所有跨层依赖都是对 `contracts/` 的依赖**(`import msim.contracts.*`),
> 而非对他人实现模块的依赖 —— 这是并行红利来源(A4/A6 无需等 A2/A3 的真实实现即可开工)。
>
> **状态(2026-06-12):波次1 全层(L0/L1/L2a/L2b/L5)实现完成并通过验收(91/91);消费者从 `msim.<层>.<模块>` import 实现(§v1.0.2,非从 contracts);跨层反向 import 静态检查全清。波次2(L3/L4)待开。**

## 分层、模块、归属
```
L0  config/config(A2)            geometry/grid(A2)
L1  geometry/window(A2)          env_static/field(A2)      env_static/leader_path(A2)
L2a physics/auv_dynamics(A3)     physics/leader(A3)        physics/follower(A3)   physics/rollout(A3)
L2b task/corridor(A4)            task/value(A4)            task/belief(A4)        task/energy_balance(A4)
L3  mdp/commitment(A5)           mdp/state(A5)             mdp/action(A5)         mdp/coop_survey_env(A5)
L4  policy/network(A5)           policy/train_ppo(A5)
L5  baselines/{ortools_vrp,exact_milp,greedy}(A6)          eval/{metrics,runner}(A6)   ← 旁路
```

## 依赖边(→ 表示"依赖其契约")
```
                 ┌─────────────────────────── contracts/ (A0, 冻结) ───────────────────────────┐
                 │  types · config · geometry · physics · task · mdp                            │
                 └──────────────────────────────────────────────────────────────────────────────┘
                         ▲          ▲             ▲              ▲             ▲
   L0  config ───────────┘          │             │              │             │
   L0  grid ────────────────────────┤             │              │             │
   L1  window ─────► grid 契约       │             │              │             │
   L1  field, leader_path ─► config 契约           │              │             │
   L2a physics(rollout/leader/follower) ─► config, geometry, env_static 契约 + auv_dynamics
   L2b task(corridor/value/belief/energy_balance) ─► config, geometry, physics(CameraSensorModel) 契约
   L3  mdp(commitment/state/action/env) ─► L0+L1+L2a+L2b 全部契约
   L4  policy ─► mdp 契约
   L5  baselines/eval ─► 仅 L0+L1+L2a+L2b 契约(不依赖 mdp/policy/RL)
```

## L3→L5 已授权跨边(§v1.1.1,2026-06-23)
- `mdp/coop_survey_env.reset` → `eval.runner.default_instance`(L5):env 借用 A6 的实例生成器产出训练用
  (targets, field) 耦合实例。**A0 在 A5 派工单中授权**;单向无环(eval 不 import mdp/policy,见下静态检查)。
  动机:RL 与基线消费**同一实例生成口径** → 公平对比。**非反向 import**(L5 是旁路,不在 L3 之上)。
  清洁化备选(未做,留后续):实例生成下沉到 L1 `env_static`,L3 与 L5 各自消费,解除此耦合。

## 关键无依赖点(防反向 import,harness §4.1)
- `task` 依赖 `physics` 的**契约**(`CameraSensorModel`),不依赖 physics 实现 → A4 可独立于 A3 完成。
- `baselines/eval` **不**依赖 `mdp/policy` → A6 与 A5 完全并行,可更早完成(基线先行当标尺)。
- 任何"第 N 层 import 第 N+1 层"即违规(A0 集成时用静态检查拦截)。

## 可并行性
| 波次 | 可并行单元 |
|---|---|
| 0(串行) | A0:冻结 contracts/ + 骨架 + 本图 + 看板 |
| 1(大并行) | **A2**(L0/L1)∥ **A3**(L2a)∥ **A4**(L2b)∥ **A6**(L5);**A1** 同步写验收测试 |
| 2(集成) | **A5**(L3/L4)+ A0 端到端 smoke;A3 可在此换 Level 2 真机口径 |
