# 架构蓝图（Architecture Blueprint）

> **本文档的身份：建议结构，非冻结契约。**
>
> 这是修士 simulation 项目推荐的代码骨架，供 **A0（契约/集成 agent）在波次 0 用作参考**。A0 应：
> 1. 对照项目根目录的 **AUV 类定义文件**，核实并敲定 `physics/` 层（尤其 `rollout`）的实际划分；
> 2. 根据真实依赖关系调整目录、文件粒度与命名；
> 3. **产出一份敲定后的最终文件结构**，连同接口契约一起冻结并向全团队广播。
>
> 调整后，本蓝图即被最终结构取代。建造 agent（A2~A6）以 A0 冻结的版本为准，不以本蓝图为准。
>
> 配套文档：`project_summary.md`（做什么）、`agent_team_harness.md`（团队如何分工与验收）。

---

## 1. 设计原则（A0 调整时须保留）

无论 A0 如何调整文件粒度，以下两条结构性原则必须保留，因为它们是整个项目可维护性与学术可信度的支柱：

**原则一：确定性几何/物理 与 RL 学习 彻底分离。**
项目中大量内容在修士开环下是确定性可解析的（可用时间、走廊覆盖、能耗、窗口几何）。它们不得与 RL 训练代码缠绕。分离后这些模块可脱离 RL 单独测试，RL 未收敛时也能先验证它们正确。

**原则二：修士/博士 的差异收敛到单一注入点。**
"信息环是否闭合（观测结果是否回写地图）"这个唯一区别，必须做成一个**可注入的策略对象**（`BeliefUpdater`），而非散落各处的 `if master/doctor`。除该注入点外，任何位置出现修士/博士分支，A0 判违规。

---

## 2. 分层总览

依赖单向自底向上。每层只依赖其下层的**契约**，不依赖其实现。

```
第0层  config / geometry        ← 无依赖，最先交付
第1层  static environment       ← 依赖第0层
第2层  physics(2a) + task(2b)   ← 依赖第0/1层；2a 含 AUV 类
第3层  mdp (Gymnasium env)      ← 依赖第0/1/2层
第4层  policy (CleanRL)         ← 依赖第3层
第5层  baselines / eval（旁路） ← 仅依赖第0/1/2层，可与第3/4层并行
```

---

## 3. 建议文件树

```
msim/
├── config/
│   └── config.py          # [第0层][A2] 世界尺度/分辨率/坐标系/全部超参（dataclass）
│
├── geometry/
│   ├── grid.py            # [第0层][A2] 栅格↔世界坐标双向变换；粗网格↔细网格
│   └── window.py          # [第1层][A2] 地图窗口生成器（滑动 mask）
│
├── env_static/
│   ├── field.py           # [第1层][A2] 概率场生成（带簇合成 / 加载真实 map）
│   └── leader_path.py     # [第1层][A2] lawnmower 轨迹生成（外生预设）
│
├── physics/                          # ===== 第2a层 [A3]：AUV 类所在 =====
│   ├── AUV_dynamics.py        # AUV 基类（集成根目录上传的 AUV 类）
│   ├── leader.py          # Leader 特化
│   ├── follower.py        # Follower 特化（相机传感器模型 p_d / p_fa / footprint）
│   └── rollout.py         # 窄接口：waypoint 序列 → (轨迹, 能耗, 用时)
│                          #   ⚠ 最终形状由 A0 对照根目录 AUV 类敲定
│
├── task/                             # ===== 第2b层 [A4]：任务抽象 =====
│   ├── corridor.py        # 走廊覆盖（footprint×0.9 扩带 → 栅格覆盖格集）
│   ├── value.py           # 熵降/价值 + "已计划覆盖"场 + 不重复计数
│   ├── belief.py          # 【信息回写开关】BeliefUpdater 接口
│   │                      #   NullUpdater(修士) / BayesianUpdater(博士)
│   └── energy_balance.py  # 跨 Follower 能耗均衡惩罚（min-max / 离散度）
│
├── mdp/                              # ===== 第3层 [A5]：Gymnasium env =====
│   ├── coop_survey_env.py # 严格 Gymnasium 接口；事件驱动 step；one-at-a-time
│   ├── state.py           # state 组装（窗口概率场降采样 + Follower 状态归一化）
│   └── action.py          # 粗网格动作 → 映射模块（纯几何）→ waypoint → 承诺队列
│
├── policy/                          # ===== 第4层 [A5]：CleanRL 风格 =====
│   ├── network.py         # actor-critic（规模受 Pi 5 推理约束）
│   └── train_ppo.py       # CleanRL 单文件式训练循环
│
├── baselines/                       # ===== 第5层旁路 [A6]：不依赖 RL =====
│   ├── ortools_vrp.py     # min-max VRP 强基线（措辞：近似最优）
│   ├── exact_milp.py      # 小实例精确解（真 ground truth）
│   └── greedy.py          # 朴素贪心基线
│
├── eval/
│   ├── metrics.py         # [A6] max / (CV 或 Jain) / 总能耗 / 推理时间
│   └── runner.py          # [A6] 消融 harness：balance on/off
│
├── contracts/             # [A0] 冻结的接口契约（签名+类型+docstring，无实现）
├── tests/                 # [A1] 按模块的验收测试
└── <AUV 类定义文件>        # 用户置于根目录，A0 波次0 读取
```

---

## 4. 关键设计意图（A0 调整文件粒度时不可丢失的语义）

文件可以拆合、改名，但以下语义必须在最终结构中有明确归属：

### 4.1 第2层必须内分 2a / 2b 两子层 —— 时间尺度隔离
- **2a 物理仿真层（AUV 类）**：连续时间、细步长（dt≈0.01~0.1 s），积分运动学/流体力学。
- **2b 任务抽象层**：走廊覆盖、熵降、价值、开关，**事件驱动、大跨度**（一个 RL step 可对应 Follower 走完整段 waypoint，几十秒）。
- 两者时间尺度差几百倍，**必须通过 `rollout` 窄接口隔离**：2b 只向 2a 索要"给定起点+waypoint 序列 → (轨迹, 能耗, 用时)"，不碰积分细节。
- 红利：① AUV 类换更精细流体力学模型，2b 与 RL 不动；② Level 1（长度+转弯）与 Level 2（流体力学积分）只是 `rollout` 的两种实现，非两套代码；③ 第②层真机复现时，可用真机数据替换 2a 输出，2b 照常工作。

### 4.2 信息回写开关（belief.py）的形态
- `BeliefUpdater` 为抽象基类，`value.py` 持有其一个实例。
- 每次走廊结算后调 `updater.update(...)`：
  - 修士注入 `NullUpdater` → 空操作，地图逐位不变（开环性）。
  - 博士注入 `BayesianUpdater` → 后验回写。
- **全代码库"修士 vs 博士"差异收敛于此一个注入点。**

### 4.3 事件驱动 ↔ 标准 Gymnasium 适配
- `step(action)` 的"一步" = 为当前空闲 Follower 做一次决策。
- env 内部维护事件队列（谁先到点/窗口推进谁先触发），step 推进到下一个需决策的事件，组装该 Follower 的 obs。
- 对外严格返回 Gym 五元组 `(obs, reward, terminated, truncated, info)`，CleanRL 无需知道内部是事件驱动。
- 忙着执行的 Follower 当拍不参与决策（天然处理"暂时不可用"）。

### 4.4 承诺队列归属
- 住在 **mdp 层（任务/调度状态）**，**不在 AUV 类**。AUV 类只负责"给一段 waypoint → 返回代价"，不记"还欠几段"。

### 4.5 动作映射为纯几何
- `action.py` 的映射模块仅把粗网格 cell 索引 → 该 cell 中心的真实世界坐标 waypoint。
- **不做**可达性检查、避障、吸附等任何额外规划。

### 4.6 价值记账的两条不变量
- **不重复计数**：同一格被任一 Follower 走廊重复覆盖，期望熵降只计一次（靠"已计划覆盖"场，只记是否拍过、不记测量结果 → 不破开环）。
- **走廊宽度 = footprint 横向宽度 × 0.9**（保守冗余，宁可少算不虚报）。

---

## 5. 与 harness 波次的对应

| 波次 | 涉及目录 | 负责 agent |
|---|---|---|
| 0（串行） | `contracts/`、根目录 AUV 类、占位骨架 | A0 |
| 1（并行） | `config/ geometry/ env_static/`（A2）、`task/`（A4）、`baselines/ eval/`（A6）、`physics/`（A3）、`tests/`（A1） | A2,A3,A4,A6 + A1 |
| 2（集成） | `mdp/ policy/`（A5）、`physics/` 真实 AUV 类替换、端到端集成 | A5,A3,A0 |

---

## 6. 给 A0 的明确指示

1. 本蓝图是**起点**，非终点。读取根目录 AUV 类后，据其暴露的方法/状态敲定 `physics/` 与 `rollout` 的实际划分。
2. 可调整：目录层级、文件拆合、命名。**不可丢弃**：§1 两原则、§4 六条设计意图的语义归属。
3. 产出一份**敲定后的最终文件结构**，写入 `contracts/`（或项目 README），作为冻结版本广播全团队。
4. 广播后，建造 agent 以你的冻结版为唯一依据；本蓝图退役为历史参考。
