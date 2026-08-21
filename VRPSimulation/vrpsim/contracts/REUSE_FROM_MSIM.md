# 允许从 msim 复用的符号（白名单，A0 维护）

依赖方向 **单向**：`VRPSimulation → msim`。msim 永不 import 本包。
不在本表内的 msim 符号，用之前须经 A0 裁决并更新本表 —— 否则等于绕过契约层建立隐式耦合。

同样的硬纪律：**本包不修改 msim 的任何文件**（CLAUDE.md 硬纪律 5，文件归属唯一）。
若发现 msim 某处不够用，提交 A0 裁决，由 msim 的所属 agent 改，不许在这边打补丁。

---

## 已在用（本步）

| 符号 | 来源 | 用途 |
|---|---|---|
| `WorldConfig` | `msim.contracts.config` | 世界尺度与栅格分辨率；`mothra_world_config()` 只是给它填真实场地的数 |
| `Field`, `FLOAT`, `Waypoint`, `Position`, `Cell` | `msim.contracts.types` | 类型别名 + **坐标系唯一真相源**（x=North / y=East / field[ix,iy]） |
| `GroundTruthInstance` | `msim.contracts.env_static` | `MothraWorld.to_gt_instance()` 的返回类型，用于对接既有链路 |
| `Grid` | `msim.geometry.grid` | 世界坐标 ↔ cell 双向变换（floor 量化、越界 clip、cell 中心） |
| `field_from_targets` | `msim.env_static.field` | 目標分布確率マップ 的**默认**构建路径（各向同性高斯 KDE + max 归一化） |

`field_from_targets` 是刻意复用而非重写：真实场与合成场走同一条代码路径，
概率语义逐位同源，两者上的消融结果才可比。
带权变体（`weight_mode="height"`）在 `vrpsim/field.py` 本地实现，
除权重外与上游逐字同构（含归一化口径）。

---

## 已在用（D7/D8 窗口层）

| 符号 | 来源 | 用途 |
|---|---|---|
| `WindowConfig`, `SensorConfig` | `msim.contracts.config` | 窗口与相机参数；`mothra_window_config()` 只是填 100×100 |
| `WindowRegion` | `msim.contracts.geometry` | 窗口纯数据类（纪律 2：不改该类） |
| `get_window`, `window_contains_world`, `window_cells` | `msim.geometry.window` | **滑动**窗口的隶属判定与栅格覆盖，直接用 |
| `lawnmower_waypoints` | `msim.env_static.leader_path` | Leader 测线；lane_spacing == East 跨度 ⇒ 单条测线 |

`msim.geometry.window` 里的 `get_window` 仍是**滑动**实现（§v1.1.5 的"离散固定块"改在
mdp/env 层，不在这个模块），正好就是 D8 要的语义，因此原样复用。

⚠ **不用** `coarse_grid_shape` / `coarse_index_to_world` —— 那是 RL 的粗网格动作空间，
本仿真改走分区域 VRP，不需要。

---

## 预期在下一步（任务调度层）复用

| 符号 | 来源 | 备注 |
|---|---|---|
| `RolloutResult`, `Level2RolloutEngine` | `msim.physics.rollout` | waypoint 序列 → (轨迹, 能耗, 用时) |
| `Follower`, `LeaderActor` | `msim.physics.*` | 相机传感器模型 p_d / p_fa / footprint |
| `corridor_cells` | `msim.task.corridor` | 走廊覆盖格集 |
| `max_energy`, `routes_to_energies`, `total_energy` | `msim.eval.metrics` | 评估指标 |

---

## 刻意**没有**复用（附理由）

| 符号 | 为什么没用 |
|---|---|
| `solve_minmax_vrp`（`msim.baselines.ortools_vrp`） | 它的车辆起点来自 `greedy._start_poses(cfg)` —— 一个由 `SimConfig` 推出的**固定布阵**（`x=0, y=(i+0.5)/n·y_max`），喂不进 Follower 的实时位置。本任务每次重解都要从两台的**当前**位置起算。故在 `vrpsim/planner.py` 按**同一套建模**（开放路径、`SetGlobalSpanCostCoefficient=1000`、距离整数化 ×100）重写了一个接受显式起点的版本，解质量口径一致，只有起点来源不同。**没有修改 msim 任何文件**（CLAUDE.md 硬纪律 5）。 |
| `RolloutEngine.execute`（`msim.physics.rollout`） | 批处理接口：一次吃完整条 waypoint 序列返回 (轨迹, 能耗, 用时)。本任务要在序列执行**中途**接受新序列（D8 覆盖语义），还要逐时刻记录位置（规格 §7），对不上。故 `vrpsim/agents.py` 实现了最小的定步长推进器。要能耗时可把实际轨迹事后交给 `msim.eval.metrics.path_cost`，口径不变。 |
| `coarse_grid_shape` / `coarse_index_to_world` | RL 的粗网格动作空间，本仿真走分区域 VRP，不需要。 |

---

## 明确**不**复用

| 符号 | 原因 |
|---|---|
| `make_clustered_instance` | 那是合成簇点过程；本仿真的目标点来自真实数据，不生成 |
| `default_instance`（`msim.eval.runner`） | 同上，且它内含 RNG |
| `msim.config.config.apply_overrides` | 本包配置面不同，用自己的 dataclass 即可 |
