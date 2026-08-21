# contracts/ — 冻结接口契约(v1.0 · FROZEN 2026-06-11;v1.0.2 口径澄清 2026-06-12)

> **状态:已冻结 v1.0(2026-06-11,经人类批准)。** 经 A0 与人类逐条裁决(见 `DECISIONS.md`)产出并冻结。
> 建造 agent(A2~A6)以本目录为**唯一依据**开工;任何接口/签名改动须经 **A0** 走变更流程并广播,
> 禁止自行扩展(纪律2)。运行环境 = `auv_py310`;NED 朝向 = North 200m 长边。
>
> **波次1 进度(2026-06-12):L0/L1/L2a/L2b/L5 全部实现完成,A0 独立验证 `pytest tests/` = 91 passed / 0 failed、零跨层反向 import。波次2(A5:L3/L4 MDP+策略)依赖已满足,待开。** 波次1 口径澄清见 `DECISIONS.md` §v1.0.1(Level1 转角口径)/§v1.0.2(import 路径、粗↔细可逆、corridor 中心法+butt cap、Bayesian 占位、balance 空序列、基线 path_cost)。看板见 `WAVE_BOARD.md`。

本目录是全项目唯一接口基准。所有跨 agent 的函数签名、数据结构(state dict 键、动作编码、
rollout 返回元组等)在此定死:**只有签名/类型/docstring,行为方法体 `raise NotImplementedError`**
(纯数据派生属性如 `WorldConfig.nx`、`CameraSensorModel.corridor_width_m`、`FollowerCommitment.is_busy`
直接给实现,不算业务逻辑)。唯一有权改本目录的角色 = **A0**。

## 文件
| 文件 | 层 | 内容 |
|---|---|---|
| `types.py` | — | 公共类型别名 + **NED 坐标系/栅格索引/数组轴序的唯一真相源** |
| `config.py` | L0 | `SimConfig` 及子配置;`apply_overrides`(异构注入) |
| `geometry.py` | L0/L1 | `GridProtocol`、`WindowRegion`、`get_window`、`coarse_grid_shape`、`coarse_index_to_world` |
| `physics.py` | L2a | `RolloutEngine`/`RolloutResult`、`CameraSensorModel`、`LeaderActorProtocol`、`FollowerProtocol` |
| `task.py` | L2b | `corridor_cells`、`expected_entropy_reduction`、`mark_planned`、`BeliefUpdater`、`balance_penalty` |
| `mdp.py` | L3 | `FollowerCommitment` + 解析量、`assemble_obs`、`decode_action`、`CoopSurveyEnvProtocol` |

## 关键接口清单(对应 harness §6)
| 接口 | 契约位置 | 关键约束 |
|---|---|---|
| 坐标变换 | `geometry.GridProtocol` | 往返无损;`(ix,iy)`/`field[ix,iy]`/NED 见 `types.py` |
| 窗口 | `geometry.get_window` | 后方、单调推进;`WindowRegion` 纯数据 |
| 粗网格 | `geometry.coarse_grid_shape` | 由 window÷footprint 推导,footprint 覆盖整格(Q5) |
| 物理 rollout | `physics.RolloutEngine.execute` | `(start_pose, waypoints)→RolloutResult`;Level1(训练)/Level2(验证) |
| 走廊覆盖 | `task.corridor_cells` | footprint×0.9;与解析解一致 |
| 价值 | `task.expected_entropy_reduction` | 不重复计数(planned_coverage);bit |
| 信息回写开关 | `task.BeliefUpdater` | Null(修士空操作)/Bayesian(博士);**唯一注入点** |
| 状态 | `mdp.assemble_obs` | Dict{map,vec};归一化 |
| 动作 | `mdp.decode_action` | 纯几何,无避障/吸附 |
| 承诺队列 | `mdp.FollowerCommitment` | 住 mdp 层;解析确定量(非估计) |
| 环境 | `mdp.CoopSurveyEnvProtocol` | 标准 Gym 五元组;env_checker 通过 |

## 同时维护的文档
- `DECISIONS.md` — 波次0 全部裁决记录(Q1~Q19 + 人类补充),契约的 rationale。
- `DEPENDENCY_GRAPH.md` — 谁依赖谁、哪些可并行。
- `WAVE_BOARD.md` — 波次0/1/2 模块归属与状态看板。
