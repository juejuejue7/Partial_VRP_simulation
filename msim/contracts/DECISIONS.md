# 波次0 契约裁决记录(DECISIONS)

> 本文件存档波次0 全部接口裁决及其 rationale,是 `contracts/` 的依据。
> 经人类逐条确认(2026-06-11 会话)。任何后续改动须经 A0 并在此追加记录。

## 人类补充约束(全局)
- AUV 类的能耗计算方法原为"AUV 运动节能"项目开发;本项目**采用为 Level 2 验证旁路**,训练不直接依赖。

## 逐条裁决

| # | 议题 | 冻结决定 |
|---|---|---|
| Q1 | rollout 返回形状 | `RolloutResult(energy_J, duration_s, end_pose, per_wp_energy_J, per_wp_duration_s, trajectory=None)`;`trajectory` 默认关,可视化时开启。 |
| Q2 | 能耗源 | 训练用 **Level 1**(路径长度+转弯惩罚);**Level 2**(AUV 类积分)保留接口,仅验证分析,不进 reward。 |
| Q3 | 时间口径 | **撤销"转弯不计时间"**(原 §10 笔误)。实际 AUV 运动按 AUV 类模式,**物理层计入转弯时间**;Level 1 用时含转弯时间近似(`Σ\|Δψ\|/nominal_yaw_rate`),仍为承诺队列解析确定量。 |
| Q4 | obs 结构 | **Dict** = `{"map": float32(Hc,Wc), "vec": float32(10,)}`;N=2 用 self/teammate 槽位隐式编码。 |
| Q5 | 动作/粗网格 | `Discrete(Hc*Wc)` + 纯几何 decode;粗网格形状**由 window÷footprint 推导**,保证相机 footprint 覆盖整格(`Wc=ceil(look_back/footprint_along)`,`Hc=ceil(width/footprint_lateral)`)。 |
| Q6 | 熵单位 + 边界 | 熵单位 **bit(log2)**;`planned_coverage`(用于 reward 不重复计数,修士也更新)与 `belief`(最终概率图,修士不更新)**分离**。 |
| Q7 | 坐标系/索引 | 世界坐标 = **NED 水平面**,与 AUV 类 `eta=[x,y,psi]` 一致(x=North、y=East、ψ 由 North 转向 East),离散化成网格;cell `(ix,iy)`、数组 `field[ix,iy]`(轴序对齐坐标序)。 |
| Q8 | 包名/归属 | 顶包 `msim/`;根目录 AUV 类移入 `msim/physics/auv_dynamics.py`(A3 拥有)。 |
| Q16 | Leader 运动 | **不做纯几何**:用 `HeterogeneousAUV("leader")` 实例 + lawnmower waypoint 经**动力学**推进、维护真实位姿;能耗累计但**不进 reward**。 |
| Q19 | 复现 | groundtruth 概率场需 `save_field/load_field`(带 seed/簇参数 metadata),供对比验证。 |
| Q9–Q15,Q17–Q18,Q20 | 见 README 清单 | 按 A0 推荐冻结(WindowRegion 纯数据 + 独立函数、field 为 ndarray、corridor 返回 set、min-max 均衡、承诺队列 list+cursor、CameraSensorModel dataclass、纯 dataclass config 无 YAML、dtype 约定、env 终止条件)。 |

## 由裁决产生的连带契约调整(人类已知会)
1. **Q3**:`RolloutConfig` 增 `nominal_yaw_rate_rps`;Level 1 `duration_s` 公式含转弯时间近似;`resolve_available_time` 同口径。
2. **Q5**:新增 `coarse_grid_shape(window,sensor)`;`action_space` 的 n 在 env 构造时算出(非硬编码)。
3. **Q7**:数组轴序定为 `field[ix,iy]`(与图像 imshow 习惯相反,可视化自行转置)。
4. **Q16**:env 采用**连续全局时钟**;`LeaderActorProtocol` 独立于 `RolloutEngine`(Leader 持续推进,不"索取代价")。

## v1.0.1 口径澄清(A0,2026-06-12;A3 提请,签名不变)

- **§v1.0.1 Level 1 转角 Σ|Δψ| 起点口径**:✅ **选 B —— 只累计队列内相邻段间转角,忽略首段转入角**。
  - 议题:`RolloutEngine.execute(start_pose, ...)` 拿得到 `start_pose.psi`,可算"当前朝向→首段"的转入角;
    而 `resolve_available_time(current_pos: Position, ...)` 只有位置、无朝向,算不出该转入角。契约要求二者"同口径"。
  - 裁决:**口径以信息量最小一方(`resolve_available_time`)为准**。两者 Σ|Δψ| 均定义为:
    设位置序列 p, w_0..w_{n-1}(execute 取 p=start_pose.position;resolve 取 p=current_pos、w=队列剩余点),
    段方向 s_0=dir(p→w_0)、s_k=dir(w_{k-1}→w_k),则
    `Σ|Δψ| = Σ_{k=1}^{n-1} |wrap(s_k − s_{k-1})|`(n−1 项;n=1 时为 0)。**不含** start_pose.psi→s_0 的首段转入角。
  - 影响:Level 1 `duration_s` 与 `energy_J` 的转角项均用此 Σ|Δψ|;`start_pose.psi` 仅供 Level 2 积分初值与 `end_pose` 朝向输出。
  - 理由:选 A 需把 `resolve_available_time` 签名由 `Position` 改 `Pose`,属破坏冻结的接口变更;选 B 是两条已冻结条款("同口径"+ Position-only)的唯一自洽解,**无签名变更**。
  - 落地:已收紧 `contracts/physics.py::RolloutEngine.execute` 与 `contracts/mdp.py::resolve_available_time` 的 docstring(行为规格,非签名)。A1 的 rollout/commitment 测试按 B 断言;A3/A5 实现按 B。

## v1.0.2 集成口径裁决(A0,2026-06-12;A2/A4/A6 提请,无既有签名变更)

- **实现 import 路径规则(全员)**:消费者(A1 测试 / A5)从 `msim.<层>.<模块>` import **实现**;`contracts/` 只供契约签名/类型/Protocol,**不从 contracts import 实现**。
  - `get_window` / `window_contains_world` / `window_contains_cell` / `window_cells` / `coarse_grid_shape` / `coarse_index_to_world` 的实现落 `msim/geometry/window.py`(A2 拥有),消费者从此 import。`coarse_*` 归 A2、住 window.py。
  - 与既有骨架一致(`Grid`∈`msim.geometry.grid`、`NullUpdater`∈`msim.task.belief`)。
- **§4.2 "粗↔细可逆" 判据(A2-Q2)**:✅ **不在 `GridProtocol` 新增 coarse_to_fine/fine_to_coarse(零契约扩展)**。判据 = `coarse_index_to_world` 在 `[0,Hc*Wc)` 上是到"粗格中心"的**解析双射**:A1 **独立**解析反推 `(row,col)=(k//Wc, k%Wc)` 与世界中心比对、recovers k(QA 不复用 A2 反函数,避免共因 bug)。细网格往返 `world_to_cell∘cell_to_world==id` 为另一半。
- **corridor 覆盖判据(A4-Q1)**:✅ **中心法**(cell 中心 ∈ 走廊矩形),与 `window_contains_cell` 中心法统一 → 全项目单一覆盖判据。端部 **butt cap(平头,无端帽)**,与 `corridor_cells` 契约"线段扩 width_m 带"一致(契约只传横向 width_m,不含 footprint_along 端延伸)。
  - **形式化(防 round cap 误读;A4-2/A1-2 冲突已据此裁,2026-06-12)**:设线段 P0→P1,单位方向 t̂,长 L。cell 中心 c 覆盖 ⟺ 投影 `s=(c−P0)·t̂ ∈ [0,L]` **且** 垂向偏移 `|(c−P0)·n̂| ≤ width/2`。**不**把投影裁剪到 [0,1]。**禁止 round cap / 胶囊**(即"点到线段距离 ≤ width/2"——它在两端各加半径 width/2 的圆帽,虚增 ~width/2 覆盖、高估 value,违"不 overclaim")。退化段(L<eps)= 以该点为中心、边长 width 的方格 ∩ 栅格。
- **belief BayesianUpdater(A4-Q2)**:✅ 本波次**仅占位**——符合 `BeliefUpdater` Protocol,`update` 体 `raise NotImplementedError`,标注 [博士]。修士活路径只经 `NullUpdater`;注入点 = 对象切换,非 if master/doctor 分支。修士验收只测 NullUpdater 开环。
- **balance_penalty 空序列(A4-Q3)**:✅ 返回 **0.0**(防 `max([])` 崩;语义中性:无 Follower 无失衡)。
- **基线代价自包含(A6-1/2)**:✅ baselines 内自实现 `path_cost(path, RolloutConfig)` + `routes_to_energies(routes, cfg)`:只读 `RolloutConfig` 标量、**不 import A3 实现**、转角口径遵 **§v1.0.1**(只计相邻段间转角);**不改任何契约签名**(`solve_*` 仍 `List[List[Waypoint]]`,per-follower 能耗由 baselines 内 helper 提供)。
  - **集成门(波次2,A0)**:断言 `A6.path_cost` 与 `A3.Level1RolloutEngine` 在同 routes 上数值一致(单一真相源,防基线/ RL 代价漂移影响对比可信度);必要时统一切到 A3 引擎。

## v1.0.3 ground-truth 耦合实例生成(A0,2026-06-12;人类裁定;**新增契约 contracts/env_static.py**)

- **议题**:发现先验概率场(`env_static.field.make_clustered_field`)与目标点(`eval.runner.default_targets`)是**两条独立生成流**——前者撒高斯簇但**丢弃簇中心**,后者**全场均匀随机**(且违 §13"避免均匀随机"),二者无共享 ground truth。先验场不来自真实目标分布。
- **人类裁定**:
  1. **耦合方向**:✅ **目标点 → KDE 派生场**。真实目标点(簇点过程生成)= **绝对 ground truth**(所有验证/下游数据的参考);先验场 = 对目标点做各向同性高斯 KDE + 归一化 [0,1]。
  2. **先验不确定性**:✅ **模糊信念**,先验∈(0,1),熵降(信息价值)非平凡、可比,契合"informative 价值"卖点与开环设定。
     - **σ 标定(2026-06-13 修订,人类据对照图裁定)**:原"σ≈簇内散布(=12m)"在场上过宽(先验>0.5 占 52%)。改为 **σ = 0.5×footprint_lateral_m = 3m**(与 footprint 挂钩、与场尺寸无关;先验>0.5 占 7%,每目标清晰光晕峰)。落点:仅 `eval.runner.default_instance` 显式传 `bandwidth_m=3m`;`contracts/env_static.make_clustered_instance` 的通用默认 `None→intra_spread` 不变(A1 契约函数用例依赖之)。对照工具:`examples/prior_bandwidth_compare.py`。
- **契约**:新增 `contracts/env_static.py`:`GroundTruthInstance(targets, field, cluster_centers)`、`field_from_targets(targets, world, *, bandwidth_m)`、`make_clustered_instance(world, rng, *, n_clusters, targets_per_cluster, intra_spread_m, bandwidth_m=None)`、`save_instance/load_instance`。RL 用 `field`、baseline 用**同一组** `targets` → 完全耦合。
- **派工**:A2 实现 `env_static`(field_from_targets / make_clustered_instance / save·load_instance;`make_clustered_field` 收敛为 `field_from_targets` 之上的便捷封装或保留);A6 把 `default_targets`/runner 改为消费 `make_clustered_instance(...).targets`(全局均匀随机退场);A1 加耦合验收(场峰值与目标簇共位、目标处均值≫非目标、双复现)。A0 验证后更新验收脚本展示耦合实例。
- **验收不变量**:① field = targets 的归一化高斯KDE;② mean(field@targets) ≫ mean(field@随机非目标);③ targets 簇结构(非均匀);④ 同 seed+参数 → (targets,field) 逐位复现。

## v1.0.4 baseline 输入改为「Leader 扫描概率图的派生检测点」(A0,2026-06-15;人类裁定;**新增契约 `env_static.detections_from_field`**)

- **议题**:发现端到端基线规划把 `inst.targets`(ground-truth 离散目标点)**直接**喂给 `solve_minmax_vrp`(示范脚本 `examples/mission_baseline_survey.py:103`,正式管线 `eval.runner.default_targets→evaluate_solver`)——等于让 Follower 规划器**预知每个真实目标精确位置**,是 **oracle/上界基线**,不符合真实探查。真实场景里规划器只能拿 **Leader 扫出来的概率图**(已揭示的 `field`,即「目標分布確率マップ」)。又因三个 baseline 都**只把 `targets` 当纯坐标消费、不读身份**(greedy:50 / ortools_vrp:51 / exact_milp:57),且 min-max VRP 本质需**离散访问点**→「直接喂概率图」不可行,必须先把概率图**处理成离散候选检测点**。
- **人类裁定**:
  1. **提取方法**:✅ **局部极大值 + NMS**(每个可分辨峰=一个候选检测点;候选数≈真实目标数,误差形态=离散/模糊偏移+近邻合并+阈值漏弱峰)。**确定性、无 RNG**。否决「阈值+连通域质心」(过度合并同簇目标)。
  2. **范围**:✅ **示范脚本 + 新契约原语 + 评估管线(A6),并保留 oracle 参照**(realistic 与 oracle 两条都报),为将来 learned-planner vs baseline 公平对比铺路。
- **契约**:新增 `contracts/env_static.py:detections_from_field(field, world, *, threshold_rel=0.3, min_separation_m) -> List[Waypoint]` —— `field_from_targets` 的**有损逆过程**。参数标定(无 magic number):`min_separation_m = 0.5*footprint_lateral_m = 3m`(= 先验模糊 σ);`threshold_rel = 0.3`。**返回点是【估计/检测】非 GT**(硬纪律#4:命名/注释须标明);GT 仅留作评估侧 TP/FN/FP 的绝对参考。归属:实现住 `msim/env_static/field.py`(A2,纯 numpy 勿引 scipy)。
- **派工**:A2 实现 `detections_from_field`(8 邻域局部极大 + 贪心 NMS,复用 field.py cell 中心网格);A6 在 `eval/runner.py` 加 `default_detections(...)` + `run_ablation(target_source="scanned_map"|"oracle_gt")`(默认 `scanned_map`=现实基线喂检测点;`oracle_gt`=GT 上界参照;`default_targets` 保持=GT 向后兼容,`evaluate_solver` 保持通用);A1 加检测原语验收 + eval realistic 路径测试。A0 改 `mission_baseline_survey.py` 集成(belief=where(revealed,field,0)→detections→VRP)+ 出「检测点 vs GT」对照图,独立验证。
- **合规说明(防硬纪律#3 误读)**:`target_source` 是**评估输入选择器**(oracle vs 现实基线,**二者皆修士基线**),**不是 master/doctor 分支、也不在活策略路径**。learned-planner(波次2)消费同一张 `field`(CNN),与基线消费其峰值检测点,是**表示差异下的公平对比**,非分支。
- **验收不变量**:① 确定性(同 field+参数→逐位复现,无 RNG);② 检测点严格在场内;③ 召回:多数 GT 目标在某检测点 min_separation_m(+裕量)内(近似覆盖,**非精确相等**);④ 精度:每个检测点处 field≥threshold_rel·max;⑤ 有损性:单孤立目标→恰 1 点且在其 res 内,两目标间距<min_separation_m→合并为 1;⑥ 单调:threshold_rel 升高→点数不增;⑦ 空场→[]。

## v1.0.5 候选点选取改为 A′ 混合(峰值锚点+覆盖补点)(A0,2026-06-16;人类裁定;**新增契约 `env_static.observation_candidates_from_field`**)

- **议题**:§v1.0.4 的纯峰值 `detections_from_field` 有**本质缺陷**——σ=3m 的 KDE 把多个近邻目标糊成一**整块**高概率区,而单峰 blob 只有**一个** 8 邻域局部极大 → 一整块被当**单点**、簇内目标数被吞掉(seed=24:9 GT 仅 5 点、recall@5m=0.67、端到端 TP 44%)。NMS 只会更糟,调阈值无效(只读扫描证实数量由 σ 下近邻几何合并主导)。
- **四方案只读原型对比(seed 24/7/3/0)**:
  - 纯覆盖采样(阈值+footprint 栅格铺点):recall→1.0,但**抛弃峰值思想**。
  - 已知 σ 去模糊(Richardson-Lucy + 峰值):seed24 仅 0.89,**否决**。裁决理由:我们的合成场**无噪声 + 生成核 σ 已知**,用同一 σ 反卷积≈把 GT 解回来 = **模型反演泄漏**(不读 `inst.targets`,但靠"恰好知道 GT 如何变成场 + 无噪声"反推 GT),**部分把已删的 oracle 请回来**,且边界处易臆造幻影峰。违"不现实/近作弊"。
  - 连通域面积+kmeans:recall 0.33~0.89 **不稳健,否决**。
  - **A′ 混合(峰值锚点 + 覆盖半径补点)**:峰值**逐点原样保留** + 大块按面积补点;recall@5m 全场 1.0、点数自适应(7~10)、**不碰生成核 σ(无作弊)**、无边界依赖。seed=0(峰值已够)补 0 点 → 自适应退化回纯峰值。
- **人类裁定**:✅ **采用 A′**。理由:既留住"基于峰值估计目标点"的思想(峰值是骨架、逐点保留),又修好"一整块被当单点";比去模糊**更纯地保留峰值**(峰值跑在原始 field 上、一字不改)且**无作弊嫌疑**。补点放法 **v1=栅格铺点**(易冻契约/易验收;后续可升级最远点采样)。
- **契约**:新增 `observation_candidates_from_field(field, world, *, threshold_rel=0.3, peak_min_separation_m, fill_radius_m) -> List[Waypoint]`,以 `detections_from_field` 为子步。**参数挂物理(无 magic number)**:`peak_min_separation_m=0.5*footprint_lateral_m=3m`(=σ)、`fill_radius_m=0.9*footprint_lateral_m=5.4m`(=相机走廊宽;点距≤R 即可走廊扫遍)。**保留 `detections_from_field` 不动**(既是子步,也留作论文「峰值 vs 混合」对照基线)。返回点是**估计/候选观测点,非 GT**。
- **派工**:A2 实现 `observation_candidates_from_field`(纯 numpy,复用 `detections_from_field`+栅格补点+贪心 dedup);A6 把 `eval.runner.default_detections` 改调新函数(传上述物理参数),`run_ablation(target_source)` 语义不变(`scanned_map` 现走 A′);A1 加验收(峰值保留⊇、自适应、覆盖改善、精度、确定性、边界、空场)。A0 改 `mission_baseline_survey.py`(峰值锚点 vs 覆盖补点分色出图)并独立验证。
- **验收不变量**:① 峰值保留:输出 ⊇ `detections_from_field` 同参输出(逐点);② 自适应:孤立目标 blob<R→输出==峰值,大簇→点数>峰值;③ 覆盖改善:耦合实例 recall@R ≥ 纯峰值,受控良分离场==1.0;④ 精度:每候选点 field≥threshold_rel·max;⑤ 确定性逐位复现、无 RNG;⑥ 点严格在场内;⑦ 空场→[]。
- **R 提升为可调试 config(2026-06-16 追补,人类要求)**:实测发现 `fill_radius_m` 是覆盖/成本权衡的关键旋钮(seed=24:R=5.4→端到端覆盖 5/9、~9 候选;R=2.7→8/9、~30 候选、路程+48%)。人类先裁定 R=5.4 默认,后改用 **R=2.7**(端到端覆盖优先)。把它**提升为可配置参数**:`contracts/config.py` 新增 `ObsCandidateConfig` + `SimConfig.obs_candidate` + `resolve_obs_candidate_params(cfg)`(None→挂 footprint:peak_sep=0.5·fl=3m、fill=0.9·fl=5.4m)。消费者(eval.runner.default_detections / mission)统一经 `resolve_obs_candidate_params` 取值,使 `cfg.obs_candidate.*` 成为调试**单一入口**。归属:config.py∈contracts(A0);eval 接入=A6;调参测试=A1。

## v1.0.6 峰值阈值与补点阈值拆开(A0,2026-06-16;人类要求)

- **议题**:§v1.0.5 中 A′ 的峰值锚点与补点**共用单一 `threshold_rel`**(圈定峰值候选格 + 圈定补点高概率区 M)。人类要求把二者拆开:峰值阈值不变、补点阈值单独设更严,**让补点只落在更确信的密集区/合并簇核心,不在弱晕区撒点**。
- **人类裁定**:✅ 峰值阈值保持 **0.3**;补点阈值新增、默认 **0.5**(约定 ≥ 峰值阈值)。
- **契约**:`observation_candidates_from_field` 新增形参 `fill_threshold_rel: float = 0.5`;`threshold_rel`(默认 0.3)语义收窄为**仅峰值锚点**阈值;补点高概率区改为 `M = {field >= fill_threshold_rel*max}`。`config.ObsCandidateConfig` 新增 `fill_threshold_rel=0.5`;`resolve_obs_candidate_params` 返回值由 3 元组 → **4 元组** `(threshold_rel, fill_threshold_rel, peak_min_separation_m, fill_radius_m)`。
- **派工**:A0 改契约/config/mission/文档;A2 改 `field.py` 实现(峰值子步用 threshold_rel、补点区 M 用 fill_threshold_rel);A6 改 `default_detections`(消费 4 元组、透传 fill_threshold_rel);A1 改/加测试(resolve 4 元组断言、补点阈值更严→补点数不增的判据)。
- **验收不变量**:① 峰值集合只随 threshold_rel 变、与 fill_threshold_rel 无关;② 提高 fill_threshold_rel → 补点数**不增**(单调);③ 其余 §v1.0.5 不变量保持(峰值保留⊇、确定性、边界、空场)。

## v1.0.7 走廊宽口径由 ×0.9 改为 ×1.0(相机探测范围与 grid cell 对齐)(A0,2026-06-19;人类要求)

- **议题**:人类按"实际探测精度"重设传感器几何——相机探测范围 = 以 Follower 为重心的 **2m×2m** 方形,且 grid cell 同设 2m×2m(`WorldConfig.res_m=2.0`)。原 `CameraSensorModel.corridor_width_m = footprint_lateral × 0.9`(§v1.0.2 的"保守冗余,宁可少算不虚报")会使 footprint=2m 时走廊宽=1.8m,**与 cell 不对齐**。
- **人类裁定**(走廊宽口径三选一,选"正好 2.0m 对齐 cell"):✅ 取消 ×0.9 保守扩带,改 **走廊宽 = footprint_lateral × 1.0**。footprint=2 → 走廊宽=2.0m=1 个 grid cell,严格对齐。**显式推翻 §v1.0.2 的 ×0.9 裁定**(footprint×0.9 扩带口径作废;corridor.py 的中心法+butt cap 覆盖算法本身不变,仅 width 取值口径变)。
- **契约**:`CameraSensorModel.corridor_width_m` property 由 `return footprint_lateral_m * 0.9` 改为 `return footprint_lateral_m`(`contracts/physics.py`)。默认 footprint_lateral=6 → 默认走廊宽由 5.4 变 **6.0**(全局生效,非仅 mission)。
- **footprint 值作用域**:2m×2m 的 footprint 值只在 mission 脚本 `cfg.sensor=SensorConfig(footprint_along_m=2.0, footprint_lateral_m=2.0)` 局部设置,**不改全局 SensorConfig 默认**(避免污染 obs_candidate 默认 R=0.9×footprint_lateral=5.4 等基于 footprint=6 的断言)。`resolve_obs_candidate_params` 里 fill_radius 的 0.9 是**另一个独立 0.9**(候选补点半径,在 config.py),**未动**。
- **连带副作用(已知会)**:① mission 中 footprint_along=2 → Phase 3 FP 采样步长 `max(along,lateral)` 6→2,虚警检查变密;② 粗网格 `coarse_grid_shape` 形状随 footprint 变(波次2 A5 未开工,暂无影响)。
- **派工**:A0 改 `contracts/physics.py` + `examples/`(acceptance_demo 断言、mission cfg);A1 同步 `test_physics_follower_leader.py`(2 个 ×0.9 断言→×1.0)与 `test_task_corridor.py`(width 5.4→6.0 断言);A3 同步 `follower.py` 注释文字。
- **验收**:`pytest tests/` 全绿(其中 3 个 corridor_width 断言由 A1 同步);`acceptance_demo.py` 全过;mission 走廊宽=2.0m、Follower 覆盖以 2m cell 为粒度。

## v1.0.8 目标场尺度从相机 footprint 解耦(独立旋钮)(A0,2026-06-19;人类裁定)

- **议题**:§v1.0.7 改相机 footprint=2 后,人类发现"声呐揭示结果"也变了。根因:`eval/runner.py` 的 `default_instance`(生成被揭示的先验场)两个**场景尺度**借用了 sensor.footprint:`bandwidth_m=0.5×footprint_lateral`(先验 KDE σ)、`intra_spread_m=2×max(footprint)`(簇内散布)。改 footprint 连带重画整个 ground-truth 实例(目标点坐标 + 先验图;实测 seed=24:footprint 6→2 使 σ 3→1m、簇散布 12→4m、先验 >0.5 占场 21%→4%、目标点坐标全变)。揭示 mask(window)本身不依赖 footprint,变的是被揭示的 field 内容。
- **裁定**:✅ 解耦。目标分布(簇多大/多密)与先验图模糊度是**海底环境属性,与相机大小无关**;§v1.0.3 借 footprint 当尺度基准属便利选择(设计缺陷)。人类选"解耦 + 独立旋钮"。
- **契约**:新增 `config.TargetFieldConfig`(`bandwidth_m=3.0`、`intra_spread_m=12.0`、`targets_per_cluster=3`)+ `SimConfig.target_field`。**默认值 = 原 footprint=6 时的值** → 默认场景逐位不变(向后兼容)。
- **派工**:A0 加 config + 文档 + mission 注释;A6 改 `eval/runner.py`(`_default_cluster_params` 的 intra_spread/targets_per_cluster、`default_instance` 的 bandwidth 改用 cfg.target_field,删 footprint 依赖);A1 加 `test_target_field_decoupling.py`(解耦不变量 + 两旋钮生效 + 默认向后兼容)。
- **验收不变量**:① 改 cfg.sensor.footprint → default_instance 的 targets/field 逐位不变;② cfg.target_field.bandwidth_m / intra_spread_m 各自单调影响先验图扩散 / 簇散布;③ default_instance(SimConfig()) 与改动前等价;④ 现有测试零依赖该耦合(全部显式传 bandwidth/intra_spread),不破。
- **边界(未解耦的部分,有意保留)**:`resolve_obs_candidate_params` 里候选提取的 `peak_sep=0.5×footprint`、`fill_radius=0.9×footprint` **仍挂相机 footprint**——那是"候选航点选取/观测精度",本就该随观测分辨率走,不属本次"目标场生成"解耦范围。

## v1.0.9 实际执行轨迹贴合 + 覆盖评估改用实际轨迹走廊(A0,2026-06-19;人类裁定)

- **议题**:人类发现 mission fig2 的 Follower 实际轨迹(Level2 积分)与规划 waypoint 有出入。检查数据流(detections→VRP→execute)确认**传递无误**:VRP 用 `_start_poses` 作 depot 起点(= execute 起点)、`routes` 点 = detections 原值(只分配不改坐标)。出入出在**执行层**:`arrive_radius_m=1.5m` 是大场景容差,而精细场景候选点间距中位数仅 2.0m(实测 seed9:F0 17/19、F1 10/13 相邻段 < 2×arrive_r,容差/点距≈0.75)→ AUV 在 1.5m 圈内提前转向、密集点处"切角"偏离规划折线;叠加 `step_motion` 动力学转弯半径。
- **裁定**(人类选 1+3 同时处理):
  - ① **到点容差 arrive_radius 1.5→0.5m**(仅 mission 局部 `follower_overrides`,< 0.25×点距 → 实际轨迹贴合 waypoint;**大场景全局默认 1.5m 不动**,acceptance/测试不受影响)。
  - ② **覆盖评估改用实际轨迹走廊**:Phase 3 的 `coverage`/TP/FN/FP 由"规划 route 折线走廊"改为 **`res.trajectory`(Level2 实际走线)的走廊并集**(`_route_corridor_union`),更贴近"相机真正扫过哪里";先 execute 得 res 再算 coverage,无轨迹回退规划 route。
- **作用域**:均在 mission 集成脚本(examples/,A0),不涉及契约/builder 模块/测试。`record_trajectory=True` 复用上一步(为 fig2)所开。
- **验收**:mission exit=0;fig2 实际轨迹在密集簇区贴合候选点(可见绕点)。arrive_radius=0.5 实测对比:**规划 route 走廊覆盖 9/9、实际轨迹走廊覆盖 6/9**——差额 3 个(T2/T7/T8,均在 x≈35–39 密集簇)是 AUV 切角 + 0.5m 到点容差下"真正扫过的 2m 走廊"未盖住规划点 cell 的**执行损失**,正是 §v1.0.9 改用实际轨迹评估要暴露的真相(规划覆盖 ≠ 实际执行覆盖;此前误报 9/9 已纠正)。§v1.0.9 改动隔离 examples、**不引入**任何测试失败(当时 repo 另有 4 个预存失败源于 `ObsCandidateConfig` 默认值被改,与本补丁无关 → 已由 **§v1.0.10** 全局定稿 0.5/0.7 + 同步测试解决)。

## v1.0.10 候选提取阈值默认定稿(threshold 0.3→0.5、fill 0.5→0.7)(A0,2026-06-19;人类裁定全局定稿)

- **议题**:人类在调 mission 时直接改了 `contracts/config.py` 的 `ObsCandidateConfig` 默认阈值(threshold_rel 0.3→0.5、fill_threshold_rel 0.5→0.7),令 `test_obs_candidate_config` 的 4 个默认值断言失效。
- **裁定**:✅ 全局定稿 **(0.5, 0.7)**——峰值锚点与补点阈值都调严:峰值候选只取 field≥0.5×max、补点只落 field≥0.7×max → 候选更集中在高概率核心、更少弱晕区杂点。约定 fill≥threshold 仍成立。**取代 §v1.0.6 的 (0.3, 0.5)**。
- **契约**:`ObsCandidateConfig.threshold_rel` 默认 0.3→0.5、`fill_threshold_rel` 默认 0.5→0.7(全局,影响所有 `default_detections` / mission)。peak_sep(0.5×footprint)、fill_radius(0.9×footprint) 口径不变。
- **派工**:A0 同步 config.py docstring/注释 + 文档;A1 同步 `test_obs_candidate_config` 的默认值断言(0.3/0.5→0.5/0.7,含 `_EXPECTED_DEFAULT_*` 常量与元组断言)。
- **验收**:`pytest tests/`=147/0(A1 同步 4 个默认断言后全绿)。

## v1.0.11 Follower 导航 waypoint 去冗余(DP 路径简化,消弹簧轨迹)(A0,2026-06-19;人类裁定)

- **议题**:人类从 fig2 发现 Follower 在密集 waypoint 处出现"弹簧圈"绕圈轨迹。A0 排查 AUV 运动控制(`waypoint_to_target_vel` LOS 制导 + 带惯性动力学):已有"偏差>20°按 cos 衰减 surge"缓解但不彻底(惯性 + 转弯半径硬约束)。**根因 = waypoint 点距(2m) < AUV 最小转弯半径(cruise/r_max=1.5/0.5236=2.86m)**:物理上无法平顺穿过密集点。实测:arrive_r=0.5 绕行 1.32x(弹簧圈)、arrive_r=1.5 绕行 0.67x(切角)、180° 掉头 1.8x——两容差都不理想,病根是点太密。
- **裁定**(人类选去冗余,治本):对 VRP 输出 route 做 **Douglas-Peucker 简化**(容差=走廊半宽 1m),移除"落在前后连线 1m 内、AUV 自然经过"的中间点,只留转折点 → 导航点距拉大、AUV 平顺跟踪。被移除点偏离简化连线 <1m<走廊半宽 → 仍被实际走廊覆盖,**覆盖不丢**。规划层(Phase 2 VRP 仍访问全部候选点)不变,只优化执行层导航。
- **实现**:mission 脚本加 `_douglas_peucker`/`_simplify_route`;Phase 3 导航前简化(Follower 点数 19/13→12/9),execute + 覆盖 + fig2 均用简化导航。
- **效果**:fig2 弹簧圈消失(平滑大弧);覆盖 6/9 → **8/9**(平顺轨迹走廊更连贯,反升);exit=0;`pytest`=147/0(改动隔离 examples)。
- **作用域**:examples/mission(A0),不涉及契约/模块/测试。备选(未采用):改 LOS 控制(假设1)治标且动 A3 物理件;回调 arrive_radius 回到切角。

## v1.0.12 候选提取【固定绝对阈值】提升为通用配置(A0,2026-06-22;人类裁定)

- **议题**:新增 windowed 协同探查模式(`examples/mission_windows_survey.py`)按 20×20 观测窗口**分块**规划——
  路径规划输入由"全局先验图"改为"当前观测窗口子块"。但候选提取(`detections_from_field` /
  `observation_candidates_from_field`)用**相对阈值**(threshold_rel × field.max()):它仅在「全局归一化场
  (max=1)」时等价于绝对值;一旦输入是**子窗口/局部场**,局部 max<1 会把空/弱窗口里 ≈0.005 的 KDE 拖尾
  相对放大成"满量程"→ 生成伪候选、Follower 被派去白扫整窗,**实际部署鲁棒性差**。改用全局 field.max()
  又等于提前使用**未揭示区**信息(违反修士开环逐步揭示,纪律4)。
- **裁定**(人类选"固定绝对阈值,提升为通用模式"):候选去留改用**固定绝对阈值**(归一化先验值口径,
  **不锚定任何 max**)——对所有窗口一致:远高于拖尾(实测空窗口块内 max≈0.005)、低于真实目标峰高
  (seed=9 实测 W0=0.618/W1=1.000/W2=0.925)→ 空窗口零候选。提升为 `ObsCandidateConfig` 配置项(**单一
  真相源**),baseline 与 windowed 统一口径。初值 peak=0.5 / fill=0.7(= baseline 全局等效,便于对照;
  后续结合识别算法/相机模块标定契合值)。
- **契约**:`ObsCandidateConfig` 新增 `peak_abs_threshold` / `fill_abs_threshold`(均 `Optional[float]=None`);
  新增 `resolve_obs_candidate_abs_thresholds(cfg)→(peak_abs, fill_abs)`。**默认 None → 退化纯相对阈值
  (向后兼容:现有 `default_detections` / A1 验收 / 全局 mission 行为逐位不变)**;消费者在「几何生成
  (相对阈值)」后按绝对阈值对候选点处先验值再筛一道。`detections_from_field` /
  `observation_candidates_from_field` 函数签名**不变**(不动 A2 实现)。
- **派工**:A0 加 config 字段 + resolve + 本记录;A0 改两个 mission 脚本(baseline/windowed)统一走
  "几何生成 + 绝对过滤(从 cfg 读)"。**待广播(未做,触他人文件需后续裁决)**:是否把绝对阈值原生集成进
  A6 `eval/runner.py::default_detections`(令 RL/全部基线默认走绝对口径)、或 A2 候选函数原生支持绝对阈值。
- **验收**:`pytest tests/`=147/0(新增字段默认 None,零回归)。baseline 候选 **32 不变**(峰值5+补点27,
  全局 max=1 → 绝对=相对,仅口径统一)、min-max 67.5m 不变。windowed 候选 33(峰值7+补点26)、全局召回
  8/9(89%)、检出 9/9(100%)。**空窗口鲁棒性实测**:seed=4 的 W0 与 seed=12 的 W2(块内无目标)在
  (A)相对/局部 max 下各产 4/3 个伪候选 → (B)固定绝对阈值下**均 0 候选**(不再白扫)。

## v1.1.0 波次2(A5:MDP/策略层)开工裁决(A0,2026-06-22;人类逐条确认)

波次1 全层 🟢(`pytest tests/`=147/0、acceptance 39/39),A5 上游依赖全部满足。开工前 A0 列 4 项待决,人类逐条裁定:

1. **训练运行环境**:✅ 采用 **`auv_py310`**(主环境,gymnasium 1.2.3 + torch 2.5.1+cu121 + numpy 2.2.6;波次0/1 全部测试在此通过)。**不**切 `cleanrl` 旧环境(gymnasium 0.28)。
   - 连带:`train_ppo` 按 **gymnasium 1.2.x API** 写(`reset(seed=)`→(obs,info)、`step`→五元组、`gymnasium.utils.env_checker.check_env`);若贴 CleanRL 原版片段须适配到 1.2.x(向量化/wrapper 接口),不得引入 0.28 依赖。
2. **reward 量纲归一化**:✅ **加权前各项先各自归一化到 [0,1]**,再用一组**占位权重**跑通,标定留到评估阶段。
   - 落地(契约):`RewardConfig` 增两个占位归一化基准 `value_norm_ref`(熵降 bit 分母,默认 1.0)、`balance_norm_ref`(均衡惩罚能耗代理分母,默认 100.0);公式 `reward = value_weight·clip(ΔH/value_norm_ref,0,1) − balance_weight·clip(balance_pen/balance_norm_ref,0,1)`。同步 `contracts/mdp.py` step docstring。改这两个 ref 不改公式结构(评估阶段标定旋钮)。
3. **波次2 验收边界**:✅ 先定到 **"PPO 能训练、每个 episode 能正常运行跑完"**;训练效果/对比由**人类验收**,后续再调。
   - 即 DoD = ① `coop_survey_env` 过 `env_checker`、忙者不决策/一次一台;② 随机策略 smoke 跑通最小 episode(全链路);③ `train(cfg)` 能驱动 PPO 连续训练若干 episode 不崩、reward 有数;④ 集成门:A6.path_cost 与 A3.Level1RolloutEngine 同 routes 数值一致。**不要求**收敛/超越基线(留待人类评估迭代)。
4. **learned policy 演示进 examples**:✅ 需要。新增 `examples/mission_learned_survey.py`(A0 拥有),**复用 `mission_windows_survey.py` 完全相同的环境设置与 windowed 调查模式**(全局 60×20、20×20 观测窗口逐块揭示、开环、即时派遣 + 并行语义、固定绝对阈值候选口径),**仅把路径规划方法从"峰值检测+A′候选+VRP"替换为"训练好的 learned policy 逐窗决策"**。其余流程/图/指标口径一致,便于与基线直接对比。

- **派工(文件归属不交叉)**:A0=`contracts/`(config.py 已改 + mdp.py 已改)+ 本记录 + 看板 + 后续 `examples/mission_learned_survey.py` 与端到端集成;A5=`msim/mdp/*`(commitment/state/action/coop_survey_env)+`msim/policy/*`(network/train_ppo);A1=`tests/`(L3/L4 验收:env_checker、忙者不决策、一次一台、commitment 解析量、obs 归一化、action 纯几何、reward 归一化公式)。
- **状态**:契约改动(§v1.1.0 的 1/2)已冻结落盘;A5 派工单待人类发话即 spawn。

## v1.1.4 reward 重构 + 协作/窗口修复(A0,2026-06-23;人类评估后逐条裁决)

15万步训练演示仍 0/9 覆盖。A0 诊断:策略沿 y≈11 扫"光晕"、擦肩目标簇 1-2m,根因是 reward 设计 + 几处协作/窗口 bug。人类与 A0 头脑风暴后定四点(**Q6 value 定义由此更新**):

1. **dense value 从 EIG 熵降 → 已解析先验概率质量(GT-free)**:二元熵在 p=0.5 最大、p=1 为零 → 旧 EIG 奖励
   光晕、躲目标中心(场峰 p≈1 熵=0)。改为:Follower 走廊扫过一格 → 该格剩余先验概率**衰减到 0**(复用
   `planned` 掩码),**reward = 衰减掉的概率值 = field[cell]**。概率质量在目标中心(field=1)最大 → 奖励对准
   目标;GT-free、部署可算、**无需熵/无需先验校准**。`dense_mass = Σ_{走廊∩窗口∩未planned} field[cell]`。
   (v2 备选:按 p_d 贝叶斯连续衰减、可重观测递减;本批用 v1 衰减到 0、复用布尔 planned 掩码。)
2. **sparse 真检出奖励(GT)**:Follower 走廊**首次确定性覆盖**一个 GT 目标格 → `+detect_weight`;`detected`
   集合去重(每目标一次)。**GT 仅入 reward**(训练信号);策略 obs 不含 GT,master/doctor 不分支 → 不违开环。
3. **value 项缩放不截断**(点2):`v = value_weight·(dense_mass/value_norm_ref)`,去掉旧 `clip(...,0,1)` 上限。
4. **降 balance 权重 + 软冗余惩罚**(点3):`balance_weight` 1.0→0.2;新增 `redundancy_weight`,惩罚本步与
   **其它 Follower** 已覆盖重叠的格数(`−redundancy_weight·n_redundant`),**软**(不终止 episode;硬碰撞后续再加)。
5. **窗口出界 bug 修复**(点4,方案 A):Leader 从 lead-in x=-3.5 起步时,look_back 窗口落在场外(x<0)→
   Follower 首航点跑到地图外。修:**reset 时把 Leader 推进到"窗口首次完全落入场内(x≈look_back)"再开始决策**
   (Leader 轨迹仍从 -3.5 记录;语义同 window 演示"扫完 W0 才派 Follower")。

- **info 契约(env 暴露,供 A1 测)**:`reward` + 分解项 `dense_mass`/`dense_term`/`n_new_detections`/`detect_term`/
  `balance_pen`/`balance_term`/`n_redundant`/`redundancy_term`;`reward == dense_term+detect_term−balance_term−redundancy_term`。
  保留 `per_follower_energy_J`/`clock_s`/`current_follower`/`n_decisions`/`leader_finished`。
- **派工(文件归属不交叉)**:A0=`contracts/`(config.py RewardConfig 已改 + mdp.py docstring 已改 + 本记录);
  A5=`msim/mdp/coop_survey_env.py`(reward 公式 dense+sparse、软冗余、窗口修复 A、info 分解项、env 存 GT 目标格);
  A1=`tests/test_mdp_coop_survey_env.py`(重写 reward 公式/info 测试 + 新增 检出去重/软冗余/窗口入场 用例)。
  **A4 `task/value.py` 不动**(env 端直接累加 field 概率质量,不再调 `expected_entropy_reduction`;该函数留作历史)。
- **状态**:契约已冻结落盘;A5 + A1 派工单同步下发(对同一份 reward/info 规格编程);改完人类统一重训一次。

## v1.1.5 离散固定块窗口(A0,2026-06-23;人类选 B 后落地;补记档案)

§v1.1.4 重训后人类裁定:窗口由"Leader 位姿连续滑窗"改为**离散固定块**(人类原话:"离散的固定块更利于演示的对比,以及实际 AUV 实装")。

- **分块**:沿 North 把场等分为固定块,`edges=[k*look_back for k in 0..n_blocks]`(windowed:60/20 → 3 块 [0,20,40,60])。窗口 = **当前活动块**(非滑窗),复用 `get_window(anchor=[edges[k+1], y_max/2, 0])` → `coarse_grid_shape/coarse_index_to_world/decode_action` 全不变。
- **门控**:块 k 可派 ⟺ `leader_x >= edges[k+1]`(Leader 扫过其右边界);块顺序作业 0→1→2,step 开头 `_activate_next_block` 跳过已 done 块、必要时小步前扫 Leader 到活动块。
- **块"完成"**:当前块连续 `2*n_followers` 次决策 `dense_mass<ε`(无进展即止),**或**块内高概率区(field≥0.5)全 planned。终止 = 全块 done。
- **单测线前提(A0 裁决)**:离散块要求 `window.width_m == world.y_max_m`(块 = 全 East 的 North 条带)。windowed 满足(width=20=y_max);**small_cfg 已对齐**(width 8→20)。多测线(width<y_max)会使 East 两侧不入任何块、语义不自洽,禁止如此配置。
- **派工**:A0=contracts(本记录);A5=`coop_survey_env.py`(块调度/门控/`_block_window`/终止);A1=`tests/`(窗口为离散块、block_k 单调 0→1→2、waypoint 落活动块、块全进程不跳块、块窗口全 East 单测线)。

## v1.1.6 obs map 扣除 planned(消观测混叠,治分工塌缩)(A0,2026-06-24;人类评估 v2 后裁决)

`windowed_real_v2.pt`(§v1.1.4+§v1.1.5 重训)演示:0/9 → **3/9**,3 个检出全贴高概率带中心(§v1.1.4 dense 奖励导向验证成功),但**两台 Follower 塌缩到同一条中间测线**(y≈13),只覆盖最亮一簇、漏掉下方带+散点 6 个。

- **A0 根因诊断**:obs 的 `_coarse_map()` 读**原始静态先验** `field[ix,iy]`,**不扣 planned**;而 reward 侧**有**扣 planned(`if not planned[cx,cy]`)+ 软冗余惩罚。后果是**观测混叠(state aliasing)**:"该带已扫"与"未扫"对应**同一 obs**、最优动作却不同 → 策略作为 obs 的函数无法区分 →(a) 两台看到相同 map → 贪心 argmax 选同一带;(b) 单台扫完亮带后 obs 不变 → 不会推进到第二带。**奖励对、策略看不见奖励依据的那半状态**。
- **人类裁定(2026-06-24)**:✅ **obs map 改为残余先验场**——`map[row,col] = field[center] · (1 − planned[center])`(粗格**中心**采样,与现有 field 采样平行;planned→**硬置 0**,与 reward 布尔 planned 掩码同口径)。obs 与 reward 对齐;GT-free、部署可算。
  - **空间不变**:map 仍 `(Hc,Wc) float32 ∈ [0,1]`,`observation_space`/`action_space` **均不变** → 旧 checkpoint 仍能 load,但输入分布改变 → **必须重训**;`windowed_real_v2.pt` 保留作对照基线(勿删)。
  - **不是开环违规**:planned 是**布尔覆盖掩码**(走廊扫过即置位,修士也更新,见 Q6 `planned_coverage` 与 `belief` 分离),**非** belief 回写;field 本身逐位不变,NullUpdater 仍开环。策略 obs 仍不含 GT。
- **契约改动**:`contracts/mdp.py::assemble_obs` docstring 的 `"map"` 语义由"窗口概率场降采样"收紧为"窗口**残余**先验概率场(已扣除 planned)降采样";`CoopSurveyEnvProtocol` obs 说明同步。**签名/形状不变**。
- **派工(文件归属不交叉)**:A0=`contracts/`(mdp.py docstring + 本记录,已冻结);A5=`msim/mdp/coop_survey_env.py`(仅改 `_coarse_map()`:中心 cell 读 `field*(1-planned)`);A1=`tests/test_mdp_coop_survey_env.py`(新增不变量:① planned 粗格在 obs map 读 0;② 一台扫过某带后,下一决策者 obs map 该带已衰减;map 仍 ∈[0,1]、形状不变)。
- **状态**:契约已冻结落盘;A5 + A1 派工单同步下发;改完 A0 跑全测试(不信自报)→ 人类重训 `windowed_real_v3.pt` → A0 复评(看两台是否分带、覆盖是否提升)。

## v1.1.7 网络空间盲视 bug:actor 头改空间卷积(治"策略不读 map")(A0,2026-06-24;人类对照实验确诊)

§v1.1.6 重训出 `windowed_real_v3.pt`:覆盖 0/9(比 v2 的 3/9 更差),两台 Follower 仍塌缩同测线(y≈11,卡两簇山谷)。

- **人类决定性对照实验**:用只训 1–2 步(≈随机权重)的 `policy_latest.pt` 跑演示,输出与充分训练的 v2/v3 **相差无几**;换演示 seed 也不变;但训练时 mean_reward 确有提升。
- **A0 决定性诊断(喂 8 张差异极大的人造 map 给 random/trained 两 ckpt)**:无论高概率质量在 map 哪一行(上半/下半/角落单峰/随机),两个 ckpt 输出**恒为中间行 row≈5**;固定 map 换 6 个不同 vec,动作**一个不变**。→ **策略输出几乎不是 obs 的函数**;random 与 trained 几乎一致,正好解释人类现象。
- **根因**:`network.py` 的 map 分支末端 `AdaptiveAvgPool2d((1,1))`(全局平均池化 GAP)把 10×10 粗网格**压成 1 个标量**(只剩平均亮度),**丢掉"亮点在哪"的空间信息**。但动作空间是**空间的**(挑粗格 `idx=row*Wc+col`)、map 本身就是这 10×10 格。actor 头从空间盲视的特征只能学一个**与 map 无关的固定空间偏置**(≈中间行),训练只把这个常数偏置 + critic 微调到一个**退化天花板**(固定扫中间带蹭到部分先验质量 → mean_reward 真涨但浅)。
- **前两轮重新定性**:§v1.1.4(reward 先验质量)/§v1.1.6(obs 扣 planned)**都对、都必要**,但被网络瓶颈掩盖——v2 的 3/9 是固定偏置恰落 row6≈y13 蹭到上簇的**运气**,非学会找目标;v3 的 0/9 是落到山谷。
- **人类裁定(2026-06-24)**:✅ **actor 头改空间对齐卷积头**——map 分支 CNN **不做 GAP、保持 (Hc,Wc) 分辨率**;vec→MLP 后**广播**成 (B,Cv,Hc,Wc) 与 map 特征 concat;actor = `1×1 Conv(C+Cv→1)` → (B,1,Hc,Wc) → flatten → (B,Hc*Wc) logits,使 **logit[r,c] 由第 (r,c) 格局部特征(含其先验概率)算出** → 策略能"指向"高概率格、两台借 §v1.1.6 残余 map 分带。critic 仍可对融合特征 GAP → 标量 V。
  - **correctness 关键**:flatten 顺序必须 `idx = row*Wc + col`(行主序,与 `coarse_index_to_world`/`decode_action` 一致),否则动作语义错乱。
  - **非契约改动**:`build_actor_critic(cfg, n_actions)` 签名、`forward`/`get_action_and_value`/`get_value` 接口、obs/action 空间**均不变** → 属 A5 `network.py` 实现改动;**旧 ckpt(v2/v3/latest)与新架构层不兼容,须重训**(产 `windowed_real_v4.pt`;旧的留作对照)。网络仍守小规模(Pi5 推理:cnn_ch≈16,1×1 conv 头很廉价)。
- **派工(文件归属不交叉)**:A0=`contracts/`(本记录;mdp/config 无需改);A5=`msim/policy/network.py`(空间卷积 actor 头,critic GAP,保持 idx 顺序与各接口);A1=`tests/`(新增回归:① 输出是 map 的函数——多张不同 map → logits/argmax 不全相同,反 GAP 盲视;② 空间对齐——`∂logits[idx]/∂map[row,col]` 在对齐格非零;③ 形状/接口不变)。
- **状态**:契约层无改动,DECISIONS 已记;A5 + A1 派工单下发;改完 A0 跑全测试 → 人类重训 `windowed_real_v4.pt` → A0 复评(看 argmax 是否跟随亮点、两台是否分带、覆盖是否提升)。

## v1.1.8(代号 §B)RL 学习失败根因 + 方向B 修复(A0,2026-06-24;人类令"先核验对齐→试方向B→不行再C")

§v1.1.7 网络改空间卷积头后重训出 `windowed_real_v4.pt`/`policy_latest.pt`(windowed,150k);人类多次测试发现 learned planner **仍达不到 baseline 逐窗 VRP 的效果**。

- **A0 决定性诊断(临时脚本,用后即删;同 seed=9/9 目标演示场景)**:
  1. **接线全对(排除致命 bug)**:network `reshape(B,1,Hc,Wc)→(B,Hc*Wc)` 行主序 == `decode_action` == `_coarse_map` 的 `idx=row*Wc+col`,逐 idx 一致、`cmap[r,c]==field(decode_action)` 偏差<3e-8。§v1.1.7 修复有效。
  2. **旧策略 ≈ 均匀随机**:策略熵=**ln(100)=4.605**(均匀)、logits↔残余先验 Pearson **−0.05**(没指向)、被选格残余分位 0.16(比随机差)、**76% 白选**、seed=9 覆盖 **0/9**、17 步提前弃窗。→ actor 几乎没离开初始化。
  3. **同 env 机制下 greedy-oracle(每步选 obs map 残余最高格)拿 9/9 100%**(残余分位 1.0、0% 白选)。→ **表示空间/动作/终止机制/reward 结构全可解**,唯一瓶颈是 **PPO 没学会**(100 维离散 + per-action 信用分配弱;且 `train_ppo` 对优势做 minibatch 标准化 → reward **重标定无效**,排除"调权重"这条路)。
- **人类裁定**:先核验对齐(已 OK),再试方向B 修复;两个 lever 一起上。
- **方向B 修复(三处改动,全部向后兼容、默认 0 → A1 验收不变)**:
  1. **`contracts/config.py`(A0)** `RewardConfig` 加 `aim_weight=0.0`(高信噪比"瞄准"奖励=所选粗格残余先验,与 greedy 同口径、去转移距离污染)+ 后续 `length_weight=0.0`/`length_norm_ref`(每步段能耗惩罚,治"单台重走自己扫过区"绕路——`n_redundant` 只罚跨台)。
  2. **`mdp/coop_survey_env.py`(A5,A0 代改)** `step()` 算 `aim_residual`(mark_planned 前取值)与 `length_term`,reward 改 `dense+detect+aim − balance − redundancy − length`;info 增 `aim_residual/aim_term/seg_energy/length_term`。**aim/length 权重=0 时逐字退化为 §v1.1.4 公式**。
  3. **`policy/train_ppo.py`(A5,A0 代改)** 加 greedy 示范 **行为克隆(BC)暖启动**(`_collect_greedy_demos`/`_behavior_clone`,独立优化器,默认 `bc_*=0` 跳过)+ PPOConfig `bc_demo_steps/bc_epochs/bc_lr`。
- **结果**(`runs/windowed_B.pt`=aim+BC+80k;`windowed_B_ft.pt`=+length/redundancy0.3/balance0.3 微调 resume+120k,**已提为 `policy_latest.pt`**):
  - 策略熵 4.605→**1.24**、相关 −0.05→**+0.87**、残余分位 0.16→**0.98**、白选 76%→**0%**;seed=9 覆盖 **0→9/9 100%、TP 9/9**,与 baseline 持平;跨 seed 78–100%。
  - **效率仍结构性到不了 VRP**:RL min-max 路程 103–218m vs VRP ~75m(greedy 170m);length 惩罚加大也只挪 15–30%。根因=**"一次一格 myopic 吐 waypoint" 无全局路线排序 → 同簇内 zigzag,上限≈greedy**。
- **人类收口裁定(2026-06-24)**:✅ **暂时接受 B,收尾波次2,不重定卖点**(project_summary §11 维持原样);**方向C(动作改"在候选点里选",继承 VRP 式路线结构以逼近效率)或其他实时路径规划方案留到下一波次**。
- **文件归属备注**:本轮 A0 为集成实验**代改了 A5 拥有的 `coop_survey_env.py` / `train_ppo.py`**(经人类授权、改动向后兼容);**须知会 A5**,后续若由 A5 接手 C 改动以此为基线。`pytest tests/`=**208/0**(零回归)。验收边界仍为 §v1.1.0 DoD(能训能跑+机制正确;效果人类裁定)。

## review 已确认(2026-06-11,冻结 v1.0)
- **A. Level 1 用时口径**:✅ 采用承诺队列**解析近似含转弯时间**:
  `t_avail = 段长/cruise + Σ|Δψ|/nominal_yaw_rate + dwell×剩余点数`。与训练 Level 1 同源、确定、快。
- **B. NED 朝向**:✅ North=200m 长边(主测线方向)、East=100m;细网格 nx=1000(North)、ny=500(East)。
- **C. 运行环境**:✅ `auv_py310`(numpy2.2 / gymnasium1.2 / torch2.5 / scipy)。波次1 前为其 `pip install ortools pulp`。
