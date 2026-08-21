# 波次看板(WAVE BOARD)

> A0 维护。状态图例:🟡 契约草拟+骨架就绪(待冻结) · 🟢 实现完成且验收测试全绿 · ⚪ 未开工 · 🔵 进行中
> 当前阶段:**波次2 收尾(§v1.1.8 / 代号 §B,2026-06-24)——learned planner 经 §v1.1.4~1.1.8 五轮真训练迭代,覆盖/检出已追平 baseline(seed9 0→100%、跨 seed 78–100%),效率结构性到不了 VRP(留方向C 下一波次)。人类裁定"暂时接受B、收尾波次2、不重定卖点"。** `runs/policy_latest.pt`=B 微调版(aim+BC+length);`pytest tests/`=**208/0**。早期:波次2 实现 + L3/L4 验收 + 训练增强经 A0 独立验收(§v1.1.1/§v1.1.2);A5 交付 6 模块 + train_ppo 增强全 🟢。波次2 开工裁决 §v1.1.0(2026-06-22)。 波次1 + §v1.0.4~12 全部完成并经 A0 独立验证(`pytest tests/`=147/0、acceptance 39/39)。§v1.0.8=目标场尺度从相机 footprint **解耦**为 `cfg.target_field`;§v1.0.9=mission 执行口径(arrive_radius 局部→0.5m + 覆盖评估改用**实际轨迹走廊**);§v1.0.10=候选阈值默认定稿 (0.3,0.5)→(0.5,0.7);§v1.0.11=Follower 导航 waypoint **DP 去冗余**;§v1.0.12=候选**固定绝对阈值**提升为通用配置(windowed 子窗口鲁棒,空窗口零伪候选)。§v1.1.0=波次2 四项开工裁决(训练环境 auv_py310 / reward 加权前归一化 [0,1] + 占位基准 / 验收边界=PPO 能训能跑完(效果人类验收) / 新增 `mission_learned_survey.py` 复用 windowed 模式仅换规划法)。

## 运行环境
| env | python | numpy | gymnasium | torch | scipy | ortools/pulp | A0 评注 |
|---|---|---|---|---|---|---|---|
| **auv_py310** ✅主环境 | 3.10 | 2.2.6 | 1.2.3 | 2.5.1+cu121 | 1.15.3 | **ortools 9.15 / pulp 3.3.2 已装** | 波次0/1 全部测试在此验证通过;全路径解释器 `D:\nixingxing\Anaconda\envs\auv_py310\python.exe` |
| cleanrl | — | 1.24.4 | 0.28.1 | 2.4.1+cu118 | ✗ | ✗ | CleanRL 经典 gymnasium 版本,A5 训练若贴 CleanRL 原版可考虑 |
| base | 3.9 | 1.21.5(**损坏**) | ✗ | ✗ | ✗ | ✗ | numpy DLL 损坏,勿用 |

## 模块看板
| 模块 | 层 | owner | 波次 | 状态 | 专属验收判据(DoD,harness §4.2) |
|---|---|---|---|---|---|
| `config/config` | L0 | A2 | 1 | 🟢 | apply_overrides 泛型字段覆盖正确 |
| `geometry/grid` | L0 | A2 | 1 | 🟢 | 栅格↔世界↔栅格往返误差 0;粗↔细可逆 |
| `geometry/window` | L1 | A2 | 1 | 🟢 | 窗口恒在 Leader 后方;单调推进;边界裁剪 |
| `env_static/field` | L1 | A2 | 1 | 🟢 | graded 概率场;groundtruth 可 save/load 复现 |
| `env_static/detections_from_field` | L1 | A2 | 1.x | 🟢 | 概率图→候选检测点(局部极大+NMS;估计非GT;确定性逐位复现)[§v1.0.4] |
| `env_static/observation_candidates_from_field` | L1 | A2 | 1.x | 🟢 | 概率图→候选观测航点(A′混合:峰值锚点+覆盖半径补点;修合并簇当单点;R 与峰值/补点阈值均可配置;估计非GT)[§v1.0.5/6] |
| `env_static/leader_path` | L1 | A2 | 1 | 🟢 | lawnmower 覆盖、与概率场解耦 |
| `physics/auv_dynamics` | L2a | A3 | — | 🟢 | (根目录 AUV 类,已就位) |
| `physics/leader` | L2a | A3 | 1 | 🟢 | Leader 动力学沿 lawnmower 推进;能耗不进 reward |
| `physics/follower` | L2a | A3 | 1 | 🟢 | 相机模型 p_d/p_fa/footprint |
| `physics/rollout` | L2a | A3 | 1 | 🟢 | Level1 用时/能耗与解析一致(§v1.0.1 转角口径);Level2 与 AUV 积分一致 |
| `task/corridor` | L2b | A4 | 1 | 🟢 | footprint×1.0 扩带(§v1.0.7,原×0.9作废);覆盖格集与解析解一致(中心法+butt cap,§v1.0.2) |
| `task/value` | L2b | A4 | 1 | 🟢 | **不重复计数**;先验期望熵降(互信息)与手算小例一致(加倍测试) |
| `task/belief` | L2b | A4 | 1 | 🟢 | NullUpdater 后地图逐位不变(开环性);Bayesian 仅占位[博士] |
| `task/energy_balance` | L2b | A4 | 1 | 🟢 | min-max;均衡与效率分离;空序列→0.0 |
| `mdp/commitment` | L3 | A5 | 2 | 🟢 | 剩余距离/可用时间解析量(非估计);含转弯时间(Q3/§v1.0.1);A0 验:5 组随机 commitment 与 Level1 逐项 |Δ|<1.5e-14 |
| `mdp/state` | L3 | A5 | 2 | 🟢 | Dict obs `{map(Hc,Wc),vec(10,)}` 全 clip[0,1];self/teammate 槽位 |
| `mdp/action` | L3 | A5 | 2 | 🟢 | 纯几何映射(委托 coarse_index_to_world),无避障/吸附 |
| `mdp/coop_survey_env` | L3 | A5 | 2 | 🟢 | A0 验:env_checker 过(small_cfg+全尺度);smoke 一次一台 0/1 交替无忙者违规;reward 公式逐字符合 §v1.1.0 |
| `policy/network` | L4 | A5 | 2 | 🟢 | 小 actor-critic(map→CNN+GAP / vec→MLP 融合,~8k 参数);Pi 5 友好;torch 延迟 import |
| `policy/train_ppo` | L4 | A5 | 2 | 🟢 | gymnasium 1.2.x API(非 cleanrl);A0 验:6 更新/1536 步/73 ep 不崩、损失全有限、v_loss 6.0→3.5、reward 有数 |
| `baselines/ortools_vrp` | L5 | A6 | 1 | 🟢 | min-max VRP(近似最优措辞,非 ground truth) |
| `baselines/exact_milp` | L5 | A6 | 1 | 🟢 | 小实例精确解(ground truth);>8 目标拒解 |
| `baselines/greedy` | L5 | A6 | 1 | 🟢 | 朴素贪心(性能地板) |
| `eval/metrics` | L5 | A6 | 1 | 🟢 | max/(CV或Jain)/总能耗/推理时间 |
| `eval/runner` | L5 | A6 | 1 | 🟢 | balance on/off 消融(weight 乘 0,非 if 分支);default_instance 目标场尺度经 `cfg.target_field`、与相机 footprint 解耦(§v1.0.8) |
| `tests/` | — | A1 | 1 | 🟢 | 12 文件 91 项全绿;value/belief 加倍判据已覆盖 |

## 波次1 验收记录(A0 独立验证,2026-06-12 / 2026-06-13)
- `D:\nixingxing\Anaconda\envs\auv_py310\python.exe -m pytest tests/ -q` → **105 passed / 0 failed**(波次1 收口 91 → §v1.0.3 耦合实例 +12 → clip 回归 +2)。
- 人工验收 harness `examples/acceptance_demo.py` → **39/39**,产出 5 图(地图/Leader 轨迹/Leader 揭示+窗口/基线路径/目标点叠加先验场)。
- 跨层反向 import 静态检查(§4.1):L0/L1→无、physics→无、task→无、baselines/eval→无。**全清**。
- 会话因 token 限额中断恢复后,in-process teammate 已终止;A0 重新 spawn **A4-2/A1-2/A2-2/A6-2** 接力收尾。
- 波次1 期间裁决:见 DECISIONS §v1.0.1(Level1 转角只计相邻段间转角)、§v1.0.2(import 路径规则 / 粗↔细可逆判据 / corridor 中心法+butt cap / Bayesian 占位 / balance 空序列 0.0 / 基线自包含 path_cost)、**§v1.0.3(ground-truth 耦合实例:真实目标点为绝对参考 → 先验场=目标点高斯KDE;新增 `contracts/env_static.py`;集成验收抓出并修复 clip→重合目标缺陷,改拒绝重采样)**。

## 波次 1.x 补丁 §v1.0.4(2026-06-15,✅ 完成并经 A0 独立验证)
- **收口**:A2 实现 `detections_from_field`(A0 独立内联验证:单点/合并/分离/空场/确定性/单调/边界全过);A6 加 `default_detections`+`run_ablation(target_source)`(默认 scanned_map,未破坏既有测试);A1 加 13 条验收(检测 9 + 评估 4);A0 独立 `pytest tests/`=**118/0**;A0 集成 `examples/mission_baseline_survey.py`(seed=3 代表性场景:9 GT→6 检测点、检测召回 89%、端到端 TP 7/9=78%、FP=0、OPEN-LOOP 标注完好、exit=0)。
- **标定定稿**:`threshold_rel=0.3`、`min_separation_m=0.5*footprint_lateral_m=3m`(=先验模糊 σ)。扫描证实提取数量对阈值/间隔不敏感,由 σ 下近邻几何合并主导;检测召回随场景在 0.56~1.0 波动(物理诚实的有损读图),零伪检。

- **议题/裁定**:baseline 原把 ground-truth `targets` 直接喂 VRP(oracle 上界,不现实)→ 改喂 **Leader 扫描概率图的派生检测点**(局部极大+NMS,估计非 GT);GT 仅留作评估侧绝对参考;评估管线保留 oracle 参照做对比。详见 DECISIONS §v1.0.4。
- **派工**:A0 冻结 `env_static.detections_from_field` 契约 + 改 `mission_baseline_survey.py`;A2 实现 field.py 中该函数;A6 加 `eval.runner.default_detections` + `run_ablation(target_source=...)`;A1 加验收 + eval realistic 测试。
- **文件归属(不交叉)**:A0=contracts/+examples/;A2=env_static/field.py;A6=eval/runner.py;A1=tests/。
- **收口判据**:`pytest tests/` 全绿(105 + 新增);`mission_baseline_survey.py` 基线消费 detections、图显示 detections≠GT、OPEN-LOOP 标注完好、exit=0。

## 波次 1.x 补丁 §v1.0.5(2026-06-16,✅ 完成并经 A0 独立验证)
- **议题/裁定**:纯峰值 `detections_from_field` 把 σ 模糊下的合并簇当成单点(簇内目标数被吞)。改用 **A′ 混合**:峰值锚点(逐点保留)+ 高概率区按「覆盖半径 R」补点。否决去模糊(无噪声+核已知→反推≈恢复 GT,作弊嫌疑)与连通域+kmeans(不稳)。详见 DECISIONS §v1.0.5。
- **收口**:A2 实现 `observation_candidates_from_field`(复用 detections_from_field 为子步;A0 独立验证:峰值保留⊇、自适应、确定性、seed24 recall@5.4→1.0);A6 `default_detections` 改调新函数并接 config 入口;A1 加 12+6 条验收;A0 独立 `pytest tests/`=**136/0**;A0 集成 mission(峰值金菱 vs 补点青圈分色,exit=0)。
- **R 可配置(人类追加)**:实测 R 是覆盖/成本权衡旋钮(seed24:R=5.4→端到端 5/9、~9 候选;R=2.7→8/9、~30 候选、路程+48%)。裁定 **R=5.4 为默认**,提升为 config:`contracts/config.py` 加 `ObsCandidateConfig` + `SimConfig.obs_candidate` + `resolve_obs_candidate_params`;改 `cfg.obs_candidate.fill_radius_m` 一处即可调,eval/mission 统一跟随。
- **文件归属(不交叉)**:A0=contracts/(env_static.py+config.py)+examples/;A2=env_static/field.py;A6=eval/runner.py;A1=tests/。

## 波次 1.x 补丁 §v1.0.6(2026-06-16,✅ 完成并经 A0 独立验证)
- **议题/裁定**:A′ 原峰值与补点共用单一阈值;人类要求拆开——峰值阈值保持 0.3,补点阈值单独设 **0.5**(更严,补点只落确信密集区/合并簇核心)。
- **收口**:A0 改契约(`observation_candidates_from_field` 加 `fill_threshold_rel=0.5`)+ config(`ObsCandidateConfig.fill_threshold_rel`;`resolve_obs_candidate_params` 3→4 元组)+ mission;A2 改 field.py 实现(补点区用 fill_threshold_rel,峰值仍用 threshold_rel);A6 改 default_detections(4 元组+透传);A1 修 5 + 新增 6 测试。A0 独立 `pytest`=**142/0**,mission(R=2.7/补点阈值0.5)exit=0。
- **验收不变量**:峰值集合只随 threshold_rel 变(与 fill 阈值无关);提高 fill 阈值→补点数不增(单调)。
- **文件归属(不交叉)**:A0=contracts/(env_static.py+config.py)+examples/;A2=env_static/field.py;A6=eval/runner.py;A1=tests/。

## 波次 1.x 补丁 §v1.0.7(2026-06-19,✅ 完成并经 A0 独立验证)
- **议题/裁定**:人类按"实际探测精度"重设传感器几何——相机探测范围=以 Follower 为重心 2m×2m,grid cell 同设 2m(res_m=2.0)。走廊宽口径三选一,裁定**取消 ×0.9 保守扩带 → 走廊宽 = footprint_lateral × 1.0**(footprint=2→走廊宽=2.0m=1 cell,严格对齐;**推翻 §v1.0.2 的 ×0.9**)。详见 DECISIONS §v1.0.7。
- **收口**:A0 改 `contracts/physics.py`(corridor_width_m ×0.9→×1.0)+ `examples/`(acceptance_demo 断言、mission `cfg.sensor=SensorConfig(footprint 2×2)`);A1 同步 `test_physics_follower_leader`(2 断言)+ `test_task_corridor`(5.4→6.0);A3 同步 follower.py 注释。A0 独立验证:`pytest tests/`=**142/0**、`acceptance_demo`=**39/39**(corridor_width=6.0m)、mission(走廊宽=2.0m、surveyed 100%、覆盖 9/9、TP=8/p_d漏检=1/FP=0)。
- **作用域**:footprint=2×2 仅 mission 局部(`cfg.sensor`),不改全局 SensorConfig 默认;`resolve_obs_candidate_params` 的 fill_radius 0.9(**另一独立 0.9**,候选补点半径)未动,obs_candidate 默认断言不受影响。
- **连带**:默认走廊宽 5.4→6.0(全局);mission footprint_along=2 → Phase3 FP 采样步长 6→2、峰值 NMS 间隔 σ=0.5×footprint=1m。
- **文件归属(不交叉)**:A0=contracts/physics.py+examples/;A1=tests/;A3=physics/follower.py(仅注释)。

## 波次 1.x 补丁 §v1.0.8(2026-06-19,✅ 完成并经 A0 独立验证)
- **议题/裁定**:§v1.0.7 改相机 footprint=2 后人类发现"声呐揭示"也变——根因 `default_instance` 的 KDE 带宽(0.5×footprint_lateral)与簇散布(2×max footprint)借用相机 footprint 当尺度,改相机连带重画 ground-truth 场景(实测 seed24:目标点全变、先验 >0.5 占场 21%→4%)。裁定**解耦 + 独立旋钮**:目标场是环境属性,与相机无关。详见 DECISIONS §v1.0.8。
- **收口**:A0 加 `config.TargetFieldConfig`(bandwidth_m=3.0/intra_spread_m=12.0/targets_per_cluster=3,**默认=原 footprint=6 值 → 默认场景逐位不变**)+ `SimConfig.target_field`;A6 改 `eval/runner.py`(`_default_cluster_params`+`default_instance` 改用 cfg.target_field、删 footprint 依赖);A1 加 `test_target_field_decoupling.py`(5 项)。A0 独立验证:`pytest tests/`=**147/0**;解耦诊断(footprint 6→2 → targets/field **逐位相同**)、bandwidth 旋钮 1/3/6m → 先验 >0.3 占场 7%/39%/70% 单调;mission 揭示场景恢复 σ=3m、目标点回到 footprint=6 历史值、覆盖 9/9。
- **边界(有意保留)**:候选提取的 peak_sep=0.5×footprint、fill_radius=0.9×footprint **仍挂相机**(那是观测/候选选取精度,该随分辨率走),不在本次解耦范围。
- **文件归属(不交叉)**:A0=contracts/config.py+examples/mission;A6=eval/runner.py;A1=tests/。

## 波次 1.x 补丁 §v1.0.9 / §v1.0.10(2026-06-19,✅ 完成并经 A0 独立验证)
- **§v1.0.9(mission 执行口径,examples 局部)**:人类发现 fig2 实际轨迹与规划 waypoint 有出入。A0 查数据流(detections→VRP→execute)确认**传递无误**(VRP 用 `_start_poses` depot=execute 起点、routes 点=detections 原值)。出入在执行层:`arrive_radius=1.5m` 与精细场景 2m 点距不匹配(实测相邻段中位 2.0m、容差/点距≈0.75)→ AUV 切角。裁定(人类选 1+3):① arrive_radius **局部** 1.5→0.5m(仅 mission `follower_overrides`,大场景默认不动);② 覆盖评估改用 **`res.trajectory` 实际走廊**(取代规划 route 折线)。**实测暴露执行损失**:规划 route 走廊 9/9 vs 实际轨迹走廊 **6/9**(T2/T7/T8 在密集簇被切角漏掉)——此前误报 9/9 已纠正。详见 DECISIONS §v1.0.9。
- **§v1.0.10(候选阈值默认定稿,契约)**:人类直接改了 `ObsCandidateConfig` 默认阈值 (0.3,0.5)→(0.5,0.7)(峰值/补点都调严),令 `test_obs_candidate_config` 4 个默认断言失效。裁定**全局定稿 0.5/0.7**(取代 §v1.0.6)。A0 同步 config docstring/注释;A1 同步测试断言(`_EXPECTED_DEFAULT_*` 常量 + 元组)。A0 独立 `pytest tests/`=**147/0**。
- **文件归属(不交叉)**:§v1.0.9=examples/mission(A0);§v1.0.10=contracts/config.py(A0)+tests/(A1)。

## 波次 1.x 补丁 §v1.0.11(2026-06-19,✅ 完成并经 A0 独立验证)
- **议题/裁定**:人类发现 Follower 在密集 waypoint 处出现弹簧圈绕圈。A0 排查根因 = waypoint 点距(2m) < AUV 最小转弯半径(cruise/r_max=2.86m),物理上转不过密集点(实测 arrive_r=0.5 绕行 1.32x/弹簧圈、arrive_r=1.5 绕行 0.67x/切角)。裁定**导航点 DP 去冗余**(Douglas-Peucker,容差=走廊半宽 1m,移除落连线内、AUV 自然经过的中间点,只留转折点;被移除点 <1m<走廊半宽 → 覆盖不丢)。详见 DECISIONS §v1.0.11。
- **收口**:mission 加 `_douglas_peucker`/`_simplify_route`,Phase3 导航前简化(Follower 点数 19/13→12/9),execute/覆盖/fig2 均用简化导航。fig2 弹簧圈消失(平滑大弧);覆盖 6/9 → **8/9**(平顺走廊更连贯,反升);`pytest`=147/0。
- **文件归属(不交叉)**:examples/mission(A0),不涉及契约/模块/测试。

## 集成总闸门(波次2,§v1.1.0 已定 DoD;待 spawn)
- A5(L3/L4)实现:commitment/state/action/coop_survey_env/network/train_ppo。
- **DoD(§v1.1.0 人类确认,验收边界=能训能跑完;收敛/超基线留人类评估)**:
  1. `coop_survey_env` 过 Gymnasium `env_checker`(auv_py310 gymnasium 1.2.x);忙者不决策、一次一台。
  2. smoke test:随机策略跑通端到端最小 episode(粗网格、N=2)全链路。
  3. `train(cfg)` 驱动 PPO 连续训练若干 episode 不崩、reward 有数(reward 按 §v1.1.0 加权前归一化 [0,1])。
  4. 集成门:断言 A6.path_cost 与 A3.Level1RolloutEngine 在同 routes 上数值一致(单一真相源,DECISIONS §v1.0.2)。
- A0 收尾:新增 `examples/mission_learned_survey.py`(复用 windowed 模式,仅换规划法为 learned policy)。
- 训练环境:`auv_py310`(gymnasium 1.2.3 + torch 2.5.1);**不**用 cleanrl(gymnasium 0.28)。

## 波次2 验收记录(§v1.1.1,A0 独立验证,2026-06-23)
- **A0 独立复跑(不复用 A5 脚本,临时脚本用后即删)**,解释器 `auv_py310`:
  1. `env_checker`:`check_env` 过 `small_cfg()` **与** 全尺度 `SimConfig()`。
  2. smoke(随机策略):small_cfg seed7 → 21~22 步 terminated;**决策者恒为最早空闲台(忙者不决策)、0/1 交替(一次一台)**,五元组形状/dtype/space-contains 全过。
  3. **集成门 DoD#4**:`eval.metrics.path_cost`(A6) vs `Level1RolloutEngine`(A3)在 200 组随机 routes 上 **max|Δenergy|=5.68e-14、max|Δduration|=2.84e-14**(数值同源,单一真相源确认)。
  4. reward 公式:逐步断言 r == value_weight·clip(ΔH/value_norm_ref) − balance_weight·clip(bal/balance_norm_ref),基准从 cfg 读(未硬编码)。
  5. PPO(DoD③):small_cfg、6 更新/1536 步/73 ep **不崩**、损失全有限、v_loss 6.0→3.5(critic 在学)、reward 有数。
  6. `pytest tests/`=**147/0**(A5 未碰 tests,无 import 破坏)。
- **跨层 import 静态检查**:L0–L2/L5 **无**反向 import mdp/policy;`policy/train_ppo` import `mdp`(合法 L4→L3 边);mdp/policy 内 **无** `if master/doctor` 分支(belief=NullUpdater 对象注入)。**全清**。
- **A0 记录的两点(非阻塞,留后续处理)**:
  - **① L3→L5 跨边**:`coop_survey_env.reset` 调 `msim.eval.runner.default_instance` 生成 (targets, field) 训练实例 → mdp(L3)依赖 eval(L5,旁路)。**此边由 A0 在 A5 派工单中亲口授权**(非 A5 越界),且单向无环、保证 RL 与基线消费同一实例生成口径(利于公平对比)。依赖图已补记此边(见 DEPENDENCY_GRAPH §L3→L5)。**清洁化备选(未做)**:把实例生成下沉到 L1 `env_static`,env 与 eval 共同消费,解除 L3→L5 耦合 —— 留作后续重构裁决。
  - **② Follower 能耗简化**:env 每段末 `phys.reset(pose)` → `energy_norm` 恒复满,obs 里电量项静态、电量耗尽截断几乎不触发;均衡/价值用**解析 energy_J 累加器**(正确)。对波次2 DoD(能训能跑)无碍,但 obs 电量信息缺失,若后续要让策略感知电量需补。已知会人类。

## 波次2 验收记录(§v1.1.2:A1 测试 + A5 训练增强,A0 独立验证,2026-06-23)
### A1 — L3/L4 验收测试(`tests/`,45 项)
- 新增 4 文件:`test_mdp_commitment.py`(9)、`test_mdp_state_action.py`(9)、`test_mdp_coop_survey_env.py`(18)、`test_policy_network_train.py`(9)。**A0 独立复跑**:L3 三文件 36/0;**全量 `pytest tests/`=192/0**(147 既有 + 45 新)。
- A0 通读 env 测试确认**真判据非共因**:决策者==独立 argmin(free_time)、忙者从不被选、reward 公式逐步核验、balance-off 消融(weight=0 非 if)、value_weight 线性缩放、截断 clock≥阈值、seed 可复现;commitment/state/action 均对照单一真相源(Level1 / coarse_index_to_world)交叉验证。无疑似 bug。
- 计数勘误:A1 报告 commitment/state_action 为 11/10,实际收集 9/9(头条 45 新/192 总正确)。
### A5 — train_ppo 训练体验增强(`msim/policy/train_ppo.py`,唯一改动)
- 4 项:① wandb(gated,延迟 import)② checkpoint save/load(+n_actions 不匹配拒绝)③ ESC 监听(msvcrt 非阻塞,StopController 可编程触发)④ CLI main()+argparse。基础签名 `train(cfg, ppo=None, stop_controller=None)` 向后兼容。
- **A0 独立运行时复跑**(临时 checkpoint/wandb 目录,验后清):① 延迟 import 干净(import train_ppo 不拉 torch/wandb/gymnasium)② 向后兼容 train(cfg) 正常 ③ save→load→resume 权重 torch.allclose 一致、resumed=True 续训不重置 ④ n_actions 不匹配→清晰 ValueError ⑤ ESC StopController 编程触发→1 update 后存盘+优雅退出、ESC ckpt 可 resume ⑥ wandb offline 免登录不崩。全过。
- 清理:A5 自验生成的 `runs/policy_latest.pt`(256 步废模型)+ `wandb/` 已被 A0 清除,避免误当真实 policy;人类真训练时干净重建。
### windowed 训练预设(A5,§v1.1.3,2026-06-23,A0 独立验收)
- **议题**:learned policy 动作空间 `Discrete(Hc×Wc)` 由 `coarse_grid_shape(window,sensor)` 决定 → 训练 config 必须与演示(`mission_windows_survey.py`)的 window/sensor 一致,否则 checkpoint 装不进演示(n_actions 不匹配)。
- **A5 落地**:`coop_survey_env.py` 加 `windowed_survey_cfg(seed=0)`(world 60×20@res2、sensor 2×2、默认 window 20×20、N=2;**与 mission_windows_survey 逐字一致**,训练/演示**单一配置真相源**);`train_ppo.py` CLI `--world` 加 `"windowed"`。
- **A0 独立验**:`coarse_grid_shape`=10×10=**Discrete(100)**;`env_checker` 过;`python -m msim.policy.train_ppo --world windowed --timesteps 512` 全路径通(entropy=4.605=ln100、存盘、不崩);**`pytest tests/`=192/0 无回归**;仓库无 runs/wandb 残留。

## 波次2 真训练迭代 §v1.1.4–§v1.1.8(2026-06-23~24,人类评估驱动;详见 DECISIONS)
人类交互式跑真实训练 + A0 写 `examples/mission_learned_survey.py`(learned vs baseline vs window 同口径)后,据演示逐轮裁决:
- **§v1.1.4** reward 重构:EIG 熵降 → **先验概率质量**(治"奖励躲目标中心")+ 稀疏真检出 + 软冗余 + 降 balance 权重。
- **§v1.1.5** 窗口由 Leader 连续滑窗 → **离散固定块**(利于演示对比 + 实装)。
- **§v1.1.6** obs map **扣除 planned**(残余先验,消观测混叠,治两台塌缩同测线)。
- **§v1.1.7** 网络 actor 头 **GAP→空间卷积**(治"策略空间盲视、不读 map"的退化天花板)。
- **§v1.1.8(代号 §B)** RL 学习失败根因诊断 + 修复(**本轮收尾**):

### 波次2 收尾记录 §B(A0,2026-06-24,人类裁定"暂时接受B、收尾、不重定卖点")
- **诊断(决定性)**:§v1.1.7 修复后 v4/`policy_latest`(150k)**策略仍≈均匀随机**(熵=ln100、logits↔残余相关−0.05、seed9 覆盖0/9);但**接线全对**(reshape==decode_action==_coarse_map)、**同 env 下 greedy-oracle 9/9 100%** → 瓶颈是 **PPO 没学会瞄准**(非接线/终止/reward-scale;优势 minibatch 标准化使调权重无效)。
- **修复(三处,默认0 向后兼容,`pytest tests/`=208/0 零回归)**:`RewardConfig` 加 `aim_weight`(高信噪比瞄准奖励)+`length_weight`(治单台绕路);`coop_survey_env.step` 入 aim/length 项;`train_ppo` 加 greedy **BC 暖启动**。
- **结果**:熵→1.24、相关→**+0.87**、seed9 覆盖 **0→100% TP9/9**(追平 baseline),跨 seed 78–100%。**效率结构性到不了 VRP**(RL 103–218m vs VRP~75m;"一次一格吐点"无全局路线排序,上限≈greedy)。
- **交付**:`runs/windowed_B.pt`(aim+BC+80k)、`runs/windowed_B_ft.pt`(+length 微调 120k)**已提为 `runs/policy_latest.pt`**(演示默认即用修复版)。旧坏策略保留为 `windowed_real_v4.pt`。
- **文件归属备注**:A0 为集成实验**代改 A5 的 `coop_survey_env.py`/`train_ppo.py`**(人类授权、向后兼容)→ **须知会 A5**;后续 C 改动以此为基线。

### 下一波次(方向C,待开工裁决)
- 人类裁定:**方向C 或其他实时路径规划方案留下一波次**。C = 动作空间 `Discrete(全窗格)` → **"在候选点里选"**(复用 baseline 峰值+补点候选,RL 只决访问顺序/分配,仿 Vashisth 方案 ii),继承 VRP 式路线结构 → 效率逼近 VRP。**改 obs/action space = 契约改动,需 A0 开工裁决 + A5 实现**。
- 备选:其他实时路径规划模式/方案(人类待定)。
