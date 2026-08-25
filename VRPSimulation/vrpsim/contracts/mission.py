"""[第0层][A0 冻结] 任务层契约 —— Leader 广域扫描 + 双 Follower 声学握手式分区域 VRP。

任务流(人工规格 2026-08-09;通信时序由 D15 于 2026-08-13 改写):
  1. 场景 = D7 裁切后的世界(500 N x 100 E,22 个 waypoint)。
  2. 部署 1 Leader + 2 Follower。Leader 测线沿长边、East=50;初始 (0, 50)。
     Follower 初始 (0, 0) 与 (0, 100)。
  3. Leader 沿测线前进;100x100 窗口的**前边界始终与 Leader 位置重合**
     (= msim `get_window` 的语义:窗口在 Leader 正后方)。长宽可调。
     ⚠ D16 起 Leader **不是匀速**:它用自己的 DR 推算判断队友跟不跟得上,
       跟不上就原地停(见下面第 8 条)。
  4. **两条互相独立的时钟**(D16;D15 之前两者是绑死的):

     (a) **USBL 声学定位**,周期 `usbl_period_s`。一个循环 = 2N 跳:

             t0+0τ  Leader → F0  定位询问
             t0+1τ  F0 采样自身位置与占用点,回复
             t0+2τ  Leader 收到 p0;  Leader → F1 定位询问
             t0+3τ  F1 采样自身位置与占用点,回复
             t0+4τ  Leader 收到 p1 —— 本循环结束,位置库刷新

     (b) **路径规划**,周期 `plan_period_s`。一轮 = 求解 + 广播 1 跳:

             t_plan       取最近一次 USBL 定位,按**承诺队列解析推算**前推到此刻
                          (陈旧度 = 从上次定位到现在,见 vrpsim/tracking.py)
                          用两台位置 + 窗口内未访问未占用的 waypoint 解一次
                          min-max VRP,两台的路线**各截到至多 5 个**,打包成广播
             t_plan+τ     两台**同一拍**收到,各自用新序列覆盖旧序列(D8)

     解耦的意义:定位一直在跑、位置一直新鲜,而规划周期可以单独拉长
     (30 / 45 / 60 …)做对比实验,不牵动定位刷新率。
     USBL 定位与指令广播视为不同频段/换能器,**不建模信道竞争**(D16)。

     可规划的 waypoint 不足 5 个时,把该台位置投影到窗口前边界,
     投影点作为序列**最后一个** waypoint,且**不占用**。
  5. Follower 依次占用队首 waypoint → 前往 → 到达后停留 `dwell_time_s` → 占用下一个。
  6. 规划器 = VRP 求解器。
  7. 逐时刻记录 waypoint 序列与各 Follower 位置。
  8. **Leader 等待策略**(D16 人工裁决)。Leader 每拍用自己的 DR 推算判两条,
     任一成立就原地停船(速度置 0),都解除才恢复前进:

       判据 A `leader_wait_on_lagging_follower`
           某台 Follower 的**推算位置**已落到窗口后沿之后。管的是"车在哪"。
           恢复要多进 `leader_wait_release_margin_m` 米(迟滞,防逐拍停走振荡)。
       判据 C `leader_wait_on_endangered_target`
           窗口里还有**尚未被分配**的目标,再走 `wait_lookahead_s`(默认一个规划
           周期)就要掉出后沿。管的是"活会不会丢" —— 覆盖率掉下去的直接原因正是
           窗口滑过去时某些点从没进过任何一次规划的池。

     两条各自可关,便于做消融。停船的时刻与原因逐拍记进
     `MissionResult.leader_holding / leader_hold_reason`。

==================================================================
一处规格留白 —— 已按"不产生自相矛盾行为"的最小解释实现,可由配置翻转
==================================================================
**已访问过的 waypoint 是否还进池?**
规格未提。默认 `revisit_visited=False`:访问过即永久出池。否则窗口内的点会被
反复重规划,调查任务无法收敛。

> 原来还有第二处留白(「其它 Follower 尚未走到的队列尾算不算占用」,开关
> `reserve_other_queue`)。**D15 起它不再存在**:新时序下两台每轮都上报、每轮都被
> 重新规划,队列尾必然全部释放回池,占用的唯一口径是"**广播落地时刻**正在前往的
> 那一个"。这同时关闭了 DECISIONS.md 的待裁决 6。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .config import MothraSimConfig

# --- 序列里的特殊 waypoint 标记 ------------------------------------------
WP_NONE: int = -1          # 无
WP_PROJECTION: int = -2    # 投影点(窗口前边界上的追赶点,不占用、不计入覆盖)

# --- Follower 状态机 ------------------------------------------------------
STATUS_IDLE: int = 0       # 队列空,原地待命
STATUS_TRANSIT: int = 1    # 正前往当前占用的 waypoint
STATUS_DWELL: int = 2      # 已到达,停留拍照中

STATUS_NAMES = {STATUS_IDLE: "idle", STATUS_TRANSIT: "transit", STATUS_DWELL: "dwell"}

# --- 声学报文类型与节点编号(D15) ----------------------------------------
# 只用于 `MissionResult.comm_events` 的记录,不参与任何决策。
MSG_POLL_REQ: int = 0      # Leader → Follower  定位请求
MSG_POLL_REP: int = 1      # Follower → Leader  定位回复(附位置与占用点)
MSG_BROADCAST: int = 2     # Leader → 全体      序列广播

NODE_LEADER: int = -1
NODE_ALL: int = -2         # 广播的收方

MSG_NAMES = {MSG_POLL_REQ: "poll_req", MSG_POLL_REP: "poll_rep",
             MSG_BROADCAST: "broadcast"}

# --- Leader 停船原因(位标志,D16) ------------------------------------------
# Leader 不再匀速:它用**自己的 DR 推算**判断队友跟不跟得上,跟不上就原地等。
HOLD_NONE: int = 0
HOLD_LAGGING: int = 1      # 判据 A:某台 Follower 的推算位置已落到窗口后沿之后
HOLD_ENDANGERED: int = 2   # 判据 C:窗口里还有没被分配的目标,再走就要掉出后沿
HOLD_TURNAROUND: int = 4   # 判据 D:车道折返在即,窗口里还有没被分配的目标

HOLD_NAMES = {HOLD_NONE: "-", HOLD_LAGGING: "lagging",
              HOLD_ENDANGERED: "endangered", HOLD_LAGGING | HOLD_ENDANGERED: "both",
              HOLD_TURNAROUND: "turnaround"}


@dataclass(frozen=True)
class MissionConfig:
    """一次任务仿真的全部输入。所有"可调参数"都在这里,脚本经 CLI 透传。"""

    # --- 场景(D7 裁切后的世界) ---------------------------------------
    sim: MothraSimConfig = field(default_factory=MothraSimConfig)

    # --- 部署(规格 §2;NED = [North, East]) ----------------------------
    leader_start_ned: Tuple[float, float] = (0.0, 50.0)
    follower_starts_ned: Tuple[Tuple[float, float], ...] = ((0.0, 0.0), (0.0, 100.0))

    # --- 窗口(规格 §3,长宽可调) --------------------------------------
    window_look_back_m: float = 100.0    # 沿测线(North)长度
    window_width_m: float = 100.0        # 横向(East)宽度

    # --- 投影点去冲突(2026-08-24 人工裁决) -----------------------------
    # 窗口里没有可派的目标时,每台 Follower 会收到一个"回到窗口前边界"的投影点。
    # 投影点**不占用**,min-max VRP 的互斥保证只覆盖真实目标、管不到它们 ⇒ 逐台
    # 独立投影会把两台送到同一个点。sparse_2 实测:314 个双投影轮里 110 轮两点
    # 完全重合(间距 <0.01 m),两台实际最近逼到 0.70 m。
    # 该值是投影点之间在窗口前边界上的**最小横向间隔**,由 `planner` 在生成投影点
    # 时保序推开来保证(见 `project_to_window_front_multi`)。
    #
    # ⚠ **作用域仅限投影点**,不是全队的安全间距保证。真实目标的分配走
    #   `solve_minmax_vrp_from`,那里没有任何间距约束 —— 它只保证互斥(同一目标不
    #   派给两台),两台同时持有的目标可以任意近。实测(greedy)取 20 m 时投影点间距
    #   恒 = 20.000 m(约束是紧的),而同时持有的真实目标最小间距是 5.1~7.5 m 的自由
    #   值,与本参数无关。字段名点明 projection,免得被读成全队保证。
    #
    # 取值 20 m(2026-08-24 人工裁决):按 contracts/metrics.py 的**效率族 + 均衡族**
    # 实测选定,不是按 ">2 m 安全间距" 这类单点判据。实测(solver=greedy):
    #   场景      间隔  時間効率(厳密)  単位目標時間  距離不均衡  時間不均衡  艦隊総航程
    #   mothra     2 m      0.872        54.0 s       6.62%      5.89%     1661 m
    #   mothra    20 m      0.909        51.8 s       4.37%      0.85%     1607 m  <- 最优
    #   mothra    50 m      0.887        53.0 s       6.20%      3.98%     1607 m
    #   sparse_2   2 m      0.963       250.0 s       1.62%      0.68%    16551 m
    #   sparse_2  20 m      0.977       246.4 s       0.70%      0.77%    16419 m
    #   sparse_2  50 m      0.991       242.8 s       0.40%      0.06%    16401 m  <- 最优
    # 50 m 在稀疏场(sparse_2)全面最优,但在密集场(mothra)反而退化 —— 强行摊开让两台
    # 跑冤枉路;且 50 m 会让 mothra 在**关掉等待机制**时也达成 100% 覆盖(其余档
    # 95.5%),推翻 D16 的支撑证据。20 m 两个场景都靠前且不触发该问题,取之。
    projection_min_separation_m: float = 20.0

    # --- 运动学 ---------------------------------------------------------
    # 两者同取 0.5 m/s(2026-08-13 人工裁决,D16)。
    #
    # ⚠ 这个组合在**匀速 Leader** 下是不可行的,实测(Mothra + greedy,2026-08-25 刷新):
    #   全局 min-max VRP 最长单机路线 467.5 m / 10 个点
    #   ⇒ 467.5/0.5 + 10x10 = 1035 s,而 Leader 以 0.5 m/s 走完 500 m 只要 1000 s。
    #   窗口会先冻结,南边的点永久错过。
    #   D16 的 **Leader 等待策略**正是为此:Leader 用自己的 DR 推算发现队友跟不上时
    #   原地停,`t_leader_finish` 从输入变成结果 ——「覆盖率约束」被换成了「时间约束」,
    #   任务时间下界 = max(测线长/v_leader, t_route_lb) = 1035 s。
    #   `feasibility_estimate()` 会对任意参数组合重算这两个量。
    leader_speed_mps: float = 0.5
    follower_speed_mps: float = 0.5
    # 目標観測(Follower 相机确认)的定点停留。规格 §5 的初值是 5 s,
    # 2026-08-13 人工裁决改为 10 s(D15)。
    # ⚠ lawnmower 对照基线经 `contracts/lawnmower.py::LawnmowerConfig.base` 内嵌本
    #   dataclass,改这里两个场景一起变,不需要也不许在那边再写一份。
    dwell_time_s: float = 10.0
    # ⚠ 取 0 而不是 msim follower_overrides 的 1.5 m。1.5 m 是**真机的接受半径**;
    #   放进这个纯运动学计时仿真会让 Follower 把最后 1.5 m 瞬移掉 ——
    #   路程记了、时间没记,每个 waypoint 白送 1 s,22 个点就是 22 s 的假余量。
    #   本仿真关心的正是时序,故取 0:走满全程、时间与距离自洽。
    arrive_radius_m: float = 0.0

    # --- 声学通信(D15) ------------------------------------------------
    # ⚠ 单跳耗时 = 传播时延 + 包时长 + 前导/保护间隔。本场地最远约 510 m、声速
    #   1500 m/s ⇒ **纯传播只有 0.34 s**;2.0 s 的主体是后两项。论文里必须写清楚,
    #   否则读者会拿声速去反推作用距离。
    acoustic_hop_s: float = 2.0

    # --- USBL 声学定位:独立于规划的自有节拍(D16) ----------------------
    # 一个完整定位循环 = 逐台「询问 → 回复」= 2N 跳。15 s ⇒ 循环 8 s + 静默 7 s。
    # ⚠ D16 之前定位与规划是绑死的(一轮握手里先轮询再规划)。现在两者各有各的时钟:
    #   定位一直在跑、位置一直在刷新;规划周期可以单独拉长做对比实验。
    usbl_period_s: float = 15.0
    first_usbl_at_s: Optional[float] = None   # None → 0.0

    # --- 路径规划:独立节拍(D16) ---------------------------------------
    # 规划轮 = 取最近一次 USBL 定位 → 按承诺队列推算到当前时刻 → 解 VRP → 广播 1 跳。
    # **不再自己轮询**,所以这个周期可以随便拉长(30 / 45 / 60 …)而不影响定位刷新率。
    plan_period_s: float = 30.0
    first_plan_at_s: Optional[float] = None   # None → 0.0(任务起点先规划一轮)
    max_sequence_len: int = 5            # 规格 §4:至多 5 个
    # 仿真时间轴上给 VRP 求解留的余量。D15 裁决:**不建模**,取 0。
    # ⚠ 与 `PlanRound.solve_wall_s`(实测挂钟耗时)是两回事,别混。
    plan_solve_s: float = 0.0

    # --- Leader 等待策略(D16) ------------------------------------------
    # Leader 不再匀速。它**只用自己的 DR 推算**(vrpsim/tracking.py 的影子模型,
    # 不读仿真真值)判断队友跟不跟得上,跟不上就原地停,等条件解除再走。
    #
    # 判据 A —— 某台 Follower 的推算位置已落到窗口后沿之后。
    #   直白、好解释,管的是"车在哪"。
    leader_wait_on_lagging_follower: bool = True
    # 判据 C —— 窗口里还有**尚未被分配**的目标,再走一个规划周期就要掉出后沿。
    #   管的是"活会不会丢"。覆盖率掉下去的直接原因正是这个:窗口滑过去了,
    #   某些点从没进过任何一次规划的池 ⇒ 再也没人会去。
    leader_wait_on_endangered_target: bool = True
    # 判据 D —— **车道折返在即**,窗口里还有尚未被分配的目标(2026-08-25 人工裁决)。
    #   多车道 boustrophedon 特有的失效形态:折返瞬间窗口整体转向 180°,原本在
    #   Leader 身后的目标一拍之内变成"在身前",`s` 由正跳负、直接出局 —— 它**不是**
    #   从后沿滑出去的,判据 C 的"再走 lookahead 会不会越过后沿"完全感知不到。
    #   实测 dense_1 的 wp57:t=1784.0 时 s=+67.4 m 稳在窗内,下一拍 s=-32.3 m 出局,
    #   此前它被下发过 9 次、进池 18 轮,始终排在队列尾部没轮到队首。
    #   ⚠ 只在**后面还有段可走**时生效:最后一条车道走完是任务结束,那由强插末轮
    #     (D15)与终止条件负责,不该在这里把 Leader 永远钉住。
    leader_wait_on_turnaround: bool = True
    # 判据 C/D 的提前量。None → 取 `plan_period_s`:留出整整一个规划周期,
    # 保证濒危的点至少还能被规划到一次。规划周期拉长时它自动跟着变。
    leader_wait_lookahead_s: Optional[float] = None
    # 判据 A 的迟滞裕度:落后者要重新进到后沿以北这么多米,Leader 才恢复前进。
    # ⚠ 不留裕度会逐拍停走振荡 —— 两者同速时 Follower 一越过后沿,Leader 一动
    #   它立刻又落后。取 5 m ⇒ 每次停船至少持续 5/0.5 = 10 s,记录才读得懂。
    leader_wait_release_margin_m: float = 5.0

    # --- 规格留白的开关(见模块 docstring) ------------------------------
    revisit_visited: bool = False
    dwell_at_projection: bool = True     # 规格 §5 字面:序列里的点到了都停 5 s

    # --- Leader 对队友位置的认知(见 vrpsim/tracking.py) -----------------
    # 每台上报的都是它**采样那一刻**的位置,而回复本身还要在水里飞 acoustic_hop_s。
    # 到了规划时刻,两台的上报分别已陈旧 3τ 与 1τ ⇒ 位置由"上次上报 + 自己发出的
    # 承诺队列 + 已知运动学"解析推算前推,**不读仿真真值**。
    #
    # nav_drift_frac_of_distance:Leader 对队友坐标的**认知误差**,按自上次上报以来
    #   行进距离的比例增长(DVL/INS 惯例,0.005 = 0.5% = 5 m/km)。
    #   ⚠ 默认 0.0 —— 此时推算与真值逐位相同,是**确定量**,论文里不许写成"估计"
    #     (CLAUDE.md 硬纪律 4)。开了它才真的是估计。
    #   ⚠ 它建模的是 Leader 的**认知**误差,不是 Follower 的执行误差,两者是两回事。
    nav_drift_frac_of_distance: float = 0.0
    seed: int = 0                        # 漂移方向的随机种子,保证可复现

    # --- 求解器(口径见 D21;两个 VRP 方法共用本节全部字段) ---------------
    #
    # D21 求解器口径(2026-08-25 人工裁决,证据见 tests/test_planner_exact.py 头注)
    # ------------------------------------------------------------------
    # `solver="exact"` 时按**问题规模**分流,不是两个方法各用各的求解器:
    #
    #     2 台车 且 1 <= 池大小 <= vrp_exact_max_targets
    #         → Held-Karp 精确解:全局最优 + 构造性确定(同输入必同输出)
    #     其余(机数 != 2,或池大小超阈值)
    #         → ortools,预算 vrp_time_limit_s,GLS 惩罚系数 vrp_gls_lambda
    #
    # 分流结果如实写进 `Assignment.solver` / `PlanRound.solver`,事后可审计
    # 每一轮到底用的哪个求解器 —— 不静默混用。
    #
    # 为什么分流而不是一律 ortools:`vrp_span_coefficient=1000` 把目标函数放大
    # 约三个数量级(由 max路程 主导),而 ortools 的 GLS 惩罚加在**弧代价**上,
    # 量级差 1e4 ⇒ 惩罚永远改变不了邻域排序 ⇒ GLS 退化成普通贪心下降(实测其解
    # 与 GREEDY_DESCENT 逐位相同),几毫秒撞到局部最优后剩余预算纯空转。
    # dense_1 169 轮实测:8.3% 的轮次非最优,最差 +43.5%,且**加时间无效**
    # (k=11 某轮给到 30 s 仍 +38.9%)。小规模问题直接精确求解才是对症的。
    #
    # 为什么分流而不是一律精确:精确解是 O(2^k·k²),k 每 +1 耗时约 ×2.15。
    #
    # 兜底的 ortools 保持独立可选(`solver="ortools"`),供消融对照。
    #
    # vrp_time_limit_s:单次 ortools 求解的**挂钟**预算。
    #   ⚠ 只对 ortools 有意义;精确解的耗时是 k 的确定函数,与本字段无关。
    #   ⚠ GLS 是跑满预算才停的元启发式 —— "实测耗时吃满预算"是它的正常行为,
    #     **不是**收敛不足的证据,别拿它当难度指标。
    #   取 30 s 的依据(2026-08-25 人工裁决):二段式全局VRP(n=46..66)在 1 s 下
    #   远未收敛,最长单机航程比 10 s 的解虚高 13~34%(mef 1405→928 m,
    #   sparse_2 1496→1323 m,dense_2 1219→959 m);10 s 已基本收敛,30 s 再留余量。
    #   相对几千秒的任务时长,这点挂钟开销可忽略。
    #   ⚠ 局部VRP 侧几乎用不到它 —— 七个场景 775 次求解,池大小上限 13,
    #     全部走精确解;它只在池顶破阈值时才生效。
    vrp_time_limit_s: float = 30.0
    vrp_exact_max_targets: int = 13      # 精确解的池上限(见下表)
    vrp_gls_lambda: float = 100.0        # ortools GLS 惩罚系数(默认 0.1 太小,见上)
    vrp_span_coefficient: int = 1000     # min-max 强度,同 msim ortools_vrp
    solver: str = "exact"                # "exact" | "ortools" | "greedy"
    #
    # vrp_exact_max_targets 取 13 的依据(2026-08-25 人工裁决):
    #   (a) 覆盖率:七个场景 775 次局部VRP 求解,池大小上限恰为 13,无一超过;
    #   (b) 耗时:实测 k=12 → 171 ms, 13 → 378 ms, 14 → 800 ms, 15 → 1.64 s,
    #       16 → 3.49 s, 17 → 7.51 s。阈值处最坏 378 ms,相对 15 s 的规划周期
    #       有 40 倍余量,机载算力上站得住。
    #   调大阈值请连同这两条依据一起更新,别只改数字。

    # --- 仿真时钟 -------------------------------------------------------
    dt_s: float = 0.5
    max_mission_time_s: float = 3600.0
    settle_time_s: float = 60.0          # Leader 走完后再多跑一段,让 Follower 收尾

    def __post_init__(self):
        if self.acoustic_hop_s < 0.0:
            raise ValueError(f"acoustic_hop_s 不能为负: {self.acoustic_hop_s}")
        if self.plan_period_s <= 0.0:
            raise ValueError(f"plan_period_s 必须为正: {self.plan_period_s}")
        if self.usbl_period_s <= 0.0:
            raise ValueError(f"usbl_period_s 必须为正: {self.usbl_period_s}")
        # 一个 USBL 循环要占满 usbl_cycle_s 秒。周期比它还短的话,第 k+1 轮的询问会
        # 压在第 k 轮的回复上 —— 同一台 USBL 换能器上物理不可能,不静默容忍。
        if self.usbl_cycle_s > self.usbl_period_s + 1e-9:
            raise ValueError(
                f"USBL 一个定位循环 {self.usbl_cycle_s:.3f} s 长于定位周期 "
                f"{self.usbl_period_s:.3f} s,两轮会重叠。"
                f"要么调大 usbl_period_s,要么调小 acoustic_hop_s。")
        # 规划轮之间也不许套娃:上一轮的序列还没落地就开下一轮,语义无从定义。
        if self.plan_latency_s > self.plan_period_s + 1e-9:
            raise ValueError(
                f"规划下发时延 {self.plan_latency_s:.3f} s 长于规划周期 "
                f"{self.plan_period_s:.3f} s。")

    @property
    def n_followers(self) -> int:
        return len(self.follower_starts_ned)

    @property
    def usbl_cycle_s(self) -> float:
        """一个完整 USBL 定位循环的时长 = 逐台「询问 → 回复」= 2N 跳。

        ⚠ 硬纪律 4:这是**由固定时序解析计算的确定量**,不是"估计"。
        """
        return 2 * self.n_followers * self.acoustic_hop_s

    @property
    def plan_latency_s(self) -> float:
        """从规划时刻到序列落到 Follower 手里 = 求解余量 + 广播 1 跳。"""
        return self.acoustic_hop_s + self.plan_solve_s

    @property
    def first_plan_s(self) -> float:
        return 0.0 if self.first_plan_at_s is None else float(self.first_plan_at_s)

    @property
    def first_usbl_s(self) -> float:
        return 0.0 if self.first_usbl_at_s is None else float(self.first_usbl_at_s)

    @property
    def wait_lookahead_s(self) -> float:
        """判据 C 的提前量。默认 = 一个规划周期(濒危点至少还能被规划到一次)。"""
        return (self.plan_period_s if self.leader_wait_lookahead_s is None
                else float(self.leader_wait_lookahead_s))

    @property
    def leader_waits(self) -> bool:
        return bool(self.leader_wait_on_lagging_follower
                    or self.leader_wait_on_endangered_target
                    or self.leader_wait_on_turnaround)


@dataclass(frozen=True)
class FollowerRequest:
    """Follower → Leader 的定位回复(规格 §4:附带自己当前占用的 waypoint)。

    ⚠ `t_s` 是 Follower **采样并发出**的时刻,不是 Leader 收到的时刻 —— 回复本身
      还要在水里飞一跳。Leader 收到的时刻记在 `t_recv_s`。
    """
    t_s: float
    follower_id: int
    position_ned: np.ndarray             # (2,) [x_N, y_E],采样时刻的位置
    occupied_wp: int                     # waypoint_id;无则 WP_NONE
    t_recv_s: float = -1.0               # Leader 收到的时刻 = t_s + acoustic_hop_s


@dataclass(frozen=True)
class Assignment:
    """Leader → Follower 的下发序列(规格 §4)。新序列覆盖旧序列(D8)。

    ⚠ `t_s` 是**规划时刻** t_ref(Leader 收齐两台位置、开始求解的那一刻),
      同一轮里两台的 `t_s` 相同。真正生效的时刻是 `t_deliver_s = t_s + 一跳`。
    """
    t_s: float
    follower_id: int
    wp_ids: Tuple[int, ...]              # 真实 waypoint_id,按访问序;不含投影点
    points_ned: np.ndarray               # (k,2) 实际下发的点(含投影点,若有)
    has_projection: bool                 # 末尾是否是投影点
    pool_ids: Tuple[int, ...]            # 本次可规划池(诊断用)
    solver: str
    leader_north_m: float
    t_deliver_s: float = -1.0            # 广播落地、真机与影子同时生效的时刻

    @property
    def n_real(self) -> int:
        return len(self.wp_ids)

    @property
    def n_points(self) -> int:
        return int(self.points_ned.shape[0])


@dataclass(frozen=True)
class PlanRound:
    """一轮联合规划的完整记录(诊断与论文用,不参与任何决策)。"""
    round_idx: int
    t_start_s: float                     # 本轮起点(D16 起 == t_plan_s,规划不再自己轮询)
    t_plan_s: float                      # 取最近一次 USBL 定位、推算到此刻、开始求解
    t_deliver_s: float                   # 广播落地 = t_plan_s + plan_latency_s
    leader_north_m: float
    pool_size: int                       # 本轮可规划池大小(= 问题规模)
    n_assigned: Tuple[int, ...]          # 每台下发的**真实** waypoint 数(不含投影点)
    n_projection: int                    # 本轮补了几个投影点
    solver: str
    # ⚠ `time.perf_counter` 实测的**挂钟**耗时,取决于宿主机性能与 ortools 时间预算。
    #   它**不进入仿真时间轴**,也不进入任何数值结论 —— 仿真侧给求解留的余量是
    #   `MissionConfig.plan_solve_s`(默认 0)。两者别混。
    solve_wall_s: float
    # 本轮是不是 Leader 走完测线后强插的那一轮(不等周期,D15)
    forced_final: bool = False
    # 规划时刻每台的 USBL 定位**陈旧度**(s):从最近一次定位到 t_plan_s 的间隔。
    # 这段位移是由承诺队列解析推算出来的(D11 口径),不是外插估计。D16 解耦之后
    # 它由 `usbl_period_s` 决定,与 `plan_period_s` 无关 —— 这正是解耦的意义。
    fix_age_s: Tuple[float, ...] = ()

    @property
    def n_assigned_total(self) -> int:
        return int(sum(self.n_assigned))

    @property
    def max_fix_age_s(self) -> float:
        return max(self.fix_age_s) if self.fix_age_s else 0.0
