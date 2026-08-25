"""[第0层][A0 冻结] 評価指標の定義 —— 各参数组之间比较「調査効率」的唯一口径。

为什么要有这个文件
================================================================================
指标名散在 `summary()` 的字典键里、含义散在各处注释里,时间一长必然出现
「论文里写的定义」与「代码里算的东西」对不上。这里把**每个指标的公式、白话定义、
方向(越大越好还是越小越好)**集中成一份契约,并由
`tests/test_metrics.py::test_every_declared_metric_exists_in_summary` 钉住:
声明了却没算 → 红。图表标签、CSV 表头、README 表格全部从这里取。

⚠ 覆盖率不再是常数(D18 修订)
================================================================================
早先本文件写着「所有参数组都是 22/22 全覆盖,终点类指标没有区分度」。
**这条已经不成立**:`leader_wait_* = False`(Leader 不停车等待)的 8 组里 Follower
跟不上 Leader,窗口把没做完的目标甩在后面,覆盖率掉到 36%~95%。所以
**`n_targets`(目標総数)与 `visited`(観測成功数)现在是必须进表的主指标**,
不是凑数的自检项 —— 少了它们,读表的人看到「waitoff 组 500 s 就收工」会以为它更快。

由此带来的**两把尺子**(不要混用):

  `t_complete_s`  任务**完成**时刻。全覆盖才有值,否则 nan。**判定任务是否达成**。
  `t_finish_s`    **実測**終了時刻。同一公式,但不看覆盖率 —— 漏了目标也给数,
                  回答「这趟实际什么时候收的工」。⚠ 漏点组天生偏小。

时间效率同样一分为三:

  `time_efficiency`      下界 / t_complete_s   —— 严格;漏点组 nan
  `time_efficiency_obs`  下界 / t_finish_s     —— 漏点组也出数,**但会被漏点抬高**
  `time_efficiency_cov`  coverage x 上一行     —— 乘覆盖率折扣,三列里最适合横比

⚠ **但効率列本身分不出「Leader 停车等待」的高下。** 实测:waitoff 漏 1/22 却早收工
60 s,覆盖折扣把虚高从 +0.046 压到 +0.005,仍未翻转;単位目標あたり時間两者也几乎
相同(53.7 vs 54.0 s)。**等待策略买到的不是效率,是完整性** —— 能分出高下的那一列
是 `visited`。这就是 D18 把它升为主指标的理由。

⚠ 三方法比较时哪些指标能用(D19)
================================================================================
被比较的三种方法:

  `vrp`        本文提案:局部VRP —— Leader 探査与 Follower 観測**并行**
  `twophase`   対照1:二段式全局VRP —— 先探明全部目标,再解一次全局 VRP,**串行**
  `lawnmower`  対照2:従来手法 —— 3 台相机机分区全覆盖扫描

**`time_efficiency` 三列不能用于三方法比较**,已由 `applies_to=(SCENARIO_VRP,)`
在结构上挡住。两条独立的理由:

  1. **循环论证。** 它们的分子 `t_mission_reference_s` **本身就是一次全局
     min-max VRP 的解**。把「全局VRP方法」列为被比较对象后,它按构造就得 ≈1.0,
     胜负在定义里就决定了。
  2. **对另外两法根本没有定义。** 分子里含 `测线长 / v_leader`,
     那是局部VRP 特有的量;lawnmower 没有 Leader 测线。

⇒ **跨方法的時間効率口径 = `t_per_target_s`(単位目標あたり時間,不归一化)。**
  配合 `t_observation_mean_s`(多早拿到结果 —— 串行 vs 并行的差别在这一列上
  比在完成时刻上更大)。`time_efficiency*` 保留,但只在**局部VRP 的参数扫描内部**
  使用,而且**只在同一个 Leader 速度内部**(见下)。

⚠ **`time_efficiency*` 不能跨 Leader 速度比**(D20 修订)
================================================================================
分子 `t_mission_reference_s = max(测线长 / v_leader, 全局 min-max VRP 路线时间)`
是个**取最大值**,它会随 Leader 速度**换主导项**:

    L = 0.3 m/s   测线 1666.7 s  >  VRP 1035.0 s   ⇒ 分子 = 1666.7(测线主导)
    L = 0.5 m/s   测线 1000.0 s  <  VRP 1035.0 s   ⇒ 分子 = 1035.0(VRP 主导)
    L = 1.0 m/s   测线  500.0 s  <  VRP 1035.0 s   ⇒ 分子 = 1035.0(VRP 主导)

(数字为 Mothra + greedy 实测,2026-08-25 刷新。"测线长"自多车道化起取
 `leader_track` 的**折线总长**,Mothra 单车道时它就等于 North 跨度 500 m。)

⇒ 早先「各组共用同一个分子常数,纯粹是 1/t_finish 的统一缩放」这句话
  **只在 L ∈ {0.5, 1.0} 时成立**,加了 0.3 档后不再成立。
  L=0.3 那 8 组的分子大了 54%,効率列会把它们**系统性地抬高** ——
  但那只说明「快不过 Leader 自己走完测线」,不说明这档配置更好。

⇒ 规则:**同一个 Leader 速度内部**比 `time_efficiency*` 有效(分子确为常数);
  **跨 Leader 速度**请一律改看 `t_per_target_s` 或 `t_complete_s`。

指标的「有効区間」
================================================================================
稼働率一类的分母取 **t <= t_finish_s** 的那一段,不是整条时间线 ——
时间线尾部有 `settle_time_s`(默认 60 s)的收尾余量,那段全是 IDLE,
算进去会凭空抬高「空转率」,而且余量长度与被比较的参数无关。
D18 起分界从 `t_complete_s` 改为 `t_finish_s`,漏点组才不会退回整条时间线
被 settle 尾巴污染;全覆盖时两者恒等,既有各组的数一个不变。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# --- 指标族 ---------------------------------------------------------------
FAMILY_TIME = "time"        # 1 時間効率
FAMILY_DUTY = "duty"        # 2 稼働率
FAMILY_TRAVEL = "travel"    # 3 移動効率(航程)
FAMILY_COMM = "comm"        # 4 通信コスト
FAMILY_PLAN = "plan"        # 5 計画品質
FAMILY_BALANCE = "balance"  # 6 負荷均衡(D19)
FAMILY_CHECK = "check"      # 正确性自检(不是效率指标,但对比表里要盯着)

FAMILY_NAMES = {
    FAMILY_TIME: "① 時間効率",
    FAMILY_DUTY: "② 稼働率",
    FAMILY_TRAVEL: "③ 移動効率",
    FAMILY_COMM: "④ 通信コスト",
    FAMILY_PLAN: "⑤ 計画品質",
    FAMILY_BALANCE: "⑥ 負荷均衡",
    FAMILY_CHECK: "自检",
}

# --- 场景(方法)标识 ------------------------------------------------------
# `MetricSpec.applies_to` 用这几个值决定某指标在哪些方法的表里出现。
# 不适用的格子记 `—`,**不记 0** —— 写 0 会被误读成"这项开销为零的更优方案"。
SCENARIO_VRP = "vrp"              # 本文提案:局部VRP(探査と観測が**並行**)
SCENARIO_TWOPHASE = "twophase"    # 対照1:二段式全局VRP(**逐次**)
SCENARIO_LAWNMOWER = "lawnmower"  # 対照2:従来の全覆盖 lawnmower
ALL_SCENARIOS = (SCENARIO_VRP, SCENARIO_TWOPHASE, SCENARIO_LAWNMOWER)

SCENARIO_NAMES = {
    SCENARIO_VRP: "局部VRP(提案)",
    SCENARIO_TWOPHASE: "二段式全局VRP",
    SCENARIO_LAWNMOWER: "lawnmower",
}

# 学会発表用の図は**中日文字を一切載せない**(D20 の口径)。方法名も
# `MetricSpec.label_en` と同じく**契約側に一箇所だけ**置く —— 出図スクリプトに
# 英訳を書くと、方法が増えたときに表と図で名前が食い違う。
SCENARIO_NAMES_EN = {
    SCENARIO_VRP: "Local VRP (proposed)",
    SCENARIO_TWOPHASE: "Two-phase global VRP",
    SCENARIO_LAWNMOWER: "Lawnmower",
}

# --- 方向 -----------------------------------------------------------------
LOWER_BETTER = "low"        # 越小越好
HIGHER_BETTER = "high"      # 越大越好
SHOULD_BE_FLAT = "flat"     # 应当在各组之间基本恒定;变了说明有别的东西被动了

BETTER_MARK = {LOWER_BETTER: "↓", HIGHER_BETTER: "↑", SHOULD_BE_FLAT: "="}


# --- 单位的英文写法 --------------------------------------------------------
# 英文图里不能出现「s/目標」这种混排。它是 `unit` 的**纯函数**,所以做成映射 +
# 派生属性,而不是让 40 个 spec 各写一遍(写一遍就多一处会漂移的地方)。
UNIT_EN: Dict[str, str] = {
    "-": "-",
    "s": "s",
    "m": "m",
    "個": "count",
    "件": "msgs",
    "回": "count",
    "点": "points",
    "点次": "issues",
    "s/目標": "s/target",
    "m/目標": "m/target",
    "件/目標": "msgs/target",
    "回/目標": "issues/target",
}


@dataclass(frozen=True)
class MetricSpec:
    """一个评价指标的完整定义。"""
    key: str            # `MissionResult.summary()` 里的键名
    label_zh: str       # 中文标签
    label_ja: str       # 论文/图里的日文标签
    label_en: str       # 学术汇报(英文图表)用的标签
    unit: str           # 单位;无量纲用 "-"
    formula: str        # 一行公式,直接对应实现
    definition: str     # 白话:这个数到底在说什么
    better: str         # LOWER_BETTER / HIGHER_BETTER / SHOULD_BE_FLAT
    family: str
    source: str = "summary"   # summary = 由 summary() 直接给
                              # sweep   = 需要额外输入,由 08_sweep.py 算
    fmt: str = "{:.2f}"       # 表格里的格式
    vector: bool = False      # 值是"每台一个"的列表(CSV 里展开成多列)
    # 这个指标在哪些方法的表里出现(D19)。默认三方法通用。
    # ⚠ 收窄它是**结构性地**防止口径滥用,不是排版偏好 —— 例如
    #   `time_efficiency*` 的分子本身就是一次全局 VRP 的解,放进三方法表
    #   会让「全局VRP方法」按构造得 1.0,是循环论证。见文件头。
    applies_to: Tuple[str, ...] = ALL_SCENARIOS

    @property
    def unit_en(self) -> str:
        """英文单位。未登记的单位原样返回 —— 宁可露出一个没翻的单位,
        也好过悄悄替换成一个错的。"""
        return UNIT_EN.get(self.unit, self.unit)

    def axis_label(self, lang: str = "en") -> str:
        """图的轴标签。`lang` ∈ {en, zh, ja}。**图表标签一律从这里取**,
        不许在脚本里手写 —— 手写的那份必然先漂移。"""
        if lang == "zh":
            return f"{self.label_zh} [{self.unit}]"
        if lang == "ja":
            return f"{self.label_ja} [{self.unit}]"
        return f"{self.label_en} [{self.unit_en}]"


METRICS: Tuple[MetricSpec, ...] = (
    # ================= ① 時間効率 =========================================
    MetricSpec(
        key="t_complete_s",
        label_zh="任务完成时刻",
        label_ja="任務完了時刻",
        label_en="Mission completion time",
        unit="s",
        formula="全覆盖 ? t_finish_s : nan",
        definition="任务真正**达成**的时刻。**判定任务是否完成、跨场景比时间用这个**。"
                   "未全覆盖时按定义是 nan —— 漏了目标就没有「完成」可言;"
                   "那种情况看 t_finish_s 并同时读 visited。"
                   "⚠ 不是 duration_s:那个含 settle_time_s 的收尾余量。",
        better=LOWER_BETTER, family=FAMILY_TIME, fmt="{:.1f}"),
    MetricSpec(
        key="t_finish_s",
        label_zh="実測終了时刻",
        label_ja="実測終了時刻",
        label_en="Observed finish time",
        unit="s",
        formula="max(全域扫遍时刻, 最后一个目标観測完了时刻)",
        definition="这趟**实际收工**的时刻,与 t_complete_s 同一公式但**不看覆盖率**,"
                   "所以漏点组也有数。全覆盖时两者逐位相同。"
                   "⚠ **不能单独横比**:漏点组少做几个目标当然早收工 —— "
                   "waitoff 组 500 s 收工不是因为它快,是因为它只做了 8/22。"
                   "必须与 visited / coverage 同读,或改看 t_per_target_s。",
        better=LOWER_BETTER, family=FAMILY_TIME, fmt="{:.1f}"),
    MetricSpec(
        key="time_efficiency",
        label_zh="时间效率(严格)",
        label_ja="時間効率(厳密)",
        label_en="Time efficiency (strict)",
        unit="-",
        formula="t_mission_reference_s / t_complete_s",
        definition="离理论最优还差多少。分子是 feasibility_estimate 给的乐观下界 "
                   "max(测线长/v_leader, 全局 min-max VRP 路线时间),分母是实测。"
                   "1.0 = 打满下界(不可能达到);0.90 = 比理论极限多花 11%。"
                   "⚠ **不能跨 Leader 速度比**(D20):分子是个 max(),"
                   "L=0.3 时由「测线长/v_leader = 1666.7 s」主导、"
                   "L≥0.5 时由「VRP 路线 1035.0 s」主导 —— 换了主导项就不再是"
                   "同一把尺子,L=0.3 那几组会被系统性抬高 54%。"
                   "**同一个 Leader 速度内部**它是常数分母的归一化量,可比;"
                   "跨速度请改看 t_per_target_s。"
                   "⚠ 未全覆盖 ⇒ nan(任务没完成,谈不上效率)。",
        better=HIGHER_BETTER, family=FAMILY_TIME, source="sweep", fmt="{:.3f}",
        applies_to=(SCENARIO_VRP,)),
    MetricSpec(
        key="time_efficiency_obs",
        label_zh="时间效率(実測)",
        label_ja="時間効率(実測)",
        label_en="Time efficiency (observed)",
        unit="-",
        formula="t_mission_reference_s / t_finish_s",
        definition="分母换成实际收工时刻,**漏点组也出数**。"
                   "⚠ **会被漏点抬高,不是能直接横比的效率**:分子是按 22 个目标全做完"
                   "算的下界,分母却只是「做完手上那几个」的收工时刻,漏得越多越好看。"
                   "⚠ **它可以超过 1.0** —— 实测 waitoff_L1_* 组只做了 8~11/22,"
                   "这一列给出 1.02~1.04,「比理论最优还快」。这不是 bug,正是"
                   "「拿全任务的下界去除半任务的耗时」这件事本身的荒谬之处,"
                   "也是最直白的证据:**这一列不能用来排名**。"
                   "它只回答:在这趟实际收工的时间里,相对全任务下界处在什么位置。"
                   "要横比请看下一行。",
        better=HIGHER_BETTER, family=FAMILY_TIME, source="sweep", fmt="{:.3f}",
        applies_to=(SCENARIO_VRP,)),
    MetricSpec(
        key="time_efficiency_cov",
        label_zh="时间效率(覆盖折扣)",
        label_ja="時間効率(被覆率補正)",
        label_en="Time efficiency (coverage-adjusted)",
        unit="-",
        formula="coverage x time_efficiency_obs = 下界 / (t_finish_s / coverage)",
        definition="把「漏点」的代价折算进时间后的效率,**三列里最适合横比的一列**。"
                   "分母 = 按本组实测的単位観測あたり時間、外推到把全部目标做完所需的时间;"
                   "漏一半就等于分母翻一倍。全覆盖时它与 time_efficiency 恒等。"
                   "⚠ **它压小虚高,但不保证翻转排序**:线性外推假设剩下的目标与已做的"
                   "一样费时,而实际漏掉的是被 Leader 甩到窗口外、只有停船等待才够得着的点 —— "
                   "所以它是**上界**不是预测。实测 greedy 下 waitoff 只漏 1/22,"
                   "折扣把虚高从 +0.046 压到 +0.005 却仍未翻转。"
                   "⇒ **効率列分不出 wait on/off 的高下,能分出的是 visited。**",
        better=HIGHER_BETTER, family=FAMILY_TIME, source="sweep", fmt="{:.3f}",
        applies_to=(SCENARIO_VRP,)),
    MetricSpec(
        key="t_observation_mean_s",
        label_zh="平均観測完了时刻",
        label_ja="平均観測完了時刻",
        label_en="Mean observation completion time",
        unit="s",
        formula="mean(visit_time_s)",
        definition="每个目标被観測完的时刻的平均。**它和 t_complete_s 会给出不同的排序**"
                   " —— 重规划越频繁,目标平均被观测得越早,但全部做完反而越晚。"
                   "关心「尽快拿到大部分结果」看它,关心「什么时候能收工」看 t_complete_s。",
        better=LOWER_BETTER, family=FAMILY_TIME, fmt="{:.1f}"),
    MetricSpec(
        key="t_observation_median_s",
        label_zh="観測完了时刻中位数",
        label_ja="観測完了時刻の中央値",
        label_en="Median observation completion time",
        unit="s",
        formula="median(visit_time_s)",
        definition="一半目标完成観測的时刻。比均值抗离群点 —— 少数几个偏远目标拖尾时,"
                   "均值会被拉高而中位数不会。",
        better=LOWER_BETTER, family=FAMILY_TIME, fmt="{:.1f}"),
    MetricSpec(
        key="t_observation_p90_s",
        label_zh="観測完了时刻P90",
        label_ja="観測完了時刻の90パーセンタイル",
        label_en="P90 observation completion time",
        unit="s",
        formula="percentile(visit_time_s, 90)",
        definition="90% 的目标完成観測的时刻。看「拖尾有多长」:它与中位数的差距,"
                   "就是最后那批难啃的目标额外花掉的时间。",
        better=LOWER_BETTER, family=FAMILY_TIME, fmt="{:.1f}"),
    MetricSpec(
        key="t_per_target_s",
        label_zh="単位目標あたり所要时间",
        label_ja="単位目標あたり所要時間",
        label_en="Time per target",
        unit="s/目標",
        formula="t_finish_s / visited",
        definition="平摊到每个**实际观测到的**目标身上的任务时间。"
                   "**对漏点中性** —— 少做几个点,分子(早收工)和分母(観測成功数)"
                   "一起变小,不像 t_finish_s 那样白送一个「更快」的假象。"
                   "所以未全覆盖的组要横比效率,看这一列或 time_efficiency_cov。",
        better=LOWER_BETTER, family=FAMILY_TIME, fmt="{:.1f}"),

    # ================= ② 稼働率 ===========================================
    MetricSpec(
        key="duty_transit_frac",
        label_zh="移動中の割合",
        label_ja="移動中の割合",
        label_en="Transit fraction",
        unit="-",
        formula="全 Follower 的 STATUS_TRANSIT 拍数 / (机数 x 有効区間拍数)",
        definition="Follower 在「赶路」上花掉的时间比例。它本身不是越低越好 —— "
                   "路总要走;真正的浪费是 idle。",
        better=SHOULD_BE_FLAT, family=FAMILY_DUTY, fmt="{:.1%}"),
    MetricSpec(
        key="duty_dwell_frac",
        label_zh="定点観測中の割合",
        label_ja="定点観測中の割合",
        label_en="Dwell (observation) fraction",
        unit="-",
        formula="全 Follower 的 STATUS_DWELL 拍数 / (机数 x 有効区間拍数)",
        definition="真正在做目標観測(相机确认)的时间比例。"
                   "⚠ **这个数应当在各组之间基本恒定**:分子约等于 目标数 x dwell_time_s,"
                   "是固定的。某一组明显跳变 = 有别的东西被动了(停留时长、投影点停留、"
                   "或者漏了目标),是一条免费的正确性自检。",
        better=SHOULD_BE_FLAT, family=FAMILY_DUTY, fmt="{:.1%}"),
    MetricSpec(
        key="duty_idle_frac",
        label_zh="待機(空転)の割合",
        label_ja="待機時間の割合",
        label_en="Idle fraction",
        unit="-",
        formula="全 Follower 的 STATUS_IDLE 拍数 / (机数 x 有効区間拍数)",
        definition="**队列空、原地干等的时间比例 —— 纯浪费,这是最直接的「調査効率」读数。**"
                   "它变大只有两个原因:规划周期太长(等下一批任务),"
                   "或者一轮里能分到的点太少(池空)。",
        better=LOWER_BETTER, family=FAMILY_DUTY, fmt="{:.1%}"),
    MetricSpec(
        key="duty_productive_frac",
        label_zh="稼働率",
        label_ja="稼働率",
        label_en="Productive fraction",
        unit="-",
        formula="1 - duty_idle_frac",
        definition="Follower 在干活(赶路或観測)的时间比例。idle 的补数,"
                   "单独列出来是因为论文里「稼働率」通常写成正向量。",
        better=HIGHER_BETTER, family=FAMILY_DUTY, fmt="{:.1%}"),
    MetricSpec(
        key="leader_hold_frac",
        label_zh="Leader停船率",
        label_ja="Leader 停船率",
        label_en="Leader wait rate",
        unit="-",
        formula="leader_holding 为真的拍数 / 全时间线拍数",
        definition="Leader 为等队友而原地停船的时间比例(D16)。"
                   "⚠ 它**不一定**是坏事:完成时刻由最后一次観測封顶时,"
                   "Leader 多停的那段根本不在关键路径上。要和 t_complete_s 一起读。",
        better=LOWER_BETTER, family=FAMILY_DUTY, fmt="{:.1%}"),
    MetricSpec(
        key="max_busy_time_s",
        label_zh="最长単機稼働時間",
        label_ja="最長単機稼働時間",
        label_en="Max vehicle busy time",
        unit="s",
        formula="max_i T_i,T_i = 第 i 台観測機の稼働時間(非 IDLE の拍数 x dt)",
        definition="**一番長く働いた観測機が、実際に動いていた時間。**"
                   "`duty_idle_frac` が「割合」なのに対しこちらは**絶対量** —— "
                   "電池・耐久の律速はこの秒数の方で決まる。"
                   "⚠ **Leader は含まない**:対象は観測を担う機体"
                   "(提案・二段式は Follower 2 台、lawnmower は 3 台全部)。"
                   "⚠ 任務長そのものではない:待機(IDLE)を差し引いた実働時間なので、"
                   "二段式のように前半ずっと待つ方式では任務長より大幅に短く出る。"
                   "⑥ 負荷均衡の `*_time*` 三列と同じ `busy_time_s` から算出、"
                   "口径は三場景で逐位同一。",
        better=LOWER_BETTER, family=FAMILY_DUTY, fmt="{:.1f}"),

    # ================= ③ 移動効率(航程) ==================================
    MetricSpec(
        key="fleet_distance_m",
        label_zh="艦隊総航程",
        label_ja="艦隊総航行距離",
        label_en="Fleet travel distance",
        unit="m",
        formula="Leader 航程 + Σ 各 Follower 航程",
        definition="**全部 AUV 加起来走了多少米 —— 跨场景比能耗就用这个。**"
                   "⚠ 与 total_distance_m 的区别是它**含 Leader**。lawnmower 场景没有 "
                   "Leader,3 台车的和就是艦隊総航程;協調探査若只报 Follower 的和,"
                   "等于白送掉 Leader 的一整条测线,对比就不公平了。",
        better=LOWER_BETTER, family=FAMILY_TRAVEL, fmt="{:.1f}"),
    MetricSpec(
        key="total_distance_m",
        label_zh="Follower総航程",
        label_ja="Follower 総航行距離",
        label_en="Follower travel distance",
        unit="m",
        formula="Σ 各 Follower 航程(**不含 Leader**)",
        definition="只统计做目標観測的那几台。看「観測作业本身的移动成本」时用它;"
                   "**跨场景比总能耗请改用 fleet_distance_m**,否则会漏掉 Leader。",
        better=LOWER_BETTER, family=FAMILY_TRAVEL, fmt="{:.1f}"),
    MetricSpec(
        key="max_distance_m",
        label_zh="最长单机航程",
        label_ja="最大単機航行距離",
        label_en="Max single-vehicle distance",
        unit="m",
        formula="max(各 Follower 航程)",
        definition="跑得最多的那一台走了多少。**这就是 min-max VRP 直接在优化的量** —— "
                   "它决定了最先耗尽电量的那台机器能撑多久,是编队续航的短板。",
        better=LOWER_BETTER, family=FAMILY_TRAVEL, fmt="{:.1f}"),
    MetricSpec(
        key="leader_distance_m",
        label_zh="Leader航程",
        label_ja="Leader 航行距離",
        label_en="Leader travel distance",
        unit="m",
        formula="leader_north_m[-1] - leader_north_m[0]",
        definition="Leader 沿测线走的距离。**应当恒等于测线长(500 m)** —— "
                   "停船不产生位移,所以它与是否等待、等多久都无关。变了说明测线被改了。",
        better=SHOULD_BE_FLAT, family=FAMILY_TRAVEL, fmt="{:.1f}"),
    MetricSpec(
        key="per_follower_distance_m",
        label_zh="各AUV航程",
        label_ja="各 AUV の航行距離",
        label_en="Per-vehicle travel distance",
        unit="m",
        formula="每台 Follower 各自的累计航程",
        definition="逐台的航程明细,CSV 里展开成 per_follower_distance_m_0/1/… 多列。"
                   "配合 jain_fairness 看负载分得均不均:两台差得多说明 min-max 没起效。",
        better=LOWER_BETTER, family=FAMILY_TRAVEL, fmt="{:.1f}", vector=True),
    MetricSpec(
        key="distance_per_target_m",
        label_zh="単位目標あたり航程",
        label_ja="単位目標あたり航行距離",
        label_en="Distance per target",
        unit="m/目標",
        formula="fleet_distance_m / visited",
        definition="**每确认一个目标,整个编队要付出多少米。**"
                   "目标数不同的场景之间比移动效率就用它 —— 这是最能直接说明"
                   "「協調探査比全覆盖省在哪」的一个数。",
        better=LOWER_BETTER, family=FAMILY_TRAVEL, fmt="{:.1f}"),
    # ================= ⑥ 負荷均衡(D19) ===================================
    # 为什么单独成族:min-max VRP 均衡的是**距离**,而任务完成时刻由**最慢那台的
    # 时间**决定。停留时长占比一高,两者就分叉 —— 把四列并排放进表里,
    # 这个隐藏假设就自己显形,不需要谁在注释里解释。
    MetricSpec(
        key="jain_fairness",
        label_zh="距離均衡度",
        label_ja="航行距離の公平性 (Jain)",
        label_en="Distance fairness (Jain)",
        unit="-",
        formula="(Σd)² / (n · Σd²),d = 各机航程",
        definition="Jain 公平性指数,1.0 = 各台航程完全相同,1/n = 全压在一台上。"
                   "min-max VRP 的直接效果就体现在这里。"
                   "⚠ **n 小时很钝**:两台差 13% 也能给出 0.995 —— "
                   "要看出差别请读 load_imbalance_* 那两列。",
        better=HIGHER_BETTER, family=FAMILY_BALANCE, fmt="{:.3f}"),
    MetricSpec(
        key="jain_fairness_time",
        label_zh="時間均衡度",
        label_ja="稼働時間の公平性 (Jain)",
        label_en="Busy-time fairness (Jain)",
        unit="-",
        formula="(ΣT)² / (n · ΣT²),T = 各机稼働時間(非 IDLE 的时长)",
        definition="与上一行**同一个公式、不同的量**。分母用的是各机真正在干活"
                   "(移動 + 停留)的时长,不是任务总时长 —— 用总时长的话所有机都一样,"
                   "量不出任何东西。**决定完成时刻的是这一列,不是距离那一列。**",
        better=HIGHER_BETTER, family=FAMILY_BALANCE, fmt="{:.3f}"),
    MetricSpec(
        key="load_imbalance_distance_frac",
        label_zh="距離不均衡",
        label_ja="航行距離の不均衡",
        label_en="Distance imbalance",
        unit="-",
        formula="(max_i d_i − min_i d_i) / max_i d_i,i 遍历全部 Follower",
        definition="相对不均衡,0 = 完全均衡。比 Jain 灵敏得多,是两台时真正能看的那列。"
                   "min-max VRP 优化的就是它,所以它通常很小。"
                   "**d_i 的口径**:第 i 台 Follower 从下水到任务结束的**累计航程** [m],"
                   "= 逐拍位移的欧氏长度之和(`follower_distance_m[-1, i]`),"
                   "含投影点腿与重规划改道,**不含 Leader**(它走的是测线,不参与均衡)。"
                   "max_i d_i <= 0(全队没动过)时按 0.0 处理。",
        better=LOWER_BETTER, family=FAMILY_BALANCE, fmt="{:.1%}"),
    MetricSpec(
        key="load_imbalance_time_frac",
        label_zh="時間不均衡",
        label_ja="稼働時間の不均衡",
        label_en="Busy-time imbalance",
        unit="-",
        formula="(max T − min T) / max T",
        definition="**本族的要害指标。** 它测的是「有机先干完、然后干等着」—— "
                   "正是 min-max 目标选错对象后的失效形态。"
                   "与上一行一起读:二段式实测切成「18 点 452.1 m / 4 点 451.2 m」—— "
                   "距離不均衡仅 0.2%,時間不均衡却有 13.1%。"
                   "⇒ **min-max VRP 均衡的是距离,而任务时间由最慢那台决定,"
                   "停留占比高时两者会分叉。**"
                   "⚠ 反过来它小**不等于**均衡算法更好:各机全程不空转时,"
                   "稼働時間自然都约等于任务时长(提案就是这种情况,Mothra 实测"
                   "距離 4.42% / 時間 2.29%,原因是逐次重规划持续把活摊开)。"
                   "跨场景看这条更清楚:提案在 7 个场景里時間不均衡 0.15%~2.94%,"
                   "而二段式 0.20%~13.09% —— 见 logs/Comparasion_0825.md。",
        better=LOWER_BETTER, family=FAMILY_BALANCE, fmt="{:.1%}"),

    # ================= ④ 通信コスト =======================================
    MetricSpec(
        key="n_comm_messages",
        label_zh="声学报文总数",
        label_ja="音響通信回数",
        label_en="Acoustic messages",
        unit="件",
        formula="USBL 询问 + 回复 + 序列广播 的条数",
        definition="整个任务发了多少条声学报文。直接对应通信能耗与信道占用。",
        better=LOWER_BETTER, family=FAMILY_COMM, fmt="{:.0f}"),
    MetricSpec(
        key="channel_duty_frac",
        label_zh="信道占空比",
        label_ja="音響チャネル占有率",
        label_en="Acoustic channel occupancy",
        unit="-",
        formula="n_comm_messages x acoustic_hop_s / duration_s",
        definition="声信道被占用的时间比例。"
                   "⚠ **超过 1.0 就说明「不建模信道竞争」这个假设已经失效** —— "
                   "报文总时长超过了任务时长,现实里必然排队。08_sweep 会对此告警。",
        better=LOWER_BETTER, family=FAMILY_COMM, fmt="{:.1%}"),
    MetricSpec(
        key="messages_per_target",
        label_zh="単位目標あたり報文数",
        label_ja="単位目標あたり通信回数",
        label_en="Messages per target",
        unit="件/目標",
        formula="n_comm_messages / visited",
        definition="平摊到每个目标身上的通信开销。用来回答「多确认一个目标要多聊几句」。",
        better=LOWER_BETTER, family=FAMILY_COMM, fmt="{:.1f}"),

    # ================= ⑤ 計画品質 =========================================
    MetricSpec(
        key="n_plan_rounds",
        label_zh="规划轮数",
        label_ja="計画回数",
        label_en="Planning rounds",
        unit="回",
        formula="整个任务解了几次 VRP",
        definition="求解次数。它乘以 solve_wall_mean_s 就是全任务的计算开销。",
        better=SHOULD_BE_FLAT, family=FAMILY_PLAN, fmt="{:.0f}"),
    MetricSpec(
        key="pool_size_mean",
        label_zh="平均池大小",
        label_ja="平均候補点数",
        label_en="Mean candidate pool size",
        unit="点",
        formula="mean(每轮可规划池的大小)",
        definition="每次求解面对的问题规模。**这才是反映求解难度的量** —— "
                   "solve_wall_s 在 ortools 下只反映时间预算(见 D16)。",
        better=SHOULD_BE_FLAT, family=FAMILY_PLAN, fmt="{:.2f}"),
    MetricSpec(
        key="wp_issue_total",
        label_zh="总下发点次",
        label_ja="総配信点数",
        label_en="Total waypoint issues",
        unit="点次",
        formula="Σ 每条 Assignment 里真实 waypoint 的个数",
        definition="所有规划轮加起来一共下发了多少个 waypoint(同一个点反复下发算多次)。"
                   "理想情况 = 目标数(每个点派一次就到位)。",
        better=LOWER_BETTER, family=FAMILY_PLAN, fmt="{:.0f}"),
    MetricSpec(
        key="wp_issues_per_target",
        label_zh="每目标下发次数",
        label_ja="目標あたり配信回数",
        label_en="Waypoint issues per target",
        unit="回/目標",
        formula="wp_issue_total / n_targets",
        definition="平均每个目标被下发了几次。**1.0 = 一次到位,越大说明计划越反复。**"
                   "D8 规定「新序列覆盖旧序列」,所以规划越频繁这个数越大。",
        better=LOWER_BETTER, family=FAMILY_PLAN, fmt="{:.2f}"),
    MetricSpec(
        key="sequence_utilisation",
        label_zh="序列利用率",
        label_ja="配信系列の利用率",
        label_en="Sequence utilisation",
        unit="-",
        formula="n_targets / wp_issue_total",
        definition="下发出去的点次里,有多大比例最终转化成了一次真实観測。"
                   "上一项的倒数,写成正向的百分比便于放进论文表格。",
        better=HIGHER_BETTER, family=FAMILY_PLAN, fmt="{:.1%}"),
    MetricSpec(
        key="reassignment_count",
        label_zh="改道次数",
        label_ja="再割当て回数",
        label_en="Reassignments",
        unit="回",
        formula="下发时该 waypoint 的上一个归属是**另一台** Follower 的次数",
        definition="**真正有代价的那种计划变更**:一个点本来派给了 A,后来改派给 B —— "
                   "A 可能已经朝它走了一段,那段路白走。"
                   "同一台重复下发同一个点不算(它本来就在往那儿去)。",
        better=LOWER_BETTER, family=FAMILY_PLAN, fmt="{:.0f}"),
    MetricSpec(
        key="reassigned_wp_count",
        label_zh="改道过的目标数",
        label_ja="再割当てされた目標数",
        label_en="Reassigned waypoints",
        unit="点",
        formula="至少被改派过一次的 waypoint 个数",
        definition="上一项按「点」而不是按「次」计数。两者一起看能区分"
                   "「少数几个点被反复改」和「很多点各改了一次」。",
        better=LOWER_BETTER, family=FAMILY_PLAN, fmt="{:.0f}"),
    MetricSpec(
        key="solve_wall_mean_s",
        label_zh="平均求解耗时",
        label_ja="平均求解時間",
        label_en="Mean solver wall time",
        unit="s",
        formula="mean(每轮 time.perf_counter 实测的 VRP 求解耗时)",
        definition="⚠ **挂钟量,不进入仿真时间轴,不构成任何数值结论**(仿真侧的余量是 "
                   "plan_solve_s = 0)。且 ortools 的元启发式**总会吃满时间预算**:"
                   "实测池大小 1 到 7 耗时全一样 —— 想看规模与耗时的真实关系必须换 "
                   "--solver greedy。反映难度请用 pool_size_mean。",
        better=SHOULD_BE_FLAT, family=FAMILY_PLAN, fmt="{:.4f}"),

    # ================= 自检(不是效率指标,但对比表里必须盯着) =============
    MetricSpec(
        key="n_targets",
        label_zh="目標総数",
        label_ja="目標総数",
        label_en="Total targets",
        unit="個",
        formula="場景内 waypoint 的个数",
        definition="本次任务里一共有多少个目标要観測。**同一场地各组恒为 22** —— "
                   "放进表里是为了让 visited 有分母、让读者一眼看到「22 里做成了几个」,"
                   "而不是只看到一个百分比。它变了就说明场景被换掉了。",
        better=SHOULD_BE_FLAT, family=FAMILY_CHECK, fmt="{:d}"),
    MetricSpec(
        key="visited",
        label_zh="観測成功数",
        label_ja="観測成功数",
        label_en="Targets observed",
        unit="個",
        formula="visit_time_s 非 nan 的目标个数",
        definition="**真正被 Follower 相机确认(目標観測)完成的目标数**,分母是 n_targets。"
                   "这是本组「干成了多少活」的绝对量。"
                   "⚠ 时间类指标必须和它一起读:waitoff 组收工早只是因为它做得少。",
        better=HIGHER_BETTER, family=FAMILY_CHECK, fmt="{:d}"),
    MetricSpec(
        key="coverage",
        label_zh="覆盖率",
        label_ja="被覆率",
        label_en="Coverage",
        unit="-",
        formula="visited / n_targets",
        definition="観測成功数占目标总数的比例。**Leader 停车等待(wait=on)时恒为 100%;"
                   "关掉等待后掉到 36%~95%** —— Follower 跟不上,目标被窗口甩在后面。"
                   "一旦小于 1.0:该组 t_complete_s 是 nan(任务未达成),"
                   "时间效率要改看 time_efficiency_cov。",
        better=HIGHER_BETTER, family=FAMILY_CHECK, fmt="{:.1%}"),
    MetricSpec(
        key="shadow_error_max_m",
        label_zh="推算残差",
        label_ja="推算残差",
        label_en="Shadow-propagation residual",
        unit="m",
        formula="max |影子推算位置 - 真值|,取遍全部规划时刻 x 全机",
        definition="⚠ 不是效率指标,是**信息边界的自检**。nav_drift = 0 时它必须是 0"
                   "(逐位相同) —— 一旦冒头就说明影子拿到了不该拿的信息,"
                   "「承诺队列解析计算是确定量」这条结论随之失效(CLAUDE.md 硬纪律 4)。",
        better=LOWER_BETTER, family=FAMILY_CHECK, fmt="{:.1e}"),
    MetricSpec(
        key="timed_out",
        label_zh="超时未终止",
        label_ja="タイムアウト",
        label_en="Timed out",
        unit="-",
        formula="跑到 max_mission_time_s 仍未满足正常终止条件",
        definition="⚠ 为真时该组**结果不可信**,不是「跑得慢」而是「没跑完」。",
        better=LOWER_BETTER, family=FAMILY_CHECK, fmt="{}"),
)

METRIC_BY_KEY: Dict[str, MetricSpec] = {m.key: m for m in METRICS}

# 由 summary() 直接提供的指标键(其余需要额外输入,由 08_sweep.py 计算)
SUMMARY_METRIC_KEYS: Tuple[str, ...] = tuple(
    m.key for m in METRICS if m.source == "summary")


def metrics_of(family: str) -> Tuple[MetricSpec, ...]:
    return tuple(m for m in METRICS if m.family == family)


def metrics_for(*scenarios: str) -> Tuple[MetricSpec, ...]:
    """给定若干方法,返回**对它们全部适用**的指标(D19)。

    三方法对比表就用 `metrics_for(*ALL_SCENARIOS)` —— 它会自动把
    `time_efficiency*` 挡在外面(那三列 `applies_to=(SCENARIO_VRP,)`),
    从结构上消灭"用全局VRP的解去评价全局VRP方法"这个循环论证。
    不传参数 = 不过滤,返回全部。
    """
    if not scenarios:
        return METRICS
    return tuple(m for m in METRICS
                 if all(s in m.applies_to for s in scenarios))
