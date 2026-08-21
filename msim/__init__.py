"""
msim — 多AUV協調探査システム 修士 simulation 顶包。

权威设计基准:project_summary.md(做什么)、agent_team_harness.md(团队分工/验收)。
本包结构由 A0 在波次0 冻结;建造 agent(A2~A6)以 contracts/ 为唯一依据。

分层(单向自底向上,每层只依赖下层契约):
    第0层  config / geometry      ← 无依赖
    第1层  env_static             ← 依赖第0层
    第2a层 physics                ← 依赖第0/1层 + 根目录 AUV 类
    第2b层 task                   ← 依赖第0/1层
    第3层  mdp (Gymnasium env)    ← 依赖第0/1/2层
    第4层  policy (CleanRL)       ← 依赖第3层
    第5层  baselines / eval(旁路)← 仅依赖第0/1/2层

硬纪律(见 CLAUDE.md):
    - 对契约编程,不对实现编程。
    - 改接口必须经 A0。
    - 修士/博士差异只许出现在 belief 注入点。
    - 开环下的"可用时间/剩余距离"是承诺队列解析的确定量,非估计。
"""

__all__ = ["__version__"]

__version__ = "0.0.0"  # 波次0:骨架 + 冻结契约草案
