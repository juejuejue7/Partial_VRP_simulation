"""
contracts/ — A0 冻结的跨模块接口契约(草案,待人类 review 后冻结)。

本目录是全项目唯一的接口基准:所有跨 agent 的函数签名、数据结构(state dict 键、
动作编码、rollout 返回元组等)在此定死,只有签名/类型/docstring,行为方法体
`raise NotImplementedError`。建造 agent 在不互看实现的前提下并行实现,只对本目录负责。

唯一有权修改本目录的角色 = A0。其他 agent 发现契约不够用 → 提交 A0 裁决并广播,
禁止自行扩展。

模块划分:
    types.py     公共类型别名 + NED 坐标系约定(全项目坐标系唯一真相源)
    config.py    SimConfig 及子配置(世界/窗口/传感器/reward/rollout)
    geometry.py  第0/1层:Grid 变换、WindowRegion、粗网格形状
    physics.py   第2a层:RolloutEngine/RolloutResult、LeaderActor、相机传感器模型
    task.py      第2b层:走廊覆盖、期望熵降、BeliefUpdater(开关)、能耗均衡
    mdp.py       第3层:承诺队列(+解析量)、obs 组装、动作解码、env 协议
"""
