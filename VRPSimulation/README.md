# VRPSimulation — 真实 Mothra 热液场上的多 AUV 協調探査仿真

修士 simulation 此前只跑在合成簇状概率场上（`msim/`，world 200×100 m）。
本目录把 Endeavour 段 **Mothra 热液场的真实烟囱分布**落成一个静态基础世界：
经纬度 → NED 平面坐标、真实目标点、目標分布確率マップ、Leader 测线与滑动观测窗口，
再在其上跑 **1 Leader + 2 Follower 的声学握手式分区域 VRP 任务仿真**（**不做 RL**）。

**当前冻结的场地与窗口**：分析域 **500(N) × 100(E) m**，22 个目标（14 chimney + 8 mound），
观测窗口 **100×100 m 沿测线滑动**，Leader 单条测线（East=50，全长 500 m）。

设计基准仍是 `project_summary.md` / `agent_team_harness.md` / `architecture_blueprint.md`；
本仿真特有的裁决在 `vrpsim/contracts/DECISIONS.md`（D1–D20）。

---

## 数据

唯一输入：`waypoints/mothra_waypoints.csv`（UTF-8-BOM，CRLF，28 行）。

来源链路（只读，本仿真不改上游）：

```
Bethy_data/TABLE_SI.xlsx (Supplemental Table S1, 572 点)
  → Bethmetory_data_process/scripts/06_extract_vents.py   按 Mothra bbox 筛出 28 点
  → Bethmetory_data_process/scripts/07_make_waypoints.py  附局部米制坐标
  → VRPSimulation/waypoints/mothra_waypoints.csv
```

| 项 | CSV 原始 | 裁切后（D7，实际使用） |
|---|---|---|
| 目标 | 28 = 20 chimney + 8 mound | **22 = 14 chimney + 8 mound** |
| NED 范围 | North 76.8–632.5，East 6.7–195.2 m | North 76.8–442.1，East 6.7–94.8 m |
| `height_m`（高出海底） | 4 – 18 m | 4 – 18 m |
| 水深 | −2286.45 … −2253.78 m | −2286.45 … −2268.30 m |
| 最近邻间距 | min 2.7 / 中位 6.6 / max 85.9 m | — |
| 已命名复合体 | + Cauldron | Stonehenge、Cuchalainn、Crab Basin、Faulty Towers、Twin Peaks |

### 裁切（D7）：North [0,500) × East [0,100)

丢掉 6 个点，**全是 chimney**：

| wp | NED | h | 名称 | 丢弃理由 |
|---|---|---|---|---|
| 23 | N529.8 / E143.7 | 12 m | （无名） | 孤点，离最近目标 85.9 m；一个点就要 Leader 多飞一整条测线 |
| 24–28 | N610–632 / E169–195 | 4–11 m | Cauldron 区 | 与主群隔 80 m 空白，为它们要把测线拉长 130 m |

裁完后 East 跨度 100 m **恰好等于窗口宽**，Leader 单条测线即可扫全场。
原点不变，所以保留点的 NED 坐标逐位不变。`full_raster_crop()` 随时能建回全幅
653×294 / 28 点的世界做对照。

---

## 坐标系（D2）

**NED：x = North，y = East，z = Down**，轴向语义继承 `msim/contracts/types.py`，不另立一套。

原点 = Mothra 1 m 水深栅格的西南角：

| | 值 |
|---|---|
| UTM 9N (EPSG:32609) | E = 491801.0 m，N = 5307442.0 m |
| WGS84 (EPSG:4326) | lon = −129.109743763°，lat = 47.920247904° |
| 垂向 | z = 0 = 海面；D 向下为正 |

```
x_N = utm9n_northing − 5307442.0     # 0 .. 653   North，长边
y_E = utm9n_easting  −  491801.0     # 0 .. 294   East ，短边
z_D = −seafloor_depth_m              # +2253 .. +2286
```

与 1 m 水深栅格逐格 bit-exact 对齐：`col = int(y_E)`、`row = 652 − int(x_N)`。

### 投影代价（实测，不得默认为零）

| 项 | 值 |
|---|---|
| 网格北 vs 真北 | 0.080° |
| 尺度因子 k | 0.999601（399 ppm，相对 h=0 的椭球面） |
| 位置残差 | 0.822 m |
| **净距离误差（海底 h≈−2270 m）** | **44 ppm ≈ 4.3 cm/km ⇒ 全场最长边 653 m 上 < 3 cm** |

最后一行才是可接受性的论据：AUV 走在海底而非椭球面上，半径 R−|h| 处的地面距离
本就比椭球面短 356 ppm，方向与 UTM 的 −399 ppm 相反，几乎抵消。
`vrpsim/geodesy.py` 同时实现了椭球局部切平面路径（**不参与运行时**），
`tests/test_geodesy.py` 把这几个数钉成回归测试。

### ⚠ 轴序陷阱

源 CSV 的 `x_local_m` 是 **East**、`y_local_m` 是 **North**，与本项目 NED **正好相反**。
写反了场地整体转置，而且不会有任何报错。
`vrpsim/dataset.py` 的策略是 **不信列值，重算再对拍**：由 `lon/lat` 重新算 NED，
再与 CSV 米制列逐点核对（容差 1 mm），不一致当场抛。

---

## 结构

```
VRPSimulation/
├── vrpsim/
│   ├── contracts/          [A0 独占] 冻结契约,其它 agent 只读
│   │   ├── frames.py       NED 帧常量 + 原点 + 投影代价的回归基准
│   │   ├── dataset.py      MothraDataset / load_waypoints 契约
│   │   ├── config.py       MothraSimConfig / CropConfig / FieldBuildConfig
│   │   ├── mission.py      MissionConfig / FollowerRequest / Assignment / PlanRound
│   │   ├── lawnmower.py    LawnmowerConfig 全覆盖对照基线(D14)
│   │   ├── twophase.py     TwoPhaseConfig 二段式全局VRP 对照基线(D19)
│   │   ├── metrics.py      評価指標の定義(公式/口径/方向/applies_to)
│   │   ├── style.py        FigureStyle 出图样式(字号/线宽/配色/版面)
│   │   ├── DECISIONS.md    裁决记录 D1–D20 + 待裁决
│   │   └── REUSE_FROM_MSIM.md  可复用的 msim 符号白名单
│   ├── geodesy.py          [第0层] WGS84 <-> UTM9N <-> NED(纯 numpy,6 阶 Krüger)
│   ├── dataset.py          [第0层] CSV -> NED 目标表(含换轴与交叉核对)
│   ├── field.py            [第1层] 目標分布確率マップ p(x,y)
│   ├── world.py            [第1层] MothraWorld 静态世界(含 D7 裁切)+ save/load
│   ├── windows.py          [第1层] Leader 测线 + 后方滑动观测窗口(D8,静态几何)
│   ├── planner.py          [第2层] 选池 + min-max VRP(显式起点)+ 截断 + 投影补点
│   ├── agents.py           [第2层] Leader / Follower 运动学状态机
│   ├── tracking.py         [第2层] Leader 对队友的承诺队列解析推算(信息边界唯一入口)
│   ├── mission.py          [第3层] 主循环:USBL/规划双时钟 + Leader 等待(D10/D15/D16)
│   ├── lawnmower.py        [第3层] 对照基线1:3 台 Follower 分区全覆盖扫描(D14)
│   ├── twophase.py         [第3层] 对照基线2:二段式全局VRP(探査→VRP→観測,逐次,D19)
│   ├── metrics_util.py     [第2层] ⑥負荷均衡的唯一实现(三场景共用)
│   ├── report.py           [第2层] 按 contracts/metrics.py 出表(08/09 共用)
│   └── viz.py              [第1层] 出图与回放
├── waypoints/              输入 CSV(只读)
├── data/                   派生产物(可重建)
├── scripts/                01 构建 / 02 世界图 / 03 窗口诊断 / 04 任务仿真 / 05 结果图
│                           / 06 全覆盖基线 / 07 两场景对比 / 08 参数扫描
│                           / 09 三方法对比(D19) / 10 扫描汇总(D20)
├── tests/                  285 项验收
└── figures/
```

**依赖方向单向：`VRPSimulation → msim`，msim 永不反向 import**（由测试守住）。
复用的 msim 符号见 `vrpsim/contracts/REUSE_FROM_MSIM.md`；本包不改 msim 任何文件。

---

## 运行

仿真跑在 `auv_py310`（有 matplotlib / gymnasium / torch）。
换算不依赖 pyproj —— 自带 6 阶 Krüger 级数实现，实测与 pyproj 差 **2.8 纳米**。

```bash
PY=D:/nixingxing/Anaconda/envs/auv_py310/python.exe

$PY -m pytest VRPSimulation/tests -q          # 285 passed, 1 skipped(pyproj 对拍在 base 环境才跑)
$PY VRPSimulation/scripts/01_build_world.py   # -> data/mothra_world.npz + 校验摘要
$PY VRPSimulation/scripts/02_plot_world.py --names    # -> figures/mothra_world_ned.png
$PY VRPSimulation/scripts/03_plot_windows.py          # -> figures/mothra_windows.png
$PY VRPSimulation/scripts/04_run_mission.py           # -> data/mission.npz + 摘要
$PY VRPSimulation/scripts/05_plot_mission.py          # -> figures/mission_overview.png
$PY VRPSimulation/scripts/06_run_lawnmower.py         # -> data/lawnmower.npz + 摘要(对照基线)
$PY VRPSimulation/scripts/07_compare_scenarios.py     # -> figures/scenario_compare.png
$PY VRPSimulation/scripts/08_sweep.py                 # 24 组参数扫描 -> logs/
$PY VRPSimulation/scripts/09_compare_methods.py       # 三方法对比(D19) -> logs/methods_compare.*
sh  VRPSimulation/logs/reps/run_seq.sh                # 5 趟串行重复批(⚠ 别用 & 并发)
$PY VRPSimulation/scripts/10_sweep_summary.py         # 扫描汇总(D20)   -> logs/sweep_summary.*
```

**04 与 05 是分开的**：04 只算数、落盘 `data/mission.npz`；05 只读盘、出图。
所以**只改样式时不用重跑 04**（几秒钟的事变成不到一秒），只有改了任务参数
（速度 / 声学单跳 / 规划周期 / 窗口尺寸 / 求解器 …）才需要重跑 04 再跑 05。

`01` 的开关：`--res-m`（默认 1.0）、`--bandwidth-m`（默认 3.0）、
`--weight-mode {uniform,height}`（默认 uniform，height 属待裁决项）、
`--advance-m`（默认 50）、`--no-crop`（建全幅 653×294 世界做对照）。
`02`：`--names`、`--no-basemap`。`03`：`--advance-m`、`--show-windows`。
`04`：`--acoustic-hop`(默认 2.0 s) / `--plan-period`(默认 30 s) / `--dwell`(默认 10 s) …
     见下方「任务仿真 → 可调参数」。
`05`：`--snapshot-interval`（默认 100 s）、`--no-time-guides`。
`06`：`--swath`（默认 6.0 m）、`--n-vehicles`（默认 3）、`--no-boustrophedon`；
见下方「全覆盖对照基线（D14）」。
`07`：`--no-paths`。`08`：见「参数扫描」。
`09`：`--leader-speed` / `--plan-period`（决定提案方法用哪组参数）；见「三方法比较（D19）」。
`02/03/05/07` 共用样式开关：`--style` / `--dump-style` / `--font-scale` / `--dpi`。

用代码直接拿世界与窗口：

```python
from vrpsim import build_mothra_world, MothraSimConfig, enumerate_windows, leader_track

cfg = MothraSimConfig()
mw = build_mothra_world(cfg)
mw.field.shape            # (500, 100)，field[ix, iy]，ix 沿 North
mw.dataset.targets_ned    # (22, 2) = [x_N, y_E]
mw.to_gt_instance()       # 直接喂 msim 的 window / rollout / VRP baseline

track = leader_track(mw.world, cfg.lane_spacing_m)             # 单条测线的两个端点
snaps = enumerate_windows(mw.world, cfg.window, mw.dataset.targets_ned)
snaps[7].target_idx       # 第 8 次重解时窗口内的目标下标（10 个，最密的一窗）
```

---

## 世界参数（D5 / D7）

```python
WorldConfig(x_max_m=500.0, y_max_m=100.0, res_m=1.0)   # nx=500, ny=100 = 50,000 cell
```

`res_m = 1.0` 与 1 m 水深栅格同构，将来加地形层零重采样。
msim 合成场默认 0.2 m；真实场用 0.2 m cell 数太大，不划算。

概率场 = 各向同性高斯 KDE（σ = 3.0 m）+ max 归一化到 [0,1]，
默认路径**直接调用** `msim.env_static.field.field_from_targets` ——
真实场与合成场概率语义逐位同源，消融对比才成立。

⚠ **该场很峰化**：>0.5 的 cell 只有 198 个（0.40%），>0.05 的 2117 个（4.23%），
总概率质量 Σp = 560。若价值结算需要更宽的先验，先动 `--bandwidth-m`
（当前值沿用 msim 默认，未经裁决不改）。

---

## 观测窗口（D8）：100×100 m，沿测线滑动

```python
WindowConfig(look_back_m=100.0, width_m=100.0, advance_threshold_m=50.0)
lane_spacing_m = 100.0        # == 窗口宽 ⇒ Leader 单条测线（East=50，全长 500 m）
```

**滑动，不是固定分块。** 窗口是 Leader 正后方 100×100 m 的矩形；每推进
`advance_threshold_m` 触发一次**重解**，重解只涉及尚未被 Follower occupied 的路径点，
新序列下发后覆盖旧序列。

实测（半窗步长 50 m）：

| 项 | 值 |
|---|---|
| 重解次数 | 10（Leader North = 50, 100, …, 500） |
| 窗口内目标数 | min 0 / 中位 4 / **max 10** |
| 空窗 | 2/10（#0 North[0,50] 与 #5 North[200,300]） |
| 每个目标的可见次数 | **恰好 2 次，且连续** |

最后一行是选推进阈值的硬判据：半窗给每个目标 2 次被分配的机会，错过即永久错过。
调到 100 m（整窗）只剩 1 次、容错为零。见 `figures/mothra_windows.png` 面板 (c)。

⚠ D9：窗口边界**严格规则**、不按目标分布调 —— 代价是 North=400 附近把
Faulty Towers 复合体切开（wp14–18 在重解 #7，wp19 要等 #8）。已裁决接受并钉进测试。

`vrpsim/windows.py` 只给**静态几何**（枚举窗口、每窗有哪些目标），用于选窗口尺寸；
运行时的重解调度在 `vrpsim/mission.py`（见下节）。

---

## 出图朝向（D6）

East 放横轴、North 放纵轴、北朝上 ⇒
`imshow(field, origin="lower", extent=[0, y_max_E, 0, x_max_N])`，**不转置**
（row = ix = North 恰好就是纵轴）。`msim/contracts/types.py` 里"可视化时自行转置"
针对的是"把 North 放横轴"的画法，本项目不采用。
朝向由 `tests/test_world_field.py::test_plot_orientation_matches_ned` 与
`tests/test_viz.py::test_plot_world_extent_is_portrait` 钉住。

---

## 出图样式（D12）：`contracts/style.py`

字号 / 线宽 / 标记大小 / 配色 / 版面全部集中在 `FigureStyle` 一个 dataclass 里，
`02` `03` `05` 三个出图脚本共用同一套开关：

```bash
# 1. 导出一份带全部字段的模板
$PY VRPSimulation/scripts/05_plot_mission.py --dump-style my_style.json

# 2. 用编辑器改里面的数值（纯 JSON，不碰代码）

# 3. 用它重新出图（不需要重跑 04）
$PY VRPSimulation/scripts/05_plot_mission.py --style my_style.json

# 只想把字整体调大：不用建文件
$PY VRPSimulation/scripts/05_plot_mission.py --font-scale 1.4
$PY VRPSimulation/scripts/05_plot_mission.py --dpi 300      # 投稿用高分辨率
```

常改的字段：

| 想改什么 | 字段 |
|---|---|
| 上排六帧的刻度数字 | `font_size_tick_snapshot`（默认 7） |
| 下排时间轴的刻度数字 | `font_size_tick_panel_b`（默认 8） |
| 02/03 的标题 / 轴标签 / 图例 | `font_size_title` / `font_size_label` / `font_size_legend` |
| 目标点大小 | `target_marker_size`（**面积** pt²，翻倍要 ×4） |
| 三台 AUV 的标记大小 | `auv_marker_scale`（整体缩放）或 `leader_marker_size` / `follower_marker_size`（**直径** pt，翻倍 ×2） |
| 轨迹线粗细 | `traj_linewidth`（快照）/ `panel_b_follower_linewidth`（下排） |
| 两台 Follower 的颜色 | `follower_colors`（长度 = Follower 台数，不够会循环取用） |
| 已确认目标的分类配色 | `chimney_color` / `mound_color` |
| 图幅比例 | `fig_width_in` / `panel_b_height_in` / `row_gap_in`（图高自动反推） |
| 六帧之间的缝隙 | `frame_fill`（默认 0.88；**改它不会破坏时间轴对齐**，帧中心不动） |
| 地形底图 | `basemap_*`，见下节 |

三条硬约束：

1. **样式不影响任何数值**。换样式重出图 = 重新渲染同一份 `mission.npz`。
2. **JSON 里写错字段名会当场报错并提示最接近的正确名**，不会静默出一张"改了但没生效"的图。
3. **契约默认值 = 接管前各处的实际取值**，所以引入本契约不改变既有图
   （`figures/mission_overview.png` 与 `mothra_world_ned.png` 逐字节复现；
   `mothra_windows.png` 仅地图图例字号由 7.5 统一到 8.0）。想让图更紧凑请改自己的
   JSON，**不要改契约默认值** —— `tests/test_style.py` 钉着这条。

代码里直接用：

```python
from vrpsim.contracts.style import DEFAULT_STYLE, load_style
from vrpsim.viz import plot_mission_snapshot

st = DEFAULT_STYLE.with_overrides(traj_linewidth=3.0, target_marker_size=80.0)
st = load_style("my_style.json", font_scale=1.2)      # 或从文件读 + 统一放大字号
plot_mission_snapshot(ax, mw, timeline, 300.0, style=st)
```

### 地形底图（D13）：与 Bethmetory_data_process 同一套渲染

底图渲染 = **haxby 测深色带 + 排名直方图均衡 + 不过曝乘性晕渲**，
与 `Bethmetory_data_process/scripts/04_mothra_plot.py` 的正式方案逐式相同
（`vrpsim/viz.py::render_terrain_rgb`）。
`tests/test_basemap_render.py` 直接 import 上游脚本做**逐元素对拍**，两边不会悄悄分叉。

| 字段 | 默认 | 说明 |
|---|---|---|
| `basemap_mode` | `"haxby"` | `"gray"` = 中性灰，底色不抢戏 |
| `basemap_norm` | `"equalize"` | 排名均衡；`"linear"` = 与水深等比、可跨图比颜色 |
| `basemap_shade_strength` | 1.0 | 0 = 不压暗 |
| `basemap_vert_exag` | 2.0 | 越小光照越柔 |
| `basemap_sun_azimuth_deg` / `_altitude_deg` | 315 / 45 | 西北高照，制图惯例 |
| `basemap_cmap_lo` / `_hi` | 0 / 1 | 截色带；`hi<1` 砍掉最暖的橙端 |
| `basemap_lighten` | 0.0 | **本项目加的一档**：朝白提亮让底图退居背景，0 = 上游原样 |

两条口径：

- **`equalize` 让颜色与水深不等比，且映射依赖当前裁切范围** ⇒ 换裁切重出图，
  同一颜色对应的水深会变，**跨图比颜色不成立**。要跨图比就用 `"linear"`。
- 与上游成图的**唯一**差异是格距：上游渲 WGS84 栅格（dx 1.000 / dy 1.488 m），
  本项目渲 UTM 1 m 栅格（dx = dy = 1.0）。晕渲依赖格距，所以阴影不会逐像素相同 ——
  这是投影不同的必然结果，不是实现偏差。

彩色底图的代价：蓝色 Follower 轨迹正好压在深蓝的轴部裂谷上会看不清。
默认给轨迹加了深色描边（`traj_halo_linewidth = 1.8`，设 0 关闭）；
嫌底图抢戏就调 `basemap_lighten: 0.35`，要彻底让路就 `basemap_mode: "gray"`。

---

## 任务仿真（D10 / D15 / D16）：Leader 广域扫描 + 双 Follower 声学握手式分区域 VRP

```bash
$PY VRPSimulation/scripts/04_run_mission.py      # -> data/mission.{npz,_assignments.json,_timeline.csv}
$PY VRPSimulation/scripts/05_plot_mission.py     # -> figures/mission_overview.png
```

部署：Leader (0, 50) 沿长边测线；Follower0 (0, 0)、Follower1 (0, 100)。
窗口 100×100 m，**前边界恒与 Leader 位置重合**。

### 两条独立的时钟（D15 声学 / D16 解耦）

**(a) USBL 声学定位**，周期 `--usbl-period`（默认 15 s）。一个循环 = 2N 跳，
每跳 `--acoustic-hop`（默认 2.0 s）：

```
t0+0τ   L→F0  定位询问                                  ← 信道占用 τ
t0+1τ   F0 采样自身位置与占用点，回复                     ← 占用 τ
t0+2τ   L 收到 p0;  L→F1 定位询问                        ← 占用 τ
t0+3τ   F1 采样自身位置与占用点，回复                     ← 占用 τ
t0+4τ   L 收到 p1 —— 循环结束（8 s），静默 7 s 后再来一轮
```

**(b) 路径规划**，周期 `--plan-period`（默认 30 s，可调 45/60/90 …）：

```
t_plan     取最近一次 USBL 定位，按承诺队列解析推算到此刻（陈旧度 ≤ usbl_period）
           联合 min-max VRP：两台位置 + 窗口内未访问未占用点
           两台序列各截 ≤5 个，打包成一条广播                ← 占用 τ
t_plan+τ   F0、F1 **同一拍**收到，各自覆盖旧序列（D8）
```

**解耦的意义**：定位一直在跑、位置一直新鲜，而规划周期可以单独拉长做对比实验，
不牵动定位刷新率。D16 之前两者绑死（一轮握手里先轮询再规划），拉长规划周期就等于
让位置变陈旧，这组对比根本做不出来。USBL 与指令下行视为不同频段/换能器，
**不建模信道竞争**。

不足 5 个时把该台位置投影到窗口前边界，投影点作为序列末尾（**不占用**）。
Follower 逐个占用并前往，到点停 `--dwell`（默认 **10 s**）做目標観測。
Leader 走完测线后会**强插一轮**（不等周期）：窗口就此冻结，那是最后的分配机会。

⚠ **单跳 2.0 s 里传播只占 0.34 s**（最远 510 m / 1500 m/s），主体是包时长与保护间隔。
论文里必须写清，否则读者会拿声速反推出错误的作用距离。

### Leader 等待策略（D16）：不再匀速

0.5/0.5 等速下，Leader 匀速走完测线只要 1000 s，而完成全部観測至少要 1084 s ——
窗口会先冻结，南边的点永久错过（实测 **21/22**）。故 Leader 每拍用**自己的 DR 推算**
判两条，任一成立就原地停船：

| 判据 | 触发 | 关掉用 |
|---|---|---|
| **A** 落后队友 | 某台的推算位置落到窗口后沿之后（恢复要多进 5 m，防振荡） | `--no-wait-lagging` |
| **C** 濒危目标 | 窗口里还有**尚未被分配**的目标，再走一个规划周期就要掉出后沿 | `--no-wait-endangered` |

于是 `t_leader_finish` 从**输入**变成**结果**：

> **等待策略把「覆盖率约束」换成了「时间约束」。**
> 任务时间下界 = max(测线长/v_leader, t_route_lb) = 1084 s（实测 1193 s）。

消融（greedy，确定性）：

| 等待策略 | 覆盖 | 完成时刻 | 停船 |
|---|---|---|---|
| A + C（默认） | 22/22 | 1187.5 s | 128.5 s |
| 只开 A | 22/22 | 1187.5 s | 51.0 s |
| 只开 C | 22/22 | 1187.5 s | 128.5 s |
| **全关（匀速 Leader）** | **21/22** | **nan** | 0 s |

两条判据在本场地互为替代：单开任一条都够，完成时刻逐位相同。A 更省（51 vs 128.5 s），
C 更对症（直接盯"目标会不会永久出局"）。两条都开是零成本的保险 —— 完成时刻由
**最后一次観測**封顶，Leader 多停的那段不在关键路径上。

默认参数结果：**覆盖 22/22 = 100%**，**任务完成 1193 s**
（全域扫遍 1193 s / 末次観測 1158 s），Leader 停船 193 s（13.9%），
47 轮规划 / 93 个定位循环 / 419 条声学报文，min-max 航程 604.1 m，
定位陈旧度 ≤ 15 s，推算残差 **0.0 m（逐位相同）**。

### 规划周期对比（USBL 固定 15 s）

```bash
$PY VRPSimulation/scripts/04_run_mission.py --plan-period 45
```

| 规划周期 | 覆盖 | 完成时刻(greedy) | 完成时刻(ortools) | 规划轮 | 定位陈旧度上限 |
|---|---|---|---|---|---|
| 30 s | 22/22 | 1187.5 s | 1193.0 s | 47 | 15.0 s |
| **45 s** | 22/22 | **1128.0 s** | **1117.5 s** | 29 | 13.5 s |
| 60 s | 22/22 | 1149.0 s | 1269.0 s | 22 | 11.0 s |
| 90 s | 22/22 | 1203.5 s | 1189.0 s | 16 | 13.0 s |

- **曲线非单调，两个求解器都指向 45 s 最好**；太密和太疏都变差。
  ⚠ 机制尚未查清，**别在论文里编解释** —— 目前只能陈述现象。
- **定位陈旧度始终 ≤ `usbl_period_s`，与规划周期无关** —— 解耦确实生效了。
- 90 s（仅 16 轮）仍能全覆盖，说明承诺队列的开环执行相当鲁棒。

### ⚠ 速度比是可行性约束，不是口味问题

`feasibility_estimate()` 给出全覆盖的**必要条件**：

```
t_route_lb = 全局 min-max VRP 最长单机路线 / v_follower + 停留时间   ← 本场地 1084 s
t_leader   = 测线长 / v_leader（Leader 不停船时）                     ← 默认 1000 s
任务时间下界 = max(两者)                                            ← 1084 s
```

**D16 之前**（Leader 匀速）：`t_route_lb > t_leader` 就是死刑 —— 窗口先冻结，
更南的目标永久错过。**D16 之后**（Leader 会停船等队友）：这条不再判死，而是告诉你
`wait_required = True`，即这套参数下 Leader **必然要等**，实际完成时刻 =
测线长/v_leader + 累计停船时长（实测 1000 + 193 = 1193 s，比下界高 10%）。

只有把等待策略**关掉**（`--no-wait-lagging --no-wait-endangered`），
速度比才重新变成硬约束 —— 那时同一套参数实测掉到 21/22。
脚本每次运行都会重算这两个量并说明该走哪条路。

### 可调参数

`04_run_mission.py`：`--window-look-back` / `--window-width`（规格 §3）、
`--acoustic-hop`、`--usbl-period` / `--plan-period` / `--first-plan-at`（D15/D16）、
`--no-wait-lagging` / `--no-wait-endangered` / `--wait-release-margin`（D16 等待策略）、
`--max-seq`（§4）、`--dwell`（§5）、`--leader-speed` / `--follower-speed`、`--dt`、
`--solver {ortools,greedy}`、`--vrp-time-limit`、`--max-time`、
`--no-dwell-at-projection`、`--nav-drift` / `--seed`（见下）。

⚠ 共享运动学参数（`--dwell` / `--follower-speed` / `--dt`）的默认值**从
`MissionConfig` 取，不在脚本里写字面量** —— 06 也一样。这样两个场景不可能悄悄跑散
（`test_script_defaults_come_from_the_contract` 钉住）。

### 每轮的规模与求解耗时（D15）

`mission_assignments.json` 的 `plan_rounds[]` 逐轮记录
`t_start_s / t_plan_s / t_deliver_s / pool_size / n_assigned / n_projection / solve_wall_s`，
npz 里另有规整数组 `plan_stats (R,9)` 与 `comm_events (E,5)`。

⚠ `solve_wall_s` 是 `time.perf_counter` 的**挂钟**耗时，取决于宿主机与
`--vrp-time-limit`，**不进入仿真时间轴、不进入任何数值结论**（仿真侧的余量是
`plan_solve_s = 0`）。副作用：同一份输入两次运行的 npz **不再逐字节相同**。

⚠ **ortools 测不出难度差异**：GLS 元启发式**总会吃满时间预算**。实测池大小 1→7，
耗时齐刷刷 ~1003 ms（`--vrp-time-limit 1.0`）；把预算压到 0.02 s，仍是齐刷刷 21.3 ms。
想看真实的规模-耗时关系必须换 `--solver greedy`（无时间预算）：

| 池大小 | 1 | 2 | 3 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| 平均耗时 [ms] | 0.013 | 0.025 | 0.038 | 0.071 | 0.095 | 0.119 |

单调、近似线性 —— 这才是问题规模的真实代价。04 会自动打印这张对照表。

### 结果图 `figures/mission_overview.png`

**上排**：按固定间隔（默认 100 s，`--snapshot-interval`）的一组快照，共 6 帧。
**下排**：North–时间图，**x 轴与上排逐帧对齐** —— 第 i 帧的水平中心正对下排 t_i 的位置。

快照里目标点是三态：

| 状态 | 画法 |
|---|---|
| 未被 Leader 窗口扫过 | **不画** |
| 已扫到、尚未被 Follower 确认 | **统一空心圈,不分类型** |
| 已确认 | 按 chimney 红圆 / mound 琥珀三角 分实心 |

中间那一行对应锁定术语：**目標探査**（Leader 声呐初查）只知道"这里有东西"，
分不出 chimney / mound；要到 **目標観測**（Follower 相机确认）之后才分得清。
空心圈画成"白圈 + 深色描边"，是因为底图是灰度晕渲——纯白在亮地形上、纯黑在暗地形上
都会消失，双层描边才两头都跳得出来。

"未扫到就不画"是有依据的：`project_summary.md` §3 里 Leader 窗口是**地图揭示器**，
没被扫到的目标系统还不知道它存在，画出来等于泄露 ground truth。
揭示时刻由 `viz.sweep_times()` 从已有记录反推，不需要重跑 04。

默认参数下各帧「已揭示 / 已观测」：0/0 → 5/0 → 8/5 → 8/8 → 18/13 → **22/22**
（North 200–300 m 那段本来就没有目标，所以 t=200→300 揭示数不变）。

全图**只有图像和坐标轴**：无图例、无标题、无轴标签、无 colorbar。
配色（无图例时靠它认人，集中定义在 `viz.FOLLOWER_COLORS` / `viz.LEADER_COLOR`）：
Leader 青、Follower0 蓝、Follower1 紫、chimney 红、mound 琥珀。

### 记录（规格 §7）

`mission.npz` 逐时刻存：`follower_pos` (T,F,2)、`follower_queue` (T,F,6) 与
`follower_queue_xy`（waypoint 序列，`-2` = 投影点）、`follower_occupied`、
`follower_status`、`leader_north_m`、`window_north`、`follower_distance_m`、
`visited_count`。`mission_assignments.json` 存每次规划事件（池、下发序列、求解器）
与摘要；`mission_timeline.csv` 是人读版宽表。

### Leader 对队友位置的认知（D11 + D15）

每台上报的都是它**采样那一刻**的位置，而回复本身还要在水里飞一跳。到了规划时刻，
先上报那台已陈旧 3 跳、后上报那台陈旧 1 跳 —— 这段的位置、占用点、已完成的点，
全部由 `vrpsim/tracking.py` 的影子模型推算，**不读仿真真值**：

```
上报时刻的位置与占用点  +  Leader 自己发出的承诺队列  +  已知速度/停留时间
        └────────────────── 解析前推到当前时刻 ──────────────────┘
```

⚠ **默认（`--nav-drift 0`）下这是确定量，不是"估计"**：推算与真值逐位相同
（实测残差 9.7e-13 m，纯浮点噪声），因为 Follower 就是严格照 Leader 发的序列执行的。
这正是 `project_summary.md` §10 的"由承诺队列解析计算"，论文措辞不许写成"估计"。
**声学延迟不破坏这一点**：Leader 精确知道自己发了什么、何时到达、对方以什么速度执行。

⚠ 前提是**影子与真机在广播落地的同一拍装上同一条序列**。D15 之前的实现在规划时刻就
更新影子，τ>0 时影子会提前一跳沿新序列走，残差涨到米级，上面那句话随即失效。
`test_shadow_and_truth_get_the_sequence_on_the_same_tick`（把单跳拉到 10 s）钉住这条。

`--nav-drift`（按行进距离的比例，0.005 = 0.5% = 5 m/km）打开后才真的是估计。
它建模的是 **Leader 的认知误差**，不是 Follower 的执行误差。实测在默认节拍下
（信息最多陈旧 3 跳 = 6 s ⇒ 行进 ≤9 m），即使 2% 漂移也只有 0.18 m 误差，
相对目标间距中位 6.6 m 可忽略，分配决策不变——**要让它起作用得同时拉大
`--plan-period` 或 `--acoustic-hop`，这个方向尚未探索。**

### 规格留白与实现选择

见 `vrpsim/contracts/DECISIONS.md` D10 / D15。规格没说死的地方按"不产生自相矛盾行为"
取默认：已观测的点不再进池（否则任务不收敛，`revisit_visited` 可翻）。

"占用"的口径由 D15 收口（关闭待裁决 6）：**只有广播落地时刻正在前往的那一个**被冻结，
已下发未走到的队列尾每轮全部释放回池 —— 因为两台每轮一起被重新规划，min-max VRP
一次解出的两条路线天然互斥，不需要额外预留。旧的 `reserve_other_queue` 开关随之删除。

---

## 全覆盖对照基线（D14）：3 台 Follower 做 lawnmower

用来量化協調探査到底快多少。**同一片场地、同一批目标、同样 3 台 AUV**，
但取消异种分工：没有 Leader、没有观测窗口、没有请求与 VRP 规划，
3 台相机机把 100 m East 分成 3 条带各扫自己那份。

```bash
$PY VRPSimulation/scripts/06_run_lawnmower.py      # -> data/lawnmower.{npz,_summary.json,_timeline.csv}
$PY VRPSimulation/scripts/07_compare_scenarios.py  # -> figures/scenario_compare.png + lawnmower_paths.png
```

### 结果（默认参数，22 个目标，两边都 100% 覆盖）

| | 協調探査（VRP） | 全覆盖（lawnmower） |
|---|---|---|
| AUV | 1 Leader + 2 Follower | 3 Follower |
| 観測幅宽 | 声呐窗口 100 m / 相机点観測 | 相机 6 m ×18 条测线 |
| **任务完成时刻** | **1193.0 s** | **6146.0 s（5.15×）** |
| 最后一次観測 | 1158.0 s | 5321.0 s |
| 最长单机航程 | 604.1 m | 3027.8 m |
| 总航程 | 1160.3 m | 9083.3 m（7.83×） |

> 数字随 D16 更新（等速 0.5 m/s + Leader 等待策略）。D15 时是 541 s vs 2109 s = 3.90×，
> D14 时是 500 s vs 2064 s = 4.13×。速度改动对两个场景同时生效
> （`LawnmowerConfig.base` 内嵌同一份 `MissionConfig`），仍是同一把尺子。
> **优势反而扩大**：等速化对全覆盖扫描的惩罚（3028 m/台 ⇒ 3×）远大于对協調探査的
> 惩罚（604 m/台 ⇒ 2.4× + 停船）。

### 三条口径（照抄 D14，改代码前先读）

1. **「配置相同」是结构保证。** `LawnmowerConfig` 内嵌一份 `MissionConfig`（字段 `base`），
   速度 / 停留 / 到达容差 / 时钟 / 场景全部从它派生。往 `LawnmowerConfig` 里另加一个
   `follower_speed_mps` 会被 `test_no_second_source_of_truth_for_shared_params` 打红。
   ⚠ D15 踩过一次坑：契约改对了，但 `06_run_lawnmower.py` 的 `--dwell` 里手抄着旧的
   5.0，于是基线仍按 5 s 跑而**没有任何测试变红**。现在两个脚本的共享参数默认值都从
   `MissionConfig()` 取，由 `test_script_defaults_come_from_the_contract` 钉住。
2. **覆盖是几何确定量**，不是感知模型——没有检出概率 / 漏检 / 误分类（硬纪律 4）。
   扫到之后车**不离开测线**，在测线上该目标 North 处原地停 `dwell_time_s`（10 s）。
3. **比的是 `t_complete_s`，不是 `duration_s`。**
   VRP = `max(Leader 走完测线, 最后一次観測)`；lawnmower = 最后一台**走完全部测线**的时刻。
   lawnmower 用的不是最后一次観測（1834 s）——全覆盖扫描不走完就不能宣称扫完。
   `MissionResult.duration_s`（590 s）含 60 s 收尾余量，拿它比会给 VRP 多算一截。
   未全覆盖时 `t_complete_s` 按定义是 `nan`，`07` 会打印"跨场景时间不可比"并跳过时间面板，
   不会拿一个漏了目标的"完成时刻"去比。

### 三处刻意让对照组占便宜的取法（论文里要写出来）

- 幅宽取 6.0 m = msim `SensorConfig.footprint_lateral_m`，不取 5.4 m 的走廊宽
  （取 5.4 每台从 6 条测线变 7 条，再慢约 1/6）。`--swath 5.4` 可一键验证。
- 不建模转弯耗时（lawnmower 的转弯远多于 VRP 路线）。
- 停留点在测线上而非目标正上方 ⇒ 不为目标绕路。

### ⚠ 论文里必须写清的一句

「Leader 100 m vs Follower 6 m」不是给对照组穿小鞋，而正是被对比的东西：
100 m 是**声呐探査**幅宽（只能做目標探査），6 m 是**相机観測**幅宽（做不到 100 m），
两者不可互换。结论应表述为：**只有相机级传感器时全覆盖要 N 条测线；
引入一台宽幅声呐做线索机，可压缩成单条测线 + 定点観測。**

---

## 評価指標（D17 / D18 / D19）：`contracts/metrics.py` + `scripts/08_sweep.py`

### 先说一个会省掉很多无用功的事实

**只改规划周期时**，`coverage` / `visited` / `n_targets` 在各组之间是常数——
拿它们做对比表只会得到一列相同的数。那个维度上真正有区分度的只有三类：
**时间**（什么时候做完）、**成本**（航程 / 通信 / 空转）、**过程**（计划改了多少次）。
这条写成了反向测试 `test_terminal_metrics_do_not_discriminate_across_planning_period`。

⚠ **换成 `wait` on/off 这个维度就完全反过来了（D18）**：关掉 Leader 停车等待后
Follower 跟不上，目标被观测窗口甩在后面，覆盖率从 100% 掉到 **36.4%**。
那里最重要的一列就是 `visited`。

### 三方法比较（D19）：`scripts/09_compare_methods.py`

被比较的三种方法，**硬件都是 3 台 AUV**，唯一的变量是探査与観測怎么组织：

| 方法 | 编成 | 探査と観測 |
|---|---|---|
| **局部VRP（提案）** | 1 Leader + 2 Follower | **並行** |
| 二段式全局VRP | 1 Leader + 2 Follower | 逐次（先探明全部，再解**一次**全局 VRP）|
| lawnmower | 3 台（全部相机机）| 探査即観測（全覆盖扫描）|

⚠ 二段式的阶段1 仍需 Leader —— 声呐才能做目標探査，相机做不到 100 m 幅宽。
所以三者台数相同，**局部VRP 与二段式之间唯一的变量就是串行化**，是干净的受控对比。

```bash
$PY VRPSimulation/scripts/09_compare_methods.py
# → data/twophase.npz、logs/methods_compare.{csv,md}
#   logs/plan_cost.csv            逐轮的池大小 + 求解挂钟耗时（局部VRP 每轮 / 二段式 1 行）
#   figures/methods_compare.png   学会発表口径：2 面・英語・无子图标题・字号 x2.5
```

正文图**只留两面**（人工指定 2026-08-15）：左 = 観測完了本数の推移（逐次 vs 並行
最直观的一张），右 = ⑥負荷均衡（距離 vs 稼働時間）。`s/目標` 与「逐次 vs 並行」
两面已从图中去掉 —— 同样的数字在上面的表和 `methods_compare.md` 里都有。
方法名取自 `SCENARIO_NAMES_EN`、轴标签取自 `MetricSpec.axis_label("en")`，
**英译只在契约里写一处**；`test_figure_carries_no_cjk_text` 扫描出图函数的运行时
字符串，任何人把日文塞回轴标签或图例都会当场变红。

实测（Leader 0.5 m/s、Follower 0.5 m/s、停留 10 s、規劃周期 15 s）：

| 方法 | 観測 | 実測終了時刻 | **s/目標** | 対提案 | 平均観測完了 | 空転率 |
|---|---|---|---|---|---|---|
| **局部VRP（提案）** | 22/22 | **1099.0 s** | **50.0** | 1.00× | **733.0 s** | 0.6% |
| 二段式全局VRP | 22/22 | 2084.5 s | 94.8 | 1.90× | 1703.8 s | **51.4%** |
| lawnmower | 22/22 | 6146.0 s | 279.4 | 5.59× | 2664.8 s | 0.3% |

二段式 = 探査 1000.0 s + 観測 1084.5 s，**两段严格不重叠**。提案把这两段叠起来跑，
省下的就是较短的那一段；空転率 51.4% 就是串行的直接代价。

⚠ **提案不是全面占优，如实记：** 二段式在 ④通信コスト（0 条 vs 430 条）与
⑤計画品質（1 轮、0 次改道、序列利用率 **100%** vs 14.5%）上**全面优于提案**。
提案赢在时间，不在这两族。

#### ⚠ 时间效率**不能**用于三方法比较

`time_efficiency` 三列已由 `MetricSpec.applies_to=(SCENARIO_VRP,)` **结构性挡住**，
两条独立的理由：

1. **循环论证** —— 分子 `t_mission_reference_s` 本身就是一次全局 min-max VRP 的解，
   把「全局VRP方法」列为被比较对象后它按构造得 ≈1.0。
2. **对另外两法没有定义** —— 分子含 `测线长/v_leader`，lawnmower 没有 Leader 测线。

⇒ **跨方法時間効率 = `t_per_target_s`（単位目標あたり時間，不归一化）**，
配合 `t_observation_mean_s`（多早拿到结果 —— 串行 vs 并行的差别在这一列上更大）。
反向测试 `test_normalised_efficiency_is_scoped_to_vrp` 防止有人日后把它塞回去。

#### ⑥ 負荷均衡（D19）：min-max VRP 均衡的是距离，不是时间

| 方法 | 距離不均衡 ↓ | 時間不均衡 ↓ |
|---|---|---|
| 局部VRP（提案）| 2.8% | 0.7% |
| 二段式全局VRP | **0.2%** | **13.1%** |
| lawnmower | 0.0% | 0.7% |

二段式实测切成 **[18 点 452.1 m，4 点 451.2 m]** —— 距离几乎相等，时间差 141.8 s。
`solve_minmax_vrp_from` 优化的是**最长距离**，但完成时刻由**最慢那台的时间**决定，
而停留时长不进 VRP 的代价函数 ⇒ 停留占比一高两者就分叉。
`test_distance_balance_and_time_balance_can_disagree` 把这个事实钉死。

⚠ 因此**本对比表对二段式偏保守**：按时间均衡（给 OR-Tools 加 service-time 维度）
它可能更快。是否改求解器是**待裁决 7**，这几列就是给那个决定用的数据。

⚠ `jain_fairness*` 在 n=2 时**太钝**（时间差 13% 仍给出 0.995），
真正看得出差别的是 `load_imbalance_*` 那两列 —— 两者并存正是这个理由。

### 漏点组怎么进对比表（D18）

覆盖率一旦 < 1.0，「完成时刻」和「时间效率」各自分成了几把尺子，**别混用**：

| 键 | 公式 | 漏点时 | 什么时候看它 |
|---|---|---|---|
| `t_complete_s` | 全覆盖 ? `t_finish_s` : `nan` | `nan` | **判定任务是否达成** |
| `t_finish_s` | `max(全域扫遍, 最后一次観測完了)` | 有值 | 「这趟实际什么时候收的工」 |
| `time_efficiency` | 下界 / `t_complete_s` | `nan` | 严格口径，全覆盖组之间比 |
| `time_efficiency_obs` | 下界 / `t_finish_s` | 有值 | ⚠ **不能排名**，见下 |
| **`time_efficiency_cov`** | `coverage` × 上一行 | 有值 | **要横比就看这一列** |
| **`t_per_target_s`** | `t_finish_s` / `visited` | 有值 | **对漏点中性**，分子分母同时缩小 |

全覆盖时 `t_finish_s ≡ t_complete_s`、三个效率列**逐位相同**——
D18 之前各组的数一个没变，`test_finish_time_equals_complete_time_under_full_coverage` 钉住。

⚠ **`time_efficiency_obs` 实测会超过 1.0**：`waitoff_L1_*` 只做了 8~11/22，
这一列给出 **1.019~1.041**，字面意思是「比理论最优还快」。这不是 bug——
是「拿全任务的下界去除半任务的耗时」本身的荒谬，也是它**不能用来排名**的最直白证据。

⚠ `time_efficiency_cov` 是**线性外推**（假设剩下的目标与已做的一样费时），而实际漏掉的
是被 Leader 甩到窗口外、只有停船等待才够得着的点 ⇒ 它是**上界**不是预测。
它压小虚高但**不保证翻转排序**：greedy 下 waitoff 只漏 1/22，折扣把虚高从 +0.046
压到 +0.005 仍未翻转（`test_coverage_discount_narrows_the_flattery_but_need_not_reverse_it`）。

> **本轮最该记住的结论：効率列分不出 Leader 停车等待的高下，能分出的是 `visited`。**
> 単位目標あたり時間两者几乎相同（greedy 53.7 vs 54.0 s）——
> **等待策略买到的不是效率，是完整性。**

### 指标定义在哪

`vrpsim/contracts/metrics.py` 是**唯一真值源**。每个指标带
`公式 / 白话定义 / 单位 / 方向（↓越小越好 ↑越大越好 =应当恒定）`，
由 `tests/test_metrics.py` 钉住「声明了却没算 → 红」。
图表标签、CSV 表头、`logs/sweep_report.md` 的定义表全部从这一处取。

| 族 | 关键指标 | 一句话 |
|---|---|---|
| ① 時間効率 | `t_complete_s` / `t_finish_s`、**`time_efficiency{,_obs,_cov}`** | 三把尺子的分工见上一节；归一化量换场地换速度仍可比 |
| | `t_observation_mean/median/p90_s` | **和 `t_complete_s` 会给出不同排序**：重规划越频繁，目标平均被观测得越早，但全部做完反而越晚 |
| | **`t_per_target_s`** | 平摊到每个**实际观测到的**目标；**对漏点中性**，是漏点组能横比的另一列 |
| ② 稼働率 | **`duty_idle_frac`** | 队列空、原地干等的比例——纯浪费，最直接的「調査効率」读数 |
| | `duty_dwell_frac` | 应当恒定（≈目标数×停留），跳变 = 有别的东西被动了，免费的自检 |
| ③ 移動効率 | **`fleet_distance_m`** | **含 Leader**，跨场景比能耗只能用它（见下） |
| | `max_distance_m` | min-max VRP 直接在优化的量 = 编队续航的短板 |
| | `distance_per_target_m` | 每确认一个目标，整个编队付出多少米 |
| ④ 通信コスト | `channel_duty_frac` | **>100% ⇒ D16「不建模信道竞争」的假设已失效**，08 会告警 |
| ⑤ 計画品質 | **`reassignment_count`** | 点本来派给 A、后来改派给 B 的次数——A 已经走的那段白走了 |
| | `wp_issues_per_target` / `sequence_utilisation` | 1.0 = 一次到位；越大说明计划越反复 |
| 自检 | **`n_targets` / `visited`** | **D18 起是主指标**：22 里做成了几个。时间列必须和它一起读 |
| | `coverage` / `shadow_error_max_m` / `timed_out` | 不是效率指标，是「这一行能不能比」的前提 |

⚠ **`total_distance_m` 只算 Follower，不含 Leader 的 500 m**，而 lawnmower 那边
3 台全算。跨场景比总能耗时这等于白送掉 Leader 一整条测线，**对協調探査有利**。
所以新增了 `fleet_distance_m`：

| | 協調探査 | lawnmower |
|---|---|---|
| Follower 総航程 | 1160.6 m | 9083.3 m |
| **艦隊総航程（含 Leader）** | **1660.6 m** | **9083.3 m** |
| 単位目標あたり航程 | 75.5 m/目標 | 412.9 m/目標 |

### 参数扫描

```bash
# 默认网格 2 x 3 x 4 = 24 组(Leader 速度三档 = 慢于/等于/快于 Follower)
$PY VRPSimulation/scripts/08_sweep.py
# 自定义
$PY VRPSimulation/scripts/08_sweep.py --wait on,off --leader-speed 0.3,0.5,1.0 \
    --plan-period 15,30,60,90 --solver greedy --vrp-time-limit 0.1
```

产出：

```
logs/<run_id>/config.json     该组完整 MissionConfig（可复现）
             metrics.json    该组全部指标
             run.log         人读版
             mission.npz / _assignments.json / _timeline.csv
logs/sweep_metrics.csv        一行一组，向量指标展开成多列，直接进 Excel
logs/sweep_report.md          带定义表的人读对比表 + 16+1 条重跑命令
figures/sweep_compare.png     8 面板；实心=全覆盖，空心=未全覆盖
logs/reps/{seq0..4}/          5 趟串行独立重复批（D20；⚠ 必须串行，见下）
logs/sweep_summary.md/.csv    论文用：7 列代表指标 × 24 组，按 wait 拆两块
figures/sweep_summary.png     论文用：2×2 四面板，均值线 + ±1 标准差带
figures/sweep_summary_leader_speed.png  补充：横轴 = Leader 速度，看最优点
  ⚠ 两张图均为**全英文标注**、字号 = contracts/style.py 默认值 **×3**（学术汇报口径，
    见 --font-scale / --label-font-scale）；wait 失效的速度上只画一条线（两条完全重合）。
    正文图四面板 = 観測成功数 / 単位目標あたり時間 / **Leader 停船率** / **距離不均衡**
```

`run_id` = `wait{on|off}_L{Leader速度}_T{规划周期}`，例如 `waiton_L0.5_T30`。

⚠ lawnmower 对照行的**通信/計画两族记 `—` 而不是 0**——它没有 Leader、没有请求、
没有 VRP，写 0 会被误读成「通信开销为零的更优方案」。时间效率三列同理记 `—`：
分子那个下界是**協調探査**的全局 min-max VRP 下界，除 lawnmower 的完成时刻没有意义。

⚠ `--from-logs` 只重出表与图（10 分钟 → 2 秒），但遇到**缺新指标键的旧
`metrics.json` 会明确报错**要求重跑——默默填 `—` 会被误读成「不适用」，
其实只是那几列从没算过。

⚠ 汇总文件被 Excel 占着时会改写到 `sweep_metrics.new.csv` 并提示，
而不是让 10 分钟的扫描因为最后一步 `PermissionError` 白跑（仿真结果早已落盘）。

### 论文用汇总（D20）：`scripts/10_sweep_summary.py`

`08` 出的是全 40 余列的**工作用**表与 8 面板诊断图。论文正文要的是收窄版：

```bash
sh  VRPSimulation/logs/reps/run_seq.sh          # 5 趟串行重复批(~37 min)
$PY VRPSimulation/scripts/10_sweep_summary.py   # 只聚合,不跑仿真(~2 s)
```

**指标选取的原则 = 每个被扫的因子配一个能解释它的指标**，不是每族凑一个：

| 支配因子 | 指标 | 一句话 |
|---|---|---|
| `wait` | `visited` ★ | L≥0.5 时 on 恒 22/22、off 掉到 36.4%~95.5%；**L=0.3 时 on/off 都是 22/22 —— wait 在那一档失效** |
| `leader_speed` | `leader_hold_frac` | L=0.3 恒 **0.0%**（追不开）／L=0.5 为 4.6%~18.3%／L=1.0 为 **44%~62%** ⇒ 提速全变成停船等待 |
| `plan_period` | `duty_idle_frac` ★ | ⚠ **只有 L≥0.5 的 4 条单调↑**；L=0.3 的两条 12.2/8.3/17.6/22.6 非单调（那一档的空转是在等 Leader 扫出目标） |
| `plan_period` | `sequence_utilisation` ★ | **六条序列全部单调↑**；T=15 时 62%~91% 的下发点次被后续序列覆盖掉 |
| `plan_period` | `channel_duty_frac` | 几乎只是 T 的函数（0.67/0.60/0.57/0.56，**六条**序列同值），全组 56%~67% 逼近「不建模信道竞争」的边界 |
| 综合 | `t_per_target_s` ★ | 24 组里唯一都有定义、且对漏点中性的时间列 |
| 达成判定 | `t_complete_s` | 未全覆盖按定义 nan ⇒ 正文表**按 `wait` 拆两块** |

★ = 同时进 2×2 正文图。

⚠ **重复批必须串行跑。** ortools 在 `vrp_time_limit_s` 这个**挂钟**预算内跑多少迭代
取决于机器负载；并行的几趟共享同一负载 ⇒ 解逐位相同 ⇒ std 假性归零。
那不是「可复现」，是**样本相关**。

实测（5 趟串行独立）：**7 指标 × 24 组 = 168 格，std 全为 0**。全 40 余列扫一遍，
唯一摆动的是 `waiton_L1_T15.reassignment_count` ∈ {20, 22}，不进图表。
⇒ 正文图的 ±std 带宽度为 0 —— 这本身就是结果，不做「零宽就不画」的特判。

**Leader 速度的最优点 = Follower 速度（0.5 m/s），但两侧失效机制不同：**

| L [m/s] | 关系 | wait=on 最好 s/目標（4 周期的幅） | 失效机制 |
|---|---|---|---|
| 0.3 | 慢于 Follower | 75.8（幅 **0.0**） | **測線律速**：`t_complete` 恒 1667.0 = 500/0.3，规划周期与 wait 双双失效 |
| 0.5 | 等速 | **50.0**（幅 7.7） | —— |
| 1.0 | 快于 Follower | 50.5（幅 **19.1**） | Leader 停船 44%~62%；wait=off 时只做成 8~11/22 |

⚠ 别把它写成陡峭的 V：L=1.0 的**最好值**（50.5）几乎追平 L=0.5（50.0）。
真正的差别是**稳健性与代价** —— 幅 19.1 vs 7.7、停船 44%~62% vs 4.6%~18.3%、
关掉等待后被覆率 36%~50% vs 82%~95%。

---

## 本步**不含**

- 水深地形栅格与坡度 / 可航行性层（D3 明确排除）。地形只以两种形式露头：
  每个目标自带 `depth_D_m`；`NEDFrame.grid_index_1m()` 冻结了 1 m 栅格索引口径。
  `data/mothra_basemap.npz` 是**出图底色**，不是环境层。
- **任务调度层**：分区域 VRP 求解、"哪些点算 occupied"、重解与覆盖下发、
  Follower / rollout / 能耗。`vrpsim/windows.py` 只给几何。

## 待裁决

见 `vrpsim/contracts/DECISIONS.md` 末尾。

- **项 5 — `advance_threshold_m`**：仍未决，但**任务主循环不依赖它** —— 重规划由时间
  周期 `plan_period_s` 触发，这个空间阈值只用于 `windows.enumerate_windows` 的静态诊断。
- ~~**项 6 — "occupied" 的确切口径**~~ —— **已由 D15 关闭**：只有**广播落地时刻正在
  前往的那一个**被冻结，队列尾每轮全部释放回池。
- 仍开着的还有：项 2（概率场是否按 `height_m` 加权）、项 3（`res_m` 与 msim 对齐）、
  项 4（继承自 msim 的两处既有不一致）、D14 的 L1（幅宽 6.0 还是 5.4 m）与 L2（转弯耗时）。
