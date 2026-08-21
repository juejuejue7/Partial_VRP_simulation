# Mothra 热液场地形数据处理

从 `Bethy_data/EndeavourAUVSouthCentral1.asc` 中裁出 Mothra 热液场子区,重投影为
各向同性 1 m 栅格,并渲染展示图。

## 源数据

### 1. 测深栅格 `EndeavourAUVSouthCentral1.asc`

| 项 | 值 |
|---|---|
| 格式 | AAIGrid `.asc`,单波段 float32 |
| 尺寸 | 6353 × 1368 |
| nodata | `-99999`(**必须先掩膜**,否则会被重采样核拖进邻域,污染滤波与形态学) |
| CRS | 无内嵌 → 手动指定 **EPSG:4326** |
| cellsize | `1.338111292e-05` deg |
| 实尺度 @lat 47.923 | 经向 **1.0001 m** / 纬向 **1.4878 m**(各向异性) |
| 有效率 | 全图仅 28.93% 是有效测深条带 |

### 2. 热液口坐标表 `TABLE_SI.xlsx`

Supplemental Table S1 — Chimneys determined from AUV map。单表 `forPublication`,
572 个数据点(487 chimney + 85 mound),覆盖整个 Endeavour 段。

表中夹着 `Segment AV1/AV2a/AV2b` 三行分节标题和一行脚注,经纬度列非数值 ——
提取时按"经纬度必须同时为数值"筛掉,**不按行号硬切**。

`Vent Field Name` 列有一处拼写不一致:`High Rise` 48 个 + `HIgh Rise` 1 个。
不影响 Mothra 的提取,但若后续要按场名分组需先归一化。

## Mothra 裁剪范围

lon `-129.10974` ~ `-129.10583`,lat `47.92026` ~ `47.92612`

范围内共 **28 个热液口**(20 chimney + 8 mound),高度 4–18 m。
按几何落点筛出的 28 个,与表中 `Vent Field Name == 'Mothra'` 的 28 个**完全一致**,
零错配 —— 这反过来印证了该 bbox 就是 Mothra 场。已命名的 17 个覆盖
Faulty Towers、Cuchalainn、Crab Basin、Stonehenge、Twin Peaks、Cauldron 等复合体。

## 产物

`outputs/`

| 文件 | 内容 |
|---|---|
| `mothra_wgs84.tif` / `.asc` + `.prj` | 裁剪结果,EPSG:4326,438 × 293,保持源分辨率 |
| **`mothra_utm9n_1m.tif`** | **分析用主产品**:EPSG:32609,653 × 294,正方形 1 m 像元 |
| `mothra_utm9n_1m.asc` + `.prj` | 同上,AAIGrid 格式 |
| `mothra_utm9n_1m.npz` | 纯 numpy 打包,供无 rasterio 的环境读取 |
| **`mothra_waypoints.csv`** | **路径规划用 waypoint 表**,28 行,附 `mothra_waypoints_meta.json` |
| **`mothra_bundle.npz`** | **水深 + 热液口打包在一起的单文件**,自洽,推荐后续分析读它 |
| `mothra_vents.csv` | 28 个热液口完整属性表(UTF-8-BOM,Excel 可直接开) |
| `mothra_vents.geojson` | 同上,EPSG:4326,供 GIS |
| `endeavour_context.npz` | 全测区 1/3 降采样底图,**仅供上下文示意,不用于定量分析** |
| `mothra_meta.json` | 元数据侧车文件 |

`mothra_bundle.npz` 的键:水深侧 `z_utm1m` / `z_wgs84` 及各自的 transform、bounds、
epsg;热液口侧 `vent_lon/lat`、`vent_easting/northing`、`vent_height_m`、
`vent_seafloor_depth_m`、`vent_morphology`、`vent_name`、`vent_sequence_id`、
`vent_col_utm1m` / `vent_row_utm1m`(可直接拿去索引 `z_utm1m`)。

### 路径规划 waypoint 表

`mothra_waypoints.csv` — 28 行纯表(无注释行,`read_csv` 直接可读):

`waypoint_id, sequence_id, lon, lat, type, height_m, x_local_m, y_local_m,
seafloor_depth_m, utm9n_easting, utm9n_northing, name`

**局部平面坐标系**(定义见 `mothra_waypoints_meta.json`):原点 = UTM 9N 栅格西南角
`E491801.0 / N5307442.0`,x 向东、y 向北,单位米。与 1 m 栅格严格对齐:

```python
col = int(x_local_m)
row = (nrows - 1) - int(y_local_m)     # nrows = 653
```

> **为什么除经纬度外还给米制坐标。** 本区 1 度经度 ≈ 74.6 km 而 1 度纬度 ≈ 111.2 km,
> 相差 1.49 倍。直接拿经纬度当平面坐标算距离,东西向会被高估约 49%。
> 规划器直接用 `x_local_m / y_local_m` 即可,不必自己处理各向异性。

> **`waypoint_id` 不是访问顺序**,只是 1..28 的稳定标签,顺序沿用源表 `sequence_id`,
> 便于回溯到原始文献表。访问次序由规划器决定。

> **两个文件的 `seafloor_depth_m` 取值来源不同。** waypoint 表取自 **UTM 1 m 栅格**
> 上述格子的值(保证规划器按局部坐标索引 `z_utm1m` 拿到的值与表逐位相同);
> `mothra_vents.csv` 取自 **WGS84 原生栅格**(溯源用)。两者中位差 0.25 m、
> 陡坡处最大 1.25 m,属重采样正常差异。

> **`height_m` 与 `seafloor_depth_m` 是两回事,不要合并。**
> 前者是表里给的烟囱高出海底的高度;后者是本项目栅格在该点采到的海底水深
> (负向下)。AUV 多波束未必分辨得出烟囱本体。

两个子区 nodata 均为 **0%**,水深 −2294.66 ~ −2197.49 m,起伏 **97.2 m**,
实际范围 294 m(E-W)× 653 m(N-S)。

`figures/`

| 文件 | 内容 |
|---|---|
| **`mothra_plot.png`** | **主图**(`04_mothra_plot.py`):haxby 晕渲地形 + 经纬度轴 + 比例尺 + 28 个热液口标记 |
| **`fig02_context_endeavour.png`** | **测区总览**(`08_context_plot.py`):同一套 haxby 配色 + Mothra 范围框 + 经纬度轴,无 colorbar/无文字 |
| `mothra_shade_compare.png` | 光照参数对比表(挑数值用) |
| `mothra_color_compare.png` | 色带截断/归一化对比表(挑数值用) |
| `mothra_plot_viridis.png` | 早期版本成图:viridis + 线性归一化(脚本已被取代) |
| `fig01_mothra_relief_{light,dark}.png` | 早期版本:蓝色阶 + 等深线 + 指北针 + 图注 |
| `fig03_mothra_3d.png` | 早期版本:三维透视 |

## 运行

本机两个环境能力互补,故流水线分两段(不需要改动任何环境):

| 环境 | rasterio / pyproj | matplotlib |
|---|---|---|
| conda `base` (py3.9) | ✅ | ❌ Pillow 9.0.1 的 `_imaging` DLL 加载失败 |
| conda `auv_py310` | ❌ | ✅ |

```bash
PY310=D:/nixingxing/Anaconda/envs/auv_py310/python.exe

# 1. 裁剪 + 重投影 (base)
python scripts/01_crop_reproject.py

# 2. 提取热液口 + 与水深打包 (base, 需 openpyxl)
python scripts/06_extract_vents.py

# 3. 生成路径规划 waypoint 表 (只需 numpy)
python scripts/07_make_waypoints.py

# 4. 主图 + 测区总览 (auv_py310)
$PY310 scripts/04_mothra_plot.py
$PY310 scripts/08_context_plot.py

# 5. 产物校验 (base) —— 44 项检查
python scripts/03_verify_outputs.py

# 可选: 参数对比表 / 早期带图注的成套图
$PY310 scripts/05_shade_compare.py --what shade
$PY310 scripts/05_shade_compare.py --what color
$PY310 scripts/02_plot_mothra.py
```

**注意执行顺序:`06` 依赖 `01` 的产物**(要用裁剪 bbox 和栅格采样水深),
`04` 若发现 `mothra_bundle.npz` 存在会优先读它以叠加热液口。

主图 `04` 的开关:

| 开关 | 默认 | 作用 |
|---|---|---|
| `--vents {morphology,uniform,none}` | `morphology` | 按 chimney-mound 分符号 / 统一符号 / 不画 |
| `--vent-labels` | 关 | 给已命名的热液口加名称标注 |
| `--no-legend` | 关 | 不画图例(分类标记下符号将无从解读,慎用) |
| `--shade` | `1.0` | 阴影混合权重,0 = 无阴影 |
| `--vert-exag` | `2.0` | 垂直夸张 |
| `--sun-alt` | `45` | 光源高度角(度) |
| `--cmap-lo` / `--cmap-hi` | `0.0` / `1.0` | 色阶截断 |
| `--gamma` | `1.0` | 归一化指数,<1 摊开深水端(**颜色不再与水深等比**) |
| `--no-hillshade` / `--name` | — | 纯色阶填色 / 换输出文件名 |

`01` 可选 `--resampling {nearest,bilinear,cubic,cubic_spline,lanczos}`,
`02` 可选 `--lang {zh,en}` 与 `--only {relief,context,3d}`。

## 处理决策

**为什么重投影到 UTM 9N。** 源数据在地理坐标下经纬向实尺度差 1.49 倍。形态学结构元
若直接在原格网上跑,东西向和南北向量到的物理尺度不一致。UTM 9N 中央经线是 −129°,
而 Mothra 在 −129.108°,几乎贴着中央经线,投影畸变可忽略。

**为什么默认 bilinear 而不是 cubic。** 纬向 1.49 m → 1.0 m 是上采样。cubic 在陡坎处
会 overshoot,凭空造出局部极大;而形态学 top-hat / opening 恰恰会把这种虚假极大
当成热液丘状体检出。bilinear 不产生新极值,是这个用途下的保守选择。

**缓冲裁剪。** 先向外多取 16 圈像元再重投影,之后裁回精确 bbox,使内部区域不会因
重采样核触边而出现 nodata 缺口。校验确认重投影后 nodata = 0。

**当前采用的配色方案。** haxby 论文风格测深色带(8 步,深蓝→青→绿→黄→橙)
+ 排名直方图均衡 + 不过曝晕渲。晕渲用乘性 `0.45 + 0.55·hs`,只压暗不提亮,
所以浅水区不会被光照冲白。

渲染核心 `render_rgb / make_cmap / normalize` 定义在 `04_mothra_plot.py`,
由 `05_shade_compare.py` 和 `08_context_plot.py` 直接 import 复用 ——
主图与总览图必然同一套渲染,不会各调各的。

早期的 viridis 版 04 已被取代(成图 `mothra_plot_viridis.png` 保留备查);
更早的蓝色阶版本在 `02_plot_mothra.py`。
色阶亮度单调性由 `scripts/_ramp_check.py` 实测(本机无 node,故用 numpy 复现了
`validate_palette.js` 的 `--ordinal` 判据与类别型 CVD 判据),不靠肉眼判断。

> **haxby 色带的已知取舍。** 实测其 OKLCH 亮度**非单调**:升到 `#eeea79`
> (L=0.917)后回落到 `#ff8c3a`(L=0.753)。代价是灰度打印/复印时若干深度会撞车 ——
> 最突出的是 `#32c8ff`(约 2272 m)与 `#ff8c3a`(约 2197 m)相差 75 m 却几乎同亮度
> (ΔL=0.028)。这是海洋地学论文的通行画法,彩色显示下没问题;若要出黑白版需另换色阶。

**热液口筛选口径。** 按几何落点(经纬度落在裁剪 bbox 内)筛,而不是按
`Vent Field Name` 标签 —— 标签是人工归属,几何才是与本项目栅格对齐的口径。
两者本次完全一致,`03_verify_outputs.py` 会独立重算并在不一致时报错。

**热液口标记配色是算出来的,不是挑的。** chimney/mound 两类用**形状 + 颜色双重
编码**(红圆 / 黄三角),识别不单靠颜色。配色用
`_ramp_check.check_categorical`(Machado-Oliveira-Fernandes 2009,severity 1.0,
OKLab ΔE×100)实测过候选组合:

| 候选 | 常视觉 ΔE(硬门 15) | protan | deutan(目标 8) | 结论 |
|---|---|---|---|---|
| 红 + 橙 | 7.1 | 7.9 | 5.6 | **FAIL** — 两个暖色在色觉障碍下基本塌成一色 |
| 红 + 品红 | 13.2 | 15.7 | 12.3 | **FAIL** |
| 橙 + 品红 | 12.9 | 14.9 | 12.5 | **FAIL** |
| **红 + 黄** | **20.8** | **21.7** | **15.3** | **PASS** ← 采用 |
| 红 + 白 | 42.3 | 47.7 | 37.5 | PASS,但白填色与白描边冲突,弃 |

标记全部落在 viridis 的 t≈0.08–0.42 段(暗紫~蓝青),暖色不与色阶撞色;
一律加白描边,保证压在暗紫上也跳得出来。

## 已知坑

`rasterio >= 1.4` 把 `Window.round_offsets()` / `round_lengths()` 的签名改成了
`**kwds`,**`op="ceil"` 参数被静默忽略**,一律按四舍五入处理。写 `op="ceil"` 拿到的
是 nearest —— 292.20 会变成 292,东边少了 0.37 m。`01_crop_reproject.py` 里改成显式
`np.floor` / `np.ceil` 计算,不依赖这两个方法;`03_verify_outputs.py` 用 pyproj 独立
正算角点复核,就是为了兜住这类静默失效。
