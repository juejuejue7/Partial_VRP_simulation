# Bethmetory_data —— 原始水深数据的获取说明

本目录下的**大体积原始栅格不入 git**(合计约 560 MB,见根目录 `.gitignore`)。
理由有二:一是 GitHub 单文件超 50 MB 即告警、仓库整体建议 <1 GB;二是 MGDS 数据
以 CC BY-NC-SA 3.0 授权发布,其条款不鼓励在公开仓库二次分发。

两批数据都是公开的,按下面的说明重新下载即可,文件名与目录布局保持不变,脚本无需改动。

> **先看这里:多数情况下你不需要下载任何东西。**
> 处理链路的产物 `Bethmetory_data_process/outputs/`(6.8 MB)已入库,
> `VRPSimulation` 与 `msim` 直接读它。只有当你要**重跑数据处理链路本身**
> (`Bethmetory_data_process/scripts/01..08`)或重做 TAG 站点的形态学探查时,
> 才需要按下文取回原始栅格。

---

## 1. M127 航次 / TAG 热液区(`Bethmetory_data/data/`,约 101 MB)

用于 `tag_bathy_explore.py`、`detect_chimnys.py` 的烟囱候选点探查。

**一条命令取回全部 4 个文件:**

```bash
cd Bethmetory_data
python tag_bathy_explore.py --download      # 自动下载到 ./data/
```

脚本内置了 URL 表并会跳过已存在的文件。手动下载的话:

| 文件 | 大小 | URL |
|---|---|---|
| `M127_AbyssBathy_WGS84_2m.tif` | 49 MB | https://hs.pangaea.de/bathy/m127/M127_AUV/M127_AbyssBathy_WGS84_2m.tif |
| `M127_AbyssBathy_Mir_WGS84_0.5m.tif` | 20 MB | https://hs.pangaea.de/bathy/m127/M127_AUV/M127_AbyssBathy_Mir_WGS84_0.5m.tif |
| `M127_AbyssBathy_ThreeMounds_Area_WGS84_0.5m.tif` | 15 MB | https://hs.pangaea.de/bathy/m127/M127_AUV/M127_AbyssBathy_ThreeMounds_Area_WGS84_0.5m.tif |
| `M127_Abyss_Magnetic_Anomaly_WGS84_10m.tif` | 16 MB | https://hs.pangaea.de/bathy/m127/M127_AUV/M127_Abyss_Magnetic_Anomaly_WGS84_10m.tif |

**授权:** CC-BY-4.0
**引用:** Petersen, S. (2019). PANGAEA, doi:10.1594/PANGAEA.899415

---

## 2. Juan de Fuca / Endeavour Segment(MGDS,约 460 MB + 多场景瓦片约 100 MB)

Mothra 场景的水深底图来源。**Mothra 单场景处理链路只依赖其中一个文件:**
`EndeavourAUVSouthCentral1.asc` → `Bethmetory_data_process/scripts/01_crop_reproject.py`。
其余是探查阶段留下的坡度图与备用格式,重跑 Mothra 主链路时不需要。

**多场景处理链路**(`Bethmetory_data_process/scripts/00_find_scenarios.py` 起,
产出 `scenarios.json` 里 mef/high_rise/sparse_*/dense_* 等场景)另外依赖三块
覆盖全段的 1 m 瓦片,见下方"多场景瓦片"小节。

三个 MGDS 数据集的 `README.txt`(含完整条款、研究者名单与引用格式)**已随库保留**在
`MGDS_Download (1)/MGDS_Download/README.txt`、`MGDS_Download (2)/.../README.txt`
与 `MGDS_Download/JdF_Endeavour_Bathymetry/README.txt`。

### 数据集 21402 —— GeoTIFF 格式
- 页面: https://www.marine-geo.org/tools/datasets/21402
- Data DOI: `10.1594/IEDA/321402`
- 涉及文件(放回 `MGDS_Download (1)/MGDS_Download/JdF_Endeavour_Bathymetry/`,
  探查阶段另在 `Bethmetory_data/` 根下留了一份副本):
  - `EndeavourAUV_ABE4_slope.tif` — 43 MB
  - `EndeavourAUV2_slope.tif` — 42 MB

### 数据集 21403 —— NetCDF:GMT 格式 ← **主链路依赖这个**
- 页面: https://www.marine-geo.org/tools/datasets/21403
- Data DOI: `10.1594/IEDA/321403`
- 涉及文件(放回 `MGDS_Download (2)/MGDS_Download/JdF_Endeavour_Bathymetry/`):
  - `EndeavourAUVSouthCentral1.asc` — 77 MB ← **必需**
  - `EndeavourAUV_ABE_Topo4mArc.grd` — 57 MB

**重跑主链路时,把 `EndeavourAUVSouthCentral1.asc` 放到:**

```
Bethmetory_data_process/Bethy_data/EndeavourAUVSouthCentral1.asc
```

然后依次执行 `Bethmetory_data_process/scripts/01_crop_reproject.py` 起的编号脚本。
同目录下的 `TABLE_SI.xlsx`(572 个热液喷口点位,48 KB)已入库,无需另取。

**授权:** CC BY-NC-SA 3.0 US — http://creativecommons.org/licenses/by-nc-sa/3.0/us/

#### 多场景瓦片(同一数据集 21403,南北三块,覆盖 MEF / High Rise / Salty Dawg / Sasquatch 等命名热液场)

`EndeavourAUVSouthCentral1.asc` 只覆盖纬向 47.9167°–47.9350°(Mothra 所在的窄条)。
其余命名热液场与滑窗搜索候选(见 `Bethmetory_data_process/scenarios.json`)落在这条
以外, 需要数据集 21403 页面(同上)下载包里的另外三块 1 m 瓦片:

| 文件 | 纬度范围 | 大小(.gz) | 覆盖的场景 |
|---|---|---|---|
| `EndeavourAUVTopoSouth1mArc.grd.gz` | 47.8820–47.9167 | 13 MB | (与 Mothra 相邻, 无命名场落入) |
| `EndeavourAUVTopoCentral1mArc.grd.gz` | 47.9350–47.9740 | 28 MB | **mef / high_rise / dense_1 / dense_2** |
| `EndeavourAUVTopoNorth1mArc.grd.gz` | 47.9740–48.0050 | 13 MB | **salty_dawg / sasquatch / sparse_2 / sparse_3** |

放到:

```
Bethmetory_data/MGDS_Download/JdF_Endeavour_Bathymetry/EndeavourAUVTopo{South,Central,North}1mArc.grd.gz
```

解压(`gzip -dk`)后即可被 `00_find_scenarios.py` 与 `01_crop_reproject.py --scenario <id>`
直接读取(rasterio 的 netCDF 驱动原生支持 `.grd`,nodata 为 NaN,与 `.asc` 的
`-99999` 哨兵值不同, 两种约定脚本内部统一处理,不需要手工转换)。

**数据引用:**
> Kelley, D.; Delaney, J.; Yoerger, D.; Caress, D.; Clague, D. and A. Denny (2015).
> Processed Bathymetry Grids (NetCDF:GMT format) derived from Multibeam Sonar Data
> from the Juan de Fuca - Endeavour Spreading Center Segment assembled as part of
> the JdF:Endeavour_Bathymetry Data Compilation. MGDS. doi:10.1594/IEDA/321403

**相关文献引用:**
> Clague, D.A., et al. (2014). Eruptive and tectonic history of the Endeavour Segment,
> Juan de Fuca Ridge, based on AUV mapping data and lava flow ages.
> Geochem. Geophys. Geosyst. doi:10.1002/2014GC005415

---

## 3. 本目录中**已入库**的内容

| 内容 | 说明 |
|---|---|
| `*.py` | 探查脚本(`tag_bathy_explore` / `detect_chimnys` / `clip` / `plot_casestudy`) |
| `*_candidates.csv` | 烟囱候选点结果表,KB 级,论文中要引用 |
| `Mothra_cropped.tif` | 96 KB 的裁剪小样,便于快速验证 |
| `MGDS_Download*/**/README.txt` | MGDS 授权条款与引用信息 |

被忽略的是:上述两批原始栅格、以及 `*_overview.png` 等可由脚本重跑再生的总览图。
