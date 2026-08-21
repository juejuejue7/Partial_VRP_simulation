"""[第0层][A0 冻结] 坐标系契约 —— 真实 Mothra 热液场的 NED 帧定义。

==================================================================
唯一真相源关系
==================================================================
本文件**不重新定义** NED 的轴向语义 —— 那是 `msim/contracts/types.py` 的职责,
本仿真严格继承之:

    x = North(指北),y = East(指东),z = Down(向下为正)
    psi = heading,从 North(+x)起、向 East(+y)为正,wrap 到 [-pi, pi)
    cell 索引 (ix, iy):ix 沿 North,iy 沿 East;数组 field[ix, iy](轴序 == 坐标序)

本文件只冻结**这个具体场地的帧参数**:原点在哪、跨度多大、与源数据的换算路径。

==================================================================
D2(人工裁决 2026-08-09):NED = UTM 9N 换轴
==================================================================
原点 = Mothra 1 m 水深栅格的**西南角**:

    UTM 9N (EPSG:32609)  E = 491801.0 m,  N = 5307442.0 m
    WGS84  (EPSG:4326)   lon = -129.109743763 deg, lat = 47.920247904 deg
    垂向    z = 0 = 海面(椭球面);D 向下为正

换算:

    x_N = utm9n_northing - 5307442.0      # 0 .. 653   (North,长边)
    y_E = utm9n_easting  -  491801.0      # 0 .. 294   (East, 短边)
    z_D = -seafloor_depth_m               # +2253 .. +2286(海面下为正)

与 `Bethmetory_data_process` 产出的 1 m 水深栅格 `mothra_utm9n_1m` 逐格 bit-exact 对齐:

    col = int(y_E)
    row = (NROWS_1M - 1) - int(x_N)       # NROWS_1M = 653

==================================================================
D2 的已量化代价(实测,不得默认为零)
==================================================================
UTM 是**投影平面**,不是真正的局部切平面。相对 WGS84 椭球局部切平面(真 ENU→NED):

    | 项                        | 实测值                                  |
    |---------------------------|-----------------------------------------|
    | 网格北 vs 真北            | 0.080 deg(子午线收敛角)               |
    | 尺度因子 k                | 0.999601(399 ppm,相对 h=0 的椭球面)  |
    | 位置残差                  | 0.822 m(28 点全域最大,契约上界 0.90)  |
    | **净距离误差(海底 h≈-2270 m)** | **44 ppm ≈ 4.3 cm / km**           |

最后一行是关键:AUV 走在海底而非椭球面上,而半径 R-|h| 处的地面距离本就比椭球面
短 356 ppm —— 方向与 UTM 的 -399 ppm 相反,几乎抵消。故本场地(最长边 653 m)上,
投影引入的路径长度误差 **< 3 cm**。方位与位置残差也都远低于 AUV 组合导航误差
(DVL/INS 米级漂移),故裁定可接受。
`vrpsim/geodesy.py` 同时实现了椭球切平面路径,`tests/test_geodesy.py` 把上述数值
**钉成回归测试** —— 若哪天原点或换算路径被改动,测试会立刻失败。

论文措辞:本仿真的平面坐标是 **UTM 9N 投影下的局部 NED**,不是严格的椭球局部切平面。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# --- 源数据 CRS ----------------------------------------------------------
EPSG_GEOG: int = 4326           # WGS84 经纬度
EPSG_UTM: int = 32609           # UTM zone 9N
UTM_ZONE: int = 9
UTM_CENTRAL_MERIDIAN_DEG: float = -129.0
UTM_K0: float = 0.9996
UTM_FALSE_EASTING_M: float = 500000.0
UTM_FALSE_NORTHING_M: float = 0.0

# --- NED 原点(D2) ------------------------------------------------------
ORIGIN_UTM9N: Tuple[float, float] = (491801.0, 5307442.0)      # (easting, northing)
ORIGIN_LLH: Tuple[float, float, float] = (-129.109743763, 47.920247904, 0.0)  # (lon, lat, h)

# --- 原始栅格跨度(= 1 m 水深栅格的外包框,由上游数据定死,不可改) --------
RASTER_NORTH_EXTENT_M: float = 653.0   # x_N 跨度
RASTER_EAST_EXTENT_M: float = 294.0    # y_E 跨度

# --- 与 1 m 水深栅格的对齐关系(D3:本步不加载地形,只冻结索引口径) -------
GRID_1M_ROWS: int = 653         # 南北向行数
GRID_1M_COLS: int = 294         # 东西向列数
GRID_1M_RES_M: float = 1.0

# --- 分析域(D7:裁切后的仿真世界) --------------------------------------
# 原点**不变**(仍是栅格西南角),只缩小外框 ⇒ 保留目标的 NED 坐标逐位不变,
# 且 col = int(y_E) 的栅格列索引照旧成立(行索引见 grid_index_1m)。
WORLD_NORTH_EXTENT_M: float = 500.0    # x_N 跨度,长边(主测线方向)
WORLD_EAST_EXTENT_M: float = 100.0     # y_E 跨度,短边 == 窗口宽度
# 向后兼容别名:老代码里的 NORTH/EAST_EXTENT_M 一律指**分析域**
NORTH_EXTENT_M: float = WORLD_NORTH_EXTENT_M
EAST_EXTENT_M: float = WORLD_EAST_EXTENT_M

# --- 观测窗口(D8:沿测线滑动,非固定分块) ------------------------------
WINDOW_LOOK_BACK_M: float = 100.0      # 沿测线(North)向后回看
WINDOW_WIDTH_M: float = 100.0          # 横向(East)宽度 == WORLD_EAST_EXTENT_M
# 窗口宽 == 场地 East 跨度 ⇒ 窗口是"全 East 的 North 条带",Leader 单条测线即可全覆盖。
# 这同时满足 msim §v1.1.5 对窗口宽的要求(虽然那条是为 RL 动作空间定的,本仿真不做 RL)。

# --- D2 代价的实测值(tests/test_geodesy.py 的回归基准) -----------------
# 位置残差:实测 0.822 m(相对 h=0 的椭球切平面,场地东北角最大),留余量取 0.90 作硬上界
UTM_VS_ELLIPSOID_MAX_RESIDUAL_M: float = 0.90
# 成对距离相对偏差 = 1/k - 1。k = k0·(1 + (x/R)²/2) = 0.9996 × 1.0000008 = 0.999601
# ⇒ 399 ppm:UTM 网格距离比**椭球面(h=0)**上的地面距离短 399 ppm。
UTM_VS_ELLIPSOID_DISTANCE_PPM: float = 399.0
UTM_SCALE_FACTOR: float = 0.999601                  # 本区实际 k(非 k0);pyproj 报 0.999600826
# 子午线收敛角 γ ≈ Δλ·sinφ = 0.108° × sin47.92° = 0.080°
MERIDIAN_CONVERGENCE_DEG: float = 0.080

# --- 深度修正:AUV 实际走在海底,不在椭球面上 -----------------------------
# 半径 R-|h| 处的地面距离比椭球面短 |h|/R;h≈-2270 m ⇒ 356 ppm。
# 它与 UTM 的 -399 ppm **方向相反、几乎抵消**:
#     UTM 网格距离 vs 海底真实航行距离 = 399 - 356 ≈ 44 ppm ≈ 4.3 cm / km
# 也就是说在本场地(最长边 653 m)上,投影带来的路径长度误差 < 3 cm。
# 这是 D2 可接受性的**最强论据**,论文里可以直接引这个数。
REFERENCE_SEAFLOOR_DEPTH_M: float = 2270.0
SEAFLOOR_DEPTH_SCALE_PPM: float = 356.0
NET_DISTANCE_ERROR_AT_SEAFLOOR_PPM: float = 44.0


@dataclass(frozen=True)
class NEDFrame:
    """NED 帧参数的只读打包(供落盘 meta / 跨模块传递)。

    默认值即 D2 冻结值;构造别的原点只应出现在单元测试里。
    """
    origin_easting_m: float = ORIGIN_UTM9N[0]
    origin_northing_m: float = ORIGIN_UTM9N[1]
    origin_lon_deg: float = ORIGIN_LLH[0]
    origin_lat_deg: float = ORIGIN_LLH[1]
    epsg_utm: int = EPSG_UTM
    epsg_geog: int = EPSG_GEOG
    north_extent_m: float = WORLD_NORTH_EXTENT_M   # 分析域,非原始栅格
    east_extent_m: float = WORLD_EAST_EXTENT_M

    def grid_index_1m(self, x_north_m: float, y_east_m: float) -> Tuple[int, int]:
        """NED → **原始** 1 m 水深栅格(653x294)的 (row, col)。

        与 `Bethmetory_data_process/outputs/mothra_waypoints_meta.json` 的
        `index_from_local` 逐字等价(那边 x=East / y=North,此处已换轴)。
        ⚠ 这是**原始栅格**的索引,与裁切无关 —— 裁切只缩小分析域外框,原点不动,
          所以这个公式在裁切前后逐位不变。裁切后数组的索引见 `crop_index`。
        D3:本步不加载地形,此方法仅冻结索引口径供将来使用。
        """
        col = int(y_east_m)
        row = (GRID_1M_ROWS - 1) - int(x_north_m)
        return row, col

    @staticmethod
    def crop_index(x_north_m: float, y_east_m: float, res_m: float = 1.0) -> Tuple[int, int]:
        """NED → **裁切后**分析域数组的 (ix, iy)(North 升序,同 field[ix, iy])。

        与 `msim.geometry.grid.Grid.world_to_cell` 同口径(floor 量化),
        原点不变 ⇒ 无偏移项,只是外框变小。
        """
        return int(x_north_m // res_m), int(y_east_m // res_m)
