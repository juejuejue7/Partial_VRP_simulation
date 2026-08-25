"""[第0层] 经纬度 <-> UTM 9N <-> NED 换算(纯 numpy,零外部依赖)。

为什么自带实现而不是调 pyproj
--------------------------------------------------------------------------------
本机两个 conda 环境能力互补:`base`(py3.9)有 pyproj/rasterio 但没有 matplotlib,
`auv_py310` 有 matplotlib/gymnasium/torch 但**没有 pyproj**。仿真必须跑在 auv_py310,
故换算自带实现。精度不是妥协:横轴墨卡托用 Krüger 级数(n 的 3 阶项),对本区实测
与 pyproj **最大差 0.50 mm** —— 那正是源 CSV 只存 3 位小数的舍入下限,即已到可测极限。

两条独立路径
--------------------------------------------------------------------------------
1. **运行时路径(授权定义,D2)**:WGS84 → UTM 9N → 换轴平移 → NED。
   `wgs84_to_utm9n` / `utm9n_to_ned` / `wgs84_to_ned` 及其逆。
2. **交叉校验路径(不参与运行时)**:WGS84 → ECEF → 椭球局部切平面 ENU → NED。
   `geodetic_to_ecef` / `ecef_to_ned` / `wgs84_to_ned_ellipsoidal`。
   这是"真"局部切平面(真北、真地面距离),用来量化 D2 的投影代价。
   ⚠ 不要拿它算仿真里的任何量 —— 那会让同一个点有两套坐标。

轴序一律遵循 `msim/contracts/types.py`:x = North, y = East, z = Down。
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .contracts.frames import (
    ORIGIN_LLH,
    ORIGIN_UTM9N,
    UTM_CENTRAL_MERIDIAN_DEG,
    UTM_FALSE_EASTING_M,
    UTM_FALSE_NORTHING_M,
    UTM_K0,
)

__all__ = [
    "wgs84_to_utm9n", "utm9n_to_wgs84",
    "utm9n_to_ned", "ned_to_utm9n",
    "wgs84_to_ned", "ned_to_wgs84",
    "geodetic_to_ecef", "ecef_to_ned", "wgs84_to_ned_ellipsoidal",
    "WGS84_A", "WGS84_F",
]

# --- WGS84 椭球 ----------------------------------------------------------
WGS84_A: float = 6378137.0                      # 长半轴 [m]
WGS84_F: float = 1.0 / 298.257223563            # 扁率
_E2: float = WGS84_F * (2.0 - WGS84_F)          # 第一偏心率平方

# --- Krüger 级数系数(第三扁率 n 的 6 阶展开,Karney 2011) ---------------
# 用 6 阶而非常见的 3 阶:3 阶的截断误差在本区约 0.5 mm,虽然已低于源数据精度,
# 但会让 forward/inverse 不是彼此的严格逆(往返残差 0.5 mm)。6 阶把往返压到
# 纳米级,`test_utm_roundtrip` 才能用干净的紧容差,而不是为迁就实现放宽判据。
_N: float = WGS84_F / (2.0 - WGS84_F)
_A_RECT: float = WGS84_A / (1.0 + _N) * (
    1.0 + _N ** 2 / 4.0 + _N ** 4 / 64.0 + _N ** 6 / 256.0)
# 正算 α_j(球面 → 横墨卡托)
_ALPHA = (
    (_N / 2.0 - 2.0 * _N ** 2 / 3.0 + 5.0 * _N ** 3 / 16.0 + 41.0 * _N ** 4 / 180.0
     - 127.0 * _N ** 5 / 288.0 + 7891.0 * _N ** 6 / 37800.0),
    (13.0 * _N ** 2 / 48.0 - 3.0 * _N ** 3 / 5.0 + 557.0 * _N ** 4 / 1440.0
     + 281.0 * _N ** 5 / 630.0 - 1983433.0 * _N ** 6 / 1935360.0),
    (61.0 * _N ** 3 / 240.0 - 103.0 * _N ** 4 / 140.0 + 15061.0 * _N ** 5 / 26880.0
     + 167603.0 * _N ** 6 / 181440.0),
    (49561.0 * _N ** 4 / 161280.0 - 179.0 * _N ** 5 / 168.0
     + 6601661.0 * _N ** 6 / 7257600.0),
    34729.0 * _N ** 5 / 80640.0 - 3418889.0 * _N ** 6 / 1995840.0,
    212378941.0 * _N ** 6 / 319334400.0,
)
# 逆算 β_j
_BETA = (
    (_N / 2.0 - 2.0 * _N ** 2 / 3.0 + 37.0 * _N ** 3 / 96.0 - _N ** 4 / 360.0
     - 81.0 * _N ** 5 / 512.0 + 96199.0 * _N ** 6 / 604800.0),
    (_N ** 2 / 48.0 + _N ** 3 / 15.0 - 437.0 * _N ** 4 / 1440.0 + 46.0 * _N ** 5 / 105.0
     - 1118711.0 * _N ** 6 / 3870720.0),
    (17.0 * _N ** 3 / 480.0 - 37.0 * _N ** 4 / 840.0 - 209.0 * _N ** 5 / 4480.0
     + 5569.0 * _N ** 6 / 90720.0),
    (4397.0 * _N ** 4 / 161280.0 - 11.0 * _N ** 5 / 504.0
     - 830251.0 * _N ** 6 / 7257600.0),
    4583.0 * _N ** 5 / 161280.0 - 108847.0 * _N ** 6 / 3991680.0,
    20648693.0 * _N ** 6 / 638668800.0,
)
# 逆算 δ_j(等角纬度 → 大地纬度)
_DELTA = (
    (2.0 * _N - 2.0 * _N ** 2 / 3.0 - 2.0 * _N ** 3 + 116.0 * _N ** 4 / 45.0
     + 26.0 * _N ** 5 / 45.0 - 2854.0 * _N ** 6 / 675.0),
    (7.0 * _N ** 2 / 3.0 - 8.0 * _N ** 3 / 5.0 - 227.0 * _N ** 4 / 45.0
     + 2704.0 * _N ** 5 / 315.0 + 2323.0 * _N ** 6 / 945.0),
    (56.0 * _N ** 3 / 15.0 - 136.0 * _N ** 4 / 35.0 - 1262.0 * _N ** 5 / 105.0
     + 73814.0 * _N ** 6 / 2835.0),
    (4279.0 * _N ** 4 / 630.0 - 332.0 * _N ** 5 / 35.0
     - 399572.0 * _N ** 6 / 14175.0),
    4174.0 * _N ** 5 / 315.0 - 144838.0 * _N ** 6 / 6237.0,
    601676.0 * _N ** 6 / 22275.0,
)

_LON0_RAD: float = math.radians(UTM_CENTRAL_MERIDIAN_DEG)


# ======================================================================
# 1. WGS84 <-> UTM 9N(横轴墨卡托,Krüger 级数)
# ======================================================================
def wgs84_to_utm9n(lon_deg, lat_deg) -> Tuple[np.ndarray, np.ndarray]:
    """经纬度(度)→ UTM 9N 的 (easting, northing)(米)。

    标量或数组均可,返回同形状 float64 数组。
    """
    lon = np.asarray(lon_deg, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)

    phi = np.radians(lat)
    dlam = np.radians(lon) - _LON0_RAD

    # 大地纬度 → 等角纬度的正切(Karney 2011 式 (8))
    sin_phi = np.sin(phi)
    t = np.sinh(np.arctanh(sin_phi)
                - 2.0 * math.sqrt(_N) / (1.0 + _N)
                * np.arctanh(2.0 * math.sqrt(_N) / (1.0 + _N) * sin_phi))

    xi = np.arctan2(t, np.cos(dlam))                    # 球面横墨卡托纵坐标
    eta = np.arctanh(np.sin(dlam) / np.hypot(1.0, t))   # 球面横墨卡托横坐标

    xi_s = xi.copy()
    eta_s = eta.copy()
    for j, aj in enumerate(_ALPHA, start=1):
        xi_s = xi_s + aj * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_s = eta_s + aj * np.cos(2 * j * xi) * np.sinh(2 * j * eta)

    easting = UTM_FALSE_EASTING_M + UTM_K0 * _A_RECT * eta_s
    northing = UTM_FALSE_NORTHING_M + UTM_K0 * _A_RECT * xi_s
    return easting, northing


def utm9n_to_wgs84(easting, northing) -> Tuple[np.ndarray, np.ndarray]:
    """UTM 9N 的 (easting, northing)(米)→ 经纬度(度)。`wgs84_to_utm9n` 的逆。"""
    e = np.asarray(easting, dtype=np.float64)
    n = np.asarray(northing, dtype=np.float64)

    xi = (n - UTM_FALSE_NORTHING_M) / (UTM_K0 * _A_RECT)
    eta = (e - UTM_FALSE_EASTING_M) / (UTM_K0 * _A_RECT)

    xi_p = xi.copy()
    eta_p = eta.copy()
    for j, bj in enumerate(_BETA, start=1):
        xi_p = xi_p - bj * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - bj * np.cos(2 * j * xi) * np.sinh(2 * j * eta)

    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))      # 等角纬度
    phi = chi.copy()
    for j, dj in enumerate(_DELTA, start=1):
        phi = phi + dj * np.sin(2 * j * chi)

    lam = _LON0_RAD + np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    return np.degrees(lam), np.degrees(phi)


# ======================================================================
# 2. UTM 9N <-> NED(D2:换轴 + 平移,无旋转无缩放)
# ======================================================================
def utm9n_to_ned(easting, northing, z_up_m=0.0, *, origin_utm9n=ORIGIN_UTM9N
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UTM 9N + 高程 → NED (x_N, y_E, z_D)。

    z_up_m 是**向上为正**的高程(海底水深即负值,如 -2283.0);
    返回的 z_D **向下为正**(= -z_up_m)。

    origin_utm9n: (easting, northing) 原点,默认 Mothra 的 D2 冻结值
    `ORIGIN_UTM9N`。其它场景(非 Mothra)传各自的栅格西南角, 不改这个默认值 ——
    Mothra 调用方不传此参数时行为逐位不变。
    """
    e = np.asarray(easting, dtype=np.float64)
    n = np.asarray(northing, dtype=np.float64)
    z_up = np.asarray(z_up_m, dtype=np.float64)

    x_north = n - origin_utm9n[1]
    y_east = e - origin_utm9n[0]
    z_down = -z_up
    return x_north, y_east, np.broadcast_to(z_down, x_north.shape).astype(np.float64)


def ned_to_utm9n(x_north_m, y_east_m, z_down_m=0.0, *, origin_utm9n=ORIGIN_UTM9N
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NED → UTM 9N + 向上为正的高程。`utm9n_to_ned` 的逆。

    origin_utm9n: 见 `utm9n_to_ned`,默认 Mothra 冻结值。
    """
    x = np.asarray(x_north_m, dtype=np.float64)
    y = np.asarray(y_east_m, dtype=np.float64)
    zd = np.asarray(z_down_m, dtype=np.float64)

    easting = y + origin_utm9n[0]
    northing = x + origin_utm9n[1]
    return easting, northing, np.broadcast_to(-zd, easting.shape).astype(np.float64)


# ======================================================================
# 3. WGS84 <-> NED(运行时授权路径:组合 1 与 2)
# ======================================================================
def wgs84_to_ned(lon_deg, lat_deg, z_up_m=0.0, *, origin_utm9n=ORIGIN_UTM9N
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """经纬度 → NED (x_N, y_E, z_D)。**本仿真的授权换算(D2)**。

    origin_utm9n: 见 `utm9n_to_ned`,默认 Mothra 冻结值; 其它场景传各自原点。
    """
    e, n = wgs84_to_utm9n(lon_deg, lat_deg)
    return utm9n_to_ned(e, n, z_up_m, origin_utm9n=origin_utm9n)


def ned_to_wgs84(x_north_m, y_east_m, z_down_m=0.0, *, origin_utm9n=ORIGIN_UTM9N
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NED → (lon, lat, z_up)。`wgs84_to_ned` 的逆。"""
    e, n, z_up = ned_to_utm9n(x_north_m, y_east_m, z_down_m, origin_utm9n=origin_utm9n)
    lon, lat = utm9n_to_wgs84(e, n)
    return lon, lat, z_up


# ======================================================================
# 4. 交叉校验路径:椭球局部切平面(**不参与运行时**)
# ======================================================================
def geodetic_to_ecef(lon_deg, lat_deg, h_m=0.0
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """经纬度 + 椭球高 → WGS84 ECEF (X, Y, Z)。仅供 `ecef_to_ned` 使用。"""
    lam = np.radians(np.asarray(lon_deg, dtype=np.float64))
    phi = np.radians(np.asarray(lat_deg, dtype=np.float64))
    h = np.asarray(h_m, dtype=np.float64)

    sin_phi = np.sin(phi)
    rn = WGS84_A / np.sqrt(1.0 - _E2 * sin_phi ** 2)    # 卯酉圈曲率半径
    x = (rn + h) * np.cos(phi) * np.cos(lam)
    y = (rn + h) * np.cos(phi) * np.sin(lam)
    z = (rn * (1.0 - _E2) + h) * sin_phi
    return x, y, z


def ecef_to_ned(x_ecef, y_ecef, z_ecef, *,
                origin_lon_deg: float = ORIGIN_LLH[0],
                origin_lat_deg: float = ORIGIN_LLH[1],
                origin_h_m: float = ORIGIN_LLH[2]
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF → 以给定原点为切点的**椭球局部切平面** NED。

    ⚠ 这是交叉校验专用。它给的是真北 + 真地面距离,与运行时的 UTM 派生 NED
    相差:方位 0.080 deg、尺度 505 ppm、位置 <= 0.82 m(见 contracts/frames.py)。
    仿真里的任何量都不要用它算,否则同一个点会有两套坐标。
    """
    x0, y0, z0 = geodetic_to_ecef(origin_lon_deg, origin_lat_deg, origin_h_m)
    dx = np.asarray(x_ecef, dtype=np.float64) - x0
    dy = np.asarray(y_ecef, dtype=np.float64) - y0
    dz = np.asarray(z_ecef, dtype=np.float64) - z0

    lam0 = math.radians(origin_lon_deg)
    phi0 = math.radians(origin_lat_deg)
    s_phi, c_phi = math.sin(phi0), math.cos(phi0)
    s_lam, c_lam = math.sin(lam0), math.cos(lam0)

    north = -s_phi * c_lam * dx - s_phi * s_lam * dy + c_phi * dz
    east = -s_lam * dx + c_lam * dy
    up = c_phi * c_lam * dx + c_phi * s_lam * dy + s_phi * dz
    return north, east, -up


def wgs84_to_ned_ellipsoidal(lon_deg, lat_deg, z_up_m=0.0
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """经纬度 → 椭球局部切平面 NED。**仅供交叉校验**,见 `ecef_to_ned`。"""
    x, y, z = geodetic_to_ecef(lon_deg, lat_deg, z_up_m)
    return ecef_to_ned(x, y, z)
