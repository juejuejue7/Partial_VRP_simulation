"""geodesy 验收:自带的纯 numpy 换算必须与权威实现一致,D2 的代价必须钉住。"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from vrpsim.contracts.config import DEFAULT_WAYPOINTS_CSV
from vrpsim.contracts.frames import (MERIDIAN_CONVERGENCE_DEG,
                                     NET_DISTANCE_ERROR_AT_SEAFLOOR_PPM, ORIGIN_LLH,
                                     ORIGIN_UTM9N, REFERENCE_SEAFLOOR_DEPTH_M,
                                     SEAFLOOR_DEPTH_SCALE_PPM, UTM_SCALE_FACTOR,
                                     UTM_VS_ELLIPSOID_DISTANCE_PPM,
                                     UTM_VS_ELLIPSOID_MAX_RESIDUAL_M)
from vrpsim.geodesy import (WGS84_A, ned_to_utm9n, ned_to_wgs84, utm9n_to_ned,
                            utm9n_to_wgs84, wgs84_to_ned, wgs84_to_ned_ellipsoidal,
                            wgs84_to_utm9n)

# CSV 的米制列只存 3 位小数 ⇒ 与之比对的分辨极限就是 0.5 mm。
CSV_ROUNDING_M = 1e-3


@pytest.fixture(scope="module")
def csv_cols():
    with open(DEFAULT_WAYPOINTS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    get = lambda k: np.array([float(r[k]) for r in rows], dtype=np.float64)  # noqa: E731
    return {k: get(k) for k in
            ("lon", "lat", "x_local_m", "y_local_m",
             "utm9n_easting", "utm9n_northing", "seafloor_depth_m")}


# ======================================================================
# 1. UTM 正逆算 —— 与 pyproj 生成的 CSV 列对拍
# ======================================================================
def test_forward_utm_matches_csv(csv_cols):
    """自带 Krüger 级数 vs CSV 的 utm9n_* 列(由 pyproj 生成)。"""
    e, n = wgs84_to_utm9n(csv_cols["lon"], csv_cols["lat"])
    assert np.abs(e - csv_cols["utm9n_easting"]).max() < CSV_ROUNDING_M
    assert np.abs(n - csv_cols["utm9n_northing"]).max() < CSV_ROUNDING_M


def test_utm_roundtrip(csv_cols):
    """6 阶 Krüger 级数下,正逆算互为严格逆(纳米级)。3 阶实现在这里会红(0.5 mm)。"""
    lon2, lat2 = utm9n_to_wgs84(csv_cols["utm9n_easting"], csv_cols["utm9n_northing"])
    e2, n2 = wgs84_to_utm9n(lon2, lat2)
    assert np.abs(e2 - csv_cols["utm9n_easting"]).max() < 1e-6
    assert np.abs(n2 - csv_cols["utm9n_northing"]).max() < 1e-6


def test_origin_llh_consistent():
    """契约里写死的 ORIGIN_LLH 必须能由 ORIGIN_UTM9N 反算出来。"""
    lon, lat = utm9n_to_wgs84(np.array([ORIGIN_UTM9N[0]]), np.array([ORIGIN_UTM9N[1]]))
    assert abs(float(lon[0]) - ORIGIN_LLH[0]) < 1e-8
    assert abs(float(lat[0]) - ORIGIN_LLH[1]) < 1e-8

    e, n = wgs84_to_utm9n(np.array([ORIGIN_LLH[0]]), np.array([ORIGIN_LLH[1]]))
    assert abs(float(e[0]) - ORIGIN_UTM9N[0]) < 1e-3
    assert abs(float(n[0]) - ORIGIN_UTM9N[1]) < 1e-3


# ======================================================================
# 2. NED 换轴(D2)
# ======================================================================
def test_ned_axis_swap_and_offset(csv_cols):
    """x_N ← northing-N0(= CSV 的 y_local_m),y_E ← easting-E0(= CSV 的 x_local_m)。"""
    x_n, y_e, z_d = wgs84_to_ned(csv_cols["lon"], csv_cols["lat"], csv_cols["seafloor_depth_m"])
    assert np.abs(x_n - csv_cols["y_local_m"]).max() < CSV_ROUNDING_M   # North ← y_local
    assert np.abs(y_e - csv_cols["x_local_m"]).max() < CSV_ROUNDING_M   # East  ← x_local
    # D 向下为正
    assert np.allclose(z_d, -csv_cols["seafloor_depth_m"])
    assert z_d.min() > 2000.0


def test_ned_roundtrip(csv_cols):
    x_n, y_e, z_d = wgs84_to_ned(csv_cols["lon"], csv_cols["lat"], csv_cols["seafloor_depth_m"])
    lon2, lat2, z_up2 = ned_to_wgs84(x_n, y_e, z_d)
    assert np.abs(lon2 - csv_cols["lon"]).max() < 1e-11
    assert np.abs(lat2 - csv_cols["lat"]).max() < 1e-11
    assert np.allclose(z_up2, csv_cols["seafloor_depth_m"])

    e, n, _ = ned_to_utm9n(x_n, y_e, z_d)
    x2, y2, _ = utm9n_to_ned(e, n)
    assert np.abs(x2 - x_n).max() < 1e-9 and np.abs(y2 - y_e).max() < 1e-9


def test_origin_maps_to_zero():
    x_n, y_e, _ = wgs84_to_ned(ORIGIN_LLH[0], ORIGIN_LLH[1], 0.0)
    assert abs(float(x_n)) < 1e-3 and abs(float(y_e)) < 1e-3


# ======================================================================
# 3. D2 的代价 —— 钉成回归测试
# ======================================================================
def test_utm_vs_ellipsoid_residual_within_contract(csv_cols):
    """UTM 派生 NED 与真椭球局部切平面的差,必须仍在 contracts/frames.py 写明的界内。

    这条测试的用途不是"验证正确",而是"原点或换算路径一旦被改动,契约里的
    0.82 m / 505 ppm / 0.080 deg 会立刻变成谎话"—— 那时这里会红。
    """
    x_n, y_e, _ = wgs84_to_ned(csv_cols["lon"], csv_cols["lat"])
    n_e, e_e, _ = wgs84_to_ned_ellipsoidal(csv_cols["lon"], csv_cols["lat"])

    res = max(float(np.abs(n_e - x_n).max()), float(np.abs(e_e - y_e).max()))
    assert res <= UTM_VS_ELLIPSOID_MAX_RESIDUAL_M
    assert res > 0.1, "残差过小 —— 椭球路径可能没真正独立于 UTM 路径"

    def pw(u, v):
        return np.hypot(u[:, None] - u[None, :], v[:, None] - v[None, :])

    # 用全精度重算值,不要用 CSV 的三位小数列 —— 最近的一对目标只相距 2.7 m,
    # 1 mm 舍入就会伪造出 ~370 ppm 的假偏差(定稿前踩过这个坑)。
    d_utm, d_ell = pw(x_n, y_e), pw(n_e, e_e)
    m = d_utm > 0.0
    ppm = float(np.abs(d_ell[m] / d_utm[m] - 1.0).max() * 1e6)
    assert abs(ppm - UTM_VS_ELLIPSOID_DISTANCE_PPM) < 20.0, \
        f"成对距离偏差 {ppm:.1f} ppm 与契约值 {UTM_VS_ELLIPSOID_DISTANCE_PPM} ppm 不符"
    # 与该点 UTM 尺度因子互为倒数关系(1/k - 1),防止契约值被随手改成别的数
    assert abs((1.0 / UTM_SCALE_FACTOR - 1.0) * 1e6 - ppm) < 5.0


def test_net_distance_error_at_seafloor(csv_cols):
    """AUV 走在海底:深度带来的 -356 ppm 与 UTM 的 -399 ppm 反向抵消,净差 ~44 ppm。

    这是 D2 可接受性的最强论据(全场最长边 653 m 上误差 < 3 cm),必须钉住。
    """
    x_n, y_e, _ = wgs84_to_ned(csv_cols["lon"], csv_cols["lat"])
    n_s, e_s, _ = wgs84_to_ned_ellipsoidal(csv_cols["lon"], csv_cols["lat"],
                                           -REFERENCE_SEAFLOOR_DEPTH_M)

    def pw(u, v):
        return np.hypot(u[:, None] - u[None, :], v[:, None] - v[None, :])

    d_utm, d_sea = pw(x_n, y_e), pw(n_s, e_s)
    m = d_utm > 0.0
    ppm = float(np.abs(d_sea[m] / d_utm[m] - 1.0).max() * 1e6)
    assert abs(ppm - NET_DISTANCE_ERROR_AT_SEAFLOOR_PPM) < 10.0
    # 深度效应本身 = |h|/R
    assert abs((REFERENCE_SEAFLOOR_DEPTH_M / WGS84_A * 1e6) - SEAFLOOR_DEPTH_SCALE_PPM) < 2.0
    # 净误差在全场最长边上 < 3 cm
    assert ppm * 1e-6 * 653.0 < 0.03


def test_meridian_convergence_matches_contract():
    """网格北 vs 真北:在原点沿 UTM 网格北走 500 m,量它在椭球切平面里偏了多少。"""
    e0, n0 = ORIGIN_UTM9N
    lon_a, lat_a = utm9n_to_wgs84(np.array([e0]), np.array([n0]))
    lon_b, lat_b = utm9n_to_wgs84(np.array([e0]), np.array([n0 + 500.0]))
    n_a, e_a, _ = wgs84_to_ned_ellipsoidal(lon_a, lat_a)
    n_b, e_b, _ = wgs84_to_ned_ellipsoidal(lon_b, lat_b)
    ang = abs(np.degrees(np.arctan2(float(e_b[0] - e_a[0]), float(n_b[0] - n_a[0]))))
    assert abs(ang - MERIDIAN_CONVERGENCE_DEG) < 0.005, \
        f"子午线收敛角实测 {ang:.4f} deg,契约写的是 {MERIDIAN_CONVERGENCE_DEG} deg"


# ======================================================================
# 4. 有 pyproj 的环境里再直接对拍一次(auv_py310 无 pyproj → 自动 skip)
# ======================================================================
def test_against_pyproj_if_available(csv_cols):
    pyproj = pytest.importorskip("pyproj", reason="auv_py310 无 pyproj;base 环境下才跑")
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32609", always_xy=True)
    e_ref, n_ref = tr.transform(csv_cols["lon"], csv_cols["lat"])
    e, n = wgs84_to_utm9n(csv_cols["lon"], csv_cols["lat"])
    assert np.abs(e - np.asarray(e_ref)).max() < 1e-3
    assert np.abs(n - np.asarray(n_ref)).max() < 1e-3
