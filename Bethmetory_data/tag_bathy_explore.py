#!/usr/bin/env python3
"""
tag_bathy_explore.py  (v2)
--------------------------
Inspect a TAG bathymetry GeoTIFF, render a hillshade, and do a FIRST-PASS
extraction of candidate mounds / chimneys (suspected discrete targets) by
removing the regional slope and finding local topographic highs.

Data (CC-BY-4.0, cite): Petersen, S. (2019), PANGAEA doi:10.1594/PANGAEA.899415
  2 m field  : https://hs.pangaea.de/bathy/m127/M127_AUV/M127_AbyssBathy_WGS84_2m.tif
  0.5 m Mir  : https://hs.pangaea.de/bathy/m127/M127_AUV/M127_AbyssBathy_Mir_WGS84_0.5m.tif
  0.5 m 3Mnd : https://hs.pangaea.de/bathy/m127/M127_AUV/M127_AbyssBathy_ThreeMounds_Area_WGS84_0.5m.tif
  10 m mag   : https://hs.pangaea.de/bathy/m127/M127_AUV/M127_Abyss_Magnetic_Anomaly_WGS84_10m.tif

The tiles store seafloor as NEGATIVE elevation (e.g. -3470 m on a mound top,
-3672 m in the deep). v2 auto-detects this so mounds are treated as HIGHS.
Extracted points are SUSPECTED targets from morphology only; activity is
confirmed later by a Follower pass / the magnetic layer / literature.

Examples (all on ONE line in PowerShell, or use backtick ` to continue):
  python tag_bathy_explore.py --download
  python tag_bathy_explore.py data/M127_AbyssBathy_ThreeMounds_Area_WGS84_0.5m.tif --high-pass-m 25 --min-height 3 --min-dist-m 8 --out out_3mnd
"""
import argparse, os, urllib.request
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform as warp_xy
from scipy.ndimage import gaussian_filter, maximum_filter, label, find_objects, binary_erosion

TARGET_CRS = "EPSG:32623"  # WGS84 / UTM 23N (TAG ~26.13N, 44.82W)
URLS = {
    "field2m":     "https://hs.pangaea.de/bathy/m127/M127_AUV/M127_AbyssBathy_WGS84_2m.tif",
    "mir":         "https://hs.pangaea.de/bathy/m127/M127_AUV/M127_AbyssBathy_Mir_WGS84_0.5m.tif",
    "threemounds": "https://hs.pangaea.de/bathy/m127/M127_AUV/M127_AbyssBathy_ThreeMounds_Area_WGS84_0.5m.tif",
    "magnetics":   "https://hs.pangaea.de/bathy/m127/M127_AUV/M127_Abyss_Magnetic_Anomaly_WGS84_10m.tif",
}

def maybe_download(outdir="data"):
    os.makedirs(outdir, exist_ok=True)
    for url in URLS.values():
        dest = os.path.join(outdir, os.path.basename(url))
        if os.path.exists(dest):
            print("  have ", dest); continue
        print("  get  ", url); urllib.request.urlretrieve(url, dest)
    print("Download complete ->", outdir)

def read_metric_dem(path, target_crs=TARGET_CRS):
    with rasterio.open(path) as src:
        src_crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
        dst_crs = CRS.from_user_input(target_crs)
        nodata = src.nodata
        if src_crs.to_epsg() == dst_crs.to_epsg():
            data = src.read(1).astype("float64"); tr = src.transform
        else:
            tr, w, h = calculate_default_transform(src_crs, dst_crs, src.width, src.height, *src.bounds)
            data = np.full((h, w), np.nan)
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src_crs,
                      dst_transform=tr, dst_crs=dst_crs,
                      src_nodata=nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
            nodata = np.nan
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        data = np.where(data == nodata, np.nan, data)
    data = np.where(np.abs(data) > 1e6, np.nan, data)  # kill stray fills
    return data, tr, dst_crs, (abs(tr.a), abs(tr.e))

def to_elevation(data, mode):
    if mode == "elevation": return data
    if mode == "depth":     return -data
    return data if np.nanmedian(data) < 0 else -data   # auto: below sea level => negative

def disk(radius):
    r = int(max(1, round(radius))); y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= r*r

def hillshade(elev, px, py, az=315.0, alt=45.0):
    az = np.deg2rad(360.0 - az + 90.0); alt = np.deg2rad(alt)
    z = np.where(np.isfinite(elev), elev, np.nanmean(elev))
    dzdy, dzdx = np.gradient(z, py, px)
    slope = np.pi/2 - np.arctan(np.hypot(dzdx, dzdy)); aspect = np.arctan2(-dzdy, dzdx)
    hs = np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope)*np.cos(az - aspect)
    return np.where(np.isfinite(elev), np.clip(hs, 0, 1), np.nan)

def residual_relief(elev, px, sigma_m, denoise_m, min_support=0.6):
    """Band-pass: light denoise minus large-sigma regional smooth. NaN where
    local valid support < min_support (kills gap-edge artifacts)."""
    finite = np.isfinite(elev)
    z = np.where(finite, elev, 0.0)
    if denoise_m and denoise_m > 0:
        s = max(0.5, denoise_m/px)
        z = gaussian_filter(z, s) / np.where(gaussian_filter(finite.astype(float), s) > 1e-6,
                                             gaussian_filter(finite.astype(float), s), np.nan)
        z = np.where(finite, z, 0.0)
    sig = max(1.0, sigma_m/px)
    den = gaussian_filter(finite.astype(float), sig)
    smooth = gaussian_filter(np.where(finite, z, 0.0), sig) / np.where(den > 1e-6, den, np.nan)
    resid = np.where(finite, z - smooth, np.nan)
    return np.where(den >= min_support, resid, np.nan)

def detect_peaks(resid, px, min_height, min_dist_m):
    rad = max(1, round(min_dist_m/px)); fp = disk(rad)
    filled = np.where(np.isfinite(resid), resid, -np.inf)
    is_max = maximum_filter(filled, footprint=fp) == filled
    valid = binary_erosion(np.isfinite(resid), structure=disk(rad), border_value=0)  # stay away from gaps
    mask = is_max & (resid >= min_height) & valid
    lbl, _ = label(mask); rows, cols, hgt = [], [], []
    for sl in find_objects(lbl):
        if sl is None: continue
        sub = np.where(lbl[sl] > 0, resid[sl], -np.inf)
        rr, cc = np.unravel_index(np.argmax(sub), sub.shape)
        rows.append(sl[0].start+rr); cols.append(sl[1].start+cc); hgt.append(float(sub[rr, cc]))
    return np.array(rows), np.array(cols), np.array(hgt)

def reject_ridges(field, px, rows, cols, ridge_scale_m=3.0, R=10.0):
    """Keep isotropic bumps (chimney-like), drop elongated ridge/scarp responses,
    via the Hessian curvature ratio at each candidate (SIFT-style edge rejection).
    Returns a boolean keep-mask aligned to rows/cols."""
    s = max(1.0, ridge_scale_m / px)
    z = np.where(np.isfinite(field), field, np.nanmean(field[np.isfinite(field)]))
    Axx = gaussian_filter(z, s, order=[0, 2])
    Ayy = gaussian_filter(z, s, order=[2, 0])
    Axy = gaussian_filter(z, s, order=[1, 1])
    thr = ((R + 1.0) ** 2) / R
    keep = np.zeros(len(rows), bool)
    for i, (r, c) in enumerate(zip(rows, cols)):
        hxx, hyy, hxy = Axx[r, c], Ayy[r, c], Axy[r, c]
        tr = hxx + hyy; det = hxx * hyy - hxy * hxy
        if det > 0 and tr < 0:                 # concave-down blob (a peak)
            keep[i] = (tr * tr) / det < thr    # small ratio => round; large => elongated
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif", nargs="?")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--sign", choices=["auto", "elevation", "depth"], default="auto",
                    help="how to read pixel values (default auto; TAG tiles are 'elevation', negative down)")
    ap.add_argument("--denoise-m", type=float, default=1.0, help="pre-smooth scale to suppress pixel noise (m)")
    ap.add_argument("--high-pass-m", type=float, default=30.0, help="regional-slope removal scale (m)")
    ap.add_argument("--min-height", type=float, default=2.0, help="min residual height for a candidate (m)")
    ap.add_argument("--min-dist-m", type=float, default=8.0, help="min separation between candidates (m)")
    ap.add_argument("--out", default="tag_out")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN","YMIN","XMAX","YMAX"),
                    help="clip detections to a box in overview-plot local metres (xmin ymin xmax ymax)")
    ap.add_argument("--reject-ridges", type=float, default=0.0,
                    help="drop elongated scarp/ridge responses; value R ~8-12 (0=off). Use for chimney-scale.")
    ap.add_argument("--ridge-scale-m", type=float, default=3.0, help="curvature scale for ridge test (m)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.download: maybe_download(); return
    if not args.tif: ap.error("give a GeoTIFF path, or use --download")

    raw, tr, crs, (px, py) = read_metric_dem(args.tif)
    elev = to_elevation(raw, args.sign)
    finite = np.isfinite(elev); h, w = elev.shape
    sign_note = "elevation(neg=deep)" if np.nanmedian(raw) < 0 else "depth(pos=deep)"
    print("="*64)
    print("FILE      :", args.tif)
    print("CRS       :", crs, "|  value convention:", args.sign, "->", sign_note)
    print(f"GRID      : {h} x {w}  |  pixel {px:.3f} x {py:.3f} m")
    print(f"EXTENT    : {w*px/1000:.3f} km (E-W) x {h*py/1000:.3f} km (N-S)")
    print(f"SEAFLOOR  : shallowest {np.nanmax(elev):.1f}  deepest {np.nanmin(elev):.1f} m  "
          f"({finite.mean()*100:.0f}% valid)")
    print("="*64)

    hs = hillshade(elev, px, py)
    resid = residual_relief(elev, px, args.high_pass_m, args.denoise_m)
    pos = resid[np.isfinite(resid) & (resid > 0)]
    if pos.size:
        qs = np.percentile(pos, [50, 75, 90, 95, 99])
        print("RESIDUAL+ percentiles (m): "
              + "  ".join(f"p{p}={v:.2f}" for p, v in zip([50, 75, 90, 95, 99], qs))
              + "   <- use to pick --min-height")

    rows, cols, hgt = detect_peaks(resid, px, args.min_height, args.min_dist_m)
    if args.reject_ridges > 0 and len(rows):
        keepr = reject_ridges(resid, px, rows, cols, args.ridge_scale_m, args.reject_ridges)
        print(f"RIDGES    : kept {int(keepr.sum())} of {len(keepr)} after ridge/scarp rejection (R={args.reject_ridges})")
        rows, cols, hgt = rows[keepr], cols[keepr], hgt[keepr]
    xloc = (cols + 0.5) * px           # overview-plot local frame (metres from W/S edge)
    yloc = (h - rows - 0.5) * py
    if args.bbox and len(rows):
        xmn, ymn, xmx, ymx = args.bbox
        keep = (xloc >= xmn) & (xloc <= xmx) & (yloc >= ymn) & (yloc <= ymx)
        print(f"BBOX      : kept {int(keep.sum())} of {len(keep)} inside {args.bbox}")
        rows, cols, hgt, xloc, yloc = rows[keep], cols[keep], hgt[keep], xloc[keep], yloc[keep]
    print(f"CANDIDATES: {len(rows)}  (denoise={args.denoise_m} high_pass={args.high_pass_m} "
          f"min_height={args.min_height} min_dist={args.min_dist_m})")
    if len(rows) > 400:
        print("  ! very high count -> raise --min-height (see percentiles) or --high-pass-m")

    xs = tr.c + (cols+0.5)*tr.a + (rows+0.5)*tr.b
    ys = tr.f + (cols+0.5)*tr.d + (rows+0.5)*tr.e
    lon, lat = warp_xy(crs, CRS.from_epsg(4326), list(xs), list(ys)) if len(rows) else ([], [])
    order = np.argsort(-hgt) if len(rows) else []
    csv = args.out + "_candidates.csv"
    with open(csv, "w") as f:
        f.write("id,x_m,y_m,utm_x_m,utm_y_m,lon,lat,residual_height_m\n")
        for i, k in enumerate(order):
            f.write(f"{i},{xloc[k]:.2f},{yloc[k]:.2f},{xs[k]:.2f},{ys[k]:.2f},{lon[k]:.6f},{lat[k]:.6f},{hgt[k]:.2f}\n")
    print("WROTE     :", csv)

    if args.no_plot: return
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("PLOT skipped (matplotlib import failed:", e, ") -- CSV is still valid.")
        print("     fix with:  pip install -U matplotlib"); return
    ext = [0, w*px, 0, h*py]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    ax[0].imshow(hs, cmap="gray", extent=ext); ax[0].set_title("Hillshade")
    v = np.nanpercentile(np.abs(resid), 98)
    ax[1].imshow(resid, cmap="RdBu_r", vmin=-v, vmax=v, extent=ext); ax[1].set_title(f"Residual relief (HP {args.high_pass_m} m)")
    ax[2].imshow(hs, cmap="gray", extent=ext)
    if len(rows):
        ax[2].scatter((cols+0.5)*px, (h-rows-0.5)*py, s=16, facecolors="none", edgecolors="red", linewidths=0.9)
    ax[2].set_title(f"{len(rows)} candidates")
    for a in ax: a.set_xlabel("m"); a.set_ylabel("m")
    png = args.out + "_overview.png"; fig.savefig(png, dpi=130); print("WROTE     :", png)

if __name__ == "__main__":
    main()