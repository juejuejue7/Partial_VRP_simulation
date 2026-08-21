import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform as warp_xy
from scipy.ndimage import gaussian_filter, maximum_filter, label, find_objects, binary_erosion

# 目标投影系设定为 Endeavour Segment 所在的 UTM 9N
TARGET_CRS = "EPSG:32609"

def read_metric_dem(path, target_crs=TARGET_CRS):
    with rasterio.open(path) as src:
        src_crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
        dst_crs = CRS.from_user_input(target_crs)
        nodata = src.nodata
        
        if src_crs.to_epsg() == dst_crs.to_epsg():
            data = src.read(1).astype("float64")
            tr = src.transform
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
    data = np.where(np.abs(data) > 1e6, np.nan, data)
    return data, tr, dst_crs, (abs(tr.a), abs(tr.e))

def to_elevation(data, mode="auto"):
    if mode == "elevation": return data
    if mode == "depth":     return -data
    return data if np.nanmedian(data) < 0 else -data

def disk(radius):
    r = int(max(1, round(radius)))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= r*r

def residual_relief(elev, px, sigma_m, denoise_m, min_support=0.6):
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
    rad = max(1, round(min_dist_m/px))
    fp = disk(rad)
    filled = np.where(np.isfinite(resid), resid, -np.inf)
    is_max = maximum_filter(filled, footprint=fp) == filled
    valid = binary_erosion(np.isfinite(resid), structure=disk(rad), border_value=0)
    mask = is_max & (resid >= min_height) & valid
    lbl, _ = label(mask)
    rows, cols, hgt = [], [], []
    for sl in find_objects(lbl):
        if sl is None: continue
        sub = np.where(lbl[sl] > 0, resid[sl], -np.inf)
        rr, cc = np.unravel_index(np.argmax(sub), sub.shape)
        rows.append(sl[0].start+rr)
        cols.append(sl[1].start+cc)
        hgt.append(float(sub[rr, cc]))
    return np.array(rows), np.array(cols), np.array(hgt)

def reject_ridges(field, px, rows, cols, ridge_scale_m=3.0, R=10.0):
    s = max(1.0, ridge_scale_m / px)
    z = np.where(np.isfinite(field), field, np.nanmean(field[np.isfinite(field)]))
    Axx = gaussian_filter(z, s, order=[0, 2])
    Ayy = gaussian_filter(z, s, order=[2, 0])
    Axy = gaussian_filter(z, s, order=[1, 1])
    thr = ((R + 1.0) ** 2) / R
    keep = np.zeros(len(rows), bool)
    for i, (r, c) in enumerate(zip(rows, cols)):
        hxx, hyy, hxy = Axx[r, c], Ayy[r, c], Axy[r, c]
        tr = hxx + hyy
        det = hxx * hyy - hxy * hxy
        if det > 0 and tr < 0:
            keep[i] = (tr * tr) / det < thr
    return keep

def main():
    class Args:
        tif = "Mothra_cropped.tif"
        sign = "auto"
        denoise_m = 1.0
        high_pass_m = 25.0
        min_height = 30.0
        min_dist_m = 5.0
        out = "Mothra_out"
        bbox = None
        reject_ridges = 10.0
        ridge_scale_m = 3.0
        no_plot = False

    args = Args()

    raw, tr, crs, (px, py) = read_metric_dem(args.tif)
    # elev 仅用于后台的数学峰值检测，保证寻找的是凸起
    elev = to_elevation(raw, args.sign)
    finite = np.isfinite(elev)
    h, w = elev.shape
    sign_note = "elevation(neg=deep)" if np.nanmedian(raw) < 0 else "depth(pos=deep)"
    
    print("="*64)
    print("FILE      :", args.tif)
    print("CRS       :", crs, "|  value convention:", args.sign, "->", sign_note)
    print(f"GRID      : {h} x {w}  |  pixel {px:.3f} x {py:.3f} m")
    print(f"EXTENT    : {w*px/1000:.3f} km (E-W) x {h*py/1000:.3f} km (N-S)")
    print(f"SEAFLOOR  : shallowest {np.nanmax(elev):.1f}  deepest {np.nanmin(elev):.1f} m  "
          f"({finite.mean()*100:.0f}% valid)")
    print("="*64)

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
        
    xloc = (cols + 0.5) * px           
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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("PLOT skipped (matplotlib import failed:", e, ") -- CSV is still valid.")
        print("     fix with:  pip install -U matplotlib"); return
        
    ext = [0, w*px, 0, h*py]
    
    # 直接生成单张验证图
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    
    # 核心修改：使用 raw 原始矩阵作为背景，不做正负值调整
    ax.imshow(raw, cmap="viridis", extent=ext)
    
    # 将检测到的坐标圈绘制在 TIF 原图数据之上
    if len(rows):
        ax.scatter((cols+0.5)*px, (h-rows-0.5)*py, s=25, facecolors="none", edgecolors="red", linewidths=1.5)
        
    ax.set_title(f"Detected {len(rows)} Chimney Waypoints on Raw TIF Data")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    
    png = args.out + "_overview.png"
    fig.savefig(png, dpi=150)
    print("WROTE     :", png)

if __name__ == "__main__":
    main()