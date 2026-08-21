import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import pandas as pd

def dec2dms(deg, is_lat=True):
    """将十进制度数转换为学术地图常用的 度分秒 (DMS) 格式"""
    d = int(deg)
    m = int(abs(deg - d) * 60)
    s = (abs(deg - d) - m/60) * 3600.0
    
    direction = ""
    if is_lat:
        direction = "N" if d >= 0 else "S"
    else:
        direction = "E" if d >= 0 else "W"
        
    return f"{abs(d)}°{m}'{s:.0f}\"{direction}"

def plot_mothra_case_study(tif_path, waypoints_path=None):
    # 1. 读取裁切后的 TIF 数据
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
            
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        crs = src.crs

        fig, ax = plt.subplots(figsize=(8, 10))
        
        # 2. 绘制水深图 (使用 viridis 配色，并添加直方图均衡化拉伸)
        # 截断极值 (2% 和 98%)，使颜色集中分布在主体地形上，增强细节对比度
        vmin, vmax = np.nanpercentile(data, [2, 98]) 
        im = ax.imshow(data, cmap="viridis", extent=extent, origin="upper", vmin=vmin, vmax=vmax)
        
        # 3. 设置经纬度坐标轴刻度
        if crs and crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        else:
            transformer = None

        def x_format(val, pos):
            if transformer:
                lon, lat = transformer.transform(val, src.bounds.bottom)
            else:
                lon = val
            return dec2dms(lon, is_lat=False)

        def y_format(val, pos):
            if transformer:
                lon, lat = transformer.transform(src.bounds.left, val)
            else:
                lat = val
            return dec2dms(lat, is_lat=True)

        ax.xaxis.set_major_formatter(FuncFormatter(x_format))
        ax.yaxis.set_major_formatter(FuncFormatter(y_format))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # 4. 添加 150m 比例尺
        if crs and crs.is_projected:
            scalebar = AnchoredSizeBar(ax.transData,
                                       150, '150 m', 'lower right', 
                                       pad=0.8,
                                       color='black',
                                       frameon=True,
                                       size_vertical=3,
                                       fontproperties=fm.FontProperties(size=12, weight='bold'))
            ax.add_artist(scalebar)

        # 5. 目标 Chimney 位置标记
        if waypoints_path:
            try:
                df = pd.read_csv(waypoints_path)
                ax.scatter(df['utm_x_m'], df['utm_y_m'], 
                           c='red', marker='o', s=40, edgecolor='black', linewidths=1.5,
                           label='Detected Chimneys (Auto)')
            except Exception as e:
                print(f"未能加载 CSV 航点数据: {e}")

        # 手动标记区 
        manual_targets = [
            # 格式示范: (utm_x, utm_y, "Chimney Name")
        ]
        
        for mx, my, name in manual_targets:
            ax.plot(mx, my, marker='*', color='gold', markersize=12, markeredgecolor='black', label='Manual Target' if name == manual_targets[0][2] else "")
            ax.annotate(name, (mx, my), xytext=(mx+5, my+5), 
                        color='black', fontsize=10, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # 去除重复图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        # 6. 图表收尾与保存
        ax.set_title("Original TIF & Mothra Bounding Box (Equalized)", fontsize=14, weight='bold', pad=15)
        fig.tight_layout()
        
        output_filename = "Mothra_CaseStudy_Map_Viridis_EQ.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"学术展示图已保存至: {output_filename}")
        plt.show()

if __name__ == "__main__":
    tif_file = "Mothra_cropped.tif"
    waypoints_csv = "Mothra_Chimneys_waypoints.csv" 
    plot_mothra_case_study(tif_file, waypoints_csv)