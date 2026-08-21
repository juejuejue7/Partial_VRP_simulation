import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import box
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def crop_tif_to_mothra_with_visualization(input_filepath, output_filepath):
    # 1. 定义 Mothra 区域的经纬度边界 (WGS84)
    min_lon = -(129 + 6/60 + 35/3600)  # -129.111111
    max_lon = -(129 + 6/60 + 21/3600)  # -129.105556
    min_lat = 47 + 55/60 + 13/3600      # 47.922222
    max_lat = 47 + 55/60 + 34/3600      # 47.925000

    try:
        with rasterio.open(input_filepath) as src:
            tif_crs = src.crs
            
            if not tif_crs:
                raise ValueError("输入的 TIF 文件缺少坐标参考系统 (CRS) 元数据，无法执行自动转换。")

            # 2. 坐标系转换 (将 WGS84 经纬度转换为 TIF 文件的原生坐标系)
            wgs84_crs = CRS.from_epsg(4326)
            
            if tif_crs != wgs84_crs:
                transformer = Transformer.from_crs(wgs84_crs, tif_crs, always_xy=True)
                min_x, min_y = transformer.transform(min_lon, min_lat)
                max_x, max_y = transformer.transform(max_lon, max_lat)
            else:
                min_x, min_y = min_lon, min_lat
                max_x, max_y = max_lon, max_lat

            # 3. 绘制原图并标记 Bounding Box
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # 使用 rasterio.plot.show 展示原始图像底层数据
            # cmap='viridis' 是一种适合深海地形/斜坡数据的配色，您也可以根据需要更改
            show(src, ax=ax, cmap='viridis', title="Original TIF & Mothra Bounding Box")
            
            # 计算 Box 的宽和高
            rect_width = max_x - min_x
            rect_height = max_y - min_y
            
            # 创建一个红色的矩形框
            rect = patches.Rectangle(
                (min_x, min_y), 
                rect_width, 
                rect_height, 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none', 
                label='Mothra Area'
            )
            
            # 将矩形框添加到图像上
            ax.add_patch(rect)
            ax.legend(loc='upper right')
            
            print("正在显示图像... 请关闭弹出的图像窗口以继续执行裁切保存。")
            plt.show()  # 程序会在此暂停，直到您关闭图片窗口

            # 4. 生成裁切用的几何多边形
            bbox = box(min_x, min_y, max_x, max_y)
            geo = [bbox]

            # 5. 执行掩膜裁切
            out_image, out_transform = mask(src, geo, crop=True)
            out_meta = src.meta.copy()

            # 更新输出文件的元数据
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # 6. 保存裁切后的 TIF 文件
            with rasterio.open(output_filepath, "w", **out_meta) as dest:
                dest.write(out_image)
                
            print(f"裁切成功！文件已保存至: {output_filepath}")
            print(f"裁切尺寸: 宽 {out_image.shape[2]} 像素, 高 {out_image.shape[1]} 像素")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_filepath}'。请确保文件路径正确。")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")

if __name__ == "__main__":
    # input_file = "EndeavourAUV_ABE4_slope.tif"
    input_file = "EndeavourAUV2_slope.tif"
    output_file = "Mothra_cropped.tif"
    crop_tif_to_mothra_with_visualization(input_file, output_file)