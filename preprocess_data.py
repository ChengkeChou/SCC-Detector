import numpy as np
import cv2
import os
from PIL import Image
from scipy import ndimage
from pathlib import Path
import argparse
import glob
from config import DATASET_PATH, SEGMENTER_DATA_PATH

def draw_line(image, x1, y1, x2, y2):
    """
    使用Bresenham线算法在二值图像上绘制线条
    
    Args:
        image: 掩码图像
        x1, y1: 起点坐标
        x2, y2: 终点坐标
    
    Returns:
        image: 添加线条后的图像
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            image[y, x] = 1
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            image[y, x] = 1
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    image[y, x] = 1
    return image

def transform_image(origin_path, image_name, mask_name, save_path):
    """
    将BMP图像和DAT掩码转换为PNG格式
    
    Args:
        origin_path: 原始文件路径
        image_name: BMP图像文件名
        mask_name: DAT掩码文件名
        save_path: 保存路径
    """
    try:
        image_path = os.path.join(origin_path, image_name)
        mask_path = os.path.join(origin_path, mask_name)
        
        # 检查文件是否存在
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"文件不存在: {image_path} 或 {mask_path}")
            return
        
        # 读取原始图像
        img = Image.open(image_path)
        img_size = img.size  # (width, height)
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 读取DAT文件中的分割信息
        with open(mask_path, 'r') as f:
            # 跳过第一行（如果有标题）
            line = f.readline()
            # 如果第一行不是坐标，跳过
            if not (',' in line and all(c.replace('.', '', 1).isdigit() for c in line.split(','))):
                line = f.readline()
            else:
                # 如果第一行是坐标，回到文件开头
                f.seek(0)
                
            # 读取边界点的坐标
            points = []
            for line in f:
                if ',' in line:
                    try:
                        x, y = line.strip().split(',')
                        x_val = int(float(x))
                        y_val = int(float(y))
                        # 确保坐标在图像范围内
                        if 0 <= x_val < img_size[0] and 0 <= y_val < img_size[1]:
                            points.append((x_val, y_val))
                    except (ValueError, IndexError) as e:
                        print(f"解析坐标出错: {line} - {e}")
                        continue
        
        # 检查是否有足够的点形成多边形
        if len(points) < 3:
            print(f"警告: {mask_path} 中的点不足以形成多边形 ({len(points)} 个点)")
            return
        
        # 创建与BMP图像尺寸相同的二值掩码
        mask = np.zeros(img_size[::-1], dtype=np.uint8)  # 注意尺寸转换 (width, height) -> (height, width)
        
        # 在掩码上绘制病变的边界
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            mask = draw_line(mask, x1, y1, x2, y2)
        
        # 保存原始BMP图像为PNG文件
        png_image_path = os.path.join(save_path, f"{os.path.splitext(image_name)[0]}.png")
        img.save(png_image_path)
        print(f"保存图像: {png_image_path}")
        
        # 转置掩码（如果需要）
        # mask = cv2.transpose(mask)
        
        # 保存未填充的掩码（保存边缘信息）
        edge_mask_path = os.path.join(save_path, f"{os.path.splitext(mask_name)[0]}_edge.png")
        cv2.imwrite(edge_mask_path, mask * 255)
        
        # 填充掩码 - 先闭合所有间隙
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 使用洪水填充算法填充闭合区域
        filled_mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        
        # 保存填充后的掩码为PNG文件
        filled_mask_path = os.path.join(save_path, f"{os.path.splitext(mask_name)[0]}.png")
        cv2.imwrite(filled_mask_path, filled_mask * 255)
        print(f"保存掩码: {filled_mask_path}")
    
    except Exception as e:
        print(f"处理 {image_name} 和 {mask_name} 时出错: {e}")

def process_batch(file_path, save_path, is_cropped=False):
    """
    批量处理BMP和DAT文件
    
    Args:
        file_path: 原始文件目录
        save_path: 保存目录
        is_cropped: 是否是CROPPED目录中的文件
    """
    print(f"开始处理批量数据: {file_path}")
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 获取目录中的文件
    try:
        filename_list = os.listdir(file_path)
    except FileNotFoundError:
        print(f"目录不存在: {file_path}")
        return
    
    # 分类文件
    bmp_files = [f for f in filename_list if f.lower().endswith('.bmp')]
    dat_files = [f for f in filename_list if f.lower().endswith('.dat')]
    
    print(f"发现 {len(bmp_files)} 个BMP文件和 {len(dat_files)} 个DAT文件")
    
    # 匹配BMP和DAT文件
    if is_cropped:
        # 对于CROPPED目录，匹配前6个字符
        prefix_length = 6
    else:
        # 对于普通目录，匹配前3个字符
        prefix_length = 3
    
    processed_count = 0
    for bmp_file in bmp_files:
        bmp_prefix = bmp_file[:prefix_length]
        matching_dat_files = [df for df in dat_files if df.startswith(bmp_prefix)]
        
        for dat_file in matching_dat_files:
            print(f'处理: {bmp_file} + {dat_file}')
            transform_image(file_path, bmp_file, dat_file, save_path)
            processed_count += 1
    
    print(f"批处理完成，共处理了 {processed_count} 对文件")

def process_recursive(root_dir, save_root, cell_type):
    """
    递归处理目录及其所有子目录中的BMP和DAT文件
    
    Args:
        root_dir: 要处理的根目录
        save_root: 保存处理结果的根目录
        cell_type: 细胞类型名称
    
    Returns:
        processed_count: 处理的文件对数量
    """
    total_processed = 0
    
    # 使用 pathlib 来递归遍历目录
    root_path = Path(root_dir)
    
    # 处理当前目录中的文件
    is_cropped = "CROPPED" in str(root_path)
    rel_path = root_path.relative_to(Path(f"F:/鳞癌自动化/鳞癌数据raw/{cell_type}")) if str(cell_type) in str(root_path) else Path("")
    save_path = Path(save_root) / rel_path
    
    # 处理当前目录中的BMP和DAT文件
    processed = process_batch(str(root_path), str(save_path), is_cropped)
    total_processed += processed if processed else 0
    
    # 遍历子目录
    for child in root_path.iterdir():
        if child.is_dir():
            # 递归处理子目录
            child_processed = process_recursive(str(child), save_root, cell_type)
            total_processed += child_processed
    
    return total_processed

def fill_mask_directory(directory):
    """
    填充目录中的所有掩码
    
    Args:
        directory: 掩码目录
    """
    print(f"开始填充掩码目录: {directory}")
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"目录不存在: {directory}")
        return
    
    mask_files = [f for f in files if f.endswith('.png') and ('_nuc' in f or '_cyt' in f) and not f.endswith('_edge.png')]
    
    for mask_file in mask_files:
        mask_path = os.path.join(directory, mask_file)
        print(f"填充掩码: {mask_file}")
        try:
            # 读取掩码
            img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img_mask is None:
                print(f"无法读取掩码: {mask_path}")
                continue
                
            # 应用二值填充
            filled_mask = ndimage.binary_fill_holes(img_mask > 0)
            filled_mask = np.uint8(filled_mask) * 255
            
            # 保存填充后的掩码
            cv2.imwrite(mask_path, filled_mask)
        except Exception as e:
            print(f"填充掩码 {mask_file} 时出错: {e}")

def preprocess_dataset(input_base_path, output_base_path):
    """
    预处理整个数据集
    
    Args:
        input_base_path: 输入数据集根目录
        output_base_path: 输出数据集根目录
    """
    # 确保输出根目录存在
    os.makedirs(output_base_path, exist_ok=True)
    
    # 细胞类型目录
    cell_types = ['im_Dyskeratotic', 'im_Koilocytotic', 'im_Metaplastic', 
                 'im_Parabasal', 'im_Superficial-Intermediate']
    
    total_processed_files = 0
    
    for cell_type in cell_types:
        input_dir = os.path.join(input_base_path, cell_type)
        output_dir = os.path.join(output_base_path, f"png_{cell_type[3:]}")
        
        print(f"\n===== 处理细胞类型: {cell_type} =====")
        
        # 递归处理所有子目录
        if os.path.exists(input_dir):
            print(f"递归处理目录: {input_dir}")
            processed = process_recursive(input_dir, output_dir, cell_type)
            total_processed_files += processed
            print(f"细胞类型 {cell_type} 共处理了 {processed} 对文件")
        else:
            print(f"目录不存在: {input_dir}")
        
        # 填充掩码
        fill_mask_directory(output_dir)
        
        # 递归填充所有子目录中的掩码
        for root, dirs, files in os.walk(output_dir):
            for d in dirs:
                subdir = os.path.join(root, d)
                print(f"填充子目录掩码: {subdir}")
                fill_mask_directory(subdir)
    
    print(f"\n预处理完成，总共处理了 {total_processed_files} 对文件")

def create_png_masks_for_segmenter(input_base_path, segmenter_data_path):
    """
    为分割器创建PNG掩码
    
    Args:
        input_base_path: 预处理后的PNG数据集根目录
        segmenter_data_path: 分割器数据路径
    """
    print(f"为分割器创建PNG掩码，源目录: {input_base_path}, 目标目录: {segmenter_data_path}")
    
    # 确保分割器数据目录存在
    os.makedirs(segmenter_data_path, exist_ok=True)
    
    # 细胞类型目录
    cell_types = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 
                  'Parabasal', 'Superficial-Intermediate']
    
    # 计数器
    processed_count = 0
    
    for cell_type in cell_types:
        png_dir = os.path.join(input_base_path, f"png_{cell_type}")
        
        # 递归处理所有子目录
        if os.path.exists(png_dir):
            print(f"处理细胞类型目录: {png_dir}")
            
            # 递归查找所有PNG图像文件
            for root, dirs, files in os.walk(png_dir):
                image_files = [f for f in files if f.endswith('.png') and not ('_nuc' in f or '_cyt' in f or '_edge' in f)]
                
                for img_file in image_files:
                    base_name = os.path.splitext(img_file)[0]
                    img_path = os.path.join(root, img_file)
                    
                    # 获取相对路径部分，用于保存文件名
                    rel_path = os.path.relpath(root, png_dir)
                    rel_path = "" if rel_path == "." else rel_path.replace("\\", "_").replace("/", "_") + "_"
                    
                    # 查找对应的掩码文件
                    cyt_files = [f for f in files if f.startswith(base_name) and '_cyt' in f and not '_edge' in f]
                    nuc_files = [f for f in files if f.startswith(base_name) and '_nuc' in f and not '_edge' in f]
                    
                    if cyt_files or nuc_files:
                        # 读取原始图像
                        image = cv2.imread(img_path)
                        if image is None:
                            print(f"无法读取图像: {img_path}")
                            continue
                        
                        # 创建分割掩码 - 0:背景, 1:细胞质, 2:细胞核
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        
                        # 添加细胞质区域
                        for cyt_file in cyt_files:
                            cyt_path = os.path.join(root, cyt_file)
                            cyt_mask = cv2.imread(cyt_path, cv2.IMREAD_GRAYSCALE)
                            if cyt_mask is not None:
                                mask[cyt_mask > 0] = 1
                            else:
                                print(f"无法读取细胞质掩码: {cyt_path}")
                        
                        # 添加细胞核区域 - 覆盖在细胞质上
                        for nuc_file in nuc_files:
                            nuc_path = os.path.join(root, nuc_file)
                            nuc_mask = cv2.imread(nuc_path, cv2.IMREAD_GRAYSCALE)
                            if nuc_mask is not None:
                                mask[nuc_mask > 0] = 2
                            else:
                                print(f"无法读取细胞核掩码: {nuc_path}")
                        
                        # 保存图像和掩码到分割器数据目录
                        img_save_name = f"{cell_type}_{rel_path}{base_name}.png"
                        mask_save_name = f"{cell_type}_{rel_path}{base_name}_mask.png"
                        
                        img_save_path = os.path.join(segmenter_data_path, img_save_name)
                        mask_save_path = os.path.join(segmenter_data_path, mask_save_name)
                        
                        # 确保文件名长度不会太长
                        if len(img_save_name) > 200:
                            img_save_name = f"{cell_type}_{base_name[-30:]}.png"
                            mask_save_name = f"{cell_type}_{base_name[-30:]}_mask.png"
                            img_save_path = os.path.join(segmenter_data_path, img_save_name)
                            mask_save_path = os.path.join(segmenter_data_path, mask_save_name)
                        
                        cv2.imwrite(img_save_path, image)
                        cv2.imwrite(mask_save_path, mask)
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            print(f"已处理 {processed_count} 对图像和掩码")
    
    print(f"为分割器创建的数据集处理完成，共生成 {processed_count} 对图像和掩码")

def main():
    parser = argparse.ArgumentParser(description="预处理SIPakMED数据集，将BMP和DAT格式转换为PNG图像和掩码")
    parser.add_argument("--input", type=str, default="F:/鳞癌自动化/鳞癌数据raw", help="输入数据集根目录")
    parser.add_argument("--output", type=str, default="F:/鳞癌自动化/代码测试/data/processed", help="输出数据集根目录")
    parser.add_argument("--segmenter", action="store_true", help="是否为分割器创建数据")
    parser.add_argument("--segmenter-output", type=str, default=SEGMENTER_DATA_PATH, help="分割器数据输出目录")
    args = parser.parse_args()
    
    # 预处理数据集
    preprocess_dataset(args.input, args.output)
    
    # 如果需要为分割器创建数据
    if args.segmenter:
        create_png_masks_for_segmenter(args.output, args.segmenter_output)
    
    print("数据预处理完成")

if __name__ == "__main__":
    main()