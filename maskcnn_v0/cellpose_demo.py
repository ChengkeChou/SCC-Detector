"""
直接使用CellPose进行细胞分割演示
不依赖于任何自定义模型，仅使用CellPose内置的预训练模型
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from tqdm import tqdm

# 确保安装了cellpose
try:
    from cellpose import models
except ImportError:
    print("缺少cellpose库，使用以下命令安装:")
    print("pip install cellpose")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('cellpose_demo.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_cellpose_demo(image_path, output_dir=None, cell_diameter=30, show=True):
    """
    使用CellPose进行细胞分割演示
    
    Args:
        image_path: 图像路径
        output_dir: 输出目录
        cell_diameter: 细胞直径参数
        show: 是否显示结果
    """
    logger.info(f"处理图像: {image_path}")
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"无法读取图像: {image_path}")
        return
    
    # 转换为RGB (CellPose需要RGB格式)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 加载CellPose模型
    logger.info("加载CellPose模型...")
    model = models.CellposeModel(
        gpu=False,  # 设置为True以使用GPU
        model_type="cyto"  # 使用"cyto"细胞质分割模型，也可以用"nuclei"进行细胞核分割
    )
    
    # 运行CellPose分割
    logger.info(f"运行CellPose分割，图像尺寸: {img_rgb.shape}")
    start_time = time.time()
    
    # CellPose参数
    channels = [0, 0]  # 使用第一个通道
    
    # 执行分割
    masks, flows, styles, diams = model.eval(
        img_rgb, 
        diameter=cell_diameter,
        channels=channels,
        flow_threshold=0.4,  # 流场阈值
        cellprob_threshold=0.5,  # 细胞概率阈值
        min_size=15  # 最小细胞大小（像素）
    )
    
    # 计算用时
    inference_time = time.time() - start_time
    
    # 获取检测到的细胞数量
    unique_cells = np.unique(masks)[1:]  # 跳过背景(0)
    num_cells = len(unique_cells)
    
    logger.info(f"检测完成: 发现 {num_cells} 个细胞，耗时 {inference_time:.2f} 秒")
    
    # 创建可视化结果
    # 1. 为每个细胞分配随机颜色
    np.random.seed(42)  # 确保每次运行颜色一致
    colors = np.random.randint(0, 255, (num_cells + 1, 3))
    colors[0] = [0, 0, 0]  # 背景为黑色
    
    # 2. 创建彩色掩码图像
    mask_colored = np.zeros_like(img_rgb)
    for i, cell_id in enumerate(unique_cells):
        mask_colored[masks == cell_id] = colors[i + 1]
    
    # 3. 将掩码与原图叠加
    alpha = 0.4  # 透明度
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, mask_colored, alpha, 0)
    
    # 4. 绘制每个细胞的边界框和编号
    result = overlay.copy()
    cell_boxes = []
    
    for i, cell_id in enumerate(unique_cells):
        # 获取当前细胞的掩码
        cell_mask = masks == cell_id
        
        # 找到边界
        y_indices, x_indices = np.where(cell_mask)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 保存边界框
        cell_boxes.append((x_min, y_min, x_max, y_max))
        
        # 绘制边界框
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 绘制编号
        cv2.putText(result, str(i+1), (x_min, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 添加总计信息
    cv2.putText(result, f"总计: {num_cells} 个细胞", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存可视化结果
        result_path = os.path.join(output_dir, f"{Path(image_path).stem}_result.png")
        cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        # 保存原始掩码 (用于后续处理)
        mask_path = os.path.join(output_dir, f"{Path(image_path).stem}_mask.png")
        cv2.imwrite(mask_path, masks.astype(np.uint16))
        
        logger.info(f"结果已保存至: {result_path}")
    
    # 显示结果
    if show:
        plt.figure(figsize=(15, 10))
        
        # 显示原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(img_rgb)
        plt.title("原始图像")
        plt.axis('off')
        
        # 显示掩码
        plt.subplot(1, 3, 2)
        plt.imshow(mask_colored)
        plt.title(f"细胞掩码 ({num_cells}个)")
        plt.axis('off')
        
        # 显示结果
        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title("分割结果")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return {
        "num_cells": num_cells,
        "masks": masks,
        "cell_boxes": cell_boxes,
        "inference_time": inference_time,
        "result_image": result
    }

def process_folder(folder_path, output_dir, cell_diameter=30):
    """批量处理文件夹中的图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_paths = []
    for ext in ['.bmp', '.jpg', '.jpeg', '.png']:
        image_paths.extend(list(Path(folder_path).glob(f"*{ext}")))
    
    if not image_paths:
        logger.warning(f"在 {folder_path} 中未找到图像文件")
        return
    
    logger.info(f"开始处理 {len(image_paths)} 个图像...")
    
    # 处理每个图像
    results = []
    for path in tqdm(image_paths, desc="处理图像"):
        try:
            result = run_cellpose_demo(str(path), output_dir, cell_diameter, show=False)
            if result:
                results.append((str(path), result))
        except Exception as e:
            logger.error(f"处理图像 {path} 时出错: {e}")
    
    # 输出统计信息
    if results:
        total_cells = sum(r[1]["num_cells"] for r in results)
        avg_cells = total_cells / len(results)
        avg_time = sum(r[1]["inference_time"] for r in results) / len(results)
        
        logger.info("\n处理完成统计:")
        logger.info(f"  - 成功处理图像: {len(results)}/{len(image_paths)}")
        logger.info(f"  - 检测到总细胞数: {total_cells}")
        logger.info(f"  - 平均每张图像细胞数: {avg_cells:.1f}")
        logger.info(f"  - 平均处理时间: {avg_time:.2f} 秒/图像")
    
    return results

def extract_cell_regions(image_path, mask_path, output_dir):
    """从掩码中提取单个细胞区域用于分类"""
    # 读取图像和掩码
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"无法读取图像: {image_path}")
        return
    
    # 掩码可能是16位存储，直接读取
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        logger.error(f"无法读取掩码: {mask_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有非零细胞ID
    unique_ids = np.unique(mask)[1:] # 跳过0背景
    
    logger.info(f"从 {image_path} 中提取 {len(unique_ids)} 个细胞区域")
    
    # 为每个细胞ID提取区域
    for cell_id in unique_ids:
        # 创建细胞掩码
        cell_mask = (mask == cell_id).astype(np.uint8)
        
        # 获取边界框
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # 添加一些边距
        margin = 10
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(image.shape[1], x + w + margin)
        y_max = min(image.shape[0], y + h + margin)
        
        # 提取细胞区域
        cell_region = image[y_min:y_max, x_min:x_max].copy()
        
        # 保存细胞图像
        cell_filename = f"cell_{Path(image_path).stem}_{cell_id}.png"
        cell_output_path = os.path.join(output_dir, cell_filename)
        cv2.imwrite(cell_output_path, cell_region)
    
    logger.info(f"细胞区域已保存到: {output_dir}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CellPose细胞分割演示")
    parser.add_argument("--image", type=str, help="单个图像路径")
    parser.add_argument("--folder", type=str, help="图像文件夹路径")
    parser.add_argument("--output", type=str, default="./cellpose_demo_results", help="输出目录")
    parser.add_argument("--diameter", type=int, default=30, help="细胞直径参数")
    parser.add_argument("--extract", action="store_true", help="提取细胞区域用于分类")
    parser.add_argument("--mask", type=str, help="用于提取细胞的掩码路径 (与--extract一起使用)")
    
    args = parser.parse_args()
    
    # 检查参数
    if args.extract:
        if not args.image or not args.mask:
            parser.error("使用--extract时必须同时提供--image和--mask参数")
        
        extract_cell_regions(args.image, args.mask, args.output)
    elif args.image:
        run_cellpose_demo(args.image, args.output, args.diameter)
    elif args.folder:
        process_folder(args.folder, args.output, args.diameter)
    else:
        parser.error("必须提供--image或--folder参数")

if __name__ == "__main__":
    main()
