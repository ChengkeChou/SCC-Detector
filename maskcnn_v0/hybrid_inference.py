"""
细胞分割混合模型推理脚本
使用训练好的混合模型直接对图像进行细胞分割和检测
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('hybrid_inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 确保能够导入混合模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid_cell_segmentation import HybridCellSegmentationModel

# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {DEVICE}")

def load_model(model_path, num_classes=5, device=None):
    """加载训练好的混合模型"""
    try:
        # 如果没有指定设备，使用全局设备
        if device is None:
            device = DEVICE
            logger.info(f"使用默认设备: {device}")
        
        # 记录详细的设备信息
        logger.info(f"加载模型到设备: {device}, 类型: {type(device)}")
        
        # 实例化模型
        model = HybridCellSegmentationModel(num_classes=num_classes, device=device)
        logger.info("成功创建HybridCellSegmentationModel实例")
        
        # 加载模型权重
        logger.info(f"尝试加载模型权重: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            logger.info(f"成功加载权重文件，检查点类型: {type(checkpoint)}")
        except Exception as e:
            logger.error(f"加载权重文件失败: {str(e)}")
            raise
        
        # 检查是否是检查点文件（包含model_state_dict）
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # 从检查点中提取模型状态字典
            state_dict = checkpoint["model_state_dict"]
            logger.info("从检查点文件中提取模型权重")
        else:
            # 假设是直接的状态字典
            state_dict = checkpoint
            logger.info("直接使用加载的状态字典")
        
        # 加载状态字典到模型
        try:
            model.load_state_dict(state_dict)
            logger.info("成功将权重加载到模型中")
        except Exception as e:
            logger.error(f"加载状态字典到模型失败: {str(e)}")
            raise
        
        # 确保模型在正确的设备上并处于评估模式
        model.to(device)
        model.eval()
        logger.info(f"成功加载模型到 {device}, 模型现在处于评估模式")
        
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise

def preprocess_image(image_path):
    """预处理图像"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return None

        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 记录原始尺寸用于后续显示
        orig_height, orig_width = image.shape[:2]
        logger.info(f"图像尺寸: {orig_width}x{orig_height}")
        
        return image
    except Exception as e:
        logger.error(f"图像预处理失败: {e}")
        return None

def visualize_results(image, results, output_path=None, confidence_threshold=0.5):
    """可视化检测和分割结果"""
    try:
        # 复制原始图像用于绘制
        vis_image = image.copy()
        
        # 获取检测结果
        boxes = results['boxes'].numpy() if isinstance(results['boxes'], torch.Tensor) else results['boxes']
        scores = results['scores'].numpy() if isinstance(results['scores'], torch.Tensor) else results['scores']
        labels = results['labels'].numpy() if isinstance(results['labels'], torch.Tensor) else results['labels']
        masks = results['masks'].numpy() if isinstance(results['masks'], torch.Tensor) else results['masks']
        
        # 颜色映射，为每个类别分配一种颜色
        colors = [
            (255, 0, 0),   # 红色
            (0, 255, 0),   # 绿色
            (0, 0, 255),   # 蓝色
            (255, 255, 0), # 黄色
            (255, 0, 255), # 紫色
            (0, 255, 255), # 青色
            (128, 0, 0),   # 深红
            (0, 128, 0),   # 深绿
            (0, 0, 128)    # 深蓝
        ]
        
        # 绘制每个检测结果
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # 只显示置信度高于阈值的结果
            if score < confidence_threshold:
                continue

            # 绘制边界框
            x1, y1, x2, y2 = map(int, box)
            color = colors[int(label) % len(colors)]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签和置信度文本
            text = f"类别 {int(label)}: {score:.2f}"
            cv2.putText(vis_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制掩码（透明叠加）
            if i < len(masks):
                mask = masks[i]
                if mask.ndim == 3:
                    mask = mask[0]  # 获取第一个通道
                
                # 创建彩色掩码
                color_mask = np.zeros_like(vis_image, dtype=np.uint8)
                color_mask[mask > 0.5] = color
                
                # 透明叠加
                alpha = 0.3  # 透明度
                mask_area = (mask > 0.5)
                vis_image[mask_area] = cv2.addWeighted(vis_image[mask_area], 1-alpha, color_mask[mask_area], alpha, 0)
        
        # 显示和保存结果
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title(f"检测到 {len(boxes)} 个细胞")
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"结果已保存至: {output_path}")
            
        plt.close()
        
        # 返回可视化结果
        return vis_image
    except Exception as e:
        logger.error(f"可视化结果失败: {e}")
        return image

def process_single_image(model, image_path, output_dir, confidence_threshold=0.5):
    """处理单张图像"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 预处理图像
        image = preprocess_image(image_path)
        if image is None:
            return None
        
        # 使用模型进行推理
        results = model.predict(image, confidence_threshold=confidence_threshold)
        
        # 统计检测到的细胞数量
        num_cells = len(results['boxes'])
        logger.info(f"在图像 {os.path.basename(image_path)} 中检测到 {num_cells} 个细胞")
        
        # 构建输出路径
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{image_name}_result.png")
        
        # 可视化并保存结果
        vis_image = visualize_results(image, results, output_path, confidence_threshold)
        
        # 将分割掩码保存为单独的图像
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        # 保存检测结果（边界框、类别、置信度）为JSON
        json_result = {
            "image_path": image_path,
            "num_cells": num_cells,
            "cells": []
        }
        
        # 保存每个细胞的掩码和详细信息
        for i, (box, score, label) in enumerate(zip(
            results['boxes'].tolist() if isinstance(results['boxes'], torch.Tensor) else results['boxes'],
            results['scores'].tolist() if isinstance(results['scores'], torch.Tensor) else results['scores'],
            results['labels'].tolist() if isinstance(results['labels'], torch.Tensor) else results['labels']
        )):
            # 只保存置信度高于阈值的结果
            if score < confidence_threshold:
                continue
                
            # 保存掩码
            mask_path = os.path.join(masks_dir, f"{image_name}_cell_{i+1}.png")
            if i < len(results['masks']):
                mask = results['masks'][i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                if mask.ndim == 3:
                    mask = mask[0]  # 获取第一个通道
                
                # 保存二值掩码
                cv2.imwrite(mask_path, (mask > 0.5).astype(np.uint8) * 255)
            
            # 添加到JSON结果
            json_result["cells"].append({
                "id": i + 1,
                "box": box,
                "class": int(label),
                "confidence": score,
                "mask_path": mask_path
            })
        
        # 保存JSON结果
        json_path = os.path.join(output_dir, f"{image_name}_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=4, ensure_ascii=False)
        
        logger.info(f"处理完成，结果已保存至 {output_dir}")
        return vis_image, results, json_result
    
    except Exception as e:
        logger.error(f"处理图像失败: {e}")
        return None

def process_directory(model, input_dir, output_dir, confidence_threshold=0.5):
    """处理目录中的所有图像"""
    try:
        # 获取目录中的所有图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(Path(input_dir).glob(f"*{ext}")))
        
        logger.info(f"在目录 {input_dir} 中找到 {len(image_paths)} 张图像")
        
        # 创建汇总报告
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_directory": input_dir,
            "output_directory": output_dir,
            "total_images": len(image_paths),
            "results": []
        }
        
        # 处理每张图像
        for image_path in image_paths:
            logger.info(f"处理图像: {image_path}")
            result = process_single_image(model, str(image_path), output_dir, confidence_threshold)
            
            if result:
                _, _, json_result = result
                summary["results"].append({
                    "image_path": str(image_path),
                    "num_cells": json_result["num_cells"]
                })
        
        # 保存汇总报告
        summary_path = os.path.join(output_dir, "summary_report.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        logger.info(f"已处理 {len(image_paths)} 张图像，汇总报告已保存至 {summary_path}")
    
    except Exception as e:
        logger.error(f"处理目录失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="细胞分割混合模型推理脚本")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth", help="模型路径")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径或目录")
    parser.add_argument("--output-dir", type=str, default="output/hybrid_results", help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--num-classes", type=int, default=5, help="类别数量")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cpu 或 cuda)")
    
    args = parser.parse_args()
    
    try:
        # 设置设备
        device = None
        if args.device:
            device = torch.device(args.device)
            
        # 加载模型
        model = load_model(args.model_path, num_classes=args.num_classes, device=device)
        
        # 检查输入是文件还是目录
        if os.path.isfile(args.input):
            # 处理单张图像
            process_single_image(model, args.input, args.output_dir, args.threshold)
        elif os.path.isdir(args.input):
            # 处理目录中的所有图像
            process_directory(model, args.input, args.output_dir, args.threshold)
        else:
            logger.error(f"输入路径 {args.input} 不存在")
            sys.exit(1)
        
        logger.info("推理完成！")
    
    except Exception as e:
        logger.error(f"推理过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
