"""
使用CellPose直接测试细胞分割的简单脚本
这个脚本直接使用CellPose来分割细胞，不依赖其他系统组件
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('cellpose_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_cellpose(image_path, output_dir=None):
    """使用CellPose进行细胞分割测试"""
    try:
        # 尝试导入cellpose
        from cellpose import models, plot
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图像: {image_path}")
            return False
            
        # 转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        logger.info(f"测试图像尺寸: {img_rgb.shape}")
        
        # 加载CellPose模型 - 使用内置的cyto模型
        model = models.CellposeModel(model_type="cyto")
        
        # 设置参数 - 根据实际情况可能需要调整
        channels = [0, 0]  # 第一个通道作为主要通道
        diameter = 30  # 估计的细胞直径
        
        # 宽松参数，提高检出率
        flow_threshold = 0.3
        cellprob_threshold = 0.0
        
        logger.info(f"使用参数: diameter={diameter}, flow_threshold={flow_threshold}, cellprob_threshold={cellprob_threshold}")
        
        # 运行分割
        masks, flows, styles, diams = model.eval(
            img_rgb, 
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        
        # 统计检测到的细胞数量
        unique_ids = np.unique(masks)[1:]  # 跳过0（背景）
        cell_count = len(unique_ids)
        
        logger.info(f"在图像 {os.path.basename(image_path)} 中检测到 {cell_count} 个细胞")
        
        if cell_count == 0:
            logger.warning("未检测到细胞，可能需要调整参数!")
        
        # 创建输出目录
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成图像名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_cellpose_result.png")
        
        # 可视化结果
        fig = plt.figure(figsize=(12, 8))
        
        # 绘制原始图像
        plt.subplot(121)
        plt.imshow(img_rgb)
        plt.title('原始图像')
        plt.axis('off')
        
        # 绘制分割结果
        plt.subplot(122)
        outlines = plot.outlines_list(masks)
        plt.imshow(img_rgb)
        for o in outlines:
            plt.plot(o[:, 0], o[:, 1], 'r')
        plt.title(f'细胞分割结果 (检测到 {cell_count} 个细胞)')
        plt.axis('off')
        
        # 保存图像
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"分割结果已保存至: {output_path}")
        
        # 也可以保存细胞掩码
        mask_path = os.path.join(output_dir, f"{base_name}_cell_masks.png")
        cv2.imwrite(mask_path, masks.astype(np.uint8) * 50)
        logger.info(f"细胞掩码已保存至: {mask_path}")
        
        return cell_count > 0
        
    except ImportError:
        logger.error("CellPose未安装。请安装: pip install cellpose")
        return False
    except Exception as e:
        logger.error(f"测试CellPose时出错: {e}")
        return False

def find_test_images():
    """查找可用的测试图像"""
    # 当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 可能的图像路径列表
    image_paths = []
    
    # 查找data目录下的图像
    data_dir = os.path.join(current_dir, "data")
    if os.path.exists(data_dir):
        # 检查是否有分类目录
        for class_dir in ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]:
            class_img_dir = os.path.join(data_dir, class_dir, "images")
            if os.path.exists(class_img_dir):
                for img_file in os.listdir(class_img_dir):
                    if img_file.endswith(('.bmp', '.jpg', '.png')):
                        image_paths.append(os.path.join(class_img_dir, img_file))
                        if len(image_paths) >= 3:  # 最多找3张图
                            break
    
    return image_paths

if __name__ == "__main__":
    # 获取命令行参数
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 如果没有提供图像路径，尝试找一些测试图像
    if not image_path:
        image_paths = find_test_images()
        if not image_paths:
            logger.error("未找到测试图像，请提供图像路径作为参数")
            sys.exit(1)
        
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cellpose_test_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试找到的图像
        success_count = 0
        for path in image_paths:
            logger.info(f"测试图像: {path}")
            if test_cellpose(path, output_dir):
                success_count += 1
        
        # 输出结果摘要
        if success_count > 0:
            logger.info(f"CellPose测试成功: {success_count}/{len(image_paths)} 张图像成功分割出细胞")
        else:
            logger.error("CellPose测试失败: 未能在任何图像中检测到细胞")
    else:
        # 测试指定图像
        if os.path.exists(image_path):
            test_cellpose(image_path)
        else:
            logger.error(f"找不到指定的图像: {image_path}")
            sys.exit(1)
