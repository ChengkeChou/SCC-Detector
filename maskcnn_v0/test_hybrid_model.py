"""
测试混合模型加载和推理脚本
确认加载检查点格式的模型文件是否正常工作
"""

import os
import sys
import logging
import torch
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('test_hybrid_model.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 确保能够导入混合模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid_cell_segmentation import HybridCellSegmentationModel

def test_model_loading(model_path, num_classes=5, device=None):
    """测试模型加载功能"""
    logger.info(f"开始测试模型加载: {model_path}")
    try:
        # 设置设备
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
            
        # 创建模型
        model = HybridCellSegmentationModel(num_classes=num_classes, device=device)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 打印检查点内容结构
        logger.info("检查点内容结构:")
        for key in checkpoint.keys():
            logger.info(f"- {key}: {type(checkpoint[key])}")
        
        # 检查是否是检查点文件（包含model_state_dict）
        if "model_state_dict" in checkpoint:
            # 从检查点中提取模型状态字典
            state_dict = checkpoint["model_state_dict"]
            logger.info("从检查点文件中提取模型权重")
        else:
            # 假设是直接的状态字典
            state_dict = checkpoint
            logger.info("直接使用状态字典")
            
        # 打印状态字典内容结构
        num_keys = len(state_dict.keys())
        logger.info(f"状态字典包含 {num_keys} 个键")
        if num_keys > 0:
            sample_keys = list(state_dict.keys())[:5]  # 只显示前5个键
            logger.info(f"示例键: {sample_keys}")
            
        # 加载状态字典
        model.load_state_dict(state_dict)
        logger.info("成功加载模型")
        
        # 设置为评估模式
        model.eval()
        logger.info("模型设置为评估模式")
        
        return True, "模型加载成功"
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return False, str(e)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试混合模型加载")
    parser.add_argument("--model", type=str, default="", help="模型文件路径")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cpu 或 cuda)")
    args = parser.parse_args()
    
    # 如果没有提供模型路径,尝试搜索默认位置
    if not args.model:
        # 搜索常见位置
        potential_paths = [
            "./models/best_model.pth",
            "./output/best_model.pth",
            "../output/best_model.pth"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                args.model = path
                break
    
    if not args.model or not os.path.exists(args.model):
        logger.error("请提供有效的模型文件路径")
        sys.exit(1)
    
    # 设置设备
    device = None
    if args.device:
        device = args.device
    
    # 测试模型加载
    success, message = test_model_loading(args.model, device=device)
    
    if success:
        logger.info("测试成功: " + message)
    else:
        logger.error("测试失败: " + message)
