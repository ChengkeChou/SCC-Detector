"""
修复 Mask R-CNN 输入格式问题的测试脚本
"""
import os
import logging
import torch
import numpy as np
from pathlib import Path
from models.hybrid_cell_segmentation import CellSegmentationDataset, get_transform

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 加载数据集
    data_dir = Path("./data")
    transform = get_transform(train=True)
    
    # 创建数据集
    try:
        dataset = CellSegmentationDataset(data_dir, split='train', transform=transform)
        logger.info(f"成功创建数据集，包含 {len(dataset)} 个样本")
        
        # 测试获取单个样本
        sample = dataset[0]
        logger.info(f"样本包含的键: {list(sample.keys())}")
        
        # 修复掩码格式问题
        if 'mask' in sample and 'masks' not in sample:
            mask = sample.pop('mask')
            sample['masks'] = mask.unsqueeze(0)  # 添加一个维度，使其变为 [1, H, W]
            logger.info("已将 'mask' 转换为 'masks'")
        
        logger.info(f"修改后的样本键: {list(sample.keys())}")
        if 'masks' in sample:
            logger.info(f"masks 形状: {sample['masks'].shape}")
            
        # 创建一个小批次
        batch = [sample, sample]  # 简化批次创建
        
        # 从批次中提取数据
        images = [item['image'] for item in batch]
        targets = [{k: v for k, v in item.items() if k != 'image'} for item in batch]
        
        logger.info(f"targets 中的键: {list(targets[0].keys())}")
        logger.info("测试成功!")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
