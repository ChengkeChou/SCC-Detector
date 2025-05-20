"""
测试数据集加载功能
"""
import os
import logging
import torch
from pathlib import Path
from models.hybrid_cell_segmentation import CellSegmentationDataset, get_transform

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dataset():
    """测试数据集加载功能"""
    data_dir = Path("./data")
    transform = get_transform(train=True)
    
    # 创建数据集
    try:
        dataset = CellSegmentationDataset(data_dir, split='train', transform=transform)
        logger.info(f"成功创建数据集，包含 {len(dataset)} 个样本")
        
        # 测试获取单个样本
        sample = dataset[0]
        logger.info(f"样本包含的键: {list(sample.keys())}")
        logger.info(f"图像形状: {sample['image'].shape}")
        logger.info(f"掩码形状: {sample['mask'].shape}")
        logger.info(f"边界框: {sample['boxes'].shape}")
        logger.info(f"标签: {sample['labels'].shape}")
        logger.info(f"图像ID: {sample['image_id']}")
        
        # 测试数据加载器
        from torch.utils.data import DataLoader
        
        def collate_fn(batch):
            """自定义数据批处理函数"""
            return batch
        
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=True, 
            num_workers=0,  # 使用0个工作进程以便于调试
            collate_fn=collate_fn
        )
        
        # 测试一个批次
        logger.info("尝试加载一个批次...")
        for batch in dataloader:
            logger.info(f"成功加载批次: {len(batch)} 个图像")
            # 提取图像和目标
            images = [item['image'] for item in batch]
            targets = [{k: v for k, v in item.items() if k != 'image'} for item in batch]
            logger.info(f"提取的图像数量: {len(images)}")
            logger.info(f"提取的目标数量: {len(targets)}")
            break
            
        logger.info("数据集加载测试成功!")
    except Exception as e:
        logger.error(f"数据集加载测试失败: {str(e)}")
        raise

if __name__ == "__main__":
    test_dataset()