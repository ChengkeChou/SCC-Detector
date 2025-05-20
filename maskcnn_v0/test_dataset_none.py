"""
测试数据集加载功能，特别是 None 值的处理
"""
import os
import sys
import logging
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from models.hybrid_cell_segmentation import CellSegmentationDataset, get_transform, collate_fn

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_dataset_loading():
    """测试数据集加载和处理"""
    data_dir = Path("./data")
    transform = get_transform(train=True)
    
    try:
        # 创建数据集
        dataset = CellSegmentationDataset(data_dir, split='train', transform=transform)
        logger.info(f"创建的数据集包含 {len(dataset)} 个样本")
        
        # 计数无效样本
        valid_samples = 0
        invalid_samples = 0
        
        for i in range(min(50, len(dataset))):
            try:
                sample = dataset[i]
                if sample is not None:
                    valid_samples += 1
                    logger.info(f"样本 {i} 有效: 图像形状 {sample['image'].shape}, 键: {list(sample.keys())}")
                else:
                    invalid_samples += 1
                    logger.warning(f"样本 {i} 无效 (返回 None)")
            except Exception as e:
                invalid_samples += 1
                logger.error(f"加载样本 {i} 时出错: {str(e)}")
        
        logger.info(f"测试结果: {valid_samples} 个有效样本, {invalid_samples} 个无效样本")
        
        # 测试数据加载器
        logger.info("测试数据加载器...")
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # 使用0个工作进程以便于调试
            collate_fn=collate_fn
        )
        
        # 测试批次加载
        valid_batches = 0
        empty_batches = 0
        
        for i, batch in enumerate(dataloader):
            if batch:
                valid_batches += 1
                logger.info(f"批次 {i} 包含 {len(batch)} 个样本")
                
                # 检查批次中的样本
                for j, item in enumerate(batch):
                    logger.info(f"  样本 {j} 键: {list(item.keys())}")
            else:
                empty_batches += 1
                logger.warning(f"批次 {i} 为空")
                
            # 只测试前5个批次
            if i >= 4:
                break
        
        logger.info(f"数据加载器测试结果: {valid_batches} 个有效批次, {empty_batches} 个空批次")
        logger.info("测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_loading()
