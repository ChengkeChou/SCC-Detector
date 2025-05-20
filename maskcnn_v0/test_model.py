"""
测试脚本 - 检查细胞分割系统的数据处理和模型训练
"""
import os
import torch
import logging
from pathlib import Path
from models.hybrid_cell_segmentation import (
    CellSegmentationDataset, 
    get_transform, 
    HybridCellSegmentationModel,
    collate_fn
)
from torch.utils.data import DataLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")

def test_dataset_and_model():
    """测试数据集加载和模型训练"""
    # 加载数据集
    data_dir = Path("./data")
    transform = get_transform(train=True)
    
    try:
        # 创建并测试数据集
        logger.info("正在加载数据集...")
        dataset = CellSegmentationDataset(data_dir, split='train', transform=transform)
        logger.info(f"成功加载数据集: {len(dataset)} 个样本")
        
        # 检查单个样本
        sample = dataset[0]
        logger.info(f"样本图像类型: {sample['image'].dtype}, 形状: {sample['image'].shape}")
        logger.info(f"样本图像值范围: {sample['image'].min().item()} - {sample['image'].max().item()}")
        
        # 创建模型
        logger.info("正在创建模型...")
        model = HybridCellSegmentationModel(num_classes=dataset.num_classes)
        model.to(device)
        logger.info("成功创建模型")
        
        # 创建 DataLoader
        logger.info("正在创建 DataLoader...")
        data_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # 使用 0 以便于调试
            collate_fn=collate_fn
        )
        logger.info("成功创建 DataLoader")
        
        # 尝试一次前向传播
        logger.info("正在测试前向传播...")
        batch = next(iter(data_loader))
        
        # 从批次中提取数据
        images = [item['image'] for item in batch]
        targets = [{k: v for k, v in item.items() if k != 'image'} for item in batch]
        
        # 确保图像是浮点类型并且值在0-1范围内
        for idx, img in enumerate(images):
            if img.dtype == torch.uint8:
                images[idx] = img.float() / 255.0
                
        logger.info(f"图像类型: {images[0].dtype}, 值范围: {images[0].min().item()} - {images[0].max().item()}")
        
        # 移动到指定设备
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # 设置为训练模式
        model.train()
        
        # 前向传播
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        logger.info(f"前向传播成功，损失: {losses.item()}")
        logger.info(f"损失字典: {', '.join([f'{k}: {v.item():.4f}' for k, v in loss_dict.items()])}")
        
        logger.info("测试完成: 数据集和模型可以正常工作！")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_and_model()
