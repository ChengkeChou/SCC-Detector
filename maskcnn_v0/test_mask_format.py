"""
测试 Mask R-CNN 的 mask 格式修复
"""
import os
import torch
import logging
import numpy as np
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

def generate_instance_masks(mask, boxes, class_ids):
    """
    将语义分割掩码转换为实例分割掩码
    
    Args:
        mask: 语义分割掩码 (H, W)
        boxes: 边界框坐标 [N, 4] (x_min, y_min, x_max, y_max)
        class_ids: 类别标签 [N]
        
    Returns:
        instance_masks: 实例分割掩码 [N, H, W]
    """
    if len(boxes) == 0:
        # 如果没有边界框，返回空掩码
        return torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
    
    h, w = mask.shape
    instance_masks = []
    
    # 将掩码转换为numpy数组（如果它是张量）
    if isinstance(mask, torch.Tensor):
        mask_np = mask.numpy()
    else:
        mask_np = mask
    
    for i, box in enumerate(boxes):
        # 创建单个实例的掩码
        instance_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 将边界框转换为整数坐标
        x_min, y_min, x_max, y_max = map(int, box)
        
        # 确保坐标在图像范围内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # 在边界框内截取语义掩码（假设非零值表示前景）
        roi_mask = mask_np[y_min:y_max, x_min:x_max]
        
        # 将边界框区域的语义掩码复制到实例掩码中
        if roi_mask.size > 0:  # 确保ROI不为空
            instance_mask[y_min:y_max, x_min:x_max] = (roi_mask > 0).astype(np.uint8)
        
        # 添加到实例掩码列表
        instance_masks.append(torch.from_numpy(instance_mask))
    
    # 将掩码堆叠为 [N, H, W] 形状的张量
    if instance_masks:
        instance_masks = torch.stack(instance_masks)
    else:
        instance_masks = torch.zeros((0, h, w), dtype=torch.uint8)
    
    return instance_masks

def custom_collate_fn(batch):
    """自定义批处理函数，确保包含masks键"""
    fixed_batch = []
    
    for item in batch:
        fixed_item = {}
        for k, v in item.items():
            if k != 'mask':
                fixed_item[k] = v
        
        # 如果有'mask'键但没有'masks'键，生成实例掩码
        if 'mask' in item and 'masks' not in item:
            mask = item['mask']
            boxes = item['boxes']
            labels = item['labels']
            
            fixed_item['masks'] = generate_instance_masks(mask, boxes, labels)
        
        fixed_batch.append(fixed_item)
    
    return fixed_batch

def test_mask_format():
    """测试掩码格式是否符合Mask R-CNN要求"""
    # 加载数据集
    data_dir = Path("./data")
    transform = get_transform(train=True)
    
    try:
        # 创建数据集
        logger.info("正在加载数据集...")
        dataset = CellSegmentationDataset(data_dir, split='train', transform=transform)
        logger.info(f"成功加载数据集: {len(dataset)} 个样本")
        
        # 检查第一个样本
        sample = dataset[0]
        logger.info(f"原始样本的键: {list(sample.keys())}")
        
        # 修复第一个样本的格式
        fixed_sample = {}
        for k, v in sample.items():
            if k != 'mask':
                fixed_sample[k] = v
        
        if 'mask' in sample and 'masks' not in sample:
            mask = sample['mask']
            boxes = sample['boxes']
            labels = sample['labels']
            
            instance_masks = generate_instance_masks(mask, boxes, labels)
            fixed_sample['masks'] = instance_masks
        
        logger.info(f"修复后的键: {list(fixed_sample.keys())}")
        if 'masks' in fixed_sample:
            logger.info(f"实例掩码形状: {fixed_sample['masks'].shape}")
        
        # 创建 DataLoader
        logger.info("创建DataLoader...")
        data_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        # 加载第一个批次
        logger.info("加载一个批次...")
        batch = next(iter(data_loader))
        logger.info(f"批次长度: {len(batch)}")
        logger.info(f"第一个样本的键: {list(batch[0].keys())}")
        
        # 检查是否包含masks键
        has_masks_key = all('masks' in item for item in batch)
        logger.info(f"所有样本都有masks键: {has_masks_key}")
        
        # 创建模型
        logger.info("创建模型...")
        model = HybridCellSegmentationModel(num_classes=dataset.num_classes)
        
        # 测试前向传播
        logger.info("执行前向传播...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 归一化图像
        images = [item['image'] for item in batch]
        for idx, img in enumerate(images):
            if img.dtype == torch.uint8:
                images[idx] = img.float() / 255.0
        
        # 准备目标
        targets = [{k: v for k, v in item.items() if k != 'image'} for item in batch]
        
        # 移动到设备
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # 设置训练模式
        model.train()
        
        # 前向传播
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        logger.info(f"损失: {losses.item()}")
        logger.info("测试成功：掩码格式已修复")
        
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mask_format()
