"""
测试 Albumentations 2.0.7 Resize 和 RandomResizedCrop 转换
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import sys

def test_albumentations():
    """测试 Albumentations 2.0.7 转换"""
    print(f"Albumentations 版本: {A.__version__}")
    
    # 创建测试图像和边界框
    test_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    test_boxes = [[10, 10, 50, 50], [100, 100, 200, 200]]
    test_labels = [0, 1]
    
    # 测试 size 参数的 Resize 转换
    print("\n测试 size 参数的 Resize 转换...")
    try:
        resize_transform = A.Compose([
            A.Resize(size=(512, 512), p=1.0),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        result = resize_transform(image=test_img, bboxes=test_boxes, class_labels=test_labels)
        print(f"Resize 成功! 转换后图像形状: {result['image'].shape}")
    except Exception as e:
        print(f"Resize 失败: {e}")
    
    # 测试 size 参数的 RandomResizedCrop 转换
    print("\n测试 size 参数的 RandomResizedCrop 转换...")
    try:
        crop_transform = A.Compose([
            A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=1.0),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        result = crop_transform(image=test_img, bboxes=test_boxes, class_labels=test_labels)
        print(f"RandomResizedCrop 成功! 转换后图像形状: {result['image'].shape}")
    except Exception as e:
        print(f"RandomResizedCrop 失败: {e}")

if __name__ == "__main__":
    test_albumentations()
