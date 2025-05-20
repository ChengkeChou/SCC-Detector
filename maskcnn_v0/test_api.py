"""
测试 Albumentations 2.0.7 API 接口正确性
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import sys

def test_transforms():
    print(f"Albumentations 版本: {A.__version__}")
    
    # 创建测试图像
    test_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    
    # 测试每种可能的参数格式
    transforms = [
        # 方式1：使用 height 和 width
        A.Resize(height=512, width=512),
        
        # 方式2：使用 size
        A.Resize(size=(512, 512)),
        
        # 方式3：使用 width 和 height
        A.RandomResizedCrop(width=512, height=512, scale=(0.8, 1.0)),
        
        # 方式4：使用 size
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0))
    ]
    
    # 测试每种转换
    for i, transform in enumerate(transforms):
        try:
            result = transform(image=test_img)
            print(f"转换 {i+1} 成功! 结果形状: {result['image'].shape}")
        except Exception as e:
            print(f"转换 {i+1} 失败: {e}")
    
    return True

if __name__ == "__main__":
    test_transforms()
