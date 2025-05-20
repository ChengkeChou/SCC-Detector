import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# 测试修复后的转换
def test_transform():
    # 创建测试图像
    test_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    test_boxes = [[10, 10, 50, 50], [100, 100, 200, 200]]
    test_labels = [0, 1]    # 使用正确的 API 参数 (size 而不是 height 和 width)
    transform = A.Compose([
        A.Resize(size=(512, 512), p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    # 应用转换
    try:
        transformed = transform(image=test_img, bboxes=test_boxes, class_labels=test_labels)
        print("转换成功！")
        print(f"原始图像尺寸: {test_img.shape}")
        print(f"转换后图像尺寸: {transformed['image'].shape}")
        print(f"转换后边界框: {transformed['bboxes']}")
        return True
    except Exception as e:
        print(f"转换失败: {e}")
        return False

if __name__ == "__main__":
    print("测试 Albumentations Resize 转换...")
    test_transform()
