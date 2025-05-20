import albumentations as A
from albumentations.pytorch import ToTensorV2
import inspect

# 检查 Resize 类的属性和方法
resize = A.Resize(height=512, width=512)
print("Resize 类属性:")
for attr in dir(resize):
    if not attr.startswith("_"):
        print(f"- {attr}")

print("\nResize 类的原始代码:")
print(inspect.getsource(A.Resize))

# 尝试使用 Resize 进行转换
print("\n测试 Resize 转换：")
import numpy as np
import cv2

# 创建一个小的测试图像
test_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
transformed = resize(image=test_img)
print(f"原始图像尺寸: {test_img.shape}")
print(f"转换后图像尺寸: {transformed['image'].shape}")

# 测试完整的转换管道
transform = A.Compose([
    A.Resize(height=512, width=512),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# 创建测试边界框
test_boxes = [[10, 10, 50, 50], [100, 100, 200, 200]]
test_labels = [0, 1]

# 应用转换
transformed = transform(image=test_img, bboxes=test_boxes, class_labels=test_labels)
print(f"\n转换后边界框: {transformed['bboxes']}")
print(f"转换后图像尺寸 (Tensor): {transformed['image'].shape}")
