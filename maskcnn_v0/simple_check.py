import albumentations as A
from albumentations.pytorch import ToTensorV2

print("测试Albumentations Resize类")

# 创建一个Resize对象
resize = A.Resize(height=512, width=512)
print("创建成功")

# 打印dir信息
print("Resize类的属性和方法:")
for attr in dir(resize):
    if not attr.startswith("_"):
        print(f"- {attr}")

print("测试完成")
