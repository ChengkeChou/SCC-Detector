import albumentations as A
import numpy as np
import inspect

# 打印版本信息
print(f"Albumentations 版本: {A.__version__}")

# 创建测试图像
img = np.ones((300, 300, 3), dtype=np.uint8) * 255

# 尝试不同的 Resize 参数
print("\n1. 尝试使用 height 和 width 参数：")
try:
    resize1 = A.Resize(height=512, width=512)
    result1 = resize1(image=img)
    print("成功! 结果尺寸:", result1["image"].shape)
except Exception as e:
    print(f"错误: {e}")

print("\n2. 尝试使用 size 参数：")
try:
    resize2 = A.Resize(size=(512, 512))
    result2 = resize2(image=img)
    print("成功! 结果尺寸:", result2["image"].shape)
except Exception as e:
    print(f"错误: {e}")

print("\n3. 查看 Resize 源码中的参数：")
try:
    resize_init_sig = inspect.signature(A.Resize.__init__)
    print("Resize.__init__ 参数:", list(resize_init_sig.parameters.keys()))
except Exception as e:
    print(f"无法获取签名: {e}")

print("\n4. 查看 Resize 类和实例的属性：")
resize = A.Resize(height=512, width=512)  # 尝试创建实例
for key in dir(resize):
    if not key.startswith("_"):
        try:
            value = getattr(resize, key)
            print(f"- {key}: {value}")
        except Exception as e:
            print(f"- {key}: <无法访问: {e}>")

# 检查 RandomResizedCrop 参数
print("\n5. 检查 RandomResizedCrop 参数：")
try:
    random_crop_sig = inspect.signature(A.RandomResizedCrop.__init__)
    print("RandomResizedCrop.__init__ 参数:", list(random_crop_sig.parameters.keys()))
except Exception as e:
    print(f"无法获取签名: {e}")
