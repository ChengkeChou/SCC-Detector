"""
测试混合模型中的get_transform函数
"""
import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath('.')))

# 导入混合模型中的get_transform函数
from models.hybrid_cell_segmentation import get_transform

# 测试训练和评估时的变换
try:
    print("测试训练变换...")
    train_transform = get_transform(train=True)
    print(f"训练变换成功: {train_transform}")
    
    print("\n测试评估变换...")
    eval_transform = get_transform(train=False)
    print(f"评估变换成功: {eval_transform}")
    
    print("\n测试完成: 变换函数工作正常")
except Exception as e:
    print(f"错误: {e}")
