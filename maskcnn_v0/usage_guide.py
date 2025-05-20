"""
细胞分割系统使用指南

本脚本提供有关如何正确使用细胞分割系统的指导，尤其是使用CellPose进行分割
"""

import os
import sys
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('usage_guide.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    print("\n" + "="*80)
    print(" "*30 + "细胞分割系统使用指南")
    print("="*80 + "\n")
    
    print("本指南将帮助您正确使用细胞分割系统，尤其是使用CellPose进行细胞分割与分类\n")
    
    # 系统概述
    print("\n## 1. 系统概述\n")
    print("细胞分割系统支持多种模型类型进行细胞分割和分类：")
    print("  - yolov8: YOLOv8模型，适合实例分割任务")
    print("  - cellpose: CellPose模型，专门用于细胞分割，推荐用于细胞图像")
    print("  - dino: DINO检测转换器模型")
    print("  - maskrcnn: Mask R-CNN实例分割模型")
    print("  - hybrid: 混合模型，结合了多种模型的优点\n")
    
    # 推荐的步骤
    print("\n## 2. 推荐的使用步骤\n")
    print("1. 启动系统 (./启动细胞分割系统.bat)")
    print("2. 从下拉菜单中选择'cellpose'作为模型类型")
    print("3. 点击'加载模型'按钮 - CellPose将使用内置的'cyto'模型，不需要额外的模型文件")
    print("4. 点击'打开图像'按钮选择要分析的细胞图像")
    print("5. 点击'运行推理'按钮进行细胞分割分析")
    print("6. 查看结果并根据需要保存\n")
    
    # 可能的问题和解决方案
    print("\n## 3. 可能的问题和解决方案\n")
    print("问题: 无法浏览选择需要的模型")
    print("解决方案: ")
    print("  - 对于CellPose，您可以不选择具体的模型文件，系统将使用内置的cyto模型")
    print("  - 如需使用自定义模型，请在'浏览...'对话框中查找系统根目录下的models文件夹，选择模型文件\n")
    
    print("问题: 无法正确识别出细胞，无法完成目标识别与检测")
    print("解决方案: ")
    print("  - 确保使用'cellpose'作为模型类型，CellPose专为细胞分割而设计")
    print("  - 尝试调整置信度阈值，降低阈值可能会检测到更多细胞，但可能包含误检")
    print("  - 对于不同类型的细胞图像，可能需要调整参数或使用适合的预训练模型\n")
    
    # 测试脚本
    print("\n## 4. 提供的测试和修复脚本\n")
    print("我们提供了几个脚本来帮助测试和修复系统：")
    print("  - test_cellpose_direct.py: 直接使用CellPose测试细胞分割，不依赖UI系统")
    print("    运行方式: python test_cellpose_direct.py [可选:图像路径]")
    print("  - fix_cellpose.py: 提供改进系统中CellPose集成的脚本")
    print("    运行方式: python fix_cellpose.py")
    print("这些脚本可以帮助验证细胞分割功能并提供改进建议\n")
    
    # 建议配置
    print("\n## 5. 推荐配置\n")
    print("最佳体验的系统配置：")
    print("  - Python 3.8 或更高版本")
    print("  - CUDA支持的NVIDIA GPU (适用于更快的处理速度)")
    print("  - 内存: 8GB或更高")
    print("  - 主要依赖包:")
    print("    - cellpose>=0.6.5")
    print("    - torch>=1.7.0")
    print("    - opencv-python>=4.5.0")
    print("    - numpy>=1.19.0")
    print("    - matplotlib>=3.3.0")
    print("    - PyQt5>=5.15.0\n")
    
    # 其他提示
    print("\n## 6. 其他提示\n")
    print("- CellPose可以找到细胞轮廓，但可能不会立即给出正确的细胞类型分类。")
    print("- 如果主要关注细胞的分割，CellPose是最佳选择。")
    print("- 对于具体的细胞类型分类，可以考虑在CellPose分割后添加单独的分类步骤。")
    print("- 定期更新系统以获取最新的改进和bug修复。\n")
    
    print("\n" + "="*80)
    print(" "*30 + "感谢使用细胞分割系统")
    print("="*80 + "\n")
    
    # 保存为文本文件
    try:
        guide_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "细胞分割系统使用指南.txt")
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("细胞分割系统使用指南\n")
            f.write("====================\n\n")
            f.write("本指南将帮助您正确使用细胞分割系统，尤其是使用CellPose进行细胞分割与分类\n\n")
            
            f.write("## 1. 系统概述\n\n")
            f.write("细胞分割系统支持多种模型类型进行细胞分割和分类：\n")
            f.write("  - yolov8: YOLOv8模型，适合实例分割任务\n")
            f.write("  - cellpose: CellPose模型，专门用于细胞分割，推荐用于细胞图像\n")
            f.write("  - dino: DINO检测转换器模型\n")
            f.write("  - maskrcnn: Mask R-CNN实例分割模型\n")
            f.write("  - hybrid: 混合模型，结合了多种模型的优点\n\n")
            
            f.write("## 2. 推荐的使用步骤\n\n")
            f.write("1. 启动系统 (./启动细胞分割系统.bat)\n")
            f.write("2. 从下拉菜单中选择'cellpose'作为模型类型\n")
            f.write("3. 点击'加载模型'按钮 - CellPose将使用内置的'cyto'模型，不需要额外的模型文件\n")
            f.write("4. 点击'打开图像'按钮选择要分析的细胞图像\n")
            f.write("5. 点击'运行推理'按钮进行细胞分割分析\n")
            f.write("6. 查看结果并根据需要保存\n\n")
            
            f.write("## 3. 可能的问题和解决方案\n\n")
            f.write("问题: 无法浏览选择需要的模型\n")
            f.write("解决方案: \n")
            f.write("  - 对于CellPose，您可以不选择具体的模型文件，系统将使用内置的cyto模型\n")
            f.write("  - 如需使用自定义模型，请在'浏览...'对话框中查找系统根目录下的models文件夹，选择模型文件\n\n")
            
            f.write("问题: 无法正确识别出细胞，无法完成目标识别与检测\n")
            f.write("解决方案: \n")
            f.write("  - 确保使用'cellpose'作为模型类型，CellPose专为细胞分割而设计\n")
            f.write("  - 尝试调整置信度阈值，降低阈值可能会检测到更多细胞，但可能包含误检\n")
            f.write("  - 对于不同类型的细胞图像，可能需要调整参数或使用适合的预训练模型\n\n")
            
            f.write("## 4. 提供的测试和修复脚本\n\n")
            f.write("我们提供了几个脚本来帮助测试和修复系统：\n")
            f.write("  - test_cellpose_direct.py: 直接使用CellPose测试细胞分割，不依赖UI系统\n")
            f.write("    运行方式: python test_cellpose_direct.py [可选:图像路径]\n")
            f.write("  - fix_cellpose.py: 提供改进系统中CellPose集成的脚本\n")
            f.write("    运行方式: python fix_cellpose.py\n")
            f.write("这些脚本可以帮助验证细胞分割功能并提供改进建议\n\n")
            
            f.write("## 5. 推荐配置\n\n")
            f.write("最佳体验的系统配置：\n")
            f.write("  - Python 3.8 或更高版本\n")
            f.write("  - CUDA支持的NVIDIA GPU (适用于更快的处理速度)\n")
            f.write("  - 内存: 8GB或更高\n")
            f.write("  - 主要依赖包:\n")
            f.write("    - cellpose>=0.6.5\n")
            f.write("    - torch>=1.7.0\n")
            f.write("    - opencv-python>=4.5.0\n")
            f.write("    - numpy>=1.19.0\n")
            f.write("    - matplotlib>=3.3.0\n")
            f.write("    - PyQt5>=5.15.0\n\n")
            
            f.write("## 6. 其他提示\n\n")
            f.write("- CellPose可以找到细胞轮廓，但可能不会立即给出正确的细胞类型分类。\n")
            f.write("- 如果主要关注细胞的分割，CellPose是最佳选择。\n")
            f.write("- 对于具体的细胞类型分类，可以考虑在CellPose分割后添加单独的分类步骤。\n")
            f.write("- 定期更新系统以获取最新的改进和bug修复。\n")
        
        logger.info(f"使用指南已保存到: {guide_path}")
    except Exception as e:
        logger.error(f"保存使用指南时出错: {e}")

if __name__ == "__main__":
    main()
