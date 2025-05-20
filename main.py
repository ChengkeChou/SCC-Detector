"""
鳞癌细胞分析系统 - 主程序入口
"""
import sys
import os

# 在导入其他库之前设置环境变量以解决OpenMP冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from PyQt5.QtWidgets import QApplication
from gui import CellAnalyzerApp

def setup_environment():
    """设置运行环境"""
    # 确保正确的工作目录
    if getattr(sys, 'frozen', False):
        # 如果是打包后的可执行文件运行
        application_path = os.path.dirname(sys.executable)
        os.chdir(application_path)
        
        # 添加程序目录到PATH
        os.environ["PATH"] = application_path + os.pathsep + os.environ["PATH"]
    
    # 设置随机种子以保证结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置PyTorch不使用多线程
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """主函数"""
    # 设置环境
    setup_environment()
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，在各平台上保持一致外观
    
    # 设置中文字体
    try:
        from PyQt5.QtGui import QFont
        font = QFont("Microsoft YaHei", 9)  # 使用微软雅黑字体
        app.setFont(font)
    except:
        pass  # 如果找不到字体，使用默认字体
    
    # 创建并显示主窗口
    main_window = CellAnalyzerApp()
    main_window.show()
    
    # 运行应用事件循环
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()