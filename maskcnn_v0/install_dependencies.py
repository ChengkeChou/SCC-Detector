"""
细胞分割系统依赖安装脚本
自动安装所有必要的依赖项
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def install_package(package, extra_args=None):
    """安装单个包"""
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if extra_args:
        cmd.extend(extra_args)
        
    cmd.append(package)
    
    print(f"安装 {package}...")
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        print(f"警告: 安装 {package} 失败")
        return False

def check_git_installed():
    """检查Git是否已安装"""
    try:
        subprocess.check_call(["git", "--version"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    
def main():
    """主安装函数"""
    print("开始安装细胞分割系统依赖...")
    
    # 安装基础依赖
    basic_packages = [
        "numpy",
        "opencv-python",
        "tqdm",
        "matplotlib",
        "Pillow",
        "PyQt5",
        "scikit-image",
        "scipy",
        "albumentations"
    ]
    
    for package in basic_packages:
        install_package(package, ["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    
    # 安装PyTorch和TorchVision
    if platform.system() == "Windows":
        # 使用CUDA 11.8版本的PyTorch
        install_package("torch", ["--index-url", "https://download.pytorch.org/whl/cu118"])
        install_package("torchvision", ["--index-url", "https://download.pytorch.org/whl/cu118"])
    else:
        install_package("torch")
        install_package("torchvision")
    
    # 安装YOLOv8
    install_package("ultralytics", ["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    
    # 安装CellPose
    try:
        install_package("cellpose", ["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    except:
        print("警告: CellPose安装失败，可能需要手动安装。")
    
    # 检查是否已安装Git
    git_installed = check_git_installed()
    
    # Detectron2安装
    if git_installed:
        try:
            print("检测到Git已安装，尝试安装Detectron2...")
            # 使用pip直接从GitHub安装
            install_package("git+https://github.com/facebookresearch/detectron2.git")
        except:
            print("警告: Detectron2安装失败。")
            print("这可能是由于CUDA版本或PyTorch版本不兼容导致的。")
    else:
        print("\n未检测到Git，无法安装Detectron2。")
        print("如果需要使用Detectron2功能，请先安装Git: https://git-scm.com/download/win")
        print("然后运行: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        print("\n不过，你仍然可以使用系统的其他功能:")
        print("- YOLOv8: 快速准确的目标检测和实例分割")
        print("- CellPose: 专门为细胞分割设计的模型")
    
    # 创建系统启动批处理文件
    create_bat_file()
    
    print("\n安装完成！")
    print("如果某些包安装失败，您可能需要手动安装它们。")
    print("请参阅README.md获取更多信息。")

def create_bat_file():
    """创建便捷的BAT文件"""
    print("创建启动文件...")
    
    # 创建启动系统的BAT文件
    bat_content = '@echo off\n'
    bat_content += 'echo 正在启动细胞分割系统...\n'
    bat_content += f'cd /d "{Path(__file__).parent.absolute()}"\n'
    bat_content += f'"{sys.executable}" run.py\n'
    bat_content += 'if %ERRORLEVEL% NEQ 0 pause\n'
    
    # 保存到上一级目录
    bat_path = Path(__file__).parent.parent / "启动细胞分割系统.bat"
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)
    
    print(f"已创建启动文件: {bat_path}")

if __name__ == "__main__":
    main()
