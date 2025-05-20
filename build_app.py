"""
打包脚本：将鳞癌细胞分析系统打包成可执行文件
使用方法：python build_app.py
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

def build_executable(debug=False, console=False, onefile=True, icon=None, name=None):
    """
    使用PyInstaller打包应用程序
    
    参数:
        debug: 是否打包调试版本
        console: 是否显示控制台窗口
        onefile: 是否打包为单个文件
        icon: 图标文件路径
        name: 可执行文件名称
    """
    # 当前项目根目录
    project_root = Path(__file__).parent
    
    # 可执行文件名称
    app_name = name or "鳞癌细胞分析系统"
    
    # 打包命令
    cmd = ["pyinstaller"]
    
    # 是否打包为单个文件
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    # 是否显示控制台窗口
    if not console:
        cmd.append("--noconsole")
        
    # 是否使用自定义图标
    if icon and os.path.exists(icon):
        cmd.append(f"--icon={icon}")
    
    # 添加主程序文件
    cmd.append("main.py")
    
    # 设置名称
    cmd.append(f"--name={app_name}")
    
    # 添加数据文件
    data_files = [
        # 添加模型文件
        ("../代码测试/model/*.pth", "model"),
        # 添加配置文件
        ("config.py", "."),
        # 添加依赖的Python文件
        ("*.py", "."),
    ]
    
    for src, dst in data_files:
        cmd.append(f"--add-data={src}{os.pathsep}{dst}")
    
    # 添加hidden imports
    hidden_imports = [
        "torch", "torchvision", "cv2", "numpy", "sklearn",
        "pandas", "matplotlib", "albumentations", "PyQt5"
    ]
    
    for imp in hidden_imports:
        cmd.append(f"--hidden-import={imp}")
    
    # 在调试模式下保留临时文件
    if debug:
        cmd.append("--debug=all")
    
    # 显示将执行的命令
    print("正在执行打包命令：")
    print(" ".join(cmd))
    
    # 执行打包命令
    try:
        subprocess.run(cmd, check=True)
        print(f"\n打包成功！可执行文件保存在：{project_root}/dist/{app_name}.exe")
    except subprocess.CalledProcessError as e:
        print(f"打包失败：{e}")
        sys.exit(1)

def create_installer():
    """
    创建安装程序（需要NSIS）
    此功能可选，需要安装NSIS（Nullsoft Scriptable Install System）
    """
    print("创建安装程序功能尚未实现。如需创建安装程序，请安装NSIS并编写安装脚本。")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="打包鳞癌细胞分析系统")
    parser.add_argument('--debug', action='store_true', help='打包为调试版本')
    parser.add_argument('--console', action='store_true', help='显示控制台窗口')
    parser.add_argument('--onedir', action='store_true', help='打包为文件夹而非单个文件')
    parser.add_argument('--icon', type=str, help='指定应用图标路径')
    parser.add_argument('--name', type=str, help='指定可执行文件名称')
    parser.add_argument('--installer', action='store_true', help='创建安装程序（需要NSIS）')

    args = parser.parse_args()

    # 确保已安装PyInstaller
    try:
        import PyInstaller
        print(f"已检测到PyInstaller版本: {PyInstaller.__version__}")
    except ImportError:
        print("未安装PyInstaller。正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("PyInstaller已安装成功。")

    # 打包可执行文件
    build_executable(
        debug=args.debug,
        console=args.console,
        onefile=not args.onedir,
        icon=args.icon,
        name=args.name
    )

    # 创建安装程序
    if args.installer:
        create_installer()

if __name__ == "__main__":
    main()