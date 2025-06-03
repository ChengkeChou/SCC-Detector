@echo off
chcp 65001 >nul
title SAM2细胞分析系统 - 快速安装

echo.
echo ========================================
echo   SAM2细胞分析系统 - 快速安装脚本
echo ========================================
echo.

echo 正在检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Python，请先安装Python 3.8-3.11
    pause
    exit /b 1
)

echo.
echo 正在升级pip...
python -m pip install --upgrade pip

echo.
echo 正在安装基础依赖...
pip install numpy opencv-python Pillow PyQt5 matplotlib pyyaml tqdm

echo.
echo 正在安装PyTorch（GPU版本）...
echo 如果您没有NVIDIA GPU，请手动安装CPU版本：
echo pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo ✅ 安装完成！
echo.
echo 现在可以运行程序了：
echo   1. 双击 "启动SAM2分析.bat"
echo   2. 或运行 "python main.py"
echo.
pause
