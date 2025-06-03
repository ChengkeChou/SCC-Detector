@echo off
chcp 65001 >nul
title SAM2细胞分析系统

echo.
echo ================================
echo   SAM2细胞分析系统 - 启动程序
echo ================================
echo.

echo 正在检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Python，请先安装Python 3.8-3.11
    pause
    exit /b 1
)

echo ✅ Python环境检查通过
echo.

echo 正在启动程序...
python main.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ 程序启动失败，请检查依赖安装
    echo 运行以下命令安装依赖:
    echo pip install -r requirements.txt
    echo.
    pause
)
