# 🔬 Cell Analysis Pro - 细胞分析专业版

一个基于 Cellpose + ResNet 的智能细胞分割与分类系统，提供现代化的图形用户界面。

![Cell Analysis Pro](https://img.shields.io/badge/版本-1.2.0-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-支持-red)
![Cellpose](https://img.shields.io/badge/Cellpose-SAM-orange)

## ✨ 主要特性

- 🧬 **强大的分割能力**: 集成 Cellpose-SAM 模型，高精度细胞分割
- 🤖 **智能分类系统**: 支持多种深度学习模型 (ResNet, EfficientNet)
- ⚙️ **实时参数调节**: 流场阈值、概率阈值等参数可视化调节
- 📊 **即时结果展示**: 实时显示分析结果和分类标签
- 🎯 **一键式操作**: 简化的工作流程，适合非编程背景用户

## 📁 项目结构

```
cellpose+resnet/
├── 📄 cell_analysis_ui.py          # 主GUI应用程序 (新设计)
├── 📄 run_cellpose_segmentation.py # 命令行分割脚本
├── 📄 train_cell_classifier.py     # 分类器训练脚本
├── 📄 README_Cellpose.md          # 本文档
├── 📄 test(1).clinerules          # AI助手配置规则
└── 📁 classifier_resnet18_20250628_161103/  # 预训练模型示例
    ├── args.txt                    # 训练参数配置
    ├── class_to_idx.txt           # 类别映射
    ├── resnet18_best.pth          # 模型权重
    ├── test_classification_report.csv
    ├── test_confusion_matrix.png
    ├── training_history.csv
    └── training_history.png
```

## 🛠️ 安装依赖

### 环境要求
- Python 3.10 或更高版本
- CUDA (可选，用于GPU加速)

### 一键安装
```bash
# 克隆项目
git clone <your-repository-url>
cd cellpose+resnet

# 创建虚拟环境
python -m venv cell_analysis_env
source cell_analysis_env/bin/activate  # Windows: cell_analysis_env\Scripts\activate

# 安装依赖
pip install torch torchvision torchaudio cellpose pyqt6 opencv-python numpy pillow
```

### 详细依赖列表
```bash
pip install -r requirements.txt
```

*requirements.txt 内容:*
```
torch>=2.0.0
torchvision>=0.15.0
cellpose>=3.0.0
PyQt6>=6.5.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
```

## 🚀 快速开始

### 1. 启动应用程序
```bash
cd cellpose+resnet
python cell_analysis_ui.py
```

### 2. 界面概览
![UI界面示意图]
- **🧬 Cellpose 分割模型**: 选择细胞质、细胞核或自定义模型
- **🤖 细胞分类模型**: 配置分类器模型文件路径
- **⚙️ Cellpose 分割参数**: 实时调节分割参数
- **📷 图像显示区域**: 原始图像和分析结果对比显示

### 3. 使用流程
1. **📁 加载模型**: 点击"🚀 加载所有模型"按钮
2. **🖼️ 导入图像**: 点击"📁 加载图像"选择待分析图像
3. **⚙️ 调节参数**: 根据图像特点调整分割参数
4. **🔍 开始分析**: 点击"🔍 开始分析"获得结果

## 📚 详细使用指南

### 模型配置

#### Cellpose 分割模型选择
- **🦠 细胞质分割 (cyto)**: 适用于细胞质明显的图像
- **🧬 细胞核分割 (nuclei)**: 适用于细胞核清晰的图像  
- **📁 自定义模型**: 使用您训练的专用Cellpose模型

#### 分类器模型配置
系统会自动尝试加载最新的分类器模型，您也可以手动指定：
- **模型文件**: `.pth` 格式的PyTorch模型权重
- **参数文件**: `args.txt` 包含训练时的参数配置
- **类别映射**: `class_to_idx.txt` 定义类别名称到索引的映射

### 参数调节指南

#### 🎯 细胞直径
- **自动检测**: 设为 `0` 让Cellpose自动估算
- **手动设置**: 输入期望的细胞直径（像素）
- **建议**: 第一次使用时选择自动检测

#### 🌊 流场阈值 (0.0-2.0)
- **默认值**: 0.4
- **调高**: 减少细胞数量，适用于细胞密集图像
- **调低**: 增加细胞数量，适用于细胞稀疏图像

#### 📊 细胞概率阈值 (-6.0-6.0)
- **默认值**: 0.0
- **调低**: 增加掩码范围，检测更多细胞
- **调高**: 减少背景误检，提高精度

#### 📺 图像通道配置
- **灰度图像**: `[0,0]`
- **RGB彩色图像**: 
  - 细胞质在绿色通道: `[1,0]`
  - 细胞质在红色通道: `[0,0]`
  - 同时有细胞质和细胞核: `[1,2]` (绿色+蓝色)
## 🔧 高级功能

### 训练自定义分类器

如果您需要训练针对特定细胞类型的分类器，可以使用 `train_cell_classifier.py` 脚本：

#### 数据准备
```
training_data/
├── ClassA/           # 类别A的图像
│   ├── image1.png
│   └── image2.jpg
├── ClassB/           # 类别B的图像
│   ├── image3.tif
│   └── ...
└── ClassC/           # 类别C的图像
    └── ...
```

#### 训练命令
```bash
python train_cell_classifier.py \
    --data_path ./training_data \
    --model_name resnet18 \
    --img_size 224 \
    --batch_size 32 \
    --epochs 50
```

### 批量处理

对于大量图像的批量处理，可以使用命令行脚本：

```bash
python run_cellpose_segmentation.py \
    --input_dir ./images \
    --output_dir ./results \
    --model_type cyto \
    --diameter 30
```

## 🤝 贡献指南

欢迎为项目做出贡献！请遵循以下步骤：

1. Fork 项目仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详细信息请查看 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Cellpose](https://github.com/MouseLand/cellpose) - 优秀的细胞分割算法
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI框架
- 所有为开源社区做出贡献的开发者们

---

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-❤️-red.svg">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/GUI-PyQt6-green.svg">
</p>

<p align="center">
  如果这个项目对您有帮助，请给我们一个 ⭐️
</p>

## 🔧 故障排除

### 常见问题及解决方案

#### 🚫 应用程序无法启动
**症状**: 点击启动后无响应或报错
**解决方案**:
```bash
# 检查Python版本
python --version  # 需要 3.10+

# 检查依赖是否正确安装
pip list | grep torch
pip list | grep cellpose
pip list | grep PyQt6

# 重新安装依赖
pip install --upgrade torch torchvision cellpose pyqt6
```

#### ❌ 模型加载失败
**症状**: 点击"加载所有模型"后出现错误提示
**解决方案**:
1. **Cellpose模型问题**:
   ```bash
   # 检查网络连接，首次使用需下载模型
   python -c "from cellpose import models; models.CellposeModel()"
   ```

2. **分类器模型问题**:
   - 确认 `.pth` 文件完整且未损坏
   - 检查 `args.txt` 格式是否正确
   - 验证 `class_to_idx.txt` 内容格式

#### 🖼️ 图像加载错误
**症状**: 选择图像后显示加载失败
**解决方案**:
- 支持格式: PNG, JPG, JPEG, BMP, TIF, TIFF
- 检查图像文件是否损坏
- 确保图像路径中无特殊字符

#### 🐌 分析速度过慢
**症状**: 分析过程耗时很长
**优化方案**:
1. **启用GPU加速**:
   ```bash
   # 检查CUDA是否可用
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **调整图像尺寸**:
   - 过大的图像会显著影响速度
   - 建议将图像缩放到合适尺寸 (如 1024x1024)

3. **优化参数设置**:
   - 适当提高流场阈值减少过度分割
   - 设置合适的细胞直径避免自动检测

#### 🎯 分割结果不理想
**症状**: 细胞分割过多、过少或不准确
**调优指南**:

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 细胞数量过多 | 流场阈值过低 | 提高流场阈值至 0.6-0.8 |
| 细胞数量过少 | 流场阈值过高 | 降低流场阈值至 0.2-0.3 |
| 背景噪点多 | 概率阈值过低 | 提高概率阈值至 1.0-2.0 |
| 细胞边界不准确 | 直径设置不当 | 手动设置合适的细胞直径 |
| 通道配置错误 | 颜色通道选择不当 | 根据图像类型调整通道配置 |

### 🆘 获取帮助

如果以上方案无法解决您的问题，请：

1. **查看控制台输出**: 启动应用时保持终端窗口开启
2. **检查系统要求**: 确认硬件和软件环境满足要求  
3. **更新依赖包**: 使用最新版本的依赖库
4. **联系支持**: 提供详细的错误信息和系统配置

## 📈 性能优化建议

### 硬件配置
- **推荐**: NVIDIA GPU (8GB+ 显存)
- **最低**: Intel i5 处理器 + 16GB 内存
- **存储**: SSD硬盘提升读取速度

### 软件优化
- 使用conda环境管理依赖
- 定期更新PyTorch和Cellpose版本
- 关闭不必要的后台程序

---

