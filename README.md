# 鳞癌细胞自动化分析系统

这是一个用于鳞状上皮癌细胞自动化分析的系统，通过深度学习进行细胞分割、分类和HPV感染风险评估。

## 目录
- [功能概述](#功能概述)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细使用指南](#详细使用指南)
  - [配置文件修改](#配置文件修改)
  - [模型训练](#模型训练)
  - [使用GUI分析图像](#使用GUI分析图像)
- [常见问题](#常见问题)
- [项目结构](#项目结构)

## 功能概述

- **细胞分割**：采用U-Net深度学习模型，自动识别并分割图像中的细胞
- **细胞分类**：使用EfficientNet模型对细胞进行分类（空泡细胞、角化不良细胞等）
- **HPV感染风险评估**：根据分类结果和预设阈值，评估样本HPV感染风险
- **批量处理**：支持对整个文件夹的图像进行批量分析
- **可视化结果**：标注各类细胞，生成分析报告，并支持训练过程可视化

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (可选，但强烈推荐用于加速)
- Windows 10/11 或 Linux 系统

## 快速开始

1. **安装依赖**

```bash
# 使用pip安装所需库
pip install -r requirements.txt

# 如果遇到albumentations安装错误，单独安装
pip install albumentations
```

2. **启动GUI应用**

```bash
python main.py
```

3. **选择图片或文件夹进行分析**
   - 点击"选择图片"按钮分析单张图像
   - 点击"选择文件夹"按钮批量处理多张图像

## 详细使用指南

### 配置文件修改

在`config.py`文件中可以修改关键配置参数：

```python
# 数据集路径（需要根据实际情况修改）
CLASSIFIER_DATA = {
    'Cervical': "F:\\鳞癌自动化\\代码测试\\data\\classifier\\Cervical",     # 宫颈鳞癌
    'Oral': "F:\\鳞癌自动化\\代码测试\\data\\classifier\\Oral",           # 口腔鳞癌
    'Urethral': "F:\\鳞癌自动化\\代码测试\\data\\classifier\\Urethral",   # 尿道鳞癌
    'Esophageal': "F:\\鳞癌自动化\\代码测试\\data\\classifier\\Esophageal" # 食管鳞癌
}

# 模型保存路径
CLASSIFY_MODEL_PATHS = {
    'Cervical': "F:\\鳞癌自动化\\代码测试\\model\\classify_model_cervical.pth",
    # ...其他器官模型路径...
}

# 训练参数
BATCH_SIZE = 8      # 根据GPU内存调整
LEARNING_RATE = 1e-4
EPOCHS = 50         # 训练轮数
```

### 模型训练

#### 数据集准备

1. 按照以下结构组织数据集：

```
data/
├─ classifier/                # 分类数据集
│  ├─ Cervical/              # 宫颈数据
│  │  ├─ Koilocytotic/       # 空泡细胞图像
│  │  ├─ Dyskeratotic/       # 角化不良细胞图像
│  │  └─ ...                 # 其他类别
│  ├─ Oral/                  # 口腔数据
│  │  ├─ Koilocytotic/
│  │  └─ ...
│  └─ ...                    # 其他器官
└─ segmenter/                # 分割数据集
   ├─ images/                # 原始图像
   ├─ *_nuc.dat              # 细胞核标注数据
   └─ *_cyt.dat              # 细胞质标注数据
```

2. 在`config.py`中设置正确的数据路径

#### 运行训练

**训练分类器**：
```bash
# 训练特定器官的分类器（推荐，每个器官分别训练）
python train_classifier.py --organ Cervical
python train_classifier.py --organ Oral
python train_classifier.py --organ Urethral
python train_classifier.py --organ Esophageal

# 使用早停机制防止过拟合（推荐）
python train_classifier.py --organ Cervical --patience 10 --val_split 0.2

# 训练组合模型（可选）
python train_classifier.py
```

**训练分割器**：
```bash
# 训练通用分割器
python train_segmenter.py

# 或使用organ参数指定特定器官（如果数据集已分开）
python train_segmenter.py --organ Cervical
```

**训练结果可视化**：
- 训练过程会自动生成可视化结果，保存在模型路径的`visualization`目录中
- 例如：`F:\鳞癌自动化\代码测试\model\visualization\classify_model_cervical_training_plot.png`

### 使用GUI分析图像

1. 启动应用程序：
```bash
python main.py
```

2. 在GUI界面中：
   - **选择图片**：分析单个图像文件
   - **选择文件夹**：批量处理整个文件夹的图像
   - **分析结果**：会自动显示在界面上，并保存在图像同目录下

3. 分析结果说明：
   - 不同颜色标记不同类型的细胞
   - 图像右上角显示HPV感染风险评分
   - 图像下方列出各类细胞的数量及比例

## 常见问题

**Q: 出现OpenMP警告信息，如何解决？**  
A: 这是由于多个OpenMP运行时库同时加载导致的。可以通过在代码开头添加以下环境变量解决：
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

**Q: 如何使用GPU加速训练？**  
A: 确保已安装CUDA和cudNN，代码将自动检测并使用GPU。可通过以下代码验证GPU是否可用：
```python
import torch
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

**Q: 训练过程中遇到"albumentations模块未找到"错误？**  
A: 单独安装albumentations库：
```bash
pip install albumentations
```

**Q: 如何避免模型过拟合？**  
A: 使用带早停机制的训练命令：
```bash
python train_classifier.py --organ Cervical --patience 10 --val_split 0.2
```

## 项目结构

- **main.py**: 程序入口，启动GUI
- **gui.py**: 图形用户界面实现
- **config.py**: 配置文件，包含路径和参数设置
- **classifier.py**: 细胞分类模型实现
- **segmenter.py**: 细胞分割模型实现
- **dataset.py**: 数据集加载和预处理
- **train_classifier.py**: 分类器训练脚本
- **train_segmenter.py**: 分割器训练脚本
- **requirements.txt**: 项目依赖库列表

## 打包为独立应用程序

1. 安装打包工具：
```bash
pip install pyinstaller
```

2. 运行打包命令：
```bash
pyinstaller --name=CervicalCancerCellAnalyzer --onefile --windowed main.py
# 或使用已有规格文件
pyinstaller CervicalCancerCellAnalyzer.spec
```

3. 打包后的可执行文件位于`dist`目录中

---