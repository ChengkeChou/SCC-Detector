# 细胞分割与检测系统

## 1. 项目概览

本项目旨在使用 YOLO (You Only Look Once) 系列模型实现细胞的实例分割。系统包含数据预处理、模型训练、以及一个图形用户界面 (GUI) 用于进行推理和结果可视化。主要针对医学图像中的细胞进行识别和精确轮廓分割。

## 2. 功能特点

*   **基于 YOLO 的实例分割**: 支持 YOLOv8, YOLOv10 等模型进行细胞实例分割。
*   **SIPakMed 数据集支持**: 提供脚本将 SIPakMed 数据集特有的 `.dat` 标注文件和 `.bmp` 图像转换为 YOLO 训练所需的格式。
*   **自动化数据准备**: 脚本化处理图像复制、格式转换以及训练集/验证集的划分。
*   **可配置化模型训练**:
    *   支持选择不同的预训练模型 (如 `yolov8n-seg.pt`, `yolov8m-seg.pt`, `yolov10m.pt` 等)。
    *   可自定义训练周期 (epochs)、图像尺寸 (img_size)、批处理大小 (batch_size)。
    *   集成多种数据增强技术 (旋转、平移、缩放、剪切、翻转、Mosaic、MixUp、Copy-Paste 等) 以提升模型泛化能力。
*   **图形化推理界面**:
    *   使用 PyQt6 构建的用户界面，方便加载图片并进行实时的细胞检测与分割。
    *   自动加载最新训练生成的 `best.pt` 模型，也可指定特定模型路径。
    *   可视化展示原始图像和带有分割结果的图像。
*   **训练过程监控**: 训练结果保存在 `CellSegmentRuns` 目录，包含各种性能指标图表和日志。

## 3. 项目结构

```
SSC/
├── cell_segmentation/                # 核心代码目录
│   ├── convert_dat_to_yolo.py      # DAT 格式转 YOLO 标签脚本
│   ├── prepare_yolo_data.py        # 准备 YOLO 训练数据集脚本
│   ├── run_yolo_training.py        # YOLO 模型训练脚本
│   ├── data/                         # 处理后的数据及YOLO数据集配置
│   │   ├── Dyskeratotic/             # (示例) 类别数据中间目录
│   │   └── processed_yolo_dataset/   # 最终用于训练的YOLO格式数据集
│   │       ├── images/
│   │       │   ├── train/
│   │       │   └── val/
│   │       ├── labels/
│   │       │   ├── train/
│   │       │   └── val/
│   │       └── sipakmed_seg.yaml     # YOLO 数据集配置文件
│   ├── models/                       # (可选) 自定义模型定义
│   └── ui/                           # 用户界面代码
│       └── inference_ui.py           # 推理界面主程序
├── raw_data/                         # 原始数据集存放目录
│   └── SIPakMed/                     # SIPakMed 原始数据
├── CellSegmentRuns/                  # YOLO 训练运行结果和模型权重
│   └── train_exp_augmented/          # (示例) 一次训练的输出
│       ├── weights/
│       │   └── best.pt               # 最佳模型权重
│       └── ...                       # 其他训练结果图表和日志
├── 安装细胞分割系统依赖.bat        # 依赖安装脚本
├── 启动细胞分割系统.bat            # 启动推理UI脚本
├── 细胞分割系统修复说明.md         # 问题修复和说明文档
├── yolov8n-seg.pt                  # (示例) 预训练模型文件
├── yolov8m-seg.pt
├── yolov10m.pt
└── ...                             # 其他模型文件
```

## 4. 环境设置与安装

1.  **Python 环境**: 推荐使用 Python 3.9+。
2.  **安装依赖**:
    *   运行根目录下的 `安装细胞分割系统依赖.bat` 脚本来自动安装所需的核心依赖库。
    *   主要依赖包括: `ultralytics`, `PyQt6`, `opencv-python`, `numpy`, `Pillow`, `torch`, `torchvision`, `torchaudio` (确保包含CUDA支持以使用GPU)。
3.  **CUDA (可选, 推荐用于GPU训练/推理)**: 如果您希望使用 NVIDIA GPU 进行加速，请确保已正确安装 NVIDIA 驱动程序、CUDA Toolkit 和 cuDNN。`ultralytics` 会自动检测并使用可用的 CUDA 环境。

## 5. 数据准备流程

1.  **放置原始数据**:
    *   将 SIPakMed 数据集的原始 `.bmp` 图像和 `.dat` 标注文件按类别存放到 `f:\SSC\raw_data\SIPakMed\<ClassName>\` 目录下。
    *   例如: `f:\SSC\raw_data\SIPakMed\Dyskeratotic\001.bmp`, `f:\SSC\raw_data\SIPakMed\Dyskeratotic\001_cyt01.dat`

2.  **转换数据格式**:
    *   打开命令行/终端，导航到 `f:\SSC\` 目录。
    *   运行转换脚本:
        ```powershell
        python cell_segmentation\convert_dat_to_yolo.py
        ```
    *   此脚本会将 `raw_data` 中的 `.bmp` 转换为 `.png` 并保存到 `cell_segmentation\data\<ClassName>\images\`，同时将 `.dat` 文件转换为 YOLOv8 分割任务所需的 `.txt` 标签格式 (每个目标一行，格式为 `class_index x1 y1 x2 y2 ... xn yn`) 并保存到 `cell_segmentation\data\<ClassName>\labels\`。脚本会优先处理 `_cytXX.dat` 文件。

3.  **准备最终训练集**:
    *   运行数据准备脚本:
        ```powershell
        python cell_segmentation\prepare_yolo_data.py
        ```
    *   此脚本会将 `cell_segmentation\data\<ClassName>\` 下的图像和标签整合、打乱，并按 80:20 的比例分割为训练集和验证集，最终存放到 `cell_segmentation\data\processed_yolo_dataset\` 目录中，同时生成 `sipakmed_seg.yaml` 配置文件。

## 6. 模型训练流程

1.  **配置训练参数 (可选)**:
    *   打开 `f:\SSC\cell_segmentation\run_yolo_training.py` 文件。
    *   您可以修改脚本开头的默认参数，如 `DEFAULT_MODEL_PT` (选择预训练模型), `DEFAULT_EPOCHS`, `DEFAULT_BATCH_SIZE` 以及各种数据增强参数。
    *   也可以通过命令行参数覆盖这些默认值。

2.  **开始训练**:
    *   打开命令行/终端，导航到 `f:\SSC\` 目录。
    *   运行训练脚本:
        ```powershell
        python cell_segmentation\run_yolo_training.py
        ```
    *   若要使用特定参数，例如指定模型和周期数：
        ```powershell
        python cell_segmentation\run_yolo_training.py --model_pt yolov8s-seg.pt --epochs 100
        ```
    *   训练过程中的输出、日志、权重文件和性能图表将保存在 `f:\SSC\CellSegmentRuns\<run_name>\` 目录下。

## 7. 模型推理与可视化

1.  **启动推理界面**:
    *   可以直接运行根目录下的 `启动细胞分割系统.bat` 脚本。
    *   或者，在命令行/终端中导航到 `f:\SSC\` 目录并运行:
        ```powershell
        python cell_segmentation\ui\inference_ui.py
        ```

2.  **使用界面**:
    *   **加载模型**: 界面启动时会自动尝试从 `f:\SSC\CellSegmentRuns\` 目录中查找并加载最新一次训练生成的 `best.pt` 模型。如果找不到或想使用特定模型，可以在 `inference_ui.py` 脚本中修改 `SPECIFIC_MODEL_PATH` 变量。
    *   **加载图片**: 点击 "加载图片" 按钮，选择需要检测的细胞图像 (支持 `.png`, `.jpg`, `.jpeg`, `.bmp` 格式)。
    *   **检测细胞**: 图片加载后，点击 "检测细胞" 按钮。模型将对图像进行推理，并在右侧显示带有分割掩码和边界框的结果。

## 8. 实用工具脚本

*   `安装细胞分割系统依赖.bat`: 快速安装项目所需 Python 库。
*   `启动细胞分割系统.bat`: 便捷启动图形化推理界面。
*   `细胞分割系统修复说明.md`: 包含一些常见问题的诊断和修复建议。

## 9. 已包含的模型文件 (示例)

项目根目录下可能包含以下预训练或已下载的模型权重文件，可用于训练或直接推理：

*   `yolov8n-seg.pt`
*   `yolov8s-seg.pt`
*   `yolov8m-seg.pt`
*   `yolov10m.pt`
*   `yolo11n.pt`
*   `yolo11m-seg.pt`

请确保在 `run_yolo_training.py` 或 `inference_ui.py` 中正确指定您希望使用的模型文件。

## 10. 常见问题与排查

*   **OMP: Error #15 / KMP_DUPLICATE_LIB_OK**:
    *   此问题通常与环境中存在多个 OpenMP 库冲突有关。`run_yolo_training.py` 脚本已尝试通过设置 `KMP_DUPLICATE_LIB_OK=TRUE` 环境变量来解决。
*   **CUDA out of memory**:
    *   显存不足。尝试减小 `run_yolo_training.py` 中的 `batch_size` 或 `img_size`。
    *   `run_yolo_training.py` 脚本已尝试设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 来优化显存分配。
*   **OSError: [WinError 1455] 页面文件太小**:
    *   Windows 虚拟内存不足。请参考 `run_yolo_training.py` 脚本顶部的注释或 `细胞分割系统修复说明.md` 来手动增加 Windows 页面文件大小。
*   **`yolo` 命令未找到**:
    *   确保 `ultralytics` 包已正确安装，并且其命令行工具所在路径已添加到系统 PATH 环境变量中。
*   **模型加载失败 (UI)**:
    *   检查 `inference_ui.py` 中的 `DEFAULT_MODEL_BASE_PATH` 是否正确指向包含训练结果的目录。
    *   确认 `CellSegmentRuns` 目录下存在有效的 `best.pt` 文件。
*   **标签文件格式错误 (corrupt image/label)**:
    *   通常是由于 `convert_dat_to_yolo.py` 生成的 `.txt` 标签文件内容不符合 YOLO 期望的格式。检查转换逻辑，确保每个多边形坐标正确，并且每行末尾只有一个换行符。

更多详细信息和特定错误处理，请参考 `细胞分割系统修复说明.md`。
