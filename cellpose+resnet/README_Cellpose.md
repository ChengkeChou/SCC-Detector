# 细胞分割与分类工具

## 概述

本项目提供一个图形用户界面 (GUI)，用于使用 Cellpose (包括 Cellpose-SAM) 对图像中的细胞进行分割，并使用预训练的 PyTorch 模型对这些细胞进行分类。

## 项目结构

```
cell_segmentation/
├── ui/
│   └── cell_analysis_ui.py  # 主要的GUI应用程序
├── models/                  # 存储训练好的分类器模型
│   └── classifier_MODELNAME_TIMESTAMP/
│       ├── MODELNAME_best.pth # 分类器权重文件
│       ├── args.txt           # 分类器的训练参数
│       └── class_to_idx.txt   # 分类器的类别映射文件
├── data/                    # 建议存放您的输入图像的目录
├── run_cellpose_segmentation.py # 用于命令行Cellpose分割的脚本 (可选)
├── train_cell_classifier.py   # 用于训练新细胞分类器的脚本 (可选)
└── README_Cellpose.md       # 本文件
```

## 先决条件

*   Python (例如 3.10+)
*   PyTorch (`torch`, `torchvision`)
*   Cellpose
*   PyQt6
*   OpenCV (`opencv-python`)
*   NumPy
*   Pillow (PIL)
*   Ast (用于安全地解析文本文件中的字典)

建议创建一个虚拟环境并安装依赖项：
```bash
pip install torch torchvision torchaudio cellpose pyqt6 opencv-python numpy pillow
```

## 所需文件和数据

### 1. 输入图像
*   将您的图像文件 (例如 `.png`, `.jpg`, `.tif`) 放置在您选择的目录中。项目中的 `data/` 文件夹是一个建议的位置。

### 2. Cellpose 模型
*   **标准模型:** UI支持 Cellpose 的标准模型，如 'cyto' (细胞质) 和 'nuclei' (细胞核)。当选择 'cyto' 或 'nuclei' 时，应用程序会默认加载 'cyto3' 模型，即 **Cellpose-SAM**，以获得更强的泛化能力。这些模型如果本地不存在，Cellpose 会尝试自动下载。
*   **自定义 Cellpose 模型:** 如果您有自定义训练的 Cellpose 模型 (通常是 `.pth` 文件)，您可以在UI中通过 "Custom Path" 选项指定其路径。

### 3. 分类器模型
*   这些模型通常是通过 `train_cell_classifier.py` 脚本或类似流程训练得到的。
*   建议将每个训练好的分类器存放在 `models/` 目录下的独立子文件夹中，例如 `models/classifier_resnet18_20250521_031546/`。
*   每个分类器子目录**必须**包含以下三个文件：
    1.  **模型权重文件:** 一个 `.pth` 文件，包含了训练好的神经网络权重 (例如 `resnet18_best.pth`)。
    2.  **参数文件 (`args.txt`):** 一个文本文件，记录了训练该模型时使用的参数。UI会读取此文件以获取必要的元数据。关键参数包括：
        *   `model_name`: 使用的模型架构 (例如 `resnet18`, `efficientnet_b0`)。
        *   `img_size`: 模型训练时使用的图像输入尺寸 (例如 `224`)。
        *   `pretrained`: 是否使用了预训练权重 (例如 `True` 或 `False`)。
    3.  **类别映射文件 (`class_to_idx.txt`):** 一个文本文件，其内容是一个Python字典的字符串表示，该字典将类别名称映射到整数索引 (例如 `{'Koilocytotic': 0, 'Metaplastic': 1, 'Parabasal': 2}` )。

### 4. 分类器训练数据 (用于 `train_cell_classifier.py`)
*   如果您希望使用 `train_cell_classifier.py` 脚本训练新的分类器模型，您需要准备相应的训练数据集。
*   **数据结构:** 训练数据应按类别存放在不同的子文件夹中。每个子文件夹的名称即为该类别的名称。
    例如：
    ```
    cell_segmentation/
    └── data/                 # 建议的训练数据根目录
        ├── ClassA/           # 类别A的图像
        │   ├── image1.png
        │   └── image2.jpg
        ├── ClassB/           # 类别B的图像
        │   ├── image3.tif
        │   └── ...
        └── ClassC/           # 类别C的图像
            └── ...
    ```
*   **图像格式:** 支持常见的图像格式，如 `.png`, `.jpg`, `.jpeg`, `.tif`, `.bmp`。
*   **使用方法:** 在运行 `train_cell_classifier.py` 脚本时，您通常需要将包含这些类别子文件夹的父目录 (例如 `cell_segmentation/data/`) 作为训练数据路径传递给脚本。

## 如何使用 `cell_analysis_ui.py` 应用程序

1.  **启动应用程序:**
    打开终端，导航到 `cell_segmentation` 目录，然后运行：
    ```bash
    python ui/cell_analysis_ui.py
    ```

2.  **加载 Cellpose 模型:**
    *   **默认 (Cellpose-SAM):** 在 "Cellpose Segmentation Model" 区域，选择 "Cytoplasm (cyto)" 或 "Nuclei (nuclei)"。应用程序将加载 `cyto3` (Cellpose-SAM) 模型。
    *   **自定义模型:** 选择 "Custom Path:"，然后点击 "Browse" 按钮，选择您的自定义 Cellpose 模型文件。

3.  **加载分类器模型:**
    *   UI会尝试自动填充在 `models/` 目录下找到的最新分类器的路径。
    *   如果需要手动指定，请在 "Cell Classifier Model" 区域：
        *   **Model:** 点击 "Browse PTH" 选择分类器的 `.pth` 权重文件。
        *   **Args:** 点击 "Browse args.txt" 选择对应的 `args.txt` 文件。
        *   **Map:** 点击 "Browse class_to_idx.txt" 选择对应的 `class_to_idx.txt` 文件。
    *   完成路径设置后，点击 "Load All Models" 按钮。状态栏会显示模型加载的进度和结果。

4.  **配置 Cellpose 参数 (可选):**
    在 "Cellpose Parameters" 区域，您可以调整以下参数：
    *   **Diameter (0 for auto):** 输入预期的细胞直径（以像素为单位）。设置为 `0` 将启用 Cellpose 的自动直径估计。
    *   **Flow Threshold:** 控制流场匹配的阈值，影响细胞的合并。默认 `0.4`。
        *   如果分割结果中的细胞数量偏少，可以尝试**提高**此值。
        *   如果分割结果中出现过多不规则形状的细胞，可以尝试**降低**此值。
    *   **Cellprob Threshold:** 控制判定为细胞的概率阈值。默认 `0.0`。
        *   如果分割结果中的细胞数量偏少或细胞掩码过小，可以尝试**降低**此值。
        *   如果分割结果中出现过多掩码（尤其是在背景较暗的区域），可以尝试**提高**此值。
    *   **Channels (Cyto, Nuclei):** 如果您使用的是多通道图像，请指定用于分割的通道索引。Cellpose的通道索引从0开始。
        *   **灰度图像:** 通常使用 `[0,0]` (细胞质通道为0，无细胞核通道)。
        *   **彩色 (RGB) 图像:**
            *   如果细胞质主要在绿色通道，且无特定细胞核通道：`[1,0]` (假设G是第二个通道，索引为1)。
            *   如果细胞质在红色通道，细胞核在蓝色通道：`[0,2]` (假设R是第一个通道索引0, B是第三个通道索引2)。
        *   `[0,0]` 表示对灰度图像进行细胞质分割。`[C,N]` 中，C是细胞质通道，N是细胞核通道。如果某个通道不使用，则设为0。

5.  **加载图像:**
    *   点击 "Load Image" 按钮，然后选择您要分析的图像文件。
    *   原始图像将显示在UI界面的左侧。

6.  **分析细胞:**
    *   当 Cellpose 模型、分类器模型和图像都成功加载后，"Analyze Cells" 按钮将被激活。
    *   点击 "Analyze Cells" 开始分割和分类过程。
    *   处理完成后，带有边界框和类别标签的已处理图像将显示在UI界面的右侧。
    *   状态栏会显示分析过程中的信息和最终结果。

## 故障排除

*   **UI无法显示:**
    *   确保所有依赖项都已正确安装在您的Python环境中。
    *   从终端运行脚本，以便查看可能出现的任何错误消息。
*   **模型加载错误:**
    *   仔细检查模型文件的路径是否正确。
    *   确保分类器所需的 `args.txt` 和 `class_to_idx.txt` 文件存在，并且格式正确。
*   **OpenMP 运行时错误:**
    *   如果您遇到类似 "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized." 的错误，请确保 `cell_analysis_ui.py` 脚本的开头包含以下代码行：
        ```python
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        ```
        (此代码已包含在当前版本的UI脚本中。)

