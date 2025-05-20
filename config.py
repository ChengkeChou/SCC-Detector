# config.py

# 数据集路径
DATASET_PATH = "F:\\鳞癌自动化\\代码测试\\data\\SIPakMED"  # 修改为你的数据集路径
EXTRA_DATASET_PATH = "F:\\鳞癌自动化\\代码测试\\data\\Extra"  # 额外数据集路径

# 分类器数据路径（按器官类型）
CLASSIFIER_DATA = {
    'Cervical': "F:\\鳞癌自动化\\代码测试\\data\\classifier\\Cervical",     # 宫颈鳞癌
    'Oral': "F:\\鳞癌自动化\\代码测试\\data\\classifier\\Oral",           # 口腔鳞癌
    'Urethral': "F:\\鳞癌自动化\\代码测试\\data\\classifier\\Urethral",   # 尿道鳞癌
    'Esophageal': "F:\\鳞癌自动化\\代码测试\\data\\classifier\\Esophageal" # 食管鳞癌
}

# 分割器数据路径（统一）
SEGMENTER_DATA_PATH = "F:\\鳞癌自动化\\代码测试\\data\\segmenter"

# 目标检测器数据路径（新增）
DETECTOR_DATA_PATH = "F:\\鳞癌自动化\\代码测试\\data\\detector"

# 模型保存路径
SEGMENT_MODEL_SAVE_PATH = "F:\\鳞癌自动化\\代码测试\\model\\segment_model.pth"
CLASSIFY_MODEL_SAVE_PATH = "F:\\鳞癌自动化\\代码测试\\model\\classify_model.pth"

# 目标检测模型路径（新增）
YOLO_MODEL_PATH = "F:\\鳞癌自动化\\代码测试\\model\\yolo_model.pt"
FRCNN_MODEL_PATH = "F:\\鳞癌自动化\\代码测试\\model\\frcnn_model.pth" 

# 模型保存路径（按器官类型保存分类器模型）
CLASSIFY_MODEL_PATHS = {
    'Cervical': "F:\\鳞癌自动化\\代码测试\\model\\classify_model_cervical.pth",
    'Oral': "F:\\鳞癌自动化\\代码测试\\model\\classify_model_oral.pth",
    'Urethral': "F:\\鳞癌自动化\\代码测试\\model\\classify_model_urethral.pth",
    'Esophageal': "F:\\鳞癌自动化\\代码测试\\model\\classify_model_esophageal.pth",
    'combined': "F:\\鳞癌自动化\\代码测试\\model\\classify_model_combined.pth"  # 组合模型
}

# 模型配置
MODEL_CONFIG = {
    'use_combined_model': True,  # 是否使用组合训练的模型
    'default_organ': 'Cervical',  # 当组合模型不可用时的默认器官类型
    'detection_method': 'yolo',  # 'yolo', 'frcnn', 'segmentation'
    'confidence_threshold': 0.5,  # 目标检测置信度阈值
    'iou_threshold': 0.45  # 目标检测NMS阈值
}

# 训练参数
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 15

# 类别列表
CLASSES = ["Koilocytotic", "Dyskeratotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

# 图像变换（根据需要进行修改）
import torchvision.transforms as transforms
import albumentations as A

# 定义图像大小
IMAGE_SIZE = 256

# 图像变换
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 使用相同的常量
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 分类器数据增强策略
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 分割器数据增强策略
SEGMENTER_TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
])

# 目标检测数据增强策略（新增）
DETECTOR_TRANSFORM = A.Compose([
    A.Resize(640, 640),  # YOLOv8默认输入尺寸
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RGBShift(p=0.2),
    A.GaussNoise(p=0.1),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# HPV感染评估标准和阈值
HPV_CRITERIA = {
    "Koilocytotic": 0.3,     # 空泡细胞
    "Dyskeratotic": 0.3,     # 角化不良细胞
    "Metaplastic": 0.2,      # 化生细胞
    "Parabasal": 0.1,        # 基底细胞
    "Superficial-Intermediate": 0.1  # 表层-中层细胞
}

HPV_THRESHOLD = 0.5  # HPV感染判定阈值

# 可视化配置
VISUALIZATION_CONFIGS = {
    'colors': {
        "Koilocytotic": (255, 0, 0),      # 红色
        "Dyskeratotic": (0, 255, 0),      # 绿色
        "Metaplastic": (0, 0, 255),       # 蓝色
        "Parabasal": (255, 255, 0),       # 黄色
        "Superficial-Intermediate": (255, 0, 255)  # 紫色
    },
    'line_thickness': 2,
    'font_scale': 0.5,
    'text_thickness': 1
}

# 分割器参数配置
SEGMENTATION_CONFIG = {
    'min_cell_size': 100,     # 最小细胞面积
    'max_cell_size': 5000,    # 最大细胞面积
    'kernel_size': 3,         # 形态学操作核大小
    'distance_threshold': 0.3  # 距离变换阈值
}

# 目标检测器参数配置（新增）
DETECTION_CONFIG = {
    'conf_threshold': 0.5,    # 置信度阈值
    'nms_threshold': 0.45,    # 非极大值抑制阈值
    'img_size': 640,          # 图像大小
    'batch_size': 16,         # 批处理大小
    'augment': True           # 是否使用增强
}