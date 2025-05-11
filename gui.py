from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout, QTreeView, QFileSystemModel, QProgressBar, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage
from PyQt5.QtCore import QDir, Qt
from segmenter import UNet  # 更新导入路径
from classifier import CellClassifier
import cv2
import torch
import os
import numpy as np
from config import CLASSIFY_MODEL_PATHS, MODEL_CONFIG, SEGMENT_MODEL_SAVE_PATH, TRANSFORM, CLASSES, HPV_CRITERIA, HPV_THRESHOLD, VISUALIZATION_CONFIGS
from train_segmenter import MaskGenerator

def analyze_image(image_path, segment_model, classify_model):
    """分析图像，进行细胞分割和分类"""
    try:
        # 读取图像
        img_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换颜色空间并进行预处理
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_image = TRANSFORM(image_rgb).unsqueeze(0)
        
        # 确保模型在正确的设备上
        device = next(segment_model.parameters()).device
        tensor_image = tensor_image.to(device)
        
        # 使用伪掩码生成器进行分割
        mask_generator = MaskGenerator()
        segmented = mask_generator.generate_consensus_mask(image_rgb)
        
        # 处理每个检测到的细胞
        marked_image = image_rgb.copy()
        contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cells_info = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < 100:  # 过滤太小的区域
                continue
                
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 提取单个细胞图像
            cell_img = image_rgb[y:y+h, x:x+w]
            
            # 分类单个细胞
            cell_tensor = TRANSFORM(cell_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = classify_model(cell_tensor)
                scores = torch.softmax(output, dim=1).squeeze()
            
            # 获取分类结果
            cell_type = CLASSES[scores.argmax().item()]
            score = scores.max().item()
            
            # 获取对应的颜色
            color = VISUALIZATION_CONFIGS['colors'].get(cell_type, (255, 255, 255))
            
            # 存储细胞信息
            cells_info.append({
                'bbox': (x, y, w, h),
                'type': cell_type,
                'score': score,
                'color': color
            })
            
            # 绘制边界框和标签
            cv2.rectangle(marked_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(marked_image, f'{cell_type}: {score:.2f}', 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        return segmented, marked_image, cells_info
        
    except Exception as e:
        print(f"分析图像时出错: {str(e)}")
        raise

def is_hpv_infected(scores):
    """
    判断是否为HPV感染
    :param scores: 分类得分
    :return: 是否为HPV感染及感染分数
    """
    scores_dict = {cls: scores[i].item() for i, cls in enumerate(CLASSES)}
    infection_score = sum(scores_dict[cls] * HPV_CRITERIA[cls] for cls in CLASSES)
    return infection_score > HPV_THRESHOLD, infection_score

def save_results(image_path, segmented, scores, save_dir):
    """
    保存分析结果
    :param image_path: 图像路径
    :param segmented: 分割结果
    :param scores: 分类得分
    :param save_dir: 保存目录
    """
    base_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"result_{base_name}")
    image = cv2.imread(image_path)
    infected, infection_score = is_hpv_infected(scores)
    cell_type_scores = ", ".join([f"{cls}: {scores[i]:.2f}" for i, cls in enumerate(CLASSES) if cls != "Superficial-Intermediate"])

    # 绘制分割结果和分类得分
    painter = QPainter(QPixmap.fromImage(image))
    painter.setPen(QPen(QColor(255, 0, 0), 2))
    painter.drawText(10, 30, f'HPV感染: {"是" if infected else "否"}')
    painter.drawText(10, 60, f'感染分数: {infection_score:.2f}')
    painter.drawText(10, 90, f'细胞类型得分: {cell_type_scores}')
    painter.end()

    # 保存结果图像
    cv2.imwrite(save_path, segmented)

def analyze_results(scores):
    """
    分析细胞分类结果并计算HPV风险评分
    :param scores: 分类得分
    :return: 分析结果字典
    """
    # 转换分数为字典
    scores_dict = {cls: scores[i].item() for i, cls in enumerate(CLASSES)}
    
    # 计算HPV感染得分
    infection_score = sum(scores_dict[cls] * HPV_CRITERIA[cls] for cls in CLASSES)
    
    # 生成详细结果
    results = {
        'scores': scores_dict,  # 各类型得分
        'infection_score': infection_score,  # HPV感染总分
        'is_infected': infection_score > HPV_THRESHOLD,  # 是否感染
        'risk_level': get_risk_level(infection_score),  # 风险等级
        'details': {  # 详细贡献分数
            cls: {
                'score': scores_dict[cls],
                'weight': HPV_CRITERIA[cls],
                'contribution': scores_dict[cls] * HPV_CRITERIA[cls]
            } for cls in CLASSES
        }
    }
    
    return results

def get_risk_level(score):
    """
    根据得分确定风险等级
    """
    if score > 0.7:
        return "高风险"
    elif score > 0.5:
        return "中风险"
    else:
        return "低风险"

def format_analysis_results(results):
    """
    格式化分析结果为显示文本
    """
    text = f"HPV感染风险分析报告\n"
    text += f"{'='*30}\n\n"
    text += f"总体评估:\n"
    text += f"感染风险: {results['risk_level']}\n"
    text += f"感染得分: {results['infection_score']:.2f}\n"
    text += f"判定结果: {'疑似感染' if results['is_infected'] else '未见明显感染'}\n\n"
    
    text += f"细胞类型分布:\n"
    text += f"{'-'*20}\n"
    for cls in CLASSES:
        details = results['details'][cls]
        text += f"{cls}:\n"
        text += f"  检出得分: {details['score']:.3f}\n"
        text += f"  权重系数: {details['weight']:.2f}\n"
        text += f"  贡献分数: {details['contribution']:.3f}\n"
        text += f"{'-'*20}\n"
    
    return text

class CellAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        # 初始化并加载分割模型
        self.segment_model = UNet(n_channels=3, n_classes=1)
        if os.path.exists(SEGMENT_MODEL_SAVE_PATH):
            self.segment_model.load_state_dict(torch.load(SEGMENT_MODEL_SAVE_PATH))
        self.segment_model.eval()

        # 初始化并加载分类模型
        self.classify_model = CellClassifier()
        if MODEL_CONFIG['use_combined_model']:
            model_path = CLASSIFY_MODEL_PATHS['combined']
            print(f"使用组合训练模型: {model_path}")
        else:
            model_path = CLASSIFY_MODEL_PATHS[MODEL_CONFIG['default_organ']]
            print(f"使用单器官模型: {model_path}")

        if os.path.exists(model_path):
            self.classify_model.load_state_dict(torch.load(model_path))
            print("模型加载成功")
        else:
            print(f"警告: 模型文件不存在 {model_path}")
        self.classify_model.eval()
        
        self.initUI()

    def initUI(self):
        """初始化UI组件"""
        self.setWindowTitle('宫颈癌细胞分析系统')
        self.setGeometry(100, 100, 1200, 800)  # 增加窗口大小
        
        # 创建水平布局来并排显示图像
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QVBoxLayout()
        
        self.button = QPushButton('选择图片', self)
        self.button.clicked.connect(self.load_image)
        control_panel.addWidget(self.button)
        
        self.batch_button = QPushButton('选择文件夹', self)
        self.batch_button.clicked.connect(self.load_folder)
        control_panel.addWidget(self.batch_button)
        
        self.progress = QProgressBar(self)
        self.progress.setAlignment(Qt.AlignCenter)
        control_panel.addWidget(self.progress)
        
        # 将控制面板添加到主布局
        control_widget = QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(200)
        main_layout.addWidget(control_widget)
        
        # 创建图像显示区域的垂直布局
        images_layout = QVBoxLayout()
        
        # 原始图像标签
        self.original_label = QLabel("原始图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(self.original_label)
        
        # 分割结果标签
        self.segment_label = QLabel("分割结果")
        self.segment_label.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(self.segment_label)
        
        # 标记结果标签
        self.marked_label = QLabel("标记结果")
        self.marked_label.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(self.marked_label)
        
        # 将图像布局添加到主布局
        images_widget = QWidget()
        images_widget.setLayout(images_layout)
        main_layout.addWidget(images_widget)
        
        self.setLayout(main_layout)

    def show_image(self, image, label):
        """显示图像到指定标签"""
        if image is None:
            return
            
        h, w = image.shape[:2]
        bytes_per_line = 3 * w
        
        if len(image.shape) == 2:  # 如果是单通道图像（掩码）
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:  # 如果是BGR格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # 缩放图像以适应标签大小
        label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def load_image(self, file_path=None):
        """加载并处理图像"""
        try:
            if not file_path:
                options = QFileDialog.Options()
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "选择图片", "", 
                    "Images (*.png *.jpg *.jpeg *.bmp *.tif)",
                    options=options
                )
            
            if not file_path:
                return
                
            # 读取原始图像
            img_array = np.fromfile(file_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图像: {file_path}")
            
            # 显示原始图像
            self.show_image(image, self.original_label)
            
            # 获取分割和分类结果
            segmented, marked_image, cells_info = analyze_image(file_path, self.segment_model, self.classify_model)
            
            # 显示分割结果
            self.show_image(segmented.astype(np.uint8), self.segment_label)
            
            # 显示标记结果
            self.show_image(marked_image, self.marked_label)
            
        except Exception as e:
            error_msg = f"处理图像时出错:\n{str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "错误", error_msg)

    def load_folder(self):
        """
        加载文件夹并批量处理图像
        """
        options = QFileDialog.Options()
        folderPath = QFileDialog.getExistingDirectory(self, "选择文件夹", options=options)
        if folderPath:
            save_dir = os.path.join(folderPath, "results")
            os.makedirs(save_dir, exist_ok=True)
            image_files = [f for f in os.listdir(folderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            self.progress.setMaximum(len(image_files))
            for i, img_name in enumerate(image_files):
                img_path = os.path.join(folderPath, img_name)
                # 使用实例化的模型
                segmented, scores = analyze_image(img_path, self.segment_model, self.classify_model)
                save_results(img_path, segmented, scores, save_dir)
                self.progress.setValue(i + 1)