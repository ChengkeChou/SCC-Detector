import glob
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, 
                         QVBoxLayout, QTreeView, QFileSystemModel, QProgressBar, QMessageBox, 
                         QHBoxLayout, QTextEdit, QSplitter, QComboBox, QTabWidget, 
                         QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QFont
from PyQt5.QtCore import QDir, Qt, QSize
from segmenter import UNet
from classifier import CellClassifier
import cv2
import torch
import os
import numpy as np
from datetime import datetime
import pandas as pd
from config import (CLASSIFY_MODEL_PATHS, MODEL_CONFIG, SEGMENT_MODEL_SAVE_PATH, 
                   TRANSFORM, CLASSES, HPV_CRITERIA, HPV_THRESHOLD, 
                   VISUALIZATION_CONFIGS)
from train_segmenter import MaskGenerator

def analyze_image(image_path, segment_model, classify_model):
    """分析图像，进行细胞分割和分类"""
    try:
        # 读取图像
        print(f"正在读取图像: {image_path}")
        img_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        print(f"图像尺寸: {image.shape}")
        
        # 转换颜色空间并进行预处理
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 确保模型在正确的设备上
        device = next(segment_model.parameters()).device
        print(f"使用设备: {device}")
        
        # 使用掩码生成器进行分割
        print("正在进行细胞分割...")
        mask_generator = MaskGenerator()
        segmented = mask_generator.generate_consensus_mask(image_rgb)
        
        # 输出掩码信息
        non_zero = np.count_nonzero(segmented)
        print(f"分割掩码非零像素数: {non_zero}，掩码尺寸: {segmented.shape}")
        
        if non_zero == 0:
            print("警告: 未检测到任何细胞区域，尝试调整分割参数...")
            # 尝试使用更宽松的参数重新生成掩码
            mask_generator = MaskGenerator({
                'hsv_s_threshold': 0.2,  # 降低饱和度阈值
                'hsv_v_threshold': 0.75,  # 降低亮度阈值
                'hed_threshold': 0.45,     # 调整HED阈值
                'min_cell_size': 50,      # 降低最小细胞大小
                'watershed_threshold': 0.6, # 调整分水岭阈值
            })
            segmented = mask_generator.generate_consensus_mask(image_rgb)
            non_zero = np.count_nonzero(segmented)
            print(f"重试后掩码非零像素数: {non_zero}")
            
        # 处理每个检测到的细胞
        marked_image = image_rgb.copy()
        print("正在查找细胞轮廓...")
        contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"检测到 {len(contours)} 个潜在轮廓")
        
        # 创建可视化：分割掩码与原图叠加
        segmented_vis = mask_generator.visualize_segmentation(image_rgb, segmented, alpha=0.4)
        
        # 初始化各类细胞的统计信息
        cell_counts = {cls: 0 for cls in CLASSES}
        total_cells = 0
        
        cells_info = []
        valid_contours = 0
        
        # 保存图像增强掩码，用于调试
        debug_dir = os.path.dirname(image_path)
        debug_path = os.path.join(debug_dir, "debug_mask.png")
        cv2.imwrite(debug_path, segmented)
        print(f"保存调试掩码到: {debug_path}")
        
        for i, contour in enumerate(contours):
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < 50:  # 降低面积阈值，增加细胞检测率
                continue
                
            valid_contours += 1
            total_cells += 1
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            print(f"细胞 #{i+1}: 位置=({x},{y}), 大小=({w},{h}), 面积={area}")
            
            # 提取单个细胞图像
            cell_img = image_rgb[y:y+h, x:x+w]
            
            # 确保细胞足够大，可以被分类
            if w < 8 or h < 8:  # 降低最小尺寸要求
                print(f"细胞 #{i+1} 太小，跳过分类")
                cell_type = "未知"
                score = 0.0
                scores = [0.0] * len(CLASSES)
            else:
                # 分类单个细胞
                try:
                    cell_tensor = TRANSFORM(cell_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = classify_model(cell_tensor)
                        scores = torch.softmax(output, dim=1).squeeze()
                    
                    # 获取分类结果
                    pred_idx = scores.argmax().item()
                    cell_type = CLASSES[pred_idx]
                    score = scores[pred_idx].item()
                    print(f"细胞 #{i+1} 分类结果: {cell_type}, 置信度: {score:.3f}")
                except Exception as ce:
                    print(f"分类细胞 #{i+1} 时出错: {str(ce)}")
                    cell_type = "分类错误"
                    score = 0.0
                    scores = torch.zeros(len(CLASSES))
            
            # 更新细胞计数
            cell_counts[cell_type] += 1
            
            # 获取对应的颜色
            color = VISUALIZATION_CONFIGS['colors'].get(cell_type, (255, 255, 255))
            
            # 存储细胞信息
            cells_info.append({
                'id': i + 1,  # 细胞ID
                'bbox': (x, y, w, h),
                'type': cell_type,
                'score': score,
                'color': color,
                'area': area,
                'scores': {CLASSES[i]: scores[i].item() for i in range(len(CLASSES))}
            })
            
            # 绘制边界框和标签
            cv2.rectangle(marked_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(marked_image, f'#{i+1}: {cell_type}', 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        print(f"检测到 {valid_contours} 个有效细胞轮廓")
        
        # 如果依然没有检测到有效细胞，尝试使用自适应阈值方法
        if valid_contours == 0:
            print("使用备用检测方法尝试查找细胞...")
            # 转为灰度图
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # 应用自适应阈值
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2)
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 重新查找轮廓
            backup_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"备用方法检测到 {len(backup_contours)} 个潜在轮廓")
            
            # 处理这些轮廓
            for i, contour in enumerate(backup_contours):
                area = cv2.contourArea(contour)
                if area < 50:
                    continue
                    
                valid_contours += 1
                total_cells += 1
                
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                # 处理同上...（这里可以添加相同的细胞处理逻辑）
                cell_img = image_rgb[y:y+h, x:x+w]
                
                # 分类逻辑...
                try:
                    if w >= 8 and h >= 8:
                        cell_tensor = TRANSFORM(cell_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = classify_model(cell_tensor)
                            scores = torch.softmax(output, dim=1).squeeze()
                        
                        pred_idx = scores.argmax().item()
                        cell_type = CLASSES[pred_idx]
                        score = scores[pred_idx].item()
                    else:
                        cell_type = "未知"
                        score = 0.0
                        scores = torch.zeros(len(CLASSES))
                except Exception:
                    cell_type = "分类错误"
                    score = 0.0
                    scores = torch.zeros(len(CLASSES))
                
                cell_counts[cell_type] += 1
                color = VISUALIZATION_CONFIGS['colors'].get(cell_type, (255, 255, 255))
                
                cells_info.append({
                    'id': len(cells_info) + 1,
                    'bbox': (x, y, w, h),
                    'type': cell_type,
                    'score': score,
                    'color': color,
                    'area': area,
                    'scores': {CLASSES[i]: scores[i].item() if i < len(scores) else 0.0 for i in range(len(CLASSES))}
                })
                
                cv2.rectangle(marked_image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(marked_image, f'#{len(cells_info)}: {cell_type}', 
                           (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
            
            # 更新分割掩码可视化
            if valid_contours > 0:
                segmented = thresh
                segmented_vis = cv2.addWeighted(image_rgb, 1, 
                                              cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB), 
                                              0.4, 0)
        
        # 计算HPV风险评分
        risk_results = analyze_hpv_risk(cells_info)
        
        # 在图像上标注风险评分
        risk_score = risk_results['infection_score']
        risk_level = risk_results['risk_level']
        
        # 在右上角添加风险评分
        text_color = (255, 0, 0) if risk_results['is_infected'] else (0, 155, 0)
        cv2.putText(marked_image, f'HPV风险: {risk_level}', 
                   (marked_image.shape[1] - 260, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, text_color, 2)
        cv2.putText(marked_image, f'风险评分: {risk_score:.2f}', 
                   (marked_image.shape[1] - 260, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, text_color, 2)
        
        # 添加细胞类型统计
        y_offset = 90
        for i, (cell_type, count) in enumerate(cell_counts.items()):
            if count > 0:
                color = VISUALIZATION_CONFIGS['colors'].get(cell_type, (255, 255, 255))
                percentage = (count / total_cells * 100) if total_cells > 0 else 0
                cv2.putText(marked_image, f'{cell_type}: {count} ({percentage:.1f}%)', 
                           (marked_image.shape[1] - 260, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # 返回结果
        return {
            'original_image': image_rgb,
            'segmented_mask': segmented,
            'segmented_vis': segmented_vis,
            'marked_image': marked_image,
            'cells_info': cells_info,
            'cell_counts': cell_counts,
            'total_cells': total_cells,
            'risk_results': risk_results
        }
        
    except Exception as e:
        print(f"分析图像时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def analyze_hpv_risk(cells_info):
    """
    基于细胞分类结果分析HPV感染风险
    """
    if not cells_info:
        return {
            'scores': {cls: 0 for cls in CLASSES},
            'infection_score': 0,
            'is_infected': False,
            'risk_level': '无风险',
            'cell_type_counts': {cls: 0 for cls in CLASSES},
            'total_cells': 0,
            'cell_type_percentages': {cls: 0 for cls in CLASSES}
        }
    
    # 统计各类型细胞数量
    cell_type_counts = {cls: 0 for cls in CLASSES}
    total_cells = len(cells_info)
    
    # 累计每种类型的可能性得分
    accumulated_scores = {cls: 0 for cls in CLASSES}
    
    for cell in cells_info:
        cell_type = cell['type']
        cell_type_counts[cell_type] += 1
        
        # 累计每种类型的分类得分
        for cls, score in cell['scores'].items():
            accumulated_scores[cls] += score
    
    # 归一化得分
    normalized_scores = {cls: score/total_cells for cls, score in accumulated_scores.items()}
    
    # 计算HPV感染得分 (基于加权平均)
    infection_score = sum(normalized_scores[cls] * HPV_CRITERIA[cls] for cls in CLASSES)
    
    # 判断风险等级
    risk_level = get_risk_level(infection_score)
    
    return {
        'scores': normalized_scores,
        'infection_score': infection_score,
        'is_infected': infection_score > HPV_THRESHOLD,
        'risk_level': risk_level,
        'cell_type_counts': cell_type_counts,
        'total_cells': total_cells,
        'cell_type_percentages': {
            cls: (count / total_cells * 100) if total_cells > 0 else 0 
            for cls, count in cell_type_counts.items()
        }
    }

def get_risk_level(score):
    """
    根据得分确定风险等级
    """
    if score > 0.7:
        return "高风险"
    elif score > 0.5:
        return "中风险"
    elif score > 0.3:
        return "低风险"
    else:
        return "无风险"

def format_analysis_results(results):
    """
    格式化分析结果为显示文本
    """
    # 确保所有需要的键都存在，防止KeyError
    default_results = {
        'risk_level': '未知',
        'infection_score': 0,
        'is_infected': False,
        'cell_type_counts': {cls: 0 for cls in CLASSES},
        'total_cells': 0,
        'cell_type_percentages': {cls: 0 for cls in CLASSES},
        'scores': {cls: 0 for cls in CLASSES}
    }
    
    # 合并提供的结果和默认值
    for key, value in default_results.items():
        if key not in results:
            results[key] = value
    
    text = f"HPV感染风险分析报告\n"
    text += f"{'='*30}\n\n"
    text += f"总体评估:\n"
    text += f"感染风险: {results['risk_level']}\n"
    text += f"风险评分: {results['infection_score']:.2f} (阈值: {HPV_THRESHOLD})\n"
    text += f"判定结果: {'疑似感染' if results['is_infected'] else '未见明显感染'}\n\n"
    
    # 添加细胞统计信息
    text += f"细胞统计信息:\n"
    text += f"{'-'*20}\n"
    text += f"总检出细胞数: {results['total_cells']}\n"
    
    for cls in CLASSES:
        count = results['cell_type_counts'].get(cls, 0)
        percentage = results['cell_type_percentages'].get(cls, 0)
        text += f"{cls}: {count} ({percentage:.1f}%)\n"
    
    text += f"{'-'*20}\n\n"
    
    # 添加各类型细胞对风险评分的贡献
    text += f"各类型细胞对风险评分贡献:\n"
    text += f"{'-'*20}\n"
    for cls in CLASSES:
        score = results['scores'].get(cls, 0)
        weight = HPV_CRITERIA.get(cls, 0)
        contribution = score * weight
        text += f"{cls}:\n"
        text += f"  细胞评分: {score:.3f}\n"
        text += f"  风险权重: {weight:.2f}\n"
        text += f"  贡献分数: {contribution:.3f}\n"
        text += f"{'-'*20}\n"
    
    return text

def save_analysis_report(results, image_path, save_dir=None):
    """
    保存分析报告
    """
    if save_dir is None:
        save_dir = os.path.dirname(image_path)
        
    # 创建保存目录
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存报告文本
    report_path = os.path.join(results_dir, f"{base_name}_report_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        # 确保risk_results存在
        if 'risk_results' not in results:
            results['risk_results'] = {
                'scores': {cls: 0 for cls in CLASSES},
                'infection_score': 0,
                'is_infected': False,
                'risk_level': '无风险',
                'cell_type_counts': {cls: 0 for cls in CLASSES},
                'total_cells': 0,
                'cell_type_percentages': {cls: 0 for cls in CLASSES}
            }
            
        f.write(format_analysis_results(results['risk_results']))
        f.write("\n\n详细细胞信息:\n")
        f.write(f"{'='*30}\n\n")
        
        # 确保cells_info存在
        if 'cells_info' not in results:
            results['cells_info'] = []
            
        for cell in results['cells_info']:
            f.write(f"细胞 #{cell.get('id', '未知')}:\n")
            f.write(f"  类型: {cell.get('type', '未知')}\n")
            f.write(f"  置信度: {cell.get('score', 0):.3f}\n")
            f.write(f"  位置: {cell.get('bbox', (0, 0, 0, 0))}\n")
            f.write(f"  面积: {cell.get('area', 0)}\n")
            f.write(f"{'-'*20}\n")
    
    # 保存图像结果
    images = {}
    for key in ['original_image', 'segmented_vis', 'marked_image']:
        if key in results:
            name = key.split('_')[0]
            images[name] = results[key]
    
    if not images:  # 如果没有图像数据，使用默认空白图像
        blank_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        images = {'blank': blank_image}
    
    image_paths = []
    for name, img in images.items():
        img_path = os.path.join(results_dir, f"{base_name}_{name}_{timestamp}.png")
        try:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            cv2.imwrite(img_path, img_bgr)
            image_paths.append(img_path)
        except Exception as e:
            print(f"保存图像 {name} 时出错: {str(e)}")
    
    # 保存CSV格式的细胞信息
    csv_path = os.path.join(results_dir, f"{base_name}_cells_{timestamp}.csv")
    data = []
    for cell in results.get('cells_info', []):
        try:
            x, y, w, h = cell.get('bbox', (0, 0, 0, 0))
            data.append({
                'ID': cell.get('id', '未知'),
                'Type': cell.get('type', '未知'),
                'Confidence': cell.get('score', 0),
                'X': x,
                'Y': y,
                'Width': w,
                'Height': h,
                'Area': cell.get('area', 0)
            })
        except Exception as e:
            print(f"处理细胞数据时出错: {str(e)}")
    
    if data:
        try:
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
        except Exception as e:
            print(f"保存CSV时出错: {str(e)}")
            csv_path = None
    else:
        csv_path = None
    
    return {
        'report_path': report_path,
        'images_paths': image_paths,
        'csv_path': csv_path
    }

class CellAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        # 初始化模型
        self._init_models()
        # 当前分析结果
        self.current_results = None
        # 当前文件路径
        self.current_file_path = None
        
        self.initUI()

    def _init_models(self):
        """初始化并加载模型"""
        # 使用CUDA如果可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化并加载分割模型
        self.segment_model = UNet(n_channels=3, n_classes=3).to(self.device)
        if os.path.exists(SEGMENT_MODEL_SAVE_PATH):
            try:
                self.segment_model.load_state_dict(
                    torch.load(SEGMENT_MODEL_SAVE_PATH, map_location=self.device)
                )
                print(f"分割模型加载成功: {SEGMENT_MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"加载分割模型出错: {str(e)}")
        else:
            print(f"警告: 分割模型文件不存在: {SEGMENT_MODEL_SAVE_PATH}")
        self.segment_model.eval()

        # 初始化并加载分类模型
        self.classify_model = CellClassifier().to(self.device)
        self.model_paths = CLASSIFY_MODEL_PATHS
        
        # 默认选择组合模型或特定器官模型
        if MODEL_CONFIG['use_combined_model']:
            model_path = CLASSIFY_MODEL_PATHS['combined']
            print(f"使用组合训练模型: {model_path}")
        else:
            model_path = CLASSIFY_MODEL_PATHS[MODEL_CONFIG['default_organ']]
            print(f"使用单器官模型: {model_path}")

        # 加载模型
        if os.path.exists(model_path):
            try:
                self.classify_model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                print(f"分类模型加载成功: {model_path}")
            except Exception as e:
                print(f"加载分类模型出错: {str(e)}")
        else:
            print(f"警告: 分类模型文件不存在: {model_path}")
        
        self.classify_model.eval()
        
        # 当前选择的模型类型
        self.current_model = 'combined' if MODEL_CONFIG['use_combined_model'] else MODEL_CONFIG['default_organ']

    def initUI(self):
        """初始化UI组件"""
        self.setWindowTitle('鳞状上皮癌细胞分析系统')
        self.setGeometry(100, 100, 1400, 900)  # 更大的窗口尺寸
        
        # 创建水平布局作为主布局
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        left_panel = QVBoxLayout()
        
        # 模型选择下拉框
        self.model_selector_label = QLabel("选择模型类型:")
        left_panel.addWidget(self.model_selector_label)
        
        self.model_selector = QComboBox()
        for organ in ['combined', 'Cervical', 'Oral', 'Urethral', 'Esophageal']:
            model_path = self.model_paths[organ]
            if (os.path.exists(model_path)):
                if organ == 'combined':
                    self.model_selector.addItem("组合模型", organ)
                else:
                    self.model_selector.addItem(f"{organ}模型", organ)
        
        # 设置当前模型
        index = self.model_selector.findData(self.current_model)
        if index >= 0:
            self.model_selector.setCurrentIndex(index)
            
        self.model_selector.currentIndexChanged.connect(self.change_model)
        left_panel.addWidget(self.model_selector)
        
        # 添加按钮
        self.button = QPushButton('选择图片分析', self)
        self.button.clicked.connect(self.load_image)
        left_panel.addWidget(self.button)
        
        self.batch_button = QPushButton('批量处理文件夹', self)
        self.batch_button.clicked.connect(self.load_folder)
        left_panel.addWidget(self.batch_button)
        
        self.save_button = QPushButton('保存分析结果', self)
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)  # 初始时禁用
        left_panel.addWidget(self.save_button)
        
        # 添加进度条
        self.progress = QProgressBar(self)
        self.progress.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.progress)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.status_label)
        
        # 添加弹性空间
        left_panel.addStretch(1)
        
        # 添加信息标签
        info_label = QLabel("鳞状上皮癌细胞自动分析系统\n版本 1.0")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: gray;")
        left_panel.addWidget(info_label)
        
        # 将左侧面板添加到主布局
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(200)
        main_layout.addWidget(left_widget)
        
        # 创建分割器，允许调整左右面板大小
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)
        
        # 中间图像区域
        image_widget = QWidget()
        images_layout = QVBoxLayout(image_widget)
        
        # 使用选项卡组织不同的图像视图
        self.image_tabs = QTabWidget()
        
        # 原始图像标签
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)
        self.original_label = QLabel("原始图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(self.original_label)
        self.image_tabs.addTab(self.original_tab, "原始图像")
        
        # 分割结果标签
        self.segment_tab = QWidget()
        segment_layout = QVBoxLayout(self.segment_tab)
        self.segment_label = QLabel("分割结果")
        self.segment_label.setAlignment(Qt.AlignCenter)
        segment_layout.addWidget(self.segment_label)
        self.image_tabs.addTab(self.segment_tab, "分割结果")
        
        # 标记结果标签
        self.marked_tab = QWidget()
        marked_layout = QVBoxLayout(self.marked_tab)
        self.marked_label = QLabel("标记结果")
        self.marked_label.setAlignment(Qt.AlignCenter)
        marked_layout.addWidget(self.marked_label)
        self.image_tabs.addTab(self.marked_tab, "细胞识别结果")
        
        images_layout.addWidget(self.image_tabs)
        
        # 右侧结果面板
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        
        # 分析结果选项卡
        self.result_tabs = QTabWidget()
        
        # 概要信息选项卡
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        self.result_tabs.addTab(self.summary_tab, "分析概要")
        
        # 详细信息选项卡 - 使用表格显示细胞详情
        self.detail_tab = QWidget()
        detail_layout = QVBoxLayout(self.detail_tab)
        
        self.cell_table = QTableWidget(0, 7)
        self.cell_table.setHorizontalHeaderLabels(["ID", "类型", "置信度", "位置X", "位置Y", "大小", "备注"])
        self.cell_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        detail_layout.addWidget(self.cell_table)
        
        self.result_tabs.addTab(self.detail_tab, "细胞详情")
        
        result_layout.addWidget(self.result_tabs)
        
        # 将图像区域和结果面板添加到分割器
        splitter.addWidget(image_widget)
        splitter.addWidget(result_widget)
        splitter.setSizes([700, 500])  # 设置初始大小
        
        self.setLayout(main_layout)

    def change_model(self, index):
        """切换分类模型"""
        selected_model = self.model_selector.currentData()
        model_path = self.model_paths[selected_model]
        
        try:
            # 加载新选择的模型
            self.classify_model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.current_model = selected_model
            self.status_label.setText(f"已加载{self.model_selector.currentText()}")
            print(f"成功加载模型: {model_path}")
            
            # 如果已经有加载的图像，使用新模型重新分析
            if self.current_file_path:
                self.status_label.setText("使用新模型重新分析...")
                self.load_image(self.current_file_path)
        except Exception as e:
            error_msg = f"加载模型时出错:\n{str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            self.status_label.setText("模型加载失败")

    def show_image(self, image, label):
        """显示图像到指定标签"""
        if image is None:
            return
            
        h, w = image.shape[:2]
        bytes_per_line = 3 * w
        
        if len(image.shape) == 2:  # 如果是单通道图像（掩码）
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)  # 归一化浮点图像转uint8
            
        # 确保图像是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # 使用标签的大小进行缩放，保持比例
        label_size = label.size()
        scaled_pixmap = pixmap.scaled(
            label_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignCenter)

    def load_image(self, file_path=None):
        """加载并处理图像"""
        try:
            if not file_path:
                options = QFileDialog.Options()
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "选择图片", "", 
                    "Images (*.png *.jpg *.jpeg *.bmp *.tif);;All Files (*)",
                    options=options
                )
            
            if not file_path:
                return
            
            self.current_file_path = file_path
            self.status_label.setText(f"正在分析: {os.path.basename(file_path)}...")
            self.progress.setValue(0)
            QApplication.processEvents()  # 更新UI
            
            # 调用analyze_image函数获取结果
            self.progress.setValue(20)
            results = analyze_image(file_path, self.segment_model, self.classify_model)
            self.progress.setValue(80)
            
            # 保存当前结果
            self.current_results = results
            
            # 显示图像结果
            self.show_image(results['original_image'], self.original_label)
            self.show_image(results['segmented_vis'], self.segment_label)  
            self.show_image(results['marked_image'], self.marked_label)
            
            # 更新概要信息
            self.update_summary_info(results)
            
            # 更新详细信息表格
            self.update_cell_details_table(results['cells_info'])
            
            # 启用保存按钮
            self.save_button.setEnabled(True)
            
            self.progress.setValue(100)
            self.status_label.setText(f"分析完成: 检测到 {results['total_cells']} 个细胞")
            
        except Exception as e:
            error_msg = f"处理图像时出错:\n{str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", error_msg)
            self.status_label.setText("分析失败")
            self.progress.setValue(0)

    def update_summary_info(self, results):
        """更新概要信息文本框"""
        try:
            if not results or 'risk_results' not in results:
                self.summary_text.setText("无分析结果或结果格式不正确")
                return
                
            report_text = format_analysis_results(results['risk_results'])
            self.summary_text.setText(report_text)
            
            # 设置风险等级对应的颜色
            risk_level = results['risk_results'].get('risk_level', '未知')
            risk_colors = {
                "高风险": "red",
                "中风险": "orange",
                "低风险": "blue",
                "无风险": "green",
                "未知": "gray"
            }
            color = risk_colors.get(risk_level, "black")
            
            # 高亮风险等级
            html = report_text.replace(
                f"感染风险: {risk_level}", 
                f"感染风险: <span style='color:{color};font-weight:bold;'>{risk_level}</span>"
            )
            self.summary_text.setHtml(html)
        except Exception as e:
            print(f"更新概要信息时出错: {str(e)}")
            self.summary_text.setText("更新分析结果时出错")
            import traceback
            traceback.print_exc()

    def update_cell_details_table(self, cells_info):
        """更新细胞详情表格"""
        try:
            # 清空表格
            self.cell_table.setRowCount(0)
            
            if not cells_info:
                return
                
            # 设置行数
            self.cell_table.setRowCount(len(cells_info))
            
            # 填充数据
            for row, cell in enumerate(cells_info):
                try:
                    # ID
                    self.cell_table.setItem(row, 0, QTableWidgetItem(str(cell.get('id', '未知'))))
                    
                    # 类型
                    cell_type = cell.get('type', '未知')
                    type_item = QTableWidgetItem(cell_type)
                    
                    # 设置类型对应的颜色
                    color = cell.get('color', (0, 0, 0))  # 默认黑色
                    type_item.setForeground(QColor(*color))
                    type_item.setFont(QFont("Arial", 10, QFont.Bold))
                    self.cell_table.setItem(row, 1, type_item)
                    
                    # 置信度
                    score = cell.get('score', 0)
                    score_item = QTableWidgetItem(f"{score:.3f}")
                    if score > 0.8:
                        score_item.setBackground(QColor(200, 255, 200))  # 绿色背景表示高置信度
                    elif score < 0.5:
                        score_item.setBackground(QColor(255, 200, 200))  # 红色背景表示低置信度
                    self.cell_table.setItem(row, 2, score_item)
                    
                    # 位置X, Y
                    bbox = cell.get('bbox', (0, 0, 0, 0))
                    if len(bbox) >= 4:  # 确保bbox有足够的元素
                        x, y, w, h = bbox
                        self.cell_table.setItem(row, 3, QTableWidgetItem(str(x)))
                        self.cell_table.setItem(row, 4, QTableWidgetItem(str(y)))
                        
                        # 大小
                        self.cell_table.setItem(row, 5, QTableWidgetItem(str(w*h)))
                    else:
                        self.cell_table.setItem(row, 3, QTableWidgetItem("0"))
                        self.cell_table.setItem(row, 4, QTableWidgetItem("0"))
                        self.cell_table.setItem(row, 5, QTableWidgetItem("0"))
                    
                    # 备注 - 细胞特征
                    area = cell.get('area', 0)
                    if area > 1000:
                        notes = "大型细胞"
                    elif area < 300:
                        notes = "小型细胞"
                    else:
                        notes = "中型细胞"
                        
                    self.cell_table.setItem(row, 6, QTableWidgetItem(notes))
                except Exception as cell_err:
                    print(f"处理细胞 #{row} 信息时出错: {str(cell_err)}")
                    # 填充默认值
                    for col in range(7):
                        self.cell_table.setItem(row, col, QTableWidgetItem("错误"))
            
            # 自动调整行高度
            self.cell_table.resizeRowsToContents()
        except Exception as e:
            print(f"更新细胞表格时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 显示错误提示到表格
            self.cell_table.setRowCount(1)
            self.cell_table.setItem(0, 0, QTableWidgetItem("更新表格时出错"))
            for col in range(1, 7):
                self.cell_table.setItem(0, col, QTableWidgetItem(""))

    def load_folder(self):
        """批量处理文件夹中的图像"""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", options=options)
        
        if not folder_path:
            return
            
        # 创建结果保存目录
        save_dir = os.path.join(folder_path, "results")
        os.makedirs(save_dir, exist_ok=True)
        
        # 查找所有图像文件
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        
        if not image_files:
            QMessageBox.information(self, "提示", "文件夹中没有找到支持的图像文件")
            return
        
        # 设置进度条
        self.progress.setMaximum(len(image_files))
        self.progress.setValue(0)
        
        # 显示批处理对话框
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"将处理 {len(image_files)} 个文件。\n是否继续?")
        msg.setWindowTitle("批量处理确认")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        if msg.exec_() != QMessageBox.Yes:
            return
            
        # 进行批处理
        processed = 0
        failed = 0
        
        for i, img_path in enumerate(image_files):
            try:
                self.status_label.setText(f"处理 {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
                QApplication.processEvents()  # 更新UI
                
                results = analyze_image(img_path, self.segment_model, self.classify_model)
                save_paths = save_analysis_report(results, img_path, save_dir)
                
                processed += 1
                
            except Exception as e:
                print(f"处理文件失败 {img_path}: {str(e)}")
                failed += 1
                
            self.progress.setValue(i + 1)
            QApplication.processEvents()  # 更新UI
        
        # 显示处理结果
        QMessageBox.information(
            self, 
            "批处理完成", 
            f"成功处理 {processed} 个文件\n失败 {failed} 个文件\n结果保存在: {save_dir}"
        )
        
        self.status_label.setText("批处理完成")

    def save_results(self):
        """保存当前分析结果"""
        if not self.current_results or not self.current_file_path:
            QMessageBox.warning(self, "警告", "没有可保存的分析结果")
            return
        
        try:
            # 打开文件夹选择对话框
            options = QFileDialog.Options()
            save_dir = QFileDialog.getExistingDirectory(
                self, "选择保存目录", os.path.dirname(self.current_file_path), options=options
            )
            
            if not save_dir:
                return
                
            # 保存分析结果
            save_paths = save_analysis_report(self.current_results, self.current_file_path, save_dir)
            
            # 显示保存成功消息
            QMessageBox.information(
                self, 
                "保存成功", 
                f"分析报告已保存至:\n{save_paths['report_path']}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存分析结果时出错:\n{str(e)}")
            self.status_label.setText("保存失败")