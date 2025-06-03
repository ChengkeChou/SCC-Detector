#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2细胞分析UI
使用SAM2进行细胞分割，ResNet进行细胞分类
"""

import os
import sys
import traceback
import json
import ast
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import Optional, List, Dict, Any

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QGroupBox, QMessageBox,
    QSlider, QSpinBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# SAM2导入
SAM2_AVAILABLE = False
SAM2_ERROR_MSG = ""

try:
    print("正在导入SAM2...")
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    print("✅ SAM2导入成功")
except ImportError as e:
    SAM2_ERROR_MSG = f"SAM2导入失败: {e}"
    print(f"❌ {SAM2_ERROR_MSG}")
    
    # 创建虚拟类防止NameError
    class SAM2ImagePredictor:
        def __init__(self, *args, **kwargs):
            pass
        def set_image(self, *args, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            return None, None, None
    
    def build_sam2(*args, **kwargs):
        return None

# 常量定义
IMG_SIZE = (224, 224)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
    (128, 255, 0), (0, 128, 255), (255, 0, 128), (128, 0, 255)
]


class AnalysisWorker(QThread):
    """分析工作线程"""
    progress_updated = pyqtSignal(int, str)
    analysis_finished = pyqtSignal(np.ndarray, int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image, sam2_predictor, classifier_model, classifier_transform, 
                 class_map, params):
        super().__init__()
        self.image = image
        self.sam2_predictor = sam2_predictor
        self.classifier_model = classifier_model
        self.classifier_transform = classifier_transform
        self.class_map = class_map
        self.params = params
    
    def run(self):
        try:
            # SAM2分割
            self.progress_updated.emit(10, "开始SAM2分割...")
            masks = self._run_sam2_segmentation()
            
            if not masks:
                self.progress_updated.emit(100, "未检测到细胞")
                self.analysis_finished.emit(self.image, 0)
                return
            
            # 分类
            self.progress_updated.emit(50, f"检测到{len(masks)}个细胞，开始分类...")
            annotated_image = self._classify_cells(masks)
            
            self.progress_updated.emit(100, f"分析完成，共分类{len(masks)}个细胞")
            self.analysis_finished.emit(annotated_image, len(masks))
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _run_sam2_segmentation(self):
        """SAM2分割"""
        # 转换为RGB格式给SAM2
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.sam2_predictor.set_image(rgb_image)
        
        # 生成点网格
        h, w = rgb_image.shape[:2]
        points_per_side = self.params['points_per_side']
        points_scale = max(h, w) / points_per_side
        
        points = []
        for i in range(points_per_side):
            for j in range(points_per_side):
                x = int((j + 0.5) * points_scale)
                y = int((i + 0.5) * points_scale)
                if x < w and y < h:
                    points.append([x, y])
        
        if not points:
            return []
        
        # 批量预测
        masks_list = []
        batch_size = 5
        
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            
            for point in batch_points:
                try:
                    masks, scores, _ = self.sam2_predictor.predict(
                        point_coords=np.array([point]),
                        point_labels=np.array([1]),
                        multimask_output=True,
                    )
                    if masks is not None and len(masks) > 0 and scores is not None:
                        best_idx = np.argmax(scores)
                        if scores[best_idx] > self.params['iou_thresh']:
                            mask = masks[best_idx]
                            area = np.sum(mask)
                            bbox = self._mask_to_bbox(mask)
                            
                            # 多重过滤条件
                            # 1. 面积过滤
                            if area < self.params['min_area']:
                                continue
                              # 2. 边界框尺寸过滤
                            x, y, w, h = bbox
                            min_bbox_size = self.params.get('min_bbox_size', 30)
                            if w < min_bbox_size or h < min_bbox_size:  # 太小的边界框
                                continue
                            if w > 500 or h > 500:  # 太大的边界框
                                continue
                            
                            # 3. 长宽比过滤（避免过于细长的形状）
                            max_aspect_ratio = self.params.get('max_aspect_ratio', 4.0)
                            aspect_ratio = max(w, h) / min(w, h)
                            if aspect_ratio > max_aspect_ratio:  # 长宽比过大
                                continue
                            
                            # 4. 掩码密度过滤（掩码在边界框中的占比）
                            bbox_area = w * h
                            mask_density = area / bbox_area if bbox_area > 0 else 0
                            min_density = self.params.get('min_density', 0.4)
                            if mask_density < min_density:  # 掩码太稀疏
                                continue
                            
                            masks_list.append({
                                'mask': mask,
                                'bbox': bbox,
                                'area': area,
                                'score': scores[best_idx]
                            })
                except Exception as e:
                    print(f"预测点{point}时出错: {e}")
                    continue
            
            progress = min(50, int((i + batch_size) * 40 / len(points)) + 10)
            self.progress_updated.emit(progress, f"SAM2分割进度: {progress-10}/40")
        
        return masks_list
    
    def _mask_to_bbox(self, mask):
        """掩码转边界框"""
        if not np.any(mask):
            return [0, 0, 0, 0]
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]
        
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            return [0, 0, 0, 0]
            
        rmin, rmax = row_indices[0], row_indices[-1]
        cmin, cmax = col_indices[0], col_indices[-1]
        
        return [cmin, rmin, cmax - cmin + 1, rmax - rmin + 1]
    
    def _classify_cells(self, masks):
        """细胞分类"""
        annotated_image = self.image.copy()
        
        for i, mask_info in enumerate(masks):
            mask = mask_info['mask']
            bbox = mask_info['bbox']
            x, y, w, h = [int(v) for v in bbox]
            
            if w == 0 or h == 0:
                continue
              # 颜色覆盖
            color = COLORS[i % len(COLORS)]
            # 确保掩码是布尔类型
            mask_bool = mask.astype(bool)
            annotated_image[mask_bool] = (annotated_image[mask_bool] * 0.6 + 
                                   np.array(color, dtype=np.uint8) * 0.4)
            
            # 裁剪细胞区域
            cropped = self.image[y:y+h, x:x+w]
            if cropped.size == 0:
                continue
              # 分类
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            try:
                tensor = self.classifier_transform(cropped_rgb).unsqueeze(0)
                # 确保张量在正确的设备上
                tensor = tensor.to(self.classifier_model.parameters().__next__().device)
                
                with torch.no_grad():
                    output = self.classifier_model(tensor)
                    _, predicted = torch.max(output, 1)
                    class_idx = predicted.item()
                    class_name = self.class_map.get(class_idx, "Unknown")
                
                # 绘制标注
                label = f"ID{i+1}:{class_name}"
                cv2.putText(annotated_image, label, 
                           (x, y - 10 if y > 20 else y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            except Exception as e:
                print(f"分类细胞{i+1}时出错: {e}")
                continue
            
            # 更新进度
            if i % 5 == 0:
                progress = 50 + int((i + 1) * 50 / len(masks))
                self.progress_updated.emit(progress, f"分类进度: {i+1}/{len(masks)}")
        
        return annotated_image


class CellAnalysisSAM2App(QWidget):
    def __init__(self):
        super().__init__()
        
        # 状态变量
        self.current_image = None
        self.original_image = None
        self.sam2_predictor = None
        self.classifier_model = None
        self.classifier_transform = None
        self.class_map = {}
          # SAM2参数
        self.sam2_params = {
            'points_per_side': 32,
            'iou_thresh': 0.88,
            'stability_thresh': 0.95,
            'min_area': 500,        # 增加最小面积阈值
            'max_area': 50000,      # 添加最大面积阈值
            'min_bbox_size': 30,    # 最小边界框尺寸
            'max_aspect_ratio': 4.0, # 最大长宽比
            'min_density': 0.4      # 最小掩码密度
        }
          # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('SAM2细胞分析系统')
        self.setGeometry(100, 100, 1500, 1000)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 配置区域
        config_layout = QHBoxLayout()
        
        # SAM2配置
        sam2_group = self._create_sam2_config_group()
        config_layout.addWidget(sam2_group)
        
        # 分类器配置
        classifier_group = self._create_classifier_config_group()
        config_layout.addWidget(classifier_group)
        
        main_layout.addLayout(config_layout)
        
        # 参数控制
        params_group = self._create_params_group()
        main_layout.addWidget(params_group)
        
        # 操作按钮
        buttons_layout = QHBoxLayout()
        
        self.load_image_btn = QPushButton('加载图像')
        self.load_image_btn.clicked.connect(self.load_image)
        buttons_layout.addWidget(self.load_image_btn)
        
        self.analyze_btn = QPushButton('开始分析')
        self.analyze_btn.clicked.connect(self.analyze_cells)
        self.analyze_btn.setEnabled(False)
        buttons_layout.addWidget(self.analyze_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)        # 图像显示
        images_layout = QHBoxLayout()
        
        # 原始图像（较小）
        self.original_label = QLabel('原始图像')
        self.original_label.setMinimumSize(350, 280)
        self.original_label.setMaximumSize(400, 320)
        self.original_label.setStyleSheet("border: 1px solid black; background-color: white;")
        self.original_label.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(self.original_label)
        
        # 分析结果（较大）
        self.result_label = QLabel('分析结果')
        self.result_label.setMinimumSize(700, 500)
        self.result_label.setStyleSheet("border: 1px solid black; background-color: white;")
        self.result_label.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(self.result_label)
        
        main_layout.addLayout(images_layout)
        
        # 状态栏
        self.status_label = QLabel('状态: 就绪')
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
    
    def _create_sam2_config_group(self):
        """创建SAM2配置组"""
        group = QGroupBox("SAM2模型配置")
        layout = QVBoxLayout()
        
        # 配置文件
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("配置文件:"))
        self.config_edit = QLineEdit()
        self.config_edit.setPlaceholderText("选择SAM2配置文件(.yaml)")
        config_btn = QPushButton("浏览")
        config_btn.clicked.connect(self.browse_sam2_config)
        config_layout.addWidget(self.config_edit)
        config_layout.addWidget(config_btn)
        layout.addLayout(config_layout)
        
        # 权重文件
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(QLabel("权重文件:"))
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("选择SAM2权重文件(.pt/.pth)")
        weights_btn = QPushButton("浏览")
        weights_btn.clicked.connect(self.browse_sam2_weights)
        weights_layout.addWidget(self.weights_edit)
        weights_layout.addWidget(weights_btn)
        layout.addLayout(weights_layout)
        
        # 加载按钮
        self.load_sam2_btn = QPushButton("加载SAM2模型")
        self.load_sam2_btn.clicked.connect(self.load_sam2_model)
        layout.addWidget(self.load_sam2_btn)
        
        group.setLayout(layout)
        return group
    
    def _create_classifier_config_group(self):
        """创建分类器配置组"""
        group = QGroupBox("ResNet分类器配置")
        layout = QVBoxLayout()
        
        # 模型文件
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型文件:"))
        self.classifier_edit = QLineEdit()
        self.classifier_edit.setPlaceholderText("选择ResNet模型文件(.pth)")
        classifier_btn = QPushButton("浏览")
        classifier_btn.clicked.connect(self.browse_classifier_model)
        model_layout.addWidget(self.classifier_edit)
        model_layout.addWidget(classifier_btn)
        layout.addLayout(model_layout)
        
        # 类别映射
        map_layout = QHBoxLayout()
        map_layout.addWidget(QLabel("类别映射:"))
        self.class_map_edit = QLineEdit()
        self.class_map_edit.setPlaceholderText("选择类别映射文件(.txt/.json)")
        map_btn = QPushButton("浏览")
        map_btn.clicked.connect(self.browse_class_map)
        map_layout.addWidget(self.class_map_edit)
        map_layout.addWidget(map_btn)
        layout.addLayout(map_layout)
        
        # 加载按钮
        self.load_classifier_btn = QPushButton("加载分类器")
        self.load_classifier_btn.clicked.connect(self.load_classifier_model)
        layout.addWidget(self.load_classifier_btn)
        group.setLayout(layout)
        return group
    
    def _create_params_group(self):
        """创建参数控制组"""
        group = QGroupBox("SAM2参数设置")
        layout = QVBoxLayout()
        
        # 第一行：基本参数
        row1_layout = QHBoxLayout()
        
        # 每边点数
        points_layout = QVBoxLayout()
        points_layout.addWidget(QLabel("每边点数:"))
        self.points_spin = QSpinBox()
        self.points_spin.setRange(8, 64)
        self.points_spin.setValue(32)
        self.points_spin.valueChanged.connect(
            lambda v: self.sam2_params.update({'points_per_side': v}))
        points_layout.addWidget(self.points_spin)
        row1_layout.addLayout(points_layout)
        
        # IoU阈值
        iou_layout = QVBoxLayout()
        iou_layout.addWidget(QLabel("IoU阈值:"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(50, 95)
        self.iou_slider.setValue(88)
        self.iou_label = QLabel("0.88")
        self.iou_slider.valueChanged.connect(self.update_iou_thresh)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_label)
        row1_layout.addLayout(iou_layout)
        
        # 最小面积
        min_area_layout = QVBoxLayout()
        min_area_layout.addWidget(QLabel("最小面积:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(100, 2000)
        self.min_area_spin.setValue(500)
        self.min_area_spin.valueChanged.connect(
            lambda v: self.sam2_params.update({'min_area': v}))
        min_area_layout.addWidget(self.min_area_spin)
        row1_layout.addLayout(min_area_layout)
        
        layout.addLayout(row1_layout)
        
        # 第二行：高级过滤参数
        row2_layout = QHBoxLayout()
        
        # 最小边界框尺寸
        min_bbox_layout = QVBoxLayout()
        min_bbox_layout.addWidget(QLabel("最小框尺寸:"))
        self.min_bbox_spin = QSpinBox()
        self.min_bbox_spin.setRange(10, 100)
        self.min_bbox_spin.setValue(30)
        self.min_bbox_spin.valueChanged.connect(
            lambda v: self.sam2_params.update({'min_bbox_size': v}))
        min_bbox_layout.addWidget(self.min_bbox_spin)
        row2_layout.addLayout(min_bbox_layout)
        
        # 最大长宽比
        aspect_layout = QVBoxLayout()
        aspect_layout.addWidget(QLabel("最大长宽比:"))
        self.aspect_spin = QSpinBox()
        self.aspect_spin.setRange(2, 10)
        self.aspect_spin.setValue(4)
        self.aspect_spin.valueChanged.connect(
            lambda v: self.sam2_params.update({'max_aspect_ratio': float(v)}))
        aspect_layout.addWidget(self.aspect_spin)
        row2_layout.addLayout(aspect_layout)
        
        # 最小密度
        density_layout = QVBoxLayout()
        density_layout.addWidget(QLabel("最小密度:"))
        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setRange(20, 80)
        self.density_slider.setValue(40)
        self.density_label = QLabel("0.40")
        self.density_slider.valueChanged.connect(self.update_density_thresh)
        density_layout.addWidget(self.density_slider)
        density_layout.addWidget(self.density_label)
        row2_layout.addLayout(density_layout)
        
        layout.addLayout(row2_layout)
        group.setLayout(layout)
        return group
    
    def update_iou_thresh(self, value):
        """更新IoU阈值"""
        thresh = value / 100.0
        self.sam2_params['iou_thresh'] = thresh
        self.iou_label.setText(f"{thresh:.2f}")
    
    def update_density_thresh(self, value):
        """更新密度阈值"""
        thresh = value / 100.0
        self.sam2_params['min_density'] = thresh
        self.density_label.setText(f"{thresh:.2f}")
    
    def browse_sam2_config(self):
        """浏览SAM2配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择SAM2配置文件", "", "YAML files (*.yaml *.yml)")
        if file_path:
            self.config_edit.setText(file_path)
    
    def browse_sam2_weights(self):
        """浏览SAM2权重文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择SAM2权重文件", "", "PyTorch files (*.pt *.pth)")
        if file_path:
            self.weights_edit.setText(file_path)
    
    def browse_classifier_model(self):
        """浏览分类器模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择ResNet模型文件", "", "PyTorch files (*.pth *.pt)")
        if file_path:
            self.classifier_edit.setText(file_path)
            # 自动寻找类别映射文件
            model_dir = os.path.dirname(file_path)
            map_path = os.path.join(model_dir, "class_to_idx.txt")
            if os.path.exists(map_path):
                self.class_map_edit.setText(map_path)
    
    def browse_class_map(self):
        """浏览类别映射文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择类别映射文件", "", "Text files (*.txt *.json)")
        if file_path:
            self.class_map_edit.setText(file_path)
    
    def load_sam2_model(self):
        """加载SAM2模型"""
        if not SAM2_AVAILABLE:
            QMessageBox.critical(self, "错误", "SAM2未安装或导入失败")
            return
        
        config_path = self.config_edit.text().strip()
        weights_path = self.weights_edit.text().strip()
        
        if not config_path or not weights_path:
            QMessageBox.warning(self, "警告", "请选择配置文件和权重文件")
            return
        
        try:
            self.status_label.setText("正在加载SAM2模型...")
            QApplication.processEvents()
            
            # 构建SAM2模型
            sam2_model = build_sam2(config_path, weights_path, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            
            self.status_label.setText("SAM2模型加载成功")
            QMessageBox.information(self, "成功", "SAM2模型加载成功")
            self._check_ready_state()
            
        except Exception as e:
            error_msg = f"SAM2模型加载失败: {e}"
            self.status_label.setText(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            traceback.print_exc()

    def load_classifier_model(self):
        """加载分类器模型"""
        model_path = self.classifier_edit.text().strip()
        map_path = self.class_map_edit.text().strip()
        
        if not model_path or not map_path:
            QMessageBox.warning(self, "警告", "请选择模型文件和类别映射文件")
            return
        
        try:
            self.status_label.setText("正在加载分类器...")
            QApplication.processEvents()
            
            # 加载类别映射
            self._load_class_map(map_path)
            
            if not self.class_map:
                QMessageBox.warning(self, "警告", "类别映射为空")
                return
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            
            num_classes = len(self.class_map)
            model = None
            
            # 处理不同的保存格式
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    # 如果保存的是完整模型
                    model = checkpoint['model']
                    if hasattr(model, 'to') and hasattr(model, 'eval'):
                        model.to(self.device)
                        model.eval()
                        self.classifier_model = model
                    else:
                        raise ValueError("保存的模型对象无效")
                else:
                    # 直接是state_dict
                    state_dict = checkpoint
                
                # 如果需要创建模型架构
                if model is None:
                    # 通过检查权重形状来判断ResNet版本
                    fc_weight_shape = None
                    if 'fc.weight' in state_dict:
                        fc_weight_shape = state_dict['fc.weight'].shape
                      # 根据fc层的输入特征数判断ResNet版本
                    if fc_weight_shape is not None:
                        fc_input_features = fc_weight_shape[1]
                        
                        if fc_input_features == 512:
                            # ResNet18 或 ResNet34
                            print("检测到ResNet18/34架构")
                            model = models.resnet18(weights=None)
                        elif fc_input_features == 2048:
                            # ResNet50, ResNet101, ResNet152
                            print("检测到ResNet50/101/152架构")
                            model = models.resnet50(weights=None)
                        else:
                            # 默认尝试ResNet18
                            print(f"未知fc输入特征数: {fc_input_features}, 默认使用ResNet18")
                            model = models.resnet18(weights=None)
                    else:
                        # 如果没有fc层，默认使用ResNet18
                        print("未找到fc层，默认使用ResNet18")
                        model = models.resnet18(weights=None)
                    
                    # 替换最后的分类层
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                      # 加载权重
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        print("权重加载成功 (允许部分不匹配)")
                    except Exception as e:
                        print(f"权重加载失败，尝试严格匹配: {e}")
                        model.load_state_dict(state_dict, strict=True)
                    
                    # 确保模型在正确的设备上
                    model.to(self.device)
                    model.eval()
                    self.classifier_model = model
            
            elif hasattr(checkpoint, 'eval'):
                # 直接是模型对象
                checkpoint.to(self.device)
                checkpoint.eval()
                self.classifier_model = checkpoint
            else:
                raise ValueError(f"不支持的模型格式: {type(checkpoint)}")
            
            # 设置变换
            self.classifier_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            self.status_label.setText("分类器加载成功")
            QMessageBox.information(self, "成功", "分类器加载成功")
            self._check_ready_state()
            
        except Exception as e:
            error_msg = f"分类器加载失败: {e}"
            self.status_label.setText(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            traceback.print_exc()
    
    def _load_class_map(self, path):
        """加载类别映射"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 尝试JSON格式
            try:
                raw_map = json.loads(content)
            except json.JSONDecodeError:
                # 尝试Python字典格式
                raw_map = ast.literal_eval(content)
            
            if not isinstance(raw_map, dict):
                raise ValueError("类别映射必须是字典格式")
            
            # 转换为idx: name格式
            if raw_map:
                first_key = next(iter(raw_map.keys()))
                first_val = next(iter(raw_map.values()))
                
                if isinstance(first_key, str) and isinstance(first_val, int):
                    # name: idx格式，转换为idx: name
                    self.class_map = {v: k for k, v in raw_map.items()}
                else:
                    # 假设是idx: name格式
                    self.class_map = {int(k): str(v) for k, v in raw_map.items()}
            else:
                self.class_map = {}
            
            print(f"加载类别映射: {self.class_map}")
            
        except Exception as e:
            print(f"加载类别映射失败: {e}")
            self.class_map = {}
            raise
    
    def load_image(self):
        """加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", 
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")
        
        if file_path:
            try:
                # 使用OpenCV加载图像
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("无法读取图像文件")
                
                self.current_image = image
                self.original_image = image.copy()
                
                # 显示图像
                self._display_image(image, self.original_label)
                
                self.status_label.setText(f"图像加载成功: {os.path.basename(file_path)}")
                self._check_ready_state()
                
            except Exception as e:
                error_msg = f"图像加载失败: {e}"
                self.status_label.setText(error_msg)
                QMessageBox.critical(self, "错误", error_msg)
    
    def _display_image(self, cv_image, label):
        """显示图像到标签"""
        try:
            # 转换BGR到RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # 缩放以适应标签
            scaled_pixmap = pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"显示图像失败: {e}")
    
    def _check_ready_state(self):
        """检查是否准备就绪"""
        ready = (self.current_image is not None and 
                self.sam2_predictor is not None and 
                self.classifier_model is not None)
        self.analyze_btn.setEnabled(ready)
    
    def analyze_cells(self):
        """分析细胞"""
        if not self._check_analysis_requirements():
            return
        
        # 禁用按钮
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 创建工作线程
        self.worker = AnalysisWorker(
            self.current_image,
            self.sam2_predictor,
            self.classifier_model,
            self.classifier_transform,
            self.class_map,
            self.sam2_params
        )
        
        # 连接信号
        self.worker.progress_updated.connect(self._update_progress)
        self.worker.analysis_finished.connect(self._analysis_finished)
        self.worker.error_occurred.connect(self._analysis_error)
        
        # 启动线程
        self.worker.start()
    
    def _check_analysis_requirements(self):
        """检查分析要求"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return False
        
        if self.sam2_predictor is None:
            QMessageBox.warning(self, "警告", "请先加载SAM2模型")
            return False
        
        if self.classifier_model is None:
            QMessageBox.warning(self, "警告", "请先加载分类器模型")
            return False
        
        return True
    
    def _update_progress(self, value, message):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.status_label.setText(f"状态: {message}")
        QApplication.processEvents()
    
    def _analysis_finished(self, result_image, cell_count):
        """分析完成"""
        self._display_image(result_image, self.result_label)
        self.status_label.setText(f"分析完成! 检测并分类了 {cell_count} 个细胞")
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setValue(100)
    
    def _analysis_error(self, error_message):
        """分析错误"""
        self.status_label.setText(f"分析失败: {error_message}")
        QMessageBox.critical(self, "错误", f"分析过程中发生错误:\n{error_message}")
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setValue(0)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用信息
    app.setApplicationName("SAM2细胞分析系统")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口
    window = CellAnalysisSAM2App()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
