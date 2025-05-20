import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import cv2
from config import CLASSES, VISUALIZATION_CONFIGS

class CellDetector:
    def __init__(self, model_path=None, confidence=0.5, nms_threshold=0.3, use_cuda=True):
        """
        初始化细胞检测器
        
        Args:
            model_path: 预训练模型路径，如果为None则使用ImageNet预训练权重
            confidence: 置信度阈值
            nms_threshold: 非极大值抑制阈值
            use_cuda: 是否使用CUDA加速
        """
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.classes = CLASSES
        self.colors = [VISUALIZATION_CONFIGS['colors'][cls] for cls in CLASSES]
        
        # 初始化模型
        self.model = self._create_model(len(self.classes) + 1)  # 加1是因为背景类
        
        # 如果指定了模型路径，加载预训练权重
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"加载模型权重: {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
    def _create_model(self, num_classes):
        """创建Faster R-CNN模型"""
        # 使用预训练的Faster R-CNN模型
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        
        # 修改分类器头，以适应我们的类别数量
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        return model
    
    def detect_image(self, image):
        """
        检测图像中的细胞
        
        Args:
            image: PIL.Image对象或numpy数组
            
        Returns:
            processed_image: 处理后的图像（带有边界框和标签）
            detections: 检测结果列表 [{'box': [x1,y1,x2,y2], 'label': class_index, 'score': confidence}]
        """
        # 确保图像是PIL图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        original_image = image.copy()
        width, height = image.size
        
        # 图像预处理
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # 获取检测结果
        boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # 应用置信度阈值过滤
        keep = scores > self.confidence
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        # 创建检测结果列表
        detections = []
        for i, box in enumerate(boxes):
            class_id = labels[i] - 1  # 减1因为模型输出包含背景类(0)
            if class_id < 0 or class_id >= len(self.classes):
                continue  # 跳过背景类或无效类别
                
            detections.append({
                'box': box.tolist(),
                'label': class_id,
                'class_name': self.classes[class_id],
                'score': float(scores[i])
            })
        
        # 在图像上绘制结果
        processed_image = self._draw_detections(original_image, detections)
        
        return processed_image, detections
    
    def _draw_detections(self, image, detections):
        """在图像上绘制检测结果"""
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体，如果不可用则使用默认字体
        try:
            font = ImageFont.truetype("simhei.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        
        for det in detections:
            box = det['box']
            label = det['class_name']
            score = det['score']
            class_id = det['label']
            
            # 获取颜色
            color = self.colors[class_id]
            
            # 绘制边界框
            draw.rectangle(box, outline=color, width=2)
            
            # 绘制标签背景
            text = f"{label}: {score:.2f}"
            text_size = draw.textbbox((0, 0), text, font=font)[2:4]
            
            # 确保标签位置在图像内
            if box[1] > text_size[1]:
                text_origin = [box[0], box[1] - text_size[1]]
            else:
                text_origin = [box[0], box[3]]
                
            # 绘制标签背景和文本
            draw.rectangle([text_origin[0], text_origin[1], 
                           text_origin[0] + text_size[0], text_origin[1] + text_size[1]], 
                           fill=color)
            
            draw.text(text_origin, text, fill=(255, 255, 255), font=font)
        
        return image
    
    def count_cells(self, detections):
        """
        计算各类细胞数量
        
        Args:
            detections: 检测结果列表
            
        Returns:
            dict: 各类细胞数量统计
        """
        counts = {cls: 0 for cls in self.classes}
        
        for det in detections:
            class_name = det['class_name']
            counts[class_name] += 1
            
        return counts