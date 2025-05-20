import os
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from config import CLASSES, VISUALIZATION_CONFIGS

class YOLOCellDetector:
    def __init__(self, model_path=None, confidence=0.5, iou_threshold=0.45, use_cuda=True):
        """
        初始化YOLOv8细胞检测器
        
        Args:
            model_path: YOLOv8模型路径
            confidence: 置信度阈值
            iou_threshold: IoU阈值（用于非极大值抑制）
            use_cuda: 是否使用CUDA加速
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.classes = CLASSES
        self.colors = [VISUALIZATION_CONFIGS['colors'][cls] for cls in CLASSES]
        
        # 加载YOLO模型，使用ultralytics包
        try:
            from ultralytics import YOLO
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"加载YOLOv8模型: {model_path}")
            else:
                # 如果没有指定模型路径，加载预训练的YOLOv8模型
                self.model = YOLO("yolov8n.pt")
                print("加载默认YOLOv8n模型")
                
            self.model_loaded = True
        except ImportError:
            print("警告: 无法导入ultralytics包，请安装: pip install ultralytics")
            self.model_loaded = False
            
    def detect_image(self, image):
        """
        使用YOLOv8检测图像中的细胞
        
        Args:
            image: PIL.Image对象或numpy数组
            
        Returns:
            processed_image: 处理后的图像（带有边界框和标签）
            detections: 检测结果列表 [{'box': [x1,y1,x2,y2], 'label': class_index, 'score': confidence}]
        """
        if not self.model_loaded:
            raise ImportError("YOLOv8模型未正确加载，请安装ultralytics包")
            
        # 确保图像是PIL图像并转换为numpy数组
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            if image_np.shape[-1] == 4:  # 处理RGBA格式
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            original_image = image.copy()
        else:
            image_np = image.copy()
            if len(image_np.shape) == 2:  # 灰度图转RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[-1] == 4:  # RGBA转RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            original_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        # 使用YOLOv8进行预测
        results = self.model(image_np, conf=self.confidence, iou=self.iou_threshold)
        
        # 解析检测结果
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # 获取坐标和类别信息
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # YOLOv8模型返回的类别ID可能需要映射到我们的类别
                if cls_id < len(self.classes):
                    class_name = self.classes[cls_id]
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'label': cls_id,
                        'class_name': class_name,
                        'score': conf
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