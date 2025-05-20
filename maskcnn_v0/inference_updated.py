"""
细胞实例分割混合架构推理模块
使用训练好的模型进行推理，支持单张图像和批量处理
支持多种模型类型: YOLOv8, Cellpose, DINO, Mask R-CNN (PyTorch)
"""

import os
import sys
import numpy as np
import torch
import cv2
import json
from pathlib import Path
import time
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
import logging

# 导入混合模型架构
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid_cell_segmentation import HybridCellSegmentationModel

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 检查CUDA是否可用
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {DEVICE}")

class CellSegmentationInference:
    """细胞分割推理类"""
    def __init__(self, model_path, model_type="maskrcnn", num_classes=5, confidence_threshold=0.5, device=None):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.device = device if device is not None else DEVICE
        
        # 加载类别映射
        self.class_names = {
            0: "Dyskeratotic",
            1: "Koilocytotic", 
            2: "Metaplastic",
            3: "Parabasal",
            4: "Superficial-Intermediate"
        }
        
        # 加载模型
        self.load_model()
        
        # 定义颜色映射（为可视化）
        self.colors = self._generate_colors(self.num_classes)
        
    def _generate_colors(self, num_classes):
        """生成用于可视化的随机颜色"""
        random.seed(42)  # 设置随机种子以确保颜色一致性
        colors = []
        for _ in range(num_classes):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            colors.append(color)
        return colors
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            logger.info(f"加载模型类型: {self.model_type}")
            
            if self.model_type == "yolov8":
                # 使用YOLOv8模型
                from models.cell_segmentation import CellSegmentationModel
                self.model = CellSegmentationModel(
                    model_path=self.model_path,
                    model_type="yolov8",
                    device=self.device
                )
                logger.info("已加载YOLOv8模型")
                
            elif self.model_type == "cellpose":
                # 使用Cellpose模型
                from models.cell_segmentation import CellSegmentationModel
                self.model = CellSegmentationModel(
                    model_path=self.model_path,
                    model_type="cellpose",
                    device=self.device
                )
                logger.info("已加载Cellpose模型")
                
            elif self.model_type == "dino":
                # 使用DINO模型
                from models.cell_segmentation import CellSegmentationModel
                self.model = CellSegmentationModel(
                    model_path=self.model_path,
                    model_type="dino",
                    device=self.device
                )
                logger.info("已加载DINO模型")
                
            elif self.model_type == "maskrcnn":
                # 使用Mask R-CNN (PyTorch)模型
                from models.maskrcnn_pytorch import TorchMaskRCNN
                self.model = TorchMaskRCNN(
                    num_classes=self.num_classes,
                    pretrained=False,
                    device=self.device
                )
                # 加载模型权重
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info("已加载PyTorch Mask R-CNN模型")
                
            else:
                # 使用混合模型
                self.model = HybridCellSegmentationModel(self.num_classes)
                
                # 加载模型权重
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            
            # 将模型移至指定设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"成功加载模型: {self.model_path}")
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            raise
    
    def preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, str):
            # 如果输入是路径字符串，读取图像
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray) and image.shape[2] == 3:
            # 如果是BGR格式的NumPy数组，转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        return image
    
    def predict(self, image):
        """执行预测"""
        # 预处理图像
        original_image = self.preprocess_image(image)
        height, width = original_image.shape[:2]
        
        # 根据模型类型不同处理
        if self.model_type in ["yolov8", "cellpose", "dino"]:
            # 对于这些模型，直接调用它们的predict方法
            with torch.no_grad():
                prediction = self.model.predict(original_image, self.confidence_threshold)
        elif self.model_type == "maskrcnn":
            # 转换为PyTorch张量
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(original_image).to(self.device).unsqueeze(0)  # 添加批次维度
            
            # 执行推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
            # 过滤置信度低的检测
            keep = outputs[0]['scores'] > self.confidence_threshold
            
            # 组装预测结果
            prediction = {
                'boxes': outputs[0]['boxes'][keep],
                'labels': outputs[0]['labels'][keep],
                'scores': outputs[0]['scores'][keep],
                'masks': outputs[0]['masks'][keep]
            }
        else:
            # 默认混合模型
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image_tensor = transform(original_image).to(self.device)
            
            # 执行推理
            with torch.no_grad():
                prediction = self.model.predict(image_tensor, self.confidence_threshold)
                
        return original_image, prediction
    
    def visualize_results(self, image, prediction, output_path=None, show=False):
        """可视化预测结果"""
        # 复制原始图像以进行绘制
        vis_image = image.copy()
        
        # 初始化类别计数和检测数
        class_counts = {}
        total_instances = 0
        
        # 根据不同的模型类型处理可视化
        if self.model_type == "maskrcnn":
            # 处理maskrcnn的预测结果
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            masks = prediction['masks'].cpu().numpy()
            
            total_instances = len(boxes)
            
            # 计算分类统计信息
            for label in labels:
                label_item = label.item() if hasattr(label, 'item') else label
                class_name = self.class_names.get(label_item, f"Unknown-{label_item}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # 为每个实例绘制掩码和边界框
            for i in range(total_instances):
                # 获取掩码
                mask = masks[i, 0]  # 取第一个通道
                mask_binary = (mask > 0.5).astype(np.uint8)
                
                # 应用掩码为实例着色
                label = labels[i]
                label_item = label.item() if hasattr(label, 'item') else label
                color = self.colors[label_item % len(self.colors)]  # 确保索引在范围内
                vis_image = self.apply_mask(vis_image, mask_binary, color, alpha=0.5)
                
                # 绘制边界框
                box = boxes[i].astype(np.int32)
                cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # 添加类别标签和置信度
                class_name = self.class_names.get(label_item, f"Unknown-{label_item}")
                score = scores[i].item() if hasattr(scores[i], 'item') else scores[i]
                label_text = f"{class_name}: {score:.2f}"
                cv2.putText(
                    vis_image, label_text, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                
        elif self.model_type in ["yolov8", "dino"]:
            # 处理YOLOv8或DINO的预测结果
            if hasattr(prediction, 'boxes'):
                boxes = prediction.boxes
                total_instances = len(boxes)
                
                # 计算分类统计信息
                for det in boxes:
                    label = int(det.cls)
                    class_name = self.class_names.get(label, f"Unknown-{label}")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # 绘制边界框
                    box = det.xyxy[0].cpu().numpy().astype(np.int32)
                    conf = float(det.conf)
                    color = self.colors[label % len(self.colors)]
                    cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                    
                    # 添加类别标签和置信度
                    label_text = f"{class_name}: {conf:.2f}"
                    cv2.putText(
                        vis_image, label_text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
                
        elif self.model_type == "cellpose":
            # 处理Cellpose的预测结果 (假设它返回masks和labels)
            masks = prediction.get('masks', [])
            labels = prediction.get('labels', [])
            total_instances = len(masks)
            
            # 计算分类统计信息
            for label in labels:
                class_name = self.class_names.get(label, f"Unknown-{label}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
            # 为每个实例绘制掩码
            for i, mask in enumerate(masks):
                if i < len(labels):
                    label = labels[i]
                    color = self.colors[label % len(self.colors)]
                    vis_image = self.apply_mask(vis_image, mask, color, alpha=0.5)
                    
                    # 找到轮廓并绘制边界框
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
                        
                        # 添加类别标签
                        class_name = self.class_names.get(label, f"Unknown-{label}")
                        cv2.putText(
                            vis_image, class_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )
        
        else:
            # 默认处理
            try:
                # 尝试获取预测结果中的各个组件
                boxes = prediction.get('boxes', [])
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu().numpy()
                
                scores = prediction.get('scores', [])
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                
                labels = prediction.get('labels', [])
                if hasattr(labels, 'cpu'):
                    labels = labels.cpu().numpy()
                
                masks = prediction.get('masks', [])
                if hasattr(masks, 'cpu'):
                    masks = masks.cpu().numpy()
                
                total_instances = len(boxes)
                
                # 计算分类统计信息
                for label in labels:
                    label_item = label.item() if hasattr(label, 'item') else label
                    class_name = self.class_names.get(label_item, f"Unknown-{label_item}")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # 为每个实例绘制边界框和掩码
                for i in range(total_instances):
                    if i < len(labels):
                        label = labels[i]
                        label_item = label.item() if hasattr(label, 'item') else label
                        color = self.colors[label_item % len(self.colors)]
                        
                        # 绘制掩码（如果有）
                        if i < len(masks):
                            mask = masks[i, 0] if masks[i].ndim > 2 else masks[i]
                            mask_binary = (mask > 0.5).astype(np.uint8)
                            vis_image = self.apply_mask(vis_image, mask_binary, color, alpha=0.5)
                        
                        # 绘制边界框
                        if i < len(boxes):
                            box = boxes[i].astype(np.int32)
                            cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                            
                            # 添加类别标签和置信度
                            class_name = self.class_names.get(label_item, f"Unknown-{label_item}")
                            
                            if i < len(scores):
                                score = scores[i].item() if hasattr(scores[i], 'item') else scores[i]
                                label_text = f"{class_name}: {score:.2f}"
                            else:
                                label_text = class_name
                                
                            cv2.putText(
                                vis_image, label_text, (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                            )
            except Exception as e:
                logger.error(f"可视化结果时出错: {e}")
                class_counts = {}
        
        # 添加统计信息
        y_pos = 30
        cv2.putText(vis_image, f"总细胞数: {total_instances}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 25
        
        for class_name, count in sorted(class_counts.items()):
            cv2.putText(vis_image, f"{class_name}: {count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 20
        
        # 如果需要保存到文件
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            logger.info(f"结果已保存到: {output_path}")
        
        # 如果需要显示
        if show:
            plt.figure(figsize=(12, 10))
            plt.imshow(vis_image)
            plt.axis('off')
            plt.show()
        
        return vis_image, class_counts
    
    def apply_mask(self, image, mask, color, alpha=0.5):
        """将掩码应用到图像上"""
        for c in range(3):
            image[:, :, c] = np.where(
                mask == 1,
                image[:, :, c] * (1 - alpha) + alpha * color[c],
                image[:, :, c]
            )
        return image
    
    def process_image(self, image_path, output_dir=None, show=False):
        """处理单张图像"""
        try:
            # 预测
            start_time = time.time()
            original_image, prediction = self.predict(image_path)
            inference_time = time.time() - start_time
            
            # 准备输出路径
            if output_dir:
                output_path = os.path.join(
                    output_dir, 
                    f"{Path(image_path).stem}_result.png"
                )
            else:
                output_path = None
            
            # 可视化
            result_image, class_counts = self.visualize_results(
                original_image, prediction, output_path, show
            )
            
            logger.info(f"处理图像: {image_path}")
            logger.info(f"推理时间: {inference_time:.4f}秒")
            logger.info(f"检测到的细胞: {sum(class_counts.values())}")
            for class_name, count in class_counts.items():
                logger.info(f"  - {class_name}: {count}")
            
            return {
                "image_path": image_path,
                "output_path": output_path,
                "inference_time": inference_time,
                "total_cells": sum(class_counts.values()),
                "class_counts": class_counts,
                "prediction": prediction
            }
            
        except Exception as e:
            logger.error(f"处理图像时出错 {image_path}: {e}")
            return None
    
    def process_batch(self, image_dir, output_dir=None, extensions=('.bmp', '.jpg', '.png')):
        """批量处理图像"""
        # 确保输出目录存在
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像路径
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(Path(image_dir).glob(f"*{ext}")))
        
        if not image_paths:
            logger.warning(f"在 {image_dir} 中未找到图像文件")
            return []
        
        logger.info(f"开始处理 {len(image_paths)} 个图像...")
        
        # 批量处理
        results = []
        for path in tqdm(image_paths, desc="处理图像"):
            result = self.process_image(str(path), output_dir)
            if result:
                results.append(result)
        
        # 计算总统计信息
        total_cells = sum(r["total_cells"] for r in results)
        class_totals = {}
        for r in results:
            for class_name, count in r["class_counts"].items():
                class_totals[class_name] = class_totals.get(class_name, 0) + count
        
        avg_time = sum(r["inference_time"] for r in results) / len(results) if results else 0
        
        # 保存批处理报告
        if output_dir:
            report_path = os.path.join(output_dir, "batch_report.json")
            report = {
                "total_images": len(results),
                "total_cells": total_cells,
                "class_totals": class_totals,
                "average_inference_time": avg_time,
                "processed_images": [
                    {
                        "image_path": r["image_path"],
                        "output_path": r["output_path"],
                        "total_cells": r["total_cells"],
                        "class_counts": r["class_counts"]
                    }
                    for r in results
                ]
            }
            
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
            logger.info(f"批处理报告已保存至: {report_path}")
        
        # 打印总结
        logger.info("批处理完成!")
        logger.info(f"处理的图像总数: {len(results)}")
        logger.info(f"检测到的细胞总数: {total_cells}")
        for class_name, count in class_totals.items():
            logger.info(f"  - {class_name}: {count}")
        logger.info(f"平均推理时间: {avg_time:.4f}秒/图像")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="细胞分割推理")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--type", type=str, default="maskrcnn", help="模型类型 (yolov8, cellpose, dino, maskrcnn)")
    parser.add_argument("--image", type=str, help="输入图像路径")
    parser.add_argument("--dir", type=str, help="输入图像目录")
    parser.add_argument("--output", type=str, default="./results", help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--show", action="store_true", help="显示结果")
    parser.add_argument("--classes", type=int, default=5, help="类别数量")
    
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        parser.error("必须提供 --image 或 --dir 参数")
    
    # 创建推理实例
    inference = CellSegmentationInference(
        model_path=args.model,
        model_type=args.type,
        num_classes=args.classes,
        confidence_threshold=args.threshold
    )
    
    # 运行推理
    if args.image:
        # 处理单张图像
        result = inference.process_image(args.image, args.output, args.show)
    else:
        # 批量处理
        results = inference.process_batch(args.dir, args.output)
