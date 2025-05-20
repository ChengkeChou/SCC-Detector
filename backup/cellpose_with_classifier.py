"""
结合CellPose和分类器的细胞检测系统

先使用CellPose进行细胞分割（无需训练，直接使用预训练模型）
然后使用分类器对分割出的细胞进行分类
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json
import logging
from tqdm import tqdm
from cellpose import models

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('cellpose_classifier.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {DEVICE}")

# 定义简单的分类器网络
class CellClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(CellClassifier, self).__init__()
        # 使用ResNet18作为特征提取器
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class CellPoseWithClassifier:
    """结合CellPose和分类器的细胞检测系统"""
    
    def __init__(self, classifier_path=None, num_classes=5, cell_size=30, device=None):
        """
        初始化系统
        
        Args:
            classifier_path: 分类器模型路径
            num_classes: 类别数量
            cell_size: CellPose细胞直径参数
            device: 计算设备
        """
        self.device = device or DEVICE
        self.cell_size = cell_size
        self.num_classes = num_classes
        
        # 类别名称
        self.class_names = {
            0: "Dyskeratotic",
            1: "Koilocytotic", 
            2: "Metaplastic",
            3: "Parabasal",
            4: "Superficial-Intermediate"
        }
        
        # 加载CellPose模型（使用预训练模型，无需自定义模型）
        logger.info("加载CellPose模型...")
        self.cellpose_model = models.CellposeModel(
            gpu=("cuda" in str(self.device)), 
            model_type="cyto",  # 'cyto' 是细胞质分割，也可以用 'nuclei' 进行细胞核分割
            pretrained_model=None  # 使用内置预训练模型
        )
        logger.info("CellPose模型加载完成")
        
        # 加载分类器
        self.classifier = CellClassifier(num_classes=num_classes).to(self.device)
        
        if classifier_path and os.path.exists(classifier_path):
            try:
                logger.info(f"加载分类器: {classifier_path}")
                checkpoint = torch.load(classifier_path, map_location=self.device)
                
                # 处理不同格式的检查点
                if 'model_state_dict' in checkpoint:
                    self.classifier.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.classifier.load_state_dict(checkpoint['state_dict'])
                else:
                    self.classifier.load_state_dict(checkpoint)
                    
                self.classifier.eval()
                logger.info("分类器加载完成")
            except Exception as e:
                logger.error(f"加载分类器失败: {e}")
                logger.warning("将使用随机初始化的分类器（仅用于演示）")
        else:
            logger.warning("未提供分类器模型路径，使用随机初始化的分类器（仅用于演示）")
            self.classifier.eval()
    
    def segment_and_classify(self, image):
        """
        使用CellPose分割细胞并进行分类
        
        Args:
            image: 输入图像(RGB格式)
            
        Returns:
            dict: 包含分割和分类结果的字典
        """
        # 确保图像是RGB格式
        if isinstance(image, str):
            img = cv2.imread(image)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3 and image.dtype == np.uint8:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
        else:
            raise ValueError("image参数必须是图像路径或NumPy数组")
        
        # 1. 使用CellPose分割细胞
        logger.info(f"使用CellPose分割细胞，图像尺寸: {img_rgb.shape}")
        start_time = time.time()
        
        channels = [0, 0]  # CellPose默认使用第一个通道
        masks, flows, styles, diams = self.cellpose_model.eval(
            img_rgb, 
            diameter=self.cell_size,
            channels=channels,
            flow_threshold=0.4,
            cellprob_threshold=0.5,
            do_3D=False
        )
        
        segmentation_time = time.time() - start_time
        logger.info(f"CellPose分割完成，用时: {segmentation_time:.2f}秒")
        
        # 检查是否检测到细胞
        unique_ids = np.unique(masks)[1:]  # 跳过0（背景）
        num_cells = len(unique_ids)
        logger.info(f"检测到 {num_cells} 个细胞")
        
        if num_cells == 0:
            logger.warning("未检测到细胞")
            return {
                "num_cells": 0,
                "boxes": np.zeros((0, 4), dtype=np.int32),
                "masks": [],
                "class_ids": np.zeros(0, dtype=np.int32),
                "scores": np.zeros(0, dtype=np.float32),
                "classes": []
            }
        
        # 2. 准备分类
        boxes = []
        extracted_cells = []
        instance_masks = []
        
        # 3. 提取每个细胞区域
        for cell_id in unique_ids:
            cell_mask = masks == cell_id
            instance_masks.append(cell_mask)
            
            # 计算边界框
            y_indices, x_indices = np.where(cell_mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # 扩展边界框以获取更多上下文
            h, w = img_rgb.shape[:2]
            margin = 5  # 像素边距
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w-1, x_max + margin)
            y_max = min(h-1, y_max + margin)
            
            # 保存边界框
            boxes.append([x_min, y_min, x_max, y_max])
            
            # 提取细胞
            cell_img = img_rgb[y_min:y_max+1, x_min:x_max+1].copy()
            
            # 应用掩码（可选）
            # cell_mask_crop = cell_mask[y_min:y_max+1, x_min:x_max+1]
            # for c in range(3):
            #     cell_img[:,:,c] = cell_img[:,:,c] * cell_mask_crop
            
            # 添加到提取的细胞列表
            extracted_cells.append(cell_img)
        
        # 4. 对提取的细胞区域进行分类
        class_ids = []
        scores = []
        
        if extracted_cells:
            start_time = time.time()
            
            # 分批处理以提高效率
            batch_size = 16
            num_batches = (len(extracted_cells) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(extracted_cells))
                
                batch_cells = extracted_cells[start_idx:end_idx]
                
                # 预处理图像为分类器的输入格式
                processed_cells = []
                for cell in batch_cells:
                    # 调整大小
                    resized = cv2.resize(cell, (224, 224))
                    # 转换为PyTorch张量
                    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
                    # 标准化
                    tensor = tensor.to(self.device)
                    processed_cells.append(tensor)
                
                # 批量分类
                with torch.no_grad():
                    input_batch = torch.stack(processed_cells)
                    outputs = self.classifier(input_batch)
                    
                    # 获取预测类别和分数
                    probs = F.softmax(outputs, dim=1)
                    batch_scores, batch_preds = torch.max(probs, dim=1)
                    
                    class_ids.extend(batch_preds.cpu().numpy())
                    scores.extend(batch_scores.cpu().numpy())
            
            classification_time = time.time() - start_time
            logger.info(f"分类完成，用时: {classification_time:.2f}秒")
        
        # 5. 创建类别名称列表
        classes = [self.class_names.get(class_id, f"未知-{class_id}") for class_id in class_ids]
        
        # 6. 统计每个类别的数量
        class_counts = {}
        for class_name in classes:
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        # 打印统计信息
        logger.info(f"类别统计:")
        for cls, count in class_counts.items():
            logger.info(f"  - {cls}: {count}")
        
        # 7. 返回结果
        return {
            "num_cells": num_cells,
            "boxes": np.array(boxes),
            "masks": instance_masks,
            "class_ids": np.array(class_ids),
            "scores": np.array(scores),
            "classes": classes,
            "class_counts": class_counts
        }
    
    def visualize_results(self, image, results):
        """
        可视化分割和分类结果
        
        Args:
            image: 输入图像
            results: 分割和分类结果
            
        Returns:
            np.ndarray: 可视化图像
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3 and image.dtype == np.uint8:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
        else:
            raise ValueError("image参数必须是图像路径或NumPy数组")
            
        # 创建可视化图像副本
        vis_image = img_rgb.copy()
        
        # 定义类别颜色
        colors = [
            (0, 0, 255),    # 红色
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红色
            (0, 255, 255),  # 黄色
            (128, 0, 0),    # 深蓝色
            (0, 128, 0),    # 深绿色
            (128, 128, 0),  # 橄榄色
            (0, 0, 128),    # 深红色
        ]
        
        # 绘制每个检测到的细胞
        boxes = results.get("boxes", [])
        masks = results.get("masks", [])
        class_ids = results.get("class_ids", [])
        scores = results.get("scores", [])
        classes = results.get("classes", [])
        
        for i in range(len(boxes)):
            # 获取实例信息
            box = boxes[i]
            mask = masks[i] if i < len(masks) else None
            class_id = class_ids[i] if i < len(class_ids) else 0
            score = scores[i] if i < len(scores) else 0
            class_name = classes[i] if i < len(classes) else "未知"
            
            # 选择颜色
            color = colors[class_id % len(colors)]
            
            # 绘制掩码
            if mask is not None:
                mask_overlay = np.zeros_like(vis_image)
                mask_overlay[mask] = color
                vis_image = cv2.addWeighted(vis_image, 1, mask_overlay, 0.5, 0)
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name} ({score:.2f})"
            cv2.putText(vis_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 添加统计信息
        y_offset = 30
        cv2.putText(vis_image, f"检测到 {len(boxes)} 个细胞", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # 显示每个类别的计数
        class_counts = results.get("class_counts", {})
        for cls, count in class_counts.items():
            cv2.putText(vis_image, f"{cls}: {count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return vis_image

    def process_image(self, image_path, output_dir=None, show=False):
        """
        处理单个图像
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
            show: 是否显示结果
            
        Returns:
            dict: 处理结果
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return None
            
            # 进行分割和分类
            results = self.segment_and_classify(image)
            
            # 可视化结果
            vis_image = self.visualize_results(image, results)
            
            # 保存结果
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存可视化结果
                output_path = os.path.join(output_dir, f"{Path(image_path).stem}_result.png")
                cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                
                # 保存结果JSON
                json_path = os.path.join(output_dir, f"{Path(image_path).stem}_result.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json_data = {
                        "image_path": image_path,
                        "num_cells": results["num_cells"],
                        "class_counts": results.get("class_counts", {}),
                        "cells": []
                    }
                    
                    # 添加每个细胞的详细信息
                    for i in range(len(results["boxes"])):
                        cell_info = {
                            "box": results["boxes"][i].tolist(),
                            "class_id": int(results["class_ids"][i]) if i < len(results["class_ids"]) else -1,
                            "class": results["classes"][i] if i < len(results["classes"]) else "未知",
                            "score": float(results["scores"][i]) if i < len(results["scores"]) else 0.0
                        }
                        json_data["cells"].append(cell_info)
                    
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"结果已保存至: {output_path} 和 {json_path}")
            
            # 显示结果
            if show:
                plt.figure(figsize=(12, 8))
                plt.imshow(vis_image)
                plt.title(f"检测到 {results['num_cells']} 个细胞")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            return {
                "image_path": image_path,
                "results": results,
                "vis_image": vis_image
            }
            
        except Exception as e:
            logger.error(f"处理图像出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_batch(self, image_dir, output_dir=None, extensions=('.bmp', '.jpg', '.jpeg', '.png')):
        """
        批量处理图像
        
        Args:
            image_dir: 图像目录
            output_dir: 输出目录
            extensions: 图像扩展名
            
        Returns:
            list: 所有处理结果
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(Path(image_dir).glob(f"*{ext}")))
        
        if not image_paths:
            logger.warning(f"在 {image_dir} 中未找到图像文件")
            return []
        
        logger.info(f"开始处理 {len(image_paths)} 个图像...")
        
        # 批量处理图像
        results = []
        for path in tqdm(image_paths, desc="处理图像"):
            result = self.process_image(str(path), output_dir, show=False)
            if result:
                results.append(result)
        
        logger.info(f"成功处理 {len(results)}/{len(image_paths)} 个图像")
        
        return results

def train_simple_classifier(train_dir, val_dir=None, output_dir="./classifier_model", 
                           num_epochs=10, batch_size=16, learning_rate=0.001):
    """
    训练简单的细胞分类器
    
    Args:
        train_dir: 训练数据目录 (每个类别一个子文件夹)
        val_dir: 验证数据目录 (如果为None则从训练集划分)
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率
        
    Returns:
        str: 最佳模型保存路径
    """
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import ImageFolder
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据扩增和预处理
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    try:
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        
        # 获取类别列表
        class_names = train_dataset.classes
        logger.info(f"发现类别: {class_names}")
        
        # 保存类别映射
        class_mapping = {i: name for i, name in enumerate(class_names)}
        with open(os.path.join(output_dir, "class_mapping.json"), 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        # 划分训练集和验证集
        if val_dir:
            val_dataset = ImageFolder(val_dir, transform=val_transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        else:
            # 从训练集划分验证集 (80% 训练, 20% 验证)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_subset = random_split(
                train_dataset, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
            
            # 更新验证集的变换
            val_subset.dataset.transform = val_transform
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
            
        logger.info(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")
        
        # 创建模型和优化器
        model = CellClassifier(num_classes=len(class_names)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        best_val_loss = float('inf')
        best_model_path = os.path.join(output_dir, "best_model.pth")
        
        logger.info("开始训练...")
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
            
            # 计算训练损失和准确率
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"验证 Epoch {epoch+1}/{num_epochs}"):
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            # 计算验证损失和准确率
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 打印结果
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_acc:.4f}, "
                       f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, best_model_path)
                logger.info(f"保存新的最佳模型，验证损失: {val_loss:.4f}")
        
        logger.info(f"训练完成，最佳模型保存于: {best_model_path}")
        return best_model_path
    
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_cellpose_only(image_path, output_dir=None, show=True, cell_size=30):
    """
    仅运行CellPose分割
    
    Args:
        image_path: 图像路径
        output_dir: 输出目录
        show: 是否显示结果
        cell_size: 细胞直径参数
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"无法读取图像: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 加载CellPose模型
    logger.info("加载CellPose模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.CellposeModel(
        gpu=("cuda" in str(device)), 
        model_type="cyto"
    )
    
    # 运行CellPose分割
    logger.info(f"开始分割，图像尺寸: {img_rgb.shape}")
    start_time = time.time()
    channels = [0, 0]
    masks, flows, styles, diams = model.eval(
        img_rgb, 
        diameter=cell_size,
        channels=channels,
        flow_threshold=0.4,
        cellprob_threshold=0.5,
        do_3D=False
    )
    inference_time = time.time() - start_time
    
    # 检查结果
    unique_masks = np.unique(masks)[1:]  # 跳过0（背景）
    num_cells = len(unique_masks)
    logger.info(f"检测到 {num_cells} 个细胞，用时 {inference_time:.2f} 秒")
    
    # 创建可视化图像
    outlines = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
    
    # 为每个细胞使用不同的颜色
    np.random.seed(42)  # 固定随机种子
    for n in unique_masks:
        color = np.random.randint(0, 256, 3).tolist()
        cell_pixels = masks == n
        outlines[cell_pixels] = color
    
    # 可视化掩码
    overlay = cv2.addWeighted(img_rgb, 0.7, outlines, 0.3, 0)
    
    # 为每个细胞绘制边界框
    vis_image = overlay.copy()
    for n in unique_masks:
        # 获取边界
        cell_mask = masks == n
        y_indices, x_indices = np.where(cell_mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 绘制
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(vis_image, f"细胞 {n}", (x_min, y_min-5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 添加信息
    cv2.putText(vis_image, f"检测到 {num_cells} 个细胞", (10, 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_cellpose.png")
        mask_path = os.path.join(output_dir, f"{Path(image_path).stem}_masks.png")
        
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, masks.astype(np.uint16))
        logger.info(f"结果已保存至: {output_path}")
    
    # 显示结果
    if show:
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("原始图像")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(vis_image)
        plt.title(f"CellPose检测结果 ({num_cells}个细胞)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CellPose与分类器结合的细胞检测系统")
    parser.add_argument("--image", type=str, help="单个图像路径")
    parser.add_argument("--folder", type=str, help="图像文件夹路径")
    parser.add_argument("--classifier", type=str, help="分类器模型路径")
    parser.add_argument("--output", type=str, default="./cellpose_results", help="输出目录")
    parser.add_argument("--cell-size", type=int, default=30, help="CellPose细胞直径参数")
    parser.add_argument("--cellpose-only", action="store_true", help="仅运行CellPose分割，不使用分类器")
    parser.add_argument("--train", action="store_true", help="训练分类器")
    parser.add_argument("--train-dir", type=str, help="训练数据目录（每个类别一个子文件夹）")
    parser.add_argument("--val-dir", type=str, help="验证数据目录（可选）")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    
    args = parser.parse_args()
    
    # 检查参数
    if args.train:
        if not args.train_dir:
            parser.error("使用 --train 时必须提供 --train-dir 参数")
        
        logger.info(f"开始训练分类器，数据目录: {args.train_dir}")
        model_path = train_simple_classifier(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_dir=args.output or "./classifier_model",
            num_epochs=args.epochs
        )
        
        if model_path:
            logger.info(f"分类器训练完成，模型保存于: {model_path}")
        
    elif args.cellpose_only:
        # 仅运行CellPose
        if args.image:
            logger.info(f"使用CellPose分割图像: {args.image}")
            run_cellpose_only(args.image, args.output, True, args.cell_size)
        elif args.folder:
            logger.info(f"使用CellPose分割文件夹: {args.folder}")
            image_paths = []
            for ext in ['.bmp', '.jpg', '.jpeg', '.png']:
                image_paths.extend(list(Path(args.folder).glob(f"*{ext}")))
            
            for path in tqdm(image_paths, desc="处理图像"):
                run_cellpose_only(str(path), args.output, False, args.cell_size)
        else:
            parser.error("必须提供 --image 或 --folder 参数")
            
    else:
        # 运行完整的CellPose+分类器系统
        if not (args.image or args.folder):
            parser.error("必须提供 --image 或 --folder 参数")
            
        # 创建系统
        system = CellPoseWithClassifier(
            classifier_path=args.classifier,
            cell_size=args.cell_size
        )
        
        # 运行
        if args.image:
            logger.info(f"分析图像: {args.image}")
            system.process_image(args.image, args.output, show=True)
        elif args.folder:
            logger.info(f"批量分析文件夹: {args.folder}")
            system.process_batch(args.folder, args.output)

if __name__ == "__main__":
    main()
