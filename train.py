import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import SIPaKMEDDataset, ExtraDataset
from segmenter import UNet
from classifier import CellClassifier
from config import (
    DATASET_PATH, 
    EXTRA_DATASET_PATH, 
    SEGMENT_MODEL_SAVE_PATH, 
    CLASSIFY_MODEL_SAVE_PATH, 
    BATCH_SIZE, 
    LEARNING_RATE, 
    EPOCHS, 
    CLASSES,
    TRANSFORM,
    IMAGE_SIZE  # 添加 IMAGE_SIZE 导入
)
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np

def create_segmentation_mask(boundaries):
    """
    根据边界坐标创建分割掩码
    """
    H, W = IMAGE_SIZE, IMAGE_SIZE
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # 处理细胞核边界
    for coords in boundaries['nuc']:
        if len(coords) > 2:
            # 转换归一化坐标回像素坐标
            pixel_coords = (coords * np.array([W, H])).astype(np.int32)
            cv2.fillPoly(mask, [pixel_coords], 1)
    
    # 处理细胞质边界
    for coords in boundaries['cyt']:
        if len(coords) > 2:
            # 转换归一化坐标回像素坐标
            pixel_coords = (coords * np.array([W, H])).astype(np.int32)
            cv2.fillPoly(mask, [coords], 1)
    
    return mask

def train_resnet(dataset_path=DATASET_PATH, extra_dataset_path=EXTRA_DATASET_PATH):
    """
    训练分类器：使用 SIPakMED 和 Extra 两个数据集
    """
    try:
        # 加载 SIPakMED 数据集，仅用于分类（不需要边界数据）
        primary_dataset = SIPaKMEDDataset(
            dataset_path, 
            transform=TRANSFORM,
            load_boundaries=False,
            include_CROPRED=True
        )
        
        # 加载额外数据集
        extra_dataset = ExtraDataset(
            extra_dataset_path,
            transform=TRANSFORM
        )
        
        # 合并数据集
        combined_dataset = ConcatDataset([primary_dataset, extra_dataset])
        dataloader = DataLoader(
            combined_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 初始化分类模型
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES)).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 训练循环
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, data in enumerate(dataloader):
                # 处理不同数据集返回的数据格式
                if len(data) == 3:  # SIPaKMEDDataset
                    images, labels, _ = data
                else:  # ExtraDataset
                    images, labels = data
                
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if i % 10 == 9:
                    accuracy = 100. * correct / total
                    print(f'Epoch [{epoch+1}/{EPOCHS}], '
                          f'Step [{i+1}/{len(dataloader)}], '
                          f'Loss: {running_loss/10:.4f}, '
                          f'Accuracy: {accuracy:.2f}%')
                    running_loss = 0.0
                    correct = 0
                    total = 0
        
        print("分类器训练完成")
        torch.save(model.state_dict(), CLASSIFY_MODEL_SAVE_PATH)
        
    except Exception as e:
        print(f"分类器训练出错: {str(e)}")
        raise

def train_unet(dataset_path=DATASET_PATH):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        dataset = SIPaKMEDDataset(
            dataset_path,
            transform=TRANSFORM,
            load_boundaries=True,
            include_CROPRED=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        
        model = UNet(n_channels=3, n_classes=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            for i, (images, _, boundaries) in enumerate(dataloader):
                try:
                    images = images.to(device)
                    batch_size = images.size(0)
                    
                    # 创建批次掩码
                    masks = torch.zeros((batch_size, 1, IMAGE_SIZE, IMAGE_SIZE)).to(device)
                    
                    # 为每个图像创建掩码
                    for batch_idx in range(batch_size):
                        mask = create_segmentation_mask(boundaries[batch_idx])
                        masks[batch_idx] = torch.from_numpy(mask).float().unsqueeze(0)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    if i % 10 == 9:
                        print(f'Epoch [{epoch+1}/{EPOCHS}], '
                              f'Step [{i+1}/{len(dataloader)}], '
                              f'Loss: {running_loss/10:.4f}')
                        running_loss = 0.0
                        
                except Exception as e:
                    print(f"处理批次 {i} 时出错: {str(e)}")
                    continue
            
        print("分割器训练完成")
        torch.save(model.state_dict(), SEGMENT_MODEL_SAVE_PATH)
        
    except Exception as e:
        print(f"分割器训练出错: {str(e)}")
        raise

if __name__ == "__main__":
    print("开始训练分类器...")
    train_resnet()
    print("\n开始训练分割器...")
    train_unet()