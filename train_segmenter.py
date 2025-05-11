import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from segmenter import UNet
from config import *

class CellDataset(Dataset):
    """细胞图像数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_files = []
        self.cyt_files = []
        self.nuc_files = []
        
        # 遍历所有细胞类型文件夹
        cell_types = ['im_Dyskeratotic', 'im_Koilocytotic', 'im_Metaplastic', 
                     'im_Parabasal', 'im_Superficial-Intermediate']
        
        for cell_type in cell_types:
            cell_type_path = self.data_dir / cell_type
            if not cell_type_path.exists():
                continue
                
            # 获取所有未裁剪的图像
            for img_file in cell_type_path.glob('*.bmp'):
                if 'CROPPED' not in str(img_file):
                    base_name = img_file.stem
                    cyt_files = list(cell_type_path.glob(f'{base_name}_cyt*.dat'))
                    nuc_files = list(cell_type_path.glob(f'{base_name}_nuc*.dat'))
                    
                    if cyt_files and nuc_files:
                        self.image_files.append(img_file)
                        self.cyt_files.append(cyt_files)
                        self.nuc_files.append(nuc_files)

    def __len__(self):
        return len(self.image_files)

    def _read_boundary_file(self, file_path):
        """读取边界坐标文件"""
        coordinates = []
        with open(file_path, 'r') as f:
            for line in f:
                x, y = map(float, line.strip().split(','))
                coordinates.append([int(x), int(y)])
        return np.array(coordinates, dtype=np.int32)

    def _create_mask_from_boundaries(self, shape, cyt_files, nuc_files):
        """从边界文件创建掩码"""
        mask = np.zeros(shape, dtype=np.uint8)
        
        # 绘制细胞质边界
        for cyt_file in cyt_files:
            boundary = self._read_boundary_file(cyt_file)
            cv2.fillPoly(mask, [boundary], 1)
            
        # 绘制细胞核边界
        for nuc_file in nuc_files:
            boundary = self._read_boundary_file(nuc_file)
            cv2.fillPoly(mask, [boundary], 2)
            
        return mask

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建掩码
        mask = self._create_mask_from_boundaries(
            image.shape[:2],
            self.cyt_files[idx],
            self.nuc_files[idx]
        )
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask

def train_segmenter():
    """训练细胞分割模型"""
    try:
        print(f"加载分割数据集: {SEGMENTER_DATA_PATH}")
        
        # 创建数据集
        dataset = CellDataset(
            SEGMENTER_DATA_PATH,
            transform=SEGMENTER_TRANSFORM
        )
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建模型
        model = UNet(n_channels=3, n_classes=3).to(device)  # 3类: 背景、细胞质、细胞核
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(EPOCHS):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for i, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if i % 10 == 9:
                    print(f'Epoch [{epoch+1}/{EPOCHS}], '
                          f'Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {train_loss/10:.4f}')
                    train_loss = 0.0
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {val_loss:.4f}')
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), SEGMENT_MODEL_SAVE_PATH)
                print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        
        print("分割器训练完成")
        
    except Exception as e:
        print(f"分割器训练出错: {str(e)}")
        raise

if __name__ == '__main__':
    train_segmenter()