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

class MaskGenerator:
    """细胞分割掩码生成器
    
    该类实现了多种细胞分割策略，并将它们组合起来以提高分割精度。
    主要技术包括：颜色空间变换、自适应阈值、分水岭分割和形态学操作。
    """
    def __init__(self, params=None):
        """初始化MaskGenerator
        
        Args:
            params: 分割参数字典，如果为None则使用默认参数
        """
        # 默认参数，可以根据实际情况调整
        self.params = params or {
            # 基本参数
            'min_cell_size': 100,  # 最小细胞面积
            'max_cell_size': 10000,  # 最大细胞面积
            'kernel_size': 5,  # 形态学操作核大小
            'blur_size': 7,  # 高斯模糊核大小
            
            # 颜色阈值
            'hed_threshold': 0.5,  # H&E染色检测阈值
            'hsv_s_threshold': 0.3,  # HSV饱和度阈值
            'hsv_v_threshold': 0.85,  # HSV亮度阈值
            
            # 分水岭参数
            'watershed_threshold': 0.7,  # 分水岭分割阈值
            'distance_threshold': 0.3,  # 距离变换阈值
            
            # 后处理
            'contour_thickness': 2,  # 轮廓线宽度
            'opening_iterations': 1,  # 开操作迭代次数
            'closing_iterations': 2,  # 闭操作迭代次数
        }
        
        # 创建形态学操作的内核
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.params['kernel_size'], self.params['kernel_size'])
        )
        
        # 小内核用于细节处理
        self.small_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (3, 3)
        )

    def generate_consensus_mask(self, image):
        """生成组合式细胞分割掩码
        
        结合多种分割策略的结果，以提高分割的鲁棒性。
        
        Args:
            image: RGB格式的原始图像
            
        Returns:
            final_mask: 二值细胞分割掩码
        """
        # 确保图像是RGB格式
        if len(image.shape) == 2 or image.shape[2] != 3:
            raise ValueError("输入图像必须是RGB格式")
        
        # 1. 创建多种掩码
        hsv_mask = self._generate_hsv_mask(image)
        hed_mask = self._generate_hed_mask(image)
        gray_mask = self._generate_adaptive_threshold_mask(image)
        
        # 2. 合并掩码(投票策略)
        combined_votes = np.zeros(hsv_mask.shape, dtype=np.uint8)
        combined_votes[hsv_mask > 0] += 1
        combined_votes[hed_mask > 0] += 2  # 给HED更高权重
        combined_votes[gray_mask > 0] += 1
        
        # 如果至少有2票，则认为是细胞
        consensus_mask = np.zeros_like(hsv_mask, dtype=np.uint8)
        consensus_mask[combined_votes >= 2] = 255
        
        # 3. 应用分水岭算法分离粘连的细胞
        separated_mask = self._watershed_separation(consensus_mask)
        
        # 4. 后处理：去除小噪点，闭合孔洞
        final_mask = self._postprocess_mask(separated_mask)
        
        return final_mask

    def _generate_hsv_mask(self, image):
        """基于HSV颜色空间的细胞分割
        
        使用HSV颜色空间来区分细胞与背景，通常对染色的细胞效果较好。
        """
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # 对S和V通道应用阈值，识别染色细胞
        _, s_thresh = cv2.threshold(
            s, 
            int(255 * self.params['hsv_s_threshold']), 
            255, 
            cv2.THRESH_BINARY
        )
        _, v_thresh = cv2.threshold(
            v, 
            int(255 * self.params['hsv_v_threshold']), 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # 组合S和V通道的结果
        hsv_mask = cv2.bitwise_and(s_thresh, v_thresh)
        
        # 应用形态学操作优化掩码
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, self.small_kernel)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, self.kernel)
        
        return hsv_mask

    def _generate_hed_mask(self, image):
        """基于H&E染色分离的细胞分割
        
        使用颜色反卷积将H&E染色的图像分离为H（苏木精）和E（伊红）成分。
        这对于病理图像特别有效。
        """
        # 归一化图像
        image_float = image.astype(np.float32) / 255.0
        
        # 避免log(0)错误
        image_float[image_float == 0] = 1e-6
        
        # 颜色反卷积矩阵（H&E染色）- 基于标准矩阵
        hed_matrix = np.array([
            [0.65, 0.70, 0.29],   # H
            [0.07, 0.99, 0.11],   # E
            [0.27, 0.57, 0.78]    # DAB
        ])
        
        # 计算光密度（Optical Density）
        od = -np.log(image_float)
        
        # 颜色反卷积
        try:
            hed = np.dot(od.reshape((-1, 3)), np.linalg.inv(hed_matrix.T))
            hed = hed.reshape(image.shape)
            
            # 提取H通道（主要标记细胞核）
            h_channel = hed[:, :, 0]
            
            # 归一化并应用阈值
            h_norm = cv2.normalize(h_channel, None, 0, 255, cv2.NORM_MINMAX)
            _, h_thresh = cv2.threshold(
                h_norm.astype(np.uint8), 
                int(255 * self.params['hed_threshold']), 
                255, 
                cv2.THRESH_BINARY
            )
            
            # 形态学操作改善掩码质量
            h_thresh = cv2.morphologyEx(h_thresh, cv2.MORPH_OPEN, self.small_kernel)
            h_thresh = cv2.morphologyEx(h_thresh, cv2.MORPH_CLOSE, self.kernel)
            
            return h_thresh
            
        except np.linalg.LinAlgError:
            # 如果反卷积失败，退回到简单的灰度阈值
            print("HED分离失败，使用灰度替代")
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, gray_thresh = cv2.threshold(
                gray, 
                int(255 * 0.7), 
                255, 
                cv2.THRESH_BINARY_INV
            )
            return gray_thresh

    def _generate_adaptive_threshold_mask(self, image):
        """使用自适应阈值的细胞分割
        
        这种方法对光照不均匀的图像效果较好。
        """
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 高斯模糊以减少噪声
        blurred = cv2.GaussianBlur(
            gray, 
            (self.params['blur_size'], self.params['blur_size']), 
            0
        )
        
        # 应用自适应阈值
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # 邻域大小
            2    # 常数减去平均值或加权平均值
        )
        
        # 应用形态学操作
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.small_kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        
        return thresh

    def _watershed_separation(self, mask):
        """使用分水岭算法分离粘连的细胞
        
        Args:
            mask: 输入的二值掩码
            
        Returns:
            分离后的掩码
        """
        # 确保输入掩码为二值图像
        if np.max(mask) != 255:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 执行距离变换
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # 应用阈值确定标记区域（细胞中心）
        _, markers = cv2.threshold(
            dist_norm, 
            self.params['watershed_threshold'], 
            255, 
            cv2.THRESH_BINARY
        )
        markers = markers.astype(np.uint8)
        
        # 寻找标记区域
        contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建标记图像
        marker_img = np.zeros(mask.shape, dtype=np.int32)
        for i, contour in enumerate(contours, start=1):
            cv2.drawContours(marker_img, [contour], -1, i, -1)
        
        # 应用分水岭算法
        # 创建3通道背景图像
        bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.watershed(bgr_mask, marker_img)
        
        # 处理分水岭结果
        separated_mask = np.zeros(mask.shape, dtype=np.uint8)
        separated_mask[marker_img > 0] = 255
        
        return separated_mask

    def _postprocess_mask(self, mask):
        """后处理掩码，去除小噪点，填充孔洞
        
        Args:
            mask: 输入的二值掩码
            
        Returns:
            处理后的掩码
        """
        # 开操作去除小噪点
        cleaned_mask = cv2.morphologyEx(
            mask, 
            cv2.MORPH_OPEN, 
            self.small_kernel, 
            iterations=self.params['opening_iterations']
        )
        
        # 闭操作填充孔洞
        cleaned_mask = cv2.morphologyEx(
            cleaned_mask, 
            cv2.MORPH_CLOSE,
            self.kernel, 
            iterations=self.params['closing_iterations']
        )
        
        # 移除太小的组件
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(cleaned_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params['min_cell_size'] <= area <= self.params['max_cell_size']:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask
    
    def visualize_segmentation(self, image, mask, alpha=0.5):
        """可视化分割结果
        
        Args:
            image: 原始RGB图像
            mask: 分割掩码
            alpha: 透明度参数
            
        Returns:
            visualization: 可视化的图像
        """
        # 创建彩色掩码用于可视化
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = [255, 0, 0]  # 红色掩码
        
        # 使用加权混合创建叠加效果
        blend = cv2.addWeighted(image, 1, color_mask, alpha, 0)
        
        # 添加轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blend, contours, -1, (0, 255, 0), self.params['contour_thickness'])
        
        return blend

class PNGCellDataset(Dataset):
    """使用预处理的PNG图像和掩码的数据集类"""
    def __init__(self, data_dir, transform=None, processed_dir=None):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.transform = transform
        self.image_files = []
        self.mask_files = []
        
        print(f"开始加载预处理PNG数据集: {data_dir}")
        
        if not self.data_dir.exists():
            print(f"警告: 数据目录不存在: {self.data_dir}")
            return
        
        # 首先检查processed文件夹（如果指定了）
        if self.processed_dir and self.processed_dir.exists():
            self._load_from_directory(self.processed_dir)
            if len(self.image_files) > 0:
                print(f"从processed文件夹加载了 {len(self.image_files)} 组图像和掩码")
                return
                
        # 如果processed文件夹不存在或为空，从主数据目录加载
        self._load_from_directory(self.data_dir)
        
        print(f"数据集加载完成，有效样本总数: {len(self.image_files)}")
        
        if len(self.image_files) == 0:
            print("警告: 没有找到有效的训练样本!")
            print("请确保已经运行了预处理脚本生成PNG格式的图像和掩码。")
            print("可以通过执行 'python preprocess_data.py --segmenter' 生成所需数据。")
    
    def _load_from_directory(self, directory):
        """从指定目录加载图像和掩码文件"""
        # 查找所有图像和对应的掩码
        for img_file in directory.glob('*.png'):
            if '_mask' not in img_file.name:  # 只选择图像文件，不包括掩码文件
                mask_file = directory / f"{img_file.stem}_mask.png"
                if mask_file.exists():
                    self.image_files.append(img_file)
                    self.mask_files.append(mask_file)
        
        print(f"从 {directory} 找到 {len(self.image_files)} 组有效的图像和掩码")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]
            mask_path = self.mask_files[idx]
            
            # 读取图像和掩码
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # 转换颜色空间
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 读取掩码 (已经是正确格式: 0=背景, 1=细胞质, 2=细胞核)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"无法读取掩码: {mask_path}")
            
            # 应用数据增强
            if self.transform:
                try:
                    augmented = self.transform(image=image, mask=mask)
                    image = augmented['image']
                    mask = augmented['mask']
                except Exception as e:
                    print(f"数据增强失败: {str(e)}")
            
            # 转换为tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
            return image, mask
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            # 返回一个空的有效样本，以避免训练中断
            empty_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            empty_mask = torch.zeros((256, 256), dtype=torch.long)
            return empty_image, empty_mask

# 保留原来的CellDataset以便向后兼容
class CellDataset(Dataset):
    """细胞图像数据集类，具有增强的错误处理"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_files = []
        self.cyt_files = []
        self.nuc_files = []
        self.valid_indices = []  # 存储有效的样本索引
        
        print(f"开始加载数据集: {data_dir}")
        print(f"检查数据目录是否存在: {'存在' if self.data_dir.exists() else '不存在'}")
        
        # 遍历所有细胞类型文件夹
        cell_types = ['im_Dyskeratotic', 'im_Koilocytotic', 'im_Metaplastic', 
                     'im_Parabasal', 'im_Superficial-Intermediate']
        
        for cell_type in cell_types:
            cell_type_path = self.data_dir / cell_type
            if not cell_type_path.exists():
                print(f"警告: 细胞类型文件夹不存在: {cell_type_path}")
                continue
                
            print(f"处理细胞类型: {cell_type}")
                
            # 获取所有未裁剪的图像
            image_count = 0
            valid_count = 0
            for img_file in cell_type_path.glob('*.bmp'):
                if 'CROPPED' not in str(img_file):
                    image_count += 1
                    base_name = img_file.stem
                    cyt_files = list(cell_type_path.glob(f'{base_name}_cyt*.dat'))
                    nuc_files = list(cell_type_path.glob(f'{base_name}_nuc*.dat'))
                    
                    if cyt_files and nuc_files:
                        # 验证图像文件是否可读取
                        try:
                            # 使用二进制模式读取文件
                            with open(img_file, 'rb') as f:
                                img_data = f.read()
                            
                            # 解码图像数据
                            img_array = np.frombuffer(img_data, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            
                            if img is None or img.size == 0:
                                print(f"警告: 无法解码图像: {img_file}")
                                continue
                                
                            # 图像有效，记录索引
                            idx = len(self.image_files)
                            self.image_files.append(img_file)
                            self.cyt_files.append(cyt_files)
                            self.nuc_files.append(nuc_files)
                            self.valid_indices.append(idx)
                            valid_count += 1
                            
                        except Exception as e:
                            print(f"读取图像出错 {img_file}: {str(e)}")
            
            print(f"  找到图像: {image_count}, 有效图像: {valid_count}")
        
        print(f"数据集加载完成，有效样本总数: {len(self.valid_indices)}")
        if len(self.valid_indices) == 0:
            print("警告: 没有找到有效的训练样本!")
            print("请确保数据集中存在可读取的图像文件和对应的边界文件。")

    def __len__(self):
        return len(self.valid_indices)

    def _read_boundary_file(self, file_path):
        """读取边界坐标文件"""
        coordinates = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        x, y = map(float, line.strip().split(','))
                        coordinates.append([int(x), int(y)])
                    except ValueError:
                        print(f"警告: 解析边界文件行失败: {line}")
                        continue
        except Exception as e:
            print(f"读取边界文件出错 {file_path}: {str(e)}")
            return np.array([[0, 0]], dtype=np.int32)  # 返回空边界
            
        if len(coordinates) < 3:
            print(f"警告: 边界文件坐标点不足: {file_path}")
            return np.array([[0, 0], [1, 1], [0, 1]], dtype=np.int32)  # 返回有效但最小的边界
            
        return np.array(coordinates, dtype=np.int32)

    def _create_mask_from_boundaries(self, shape, cyt_files, nuc_files):
        """从边界文件创建掩码"""
        mask = np.zeros(shape, dtype=np.uint8)
        
        try:
            # 绘制细胞质边界
            for cyt_file in cyt_files:
                boundary = self._read_boundary_file(cyt_file)
                if boundary.size > 3:  # 确保有足够的点形成多边形
                    cv2.fillPoly(mask, [boundary], 1)
                
            # 绘制细胞核边界
            for nuc_file in nuc_files:
                boundary = self._read_boundary_file(nuc_file)
                if boundary.size > 3:  # 确保有足够的点形成多边形
                    cv2.fillPoly(mask, [boundary], 2)
        except Exception as e:
            print(f"创建掩码出错: {str(e)}")
            # 返回空掩码但不抛出异常
        
        return mask

    def __getitem__(self, idx):
        try:
            # 使用有效索引获取数据
            valid_idx = self.valid_indices[idx]
            img_path = self.image_files[valid_idx]
            
            # 使用二进制方式打开图像文件
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            # 解码图像数据
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None or image.size == 0:
                raise ValueError(f"无法解码图像: {img_path}")
                
            # 转换颜色空间
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 创建掩码
            mask = self._create_mask_from_boundaries(
                image.shape[:2],
                self.cyt_files[valid_idx],
                self.nuc_files[valid_idx]
            )
            
            # 应用数据增强
            if self.transform:
                try:
                    augmented = self.transform(image=image, mask=mask)
                    image = augmented['image']
                    mask = augmented['mask']
                except Exception as e:
                    print(f"数据增强失败: {str(e)}")
                    # 如果增强失败，使用原始图像和掩码
            
            # 转换为tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
            return image, mask
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            # 返回一个空的有效样本，以避免训练中断
            empty_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            empty_mask = torch.zeros((256, 256), dtype=torch.long)
            return empty_image, empty_mask

def train_segmenter(use_preprocessed=True):
    """训练细胞分割模型
    
    Args:
        use_preprocessed: 是否使用预处理的PNG数据
    """
    try:
        print(f"加载分割数据集: {SEGMENTER_DATA_PATH}")
        
        # 根据参数选择数据集类
        if use_preprocessed:
            print("使用预处理的PNG格式数据集")
            dataset = PNGCellDataset(
                SEGMENTER_DATA_PATH,
                transform=SEGMENTER_TRANSFORM
            )
        else:
            print("使用原始BMP和DAT文件数据集")
            dataset = CellDataset(
                SEGMENTER_DATA_PATH,
                transform=SEGMENTER_TRANSFORM
            )
        
        # 确保数据集不为空
        if len(dataset) == 0:
            print("错误: 数据集为空，无法进行训练")
            print("如果使用预处理数据，请先运行: python preprocess_data.py --segmenter")
            return
        
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
        
        # 初始化训练跟踪指标
        train_losses = []
        val_losses = []
        train_dice_scores = []
        val_dice_scores = []
        
        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(EPOCHS):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_dice = 0.0
            batch_count = 0
            
            for i, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 计算Dice系数（评估分割质量）
                pred_masks = torch.argmax(outputs, dim=1)
                dice_score = calculate_dice_score(pred_masks, masks)
                train_dice += dice_score
                
                batch_count += 1
                
                if i % 10 == 9:
                    print(f'Epoch [{epoch+1}/{EPOCHS}], '
                          f'Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {train_loss/10:.4f}, '
                          f'Dice: {train_dice/batch_count:.4f}')
                    
            # 计算平均训练指标
            avg_train_loss = train_loss / batch_count
            avg_train_dice = train_dice / batch_count
            train_losses.append(avg_train_loss)
            train_dice_scores.append(avg_train_dice)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    # 计算验证集的Dice系数
                    pred_masks = torch.argmax(outputs, dim=1)
                    dice_score = calculate_dice_score(pred_masks, masks)
                    val_dice += dice_score
                    
                    val_batch_count += 1
            
            # 计算平均验证指标
            avg_val_loss = val_loss / val_batch_count
            avg_val_dice = val_dice / val_batch_count
            val_losses.append(avg_val_loss)
            val_dice_scores.append(avg_val_dice)
            
            print(f'Epoch [{epoch+1}/{EPOCHS}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}')
            
            # 学习率调整
            scheduler.step(avg_val_loss)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), SEGMENT_MODEL_SAVE_PATH)
                print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        
        print("分割器训练完成")
        
        # 可视化训练结果
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Dice系数曲线
        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(train_dice_scores) + 1), train_dice_scores, 'b-', label='训练Dice系数')
        plt.plot(range(1, len(val_dice_scores) + 1), val_dice_scores, 'r-', label='验证Dice系数')
        plt.title('Dice系数曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.grid(True)
        plt.legend()
        
        # 训练/验证损失差值曲线（用于监测过拟合）
        plt.subplot(2, 2, 3)
        loss_diff = [val - train for train, val in zip(train_losses, val_losses)]
        plt.plot(range(1, len(loss_diff) + 1), loss_diff, 'g-', label='验证损失-训练损失')
        plt.title('过拟合监测')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.legend()
        
        # 保存可视化结果
        vis_dir = os.path.join(os.path.dirname(SEGMENT_MODEL_SAVE_PATH), "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        plt_save_path = os.path.join(vis_dir, "segmenter_training_plot.png")
        plt.savefig(plt_save_path)
        print(f"可视化结果已保存至: {plt_save_path}")
        plt.show()
        
    except Exception as e:
        print(f"分割器训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def calculate_dice_score(pred, target):
    """计算Dice系数
    
    Args:
        pred: 预测的分割掩码
        target: 目标分割掩码
        
    Returns:
        dice_score: Dice系数 (0-1之间)
    """
    smooth = 1e-5  # 防止除零
    
    # 对每个类别分别计算Dice系数
    dice_scores = []
    
    # 背景通常被忽略（索引0）
    for cls in range(1, 3):  # 对于细胞质(1)和细胞核(2)分别计算
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    
    # 返回所有类别的平均Dice系数
    return np.mean(dice_scores)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练细胞分割模型')
    parser.add_argument('--use-raw', action='store_true', help='使用原始BMP和DAT格式数据，而不是预处理的PNG格式数据')
    args = parser.parse_args()
    
    train_segmenter(use_preprocessed=not args.use_raw)