import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import (
    TRANSFORM, 
    CLASSIFIER_TRANSFORM,
    SEGMENTER_TRANSFORM,
    IMAGE_SIZE
)

class SIPaKMEDDataset(Dataset):
    def __init__(self, root_dir, transform=TRANSFORM, load_boundaries=True, include_CROPRED=False, only_CROPRED=False):
        """
        初始化数据集类
        :param root_dir: 数据集根目录
        :param transform: 图像变换
        :param load_boundaries: 是否加载边界数据
        :param include_CROPRED: 是否包含CROPRED子文件夹数据
        :param only_CROPRED: 是否只使用CROPRED子文件夹数据
        """
        self.root_dir = root_dir
        self.transform = transform
        self.load_boundaries = load_boundaries
        self.include_CROPRED = include_CROPRED
        self.only_CROPRED = only_CROPRED
        self.image_paths = []
        self.labels = []
        self.boundaries_dict = {}

        # 获取所有类别文件夹
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        
        # 初始化数据集
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            
            if only_CROPRED:
                # 只处理 CROPRED 目录
                CROPRED_dir = os.path.join(cls_dir, "CROPRED")
                if os.path.isdir(CROPRED_dir):
                    self._process_directory(CROPRED_dir, label)
            else:
                # 处理主目录
                if not include_CROPRED:
                    self._process_directory(cls_dir, label)
                else:
                    # 处理主目录和 CROPRED 目录
                    self._process_directory(cls_dir, label)
                    CROPRED_dir = os.path.join(cls_dir, "CROPRED")
                    if os.path.isdir(CROPRED_dir):
                        self._process_directory(CROPRED_dir, label)

    def _process_directory(self, directory, label):
        """
        处理指定目录中的图像和边界文件
        """
        allowed_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif')
        
        for img_name in os.listdir(directory):
            if img_name.lower().endswith(allowed_extensions):
                img_path = os.path.join(directory, img_name)
                img_base = os.path.splitext(img_name)[0]
                
                # 收集该图像对应的所有边界文件
                boundaries = {
                    'nuc': sorted([f for f in os.listdir(directory) 
                                 if f.startswith(img_base) and '_nuc' in f and f.endswith('.dat')]),
                    'cyt': sorted([f for f in os.listdir(directory) 
                                 if f.startswith(img_base) and '_cyt' in f and f.endswith('.dat')])
                }
                
                self.image_paths.append(img_path)
                self.labels.append(label)
                self.boundaries_dict[img_path] = boundaries

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        """
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # 读取原始图像并获取原始尺寸
            img_array = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            original_h, original_w = image.shape[:2]
            
            # 转换颜色空间并调整大小
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            
            if self.transform:
                image = self.transform(image)
            
            # 如果需要加载边界数据
            if self.load_boundaries:
                boundaries = {'nuc': [], 'cyt': []}
                boundary_files = self.boundaries_dict.get(img_path, {'nuc': [], 'cyt': []})
                directory = os.path.dirname(img_path)
                
                # 读取边界文件
                for key in ['nuc', 'cyt']:
                    for boundary_name in boundary_files[key]:
                        boundary_path = os.path.join(directory, boundary_name)
                        try:
                            coords = []
                            with open(boundary_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        try:
                                            x, y = map(float, line.strip().split(','))
                                            # 归一化坐标到 [0, 1] 范围
                                            x = x / original_w
                                            y = y / original_h
                                            coords.append([x, y])
                                        except ValueError:
                                            continue
                            if coords:
                                boundaries[key].append(np.array(coords))
                        except Exception as e:
                            print(f"读取边界文件失败 {boundary_path}: {str(e)}")
                            continue
                
                # 确保返回的边界数据格式一致
                return image, label, {
                    'nuc': boundaries['nuc'],
                    'cyt': boundaries['cyt'],
                    'original_size': (original_h, original_w)
                }
            
            return image, label, {
                'nuc': [],
                'cyt': [],
                'original_size': (IMAGE_SIZE, IMAGE_SIZE)
            }
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            # 返回默认值
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), label, {
                'nuc': [],
                'cyt': [],
                'original_size': (IMAGE_SIZE, IMAGE_SIZE)
            }

    def __len__(self):
        """
        返回数据集的大小
        :return: 数据集中样本的数量
        """
        return len(self.image_paths)

class ExtraDataset(Dataset):
    def __init__(self, root_dir, transform=TRANSFORM):
        """
        初始化额外数据集(仅包含图片)
        :param root_dir: 数据集根目录
        :param transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 获取所有类别文件夹
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        
        # 遍历每个类别文件夹
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            
            # 只收集图片文件
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tif')):
                    img_path = os.path.join(cls_dir, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取一个样本
        :param idx: 样本索引
        :return: (image, label, boundaries) - 与 SIPaKMEDDataset 保持一致的返回格式
        """
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # 读取图像
            img_array = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            original_h, original_w = image.shape[:2]
            
            # 转换颜色空间并调整大小
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            
            if self.transform:
                image = self.transform(image)
            
            # 返回与 SIPaKMEDDataset 相同格式的数据
            return image, label, {
                'nuc': [],
                'cyt': [],
                'original_size': (original_h, original_w)
            }
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            # 返回默认值
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), label, {
                'nuc': [],
                'cyt': [],
                'original_size': (IMAGE_SIZE, IMAGE_SIZE)
            }

class ClassifierDataset(Dataset):
    """用于分类器训练的数据集类"""
    def __init__(self, root_dir, transform=CLASSIFIER_TRANSFORM):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 获取所有类别文件夹
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        
        print(f"初始化分类数据集，根目录: {root_dir}")
        print(f"找到类别: {self.classes}")
        
        # 收集图片
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            print(f"处理类别 {cls}")
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tif')):
                    img_path = os.path.join(cls_dir, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
        
        print(f"数据集初始化完成，共加载 {len(self.image_paths)} 个样本")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # 读取图像
            img_array = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), label

class SegmenterDataset(Dataset):
    """分割器数据集类"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # 收集所有支持的图像文件
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(valid_extensions):
                self.image_paths.append(os.path.join(root_dir, filename))
                
        print(f"找到 {len(self.image_paths)} 个图像文件")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取数据集项
        返回:
            image: 预处理后的图像张量
            shape: 原始图像尺寸的张量
        """
        try:
            # 读取图像
            img_path = self.image_paths[idx]
            img_array = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # 记录原始尺寸
            original_h, original_w = image.shape[:2]
            
            # 转换颜色空间
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            # 返回图像和尺寸信息(转换为tensor)
            shape = torch.tensor([original_h, original_w], dtype=torch.long)
            return image, shape
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            # 返回空白图像和默认尺寸
            return (torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), 
                   torch.tensor([IMAGE_SIZE, IMAGE_SIZE], dtype=torch.long))

def read_boundary_file(file_path):
    """
    读取边界文件的坐标点
    :param file_path: 边界文件路径
    :return: 坐标点列表
    """
    try:
        coords = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # 忽略空行
                    try:
                        x, y = map(float, line.strip().split(','))
                        coords.append([x, y])
                    except ValueError as e:
                        print(f"无法解析坐标行: {line.strip()}, 错误: {str(e)}")
                        continue
        return np.array(coords) if coords else None
    except Exception as e:
        print(f"读取边界文件失败 {file_path}: {str(e)}")
        return None