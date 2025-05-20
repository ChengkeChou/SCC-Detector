import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import CLASSES

class CellClassifier(nn.Module):
    def __init__(self):
        super(CellClassifier, self).__init__()
        # 使用新的权重初始化方式
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # 或者使用 DEFAULT 获取最新权重
        # self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # 修改分类器层以匹配我们的类别数
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))

    def forward(self, x):
        return self.model(x)