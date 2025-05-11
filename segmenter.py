import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """双重卷积模块
    包含两个连续的卷积层，每个卷积后跟批归一化和ReLU激活
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        mid_channels (int, optional): 中间层通道数，默认等于输出通道数
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 第二个卷积层
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样模块
    使用最大池化进行下采样，然后进行双重卷积
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2最大池化，步长为2
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样模块
    可以选择使用双线性插值或转置卷积进行上采样
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        bilinear (bool): 是否使用双线性插值进行上采样
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            # 使用双线性插值上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 使用转置卷积上采样
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 计算特征图大小差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 对x1进行padding以匹配x2的大小
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 连接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积模块
    最终的1x1卷积层，用于生成预测图
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数（通常等于类别数）
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet模型实现
    用于医学图像分割的U形网络结构
    
    参数:
        n_channels (int): 输入图像的通道数（默认为3，RGB图像）
        n_classes (int): 分割类别数（默认为1，二分类）
        bilinear (bool): 是否使用双线性插值进行上采样
    
    网络结构:
        - 编码器部分：4次下采样，特征图尺寸逐渐减小
        - 解码器部分：4次上采样，特征图尺寸逐渐增大
        - 跳跃连接：将编码器的特征图与解码器对应层连接
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 解码器部分
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 输出层
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """前向传播
        Args:
            x (tensor): 输入图像，形状为(batch_size, channels, height, width)
        Returns:
            tensor: 分割预测图，形状为(batch_size, n_classes, height, width)
        """
        # 编码器路径
        x1 = self.inc(x)      # 第一层特征
        x2 = self.down1(x1)   # 第一次下采样
        x3 = self.down2(x2)   # 第二次下采样
        x4 = self.down3(x3)   # 第三次下采样
        x5 = self.down4(x4)   # 第四次下采样
        
        # 解码器路径（带跳跃连接）
        x = self.up1(x5, x4)  # 第一次上采样
        x = self.up2(x, x3)   # 第二次上采样
        x = self.up3(x, x2)   # 第三次上采样
        x = self.up4(x, x1)   # 第四次上采样
        logits = self.outc(x) # 生成最终预测
        return logits