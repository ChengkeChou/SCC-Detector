# SAM2细胞分析系统 🔬

基于SAM2的智能细胞分析工具，提供分割、分类和可视化功能。

## 📋 文件说明

- `main.py` - 主程序（GUI界面）
- `requirements.txt` - Python依赖包
- `README.md` - 使用说明

## 🚀 快速开始

### 1. 安装依赖
```bash
# 安装Python依赖
pip install -r requirements.txt

# GPU加速版本（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 启动程序
```bash
python main.py
```

## ⚙️ 环境要求

- **Python**: 3.8-3.11
- **内存**: 最少8GB
- **显卡**: 可选，NVIDIA GPU可加速
- **存储**: 至少2GB可用空间

## 🔧 使用说明

1. **启动程序**: 运行 `python main.py`
2. **加载模型**: 
   - 选择SAM2配置文件（.yaml）
   - 选择模型权重文件（.pt）
3. **加载图像**: 点击"加载图像"选择细胞图像
4. **开始分析**: 点击"开始分析"进行处理
5. **查看结果**: 在界面中查看分割结果

## 📝 支持格式

- **图像**: PNG, JPG, JPEG, BMP, TIFF
- **模型**: .pt, .pth (PyTorch权重)
- **配置**: .yaml (SAM2配置文件)

## ❗ 注意事项

1. 首次运行需要下载SAM2模型文件（约1-2GB）
2. 建议图像分辨率不超过2048x2048
3. 如遇到问题，请检查Python环境和依赖版本

## 🆘 常见问题

**Q: 程序无法启动？**
A: 检查Python版本和依赖安装：`python --version` 和 `pip list`

**Q: 分析速度慢？**
A: 安装GPU版本的PyTorch，或降低图像分辨率

**Q: 内存不足？**
A: 减少分析参数或使用更小的图像

---

**感谢使用SAM2细胞分析系统！** 🌟
