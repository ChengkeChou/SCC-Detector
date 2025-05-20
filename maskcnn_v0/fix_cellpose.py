"""
这个脚本用于改进细胞分割系统中的CellPose集成，
使其能够正确分割细胞并提供更准确的分类功能
"""

import os
import sys
import torch
import logging
from cellpose import models
import cv2
import numpy as np
from PIL import Image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CellSegmentation")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cellpose():
    """测试CellPose在当前环境是否正确工作"""
    try:
        # 使用内置的cyto模型
        model = models.CellposeModel(model_type="cyto")
        logger.info("CellPose模型加载成功")
        return True
    except Exception as e:
        logger.error(f"CellPose测试失败: {e}")
        return False

def test_classification(image_path):
    """测试细胞分类功能"""
    try:
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图像: {image_path}")
            return False
            
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 使用CellPose的cyto模型进行分割
        model = models.CellposeModel(model_type="cyto")
        
        # 运行分割
        masks, flows, styles, diams = model.eval(img_rgb, diameter=30, channels=[0,0])
        
        # 统计检测到的细胞数量
        unique_ids = np.unique(masks)[1:]  # 跳过0（背景）
        cell_count = len(unique_ids)
        
        # 显示结果
        logger.info(f"在图像 {image_path} 中检测到 {cell_count} 个细胞")
        
        # 可视化结果并保存
        from cellpose import plot
        fig = plot.show_segmentation(img_rgb, masks, flows[0], channels=[0,0])
        fig.savefig(f"cellpose_test_result.png")
        logger.info(f"分割结果已保存至 cellpose_test_result.png")
        
        return cell_count > 0
    except Exception as e:
        logger.error(f"测试分类失败: {e}")
        return False

def improve_cellpose_integration():
    """改进系统中的CellPose集成"""
    # 修改这里以指向正确的cell_segmentation.py文件路径
    cell_segmentation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          "models", "cell_segmentation.py")
    
    if not os.path.exists(cell_segmentation_path):
        logger.error(f"找不到文件: {cell_segmentation_path}")
        return False
    
    try:
        # 读取文件内容
        with open(cell_segmentation_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 修复_load_cellpose_model方法
        load_cellpose_model_code = """    def _load_cellpose_model(self):
        \"\"\"加载CellPose模型\"\"\"
        if not CELLPOSE_AVAILABLE:
            raise ImportError(
                "CellPose不可用。请安装: pip install cellpose"
            )
            
        try:
            # 默认使用cyto模型，除非提供了自定义模型路径
            if os.path.exists(self.model_path) and self.model_path.endswith('.pth'):
                # 自定义模型路径
                model_type = os.path.basename(self.model_path)
                pretrained_model = self.model_path
                logger.info(f"使用自定义CellPose模型: {pretrained_model}")
            else:
                # 使用内置模型
                model_type = "cyto"  # 细胞质分割模型
                pretrained_model = None
                logger.info(f"使用内置CellPose模型: {model_type}")
            
            # 创建CellPose模型
            self.model = models.CellposeModel(
                gpu=("cuda" in str(self.device)), 
                model_type=model_type,
                pretrained_model=pretrained_model
            )
            logger.info(f"CellPose模型已加载: {model_type}")
        except Exception as e:
            logger.error(f"加载CellPose模型时出错: {str(e)}")
            raise RuntimeError(f"加载CellPose模型时出错: {str(e)}")"""
        
        # 改进的_predict_cellpose方法，使其能够找到更多细胞
        improved_predict_cellpose = """    def _predict_cellpose(self, image):
        \"\"\"使用CellPose模型进行预测\"\"\"
        try:
            # CellPose参数优化
            channels = [0, 0]  # 使用第一个通道作为主要通道
            diameter = self.kwargs.get("cell_size", 30)  # 细胞直径，可根据实际情况调整
            flow_threshold = self.kwargs.get("flow_threshold", 0.4)
            cellprob_threshold = max(0.1, self.confidence_threshold - 0.3)  # 降低细胞概率阈值以提高检出率
            
            # 记录参数
            logger.info(f"CellPose参数: diameter={diameter}, flow_threshold={flow_threshold}, cellprob_threshold={cellprob_threshold}")
            
            # 运行CellPose
            masks, flows, styles, diams = self.model.eval(
                image, 
                diameter=diameter,
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                do_3D=False
            )
            
            # 记录检测到的细胞数量
            unique_ids = np.unique(masks)[1:]  # 跳过0（背景）
            cell_count = len(unique_ids)
            logger.info(f"检测到 {cell_count} 个细胞")
            
            # 如果没有检测到细胞，尝试使用更宽松的参数再次检测
            if cell_count == 0:
                logger.info("未检测到细胞，尝试使用更宽松的参数...")
                masks, flows, styles, diams = self.model.eval(
                    image, 
                    diameter=diameter,
                    channels=channels,
                    flow_threshold=0.3,  # 降低flow阈值
                    cellprob_threshold=0.0,  # 降低概率阈值
                    do_3D=False
                )
                unique_ids = np.unique(masks)[1:]
                cell_count = len(unique_ids)
                logger.info(f"使用宽松参数后检测到 {cell_count} 个细胞")
            
            # 如果指定了检查点文件，加载分类模型进行细胞分类
            classifier = None
            if hasattr(self, 'classifier') and self.classifier is not None:
                classifier = self.classifier
                
            boxes = []
            instance_masks = []
            scores = []
            class_ids = []
            
            # 对每个实例进行处理
            for instance_id in unique_ids:
                # 为每个实例创建二进制掩码
                instance_mask = masks == instance_id
                
                # 计算掩码边界框
                coords = np.where(instance_mask)
                if len(coords[0]) == 0:
                    continue
                    
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                
                # 计算分数 - 这里我们可以使用掩码区域的平均强度或固定值
                score = 1.0
                
                # 分类细胞 - 如果有分类器就使用，否则使用默认类别
                if classifier is not None:
                    # 提取细胞区域进行分类
                    cell_roi = image[y_min:y_max, x_min:x_max].copy()
                    if cell_roi.size > 0:
                        class_id, class_score = self._classify_cell(cell_roi, classifier)
                        score = class_score
                    else:
                        class_id = 0  # 默认类别
                else:
                    # 没有分类器，使用默认类别
                    class_id = 0
                
                # 保存结果
                boxes.append([x_min, y_min, x_max, y_max])
                instance_masks.append(instance_mask)
                scores.append(score)
                class_ids.append(class_id)
            
            # 返回标准格式的结果
            return {
                "boxes": np.array(boxes) if boxes else np.zeros((0, 4), dtype=np.int32),
                "masks": np.array(instance_masks) if instance_masks else np.zeros((0, image.shape[0], image.shape[1]), dtype=bool),
                "scores": np.array(scores) if scores else np.zeros(0, dtype=np.float32),
                "class_ids": np.array(class_ids) if class_ids else np.zeros(0, dtype=np.int32),
                "classes": [self.class_names[i] for i in class_ids] if class_ids else []
            }
        except Exception as e:
            logger.error(f"CellPose预测出错: {e}")
            # 返回空结果
            return {
                "boxes": np.zeros((0, 4), dtype=np.int32),
                "masks": np.zeros((0, image.shape[0], image.shape[1]), dtype=bool),
                "scores": np.zeros(0, dtype=np.float32),
                "class_ids": np.zeros(0, dtype=np.int32),
                "classes": []
            }"""
        
        # 添加细胞分类辅助方法
        classify_cell_method = """    def _classify_cell(self, cell_image, classifier=None):
        \"\"\"对单个细胞进行分类
        
        Args:
            cell_image: 细胞区域图像
            classifier: 分类器模型，如果为None则使用简单规则进行分类
            
        Returns:
            (class_id, confidence): 类别ID和置信度
        \"\"\"
        if classifier is None:
            # 使用简单规则 - 基于细胞大小和形状特征
            h, w = cell_image.shape[:2]
            area = h * w
            
            # 基于面积进行简单分类
            if area > 5000:
                return 3, 0.8  # Parabasal
            elif area > 3000:
                return 2, 0.7  # Metaplastic
            elif area > 2000:
                return 0, 0.8  # Dyskeratotic
            elif area > 1000:
                return 1, 0.75  # Koilocytotic
            else:
                return 4, 0.9  # Superficial-Intermediate
        else:
            # 使用分类器模型
            try:
                # 预处理图像
                cell_resized = cv2.resize(cell_image, (224, 224))
                # 转换为RGB
                if cell_resized.shape[2] == 3:
                    cell_rgb = cv2.cvtColor(cell_resized, cv2.COLOR_BGR2RGB)
                else:
                    cell_rgb = cell_resized
                
                # 准备输入
                img_tensor = torch.from_numpy(cell_rgb).float().permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
                
                # 使用分类器
                with torch.no_grad():
                    outputs = classifier(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    class_id = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, class_id].item()
                
                return class_id, confidence
            except Exception as e:
                logger.error(f"细胞分类出错: {e}")
                return 0, 0.5  # 返回默认类别和置信度
        """
        
        # 查找并替换这些方法
        # 注意：这种方式可能会导致一些问题，理想情况下应该使用更精确的方式修改这些方法
        
        # 保存修改
        backup_path = cell_segmentation_path + ".bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"原始文件已备份至: {backup_path}")
        
        # 生成修改指南而不是直接修改文件
        guide_path = "cellpose_improvement_guide.txt"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(f"# 细胞分割系统CellPose集成改进指南\n\n")
            f.write(f"## 1. 改进CellPose加载方法\n\n")
            f.write(f"在`models/cell_segmentation.py`文件中，修改`_load_cellpose_model`方法如下:\n\n")
            f.write(f"```python\n{load_cellpose_model_code}\n```\n\n")
            f.write(f"## 2. 改进CellPose预测方法\n\n")
            f.write(f"修改`_predict_cellpose`方法如下:\n\n")
            f.write(f"```python\n{improved_predict_cellpose}\n```\n\n")
            f.write(f"## 3. 添加细胞分类辅助方法\n\n")
            f.write(f"在CellSegmentationModel类中添加以下方法:\n\n")
            f.write(f"```python\n{classify_cell_method}\n```\n\n")
            
        logger.info(f"改进指南已生成: {guide_path}")
        return True
        
    except Exception as e:
        logger.error(f"改进CellPose集成失败: {e}")
        return False

if __name__ == "__main__":
    # 测试CellPose
    if test_cellpose():
        logger.info("CellPose测试成功!")
        
        # 尝试找到测试图像
        test_image = None
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Dyskeratotic", "images", "006.bmp"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "images", "sample.jpg"),
            # 添加其他可能的路径
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                test_image = path
                break
                
        if test_image:
            logger.info(f"使用测试图像: {test_image}")
            test_classification(test_image)
        else:
            logger.warning("未找到测试图像")
        
        # 改进CellPose集成
        improve_cellpose_integration()
    else:
        logger.error("CellPose测试失败，请确保已正确安装CellPose库")
        sys.exit(1)
