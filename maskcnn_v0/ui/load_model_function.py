def load_model(self):
    """加载选定的模型"""
    import os
    import logging
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
    from cell_segmentation.inference import CellSegmentationInference  # Ensure this is the correct import path

    # Configure logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.ERROR)

    model_path = self.model_combo.currentText()
    
    # 如果选择了"浏览..."，打开文件对话框
    if model_path == "浏览...":
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pth)"
        )
        if not model_path:
            return
        
        # 添加到下拉列表
        self.model_combo.insertItem(0, model_path)
        self.model_combo.setCurrentIndex(0)
        
        # 检查模型文件是否存在
        # 检查模型文件是否存在
        model_path = self.model_combo.currentText()  # 确保 model_path 已定义
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", f"模型文件不存在: {model_path}")
            return
        
        try:
            # 更新状态
            self.statusbar.showMessage(f"正在加载模型: {model_path}")
            
            # 获取选定的模型类型
            model_type = self.model_type_combo.currentText()
            
            # 创建推理对象
            self.inference = CellSegmentationInference(
                model_path=model_path,
                model_type=model_type,
                num_classes=5,  # 假设有5个类别
                confidence_threshold=self.conf_threshold.value()
            )
            
            # 更新UI状态
            self.run_inference_btn.setEnabled(True)
            self.statusbar.showMessage(f"模型加载成功: {model_path} (类型: {model_type})")
            
            # 显示成功消息
            QMessageBox.information(self, "成功", f"{model_type}模型加载成功！")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            logger.error(f"加载模型失败: {e}")
            self.statusbar.showMessage("模型加载失败")
