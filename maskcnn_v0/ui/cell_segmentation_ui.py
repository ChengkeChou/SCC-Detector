"""
细胞实例分割系统用户界面
提供图形界面进行实例分割、分类和统计
"""

# 解决 OpenMP 多运行时问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import cv2
import numpy as np
import json
import torch
from pathlib import Path
import time
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, 
    QCheckBox, QTabWidget, QWidget, QGroupBox, QFormLayout, QListWidget,
    QListWidgetItem, QProgressBar, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QFont, QIcon, QBrush, QLinearGradient
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QDir, QEvent

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid_cell_segmentation import HybridCellSegmentationModel
from inference import CellSegmentationInference

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ui.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 工作线程，用于后台处理以避免UI冻结
class WorkerThread(QThread):
    # 定义信号
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    complete_signal = pyqtSignal()
    
    def __init__(self, task_type, args=None):
        super().__init__()
        self.task_type = task_type  # 'single', 'batch', 'preprocess', 'train'
        self.args = args or {}
        
    def run(self):
        try:
            if self.task_type == 'single':
                # 处理单张图像
                inference = self.args.get('inference')
                image_path = self.args.get('image_path')
                output_dir = self.args.get('output_dir')
                
                result = inference.process_image(image_path, output_dir)
                if result:
                    self.result_signal.emit(result)
                
            elif self.task_type == 'batch':
                # 批量处理图像
                inference = self.args.get('inference')
                image_dir = self.args.get('image_dir')
                output_dir = self.args.get('output_dir')
                
                # 获取所有图像路径
                image_paths = []
                for ext in ('.bmp', '.jpg', '.png'):
                    image_paths.extend(list(Path(image_dir).glob(f"*{ext}")))
                
                if not image_paths:
                    self.error_signal.emit(f"在 {image_dir} 中未找到图像文件")
                    return
                
                total = len(image_paths)
                results = []
                
                for i, path in enumerate(image_paths):
                    result = inference.process_image(str(path), output_dir)
                    if result:
                        results.append(result)
                        self.result_signal.emit(result)
                    
                    # 更新进度
                    progress = int((i + 1) / total * 100)
                    self.progress_signal.emit(progress)
                
                # 完成信号
                self.complete_signal.emit()
                
            elif self.task_type == 'preprocess':
                # 预处理数据
                from utils.dat_to_masks import convert_dataset
                
                input_dir = self.args.get('input_dir')
                output_dir = self.args.get('output_dir')
                format = self.args.get('format', 'yolo')
                num_workers = self.args.get('num_workers', 4)
                
                try:
                    convert_dataset(input_dir, output_dir, format, num_workers)
                    self.complete_signal.emit()
                except Exception as e:
                    self.error_signal.emit(f"预处理失败: {str(e)}")
            
            elif self.task_type == 'train':
                # 训练模型
                try:
                    from models.hybrid_cell_segmentation import train_model
                    
                    data_dir = self.args.get('data_dir')
                    output_dir = self.args.get('output_dir')
                    num_epochs = self.args.get('num_epochs', 50)
                    batch_size = self.args.get('batch_size', 4)
                    learning_rate = self.args.get('learning_rate', 0.001)
                    resume = self.args.get('resume', False)
                    
                    # 开始训练
                    best_model_path = train_model(
                        data_dir,
                        output_dir,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        resume=resume
                    )
                    
                    # 发送结果
                    self.result_signal.emit({"best_model_path": best_model_path})
                    self.complete_signal.emit()
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.error_signal.emit(f"训练失败: {str(e)}")
            
        except Exception as e:
            logger.error(f"工作线程错误: {e}")
            self.error_signal.emit(f"操作失败: {str(e)}")

class CellSegmentationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("细胞实例分割系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化内部变量
        self.current_image_path = None
        self.current_result = None
        self.inference = None
        self.worker_thread = None
        
        # 创建UI组件
        self.init_ui()
        
        # 检查CUDA是否可用
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.statusbar.showMessage(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            self.statusbar.showMessage("CUDA不可用，将使用CPU")
        
        # 扫描可用的模型
        self.scan_models()
        
    def init_ui(self):
        # 创建主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # 创建推理选项卡
        inference_tab = QWidget()
        tab_widget.addTab(inference_tab, "推理")
        
        # 创建数据准备选项卡
        data_tab = QWidget()
        tab_widget.addTab(data_tab, "数据准备")
        
        # 创建训练选项卡
        training_tab = QWidget()
        tab_widget.addTab(training_tab, "训练")
        
        # 设置推理选项卡
        self.setup_inference_tab(inference_tab)
        
        # 设置数据准备选项卡
        self.setup_data_tab(data_tab)
        
        # 设置训练选项卡
        self.setup_training_tab(training_tab)
        
        # 创建状态栏
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("就绪")
        
    def setup_inference_tab(self, tab):
        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
          # 模型组
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout()
          # 模型类型选择
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["yolov8", "cellpose", "dino", "maskrcnn", "hybrid"])
        self.model_type_combo.setCurrentText("hybrid")  # 默认选择混合模型
        self.model_type_combo.setStyleSheet("QComboBox { padding: 3px; border: 1px solid #aaa; border-radius: 3px; }")
        model_layout.addRow("模型类型:", self.model_type_combo)
        
        # 模型选择
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("QComboBox { padding: 3px; border: 1px solid #aaa; border-radius: 3px; }")
        model_layout.addRow("选择模型:", self.model_combo)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setStyleSheet(
            "QPushButton {background-color: #4CAF50; color: white; border-radius: 4px; padding: 6px;}"
            "QPushButton:hover {background-color: #45a049;}"
            "QPushButton:pressed {background-color: #3d8b40;}"
        )
        model_layout.addRow("", self.load_model_btn)
        
        # 置信度阈值
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.1, 1.0)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.setValue(0.5)
        self.conf_threshold.setStyleSheet("QDoubleSpinBox { padding: 2px; border: 1px solid #aaa; border-radius: 3px; }")
        model_layout.addRow("置信度阈值:", self.conf_threshold)
        
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)
          # 图像组
        image_group = QGroupBox("图像")
        image_layout = QVBoxLayout()
        
        # 图像按钮
        self.open_image_btn = QPushButton("打开图像")
        self.open_image_btn.clicked.connect(self.open_image)
        self.open_image_btn.setStyleSheet(
            "QPushButton {background-color: #3498db; color: white; border-radius: 4px; padding: 6px;}"
            "QPushButton:hover {background-color: #2980b9;}"
            "QPushButton:pressed {background-color: #1c6ea4;}"
        )
        
        self.open_dir_btn = QPushButton("打开文件夹")
        self.open_dir_btn.clicked.connect(self.open_directory)
        self.open_dir_btn.setStyleSheet(
            "QPushButton {background-color: #3498db; color: white; border-radius: 4px; padding: 6px;}"
            "QPushButton:hover {background-color: #2980b9;}"
            "QPushButton:pressed {background-color: #1c6ea4;}"
        )
        
        # 运行按钮
        self.run_inference_btn = QPushButton("运行推理")
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.run_inference_btn.setEnabled(False)
        self.run_inference_btn.setStyleSheet(
            "QPushButton {background-color: #e74c3c; color: white; border-radius: 4px; padding: 6px;}"
            "QPushButton:hover {background-color: #c0392b;}"
            "QPushButton:pressed {background-color: #a93226;}"
            "QPushButton:disabled {background-color: #d0d0d0; color: #a0a0a0;}"
        )
          # 保存结果按钮
        self.save_result_btn = QPushButton("保存结果")
        self.save_result_btn.clicked.connect(self.save_result)
        self.save_result_btn.setEnabled(False)
        self.save_result_btn.setStyleSheet(
            "QPushButton {background-color: #f39c12; color: white; border-radius: 4px; padding: 6px;}"
            "QPushButton:hover {background-color: #d68910;}"
            "QPushButton:pressed {background-color: #b9770e;}"
            "QPushButton:disabled {background-color: #d0d0d0; color: #a0a0a0;}"
        )
        
        image_layout.addWidget(self.open_image_btn)
        image_layout.addWidget(self.open_dir_btn)
        image_layout.addWidget(self.run_inference_btn)
        image_layout.addWidget(self.save_result_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar {border: 1px solid #aaa; border-radius: 3px; text-align: center;}"
            "QProgressBar::chunk {background-color: #4CAF50; width: 10px;}"
        )
        image_layout.addWidget(self.progress_bar)
        
        image_group.setLayout(image_layout)
        control_layout.addWidget(image_group)
        
        # 结果组
        results_group = QGroupBox("统计结果")
        results_layout = QVBoxLayout()
        
        # 结果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["类别", "数量"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        control_layout.addWidget(results_group)
        
        # 添加控制面板到分割器
        splitter.addWidget(control_panel)
          # 右侧图像显示
        image_display = QWidget()
        image_layout = QVBoxLayout(image_display)
        
        # 原始图像和结果图像标签
        self.original_image_label = QLabel("原始图像")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 300)
        self.original_image_label.setStyleSheet(
            "QLabel {border: 2px solid #3498db; border-radius: 6px; background-color: #f0f0f0; font-size: 14px;}"
        )
        
        self.result_image_label = QLabel("结果图像")
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setMinimumSize(400, 300)
        self.result_image_label.setStyleSheet(
            "QLabel {border: 2px solid #e74c3c; border-radius: 6px; background-color: #f0f0f0; font-size: 14px;}"
        )
        
        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.result_image_label)
        
        # 添加图像显示到分割器
        splitter.addWidget(image_display)
        
        # 设置分割器比例
        splitter.setSizes([300, 800])
        
        # 将分割器添加到主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        tab.setLayout(main_layout)
        
    def setup_data_tab(self, tab):
        layout = QVBoxLayout()
        
        # 数据预处理组
        preprocess_group = QGroupBox("数据预处理")
        preprocess_layout = QFormLayout()
        
        # 输入目录
        self.input_dir_edit = QLabel("未选择")
        self.input_dir_btn = QPushButton("选择输入目录")
        self.input_dir_btn.clicked.connect(self.select_input_dir)
        preprocess_layout.addRow(self.input_dir_btn, self.input_dir_edit)
        
        # 输出目录
        self.output_dir_edit = QLabel("未选择")
        self.output_dir_btn = QPushButton("选择输出目录")
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        preprocess_layout.addRow(self.output_dir_btn, self.output_dir_edit)
        
        # 格式选择
        self.format_combo = QComboBox()
        self.format_combo.addItems(["yolo", "coco", "mask"])
        preprocess_layout.addRow("输出格式:", self.format_combo)
        
        # 线程数
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        preprocess_layout.addRow("工作线程数:", self.workers_spin)
        
        # 运行预处理按钮
        self.run_preprocess_btn = QPushButton("运行预处理")
        self.run_preprocess_btn.clicked.connect(self.run_preprocessing)
        preprocess_layout.addRow("", self.run_preprocess_btn)
        
        # 预处理进度条
        self.preprocess_progress = QProgressBar()
        self.preprocess_progress.setVisible(False)
        preprocess_layout.addRow("进度:", self.preprocess_progress)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # 数据统计
        stats_group = QGroupBox("数据集统计")
        stats_layout = QFormLayout()
        
        # 选择数据集目录
        self.dataset_dir_edit = QLabel("未选择")
        self.dataset_dir_btn = QPushButton("选择数据集")
        self.dataset_dir_btn.clicked.connect(self.select_dataset_dir)
        stats_layout.addRow(self.dataset_dir_btn, self.dataset_dir_edit)
        
        # 加载统计按钮
        self.load_stats_btn = QPushButton("加载统计信息")
        self.load_stats_btn.clicked.connect(self.load_dataset_stats)
        stats_layout.addRow("", self.load_stats_btn)
        
        # 统计表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["类别", "数量"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addRow(self.stats_table)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        tab.setLayout(layout)
        
    def setup_training_tab(self, tab):
        layout = QVBoxLayout()
        
        # 训练设置组
        training_group = QGroupBox("训练设置")
        training_layout = QFormLayout()
        
        # 数据集目录
        self.train_data_dir_edit = QLabel("未选择")
        self.train_data_dir_btn = QPushButton("选择数据集目录")
        self.train_data_dir_btn.clicked.connect(self.select_train_data_dir)
        training_layout.addRow(self.train_data_dir_btn, self.train_data_dir_edit)
        
        # 输出目录
        self.train_output_dir_edit = QLabel("未选择")
        self.train_output_dir_btn = QPushButton("选择输出目录")
        self.train_output_dir_btn.clicked.connect(self.select_train_output_dir)
        training_layout.addRow(self.train_output_dir_btn, self.train_output_dir_edit)
        
        # 训练参数
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(50)
        training_layout.addRow("训练周期:", self.epochs_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        training_layout.addRow("批量大小:", self.batch_size_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        training_layout.addRow("学习率:", self.lr_spin)
        
        self.resume_check = QCheckBox("从检查点恢复")
        training_layout.addRow("", self.resume_check)
        
        # 开始训练按钮
        self.start_training_btn = QPushButton("开始训练")
        self.start_training_btn.clicked.connect(self.start_training)
        training_layout.addRow("", self.start_training_btn)
        
        # 训练进度
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        training_layout.addRow("进度:", self.training_progress)
        
        # 训练日志
        self.training_log = QListWidget()
        training_layout.addRow("训练日志:", self.training_log)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        tab.setLayout(layout)
    
    def scan_models(self):
        """扫描可用的模型"""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "checkpoints")
        
        # 如果目录不存在，尝试创建
        if not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir)
            except Exception as e:
                logger.error(f"创建模型目录失败: {e}")
                
        # 默认推荐路径
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        if os.path.exists(output_dir):
            # 查找best_model.pth
            best_model = os.path.join(output_dir, "best_model.pth")
            if os.path.exists(best_model):
                self.model_combo.addItem(best_model)
        
        # 扫描模型目录
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            for model_file in model_files:
                self.model_combo.addItem(os.path.join(models_dir, model_file))                
        # 添加一个自定义选项
        self.model_combo.addItem("浏览...")
    
    def load_model(self):
        """加载选定的模型"""
        model_path = self.model_combo.currentText()        # 如果选择了"浏览..."，打开文件对话框
        if model_path == "浏览...":
            initial_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            model_path, _ = QFileDialog.getOpenFileName(
                self, "选择模型文件", initial_dir, "PyTorch模型 (*.pth);;所有文件 (*)"
            )
            if not model_path:
                return
            
            # 添加到下拉列表
            self.model_combo.insertItem(0, model_path)
            self.model_combo.setCurrentIndex(0)
        
        # 检查模型文件是否存在
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
    
    def open_image(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.bmp *.jpg *.png)"
        )
        
        if not file_path:
            return
            
        try:
            # 加载图像
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "错误", f"无法读取图像: {file_path}")
                return
                
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 显示原始图像
            self.show_image(image, self.original_image_label)
            
            # 存储当前图像路径
            self.current_image_path = file_path
            
            # 清除结果
            self.result_image_label.clear()
            self.result_image_label.setText("结果图像")
            self.current_result = None
            
            # 更新UI状态
            if self.inference:
                self.run_inference_btn.setEnabled(True)
            
            self.statusbar.showMessage(f"已加载图像: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开图像失败: {str(e)}")
            logger.error(f"打开图像失败: {e}")
    
    def open_directory(self):
        """打开图像目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择图像目录")
        
        if not dir_path:
            return
            
        try:
            # 检查目录中是否有图像文件
            image_count = 0
            for ext in ['.bmp', '.jpg', '.png']:
                image_count += len(list(Path(dir_path).glob(f"*{ext}")))
                
            if image_count == 0:
                QMessageBox.warning(self, "警告", f"所选目录中没有支持的图像文件")
                return
                
            # 提示用户确认
            reply = QMessageBox.question(
                self, "批量处理",
                f"找到{image_count}个图像文件。确定要处理这个目录吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 如果未加载模型，提示用户
                if not self.inference:
                    QMessageBox.warning(self, "警告", "请先加载模型")
                    return
                    
                # 询问输出目录
                output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
                if not output_dir:
                    return
                
                # 开始批处理
                self.statusbar.showMessage(f"开始批量处理 {dir_path}")
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                # 创建工作线程
                self.worker_thread = WorkerThread(
                    task_type='batch',
                    args={
                        'inference': self.inference,
                        'image_dir': dir_path,
                        'output_dir': output_dir
                    }
                )
                
                # 连接信号
                self.worker_thread.progress_signal.connect(self.update_progress)
                self.worker_thread.result_signal.connect(self.handle_batch_result)
                self.worker_thread.error_signal.connect(self.show_error)
                self.worker_thread.complete_signal.connect(self.batch_complete)
                
                # 启动线程
                self.worker_thread.start()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开目录失败: {str(e)}")
            logger.error(f"打开目录失败: {e}")
    
    def run_inference(self):
        """运行推理"""
        if not self.inference:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择图像")
            return
            
        try:
            # 更新置信度阈值
            self.inference.confidence_threshold = self.conf_threshold.value()
            
            # 更新状态
            self.statusbar.showMessage("正在进行推理...")
            
            # 创建工作线程
            self.worker_thread = WorkerThread(
                task_type='single',
                args={
                    'inference': self.inference,
                    'image_path': self.current_image_path,
                    'output_dir': None  # 不需要保存到文件
                }
            )
            
            # 连接信号
            self.worker_thread.result_signal.connect(self.handle_result)
            self.worker_thread.error_signal.connect(self.show_error)
            
            # 启动线程
            self.worker_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理失败: {str(e)}")
            logger.error(f"推理失败: {e}")
            self.statusbar.showMessage("推理失败")
    
    def handle_result(self, result):
        """处理推理结果"""
        if not result:
            return
            
        # 存储当前结果
        self.current_result = result
        
        # 从原始图像和预测创建可视化结果
        original_image = cv2.imread(self.current_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        result_image, class_counts = self.inference.visualize_results(
            original_image, result["prediction"]
        )
        
        # 显示结果图像
        self.show_image(result_image, self.result_image_label)
        
        # 更新结果表格
        self.update_results_table(class_counts)
        
        # 更新UI状态
        self.save_result_btn.setEnabled(True)
        
        # 更新状态栏
        total_cells = sum(class_counts.values())
        inference_time = result["inference_time"]
        self.statusbar.showMessage(f"推理完成: 检测到{total_cells}个细胞，用时{inference_time:.2f}秒")
    
    def handle_batch_result(self, result):
        """处理批处理中的单个结果"""
        # 可以在这里更新批处理结果列表或其他UI元素
        pass
    
    def batch_complete(self):
        """批处理完成"""
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "完成", "批量处理已完成")
        self.statusbar.showMessage("批量处理完成")
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def show_error(self, message):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", message)
        self.statusbar.showMessage("发生错误")
    
    def save_result(self):
        """保存结果"""
        if not self.current_result:
            return
            
        try:
            # 获取保存路径
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存结果", "", "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)"
            )
            
            if not save_path:
                return
                
            # 重新生成可视化结果
            original_image = cv2.imread(self.current_image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            result_image, _ = self.inference.visualize_results(
                original_image, self.current_result["prediction"]
            )
            
            # 保存图像
            cv2.imwrite(save_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            
            self.statusbar.showMessage(f"结果已保存至: {save_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存结果失败: {str(e)}")
            logger.error(f"保存结果失败: {e}")
    
    def update_results_table(self, class_counts):
        """更新结果表格"""
        # 清空表格
        self.results_table.setRowCount(0)
        
        # 添加总数
        total = sum(class_counts.values())
        self.results_table.insertRow(0)
        self.results_table.setItem(0, 0, QTableWidgetItem("总计"))
        self.results_table.setItem(0, 1, QTableWidgetItem(str(total)))
        
        # 添加每个类别的数量
        row = 1
        for class_name, count in sorted(class_counts.items()):
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(class_name))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(count)))
            row += 1
    
    def show_image(self, image, label):
        """在标签中显示图像"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 计算合适的缩放比例
        label_size = label.size()
        scale_w = label_size.width() / width
        scale_h = label_size.height() / height
        scale = min(scale_w, scale_h)
        
        # 缩放图像
        new_width = int(width * scale)
        new_height = int(height * scale)
        pixmap = QPixmap.fromImage(q_image).scaled(new_width, new_height, Qt.KeepAspectRatio)
        
        label.setPixmap(pixmap)
    
    def select_input_dir(self):
        """选择输入目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.input_dir_edit.setText(dir_path)
    
    def select_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def select_dataset_dir(self):
        """选择数据集目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if dir_path:
            self.dataset_dir_edit.setText(dir_path)
    
    def select_train_data_dir(self):
        """选择训练数据集目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择训练数据集目录")
        if dir_path:
            self.train_data_dir_edit.setText(dir_path)
    
    def select_train_output_dir(self):
        """选择训练输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择训练输出目录")
        if dir_path:
            self.train_output_dir_edit.setText(dir_path)
    
    def run_preprocessing(self):
        """运行数据预处理"""
        input_dir = self.input_dir_edit.text()
        output_dir = self.output_dir_edit.text()
        
        if input_dir == "未选择" or output_dir == "未选择":
            QMessageBox.warning(self, "警告", "请选择输入和输出目录")
            return
            
        # 提示用户确认
        reply = QMessageBox.question(
            self, "数据预处理",
            f"确定要处理目录{input_dir}中的数据吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 设置UI状态
            self.preprocess_progress.setVisible(True)
            self.run_preprocess_btn.setEnabled(False)
            self.statusbar.showMessage("正在预处理数据...")
            
            # 创建工作线程
            self.worker_thread = WorkerThread(
                task_type='preprocess',
                args={
                    'input_dir': input_dir,
                    'output_dir': output_dir,
                    'format': self.format_combo.currentText(),
                    'num_workers': self.workers_spin.value()
                }
            )
            
            # 连接信号
            self.worker_thread.progress_signal.connect(lambda v: self.preprocess_progress.setValue(v))
            self.worker_thread.error_signal.connect(self.show_error)
            self.worker_thread.complete_signal.connect(self.preprocess_complete)
            
            # 启动线程
            self.worker_thread.start()
    
    def preprocess_complete(self):
        """预处理完成"""
        self.preprocess_progress.setVisible(False)
        self.run_preprocess_btn.setEnabled(True)
        QMessageBox.information(self, "完成", "数据预处理已完成")
        self.statusbar.showMessage("数据预处理完成")
    
    def load_dataset_stats(self):
        """加载数据集统计信息"""
        dataset_dir = self.dataset_dir_edit.text()
        
        if dataset_dir == "未选择":
            QMessageBox.warning(self, "警告", "请选择数据集目录")
            return
            
        try:
            # 加载元数据
            metadata_path = os.path.join(dataset_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                QMessageBox.warning(self, "警告", f"未找到元数据文件: {metadata_path}")
                return
                
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
            # 计算统计信息
            class_counts = {}
            for img in metadata.get("images", []):
                for instance in img.get("instances", []):
                    class_name = instance.get("class_name", "Unknown")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # 更新表格
            self.stats_table.setRowCount(0)
            
            # 添加总数
            total = sum(class_counts.values())
            self.stats_table.insertRow(0)
            self.stats_table.setItem(0, 0, QTableWidgetItem("总计"))
            self.stats_table.setItem(0, 1, QTableWidgetItem(str(total)))
            
            # 添加每个类别的数量
            row = 1
            for class_name, count in sorted(class_counts.items()):
                self.stats_table.insertRow(row)
                self.stats_table.setItem(row, 0, QTableWidgetItem(class_name))
                self.stats_table.setItem(row, 1, QTableWidgetItem(str(count)))
                row += 1
                
            self.statusbar.showMessage(f"加载了{len(metadata.get('images', []))}个图像的统计信息")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载统计信息失败: {str(e)}")
            logger.error(f"加载统计信息失败: {e}")
    
    def start_training(self):
        """开始训练"""
        data_dir = self.train_data_dir_edit.text()
        output_dir = self.train_output_dir_edit.text()
        
        if data_dir == "未选择" or output_dir == "未选择":
            QMessageBox.warning(self, "警告", "请选择数据集和输出目录")
            return
            
        # 检查数据集是否有效
        metadata_path = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            QMessageBox.warning(self, "警告", f"无效的数据集: 未找到元数据文件")
            return
            
        # 提示用户确认
        reply = QMessageBox.question(
            self, "训练",
            f"确定要使用以下参数开始训练吗？\n\n"
            f"数据集: {data_dir}\n"
            f"输出目录: {output_dir}\n"
            f"训练周期: {self.epochs_spin.value()}\n"
            f"批量大小: {self.batch_size_spin.value()}\n"
            f"学习率: {self.lr_spin.value()}\n"
            f"从检查点恢复: {'是' if self.resume_check.isChecked() else '否'}\n\n"
            f"注意: 训练过程可能需要较长时间。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 设置UI状态
            self.training_progress.setVisible(True)
            self.start_training_btn.setEnabled(False)
            self.training_log.clear()
            self.statusbar.showMessage("正在准备训练...")
            
            # 添加日志处理器
            class QListWidgetHandler(logging.Handler):
                def __init__(self, list_widget):
                    super().__init__()
                    self.list_widget = list_widget
                    
                def emit(self, record):
                    msg = self.format(record)
                    # 在UI线程中更新
                    QApplication.instance().postEvent(
                        self.list_widget,
                        LogEvent(msg)
                    )
            
            # 自定义事件类型
            class LogEvent(QEvent):
                EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
                
                def __init__(self, message):
                    super().__init__(self.EVENT_TYPE)
                    self.message = message
                    
            # 重写list widget的事件处理
            self.training_log.event = lambda event: self.handle_log_event(event) \
                if hasattr(event, 'EVENT_TYPE') and event.type() == LogEvent.EVENT_TYPE \
                else QListWidget.event(self.training_log, event)
                
            self.handle_log_event = lambda event: self.training_log.addItem(event.message)
            
            # 添加日志处理器
            list_handler = QListWidgetHandler(self.training_log)
            list_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logging.getLogger().addHandler(list_handler)
            
            # 创建工作线程
            self.worker_thread = WorkerThread(
                task_type='train',
                args={
                    'data_dir': data_dir,
                    'output_dir': output_dir,
                    'num_epochs': self.epochs_spin.value(),
                    'batch_size': self.batch_size_spin.value(),
                    'learning_rate': self.lr_spin.value(),
                    'resume': self.resume_check.isChecked()
                }
            )
            
            # 连接信号
            self.worker_thread.result_signal.connect(self.handle_training_result)
            self.worker_thread.error_signal.connect(self.show_error)
            self.worker_thread.complete_signal.connect(self.training_complete)
            
            # 启动线程
            self.worker_thread.start()
            
            # 添加日志
            self.training_log.addItem("训练已开始...")
            
    def handle_training_result(self, result):
        """处理训练结果"""
        best_model_path = result.get("best_model_path")
        if best_model_path:
            # 更新模型下拉列表
            self.model_combo.insertItem(0, best_model_path)
            self.model_combo.setCurrentIndex(0)
            
            # 添加日志
            self.training_log.addItem(f"最佳模型已保存: {best_model_path}")
    
    def training_complete(self):
        """训练完成"""
        self.training_progress.setVisible(False)
        self.start_training_btn.setEnabled(True)
        QMessageBox.information(self, "完成", "训练已完成")
        self.statusbar.showMessage("训练完成")
        
        # 添加日志
        self.training_log.addItem("训练完成!")

def main():
    app = QApplication(sys.argv)
    window = CellSegmentationUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
