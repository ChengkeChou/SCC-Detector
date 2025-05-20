import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QFrame
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt6.QtCore import Qt
from ultralytics import YOLO
import cv2
import numpy as np

# --- 配置 ---
# 尝试自动定位最新的 best.pt 模型文件
# 您可能需要根据实际的训练输出路径进行调整
DEFAULT_MODEL_BASE_PATH = r"f:\SSC\CellSegmentRuns" 
# 如果上面的路径找不到模型，或者您想指定一个特定的模型，请取消注释并修改下面这行
# SPECIFIC_MODEL_PATH = r"f:\SSC\CellSegmentRuns\train_exp\weights\best.pt" 
SPECIFIC_MODEL_PATH = None  # 如果未指定特定模型，则将其设置为None

def find_latest_best_model(base_path):
    """查找最新的 best.pt 模型文件"""
    latest_run_dir = None
    latest_mtime = 0

    if not os.path.exists(base_path):
        return None

    for run_dir_name in os.listdir(base_path):
        run_dir_path = os.path.join(base_path, run_dir_name)
        if os.path.isdir(run_dir_path):
            weights_dir = os.path.join(run_dir_path, "weights")
            best_pt_path = os.path.join(weights_dir, "best.pt")
            if os.path.exists(best_pt_path):
                mtime = os.path.getmtime(best_pt_path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_run_dir = best_pt_path
    return latest_run_dir

class InferenceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image_path = None
        self.processed_image = None
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0; /* Light gray background */
                font-family: 'Segoe UI', Arial, sans-serif; /* Modern font */
            }
            QLabel {
                font-size: 14px;
                color: #333; /* Dark gray text */
            }
            QPushButton {
                background-color: #0078d4; /* Blue */
                color: white;
                border: none;
                padding: 10px 15px;
                font-size: 14px;
                border-radius: 5px; /* Rounded corners */
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #005a9e; /* Darker blue on hover */
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            #ImageLabel { /* Specific styling for image labels */
                border: 2px dashed #0078d4; /* Blue dashed border */
                border-radius: 5px;
                background-color: white; /* White background for image area */
            }
            #TitleLabel {
                font-size: 20px;
                font-weight: bold;
                color: #0078d4; /* Blue title */
                padding-bottom: 10px;
            }
            #InfoLabel {
                font-size: 12px;
                color: #555;
            }
        """)

        # 尝试加载模型
        try:
            if SPECIFIC_MODEL_PATH is not None and os.path.exists(SPECIFIC_MODEL_PATH):
                model_path_to_load = SPECIFIC_MODEL_PATH
                print(f"使用指定模型: {model_path_to_load}")
                print(f"使用指定模型: {model_path_to_load}")
            else:
                print(f"在 {DEFAULT_MODEL_BASE_PATH} 中搜索最新模型...")
                model_path_to_load = find_latest_best_model(DEFAULT_MODEL_BASE_PATH)
                if model_path_to_load:
                    print(f"自动找到最新模型: {model_path_to_load}")
                else:
                    print(f"未在 {DEFAULT_MODEL_BASE_PATH} 中找到 best.pt。请检查路径或指定 SPECIFIC_MODEL_PATH。")
            
            if model_path_to_load:
                self.model = YOLO(model_path_to_load)
                print(f"模型 '{model_path_to_load}' 加载成功。")
            else:
                 QMessageBox.warning(self, "模型加载失败", f"无法找到或加载模型。请检查脚本中的 DEFAULT_MODEL_BASE_PATH 或 SPECIFIC_MODEL_PATH 设置。")
                 print("模型加载失败。UI 将继续运行，但检测功能将不可用，直到模型被正确指定。")

        except Exception as e:
            QMessageBox.critical(self, "模型加载错误", f"加载模型时发生错误: {e}")
            print(f"加载模型时发生错误: {e}")

        self.initUI()

    def initUI(self):
        self.setWindowTitle('YOLOv8 细胞分割推理')
        self.setGeometry(100, 100, 1280, 720) # Increased size slightly

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20) # Add margins around the main layout
        main_layout.setSpacing(15) # Add spacing between widgets

        # Title
        title_label = QLabel("细胞分割与检测")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        self.load_image_btn = QPushButton(' 加载图片')
        # self.load_image_btn.setIcon(QIcon.fromTheme("document-open")) # Example: Needs icon theme or path
        self.load_image_btn.clicked.connect(self.loadImage)
        
        self.detect_btn = QPushButton(' 检测细胞')
        # self.detect_btn.setIcon(QIcon.fromTheme("system-search")) # Example
        self.detect_btn.clicked.connect(self.detectCells)
        self.detect_btn.setEnabled(False)

        button_layout.addStretch() # Push buttons to the center or one side
        button_layout.addWidget(self.load_image_btn)
        button_layout.addWidget(self.detect_btn)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator1)

        # 图片显示区域
        image_display_layout = QHBoxLayout()
        image_display_layout.setSpacing(20)

        # Original Image Area
        original_image_container = QVBoxLayout()
        original_title = QLabel("原始图片")
        original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label = QLabel('\\n\\n请点击 "加载图片" 选择一张图片\\n\\n')
        self.original_image_label.setObjectName("ImageLabel")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setMinimumSize(500, 400) # Adjusted size
        # self.original_image_label.setStyleSheet("border: 1px solid black;") # Replaced by #ImageLabel style
        original_image_container.addWidget(original_title)
        original_image_container.addWidget(self.original_image_label)
        
        # Processed Image Area
        processed_image_container = QVBoxLayout()
        processed_title = QLabel("检测结果")
        processed_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_image_label = QLabel('\\n\\n检测结果将在此处显示\\n\\n')
        self.processed_image_label.setObjectName("ImageLabel")
        self.processed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_image_label.setMinimumSize(500, 400) # Adjusted size
        # self.processed_image_label.setStyleSheet("border: 1px solid black;") # Replaced by #ImageLabel style
        processed_image_container.addWidget(processed_title)
        processed_image_container.addWidget(self.processed_image_label)

        image_display_layout.addLayout(original_image_container)
        image_display_layout.addLayout(processed_image_container)
        main_layout.addLayout(image_display_layout)

        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator2)
        
        # 模型信息
        model_info_text = f"当前模型: {self.model.ckpt_path if self.model and hasattr(self.model, 'ckpt_path') else '未加载或路径不可用'}"
        self.model_info_label = QLabel(model_info_text)
        self.model_info_label.setObjectName("InfoLabel")
        self.model_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.model_info_label)


        self.setLayout(main_layout)
        self.show()

    def loadImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", 
                                                  "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)")
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(self.current_image_path)
            self.original_image_label.setPixmap(pixmap.scaled(self.original_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.processed_image_label.setText('检测结果将显示在此处') # 重置结果显示
            self.detect_btn.setEnabled(True)
            self.processed_image = None

    def detectCells(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "无图片", "请先加载一张图片。")
            return

        if not self.model:
            QMessageBox.critical(self, "无模型", "模型未加载，无法进行检测。")
            return

        try:
            self.processed_image_label.setText("正在检测...")
            QApplication.processEvents() # 更新UI

            results = self.model.predict(source=self.current_image_path, save=False, stream=False) # stream=False for single image

            if results and len(results) > 0:
                # results[0].plot() 返回 BGR 格式的 numpy 数组
                annotated_image_bgr = results[0].plot() 
                
                # 将 BGR 转换为 RGB
                annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
                self.processed_image = annotated_image_rgb # 保存处理后的图像

                h, w, ch = annotated_image_rgb.shape
                bytes_per_line = ch * w
                q_image = QImage(annotated_image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888) # 修正 Format_RGB888
                
                pixmap = QPixmap.fromImage(q_image)
                self.processed_image_label.setPixmap(pixmap.scaled(self.processed_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                self.processed_image_label.setText("未检测到目标或处理出错。")
                QMessageBox.information(self, "检测完成", "未在图片中检测到任何目标，或者处理过程中出现问题。")

        except Exception as e:
            self.processed_image_label.setText("检测时发生错误。")
            QMessageBox.critical(self, "检测错误", f"进行细胞检测时发生错误: {e}")
            print(f"检测错误: {e}")
if __name__ == '__main__':
    # 确保 PyQt6 可以找到其插件
    try:
        from PyQt6 import sip
    except ImportError:
        pass # Python 3.9+ and PyQt6 might not need this explicitly

    app = QApplication(sys.argv)
    ex = InferenceApp()
    sys.exit(app.exec())
