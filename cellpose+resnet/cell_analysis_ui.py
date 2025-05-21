from __future__ import annotations # Added for forward type hint evaluation
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json # For reading class_to_idx.txt if it's simple JSON
import ast  # For safely evaluating string representation of dict from class_to_idx.txt
import argparse # To parse args.txt or define a similar structure

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QMessageBox, QLineEdit, QGroupBox, QRadioButton,
                             QSizePolicy, QSlider) # Added QSlider
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt

import cv2
import numpy as np
from PIL import Image

# Cellpose imports
from cellpose import models as cellpose_models
from cellpose import io as cellpose_io
# from cellpose import utils as cellpose_utils # If needed for outlines

# PyTorch imports for classifier
import torch
from torchvision import transforms
import torch.nn as nn # Added
from torchvision import models as torchvision_models # Changed from 'models' to 'torchvision_models' to avoid conflict

# We'll need the build_model function from train_cell_classifier.py
# For now, let's assume it will be copied or imported here.

# Actual build_model function from train_cell_classifier.py
def build_model(model_name, num_classes, pretrained=True):
    """Builds a specified model."""
    if model_name == 'resnet18':
        model = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = torchvision_models.resnet50(weights=torchvision_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = torchvision_models.efficientnet_b0(weights=torchvision_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b3':
        model = torchvision_models.efficientnet_b3(weights=torchvision_models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

# Helper function from run_cellpose_segmentation.py
def _get_scalar_diameter(diams_value, input_diameter_val=None):
    """Converts a diameter value (which could be float, int, ndarray, or None) to a scalar float."""
    scalar_val = None
    if isinstance(diams_value, np.ndarray):
        if diams_value.size > 0:
            scalar_val = float(np.mean(diams_value))
    elif isinstance(diams_value, (float, int, np.floating, np.integer)):
        scalar_val = float(diams_value)
    elif diams_value is None:
        pass
    else:
        try:
            scalar_val = float(diams_value)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert diams_value of type {type(diams_value)} to float. Will use fallback.")
            scalar_val = None

    if scalar_val is None or scalar_val <= 0:
        if input_diameter_val is not None and input_diameter_val > 0:
            scalar_val = float(input_diameter_val)
        else:
            scalar_val = 30.0 # Default fallback diameter
    return scalar_val

# Actual segment_image_and_get_instances function (adapted from run_cellpose_segmentation.py and placeholder)
def segment_image_and_get_instances(model: 'cellpose_models.CellposeModel' | 'cellpose_models.Cellpose',
                                   image_np: np.ndarray, 
                                   diameter: float | None = None, 
                                   channels: list | None = None,
                                   flow_threshold: float = 0.4, 
                                   cellprob_threshold: float = 0.0,
                                   tile_norm_blocksize: int = 0,
                                   show_visualization: bool = False):
    """
    Performs cell segmentation on a single image using a pre-initialized CellPose model
    and returns information about each detected cell instance.
    """
    if image_np is None:
        print("Error: Input image is None.")
        return [], 0.0

    if len(image_np.shape) > 2 and image_np.shape[-1] > 3 and image_np.shape[-1] != 4 :
        print(f"Warning: Image has unexpected shape {image_np.shape}. Expected 2D (H,W) or 3D (H,W,C) with C<=4.")
        if image_np.shape[-1] > 4: # Attempt to take first 3 channels if more than 4
            image_np = image_np[..., :3]
    
    normalize_params = {"tile_norm_blocksize": tile_norm_blocksize} if tile_norm_blocksize > 0 else True

    masks, flows, diams = model.eval(image_np, 
                                     diameter=diameter,
                                     channels=channels, # Uncommented: Use channels from UI
                                     flow_threshold=flow_threshold, 
                                     cellprob_threshold=cellprob_threshold,
                                     normalize=normalize_params,
                                     batch_size=8) 
    
    scalar_diams = _get_scalar_diameter(diams, diameter)

    instances = []
    unique_mask_ids = np.unique(masks)
    for mask_id in unique_mask_ids:
        if mask_id == 0:
            continue
        
        instance_mask_full = (masks == mask_id)
        
        rows, cols = np.where(instance_mask_full)
        if rows.size == 0 or cols.size == 0:
            continue
        ymin, ymax = np.min(rows), np.max(rows)
        xmin, xmax = np.min(cols), np.max(cols)
        bbox = [xmin, ymin, xmax, ymax]

        cropped_image_data = image_np[ymin:ymax+1, xmin:xmax+1].copy()

        if cropped_image_data.ndim == 2:
            cropped_image_data = cv2.cvtColor(cropped_image_data, cv2.COLOR_GRAY2RGB)
        elif cropped_image_data.ndim == 3 and cropped_image_data.shape[-1] == 1:
             cropped_image_data = cv2.cvtColor(cropped_image_data, cv2.COLOR_GRAY2RGB)
        elif cropped_image_data.ndim == 3 and cropped_image_data.shape[-1] == 4: # Handle RGBA from original
             cropped_image_data = cv2.cvtColor(cropped_image_data, cv2.COLOR_RGBA2RGB)

        instances.append({
            'cropped_image': cropped_image_data,
            'bbox': bbox,
            'mask_full': instance_mask_full 
        })
    
    if show_visualization and masks is not None:
        try:
            from cellpose import plot # Dynamic import for plotting
            img_for_plot = image_np.copy()
            if img_for_plot.ndim == 3 and img_for_plot.shape[-1] > 3:
                img_for_plot = img_for_plot[...,:3]
            mask_overlay = plot.mask_overlay(img_for_plot, masks)
            # In a UI, direct plt.show() is problematic.
            # This visualization part is more for command-line debugging.
            # For UI, the main `analyze_cells` method handles drawing on QPixmap.
            print("Cellpose visualization requested (debug: overlay created, not shown directly in UI by this function).")
            # Example: cv2.imshow("Debug Overlay", cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyWindow("Debug Overlay")
        except ImportError:
            print("Plotting libraries (e.g., matplotlib) not available for direct visualization here.")
        except Exception as e:
            print(f"Error during visualization attempt in segment_image_and_get_instances: {e}")

    return instances, scalar_diams

DEFAULT_CELLPOSE_MODEL_TYPE = 'cyto'
DEFAULT_CLASSIFIER_IMG_SIZE = 224 # Fallback, should be read from args.txt

# Define some colors for classes (add more if you have more classes)
CLASS_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
]


class CellAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.original_qimage = None
        self.original_cv_image = None
        self.annotated_cv_image = None

        self.cellpose_model_type = DEFAULT_CELLPOSE_MODEL_TYPE
        self.cellpose_custom_path = None
        self.cellpose_model = None
        # Cellpose analysis parameters
        self.cellpose_diameter = 0.0 # 0.0 for auto-detect
        self.cellpose_flow_threshold = 0.4
        self.cellpose_cellprob_threshold = 0.0
        self.cellpose_channels = [0, 0] # Default for grayscale cyto

        self.classifier_model_path = None
        self.classifier_args_path = None
        self.classifier_class_to_idx_path = None
        
        self.classifier_model = None
        self.classifier_class_names = []
        self.classifier_idx_to_class = {}
        self.classifier_transform = None
        self.classifier_img_size = DEFAULT_CLASSIFIER_IMG_SIZE
        self.classifier_model_name = "resnet18" # Default, override from args.txt
        self.classifier_pretrained = True # Default, override from args.txt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.initUI()
        self._try_load_latest_classifier_defaults() # Attempt to load defaults after UI is initialized

    def initUI(self):
        self.setWindowTitle('Cell Segmentation and Classification UI')
        self.setGeometry(50, 50, 1600, 950) # Adjusted size for new controls

        main_layout = QVBoxLayout()

        # --- Top: Model Configuration ---
        top_config_layout = QHBoxLayout()
        
        # Left side: Model Selection (Cellpose & Classifier)
        model_selection_layout = QVBoxLayout()

        # Cellpose Config
        cp_group = QGroupBox("Cellpose Segmentation Model")
        cp_layout = QVBoxLayout()
        self.cp_radio_cyto = QRadioButton("Cytoplasm (cyto)")
        self.cp_radio_cyto.setChecked(self.cellpose_model_type == 'cyto')
        self.cp_radio_cyto.toggled.connect(lambda: self.set_cellpose_model_type('cyto'))
        self.cp_radio_nuclei = QRadioButton("Nuclei (nuclei)")
        self.cp_radio_nuclei.toggled.connect(lambda: self.set_cellpose_model_type('nuclei'))
        
        cp_custom_layout = QHBoxLayout()
        self.cp_radio_custom = QRadioButton("Custom Path:")
        self.cp_radio_custom.toggled.connect(lambda: self.set_cellpose_model_type('custom'))
        self.cp_custom_path_edit = QLineEdit()
        self.cp_custom_path_edit.setPlaceholderText("Path to Cellpose model file")
        self.cp_custom_path_btn = QPushButton("Browse")
        self.cp_custom_path_btn.clicked.connect(self.browse_cellpose_custom_path)
        cp_custom_layout.addWidget(self.cp_radio_custom)
        cp_custom_layout.addWidget(self.cp_custom_path_edit)
        cp_custom_layout.addWidget(self.cp_custom_path_btn)

        cp_layout.addWidget(self.cp_radio_cyto)
        cp_layout.addWidget(self.cp_radio_nuclei)
        cp_layout.addLayout(cp_custom_layout)
        cp_group.setLayout(cp_layout)
        model_selection_layout.addWidget(cp_group)

        # Classifier Config
        clf_group = QGroupBox("Cell Classifier Model")
        clf_layout = QVBoxLayout()
        
        self.clf_model_path_edit = QLineEdit()
        self.clf_model_path_edit.setPlaceholderText("Path to classifier model (.pth)")
        clf_model_btn = QPushButton("Browse PTH")
        clf_model_btn.clicked.connect(self.browse_classifier_model_path)
        clf_path_layout1 = QHBoxLayout()
        clf_path_layout1.addWidget(QLabel("Model:"))
        clf_path_layout1.addWidget(self.clf_model_path_edit)
        clf_path_layout1.addWidget(clf_model_btn)
        clf_layout.addLayout(clf_path_layout1)

        self.clf_args_path_edit = QLineEdit()
        self.clf_args_path_edit.setPlaceholderText("Path to training args.txt")
        clf_args_btn = QPushButton("Browse args.txt")
        clf_args_btn.clicked.connect(self.browse_classifier_args_path)
        clf_path_layout2 = QHBoxLayout()
        clf_path_layout2.addWidget(QLabel("Args:"))
        clf_path_layout2.addWidget(self.clf_args_path_edit)
        clf_path_layout2.addWidget(clf_args_btn)
        clf_layout.addLayout(clf_path_layout2)
        
        self.clf_class_map_path_edit = QLineEdit()
        self.clf_class_map_path_edit.setPlaceholderText("Path to class_to_idx.txt")
        clf_map_btn = QPushButton("Browse class_to_idx.txt")
        clf_map_btn.clicked.connect(self.browse_classifier_class_map_path)
        clf_path_layout3 = QHBoxLayout()
        clf_path_layout3.addWidget(QLabel("Map:"))
        clf_path_layout3.addWidget(self.clf_class_map_path_edit)
        clf_path_layout3.addWidget(clf_map_btn)
        clf_layout.addLayout(clf_path_layout3)

        self.load_models_btn = QPushButton("Load All Models")
        self.load_models_btn.clicked.connect(self.load_all_models)
        clf_layout.addWidget(self.load_models_btn)
        clf_group.setLayout(clf_layout)
        model_selection_layout.addWidget(clf_group)
        
        top_config_layout.addLayout(model_selection_layout)

        # Right side: Cellpose Parameters
        cp_params_group = QGroupBox("Cellpose Parameters")
        cp_params_layout = QVBoxLayout()

        # Diameter
        diameter_layout = QHBoxLayout()
        diameter_layout.addWidget(QLabel("Diameter (0 for auto):"))
        self.cp_diameter_edit = QLineEdit(str(self.cellpose_diameter))
        self.cp_diameter_edit.setPlaceholderText("e.g., 30.0 or 0")
        diameter_layout.addWidget(self.cp_diameter_edit)
        cp_params_layout.addLayout(diameter_layout)

        # Flow Threshold
        flow_layout = QHBoxLayout()
        flow_layout.addWidget(QLabel("Flow Threshold:"))
        self.cp_flow_slider = QSlider(Qt.Orientation.Horizontal)
        self.cp_flow_slider.setMinimum(0)
        self.cp_flow_slider.setMaximum(200) # Represents 0.0 to 2.0
        self.cp_flow_slider.setValue(int(self.cellpose_flow_threshold * 100))
        self.cp_flow_value_label = QLabel(f"{self.cellpose_flow_threshold:.2f}")
        self.cp_flow_slider.valueChanged.connect(self._update_flow_threshold)
        flow_layout.addWidget(self.cp_flow_slider)
        flow_layout.addWidget(self.cp_flow_value_label)
        cp_params_layout.addLayout(flow_layout)

        # Cellprob Threshold
        cellprob_layout = QHBoxLayout()
        cellprob_layout.addWidget(QLabel("Cellprob Threshold:"))
        self.cp_cellprob_slider = QSlider(Qt.Orientation.Horizontal)
        self.cp_cellprob_slider.setMinimum(-600) # Represents -6.0
        self.cp_cellprob_slider.setMaximum(600)  # Represents 6.0
        self.cp_cellprob_slider.setValue(int(self.cellpose_cellprob_threshold * 100))
        self.cp_cellprob_value_label = QLabel(f"{self.cellpose_cellprob_threshold:.2f}")
        self.cp_cellprob_slider.valueChanged.connect(self._update_cellprob_threshold)
        cellprob_layout.addWidget(self.cp_cellprob_slider)
        cellprob_layout.addWidget(self.cp_cellprob_value_label)
        cp_params_layout.addLayout(cellprob_layout)
        
        # Channels
        channels_layout = QHBoxLayout()
        channels_layout.addWidget(QLabel("Channels (Cyto, Nuclei):"))
        self.cp_channel1_edit = QLineEdit(str(self.cellpose_channels[0]))
        self.cp_channel1_edit.setPlaceholderText("Ch1 (Cyto)")
        self.cp_channel2_edit = QLineEdit(str(self.cellpose_channels[1]))
        self.cp_channel2_edit.setPlaceholderText("Ch2 (Nuclei, 0 if none)")
        channels_layout.addWidget(self.cp_channel1_edit)
        channels_layout.addWidget(self.cp_channel2_edit)
        cp_params_layout.addLayout(channels_layout)

        cp_params_group.setLayout(cp_params_layout)
        top_config_layout.addWidget(cp_params_group)
        
        main_layout.addLayout(top_config_layout)

        # --- Middle: Image Loading and Action ---
        action_layout = QHBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.analyze_btn = QPushButton("Analyze Cells")
        self.analyze_btn.clicked.connect(self.analyze_cells)
        self.analyze_btn.setEnabled(False) # Disabled until models and image are loaded
        action_layout.addWidget(self.load_image_btn)
        action_layout.addWidget(self.analyze_btn)
        main_layout.addLayout(action_layout)

        # --- Image Display Area ---
        image_display_layout = QHBoxLayout()
        self.original_image_label = QLabel("Original Image: Load an image to start.")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setMinimumSize(600, 400)
        self.original_image_label.setStyleSheet("border: 1px solid black; background-color: white;")
        
        self.processed_image_label = QLabel("Processed Image: Results will appear here.")
        self.processed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_image_label.setMinimumSize(600, 400)
        self.processed_image_label.setStyleSheet("border: 1px solid black; background-color: white;")
        
        image_display_layout.addWidget(self.original_image_label)
        image_display_layout.addWidget(self.processed_image_label)
        main_layout.addLayout(image_display_layout)

        # --- Bottom: Status Bar ---
        self.status_label = QLabel("Status: Ready. Please load models and an image.")
        self.status_label.setFixedHeight(30)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)
        self.show()

    def _update_flow_threshold(self, value):
        self.cellpose_flow_threshold = value / 100.0
        self.cp_flow_value_label.setText(f"{self.cellpose_flow_threshold:.2f}")

    def _update_cellprob_threshold(self, value):
        self.cellpose_cellprob_threshold = value / 100.0
        self.cp_cellprob_value_label.setText(f"{self.cellpose_cellprob_threshold:.2f}")

    def set_cellpose_model_type(self, type_name):
        if self.cp_radio_cyto.isChecked() and type_name == 'cyto':
            self.cellpose_model_type = 'cyto'
            self.cp_custom_path_edit.setEnabled(False)
        elif self.cp_radio_nuclei.isChecked() and type_name == 'nuclei':
            self.cellpose_model_type = 'nuclei'
            self.cp_custom_path_edit.setEnabled(False)
        elif self.cp_radio_custom.isChecked() and type_name == 'custom':
            self.cellpose_model_type = 'custom'
            self.cp_custom_path_edit.setEnabled(True)

    def browse_file(self, caption, filter_str):
        file_path, _ = QFileDialog.getOpenFileName(self, caption, "", filter_str)
        return file_path

    def browse_cellpose_custom_path(self):
        path = self.browse_file("Select Cellpose Model", "Model files (*.pth *.pt *.pb);;All files (*)")
        if path:
            self.cp_custom_path_edit.setText(path)
            self.cellpose_custom_path = path
            self.cp_radio_custom.setChecked(True) # Ensure custom is selected

    def browse_classifier_model_path(self):
        path = self.browse_file("Select Classifier Model", "PyTorch Model (*.pth)")
        if path:
            self.clf_model_path_edit.setText(path)
            self.classifier_model_path = path
            # Try to infer args and class_map paths
            base_dir = os.path.dirname(path)
            model_name_no_ext = os.path.splitext(os.path.basename(path))[0]
            
            # Common pattern: models/classifier_MODELNAME_TIMESTAMP/MODELNAME_best.pth
            # args.txt and class_to_idx.txt are in models/classifier_MODELNAME_TIMESTAMP/
            
            # Heuristic 1: files are in the same directory as the model
            potential_args_path = os.path.join(base_dir, "args.txt")
            potential_map_path = os.path.join(base_dir, "class_to_idx.txt")

            if os.path.exists(potential_args_path) and not self.clf_args_path_edit.text():
                self.clf_args_path_edit.setText(potential_args_path)
                self.classifier_args_path = potential_args_path
            if os.path.exists(potential_map_path) and not self.clf_class_map_path_edit.text():
                self.clf_class_map_path_edit.setText(potential_map_path)
                self.classifier_class_to_idx_path = potential_map_path


    def browse_classifier_args_path(self):
        path = self.browse_file("Select Classifier Args File", "Text files (*.txt)")
        if path:
            self.clf_args_path_edit.setText(path)
            self.classifier_args_path = path

    def browse_classifier_class_map_path(self):
        path = self.browse_file("Select Classifier Class Map File", "Text files (*.txt)")
        if path:
            self.clf_class_map_path_edit.setText(path)
            self.classifier_class_to_idx_path = path
            
    def _parse_args_txt(self, filepath):
        args_dict = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        args_dict[key.strip()] = value.strip()
            
            # Extract specific values, converting type as necessary
            self.classifier_img_size = int(args_dict.get('img_size', DEFAULT_CLASSIFIER_IMG_SIZE))
            self.classifier_model_name = args_dict.get('model_name', 'resnet18') # Default if not found
            # Add any other args you need, e.g., pretrained status
            self.classifier_pretrained = ast.literal_eval(args_dict.get('pretrained', 'True').capitalize())


        except Exception as e:
            QMessageBox.warning(self, "Args File Error", f"Could not parse args.txt: {e}. Using defaults.")
            self.classifier_img_size = DEFAULT_CLASSIFIER_IMG_SIZE
            self.classifier_model_name = "resnet18"


    def load_all_models(self):
        self.status_label.setText("Status: Loading models...")
        QApplication.processEvents()

        # Load Cellpose Model
        try:
            use_gpu = torch.cuda.is_available()
            model_to_load_identifier = None # Will be 'cyto', 'nuclei', or a file path

            if self.cellpose_model_type == 'custom':
                self.cellpose_custom_path = self.cp_custom_path_edit.text()
                if not self.cellpose_custom_path or not os.path.exists(self.cellpose_custom_path):
                    raise ValueError("Custom Cellpose model path is invalid or not set.")
                model_to_load_identifier = self.cellpose_custom_path
            else: # 'cyto' or 'nuclei'
                model_to_load_identifier = self.cellpose_model_type

            # Use CellposeModel for all cases.
            # Its pretrained_model argument can take a model type string (e.g., 'cyto', 'nuclei')
            # or a path to a custom model file.
            self.cellpose_model = cellpose_models.CellposeModel(gpu=use_gpu, pretrained_model=model_to_load_identifier)
            
            if self.cellpose_model_type == 'custom':
                self.status_label.setText(f"Status: Loaded custom Cellpose model: {os.path.basename(model_to_load_identifier)}")
            else:
                self.status_label.setText(f"Status: Loaded Cellpose model: {model_to_load_identifier}")

        except Exception as e:
            QMessageBox.critical(self, "Cellpose Model Error", f"Failed to load Cellpose model: {e}")
            self.status_label.setText("Status: Error loading Cellpose model.")
            self.cellpose_model = None
            return

        # Load Classifier Model
        self.classifier_model_path = self.clf_model_path_edit.text()
        self.classifier_args_path = self.clf_args_path_edit.text()
        self.classifier_class_to_idx_path = self.clf_class_map_path_edit.text()

        if not all([self.classifier_model_path, self.classifier_args_path, self.classifier_class_to_idx_path]):
            QMessageBox.warning(self, "Classifier Info Missing", "Please provide paths for classifier model, args.txt, and class_to_idx.txt.")
            self.status_label.setText("Status: Classifier model info missing.")
            return
        
        try:
            # 1. Parse args.txt
            self._parse_args_txt(self.classifier_args_path)

            # 2. Parse class_to_idx.txt
            with open(self.classifier_class_to_idx_path, 'r') as f:
                class_to_idx_str = f.read()
            class_to_idx = ast.literal_eval(class_to_idx_str) # Safely evaluate string to dict
            self.classifier_class_names = sorted(class_to_idx.keys(), key=lambda k: class_to_idx[k])
            self.classifier_idx_to_class = {i: name for name, i in class_to_idx.items()}
            num_classes = len(self.classifier_class_names)

            # 3. Build and load classifier model
            # Ensure build_model is correctly defined/imported
            self.classifier_model = build_model(self.classifier_model_name, num_classes, pretrained=self.classifier_pretrained) # Use parsed pretrained status
            self.classifier_model.load_state_dict(torch.load(self.classifier_model_path, map_location=self.device))
            self.classifier_model.to(self.device)
            self.classifier_model.eval()

            # 4. Define classifier transforms (similar to val_test_transform)
            self.classifier_transform = transforms.Compose([
                transforms.Resize((self.classifier_img_size, self.classifier_img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.status_label.setText(f"Status: Loaded Cellpose ({self.cellpose_model_type}) and Classifier ({self.classifier_model_name}). Ready for image.")
            self.check_enable_analyze()

        except Exception as e:
            QMessageBox.critical(self, "Classifier Model Error", f"Failed to load classifier model or its assets: {e}")
            self.status_label.setText("Status: Error loading classifier model.")
            self.classifier_model = None
            return
        
        self.check_enable_analyze()


    def load_image(self):
        self.current_image_path = self.browse_file("Select Image", "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)")
        if self.current_image_path:
            try:
                # Using cellpose_io.imread handles various image formats better
                self.original_cv_image = cellpose_io.imread(self.current_image_path)
                if self.original_cv_image is None:
                    raise ValueError("Image could not be read by cellpose.io.imread.")

                # Ensure it's BGR for OpenCV display consistency if it's grayscale
                if len(self.original_cv_image.shape) == 2: # Grayscale
                    self.original_cv_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_GRAY2BGR)
                # If it has an alpha channel, remove it for consistency or handle as needed
                elif self.original_cv_image.shape[2] == 4: # RGBA
                    self.original_cv_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_RGBA2BGR)
                # If it's RGB, convert to BGR for OpenCV consistency (though cellpose might handle it)
                # For display, QImage expects RGB, so we'll convert back later.
                # Cellpose eval usually expects RGB or Grayscale.
                # Let's keep original_cv_image as BGR for cv2 drawing, and convert to RGB for QPixmap.

                display_image_rgb = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)
                h, w, ch = display_image_rgb.shape
                bytes_per_line = ch * w
                self.original_qimage = QImage(display_image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                pixmap = QPixmap.fromImage(self.original_qimage)
                self.original_image_label.setPixmap(pixmap.scaled(self.original_image_label.size(), 
                                                                  Qt.AspectRatioMode.KeepAspectRatio, 
                                                                  Qt.TransformationMode.SmoothTransformation))
                self.status_label.setText(f"Status: Loaded image: {os.path.basename(self.current_image_path)}")
                self.processed_image_label.setText("Processed Image: Results will appear here.") # Reset
                self.check_enable_analyze()
            except Exception as e:
                QMessageBox.critical(self, "Image Load Error", f"Failed to load image: {e}")
                self.current_image_path = None
                self.original_cv_image = None
                self.original_image_label.setText("Failed to load image.")


    def check_enable_analyze(self):
        if self.cellpose_model and self.classifier_model and self.current_image_path:
            self.analyze_btn.setEnabled(True)
        else:
            self.analyze_btn.setEnabled(False)

    def analyze_cells(self):
        if not self.current_image_path or not self.cellpose_model or not self.classifier_model:
            QMessageBox.warning(self, "Not Ready", "Please load models and an image first.")
            return

        self.status_label.setText("Status: Analyzing cells... (Segmentation)")
        QApplication.processEvents()

        try:
            # Get Cellpose parameters from UI
            try:
                diameter_text = self.cp_diameter_edit.text()
                self.cellpose_diameter = float(diameter_text) if diameter_text else 0.0
                if self.cellpose_diameter < 0: self.cellpose_diameter = 0.0 # Ensure non-negative, 0 for auto
            except ValueError:
                QMessageBox.warning(self, "Parameter Error", "Invalid diameter value. Using 0 (auto).")
                self.cellpose_diameter = 0.0
            
            # Flow and Cellprob thresholds are updated by sliders directly to self.cellpose_flow_threshold etc.

            try:
                ch1 = int(self.cp_channel1_edit.text())
                ch2 = int(self.cp_channel2_edit.text())
                self.cellpose_channels = [ch1, ch2]
            except ValueError:
                QMessageBox.warning(self, "Parameter Error", "Invalid channel values. Using [0,0].")
                self.cellpose_channels = [0,0]


            # Cellpose expects RGB or Grayscale. Our original_cv_image is BGR. Convert for cellpose.
            # Or, if cellpose.io.imread already gives a suitable format, use that directly.
            # Let's assume cellpose.io.imread gives something cellpose.eval can handle.
            # If original_cv_image was read by cv2.imread, it's BGR.
            # If by cellpose_io.imread, it might be RGB or Grayscale.
            # For safety, let's prepare an RGB version for Cellpose if it's BGR.
            
            # image_for_cellpose = self.original_cv_image # if cellpose_io.imread was used and it's fine
            # If original_cv_image is BGR:
            if self.original_cv_image.ndim == 3 and self.original_cv_image.shape[2] == 3:
                 image_for_cellpose = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)
            else: # Grayscale or already RGB
                 image_for_cellpose = self.original_cv_image.copy()


            # --- Segmentation ---
            self.log_message(f"Running Cellpose with: Diameter={self.cellpose_diameter if self.cellpose_diameter > 0 else 'auto'}, "
                             f"FlowThresh={self.cellpose_flow_threshold}, CellprobThresh={self.cellpose_cellprob_threshold}, "
                             f"Channels={self.cellpose_channels}")

            cell_instances, _ = segment_image_and_get_instances(
                self.cellpose_model, 
                image_for_cellpose, 
                diameter=self.cellpose_diameter if self.cellpose_diameter > 0 else None,
                channels=self.cellpose_channels,
                flow_threshold=self.cellpose_flow_threshold,
                cellprob_threshold=self.cellpose_cellprob_threshold
                # tile_norm_blocksize can be added here if a UI control is made for it
            )

            if not cell_instances:
                self.status_label.setText("Status: No cells found by Cellpose.")
                self.processed_image_label.setText("No cells detected.")
                return

            self.status_label.setText(f"Status: Found {len(cell_instances)} cells. Classifying...")
            QApplication.processEvents()

            # --- Classification ---
            self.annotated_cv_image = self.original_cv_image.copy() # Draw on BGR image

            for i, instance_info in enumerate(cell_instances):
                cropped_np = instance_info['cropped_image'] # This should be RGB or Grayscale from cellpose
                bbox = instance_info['bbox'] # [xmin, ymin, xmax, ymax]

                # Convert cropped NumPy array to PIL Image for PyTorch transform
                # Ensure cropped_np is in a format PIL.Image.fromarray can handle (e.g., RGB uint8)
                if cropped_np.ndim == 2: # Grayscale
                    pil_image = Image.fromarray(cropped_np, mode='L').convert('RGB')
                elif cropped_np.ndim == 3 and cropped_np.shape[2] == 1: # Single channel
                    pil_image = Image.fromarray(cropped_np.squeeze(), mode='L').convert('RGB')
                elif cropped_np.ndim == 3 and cropped_np.shape[2] == 3: # RGB
                    pil_image = Image.fromarray(cropped_np, mode='RGB')
                else:
                    print(f"Warning: Cropped image has unexpected shape {cropped_np.shape}, skipping.")
                    continue
                
                img_tensor = self.classifier_transform(pil_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.classifier_model(img_tensor)
                    _, predicted_idx = torch.max(outputs, 1)
                
                predicted_class_name = self.classifier_idx_to_class.get(predicted_idx.item(), "Unknown")
                class_color = CLASS_COLORS[predicted_idx.item() % len(CLASS_COLORS)] # Cycle through colors

                # Draw on the annotated image (which is BGR)
                cv2.rectangle(self.annotated_cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), class_color, 2)
                cv2.putText(self.annotated_cv_image, f"{predicted_class_name}", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 1)
                
                if (i + 1) % 10 == 0: # Update status periodically
                    self.status_label.setText(f"Status: Classified {i+1}/{len(cell_instances)} cells.")
                    QApplication.processEvents()


            # Display annotated image
            display_annotated_rgb = cv2.cvtColor(self.annotated_cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = display_annotated_rgb.shape
            bytes_per_line = ch * w
            q_image_processed = QImage(display_annotated_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            pixmap_processed = QPixmap.fromImage(q_image_processed)
            self.processed_image_label.setPixmap(pixmap_processed.scaled(self.processed_image_label.size(), 
                                                                        Qt.AspectRatioMode.KeepAspectRatio, 
                                                                        Qt.TransformationMode.SmoothTransformation))
            self.status_label.setText(f"Status: Analysis complete. Processed {len(cell_instances)} cells.")

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis: {e}")
            self.status_label.setText(f"Status: Error during analysis: {e}")
            print(f"Detailed error: {e}") # For console debugging
            import traceback
            traceback.print_exc()

    def log_message(self, message, level='info'): # Added log_message stub
        print(f"[{level.upper()}] {message}") # Corrected f-string syntax
        # In a real scenario, this might write to a log file or a QTextEdit in the UI.
        if level == 'error':
            self.status_label.setText(f"Error: {message[:100]}...") # Show brief error on status
        elif level == 'warning':
             self.status_label.setText(f"Warning: {message[:100]}...")
        # else:
        #     self.status_label.setText(f"Status: {message[:100]}...")


    def _try_load_latest_classifier_defaults(self):
        try:
            # Corrected base directory to be relative to this UI script's location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Assuming the 'models' directory is one level up from the 'ui' directory
            default_classifier_base_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))

            if not os.path.isdir(default_classifier_base_dir):
                print(f"Default classifier base directory not found: {default_classifier_base_dir}")
                return

            latest_classifier_dir = None
            latest_mtime = 0

            for item_name in os.listdir(default_classifier_base_dir):
                item_path = os.path.join(default_classifier_base_dir, item_name)
                if os.path.isdir(item_path) and item_name.startswith("classifier_"):
                    try:
                        mtime = os.path.getmtime(item_path)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_classifier_dir = item_path
                    except OSError:
                        continue
            
            if latest_classifier_dir:
                print(f"Found latest classifier directory: {latest_classifier_dir}")
                potential_args_path = os.path.join(latest_classifier_dir, "args.txt")
                potential_map_path = os.path.join(latest_classifier_dir, "class_to_idx.txt")
                
                temp_model_name = 'resnet18' # Default fallback
                if os.path.exists(potential_args_path):
                    args_dict_temp = {}
                    with open(potential_args_path, 'r') as f_args:
                        for line in f_args:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                args_dict_temp[key.strip()] = value.strip()
                    temp_model_name = args_dict_temp.get('model_name', temp_model_name)
                
                potential_model_path = os.path.join(latest_classifier_dir, f"{temp_model_name}_best.pth")

                paths_to_set = {}
                if os.path.exists(potential_model_path):
                    paths_to_set['model'] = potential_model_path
                else:
                    print(f"Model file not found: {potential_model_path}")
                
                if os.path.exists(potential_args_path):
                    paths_to_set['args'] = potential_args_path
                else:
                    print(f"Args file not found: {potential_args_path}")

                if os.path.exists(potential_map_path):
                    paths_to_set['map'] = potential_map_path
                else:
                    print(f"Class map file not found: {potential_map_path}")

                # Set paths if found
                if 'model' in paths_to_set:
                    self.clf_model_path_edit.setText(paths_to_set['model'])
                    self.classifier_model_path = paths_to_set['model']
                if 'args' in paths_to_set:
                    self.clf_args_path_edit.setText(paths_to_set['args'])
                    self.classifier_args_path = paths_to_set['args']
                if 'map' in paths_to_set:
                    self.clf_class_map_path_edit.setText(paths_to_set['map'])
                    self.classifier_class_to_idx_path = paths_to_set['map']
                
                if all(k in paths_to_set for k in ['model', 'args', 'map']):
                    self.status_label.setText("Status: Automatically filled paths for the latest classifier. Press 'Load All Models'.")
                else:
                    self.status_label.setText("Status: Found latest classifier directory, but some asset files were missing. Paths partially filled.")
            else:
                print(f"No classifier directories found in {default_classifier_base_dir}")

        except Exception as e:
            print(f"Error trying to load latest classifier defaults: {e}")
            # Do not show a QMessageBox here, as it's an optional startup feature

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CellAnalysisApp()
    # ex.show() is called within CellAnalysisApp.initUI()
    sys.exit(app.exec())
