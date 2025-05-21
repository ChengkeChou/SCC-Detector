from __future__ import annotations # Defer type hint evaluation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Add this line to address OMP Error #15

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import cv2
import pandas as pd
import traceback
import torch # Added import
from pathlib import Path # Added import
from tqdm import tqdm # Added import
import sys # Added import

from cellpose import models, io, plot # Standard way to import models and io
from torchvision import models as torch_models, transforms
from PIL import Image

# 预定义细胞类别（根据实际项目数据调整）
CELL_CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate"
]

def _get_scalar_diameter(diams_value, input_diameter_val=None):
    """Converts a diameter value (which could be float, int, ndarray, or None) to a scalar float."""
    scalar_val = None
    if isinstance(diams_value, np.ndarray):
        if diams_value.size == 1:
            scalar_val = diams_value.item()
        elif diams_value.size > 1:
            scalar_val = np.mean(diams_value)
        else: # diams_value.size == 0 (empty array)
            print("    Warning: Cellpose returned an empty diameter array.")
            # Fallback logic handled below
    elif isinstance(diams_value, (float, int, np.floating, np.integer)): # More robust check for numpy scalars
        scalar_val = float(diams_value)
    elif diams_value is None:
        print("    Warning: Cellpose returned None for diameter.")
        # Fallback logic handled below
    else:
        print(f"    Warning: Unexpected type for diameter value: {type(diams_value)}. Value: {diams_value}.")
        # Fallback logic handled below

    # Validate and apply fallback for scalar_val if it's problematic or diams_value was unhandled
    if scalar_val is None or scalar_val <= 0: # Diameter must be positive
        if scalar_val is not None: # Only print if it was calculated but invalid
            print(f"    Diameter value {scalar_val:.4f} is non-positive. Using fallback.")
        
        if input_diameter_val is not None:
            try:
                fallback_diam = float(input_diameter_val)
                if fallback_diam > 0:
                    print(f"    Using input diameter as fallback: {fallback_diam}")
                    return fallback_diam
            except ValueError:
                print(f"    Warning: Could not parse input_diameter_val '{input_diameter_val}' as float.")
        
        # Default fallback if input_diameter_val is not suitable
        default_fallback = 30.0
        print(f"    Using default fallback diameter: {default_fallback}")
        return default_fallback
        
    return scalar_val

def segment_image_and_get_instances(model: 'models.CellposeModel' | 'models.Cellpose', image_np: np.ndarray, 
                                   diameter: float | None = None, channels: list | None = None,
                                   flow_threshold: float = 0.4, cellprob_threshold: float = 0.0,
                                   tile_norm_blocksize: int = 0, show_visualization: bool = False):
    """
    Performs cell segmentation on a single image using a pre-initialized CellPose model
    and returns information about each detected cell instance.

    Args:
        model ('models.CellposeModel' | 'models.Cellpose'): Pre-initialized CellPose model.
        image_np (np.ndarray): The input image as a NumPy array.
        diameter (float, optional): Expected cell diameter in pixels. If None, CellPose estimates it.
                                   This overrides the model's default diameter if provided.
        channels (list, optional): List of channels to use for segmentation.
                                  e.g., [0,0] for grayscale, [2,1] for cytoplasm (red) and nucleus (green).
                                  If None, CellPose uses its default.
        flow_threshold (float): Maximum allowed error of the flows for each mask. Default is 0.4.
                               Increase this threshold if cellpose is not returning as many masks as you'd expect.
                               Decrease this threshold if cellpose is returning too many ill-shaped masks.
        cellprob_threshold (float): Threshold on cellular probability. Default is 0.0.
                                   Decrease this threshold if cellpose is not returning as many masks as you'd expect.
                                   Increase this threshold if cellpose is returning too many masks from dim areas.
        tile_norm_blocksize (int): Blocksize for normalizing the image in tiles. Default is 0 (normalize whole image).
                                  Use values like 100-200 for inhomogeneous brightness across the image.
        show_visualization (bool): Whether to show a visualization of the segmentation. Default is False.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected cell instance
              and contains:
              - 'cropped_image': NumPy array of the cropped cell.
              - 'bbox': [xmin, ymin, xmax, ymax] bounding box coordinates in the original image.
              - 'mask_full': Boolean NumPy array of the cell's mask in the full original image dimensions.
             - 'mask_cropped': Boolean NumPy array of the cell's mask in the cropped image dimensions.
              - 'estimated_diameter': Diameter estimated by CellPose for the input image.
        float: The diameter estimated by CellPose for this image.
    """
    if image_np is None:
        print("Error: Input image is None.")
        return [], 0.0

    # Ensure image is 2D (grayscale) or 3D (RGB)
    if len(image_np.shape) > 2 and image_np.shape[-1] > 3:  # e.g. RGBA or multi-channel beyond RGB
        image_np = image_np[..., :3]  # Keep only RGB
    
    # Default channels if not specified
    if channels is None:
        if len(image_np.shape) == 2 or image_np.shape[-1] == 1:  # Grayscale
            channels = [0, 0]  # Use the same channel for both cytoplasm and nucleus
        else:  # RGB/Multichannel
            channels = [0, 0]  # Default for RGB is first channel
            # Can be customized based on staining, e.g., [2, 1] for RED=cytoplasm, GREEN=nucleus
    
    # Normalize image for better segmentation
    normalize_params = {"tile_norm_blocksize": tile_norm_blocksize} if tile_norm_blocksize > 0 else True

    # Run CellPose-SAM segmentation
    # Run CellPose-SAM segmentation with optimized parameters
    # masks, flows, styles, diams_est = model.eval(image_np,
    #                                            diameter=diameter, 
    #                                            channels=channels, # channels is deprecated
    #                                            flow_threshold=flow_threshold, 
    #                                            cellprob_threshold=cellprob_threshold,
    #                                            normalize=normalize_params,
    #                                            batch_size=32)  # Optimized batch size for faster processing
    
    # For Cellpose 4.0.1+, eval returns masks, flows, diams
    # The 'styles' output has been removed.
    # The 'channels' argument to eval is also deprecated. Cellpose handles channel selection internally.
    masks, flows, diams = model.eval(image_np, 
                                           diameter=diameter,
                                           # channels=channels, # Deprecated
                                           flow_threshold=flow_threshold, 
                                           cellprob_threshold=cellprob_threshold,
                                           normalize=normalize_params,
                                           batch_size=32)
    
    num_cells_in_image = np.max(masks)
    # Ensure diams is a scalar float for formatting
    scalar_diams = _get_scalar_diameter(diams, diameter)
    print(f"    Found {num_cells_in_image} cells. Estimated diameter: {scalar_diams:.2f} px")

    # Visualize the segmentation if requested
    if show_visualization:
        # Create a nice matplotlib visualization with improved layout
        fig = plt.figure(figsize=(12, 5))
        # plot.show_segmentation(fig, image_np, masks, flows[0]) # flows[0] might be problematic if flows is None or not structured as expected
        # Check if flows is not None and has the expected structure
        flow_to_show = None
        if flows is not None and len(flows) > 0 and isinstance(flows[0], np.ndarray):
            flow_to_show = flows[0]
        elif flows is not None and isinstance(flows, tuple) and len(flows) > 0 and isinstance(flows[0], np.ndarray): # flows can be a tuple e.g. (flow_x, flow_y)
             # For simplicity, just show the first component or combine them if necessary
             # This part might need adjustment based on the exact structure of 'flows' in Cellpose 4.x
            flow_to_show = flows[0]


        plot.show_segmentation(fig, image_np, masks, flow_to_show, channels=channels if channels != [0,0] else None) # Pass original channels for viz if meaningful
        plt.tight_layout()
        plt.show()

    detected_instances = []

    if num_cells_in_image == 0:
        return [], diams

    for i in range(1, num_cells_in_image + 1):
        cell_mask_full = (masks == i)
        if not np.any(cell_mask_full):
            continue

        # Get bounding box for the cell
        rows = np.any(cell_mask_full, axis=1)
        cols = np.any(cell_mask_full, axis=0)
        if not np.any(rows) or not np.any(cols):
            continue

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add a small padding
        padding = 2
        rmin_pad = max(0, rmin - padding)
        rmax_pad = min(image_np.shape[0], rmax + 1 + padding)
        cmin_pad = max(0, cmin - padding)
        cmax_pad = min(image_np.shape[1], cmax + 1 + padding)
        
        cropped_cell_img = image_np[rmin_pad:rmax_pad, cmin_pad:cmax_pad]
        
        # The mask within the cropped image (useful for some classification models)
        cropped_mask_relative = cell_mask_full[rmin_pad:rmax_pad, cmin_pad:cmax_pad]

        if cropped_cell_img.size == 0:
            print(f"    Warning: Cropped cell {i} is empty. Skipping.")
            continue
        
        instance_info = {
            'cropped_image': cropped_cell_img,
            'bbox': [cmin, rmin, cmax, rmax], # xmin, ymin, xmax, ymax
            'mask_full': cell_mask_full,
            'mask_cropped': cropped_mask_relative, # Mask relative to the crop
            'estimated_diameter': scalar_diams # Use the scalar diameter here as well
        }
        detected_instances.append(instance_info)

    return detected_instances, scalar_diams # Return scalar diameter

# --- Main execution for testing (single image) ---
# This block can be removed or commented out if the primary execution is directory-based
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Test CellPose segmentation on a single image.")
#     parser.add_argument("--image_path", type=str, required=True,
#                         help="Path to the input image file for testing.")
#     parser.add_argument("--model_type", type=str, default="cyto",
#                         help="CellPose model type ('cyto', 'nuclei') or path to a custom model.")
#     parser.add_argument("--diameter", type=float, default=None,
#                         help="Expected cell diameter in pixels. CellPose estimates if None. (e.g., 30.0)")
#     parser.add_argument("--flow_threshold", type=float, default=0.4,
#                         help="Flow threshold for CellPose - adjust to control mask sensitivity")
#     parser.add_argument("--cellprob_threshold", type=float, default=0.0,
#                         help="Cell probability threshold - adjust to control mask detection sensitivity")
#     parser.add_argument("--tile_norm_blocksize", type=int, default=0,
#                         help="Block size for tile normalization - use for inhomogeneous brightness")
#     parser.add_argument("--visualize", action="store_true", 
#                         help="Show visualization of the segmentation results")
#     parser.add_argument("--no_gpu", action="store_true", help="Disable GPU usage for CellPose.")
    
#     args = parser.parse_args()

#     print(f"--- Initializing CellPose Model ({args.model_type}) ---")
#     use_gpu = not args.no_gpu
#     if args.model_type in ['cyto', 'nuclei']:
#         cp_model = models.CellposeModel(gpu=use_gpu, model_type=args.model_type)
#     else: 
#         cp_model = models.CellposeModel(gpu=use_gpu, pretrained_model=args.model_type)
#     print(f"Model initialized. Using GPU: {use_gpu}")

#     print(f"--- Loading Test Image: {args.image_path} ---")
#     try:
#         test_image = io.imread(args.image_path)
#         if test_image is None:
#             raise ValueError("Image could not be read.")
#     except Exception as e:
#         print(f"Error loading image: {e}")
#         exit()
    
#     print(f"Image loaded successfully. Shape: {test_image.shape}")

#     print(f"--- Running Segmentation ---")
#     start_time = time.time()
    
#     instances, est_diameter = segment_image_and_get_instances(
#         cp_model, 
#         test_image, 
#         diameter=args.diameter,
#         flow_threshold=args.flow_threshold,
#         cellprob_threshold=args.cellprob_threshold,
#         tile_norm_blocksize=args.tile_norm_blocksize,
#         show_visualization=args.visualize
#     )
#     end_time = time.time()
#     print(f"Segmentation completed in {end_time - start_time:.2f} seconds.")
#     scalar_est_diameter = _get_scalar_diameter(est_diameter, args.diameter)
#     print(f"Found {len(instances)} cells with estimated diameter: {scalar_est_diameter:.2f}px")

# The segment_and_crop_cells function is removed as its functionality is covered by run_cellpose_segmentation_on_directory
# def segment_and_crop_cells(input_dir, output_dir, model_type, diameter=None, ...):
#     ...

def run_cellpose_segmentation_on_directory(input_dir, output_dir, model_type,
                                      diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
                                      tile_norm_blocksize=0, gpu=True, save_visualizations=True):
    """
    运行Cellpose分割并将裁剪的细胞和掩码保存到目录。
    """
    print(f"初始化CellPose模型({model_type})...")
    cp_model = models.CellposeModel(gpu=gpu, model_type=model_type)
    
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    crops_dir = output_dir / "crops"
    results_dir = output_dir / "results" # For segmentation visualizations and CSVs
    
    for dir_path in [output_dir, masks_dir, crops_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    stats = {
        "total_images": 0,
        "total_cells_segmented": 0,
    }
    
    all_segmentation_results = []
    
    input_dir = Path(input_dir)
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"**/*{ext}"))) 
    
    if not image_files:
        print(f"错误: 在{input_dir}中没有找到图像")
        return
        
    print(f"找到{len(image_files)}个图像文件进行分割")
    
    for img_path in tqdm(image_files, desc="分割图像"):
        try:
            img_name = img_path.name
            print(f"\\n处理图像: {img_name}")
            
            img = io.imread(str(img_path))
            
            instances, est_diameter = segment_image_and_get_instances(
                cp_model, 
                img, 
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                tile_norm_blocksize=tile_norm_blocksize
            )
            
            if instances and save_visualizations:
                masks = np.zeros(img.shape[:2], dtype=np.int32)
                for i, instance in enumerate(instances, 1):
                    masks[instance['mask_full']] = i
                
                masks_path_file = masks_dir / f"{img_path.stem}_masks.tif"
                io.imsave(str(masks_path_file), masks)
                
                fig = plt.figure(figsize=(12, 8))
                plot.show_segmentation(fig, img, masks, None) # No flow for simplicity here
                plt.tight_layout()
                plt.savefig(str(results_dir / f"{img_path.stem}_segmentation.png"), dpi=150)
                plt.close(fig)
            
            stats["total_images"] += 1
            stats["total_cells_segmented"] += len(instances)
            
            segmentation_results_for_image = []
            
            # Create a subdirectory in crops_dir for the current image
            image_specific_crops_dir = crops_dir / img_path.stem
            image_specific_crops_dir.mkdir(parents=True, exist_ok=True)

            for i, instance in enumerate(instances, 1):
                cell_img = instance['cropped_image']
                bbox = instance['bbox']
                
                cell_filename = f"cell_{i:03d}.png"
                cell_save_path = image_specific_crops_dir / cell_filename
                
                try:
                    if len(cell_img.shape) == 3 and cell_img.shape[2] == 3: # RGB
                        cv2.imwrite(str(cell_save_path), cv2.cvtColor(cell_img, cv2.COLOR_RGB2BGR))
                    elif len(cell_img.shape) == 2: # Grayscale
                        cv2.imwrite(str(cell_save_path), cell_img)
                    else: # Other formats, try to save directly
                         cv2.imwrite(str(cell_save_path), cell_img)
                except Exception as e_write:
                    print(f"  保存裁剪的细胞图像失败 {cell_save_path}: {e_write}")
                
                result = {
                    "image_path": str(img_path),
                    "original_image_filename": img_name,
                    "cell_id_in_image": i,
                    "bbox_xmin_ymin_xmax_ymax": bbox,
                    "estimated_diameter_for_image": est_diameter,
                    "cropped_cell_path": str(cell_save_path)
                }
                segmentation_results_for_image.append(result)
                all_segmentation_results.append(result)
            
            if segmentation_results_for_image:
                df = pd.DataFrame(segmentation_results_for_image)
                df.to_csv(str(results_dir / f"{img_path.stem}_segmentation_details.csv"), index=False)
                
        except Exception as e_proc:
            print(f"处理图像 {img_path} 失败: {e_proc}")
            traceback.print_exc()
    
    if all_segmentation_results:
        all_df = pd.DataFrame(all_segmentation_results)
        all_df.to_csv(str(output_dir / "all_segmentation_results.csv"), index=False)
    
    print("\\n=== 分割完成 ===")
    print(f"处理的图像总数: {stats['total_images']}")
    print(f"分割的细胞总数: {stats['total_cells_segmented']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用CellPose进行宫颈细胞分割")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="包含输入图像的目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存分割结果的目录")
    parser.add_argument("--model_type", type=str, default="cyto2",
                        help="CellPose模型类型 ('cyto2', 'cyto', 'nuclei' 等)")
    parser.add_argument("--diameter", type=float, default=None,
                        help="预期细胞直径（像素）。如果为None，CellPose会估计。")
    parser.add_argument("--flow_threshold", type=float, default=0.4,
                        help="CellPose的流阈值")
    parser.add_argument("--cellprob_threshold", type=float, default=0.0,
                        help="CellPose的细胞概率阈值")
    parser.add_argument("--tile_norm_blocksize", type=int, default=0,
                        help="瓦片归一化块大小 (0表示不进行瓦片归一化)")
    parser.add_argument("--no_gpu", action="store_true", 
                        help="禁用GPU")
    parser.add_argument("--no_vis", action="store_true",
                        help="不保存分割可视化结果")
    
    args = parser.parse_args()
    
    if len(sys.argv) <= 1: # Show help if no arguments are passed
        parser.print_help(sys.stderr)
        sys.exit(1)

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"CellPose模型: {args.model_type}")
    print(f"使用GPU: {not args.no_gpu}")
    print(f"保存可视化: {not args.no_vis}")

    start_time_main = time.time()
    run_cellpose_segmentation_on_directory(
        args.input_dir, 
        args.output_dir, 
        args.model_type,
        diameter=args.diameter, 
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        tile_norm_blocksize=args.tile_norm_blocksize,
        gpu=not args.no_gpu,
        save_visualizations=not args.no_vis
    )
    end_time_main = time.time()
    print(f"总处理时间: {end_time_main - start_time_main:.2f} 秒")
