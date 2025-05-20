"""
数据预处理模块
将.dat边界坐标文件转换为分割掩码格式，支持COCO、YOLO和自定义格式
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re

# 类别映射
CLASS_MAPPING = {
    "Dyskeratotic": 0,
    "Koilocytotic": 1, 
    "Metaplastic": 2,
    "Parabasal": 3,
    "Superficial-Intermediate": 4
}

# 创建JSON可序列化的类型转换函数
def json_serialize(obj):
    """将NumPy类型转换为可JSON序列化的类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple) and all(isinstance(i, (int, float)) for i in obj):
        return list(obj)
    return obj

def read_dat_file(dat_file_path):
    """读取.dat文件并提取坐标"""
    try:
        with open(dat_file_path, 'r') as f:
            lines = f.readlines()
        
        # 提取坐标信息
        coordinates = []
        for line in lines:
            # 跳过注释行
            if line.strip().startswith('#') or not line.strip():
                continue
                
            # 提取坐标对 - 支持多种分隔符格式 (空格, 逗号, 分号等)
            try:
                # 尝试不同的分隔符
                if ',' in line:
                    x, y = map(float, line.strip().split(','))
                elif ';' in line:
                    x, y = map(float, line.strip().split(';'))
                else:
                    # 默认使用空格或制表符
                    x, y = map(float, line.strip().split())
                
                coordinates.append((int(x), int(y)))
            except ValueError:
                print(f"无法解析坐标行: {line.strip()} 在文件 {dat_file_path}")
                
        return coordinates
    except Exception as e:
        print(f"读取.dat文件时出错 {dat_file_path}: {str(e)}")
        return []

def create_mask_from_coordinates(coordinates, image_shape):
    """基于多边形坐标创建二值掩码"""
    if not coordinates or len(coordinates) < 3:
        return None
    
    # 创建空掩码
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # 将坐标转换为NumPy数组并重塑为适合cv2.fillPoly的格式
    points = np.array([coordinates], dtype=np.int32)
    
    # 填充多边形
    cv2.fillPoly(mask, points, 255)
    
    return mask

def extract_image_id(filename):
    """从文件名中提取图像ID"""
    match = re.search(r'(\d+)_[a-z]+\d+', filename)
    if match:
        return match.group(1)
    return None

def get_cell_type_from_path(file_path):
    """从文件路径中提取细胞类型"""
    path_parts = Path(file_path).parts
    for part in path_parts:
        if part in CLASS_MAPPING:
            return part
    return None

def process_image_with_annotations(image_path, output_dir, format="yolo", keep_structure=True, all_formats=False):
    """
    处理单个图像及其所有注释
    
    Args:
        image_path: 图像路径
        output_dir: 输出目录
        format: 输出格式 (yolo, coco, mask)
        keep_structure: 是否保持原始文件夹结构
        all_formats: 是否同时输出所有格式
    
    Returns:
        dict: 处理的细胞实例信息
    """
    try:
        # 获取必要的路径和基本信息
        image_name = Path(image_path).name
        base_name = Path(image_path).stem
        cell_type = get_cell_type_from_path(image_path)
        class_id = CLASS_MAPPING.get(cell_type, 0)
        
        # 确定要处理的格式
        formats_to_process = ["yolo", "coco", "mask"] if all_formats else [format]
        
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
            
        image_height, image_width = image.shape[:2]
        
        # 查找相关的.dat文件（细胞质和细胞核）
        dat_dir = Path(image_path).parent
        cyt_files = list(dat_dir.glob(f"{base_name}_cyt*.dat"))
        nuc_files = list(dat_dir.glob(f"{base_name}_nuc*.dat"))
        
        if not cyt_files or not nuc_files:
            print(f"缺少.dat文件: {image_path}")
            return None
            
        # 确保数量匹配
        if len(cyt_files) != len(nuc_files):
            print(f"细胞质和细胞核文件数量不匹配: {len(cyt_files)} vs {len(nuc_files)} in {image_path}")
            # 尝试找出匹配的对
            matched_pairs = []
            for cyt_file in cyt_files:
                cyt_id = re.search(r'cyt(\d+)', cyt_file.name).group(1)
                matching_nuc = [n for n in nuc_files if re.search(r'nuc(\d+)', n.name).group(1) == cyt_id]
                if matching_nuc:
                    matched_pairs.append((cyt_file, matching_nuc[0]))
            if matched_pairs:
                print(f"找到了{len(matched_pairs)}对匹配的文件")
                cyt_files = [pair[0] for pair in matched_pairs]
                nuc_files = [pair[1] for pair in matched_pairs]
            else:
                return None
        
        # 排序以确保匹配
        cyt_files.sort()
        nuc_files.sort()
        
        # 创建组合掩码 - 背景为0，每个实例有唯一ID
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # 存储实例信息
        instances = []
        
        # 为每种格式创建输出目录和文件
        output_paths = {}
        label_files = {}
        
        for fmt in formats_to_process:
            # 确定此格式的输出目录
            if all_formats:
                fmt_output_dir = os.path.join(output_dir, fmt)
            else:
                fmt_output_dir = output_dir
                
            # 如果保持原始结构，添加类别子目录
            if keep_structure:
                cell_subdir = os.path.join(fmt_output_dir, cell_type)
                os.makedirs(cell_subdir, exist_ok=True)
                
                # 创建格式所需的各个子目录
                images_dir = os.path.join(cell_subdir, "images")
                os.makedirs(images_dir, exist_ok=True)
                
                if fmt == "yolo":
                    labels_dir = os.path.join(cell_subdir, "labels")
                    os.makedirs(labels_dir, exist_ok=True)
                    label_files[fmt] = os.path.join(labels_dir, f"{base_name}.txt")
                elif fmt == "coco":
                    annotations_dir = os.path.join(cell_subdir, "annotations")
                    os.makedirs(annotations_dir, exist_ok=True)
                elif fmt == "mask":
                    masks_dir = os.path.join(cell_subdir, "masks")
                    os.makedirs(masks_dir, exist_ok=True)
                
                # 设置输出路径
                output_paths[fmt] = {
                    "image": os.path.join(images_dir, image_name),
                    "mask": os.path.join(masks_dir, f"{base_name}_mask.png") if fmt == "mask" else None
                }
            else:
                # 不保持结构，使用标准目录布局
                images_dir = os.path.join(fmt_output_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                
                if fmt == "yolo":
                    labels_dir = os.path.join(fmt_output_dir, "labels")
                    os.makedirs(labels_dir, exist_ok=True)
                    label_files[fmt] = os.path.join(labels_dir, f"{base_name}.txt")
                elif fmt == "coco":
                    annotations_dir = os.path.join(fmt_output_dir, "annotations")
                    os.makedirs(annotations_dir, exist_ok=True)
                elif fmt == "mask":
                    masks_dir = os.path.join(fmt_output_dir, "masks")
                    os.makedirs(masks_dir, exist_ok=True)
                
                # 设置输出路径
                output_paths[fmt] = {
                    "image": os.path.join(images_dir, image_name),
                    "mask": os.path.join(masks_dir, f"{base_name}_mask.png") if fmt == "mask" else None
                }
        
        # 准备YOLO格式的标签行
        yolo_label_lines = [] if "yolo" in formats_to_process else None
        
        # 复制原始图像到各个输出目录
        for fmt in formats_to_process:
            shutil.copy(image_path, output_paths[fmt]["image"])
        
        # 处理每个细胞
        for i, (cyt_file, nuc_file) in enumerate(zip(cyt_files, nuc_files)):
            # 确保核和细胞质匹配
            cyt_id = re.search(r'cyt(\d+)', cyt_file.name).group(1)
            nuc_id = re.search(r'nuc(\d+)', nuc_file.name).group(1)
            
            if cyt_id != nuc_id:
                print(f"警告: 细胞质和细胞核ID不匹配: {cyt_file.name} vs {nuc_file.name}")
                continue
                
            # 读取坐标
            cyt_coords = read_dat_file(str(cyt_file))
            nuc_coords = read_dat_file(str(nuc_file))
            
            if not cyt_coords or not nuc_coords:
                continue
                
            # 创建掩码
            cyt_mask = create_mask_from_coordinates(cyt_coords, image.shape)
            nuc_mask = create_mask_from_coordinates(nuc_coords, image.shape)
            
            if cyt_mask is None or nuc_mask is None:
                continue
                
            # 为这个细胞实例分配一个唯一ID
            instance_id = i + 1
            
            # 添加到组合掩码 (实例分割)
            combined_mask[cyt_mask > 0] = instance_id
            
            # 找到掩码的边界框
            y_indices, x_indices = np.where(cyt_mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # 为YOLO格式处理
            if "yolo" in formats_to_process:
                # 计算YOLO格式的归一化坐标
                x_center = (x_min + x_max) / (2 * image_width)
                y_center = (y_min + y_max) / (2 * image_height)
                box_width = (x_max - x_min) / image_width
                box_height = (y_max - y_min) / image_height
                
                # 添加到YOLO标签
                yolo_label_lines.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")
              # 存储实例信息
            instances.append({
                "instance_id": int(instance_id),
                "class_id": int(class_id),
                "class_name": cell_type,
                "cyt_file": str(cyt_file),
                "nuc_file": str(nuc_file),
                "cyt_coords": [[int(x), int(y)] for x, y in cyt_coords],
                "nuc_coords": [[int(x), int(y)] for x, y in nuc_coords],
                "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            })
        
        # 保存各种格式的输出
        for fmt in formats_to_process:
            # 保存掩码
            if fmt == "mask":
                cv2.imwrite(output_paths[fmt]["mask"], combined_mask)
            
            # 保存YOLO格式标签
            if fmt == "yolo" and yolo_label_lines:
                with open(label_files[fmt], "w") as f:
                    f.write("\n".join(yolo_label_lines))
        
        # 返回处理结果信息
        result = {
            "image_path": image_path,
            "output_paths": output_paths,
            "cell_type": cell_type,
            "instance_count": len(instances),
            "instances": instances,
            "image_width": image_width,
            "image_height": image_height
        }
        
        return result
                
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def convert_dataset(input_dir, output_dir, format="yolo", num_workers=4, keep_structure=True, all_formats=False):
    """
    转换整个数据集
    
    Args:
        input_dir: 输入数据集根目录
        output_dir: 输出目录
        format: 输出格式 (yolo, coco, mask)或'all'表示所有格式
        num_workers: 并行处理的工作线程数
        keep_structure: 是否保持原始文件夹结构
        all_formats: 是否同时输出所有格式
    """
    # 获取所有.bmp图像
    image_paths = []
    for cell_type in CLASS_MAPPING.keys():
        cell_type_dir = os.path.join(input_dir, cell_type)
        if os.path.exists(cell_type_dir):
            image_paths.extend(glob.glob(os.path.join(cell_type_dir, "*.bmp")))
    
    print(f"找到{len(image_paths)}个图像文件")
    
    # 确定要处理的格式
    formats_to_process = ["yolo", "coco", "mask"] if all_formats else [format]
    
    # 为每种格式创建输出目录
    for fmt in formats_to_process:
        if all_formats:
            fmt_output_dir = os.path.join(output_dir, fmt)
        else:
            fmt_output_dir = output_dir
        os.makedirs(fmt_output_dir, exist_ok=True)
    
    # 创建元数据文件
    metadata = {
        "dataset_info": {
            "name": "SIPakMed细胞数据集",
            "description": "脱落细胞学图像的实例分割数据集",
            "format": format if not all_formats else "multiple",
            "classes": list(CLASS_MAPPING.keys()),
            "class_mapping": CLASS_MAPPING,
            "image_count": len(image_paths)
        },
        "images": []
    }
    
    # 并行处理图像
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(
                lambda p: process_image_with_annotations(p, output_dir, format, keep_structure, all_formats), 
                image_paths
            ),
            total=len(image_paths),
            desc="处理图像"
        ))
    
    # 过滤掉None结果并添加到元数据
    valid_results = [r for r in results if r is not None]
    metadata["images"] = valid_results
      # 保存元数据到每个格式的输出目录
    for fmt in formats_to_process:
        if all_formats:
            fmt_output_dir = os.path.join(output_dir, fmt)
        else:
            fmt_output_dir = output_dir
        
        with open(os.path.join(fmt_output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, default=json_serialize, ensure_ascii=False, indent=2)
    
    # 创建训练/验证/测试分割
    image_ids = [Path(img["image_path"]).stem for img in valid_results]
    np.random.shuffle(image_ids)
    
    train_split = 0.7
    val_split = 0.15
    
    train_size = int(len(image_ids) * train_split)
    val_size = int(len(image_ids) * val_split)
    
    train_ids = image_ids[:train_size]
    val_ids = image_ids[train_size:train_size+val_size]
    test_ids = image_ids[train_size+val_size:]
    
    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
      # 保存分割信息到每个格式的输出目录
    for fmt in formats_to_process:
        if all_formats:
            fmt_output_dir = os.path.join(output_dir, fmt)
        else:
            fmt_output_dir = output_dir
        
        with open(os.path.join(fmt_output_dir, "splits.json"), "w", encoding="utf-8") as f:
            json.dump(splits, f, default=json_serialize, ensure_ascii=False, indent=2)
    
    print(f"数据集转换完成。")
    print(f"  - 训练集: {len(train_ids)}个图像")
    print(f"  - 验证集: {len(val_ids)}个图像")
    print(f"  - 测试集: {len(test_ids)}个图像")
    print(f"  - 总计: {len(valid_results)}个有效图像 (有{len(image_paths)-len(valid_results)}个处理失败)")
    
    if all_formats:
        print(f"输出格式: {', '.join(formats_to_process)}")
        print(f"输出保存到: {output_dir}")
        for fmt in formats_to_process:
            print(f"  - {fmt}: {os.path.join(output_dir, fmt)}")
    else:
        print(f"输出格式: {format}")
        print(f"输出保存到: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将.dat坐标文件转换为细胞分割数据集")
    parser.add_argument("--input", type=str, required=True, help="输入数据集目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--format", type=str, default="yolo", choices=["yolo", "coco", "mask", "all"],
                      help="输出格式 (yolo, coco, mask, 或 all表示全部)")
    parser.add_argument("--workers", type=int, default=4, help="并行工作线程数")
    parser.add_argument("--keep-structure", action="store_true", help="保持原始文件夹结构")
    
    args = parser.parse_args()
    
    # 检查是否要输出所有格式
    all_formats = args.format == "all"
    format_to_use = "yolo" if all_formats else args.format
    
    convert_dataset(args.input, args.output, format_to_use, args.workers, args.keep_structure, all_formats)
