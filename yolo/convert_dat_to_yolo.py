import os
from PIL import Image
import shutil

# Class name to index mapping from sipakmed_seg.yaml
CLASS_MAPPING = {
    "Dyskeratotic": 0,
    "Koilocytotic": 1,
    "Metaplastic": 2,
    "Parabasal": 3,
    "Superficial-Intermediate": 4
}

RAW_DATA_BASE_PATH = r"f:\SSC\raw_data\SIPakMed"
PROCESSED_DATA_BASE_PATH = r"f:\SSC\cell_segmentation\data"

def convert_dat_to_yolo_format(dat_file_path, image_width, image_height, class_index):
    polygons_yolo = []
    try:
        with open(dat_file_path, 'r') as f:
            points_str_list = []
            for line_number, line in enumerate(f):
                try:
                    x_str, y_str = line.strip().split(',')
                    x = float(x_str)
                    y = float(y_str)
                    
                    # Normalize coordinates
                    norm_x = x / image_width
                    norm_y = y / image_height
                    
                    # Clip values to be within [0, 1]
                    norm_x = max(0.0, min(1.0, norm_x))
                    norm_y = max(0.0, min(1.0, norm_y))
                    
                    points_str_list.append(f"{norm_x:.6f} {norm_y:.6f}")
                except ValueError:
                    print(f"警告: 跳过 {dat_file_path} 中格式错误的行 {line_number + 1}: {line.strip()}")
                    continue
            
            if len(points_str_list) > 2: # A polygon needs at least 3 points
                polygon_str = " ".join(points_str_list)
                yolo_line = f"{class_index} {polygon_str}"
                polygons_yolo.append(yolo_line)
            else:
                print(f"警告: {dat_file_path} 中的点不足以形成多边形 (找到 {len(points_str_list)} 个点)。已跳过。")

    except Exception as e:
        print(f"处理 DAT 文件 {dat_file_path} 时出错: {e}")
    return polygons_yolo

def main():
    if not os.path.exists(RAW_DATA_BASE_PATH):
        print(f"错误: 原始数据基本路径未找到: {RAW_DATA_BASE_PATH}")
        return

    for class_name, class_idx in CLASS_MAPPING.items():
        raw_class_path = os.path.join(RAW_DATA_BASE_PATH, class_name)
        
        processed_class_images_path = os.path.join(PROCESSED_DATA_BASE_PATH, class_name, "images")
        processed_class_labels_path = os.path.join(PROCESSED_DATA_BASE_PATH, class_name, "labels")

        # 清理并创建目标目录
        if os.path.exists(processed_class_images_path):
            shutil.rmtree(processed_class_images_path)
        os.makedirs(processed_class_images_path, exist_ok=True)
        
        if os.path.exists(processed_class_labels_path):
            shutil.rmtree(processed_class_labels_path)
        os.makedirs(processed_class_labels_path, exist_ok=True)

        if not os.path.isdir(raw_class_path):
            print(f"原始数据类别路径 {class_name} 未找到: {raw_class_path}，跳过此类别。")
            continue

        print(f"正在处理类别: {class_name}")
        # New: Print total BMP files to be processed for the current class
        all_bmp_files = []
        dat_files_by_base_name = {} # key: base_image_name, value: {'cyt': [list_of_cyt_paths], 'nuc': [list_of_nuc_paths]}

        for filename in os.listdir(raw_class_path):
            file_lower = filename.lower()
            base_name, ext = os.path.splitext(filename)

            if file_lower.endswith(".bmp"):
                all_bmp_files.append(filename)
            elif file_lower.endswith(".dat"):
                parts = base_name.split('_')
                if len(parts) >= 2:
                    image_base_name = "_".join(parts[:-1])
                    dat_type_full = parts[-1].lower() # e.g., "cyt01", "nuc", "cyt"
                    
                    current_dat_type = None
                    if dat_type_full.startswith("cyt"):
                        current_dat_type = "cyt"
                    elif dat_type_full.startswith("nuc"):
                        current_dat_type = "nuc"
                    
                    if current_dat_type:
                        if image_base_name not in dat_files_by_base_name:
                            dat_files_by_base_name[image_base_name] = {'cyt': [], 'nuc': []}
                        dat_files_by_base_name[image_base_name][current_dat_type].append(os.path.join(raw_class_path, filename))
                    # else:
                        # print(f"  警告: 未知 DAT 文件类型或格式 (不以 cyt 或 nuc 开头): {filename}")
                # else:
                    # print(f"  警告: DAT 文件名格式无法解析: {filename}，期望格式 'imagename_cytXX.dat' 或 'imagename_nucXX.dat'")

        if not all_bmp_files:
            print(f"  在 {raw_class_path} 中未找到 BMP 文件。")
            print("-" * 30)
            continue
        
        processed_count_in_class = 0
        for i, bmp_filename in enumerate(all_bmp_files): # Modified
            base_image_name = bmp_filename[:-4] # Remove .bmp extension
            raw_bmp_path = os.path.join(raw_class_path, bmp_filename)
            
            # New: Print progress for each image
            print(f"  [{i+1}/{len(all_bmp_files)}] 正在处理图像: {bmp_filename}")

            try:
                img = Image.open(raw_bmp_path)
                img_width, img_height = img.size
                if img_width == 0 or img_height == 0:
                    print(f"  警告: 图像 {raw_bmp_path} 的尺寸为零。跳过此图像。")
                    continue
            except Exception as e:
                print(f"  打开或读取 BMP 文件 {raw_bmp_path} 时出错: {e}。跳过此图像。")
                continue

            png_filename = base_image_name + ".png"
            processed_png_path = os.path.join(processed_class_images_path, png_filename)
            try:
                img.save(processed_png_path)
                # print(f"  已转换并保存 {raw_bmp_path} 到 {processed_png_path}")
            except Exception as e:
                # Indented error message
                print(f"    保存 PNG {processed_png_path} 时出错: {e}")
                continue
            
            yolo_label_filename = base_image_name + ".txt"
            yolo_label_filepath = os.path.join(processed_class_labels_path, yolo_label_filename)
            
            all_yolo_lines_for_image = []
            
            # 优先使用 _cyt.dat 文件
            cyt_files_to_process = []
            if base_image_name in dat_files_by_base_name:
                if 'cyt' in dat_files_by_base_name[base_image_name] and dat_files_by_base_name[base_image_name]['cyt']:
                    cyt_files_to_process.extend(dat_files_by_base_name[base_image_name]['cyt'])
                    # print(f"  找到 {base_image_name} 的 cyt 文件: {cyt_files_to_process}")
                # elif 'nuc' in dat_files_by_base_name[base_image_name] and dat_files_by_base_name[base_image_name]['nuc']:
                    # pass # 明确跳过nuc，如果只想要cyt的话, 并且nuc不是主要目标
            # else:
                # print(f"  未在 dat_files_by_base_name 字典中找到键 {base_image_name}")

            if cyt_files_to_process:
                for dat_file_path in cyt_files_to_process: # Iterate through all found cyt files for this image
                    yolo_lines = convert_dat_to_yolo_format(dat_file_path, img_width, img_height, class_idx)
                    all_yolo_lines_for_image.extend(yolo_lines) # Add polygons from each cyt file
            
            if all_yolo_lines_for_image:
                with open(yolo_label_filepath, 'w') as f_label:
                    for line in all_yolo_lines_for_image:
                        f_label.write(line + "\n") # <--- 修改在这里
                # Indented success message and count
                print(f"    为 {png_filename} 生成了 YOLO 标签: {yolo_label_filepath} (包含 {len(all_yolo_lines_for_image)} 个对象)")
                processed_count_in_class += 1
            elif base_image_name in dat_files_by_base_name:
                # Indented warning
                print(f"    警告: 未从 {base_image_name} 的 DAT 文件中找到有效多边形。未生成 TXT 文件。")
            # If no DAT files were associated, no TXT file is generated, which is fine.

        # Modified: Print summary for the class
        print(f"类别 {class_name} 处理完成。共处理 {processed_count_in_class} 张有效图片。")
        print("-" * 30)
    print("所有类别 DAT 到 YOLO 的转换已完成。")

if __name__ == "__main__":
    main()

