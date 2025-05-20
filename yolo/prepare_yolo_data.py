import os
import shutil
import random # Added for shuffling

def main():
    base_data_path = r"f:\SSC\cell_segmentation\data"
    yolo_dataset_path = os.path.join(base_data_path, "processed_yolo_dataset")
    val_split_ratio = 0.2 # 20% for validation

    # Define class names based on your folder structure
    class_names = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

    # Output directories for YOLO
    yolo_images_train_path = os.path.join(yolo_dataset_path, "images", "train")
    yolo_labels_train_path = os.path.join(yolo_dataset_path, "labels", "train")
    yolo_images_val_path = os.path.join(yolo_dataset_path, "images", "val")
    yolo_labels_val_path = os.path.join(yolo_dataset_path, "labels", "val")

    # Create output directories if they don't exist
    # Clear existing directories to prevent accumulation from previous runs
    for path in [yolo_images_train_path, yolo_labels_train_path, yolo_images_val_path, yolo_labels_val_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    print(f"目标 YOLO 训练图片目录: {yolo_images_train_path}")
    print(f"目标 YOLO 训练标签目录: {yolo_labels_train_path}")
    print(f"目标 YOLO 验证图片目录: {yolo_images_val_path}")
    print(f"目标 YOLO 验证标签目录: {yolo_labels_val_path}")
    print("-" * 30)

    for class_name in class_names:
        # Define potential source paths
        paths_to_try = [
            # 1. Primary data path with images/labels subdirectories
            (os.path.join(base_data_path, class_name, "images"), 
             os.path.join(base_data_path, class_name, "labels"), 
             f"主数据路径 {os.path.join(base_data_path, class_name)} 的 images/labels 子目录"),
            
            # 2. Primary data path with files directly in class folder
            (os.path.join(base_data_path, class_name), 
             os.path.join(base_data_path, class_name), 
             f"主数据路径 {os.path.join(base_data_path, class_name)} (直接文件)"),
            
            # 3. Raw data path with images/labels subdirectories
            (os.path.join(r"f:\SSC\raw_data\SIPakMed", class_name, "images"), 
             os.path.join(r"f:\SSC\raw_data\SIPakMed", class_name, "labels"), 
             f"原始数据路径 {os.path.join(r'f:\SSC\raw_data\SIPakMed', class_name)} 的 images/labels 子目录"),
            
            # 4. Raw data path with files directly in class folder
            (os.path.join(r"f:\SSC\raw_data\SIPakMed", class_name), 
             os.path.join(r"f:\SSC\raw_data\SIPakMed", class_name), 
             f"原始数据路径 {os.path.join(r'f:\SSC\raw_data\SIPakMed', class_name)} (直接文件)")
        ]

        source_images_path = None
        source_labels_path = None
        path_description = ""

        for img_candidate, lbl_candidate, desc in paths_to_try:
            if os.path.isdir(img_candidate):
                # Check if the image directory actually contains image files
                has_images_in_candidate = False
                for f_check in os.listdir(img_candidate):
                    if f_check.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): # Added .bmp
                        has_images_in_candidate = True
                        break
                
                if not has_images_in_candidate:
                    # print(f"  调试: 路径 {img_candidate} 是目录但没有图片文件，跳过。")
                    continue # No images found in this candidate path, try next

                # If labels are expected in a different directory (subfolder case), that must also be a directory.
                # If img_candidate and lbl_candidate are the same, it means we are checking for files directly in a class folder,
                # so lbl_candidate being a directory is implicitly true (it's same as img_candidate).
                if img_candidate != lbl_candidate and not os.path.isdir(lbl_candidate):
                    # print(f"  调试: 图片目录 {img_candidate} 有效，但标签目录 {lbl_candidate} 不是目录，跳过。")
                    continue # Specified labels subdir not found, try next candidate

                source_images_path = img_candidate
                source_labels_path = lbl_candidate # This could be the same as source_images_path
                path_description = desc
                print(f"正在处理类别: {class_name} 从 {path_description}")
                break # Found a valid path combination
            # else:
                # print(f"  调试: 路径 {img_candidate} 不是目录，跳过。")
        
        if not source_images_path:
            print(f"警告: 类别 {class_name} 在所有检查路径中均未找到有效的数据源。跳过。")
            print("-" * 30)
            continue

        # List image files from the determined source_images_path
        image_files = []
        if os.path.isdir(source_images_path):
            for f_name in os.listdir(source_images_path):
                if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): # Added .bmp
                    image_files.append(f_name)
        else:
            print(f"  警告: 图片源路径 {source_images_path} 不存在。跳过 {class_name} 的图片处理。")
            # continue # If images path doesn't exist, we can't process this class

        if not image_files:
            print(f"  警告: 在 {source_images_path} 中没有找到图片文件。跳过 {class_name}。")
            print("-" * 30)
            continue

        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_split_ratio))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        print(f"  类别 {class_name}: 总计 {len(image_files)} 张图片, 训练 {len(train_files)} 张, 验证 {len(val_files)} 张")

        # Function to copy files
        def copy_files(files, dest_img_path, dest_lbl_path):
            copied_images_count = 0
            copied_labels_count = 0
            for img_filename in files:
                base_filename, _ = os.path.splitext(img_filename)
                label_filename = base_filename + ".txt"

                src_img_file = os.path.join(source_images_path, img_filename)
                src_lbl_file = os.path.join(source_labels_path, label_filename)

                dst_img_file = os.path.join(dest_img_path, img_filename)
                dst_lbl_file = os.path.join(dest_lbl_path, label_filename)

                if os.path.isfile(src_img_file):
                    shutil.copy2(src_img_file, dst_img_file)
                    copied_images_count += 1
                else:
                    print(f"    警告: 图片文件未找到 {src_img_file}")

                if os.path.isfile(src_lbl_file):
                    shutil.copy2(src_lbl_file, dst_lbl_file)
                    copied_labels_count += 1
                else:
                    # This is a common scenario if not all images have corresponding labels
                    # Or if label naming convention is different / labels are in a different place for some files
                    print(f"    信息: 标签文件未找到 {src_lbl_file} (图片: {img_filename}). 这可能是预期的。")
            return copied_images_count, copied_labels_count

        # Copy training files
        img_c, lbl_c = copy_files(train_files, yolo_images_train_path, yolo_labels_train_path)
        print(f"    已复制 {img_c} 张训练图片和 {lbl_c} 个训练标签到训练集目录")

        # Copy validation files
        img_c, lbl_c = copy_files(val_files, yolo_images_val_path, yolo_labels_val_path)
        print(f"    已复制 {img_c} 张验证图片和 {lbl_c} 个验证标签到验证集目录")
        print("-" * 30)

    print("数据准备和划分完成。")
    print(f"请检查以下目录中的文件:")
    print(f"  训练图片: {yolo_images_train_path}")
    print(f"  训练标签: {yolo_labels_train_path}")
    print(f"  验证图片: {yolo_images_val_path}")
    print(f"  验证标签: {yolo_labels_val_path}")

if __name__ == "__main__":
    main()