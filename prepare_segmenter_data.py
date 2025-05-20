import os
import shutil
import cv2
import numpy as np
import glob
import re

def prepare_segmenter_data():
    """
    准备分割器训练数据：
    1. 从原始数据目录递归查找并复制图像和边界文件到训练目录
    2. 验证数据的完整性和可读性
    """
    # 修复中文路径问题：使用原始字符串和os.path模块
    src_root = r"F:\鳞癌自动化\鳞癌数据raw"
    dst_root = r"F:\鳞癌自动化\代码测试\data\segmenter"
    
    # 确保目标目录存在
    os.makedirs(dst_root, exist_ok=True)
    
    # 细胞类型列表
    cell_types = [
        'im_Dyskeratotic', 
        'im_Koilocytotic', 
        'im_Metaplastic', 
        'im_Parabasal', 
        'im_Superficial-Intermediate'
    ]
    
    valid_files = 0
    invalid_files = 0
    
    for cell_type in cell_types:
        src_cell_dir = os.path.join(src_root, cell_type)
        dst_cell_dir = os.path.join(dst_root, cell_type)
        
        # 如果源目录不存在，跳过
        if not os.path.exists(src_cell_dir):
            print(f"警告: 源目录不存在: {src_cell_dir}")
            continue
            
        # 确保目标目录存在
        os.makedirs(dst_cell_dir, exist_ok=True)
        
        print(f"处理 {cell_type} 类型的细胞图像...")
        
        # 递归查找所有.bmp图像文件
        img_paths = []
        for root, dirs, files in os.walk(src_cell_dir):
            for file in files:
                if file.endswith('.bmp') and "CROPPED" not in file:
                    img_paths.append(os.path.join(root, file))
                    
        print(f"找到 {len(img_paths)} 个图像文件")
        
        # 创建一个字典，用于跟踪每个图像的边界文件
        img_data = {}
        
        # 首先查找所有图像和边界文件
        for root, dirs, files in os.walk(src_cell_dir):
            for file in files:
                if file.endswith('.dat'):  # 边界文件
                    full_path = os.path.join(root, file)
                    # 提取基本图像名称
                    match = re.match(r'(.*?)(?:_cyt|_nuc)', file)
                    if match:
                        base_name = match.group(1)
                        if base_name not in img_data:
                            img_data[base_name] = {'cyt_files': [], 'nuc_files': [], 'img_file': None}
                        
                        if '_cyt' in file:
                            img_data[base_name]['cyt_files'].append(full_path)
                        elif '_nuc' in file:
                            img_data[base_name]['nuc_files'].append(full_path)
                
                elif file.endswith('.bmp') and "CROPPED" not in file:
                    full_path = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]
                    if base_name not in img_data:
                        img_data[base_name] = {'cyt_files': [], 'nuc_files': [], 'img_file': None}
                    img_data[base_name]['img_file'] = full_path
        
        # 处理找到的文件
        for base_name, data in img_data.items():
            if data['img_file'] and data['cyt_files'] and data['nuc_files']:
                img_path = data['img_file']
                img_name = os.path.basename(img_path)
                dst_img_path = os.path.join(dst_cell_dir, img_name)
                
                # 显示找到的边界文件数量
                print(f"图像: {img_name}, 找到 {len(data['cyt_files'])} 个细胞质边界文件, {len(data['nuc_files'])} 个细胞核边界文件")
                
                # 尝试读取图像文件验证其完整性
                try:
                    # 使用二进制模式打开文件，确认它是否可访问
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                    
                    # 将二进制数据转换为NumPy数组
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        print(f"警告: 无法解码图像: {img_path}")
                        invalid_files += 1
                        continue
                        
                    # 复制图像文件
                    try:
                        shutil.copy2(img_path, dst_img_path)
                        print(f"成功复制图像: {img_name}")
                    except Exception as e:
                        print(f"复制图像文件时出错 {img_path}: {str(e)}")
                        invalid_files += 1
                        continue
                    
                    # 复制边界文件
                    all_copied = True
                    for cf in data['cyt_files']:
                        cf_name = os.path.basename(cf)
                        dst_cf_path = os.path.join(dst_cell_dir, cf_name)
                        try:
                            shutil.copy2(cf, dst_cf_path)
                        except Exception as e:
                            print(f"复制细胞质文件时出错 {cf}: {str(e)}")
                            all_copied = False
                            
                    for nf in data['nuc_files']:
                        nf_name = os.path.basename(nf)
                        dst_nf_path = os.path.join(dst_cell_dir, nf_name)
                        try:
                            shutil.copy2(nf, dst_nf_path)
                        except Exception as e:
                            print(f"复制细胞核文件时出错 {nf}: {str(e)}")
                            all_copied = False
                    
                    if all_copied:
                        valid_files += 1
                        print(f"成功添加训练样本: {img_name}")
                    else:
                        invalid_files += 1
                
                except Exception as e:
                    print(f"处理文件时出错 {img_path}: {str(e)}")
                    invalid_files += 1
            elif data['img_file']:
                print(f"警告: 图像 {os.path.basename(data['img_file'])} 缺少边界文件")
                invalid_files += 1
    
    print(f"\n数据准备完成!")
    print(f"有效文件: {valid_files}")
    print(f"无效文件: {invalid_files}")
    print(f"\n分割器训练数据路径: {dst_root}")
    
    # 验证数据集大小
    if valid_files == 0:
        print("错误: 没有找到有效的训练数据!")
        print("请检查源目录是否包含正确的数据文件结构。")
        
        # 检查目录结构，查找二级子文件夹
        print("\n检查目录结构:")
        for cell_type in cell_types:
            type_dir = os.path.join(src_root, cell_type)
            if os.path.exists(type_dir):
                subdirs = [d for d in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, d))]
                print(f"  - {cell_type}/ 包含子文件夹: {subdirs}")
                
                # 如果有子文件夹，随机检查一些文件
                for subdir in subdirs[:2]:  # 只检查前两个子文件夹
                    subdir_path = os.path.join(type_dir, subdir)
                    files = os.listdir(subdir_path)[:5]  # 只列出前5个文件
                    print(f"    - {subdir}/ 包含文件: {files}")
    else:
        print("数据集已准备就绪，可以开始训练分割器模型。")
        print("运行命令: python train_segmenter.py")

if __name__ == "__main__":
    prepare_segmenter_data()