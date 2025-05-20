import os
import subprocess
import argparse

# --- 默认训练参数 (可以根据需要修改或通过命令行参数覆盖) ---
DEFAULT_DATA_YAML = r"f:\SSC\cell_segmentation\data\processed_yolo_dataset\sipakmed_seg.yaml"
DEFAULT_MODEL_PT = "yolo11l-seg.pt"  # 或者您训练好的模型的 .pt 文件路径，如 runs/segment/trainX/weights/best.pt
DEFAULT_EPOCHS = 50
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 4  # 从一个较小的值开始，以避免OOM，您可以尝试增加它
DEFAULT_PROJECT_NAME = "CellSegmentRuns"
DEFAULT_RUN_NAME = "train_exp_augmented" # 改个名以区分
DEFAULT_DEVICE = "0"  # 使用 GPU 0

# --- 默认数据增强参数 ---
DEFAULT_DEGREES = 0.0  # 图像旋转角度 (+/- deg)
DEFAULT_TRANSLATE = 0.1 # 图像平移比例 (+/- fraction)
DEFAULT_SCALE = 0.5     # 图像缩放比例 (+/- gain)
DEFAULT_SHEAR = 0.0     # 图像剪切角度 (+/- deg)
DEFAULT_PERSPECTIVE = 0.0 # 图像透视变换 (+/- fraction), range 0-0.001
DEFAULT_FLIPUD = 0.0    # 上下翻转概率 (0.0-1.0)
DEFAULT_FLIPLR = 0.5    # 左右翻转概率 (0.0-1.0)
DEFAULT_MOSAIC = 1.0    # Mosaic 数据增强概率 (0.0-1.0)
DEFAULT_MIXUP = 0.0     # MixUp 数据增强概率 (0.0-1.0)
DEFAULT_COPY_PASTE = 0.0 # Copy-paste 数据增强概率 (segmentation only)

# --- 针对特定错误的建议 ---
# OSError: [WinError 1455] 页面文件太小，无法完成操作。
# 这个错误通常表示 Windows 虚拟内存（页面文件）不足。
# 脚本无法直接解决此问题。您可能需要手动增加 Windows 页面文件的大小。
# 步骤通常是：
# 1. 右键点击“此电脑” -> 属性 -> 高级系统设置 -> “高级”选项卡下的“性能”设置 -> “高级”选项卡 -> “虚拟内存”更改。
# 2. 取消“自动管理所有驱动器的分页文件大小”。
# 3. 选择一个驱动器，设置“自定义大小”，初始大小和最大大小可以设置得大一些（例如，RAM的1.5到3倍）。
# 4. 点击“设置”，然后确定，并重启电脑。

def run_training(data_yaml, model_pt, epochs, img_size, batch_size, project_name, run_name, device,
                 degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste): # 新增增强参数
    """运行 YOLOv8 分割训练"""
    print("--- 开始 YOLOv8 分割训练 ---")
    print(f"数据配置文件: {data_yaml}")
    print(f"模型: {model_pt}")
    print(f"周期: {epochs}")
    print(f"图像尺寸: {img_size}")
    print(f"批次大小: {batch_size}")
    print(f"项目名称: {project_name}")
    print(f"运行名称: {run_name}")
    print(f"设备: {device}")
    print("--- 数据增强参数 ---")
    print(f"Degrees: {degrees}")
    print(f"Translate: {translate}")
    print(f"Scale: {scale}")
    print(f"Shear: {shear}")
    print(f"Perspective: {perspective}")
    print(f"Flipud: {flipud}")
    print(f"Fliplr: {fliplr}")
    print(f"Mosaic: {mosaic}")
    print(f"Mixup: {mixup}")
    print(f"Copy Paste: {copy_paste}")

    # 1. 设置环境变量来尝试解决 OMP 和 CUDA 内存问题
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("设置环境变量: KMP_DUPLICATE_LIB_OK=TRUE")
    print("设置环境变量: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # 2. 构建 YOLO 训练命令
    # yolo executable path might need adjustment if 'yolo' is not directly in PATH
    # For example, if it's a script: ["python", "path/to/yolo_script.py", ...]
    # Assuming 'yolo' is an executable or script accessible via PATH
    command = [
        "yolo",
        "segment",
        "train",
        f"data={data_yaml}",
        f"model={model_pt}",
        f"epochs={epochs}",
        f"imgsz={img_size}",
        f"batch={batch_size}",
        f"project={project_name}",
        f"name={run_name}",
        f"device={device}",
        # 添加数据增强参数到命令中
        f"degrees={degrees}",
        f"translate={translate}",
        f"scale={scale}",
        f"shear={shear}",
        f"perspective={perspective}",
        f"flipud={flipud}",
        f"fliplr={fliplr}",
        f"mosaic={mosaic}",
        f"mixup={mixup}",
        f"copy_paste={copy_paste}"
        # 可以添加其他参数，例如：
        # "patience=20",  # Early stopping patience
        # "optimizer=AdamW",
        # "lr0=0.001",
        # "amp=True" # Mixed precision training (usually default True for YOLOv8)
    ]

    print(f"\n执行命令: {' '.join(command)}\n")

    # 3. 执行命令
    try:
        # 使用 shell=True 可能会更容易找到 yolo 命令，但如果 yolo 在 PATH 中，则不需要。
        # 为了安全性和可移植性，最好避免 shell=True，并确保 yolo 可执行文件在系统 PATH 中，
        # 或者提供 yolo 可执行文件的完整路径。
        # 如果 'yolo' 是通过 anaconda 安装的，它通常在 anaconda 的 Scripts 目录下，该目录应在 PATH 中。
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc == 0:
            print("\n--- 训练成功完成 ---")
        else:
            print(f"\n--- 训练失败，退出代码: {rc} ---")
            if "OutOfMemoryError" in str(process.stderr.read() if process.stderr else "") :
                 print("检测到 OutOfMemoryError。尝试进一步减小批次大小 (batch_size) 或图像尺寸 (img_size)。")
            elif "页面文件太小" in str(process.stderr.read() if process.stderr else ""):
                 print("检测到页面文件太小错误。请参考脚本顶部的注释来增加 Windows 页面文件大小。")

    except FileNotFoundError:
        print("错误: 'yolo' 命令未找到。请确保 Ultralytics YOLOv8 已正确安装并且 'yolo' 在您的系统 PATH 中。")
        print("或者，您可能需要指定 yolo 可执行文件的完整路径。")
    except Exception as e:
        print(f"执行训练时发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 分割训练脚本")
    parser.add_argument("--data_yaml", type=str, default=DEFAULT_DATA_YAML, help="数据 YAML 文件路径")
    parser.add_argument("--model_pt", type=str, default=DEFAULT_MODEL_PT, help="预训练模型 .pt 文件路径或官方模型名称")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练周期数")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE, help="输入图像尺寸")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="批次大小")
    parser.add_argument("--project_name", type=str, default=DEFAULT_PROJECT_NAME, help="YOLO 结果保存的项目文件夹名称")
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME, help="本次训练的运行名称")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="训练设备 (例如, 0, 1, cpu)")

    # 新增数据增强参数的 argparse 定义
    parser.add_argument("--degrees", type=float, default=DEFAULT_DEGREES, help="图像旋转角度 (+/- deg)")
    parser.add_argument("--translate", type=float, default=DEFAULT_TRANSLATE, help="图像平移比例 (+/- fraction)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="图像缩放比例 (+/- gain)")
    parser.add_argument("--shear", type=float, default=DEFAULT_SHEAR, help="图像剪切角度 (+/- deg)")
    parser.add_argument("--perspective", type=float, default=DEFAULT_PERSPECTIVE, help="图像透视变换 (+/- fraction)")
    parser.add_argument("--flipud", type=float, default=DEFAULT_FLIPUD, help="上下翻转概率")
    parser.add_argument("--fliplr", type=float, default=DEFAULT_FLIPLR, help="左右翻转概率")
    parser.add_argument("--mosaic", type=float, default=DEFAULT_MOSAIC, help="Mosaic 数据增强概率")
    parser.add_argument("--mixup", type=float, default=DEFAULT_MIXUP, help="MixUp 数据增强概率")
    parser.add_argument("--copy_paste", type=float, default=DEFAULT_COPY_PASTE, help="Copy-paste 数据增强概率 (仅分割)")

    args = parser.parse_args()

    run_training(
        data_yaml=args.data_yaml,
        model_pt=args.model_pt,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        project_name=args.project_name,
        run_name=args.run_name,
        device=args.device,
        # 传递增强参数
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste
    )
