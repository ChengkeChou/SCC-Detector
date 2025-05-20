"""
细胞实例分割系统入口脚本
用于启动系统UI或运行命令行任务
"""

import os
import sys
import argparse
from pathlib import Path

def setup_environment():
    """设置环境变量和路径"""
    # 添加项目根目录到路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # 创建必要的目录
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "models", "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "output"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)

def launch_ui():
    """启动图形用户界面"""
    try:
        from ui.cell_segmentation_ui import main
        main()
    except ImportError:
        print("错误: 无法导入UI模块。请确保已安装PyQt5。")
        print("安装命令: pip install PyQt5")
        sys.exit(1)
    except Exception as e:
        print(f"启动UI时出错: {e}")
        sys.exit(1)

def run_preprocessing(args):
    """运行数据预处理"""
    try:
        from utils.dat_to_masks import convert_dataset
        convert_dataset(args.input, args.output, args.format, args.workers)
    except ImportError:
        print("错误: 无法导入预处理模块。")
        sys.exit(1)
    except Exception as e:
        print(f"预处理数据时出错: {e}")
        sys.exit(1)

def run_training(args):
    """运行模型训练"""
    try:
        from models.hybrid_cell_segmentation import train_model
        best_model_path = train_model(
            args.data,
            args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume=args.resume
        )
        print(f"训练完成! 最佳模型保存在: {best_model_path}")
    except ImportError:
        print("错误: 无法导入训练模块。")
        sys.exit(1)
    except Exception as e:
        print(f"训练模型时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_inference(args):
    """运行推理"""
    try:
        from inference import CellSegmentationInference
        
        # 创建推理对象
        inference = CellSegmentationInference(
            model_path=args.model,
            num_classes=args.classes,
            confidence_threshold=args.threshold
        )
        
        if args.image:
            # 处理单张图像
            result = inference.process_image(args.image, args.output, args.show)
            if result:
                print(f"处理完成: {args.image}")
                print(f"检测到{result['total_cells']}个细胞:")
                for class_name, count in result['class_counts'].items():
                    print(f"  - {class_name}: {count}")
        else:
            # 批量处理
            results = inference.process_batch(args.dir, args.output)
            if results:
                print(f"批处理完成: 处理了{len(results)}个图像")
    except ImportError:
        print("错误: 无法导入推理模块。")
        sys.exit(1)
    except Exception as e:
        print(f"推理时出错: {e}")
        sys.exit(1)

def main():
    # 设置环境
    setup_environment()
    
    # 创建解析器
    parser = argparse.ArgumentParser(description="细胞实例分割系统")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # UI命令
    ui_parser = subparsers.add_parser("ui", help="启动图形用户界面")
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser("preprocess", help="预处理数据")
    preprocess_parser.add_argument("--input", type=str, required=True, help="输入数据集目录")
    preprocess_parser.add_argument("--output", type=str, required=True, help="输出目录")
    preprocess_parser.add_argument("--format", type=str, default="yolo", 
                                 choices=["yolo", "coco", "mask"],
                                 help="输出格式")
    preprocess_parser.add_argument("--workers", type=int, default=4, 
                                 help="并行工作线程数")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data", type=str, required=True, help="预处理后的数据集目录")
    train_parser.add_argument("--output", type=str, default="./output", help="输出目录")
    train_parser.add_argument("--epochs", type=int, default=50, help="训练周期数")
    train_parser.add_argument("--batch-size", type=int, default=4, help="批量大小")
    train_parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    train_parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    
    # 推理命令
    inference_parser = subparsers.add_parser("inference", help="运行推理")
    inference_parser.add_argument("--model", type=str, required=True, help="模型路径")
    inference_parser.add_argument("--image", type=str, help="输入图像路径")
    inference_parser.add_argument("--dir", type=str, help="输入图像目录")
    inference_parser.add_argument("--output", type=str, default="./results", help="输出目录")
    inference_parser.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    inference_parser.add_argument("--show", action="store_true", help="显示结果")
    inference_parser.add_argument("--classes", type=int, default=5, help="类别数量")
    
    # 解析参数
    args = parser.parse_args()
    
    # 处理命令
    if args.command == "ui" or not args.command:
        launch_ui()
    elif args.command == "preprocess":
        run_preprocessing(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "inference":
        if not args.image and not args.dir:
            inference_parser.error("必须提供 --image 或 --dir 参数")
        run_inference(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
