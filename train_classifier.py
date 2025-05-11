import os
# 设置环境变量解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset import ClassifierDataset
from classifier import CellClassifier
import matplotlib.pyplot as plt
import numpy as np
from config import (
    CLASSIFIER_DATA,
    CLASSIFY_MODEL_PATHS,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    CLASSIFIER_TRANSFORM
)

def train_classifier(organ_type=None):
    """
    训练细胞分类器
    :param organ_type: 器官类型，如果为None则组合所有数据训练
    """
    try:
        datasets = []
        if organ_type:
            # 单个器官类型训练
            if organ_type not in CLASSIFIER_DATA:
                raise ValueError(f"不支持的器官类型: {organ_type}")
            data_path = CLASSIFIER_DATA[organ_type]
            datasets.append(ClassifierDataset(data_path, transform=CLASSIFIER_TRANSFORM))
            model_save_path = CLASSIFY_MODEL_PATHS[organ_type]
            plot_title = f"{organ_type}细胞分类器训练结果"
            print(f"加载 {organ_type} 分类数据集: {data_path}")
        else:
            # 组合所有数据训练
            print("组合所有器官类型数据进行训练")
            for organ, data_path in CLASSIFIER_DATA.items():
                print(f"加载 {organ} 数据集: {data_path}")
                datasets.append(ClassifierDataset(data_path, transform=CLASSIFIER_TRANSFORM))
            model_save_path = CLASSIFY_MODEL_PATHS['combined']  # 需要在config中添加
            plot_title = "组合细胞分类器训练结果"

        # 组合数据集
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]
        
        print(f"数据集大小: {len(dataset)}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        # 检查GPU是否可用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        if device.type == 'cuda':
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"可用GPU数: {torch.cuda.device_count()}")
            print(f"当前GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

        model = CellClassifier().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 初始化训练跟踪指标
        epoch_losses = []
        epoch_accuracies = []
        
        # 训练循环
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            batch_losses = []
            
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if i % 10 == 9:
                    accuracy = 100. * correct / total
                    print(f'Epoch [{epoch+1}/{EPOCHS}], '
                          f'Step [{i+1}/{len(dataloader)}], '
                          f'Loss: {running_loss/10:.4f}, '
                          f'Accuracy: {accuracy:.2f}%')
                    running_loss = 0.0
            
            # 每个epoch结束后计算指标
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_accuracy = 100. * correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            
            print(f'Epoch [{epoch+1}/{EPOCHS}] - '
                  f'Loss: {epoch_loss:.4f}, '
                  f'Accuracy: {epoch_accuracy:.2f}%')

        print("分类器训练完成")
        torch.save(model.state_dict(), model_save_path)
        
        # 确保即使有警告也会保存可视化图片
        try:
            # 可视化训练结果
            plt.figure(figsize=(12, 5))
            
            # 损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(range(1, EPOCHS + 1), epoch_losses, 'b-', label='训练损失')
            plt.title('训练损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # 准确率曲线
            plt.subplot(1, 2, 2)
            plt.plot(range(1, EPOCHS + 1), epoch_accuracies, 'r-', label='训练准确率')
            plt.title('训练准确率')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.grid(True)
            plt.legend()
            
            plt.suptitle(plot_title)
            plt.tight_layout()
            
            # 创建可视化结果保存目录
            vis_dir = os.path.join(os.path.dirname(model_save_path), "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 保存图像
            model_name = os.path.basename(model_save_path).split('.')[0]
            plt_save_path = os.path.join(vis_dir, f"{model_name}_training_plot.png")
            plt.savefig(plt_save_path)
            print(f"可视化结果已保存至: {plt_save_path}")
            
            # 显示图表（如果在交互式环境中）
            try:
                plt.show()
            except Exception as show_err:
                print(f"无法显示图表（这在非交互式环境中是正常的）: {show_err}")
            
            # 即使plt.show()失败，也确保关闭图表以释放内存
            plt.close()
            
            # 确保路径正确
            print(f"确认可视化图片保存位置: {os.path.abspath(plt_save_path)}")
            
        except Exception as vis_err:
            print(f"可视化保存时发生错误: {vis_err}")
            # 尝试最简单的保存方式
            try:
                simple_plt_save_path = os.path.join(os.path.dirname(model_save_path), f"{model_name}_training_simple.png")
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, EPOCHS + 1), epoch_losses, 'b-', label='训练损失')
                plt.plot(range(1, EPOCHS + 1), epoch_accuracies, 'r-', label='训练准确率')
                plt.title(f"{plot_title} - 简化版")
                plt.legend()
                plt.grid(True)
                plt.savefig(simple_plt_save_path)
                print(f"简化版可视化结果已保存至: {simple_plt_save_path}")
                plt.close()
            except Exception as simple_err:
                print(f"无法保存简化版可视化: {simple_err}")
                
            # 保存原始训练数据到CSV，以便稍后手动绘图
            try:
                import pandas as pd
                csv_dir = os.path.join(os.path.dirname(model_save_path), "training_data")
                os.makedirs(csv_dir, exist_ok=True)
                csv_path = os.path.join(csv_dir, f"{model_name}_training_data.csv")
                
                df = pd.DataFrame({
                    'epoch': list(range(1, EPOCHS + 1)),
                    'loss': epoch_losses,
                    'accuracy': epoch_accuracies
                })
                df.to_csv(csv_path, index=False)
                print(f"训练数据已保存至CSV文件: {csv_path}")
            except Exception as csv_err:
                print(f"无法保存训练数据到CSV: {csv_err}")

    except Exception as e:
        print(f"分类器训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
        # 尝试保存已有数据
        if 'epoch_losses' in locals() and len(epoch_losses) > 0:
            try:
                print("尝试保存已训练的数据...")
                import pandas as pd
                recovery_dir = os.path.join(os.path.dirname(model_save_path) if 'model_save_path' in locals() else ".", "recovery")
                os.makedirs(recovery_dir, exist_ok=True)
                recovery_path = os.path.join(recovery_dir, "training_recovery_data.csv")
                
                epochs_completed = len(epoch_losses)
                df = pd.DataFrame({
                    'epoch': list(range(1, epochs_completed + 1)),
                    'loss': epoch_losses,
                    'accuracy': epoch_accuracies if 'epoch_accuracies' in locals() else [0] * epochs_completed
                })
                df.to_csv(recovery_path, index=False)
                print(f"恢复数据已保存至: {recovery_path}")
            except Exception as recovery_err:
                print(f"恢复数据保存失败: {recovery_err}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--organ', 
                       choices=['Cervical', 'Oral', 'Urethral', 'Esophageal'],
                       help='选择器官类型（可选）')
    args = parser.parse_args()
    
    train_classifier(args.organ)