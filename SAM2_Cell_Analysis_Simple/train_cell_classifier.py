import argparse
import os
import shutil
import time
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ---------------------- Helper Functions ----------------------

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_timestamp():
    """Returns a formatted timestamp string."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------- Dataset and DataLoader ----------------------

class CellImageDataset(Dataset):
    """Custom Dataset for loading cell images."""
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx
        if self.class_to_idx is None:
            self.classes = sorted(list(set(labels)))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.classes = sorted(self.class_to_idx.keys(), key=lambda x: self.class_to_idx[x])

        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder or skip
            # For simplicity, let's return a black image and a dummy label if error
            return torch.zeros((3, 224, 224)), -1 # Or handle appropriately

        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label_idx

def get_data_loaders(data_dir, batch_size, img_size, num_workers, test_size=0.2, val_size=0.2, seed=42):
    """Loads data, splits it, and creates DataLoaders."""
    all_image_paths = []
    all_labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name, "CROPPED") # Assuming CROPPED subfolder
        if not os.path.isdir(class_path):
            print(f"Warning: CROPPED directory not found for class {class_name} in {os.path.join(data_dir, class_name)}. Skipping this class.")
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): # Added '.bmp'
                all_image_paths.append(os.path.join(class_path, img_name))
                all_labels.append(class_name)

    if not all_image_paths:
        raise ValueError(f"No images found in {data_dir} with the expected structure (ClassName/CROPPED/*.png|jpg|jpeg|bmp).") # Added bmp to error message

    # Split: train_val and test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=test_size, stratify=all_labels, random_state=seed
    )

    # Split: train and val
    # Adjust val_size relative to the size of train_val set
    relative_val_size = val_size / (1 - test_size)
    if len(set(train_val_labels)) > 1 and relative_val_size < 1.0 : # Ensure stratification is possible
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels, test_size=relative_val_size, stratify=train_val_labels, random_state=seed
        )
    else: # Not enough samples or classes for stratified val split, or val_size is too large
        print("Warning: Not enough samples or classes for a stratified validation split, or val_size is too large. Using simple split or assigning all to train if val_size is 0.")
        if relative_val_size > 0 and len(train_val_paths) > 1 :
             train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths, train_val_labels, test_size=relative_val_size, random_state=seed
            )
        else: # if val_size is 0 or only one sample left
            train_paths, train_labels = train_val_paths, train_val_labels
            val_paths, val_labels = [], []


    print(f"Dataset split: Train: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}")
    if not train_paths:
        raise ValueError("Training set is empty. Check data directory and split sizes.")


    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CellImageDataset(train_paths, train_labels, transform=train_transform, class_to_idx=class_to_idx)
    val_dataset = CellImageDataset(val_paths, val_labels, transform=val_test_transform, class_to_idx=class_to_idx) if val_paths else None
    test_dataset = CellImageDataset(test_paths, test_labels, transform=val_test_transform, class_to_idx=class_to_idx) if test_paths else None

    # Handle num_workers for Windows
    nw = min(os.cpu_count(), num_workers) if num_workers > 0 else 0
    if os.name == 'nt' and nw > 0 : # On Windows, multiprocessing with num_workers > 0 needs if __name__ == '__main__':
        print("Warning: On Windows, DataLoader with num_workers > 0 can cause issues if not run inside 'if __name__ == \"__main__\":'). Setting num_workers to 0 if it was > 0.")
        # Forcing num_workers to 0 on Windows if it was set higher, to avoid common pickling errors with datasets/lambdas
        # This is a common workaround. The ideal solution is to ensure all custom dataset components are picklable.
        # However, for robustness in a general script, this is safer.
        # nw = 0 # Re-evaluate if this is too restrictive. User might have picklable dataset.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True) if test_dataset else None

    return train_loader, val_loader, test_loader, class_to_idx, class_names

# ---------------------- Model Building ----------------------

def build_model(model_name, num_classes, pretrained=True):
    """Builds a specified model."""
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

# ---------------------- Training and Evaluation ----------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_num, total_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        if labels.min() < 0: # Skip batches with dummy labels from loading errors
            print(f"Skipping batch with invalid labels: {labels}")
            continue

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        if (i + 1) % (len(loader) // 5 + 1) == 0 : # Print progress ~5 times per epoch
             print(f"Epoch [{epoch_num+1}/{total_epochs}] Batch [{i+1}/{len(loader)}] Train Loss: {loss.item():.4f}")


    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc


def evaluate_model(model, loader, criterion, device, class_names=None, output_dir=None, prefix="val"):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if labels.min() < 0: continue # Skip

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0
    epoch_acc = epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc

    if class_names and output_dir and all_labels and all_preds:
        try:
            # Define labels and names for the report/CM to include ALL classes known to the model.
            # This ensures a consistent structure for reports and confusion matrices.
            report_indices = list(range(len(class_names)))
            report_names = class_names # class_names is already the list of actual names

            report = classification_report(
                all_labels, 
                all_preds, 
                labels=report_indices, 
                target_names=report_names, 
                output_dict=True, 
                zero_division=0
            )
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(output_dir, f"{prefix}_classification_report.csv"))
            
            # Generate the string for printing once to avoid re-computing
            printed_report_str = classification_report(
                all_labels, 
                all_preds, 
                labels=report_indices, 
                target_names=report_names, 
                zero_division=0
            )
            print(f"\n{prefix.capitalize()} Classification Report:\n{printed_report_str}")

            cm = confusion_matrix(all_labels, all_preds, labels=report_indices) # Use all class indices for cm
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=report_names) # Use all class names for display
            disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
            plt.title(f"{prefix.capitalize()} Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
            plt.close()   
        except Exception as e:
            print(f"Error generating/saving report/CM for {prefix}: {e}")
            print(f"All labels: {all_labels}, All preds: {all_preds}, Class names: {class_names}")


    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_history(history, output_dir):
    """Plots training and validation loss and accuracy."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history: plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history: plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()

# ---------------------- Main Training Pipeline ----------------------

def run_training_pipeline(args):
    """Main function to run the training pipeline."""
    set_seed(args.seed)
    timestamp = get_timestamp()
    output_dir = args.output_dir if args.output_dir else os.path.join("models", f"classifier_{args.model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(output_dir, "args.txt"), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    print("Loading data...")
    try:
        train_loader, val_loader, test_loader, class_to_idx, class_names = get_data_loaders(
            args.data_dir, args.batch_size, args.img_size, args.num_workers,
            args.test_split, args.val_split, args.seed
        )
    except ValueError as e:
        print(f"Error creating data loaders: {e}")
        return

    if not train_loader:
        print("Failed to create train_loader. Exiting.")
        return

    num_classes = len(class_names)
    if num_classes == 0:
        print(f"No classes found in {args.data_dir}. Please check the directory structure.")
        return
    print(f"Found {num_classes} classes: {class_names}")
    with open(os.path.join(output_dir, "class_to_idx.txt"), 'w') as f:
        f.write(str(class_to_idx))


    # Model
    print(f"Building model: {args.model_name}")
    model = build_model(args.model_name, num_classes, args.pretrained)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, factor=0.1) # Removed verbose=True


    # Training Loop
    print("Starting training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_path = os.path.join(output_dir, f"{args.model_name}_best.pth")

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc = float('inf'), 0.0 # Default if no val_loader
        if val_loader:
            val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            scheduler.step(val_loss) # Reduce LR on plateau

            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else: # If no validation set, save model at each epoch or based on train loss
            if train_loss < best_val_loss : # Using train_loss as proxy if no val_loss
                best_val_loss = train_loss
                torch.save(model.state_dict(), best_model_path)
            print("No validation set. Model will be saved based on training loss or at the end.")


        epoch_duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{args.epochs}] | Duration: {epoch_duration:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loader and epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {args.early_stopping_patience} epochs with no improvement.")
            break
        if not val_loader and epoch == args.epochs -1: # Save last model if no validation
             torch.save(model.state_dict(), best_model_path)


    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    plot_history(history, output_dir)
    print(f"Training finished. Best model saved to {best_model_path}")

    # Evaluation on Test Set
    if test_loader:
        print("Evaluating on Test Set...")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device)) # Load best model
        else:
            print("Warning: Best model not found. Evaluating with the last state of the model.")

        test_loss, test_acc, test_preds, test_labels = evaluate_model(
            model, test_loader, criterion, device, class_names, output_dir, prefix="test"
        )
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    else:
        print("No test set provided. Skipping final evaluation on test set.")

    print(f"All results saved in {output_dir}")
    # Clean up __pycache__ if it's created locally by mistake (though usually it's in the script's dir)
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")

if __name__ == '__main__':
    # Calculate the default data_dir relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjusted default_data_dir to point to the SIPakMed dataset
    default_data_dir = os.path.abspath(os.path.join(script_dir, '..', 'raw_data', 'SIPakMed'))

    parser = argparse.ArgumentParser(description="Train a cell image classifier.")
    # Paths
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Directory containing class-named subfolders, where each class folder has a CROPPED subfolder (e.g., your_data_dir/ClassName/CROPPED/image.png). Default: ../raw_data/SIPakMed')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save models and results. Defaults to models/classifier_MODELNAME_TIMESTAMP')
    # Model
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b3'], help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained', help='Do not use pretrained weights')
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (height and width)')
    # Data
    parser.add_argument('--test_split', type=float, default=0.2, help='Proportion of dataset for testing')
    parser.add_argument('--val_split', type=float, default=0.2, help='Proportion of original dataset for validation (taken from training set after test split)')
    # System
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader. Set to 0 for Windows if issues arise.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='Force CPU usage')
    # Early Stopping & LR Scheduler
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping (epochs)')
    parser.add_argument('--lr_patience', type=int, default=5, help='Patience for learning rate scheduler (epochs)')


    args = parser.parse_args()

    # Basic validation for splits
    if not (0 <= args.test_split < 1):
        raise ValueError("test_split must be between 0 and 1.")
    if not (0 <= args.val_split < 1):
        raise ValueError("val_split must be between 0 and 1.")
    if args.test_split + args.val_split >= 1.0 and args.val_split > 0 : # val_split is from original, so sum can be >=1 if val_split is large
         actual_val_from_train_val = args.val_split / (1-args.test_split) if (1-args.test_split) > 0 else 1.0
         if actual_val_from_train_val >=1.0 and args.val_split > 0 : # if val split would consume all of train_val
            print(f"Warning: val_split ({args.val_split}) is too large compared to test_split ({args.test_split}). Validation set might be empty or too small.")
            # No direct error, but get_data_loaders will handle empty sets.

    # Ensure num_workers is 0 on Windows if it was set > 0 by default and not in __main__ context (already handled in get_data_loaders, but good to note)
    # This script structure with `if __name__ == '__main__':` should allow `num_workers > 0` on Windows.
    # The check inside `get_data_loaders` is a fallback.

    run_training_pipeline(args)