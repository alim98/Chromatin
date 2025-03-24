import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from dataloader.nuclei_dataloader import get_nuclei_dataloader
from model.vgg3d import Vgg3D, load_model_from_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune VGG3D model on nuclei dataset')
    
    # Data params
    parser.add_argument('--data_dir', type=str, default=config.DATA_ROOT, 
                        help='Path to nuclei dataset')
    parser.add_argument('--class_csv', type=str, default=config.CLASS_CSV_PATH,
                        help='Path to class CSV file')
    parser.add_argument('--output_dir', type=str, default=config.VISUALIZATION_OUTPUT_DIR,
                        help='Directory to save results and model checkpoints')
    parser.add_argument('--class_id', type=int, nargs='+', default=None,
                        help='Filter by class ID(s). If not specified, all classes will be included.')
    
    # Model params
    parser.add_argument('--checkpoint', type=str, default='model/hemibrain_production.checkpoint',
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--target_size', type=int, nargs=3, default=[80, 80, 80],
                        help='Target size for volumes (depth, height, width)')
    parser.add_argument('--output_classes', type=int, default=None,
                        help='Number of output classes. If not specified, will be determined from the dataset')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam',
                        help='Optimizer to use for training')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--freeze_features', action='store_true', 
                        help='Freeze feature extractor layers and only train classifier')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data to use for training (vs validation)')
    
    # Training process
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda/cpu)')
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, criterion, optimizer, args):
    """
    Train the model with the given parameters
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        args: Command line arguments
        
    Returns:
        Trained model and training history
    """
    device = torch.device(args.device)
    model = model.to(device)
    
    # For early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # History to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch in progress_bar:
            # Get data
            volumes = batch['volume']
            labels = batch['label']
            
            # Move to device
            volumes = torch.stack(volumes).to(device)
            labels = torch.tensor(labels).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * volumes.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for batch in progress_bar:
                # Get data
                volumes = batch['volume']
                labels = batch['label']
                
                # Move to device
                volumes = torch.stack(volumes).to(device)
                labels = torch.tensor(labels).to(device)
                
                # Forward pass
                outputs = model(volumes)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * volumes.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Save predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Check for improvement for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save the model checkpoint
            checkpoint_path = os.path.join(args.output_dir, f'vgg3d_finetuned_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f'  Saved best model checkpoint to {checkpoint_path}')
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def plot_training_history(history, save_path):
    """
    Plot the training history
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved training history plot to {save_path}')

def evaluate_model(model, test_loader, args):
    """
    Evaluate the model on the test set
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # Get data
            volumes = batch['volume']
            labels = batch['label']
            sample_ids = batch['metadata']['sample_id']
            
            # Move to device
            volumes = torch.stack(volumes).to(device)
            labels = torch.tensor(labels).to(device)
            
            # Forward pass
            outputs = model(volumes)
            _, predicted = torch.max(outputs, 1)
            
            # Save predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sample_ids.extend(sample_ids)
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Create results dictionary
    results = {
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': list(zip(all_sample_ids, all_labels, all_preds))
    }
    
    return results

def save_evaluation_results(results, save_dir):
    """
    Save evaluation results to files
    
    Args:
        results: Dictionary with evaluation metrics
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = results['confusion_matrix']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = np.unique(np.array([label for _, label, _ in results['predictions']]))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    # Save predictions to CSV
    with open(os.path.join(save_dir, 'predictions.csv'), 'w') as f:
        f.write('sample_id,true_label,predicted_label\n')
        for sample_id, true_label, pred_label in results['predictions']:
            f.write(f'{sample_id},{true_label},{pred_label}\n')
    
    # Save classification report
    report = results['classification_report']
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write('Class\tPrecision\tRecall\tF1-Score\tSupport\n')
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            f.write(f"{class_name}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t{metrics['f1-score']:.4f}\t{metrics['support']}\n")
        
        f.write(f"\nAccuracy: {report['accuracy']:.4f}\n")
        f.write(f"Macro Avg: {report['macro avg']['precision']:.4f}\t{report['macro avg']['recall']:.4f}\t{report['macro avg']['f1-score']:.4f}\t{report['macro avg']['support']}\n")
        f.write(f"Weighted Avg: {report['weighted avg']['precision']:.4f}\t{report['weighted avg']['recall']:.4f}\t{report['weighted avg']['f1-score']:.4f}\t{report['weighted avg']['support']}\n")
    
    print(f'Saved evaluation results to {save_dir}')

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Define transforms as functions instead of nn.Sequential with lambdas
    def transform(x):
        x_tensor = torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0)  # Add channel dimension
        if x_tensor.max() > 1.0:
            x_tensor = x_tensor / 255.0  # Normalize to [0, 1] if needed
        return x_tensor
    
    def mask_transform(x):
        if x is None:
            return None
        x_tensor = torch.from_numpy(x).float()
        x_tensor = x_tensor.unsqueeze(0)  # Add channel dimension
        return x_tensor
    
    # Split dataset by sample ID for train/val
    dataset = get_nuclei_dataloader(
        root_dir=args.data_dir,
        batch_size=1,  # Get full dataset first
        transform=None,
        mask_transform=None,
        class_csv_path=args.class_csv,
        filter_by_class=args.class_id,
        return_paths=True,
        load_volumes=True,
        target_size=tuple(args.target_size)
    ).dataset
    
    # Get all sample IDs
    sample_ids = []
    class_counts = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_id = sample['metadata']['sample_id']
        label = sample['label']
        
        sample_ids.append((sample_id, label))
        
        # Count class distribution
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # Print class distribution
    print("Class distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  Class {class_id}: {count} samples")
    
    # Shuffle and split by sample ID to avoid data leakage
    np.random.shuffle(sample_ids)
    train_size = int(len(sample_ids) * args.train_split)
    train_sample_ids = [item[0] for item in sample_ids[:train_size]]
    val_sample_ids = [item[0] for item in sample_ids[train_size:]]
    
    print(f"Training samples: {len(train_sample_ids)}")
    print(f"Validation samples: {len(val_sample_ids)}")
    
    # Create data loaders
    train_loader = get_nuclei_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        transform=transform,
        mask_transform=mask_transform,
        class_csv_path=args.class_csv,
        filter_by_class=args.class_id,
        sample_ids=train_sample_ids,
        return_paths=True,
        load_volumes=True,
        target_size=tuple(args.target_size),
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = get_nuclei_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        transform=transform,
        mask_transform=mask_transform,
        class_csv_path=args.class_csv,
        filter_by_class=args.class_id,
        sample_ids=val_sample_ids,
        return_paths=True,
        load_volumes=True,
        target_size=tuple(args.target_size),
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # Determine number of output classes if not specified
    if args.output_classes is None:
        args.output_classes = len(class_counts)
        print(f"Auto-detected {args.output_classes} output classes")
    
    # Create and load model
    model = Vgg3D(
        input_size=tuple(args.target_size),
        output_classes=args.output_classes,
        input_fmaps=1  # Single channel input
    )
    
    # Load pre-trained weights if checkpoint exists
    if os.path.exists(args.checkpoint):
        model = load_model_from_checkpoint(model, args.checkpoint)
        print(f"Loaded pre-trained model from {args.checkpoint}")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Starting with random weights.")
    
    # Freeze feature extractor if requested
    if args.freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
        print("Froze feature extractor layers - only training classifier")
    
    # Update the classifier output layer to match our number of classes
    if model.classifier[-1].out_features != args.output_classes:
        print(f"Changing classifier output from {model.classifier[-1].out_features} to {args.output_classes} classes")
        # Create a new Linear layer while keeping the model's existing weights
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, args.output_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:  # adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # Train the model
    print("Starting training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, args)
    
    # Plot training history
    plot_training_history(history, os.path.join(args.output_dir, 'training_history.png'))
    
    # Evaluate the model on validation set
    print("Evaluating model...")
    results = evaluate_model(model, val_loader, args)
    
    # Save evaluation results
    save_evaluation_results(results, args.output_dir)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'vgg3d_finetuned_final.pth')
    torch.save({
        'epochs': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    print("Finetuning complete!")

if __name__ == "__main__":
    main() 