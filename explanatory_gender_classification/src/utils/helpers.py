"""
Helper functions and utilities.
"""

import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device


def save_checkpoint(model, optimizer, epoch, loss, filename):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filename: Checkpoint filename
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filename: Checkpoint filename
        
    Returns:
        Loaded epoch, loss
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def plot_training_history(history_df, save_path=None):
    """
    Plot training history.
    
    Args:
        history_df: DataFrame containing training history
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    axes[0, 0].plot(history_df['Epoch'], history_df['Train Loss'], label='Training Loss')
    axes[0, 0].plot(history_df['Epoch'], history_df['Validation Loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training and validation accuracy
    axes[0, 1].plot(history_df['Epoch'], history_df['Train Accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history_df['Epoch'], history_df['Validation Accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Test accuracy
    if 'Test Accuracy' in history_df.columns:
        axes[1, 0].plot(history_df['Epoch'], history_df['Test Accuracy'], label='Test Accuracy', color='red')
        axes[1, 0].set_title('Test Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate
    if 'Learning Rate' in history_df.columns:
        axes[1, 1].plot(history_df['Epoch'], history_df['Learning Rate'], label='Learning Rate', color='orange')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of labels
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    return {classes[i]: weights[i] for i in range(len(classes))}


def create_directory(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def get_model_summary(model, input_size):
    """
    Get model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
        
    Returns:
        Model summary string
    """
    from torchsummary import summary
    
    try:
        summary_str = summary(model, input_size, verbose=0)
        return summary_str
    except Exception as e:
        return f"Could not generate summary: {e}"


def format_time(seconds):
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"