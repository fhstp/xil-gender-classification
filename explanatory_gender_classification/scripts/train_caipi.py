"""
CAIPI training script for gender classification.

This script implements CAIPI-only training (without RRR regularization)
as described in the paper experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import os
import sys
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import GenderDataset
from models.cnn_models import create_model
from explainability.gradcam import GradCAMWrapper
from explainability.bla import BLAWrapper, create_bla_model
from augmentation.caipi import apply_caipi_sampling, CAIPIAugmentation
from evaluation.bias_metrics import evaluate_model_bias, BiasMetricsTracker


def train_caipi_model(model, train_dataset, val_dataset, explainer, 
                     k=3, num_samples=50, sampling_strategy='uncertainty',
                     num_epochs=20, batch_size=32, device='cpu', save_path=None):
    """
    Train model using CAIPI augmentation only (no RRR).
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        explainer: Explainability method
        k: Number of counterexamples per sample
        num_samples: Number of samples to select for CAIPI
        sampling_strategy: 'uncertainty' or 'high_confidence'
        num_epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        save_path: Path to save model
    """
    model = model.to(device)
    
    print(f"Training CAIPI model with k={k}, strategy={sampling_strategy}")
    
    # Step 1: Pre-train model on original data (baseline training)
    print("Phase 1: Pre-training on original dataset...")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Pre-train for a few epochs
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_dataloader, desc=f"Pre-training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'forward') and len(model(images)) == 2:
                logits, _ = model(images)  # BLA model
            else:
                logits = model(images)
                
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_dataloader)
        print(f"Pre-training Epoch {epoch+1}: Loss {avg_loss:.4f}, Accuracy {accuracy:.2f}%")
    
    # Step 2: Generate CAIPI augmented dataset
    print("Phase 2: Generating CAIPI augmented samples...")
    
    augmented_samples = apply_caipi_sampling(
        dataset=train_dataset,
        model=model,
        explainer=explainer,
        num_samples=num_samples,
        k=k,
        sampling_strategy=sampling_strategy,
        device=device
    )
    
    # Create augmented dataset
    aug_images = []
    aug_labels = []
    
    for img, _, label in augmented_samples:
        aug_images.append(img)
        aug_labels.append(label)
    
    # Convert to tensor dataset
    from torch.utils.data import TensorDataset
    aug_images_tensor = torch.stack(aug_images)
    aug_labels_tensor = torch.tensor(aug_labels, dtype=torch.long)
    augmented_dataset = TensorDataset(aug_images_tensor, aug_labels_tensor)
    
    # Combine original and augmented datasets
    combined_dataset = ConcatDataset([train_dataset, augmented_dataset])
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Combined dataset size: {len(combined_dataset)} "
          f"(Original: {len(train_dataset)}, Augmented: {len(augmented_dataset)})")
    
    # Step 3: Fine-tune on combined dataset
    print("Phase 3: Fine-tuning on combined dataset...")
    
    metrics_tracker = BiasMetricsTracker()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(combined_dataloader, desc=f"Fine-tuning Epoch {epoch+1}"):
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, _ = batch  # If masks are included
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'forward') and len(model(images)) == 2:
                logits, _ = model(images)  # BLA model
            else:
                logits = model(images)
                
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(combined_dataloader)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Accuracy {accuracy:.2f}%")
        
        # Evaluate bias metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Evaluating bias metrics...")
            bias_metrics = evaluate_model_bias(
                model=model,
                dataloader=val_dataloader,
                explainer=explainer,
                device=device
            )
            
            metrics_tracker.update(bias_metrics)
            print(f"FFP: {bias_metrics['FFP']:.3f}, BFP: {bias_metrics['BFP']:.3f}, "
                  f"BSR: {bias_metrics['BSR']:.3f}, DICE: {bias_metrics['DICE']:.3f}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_model_bias(
        model=model,
        dataloader=val_dataloader,
        explainer=explainer,
        device=device
    )
    
    print("\nFinal Bias Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Save model
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'k': k,
                'num_samples': num_samples,
                'sampling_strategy': sampling_strategy,
                'num_epochs': num_epochs
            },
            'final_metrics': final_metrics,
            'metrics_history': metrics_tracker.metrics_history
        }, save_path)
        print(f"Model saved to: {save_path}")
    
    return model, final_metrics


def main():
    parser = argparse.ArgumentParser(description='Train CAIPI model for gender classification')
    parser.add_argument('--data_dir', type=str, default='gender_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                       choices=['efficientnet_b0', 'vgg16', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--explainer', type=str, default='gradcam',
                       choices=['gradcam', 'bla'], help='Explainability method')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of counterexamples per sample')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to select for CAIPI')
    parser.add_argument('--sampling_strategy', type=str, default='uncertainty',
                       choices=['uncertainty', 'high_confidence'],
                       help='Sampling strategy')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='caipi_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    from data.dataset import prepare_data_splits_from_dataset_folder
    from torchvision import transforms
    
    # Prepare data splits
    train_df, val_df, test_df, label_encoder = prepare_data_splits_from_dataset_folder(args.data_dir)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = GenderDataset(train_df, transform=train_transform)
    val_dataset = GenderDataset(val_df, transform=val_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"Creating {args.model_name} model...")
    if args.explainer == 'bla':
        model = create_bla_model(args.model_name, num_classes=2)
        explainer = model.get_explanation_wrapper()
    else:
        model = create_model(args.model_name, num_classes=2, pretrained=True)
        explainer = GradCAMWrapper(model, target_layer_name='features')
    
    # Set up save path
    save_path = os.path.join(
        args.save_dir, 
        f'caipi_{args.explainer}_{args.sampling_strategy}_k{args.k}.pth'
    )
    
    # Train model
    trained_model, final_metrics = train_caipi_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        explainer=explainer,
        k=args.k,
        num_samples=args.num_samples,
        sampling_strategy=args.sampling_strategy,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        save_path=save_path
    )
    
    print(f"\nTraining completed! Results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()