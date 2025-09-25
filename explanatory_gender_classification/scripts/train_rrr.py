"""
Train gender classification models with Right for Right Reasons (RRR).

Usage:
    python train_rrr.py --model efficientnet_b0 --epochs 20 --l2_grads 1000
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import prepare_data_splits, prepare_data_splits_from_dataset_folder, create_data_loaders, get_mask_directories
from models.rrr_model import RRRGenderClassifier, rrr_loss_function
from training.trainer import EarlyStopper
from utils.helpers import set_random_seeds, get_device, create_directory, count_parameters
from utils.settings import Config


class RRRTrainer:
    """Trainer specifically for RRR models."""
    
    def __init__(self, model, criterion, optimizer, scheduler=None, device='cuda', 
                 l2_grads=1000, class_weights=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.l2_grads = l2_grads
        self.class_weights = class_weights
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_answer_loss': [],
            'train_reason_loss': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': []
        }
        
    def train_epoch(self, train_loader):
        """Train for one epoch with RRR loss."""
        self.model.train()
        
        running_loss = 0.0
        running_answer_loss = 0.0
        running_reason_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, masks, labels in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # Ensure images require gradients for RRR
            images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute RRR loss
            loss, answer_loss, reason_loss = rrr_loss_function(
                A=masks,
                X=images, 
                y=labels,
                logits=outputs,
                criterion=self.criterion,
                class_weights=self.class_weights,
                l2_grads=self.l2_grads
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            running_answer_loss += answer_loss.item() * images.size(0)
            running_reason_loss += reason_loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': loss.item(),
                'Answer': answer_loss.item(),
                'Reason': reason_loss.item()
            })
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_answer_loss = running_answer_loss / len(train_loader.dataset)
        epoch_reason_loss = running_reason_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        return epoch_loss, epoch_acc.item(), epoch_answer_loss, epoch_reason_loss
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def test_epoch(self, test_loader):
        """Test for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for images, masks, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = running_corrects.double() / len(test_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def train(self, train_loader, val_loader, test_loader, epochs, early_stopper=None, save_path=None):
        """Full training loop."""
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 10)
            
            # Training
            train_loss, train_acc, train_answer_loss, train_reason_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Testing
            test_loss, test_acc = self.test_epoch(test_loader)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_answer_loss'].append(train_answer_loss)
            self.history['train_reason_loss'].append(train_reason_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)
            
            # Print statistics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Answer Loss: {train_answer_loss:.4f}, Reason Loss: {train_reason_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved new best model with val_acc: {val_acc:.4f}')
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            if early_stopper:
                if early_stopper.early_stop(val_loss):
                    print("Early stopping triggered")
                    break
                    
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Epoch': range(1, len(self.history['train_loss']) + 1),
            'Train Loss': self.history['train_loss'],
            'Train Accuracy': self.history['train_acc'],
            'Train Answer Loss': self.history['train_answer_loss'],
            'Train Reason Loss': self.history['train_reason_loss'],
            'Validation Loss': self.history['val_loss'],
            'Validation Accuracy': self.history['val_acc'],
            'Test Loss': self.history['test_loss'],
            'Test Accuracy': self.history['test_acc'],
            'Learning Rate': self.history['lr']
        })
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description='Train RRR gender classification model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       choices=Config.SUPPORTED_MODELS,
                       help='Model architecture to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED,
                       help='Random seed for reproducibility')
    
    # RRR specific arguments
    parser.add_argument('--l2_grads', type=float, default=Config.L2_GRADS,
                       help='Lambda coefficient for gradient regularization')
    
    # Data arguments (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--gender_dataset_path', type=str,
                           help='Path to gender_dataset folder containing female/ and male/ subdirs')
    data_group.add_argument('--separate_paths', nargs=2, metavar=('FEMALE_PATH', 'MALE_PATH'),
                           help='Separate paths to female and male image directories')
    parser.add_argument('--masks_dir', type=str, required=False,
                       help='Path to masks directory for RRR training (optional if using gender_dataset_path)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments/rrr',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (defaults to model_rrr)')
    
    # Training options
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=Config.PATIENCE,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Set device
    device = get_device()
    
    # Set experiment name
    if args.experiment_name is None:
        args.experiment_name = f'{args.model}_rrr_l2_{args.l2_grads}'
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    create_directory(output_dir)
    
    print(f"Starting RRR training for {args.model}")
    print(f"L2 gradients lambda: {args.l2_grads}")
    print(f"Output directory: {output_dir}")
    
    # Prepare data
    print("Preparing data splits...")
    
    if args.gender_dataset_path:
        # Use gender_dataset folder with pre-split data
        train_df, val_df, test_df, label_encoder = prepare_data_splits_from_dataset_folder(
            args.gender_dataset_path
        )
        # Get image and mask paths
        female_path = os.path.join(args.gender_dataset_path, 'resized_female_images')
        male_path = os.path.join(args.gender_dataset_path, 'resized_male_images')
        
        # Get mask directories from gender_dataset folder
        try:
            female_masks_path, male_masks_path = get_mask_directories(args.gender_dataset_path)
            print(f"Using masks from gender_dataset:")
            print(f"  Female masks: {female_masks_path}")
            print(f"  Male masks: {male_masks_path}")
        except ValueError as e:
            print(f"Warning: {e}")
            if not args.masks_dir:
                print("Error: No masks available. Please provide either:")
                print("  1. A gender_dataset folder with resized_female_masks/ and resized_male_masks/ directories")
                print("  2. A --masks_dir argument pointing to a masks directory")
                return
            print("Falling back to --masks_dir")
            female_masks_path, male_masks_path = None, None
    else:
        # Use separate paths
        female_path, male_path = args.separate_paths
        train_df, val_df, test_df, label_encoder = prepare_data_splits(
            female_path, male_path
        )
        female_masks_path, male_masks_path = None, None
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create data loaders with masks
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        batch_size=args.batch_size,
        use_masks=True,  # Enable masks for RRR
        masks_dir=args.masks_dir,
        female_path=female_path,
        male_path=male_path,
        female_masks_path=female_masks_path,
        male_masks_path=male_masks_path
    )
    
    # Create RRR model
    print(f"Creating RRR {args.model} model...")
    model = RRRGenderClassifier(
        architecture=args.model,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True
    )
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=Config.SCHEDULER_STEP_SIZE, 
                      gamma=Config.SCHEDULER_GAMMA)
    
    # Create RRR trainer
    trainer = RRRTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        l2_grads=args.l2_grads
    )
    
    # Early stopping
    early_stopper = None
    if args.early_stopping:
        early_stopper = EarlyStopper(patience=args.patience, min_delta=Config.MIN_DELTA)
    
    # Train model
    print("Starting RRR training...")
    model_save_path = os.path.join(output_dir, f'{args.experiment_name}_best.pth')
    
    results_df = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader, 
        test_loader=test_loader,
        epochs=args.epochs,
        early_stopper=early_stopper,
        save_path=model_save_path
    )
    
    # Save results
    results_path = os.path.join(output_dir, f'{args.experiment_name}_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f'{args.experiment_name}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training history
    from utils.helpers import plot_training_history
    plot_path = os.path.join(output_dir, f'{args.experiment_name}_training_plot.png')
    plot_training_history(results_df, save_path=plot_path)
    
    # Print final results
    final_results = results_df.iloc[-1]
    print("\nFinal Results:")
    print(f"Train Accuracy: {final_results['Train Accuracy']:.4f}")
    print(f"  Answer Loss: {final_results['Train Answer Loss']:.4f}")
    print(f"  Reason Loss: {final_results['Train Reason Loss']:.4f}")
    print(f"Validation Accuracy: {final_results['Validation Accuracy']:.4f}")
    print(f"Test Accuracy: {final_results['Test Accuracy']:.4f}")
    
    print(f"\nRRR training completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()