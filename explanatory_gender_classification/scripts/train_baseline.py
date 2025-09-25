"""
Train baseline gender classification models.

Usage:
    python train_baseline.py --model efficientnet_b0 --epochs 20 --batch_size 16
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import prepare_data_splits, prepare_data_splits_from_dataset_folder, create_data_loaders
from models.architectures import create_model
from training.trainer import BaseTrainer, EarlyStopper
from utils.helpers import set_random_seeds, get_device, create_directory
from utils.settings import Config


def main():
    parser = argparse.ArgumentParser(description='Train baseline gender classification model')
    
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
    
    # Data arguments (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--gender_dataset_path', type=str,
                           help='Path to gender_dataset folder containing female/ and male/ subdirs')
    data_group.add_argument('--separate_paths', nargs=2, metavar=('FEMALE_PATH', 'MALE_PATH'),
                           help='Separate paths to female and male image directories')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments/baseline',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (defaults to model name)')
    
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
        args.experiment_name = f'{args.model}_baseline'
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    create_directory(output_dir)
    
    print(f"Starting baseline training for {args.model}")
    print(f"Output directory: {output_dir}")
    
    # Prepare data
    print("Preparing data splits...")
    
    if args.gender_dataset_path:
        # Use gender_dataset folder with pre-split data
        train_df, val_df, test_df, label_encoder = prepare_data_splits_from_dataset_folder(
            args.gender_dataset_path
        )
        # Paths are already included in the DataFrames as 'full_path'
        female_path = os.path.join(args.gender_dataset_path, 'resized_female_images')
        male_path = os.path.join(args.gender_dataset_path, 'resized_male_images')
    else:
        # Use separate paths
        female_path, male_path = args.separate_paths
        train_df, val_df, test_df, label_encoder = prepare_data_splits(
            female_path, male_path
        )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        batch_size=args.batch_size,
        female_path=female_path,
        male_path=male_path
    )
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(
        architecture=args.model,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True
    )
    model = model.to(device)
    
    # Print model info
    from utils.helpers import count_parameters
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=Config.SCHEDULER_STEP_SIZE, 
                      gamma=Config.SCHEDULER_GAMMA)
    
    # Create trainer
    trainer = BaseTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # Early stopping
    early_stopper = None
    if args.early_stopping:
        early_stopper = EarlyStopper(patience=args.patience, min_delta=Config.MIN_DELTA)
    
    # Train model
    print("Starting training...")
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
    print(f"Validation Accuracy: {final_results['Validation Accuracy']:.4f}")
    print(f"Test Accuracy: {final_results['Test Accuracy']:.4f}")
    
    print(f"\nTraining completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()