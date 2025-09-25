"""
Train hybrid XIL models combining CAIPI and RRR.

Usage:
    python train_hybrid.py --data_dir ../gender_dataset --k 3 --sampling uncertainty --explainer gradcam
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import prepare_data_splits_from_dataset_folder, create_data_loaders
from models.architectures import create_model
from explainability.gradcam import GradCAMWrapper
from explainability.bla import create_bla_model
from training.hybrid_trainer import HybridXILTrainer
from evaluation.bias_metrics import evaluate_model_bias
from utils.helpers import set_random_seeds, get_device, create_directory


def main():
    parser = argparse.ArgumentParser(description='Train hybrid XIL gender classification model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to gender dataset directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                       choices=['densenet121', 'efficientnet_b0', 'googlenet', 
                               'mobilenet_v2', 'resnet50', 'vgg16'],
                       help='Model architecture')
    
    # Hybrid training arguments
    parser.add_argument('--k', type=int, default=3,
                       help='Number of CAIPI counterexamples per sample')
    parser.add_argument('--sampling', type=str, default='uncertainty',
                       choices=['uncertainty', 'high_confidence'],
                       help='CAIPI sampling strategy')
    parser.add_argument('--explainer', type=str, default='gradcam',
                       choices=['gradcam', 'bla'],
                       help='Explainability method')
    parser.add_argument('--num_caipi_samples', type=int, default=100,
                       help='Number of samples to select for CAIPI augmentation')
    parser.add_argument('--rrr_lambda', type=float, default=10.0,
                       help='RRR regularization weight')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments/hybrid',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (defaults to auto-generated)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Get device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f'{args.model}_hybrid_k{args.k}_{args.sampling}_{args.explainer}_rrr{args.rrr_lambda}'
    
    # Create output directory
    output_path = os.path.join(args.output_dir, args.experiment_name)
    create_directory(output_path)
    print(f"Output directory: {output_path}")
    
    print(f"Starting hybrid XIL training for {args.model}")
    print(f"Configuration: k={args.k}, sampling={args.sampling}, explainer={args.explainer}, RRR λ={args.rrr_lambda}")
    
    try:
        # Prepare data splits
        print("Preparing data splits...")
        train_df, val_df, test_df, label_encoder = prepare_data_splits_from_dataset_folder(args.data_dir)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_df, val_df, test_df,
            batch_size=args.batch_size,
            use_masks=True  # Required for RRR
        )
        
        print(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create base model
        model = create_model(args.model, num_classes=2, pretrained=True)
        
        # Create explainer based on choice
        if args.explainer == 'gradcam':
            explainer = GradCAMWrapper(model)
        else:  # bla
            explainer = create_bla_model(args.model, num_classes=2)
            model = explainer  # BLA model replaces the base model
        
        # Create hybrid trainer
        hybrid_trainer = HybridXILTrainer(
            model=model,
            base_dataset=train_loader.dataset,  # Pass the dataset
            explainer=explainer,
            device=device,
            learning_rate=args.lr,
            rrr_lambda=args.rrr_lambda,
            caipi_k=args.k,
            num_caipi_samples=args.num_caipi_samples,
            sampling_strategy=args.sampling
        )
        
        print("Starting hybrid training...")
        
        # Train the hybrid model
        history = hybrid_trainer.train(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            eval_dataloader=val_loader,
            save_path=os.path.join(output_path, 'best_model.pt')
        )
        
        print("Training completed! Evaluating model...")
        
        # Evaluate the trained model
        model.eval()
        test_accuracy = 0.0
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, masks, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        test_loss = test_loss / len(test_loader)
        
        # Evaluate bias metrics
        print("Computing bias metrics...")
        bias_metrics = evaluate_model_bias(
            model=model,
            test_loader=test_loader,
            explainer=explainer,
            device=device
        )
        
        # Save results
        results = {
            'experiment_config': {
                'model': args.model,
                'k': args.k,
                'sampling_strategy': args.sampling,
                'explainer': args.explainer,
                'rrr_lambda': args.rrr_lambda,
                'num_caipi_samples': args.num_caipi_samples,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'seed': args.seed
            },
            'final_metrics': {
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                **bias_metrics
            },
            'training_history': history
        }
        
        # Save results to JSON
        results_file = os.path.join(output_path, 'results.json')
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"HYBRID XIL TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Model: {args.model}")
        print(f"Configuration: k={args.k}, {args.sampling}, {args.explainer}, RRR λ={args.rrr_lambda}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"FFP: {bias_metrics.get('ffp', 'N/A'):.3f}")
        print(f"BFP: {bias_metrics.get('bfp', 'N/A'):.3f}")
        print(f"BSR: {bias_metrics.get('bsr', 'N/A'):.3f}")
        print(f"DICE: {bias_metrics.get('dice', 'N/A'):.3f}")
        print(f"\nResults saved to: {output_path}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)