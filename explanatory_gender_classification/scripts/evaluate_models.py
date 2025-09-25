"""
Evaluate trained gender classification models.

Usage:
    python evaluate_models.py --model_path experiments/baseline/efficientnet_b0_baseline_best.pth --architecture efficientnet_b0
"""

import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import prepare_data_splits, create_data_loaders
from models.architectures import create_model
from models.rrr_model import RRRGenderClassifier
from utils.helpers import set_random_seeds, get_device, plot_confusion_matrix
from utils.settings import Config


def evaluate_model(model, test_loader, device, label_encoder):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device for inference
        label_encoder: Label encoder for class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # RRR model with masks
                images, masks, labels = batch
            else:  # Standard model
                images, labels = batch
                
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels)
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Classification report
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate gender classification model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--architecture', type=str, default='efficientnet_b0',
                       choices=Config.SUPPORTED_MODELS,
                       help='Model architecture')
    parser.add_argument('--model_type', type=str, default='baseline',
                       choices=['baseline', 'rrr'],
                       help='Type of model (baseline or RRR)')
    
    # Data arguments
    parser.add_argument('--female_path', type=str, required=True,
                       help='Path to female images directory')
    parser.add_argument('--male_path', type=str, required=True,
                       help='Path to male images directory')
    parser.add_argument('--masks_dir', type=str, default=None,
                       help='Path to masks directory (required for RRR models)')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                       help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Set device
    device = get_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Evaluating {args.architecture} {args.model_type} model")
    print(f"Model path: {args.model_path}")
    
    # Prepare data
    print("Preparing data...")
    train_df, val_df, test_df, label_encoder = prepare_data_splits(
        args.female_path, 
        args.male_path
    )
    
    # Create data loaders
    use_masks = (args.model_type == 'rrr')
    if use_masks and args.masks_dir is None:
        raise ValueError("RRR models require --masks_dir argument")
    
    _, _, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        batch_size=args.batch_size,
        use_masks=use_masks,
        masks_dir=args.masks_dir,
        female_path=args.female_path,
        male_path=args.male_path
    )
    
    # Load model
    print("Loading model...")
    if args.model_type == 'baseline':
        model = create_model(
            architecture=args.architecture,
            num_classes=Config.NUM_CLASSES,
            pretrained=False,
            freeze_backbone=False
        )
    else:  # RRR model
        model = RRRGenderClassifier(
            architecture=args.architecture,
            num_classes=Config.NUM_CLASSES,
            pretrained=False,
            freeze_backbone=False
        )
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    print(f"Model loaded successfully")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, label_encoder)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"\nClassification Report:")
    
    report_df = pd.DataFrame(results['classification_report']).transpose()
    print(report_df)
    
    # Save results
    model_name = os.path.basename(args.model_path).replace('.pth', '')
    
    # Save classification report
    report_path = os.path.join(args.output_dir, f'{model_name}_classification_report.csv')
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")
    
    # Save detailed results
    detailed_results = pd.DataFrame({
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'female_prob': results['probabilities'][:, 0],
        'male_prob': results['probabilities'][:, 1]
    })
    
    detailed_path = os.path.join(args.output_dir, f'{model_name}_detailed_results.csv')
    detailed_results.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to {detailed_path}")
    
    # Plot and save confusion matrix
    cm_path = os.path.join(args.output_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(
        results['labels'], 
        results['predictions'],
        results['class_names'],
        save_path=cm_path
    )
    
    # Calculate per-class metrics
    print(f"\nPer-class Metrics:")
    for i, class_name in enumerate(results['class_names']):
        class_mask = results['labels'] == i
        class_accuracy = np.mean(results['predictions'][class_mask] == results['labels'][class_mask])
        print(f"{class_name}: {class_accuracy:.4f}")
    
    # Calculate bias metrics (difference in accuracy between classes)
    female_mask = results['labels'] == 0  # Assuming female is class 0
    male_mask = results['labels'] == 1    # Assuming male is class 1
    
    female_accuracy = np.mean(results['predictions'][female_mask] == results['labels'][female_mask])
    male_accuracy = np.mean(results['predictions'][male_mask] == results['labels'][male_mask])
    
    bias_metric = abs(female_accuracy - male_accuracy)
    
    print(f"\nBias Metrics:")
    print(f"Female Accuracy: {female_accuracy:.4f}")
    print(f"Male Accuracy: {male_accuracy:.4f}")
    print(f"Accuracy Difference (bias metric): {bias_metric:.4f}")
    
    # Save summary metrics
    summary_metrics = {
        'model_path': args.model_path,
        'architecture': args.architecture,
        'model_type': args.model_type,
        'overall_accuracy': results['accuracy'],
        'female_accuracy': female_accuracy,
        'male_accuracy': male_accuracy,
        'bias_metric': bias_metric,
        'precision_macro': results['classification_report']['macro avg']['precision'],
        'recall_macro': results['classification_report']['macro avg']['recall'],
        'f1_macro': results['classification_report']['macro avg']['f1-score']
    }
    
    summary_df = pd.DataFrame([summary_metrics])
    summary_path = os.path.join(args.output_dir, f'{model_name}_summary_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()