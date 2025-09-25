"""
Generate explanations (GradCAM and LIME) for trained models.

Usage:
    python generate_explanations.py --model_path experiments/baseline/efficientnet_b0_baseline_best.pth --architecture efficientnet_b0 --method gradcam
"""

import argparse
import torch
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import prepare_data_splits, create_data_loaders
from models.architectures import create_model
from models.rrr_model import RRRGenderClassifier
from evaluation.explainability import (
    GradCAM, LIMEExplainer, get_gradcam_layer,
    generate_explanations_for_dataset
)
from utils.helpers import set_random_seeds, get_device, create_directory
from utils.settings import Config


def main():
    parser = argparse.ArgumentParser(description='Generate explanations for gender classification model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--architecture', type=str, default='efficientnet_b0',
                       choices=Config.SUPPORTED_MODELS,
                       help='Model architecture')
    parser.add_argument('--model_type', type=str, default='baseline',
                       choices=['baseline', 'rrr'],
                       help='Type of model (baseline or RRR)')
    
    # Explanation arguments
    parser.add_argument('--method', type=str, default='both',
                       choices=['gradcam', 'lime', 'both'],
                       help='Explanation method to use')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to explain')
    parser.add_argument('--dataset_split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to explain')
    
    # Data arguments
    parser.add_argument('--female_path', type=str, required=True,
                       help='Path to female images directory')
    parser.add_argument('--male_path', type=str, required=True,
                       help='Path to male images directory')
    parser.add_argument('--masks_dir', type=str, default=None,
                       help='Path to masks directory (required for RRR models)')
    
    # LIME specific arguments
    parser.add_argument('--lime_samples', type=int, default=1000,
                       help='Number of samples for LIME explanation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='explanations',
                       help='Output directory for explanations')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                       help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Set device
    device = get_device()
    
    # Create output directory
    model_name = os.path.basename(args.model_path).replace('.pth', '')
    output_dir = os.path.join(args.output_dir, model_name, args.dataset_split)
    create_directory(output_dir)
    
    print(f"Generating {args.method} explanations for {args.architecture} {args.model_type} model")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {output_dir}")
    
    # Prepare data
    print("Preparing data...")
    train_df, val_df, test_df, label_encoder = prepare_data_splits(
        args.female_path, 
        args.male_path
    )
    
    # Select dataset split
    if args.dataset_split == 'train':
        dataset_df = train_df
    elif args.dataset_split == 'val':
        dataset_df = val_df
    else:
        dataset_df = test_df
    
    print(f"Using {args.dataset_split} split with {len(dataset_df)} samples")
    
    # Create data loaders
    use_masks = (args.model_type == 'rrr')
    if use_masks and args.masks_dir is None:
        raise ValueError("RRR models require --masks_dir argument")
    
    if args.dataset_split == 'train':
        data_loader, _, _ = create_data_loaders(
            train_df, val_df, test_df,
            batch_size=args.batch_size,
            use_masks=use_masks,
            masks_dir=args.masks_dir,
            female_path=args.female_path,
            male_path=args.male_path
        )
    elif args.dataset_split == 'val':
        _, data_loader, _ = create_data_loaders(
            train_df, val_df, test_df,
            batch_size=args.batch_size,
            use_masks=use_masks,
            masks_dir=args.masks_dir,
            female_path=args.female_path,
            male_path=args.male_path
        )
    else:
        _, _, data_loader = create_data_loaders(
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
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Generate explanations using the built-in function
    print(f"Generating {args.method} explanations for {args.num_samples} samples...")
    
    generate_explanations_for_dataset(
        model=model,
        data_loader=data_loader,
        architecture=args.architecture,
        output_dir=output_dir,
        num_samples=args.num_samples,
        method=args.method
    )
    
    print(f"Explanation generation completed! Results saved to {output_dir}")
    
    # Generate individual examples for demonstration
    if args.method in ['gradcam', 'both']:
        print("Generating individual GradCAM examples...")
        
        # Initialize GradCAM
        target_layer = get_gradcam_layer(model, args.architecture)
        gradcam = GradCAM(model, target_layer)
        
        # Process a few individual samples
        count = 0
        for batch_idx, batch in enumerate(data_loader):
            if count >= 5:  # Just generate 5 individual examples
                break
                
            if len(batch) == 3:  # RRR model with masks
                images, masks, labels = batch
            else:  # Standard model
                images, labels = batch
                
            images = images.to(device)
            
            for i in range(images.shape[0]):
                if count >= 5:
                    break
                    
                image = images[i:i+1]
                label = labels[i].item()
                
                # Generate GradCAM
                gradcam_viz, cam = gradcam.visualize_cam(image)
                
                # Save individual example
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image (denormalized)
                img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                for c in range(3):
                    img_np[:, :, c] = img_np[:, :, c] * std[c] + mean[c]
                img_np = torch.clamp(torch.tensor(img_np), 0, 1).numpy()
                
                axes[0].imshow(img_np)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # GradCAM heatmap
                axes[1].imshow(cam, cmap='jet')
                axes[1].set_title('GradCAM Heatmap')
                axes[1].axis('off')
                
                # Overlay
                axes[2].imshow(gradcam_viz)
                axes[2].set_title('GradCAM Overlay')
                axes[2].axis('off')
                
                plt.suptitle(f'Sample {count + 1} - True Label: {label_encoder.classes_[label]}')
                plt.tight_layout()
                
                example_path = os.path.join(output_dir, f'gradcam_example_{count + 1}_label_{label}.png')
                plt.savefig(example_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                count += 1
    
    if args.method in ['lime', 'both']:
        print("Generating individual LIME examples...")
        
        # Initialize LIME
        lime_explainer = LIMEExplainer(model, device)
        
        # Process a few individual samples
        count = 0
        for batch_idx, batch in enumerate(data_loader):
            if count >= 5:  # Just generate 5 individual examples
                break
                
            if len(batch) == 3:  # RRR model with masks
                images, masks, labels = batch
            else:  # Standard model
                images, labels = batch
                
            for i in range(images.shape[0]):
                if count >= 5:
                    break
                    
                image = images[i]
                label = labels[i].item()
                
                # Convert to numpy for LIME (denormalized)
                img_np = image.permute(1, 2, 0).cpu().numpy()
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                for c in range(3):
                    img_np[:, :, c] = img_np[:, :, c] * std[c] + mean[c]
                img_np = (torch.clamp(torch.tensor(img_np), 0, 1).numpy() * 255).astype('uint8')
                
                # Generate LIME explanation
                explanation = lime_explainer.explain_instance(
                    img_np, 
                    num_samples=args.lime_samples
                )
                
                # Visualize explanation
                lime_viz, mask = lime_explainer.visualize_explanation(
                    explanation, img_np, label_idx=label
                )
                
                # Save individual example
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(img_np)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # LIME mask
                axes[1].imshow(mask, cmap='RdYlBu')
                axes[1].set_title('LIME Mask')
                axes[1].axis('off')
                
                # LIME visualization
                axes[2].imshow(lime_viz)
                axes[2].set_title('LIME Explanation')
                axes[2].axis('off')
                
                plt.suptitle(f'Sample {count + 1} - True Label: {label_encoder.classes_[label]}')
                plt.tight_layout()
                
                example_path = os.path.join(output_dir, f'lime_example_{count + 1}_label_{label}.png')
                plt.savefig(example_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                count += 1
    
    print(f"Individual examples generated successfully!")
    print(f"All explanations saved to {output_dir}")


if __name__ == '__main__':
    main()