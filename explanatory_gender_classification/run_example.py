#!/usr/bin/env python3
"""
Example usage script for the gender classification project with gender_dataset folder.

This script demonstrates how to:
1. Set up data from the gender_dataset folder
2. Train a baseline model
3. Train an RRR model
4. Evaluate the models
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and print its description."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*50)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error output: {result.stderr}")
        return False
    else:
        print("Command completed successfully!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

def main():
    """Main example workflow."""
    
    # Configuration
    GENDER_DATASET_PATH = "../gender_dataset"  # Adjust this path to your gender_dataset folder
    MASKS_DIR = "../masks"  # Adjust this path to your masks folder (for RRR training)
    
    # Model configurations to try
    models = ["efficientnet_b0"]  # You can add more models: "resnet50", "densenet121", etc.
    
    print("Gender Classification Training Example")
    print("=" * 50)
    
    # Check if paths exist
    if not os.path.exists(GENDER_DATASET_PATH):
        print(f"Error: Gender dataset path does not exist: {GENDER_DATASET_PATH}")
        print("Please update the GENDER_DATASET_PATH variable in this script.")
        return
        
    # Check for new structure
    dataset_split_path = os.path.join(GENDER_DATASET_PATH, 'dataset_split')
    female_path = os.path.join(GENDER_DATASET_PATH, 'resized_female_images')
    male_path = os.path.join(GENDER_DATASET_PATH, 'resized_male_images')
    
    if os.path.exists(dataset_split_path) and os.path.exists(female_path) and os.path.exists(male_path):
        print(f"Using structured gender dataset from: {GENDER_DATASET_PATH}")
        print("Dataset structure:")
        print(f"  ✅ dataset_split/ (with CSV files)")
        print(f"  ✅ resized_female_images/ ({len([f for f in os.listdir(female_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])} files)")
        print(f"  ✅ resized_male_images/ ({len([f for f in os.listdir(male_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])} files)")
        
        # Check masks
        female_masks_path = os.path.join(GENDER_DATASET_PATH, 'resized_female_masks')
        male_masks_path = os.path.join(GENDER_DATASET_PATH, 'resized_male_masks')
        if os.path.exists(female_masks_path) and os.path.exists(male_masks_path):
            female_mask_count = len([f for f in os.listdir(female_masks_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            male_mask_count = len([f for f in os.listdir(male_masks_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  ✅ resized_female_masks/ ({female_mask_count} files)")
            print(f"  ✅ resized_male_masks/ ({male_mask_count} files)")
    else:
        print(f"Error: Required directories not found in {GENDER_DATASET_PATH}")
        print("Expected structure:")
        print(f"  {GENDER_DATASET_PATH}/")
        print("    ├── dataset_split/")
        print("    ├── resized_female_images/")
        print("    ├── resized_male_images/")
        print("    ├── resized_female_masks/")
        print("    └── resized_male_masks/")
        return
    
    for model in models:
        # Train baseline model
        baseline_cmd = [
            "python", "scripts/train_baseline.py",
            "--gender_dataset_path", GENDER_DATASET_PATH,
            "--model", model,
            "--epochs", "5",  # Reduced for quick testing
            "--batch_size", "16",
            "--output_dir", f"experiments/example_{model}"
        ]
        
        success = run_command(baseline_cmd, f"Training baseline {model} model")
        if not success:
            print(f"Skipping RRR training for {model} due to baseline training failure")
            continue
        
        # Train RRR model (masks are automatically detected from gender_dataset structure)
        female_masks_path = os.path.join(GENDER_DATASET_PATH, 'resized_female_masks')
        male_masks_path = os.path.join(GENDER_DATASET_PATH, 'resized_male_masks')
        
        if os.path.exists(female_masks_path) and os.path.exists(male_masks_path):
            rrr_cmd = [
                "python", "scripts/train_rrr.py",
                "--gender_dataset_path", GENDER_DATASET_PATH,
                "--model", model,
                "--epochs", "5",  # Reduced for quick testing
                "--batch_size", "16",
                "--l2_grads", "1000",
                "--output_dir", f"experiments/example_rrr_{model}"
            ]
            
            run_command(rrr_cmd, f"Training RRR {model} model")
        elif os.path.exists(MASKS_DIR):
            # Fall back to external masks directory
            rrr_cmd = [
                "python", "scripts/train_rrr.py",
                "--gender_dataset_path", GENDER_DATASET_PATH,
                "--masks_dir", MASKS_DIR,
                "--model", model,
                "--epochs", "5",  # Reduced for quick testing
                "--batch_size", "16",
                "--l2_grads", "1000",
                "--output_dir", f"experiments/example_rrr_{model}"
            ]
            
            run_command(rrr_cmd, f"Training RRR {model} model")
        else:
            print(f"\nSkipping RRR training - no masks found in dataset or external directory")
    
    print("\n" + "="*50)
    print("Example training completed!")
    print("Check the experiments/ directory for results.")
    print("="*50)

if __name__ == "__main__":
    main()