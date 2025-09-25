#!/usr/bin/env python3
"""
Data verification script for gender_dataset folder.

This script helps you verify that your gender_dataset folder is properly set up
and provides statistics about your dataset.
"""

import os
import sys
from collections import Counter

def get_image_files(directory):
    """Get list of image files in a directory."""
    if not os.path.exists(directory):
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = []
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            files.append(filename)
    
    return sorted(files)

def get_mask_files(directory):
    """Get list of mask files in a directory (JPG files)."""
    if not os.path.exists(directory):
        return []
    
    # Masks are stored as JPG files in your dataset
    mask_extensions = {'.jpg', '.jpeg', '.png'}
    files = []
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in mask_extensions):
            files.append(filename)
    
    return sorted(files)

def analyze_filenames(files):
    """Analyze filename patterns."""
    extensions = Counter()
    lengths = []
    
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        extensions[ext] += 1
        lengths.append(len(filename))
    
    return extensions, lengths

def verify_gender_dataset(gender_dataset_path):
    """Verify the structure and content of gender_dataset folder."""
    
    print("Gender Dataset Verification")
    print("=" * 50)
    
    if not os.path.exists(gender_dataset_path):
        print(f"❌ Error: Gender dataset path does not exist: {gender_dataset_path}")
        return False
    
    print(f"✅ Dataset path exists: {gender_dataset_path}")
    
    # Check required subdirectories
    required_dirs = [
        'dataset_split',
        'resized_female_images',
        'resized_female_masks', 
        'resized_male_images',
        'resized_male_masks'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(gender_dataset_path, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name} directory exists")
        else:
            print(f"❌ {dir_name} directory missing")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    # Check CSV files in dataset_split
    dataset_split_path = os.path.join(gender_dataset_path, 'dataset_split')
    required_csvs = ['train_set.csv', 'val_set.csv', 'test_set.csv']
    
    for csv_file in required_csvs:
        csv_path = os.path.join(dataset_split_path, csv_file)
        if os.path.exists(csv_path):
            print(f"✅ {csv_file} exists")
        else:
            print(f"❌ {csv_file} missing in dataset_split")
            return False
    
    # Get image files
    female_path = os.path.join(gender_dataset_path, 'resized_female_images')
    male_path = os.path.join(gender_dataset_path, 'resized_male_images')
    female_files = get_image_files(female_path)
    male_files = get_image_files(male_path)
    
    # Check mask files
    female_masks_path = os.path.join(gender_dataset_path, 'resized_female_masks')
    male_masks_path = os.path.join(gender_dataset_path, 'resized_male_masks')
    female_masks = get_mask_files(female_masks_path)
    male_masks = get_mask_files(male_masks_path)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Female images: {len(female_files)}")
    print(f"Male images: {len(male_files)}")
    print(f"Total images: {len(female_files) + len(male_files)}")
    print(f"Female masks: {len(female_masks)}")
    print(f"Male masks: {len(male_masks)}")
    print(f"Total masks: {len(female_masks) + len(male_masks)}")
    
    if len(female_files) == 0:
        print("⚠️  Warning: No female images found!")
    
    if len(male_files) == 0:
        print("⚠️  Warning: No male images found!")
    
    # Check CSV data
    try:
        import pandas as pd
        train_df = pd.read_csv(os.path.join(dataset_split_path, 'train_set.csv'))
        val_df = pd.read_csv(os.path.join(dataset_split_path, 'val_set.csv'))
        test_df = pd.read_csv(os.path.join(dataset_split_path, 'test_set.csv'))
        
        print(f"\n📋 CSV Split Information:")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Total CSV samples: {len(train_df) + len(val_df) + len(test_df)}")
        
        # Show sample of CSV structure
        if not train_df.empty:
            print(f"\n📝 CSV Structure (sample from train_set.csv):")
            print(f"Columns: {list(train_df.columns)}")
            print(f"First few rows:")
            print(train_df.head(3).to_string())
            
    except Exception as e:
        print(f"⚠️  Warning: Could not read CSV files: {e}")
    
    # Class balance
    total_images = len(female_files) + len(male_files)
    if total_images > 0:
        female_ratio = len(female_files) / total_images
        male_ratio = len(male_files) / total_images
        
        print(f"\n⚖️  Class Balance:")
        print(f"Female: {female_ratio:.1%} ({len(female_files)} images)")
        print(f"Male: {male_ratio:.1%} ({len(male_files)} images)")
        
        if abs(female_ratio - 0.5) > 0.2:  # More than 20% imbalance
            print("⚠️  Warning: Significant class imbalance detected!")
    
    # Analyze file types
    if female_files:
        female_ext, female_lengths = analyze_filenames(female_files)
        print(f"\n📁 Female Images:")
        print(f"File extensions: {dict(female_ext)}")
        print(f"Filename length range: {min(female_lengths)}-{max(female_lengths)} characters")
        print(f"Sample files: {female_files[:5]}")
    
    if male_files:
        male_ext, male_lengths = analyze_filenames(male_files)
        print(f"\n📁 Male Images:")
        print(f"File extensions: {dict(male_ext)}")
        print(f"Filename length range: {min(male_lengths)}-{max(male_lengths)} characters")
        print(f"Sample files: {male_files[:5]}")
    
    # Check for potential issues
    print(f"\n🔍 Potential Issues Check:")
    
    # Check for duplicate filenames across classes
    common_names = set(female_files) & set(male_files)
    if common_names:
        print(f"⚠️  Warning: {len(common_names)} duplicate filenames across classes: {list(common_names)[:5]}...")
    else:
        print("✅ No duplicate filenames across classes")
    
    # Check for very small dataset
    if total_images < 100:
        print("⚠️  Warning: Very small dataset. Consider using more images for better results.")
    elif total_images < 1000:
        print("⚠️  Note: Small dataset. Results may vary significantly.")
    else:
        print("✅ Dataset size looks good for training")
    
    print(f"\n🎯 Ready for Training:")
    if total_images > 0 and len(female_files) > 0 and len(male_files) > 0:
        print("✅ Dataset is ready for training!")
        
        # Suggest train/val/test splits
        train_size = int(total_images * 0.7)
        val_size = int(total_images * 0.15)
        test_size = total_images - train_size - val_size
        
        print(f"\n📈 Suggested data splits (70/15/15):")
        print(f"Training: {train_size} images")
        print(f"Validation: {val_size} images") 
        print(f"Testing: {test_size} images")
        
        return True
    else:
        print("❌ Dataset is not ready for training")
        return False

def main():
    """Main verification function."""
    
    # Default path - adjust this to point to your gender_dataset folder
    default_path = "../gender_dataset"
    
    if len(sys.argv) > 1:
        gender_dataset_path = sys.argv[1]
    else:
        gender_dataset_path = default_path
        print(f"Using default path: {gender_dataset_path}")
        print("You can also specify a path: python verify_data.py /path/to/gender_dataset")
        print()
    
    success = verify_gender_dataset(gender_dataset_path)
    
    if success:
        print(f"\n🚀 Next steps:")
        print(f"1. Run training: python run_example.py")
        print(f"2. Or use custom command:")
        print(f"   python scripts/train_baseline.py --gender_dataset_path {gender_dataset_path} --model efficientnet_b0")
    else:
        print(f"\n🔧 Please fix the issues above before proceeding with training.")

if __name__ == "__main__":
    main()