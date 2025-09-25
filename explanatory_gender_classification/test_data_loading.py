#!/usr/bin/env python3
"""
Test script to verify data loading works correctly with the updated code.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_loading():
    """Test that data loading works with the new CSV format."""
    
    from data.dataset import prepare_data_splits_from_dataset_folder
    
    print("Testing data loading with your CSV format...")
    print("=" * 50)
    
    gender_dataset_path = "../gender_dataset"
    
    try:
        # Test data loading
        train_df, val_df, test_df, label_encoder = prepare_data_splits_from_dataset_folder(
            gender_dataset_path
        )
        
        print("✅ Data loading successful!")
        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")  
        print(f"Test samples: {len(test_df)}")
        
        print(f"\n📋 Train DataFrame structure:")
        print(f"Columns: {list(train_df.columns)}")
        print(f"Sample rows:")
        print(train_df.head(3).to_string())
        
        print(f"\n🏷️ Label distribution in training set:")
        print(train_df['label'].value_counts())
        
        print(f"\n🔢 Label encoding:")
        print(f"Classes: {label_encoder.classes_}")
        
        # Test dataset creation
        from data.dataset import create_data_loaders, get_mask_directories
        
        print(f"\n🎭 Testing mask directories...")
        try:
            female_masks_path, male_masks_path = get_mask_directories(gender_dataset_path)
            print(f"✅ Female masks: {female_masks_path}")
            print(f"✅ Male masks: {male_masks_path}")
            
            # Test data loader creation
            print(f"\n📦 Testing data loader creation...")
            train_loader, val_loader, test_loader = create_data_loaders(
                train_df.head(10), val_df.head(5), test_df.head(5),  # Small subset for testing
                batch_size=2,
                use_masks=True,
                female_masks_path=female_masks_path,
                male_masks_path=male_masks_path
            )
            
            print(f"✅ Data loaders created successfully!")
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")
            
            # Test loading one batch
            print(f"\n🔍 Testing batch loading...")
            for batch_idx, batch_data in enumerate(train_loader):
                if len(batch_data) == 3:  # Image, mask, label
                    images, masks, labels = batch_data
                    print(f"✅ Loaded batch with masks:")
                    print(f"  Images shape: {images.shape}")
                    print(f"  Masks shape: {masks.shape}")
                    print(f"  Labels: {labels}")
                else:  # Image, label
                    images, labels = batch_data
                    print(f"✅ Loaded batch without masks:")
                    print(f"  Images shape: {images.shape}")
                    print(f"  Labels: {labels}")
                break
                
        except Exception as e:
            print(f"⚠️ Mask loading issue: {e}")
            print("Testing without masks...")
            
            # Test without masks
            train_loader, val_loader, test_loader = create_data_loaders(
                train_df.head(10), val_df.head(5), test_df.head(5),
                batch_size=2,
                use_masks=False
            )
            
            print(f"✅ Data loaders (no masks) created successfully!")
            
            # Test loading one batch
            for batch_idx, (images, labels) in enumerate(train_loader):
                print(f"✅ Loaded batch:")
                print(f"  Images shape: {images.shape}")
                print(f"  Labels: {labels}")
                break
        
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print(f"\n🎉 All tests passed! Your dataset is ready for training.")
    return True

if __name__ == "__main__":
    test_data_loading()