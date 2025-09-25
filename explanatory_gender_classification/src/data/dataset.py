"""
Dataset classes and data loading utilities for gender classification.
"""

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class GenderDataset(Dataset):
    """Custom dataset for gender classification."""
    
    def __init__(self, data, transform=None, mask_transform=None, use_masks=False, 
                 masks_dir=None, female_path=None, male_path=None, 
                 female_masks_path=None, male_masks_path=None):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame containing image paths and labels
            transform: Image transformations
            mask_transform: Mask transformations for RRR training
            use_masks: Whether to load masks for RRR training
            masks_dir: Directory containing masks (legacy support)
            female_path: Path to female images directory
            male_path: Path to male images directory
            female_masks_path: Path to female masks directory
            male_masks_path: Path to male masks directory
        """
        # Keep DataFrame as is, don't convert to values
        self.data = data
        self.transform = transform
        self.mask_transform = mask_transform
        self.use_masks = use_masks
        self.masks_dir = masks_dir  # Legacy support
        self.female_path = female_path
        self.male_path = male_path
        self.female_masks_path = female_masks_path
        self.male_masks_path = male_masks_path
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Handle both DataFrame and array inputs
        if hasattr(self.data, 'iloc'):
            # DataFrame input
            row = self.data.iloc[idx]
            image_name = row['image'] if 'image' in row else row.iloc[0]
            label = row['encoded_label'] if 'encoded_label' in row else (row['label'] if 'label' in row else row.iloc[1])
            image_path = row.get('full_path', image_name)
            
            # Ensure we have the correct data types
            image_name = str(image_name)
            image_path = str(image_path)
            if isinstance(label, str):
                # Convert string labels to encoded values if needed
                if hasattr(self, 'label_encoder'):
                    label = self.label_encoder.transform([label])[0]
                else:
                    # Manual encoding for common cases
                    label = 0 if label.lower() == 'female' else 1
            else:
                label = int(label)
        else:
            # Array input (legacy)
            image_name = str(self.data[idx][0])
            label = int(self.data[idx][1])
            
            # Construct full image path
            if self.female_path and self.male_path:
                if 'female' in str(label).lower():
                    image_path = os.path.join(self.female_path, image_name)
                else:
                    image_path = os.path.join(self.male_path, image_name)
            else:
                image_path = image_name
            
        # Load image
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        if not self.use_masks:
            return image, label
            
        # Load mask for RRR training
        mask = None
        if self.use_masks:
            # Keep the original extension for masks (they are JPG files)
            mask_name = os.path.basename(str(image_name))
            
            # Try separate mask directories first (new structure)
            if self.female_masks_path and self.male_masks_path:
                if hasattr(self.data, 'iloc'):
                    # Get label from DataFrame
                    row = self.data.iloc[idx]
                    current_label = row.get('label', 'unknown')
                else:
                    current_label = str(label).lower()
                
                if 'female' in str(current_label).lower():
                    mask_path = os.path.join(str(self.female_masks_path), mask_name)
                else:
                    mask_path = os.path.join(str(self.male_masks_path), mask_name)
            
            # Fall back to single masks directory (legacy)
            elif self.masks_dir:
                mask_path = os.path.join(str(self.masks_dir), mask_name)
            else:
                mask_path = None
            
            if mask_path:
                try:
                    mask = Image.open(mask_path).convert('L')  # Grayscale
                    if self.mask_transform:
                        mask = self.mask_transform(mask)
                    else:
                        mask = transforms.ToTensor()(mask)
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
                    # Create dummy mask
                    mask = torch.zeros((1, image.shape[1], image.shape[2]))
            else:
                # Create dummy mask if no mask path available
                mask = torch.zeros((1, image.shape[1], image.shape[2]))
                
        return image, mask, label


def create_data_loaders(train_df, val_df, test_df, batch_size=16, use_masks=False,
                       masks_dir=None, female_path=None, male_path=None,
                       female_masks_path=None, male_masks_path=None):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        test_df: Test DataFrame
        batch_size: Batch size for data loaders
        use_masks: Whether to use masks for RRR training
        masks_dir: Directory containing masks
        female_path: Path to female images
        male_path: Path to male images
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]) if use_masks else None
    
    # Create datasets
    train_dataset = GenderDataset(
        train_df, 
        transform=train_transform,
        mask_transform=mask_transform,
        use_masks=use_masks,
        masks_dir=masks_dir,
        female_path=female_path,
        male_path=male_path,
        female_masks_path=female_masks_path,
        male_masks_path=male_masks_path
    )
    
    val_dataset = GenderDataset(
        val_df, 
        transform=val_test_transform,
        mask_transform=mask_transform,
        use_masks=use_masks,
        masks_dir=masks_dir,
        female_path=female_path,
        male_path=male_path,
        female_masks_path=female_masks_path,
        male_masks_path=male_masks_path
    )
    
    test_dataset = GenderDataset(
        test_df, 
        transform=val_test_transform,
        mask_transform=mask_transform,
        use_masks=use_masks,
        masks_dir=masks_dir,
        female_path=female_path,
        male_path=male_path,
        female_masks_path=female_masks_path,
        male_masks_path=male_masks_path
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def prepare_data_splits_from_dataset_folder(gender_dataset_path):
    """
    Prepare train, validation, and test splits from gender_dataset folder with pre-split CSV files.
    
    Args:
        gender_dataset_path: Path to gender_dataset folder containing:
            - dataset_split/ (with train_set.csv, val_set.csv, test_set.csv)
            - resized_female_images/
            - resized_female_masks/
            - resized_male_images/
            - resized_male_masks/
        
    Returns:
        Tuple of (train_df, val_df, test_df, label_encoder)
    """
    dataset_split_path = os.path.join(gender_dataset_path, 'dataset_split')
    female_images_path = os.path.join(gender_dataset_path, 'resized_female_images')
    male_images_path = os.path.join(gender_dataset_path, 'resized_male_images')
    
    # Check if required directories exist
    if not os.path.exists(dataset_split_path):
        raise ValueError(f"dataset_split folder not found in {gender_dataset_path}")
    
    if not os.path.exists(female_images_path) or not os.path.exists(male_images_path):
        raise ValueError(f"Image folders not found in {gender_dataset_path}")
    
    # Load pre-split CSV files
    train_csv = os.path.join(dataset_split_path, 'train_set.csv')
    val_csv = os.path.join(dataset_split_path, 'val_set.csv')
    test_csv = os.path.join(dataset_split_path, 'test_set.csv')
    
    if not all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
        raise ValueError(f"Required CSV files not found in {dataset_split_path}")
    
    # Load dataframes
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"Loaded pre-split data:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Add full paths to images and extract labels from paths
    def add_full_paths(df):
        df = df.copy()
        
        # Extract label from image path and create full path
        def process_row(row):
            image_path = row['image']
            
            # Extract gender from path
            if 'female' in image_path.lower():
                gender = 'female'
                # Create full path - the image path already includes the directory
                full_path = os.path.join(gender_dataset_path, image_path)
            elif 'male' in image_path.lower():
                gender = 'male'
                # Create full path - the image path already includes the directory
                full_path = os.path.join(gender_dataset_path, image_path)
            else:
                # Fallback: try to infer from filename
                if 'f_' in image_path.lower():
                    gender = 'female'
                    full_path = os.path.join(female_images_path, os.path.basename(image_path))
                else:
                    gender = 'male'
                    full_path = os.path.join(male_images_path, os.path.basename(image_path))
            
            return pd.Series({
                'img_id': row['img_id'],
                'image': os.path.basename(image_path),  # Just the filename
                'image_path': image_path,  # Original path from CSV
                'full_path': full_path,  # Absolute path to image file
                'label': gender
            })
        
        df = df.apply(process_row, axis=1)
        
        return df
    
    train_df = add_full_paths(train_df)
    val_df = add_full_paths(val_df)
    test_df = add_full_paths(test_df)
    
    # Create label encoder and encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    
    # Fit on all labels
    all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']])
    label_encoder.fit(all_labels)
    
    # Encode labels
    train_df['encoded_label'] = label_encoder.transform(train_df['label'])
    val_df['encoded_label'] = label_encoder.transform(val_df['label'])
    test_df['encoded_label'] = label_encoder.transform(test_df['label'])
    
    # Add image ID if not present
    for df in [train_df, val_df, test_df]:
        if 'img_id' not in df.columns:
            df['img_id'] = df['image'].apply(
                lambda x: os.path.splitext(x)[0].lstrip('0') if os.path.splitext(x)[0].lstrip('0') else '0'
            )
    
    print(f"Class distribution:")
    print("Training:")
    print(train_df['label'].value_counts())
    print("Validation:")
    print(val_df['label'].value_counts()) 
    print("Test:")
    print(test_df['label'].value_counts())
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), label_encoder


def get_mask_directories(gender_dataset_path):
    """
    Get mask directories from gender_dataset folder.
    
    Args:
        gender_dataset_path: Path to gender_dataset folder
        
    Returns:
        Tuple of (female_masks_path, male_masks_path)
    """
    female_masks_path = os.path.join(gender_dataset_path, 'resized_female_masks')
    male_masks_path = os.path.join(gender_dataset_path, 'resized_male_masks')
    
    if not os.path.exists(female_masks_path) or not os.path.exists(male_masks_path):
        raise ValueError(f"Mask directories not found in {gender_dataset_path}")
    
    return female_masks_path, male_masks_path


def prepare_data_splits(female_path, male_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare train, validation, and test splits from image directories.
    
    Args:
        female_path: Path to female images directory
        male_path: Path to male images directory
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df, label_encoder)
    """
    
    # Get file lists
    female_files = [f for f in os.listdir(female_path) if f.endswith('.jpg')]
    male_files = [f for f in os.listdir(male_path) if f.endswith('.jpg')]
    
    # Sort files
    female_files.sort()
    male_files.sort()
    
    # Create DataFrames
    df_female = pd.DataFrame(female_files, columns=['image'])
    df_female['label'] = 'female'
    
    df_male = pd.DataFrame(male_files, columns=['image'])
    df_male['label'] = 'male'
    
    # Combine DataFrames
    df = pd.concat([df_female, df_male], ignore_index=True)
    
    # Add image ID
    df['img_id'] = df['image'].apply(
        lambda x: x.split('.')[0].lstrip('0') if x.split('.')[0].lstrip('0') else '0'
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])
    
    # Split data
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, 
        stratify=df['label']
    )
    
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=random_state,
        stratify=train_val['label']
    )
    
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), label_encoder