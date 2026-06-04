"""XIL-aware dataset that returns images, labels, and feedback masks for training."""
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from .feedback_masks import bbox_to_foreground_mask, mask_to_bbox_mask

class XILGenderDataset(Dataset):
    """
    Standardized dataset for journal extension.
    Always returns:
    image, label, feedback_mask, foreground_mask, metadata
    """
    
    def __init__(
        self,
        data,
        transform=None,
        mask_transform=None,
        use_masks=False,
        mask_source="segmentation",  # segmentation | bbox | bbox_from_segmentation
        bbox_column=None,
        female_masks_path=None,
        male_masks_path=None
    ):
        self.data = data
        self.transform = transform
        self.mask_transform = mask_transform
        self.use_masks = use_masks
        self.mask_source = mask_source
        self.bbox_column = bbox_column
        self.female_masks_path = female_masks_path
        self.male_masks_path = male_masks_path
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.data.iloc[idx]
        image_name = str(row.get('image', ''))
        label = row.get('encoded_label', row.get('label'))
        image_path = str(row.get('full_path', image_name))
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        orig_size = image.size # (W, H)
            
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        metadata = row.to_dict()
        
        if not self.use_masks:
            # Return dummy masks if not using masks
            H, W = image_tensor.shape[1], image_tensor.shape[2]
            return image_tensor, label, torch.ones((1, H, W)), torch.ones((1, H, W)), metadata
            
        # Determine masks
        feedback_mask = None
        foreground_mask = None
        
        if self.mask_source in ["segmentation", "bbox_from_segmentation"]:
            # Load from mask files
            mask_name = os.path.basename(image_name)
            current_label = str(row.get('label', 'unknown')).lower()
            
            mask_path = None
            if self.female_masks_path and self.male_masks_path:
                if 'female' in current_label:
                    mask_path = os.path.join(str(self.female_masks_path), mask_name)
                else:
                    mask_path = os.path.join(str(self.male_masks_path), mask_name)
                    
            if mask_path and os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
                if self.mask_transform:
                    seg_mask_tensor = self.mask_transform(mask_img)
                else:
                    seg_mask_tensor = transforms.ToTensor()(mask_img)
            else:
                seg_mask_tensor = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]))
                
            if self.mask_source == "segmentation":
                foreground_mask = seg_mask_tensor
                feedback_mask = 1.0 - foreground_mask
            elif self.mask_source == "bbox_from_segmentation":
                foreground_mask = mask_to_bbox_mask(seg_mask_tensor)
                feedback_mask = 1.0 - foreground_mask
                
        elif self.mask_source == "bbox":
            bbox = row.get(self.bbox_column) if self.bbox_column else None
            # Need to map bbox to mask
            if bbox is not None:
                # Assuming bbox is a list or tuple of [x, y, w, h] in original image coordinates
                fg_mask_np = bbox_to_foreground_mask(bbox, (orig_size[1], orig_size[0]))
                fg_mask_pil = Image.fromarray((fg_mask_np * 255).astype(np.uint8))
                if self.mask_transform:
                    foreground_mask = self.mask_transform(fg_mask_pil)
                else:
                    foreground_mask = transforms.ToTensor()(fg_mask_pil)
            else:
                foreground_mask = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]))
            
            feedback_mask = 1.0 - foreground_mask
            
        return image_tensor, label, feedback_mask, foreground_mask, metadata

def create_xil_data_loaders(train_df, val_df, test_df, batch_size=16, use_masks=False,
                           mask_source="segmentation", bbox_column=None,
                           female_masks_path=None, male_masks_path=None):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]) if use_masks else None
    
    train_dataset = XILGenderDataset(train_df, transform=train_transform, mask_transform=mask_transform,
                                     use_masks=use_masks, mask_source=mask_source, bbox_column=bbox_column,
                                     female_masks_path=female_masks_path, male_masks_path=male_masks_path)
    val_dataset = XILGenderDataset(val_df, transform=val_test_transform, mask_transform=mask_transform,
                                   use_masks=use_masks, mask_source=mask_source, bbox_column=bbox_column,
                                   female_masks_path=female_masks_path, male_masks_path=male_masks_path)
    test_dataset = XILGenderDataset(test_df, transform=val_test_transform, mask_transform=mask_transform,
                                    use_masks=use_masks, mask_source=mask_source, bbox_column=bbox_column,
                                    female_masks_path=female_masks_path, male_masks_path=male_masks_path)
                                    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader