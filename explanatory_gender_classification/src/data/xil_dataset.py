import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image, ImageOps
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
        male_masks_path=None,
        joint_augment=False,
        max_rotation_deg=10.0,
        feedback_noise='none',
        feedback_noise_level=0.0
    ):
        # joint_augment=True applies ONE random geometric augmentation (horizontal flip with
        # p=0.5 and a rotation in [-max_rotation_deg, +max_rotation_deg]) to the image AND to
        # its segmentation mask, so that feedback masks stay spatially aligned with the image.
        # In that mode `transform` must be deterministic (resize/normalise only).
        # joint_augment=False reproduces the earlier behaviour in which random flip/rotation
        # were part of `transform` and therefore applied to the image only.
        # Simulated imperfect user feedback (training data only; evaluation masks are never perturbed):
        #   shift  : the person region is translated by up to level*224 pixels in x and y
        #   dilate : the person region is randomly dilated or eroded by up to level*224/2 pixels
        #   drop   : with probability `level` the feedback of a sample is missing (all-zero mask)
        #   bbox_centroid : the centre of the person's bounding box is moved by a Gaussian offset with
        #                   std = level * box width (x) and level * box height (y); box size unchanged
        #   bbox_jitter   : as bbox_centroid and, in addition, width and height are scaled by
        #                   (1 + N(0, level)); simulates a sloppy hand-drawn box
        self.feedback_noise = feedback_noise
        self.feedback_noise_level = feedback_noise_level
        self.joint_augment = joint_augment
        self.max_rotation_deg = max_rotation_deg
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

    def _perturb_feedback(self, foreground_mask):
        """Return (foreground_mask, drop) after applying the configured feedback noise."""
        mode, lvl = self.feedback_noise, self.feedback_noise_level
        if mode == 'none' or lvl <= 0:
            return foreground_mask, False
        if mode == 'drop':
            return foreground_mask, random.random() < lvl
        H, W = foreground_mask.shape[-2:]
        if mode in ('bbox_centroid', 'bbox_jitter'):
            ys, xs = torch.where(foreground_mask[0] > 0.5)
            if len(xs) == 0:
                return foreground_mask, False
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            w, h = x1 - x0, y1 - y0
            cx = (x0 + x1) / 2 + random.gauss(0, lvl) * w
            cy = (y0 + y1) / 2 + random.gauss(0, lvl) * h
            if mode == 'bbox_jitter':
                w = w * max(0.2, 1 + random.gauss(0, lvl))
                h = h * max(0.2, 1 + random.gauss(0, lvl))
            nx0, nx1 = int(round(max(0, cx - w / 2))), int(round(min(W, cx + w / 2)))
            ny0, ny1 = int(round(max(0, cy - h / 2))), int(round(min(H, cy + h / 2)))
            new = torch.zeros_like(foreground_mask)
            if nx1 > nx0 and ny1 > ny0:
                new[..., ny0:ny1, nx0:nx1] = 1.0
            return new, False
        if mode == 'shift':
            dx = int(random.uniform(-lvl, lvl) * W)
            dy = int(random.uniform(-lvl, lvl) * H)
            return torch.roll(foreground_mask, shifts=(dy, dx), dims=(-2, -1)), False
        if mode == 'dilate':
            r = int(random.uniform(-lvl, lvl) * W / 2)
            m = foreground_mask.unsqueeze(0)
            if r > 0:
                m = torch.nn.functional.max_pool2d(m, 2 * r + 1, stride=1, padding=r)
            elif r < 0:
                m = -torch.nn.functional.max_pool2d(-m, 2 * (-r) + 1, stride=1, padding=-r)
            return m.squeeze(0), False
        raise ValueError(f'unknown feedback_noise: {mode}')

    def _sample_geometry(self):
        flip = random.random() < 0.5
        angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
        return flip, angle

    @staticmethod
    def _apply_geometry(img, geom, resample):
        flip, angle = geom
        if flip:
            img = ImageOps.mirror(img)
        if angle != 0.0:
            img = img.rotate(angle, resample=resample, fillcolor=0 if img.mode == 'L' else (0, 0, 0))
        return img
        
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

        geom = self._sample_geometry() if self.joint_augment else None
        if geom is not None:
            image = self._apply_geometry(image, geom, Image.BILINEAR)
            
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
                if geom is not None:
                    mask_img = self._apply_geometry(mask_img, geom, Image.NEAREST)
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
            
        if self.feedback_noise != 'none' and foreground_mask is not None:
            # the FEEDBACK is perturbed; foreground_mask (used for evaluation) stays exact
            noisy_fg, drop = self._perturb_feedback(foreground_mask)
            feedback_mask = torch.zeros_like(feedback_mask) if drop else 1.0 - noisy_fg
        return image_tensor, label, feedback_mask, foreground_mask, metadata

def create_xil_data_loaders(train_df, val_df, test_df, batch_size=16, use_masks=False,
                           mask_source="segmentation", bbox_column=None,
                           female_masks_path=None, male_masks_path=None,
                           legacy_augmentation=False, feedback_noise='none', feedback_noise_level=0.0):
    """Build train/val/test loaders.

    By default random flip/rotation are applied jointly to image and feedback mask
    (`XILGenderDataset(joint_augment=True)`), keeping the feedback aligned with the image.
    `legacy_augmentation=True` restores the earlier image-only augmentation (kept only so that
    earlier runs can be reproduced; it misaligns masks for flipped/rotated training images).
    """
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
    
    if legacy_augmentation:
        train_dataset = XILGenderDataset(train_df, transform=train_transform, mask_transform=mask_transform,
                                         use_masks=use_masks, mask_source=mask_source, bbox_column=bbox_column,
                                         female_masks_path=female_masks_path, male_masks_path=male_masks_path,
                                         feedback_noise=feedback_noise, feedback_noise_level=feedback_noise_level)
    else:
        train_dataset = XILGenderDataset(train_df, transform=val_test_transform, mask_transform=mask_transform,
                                         use_masks=use_masks, mask_source=mask_source, bbox_column=bbox_column,
                                         female_masks_path=female_masks_path, male_masks_path=male_masks_path,
                                         joint_augment=True,
                                         feedback_noise=feedback_noise,
                                         feedback_noise_level=feedback_noise_level)
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