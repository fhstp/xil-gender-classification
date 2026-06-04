"""Feedback mask construction utilities (segmentation and bounding-box to irrelevance masks)."""
import numpy as np
import torch
from PIL import Image

def segmentation_to_feedback_mask(seg_mask, relevant_is_foreground=True):
    """
    Converts a person segmentation mask into the RRR/CAIPI feedback mask convention.
    For RRR/CAIPI, mask should be:
    0 = relevant region, 1 = irrelevant region
    """
    if isinstance(seg_mask, torch.Tensor):
        mask = (seg_mask > 0.5).float()
    else:
        mask = (seg_mask > 0.5).astype(np.float32)
        
    if relevant_is_foreground:
        # Foreground is relevant (0), background is irrelevant (1)
        return 1.0 - mask
    else:
        # Background is relevant (0), foreground is irrelevant (1)
        return mask

def bbox_to_foreground_mask(bbox, image_size):
    """
    bbox: [x_min, y_min, width, height] or [x1, y1, x2, y2]
    returns binary foreground mask:
    1 = inside bbox, 0 = outside bbox
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.float32)
    
    if bbox is None or len(bbox) != 4:
        return mask
        
    # Assume [x1, y1, x2, y2] or [x, y, w, h] format, checking format:
    # If the 3rd and 4th values are smaller than W, H and we know they might be w, h
    # We will assume [x_min, y_min, width, height] as standard COCO format
    # Let's standardize on [x_min, y_min, width, height] for this project if not specified.
    # We will accept [x1, y1, width, height]
    x, y, w, h = [int(v) for v in bbox]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    
    mask[y1:y2, x1:x2] = 1.0
    return mask

def bbox_to_feedback_mask(bbox, image_size):
    """
    returns mask in XIL convention:
    0 = relevant, 1 = irrelevant
    """
    fg_mask = bbox_to_foreground_mask(bbox, image_size)
    return 1.0 - fg_mask

def mask_to_bbox_mask(seg_mask):
    """
    Derive bounding box from an existing segmentation mask.
    Useful for COCO because exact masks are already available.
    """
    if isinstance(seg_mask, torch.Tensor):
        mask_np = seg_mask.squeeze().cpu().numpy()
    else:
        mask_np = np.array(seg_mask)
        
    rows = np.any(mask_np > 0.5, axis=1)
    cols = np.any(mask_np > 0.5, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Empty mask
        if isinstance(seg_mask, torch.Tensor):
            return torch.zeros_like(seg_mask)
        return np.zeros_like(mask_np)
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    bbox_mask = np.zeros_like(mask_np)
    bbox_mask[ymin:ymax+1, xmin:xmax+1] = 1.0
    
    if isinstance(seg_mask, torch.Tensor):
        return torch.from_numpy(bbox_mask).to(seg_mask.device).float().unsqueeze(0)
    return bbox_mask

def resize_mask(mask, size=(224, 224)):
    """
    Make sure all feedback masks match the model input resolution.
    """
    if isinstance(mask, torch.Tensor):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)
        return torch.nn.functional.interpolate(mask, size=size, mode='nearest').squeeze()
    else:
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        resized = pil_mask.resize(size, Image.NEAREST)
        return np.array(resized, dtype=np.float32) / 255.0