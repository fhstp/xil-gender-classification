"""Object cut-paste augmentation utilities (experimental, not used in main pipeline)."""
import torch
import numpy as np
from PIL import Image

def insert_object(image: torch.Tensor, obj_img: torch.Tensor, obj_mask: torch.Tensor, x: int, y: int) -> torch.Tensor:
    """
    Insert an object into the image at the specified coordinates.
    """
    _, h, w = image.shape
    _, obj_h, obj_w = obj_img.shape
    
    # Calculate bounds
    y1, y2 = max(0, y), min(h, y + obj_h)
    x1, x2 = max(0, x), min(w, x + obj_w)
    
    obj_y1, obj_y2 = max(0, -y), obj_h - max(0, y + obj_h - h)
    obj_x1, obj_x2 = max(0, -x), obj_w - max(0, x + obj_w - w)
    
    if y1 >= y2 or x1 >= x2:
        return image
        
    out_img = image.clone()
    alpha = obj_mask[:, obj_y1:obj_y2, obj_x1:obj_x2]
    
    out_img[:, y1:y2, x1:x2] = (1 - alpha) * out_img[:, y1:y2, x1:x2] + alpha * obj_img[:, obj_y1:obj_y2, obj_x1:obj_x2]
    
    return out_img
