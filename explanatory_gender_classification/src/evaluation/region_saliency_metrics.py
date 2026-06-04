"""Region-level saliency decomposition (body, clothing, hair, background)."""
import torch
from typing import Dict

def compute_region_saliency(saliency_map: torch.Tensor, region_masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute proportion of saliency allocated to specific regions.
    
    Args:
        saliency_map: Saliency map (H, W)
        region_masks: Dict mapping region name to binary mask (H, W)
    """
    total_saliency = saliency_map.sum().item()
    if total_saliency == 0:
        return {f"{k}_saliency_ratio": 0.0 for k in region_masks}
        
    results = {}
    for region_name, mask in region_masks.items():
        region_saliency = (saliency_map * (mask > 0.5).float()).sum().item()
        results[f"{region_name}_saliency_ratio"] = region_saliency / total_saliency
        
    return results
