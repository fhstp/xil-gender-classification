"""
Bias evaluation metrics for gender classification models.

This module implements the bias metrics described in the paper:
- FFP (Foreground Focus Proportion) 
- BFP (Background Focus Proportion)
- BSR (Background Saliency Ratio)
- DICE Score

These metrics evaluate how well the model focuses on relevant vs irrelevant regions.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Union, Optional
import torch.nn.functional as F


def compute_dice_score(prediction_mask: torch.Tensor, 
                      ground_truth_mask: torch.Tensor,
                      smooth: float = 1e-6) -> float:
    """
    Compute DICE score between prediction and ground truth masks.
    
    DICE = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        prediction_mask: Predicted relevance mask (H, W) or (B, H, W)
        ground_truth_mask: Ground truth segmentation mask (H, W) or (B, H, W) 
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        DICE score (0-1, higher is better)
    """
    # Ensure masks are binary
    pred_binary = (prediction_mask > 0.5).float()
    gt_binary = (ground_truth_mask > 0.5).float()
    
    # Flatten masks
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()
    
    # Calculate intersection and union
    intersection = (pred_flat * gt_flat).sum()
    total = pred_flat.sum() + gt_flat.sum()
    
    # Compute DICE
    dice = (2.0 * intersection + smooth) / (total + smooth)
    
    return dice.item()


def compute_ffp(saliency_map: torch.Tensor, 
                foreground_mask: torch.Tensor,
                threshold_percentile: float = 25.0) -> float:
    """
    Compute Foreground Focus Proportion (FFP).
    
    FFP = sum(I(S(i) > T) for i in F) / |F|
    
    Measures proportion of foreground pixels with saliency above threshold.
    
    Args:
        saliency_map: Saliency/attention map (H, W)
        foreground_mask: Binary mask indicating foreground (person) pixels (H, W)
        threshold_percentile: Percentile for threshold (0-100)
        
    Returns:
        FFP score (0-1, higher is better)
    """
    # Calculate threshold (e.g., 25th percentile)
    threshold = torch.quantile(saliency_map.flatten(), threshold_percentile / 100.0)
    
    # Get foreground pixels
    foreground_pixels = (foreground_mask > 0.5)
    
    if foreground_pixels.sum() == 0:
        return 0.0
    
    # Count foreground pixels above threshold
    high_saliency_foreground = (saliency_map > threshold) & foreground_pixels
    
    # Calculate FFP
    ffp = high_saliency_foreground.sum().float() / foreground_pixels.sum().float()
    
    return ffp.item()


def compute_bfp(saliency_map: torch.Tensor, 
                foreground_mask: torch.Tensor,
                threshold_percentile: float = 25.0) -> float:
    """
    Compute Background Focus Proportion (BFP).
    
    BFP = sum(I(S(i) > T) for i in B) / |B|
    
    Measures proportion of background pixels with saliency above threshold.
    
    Args:
        saliency_map: Saliency/attention map (H, W)
        foreground_mask: Binary mask indicating foreground (person) pixels (H, W)
        threshold_percentile: Percentile for threshold (0-100)
        
    Returns:
        BFP score (0-1, lower is better)
    """
    # Calculate threshold (e.g., 25th percentile)
    threshold = torch.quantile(saliency_map.flatten(), threshold_percentile / 100.0)
    
    # Get background pixels (inverse of foreground)
    background_pixels = (foreground_mask <= 0.5)
    
    if background_pixels.sum() == 0:
        return 0.0
    
    # Count background pixels above threshold
    high_saliency_background = (saliency_map > threshold) & background_pixels
    
    # Calculate BFP
    bfp = high_saliency_background.sum().float() / background_pixels.sum().float()
    
    return bfp.item()


def compute_bsr(saliency_map: torch.Tensor, 
                foreground_mask: torch.Tensor) -> float:
    """
    Compute Background Saliency Ratio (BSR).
    
    BSR = sum(S(i) for i in B) / sum(S(i) for i in I)
    
    Measures proportion of total saliency attributed to background.
    
    Args:
        saliency_map: Saliency/attention map (H, W)
        foreground_mask: Binary mask indicating foreground (person) pixels (H, W)
        
    Returns:
        BSR score (0-1, lower is better)
    """
    # Get background pixels
    background_pixels = (foreground_mask <= 0.5)
    
    # Calculate total saliency in background vs entire image
    background_saliency = (saliency_map * background_pixels.float()).sum()
    total_saliency = saliency_map.sum()
    
    if total_saliency == 0:
        return 0.0
    
    # Calculate BSR
    bsr = background_saliency / total_saliency
    
    return bsr.item()


def compute_all_bias_metrics(saliency_map: torch.Tensor,
                           foreground_mask: torch.Tensor,
                           explanation_mask: Optional[torch.Tensor] = None,
                           threshold_percentile: float = 25.0) -> Dict[str, float]:
    """
    Compute all bias metrics for a single sample.
    
    Args:
        saliency_map: Saliency/attention map (H, W)
        foreground_mask: Binary mask indicating foreground pixels (H, W) 
        explanation_mask: Optional binary explanation mask for DICE (H, W)
        threshold_percentile: Percentile for FFP/BFP threshold (0-100)
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Compute FFP, BFP, BSR
    metrics['FFP'] = compute_ffp(saliency_map, foreground_mask, threshold_percentile)
    metrics['BFP'] = compute_bfp(saliency_map, foreground_mask, threshold_percentile)
    metrics['BSR'] = compute_bsr(saliency_map, foreground_mask)
    
    # Compute DICE if explanation mask is provided
    if explanation_mask is not None:
        # Convert saliency map to binary mask for DICE calculation
        saliency_threshold = torch.quantile(saliency_map.flatten(), 0.5)  # Median threshold
        binary_saliency = (saliency_map >= saliency_threshold).float()
        
        metrics['DICE'] = compute_dice_score(binary_saliency, explanation_mask)
    
    return metrics


def evaluate_model_bias(model,
                       dataloader,
                       explainer,
                       device: str = 'cpu',
                       threshold_percentile: float = 25.0) -> Dict[str, float]:
    """
    Evaluate bias metrics for an entire dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing (images, labels, masks) 
        explainer: Explainability method (GradCAM or BLA)
        device: Device to run evaluation on
        threshold_percentile: Percentile for FFP/BFP threshold
        
    Returns:
        Dictionary of averaged metrics
    """
    model.eval()
    
    all_ffp = []
    all_bfp = []
    all_bsr = []
    all_dice = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, masks = batch
            else:
                images, labels = batch
                masks = None
                
            images = images.to(device)
            labels = labels.to(device)
            if masks is not None:
                masks = masks.to(device)
            
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                image = images[i:i+1]  # Keep batch dimension
                label = labels[i]
                
                # Generate explanation
                if hasattr(explainer, 'generate_cam'):
                    # GradCAM
                    saliency_map = explainer.generate_cam(image, label.item())
                elif hasattr(explainer, 'generate_explanation'):
                    # BLA wrapper
                    saliency_map = explainer.generate_explanation(image)[0]
                else:
                    # Direct model call for BLA
                    if len(model(image)) == 2:
                        _, saliency_map = model(image)
                        saliency_map = saliency_map[0]
                    else:
                        raise ValueError("Cannot generate explanations from this model")
                
                # Get ground truth mask if available
                if masks is not None:
                    if len(masks.shape) == 4:  # (B, C, H, W)
                        gt_mask = masks[i, 0]  # Take first channel
                    else:  # (B, H, W)
                        gt_mask = masks[i]
                else:
                    # If no masks provided, assume entire image is foreground
                    gt_mask = torch.ones_like(saliency_map)
                
                # Resize saliency map to match mask size if needed
                if saliency_map.shape != gt_mask.shape:
                    saliency_map = F.interpolate(
                        saliency_map.unsqueeze(0).unsqueeze(0),
                        size=gt_mask.shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                
                # Compute metrics
                metrics = compute_all_bias_metrics(
                    saliency_map, gt_mask, gt_mask, threshold_percentile
                )
                
                all_ffp.append(metrics['FFP'])
                all_bfp.append(metrics['BFP'])
                all_bsr.append(metrics['BSR'])
                all_dice.append(metrics['DICE'])
    
    # Return averaged metrics
    return {
        'FFP': np.mean(all_ffp),
        'BFP': np.mean(all_bfp), 
        'BSR': np.mean(all_bsr),
        'DICE': np.mean(all_dice)
    }


class BiasMetricsTracker:
    """
    Class to track bias metrics during training and evaluation.
    """
    
    def __init__(self):
        self.metrics_history = {
            'FFP': [],
            'BFP': [],
            'BSR': [],
            'DICE': []
        }
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics history."""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest metrics."""
        return {key: values[-1] if values else 0.0 
                for key, values in self.metrics_history.items()}
    
    def get_average(self) -> Dict[str, float]:
        """Get average of all metrics."""
        return {key: np.mean(values) if values else 0.0 
                for key, values in self.metrics_history.items()}
    
    def print_summary(self):
        """Print summary of metrics."""
        latest = self.get_latest()
        average = self.get_average()
        
        print("\n" + "="*50)
        print("BIAS METRICS SUMMARY")
        print("="*50)
        print(f"{'Metric':<10} {'Latest':<10} {'Average':<10}")
        print("-"*30)
        for metric in ['FFP', 'BFP', 'BSR', 'DICE']:
            print(f"{metric:<10} {latest[metric]:<10.3f} {average[metric]:<10.3f}")
        print("="*50)
        print("Higher is better: FFP, DICE")
        print("Lower is better: BFP, BSR")
        print("="*50)