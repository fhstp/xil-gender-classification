"""
GradCAM implementation for explainable AI.

This module provides GradCAM (Gradient-weighted Class Activation Mapping) 
functionality for generating visual explanations of CNN predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM) implementation.
    
    GradCAM generates visual explanations by highlighting the regions of an input image
    that are important for predictions at a specific layer of a CNN.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for gradient extraction (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture activations and gradients
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, 
                    input_image: torch.Tensor, 
                    class_idx: Optional[int] = None) -> torch.Tensor:
        """
        Generate Class Activation Map.
        
        Args:
            input_image: Input tensor (1, C, H, W) or (C, H, W)
            class_idx: Target class index. If None, uses predicted class.
            
        Returns:
            CAM heatmap as tensor (H, W)
        """
        # Ensure input has batch dimension
        if len(input_image.shape) == 3:
            input_image = input_image.unsqueeze(0)
            
        input_image = input_image.requires_grad_()
        
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)  # (H, W)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize CAM
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam
    
    def generate_visualization(self, 
                             input_image: torch.Tensor,
                             class_idx: Optional[int] = None,
                             alpha: float = 0.4) -> np.ndarray:
        """
        Generate GradCAM visualization overlayed on input image.
        
        Args:
            input_image: Input tensor (1, C, H, W) or (C, H, W)
            class_idx: Target class index
            alpha: Transparency for overlay (0-1)
            
        Returns:
            Visualization as numpy array (H, W, 3)
        """
        # Generate CAM
        cam = self.generate_cam(input_image, class_idx)
        
        # Convert input image to numpy
        if len(input_image.shape) == 4:
            img_np = input_image[0].detach().cpu().permute(1, 2, 0).numpy()
        else:
            img_np = input_image.detach().cpu().permute(1, 2, 0).numpy()
            
        # Normalize image to 0-1
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Resize CAM to match image size
        cam_np = cam.detach().cpu().numpy()
        h, w = img_np.shape[:2]
        cam_resized = cv2.resize(cam_np, (w, h))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Overlay on original image
        visualization = alpha * heatmap + (1 - alpha) * img_np
        visualization = np.clip(visualization, 0, 1)
        
        return visualization


class GradCAMWrapper:
    """
    Wrapper class for GradCAM to provide a unified interface for different model types.
    """
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 target_layer_name: str = 'features'):
        """
        Initialize GradCAM wrapper.
        
        Args:
            model: PyTorch model
            target_layer_name: Name of target layer (e.g., 'features', 'layer4')
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # Find target layer
        self.target_layer = self._get_target_layer(model, target_layer_name)
        
        # Initialize GradCAM
        self.gradcam = GradCAM(model, self.target_layer)
    
    def _get_target_layer(self, model, layer_name):
        """Get target layer from model by name."""
        # Try direct attribute access first
        if hasattr(model, layer_name):
            return getattr(model, layer_name)
            
        # For models with sequential features
        if hasattr(model, 'features'):
            if isinstance(model.features, torch.nn.Sequential):
                return model.features[-1]  # Last layer in features
            return model.features
            
        # For ResNet-style models
        if hasattr(model, 'layer4'):
            return model.layer4
            
        # Fallback: find last convolutional layer
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
        
        if conv_layers:
            return conv_layers[-1][1]  # Return last conv layer
            
        raise ValueError(f"Could not find target layer '{layer_name}' in model")
    
    def generate_cam(self, 
                    images: torch.Tensor, 
                    target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate CAM for input images.
        
        Args:
            images: Input tensor (B, C, H, W) or (C, H, W)
            target_class: Target class for explanation
            
        Returns:
            CAM tensor (B, H, W) or (H, W)
        """
        was_single = len(images.shape) == 3
        if was_single:
            images = images.unsqueeze(0)
            
        images = images.to(self.device)
        
        cams = []
        for i in range(images.shape[0]):
            cam = self.gradcam.generate_cam(images[i:i+1], target_class)
            cams.append(cam)
            
        result = torch.stack(cams)
        
        if was_single:
            result = result[0]
            
        return result
    
    def generate_explanation(self, 
                           images: torch.Tensor, 
                           target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate explanation (alias for generate_cam for compatibility).
        
        Args:
            images: Input tensor
            target_class: Target class
            
        Returns:
            Explanation tensor
        """
        return self.generate_cam(images, target_class)
    
    def visualize(self, 
                 image: torch.Tensor,
                 target_class: Optional[int] = None,
                 save_path: Optional[str] = None,
                 title: str = "GradCAM Explanation"):
        """
        Create and display GradCAM visualization.
        
        Args:
            image: Input image tensor (C, H, W)
            target_class: Target class for explanation
            save_path: Path to save visualization
            title: Plot title
        """
        # Generate visualization
        vis = self.gradcam.generate_visualization(image, target_class)
        cam = self.gradcam.generate_cam(image, target_class)
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img_np = image.detach().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # CAM heatmap
        im1 = axes[1].imshow(cam.detach().cpu().numpy(), cmap='viridis')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(vis)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


def create_gradcam_explainer(model: torch.nn.Module, 
                           target_layer_name: str = 'features') -> GradCAMWrapper:
    """
    Factory function to create GradCAM explainer.
    
    Args:
        model: PyTorch model
        target_layer_name: Name of target layer
        
    Returns:
        GradCAMWrapper instance
    """
    return GradCAMWrapper(model, target_layer_name)