"""
Bounded Logit Attention (BLA) explainability method implementation.

This module implements the BLA method as described in the paper and 
adapted from the user's notebook implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class BLA(nn.Module):
    """
    Bounded Logit Attention module that generates self-explaining feature attention.
    
    BLA uses a 1x1 convolution to generate logits, bounds them with beta function
    (inverted ReLU), and applies softmax to create attention weights.
    """
    
    def __init__(self, feature_map_size: int, temperature: float = 0.1):
        """
        Initialize BLA module.
        
        Args:
            feature_map_size: Number of channels in the feature maps
            temperature: Temperature parameter for softmax attention
        """
        super(BLA, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels=feature_map_size, 
            out_channels=1, 
            kernel_size=1
        )
        self.temperature = temperature

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through BLA module.
        
        Args:
            feature_map: Input feature maps (B, C, H, W)
            
        Returns:
            Tuple of (reweighed_features, attention_weights)
        """
        # Generate logits using 1x1 convolution
        logits = self.conv1x1(feature_map).squeeze(1)  # (B, H, W)
        
        # Apply beta function (bound logits to be <= 0)
        logits = torch.min(logits, torch.zeros_like(logits))
        
        # Flatten for softmax
        batch_size, height, width = logits.shape
        logits_flat = logits.view(batch_size, -1)  # (B, H*W)
        
        # Apply softmax to get attention weights
        attention_weights_flat = F.softmax(logits_flat / self.temperature, dim=-1)
        
        # Reshape back to spatial dimensions
        attention_weights = attention_weights_flat.view(batch_size, height, width)
        
        # Apply attention to feature maps
        # Expand attention weights to match feature map dimensions
        attention_expanded = attention_weights.unsqueeze(1)  # (B, 1, H, W)
        reweighed_features = attention_expanded * feature_map  # (B, C, H, W)
        
        return reweighed_features, attention_weights


class BLAWrapper:
    """
    Wrapper class for BLA explainability method to generate explanations
    compatible with the XIL framework.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize BLA wrapper.
        
        Args:
            model: Model that includes BLA module
        """
        self.model = model
        self.device = next(model.parameters()).device
        
    def generate_explanation(self, 
                           images: torch.Tensor, 
                           target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate BLA explanation for input images.
        
        Args:
            images: Input images (B, C, H, W)
            target_class: Target class for explanation (ignored for BLA)
            
        Returns:
            Attention maps (B, H, W)
        """
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'forward') and len(self.model(images)) == 2:
                # Model returns (logits, attention_weights)
                _, attention_weights = self.model(images)
            else:
                raise ValueError("Model must return attention weights for BLA explanation")
                
        return attention_weights
    
    def generate_binary_mask(self, 
                           images: torch.Tensor, 
                           threshold_percentile: float = 50.0) -> torch.Tensor:
        """
        Generate binary relevance mask from BLA attention.
        
        Args:
            images: Input images (B, C, H, W)
            threshold_percentile: Percentile for thresholding (0-100)
            
        Returns:
            Binary masks (B, H, W) - 1 for relevant, 0 for irrelevant
        """
        attention_weights = self.generate_explanation(images)
        
        batch_size = attention_weights.shape[0]
        binary_masks = []
        
        for i in range(batch_size):
            attention_map = attention_weights[i]
            
            # Calculate threshold based on percentile
            threshold = torch.quantile(attention_map.flatten(), threshold_percentile / 100.0)
            
            # Create binary mask
            binary_mask = (attention_map >= threshold).float()
            binary_masks.append(binary_mask)
            
        return torch.stack(binary_masks)
    
    def visualize_explanation(self, 
                            image: torch.Tensor, 
                            save_path: Optional[str] = None,
                            title: str = "BLA Explanation") -> None:
        """
        Visualize BLA explanation for a single image.
        
        Args:
            image: Input image (C, H, W)
            save_path: Path to save visualization
            title: Title for the plot
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension
            
        image = image.to(self.device)
        
        # Generate explanation
        attention_weights = self.generate_explanation(image)
        attention_map = attention_weights[0].cpu().numpy()  # Remove batch dimension
        
        # Prepare image for visualization
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        
        # Normalize image to 0-1 if needed
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention heatmap
        im1 = axes[1].imshow(attention_map, cmap='viridis')
        axes[1].set_title('BLA Attention')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Binary mask
        threshold = np.median(attention_map)
        binary_mask = (attention_map >= threshold).astype(np.float32)
        im2 = axes[2].imshow(binary_mask, cmap='viridis')
        axes[2].set_title('Binary Mask')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Overlay
        height, width = img_np.shape[:2]
        attention_resized = cv2.resize(attention_map, (width, height))
        
        # Create colored attention map
        attention_colored = cv2.applyColorMap(
            (attention_resized * 255).astype('uint8'), cv2.COLORMAP_TURBO
        )
        attention_colored = attention_colored[:, :, ::-1]  # BGR to RGB
        attention_colored = attention_colored.astype(np.float32) / 255.0
        
        # Overlay
        overlay = 0.6 * img_np + 0.4 * attention_colored
        overlay = np.clip(overlay, 0, 1)
        
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


class ModelWithBLA(nn.Module):
    """
    Wrapper to add BLA to existing models for self-explaining predictions.
    """
    
    def __init__(self, 
                 base_model: nn.Module, 
                 feature_map_size: int,
                 num_classes: int,
                 temperature: float = 0.1,
                 insert_after_layer: str = 'features'):
        """
        Initialize model with BLA.
        
        Args:
            base_model: Base model (e.g., EfficientNet, ResNet)
            feature_map_size: Number of feature map channels where BLA is inserted
            num_classes: Number of output classes
            temperature: Temperature for BLA attention
            insert_after_layer: Layer name after which to insert BLA
        """
        super(ModelWithBLA, self).__init__()
        
        # Extract feature extractor
        if hasattr(base_model, insert_after_layer):
            self.feature_extractor = getattr(base_model, insert_after_layer)
        else:
            # Assume it's the features part for models like EfficientNet
            self.feature_extractor = base_model.features
            
        # BLA module
        self.bla = BLA(feature_map_size, temperature)
        
        # Final classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_map_size, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through model with BLA.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply BLA
        reweighed_features, attention_weights = self.bla(features)
        
        # Global pooling and classification
        pooled_features = self.global_pool(reweighed_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        logits = self.classifier(pooled_features)
        
        return logits, attention_weights
    
    def get_explanation_wrapper(self) -> BLAWrapper:
        """
        Get BLA wrapper for explanation generation.
        
        Returns:
            BLAWrapper instance
        """
        return BLAWrapper(self)


def create_bla_model(model_name: str, 
                    num_classes: int = 2, 
                    pretrained: bool = True,
                    temperature: float = 0.1) -> ModelWithBLA:
    """
    Create a model with BLA based on the model name.
    
    Args:
        model_name: Name of the base model ('efficientnet_b0', 'resnet50', etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        temperature: Temperature for BLA attention
        
    Returns:
        ModelWithBLA instance
    """
    import torchvision.models as models
    
    if model_name == 'efficientnet_b0':
        base_model = models.efficientnet_b0(pretrained=pretrained)
        feature_map_size = 1280
        
    elif model_name == 'resnet50':
        base_model = models.resnet50(pretrained=pretrained)
        # For ResNet, we need to modify to get features before final layers
        feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        base_model = nn.Module()
        base_model.features = feature_extractor
        feature_map_size = 2048
        
    elif model_name == 'vgg16':
        base_model = models.vgg16(pretrained=pretrained)
        feature_map_size = 512
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return ModelWithBLA(
        base_model=base_model,
        feature_map_size=feature_map_size,
        num_classes=num_classes,
        temperature=temperature
    )