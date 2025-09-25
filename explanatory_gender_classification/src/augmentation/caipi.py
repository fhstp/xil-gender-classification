"""
CAIPI (Counterfactual Augmented Interactive Perturbation-based Inference)
Implementation for data augmentation based on explainability methods.

This module implements the CAIPI data augmentation technique described in the paper
"Explanatory Interactive Machine Learning for Bias Mitigation in Visual Gender Classification".
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
import random
from typing import List, Tuple, Optional
from PIL import Image
import cv2


class CAIPIAugmentation:
    """
    CAIPI data augmentation module that generates counterexamples by applying
    transformations to irrelevant regions of images while keeping relevant regions intact.
    """
    
    def __init__(self, k: int = 3, transformations: Optional[List[str]] = None):
        """
        Initialize CAIPI augmentation.
        
        Args:
            k: Number of counterexamples to generate per sample
            transformations: List of transformation names to apply
        """
        self.k = k
        
        if transformations is None:
            self.transformations = [
                'inversion', 'posterization', 'equalization', 
                'color_jittering', 'solarization'
            ]
        else:
            self.transformations = transformations
            
        # Define transformation parameters
        self.transform_configs = {
            'color_jittering': {
                'brightness': 0.5,
                'contrast': 0.5,
                'saturation': 0.5,
                'hue': 0.1
            },
            'posterization': {
                'bits': [2, 4, 6]  # Different posterization levels
            },
            'solarization': {
                'threshold': [128, 200, 255]  # Different thresholds
            }
        }
        
    def apply_transformation(self, image: np.ndarray, transform_name: str) -> np.ndarray:
        """
        Apply a specific transformation to the image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            transform_name: Name of transformation to apply
            
        Returns:
            Transformed image as numpy array
        """
        # Convert to PIL for transformations
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = image
            
        if transform_name == 'inversion':
            # Invert colors
            inverted = TF.invert(pil_image)
            return np.array(inverted)
            
        elif transform_name == 'posterization':
            # Posterize with random bits
            bits = random.choice(self.transform_configs['posterization']['bits'])
            posterized = TF.posterize(pil_image, bits=bits)
            return np.array(posterized)
            
        elif transform_name == 'equalization':
            # Histogram equalization
            equalized = TF.equalize(pil_image)
            return np.array(equalized)
            
        elif transform_name == 'color_jittering':
            # Color jittering
            jitter = transforms.ColorJitter(
                brightness=self.transform_configs['color_jittering']['brightness'],
                contrast=self.transform_configs['color_jittering']['contrast'],
                saturation=self.transform_configs['color_jittering']['saturation'],
                hue=self.transform_configs['color_jittering']['hue']
            )
            jittered = jitter(pil_image)
            return np.array(jittered)
            
        elif transform_name == 'solarization':
            # Solarization with random threshold
            threshold = random.choice(self.transform_configs['solarization']['threshold'])
            solarized = TF.solarize(pil_image, threshold=threshold)
            return np.array(solarized)
            
        else:
            # If transformation not recognized, return original
            return np.array(pil_image)
    
    def generate_counterexamples(self, 
                               image: torch.Tensor, 
                               mask: torch.Tensor, 
                               label: int) -> List[Tuple[torch.Tensor, int]]:
        """
        Generate k counterexamples by applying transformations to irrelevant regions.
        
        Args:
            image: Original image tensor (C, H, W)
            mask: Relevance mask (1, H, W) - 0 for relevant, 1 for irrelevant
            label: Original label (unchanged in counterexamples)
            
        Returns:
            List of (augmented_image, label) tuples
        """
        counterexamples = []
        
        # Convert tensors to numpy for processing
        if len(image.shape) == 3:
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.cpu().numpy()
            
        if len(mask.shape) == 3:
            mask_np = mask.squeeze(0).cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()
            
        # Normalize image to 0-255 range for PIL operations
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
            
        # Ensure mask is binary
        mask_np = (mask_np > 0.5).astype(np.float32)
        
        for i in range(self.k):
            # Create a copy of the original image
            augmented_image = image_np.copy()
            
            # Select random transformation
            transform_name = random.choice(self.transformations)
            
            # Apply transformation to the entire image
            transformed_image = self.apply_transformation(augmented_image, transform_name)
            
            # Combine original and transformed image using mask
            # Keep relevant regions (mask=0) from original, irrelevant regions (mask=1) from transformed
            for c in range(3):  # For RGB channels
                augmented_image[:, :, c] = (
                    (1 - mask_np) * image_np[:, :, c] + 
                    mask_np * transformed_image[:, :, c]
                )
            
            # Convert back to tensor
            augmented_tensor = torch.from_numpy(augmented_image).permute(2, 0, 1).float()
            
            # Normalize back to 0-1 range if needed
            if image.max() <= 1.0:
                augmented_tensor = augmented_tensor / 255.0
                
            counterexamples.append((augmented_tensor, label))
            
        return counterexamples
    
    def augment_batch(self, 
                     images: torch.Tensor, 
                     masks: torch.Tensor, 
                     labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CAIPI augmentation to a batch of images.
        
        Args:
            images: Batch of images (B, C, H, W)
            masks: Batch of relevance masks (B, 1, H, W)
            labels: Batch of labels (B,)
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        batch_size = images.shape[0]
        all_augmented_images = []
        all_augmented_labels = []
        
        for i in range(batch_size):
            counterexamples = self.generate_counterexamples(
                images[i], masks[i], labels[i].item()
            )
            
            for aug_image, aug_label in counterexamples:
                all_augmented_images.append(aug_image)
                all_augmented_labels.append(aug_label)
        
        # Stack into tensors
        augmented_images = torch.stack(all_augmented_images)
        augmented_labels = torch.tensor(all_augmented_labels)
        
        return augmented_images, augmented_labels


def apply_caipi_sampling(dataset, 
                        model, 
                        explainer, 
                        num_samples: int = 50, 
                        k: int = 3,
                        sampling_strategy: str = 'uncertainty',
                        device: str = 'cpu') -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Apply CAIPI sampling to select uncertain/confident samples and generate counterexamples.
    
    Args:
        dataset: Dataset to sample from
        model: Trained model for prediction
        explainer: Explainability method (GradCAM or BLA)
        num_samples: Number of samples to select for augmentation
        k: Number of counterexamples per selected sample
        sampling_strategy: 'uncertainty' or 'high_confidence'
        device: Device to run computations on
        
    Returns:
        List of (image, mask, label) tuples for augmented samples
    """
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    
    # Create dataloader for sampling
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model.eval()
    sample_scores = []
    sample_indices = []
    
    # Calculate prediction confidence/uncertainty for all samples
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # Handle different batch formats (with or without masks)
            if len(batch) == 3:
                images, masks, labels = batch
            else:
                images, labels = batch
                
            images = images.to(device)
            
            if hasattr(model, 'forward') and len(model(images)) == 2:
                # Model with BLA returns (logits, attention)
                logits, _ = model(images)
            else:
                # Regular model
                logits = model(images)
                
            probs = F.softmax(logits, dim=1)
            max_prob = torch.max(probs, dim=1)[0].item()
            
            if sampling_strategy == 'uncertainty':
                # Lower probability = higher uncertainty
                score = 1 - max_prob
            else:  # high_confidence
                # Higher probability = higher confidence
                score = max_prob
                
            sample_scores.append(score)
            sample_indices.append(idx)
    
    # Select top samples based on strategy
    sorted_indices = sorted(range(len(sample_scores)), 
                          key=lambda i: sample_scores[i], reverse=True)
    selected_indices = sorted_indices[:num_samples]
    
    # Generate counterexamples for selected samples
    caipi = CAIPIAugmentation(k=k)
    augmented_samples = []
    
    model.eval()
    with torch.no_grad():
        for idx in selected_indices:
            images, labels = dataset[idx]
            images = images.unsqueeze(0).to(device)  # Add batch dimension
            
            # Generate explanation mask
            if hasattr(explainer, 'generate_cam'):
                # GradCAM
                explanation_map = explainer.generate_cam(images, labels)
            else:
                # BLA - extract attention weights
                _, attention_weights = model(images)
                explanation_map = attention_weights.squeeze(0)
                
            # Convert explanation to binary mask (threshold at median)
            explanation_flat = explanation_map.flatten()
            threshold = torch.median(explanation_flat)
            binary_mask = (explanation_map < threshold).float().unsqueeze(0)
            
            # Generate counterexamples
            counterexamples = caipi.generate_counterexamples(
                images.squeeze(0), binary_mask, labels
            )
            
            for aug_image, aug_label in counterexamples:
                augmented_samples.append((aug_image, binary_mask.squeeze(0), aug_label))
    
    return augmented_samples