"""
Explainability methods including GradCAM and LIME.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import warnings
warnings.filterwarnings('ignore')


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM) implementation.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for gradient extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        """Save forward pass activations."""
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients."""
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_image, class_idx=None):
        """
        Generate Class Activation Map.
        
        Args:
            input_image: Input image tensor (1 x C x H x W)
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            GradCAM heatmap
        """
        # Forward pass
        model_output = self.model(input_image)
        
        if class_idx is None:
            class_idx = model_output.argmax(dim=1).item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = model_output[:, class_idx]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam.cpu().numpy()
    
    def visualize_cam(self, input_image, class_idx=None, alpha=0.4):
        """
        Visualize GradCAM overlaid on input image.
        
        Args:
            input_image: Input image tensor (1 x C x H x W)
            class_idx: Target class index
            alpha: Opacity of overlay
            
        Returns:
            Visualization as numpy array
        """
        # Generate CAM
        cam = self.generate_cam(input_image, class_idx)
        
        # Convert input image to numpy and denormalize
        img = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Denormalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255
        
        # Overlay on original image
        superimposed = heatmap * alpha + img * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 1)
        
        return superimposed, cam_resized


def get_gradcam_layer(model, architecture):
    """
    Get the appropriate layer for GradCAM based on model architecture.
    
    Args:
        model: PyTorch model
        architecture: Model architecture name
        
    Returns:
        Target layer for GradCAM
    """
    if architecture == 'resnet50':
        return model.backbone.layer4[-1].conv3
    elif architecture == 'efficientnet_b0':
        return model.backbone._blocks[-1]._project_conv
    elif architecture == 'densenet121':
        return model.backbone.features.denseblock4.denselayer16.conv2
    elif architecture == 'vgg16':
        return model.backbone.features[28]  # Last conv layer
    elif architecture == 'mobilenet_v2':
        return model.backbone.features[-1][0]
    elif architecture == 'googlenet':
        return model.backbone.inception5b.branch4[1].conv
    else:
        raise ValueError(f"GradCAM layer not defined for {architecture}")


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for image classification.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize LIME explainer.
        
        Args:
            model: PyTorch model
            device: Device for model inference
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Image preprocessing
        self.preprocess = lambda x: self._preprocess_image(x)
        
    def _preprocess_image(self, images):
        """
        Preprocess images for model input.
        
        Args:
            images: Batch of images as numpy arrays
            
        Returns:
            Preprocessed tensor
        """
        # Convert to tensor and normalize
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        batch = []
        for img in images:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            pil_img = Image.fromarray(img)
            tensor_img = transform(pil_img)
            batch.append(tensor_img)
            
        return torch.stack(batch).to(self.device)
    
    def predict_batch(self, images):
        """
        Predict probabilities for a batch of images.
        
        Args:
            images: Batch of images as numpy arrays
            
        Returns:
            Prediction probabilities
        """
        with torch.no_grad():
            tensor_images = self._preprocess_image(images)
            logits = self.model(tensor_images)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()
    
    def explain_instance(self, image, num_samples=1000, num_features=100000):
        """
        Explain a single image prediction using LIME.
        
        Args:
            image: Input image as numpy array (H x W x C)
            num_samples: Number of samples for LIME
            num_features: Number of features for LIME
            
        Returns:
            LIME explanation
        """
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Generate explanation
        explanation = explainer.explain_instance(
            image, 
            self.predict_batch,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
        )
        
        return explanation
    
    def visualize_explanation(self, explanation, image, label_idx=1, positive_only=True, hide_rest=True):
        """
        Visualize LIME explanation.
        
        Args:
            explanation: LIME explanation object
            image: Original image
            label_idx: Label index to explain
            positive_only: Show only positive contributions
            hide_rest: Hide non-contributing regions
            
        Returns:
            Visualization as numpy array
        """
        temp, mask = explanation.get_image_and_mask(
            label_idx, 
            positive_only=positive_only,
            hide_rest=hide_rest,
            num_features=10
        )
        
        return temp, mask


def compare_explanations(original_image, gradcam_result, lime_result, save_path=None):
    """
    Compare GradCAM and LIME explanations side by side.
    
    Args:
        original_image: Original input image
        gradcam_result: GradCAM visualization
        lime_result: LIME visualization  
        save_path: Optional path to save comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GradCAM
    axes[1].imshow(gradcam_result)
    axes[1].set_title('GradCAM')
    axes[1].axis('off')
    
    # LIME
    axes[2].imshow(lime_result)
    axes[2].set_title('LIME')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Explanation comparison saved to {save_path}")
    
    plt.show()


def generate_explanations_for_dataset(model, data_loader, architecture, output_dir, 
                                    num_samples=50, method='both'):
    """
    Generate explanations for a subset of the dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        architecture: Model architecture name
        output_dir: Directory to save explanations
        num_samples: Number of samples to explain
        method: 'gradcam', 'lime', or 'both'
    """
    import os
    from ..utils.helpers import create_directory
    
    create_directory(output_dir)
    
    # Initialize explainers
    gradcam = None
    lime_explainer = None
    
    if method in ['gradcam', 'both']:
        target_layer = get_gradcam_layer(model, architecture)
        gradcam = GradCAM(model, target_layer)
        
    if method in ['lime', 'both']:
        lime_explainer = LIMEExplainer(model)
    
    model.eval()
    count = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if count >= num_samples:
                break
                
            for i in range(images.shape[0]):
                if count >= num_samples:
                    break
                    
                image = images[i:i+1]  # Keep batch dimension
                label = labels[i].item()
                
                # Generate explanations
                if gradcam:
                    gradcam_viz, cam = gradcam.visualize_cam(image)
                    
                    # Save GradCAM
                    gradcam_path = os.path.join(output_dir, f'gradcam_{count}_label_{label}.png')
                    plt.imsave(gradcam_path, gradcam_viz)
                
                if lime_explainer:
                    # Convert to numpy for LIME
                    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
                    
                    # Denormalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                    
                    # Generate LIME explanation
                    explanation = lime_explainer.explain_instance(img_np)
                    lime_viz, _ = lime_explainer.visualize_explanation(explanation, img_np, label)
                    
                    # Save LIME
                    lime_path = os.path.join(output_dir, f'lime_{count}_label_{label}.png')
                    plt.imsave(lime_path, lime_viz)
                
                count += 1
                
                if count % 10 == 0:
                    print(f"Generated explanations for {count}/{num_samples} samples")
    
    print(f"Explanation generation complete. Saved to {output_dir}")