"""
Model architectures for gender classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class GenderClassifier(nn.Module):
    """Base gender classifier with configurable backbone."""
    
    def __init__(self, architecture='efficientnet_b0', num_classes=2, pretrained=True, freeze_backbone=True):
        """
        Initialize the gender classifier.
        
        Args:
            architecture: Name of the backbone architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
        """
        super(GenderClassifier, self).__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        
        # Create backbone model
        self.backbone = self._create_backbone(architecture, pretrained)
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            self._freeze_backbone()
            
        # Get the number of input features for the classifier
        fc_inputs = self._get_fc_inputs()
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)  # For NLLLoss
        )
        
        # Replace the original classifier
        self._replace_classifier()
        
    def _create_backbone(self, architecture, pretrained):
        """Create the backbone model based on architecture name."""
        
        if architecture == 'efficientnet_b0':
            if pretrained:
                model = EfficientNet.from_pretrained('efficientnet-b0')
            else:
                model = EfficientNet.from_name('efficientnet-b0')
        elif architecture == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif architecture == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
        elif architecture == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
        elif architecture == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
        elif architecture == 'googlenet':
            model = models.googlenet(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
            
        return model
    
    def _freeze_backbone(self):
        """Freeze backbone parameters except the final layer."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze the final layer based on architecture
        if self.architecture == 'efficientnet_b0':
            for param in self.backbone._fc.parameters():
                param.requires_grad = True
        elif self.architecture in ['resnet50']:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.architecture == 'densenet121':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.architecture == 'vgg16':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.architecture == 'mobilenet_v2':
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif self.architecture == 'googlenet':
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
    
    def _get_fc_inputs(self):
        """Get the number of input features for the final classifier."""
        if self.architecture == 'efficientnet_b0':
            return self.backbone._fc.in_features
        elif self.architecture == 'resnet50':
            return self.backbone.fc.in_features
        elif self.architecture == 'densenet121':
            return self.backbone.classifier.in_features
        elif self.architecture == 'vgg16':
            return self.backbone.classifier[0].in_features
        elif self.architecture == 'mobilenet_v2':
            return self.backbone.classifier[1].in_features
        elif self.architecture == 'googlenet':
            return self.backbone.fc.in_features
        else:
            raise ValueError(f"FC inputs not defined for architecture: {self.architecture}")
    
    def _replace_classifier(self):
        """Replace the original classifier with our custom classifier."""
        if self.architecture == 'efficientnet_b0':
            self.backbone._fc = self.classifier
        elif self.architecture == 'resnet50':
            self.backbone.fc = self.classifier
        elif self.architecture == 'densenet121':
            self.backbone.classifier = self.classifier
        elif self.architecture == 'vgg16':
            self.backbone.classifier = self.classifier
        elif self.architecture == 'mobilenet_v2':
            self.backbone.classifier = self.classifier
        elif self.architecture == 'googlenet':
            self.backbone.fc = self.classifier
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.backbone(x)
    
    def get_features(self, x):
        """Get feature maps from the backbone (useful for explainability)."""
        if self.architecture == 'efficientnet_b0':
            x = self.backbone.extract_features(x)
        elif self.architecture == 'resnet50':
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
        else:
            # For other architectures, just use the features method if available
            if hasattr(self.backbone, 'features'):
                x = self.backbone.features(x)
            else:
                raise NotImplementedError(f"Feature extraction not implemented for {self.architecture}")
        
        return x


def create_model(architecture='efficientnet_b0', num_classes=2, pretrained=True, freeze_backbone=True):
    """
    Factory function to create a gender classifier model.
    
    Args:
        architecture: Name of the backbone architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        
    Returns:
        GenderClassifier model
    """
    return GenderClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


def load_model(model_path, architecture='efficientnet_b0', num_classes=2):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model
        architecture: Model architecture
        num_classes: Number of output classes
        
    Returns:
        Loaded model
    """
    model = create_model(architecture, num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model