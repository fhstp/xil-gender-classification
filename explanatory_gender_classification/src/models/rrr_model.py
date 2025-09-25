"""
Right for the Right Reasons (RRR) model implementation.
"""

import torch
import torch.nn as nn
from .architectures import GenderClassifier


class RRRGenderClassifier(GenderClassifier):
    """Gender classifier with Right for Right Reasons training."""
    
    def __init__(self, architecture='efficientnet_b0', num_classes=2, pretrained=True, freeze_backbone=True):
        """Initialize RRR gender classifier."""
        super(RRRGenderClassifier, self).__init__(
            architecture=architecture,
            num_classes=num_classes, 
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # Log softmax layer for RRR loss computation
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature return for RRR.
        
        Args:
            x: Input tensor
            return_features: Whether to return intermediate features
            
        Returns:
            Model output, optionally with features
        """
        if return_features:
            features = self.get_features(x)
            output = self.backbone(x)
            return output, features
        else:
            return self.backbone(x)


def rrr_loss_function(A, X, y, logits, criterion, class_weights=None, l2_grads=1000, reduce_func=torch.sum):
    """
    Right for the Right Reasons loss function.
    
    This implements the RRR loss from the paper which regularizes input gradients
    in locations specified by annotation masks.
    
    Args:
        A: Annotation masks (N x H x W) - indicates regions of interest
        X: Input images (N x C x H x W) - requires gradients
        y: Target labels (N,)
        logits: Model outputs before final activation (N x num_classes)
        criterion: Base loss function (e.g., NLLLoss)
        class_weights: Optional class weights for balancing
        l2_grads: Lambda coefficient for gradient regularization
        reduce_func: Function to reduce the gradient penalty (sum or mean)
        
    Returns:
        Tuple of (total_loss, right_answer_loss, right_reason_loss)
    """
    
    # Apply log softmax to logits
    log_softmax = nn.LogSoftmax(dim=1)
    log_prob_ys = log_softmax(logits)
    
    # Right answers loss - standard classification loss
    right_answer_loss = criterion(log_prob_ys, y)
    
    # Right reasons loss - gradient regularization
    # Calculate gradients of log probabilities w.r.t. input
    gradXes = torch.autograd.grad(
        outputs=log_prob_ys,
        inputs=X,
        grad_outputs=torch.ones_like(log_prob_ys),
        create_graph=True
    )[0]
    
    # Expand annotation masks to match gradient dimensions
    # A might be (N x 1 x H x W) or (N x H x W), gradXes is (N x C x H x W)
    
    # Squeeze any extra dimensions from A
    while A.dim() > 3:
        A = A.squeeze(1)
    
    # Now A should be (N x H x W)
    A_expanded = A.unsqueeze(1)  # (N x 1 x H x W)
    A_expanded = A_expanded.expand(-1, gradXes.shape[1], -1, -1)  # (N x C x H x W)
    
    # Element-wise multiplication of annotations and gradients, then square
    A_gradX = torch.mul(A_expanded, gradXes) ** 2
    
    # Apply class weights if provided
    if class_weights is not None:
        # Create class weight tensor for each sample in batch
        class_weights_batch = torch.zeros_like(y, dtype=torch.float)
        for i, class_weight in enumerate(class_weights):
            class_weights_batch[y == i] = class_weight
    else:
        class_weights_batch = torch.ones_like(y, dtype=torch.float)
    
    # Sum over spatial and channel dimensions to get penalty per sample
    # Sum over dimensions 1, 2, 3 (C, H, W) keeping batch dimension
    right_reason_penalty = torch.sum(A_gradX, dim=(1, 2, 3))
    
    # Apply class weights and reduction
    weighted_penalty = class_weights_batch * right_reason_penalty
    right_reason_loss = reduce_func(weighted_penalty) * l2_grads
    
    # Total loss is sum of both components
    total_loss = right_answer_loss + right_reason_loss
    
    return total_loss, right_answer_loss, right_reason_loss


class RRRTrainer:
    """Trainer specifically for RRR models."""
    
    def __init__(self, model, criterion, optimizer, scheduler=None, device='cuda', 
                 l2_grads=1000, class_weights=None):
        """
        Initialize RRR trainer.
        
        Args:
            model: RRR model to train
            criterion: Base loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Training device
            l2_grads: Lambda for gradient regularization
            class_weights: Optional class weights
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.l2_grads = l2_grads
        self.class_weights = class_weights
        
    def train_epoch(self, train_loader):
        """Train for one epoch with RRR loss."""
        self.model.train()
        
        running_loss = 0.0
        running_answer_loss = 0.0
        running_reason_loss = 0.0
        running_corrects = 0
        
        for images, masks, labels in train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # Ensure images require gradients for RRR
            images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute RRR loss
            loss, answer_loss, reason_loss = rrr_loss_function(
                A=masks,
                X=images, 
                y=labels,
                logits=outputs,
                criterion=self.criterion,
                class_weights=self.class_weights,
                l2_grads=self.l2_grads
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            running_answer_loss += answer_loss.item() * images.size(0)
            running_reason_loss += reason_loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_answer_loss = running_answer_loss / len(train_loader.dataset)
        epoch_reason_loss = running_reason_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        return {
            'loss': epoch_loss,
            'answer_loss': epoch_answer_loss,
            'reason_loss': epoch_reason_loss,
            'accuracy': epoch_acc.item()
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc.item()
        }