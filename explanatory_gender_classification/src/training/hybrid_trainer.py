"""
Hybrid XIL training approach combining CAIPI and RRR.

This module implements the hybrid system that integrates both CAIPI data augmentation
and RRR regularization for improved bias mitigation in gender classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
from tqdm import tqdm
import sys
import os

# Add the src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from models.rrr_model import RRRTrainer
from augmentation.caipi import CAIPIAugmentation, apply_caipi_sampling
from evaluation.bias_metrics import evaluate_model_bias, BiasMetricsTracker


class HybridXILTrainer:
    """
    Hybrid trainer that combines CAIPI augmentation with RRR regularization.
    
    The hybrid approach:
    1. Selects samples using uncertainty/confidence sampling
    2. Generates CAIPI counterexamples for selected samples
    3. Trains with both augmented data and RRR regularization
    """
    
    def __init__(self,
                 model: nn.Module,
                 base_dataset,
                 explainer,
                 device: str = 'cpu',
                 learning_rate: float = 1e-4,
                 rrr_lambda: float = 10.0,
                 caipi_k: int = 3,
                 num_caipi_samples: int = 50,
                 sampling_strategy: str = 'uncertainty'):
        """
        Initialize hybrid XIL trainer.
        
        Args:
            model: Model to train (should support explanations)
            base_dataset: Original training dataset
            explainer: Explainability method (GradCAM or BLA)
            device: Device to run training on
            learning_rate: Learning rate for optimizer
            rrr_lambda: Weight for RRR regularization loss
            caipi_k: Number of counterexamples per CAIPI sample
            num_caipi_samples: Number of samples to select for CAIPI
            sampling_strategy: 'uncertainty' or 'high_confidence'
        """
        self.model = model.to(device)
        self.base_dataset = base_dataset
        self.explainer = explainer
        self.device = device
        self.rrr_lambda = rrr_lambda
        self.caipi_k = caipi_k
        self.num_caipi_samples = num_caipi_samples
        self.sampling_strategy = sampling_strategy
        
        # Initialize optimizer and loss for RRR trainer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize RRR trainer for regularization
        self.rrr_trainer = RRRTrainer(
            model=model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=device,
            l2_grads=rrr_lambda
        )
        
        # Initialize CAIPI augmentation
        self.caipi = CAIPIAugmentation(k=caipi_k)
        
        # Metrics tracking
        self.metrics_tracker = BiasMetricsTracker()
        
        # Training history
        self.train_history = {
            'epoch': [],
            'classification_loss': [],
            'rrr_loss': [],
            'total_loss': [],
            'accuracy': [],
            'bias_metrics': []
        }
        
    def generate_caipi_augmented_dataset(self) -> TensorDataset:
        """
        Generate CAIPI augmented dataset from selected samples.
        
        Returns:
            TensorDataset with augmented samples
        """
        print(f"Generating CAIPI augmented dataset with k={self.caipi_k}, "
              f"samples={self.num_caipi_samples}, strategy={self.sampling_strategy}")
        
        # Apply CAIPI sampling to get augmented samples
        augmented_samples = apply_caipi_sampling(
            dataset=self.base_dataset,
            model=self.model,
            explainer=self.explainer,
            num_samples=self.num_caipi_samples,
            k=self.caipi_k,
            sampling_strategy=self.sampling_strategy,
            device=self.device
        )
        
        # Separate images, masks, and labels
        aug_images = []
        aug_labels = []
        aug_masks = []
        
        for img, mask, label in augmented_samples:
            aug_images.append(img)
            aug_labels.append(label)
            aug_masks.append(mask)
        
        # Convert to tensors
        aug_images_tensor = torch.stack(aug_images)
        aug_labels_tensor = torch.tensor(aug_labels, dtype=torch.long)
        aug_masks_tensor = torch.stack(aug_masks)
        
        print(f"Generated {len(aug_images)} augmented samples")
        
        return TensorDataset(aug_images_tensor, aug_labels_tensor, aug_masks_tensor)
    
    def create_hybrid_dataloader(self, 
                                batch_size: int = 32, 
                                include_original: bool = True) -> DataLoader:
        """
        Create dataloader that combines original and CAIPI-augmented data.
        
        Args:
            batch_size: Batch size for training
            include_original: Whether to include original training data
            
        Returns:
            Combined DataLoader
        """
        datasets_to_combine = []
        
        # Add original dataset if requested
        if include_original:
            datasets_to_combine.append(self.base_dataset)
            
        # Add CAIPI augmented dataset
        caipi_dataset = self.generate_caipi_augmented_dataset()
        datasets_to_combine.append(caipi_dataset)
        
        # Combine datasets
        combined_dataset = ConcatDataset(datasets_to_combine)
        
        return DataLoader(
            combined_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True
        )
    
    def train_epoch(self, 
                   dataloader: DataLoader, 
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Train one epoch with hybrid approach.
        
        Args:
            dataloader: Training dataloader
            optimizer: Optimizer for training
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_classification_loss = 0.0
        total_rrr_loss = 0.0
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        classification_criterion = nn.CrossEntropyLoss()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            if len(batch) == 3:
                images, labels, masks = batch
            else:
                images, labels = batch
                # If no masks, create dummy masks (all relevant)
                masks = torch.zeros(images.shape[0], images.shape[2], images.shape[3])
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and len(self.model(images)) == 2:
                # Model with explanations (BLA)
                logits, explanations = self.model(images)
            else:
                # Regular model
                logits = self.model(images)
                explanations = None
            
            # Classification loss
            classification_loss = classification_criterion(logits, labels)
            
            # RRR regularization loss
            if explanations is not None:
                # Use explanations directly for RRR
                rrr_loss = self.rrr_trainer.compute_rrr_loss_from_explanations(
                    images, logits, explanations, masks
                )
            else:
                # Compute RRR loss using gradients
                rrr_loss = self.rrr_trainer.compute_rrr_loss(images, logits, masks)
            
            # Combined loss
            total_batch_loss = classification_loss + self.rrr_lambda * rrr_loss
            
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()
            
            # Track metrics
            total_classification_loss += classification_loss.item()
            total_rrr_loss += rrr_loss.item()
            total_loss += total_batch_loss.item()
            
            # Accuracy
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # Calculate average metrics
        num_batches = len(dataloader)
        epoch_metrics = {
            'classification_loss': total_classification_loss / num_batches,
            'rrr_loss': total_rrr_loss / num_batches,
            'total_loss': total_loss / num_batches,
            'accuracy': correct_predictions / total_samples
        }
        
        return epoch_metrics
    
    def evaluate_bias_metrics(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate bias metrics on validation/test set.
        
        Args:
            eval_dataloader: Evaluation dataloader
            
        Returns:
            Dictionary of bias metrics
        """
        return evaluate_model_bias(
            model=self.model,
            dataloader=eval_dataloader,
            explainer=self.explainer,
            device=self.device
        )
    
    def train(self, 
             num_epochs: int = 10,
             batch_size: int = 32,
             eval_dataloader: Optional[DataLoader] = None,
             save_path: Optional[str] = None,
             evaluate_every: int = 5) -> Dict[str, List]:
        """
        Train the model using hybrid CAIPI + RRR approach.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            eval_dataloader: Optional evaluation dataloader
            save_path: Optional path to save model checkpoints
            evaluate_every: Evaluate bias metrics every N epochs
            
        Returns:
            Training history dictionary
        """
        print(f"Starting Hybrid XIL Training")
        print(f"Configuration:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - RRR lambda: {self.rrr_lambda}")
        print(f"  - CAIPI k: {self.caipi_k}")
        print(f"  - CAIPI samples: {self.num_caipi_samples}")
        print(f"  - Sampling strategy: {self.sampling_strategy}")
        
        # Create hybrid dataloader
        train_dataloader = self.create_hybrid_dataloader(
            batch_size=batch_size, 
            include_original=True
        )
        
        # Initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train one epoch
            epoch_metrics = self.train_epoch(train_dataloader, optimizer)
            
            # Update history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['classification_loss'].append(epoch_metrics['classification_loss'])
            self.train_history['rrr_loss'].append(epoch_metrics['rrr_loss'])
            self.train_history['total_loss'].append(epoch_metrics['total_loss'])
            self.train_history['accuracy'].append(epoch_metrics['accuracy'])
            
            # Print epoch results
            print(f"Classification Loss: {epoch_metrics['classification_loss']:.4f}")
            print(f"RRR Loss: {epoch_metrics['rrr_loss']:.4f}")
            print(f"Total Loss: {epoch_metrics['total_loss']:.4f}")
            print(f"Accuracy: {epoch_metrics['accuracy']:.4f}")
            
            # Evaluate bias metrics periodically
            if eval_dataloader is not None and (epoch + 1) % evaluate_every == 0:
                print("\nEvaluating bias metrics...")
                bias_metrics = self.evaluate_bias_metrics(eval_dataloader)
                
                self.train_history['bias_metrics'].append(bias_metrics)
                self.metrics_tracker.update(bias_metrics)
                
                print(f"FFP: {bias_metrics['FFP']:.3f} (higher better)")
                print(f"BFP: {bias_metrics['BFP']:.3f} (lower better)")
                print(f"BSR: {bias_metrics['BSR']:.3f} (lower better)")
                print(f"DICE: {bias_metrics['DICE']:.3f} (higher better)")
            
            # Save checkpoint
            if save_path is not None and (epoch + 1) % 5 == 0:
                checkpoint_path = f"{save_path}_epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_history': self.train_history,
                    'config': {
                        'rrr_lambda': self.rrr_lambda,
                        'caipi_k': self.caipi_k,
                        'num_caipi_samples': self.num_caipi_samples,
                        'sampling_strategy': self.sampling_strategy
                    }
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        
        # Final evaluation
        if eval_dataloader is not None:
            print("\nFinal bias metrics evaluation:")
            final_bias_metrics = self.evaluate_bias_metrics(eval_dataloader)
            self.metrics_tracker.update(final_bias_metrics)
            self.metrics_tracker.print_summary()
        
        return self.train_history
    
    def save_model(self, save_path: str):
        """Save the trained model and training history."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_history': self.train_history,
            'metrics_history': self.metrics_tracker.metrics_history,
            'config': {
                'rrr_lambda': self.rrr_lambda,
                'caipi_k': self.caipi_k,
                'num_caipi_samples': self.num_caipi_samples,
                'sampling_strategy': self.sampling_strategy
            }
        }, save_path)
        print(f"Model saved to: {save_path}")


def run_hybrid_experiments(model,
                         train_dataset,
                         val_dataset,
                         explainer,
                         device: str = 'cpu',
                         save_dir: str = 'hybrid_experiments') -> Dict:
    """
    Run comprehensive hybrid experiments with different configurations.
    
    Args:
        model: Base model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        explainer: Explainability method
        device: Device to run on
        save_dir: Directory to save results
        
    Returns:
        Dictionary of experimental results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Experiment configurations from the paper
    configurations = [
        # Uncertain sampling
        {'sampling_strategy': 'uncertainty', 'k': 1},
        {'sampling_strategy': 'uncertainty', 'k': 3},
        {'sampling_strategy': 'uncertainty', 'k': 5},
        # High confidence sampling
        {'sampling_strategy': 'high_confidence', 'k': 1},
        {'sampling_strategy': 'high_confidence', 'k': 3},
        {'sampling_strategy': 'high_confidence', 'k': 5},
    ]
    
    results = {}
    
    # Create validation dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    for i, config in enumerate(configurations):
        print(f"\n{'='*60}")
        print(f"Running Hybrid Experiment {i+1}/{len(configurations)}")
        print(f"Configuration: {config}")
        print(f"{'='*60}")
        
        # Initialize trainer with current configuration
        trainer = HybridXILTrainer(
            model=model,
            base_dataset=train_dataset,
            explainer=explainer,
            device=device,
            caipi_k=config['k'],
            sampling_strategy=config['sampling_strategy']
        )
        
        # Train model
        history = trainer.train(
            num_epochs=20,
            batch_size=32,
            eval_dataloader=val_dataloader,
            save_path=os.path.join(save_dir, f"hybrid_{config['sampling_strategy']}_k{config['k']}"),
            evaluate_every=5
        )
        
        # Store results
        config_name = f"{config['sampling_strategy']}_k{config['k']}"
        results[config_name] = {
            'config': config,
            'history': history,
            'final_metrics': trainer.metrics_tracker.get_latest()
        }
        
        print(f"\nCompleted experiment: {config_name}")
    
    # Save consolidated results
    results_path = os.path.join(save_dir, 'hybrid_experiments_results.pth')
    torch.save(results, results_path)
    print(f"\nAll hybrid experiments completed. Results saved to: {results_path}")
    
    return results