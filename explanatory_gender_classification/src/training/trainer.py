"""
Training utilities and trainer classes.
"""

import torch
import pandas as pd
from tqdm import tqdm
import os


class BaseTrainer:
    """Base trainer class for standard classification training."""
    
    def __init__(self, model, criterion, optimizer, scheduler=None, device='cuda'):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Training device
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': []
        }
        
    def train_epoch(self, train_loader):
        """Train the model for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar
            pbar.set_postfix({'Loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self, val_loader):
        """Validate the model for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def test_epoch(self, test_loader):
        """Test the model."""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = running_corrects.double() / len(test_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def train(self, train_loader, val_loader, test_loader, epochs, early_stopper=None, save_path=None):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            epochs: Number of training epochs
            early_stopper: Early stopping object
            save_path: Path to save the best model
            
        Returns:
            Training history DataFrame
        """
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 10)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Testing
            test_loss, test_acc = self.test_epoch(test_loader)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)
            
            # Print statistics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved new best model with val_acc: {val_acc:.4f}')
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            if early_stopper:
                if early_stopper.early_stop(val_loss):
                    print("Early stopping triggered")
                    break
                    
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Epoch': range(1, len(self.history['train_loss']) + 1),
            'Train Loss': self.history['train_loss'],
            'Train Accuracy': self.history['train_acc'],
            'Validation Loss': self.history['val_loss'], 
            'Validation Accuracy': self.history['val_acc'],
            'Test Loss': self.history['test_loss'],
            'Test Accuracy': self.history['test_acc'],
            'Learning Rate': self.history['lr']
        })
        
        return results_df


class EarlyStopper:
    """Early stopping utility."""
    
    def __init__(self, patience=3, min_delta=0):
        """
        Initialize early stopper.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        
    def early_stop(self, validation_loss):
        """
        Check if training should stop early.
        
        Args:
            validation_loss: Current validation loss
            
        Returns:
            True if should stop, False otherwise
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def save_results(results_df, save_dir, experiment_name):
    """
    Save training results to CSV.
    
    Args:
        results_df: Results DataFrame
        save_dir: Directory to save results
        experiment_name: Name of the experiment
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{experiment_name}_training_results.csv')
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
    
def load_results(results_path):
    """
    Load training results from CSV.
    
    Args:
        results_path: Path to results CSV file
        
    Returns:
        Results DataFrame
    """
    return pd.read_csv(results_path)