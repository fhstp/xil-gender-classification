"""CAIPI (Counterexample-Augmented Interactive Learning) trainer."""
import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer import BaseTrainer

class CAIPITrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, caipi_augmenter, scheduler=None, device='cuda'):
        super().__init__(model, criterion, optimizer, scheduler, device)
        self.caipi = caipi_augmenter
        
    def train_epoch(self, train_loader):
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc="Training CAIPI", leave=False)
        for images, masks, labels in pbar:
            # Generate counterfactuals via CAIPI
            # CAIPI expects mask=0 for relevant, mask=1 for irrelevant
            aug_images_list = []
            aug_labels_list = []
            
            for i in range(len(images)):
                counterexamples = self.caipi.generate_counterexamples(images[i], masks[i], labels[i].item())
                aug_images_list.extend([c[0] for c in counterexamples])
                aug_labels_list.extend([c[1] for c in counterexamples])
                
            if len(aug_images_list) > 0:
                aug_images = torch.stack(aug_images_list)
                aug_labels = torch.tensor(aug_labels_list)
                
                # Combine original and augmented
                batch_images = torch.cat([images, aug_images], dim=0)
                batch_labels = torch.cat([labels, aug_labels], dim=0)
            else:
                batch_images = images
                batch_labels = labels
                
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * batch_images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == batch_labels.data).item()
            total_samples += batch_images.size(0)
            
            pbar.set_postfix({'Loss': loss.item()})
            
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = running_corrects / total_samples if total_samples > 0 else 0
        return epoch_loss, epoch_acc
            
    def validate_epoch(self, val_loader):
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Accept varying batch unpacks based on the dataset logic
                images = batch[0]
                labels = batch[1] if len(batch) == 2 else batch[-1]
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += images.size(0)
                
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = running_corrects / total_samples if total_samples > 0 else 0
        
        return epoch_loss, epoch_acc
        
    def test_epoch(self, test_loader):
        return self.validate_epoch(test_loader)
        
    def test(self, test_loader):
        return self.test_epoch(test_loader)
