"""
Hybrid XIL trainer combining CAIPI data augmentation with RRR gradient regularization.

Strategy:
  1. For each batch, generate CAIPI counterexamples (perturbed irrelevant regions).
  2. Compute CE loss on original images (with grad enabled) + augmented images.
  3. Add RRR gradient penalty on original images weighted by the feedback mask.

This matches the paper description:
  "We first augment the dataset using CAIPI, then train the model
   incorporating the RRR loss function."
"""

import torch
import torch.nn as nn
from tqdm import tqdm

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.trainer import BaseTrainer
from augmentation.caipi import CAIPIAugmentation


class HybridXILTrainer(BaseTrainer):
    """
    Trainer that combines CAIPI augmentation with RRR gradient regularization.

    Constructor mirrors BaseTrainer so the experiment runner can create it
    with the same (model, criterion, optimizer, device) pattern as other trainers.
    Extra keyword-only args:
        caipi_k     -- counterexamples per selected sample (default 3)
        rrr_lambda  -- weight for the RRR gradient penalty (default 1000.0)
    """

    def __init__(self, model, criterion, optimizer, scheduler=None, device='cuda',
                 caipi_k: int = 3, rrr_lambda: float = 1000.0):
        super().__init__(model, criterion, optimizer, scheduler, device)
        self.caipi = CAIPIAugmentation(k=caipi_k)
        self.rrr_lambda = rrr_lambda

    # ------------------------------------------------------------------
    # Training epoch: CAIPI augmentation + RRR loss
    # ------------------------------------------------------------------
    def train_epoch(self, train_loader):
        """Train one epoch with CAIPI counterexamples + RRR gradient penalty."""
        self.model.train()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc="Training Hybrid", leave=False)
        for batch in pbar:
            # Batch format from TrainLoaderWrapper (hybrid):
            #   batch[0] = images, batch[1] = feedback_mask (0=person, 1=bg), batch[2] = labels
            images = batch[0]
            masks  = batch[1] if len(batch) >= 3 else torch.zeros(
                images.shape[0], images.shape[2], images.shape[3])
            labels = batch[-1]

            # ── 1. Generate CAIPI counterexamples ─────────────────────
            aug_images_list: list = []
            aug_labels_list: list = []
            for i in range(len(images)):
                ces = self.caipi.generate_counterexamples(
                    images[i], masks[i], labels[i].item())
                aug_images_list.extend(c[0] for c in ces)
                aug_labels_list.extend(c[1] for c in ces)

            # ── 2. CE loss on originals (requires_grad for RRR penalty) ─
            orig_imgs = images.to(self.device)
            orig_imgs.requires_grad_(True)
            orig_labels = labels.to(self.device)
            orig_masks  = masks.to(self.device)

            self.optimizer.zero_grad()

            orig_out = self.model(orig_imgs)
            if isinstance(orig_out, tuple):
                orig_out = orig_out[0]
            ce_loss = self.criterion(orig_out, orig_labels)

            # ── 3. RRR gradient penalty ────────────────────────────────
            # d(sum(logits)) / d(input pixels) — penalised in irrelevant regions
            grad_x = torch.autograd.grad(
                outputs=orig_out.sum(),
                inputs=orig_imgs,
                create_graph=True,
            )[0]  # (B, C, H, W)

            # Normalise mask to (B, H, W) then expand to (B, C, H, W)
            m = orig_masks
            if m.dim() == 4:
                m = m.squeeze(1)
            elif m.dim() == 2:
                m = m.unsqueeze(0)
            m = m.unsqueeze(1).expand_as(grad_x)   # (B, C, H, W)

            rrr_penalty = self.rrr_lambda * (m * grad_x).pow(2).mean()

            # ── 4. CE loss on CAIPI augmented counterexamples ──────────
            aug_ce_loss = torch.tensor(0.0, device=self.device)
            if aug_images_list:
                aug_imgs_t = torch.stack(aug_images_list).to(self.device)
                aug_lbs_t  = torch.tensor(aug_labels_list, dtype=torch.long,
                                          device=self.device)
                aug_out = self.model(aug_imgs_t)
                if isinstance(aug_out, tuple):
                    aug_out = aug_out[0]
                aug_ce_loss = self.criterion(aug_out, aug_lbs_t)

            total_loss = ce_loss + rrr_penalty + aug_ce_loss

            total_loss.backward()
            self.optimizer.step()

            # ── Metrics (on original images only) ─────────────────────
            with torch.no_grad():
                _, preds = torch.max(orig_out.detach(), 1)
                running_corrects += (preds == orig_labels).sum().item()
            running_loss  += total_loss.item() * images.size(0)
            total_samples += images.size(0)

            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        epoch_loss = running_loss  / total_samples if total_samples else 0.0
        epoch_acc  = running_corrects / total_samples if total_samples else 0.0
        return epoch_loss, epoch_acc

    # ------------------------------------------------------------------
    # Validation / test: no masks needed, flexible batch size
    # ------------------------------------------------------------------
    def validate_epoch(self, val_loader):
        self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(self.device)
                labels = batch[-1].to(self.device)

                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = self.criterion(outputs, labels)
                running_loss  += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += (preds == labels).sum().item()
                total_samples += images.size(0)

        epoch_loss = running_loss  / total_samples if total_samples else 0.0
        epoch_acc  = running_corrects / total_samples if total_samples else 0.0
        return epoch_loss, epoch_acc

    def test_epoch(self, test_loader):
        return self.validate_epoch(test_loader)
