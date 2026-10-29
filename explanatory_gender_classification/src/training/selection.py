"""
Budgeted-feedback sampling for XIL (CAIPI / RRR / hybrid).

In the protocol described in the CBMI paper and in the journal manuscript, only a small number of
training samples receive feedback per epoch; they are chosen by a sampling strategy:

  * uncertainty:      the samples with the LOWEST maximum class probability
  * high_confidence:  the correctly classified samples with the HIGHEST maximum class probability
                      (CBMI: correct predictions with confidence > 0.9; if fewer than `n_select`
                      samples qualify, the remaining slots are filled with the next most confident
                      correctly classified samples, then with the most confident remaining ones)

Samples that are not selected keep contributing to the classification loss but receive an all-zero
feedback mask, i.e. no explanation penalty (RRR) and no counterexamples (CAIPI).

`--sampling_mode all` in run_bbox_feedback_experiments.py (the earlier behaviour) gives feedback for
every training image in every epoch and does not use this module.
"""

import torch
from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    """Wraps a dataset so that every item additionally returns its dataset index (last element)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return (*item, idx)


@torch.no_grad()
def score_samples(model, score_loader, device):
    """Return (max_prob, correct) tensors for every sample of `score_loader` (no augmentation, fixed order)."""
    model.eval()
    max_probs, corrects = [], []
    for batch in score_loader:
        images, labels = batch[0].to(device), batch[1]
        out = model(images)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)  # softmax of log-probabilities equals the probabilities
        mp, pred = probs.max(dim=1)
        max_probs.append(mp.cpu())
        corrects.append((pred.cpu() == torch.as_tensor(labels)))
    return torch.cat(max_probs), torch.cat(corrects)


def select_indices(max_prob, correct, strategy, n_select, confidence_threshold=0.9):
    """Pick `n_select` dataset indices according to `strategy` (see module docstring)."""
    n = len(max_prob)
    n_select = min(n_select, n)
    if strategy == 'uncertainty':
        order = torch.argsort(max_prob, stable=True)  # ascending: least confident first
        return set(order[:n_select].tolist())
    if strategy == 'high_confidence':
        # priority 0: correct and > threshold, 1: correct, 2: everything else; within a group by confidence
        priority = torch.full((n,), 2, dtype=torch.long)
        priority[correct] = 1
        priority[correct & (max_prob > confidence_threshold)] = 0
        key = priority.double() * 10.0 - max_prob.double()
        order = torch.argsort(key, stable=True)
        return set(order[:n_select].tolist())
    raise ValueError(f"unknown sampling strategy: {strategy}")


class BudgetedFeedbackLoader:
    """
    Training-loader wrapper for `--sampling_mode selected`.

    At the start of every epoch (every call to __iter__) the CURRENT model scores the training set,
    `n_select` samples are chosen, and only those samples keep their feedback mask. Yields
    (images, feedback_mask, labels), the tuple layout that the XIL trainers expect.
    The selected indices of every epoch are kept in `self.history` for auditing.
    """

    def __init__(self, indexed_loader, score_loader, model, strategy, n_select, device):
        self.loader = indexed_loader
        self.score_loader = score_loader
        self.model = model
        self.strategy = strategy
        self.n_select = n_select
        self.device = device
        self.dataset = indexed_loader.dataset
        self.history = []

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        max_prob, correct = score_samples(self.model, self.score_loader, self.device)
        selected = select_indices(max_prob, correct, self.strategy, self.n_select)
        self.history.append(sorted(selected))
        for batch in self.loader:
            images, labels, feedback_mask, _fg, _meta, idx = batch
            gate = torch.tensor([int(i) in selected for i in idx], dtype=feedback_mask.dtype)
            yield images, feedback_mask * gate.view(-1, *([1] * (feedback_mask.dim() - 1))), labels
