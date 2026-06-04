"""Quick-start training script for baseline gender classification with BLA."""
import torch
import torch.nn as nn
import os
from src.data.dataset import GenderDataset
from src.explainability.bla import create_bla_model
from src.evaluation.bias_metrics import evaluate_model_bias
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('results/journal_extension/coco_bbox/checkpoints', exist_ok=True)

train_dataset = GenderDataset('gender_dataset', split='train')
test_dataset = GenderDataset('gender_dataset', split='test')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3 Seeds for the baseline
for seed in [42, 1, 2]:
    torch.manual_seed(seed)
    model = create_bla_model('efficientnet_b0', num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(20):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            
    model.eval()
    explainer = model.get_explanation_wrapper()
    metrics = evaluate_model_bias(model, test_loader, explainer, device)
    print(f'Seed {seed} Metrics:', metrics)
    torch.save(model.state_dict(), f'results/journal_extension/coco_bbox/checkpoints/baseline_none_k0_none_bla_{seed}.pth')