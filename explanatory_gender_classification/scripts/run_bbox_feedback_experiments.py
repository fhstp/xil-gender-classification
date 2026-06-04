"""Run bounding-box feedback experiments comparing bbox vs segmentation mask feedback."""
import os
import argparse
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import time
import json
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import prepare_data_splits, prepare_data_splits_from_dataset_folder, get_mask_directories
from data.xil_dataset import create_xil_data_loaders
from models.architectures import create_model
from models.rrr_model import RRRGenderClassifier
from training.trainer import BaseTrainer
from training.rrr_trainer import RRRTrainer
from training.hybrid_trainer import HybridXILTrainer
from training.caipi_trainer import CAIPITrainer
from augmentation.caipi import CAIPIAugmentation
from utils.helpers import set_random_seeds, get_device
from utils.settings import Config
from evaluation.bias_metrics import evaluate_model_bias
from evaluation.classification_metrics import compute_classification_metrics
from explainability.gradcam import GradCAM
from explainability.bla import BLAWrapper, create_bla_model

def evaluate_and_log(model, test_loader, device, explainer, results_dir, exp_config, run_id):
    """Evaluate model and save detailed metrics."""
    print(f"  -> Classification pass...", flush=True)
    
    # Standard classification metrics
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 5: # XIL Dataset
                images, labels, _, _, metadata = batch
            else:
                images, labels = batch[:2]
                metadata = {}
                
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            # GenderClassifier outputs log-probs (max ≤ 0); ModelWithBLA outputs raw logits
            probs = (torch.exp(outputs)
                     if outputs.max().item() <= 0.0
                     else torch.softmax(outputs, dim=1))
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Prob of male (class 1)
            
            # Reconstruct metadata dict of lists
            batch_metadata = []
            for i in range(len(preds)):
                item_meta = {k: v[i] if isinstance(v, (list, tuple, torch.Tensor)) else v for k, v in metadata.items()}
                batch_metadata.append(item_meta)
            all_metadata.extend(batch_metadata)
            
    # Compute Classification Metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    clf_metrics = compute_classification_metrics(all_labels, all_preds)
    
    # Compute Bias Metrics (Requires Explanations)
    n_test = sum(len(b[0]) for b in test_loader)
    print(f"  -> Bias metrics via {type(explainer).__name__} on {n_test} test samples...", flush=True)
    bias_metrics = evaluate_model_bias(
        model, test_loader, explainer, device=device
    )
    print(f"  -> Bias metrics done: FFP={bias_metrics.get('FFP',0):.3f}, BSR={bias_metrics.get('BSR',0):.3f}", flush=True)
    
    # Combine results
    results = {
        'run_id': run_id,
        'dataset': 'coco',
        'method': exp_config['method'],
        'feedback_type': exp_config['feedback_type'],
        'sampling_strategy': exp_config['sampling_strategy'],
        'k': exp_config['k'],
        'explainer': exp_config['explainer'],
        'seed': exp_config['seed'],
        **clf_metrics,
        **bias_metrics
    }
    
    # Flatten confusion matrix for CSV
    cm = results.pop('confusion_matrix')
    results['cm_tn'] = cm[0][0]
    results['cm_fp'] = cm[0][1]
    results['cm_fn'] = cm[1][0]
    results['cm_tp'] = cm[1][1]
    
    # Calculate female/male error rates
    female_total = cm[0][0] + cm[0][1]
    male_total = cm[1][0] + cm[1][1]
    results['female_error_rate'] = cm[0][1] / female_total if female_total > 0 else 0
    results['male_error_rate'] = cm[1][0] / male_total if male_total > 0 else 0
    results['error_rate_gap'] = abs(results['female_error_rate'] - results['male_error_rate'])
    
    # Log individual predictions
    df_preds = pd.DataFrame(all_metadata)
    if not df_preds.empty:
        df_preds['run_id'] = run_id
        df_preds['label_gt'] = all_labels
        df_preds['prediction'] = all_preds
        df_preds['prob_male'] = all_probs
        df_preds['correct'] = (all_labels == all_preds).astype(int)
        
        preds_path = os.path.join(results_dir, 'per_sample_predictions.csv')
        mode = 'a' if os.path.exists(preds_path) else 'w'
        header = not os.path.exists(preds_path)
        df_preds.to_csv(preds_path, mode=mode, header=header, index=False)
    
    # Log main results
    results_path = os.path.join(results_dir, 'results.csv')
    df_res = pd.DataFrame([results])
    mode = 'a' if os.path.exists(results_path) else 'w'
    header = not os.path.exists(results_path)
    df_res.to_csv(results_path, mode=mode, header=header, index=False)
    
    print(f"Results saved for run {run_id}: Acc={results['accuracy']:.4f}, BSR={results.get('BSR', 0):.4f}")
    return results

def get_explainer(model, explainer_name, device):
    """Return the explainer requested by explainer_name ('gradcam' or 'bla')."""
    if explainer_name == 'bla':
        return BLAWrapper(model, device=device)
    # GradCAM — find the last conv layer dynamically
    target_layer = None
    base_model = model.backbone if hasattr(model, 'backbone') else model
    if hasattr(base_model, '_conv_head'):
        target_layer = base_model._conv_head
    elif hasattr(base_model, 'features'):
        target_layer = base_model.features[-1]
    elif hasattr(base_model, 'layer4'):
        target_layer = base_model.layer4
    return GradCAM(model, target_layer=target_layer)

def run_experiment(exp_config, data_splits, device, args):
    """Run a single experiment configuration."""
    train_df, val_df, test_df = data_splits
    run_id = f"{exp_config['method']}_{exp_config['feedback_type']}_k{exp_config['k']}_{exp_config['sampling_strategy']}_{exp_config['explainer']}_{exp_config['seed']}"

    # Skip if this run_id already exists in results.csv (allows safe re-runs after crash)
    results_path = os.path.join(args.results_dir, 'results.csv')
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        if run_id in existing['run_id'].values:
            print(f"\nSkipping {run_id} — already in results.csv")
            return None

    print(f"\n{'='*60}", flush=True)
    print(f"  Experiment: {run_id}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Setup DataLoaders based on feedback type
    # Always load masks so bias metrics are meaningful even for the baseline.
    # The TrainLoaderWrapper strips masks from batches when the method doesn't need them.
    use_masks = True

    mask_source = "segmentation"
    if exp_config['feedback_type'] == 'bbox':
        mask_source = "bbox_from_segmentation" # Using COCO segmentation masks to derive bounding boxes

    print(f"[1/5] Creating data loaders (mask_source={mask_source})...", flush=True)
    female_masks_path, male_masks_path = get_mask_directories(args.data_dir)

    train_loader, val_loader, test_loader = create_xil_data_loaders(
        train_df, val_df, test_df,
        batch_size=Config.BATCH_SIZE,
        use_masks=use_masks,
        mask_source=mask_source,
        female_masks_path=female_masks_path,
        male_masks_path=male_masks_path
    )
    print(f"     train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)} samples", flush=True)

    # 2. Setup Model & Trainer
    start_time = time.time()
    print(f"[2/5] Setting up model and trainer (method={exp_config['method']})...", flush=True)

    if exp_config['method'] == 'baseline':
        model = create_model(pretrained=True).to(device)
        criterion = nn.NLLLoss()
        optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
        trainer = BaseTrainer(model, criterion=criterion, optimizer=optimizer, device=device)

    elif exp_config['method'] == 'caipi':
        model = create_model(pretrained=True).to(device)
        criterion = nn.NLLLoss()
        optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)

        caipi_aug = CAIPIAugmentation(k=exp_config['k'])
        trainer = CAIPITrainer(model, criterion=criterion, optimizer=optimizer, caipi_augmenter=caipi_aug, device=device)

    elif exp_config['method'] == 'rrr':
        if exp_config.get('explainer') == 'bla':
            model = create_bla_model('efficientnet_b0', num_classes=2, pretrained=True).to(device)
            criterion = nn.CrossEntropyLoss()
        else:
            model = RRRGenderClassifier(pretrained=True).to(device)
            criterion = nn.NLLLoss()
        optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
        trainer = RRRTrainer(model, criterion=criterion, optimizer=optimizer, device=device)

    elif exp_config['method'] == 'hybrid':
        if exp_config.get('explainer') == 'bla':
            # BLA needs a model that returns (logits, attention_weights)
            model = create_bla_model('efficientnet_b0', num_classes=2, pretrained=True).to(device)
            criterion = nn.CrossEntropyLoss()   # ModelWithBLA outputs raw logits
        else:
            model = create_model(pretrained=True).to(device)
            criterion = nn.NLLLoss()
        optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
        trainer = HybridXILTrainer(
            model, criterion=criterion, optimizer=optimizer, device=device,
            caipi_k=exp_config['k'], rrr_lambda=1000.0
        )
        
    class TrainLoaderWrapper:
        def __init__(self, loader, method): 
            self.loader = loader
            self.method = method
            self.dataset = loader.dataset
        def __iter__(self):
            for batch in self.loader:
                if self.method == 'baseline':
                    yield batch[0], batch[1]
                elif self.method in ['rrr', 'hybrid']:
                    # feedback_mask: 0=person (relevant), 1=background (irrelevant) — matches RRR's A convention
                    yield batch[0], batch[2], batch[1]  # image, feedback_mask, label
                else:  # caipi
                    yield batch[0], batch[2], batch[1]  # image, feedback_mask, label
        def __len__(self): return len(self.loader)

    wrapped_train_loader = TrainLoaderWrapper(train_loader, exp_config['method'])
    wrapped_val_loader = TrainLoaderWrapper(val_loader, exp_config['method'])
    wrapped_test_loader = TrainLoaderWrapper(test_loader, exp_config['method'])

    # 3. Train Model
    epochs = args.epochs if (hasattr(args, 'epochs') and args.epochs is not None) else Config.EPOCHS
    print(f"[3/5] Training for {epochs} epochs...", flush=True)
    history = trainer.train(wrapped_train_loader, wrapped_val_loader, wrapped_test_loader, epochs=epochs)
    elapsed = (time.time() - start_time) / 60
    print(f"     Training complete ({elapsed:.1f} min elapsed)", flush=True)

    # Save per-epoch training history
    if isinstance(history, pd.DataFrame) and not history.empty:
        history_dir = os.path.join(args.results_dir, 'training_history')
        os.makedirs(history_dir, exist_ok=True)
        history.insert(0, 'run_id', run_id)
        history.to_csv(os.path.join(history_dir, f'{run_id}.csv'), index=False)

    # 4. Save Checkpoint
    print(f"[4/5] Saving checkpoint...", flush=True)
    checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_id}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"     Saved to {checkpoint_path}", flush=True)

    # 5. Evaluate Model
    print(f"[5/5] Evaluating model (classification + bias metrics)...", flush=True)
    explainer = get_explainer(model, exp_config['explainer'], device)
    print(f"     Explainer: {type(explainer).__name__}", flush=True)
    
    # Wrap standard test_loader if it outputs 5 items but original expects 3 or 2
    # The bias metrics handle `if len(batch) == 3`, so we need to wrap the XIL DataLoader for evaluation.
    class EvalLoaderWrapper:
        def __init__(self, loader): self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                # XIL returns: image, label, feedback_mask, foreground_mask, metadata
                # evaluate_model_bias expects: image, label, mask (where mask is the ground truth foreground)
                yield batch[0], batch[1], batch[3] # Return foreground_mask for bias eval
        def __len__(self): return len(self.loader)

    eval_loader = EvalLoaderWrapper(test_loader)
    
    results = evaluate_and_log(
        model=model,
        test_loader=test_loader, # Pass full loader for our custom evaluate_and_log
        device=device,
        explainer=explainer,
        results_dir=args.results_dir,
        exp_config=exp_config,
        run_id=run_id
    )
    
    results['runtime_minutes'] = (time.time() - start_time) / 60
    results['model_checkpoint_path'] = checkpoint_path
    
    # Update row with runtime
    df_res = pd.read_csv(os.path.join(args.results_dir, 'results.csv'))
    df_res.loc[df_res['run_id'] == run_id, 'runtime_minutes'] = results['runtime_minutes']
    df_res.loc[df_res['run_id'] == run_id, 'model_checkpoint_path'] = results['model_checkpoint_path']
    df_res.to_csv(os.path.join(args.results_dir, 'results.csv'), index=False)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='gender_dataset')
    parser.add_argument('--results_dir', type=str, default='results/journal_extension/coco_bbox')
    parser.add_argument('--methods', nargs='+', default=['caipi', 'rrr', 'hybrid'])
    parser.add_argument('--feedback_types', nargs='+', default=['segmentation', 'bbox'])
    parser.add_argument('--sampling_strategies', nargs='+', default=['high_confidence', 'uncertainty'])
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--rrr_explainers', nargs='+', default=['gradcam'],
                        help='Explainers to use for RRR (gradcam, bla)')
    parser.add_argument('--hybrid_explainers', nargs='+', default=['bla'],
                        help='Explainers to use for Hybrid (gradcam, bla)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None, help='Override default epochs')
    parser.add_argument('--dry_run', action='store_true', help='Set up configs but do not train')
    args = parser.parse_args()
    
    print(f"Starting experiments with args: {args}")
    os.makedirs(args.results_dir, exist_ok=True)
    
    device = get_device() if args.device == 'cuda' else torch.device('cpu')
    set_random_seeds(args.seed)
    
    print(f"Device: {get_device()}", flush=True)
    print("Loading datasets...", flush=True)
    if os.path.exists(os.path.join(args.data_dir, 'splits', 'train.csv')):
        train_df = pd.read_csv(os.path.join(args.data_dir, 'splits', 'train.csv'))
        val_df = pd.read_csv(os.path.join(args.data_dir, 'splits', 'val.csv'))
        test_df = pd.read_csv(os.path.join(args.data_dir, 'splits', 'test.csv'))
        for df in [train_df, val_df, test_df]:
            df['full_path'] = df['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))
    else:
        # Check if dataset_split folder exists for pre-split
        if os.path.exists(os.path.join(args.data_dir, 'dataset_split')):
            train_df, val_df, test_df, label_encoder = prepare_data_splits_from_dataset_folder(args.data_dir)
        else:
            female_path = os.path.join(args.data_dir, 'resized_female_images')
            male_path = os.path.join(args.data_dir, 'resized_male_images')
            train_df, val_df, test_df, label_encoder = prepare_data_splits(female_path, male_path)
            
            # Add full_path for prepare_data_splits fallback
            for df in [train_df, val_df, test_df]:
                df['full_path'] = df.apply(lambda row: os.path.join(female_path if 'female' in str(row['label']).lower() else male_path, row['image']), axis=1)
            
    data_splits = (train_df, val_df, test_df)
    
    # Build experiment configurations
    experiments = []
    
    # 1. Baseline
    if 'baseline' in args.methods or 'all' in args.methods:
        experiments.append({
            'method': 'baseline',
            'feedback_type': 'none',
            'sampling_strategy': 'none',
            'k': 0,
            'explainer': 'gradcam',
            'seed': args.seed
        })
        
    # 2. XIL Methods
    for method in ['caipi', 'rrr', 'hybrid']:
        if method in args.methods or 'all' in args.methods:
            explainers = (args.rrr_explainers    if method == 'rrr'
                          else args.hybrid_explainers if method == 'hybrid'
                          else ['gradcam'])
            for feedback in args.feedback_types:
                for strategy in args.sampling_strategies:
                    for explainer in explainers:
                        if method == 'rrr':
                            experiments.append({
                                'method': method,
                                'feedback_type': feedback,
                                'sampling_strategy': strategy,
                                'k': 0,
                                'explainer': explainer,
                                'seed': args.seed
                            })
                        else:
                            for k in args.k_values:
                                experiments.append({
                                    'method': method,
                                    'feedback_type': feedback,
                                    'sampling_strategy': strategy,
                                    'k': k,
                                    'explainer': explainer,
                                    'seed': args.seed
                                })
                            
    print(f"Total experiments configured: {len(experiments)}", flush=True)
    for i, e in enumerate(experiments):
        print(f"  [{i+1}] {e['method']}_{e['feedback_type']}_k{e['k']}_{e['sampling_strategy']}", flush=True)
    
    if args.dry_run:
        print("Dry run complete. Exiting.")
        return
        
    for exp in experiments:
        try:
            run_experiment(exp, data_splits, device, args)
        except Exception as e:
            print(f"Error running experiment {exp}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
