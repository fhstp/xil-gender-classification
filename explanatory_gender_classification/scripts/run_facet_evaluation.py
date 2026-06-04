"""
Phase 5: FACET Generalization Evaluation

Loads the best trained checkpoint for each (method, feedback_type) pair from the
COCO bbox experiments and evaluates it on the FACET test set.

Computed metrics:
  - accuracy, balanced_accuracy, macro_f1, per-class error rates
  - error_rate_gap (male vs female)
  - FFP, BFP, BSR (saliency-based bias metrics using bbox foreground masks)
  - per_sample_predictions.csv with FACET attribute columns for intersectional analysis
"""

import os
import ast
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.architectures import create_model
from models.rrr_model import RRRGenderClassifier
from explainability.gradcam import GradCAM
from explainability.bla import BLAWrapper, create_bla_model
from evaluation.classification_metrics import compute_classification_metrics
from utils.helpers import set_random_seeds, get_device
from utils.settings import Config


# ---------------------------------------------------------------------------
# FACET dataset
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _parse_bbox(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return None
    return None


def _bbox_to_foreground_mask(bbox, target_hw=(224, 224), orig_wh=None):
    """Return a (1, H, W) float tensor, 1 inside bbox, 0 outside."""
    H, W = target_hw
    mask = torch.zeros(1, H, W)
    if bbox is None:
        return mask
    x, y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
    if orig_wh is not None:
        ow, oh = orig_wh
        x  = x  / ow * W
        y  = y  / oh * H
        bw = bw / ow * W
        bh = bh / oh * H
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + bw))
    y2 = min(H, int(y + bh))
    mask[0, y1:y2, x1:x2] = 1.0
    return mask


class FACETEvalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform or _DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['image_name']))
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))

        orig_wh = image.size  # (W, H)
        image_tensor = self.transform(image)

        bbox = _parse_bbox(row.get('bbox'))
        foreground_mask = _bbox_to_foreground_mask(bbox, target_hw=(224, 224), orig_wh=orig_wh)

        label = int(row['encoded_label'])
        return image_tensor, label, foreground_mask, row.to_dict()


def _facet_collate(batch):
    images     = torch.stack([b[0] for b in batch])
    labels     = torch.tensor([b[1] for b in batch], dtype=torch.long)
    masks      = torch.stack([b[2] for b in batch])
    metadata   = [b[3] for b in batch]
    return images, labels, masks, metadata


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, method, device, explainer='gradcam'):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Detect the architecture from the checkpoint's own keys so we never depend
    # on a possibly-wrong explainer flag. BLA models (create_bla_model) save keys
    # under 'feature_extractor.*'; the GradCAM GenderClassifier saves 'backbone._conv*'.
    keys = list(state_dict.keys())
    is_bla = (explainer == 'bla'
              or any(k.startswith('feature_extractor') for k in keys)
              or any('bla' in k.lower() for k in keys))

    if is_bla:
        model = create_bla_model('efficientnet_b0', num_classes=2, pretrained=False)
    elif method == 'rrr':
        model = RRRGenderClassifier(pretrained=False)
    else:
        model = create_model(pretrained=False)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_explainer(model, method, device, explainer='gradcam'):
    if explainer == 'bla':
        return BLAWrapper(model, device=device)
    base = model.backbone if hasattr(model, 'backbone') else model
    target_layer = None
    if hasattr(base, '_conv_head'):
        target_layer = base._conv_head
    elif hasattr(base, 'features'):
        target_layer = base.features[-1]
    elif hasattr(base, 'layer4'):
        target_layer = base.layer4
    return GradCAM(model, target_layer=target_layer)


# ---------------------------------------------------------------------------
# Model selection from COCO results
# ---------------------------------------------------------------------------

def select_models(results_csv, checkpoints_dir):
    """
    Return a list of dicts describing which checkpoint to evaluate on FACET.
    Selects: 1 baseline + best (method, feedback_type) pair for each of
    caipi/rrr/hybrid × segmentation/bbox = up to 7 models total.
    """
    if not os.path.exists(results_csv):
        print(f"[WARN] {results_csv} not found — cannot auto-select models.")
        return []

    df = pd.read_csv(results_csv)
    selected = []

    # Baseline
    bl = df[df['method'] == 'baseline']
    if not bl.empty:
        row = bl.sort_values('accuracy', ascending=False).iloc[0]
        ckpt = os.path.join(checkpoints_dir, f"{row['run_id']}.pth")
        if os.path.exists(ckpt):
            selected.append({'label': 'baseline', 'method': 'baseline',
                             'feedback_type': 'none', 'checkpoint': ckpt,
                             'coco_run_id': row['run_id'],
                             'coco_accuracy': row['accuracy']})

    for method in ['caipi', 'rrr', 'hybrid']:
        for fb in ['segmentation', 'bbox']:
            sub = df[(df['method'] == method) & (df['feedback_type'] == fb)]
            if sub.empty:
                print(f"[WARN] No results for {method}/{fb} — skipping.")
                continue
            row = sub.sort_values('accuracy', ascending=False).iloc[0]
            ckpt = os.path.join(checkpoints_dir, f"{row['run_id']}.pth")
            if not os.path.exists(ckpt):
                print(f"[WARN] Checkpoint missing: {ckpt}")
                continue
            selected.append({
                'label': f"{method}_{fb}",
                'method': method,
                'feedback_type': fb,
                'checkpoint': ckpt,
                'coco_run_id': row['run_id'],
                'coco_accuracy': row['accuracy'],
            })

    return selected


def select_all_models(results_csv, checkpoints_dir):
    """Return one entry per row in results.csv (all runs, not just best per group)."""
    if not os.path.exists(results_csv):
        print(f"[WARN] {results_csv} not found.")
        return []

    df = pd.read_csv(results_csv)
    selected = []
    for _, row in df.iterrows():
        ckpt = os.path.join(checkpoints_dir, f"{row['run_id']}.pth")
        if not os.path.exists(ckpt):
            print(f"[WARN] Checkpoint missing, skipping: {row['run_id']}")
            continue
        selected.append({
            'label':         row['run_id'],
            'method':        row['method'],
            'feedback_type': row['feedback_type'],
            'explainer':     row.get('explainer', 'gradcam'),
            'checkpoint':    ckpt,
            'coco_run_id':   row['run_id'],
            'coco_accuracy': row['accuracy'],
        })
    return selected


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_info, data_loader, device):
    method   = model_info['method']
    ckpt     = model_info['checkpoint']
    # Derive the explainer from the checkpoint filename (run_id) so BLA models are
    # always loaded/explained with BLA, regardless of the model_info flag.
    base = os.path.basename(ckpt)
    explainer_name = 'bla' if '_bla' in base else model_info.get('explainer', 'gradcam')

    print(f"  Loading model from {os.path.basename(ckpt)} ...")
    model    = load_model(ckpt, method, device, explainer=explainer_name)
    explainer = get_explainer(model, method, device, explainer=explainer_name)

    all_preds, all_labels, all_probs = [], [], []
    all_ffp, all_bfp, all_bsr = [], [], []
    all_metadata = []

    n_total   = len(data_loader.dataset)
    n_done    = 0
    print(f"  Running inference + saliency on {n_total} images ...", flush=True)

    for images, labels, fg_masks, metadata in data_loader:
        images   = images.to(device)
        labels   = labels.to(device)
        fg_masks = fg_masks.to(device)   # (B, 1, H, W)

        with torch.no_grad():
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        probs = (torch.exp(outputs)
                 if outputs.max().item() <= 0.0
                 else torch.softmax(outputs, dim=1))
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

        # Per-image saliency metrics
        for i in range(len(images)):
            n_done += 1
            if n_done % 200 == 0 or n_done == n_total:
                print(f"    saliency {n_done}/{n_total}", flush=True)
            img_i  = images[i:i+1]
            lbl_i  = labels[i]
            fgm_i  = fg_masks[i, 0].cpu()   # (H, W) foreground mask

            try:
                if hasattr(explainer, 'generate_cam'):
                    sal = explainer.generate_cam(img_i, lbl_i.item())
                else:
                    sal = explainer.generate_explanation(img_i)[0]

                sal = sal.detach().cpu()
                if sal.shape != fgm_i.shape:
                    sal = F.interpolate(
                        sal.unsqueeze(0).unsqueeze(0),
                        size=fgm_i.shape,
                        mode='bilinear', align_corners=False
                    ).squeeze()

                fg  = (fgm_i > 0.5)
                bg  = ~fg
                thr = torch.quantile(sal.flatten(), 0.25)
                ffp = ((sal > thr) & fg).float().sum() / (fg.float().sum() + 1e-8)
                bfp = ((sal > thr) & bg).float().sum() / (bg.float().sum() + 1e-8)
                bsr = (sal * bg.float()).sum() / (sal.sum() + 1e-8)
            except Exception as e:
                ffp = bfp = bsr = torch.tensor(0.0)

            all_ffp.append(ffp.item() if isinstance(ffp, torch.Tensor) else ffp)
            all_bfp.append(bfp.item() if isinstance(bfp, torch.Tensor) else bfp)
            all_bsr.append(bsr.item() if isinstance(bsr, torch.Tensor) else bsr)

        all_metadata.extend(metadata)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    clf = compute_classification_metrics(all_labels, all_preds)
    cm  = clf.pop('confusion_matrix')

    f_tot = cm[0][0] + cm[0][1]
    m_tot = cm[1][0] + cm[1][1]
    f_err = cm[0][1] / f_tot if f_tot > 0 else 0.0
    m_err = cm[1][0] / m_tot if m_tot > 0 else 0.0

    metrics = {
        'label':         model_info['label'],
        'method':        method,
        'feedback_type': model_info['feedback_type'],
        'coco_run_id':   model_info.get('coco_run_id', ''),
        'coco_accuracy': model_info.get('coco_accuracy', float('nan')),
        'dataset':       'facet',
        **clf,
        'cm_tn': cm[0][0], 'cm_fp': cm[0][1],
        'cm_fn': cm[1][0], 'cm_tp': cm[1][1],
        'female_error_rate': f_err,
        'male_error_rate':   m_err,
        'error_rate_gap':    abs(f_err - m_err),
        'FFP': float(np.mean(all_ffp)),
        'BFP': float(np.mean(all_bfp)),
        'BSR': float(np.mean(all_bsr)),
    }

    # Per-sample predictions
    per_sample = pd.DataFrame(all_metadata)
    per_sample['model_label']   = model_info['label']
    per_sample['label_gt']      = all_labels
    per_sample['prediction']    = all_preds
    per_sample['prob_male']     = all_probs
    per_sample['correct']       = (all_labels == all_preds).astype(int)

    return metrics, per_sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate best COCO models on FACET")
    parser.add_argument('--facet_dir',   type=str,
                        default=r'c:\Users\Queby\Research\xil-gender-classification\facet_dataset\facet_processed',
                        help='Directory containing facet_processed (with splits/ and images/)')
    parser.add_argument('--coco_results_csv', type=str,
                        default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\coco_bbox\results.csv')
    parser.add_argument('--checkpoints_dir', type=str,
                        default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\coco_bbox\checkpoints')
    parser.add_argument('--results_dir', type=str,
                        default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\facet_eval')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which FACET split to evaluate on')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific model labels to evaluate (default: auto-select from results.csv)')
    parser.add_argument('--all_models', action='store_true',
                        help='Evaluate every run in results.csv individually (not just best per group)')
    parser.add_argument('--curated_dir', type=str,
                        default=r'c:\Users\Queby\Research\xil-gender-classification\facet_dataset\labeled_images',
                        help='If set, restrict evaluation to images present in <curated_dir>/female/ '
                             'and <curated_dir>/male/ (curated balanced subset).')
    args = parser.parse_args()

    set_random_seeds(args.seed)
    device = get_device() if args.device == 'cuda' else torch.device('cpu')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load FACET split
    split_csv = os.path.join(args.facet_dir, 'splits', f'{args.split}.csv')
    img_dir   = os.path.join(args.facet_dir, 'images')
    if not os.path.exists(split_csv):
        print(f"[ERROR] Split CSV not found: {split_csv}")
        return

    df = pd.read_csv(split_csv)
    print(f"Loaded FACET {args.split} set: {len(df)} samples")

    # Filter to curated balanced subset if requested
    if args.curated_dir and os.path.isdir(args.curated_dir):
        female_names = set(os.listdir(os.path.join(args.curated_dir, 'female')))
        male_names   = set(os.listdir(os.path.join(args.curated_dir, 'male')))
        curated_names = female_names | male_names
        before = len(df)
        df = df[df['image_name'].isin(curated_names)].reset_index(drop=True)
        print(f"Curated filter applied: {before} -> {len(df)} samples "
              f"({len(female_names)} female, {len(male_names)} male)")

    print(df['label'].value_counts().to_string())

    dataset    = FACETEvalDataset(df, img_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0,
                            collate_fn=_facet_collate)

    # Select models
    if args.all_models:
        model_list = select_all_models(args.coco_results_csv, args.checkpoints_dir)
    else:
        model_list = select_models(args.coco_results_csv, args.checkpoints_dir)
        if args.models:
            model_list = [m for m in model_list if m['label'] in args.models]

    if not model_list:
        print("[ERROR] No models to evaluate. Check --coco_results_csv and --checkpoints_dir.")
        return

    # Load already-evaluated labels to enable incremental runs
    results_path = os.path.join(args.results_dir, f'facet_{args.split}_results.csv')
    preds_path   = os.path.join(args.results_dir, f'facet_{args.split}_per_sample.csv')
    done_labels  = set()
    if os.path.exists(results_path):
        done_labels = set(pd.read_csv(results_path)['label'].tolist())
        print(f"Resuming — {len(done_labels)} model(s) already evaluated, skipping them.")

    pending = [m for m in model_list if m['label'] not in done_labels]
    skipped = len(model_list) - len(pending)

    print(f"\nEvaluating {len(pending)} model(s) on FACET {args.split} set "
          f"({skipped} already done, skipped):")
    for m in pending:
        print(f"  {m['label']:50s}  coco_acc={m.get('coco_accuracy', float('nan')):.4f}")

    all_results    = []
    all_per_sample = []

    for model_info in pending:
        print(f"\n{'='*55}\nEvaluating: {model_info['label']}\n{'='*55}")
        try:
            metrics, per_sample = evaluate_model(model_info, dataloader, device)
            all_results.append(metrics)
            all_per_sample.append(per_sample)
            print(f"  Accuracy={metrics['accuracy']:.4f}  "
                  f"ERR_gap={metrics['error_rate_gap']:.4f}  "
                  f"BSR={metrics['BSR']:.4f}")

            # Append after each model so a crash doesn't lose progress
            mode   = 'a' if os.path.exists(results_path) else 'w'
            header = not os.path.exists(results_path)
            pd.DataFrame([metrics]).to_csv(results_path, mode=mode, header=header, index=False)

            p_mode   = 'a' if os.path.exists(preds_path) else 'w'
            p_header = not os.path.exists(preds_path)
            per_sample.to_csv(preds_path, mode=p_mode, header=p_header, index=False)

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    print(f"\nResults saved to {results_path}")
    print(f"Per-sample predictions saved to {preds_path}")
    print("\nFACET evaluation complete.")


if __name__ == '__main__':
    main()
