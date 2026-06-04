"""
Qualitative visualization for FACET evaluation results.

Generates:
  1. Grid figures: same images × multiple models, with GradCAM/BLA explanation
     overlays and bounding boxes.
  2. A per-image saliency CSV (FFP, BFP, BSR) for the selected images/models.

Usage:
  python scripts/visualize_facet_explanations.py

Optional flags:
  --n_images        Number of images per gender per selection criterion (default 3)
  --models          Space-separated list of run_ids to compare (default: 4 representative ones)
  --output_dir      Where to save figures (default: results/journal_extension/facet_eval/figures)
"""

import os
import ast
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.architectures import create_model
from models.rrr_model import RRRGenderClassifier
from explainability.gradcam import GradCAM
from explainability.bla import BLAWrapper, create_bla_model
from utils.helpers import set_random_seeds, get_device

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Human-readable labels for the figure
MODEL_LABELS = {
    'baseline_none_k0_none_gradcam_42':                  'Baseline\n(GradCAM)',
    'caipi_segmentation_k1_high_confidence_gradcam_42':  'CAIPI\nseg-k1 (GradCAM)',
    'caipi_bbox_k1_high_confidence_gradcam_42':          'CAIPI\nbbox-k1 (GradCAM)',
    'rrr_segmentation_k0_high_confidence_gradcam_42':    'RRR\nseg (GradCAM)',
    'rrr_bbox_k0_high_confidence_gradcam_42':            'RRR\nbbox (GradCAM)',
    'rrr_segmentation_k0_high_confidence_bla_42':        'RRR\nseg (BLA)',
    'rrr_bbox_k0_high_confidence_bla_42':                'RRR\nbbox (BLA)',
    'hybrid_segmentation_k1_high_confidence_bla_42':     'Hybrid\nseg-k1 (BLA)',
    'hybrid_bbox_k1_high_confidence_bla_42':             'Hybrid\nbbox-k1 (BLA)',
}

# Mirrors the COCO grid: baseline + CAIPI seg/bbox, RRR seg/bbox (GradCAM + BLA), Hybrid seg/bbox
DEFAULT_MODELS = [
    'baseline_none_k0_none_gradcam_42',
    'caipi_segmentation_k1_high_confidence_gradcam_42',
    'caipi_bbox_k1_high_confidence_gradcam_42',
    'rrr_segmentation_k0_high_confidence_gradcam_42',
    'rrr_bbox_k0_high_confidence_gradcam_42',
    'rrr_segmentation_k0_high_confidence_bla_42',
    'rrr_bbox_k0_high_confidence_bla_42',
    'hybrid_segmentation_k1_high_confidence_bla_42',
    'hybrid_bbox_k1_high_confidence_bla_42',
]

# ---------------------------------------------------------------------------
# Model loading (mirrors run_facet_evaluation.py)
# ---------------------------------------------------------------------------

def _explainer_from_run_id(run_id):
    """Extract explainer name from the run_id string."""
    if run_id.endswith('_bla_42') or '_bla_' in run_id:
        return 'bla'
    return 'gradcam'


def _method_from_run_id(run_id):
    for m in ('baseline', 'caipi', 'rrr', 'hybrid'):
        if run_id.startswith(m):
            return m
    return 'baseline'


def load_model(checkpoint_path, method, device, explainer='gradcam'):
    if explainer == 'bla':
        model = create_bla_model('efficientnet_b0', num_classes=2, pretrained=False)
    elif method == 'rrr':
        model = RRRGenderClassifier(pretrained=False)
    else:
        model = create_model(pretrained=False)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
# Image helpers
# ---------------------------------------------------------------------------

def _parse_bbox(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def load_image_raw(img_path):
    """Return PIL image or a black placeholder."""
    try:
        return Image.open(img_path).convert('RGB')
    except Exception:
        return Image.new('RGB', (224, 224))


def compute_saliency(explainer, img_tensor, label):
    """Return a (H, W) numpy saliency map in [0, 1]."""
    try:
        if hasattr(explainer, 'generate_cam'):
            sal = explainer.generate_cam(img_tensor, label)
        else:
            sal = explainer.generate_explanation(img_tensor)[0]
        sal = sal.detach().cpu()
        if sal.shape != torch.Size([224, 224]):
            sal = F.interpolate(
                sal.unsqueeze(0).unsqueeze(0).float(),
                size=(224, 224),
                mode='bilinear', align_corners=False
            ).squeeze()
        sal = sal.float()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        return sal.numpy()
    except Exception as e:
        print(f"    [WARN] saliency failed: {e}")
        return np.zeros((224, 224))


def saliency_metrics(sal_np, fg_mask_np):
    """Compute FFP, BFP, BSR given saliency and binary foreground mask."""
    sal   = torch.tensor(sal_np)
    fg    = torch.tensor(fg_mask_np, dtype=torch.bool)
    bg    = ~fg
    thr   = torch.quantile(sal.flatten(), 0.25)
    ffp   = ((sal > thr) & fg).float().sum() / (fg.float().sum() + 1e-8)
    bfp   = ((sal > thr) & bg).float().sum() / (bg.float().sum() + 1e-8)
    bsr   = (sal * bg.float()).sum() / (sal.sum() + 1e-8)
    return ffp.item(), bfp.item(), bsr.item()


def bbox_to_mask(bbox, img_wh, target_hw=(224, 224)):
    """Convert [x, y, w, h] bbox to a binary (H, W) foreground mask."""
    mask = np.zeros(target_hw, dtype=np.float32)
    if bbox is None:
        return mask
    orig_w, orig_h = img_wh
    x, y, bw, bh = bbox[:4]
    sx = target_hw[1] / orig_w
    sy = target_hw[0] / orig_h
    x1 = int(np.clip(x * sx, 0, target_hw[1] - 1))
    y1 = int(np.clip(y * sy, 0, target_hw[0] - 1))
    x2 = int(np.clip((x + bw) * sx, 0, target_hw[1]))
    y2 = int(np.clip((y + bh) * sy, 0, target_hw[0]))
    mask[y1:y2, x1:x2] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def select_images(per_sample_df, baseline_id, facet_split_df,
                  n_per_group=3, seed=42, min_bbox_area=150000):
    """
    Select interesting images for the qualitative figure.

    Filters to 'prominent person' images where the annotated person's bounding
    box covers at least min_bbox_area pixels in the original image coordinates
    (default 150k px ≈ ~400×375 — a person filling roughly 1/4 of a 1500×2000
    SA-1B image). This reduces cluttered multi-person backgrounds.

    Criteria returned:
      - 'baseline_wrong_female' / 'baseline_wrong_male'
      - 'baseline_correct_female' / 'baseline_correct_male'
    """
    rng = np.random.default_rng(seed)
    bl = per_sample_df[per_sample_df['model_label'] == baseline_id].copy()

    # Build a bbox-area lookup from the split CSV
    def _bbox_area(bbox_val):
        b = _parse_bbox(bbox_val)
        if b is None or len(b) < 4:
            return 0
        return b[2] * b[3]  # w * h

    area_map = {
        row['image_name']: _bbox_area(row.get('bbox'))
        for _, row in facet_split_df.iterrows()
    }
    prominent = {img for img, area in area_map.items() if area >= min_bbox_area}
    bl = bl[bl['image_name'].isin(prominent)]

    groups = {}
    for gender_val, gender_name in [(0, 'female'), (1, 'male')]:
        gender_rows = bl[bl['encoded_label'] == gender_val]
        wrong   = gender_rows[gender_rows['correct'] == 0]['image_name'].unique()
        correct = gender_rows[gender_rows['correct'] == 1]['image_name'].unique()
        rng.shuffle(wrong)
        rng.shuffle(correct)
        groups[f'baseline_wrong_{gender_name}']   = list(wrong[:n_per_group])
        groups[f'baseline_correct_{gender_name}'] = list(correct[:n_per_group])

    return groups


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def overlay_heatmap(raw_img_224, sal_np, alpha=0.45):
    """Blend a jet heatmap over the image. Returns H×W×3 uint8 array."""
    heat = cm.jet(sal_np)[:, :, :3]           # H×W×3, float [0,1]
    img  = np.array(raw_img_224) / 255.0       # H×W×3, float [0,1]
    blended = (1 - alpha) * img + alpha * heat
    return (blended * 255).astype(np.uint8)


def draw_bbox_on_img(raw_img_224_arr, bbox, orig_wh, color='lime', lw=2):
    """Return a copy with a rectangle drawn for the bbox."""
    import copy
    fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=112)
    ax.imshow(raw_img_224_arr)
    ax.axis('off')
    if bbox is not None:
        orig_w, orig_h = orig_wh
        x, y, bw, bh = bbox[:4]
        sx = 224 / orig_w
        sy = 224 / orig_h
        rect = mpatches.Rectangle(
            (x * sx, y * sy), bw * sx, bh * sy,
            linewidth=lw, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return arr


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_explanation_grid(
    image_names, model_configs, img_dir, per_sample_df,
    facet_split_df, device, output_path, title=None
):
    """
    Render a grid:  rows = images,  cols = [original+bbox] + [model explanations]

    model_configs: list of dicts with keys: run_id, checkpoint, label, method, explainer
    """
    n_images = len(image_names)
    n_cols   = 1 + len(model_configs)
    fig_w    = 2.5 * n_cols
    fig_h    = 2.8 * n_images

    fig, axes = plt.subplots(n_images, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    # Column headers
    axes[0, 0].set_title('Original\n+ BBox', fontsize=8, fontweight='bold')
    for j, mc in enumerate(model_configs):
        axes[0, j + 1].set_title(mc['label'], fontsize=7, fontweight='bold')

    saliency_records = []

    # Load all models first
    print(f"  Loading {len(model_configs)} model(s)...")
    loaded = []
    for mc in model_configs:
        print(f"    {mc['run_id']} ...", flush=True)
        m   = load_model(mc['checkpoint'], mc['method'], device, mc['explainer'])
        exp = get_explainer(m, mc['method'], device, mc['explainer'])
        loaded.append((m, exp))

    for i, img_name in enumerate(image_names):
        # Find the row in the split CSV for this image
        split_row = facet_split_df[facet_split_df['image_name'] == img_name]
        if split_row.empty:
            print(f"  [WARN] {img_name} not found in split CSV, skipping.")
            for ax in axes[i]:
                ax.axis('off')
            continue
        split_row = split_row.iloc[0]

        gender_label = split_row.get('label', '?')
        bbox         = _parse_bbox(split_row.get('bbox'))
        img_path     = os.path.join(img_dir, img_name)
        raw_pil      = load_image_raw(img_path)
        orig_wh      = raw_pil.size
        raw_224      = raw_pil.resize((224, 224))
        raw_224_np   = np.array(raw_224)
        fg_mask      = bbox_to_mask(bbox, orig_wh)

        img_tensor = TRANSFORM(raw_pil).unsqueeze(0).to(device)

        # Column 0: original image + bbox
        ax0 = axes[i, 0]
        ax0.imshow(raw_224_np)
        if bbox is not None:
            x, y, bw, bh = bbox[:4]
            sx = 224 / orig_wh[0]
            sy = 224 / orig_wh[1]
            rect = mpatches.Rectangle(
                (x * sx, y * sy), bw * sx, bh * sy,
                linewidth=1.5, edgecolor='lime', facecolor='none'
            )
            ax0.add_patch(rect)
        ax0.set_ylabel(f"{img_name[:16]}\n({gender_label})",
                       fontsize=6, rotation=0, labelpad=50, va='center')
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Explanation columns
        for j, (mc, (model, explainer)) in enumerate(zip(model_configs, loaded)):
            with torch.no_grad():
                out = model(img_tensor)
                if isinstance(out, tuple):
                    out = out[0]
            probs   = torch.softmax(out, dim=1) if out.max().item() > 0 else torch.exp(out)
            pred    = int(torch.argmax(out, dim=1).item())
            conf    = probs[0, pred].item()

            sal_np = compute_saliency(explainer, img_tensor, pred)
            ffp, bfp, bsr = saliency_metrics(sal_np, fg_mask)

            blended = overlay_heatmap(raw_224_np, sal_np)
            ax = axes[i, j + 1]
            ax.imshow(blended)
            if bbox is not None:
                rect = mpatches.Rectangle(
                    (x * sx, y * sy), bw * sx, bh * sy,
                    linewidth=1.5, edgecolor='lime', facecolor='none'
                )
                ax.add_patch(rect)

            gt_label = int(split_row.get('encoded_label', -1))
            correct  = (pred == gt_label)
            tick_sym = '✓' if correct else '✗'
            pred_str = 'M' if pred == 1 else 'F'
            ax.set_xlabel(
                f"{tick_sym}{pred_str}  BSR={bsr:.2f}",
                fontsize=6,
                color='green' if correct else 'red'
            )
            ax.set_xticks([])
            ax.set_yticks([])

            saliency_records.append({
                'image_name':  img_name,
                'gender_label': gender_label,
                'run_id':      mc['run_id'],
                'method':      mc['method'],
                'explainer':   mc['explainer'],
                'prediction':  pred_str,
                'gt':          'M' if gt_label == 1 else 'F',
                'correct':     int(correct),
                'confidence':  conf,
                'FFP':         ffp,
                'BFP':         bfp,
                'BSR':         bsr,
            })

    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold', y=1.01)

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return pd.DataFrame(saliency_records)


# ---------------------------------------------------------------------------
# BSR bar chart
# ---------------------------------------------------------------------------

def plot_bsr_comparison(facet_results_df, output_path):
    """Bar chart of BSR per model, colored by explainer type."""
    df = facet_results_df.sort_values('BSR')
    labels = [
        MODEL_LABELS.get(r, r.replace('_42', '').replace('_', '\n'))
        for r in df['label']
    ]
    colors = ['#2196F3' if 'bla' in r else '#FF9800' for r in df['label']]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(df)), df['BSR'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('BSR (Background Saliency Ratio)')
    ax.set_title('Background Saliency Ratio across all models\n'
                 '(lower = model attends more to person foreground)')
    ax.set_ylim(0, 1)
    ax.axhline(df[df['label'].str.contains('baseline')]['BSR'].values[0],
               color='gray', linestyle='--', linewidth=1, label='Baseline BSR')

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor='#2196F3', label='BLA explainer'),
                  Patch(facecolor='#FF9800', label='GradCAM explainer')]
    ax.legend(handles=legend_els, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_err_gap_vs_bsr(facet_results_df, output_path):
    """
    Scatter plot: X = BSR (how much the model attends to background, lower is better),
    Y = Error Rate Gap (|female_err − male_err|, lower is fairer).
    Bottom-left corner = ideal: model attends to the person AND is gender-fair.
    Each dot is one trained model run on the FACET test set.
    """
    method_colors = {
        'baseline': '#555555',
        'caipi':    '#4CAF50',
        'rrr':      '#2196F3',
        'hybrid':   '#E91E63',
    }
    marker_map = {'gradcam': 'o', 'bla': 's'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Shade the ideal region (bottom-left quadrant relative to baseline)
    bl_row = facet_results_df[facet_results_df['method'] == 'baseline']
    if not bl_row.empty:
        bl_bsr = bl_row['BSR'].values[0]
        bl_gap = bl_row['error_rate_gap'].values[0]
        ax.axhline(bl_gap, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(bl_bsr, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.fill_between([0, bl_bsr], [0, 0], [bl_gap, bl_gap],
                        color='#c8e6c9', alpha=0.35, label='Better than baseline (ideal region)')

    for _, row in facet_results_df.iterrows():
        explainer = 'bla' if 'bla' in str(row['label']) else 'gradcam'
        color  = method_colors.get(row['method'], '#999999')
        marker = marker_map.get(explainer, 'o')
        ax.scatter(row['BSR'], row['error_rate_gap'],
                   color=color, marker=marker, s=80, zorder=4,
                   edgecolors='white', linewidths=0.8)

    # Annotate key models
    def _annotate(row, text, offset=(0.01, 0.01)):
        ax.annotate(text, (row['BSR'], row['error_rate_gap']),
                    xytext=(row['BSR'] + offset[0], row['error_rate_gap'] + offset[1]),
                    fontsize=6.5, color='#333333',
                    arrowprops=dict(arrowstyle='->', color='#777777', lw=0.7))

    if not bl_row.empty:
        _annotate(bl_row.iloc[0], 'Baseline', offset=(0.01, 0.015))

    best_acc_row = facet_results_df.loc[facet_results_df['accuracy'].idxmax()]
    _annotate(best_acc_row, 'Best Acc.', offset=(0.01, -0.03))

    best_fair_row = facet_results_df.loc[facet_results_df['error_rate_gap'].idxmin()]
    _annotate(best_fair_row, 'Fairest', offset=(-0.08, 0.02))

    ax.set_xlabel('BSR — Background Saliency Ratio\n(fraction of saliency mass on background; '
                  'lower = model attends to the person)', fontsize=9)
    ax.set_ylabel('Error Rate Gap  |female_err − male_err|\n(lower = fairer predictions)', fontsize=9)
    ax.set_title('BSR vs. Gender Fairness on FACET\n'
                 'Each point is one trained model. '
                 'Best models are in the lower-left corner.', fontsize=10)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=9, label=m.capitalize())
        for m, c in method_colors.items()
    ] + [
        Line2D([0], [0], marker='o', color='gray', markersize=8, label='GradCAM'),
        Line2D([0], [0], marker='s', color='gray', markersize=8, label='BLA'),
        Patch(facecolor='#c8e6c9', alpha=0.5, label='Ideal region'),
    ]
    ax.legend(handles=legend_els, fontsize=8, ncol=2, loc='upper right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--facet_dir',    default=r'c:\Users\Queby\Research\xil-gender-classification\facet_dataset\facet_processed')
    parser.add_argument('--checkpoints_dir', default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\coco_bbox\checkpoints')
    parser.add_argument('--results_dir',  default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\facet_eval')
    parser.add_argument('--output_dir',   default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\facet_eval\figures')
    parser.add_argument('--split',        default='test')
    parser.add_argument('--n_images',      type=int, default=3)
    parser.add_argument('--min_bbox_area', type=int, default=150000,
                        help='Min bbox w*h in original px to filter for prominent-person images')
    parser.add_argument('--models',        nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--device',       default='cuda')
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--curated_dir',
                        default=r'c:\Users\Queby\Research\xil-gender-classification\facet_dataset\labeled_images',
                        help='If set, restrict to images present in <curated_dir>/female/ and /male/.')
    args = parser.parse_args()

    set_random_seeds(args.seed)
    device = get_device() if args.device == 'cuda' else torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    split_csv      = os.path.join(args.facet_dir, 'splits', f'{args.split}.csv')
    per_sample_csv = os.path.join(args.results_dir, f'facet_{args.split}_per_sample.csv')
    results_csv    = os.path.join(args.results_dir, f'facet_{args.split}_results.csv')
    img_dir        = os.path.join(args.facet_dir, 'images')

    facet_split_df  = pd.read_csv(split_csv)
    per_sample_df   = pd.read_csv(per_sample_csv)
    facet_results   = pd.read_csv(results_csv)

    # Filter to curated balanced subset if curated_dir is provided
    if args.curated_dir and os.path.isdir(args.curated_dir):
        female_names  = set(os.listdir(os.path.join(args.curated_dir, 'female')))
        male_names    = set(os.listdir(os.path.join(args.curated_dir, 'male')))
        curated_names = female_names | male_names
        facet_split_df = facet_split_df[facet_split_df['image_name'].isin(curated_names)].reset_index(drop=True)
        per_sample_df  = per_sample_df[per_sample_df['image_name'].isin(curated_names)].reset_index(drop=True)
        print(f"Curated filter: {len(facet_split_df)} images "
              f"({len(female_names)} female, {len(male_names)} male)")

    print(f"Loaded {len(facet_split_df)} FACET images, "
          f"{len(per_sample_df)} per-sample rows, "
          f"{len(facet_results)} model results.")

    # --- Metric comparison figures ---
    print("\nGenerating metric plots...")
    plot_bsr_comparison(facet_results,
                        os.path.join(args.output_dir, 'bsr_comparison.png'))
    plot_err_gap_vs_bsr(facet_results,
                        os.path.join(args.output_dir, 'err_gap_vs_bsr_scatter.png'))

    # --- Build model config list ---
    coco_results_csv = os.path.join(
        os.path.dirname(args.checkpoints_dir), 'results.csv')
    coco_df = pd.read_csv(coco_results_csv) if os.path.exists(coco_results_csv) else pd.DataFrame()

    model_configs = []
    for run_id in args.models:
        ckpt = os.path.join(args.checkpoints_dir, f'{run_id}.pth')
        if not os.path.exists(ckpt):
            print(f"  [WARN] Checkpoint not found, skipping: {run_id}")
            continue
        explainer = _explainer_from_run_id(run_id)
        method    = _method_from_run_id(run_id)
        label     = MODEL_LABELS.get(run_id, run_id.replace('_42', '').replace('_', '\n'))
        model_configs.append({
            'run_id':    run_id,
            'checkpoint': ckpt,
            'method':    method,
            'explainer': explainer,
            'label':     label,
        })

    if not model_configs:
        print("[ERROR] No valid checkpoints found. Check --models and --checkpoints_dir.")
        return

    # --- Image selection ---
    baseline_id = args.models[0]
    groups = select_images(per_sample_df, baseline_id, facet_split_df,
                           n_per_group=args.n_images, seed=args.seed)

    all_saliency = []

    for criterion, image_names in groups.items():
        if not image_names:
            continue
        print(f"\nGenerating figure for: {criterion} ({len(image_names)} images) ...")
        out_path  = os.path.join(args.output_dir, f'explanations_{criterion}.png')
        sal_df    = make_explanation_grid(
            image_names=image_names,
            model_configs=model_configs,
            img_dir=img_dir,
            per_sample_df=per_sample_df,
            facet_split_df=facet_split_df,
            device=device,
            output_path=out_path,
            title=criterion.replace('_', ' ').title(),
        )
        all_saliency.append(sal_df)

    # Save per-image saliency metrics
    if all_saliency:
        sal_out = os.path.join(args.output_dir, 'per_image_saliency.csv')
        pd.concat(all_saliency, ignore_index=True).to_csv(sal_out, index=False)
        print(f"\nPer-image saliency metrics saved: {sal_out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
