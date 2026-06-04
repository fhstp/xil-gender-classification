"""
Qualitative visualization for COCO test set.

Generates a grid figure comparing explanation heatmaps across models on the
same COCO test images. Optionally shows the segmentation mask used during training.

Also produces a side-by-side segmentation vs bbox comparison to show what
foreground feedback looks like for the same image.

Usage:
  python scripts/visualize_coco_explanations.py
"""

import os
import sys
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

GENDER_LABEL = {0: 'female', 1: 'male'}

MODEL_LABELS = {
    'baseline_none_k0_none_gradcam_42':                      'Baseline\n(GradCAM)',
    'caipi_segmentation_k1_high_confidence_gradcam_42':      'CAIPI\nseg-k1 (GradCAM)',
    'caipi_bbox_k1_high_confidence_gradcam_42':              'CAIPI\nbbox-k1 (GradCAM)',
    'rrr_segmentation_k0_high_confidence_gradcam_42':        'RRR\nseg (GradCAM)',
    'rrr_bbox_k0_high_confidence_gradcam_42':                'RRR\nbbox (GradCAM)',
    'rrr_segmentation_k0_high_confidence_bla_42':            'RRR\nseg (BLA)',
    'rrr_bbox_k0_high_confidence_bla_42':                    'RRR\nbbox (BLA)',
    'hybrid_segmentation_k1_high_confidence_bla_42':         'Hybrid\nseg-k1 (BLA)',
    'hybrid_bbox_k1_high_confidence_bla_42':                 'Hybrid\nbbox-k1 (BLA)',
}

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
# Model loading
# ---------------------------------------------------------------------------

def _explainer_from_run_id(run_id):
    return 'bla' if ('_bla_' in run_id or run_id.endswith('_bla_42')) else 'gradcam'


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
    model = model.to(device).eval()
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
# Saliency helpers
# ---------------------------------------------------------------------------

def compute_saliency(explainer, img_tensor, label):
    try:
        if hasattr(explainer, 'generate_cam'):
            sal = explainer.generate_cam(img_tensor, label)
        else:
            sal = explainer.generate_explanation(img_tensor)[0]
        sal = sal.detach().cpu()
        if sal.shape != torch.Size([224, 224]):
            sal = F.interpolate(
                sal.unsqueeze(0).unsqueeze(0).float(),
                size=(224, 224), mode='bilinear', align_corners=False
            ).squeeze()
        sal = sal.float()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        return sal.numpy()
    except Exception as e:
        print(f"    [WARN] saliency failed: {e}")
        return np.zeros((224, 224))


def overlay_heatmap(img_arr, sal_np, alpha=0.45):
    heat = cm.jet(sal_np)[:, :, :3]
    img  = img_arr / 255.0
    blended = (1 - alpha) * img + alpha * heat
    return (blended * 255).astype(np.uint8)


def mask_to_rgba_overlay(mask_224, color=(0, 1, 0), alpha=0.4):
    """Return H×W×4 RGBA array for overlaying a binary mask."""
    overlay = np.zeros((224, 224, 4), dtype=np.float32)
    overlay[mask_224 > 0.5, :3] = color
    overlay[mask_224 > 0.5, 3]  = alpha
    return overlay


# ---------------------------------------------------------------------------
# COCO dataset loader
# ---------------------------------------------------------------------------

def load_coco_test_split(dataset_dir, split_csv='dataset_split/test_set.csv'):
    df = pd.read_csv(os.path.join(dataset_dir, split_csv))
    records = []
    for _, row in df.iterrows():
        img_rel = str(row['image']).replace('\\', os.sep).replace('/', os.sep)
        img_path = os.path.join(dataset_dir, img_rel)
        # Derive mask path: same filename, replace images→masks folder
        mask_rel = img_rel.replace('resized_female_images', 'resized_female_masks') \
                          .replace('resized_male_images', 'resized_male_masks')
        mask_path = os.path.join(dataset_dir, mask_rel)
        gender = 0 if 'female' in img_rel else 1
        records.append({
            'img_id':    row['img_id'],
            'img_path':  img_path,
            'mask_path': mask_path,
            'gender':    gender,
            'label_str': GENDER_LABEL[gender],
        })
    return pd.DataFrame(records)


def load_coco_image_and_mask(img_path, mask_path):
    """Return (img_224_np, mask_224_np, bbox_224) — mask is float32 [0,1],
    bbox_224 is [x1, y1, x2, y2] in 224×224 coordinates (or None)."""
    try:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
    except Exception:
        img = Image.new('RGB', (224, 224))
    img_np = np.array(img)

    try:
        msk = Image.open(mask_path).convert('L').resize((224, 224))
        msk_np = (np.array(msk) > 127).astype(np.float32)
    except Exception:
        msk_np = np.zeros((224, 224), dtype=np.float32)

    # Derive bounding box from the segmentation mask
    if msk_np.max() > 0:
        ys, xs = np.where(msk_np > 0)
        bbox_224 = [xs.min(), ys.min(), xs.max(), ys.max()]  # x1,y1,x2,y2
    else:
        bbox_224 = None

    return img_np, msk_np, bbox_224


def draw_bbox_rect(ax, bbox_224, color='lime', lw=1.5):
    """Draw a rectangle on ax given [x1,y1,x2,y2] in 224-px coords."""
    if bbox_224 is None:
        return
    x1, y1, x2, y2 = bbox_224
    rect = mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def select_coco_images(test_df, n_per_gender=4, seed=42):
    """Random stratified selection (fallback)."""
    rng = np.random.default_rng(seed)
    selected = []
    for g in [0, 1]:
        subset = test_df[test_df['gender'] == g]
        idx = rng.choice(len(subset), size=min(n_per_gender, len(subset)), replace=False)
        selected.append(subset.iloc[idx])
    return pd.concat(selected, ignore_index=True)


def select_coco_images_high_background(test_df, n_per_gender=4, n_candidates=120, seed=42):
    """
    Select images where the person occupies a small fraction of the frame —
    a proxy for high background saliency (bad explanations).

    Samples n_candidates images per gender, computes foreground fraction from
    the segmentation mask (faster than running model inference), then returns
    the n_per_gender images with the smallest foreground area.
    """
    rng = np.random.default_rng(seed)
    selected = []
    for g in [0, 1]:
        subset = test_df[test_df['gender'] == g].reset_index(drop=True)
        n_cand = min(n_candidates, len(subset))
        cand_idx = rng.choice(len(subset), size=n_cand, replace=False)
        candidates = subset.iloc[cand_idx].copy().reset_index(drop=True)

        fg_fracs = []
        for row in candidates.itertuples():
            _, seg_mask, _ = load_coco_image_and_mask(row.img_path, row.mask_path)
            fg_fracs.append(float(seg_mask.mean()))

        candidates['_fg_frac'] = fg_fracs
        # Smallest fg fraction = most background = likely highest BSR
        top = candidates.sort_values('_fg_frac').head(n_per_gender).drop(columns=['_fg_frac'])
        selected.append(top)

    return pd.concat(selected, ignore_index=True)


# ---------------------------------------------------------------------------
# Mask contrast figure: show segmentation vs bbox side-by-side for a few images
# ---------------------------------------------------------------------------

def plot_mask_contrast(image_rows, output_path):
    """
    Grid: rows=images, cols=[original + bbox | segmentation overlay | bbox rectangle overlay]

    Uniform style with FACET figures: original image always visible in background,
    lime green bounding box drawn the same way, masks shown as semi-transparent overlays.
    """
    n = len(image_rows)
    fig, axes = plt.subplots(n, 3, figsize=(7, 2.5 * n), squeeze=False)
    axes[0, 0].set_title('Original\n+ bbox', fontsize=8, fontweight='bold')
    axes[0, 1].set_title('Segmentation mask\n(used in seg feedback)', fontsize=8, fontweight='bold')
    axes[0, 2].set_title('Bounding box mask\n(used in bbox feedback)', fontsize=8, fontweight='bold')

    for i, row in enumerate(image_rows.itertuples()):
        img_np, seg_mask, bbox_224 = load_coco_image_and_mask(row.img_path, row.mask_path)

        ax0, ax1, ax2 = axes[i, 0], axes[i, 1], axes[i, 2]

        # Col 0: original + lime bbox
        ax0.imshow(img_np)
        draw_bbox_rect(ax0, bbox_224)
        ax0.set_ylabel(f"{row.label_str}\n{os.path.basename(row.img_path)[:15]}",
                       fontsize=6, rotation=0, labelpad=55, va='center')

        # Col 1: original + segmentation overlay + bbox
        ax1.imshow(img_np)
        if seg_mask.max() > 0:
            ax1.imshow(mask_to_rgba_overlay(seg_mask, color=(0, 1, 0)))
        draw_bbox_rect(ax1, bbox_224)

        # Col 2: original + bbox-rectangle overlay + bbox
        ax2.imshow(img_np)
        if bbox_224 is not None:
            x1, y1, x2, y2 = bbox_224
            bbox_mask = np.zeros((224, 224), dtype=np.float32)
            bbox_mask[y1:y2+1, x1:x2+1] = 1.0
            ax2.imshow(mask_to_rgba_overlay(bbox_mask, color=(1, 0.5, 0)))
        draw_bbox_rect(ax2, bbox_224)

        for ax in [ax0, ax1, ax2]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main explanation grid for COCO
# ---------------------------------------------------------------------------

def make_coco_explanation_grid(
    image_rows, model_configs, device, output_path, title=None
):
    """
    rows = images (female then male)
    cols = [original + mask | model1 | model2 | ...]
    """
    n_images = len(image_rows)
    n_cols   = 1 + len(model_configs)
    fig, axes = plt.subplots(n_images, n_cols,
                              figsize=(2.5 * n_cols, 2.8 * n_images),
                              squeeze=False)

    axes[0, 0].set_title('Original\n+ BBox', fontsize=8, fontweight='bold')
    for j, mc in enumerate(model_configs):
        axes[0, j + 1].set_title(mc['label'], fontsize=7, fontweight='bold')

    print(f"  Loading {len(model_configs)} model(s)...")
    loaded = []
    for mc in model_configs:
        print(f"    {mc['run_id']} ...", flush=True)
        m   = load_model(mc['checkpoint'], mc['method'], device, mc['explainer'])
        exp = get_explainer(m, mc['method'], device, mc['explainer'])
        loaded.append((m, exp))

    per_image_records = []

    for i, row in enumerate(image_rows.itertuples()):
        img_np, seg_mask, bbox_224 = load_coco_image_and_mask(row.img_path, row.mask_path)
        gt_label = row.gender

        try:
            pil_img = Image.open(row.img_path).convert('RGB')
        except Exception:
            pil_img = Image.new('RGB', (224, 224))
        img_tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)

        # Column 0: original image + lime green bbox (same style as FACET)
        ax0 = axes[i, 0]
        ax0.imshow(img_np)
        draw_bbox_rect(ax0, bbox_224)
        ax0.set_ylabel(f"{row.label_str}\n{os.path.basename(row.img_path)[:15]}",
                       fontsize=6, rotation=0, labelpad=55, va='center')
        ax0.set_xticks([])
        ax0.set_yticks([])

        for j, (mc, (model, explainer)) in enumerate(zip(model_configs, loaded)):
            with torch.no_grad():
                out = model(img_tensor)
                if isinstance(out, tuple):
                    out = out[0]
            probs = torch.softmax(out, dim=1) if out.max().item() > 0 else torch.exp(out)
            pred  = int(torch.argmax(out, dim=1).item())
            conf  = probs[0, pred].item()

            sal_np = compute_saliency(explainer, img_tensor, pred)

            # BSR computed with segmentation mask as foreground (more precise than bbox)
            sal_t = torch.tensor(sal_np)
            fg_t  = torch.tensor(seg_mask, dtype=torch.bool)
            bg_t  = ~fg_t
            bsr   = (sal_t * bg_t.float()).sum() / (sal_t.sum() + 1e-8)

            # Explanation overlay on original image + lime bbox (FACET-uniform style)
            blended = overlay_heatmap(img_np, sal_np)
            ax = axes[i, j + 1]
            ax.imshow(blended)
            draw_bbox_rect(ax, bbox_224)

            correct  = (pred == gt_label)
            pred_str = GENDER_LABEL[pred]
            tick_sym = '✓' if correct else '✗'
            ax.set_xlabel(
                f"{tick_sym}{pred_str}  BSR={bsr:.2f}",
                fontsize=6, color='green' if correct else 'red'
            )
            ax.set_xticks([])
            ax.set_yticks([])

            per_image_records.append({
                'img_path':   row.img_path,
                'gender':     row.label_str,
                'run_id':     mc['run_id'],
                'method':     mc['method'],
                'explainer':  mc['explainer'],
                'prediction': pred_str,
                'correct':    int(correct),
                'confidence': conf,
                'BSR':        bsr.item(),
            })

    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold', y=1.01)

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return pd.DataFrame(per_image_records)


# ---------------------------------------------------------------------------
# Seg vs bbox comparison: same model, segmentation mask vs bbox mask as feedback
# ---------------------------------------------------------------------------

def plot_seg_vs_bbox_metrics(facet_results_csv, output_path):
    """
    Bar chart: compare paired seg vs bbox runs on ERR_gap and BSR.
    Pairs: RRR-seg vs RRR-bbox, Hybrid-seg vs Hybrid-bbox, CAIPI-seg vs CAIPI-bbox.
    """
    df = pd.read_csv(facet_results_csv)

    pairs = [
        ('caipi_segmentation_k1_high_confidence_gradcam_42', 'caipi_bbox_k1_high_confidence_gradcam_42', 'CAIPI k1\nhigh_conf (GradCAM)'),
        ('caipi_segmentation_k1_uncertainty_gradcam_42',     'caipi_bbox_k1_uncertainty_gradcam_42',     'CAIPI k1\nuncertainty (GradCAM)'),
        ('rrr_segmentation_k0_high_confidence_gradcam_42',   'rrr_bbox_k0_high_confidence_gradcam_42',   'RRR high_conf\n(GradCAM)'),
        ('rrr_segmentation_k0_high_confidence_bla_42',       'rrr_bbox_k0_high_confidence_bla_42',       'RRR high_conf\n(BLA)'),
        ('hybrid_segmentation_k1_high_confidence_bla_42',    'hybrid_bbox_k1_high_confidence_bla_42',    'Hybrid k1\nhigh_conf (BLA)'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(pairs))
    width = 0.35

    for metric, ax, ylabel, better in [
        ('error_rate_gap', ax1, 'Error Rate Gap (lower = fairer)', 'lower'),
        ('BSR',            ax2, 'Background Saliency Ratio (lower = more focused on person)', 'lower'),
    ]:
        seg_vals, bbox_vals, labels = [], [], []
        for seg_id, bbox_id, label in pairs:
            seg_row  = df[df['label'] == seg_id]
            bbox_row = df[df['label'] == bbox_id]
            if seg_row.empty or bbox_row.empty:
                continue
            seg_vals.append(seg_row[metric].values[0])
            bbox_vals.append(bbox_row[metric].values[0])
            labels.append(label)

        x_pos = np.arange(len(labels))
        ax.bar(x_pos - width/2, seg_vals,  width, label='Segmentation mask', color='#5C85D6', edgecolor='white')
        ax.bar(x_pos + width/2, bbox_vals, width, label='Bounding box mask',  color='#D6875C', edgecolor='white')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f'{metric} — Segmentation vs Bbox feedback', fontsize=9, fontweight='bold')
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',     default=r'c:\Users\Queby\Research\xil-gender-classification\gender_dataset')
    parser.add_argument('--checkpoints_dir', default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\coco_bbox\checkpoints')
    parser.add_argument('--facet_results',   default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\facet_eval\facet_test_results.csv')
    parser.add_argument('--output_dir',      default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\facet_eval\figures')
    parser.add_argument('--n_images',        type=int, default=4,
                        help='Images per gender to show in explanation grid')
    parser.add_argument('--models',          nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--device',          default='cuda')
    parser.add_argument('--seed',            type=int, default=42)
    args = parser.parse_args()

    set_random_seeds(args.seed)
    device = get_device() if args.device == 'cuda' else torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Seg vs bbox metric comparison (no model loading needed) ---
    print("Generating segmentation vs bbox metric comparison...")
    plot_seg_vs_bbox_metrics(
        args.facet_results,
        os.path.join(args.output_dir, 'seg_vs_bbox_metrics.png')
    )

    # --- Load COCO test split ---
    print("\nLoading COCO test set...")
    test_df = load_coco_test_split(args.dataset_dir)
    print(f"  {len(test_df)} test images  "
          f"({(test_df.gender==0).sum()} female, {(test_df.gender==1).sum()} male)")

    # --- Mask contrast figure (no model loading) ---
    sample_rows = select_coco_images(test_df, n_per_gender=3, seed=args.seed)
    print("\nGenerating mask contrast figure (segmentation vs bbox)...")
    plot_mask_contrast(
        sample_rows,
        os.path.join(args.output_dir, 'coco_mask_contrast.png')
    )

    # --- Select high-background images for explanation grid ---
    print("\nSelecting images with high background area (proxy for bad explanations)...")
    image_rows = select_coco_images_high_background(
        test_df, n_per_gender=args.n_images, n_candidates=120, seed=args.seed
    )
    print(f"  Selected {len(image_rows)} images ({(image_rows.gender==0).sum()} female, "
          f"{(image_rows.gender==1).sum()} male)")

    # --- Build model config list ---
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
            'run_id':     run_id,
            'checkpoint': ckpt,
            'method':     method,
            'explainer':  explainer,
            'label':      label,
        })

    if not model_configs:
        print("[ERROR] No valid checkpoints found.")
        return

    # --- Explanation grid ---
    print(f"\nGenerating COCO explanation grid ({len(image_rows)} images × {len(model_configs)} models)...")
    sal_df = make_coco_explanation_grid(
        image_rows=image_rows,
        model_configs=model_configs,
        device=device,
        output_path=os.path.join(args.output_dir, 'coco_explanation_grid.png'),
        title='COCO Test Set — Explanation Overlays (high-background images)',
    )

    sal_out = os.path.join(args.output_dir, 'coco_per_image_saliency.csv')
    sal_df.to_csv(sal_out, index=False)
    print(f"  Saliency metrics saved: {sal_out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
