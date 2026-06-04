"""
Extended FACET evaluation using SAM segmentation masks (person / clothing / hair).

For each image in the FACET test split that has a SAM mask in coco_masks.json,
computes saliency within three sub-regions:
  - person body (excluding clothing/hair)
  - clothing
  - hair
  - background (outside all three)

This gives a more granular view of what the model is actually attending to,
beyond the coarse bounding-box BSR.

Requires: pycocotools  (pip install pycocotools)

Usage:
  python scripts/run_facet_mask_evaluation.py [--all_models]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask_util
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.architectures import create_model
from models.rrr_model import RRRGenderClassifier
from explainability.gradcam import GradCAM
from explainability.bla import BLAWrapper, create_bla_model
from utils.helpers import set_random_seeds, get_device
from evaluation.bias_metrics import compute_dice_score

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# COCO mask category names as defined in FACET
MASK_CATS = {1: 'person', 2: 'clothing', 3: 'hair'}


# ---------------------------------------------------------------------------
# Model loading (mirrors run_facet_evaluation.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, method, device, explainer='gradcam'):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    # Detect architecture from the checkpoint's own keys (BLA -> 'feature_extractor.*').
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
    return model.to(device).eval()


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
# SAM mask loading
# ---------------------------------------------------------------------------

def resize_mask(mask_np, target_hw=(224, 224)):
    """Resize a binary mask to target size using nearest neighbour."""
    pil = Image.fromarray(mask_np * 255)
    pil = pil.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)


def build_person_mask_index(coco_masks_json):
    """
    Parse coco_masks.json and return a dict:
      facet_person_id -> {cat_name: binary_mask_np (224 x 224, float32)}

    Masks are resized to 224x224 immediately on decode to avoid OOM from
    storing thousands of full-resolution (1500x2250) arrays in RAM.
    """
    if not HAS_PYCOCOTOOLS:
        raise RuntimeError("pycocotools not installed. Run: pip install pycocotools")

    print(f"Loading SAM masks from {coco_masks_json} ...", flush=True)
    with open(coco_masks_json) as f:
        data = json.load(f)

    cat_map = {c['id']: c['name'] for c in data.get('categories', [])}
    print(f"  Categories: {cat_map}", flush=True)

    index = {}
    for i, ann in enumerate(data.get('annotations', [])):
        if i % 10000 == 0:
            print(f"  Processing annotation {i}/{len(data['annotations'])} ...", flush=True)
        pid  = ann.get('facet_person_id')
        cid  = ann.get('category_id')
        seg  = ann.get('segmentation')
        if pid is None or cid is None or seg is None:
            continue
        cat_name = cat_map.get(cid, str(cid))

        try:
            if isinstance(seg, list):
                h, w = ann.get('height', 0), ann.get('width', 0)
                if h == 0 or w == 0:
                    continue
                rle = coco_mask_util.frPyObjects(seg, h, w)
                rle = coco_mask_util.merge(rle)
            else:
                rle = seg
            m = coco_mask_util.decode(rle).astype(np.uint8)
            # Resize immediately — avoid storing full-res masks (OOM risk)
            m = resize_mask(m)
        except Exception:
            continue

        if pid not in index:
            index[pid] = {}
        if cat_name in index[pid]:
            index[pid][cat_name] = np.maximum(index[pid][cat_name], m)
        else:
            index[pid][cat_name] = m

    print(f"  Loaded masks for {len(index)} people.", flush=True)
    return index


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
        return sal
    except Exception as e:
        print(f"    [WARN] {e}")
        return torch.zeros(224, 224)


def region_saliency_fraction(sal_t, region_mask_np):
    """Fraction of total saliency mass that falls within region_mask."""
    r = torch.tensor(region_mask_np, dtype=torch.float32)
    return ((sal_t * r).sum() / (sal_t.sum() + 1e-8)).item()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_model_with_masks(model_info, facet_split_df, mask_index,
                               img_dir, device):
    """
    For each test image that has SAM masks (matched by person_id),
    compute per-region saliency fractions.

    Returns a list of dicts (one per image).
    """
    method       = model_info['method']
    ckpt         = model_info['checkpoint']
    # Derive explainer from the checkpoint filename so BLA models are always
    # loaded/explained as BLA regardless of the model_info flag.
    explainer_nm = 'bla' if '_bla' in os.path.basename(ckpt) else model_info.get('explainer', 'gradcam')

    print(f"  Loading {os.path.basename(ckpt)} ...", flush=True)
    model    = load_model(ckpt, method, device, explainer_nm)
    explainer = get_explainer(model, method, device, explainer_nm)

    records = []
    n_with_mask = 0

    for _, row in facet_split_df.iterrows():
        pid  = int(row['person_id'])
        if pid not in mask_index:
            continue
        n_with_mask += 1

        img_path = os.path.join(img_dir, row['image_name'])
        try:
            pil = Image.open(img_path).convert('RGB')
        except Exception:
            pil = Image.new('RGB', (224, 224))

        img_tensor = TRANSFORM(pil).unsqueeze(0).to(device)
        gt_label   = int(row['encoded_label'])

        with torch.no_grad():
            out = model(img_tensor)
            if isinstance(out, tuple):
                out = out[0]

        pred = int(torch.argmax(out, dim=1).item())
        sal  = compute_saliency(explainer, img_tensor, pred)

        masks_for_person = mask_index[pid]  # dict: cat_name -> np array (224x224, pre-resized)

        # Person-body: person minus clothing and hair
        person_m   = masks_for_person.get('person',   np.zeros((224, 224)))
        clothing_m = masks_for_person.get('clothing', np.zeros((224, 224)))
        hair_m     = masks_for_person.get('hair',     np.zeros((224, 224)))
        body_m     = np.clip(person_m - clothing_m - hair_m, 0, 1)

        # Background: outside all masks
        all_fg = np.clip(person_m + clothing_m + hair_m, 0, 1)
        bg_m   = 1.0 - all_fg

        # DICE: binarize saliency at median, compare to person foreground
        sal_np = sal.numpy()
        sal_binary = (sal_np > np.median(sal_np)).astype(np.float32)
        dice = compute_dice_score(
            torch.tensor(sal_binary), torch.tensor(all_fg)
        )

        records.append({
            'run_id':          model_info['label'],
            'method':          method,
            'explainer':       explainer_nm,
            'image_name':      row['image_name'],
            'person_id':       pid,
            'gender_label':    row['label'],
            'encoded_label':   gt_label,
            'prediction':      pred,
            'correct':         int(pred == gt_label),
            # Region saliency fractions (sum to ~1 if masks cover everything)
            'sal_body':       region_saliency_fraction(sal, body_m),
            'sal_clothing':   region_saliency_fraction(sal, clothing_m),
            'sal_hair':       region_saliency_fraction(sal, hair_m),
            'sal_background': region_saliency_fraction(sal, bg_m),
            # Classic BSR for comparison
            'BSR': region_saliency_fraction(sal, bg_m),
            # Person-foreground (body + clothing + hair)
            'sal_person_total': region_saliency_fraction(sal, all_fg),
            # DICE against SAM person mask
            'DICE': dice,
        })

    print(f"    Evaluated {n_with_mask} images with SAM masks.")
    return records


def select_models(coco_results_csv, checkpoints_dir, all_models=True):
    df = pd.read_csv(coco_results_csv)
    selected = []
    for _, row in df.iterrows():
        ckpt = os.path.join(checkpoints_dir, f"{row['run_id']}.pth")
        if not os.path.exists(ckpt):
            continue
        selected.append({
            'label':      row['run_id'],
            'method':     row['method'],
            'explainer':  row.get('explainer', 'gradcam'),
            'checkpoint': ckpt,
        })
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--facet_dir',   default=r'c:\Users\Queby\Research\xil-gender-classification\facet_dataset\facet_processed')
    parser.add_argument('--masks_json',  default=r'c:\Users\Queby\Research\xil-gender-classification\facet_dataset\annotations\coco_masks.json')
    parser.add_argument('--coco_results_csv', default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\coco_bbox\results.csv')
    parser.add_argument('--checkpoints_dir',  default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\coco_bbox\checkpoints')
    parser.add_argument('--results_dir', default=r'c:\Users\Queby\Research\xil-gender-classification\results\journal_extension\facet_eval')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--curated_dir', type=str,
                        default=r'c:\Users\Queby\Research\xil-gender-classification\facet_dataset\labeled_images',
                        help='If set, restrict evaluation to images present in <curated_dir>/female/ '
                             'and <curated_dir>/male/ (curated balanced subset).')
    args = parser.parse_args()

    if not HAS_PYCOCOTOOLS:
        print("[ERROR] pycocotools not installed. Run: pip install pycocotools")
        return

    set_random_seeds(args.seed)
    device = get_device() if args.device == 'cuda' else torch.device('cpu')
    os.makedirs(args.results_dir, exist_ok=True)

    # Load FACET test split
    split_csv = os.path.join(args.facet_dir, 'splits', 'test.csv')
    test_df   = pd.read_csv(split_csv)
    img_dir   = os.path.join(args.facet_dir, 'images')
    print(f"Test set: {len(test_df)} rows")

    # Filter to curated balanced subset if requested
    if args.curated_dir and os.path.isdir(args.curated_dir):
        female_names = set(os.listdir(os.path.join(args.curated_dir, 'female')))
        male_names   = set(os.listdir(os.path.join(args.curated_dir, 'male')))
        curated_names = female_names | male_names
        before = len(test_df)
        test_df = test_df[test_df['image_name'].isin(curated_names)].reset_index(drop=True)
        print(f"Curated filter applied: {before} -> {len(test_df)} samples "
              f"({len(female_names)} female, {len(male_names)} male)")

    # Build SAM mask index
    mask_index = build_person_mask_index(args.masks_json)

    # Match test images to masks
    test_with_masks = test_df[test_df['person_id'].isin(mask_index.keys())]
    print(f"Test images with SAM masks: {len(test_with_masks)} "
          f"({len(test_with_masks)/len(test_df)*100:.1f}% of test set)")

    if test_with_masks.empty:
        print("[WARN] No test images have SAM masks. "
              "person_id values may not match. Check coco_masks.json format.")
        return

    # Select models
    model_list = select_models(args.coco_results_csv, args.checkpoints_dir)
    print(f"Models to evaluate: {len(model_list)}")

    # Incremental skip
    out_csv = os.path.join(args.results_dir, 'facet_test_sam_mask_metrics.csv')
    done_ids = set()
    if os.path.exists(out_csv):
        done_ids = set(pd.read_csv(out_csv)['run_id'].unique())
        print(f"  Resuming — {len(done_ids)} model(s) already done, skipping.")

    pending = [m for m in model_list if m['label'] not in done_ids]

    for model_info in pending:
        print(f"\n{'='*55}\n{model_info['label']}\n{'='*55}")
        records = evaluate_model_with_masks(
            model_info, test_with_masks, mask_index, img_dir, device
        )
        if not records:
            continue

        df_out = pd.DataFrame(records)

        # Per-model summary
        mean_sal = df_out[['sal_body', 'sal_clothing', 'sal_hair',
                            'sal_background', 'sal_person_total', 'DICE']].mean()
        print(f"  Mean saliency fractions:")
        for col in mean_sal.index:
            print(f"    {col:20s}: {mean_sal[col]:.3f}")

        mode   = 'a' if os.path.exists(out_csv) else 'w'
        header = not os.path.exists(out_csv)
        df_out.to_csv(out_csv, mode=mode, header=header, index=False)
        print(f"  Saved incremental results to {out_csv}")

    print(f"\nDone. Results in {out_csv}")


if __name__ == '__main__':
    main()
