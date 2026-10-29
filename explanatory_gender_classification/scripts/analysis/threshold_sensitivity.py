"""
Sensitivity of the threshold-based saliency metrics (FFP, BFP, DICE) to the binarisation threshold.

The main analysis binarises saliency maps at the 25th percentile (FFP/BFP) and at the median (DICE). This script
computes every saliency map ONCE and then evaluates, against the person segmentation mask of the test images:

  FFP, BFP : thresholds at percentiles 10, 25, 50, 75, 90 of the saliency map
  DICE     : thresholds at percentiles 25, 50, 75, 90 and "area-matched" (the threshold is chosen so that the
             binarised saliency covers exactly as many pixels as the person mask, which removes the dependence of
             DICE on the footprint of the region)
  BSR      : threshold-free (reported once)

Output: <out> (long format: run_id, metric, threshold, mean, std over test images).

Usage (repository root; run ids are checkpoint names without .pth):
    python explanatory_gender_classification/scripts/analysis/threshold_sensitivity.py \
        --results_dir results/journal_extension/unified/<shard> \
        --run_ids baseline_none_k0_none_gradcam_42 hybrid_bbox_k3_none_bla_42 \
        --method_of baseline:gradcam hybrid:bla --out results/journal_extension/analysis/threshold_sensitivity.csv
(method/explainer are parsed from the run id: <method>_<feedback>_k<k>_<sampling>_<explainer>_<seed>)
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))

FFP_BFP_PCT = [10, 25, 50, 75, 90]
DICE_PCT = [25, 50, 75, 90]


def saliency_for(explainer, model, image, label):
    if hasattr(explainer, 'generate_cam'):
        return explainer.generate_cam(image, label)
    return explainer.generate_explanation(image)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--run_ids', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    import run_bbox_feedback_experiments as launcher
    from data.dataset import prepare_data_splits_from_dataset_folder, get_mask_directories
    from data.xil_dataset import create_xil_data_loaders
    from evaluation.bias_metrics import compute_ffp, compute_bfp, compute_bsr, compute_dice_score

    train_df, val_df, test_df, _ = prepare_data_splits_from_dataset_folder('gender_dataset')
    fm, mm = get_mask_directories('gender_dataset')
    _, _, test_loader = create_xil_data_loaders(train_df, val_df, test_df, batch_size=16, use_masks=True,
                                                mask_source='segmentation', female_masks_path=fm, male_masks_path=mm)
    device = torch.device('cpu')
    rows = []
    for run_id in a.run_ids:
        tokens = run_id.split('__')[0].split('_')
        method, explainer = tokens[0], tokens[-2]  # <method>_..._<explainer>_<seed>
        model, _ = launcher.build_models_for({'method': 'hybrid' if explainer == 'bla' else method, 'explainer': explainer},
                                             SimpleNamespace(backbone_mode='legacy'), device)
        ck = torch.load(os.path.join(a.results_dir, 'checkpoints', f'{run_id}.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck)
        model.eval()
        expl = launcher.get_explainer(model, explainer, device)
        acc = {}
        n = 0
        for batch in test_loader:
            images, labels, masks = batch[0], batch[1], batch[3]
            for i in range(len(images)):
                sal = saliency_for(expl, model, images[i:i + 1], int(labels[i]))
                mask = masks[i, 0] if masks.dim() == 4 else masks[i]
                if sal.shape != mask.shape:
                    sal = F.interpolate(sal[None, None], size=mask.shape, mode='bilinear', align_corners=False)[0, 0]
                n += 1
                acc.setdefault(('BSR', 'none'), []).append(float(compute_bsr(sal, mask)))
                for p in FFP_BFP_PCT:
                    acc.setdefault(('FFP', f'p{p}'), []).append(float(compute_ffp(sal, mask, float(p))))
                    acc.setdefault(('BFP', f'p{p}'), []).append(float(compute_bfp(sal, mask, float(p))))
                for p in DICE_PCT:
                    thr = torch.quantile(sal.flatten(), p / 100.0)
                    acc.setdefault(('DICE', f'p{p}'), []).append(float(compute_dice_score((sal >= thr).float(), mask)))
                area = float((mask > 0.5).float().mean())  # area-matched: same number of pixels as the person mask
                thr = torch.quantile(sal.flatten(), min(max(1.0 - area, 0.0), 1.0))
                acc.setdefault(('DICE', 'area_matched'), []).append(float(compute_dice_score((sal >= thr).float(), mask)))
        for (metric, thr_name), vals in acc.items():
            rows.append(dict(run_id=run_id, metric=metric, threshold=thr_name, mean=float(np.mean(vals)),
                             std=float(np.std(vals, ddof=1)), n_images=n))
        print(run_id, 'done', n, 'images', flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.pivot_table(index=['metric', 'threshold'], columns='run_id', values='mean').round(3).to_string())


if __name__ == '__main__':
    main()
