"""
Figure and IoU statistics for the box-jitter feedback noise (feedback_noise = bbox_jitter, see data/xil_dataset.py).

The perturbation is re-implemented here with the same formula as `XILGenderDataset._perturb_feedback` and applied to the
real bounding boxes (derived from the person masks at 224 x 224) of the training images, so that the shown boxes and the
reported intersection-over-union (IoU) are those of the experiment.

Outputs:  fig_bbox_noise_examples.pdf (+ slides_assets/*.png),  results/journal_extension/analysis_noise/bbox_noise_iou.csv

Usage (repository root):
    python explanatory_gender_classification/scripts/analysis/make_bbox_noise_figure.py --out .
"""

import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import pandas as pd
from PIL import Image

LEVELS = [0.05, 0.1, 0.2, 0.3]
ROOT = 'gender_dataset'


def true_box(mask):
    ys, xs = np.where(mask > 0.5)
    return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1


def perturb(box, lvl, W=224, H=224):
    """Same arithmetic as XILGenderDataset._perturb_feedback (mode bbox_jitter)."""
    x0, x1, y0, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2 + random.gauss(0, lvl) * w
    cy = (y0 + y1) / 2 + random.gauss(0, lvl) * h
    w = w * max(0.2, 1 + random.gauss(0, lvl))
    h = h * max(0.2, 1 + random.gauss(0, lvl))
    nx0, nx1 = int(round(max(0, cx - w / 2))), int(round(min(W, cx + w / 2)))
    ny0, ny1 = int(round(max(0, cy - h / 2))), int(round(min(H, cy + h / 2)))
    return nx0, nx1, ny0, ny1


def iou(a, b):
    ix = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


def dice_coef(a, b):
    """Dice coefficient of two boxes, 2|A and B| / (|A| + |B|) = 2 IoU / (1 + IoU); the same overlap measure as DICE between saliency and mask"""
    ix = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    tot = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2])
    return 2 * ix * iy / tot if tot > 0 else 0.0


def load(row):
    rel = row.image.replace('\\', '/')
    kind = 'female' if 'female' in rel else 'male'
    name = os.path.basename(rel)
    img = Image.open(os.path.join(ROOT, rel)).convert('RGB').resize((224, 224), Image.BILINEAR)
    m = Image.open(os.path.join(ROOT, f'resized_{kind}_masks', name)).convert('L').resize((224, 224), Image.BILINEAR)
    return img, (np.asarray(m, dtype=np.float32) / 255.0), kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.')
    ap.add_argument('--examples', nargs='*', default=None, help='image file names to show (default: three fixed examples)')
    a = ap.parse_args()
    random.seed(0)
    train = pd.read_csv(f'{ROOT}/dataset_split/train_set.csv')

    # IoU statistics over all training boxes (10 draws per box and level)
    rows = []
    for _, r in train.iterrows():
        try:
            _, m, _ = load(r)
            box = true_box(m)
        except Exception:
            continue
        for lvl in LEVELS:
            for _ in range(10):
                b = perturb(box, lvl)  # one draw: IoU and Dice are computed from the same perturbed box
                rows.append((lvl, iou(box, b), dice_coef(box, b)))
    st = pd.DataFrame(rows, columns=['level', 'iou', 'dice'])
    summary = st.groupby('level').iou.agg(['mean', 'median', lambda s: s.quantile(.1), lambda s: s.quantile(.9)])
    summary.columns = ['mean', 'median', 'p10', 'p90']
    summary['dice_mean'] = st.groupby('level').dice.mean()
    summary['dice_median'] = st.groupby('level').dice.median()
    os.makedirs('results/journal_extension/analysis_noise', exist_ok=True)
    summary.to_csv('results/journal_extension/analysis_noise/bbox_noise_iou.csv')
    print(summary.round(3))

    # example images: fixed picks spread over the training set (one female-, two male-presenting if available)
    fem = train[train.image.str.contains('female')].reset_index(drop=True)
    mal = train[~train.image.str.contains('female')].reset_index(drop=True)
    picks = [fem.iloc[3], mal.iloc[7], fem.iloc[40]] if a.examples is None else [train[train.image.str.endswith(n)].iloc[0] for n in a.examples]

    plt.rcParams.update({'font.size': 10, 'savefig.bbox': 'tight'})
    fig, axs = plt.subplots(len(picks), 5, figsize=(11.2, 2.35 * len(picks) + .8))
    cols = ['#d9822b', '#c0392b', '#7d3c98']
    titles = ['true box'] + [f'noise level {l:g}\n(mean IoU {summary.loc[l, "mean"]:.2f}, Dice {summary.loc[l, "dice_mean"]:.2f})' for l in LEVELS]
    for i, r in enumerate(picks):
        img, m, _ = load(r)
        box = true_box(m)
        for j in range(5):
            ax = axs[i, j]
            ax.imshow(img)
            ax.add_patch(mp.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec='#2e8b57', lw=2.2, ls='--' if j else '-'))
            if j:
                for c in cols:
                    b = perturb(box, LEVELS[j - 1])
                    ax.add_patch(mp.Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False, ec=c, lw=1.6))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(titles[j], fontsize=9.5)
    fig.suptitle('Simulated imprecise feedback: bounding boxes perturbed in every epoch (green: true box; orange/red/purple: three independent draws)',
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'fig_bbox_noise_examples.pdf'))
    os.makedirs(os.path.join(a.out, 'slides_assets'), exist_ok=True)
    fig.savefig(os.path.join(a.out, 'slides_assets', 'fig_bbox_noise_examples.png'), dpi=190)
    print('wrote fig_bbox_noise_examples')


if __name__ == '__main__':
    main()
