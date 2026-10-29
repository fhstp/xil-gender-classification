"""
Re-evaluate saved checkpoints with a FIXED evaluation protocol.

For every checkpoint (by default every run in <results_dir>/results.csv plus the three BLA models trained
without XIL) this computes, on the CURRENT train/val/test split files:

  * validation and test accuracy (validation accuracy allows configuration selection without touching the test set)
  * FFP / BFP / BSR / DICE against the person SEGMENTATION mask (same reference for every model, including
    the unsteered baselines and the bounding-box-trained models) and, with --with_bbox_ref, against the person BOUNDING BOX
    derived from that mask (suffix _bbox)

Output: <results_dir>/reevaluation.csv (one row per checkpoint, written incrementally; reruns skip finished rows).

Usage (from the repository root):
    python explanatory_gender_classification/scripts/analysis/reevaluate_checkpoints.py \
        --results_dir results/journal_extension/coco_bbox --workers 8
"""

import argparse
import os
import sys
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))           # scripts/  (for the launcher helpers)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))

DATA_DIR = 'gender_dataset'
RESULTS_DIR = 'results/journal_extension/coco_bbox'
_STATE = {}


def _init(threads, results_dir, with_bbox_ref):
    global RESULTS_DIR
    RESULTS_DIR = results_dir
    _STATE['with_bbox_ref'] = with_bbox_ref
    torch.set_num_threads(threads)


def _load_data(batch_size):
    from data.dataset import prepare_data_splits_from_dataset_folder, get_mask_directories
    from data.xil_dataset import create_xil_data_loaders
    train_df, val_df, test_df, _ = prepare_data_splits_from_dataset_folder(DATA_DIR)
    fm, mm = get_mask_directories(DATA_DIR)
    loaders = {}
    for ref in ('segmentation', 'bbox_from_segmentation'):
        _, v, t = create_xil_data_loaders(train_df, val_df, test_df, batch_size=batch_size, use_masks=True,
                                          mask_source=ref, female_masks_path=fm, male_masks_path=mm)
        loaders[ref] = (v, t)
    return loaders


@torch.no_grad()
def _accuracy(model, loader, collect=None):
    model.eval()
    correct = total = 0
    for batch in loader:
        images, labels = batch[0], batch[1]
        out = model(images)
        if isinstance(out, tuple):
            out = out[0]
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)
        if collect is not None:
            prob_male = torch.softmax(out, dim=1)[:, 1]
            meta = batch[4]
            for i in range(len(labels)):
                collect.append({'img_id': int(meta['img_id'][i]) if 'img_id' in meta else -1,
                                'label_gt': int(labels[i]), 'prediction': int(out[i].argmax()),
                                'prob_male': float(prob_male[i])})
    return correct / total, total


def evaluate_one(job):
    from types import SimpleNamespace
    import run_bbox_feedback_experiments as launcher
    from evaluation.bias_metrics import evaluate_model_bias

    run_id, method, explainer, ckpt = job
    if 'loaders' not in _STATE:
        _STATE['loaders'] = _load_data(16)
    loaders = _STATE['loaders']
    device = torch.device('cpu')

    args = SimpleNamespace(backbone_mode='legacy')
    model, _ = launcher.build_models_for({'method': method, 'explainer': explainer}, args, device)
    obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = obj['model_state_dict'] if isinstance(obj, dict) and 'model_state_dict' in obj else obj
    model.load_state_dict(state)
    model.eval()

    seg_val, seg_test = loaders['segmentation']
    row = {'run_id': run_id}
    row['val_accuracy'], row['n_val'] = _accuracy(model, seg_val)
    preds = []
    row['test_accuracy'], row['n_test'] = _accuracy(model, seg_test, collect=preds)
    os.makedirs(os.path.join(RESULTS_DIR, 'reeval_predictions'), exist_ok=True)
    pd.DataFrame(preds).assign(run_id=run_id).to_csv(
        os.path.join(RESULTS_DIR, 'reeval_predictions', f'{run_id}.csv'), index=False)
    expl = launcher.get_explainer(model, explainer, device)
    refs = [('', 'segmentation')] + ([('_bbox', 'bbox_from_segmentation')] if _STATE.get('with_bbox_ref') else [])
    for tag, ref in refs:
        m = evaluate_model_bias(model, loaders[ref][1], expl, device=device)
        row.update({f'{k}{tag}': float(v) for k, v in m.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results/journal_extension/coco_bbox')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--with_bbox_ref', action='store_true', help='also compute metrics against the bounding-box reference (2x slower)')
    ap.add_argument('--only', default='', help='substring filter on run ids (debugging)')
    args = ap.parse_args()
    global RESULTS_DIR
    RESULTS_DIR = args.results_dir

    res = pd.read_csv(os.path.join(args.results_dir, 'results.csv'))
    res = res[res.run_id != 'run_id']
    ck_dir = os.path.join(args.results_dir, 'checkpoints')
    jobs = [(r.run_id, 'hybrid' if (r.method == 'baseline' and r.explainer == 'bla') else r.method, r.explainer, os.path.join(ck_dir, f'{r.run_id}.pth')) for r in res.itertuples()]
    import glob as _glob
    bla_seeds = sorted({int(os.path.basename(p).rsplit('_', 1)[1][:-4]) for p in _glob.glob(os.path.join(ck_dir, 'baseline_none_k0_none_bla_*.pth'))}) or [42, 1, 2]
    for seed in bla_seeds:  # BLA trained without XIL (checkpoints of any seed; not necessarily part of results.csv)
        jobs.append((f'baseline_none_k0_none_bla_{seed}', 'hybrid', 'bla',
                     os.path.join(ck_dir, f'baseline_none_k0_none_bla_{seed}.pth')))
    seen_ids = set()
    jobs = [j for j in jobs if os.path.exists(j[3]) and args.only in j[0] and not (j[0] in seen_ids or seen_ids.add(j[0]))]

    out = os.path.join(args.results_dir, 'reevaluation.csv')
    done = set(pd.read_csv(out).run_id) if os.path.exists(out) else set()
    jobs = [j for j in jobs if j[0] not in done]
    print(f'{len(jobs)} checkpoints to evaluate, {len(done)} already done', flush=True)

    with mp.Pool(args.workers, initializer=_init, initargs=(args.threads, args.results_dir, args.with_bbox_ref)) as pool:
        for i, row in enumerate(pool.imap_unordered(evaluate_one, jobs), 1):
            pd.DataFrame([row]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
            print(f"[{i}/{len(jobs)}] {row['run_id']} val={row['val_accuracy']:.4f} test={row['test_accuracy']:.4f} "
                  f"BSR={row['BSR']:.3f}", flush=True)


if __name__ == '__main__':
    main()
