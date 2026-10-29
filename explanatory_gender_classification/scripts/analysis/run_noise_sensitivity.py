"""
Sensitivity of one XIL configuration to imperfect (noisy) box feedback.

Simulates annotators who draw imprecise boxes: in every training epoch each box is perturbed
(`--feedback_noise bbox_jitter`: centroid offset ~ N(0, level * box size) and width/height scaled by 1 + N(0, level);
 `--feedback_noise bbox_centroid`: centroid offset only). Evaluation masks are never perturbed.
Mean IoU between the perturbed and the true box (simulation with random boxes): about 0.85 (level 0.05), 0.74 (0.1), 0.56 (0.2), 0.43 (0.3).
The noise-free reference is the same configuration from the unified grid (level 0).

One shard (= separate process and results directory) per (level, seed). Use the same protocol flags as the unified grid.

Usage (repository root), example for the hybrid + BLA + bounding-box configuration with k=3:
    python explanatory_gender_classification/scripts/analysis/run_noise_sensitivity.py \
        --out results/journal_extension/noise --method hybrid --explainer bla --feedback bbox --k 3 \
        --mode bbox_jitter --levels 0.05 0.1 0.2 0.3 --seeds 42 1 2 --parallel 2 --threads 12
    python .../run_noise_sensitivity.py --out results/journal_extension/noise --merge
"""

import argparse
import glob
import os
import subprocess
import sys
import time

import pandas as pd

DEVICE = os.environ.get('XIL_DEVICE', 'cpu')  # set XIL_DEVICE=cuda to train on a GPU
HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.normpath(os.path.join(HERE, '..', 'run_bbox_feedback_experiments.py'))


def build_jobs(a):
    jobs = []
    for level in a.levels:
        for seed in a.seeds:
            name = f'{a.mode}_{level}_s{seed}'
            cmd = [sys.executable, LAUNCHER, '--methods', a.method, '--feedback_types', a.feedback,
                   '--k_values', str(a.k), '--seed', str(seed),
                   '--feedback_noise', a.mode, '--feedback_noise_level', str(level),
                   '--rrr_lambda', str(a.rrr_lambda),
                   *(['--backbone_mode', 'legacy', '--legacy_augmentation', '--rrr_output', 'native',
                      '--sampling_mode', 'selected', '--samples_per_epoch', str(a.samples_per_epoch),
                      '--sampling_strategies', a.sampling_label, '--eval_reference', 'both', '--device', DEVICE]
                     if a.protocol == 'legacy_sampling' else
                     ['--backbone_mode', 'legacy', '--rrr_output', 'native',
                      '--sampling_mode', 'selected', '--samples_per_epoch', str(a.samples_per_epoch),
                      '--sampling_strategies', a.sampling_label, '--eval_reference', 'feedback', '--device', DEVICE]
                     if a.protocol == 'aligned_sampling' else
                     ['--backbone_mode', 'legacy', '--legacy_augmentation', '--rrr_output', 'native', '--sampling_mode', 'all',
                      '--sampling_strategies', 'uncertainty', '--eval_reference', 'feedback', '--device', DEVICE]
                     if a.protocol == 'legacy' else
                     ['--backbone_mode', 'trainable', '--rrr_output', 'log_softmax', '--sampling_mode', 'all',
                      '--sampling_strategies', 'none', '--eval_reference', 'both', '--device', DEVICE]),
                   '--results_dir', os.path.join(a.out, name)]
            cmd += ['--rrr_explainers' if a.method == 'rrr' else '--hybrid_explainers', a.explainer]
            jobs.append((name, cmd))
    return jobs


def merge(out):
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out, '*', 'results.csv')))]
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(out, 'results.csv'), index=False)
        print(f'merged {len(df)} rows')
        cols = ['feedback_noise', 'feedback_noise_level']
        print(df.groupby(cols)[['accuracy', 'error_rate_gap', 'BSR', 'DICE']].agg(['mean', 'std']).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--method', default='hybrid', choices=['caipi', 'rrr', 'hybrid'])
    ap.add_argument('--explainer', default='bla', choices=['gradcam', 'bla'])
    ap.add_argument('--feedback', default='bbox', choices=['bbox', 'segmentation'])
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--mode', default='bbox_jitter', choices=['bbox_jitter', 'bbox_centroid', 'shift', 'dilate', 'drop'])
    ap.add_argument('--levels', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 1, 2])
    ap.add_argument('--parallel', type=int, default=2)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--rrr_lambda', type=float, default=1000.0, help='use the same value as the unified grid')
    ap.add_argument('--protocol', choices=['unified', 'legacy', 'legacy_sampling', 'aligned_sampling'], default='unified',
                    help='legacy = settings of the first journal runs (comparable with the values in the paper); '
                         'aligned_sampling = budgeted sampling with image and feedback mask augmented jointly (same as the unified-grid protocol)')
    ap.add_argument('--samples_per_epoch', type=int, default=50, help='legacy_sampling / aligned_sampling only')
    ap.add_argument('--sampling_label', default='uncertainty', choices=['uncertainty', 'high_confidence'],
                    help='legacy_sampling / aligned_sampling only: sampling strategy of the noisy runs')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    if a.merge:
        merge(a.out)
        return
    os.makedirs(a.out, exist_ok=True)
    pending = build_jobs(a)
    if a.dry_run:
        for name, cmd in pending:
            print(name, '\n   ', ' '.join(cmd))
        return
    env = {**os.environ, 'OMP_NUM_THREADS': str(a.threads), 'MKL_NUM_THREADS': str(a.threads)}
    running = []
    while pending or running:
        running = [(n, p) for n, p in running if p.poll() is None]
        while pending and len(running) < a.parallel:
            name, cmd = pending.pop(0)
            print('start', name, flush=True)
            running.append((name, subprocess.Popen(cmd, stdout=open(os.path.join(a.out, f'{name}.log'), 'w'),
                                                   stderr=subprocess.STDOUT, env=env)))
        time.sleep(30)
    merge(a.out)


if __name__ == '__main__':
    main()
