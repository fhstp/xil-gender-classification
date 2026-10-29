"""
Analysis of the feedback-noise sensitivity runs (run_noise_sensitivity.py) against the noise-free reference.

* The reference (level 0) is the SAME configuration from the reference grid (same method, explainer, feedback type, k,
  sampling label, augmentation) with the same seeds, so every noisy run has a matched noise-free run.
* A run is flagged `collapsed` when it (almost) always predicts one class (min(female_recall, male_recall) < 0.2).
  Collapsed runs are reported, never dropped; every summary is given with and without them.
* Effects are paired differences (noisy - noise-free, same seed) with a percentile-bootstrap 95 % CI over seeds.
  With few seeds the interval is wide; the per-seed table is printed so the reader can see every run.

Outcome metrics (columns of noise_runs.csv / noise_summary.csv):
  accuracy, error_rate_gap, worst_class_error (= max(female error, male error)), BSR and DICE (scored against the same mask the
  run was trained with, i.e. the clean bounding box; the noise-free reference is scored the same way), and, if the checkpoints of
  the runs were re-evaluated (reevaluate_checkpoints.py --results_dir <shard>), BSR_person / DICE_person against the person
  segmentation mask for noisy and reference runs alike (--reeval / --ref_reeval).

Besides the CSV files the script writes noise_table.tex (rows = noise levels) for the appendix, using the mean overlap of the perturbed
and the true box (IoU and Dice coefficient, from bbox_noise_iou.csv written by make_bbox_noise_figure.py).

Usage (repository root):
    python explanatory_gender_classification/scripts/analysis/noise_analysis.py \
        --noise results/journal_extension/noise_aligned/results.csv \
        --ref "results/pods/pod*/results/journal_extension/aligned_main*/*/results.csv" \
        --reeval "results/journal_extension/noise_aligned/*/reevaluation.csv" \
        --ref_reeval "results/pods/pod*/results/journal_extension/aligned_main*/reevaluation.csv" \
        --method hybrid --explainer bla --feedback bbox --k 3 --sampling uncertainty \
        --out results/journal_extension/analysis_noise
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
BASE_METRICS = ['accuracy', 'error_rate_gap', 'worst_class_error', 'BSR', 'DICE']
PERSON_METRICS = ['BSR_person', 'DICE_person']
NUMERIC = ['accuracy', 'error_rate_gap', 'BSR', 'DICE', 'k', 'seed', 'feedback_noise_level', 'female_recall', 'male_recall',
           'female_error_rate', 'male_error_rate']


def load(pattern_or_file):
    files = sorted(glob.glob(pattern_or_file)) or [pattern_or_file]
    parts = [pd.read_csv(f) for f in files if os.path.exists(f)]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.run_id != 'run_id'].copy()
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'female_error_rate' in df and 'male_error_rate' in df:
        df['worst_class_error'] = df[['female_error_rate', 'male_error_rate']].max(axis=1)
    return df


def attach_person_metrics(noisy, ref, reeval_glob, ref_reeval_glob):
    """BSR/DICE against the person mask from reevaluation.csv files; noisy shards are matched through their own results.csv."""
    rows = []
    for f in sorted(glob.glob(reeval_glob)) if reeval_glob else []:
        res = os.path.join(os.path.dirname(f), 'results.csv')
        if not os.path.exists(res):
            continue
        r, s = pd.read_csv(f), pd.read_csv(res)
        s = s[s.run_id != 'run_id']
        m = s[['run_id', 'feedback_noise_level', 'seed']].merge(r[['run_id', 'BSR', 'DICE']].rename(columns={'BSR': 'BSR_person', 'DICE': 'DICE_person'}), on='run_id')
        rows.append(m)
    if rows:
        p = pd.concat(rows, ignore_index=True)
        for c in ('feedback_noise_level', 'seed'):
            p[c] = pd.to_numeric(p[c])
        noisy = noisy.merge(p[['feedback_noise_level', 'seed', 'BSR_person', 'DICE_person']], on=['feedback_noise_level', 'seed'], how='left')
    if ref_reeval_glob:
        parts = [pd.read_csv(f) for f in sorted(glob.glob(ref_reeval_glob))]
        if parts:
            r = pd.concat(parts, ignore_index=True).drop_duplicates('run_id')
            r = r[['run_id', 'BSR', 'DICE']].rename(columns={'BSR': 'BSR_person', 'DICE': 'DICE_person'})
            ref = ref.merge(r, on='run_id', how='left')
    return noisy, ref


def boot_ci(d, n=20000):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return np.nan, np.nan
    return tuple(np.percentile([rng.choice(d, len(d)).mean() for _ in range(n)], [2.5, 97.5]))


def write_table(res, runs, iou_csv, out):
    """noise_table.tex: one row per level; means over the runs that trained normally and paired differences to the noise-free run of the same seed"""
    B = chr(92)
    iou = pd.read_csv(iou_csv).set_index('level') if iou_csv and os.path.exists(iou_csv) else None
    has_person = 'DICE_person_mean_noncollapsed' in res and bool(res['DICE_person_mean_noncollapsed'].notna().any())
    dice_col = 'DICE_person' if has_person else 'DICE'
    cols = [('accuracy', 100, 1, 'Acc.' + B + ' (' + B + '%)'), ('error_rate_gap', 1, 3, B + '(' + B + 'mathrm{ERR}_{' + B + 'mathrm{gap}}' + B + ')'),
            ('worst_class_error', 1, 3, 'Worst-class err.'), (dice_col, 1, 3, 'DICE')]
    has_c = int(res['n_collapsed'].sum()) > 0   # the column with the number of single-class runs is shown only if there are any
    head = ['Noise level', 'Box IoU', 'Box Dice'] + (['Single-class runs'] if has_c else []) + [c[3] for c in cols] + [B + '(' + B + 'Delta' + B + ') ' + (c[3] if c[0] != 'accuracy' else 'Acc.' + B + ' (pp)') for c in cols]
    lines = []
    for _, r in res.sort_values('level').iterrows():
        lvl = r['level']
        ok = runs[(runs.feedback_noise_level == lvl) & (~runs.collapsed)]
        i_ = '1.00' if lvl == 0 else (f"{iou.loc[lvl, 'mean']:.2f}" if iou is not None and lvl in iou.index else '--')
        d_ = '1.00' if lvl == 0 else (f"{iou.loc[lvl, 'dice_mean']:.2f}" if iou is not None and 'dice_mean' in iou and lvl in iou.index else '--')
        cells = [f'{lvl:g}', i_, d_] + ([f"{int(r['n_collapsed'])} of {int(r['n'])}"] if has_c else [])
        cells += [f'${ok[m].mean() * sc:.{nd}f}$' for m, sc, nd, _ in cols]
        cells += ['--' if lvl == 0 else f"${r[f'{m}_paired_diff_noncollapsed'] * sc:+.{max(nd - 1, 1) if m == 'accuracy' else nd}f}$" for m, sc, nd, _ in cols]
        lines.append(' & '.join(cells) + ' ' + B + B)
    tab = ('% generated by noise_analysis.py; DICE is scored against the ' + ('person segmentation mask' if has_person else 'bounding box used as feedback') + '\n'
           + B + 'begin{tabular}{ccc' + ('c' if has_c else '') + 'c' * len(cols) + 'r' * len(cols) + '}\n' + B + 'toprule\n' + ' & '.join(head) + ' ' + B + B + '\n' + B + 'midrule\n' + '\n'.join(lines)
           + '\n' + B + 'bottomrule\n' + B + 'end{tabular}\n')
    open(os.path.join(out, 'noise_table.tex'), 'w', encoding='utf8').write(tab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise', required=True)
    ap.add_argument('--ref', required=True, help='glob of reference-grid results.csv files (noise-free reference)')
    ap.add_argument('--ref_augmentation', default='joint_image_mask', help="augmentation label of the reference runs ('' for the original runs that have none)")
    ap.add_argument('--reeval', default='', help='glob of the reevaluation.csv files of the noisy shards (BSR/DICE against the person mask)')
    ap.add_argument('--ref_reeval', default='', help='glob of the reevaluation.csv files of the reference runs')
    ap.add_argument('--iou_csv', default='results/journal_extension/analysis_noise/bbox_noise_iou.csv')
    ap.add_argument('--method', default='hybrid')
    ap.add_argument('--explainer', default='bla')
    ap.add_argument('--feedback', default='bbox')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--sampling', default='uncertainty')
    ap.add_argument('--collapse_recall', type=float, default=0.2)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    noisy = load(a.noise)
    ref = load(a.ref)
    if a.ref_augmentation and 'augmentation' in ref:
        ref = ref[ref.augmentation == a.ref_augmentation]
    if 'feedback_noise' in ref:
        ref = ref[ref.feedback_noise.fillna('none') == 'none']   # rows of the original runs have no value in this column
    ref = ref[(ref.method == a.method) & (ref.explainer == a.explainer) & (ref.feedback_type == a.feedback) &
              (ref.k == a.k) & (ref.sampling_strategy == a.sampling)].copy()
    ref['feedback_noise_level'] = 0.0
    ref['feedback_noise'] = 'none'
    noisy, ref = attach_person_metrics(noisy, ref, a.reeval, a.ref_reeval)
    df = pd.concat([ref, noisy], ignore_index=True)
    df['collapsed'] = df[['female_recall', 'male_recall']].min(axis=1) < a.collapse_recall
    df = df.drop_duplicates(['feedback_noise_level', 'seed'])
    df.to_csv(os.path.join(a.out, 'noise_runs.csv'), index=False)
    metrics = BASE_METRICS + [m for m in PERSON_METRICS if m in df and df[m].notna().any()]

    pd.set_option('display.width', 220)
    print('seeds in reference:', sorted(ref.seed.astype(int).unique()), '| seeds in noisy runs:', sorted(noisy.seed.astype(int).unique()))
    for m in ('accuracy', 'error_rate_gap', 'worst_class_error', 'DICE'):
        print(f'\n{m} per run (rows = seed, columns = noise level; * = collapsed)')
        p = df.pivot(index='seed', columns='feedback_noise_level', values=m).round(3).astype(str)
        c = df.pivot(index='seed', columns='feedback_noise_level', values='collapsed').fillna(False).astype(bool)
        p = p.where(df.pivot(index='seed', columns='feedback_noise_level', values=m).notna(), '-')
        print(p.where(~c, p + '*').to_string())

    rows = []
    base = df[df.feedback_noise_level == 0].set_index('seed')
    for lvl, g in df.groupby('feedback_noise_level'):
        g = g.set_index('seed')
        ok = g[~g.collapsed]
        r = dict(level=lvl, n=len(g), n_collapsed=int(g.collapsed.sum()))
        for m in metrics:
            r[f'{m}_mean_all'] = g[m].mean()
            r[f'{m}_median_all'] = g[m].median()
            r[f'{m}_mean_noncollapsed'] = ok[m].mean() if len(ok) else np.nan
            if lvl > 0:
                common = g.index.intersection(base.index)
                d = (g.loc[common, m] - base.loc[common, m]).values
                lo, hi = boot_ci(d[~np.isnan(d)])
                r[f'{m}_paired_diff'], r[f'{m}_ci_low'], r[f'{m}_ci_high'] = np.nanmean(d) if len(d) else np.nan, lo, hi
                both = [s for s in common if not g.loc[s, 'collapsed'] and not base.loc[s, 'collapsed']]
                dn = (g.loc[both, m] - base.loc[both, m]).values
                r[f'{m}_paired_diff_noncollapsed'] = np.nanmean(dn) if len(dn) else np.nan
                r['n_pairs_noncollapsed'] = len(both)
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, 'noise_summary.csv'), index=False)
    write_table(res, df, a.iou_csv, a.out)
    show = ['level', 'n', 'n_collapsed'] + [f'{m}_paired_diff_noncollapsed' for m in metrics]
    print('\nsummary (paired differences to the noise-free run of the same seed, runs that trained normally)')
    print(res[[c for c in show if c in res]].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
