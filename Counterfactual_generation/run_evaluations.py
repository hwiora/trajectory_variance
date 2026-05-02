#!/usr/bin/env python
"""
run_evaluations.py — Unified evaluation for Interspeech 2026.

Computes ALL numbers for Tables 1 and 2 from one consistent model generation
(Feb 2024 VAE + displacement models).

For each bird (R4634 / R4951 / R5018):

  Table 1 — Raw Pearson correlations with trajectory variance (displacement model):
    6 acoustic features × 3 birds: r value and p-value
    + R²(duration) for transparency

  Table 2 — Baseline comparison (same eval subset, same target ages):
    |r|     : |Pearson corr(spectral flatness, V)| per method
    AUC     : song vs call AUC on duration-residualized variance
    d       : Cohen's d for song vs call on duration-residualized variance

Methods:
  1. Displacement (our learned model)
  2. Gaussian OT  (closed-form Monge map between per-age Gaussians)
  3. kNN (k=10)   (mean of 10 nearest neighbors at each target age)
  4. Per-age OT   (Hungarian assignment per source-target age pair)

Usage:
  python run_evaluations.py                  # all 3 birds, full evaluation
  python run_evaluations.py --bird R4634     # single bird
  python run_evaluations.py --quick          # fast run (n_eval=2000)
  python run_evaluations.py --skip_hard_ot   # skip per-age OT (slow)

Output:
  models/paper_eval_results.json             # complete JSON
  Console: LaTeX-ready table rows for copy-paste into paper
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score

# ── Local imports (heavy computation lives in existing modules) ──
from .baseline_comparison import (
    load_data,
    make_target_ages,
    _draw_eval_subset,
    _strict_ot_subset,
    knn_trajectory_variance,
    gaussian_ot_trajectory_variance,
    ot_trajectory_variance,
    neural_trajectory_variance,
    trajectory_metrics,
)
from .analyze_plasticity import (
    compute_acoustic_features_streaming,
)


# ═════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════

BIRDS = ["R4634", "R4951", "R5018"]
BIRD_LABELS = {"R4634": "A", "R4951": "B", "R5018": "C"}

_MODELS = Path(__file__).parent / "models"

FLOW_DIRS_OT = {
    "R4634": _MODELS / "ot_flow_R4634_ot",
    "R4951": _MODELS / "ot_flow_R4951_ot",
    "R5018": _MODELS / "ot_flow_R5018_ot",
}

FLOW_DIRS_KNN = {
    "R4634": _MODELS / "ot_flow_R4634_knn",
    "R4951": _MODELS / "ot_flow_R4951_knn",
    "R5018": _MODELS / "ot_flow_R5018_knn",
}

# Default to OT models (backward compat); --coupling flag selects
FLOW_DIRS = FLOW_DIRS_OT

ACOUSTIC_FEATURES = ["spectral_flatness"]

# Display names for LaTeX
FEATURE_DISPLAY = {
    "spectral_flatness": "Spectral flatness",
}

OUTPUT_DIR = Path(__file__).parent / "models"


# ═════════════════════════════════════════════════════════════════
# Acoustic feature loading (with caching)
# ═════════════════════════════════════════════════════════════════

def load_or_compute_acoustic_features(bird: str, N: int) -> dict[str, np.ndarray]:
    """Load cached spectral flatness or compute from spectrograms.

    Returns dict with key 'spectral_flatness' -> (N_lat,) numpy array.
    NaN entries indicate segments without a corresponding spectrogram in
    the preprocessing snapshot used to compute these features.
    """
    cache_path = OUTPUT_DIR / f"spectral_flatness_{bird}.npz"
    if cache_path.exists():
        print(f"  Loading cached spectral flatness: {cache_path}")
        data = np.load(cache_path)
        return {"spectral_flatness": data["spectral_flatness"]}

    print(f"  Computing acoustic features from spectrograms (streaming)...")
    t0 = time.time()
    full = compute_acoustic_features_streaming(bird, N)
    dt = time.time() - t0
    sf = np.full(N, np.nan, dtype=np.float64)
    sf[:len(full['spectral_flatness'])] = full['spectral_flatness']
    print(f"  Done in {dt:.1f}s.  Caching to {cache_path}")
    np.savez(cache_path, spectral_flatness=sf)
    return {"spectral_flatness": sf}


# ═════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═════════════════════════════════════════════════════════════════

def pearson_corr(x: np.ndarray, y: np.ndarray):
    """Raw Pearson correlation. Returns (r, p)."""
    r, p = scipy_stats.pearsonr(x, y)
    return float(r), float(p)


def cohens_d_residualized(values: np.ndarray, is_song: np.ndarray,
                          durations: np.ndarray) -> float:
    """Cohen's d for song vs call after linearly regressing out duration."""
    slope, intercept = np.polyfit(durations, values, 1)
    resid = values - (slope * durations + intercept)
    s, c = resid[is_song], resid[~is_song]
    n_s, n_c = len(s), len(c)
    pool_std = np.sqrt(
        (s.var() * (n_s - 1) + c.var() * (n_c - 1)) / (n_s + n_c - 2))
    return float((s.mean() - c.mean()) / (pool_std + 1e-10))


def auc_residualized(values: np.ndarray, is_song: np.ndarray,
                     durations: np.ndarray) -> float:
    """AUC for song vs call after regressing out duration."""
    slope, intercept = np.polyfit(durations, values, 1)
    resid = values - (slope * durations + intercept)
    try:
        return float(roc_auc_score(is_song.astype(int), resid))
    except Exception:
        return 0.5


def r2_duration(values: np.ndarray, durations: np.ndarray) -> float:
    """R² of variance explained by duration alone."""
    X = durations.reshape(-1, 1)
    reg = LinearRegression().fit(X, values)
    return float(r2_score(values, reg.predict(X)))


# ═════════════════════════════════════════════════════════════════
# Core evaluation per bird
# ═════════════════════════════════════════════════════════════════

def evaluate_bird(bird: str, n_eval: int = 10_000,
                  num_target_ages: int = 7, k: int = 10,
                  skip_hard_ot: bool = False,
                  coupling: str = "ot") -> dict:
    """Run complete evaluation for one bird.

    Args:
        coupling: Which trained displacement model to use.
                  "ot"  = Feb 24 models (OT/Hungarian coupling during training)
                  "knn" = Feb 26 models (kNN coupling during training)

    Returns a dict with all Table 1 and Table 2 entries.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_dirs_map = FLOW_DIRS_KNN if coupling == "knn" else FLOW_DIRS_OT
    flow_dir = flow_dirs_map[bird]

    print(f"\n{'='*70}")
    print(f"  EVALUATING {bird} (Bird {BIRD_LABELS[bird]}) — coupling={coupling}")
    print(f"{'='*70}")
    print(f"  Flow dir: {flow_dir}")
    print(f"  Device:   {device}")
    print(f"  n_eval:   {n_eval}")

    # ── Load data ──
    print("\n  Loading data...")
    data = load_data(flow_dir, bird, device,
                     labels_dir=Path(__file__).parent.parent / "gold_standard_labels")
    data['flow_dir'] = flow_dir
    N = len(data['latents_norm'])

    print(f"  N = {N:,}  ({data['is_song'].sum():,} song, "
          f"{(~data['is_song']).sum():,} call)")

    # ── Load/compute acoustic features ──
    features = load_or_compute_acoustic_features(bird, N)
    # NaN entries mark segments outside the feature-computation snapshot.
    N_feat = int(np.sum(~np.isnan(features['spectral_flatness'])))

    # ── Draw reproducible eval subset ──
    # Restrict to indices that have both latents and acoustic features
    sub_idx, sub_is_song, sub_dur = _draw_eval_subset(data, n_eval, seed=42)
    if N_feat < N:
        sub_idx = sub_idx[sub_idx < N_feat]
        sub_is_song = data['is_song'][sub_idx]
        sub_dur = data['durations'][sub_idx]
    N_eval = len(sub_idx)
    print(f"\n  Eval subset: {N_eval:,} samples "
          f"({sub_is_song.sum():,} song, {(~sub_is_song).sum():,} call)")

    # Acoustic features for the eval subset
    sub_features = {f: features[f][sub_idx] for f in ACOUSTIC_FEATURES}

    target_ages = make_target_ages(data['age_min'], data['age_max'],
                                   num_target_ages)

    # ── Compute trajectory variance for all methods ──
    methods = {}

    # 1. Displacement (our model)
    print("\n  [1/4] Displacement model...")
    t0 = time.time()
    traj_neural = neural_trajectory_variance(
        data['model'], data['latents_norm'], data['ages'],
        data['norm_min'], data['norm_max'], data['device'],
        target_ages, subset_idx=sub_idx)
    m_neural = trajectory_metrics(traj_neural)
    dt = time.time() - t0
    methods['Displacement'] = {
        'variance': m_neural['variance'],
        'time_sec': dt,
    }
    print(f"    Done in {dt:.1f}s. Mean var = {m_neural['variance'].mean():.3f}")

    # 2. Gaussian OT
    print("  [2/4] Gaussian OT...")
    t0 = time.time()
    traj_gauss = gaussian_ot_trajectory_variance(
        data['latents_norm'], data['ages'], target_ages,
        subset_idx=sub_idx)
    m_gauss = trajectory_metrics(traj_gauss)
    dt = time.time() - t0
    methods['Gaussian OT'] = {
        'variance': m_gauss['variance'],
        'time_sec': dt,
    }
    print(f"    Done in {dt:.1f}s. Mean var = {m_gauss['variance'].mean():.3f}")

    # 3. kNN (k=10)
    print(f"  [3/4] kNN (k={k})...")
    t0 = time.time()
    traj_knn = knn_trajectory_variance(
        data['latents_norm'], data['ages'], target_ages,
        k=k, subset_idx=sub_idx)
    m_knn = trajectory_metrics(traj_knn)
    dt = time.time() - t0
    methods[f'kNN(k={k})'] = {
        'variance': m_knn['variance'],
        'time_sec': dt,
    }
    print(f"    Done in {dt:.1f}s. Mean var = {m_knn['variance'].mean():.3f}")

    # 4. Per-age OT (optional — slow)
    if not skip_hard_ot:
        sub_idx_ot, ot_cov = _strict_ot_subset(
            sub_idx, data['ages'], target_ages, seed=42)
        if len(sub_idx_ot) > 0:
            print(f"  [4/4] Per-age OT (coverage={ot_cov:.1%})...")
            t0 = time.time()
            traj_ot = ot_trajectory_variance(
                data['latents_norm'], data['ages'], target_ages,
                subset_idx=sub_idx_ot)
            m_ot = trajectory_metrics(traj_ot)
            dt = time.time() - t0
            methods['Per-age OT'] = {
                'variance': m_ot['variance'],
                'time_sec': dt,
                'ot_sub_idx': sub_idx_ot,
                'ot_coverage': ot_cov,
            }
            print(f"    Done in {dt:.1f}s. Mean var = {m_ot['variance'].mean():.3f}")
        else:
            print("  [4/4] Per-age OT — skipped (empty strict subset)")
    else:
        print("  [4/4] Per-age OT — skipped by --skip_hard_ot")

    # ── Evaluate all methods ──
    print(f"\n{'─'*70}")
    print(f"  EVALUATION METRICS")
    print(f"{'─'*70}")

    results_by_method = {}

    for method_name, mdata in methods.items():
        var = mdata['variance']

        # Per-age OT may use a different subset
        if method_name == 'Per-age OT':
            ot_idx = mdata['ot_sub_idx']
            m_is_song = data['is_song'][ot_idx]
            m_dur = data['durations'][ot_idx]
            m_sub_idx = ot_idx
            m_features = {f: features[f][ot_idx] for f in ACOUSTIC_FEATURES}
        else:
            m_is_song = sub_is_song
            m_dur = sub_dur
            m_sub_idx = sub_idx
            m_features = sub_features

        # Raw Pearson correlations with all 6 acoustic features
        feature_correlations = {}
        for fname in ACOUSTIC_FEATURES:
            r_val, p_val = pearson_corr(var, m_features[fname])
            feature_correlations[fname] = {'r': r_val, 'p': p_val}

        # Cohen's d (song vs call, duration-residualized)
        d_resid = cohens_d_residualized(var, m_is_song, m_dur.astype(float))

        # AUC (song vs call, duration-residualized)
        auc_res = auc_residualized(var, m_is_song, m_dur.astype(float))

        # R²(duration) — for transparency
        r2_d = r2_duration(var, m_dur.astype(float))

        results_by_method[method_name] = {
            'feature_correlations': feature_correlations,
            'cohens_d_resid': d_resid,
            'auc_resid': auc_res,
            'r2_duration': r2_d,
            'time_sec': mdata['time_sec'],
            'n_eval': len(var),
        }

        # Print summary
        print(f"\n  {method_name}:")
        print(f"    R²(duration)  = {r2_d:.3f}")
        for fname in ACOUSTIC_FEATURES:
            fc = feature_correlations[fname]
            print(f"    r({fname:20s}) = {fc['r']:+.3f}  (p={fc['p']:.2e})")
        print(f"    Cohen's d_res  = {d_resid:+.3f}")
        print(f"    AUC_res        = {auc_res:.3f}")

    # ── Compile bird results ──
    bird_results = {
        'bird': bird,
        'label': BIRD_LABELS[bird],
        'coupling': coupling,
        'N': N,
        'n_eval': N_eval,
        'flow_dir': flow_dir,
        'table2': results_by_method,
    }
    return bird_results


# ═════════════════════════════════════════════════════════════════
# LaTeX table formatting
# ═════════════════════════════════════════════════════════════════

def print_latex_tables(all_results: dict, k: int = 10):
    """Print LaTeX-ready tables for the paper."""

    birds = [b for b in BIRDS if b in all_results]

    # ── Table 1: Feature correlations (displacement model only) ──
    print("\n" + "=" * 80)
    print("TABLE 1 — Raw Pearson correlations with trajectory variance (displacement)")
    print("=" * 80)

    header_labels = " & ".join(
        [f"\\textbf{{{all_results[b]['label']}}}" for b in birds])
    print(f"    \\textbf{{Feature}} & {header_labels} \\\\")
    print(r"    \midrule")

    for fname in ACOUSTIC_FEATURES:
        display = FEATURE_DISPLAY.get(fname, fname)
        parts = []
        for bird_id in birds:
            t2 = all_results[bird_id]['table2']
            if 'Displacement' in t2:
                fc = t2['Displacement']['feature_correlations']
                r_val = fc[fname]['r']
                parts.append(f"${r_val:+.2f}$")
            else:
                parts.append("--")
        row = f"    {display} & " + " & ".join(parts) + " \\\\"
        print(row)

    # Duration R² row
    print(r"    \midrule")
    parts_r2 = []
    for bird_id in birds:
        t2 = all_results[bird_id]['table2']
        if 'Displacement' in t2:
            r2 = t2['Displacement']['r2_duration']
            parts_r2.append(f"${r2:.2f}$")
        else:
            parts_r2.append("--")
    print(f"    $R^2$(duration) & " + " & ".join(parts_r2) + " \\\\")
    print(r"    \bottomrule")

    # ── Table 2: Baseline comparison ──
    print("\n" + "=" * 80)
    print("TABLE 2 — Baseline comparison")
    print("=" * 80)
    print(r"    & \multicolumn{" + str(len(birds)) + r"}{c}{$|r|$ (spec.\ flatness)} "
          r"& \multicolumn{" + str(len(birds)) + r"}{c}{$d_r$ (song vs.\ call)} \\")
    cmidrule_1 = f"\\cmidrule(lr){{2-{1+len(birds)}}}"
    cmidrule_2 = f"\\cmidrule(lr){{{2+len(birds)}-{1+2*len(birds)}}}"
    print(f"    {cmidrule_1} {cmidrule_2}")

    header_labels_2 = " & ".join(
        [f"\\textbf{{{all_results[b]['label']}}}" for b in birds])
    print(f"    \\textbf{{Method}} & {header_labels_2} & {header_labels_2} \\\\")
    print(r"    \midrule")

    knn_key = f'kNN(k={k})'
    method_display = {
        'Displacement': 'Displacement (ours)',
        'Gaussian OT': 'Gaussian OT',
        knn_key: f'$k$-NN ($k\\!=\\!{k}$)',
        'Per-age OT': 'Per-age OT',
    }

    for method_key in ['Displacement', 'Gaussian OT', knn_key, 'Per-age OT']:
        parts_r = []
        parts_d = []
        for bird_id in birds:
            t2 = all_results[bird_id]['table2']
            if method_key in t2:
                fc = t2[method_key]['feature_correlations']
                rp = abs(fc['spectral_flatness']['r'])
                cd = t2[method_key]['cohens_d_resid']
                parts_r.append(f".{rp:.2f}"[1:] if rp < 1 else f"{rp:.2f}")
                parts_d.append(f"{cd:.2f}")
            else:
                parts_r.append("--")
                parts_d.append("--")

        display = method_display.get(method_key, method_key)
        row = f"    {display} & " + " & ".join(parts_r + parts_d) + " \\\\"
        print(row)

    print(r"    \bottomrule")


def print_summary_table(all_results: dict, k: int = 10):
    """Print a console-friendly summary table."""
    knn_key = f'kNN(k={k})'

    print(f"\n{'═'*110}")
    print(f"  {'Method':<22} ", end="")
    for bird_id in BIRDS:
        if bird_id in all_results:
            lbl = all_results[bird_id]['label']
            print(f"│ r(flat)_{lbl}  d_res_{lbl}  AUC_{lbl} ", end="")
    print()
    print(f"  {'─'*22} ", end="")
    for bird_id in BIRDS:
        if bird_id in all_results:
            print(f"│{'─'*32}", end="")
    print()

    for method_key in ['Displacement', 'Gaussian OT', knn_key, 'Per-age OT']:
        print(f"  {method_key:<22} ", end="")
        for bird_id in BIRDS:
            if bird_id not in all_results:
                continue
            t2 = all_results[bird_id]['table2']
            if method_key in t2:
                fc = t2[method_key]['feature_correlations']
                rp = fc['spectral_flatness']['r']
                cd = t2[method_key]['cohens_d_resid']
                auc = t2[method_key]['auc_resid']
                print(f"│ {rp:+.3f}    {cd:+.2f}    {auc:.3f} ", end="")
            else:
                print(f"│   --       --       --  ", end="")
        print()
    print(f"{'═'*110}")


def print_coupling_comparison(all_results: dict, birds: list[str], k: int = 10):
    """Print side-by-side comparison of OT vs kNN coupling for the displacement model."""
    print(f"\n{'═'*90}")
    print(f"  OT vs kNN COUPLING COMPARISON (Displacement model only)")
    print(f"{'═'*90}")
    print(f"  {'Bird':<8} {'Metric':<22} {'OT coupling':>14} {'kNN coupling':>14} {'Δ':>10}")
    print(f"  {'─'*8} {'─'*22} {'─'*14} {'─'*14} {'─'*10}")

    for bird in birds:
        key_ot = f"{bird}_ot"
        key_knn = f"{bird}_knn"
        if key_ot not in all_results or key_knn not in all_results:
            continue

        t2_ot = all_results[key_ot]['table2'].get('Displacement', {})
        t2_knn = all_results[key_knn]['table2'].get('Displacement', {})

        # Spectral flatness correlation
        r_ot = t2_ot.get('feature_correlations', {}).get('spectral_flatness', {}).get('r', float('nan'))
        r_knn = t2_knn.get('feature_correlations', {}).get('spectral_flatness', {}).get('r', float('nan'))
        print(f"  {bird:<8} {'r(spec. flatness)':<22} {r_ot:+14.3f} {r_knn:+14.3f} {r_knn - r_ot:+10.3f}")

        # Cohen's d
        d_ot = t2_ot.get('cohens_d_resid', float('nan'))
        d_knn = t2_knn.get('cohens_d_resid', float('nan'))
        print(f"  {'':8} {'d_res (song/call)':<22} {d_ot:+14.2f} {d_knn:+14.2f} {d_knn - d_ot:+10.2f}")

        # AUC
        auc_ot = t2_ot.get('auc_resid', float('nan'))
        auc_knn = t2_knn.get('auc_resid', float('nan'))
        print(f"  {'':8} {'AUC_res':<22} {auc_ot:14.3f} {auc_knn:14.3f} {auc_knn - auc_ot:+10.3f}")

        # R²(duration)
        r2_ot = t2_ot.get('r2_duration', float('nan'))
        r2_knn = t2_knn.get('r2_duration', float('nan'))
        print(f"  {'':8} {'R²(duration)':<22} {r2_ot:14.3f} {r2_knn:14.3f} {r2_knn - r2_ot:+10.3f}")

        print(f"  {'─'*8}{'─'*22}{'─'*14}{'─'*14}{'─'*10}")

    print(f"{'═'*90}")


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for Interspeech 2026 paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--bird", type=str, choices=BIRDS, default=None,
                        help="Evaluate a single bird (default: all)")
    parser.add_argument("--coupling", type=str, choices=["ot", "knn", "both"],
                        default="ot",
                        help="Which displacement model coupling to evaluate "
                             "(default: ot). Use 'both' to run OT and kNN side-by-side.")
    parser.add_argument("--n_eval", type=int, default=10_000,
                        help="Evaluation subset size (default: 10K)")
    parser.add_argument("--num_target_ages", type=int, default=7,
                        help="Number of evenly spaced target ages (default: 7)")
    parser.add_argument("--k", type=int, default=10,
                        help="k for kNN baseline (default: 10)")
    parser.add_argument("--skip_hard_ot", action="store_true",
                        help="Skip per-age OT baseline (slow)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: n_eval=2000, skip per-age OT")
    parser.add_argument("--output", type=str,
                        default="models/paper_eval_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if args.quick:
        args.n_eval = 2000
        args.skip_hard_ot = True

    birds_to_run = [args.bird] if args.bird else BIRDS
    couplings = ["ot", "knn"] if args.coupling == "both" else [args.coupling]

    all_results = {}

    t_total = time.time()
    for coupling in couplings:
        for bird in birds_to_run:
            t_bird = time.time()
            result = evaluate_bird(
                bird,
                n_eval=args.n_eval,
                num_target_ages=args.num_target_ages,
                k=args.k,
                skip_hard_ot=args.skip_hard_ot,
                coupling=coupling,
            )
            result['time_total_sec'] = time.time() - t_bird
            # Key: "R4634_ot" or "R4634_knn" when comparing, else just "R4634"
            key = f"{bird}_{coupling}" if len(couplings) > 1 else bird
            all_results[key] = result

    total_time = time.time() - t_total
    print(f"\n{'#'*70}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'#'*70}")

    # ── Console summary ──
    if len(couplings) == 1:
        print_summary_table(all_results, k=args.k)
        print_latex_tables(all_results, k=args.k)
    else:
        # Side-by-side comparison for --coupling=both
        print_coupling_comparison(all_results, birds_to_run, k=args.k)

    # ── Save JSON ──
    save_results = {}
    for result_key, r in all_results.items():
        sr = dict(r)
        clean_t2 = {}
        for method_name, mdata in sr['table2'].items():
            clean = {k: v for k, v in mdata.items()
                     if k not in ('ot_sub_idx',)}
            clean_t2[method_name] = clean
        sr['table2'] = clean_t2
        save_results[result_key] = sr

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
