"""
Nonparametric Baselines for Trajectory Variance.

Core question: the MLP is trained on OT-paired data -- can the same
OT/kNN search used during training replace the model at inference time?

Methods compared:
  1. k-NN Search (configurable k): find k nearest neighbors at each target
     age in normalized latent space, average them, compute trajectory variance.
  2. Mini-batch OT: per (source_age, target_age) pair, solve Hungarian
     assignment from eval samples to the full target-age pool.
  3. Gaussian OT (Monge map): closed-form optimal affine transport between
     per-age Gaussians. Same transform for all vocalizations at a given age.
  4. Neural Transport: the trained MLP (z_cf = z_src + f_theta(...)).

Design principles:
  - ALL methods operate in z-score normalized latent space (matching the MLP).
  - ALL use the same T evenly spaced target ages (default T=7, matching paper).
  - The full dataset is the search pool for nonparametric methods (same
    information available to the MLP during training).
  - Target ages are snapped to the nearest actual age in the data -- no
    bandwidth parameter, deterministic and parameter-free.
  - Evaluation is subsetted for tractability; the pool is always full.

Usage:
    # Default comparison (all methods, k=5) — auto-discovers flow_dir
    python baseline_comparison.py --bird R2915

    # Explicit flow_dir
    python baseline_comparison.py --bird R2915 --flow_dir models/ot_flow_R2915_20260222_110110

    # k-sweep experiment
    python baseline_comparison.py --bird R2915 --k_sweep

    # Sample-size experiment
    python baseline_comparison.py --bird R2915 --sample_size_experiment

    # Pixel-space MSE diagnostic
    python baseline_comparison.py --bird R2915 --pixel_mse

    # MLP diagnostic (what does the model learn?)
    python baseline_comparison.py ... --diagnose
"""

import argparse
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy import stats as scipy_stats
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .models.flow import load_transport_model
from .train_ae import SpectrogramAE, SpectrogramVAE


# ============================================================================
# Shared Utilities
# ============================================================================

def build_age_pools(latents, ages):
    """Group latents by age for fast lookup.

    Returns:
        pools:       dict[age -> (N_age, D) tensor]
        pool_idx:    dict[age -> array of global indices]
        unique_ages: sorted list of unique age values
    """
    ages_np = ages.numpy() if torch.is_tensor(ages) else ages
    unique = sorted(set(ages_np.tolist()))
    pools, pool_idx = {}, {}
    for a in unique:
        idx = np.where(ages_np == a)[0]
        pool_idx[a] = idx
        pools[a] = latents[idx]
    return pools, pool_idx, unique


def snap_target_ages(target_ages, unique_ages):
    """Map each target age to the nearest actual age in the data."""
    return [min(unique_ages, key=lambda a: abs(a - ta)) for ta in target_ages]


def make_target_ages(age_min, age_max, num_targets=7):
    """Create evenly spaced target ages across the developmental range."""
    return np.linspace(age_min, age_max, num_targets).tolist()


def _draw_eval_subset(data, n_eval, seed=42):
    """Draw a reproducible evaluation subset.

    Returns (sub_idx, sub_is_song, sub_dur) where sub_idx indexes into
    latents_norm and the others are aligned label/duration arrays.
    All arrays are in pipeline order (same as latents.pt).
    """
    N = len(data['latents_norm'])
    np.random.seed(seed)
    sub_idx = np.sort(np.random.choice(N, min(n_eval, N), replace=False))
    return sub_idx, data['is_song'][sub_idx], data['durations'][sub_idx]


def _strict_ot_subset(sub_idx, ages, target_ages, seed=42):
    """Build a strict one-to-one-feasible eval subset for hard OT.

    For each source-age group in sub_idx, cap group size to the minimum
    available target-age pool among requested target ages. This guarantees
    N_src_group <= N_tgt for every target age in hard OT.

    Returns:
        sub_idx_ot: filtered subset indices
        coverage: fraction retained from original sub_idx
    """
    ages_np = ages.numpy() if torch.is_tensor(ages) else ages
    _, _, unique_ages = build_age_pools(torch.empty((len(ages_np), 1)), ages)
    snapped = snap_target_ages(target_ages, unique_ages)

    # Global counts by age over full pool
    pool_counts = {a: int(np.sum(ages_np == a)) for a in unique_ages}
    min_tgt_count = min(pool_counts[a] for a in snapped)

    sub_ages = ages_np[sub_idx]
    rng = np.random.RandomState(seed)
    keep_local = []
    for src_age in sorted(set(sub_ages.tolist())):
        local_idx = np.where(sub_ages == src_age)[0]
        if len(local_idx) == 0:
            continue
        keep_n = min(len(local_idx), min_tgt_count)
        if keep_n < len(local_idx):
            chosen = np.sort(rng.choice(local_idx, size=keep_n, replace=False))
        else:
            chosen = local_idx
        keep_local.append(chosen)

    if not keep_local:
        return np.array([], dtype=int), 0.0

    keep_local = np.sort(np.concatenate(keep_local))
    sub_idx_ot = sub_idx[keep_local]
    coverage = len(sub_idx_ot) / max(1, len(sub_idx))
    return sub_idx_ot, coverage


# ============================================================================
# Data Loading
# ============================================================================

def load_data(flow_dir, bird, device, labels_dir='gold_standard_labels',
              ae_dir_override=None):
    """Load latents, ages, labels, durations, and models.

    Song/call labels come from gold-standard bout-based labels
    (pipeline order NPZ), not H5 cluster_id.

    Returns a dict with everything needed for baseline comparison.
    """
    flow_dir = Path(flow_dir)
    flow_config = json.load(open(flow_dir / 'config.json'))

    # Resolve AE directory
    if ae_dir_override:
        ae_dir = Path(ae_dir_override)
    else:
        ae_dir = Path(flow_config['ae_dir'])
    if not ae_dir.exists():
        ae_dir = Path(__file__).parent / 'models' / ae_dir.parts[-1]
    if not ae_dir.exists():
        # Last resort: find most recent ae_{bird}_* with latents.pt
        models_dir = Path(__file__).parent / 'models'
        candidates = sorted(models_dir.glob(f'ae_{bird}_*/latents.pt'))
        if candidates:
            ae_dir = candidates[-1].parent
        else:
            raise FileNotFoundError(f"Cannot find AE dir for {bird}")
    print(f"  AE dir: {ae_dir}")
    ae_config = json.load(open(ae_dir / 'config.json'))

    # Latents and z-score normalization
    ld = torch.load(ae_dir / 'latents.pt', weights_only=True)
    latents, ages, lengths = ld['z'], ld['ages'], ld['lengths']
    z_mean = torch.tensor(flow_config['z_mean']).unsqueeze(0)
    z_std  = torch.tensor(flow_config['z_std']).unsqueeze(0)
    latents_norm = (latents - z_mean) / z_std

    # Age normalization (matching train_ot_flow.py)
    age_min, age_max = flow_config['age_min'], flow_config['age_max']
    age_margin = flow_config.get('age_margin', 0.0)
    _m = age_margin * (age_max - age_min)
    norm_min, norm_max = age_min - _m, age_max + _m

    # Neural transport model
    model = load_transport_model(flow_config, device,
                                 weights_path=flow_dir / 'best.pt')

    # Autoencoder (for optional pixel-space diagnostics)
    ae_model_type = ae_config.get('model_type', 'ae')
    AEClass = SpectrogramVAE if ae_model_type == 'vae' else SpectrogramAE
    ae = AEClass(ae_config['n_mels'], ae_config['n_time'],
                 ae_config['latent_dim']).to(device)
    ae.load_state_dict(torch.load(ae_dir / 'best.pt', map_location=device,
                                  weights_only=True))
    ae.eval()

    # Gold-standard song/call labels (pipeline order, aligned with latents)
    labels_path = Path(labels_dir) / f'{bird}_song_labels_pipeline_order.npz'
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Gold-standard labels not found: {labels_path}\n"
            f"Run: python label_song_calls.py --bird {bird}")
    lbl = np.load(labels_path)
    is_song = lbl['is_song']
    assert len(is_song) == len(latents), (
        f"Label count ({len(is_song)}) != latent count ({len(latents)}). "
        f"This usually means labels and latents were built from different H5 versions. "
        f"Regenerate labels (label_song_calls.py --bird {bird}) and/or re-export latents "
        f"(train_ae.py --encode_only with the AE checkpoint trained on the current H5)."
    )

    # Durations from latents.pt lengths (in frames -> seconds)
    # Or from H5 if available. Use lengths as proxy (proportional to duration).
    durations = lengths.numpy().astype(np.float32)

    print(f"  Gold-standard labels: {labels_path}")
    print(f"  Song: {is_song.sum():,} / {len(is_song):,} "
          f"({is_song.mean()*100:.1f}%)")

    return {
        'latents': latents, 'latents_norm': latents_norm,
        'ages': ages, 'lengths': lengths,
        'z_mean': z_mean, 'z_std': z_std,
        'age_min': age_min, 'age_max': age_max,
        'norm_min': norm_min, 'norm_max': norm_max,
        'durations': durations, 'is_song': is_song,
        'model': model, 'ae': ae,
        'ae_config': ae_config, 'flow_config': flow_config,
        'device': device, 'bird': bird,
        'flow_dir': flow_dir,
    }


# ============================================================================
# Baseline 1: k-NN Search
# ============================================================================

def knn_trajectory_variance(latents_norm, ages, target_ages, k=1,
                            subset_idx=None, chunk_size=2048):
    """k-NN baseline in normalized latent space.

    For each eval sample z_i, the 'counterfactual' at target age a_t is the
    mean of its k nearest neighbors among all samples at the closest actual
    age to a_t.  Trajectory variance = sum_d Var_t[cf_i(a_t, d)].

    Args:
        latents_norm: (N, D) z-score normalized latents (full dataset = pool)
        ages:         (N,) ages in days
        target_ages:  T target ages (will be snapped to nearest actual age)
        k:            number of neighbors (k=1 is plain NN; higher k smooths)
        subset_idx:   which samples to evaluate (indices into latents_norm)
        chunk_size:   batch size for distance computation (memory control)

    Returns:
        variances: (N_sub,) numpy array
    """
    pools, _, unique_ages = build_age_pools(latents_norm, ages)
    snapped = snap_target_ages(target_ages, unique_ages)

    if subset_idx is None:
        subset_idx = np.arange(len(latents_norm))
    N_sub, D = len(subset_idx), latents_norm.shape[1]

    trajectories = torch.zeros(len(target_ages), N_sub, D)

    for t_idx, ta in enumerate(tqdm(snapped, desc=f'kNN(k={k})')):
        z_pool = pools[ta]
        k_eff = min(k, len(z_pool))

        for start in range(0, N_sub, chunk_size):
            end = min(start + chunk_size, N_sub)
            z_q = latents_norm[subset_idx[start:end]]
            dists = torch.cdist(z_q, z_pool)

            if k_eff == 1:
                trajectories[t_idx, start:end] = z_pool[dists.argmin(dim=1)]
            else:
                _, topk_idx = dists.topk(k_eff, dim=1, largest=False)
                trajectories[t_idx, start:end] = z_pool[topk_idx].mean(dim=1)

    return trajectories


# ============================================================================
# Baseline 2: Mini-batch OT (per age pair)
# ============================================================================

def ot_trajectory_variance(latents_norm, ages, target_ages, subset_idx=None,
                           max_ot_size=5000, avg_repeats=3):
    """Per-age-pair OT baseline in normalized latent space.

    For each (source_age, target_age) pair, solves Hungarian assignment
    from eval samples at source_age to the full pool at target_age.

    For large target pools (> max_ot_size), subsamples the pool multiple
    times and averages the matched targets to reduce stochastic noise.

    Args:
        latents_norm: (N, D) normalized latents (full dataset)
        ages:         (N,) ages in days
        target_ages:  T target ages
        subset_idx:   indices to evaluate
        max_ot_size:  max target pool size for exact Hungarian
        avg_repeats:  how many subsampled OT solutions to average (large pools)
    """
    pools, _, unique_ages = build_age_pools(latents_norm, ages)
    snapped = snap_target_ages(target_ages, unique_ages)
    ages_np = ages.numpy() if torch.is_tensor(ages) else ages

    if subset_idx is None:
        subset_idx = np.arange(len(latents_norm))
    N_sub, D = len(subset_idx), latents_norm.shape[1]
    sub_ages = ages_np[subset_idx]

    trajectories = torch.zeros(len(target_ages), N_sub, D)

    for t_idx, ta in enumerate(tqdm(snapped, desc='OT per-age-pair')):
        z_tgt_pool = pools[ta]
        N_tgt = len(z_tgt_pool)

        for src_age in sorted(set(sub_ages.tolist())):
            local_idx = np.where(sub_ages == src_age)[0]
            if len(local_idx) == 0:
                continue

            z_src = latents_norm[subset_idx[local_idx]]
            N_src = len(z_src)

            if src_age == ta:
                trajectories[t_idx, local_idx] = z_src
                continue

            if N_tgt <= max_ot_size:
                # Exact rectangular Hungarian (N_src x N_tgt)
                C = torch.cdist(z_src, z_tgt_pool, p=2).pow(2)
                _, col = linear_sum_assignment(C.numpy())
                trajectories[t_idx, local_idx] = z_tgt_pool[col]
            else:
                # Average over multiple subsampled OT solutions
                matched = torch.zeros(N_src, D)
                for _ in range(avg_repeats):
                    sample = np.random.choice(N_tgt, max_ot_size, replace=False)
                    z_sub = z_tgt_pool[sample]
                    C = torch.cdist(z_src, z_sub, p=2).pow(2)
                    _, col = linear_sum_assignment(C.numpy())
                    matched += z_sub[col]
                trajectories[t_idx, local_idx] = matched / avg_repeats

    return trajectories


# ============================================================================
# Baseline 2b: Soft Transport Plans
# ============================================================================

def _sinkhorn_plan(C, eps=0.1, n_iters=50):
    """Balanced Sinkhorn transport plan for a cost matrix C (n_src x n_tgt).

    Returns a dense plan P whose row/column sums approximate uniform marginals.
    """
    n_src, n_tgt = C.shape
    a = torch.full((n_src,), 1.0 / n_src, dtype=C.dtype, device=C.device)
    b = torch.full((n_tgt,), 1.0 / n_tgt, dtype=C.dtype, device=C.device)

    # Cost normalization improves numeric stability across birds/runs.
    Cn = C / (C.mean() + 1e-8)
    K = torch.exp(-Cn / max(eps, 1e-6))
    K = torch.clamp(K, min=1e-30)

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.t() @ u + 1e-12)

    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return P


def soft_transport_trajectory_variance(latents_norm, ages, target_ages,
                                       subset_idx=None,
                                       method='sinkhorn',
                                       max_ot_size=2048,
                                       chunk_size=1024,
                                       sinkhorn_eps=0.1,
                                       sinkhorn_iters=50,
                                       rowsoftmax_tau=0.1):
    """Soft transport baseline via barycentric projection.

    method='sinkhorn': balanced Sinkhorn plan per source-target age pair.
    method='rowsoftmax': per-source softmax weights over target pool.
    """
    if method not in {'sinkhorn', 'rowsoftmax'}:
        raise ValueError(f"Unknown soft transport method: {method}")

    pools, _, unique_ages = build_age_pools(latents_norm, ages)
    snapped = snap_target_ages(target_ages, unique_ages)
    ages_np = ages.numpy() if torch.is_tensor(ages) else ages

    if subset_idx is None:
        subset_idx = np.arange(len(latents_norm))
    N_sub, D = len(subset_idx), latents_norm.shape[1]
    sub_ages = ages_np[subset_idx]

    trajectories = torch.zeros(len(target_ages), N_sub, D)

    for t_idx, ta in enumerate(tqdm(snapped, desc=f'Soft-{method}')):
        z_tgt_pool_full = pools[ta]
        N_tgt_full = len(z_tgt_pool_full)

        if N_tgt_full > max_ot_size:
            sample = np.random.choice(N_tgt_full, max_ot_size, replace=False)
            z_tgt_pool = z_tgt_pool_full[sample]
        else:
            z_tgt_pool = z_tgt_pool_full

        for src_age in sorted(set(sub_ages.tolist())):
            local_idx = np.where(sub_ages == src_age)[0]
            if len(local_idx) == 0:
                continue

            z_src_all = latents_norm[subset_idx[local_idx]]

            if src_age == ta:
                trajectories[t_idx, local_idx] = z_src_all
                continue

            # Chunk source rows to keep memory bounded for large n_eval.
            for st in range(0, len(local_idx), chunk_size):
                ed = min(st + chunk_size, len(local_idx))
                z_src = z_src_all[st:ed]
                C = torch.cdist(z_src, z_tgt_pool, p=2).pow(2)

                if method == 'sinkhorn':
                    P = _sinkhorn_plan(C, eps=sinkhorn_eps, n_iters=sinkhorn_iters)
                    row_mass = P.sum(dim=1, keepdim=True)
                    z_cf = (P @ z_tgt_pool) / (row_mass + 1e-12)
                else:
                    Cn = C / (C.mean() + 1e-8)
                    W = torch.softmax(-Cn / max(rowsoftmax_tau, 1e-6), dim=1)
                    z_cf = W @ z_tgt_pool

                trajectories[t_idx, local_idx[st:ed]] = z_cf

    return trajectories


# ============================================================================
# Baseline 3: Gaussian OT (Monge Map)
# ============================================================================

def gaussian_ot_trajectory_variance(latents_norm, ages, target_ages,
                                    subset_idx=None):
    """Gaussian OT: closed-form Monge map between per-age Gaussians.

    Fits N(mu_s, Sigma_s) per source age, N(mu_t, Sigma_t) per target age.
    Optimal affine transport:
        T(z) = mu_t + A (z - mu_s)
        A = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}

    Key limitation: EVERY vocalization at the same source age gets the SAME
    affine transform -- no type-specific transport.
    """
    N, D = latents_norm.shape
    z_np   = latents_norm.numpy() if torch.is_tensor(latents_norm) else latents_norm
    ages_np = ages.numpy() if torch.is_tensor(ages) else ages

    if subset_idx is None:
        subset_idx = np.arange(N)
    N_sub = len(subset_idx)

    unique_ages = sorted(set(ages_np.tolist()))
    snapped = snap_target_ages(target_ages, unique_ages)

    # --- Fit per-age Gaussians with eigen-decomposition for sqrt/inv_sqrt ---
    stats = {}
    for a in unique_ages:
        z_a = z_np[ages_np == a]
        if len(z_a) < D + 1:
            continue
        mu  = z_a.mean(axis=0)
        cov = np.cov(z_a, rowvar=False) + 1e-5 * np.eye(D)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        sq = np.sqrt(eigvals)
        S_half  = eigvecs @ np.diag(sq)       @ eigvecs.T
        S_ihalf = eigvecs @ np.diag(1.0 / sq) @ eigvecs.T
        stats[a] = dict(mu=mu, cov=cov, S_half=S_half, S_ihalf=S_ihalf)

    # --- Precompute transport matrices for needed (src, tgt) pairs ---
    sub_ages = ages_np[subset_idx]
    transport = {}

    for a_s in tqdm(sorted(set(sub_ages.tolist())),
                    desc='Gaussian OT: transport maps'):
        if a_s not in stats:
            continue
        s = stats[a_s]
        for a_t in set(snapped):
            if (a_s, a_t) in transport or a_t not in stats:
                continue
            if a_s == a_t:
                transport[(a_s, a_t)] = (np.eye(D), np.zeros(D))
                continue
            t = stats[a_t]
            M = s['S_half'] @ t['cov'] @ s['S_half']
            eig_M, vec_M = np.linalg.eigh(M)
            M_half = vec_M @ np.diag(np.sqrt(np.maximum(eig_M, 1e-8))) @ vec_M.T
            A = s['S_ihalf'] @ M_half @ s['S_ihalf']
            b = t['mu'] - A @ s['mu']
            transport[(a_s, a_t)] = (A, b)

    # --- Apply transport ---
    trajectories = np.zeros((len(target_ages), N_sub, D), dtype=np.float32)

    for t_idx, a_t in enumerate(snapped):
        for a_s in sorted(set(sub_ages.tolist())):
            mask = (sub_ages == a_s)
            z_src = z_np[subset_idx[mask]]
            key = (a_s, a_t)
            if key in transport:
                A, b = transport[key]
                trajectories[t_idx, mask] = (z_src @ A.T + b).astype(np.float32)
            else:
                trajectories[t_idx, mask] = z_src  # identity fallback

    return torch.from_numpy(trajectories)


# ============================================================================
# Neural Transport (our model)
# ============================================================================

@torch.no_grad()
def neural_trajectory_variance(model, latents_norm, ages, norm_min, norm_max,
                               device, target_ages, subset_idx=None,
                               batch_size=512):
    """Trajectory variance from the trained MLP: z_cf = z_src + f_theta(z_src, c).

    Already operates in normalized latent space by construction.
    """
    model.eval()
    if subset_idx is None:
        subset_idx = np.arange(len(latents_norm))
    N_sub, D = len(subset_idx), latents_norm.shape[1]

    ages_norm = (ages - norm_min) / (norm_max - norm_min)
    tgt_norms = [(ta - norm_min) / (norm_max - norm_min) for ta in target_ages]

    trajectories = torch.zeros(len(target_ages), N_sub, D)

    for t_idx, ta_norm in enumerate(tgt_norms):
        for start in range(0, N_sub, batch_size):
            end = min(start + batch_size, N_sub)
            idx = subset_idx[start:end]
            z_src = latents_norm[idx].to(device)
            a_src = ages_norm[idx].to(device)
            a_tgt = torch.full((len(idx),), ta_norm, device=device)
            c = torch.stack([a_src, a_tgt], dim=1)
            trajectories[t_idx, start:end] = model.generate(z_src, c).cpu()

    return trajectories


def neural_trajectory_variance_gaprel(model, latents_norm, ages, norm_min, norm_max,
                                       device, gap_window=25.0, num_gaps=11,
                                       subset_idx=None, batch_size=512):
    """Gap-relative trajectory variance: same symmetric gap window for all samples.

    Removes position-dependent bias from absolute target ages.
    """
    model.eval()
    if subset_idx is None:
        subset_idx = np.arange(len(latents_norm))
    N_sub, D = len(subset_idx), latents_norm.shape[1]

    norm_range = norm_max - norm_min
    ages_norm = (ages - norm_min) / norm_range
    gaps_days = torch.linspace(-gap_window, gap_window, num_gaps)
    gaps_norm = gaps_days / norm_range

    trajectories = torch.zeros(num_gaps, N_sub, D)

    for g_idx, g_norm in enumerate(gaps_norm):
        for start in range(0, N_sub, batch_size):
            end = min(start + batch_size, N_sub)
            idx = subset_idx[start:end]
            z_src = latents_norm[idx].to(device)
            a_src = ages_norm[idx].to(device)
            a_tgt = (a_src + g_norm).clamp(0.0, 1.0)
            c = torch.stack([a_src, a_tgt], dim=1)
            trajectories[g_idx, start:end] = model.generate(z_src, c).cpu()

    return trajectories


# ============================================================================
# Trajectory Metrics
# ============================================================================

def trajectory_metrics(trajectories):
    """Compute variance, path length, and monotonicity from trajectories.

    Args:
        trajectories: (T, N, D) tensor of counterfactual positions

    Returns:
        dict with 'variance', 'path_length', 'monotonicity' arrays, each (N,)
    """
    if not torch.is_tensor(trajectories):
        trajectories = torch.from_numpy(trajectories)

    # Trajectory variance: sum_d Var_t[CF(a_t, d)]
    variance = trajectories.var(dim=0).sum(dim=1).numpy()

    # Path length: sum of step-wise displacements
    diffs = trajectories[1:] - trajectories[:-1]  # (T-1, N, D)
    step_lengths = diffs.norm(dim=2)               # (T-1, N)
    path_length = step_lengths.sum(dim=0).numpy()  # (N,)

    # Monotonicity: net displacement / path length
    # M=1 means straight-line trajectory; M~0 means wandering
    net_disp = (trajectories[-1] - trajectories[0]).norm(dim=1).numpy()  # (N,)
    monotonicity = net_disp / (path_length + 1e-10)  # (N,)

    return {
        'variance': variance,
        'path_length': path_length,
        'monotonicity': monotonicity,
    }


# ============================================================================
# Evaluation Metrics
# ============================================================================

def _evaluate_one_metric(values, is_song, durations, higher_for_song=True):
    """Evaluate a single metric array for song/call separation.

    Returns dict with cohens_d, AUC, partial_r, p-values (raw and duration-residualized).
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import mannwhitneyu

    var = values
    dur = durations * 1000  # seconds -> ms
    is_f = is_song.astype(float)
    v_song, v_call = var[is_song], var[~is_song]

    # --- Cohen's d (raw) ---
    n_s, n_c = len(v_song), len(v_call)
    ps = np.sqrt((v_song.var() * (n_s - 1) + v_call.var() * (n_c - 1))
                 / (n_s + n_c - 2))
    d_raw = float((v_song.mean() - v_call.mean()) / (ps + 1e-10))

    # --- Mann-Whitney U test ---
    alt = 'greater' if higher_for_song else 'less'
    try:
        stat, p_raw = mannwhitneyu(v_song, v_call, alternative=alt)
        p_raw = float(p_raw)
    except Exception:
        p_raw = 1.0

    # --- AUC (raw) ---
    try:
        auc_raw = float(roc_auc_score(is_song.astype(int), var))
    except Exception:
        auc_raw = 0.5

    # --- Duration-controlled residual ---
    slope, intercept = np.polyfit(dur, var, 1)
    var_r = var - (slope * dur + intercept)
    rs, rc = var_r[is_song], var_r[~is_song]
    ps_r = np.sqrt((rs.var() * (n_s - 1) + rc.var() * (n_c - 1))
                   / (n_s + n_c - 2))
    d_resid = float((rs.mean() - rc.mean()) / (ps_r + 1e-10))

    # --- Mann-Whitney U test on residuals ---
    try:
        stat_r, p_resid = mannwhitneyu(rs, rc, alternative=alt)
        p_resid = float(p_resid)
    except Exception:
        p_resid = 1.0

    try:
        auc_resid = float(roc_auc_score(is_song.astype(int), var_r))
    except Exception:
        auc_resid = 0.5

    # --- Partial correlation: r(is_song, V | dur) ---
    r_vd = np.corrcoef(var, dur)[0, 1]
    r_sd = np.corrcoef(is_f, dur)[0, 1]
    r_vs = np.corrcoef(var, is_f)[0, 1]
    denom = np.sqrt((1 - r_vd**2) * (1 - r_sd**2))
    partial_r = float((r_vs - r_vd * r_sd) / denom) if denom > 1e-10 else 0.0

    return {
        'cohens_d_raw': d_raw, 'cohens_d_resid': d_resid,
        'auc_raw': auc_raw, 'auc_resid': auc_resid,
        'partial_r': partial_r, 'raw_r': float(r_vs),
        'r2_duration': float(r_vd**2),
        'song_mean': float(v_song.mean()), 'call_mean': float(v_call.mean()),
        'p_value_raw': p_raw, 'p_value_resid': p_resid,
    }


def evaluate_trajectories(trajectories, is_song, durations, method_name):
    """Evaluate all trajectory metrics (variance, path length, monotonicity).

    Args:
        trajectories: (T, N, D) tensor of counterfactual positions
        is_song: (N,) boolean array
        durations: (N,) float array

    Returns dict with per-metric evaluation results.
    """
    metrics = trajectory_metrics(trajectories)

    # Variance: song expected higher (more developmental change)
    var_eval = _evaluate_one_metric(
        metrics['variance'], is_song, durations, higher_for_song=True)

    # Path length: song expected higher (longer developmental path)
    pl_eval = _evaluate_one_metric(
        metrics['path_length'], is_song, durations, higher_for_song=True)

    # Monotonicity: song expected higher (more directional change)
    mono_eval = _evaluate_one_metric(
        metrics['monotonicity'], is_song, durations, higher_for_song=True)

    result = {'method': method_name}
    for key, val in var_eval.items():
        result[f'var_{key}'] = val
    for key, val in pl_eval.items():
        result[f'pl_{key}'] = val
    for key, val in mono_eval.items():
        result[f'mono_{key}'] = val

    return result


def evaluate_variance(variances, is_song, durations, method_name):
    """Legacy wrapper: evaluate only variance (for backward compatibility).

    Accepts either a (N,) variance array or (T, N, D) trajectories tensor.
    """
    if variances.ndim >= 3:
        # Got trajectories, extract variance
        metrics = trajectory_metrics(variances)
        variances = metrics['variance']

    r = _evaluate_one_metric(variances, is_song, durations, higher_for_song=True)
    r['method'] = method_name
    return r


def print_comparison_table(results):
    """Print a formatted comparison table with all three trajectory metrics."""
    # Detect if results use the new multi-metric format (var_/pl_/mono_ prefixed keys)
    is_multi = any(k.startswith('var_') for k in results[0]) if results else False

    if is_multi:
        _print_multi_metric_table(results)
    else:
        _print_legacy_table(results)


def _print_legacy_table(results):
    """Print table for legacy single-metric (variance-only) results."""
    width = 132
    print("\n" + "=" * width)
    print(f"{'Method':<22} {'d_raw':>7} {'d_res':>7} {'p_raw':>9} {'p_res':>9} "
          f"{'AUC_r':>7} {'AUC_res':>7} {'r_part':>7} {'r_raw':>7} "
          f"{'R^2_dur':>7} {'Song_M':>8} {'Call_M':>8}")
    print("-" * width)
    for r in results:
        p_raw_str = "<1e-10" if r['p_value_raw'] < 1e-10 else f"{r['p_value_raw']:.2e}"
        p_res_str = "<1e-10" if r['p_value_resid'] < 1e-10 else f"{r['p_value_resid']:.2e}"

        print(f"{r['method']:<22} {r['cohens_d_raw']:>7.3f} "
              f"{r['cohens_d_resid']:>7.3f} {p_raw_str:>9} {p_res_str:>9} "
              f"{r['auc_raw']:>7.3f} {r['auc_resid']:>7.3f} "
              f"{r['partial_r']:>7.3f} {r['raw_r']:>7.3f} "
              f"{r['r2_duration']:>7.3f} "
              f"{r['song_mean']:>8.2f} {r['call_mean']:>8.2f}")
    print("=" * width)


def _print_multi_metric_table(results):
    """Print table for multi-metric (variance + path length + monotonicity) results."""
    metrics_info = [
        ('var',  'Traj. Variance'),
        ('pl',   'Path Length'),
        ('mono', 'Monotonicity'),
    ]

    for prefix, label in metrics_info:
        width = 120
        print(f"\n{'─'*width}")
        print(f"  {label}")
        print(f"{'─'*width}")
        print(f"  {'Method':<22} {'d_raw':>7} {'d_res':>7} {'p_raw':>9} {'p_res':>9} "
              f"{'AUC_r':>7} {'AUC_res':>7} {'r_part':>7} "
              f"{'Song_M':>8} {'Call_M':>8}")
        print(f"  {'-'*(width-2)}")
        for r in results:
            d_raw = r[f'{prefix}_cohens_d_raw']
            d_res = r[f'{prefix}_cohens_d_resid']
            p_raw = r[f'{prefix}_p_value_raw']
            p_res = r[f'{prefix}_p_value_resid']
            p_raw_str = "<1e-10" if p_raw < 1e-10 else f"{p_raw:.2e}"
            p_res_str = "<1e-10" if p_res < 1e-10 else f"{p_res:.2e}"

            print(f"  {r['method']:<22} {d_raw:>7.3f} {d_res:>7.3f} "
                  f"{p_raw_str:>9} {p_res_str:>9} "
                  f"{r[f'{prefix}_auc_raw']:>7.3f} {r[f'{prefix}_auc_resid']:>7.3f} "
                  f"{r[f'{prefix}_partial_r']:>7.3f} "
                  f"{r[f'{prefix}_song_mean']:>8.3f} {r[f'{prefix}_call_mean']:>8.3f}")
    print(f"{'═'*width}")


# ============================================================================
# Experiment: Default Comparison
# ============================================================================

def run_comparison(data, n_eval=10000, num_target_ages=7, k=5, quick=False,
                   soft_transport='none',
                   soft_max_ot_size=2048,
                   soft_chunk_size=1024,
                   sinkhorn_eps=0.1,
                   sinkhorn_iters=50,
                   rowsoftmax_tau=0.1,
                   skip_hard_ot=False):
    """Run all methods on the same evaluation subset and compare."""
    if quick:
        n_eval = 2000

    N = len(data['latents_norm'])
    target_ages = make_target_ages(data['age_min'], data['age_max'],
                                   num_target_ages)
    sub_idx, sub_is_song, sub_dur = _draw_eval_subset(data, n_eval)

    print(f"\n=== Comparison (N_eval={len(sub_idx):,}, pool={N:,}, "
          f"T={num_target_ages} target ages, k={k}) ===")

    methods = [
        (f'kNN(k={k})',
         lambda: knn_trajectory_variance(
             data['latents_norm'], data['ages'], target_ages,
             k=k, subset_idx=sub_idx)),
        ('Gaussian OT',
         lambda: gaussian_ot_trajectory_variance(
             data['latents_norm'], data['ages'], target_ages,
             subset_idx=sub_idx)),
    ]

    if soft_transport in {'sinkhorn', 'both'}:
        methods.append(
            ('Soft OT (Sinkhorn)',
             lambda: soft_transport_trajectory_variance(
                 data['latents_norm'], data['ages'], target_ages,
                 subset_idx=sub_idx, method='sinkhorn',
                 max_ot_size=soft_max_ot_size,
                 chunk_size=soft_chunk_size,
                 sinkhorn_eps=sinkhorn_eps,
                 sinkhorn_iters=sinkhorn_iters,
                 rowsoftmax_tau=rowsoftmax_tau))
        )

    if soft_transport in {'rowsoftmax', 'both'}:
        methods.append(
            ('Soft Plan (row-softmax)',
             lambda: soft_transport_trajectory_variance(
                 data['latents_norm'], data['ages'], target_ages,
                 subset_idx=sub_idx, method='rowsoftmax',
                 max_ot_size=soft_max_ot_size,
                 chunk_size=soft_chunk_size,
                 sinkhorn_eps=sinkhorn_eps,
                 sinkhorn_iters=sinkhorn_iters,
                 rowsoftmax_tau=rowsoftmax_tau))
        )

    methods.append(
        ('Neural',
         lambda: neural_trajectory_variance(
             data['model'], data['latents_norm'], data['ages'],
             data['norm_min'], data['norm_max'], data['device'],
             target_ages, subset_idx=sub_idx))
    )

    results = []

    # Hard OT reference (strict one-to-one subset) unless explicitly skipped
    if not skip_hard_ot:
        sub_idx_ot, ot_cov = _strict_ot_subset(sub_idx, data['ages'], target_ages, seed=42)
        if len(sub_idx_ot) > 0:
            print(f"\n--- OT (per-age) [strict 1:1 subset, coverage={ot_cov:.1%}] ---")
            t0 = time.time()
            traj = ot_trajectory_variance(
                data['latents_norm'], data['ages'], target_ages,
                subset_idx=sub_idx_ot)
            dt = time.time() - t0
            m = trajectory_metrics(traj)
            print(f"  Time: {dt:.1f}s, Mean var: {m['variance'].mean():.3f}")
            r = evaluate_trajectories(traj, data['is_song'][sub_idx_ot],
                                      data['durations'][sub_idx_ot], 'OT (per-age)')
            r['time_sec'] = dt
            r['ot_eval_coverage'] = float(ot_cov)
            results.append(r)
        else:
            print("\n--- OT (per-age) ---")
            print("  Skipped: strict OT subset is empty.")

    for name, func in methods:
        print(f"\n--- {name} ---")
        t0 = time.time()
        traj = func()
        dt = time.time() - t0
        m = trajectory_metrics(traj)
        print(f"  Time: {dt:.1f}s, Mean var: {m['variance'].mean():.3f}, "
              f"Mean PL: {m['path_length'].mean():.3f}, "
              f"Mean mono: {m['monotonicity'].mean():.3f}")
        r = evaluate_trajectories(traj, sub_is_song, sub_dur, name)
        r['time_sec'] = dt
        results.append(r)

        # Save exact arrays for Figure 1 plotting if it's the Neural model
        if name == 'Neural':
            out_npz = Path(data['flow_dir']) / "fig1_data.npz"
            np.savez(out_npz,
                     variances=m['variance'],
                     path_length=m['path_length'],
                     monotonicity=m['monotonicity'],
                     is_song=sub_is_song, durations=sub_dur)
            print(f"  Saved raw Neural metric vectors to {out_npz}")

    print_comparison_table(results)

    # Save results
    out = Path(data['flow_dir']) / 'baseline_comparison.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved to {out}")
    return results


# ============================================================================
# Experiment: k-Sweep
# ============================================================================

def run_k_sweep(data, k_values=None, n_eval=10000, num_target_ages=7,
                quick=False, soft_transport='none',
                soft_max_ot_size=2048,
                soft_chunk_size=1024,
                sinkhorn_eps=0.1,
                sinkhorn_iters=50,
                rowsoftmax_tau=0.1,
                skip_hard_ot=False):
    """Sweep k for kNN and compare against neural transport.

    If kNN with large k matches the MLP, the model adds only computational
    convenience.  If the MLP wins at all k, it provides statistical
    superiority via its conditional-expectation smoothing.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 25, 50, 100, 200, 500] if not quick else [1, 5, 25]
    if quick:
        n_eval = 2000

    N = len(data['latents_norm'])
    target_ages = make_target_ages(data['age_min'], data['age_max'],
                                   num_target_ages)
    sub_idx, sub_is_song, sub_dur = _draw_eval_subset(data, n_eval)

    print(f"\n=== k-Sweep (N_eval={len(sub_idx):,}, pool={N:,}, "
          f"T={num_target_ages}) ===")

    results = []

    # kNN at each k
    for k in k_values:
        print(f"\n--- kNN(k={k}) ---")
        t0 = time.time()
        traj = knn_trajectory_variance(data['latents_norm'], data['ages'],
                                       target_ages, k=k, subset_idx=sub_idx)
        dt = time.time() - t0
        m = trajectory_metrics(traj)
        print(f"  Time: {dt:.1f}s, Mean var: {m['variance'].mean():.3f}")
        r = evaluate_variance(traj, sub_is_song, sub_dur, f'kNN(k={k})')
        r['k'] = k
        r['time_sec'] = dt
        results.append(r)

    # OT baseline for reference (strict one-to-one subset) unless skipped
    if not skip_hard_ot:
        sub_idx_ot, ot_cov = _strict_ot_subset(sub_idx, data['ages'], target_ages, seed=42)
        if len(sub_idx_ot) > 0:
            print(f"\n--- OT (per-age) [strict 1:1 subset, coverage={ot_cov:.1%}] ---")
            t0 = time.time()
            traj = ot_trajectory_variance(data['latents_norm'], data['ages'],
                                          target_ages, subset_idx=sub_idx_ot)
            dt = time.time() - t0
            r = evaluate_variance(traj, data['is_song'][sub_idx_ot], data['durations'][sub_idx_ot],
                                  'OT (per-age)')
            r['k'] = None
            r['time_sec'] = dt
            r['ot_eval_coverage'] = float(ot_cov)
            results.append(r)
        else:
            print("\n--- OT (per-age) ---")
            print("  Skipped: strict OT subset is empty.")
    else:
        print("\n--- OT (per-age) ---")
        print("  Skipped by --skip_hard_ot")

    # Gaussian OT for reference
    print("\n--- Gaussian OT ---")
    t0 = time.time()
    v = gaussian_ot_trajectory_variance(data['latents_norm'], data['ages'],
                                        target_ages, subset_idx=sub_idx)
    dt = time.time() - t0
    r = evaluate_variance(v, sub_is_song, sub_dur, 'Gaussian OT')
    r['k'] = None
    r['time_sec'] = dt
    results.append(r)

    if soft_transport in {'sinkhorn', 'both'}:
        print("\n--- Soft OT (Sinkhorn) ---")
        t0 = time.time()
        v = soft_transport_trajectory_variance(
            data['latents_norm'], data['ages'], target_ages,
            subset_idx=sub_idx, method='sinkhorn',
            max_ot_size=soft_max_ot_size,
            chunk_size=soft_chunk_size,
            sinkhorn_eps=sinkhorn_eps,
            sinkhorn_iters=sinkhorn_iters,
            rowsoftmax_tau=rowsoftmax_tau)
        dt = time.time() - t0
        r = evaluate_variance(v, sub_is_song, sub_dur, 'Soft OT (Sinkhorn)')
        r['k'] = None
        r['time_sec'] = dt
        results.append(r)

    if soft_transport in {'rowsoftmax', 'both'}:
        print("\n--- Soft Plan (row-softmax) ---")
        t0 = time.time()
        v = soft_transport_trajectory_variance(
            data['latents_norm'], data['ages'], target_ages,
            subset_idx=sub_idx, method='rowsoftmax',
            max_ot_size=soft_max_ot_size,
            chunk_size=soft_chunk_size,
            sinkhorn_eps=sinkhorn_eps,
            sinkhorn_iters=sinkhorn_iters,
            rowsoftmax_tau=rowsoftmax_tau)
        dt = time.time() - t0
        r = evaluate_variance(v, sub_is_song, sub_dur, 'Soft Plan (row-softmax)')
        r['k'] = None
        r['time_sec'] = dt
        results.append(r)

    # Neural transport for reference
    print("\n--- Neural Transport ---")
    t0 = time.time()
    v = neural_trajectory_variance(
        data['model'], data['latents_norm'], data['ages'],
        data['norm_min'], data['norm_max'], data['device'],
        target_ages, subset_idx=sub_idx)
    dt = time.time() - t0
    r = evaluate_variance(v, sub_is_song, sub_dur, 'Neural')
    r['k'] = None
    r['time_sec'] = dt
    results.append(r)

    print_comparison_table(results)

    # Plot
    _plot_k_sweep(results, Path(data['flow_dir']))

    # Save
    out = Path(data['flow_dir']) / 'k_sweep_results.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved to {out}")
    return results


def _plot_k_sweep(results, output_dir):
    """Plot k-sweep: metrics vs k with baselines as horizontal lines."""
    knn  = [r for r in results if r.get('k') is not None]
    refs = [r for r in results if r.get('k') is None]
    if not knn:
        return

    ks = [r['k'] for r in knn]
    metrics = [
        ('cohens_d_resid', "Cohen's d (residual)"),
        ('auc_resid',      'AUC (residual)'),
        ('partial_r',      'Partial r(song, V|dur)'),
    ]
    ref_colors = {'Neural': 'crimson', 'OT (per-age)': 'forestgreen',
                  'Gaussian OT': 'darkorange',
                  'Soft OT (Sinkhorn)': 'purple',
                  'Soft Plan (row-softmax)': 'teal'}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, metrics):
        vals = [r[metric] for r in knn]
        ax.plot(ks, vals, 'o-', color='royalblue', label='kNN', ms=6, lw=2)

        for ref in refs:
            c = ref_colors.get(ref['method'], 'gray')
            ax.axhline(ref[metric], color=c, ls='--', lw=1.5,
                       label=f"{ref['method']} ({ref[metric]:.3f})")

        ax.set_xlabel('k (neighbors)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    plt.suptitle('k-Sweep: kNN Baseline vs Neural Transport', fontsize=13)
    plt.tight_layout()
    out = output_dir / 'k_sweep.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================================
# Experiment: Sample-Size
# ============================================================================

def run_sample_size_experiment(data, fractions=None, n_repeats=3,
                               num_target_ages=7, quick=False):
    """Test whether more data closes the gap between baselines and neural.

    Subsamples the data at different fractions.  For each subsample:
      - kNN and OT use the subsampled pool (less data to search)
      - Neural uses the full model (trained on full data)

    This tests: "With enough search data, do baselines match neural transport?"
    """
    if fractions is None:
        fractions = [0.01, 0.05, 0.25, 1.0] if quick else \
                    [0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
    if quick:
        n_repeats = 1

    N = len(data['latents_norm'])
    target_ages = make_target_ages(data['age_min'], data['age_max'],
                                   num_target_ages)
    all_results = []

    for frac in fractions:
        n_sub = max(200, int(N * frac))
        print(f"\n{'=' * 60}")
        print(f"  Fraction: {frac:.1%} ({n_sub:,} samples)")
        print(f"{'=' * 60}")

        for rep in range(n_repeats):
            np.random.seed(42 + rep)
            sub_idx = np.sort(np.random.choice(N, n_sub, replace=False))

            sub_is_song = data['is_song'][sub_idx]
            sub_dur     = data['durations'][sub_idx]
            if sub_is_song.sum() < 10 or (~sub_is_song).sum() < 10:
                continue

            # Subsampled latents and ages for baselines
            latents_sub = data['latents_norm'][sub_idx]
            ages_sub    = data['ages'][sub_idx]

            row_base = {'fraction': frac, 'n_samples': n_sub, 'repeat': rep}

            # --- kNN (k=5) on subsampled pool ---
            t0 = time.time()
            v = knn_trajectory_variance(latents_sub, ages_sub, target_ages,
                                        k=5)
            r = evaluate_variance(v, sub_is_song, sub_dur, 'kNN(k=5)')
            r.update(row_base)
            r['time_sec'] = time.time() - t0
            all_results.append(r)

            # --- OT on subsampled pool (skip if too large for tractability) ---
            if n_sub <= 30000:
                t0 = time.time()
                v = ot_trajectory_variance(latents_sub, ages_sub, target_ages)
                r = evaluate_variance(v, sub_is_song, sub_dur, 'OT')
                r.update(row_base)
                r['time_sec'] = time.time() - t0
                all_results.append(r)

            # --- Neural (full model, eval on subsampled indices) ---
            t0 = time.time()
            v = neural_trajectory_variance(
                data['model'], data['latents_norm'], data['ages'],
                data['norm_min'], data['norm_max'], data['device'],
                target_ages, subset_idx=sub_idx)
            r = evaluate_variance(v, sub_is_song, sub_dur, 'Neural')
            r.update(row_base)
            r['time_sec'] = time.time() - t0
            all_results.append(r)

            # Print summary for this repeat
            last3 = all_results[-3:] if n_sub <= 30000 else all_results[-2:]
            summary = ", ".join(f"{r['method']}={r['partial_r']:.3f}"
                                for r in last3)
            print(f"  Rep {rep+1}: partial_r: {summary}")

    _plot_sample_size(all_results, Path(data['flow_dir']))

    out = Path(data['flow_dir']) / 'sample_size_results.json'
    json.dump(all_results, open(out, 'w'), indent=2)
    print(f"\nSaved to {out}")
    return all_results


def _plot_sample_size(results, output_dir):
    """Plot sample-size vs performance for all methods."""
    methods = sorted(set(r['method'] for r in results))
    colors = {'kNN(k=5)': 'royalblue', 'OT': 'forestgreen',
              'Neural': 'crimson'}

    metrics = [
        ('cohens_d_resid', "Cohen's d (residual)"),
        ('auc_resid',      'AUC (residual)'),
        ('partial_r',      'Partial r(song, V|dur)'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, metrics):
        for method in methods:
            mr = [r for r in results if r['method'] == method]
            fracs = sorted(set(r['fraction'] for r in mr))
            means = [np.mean([r[metric] for r in mr if r['fraction'] == f])
                     for f in fracs]
            stds  = [np.std([r[metric] for r in mr if r['fraction'] == f])
                     for f in fracs]
            c = colors.get(method, 'gray')
            ax.errorbar(fracs, means, yerr=stds, fmt='o-', color=c,
                        label=method, ms=4, capsize=2, lw=1.5)
        ax.set_xlabel('Data Fraction')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Sample Size vs Performance', fontsize=13)
    plt.tight_layout()
    out = output_dir / 'sample_size_experiment.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================================
# Distributional Validation: MMD between counterfactual and real
# ============================================================================

def _mmd_rbf(X, Y, bandwidths=None):
    """Compute MMD^2 with RBF kernel between two sample sets.

    Args:
        X: (n, D) tensor
        Y: (m, D) tensor
        bandwidths: list of kernel bandwidths (auto if None)

    Returns:
        float: MMD^2 estimate
    """
    if bandwidths is None:
        # Median heuristic on pooled data
        with torch.no_grad():
            Z = torch.cat([X[:500], Y[:500]], dim=0)
            dists = torch.cdist(Z, Z)
            median_dist = dists[dists > 0].median().item()
            bandwidths = [median_dist * f for f in [0.5, 1.0, 2.0]]

    def rbf_kernel(A, B, bw):
        d2 = torch.cdist(A, B).pow(2)
        return torch.exp(-d2 / (2 * bw**2))

    mmd2 = 0.0
    for bw in bandwidths:
        Kxx = rbf_kernel(X, X, bw).mean()
        Kyy = rbf_kernel(Y, Y, bw).mean()
        Kxy = rbf_kernel(X, Y, bw).mean()
        mmd2 += float(Kxx + Kyy - 2 * Kxy)
    return mmd2 / len(bandwidths)


@torch.no_grad()
def run_distributional_validation(data, n_eval=5000, num_target_ages=7,
                                  max_mmd_samples=2000):
    """Validate counterfactual plausibility via distributional matching.

    For each target age, compares the distribution of neural counterfactuals
    (transported from other ages) against the real distribution at that age.
    Uses MMD (Maximum Mean Discrepancy) with RBF kernel.

    A low MMD means the counterfactual distribution is plausible —
    it looks like what we actually observe at that age.
    """
    model = data['model']
    latents_norm = data['latents_norm']
    ages = data['ages']
    device = data['device']
    norm_min, norm_max = data['norm_min'], data['norm_max']
    ages_norm = (ages - norm_min) / (norm_max - norm_min)

    target_ages = make_target_ages(data['age_min'], data['age_max'],
                                   num_target_ages)
    pools, _, unique_ages = build_age_pools(latents_norm, ages)
    snapped = snap_target_ages(target_ages, unique_ages)

    # Eval subset: samples NOT at the target age (so transport is non-trivial)
    np.random.seed(42)
    all_idx = np.arange(len(latents_norm))

    results = []
    ages_np = ages.numpy() if torch.is_tensor(ages) else ages

    print(f"\n=== Distributional Validation (MMD) ===")

    for ta_orig, ta in zip(target_ages, snapped):
        # Real samples at target age
        z_real = pools[ta]
        n_real = len(z_real)
        if n_real < 20:
            print(f"  Age {ta:.1f}d: too few real samples ({n_real}), skipping")
            continue

        # Source samples: not at target age
        source_mask = ages_np != ta
        source_idx = np.where(source_mask)[0]
        n_src = min(n_eval, len(source_idx))
        src_sub = np.sort(np.random.choice(source_idx, n_src, replace=False))

        # Generate counterfactuals to target age
        ta_norm = (ta - norm_min) / (norm_max - norm_min)
        z_cf_list = []
        batch_size = 512
        model.eval()
        for start in range(0, n_src, batch_size):
            end = min(start + batch_size, n_src)
            idx = src_sub[start:end]
            z_src = latents_norm[idx].to(device)
            a_src = ages_norm[idx].to(device)
            a_tgt = torch.full((len(idx),), ta_norm, device=device)
            c = torch.stack([a_src, a_tgt], dim=1)
            z_cf_list.append(model.generate(z_src, c).cpu())

        z_cf = torch.cat(z_cf_list, dim=0)

        # Subsample for MMD (O(n^2) kernel)
        n_mmd = min(max_mmd_samples, len(z_cf), n_real)
        cf_sub = z_cf[np.random.choice(len(z_cf), n_mmd, replace=False)]
        real_sub = z_real[np.random.choice(n_real, n_mmd, replace=False)]

        mmd2 = _mmd_rbf(cf_sub, real_sub)

        # Null baseline: MMD between two halves of real data
        # Use half-size samples if not enough for full n_mmd splits
        n_null = min(n_mmd, n_real // 2)
        if n_null >= 20:
            perm = np.random.permutation(n_real)
            real_a = z_real[perm[:n_null]]
            real_b = z_real[perm[n_null:2*n_null]]
            mmd2_null = _mmd_rbf(real_a, real_b)
        else:
            mmd2_null = float('nan')

        print(f"  Age {ta:.1f}d: MMD²(CF, real) = {mmd2:.6f}, "
              f"MMD²(real, real) = {mmd2_null:.6f}, "
              f"ratio = {mmd2 / (mmd2_null + 1e-10):.2f}")

        results.append({
            'target_age': float(ta),
            'mmd2_cf_real': float(mmd2),
            'mmd2_null': float(mmd2_null),
            'ratio': float(mmd2 / (mmd2_null + 1e-10)),
            'n_cf': int(n_mmd),
            'n_real': int(n_real),
        })

    # Summary
    ratios = [r['ratio'] for r in results if not np.isnan(r['mmd2_null'])]
    if ratios:
        print(f"\n  Mean MMD ratio (CF/null): {np.mean(ratios):.2f} "
              f"(1.0 = indistinguishable from real)")

    # Save
    out = Path(data['flow_dir']) / 'distributional_validation.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f"  Saved to {out}")
    return results


# ============================================================================
# Optional Diagnostic: Pixel-Space MSE
# ============================================================================

@torch.no_grad()
def run_pixel_mse(data, n_samples=2000, num_target_ages=7):
    """Pixel-space validation of the AE and transport model.

    Reports:
      1. AE/VAE reconstruction MSE (encode -> decode vs original)
      2. Neural vs kNN counterfactual pixel MSE
      3. Self-transport MSE (identity check)
    """
    model     = data['model']
    ae        = data['ae']
    ae_config = data['ae_config']
    latents_norm = data['latents_norm']
    latents   = data['latents']
    ages      = data['ages']
    lengths   = data['lengths']
    device    = data['device']
    z_mean    = data['z_mean'].to(device)
    z_std     = data['z_std'].to(device)

    data_mean = ae_config['data_mean']
    data_std  = ae_config['data_std']
    n_time    = ae_config['n_time']

    def safe_decode(z_batch):
        """Decode ensuring batch_size >= 2 for BatchNorm."""
        if z_batch.shape[0] == 1:
            z_padded = z_batch.repeat(2, 1)
            return ae.decode(z_padded.to(device)).cpu()[:1]
        return ae.decode(z_batch.to(device)).cpu()

    # ── Part 1: AE/VAE Reconstruction MSE ──────────────────────────────
    print("\n=== AE/VAE Reconstruction MSE ===")
    print(f"  Model type: {ae_config.get('model_type', 'ae').upper()}")
    print(f"  Latent dim: {ae_config['latent_dim']}, "
          f"Input: ({ae_config['n_mels']}, {n_time})")

    # Load original spectrograms for the sample
    from train_ae import load_spectrograms_h5
    bird = data['bird']
    specs_all, _, lengths_all, _, _ = load_spectrograms_h5(bird, n_time=n_time)
    specs_normed = (specs_all - data_mean) / (data_std + 1e-6)

    np.random.seed(42)
    recon_idx = np.random.choice(len(specs_normed),
                                 min(n_samples, len(specs_normed)),
                                 replace=False)

    recon_mses = []
    batch_size = 64
    ae.eval()
    with torch.no_grad():
        for b_start in range(0, len(recon_idx), batch_size):
            b_end = min(b_start + batch_size, len(recon_idx))
            b_idx = recon_idx[b_start:b_end]
            x = specs_normed[b_idx].to(device)
            out = ae(x)
            x_hat = out[0]  # works for both AE and VAE
            lens = lengths_all[b_idx].int()
            for i in range(len(b_idx)):
                sl = min(int(lens[i].item()), n_time)
                mse = F.mse_loss(x_hat[i, :, :sl], x[i, :, :sl]).item()
                recon_mses.append(mse)

    recon_mses = np.array(recon_mses)
    # Convert from normalized to original scale: MSE_pixel = MSE_norm * data_std^2
    recon_mses_pixel = recon_mses * (data_std ** 2)

    print(f"\n  Reconstruction MSE (normalized space):")
    print(f"    Mean:   {recon_mses.mean():.6f}")
    print(f"    Median: {np.median(recon_mses):.6f}")
    print(f"    Std:    {recon_mses.std():.6f}")
    print(f"  Reconstruction MSE (pixel/dB space):")
    print(f"    Mean:   {recon_mses_pixel.mean():.2f}")
    print(f"    Median: {np.median(recon_mses_pixel):.2f}")
    print(f"    RMSE:   {np.sqrt(recon_mses_pixel.mean()):.2f} dB")

    # ── Part 2: Counterfactual Pixel MSE ────────────────────────────────
    print("\n=== Counterfactual Pixel MSE ===")
    target_ages = make_target_ages(data['age_min'], data['age_max'],
                                   num_target_ages)
    _, _, unique_ages = build_age_pools(data['latents'], ages)
    norm_min, norm_max = data['norm_min'], data['norm_max']
    ages_norm = (ages - norm_min) / (norm_max - norm_min)
    ages_np = ages.numpy()

    # Pool in normalized space for kNN
    pools_norm, _, _ = build_age_pools(latents_norm, ages)

    sample_idx = recon_idx  # reuse the same sample

    results = {k: [] for k in [
        'neural_mse', 'nn_mse', 'self_transport_mse',
        'source_age', 'target_age',
        'recon_mse_norm', 'recon_mse_pixel']}

    snapped = snap_target_ages(target_ages, unique_ages)

    for _, ta_actual in zip(tqdm(target_ages, desc='Pixel MSE'), snapped):
        ta_norm = (ta_actual - norm_min) / (norm_max - norm_min)
        z_pool_norm = pools_norm[ta_actual]

        batch_size = 32
        for b_start in range(0, len(sample_idx), batch_size):
            b_end = min(b_start + batch_size, len(sample_idx))
            b_idx = sample_idx[b_start:b_end]
            B = len(b_idx)

            z_src_norm = latents_norm[b_idx].to(device)
            src_lens = lengths[b_idx].int().numpy()

            # kNN in normalized space
            dists = torch.cdist(latents_norm[b_idx], z_pool_norm)
            nn_idx = dists.argmin(dim=1)
            z_nn_norm = z_pool_norm[nn_idx]

            # Neural transport
            a_src = ages_norm[b_idx].to(device)
            a_tgt = torch.full((B,), ta_norm, device=device)
            c = torch.stack([a_src, a_tgt], dim=1)
            z_cf_norm = model.generate(z_src_norm, c)

            # Unnormalize for decoding
            z_src_raw = z_src_norm * z_std + z_mean
            z_nn_raw  = z_nn_norm.to(device) * z_std + z_mean
            z_cf_raw  = z_cf_norm * z_std + z_mean

            spec_src    = safe_decode(z_src_raw.cpu()) * data_std + data_mean
            spec_nn     = safe_decode(z_nn_raw.cpu())  * data_std + data_mean
            spec_neural = safe_decode(z_cf_raw.cpu())  * data_std + data_mean

            for i in range(B):
                sl = min(int(src_lens[i]), n_time)
                s = spec_src[i, :, :sl]
                results['neural_mse'].append(
                    F.mse_loss(spec_neural[i, :, :sl], s).item())
                results['nn_mse'].append(
                    F.mse_loss(spec_nn[i, :, :sl], s).item())
                results['source_age'].append(float(ages_np[b_idx[i]]))
                results['target_age'].append(float(ta_actual))

                # Self-transport check
                if abs(ages_np[b_idx[i]] - ta_actual) < 1:
                    c_self = torch.stack([a_src[i:i+1], a_src[i:i+1]], dim=1)
                    z_self = model.generate(z_src_norm[i:i+1], c_self)
                    # Latent-space self-transport error
                    results.setdefault('self_transport_latent_mse', []).append(
                        F.mse_loss(z_self, z_src_norm[i:i+1]).item())
                    z_self_raw = z_self * z_std + z_mean
                    spec_self = safe_decode(z_self_raw.cpu()) * data_std + data_mean
                    results['self_transport_mse'].append(
                        F.mse_loss(spec_self[0, :, :sl], s).item())
                    # Also: pure decode-roundtrip (no flow) for comparison
                    spec_roundtrip = safe_decode(z_src_raw[i:i+1].cpu()) * data_std + data_mean
                    results.setdefault('roundtrip_mse', []).append(
                        F.mse_loss(spec_roundtrip[0, :, :sl], s).item())

    # Store reconstruction stats in results
    results['recon_mse_norm'] = recon_mses.tolist()
    results['recon_mse_pixel'] = recon_mses_pixel.tolist()

    # ── Summary ─────────────────────────────────────────────────────────
    neural_mse = np.array(results['neural_mse'])
    nn_mse = np.array(results['nn_mse'])
    self_mse = np.array(results['self_transport_mse'])
    age_gaps = np.abs(np.array(results['target_age']) - np.array(results['source_age']))

    print(f"\n{'═'*65}")
    print(f"  PIXEL-SPACE VALIDATION SUMMARY")
    print(f"{'═'*65}")
    print(f"  AE/VAE Reconstruction:")
    print(f"    MSE (norm):   {recon_mses.mean():.6f} ± {recon_mses.std():.6f}")
    print(f"    RMSE (dB):    {np.sqrt(recon_mses_pixel.mean()):.2f}")
    roundtrip_mse = np.array(results.get('roundtrip_mse', []))
    self_latent = np.array(results.get('self_transport_latent_mse', []))
    print(f"  Self-Transport (identity check, n={len(self_mse)}):")
    if len(self_mse) > 0:
        print(f"    Decode roundtrip MSE:    {roundtrip_mse.mean():.4f}  (AE only, no flow)")
        print(f"    Self-transport MSE:      {self_mse.mean():.4f}  (AE + flow at zero gap)")
        print(f"    Flow-only contribution:  {self_mse.mean() - roundtrip_mse.mean():.4f}  (difference)")
        print(f"    Latent-space self MSE:   {self_latent.mean():.6f}  (z_cf vs z_src in norm space)")
    else:
        print(f"    (no same-age pairs in sample)")
    print(f"  Counterfactual (Neural, n={len(neural_mse)}):")
    print(f"    Mean MSE:     {neural_mse.mean():.4f}")
    print(f"    Median MSE:   {np.median(neural_mse):.4f}")
    print(f"  Counterfactual (kNN, n={len(nn_mse)}):")
    print(f"    Mean MSE:     {nn_mse.mean():.4f}")
    print(f"    Median MSE:   {np.median(nn_mse):.4f}")
    print(f"  Neural/kNN ratio: {neural_mse.mean() / (nn_mse.mean() + 1e-10):.3f}")
    # MSE by age gap bucket
    print(f"\n  MSE by age gap (days):")
    print(f"  {'Gap':>8s}  {'Neural':>10s}  {'kNN':>10s}  {'N':>6s}")
    for lo, hi in [(0,10),(10,20),(20,30),(30,50),(50,100)]:
        mask = (age_gaps >= lo) & (age_gaps < hi)
        if mask.sum() > 0:
            print(f"  {lo:3d}-{hi:<3d}   {neural_mse[mask].mean():10.4f}  "
                  f"{nn_mse[mask].mean():10.4f}  {mask.sum():6d}")
    print(f"{'═'*65}")

    # Plot
    _plot_pixel_mse(results, Path(data['flow_dir']))

    out = Path(data['flow_dir']) / 'pixel_mse_results.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved to {out}")
    return results


def _plot_pixel_mse(results, output_dir):
    """Plot pixel MSE: reconstruction + Neural vs kNN across age gaps."""
    ages_src = np.array(results['source_age'])
    ages_tgt = np.array(results['target_age'])
    age_gaps = np.abs(ages_tgt - ages_src)
    neural   = np.array(results['neural_mse'])
    nn       = np.array(results['nn_mse'])
    recon    = np.array(results.get('recon_mse_norm', []))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    # Panel A: AE reconstruction MSE histogram
    ax = axes[0]
    if len(recon) > 0:
        ax.hist(recon, bins=50, color='forestgreen', alpha=0.7, edgecolor='none')
        ax.axvline(recon.mean(), color='k', ls='--', lw=1.5,
                   label=f'mean={recon.mean():.5f}')
        ax.set_title('AE/VAE Reconstruction MSE')
        ax.legend(fontsize=8)
    else:
        ax.set_title('AE/VAE Recon (no data)')
    ax.set_xlabel('MSE (normalized)')
    ax.set_ylabel('Count')

    # Panel B: MSE vs age gap
    ax = axes[1]
    bins = np.arange(0, 65, 5)
    for mse, label, c in [(neural, 'Neural', 'crimson'), (nn, 'kNN', 'royalblue')]:
        means, ses, centers = [], [], []
        for i in range(len(bins) - 1):
            mask = (age_gaps >= bins[i]) & (age_gaps < bins[i + 1])
            if mask.sum() > 5:
                means.append(mse[mask].mean())
                ses.append(mse[mask].std() / np.sqrt(mask.sum()))
                centers.append((bins[i] + bins[i + 1]) / 2)
        ax.errorbar(centers, means, yerr=ses, fmt='o-', label=label, color=c,
                    ms=4, capsize=2)
    ax.set_xlabel('Age Gap (days)')
    ax.set_ylabel('Pixel MSE')
    ax.set_title('CF Pixel MSE vs Age Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Scatter neural vs kNN
    ax = axes[2]
    ax.scatter(nn, neural, alpha=0.05, s=2, c='gray')
    lim = max(nn.max(), neural.max())
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
    ax.set_xlabel('kNN Pixel MSE')
    ax.set_ylabel('Neural Pixel MSE')
    ax.set_title('Neural vs kNN')
    ax.set_aspect('equal')

    # Panel D: Self-transport histogram
    ax = axes[3]
    self_mse = np.array(results.get('self_transport_mse', []))
    if len(self_mse) > 0:
        ax.hist(self_mse, bins=50, color='teal', alpha=0.7)
        ax.set_title(f'Self-Transport MSE (mean={self_mse.mean():.5f})')
    else:
        ax.set_title('Self-Transport (no data)')
    ax.set_xlabel('Pixel MSE')
    ax.set_ylabel('Count')

    plt.suptitle('Pixel-Space Validation', fontsize=13)
    plt.tight_layout()
    out = output_dir / 'pixel_mse_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================================
# Optional Diagnostic: What Does the MLP Learn?
# ============================================================================

@torch.no_grad()
def run_mlp_diagnostic(data, n_samples=5000):
    """Analyze whether the MLP produces type-specific transport.

    Key questions:
      1. Does displacement magnitude differ for song vs call?
      2. Are displacement directions consistent within vs across types?
      3. Is self-transport (same age) near zero?
    """
    model = data['model']
    latents_norm = data['latents_norm']
    ages  = data['ages']
    is_song   = data['is_song']
    device    = data['device']
    norm_min, norm_max = data['norm_min'], data['norm_max']
    age_min, age_max   = data['age_min'], data['age_max']
    ages_norm = (ages - norm_min) / (norm_max - norm_min)

    np.random.seed(42)
    sub_idx = np.sort(np.random.choice(len(latents_norm),
                                      min(n_samples, len(latents_norm)),
                                      replace=False))
    sub_is_song = is_song[sub_idx]

    # --- Transport everything to age_max ---
    ta_norm = (age_max - norm_min) / (norm_max - norm_min)
    z_src = latents_norm[sub_idx].to(device)
    a_src = ages_norm[sub_idx].to(device)
    a_tgt = torch.full((len(sub_idx),), ta_norm, device=device)
    c = torch.stack([a_src, a_tgt], dim=1)

    delta = model(z_src, c).cpu()
    delta_mag = delta.norm(dim=1).numpy()

    mag_song = delta_mag[sub_is_song]
    mag_call = delta_mag[~sub_is_song]

    # --- Direction consistency ---
    delta_unit = F.normalize(delta, dim=1)
    song_mask = torch.tensor(sub_is_song)
    call_mask = ~song_mask

    def pairwise_cos(vecs, n=500):
        if len(vecs) < 2:
            return 0.0
        idx = np.random.choice(len(vecs), (min(n, len(vecs)), 2), replace=True)
        return F.cosine_similarity(vecs[idx[:, 0]], vecs[idx[:, 1]],
                                   dim=1).mean().item()

    within_song  = pairwise_cos(delta_unit[song_mask])
    within_call  = pairwise_cos(delta_unit[call_mask])
    n_cross = min(500, song_mask.sum().item(), call_mask.sum().item())
    cross = F.cosine_similarity(
        delta_unit[song_mask][:n_cross],
        delta_unit[call_mask][:n_cross], dim=1).mean().item() \
        if n_cross > 0 else 0.0

    # --- Displacement magnitude vs target age ---
    test_ages = np.linspace(age_min, age_max, 6)
    mag_by_age = {'song': [], 'call': []}
    for ta in test_ages:
        ta_n = (ta - norm_min) / (norm_max - norm_min)
        a_t = torch.full((len(sub_idx),), ta_n, device=device)
        c_t = torch.stack([a_src, a_t], dim=1)
        d = model(z_src, c_t).cpu().norm(dim=1).numpy()
        mag_by_age['song'].append(d[sub_is_song].mean())
        mag_by_age['call'].append(d[~sub_is_song].mean())

    # --- Self-transport check ---
    self_mags = []
    ages_np = ages.numpy()
    for ta in np.linspace(age_min, age_max, 10):
        ta_n = (ta - norm_min) / (norm_max - norm_min)
        mask = torch.abs(ages[sub_idx].float() - ta) < 1.0
        if mask.sum() < 10:
            continue
        z_at = latents_norm[sub_idx][mask].to(device)
        a_self = torch.full((mask.sum(),), ta_n, device=device)
        c_self = torch.stack([a_self, a_self], dim=1)
        d_self = model(z_at, c_self).cpu().norm(dim=1)
        self_mags.append((float(ta), d_self.mean().item(), d_self.std().item()))

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # A: Displacement magnitude histogram
    ax = axes[0, 0]
    ax.hist(mag_song, bins=50, alpha=0.7, color='crimson', density=True,
            label=f'Song ({mag_song.mean():.2f})')
    ax.hist(mag_call, bins=50, alpha=0.5, color='royalblue', density=True,
            label=f'Call ({mag_call.mean():.2f})')
    ax.set_xlabel('||delta||')
    ax.set_title(f'Displacement to age {age_max:.0f}d')
    ax.legend(fontsize=8)

    # B: Displacement vs target age
    ax = axes[0, 1]
    ax.plot(test_ages, mag_by_age['song'], 'o-', color='crimson', label='Song')
    ax.plot(test_ages, mag_by_age['call'], 's--', color='royalblue', label='Call')
    ax.set_xlabel('Target Age')
    ax.set_ylabel('Mean ||delta||')
    ax.set_title('Displacement vs Target Age')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # C: Direction consistency
    ax = axes[0, 2]
    cats = ['Song\n(within)', 'Call\n(within)', 'Cross']
    vals = [within_song, within_call, cross]
    bars = ax.bar(cats, vals, color=['crimson', 'royalblue', 'purple'],
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Displacement Direction Consistency')
    ax.set_ylim([0, 1])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.3f}', ha='center', fontsize=9)

    # D: Self-transport
    ax = axes[1, 0]
    if self_mags:
        sa, sm, ss = zip(*self_mags)
        ax.errorbar(sa, sm, yerr=ss, fmt='o-', color='teal', capsize=2)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Age')
    ax.set_ylabel('Self-transport ||delta||')
    ax.set_title('Self-Transport (should be ~0)')
    ax.grid(True, alpha=0.3)

    # E: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    txt = (
        f"Displacement to age {age_max:.0f}d:\n"
        f"  Song ||delta||: {mag_song.mean():.2f} +/- {mag_song.std():.2f}\n"
        f"  Call ||delta||: {mag_call.mean():.2f} +/- {mag_call.std():.2f}\n"
        f"  Ratio: {mag_song.mean() / (mag_call.mean() + 1e-10):.2f}x\n\n"
        f"Direction consistency (cos sim):\n"
        f"  Within song: {within_song:.3f}\n"
        f"  Within call: {within_call:.3f}\n"
        f"  Cross:       {cross:.3f}\n\n"
        f"Self-transport ||delta||:\n"
        f"  Mean: {np.mean([s[1] for s in self_mags]):.4f}"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    axes[1, 2].axis('off')

    plt.suptitle('MLP Diagnostic: What Does the Transport Model Learn?',
                 fontsize=13)
    plt.tight_layout()
    out = Path(data['flow_dir']) / 'mlp_diagnostic.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    return {
        'song_displacement_mean': float(mag_song.mean()),
        'call_displacement_mean': float(mag_call.mean()),
        'displacement_ratio': float(mag_song.mean() / (mag_call.mean() + 1e-10)),
        'within_song_consistency': within_song,
        'within_call_consistency': within_call,
        'cross_consistency': cross,
    }


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Nonparametric Baselines for Trajectory Variance',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required
    parser.add_argument('--bird', type=str, required=True,
                        help='Bird ID (e.g. R2915). Gold-standard labels must exist.')
    parser.add_argument('--flow_dir', type=str, default=None,
                        help='OT flow model dir. Auto-detected if not provided.')
    parser.add_argument('--labels_dir', type=str, default='gold_standard_labels',
                        help='Directory with song labels (default: gold_standard_labels/)')
    parser.add_argument('--ae_dir', type=str, default=None,
                        help='Override AE directory (default: from flow config)')

    # Experiment selection (mutually exclusive modes)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--k_sweep', action='store_true',
                      help='Sweep k for kNN baseline (k=1..50)')
    mode.add_argument('--sample_size_experiment', action='store_true',
                      help='Subsample data at different fractions')
    mode.add_argument('--pixel_mse', action='store_true',
                      help='Pixel-space MSE diagnostic')
    mode.add_argument('--diagnose', action='store_true',
                      help='MLP diagnostic (type-specific transport)')
    mode.add_argument('--dist_validation', action='store_true',
                      help='Distributional validation (MMD: CF vs real)')

    # Settings
    parser.add_argument('--quick', action='store_true',
                        help='Smaller subsets for fast testing')
    parser.add_argument('--n_eval', type=int, default=10000,
                        help='Evaluation subset size (default: 10K)')
    parser.add_argument('--k', type=int, default=5,
                        help='k for kNN in default comparison (default: 5)')
    parser.add_argument('--num_target_ages', type=int, default=7,
                        help='Number of evenly spaced target ages (default: 7)')
    parser.add_argument('--soft_transport', type=str,
                        choices=['none', 'sinkhorn', 'rowsoftmax', 'both'],
                        default='none',
                        help='Enable soft transport baseline(s)')
    parser.add_argument('--soft_max_ot_size', type=int, default=2048,
                        help='Max target pool size for soft transport (default: 2048)')
    parser.add_argument('--soft_chunk_size', type=int, default=1024,
                        help='Source chunk size for soft transport (default: 1024)')
    parser.add_argument('--sinkhorn_eps', type=float, default=0.1,
                        help='Sinkhorn entropic regularization epsilon')
    parser.add_argument('--sinkhorn_iters', type=int, default=50,
                        help='Sinkhorn iterations (default: 50)')
    parser.add_argument('--rowsoftmax_tau', type=float, default=0.1,
                        help='Temperature for row-softmax transport')
    parser.add_argument('--skip_hard_ot', action='store_true',
                        help='Skip hard OT baseline (kNN/soft OT/Gaussian/Neural only)')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Auto-discover flow_dir if not provided
    flow_dir = args.flow_dir
    if flow_dir is None:
        models_dir = Path(__file__).parent / 'models'
        candidates = sorted(models_dir.glob(f'ot_flow_{args.bird}_*'))
        if not candidates:
            raise FileNotFoundError(
                f"No ot_flow_{args.bird}_* found in {models_dir}. "
                f"Provide --flow_dir explicitly.")
        flow_dir = str(max(candidates, key=lambda p: p.stat().st_mtime))
        print(f"  Auto-detected flow_dir: {flow_dir}")

    # Load data
    print("\n=== Loading Data ===")
    data = load_data(flow_dir, args.bird, device, labels_dir=args.labels_dir,
                     ae_dir_override=args.ae_dir)
    data['flow_dir'] = flow_dir
    N = len(data['latents_norm'])
    print(f"  {N:,} samples, {data['is_song'].sum():,} song, "
          f"{(~data['is_song']).sum():,} call")

    # Dispatch
    if args.k_sweep:
        run_k_sweep(
            data, n_eval=args.n_eval,
            num_target_ages=args.num_target_ages, quick=args.quick,
            soft_transport=args.soft_transport,
            soft_max_ot_size=args.soft_max_ot_size,
            soft_chunk_size=args.soft_chunk_size,
            sinkhorn_eps=args.sinkhorn_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            rowsoftmax_tau=args.rowsoftmax_tau,
            skip_hard_ot=args.skip_hard_ot,
        )

    elif args.sample_size_experiment:
        run_sample_size_experiment(data, num_target_ages=args.num_target_ages,
                                  quick=args.quick)

    elif args.pixel_mse:
        n_px = 500 if args.quick else 2000
        run_pixel_mse(data, n_samples=n_px,
                      num_target_ages=args.num_target_ages)

    elif args.diagnose:
        n_diag = 2000 if args.quick else 5000
        results = run_mlp_diagnostic(data, n_samples=n_diag)
        print(json.dumps(results, indent=2))

    elif args.dist_validation:
        n_dv = 2000 if args.quick else 5000
        run_distributional_validation(data, n_eval=n_dv,
                                      num_target_ages=args.num_target_ages)

    else:
        run_comparison(
            data, n_eval=args.n_eval,
            num_target_ages=args.num_target_ages,
            k=args.k, quick=args.quick,
            soft_transport=args.soft_transport,
            soft_max_ot_size=args.soft_max_ot_size,
            soft_chunk_size=args.soft_chunk_size,
            sinkhorn_eps=args.sinkhorn_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            rowsoftmax_tau=args.rowsoftmax_tau,
            skip_hard_ot=args.skip_hard_ot,
        )


if __name__ == '__main__':
    main()
