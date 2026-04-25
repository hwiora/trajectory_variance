"""
Data-to-Data Transport with OT or k-NN Coupling.

Two architectures (--arch):
  flow   : Flow matching — learns velocity v(z_t, t | ages), ODE integration.
           Suffers from velocity averaging when trajectories cross at intermediate
           points z_t (vm_pred/vm_tgt ≈ 0.5 with diverse k-NN couplings).
  direct : Direct displacement — learns delta = f(z_src | ages), single pass.
           No time variable, no ODE, no crossing trajectories. Preferred.

Coupling strategies (--coupling):
  ot  : Mini-batch OT (Hungarian) — globally optimal within batch.
  knn : Pre-computed k-NN — each source pairs with one of its k nearest
        neighbors (cross-age). Purely local, never crosses type boundaries.

Usage:
    python Counterfactual_generation/train_ot_flow.py \
        --ae_dir Counterfactual_generation/models/ae_R5018_20260211_161856 \
        --arch direct --coupling knn --knn_k 10 --min_age_gap 5
"""

import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .train_ae import SpectrogramAE, SpectrogramVAE
from .models.flow import (
    FlowMLP, TransportMLP, TransportMLPv2, HeteroscedasticTransportMLP,
)


# ========================== OT Coupling ==========================

def ot_coupling(z_src, z_tgt):
    """Mini-batch OT coupling via Hungarian algorithm.

    Finds the permutation of z_tgt that minimizes total squared L2 distance
    to z_src. This implicitly pairs similar syllables across ages.

    Args:
        z_src: (N, D) source latents
        z_tgt: (N, D) target latents

    Returns:
        col_ind: (N,) permutation indices for z_tgt
    """
    with torch.no_grad():
        C = torch.cdist(z_src, z_tgt, p=2).pow(2)
    _, col_ind = linear_sum_assignment(C.cpu().numpy())
    return torch.tensor(col_ind, dtype=torch.long)


# ========================== k-NN Coupling ==========================

def knn_coupling(z_src, z_tgt, k, ages_src=None, ages_tgt=None,
                 min_age_gap=0):
    """Mini-batch k-NN coupling: for each source, pick a random neighbor
    from its k nearest targets.

    Mirrors ot_coupling() in interface — operates on the current batch only,
    not a precomputed global index.  This makes kNN and OT couplings
    structurally comparable (both are mini-batch stochastic).

    Args:
        z_src: (B, D) source latents
        z_tgt: (B, D) candidate target latents
        k: number of nearest neighbors to consider
        ages_src: (B,) source ages (optional, for age-gap filtering)
        ages_tgt: (B,) target ages (optional, for age-gap filtering)
        min_age_gap: minimum age difference in normalized units (default 0)

    Returns:
        col_ind: (B,) indices into z_tgt — one per source
    """
    with torch.no_grad():
        dists = torch.cdist(z_src, z_tgt, p=2).pow(2)  # (B, B)

        # Exclude self-matches (diagonal)
        B = z_src.shape[0]
        dists[torch.arange(B), torch.arange(B)] = float('inf')

        # Filter by minimum age gap if provided
        if min_age_gap > 0 and ages_src is not None and ages_tgt is not None:
            age_diffs = (ages_src.unsqueeze(1) - ages_tgt.unsqueeze(0)).abs()
            dists[age_diffs < min_age_gap] = float('inf')

        # Clamp k to available valid targets
        valid_per_row = (dists < float('inf')).sum(dim=1).min().item()
        effective_k = max(1, min(k, int(valid_per_row)))

        # Top-k nearest for each source
        _, topk_idx = dists.topk(effective_k, dim=1, largest=False)  # (B, k)

        # Randomly pick one of the k neighbors per source
        random_k = torch.randint(0, effective_k, (B,))
        col_ind = topk_idx[torch.arange(B), random_k]

    return col_ind


# Legacy global kNN — kept for backward compatibility / analysis scripts
@torch.no_grad()
def precompute_knn(latents_norm, k, device, ages=None, min_age_gap=0,
                   chunk_size=2048):
    """Pre-compute k nearest neighbors for all samples (GPU-accelerated).

    NOTE: For training, prefer knn_coupling() (mini-batch) instead.
    This function is retained for baseline_comparison.py and analysis scripts.

    Uses chunked torch.cdist on GPU — fast and no extra dependencies.
    For 181K x 128 with chunk_size=2048: ~1 min on GPU.
    """
    z = latents_norm.to(device)
    N = len(z)
    nn_idx = torch.zeros(N, k, dtype=torch.long)

    ages_device = None
    if min_age_gap > 0 and ages is not None:
        ages_t = ages.float() if torch.is_tensor(ages) else torch.tensor(ages, dtype=torch.float32)
        ages_device = ages_t.to(device)

    for start in tqdm(range(0, N, chunk_size), desc='  k-NN precompute'):
        end = min(start + chunk_size, N)
        dists = torch.cdist(z[start:end], z)  # (chunk, N)
        dists[torch.arange(end - start, device=device),
              torch.arange(start, end, device=device)] = float('inf')
        if ages_device is not None:
            chunk_ages = ages_device[start:end].unsqueeze(1)
            age_diffs = (chunk_ages - ages_device.unsqueeze(0)).abs()
            dists[age_diffs < min_age_gap] = float('inf')
            del age_diffs
        _, topk = dists.topk(k, dim=1, largest=False)
        nn_idx[start:end] = topk.cpu()

    return nn_idx.numpy()


def knn_sample_targets(idx_src, nn_indices, rng):
    """For each source index, randomly pick one of its k pre-computed neighbors.

    NOTE: Legacy function for global precomputed kNN. For training, use
    knn_coupling() instead.

    Args:
        idx_src: (B,) source indices (numpy or tensor)
        nn_indices: (N, k) pre-computed neighbor indices
        rng: numpy RandomState

    Returns:
        idx_tgt: (B,) target indices as LongTensor
    """
    src = idx_src.numpy() if torch.is_tensor(idx_src) else idx_src
    k = nn_indices.shape[1]
    random_k = rng.randint(0, k, size=len(src))
    return torch.tensor(nn_indices[src, random_k], dtype=torch.long)


def report_knn_pairing_stats(nn_indices, ages, age_min, age_max, k=None):
    """Print statistics about the k-NN pairings to help tune k."""
    ages_np = ages.numpy() if torch.is_tensor(ages) else ages
    N = len(ages_np)

    # Sample 10K pairs to estimate statistics
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(N, min(10000, N), replace=False)
    random_k = rng.randint(0, nn_indices.shape[1], size=len(sample_idx))
    tgt_idx = nn_indices[sample_idx, random_k]

    age_diffs = np.abs(ages_np[sample_idx] - ages_np[tgt_idx])
    same_age = (age_diffs == 0).mean() * 100
    within_5d = (age_diffs <= 5).mean() * 100
    within_10d = (age_diffs <= 10).mean() * 100

    print(f'  k-NN pairing stats (k={nn_indices.shape[1]}):')
    print(f'    Same age:  {same_age:.1f}%')
    print(f'    Within 5d: {within_5d:.1f}%')
    print(f'    Within 10d: {within_10d:.1f}%')
    print(f'    Age gap: mean={age_diffs.mean():.1f}d, '
          f'median={np.median(age_diffs):.1f}d, max={age_diffs.max():.0f}d')


# ========================== Helpers ==========================

def crop_to_length(spec, length, n_time):
    if length >= n_time:
        return spec
    return spec[:, :length]


# ========================== Visualization ==========================

@torch.no_grad()
def visualize(model, ae, latents_norm, ages, lengths, z_mean, z_std,
              age_min, age_max, ae_config, device, output_path, epoch,
              num_samples=4, num_steps=50, age_margin=0.0):
    """Visualize data-to-data counterfactuals (no inversion needed)."""
    model.eval(); ae.eval()
    n_time = ae_config['n_time']
    data_mean, data_std = ae_config['data_mean'], ae_config['data_std']

    # Age normalization with margin
    _m = age_margin * (age_max - age_min)
    norm_min, norm_max = age_min - _m, age_max + _m

    target_ages = np.linspace(age_min, age_max, 7)

    # Pick source samples at diverse ages
    unique_ages = sorted(torch.unique(ages).tolist())
    step_a = max(1, len(unique_ages) // num_samples)
    viz_ages = [unique_ages[i * step_a] for i in range(min(num_samples, len(unique_ages)))]

    sources = []
    for age in viz_ages:
        idxs = torch.where(ages == age)[0]
        good = [i.item() for i in idxs if lengths[i].item() >= 40]
        if good:
            sources.append({'idx': random.choice(good), 'age': age})
    if not sources:
        return

    n_cols = len(target_ages) + 1
    n_src = len(sources)

    all_specs = []
    all_lengths = []

    for src in sources:
        z_src = latents_norm[src['idx']].unsqueeze(0).to(device)
        sl = int(lengths[src['idx']].item())
        all_lengths.append(sl)
        age_src_norm = (src['age'] - norm_min) / (norm_max - norm_min)

        # Original reconstruction
        recon = ae.decode(z_src * z_std + z_mean).squeeze(0).cpu() * data_std + data_mean
        rc = crop_to_length(recon, sl, n_time)

        # Direct counterfactuals at each target age
        cfs = []
        for ta in target_ages:
            age_tgt_norm = (ta - norm_min) / (norm_max - norm_min)
            c = torch.tensor([[age_src_norm, age_tgt_norm]], device=device, dtype=torch.float32)
            z_cf = model.generate(z_src, c, steps=num_steps, cfg_scale=1.0, solver='heun')
            spec = ae.decode(z_cf * z_std + z_mean).squeeze(0).cpu() * data_std + data_mean
            cfs.append(crop_to_length(spec, sl, n_time))

        all_specs.append({'orig': rc, 'cfs': cfs})

    # Plot with proportional widths
    max_len = max(all_lengths)
    margin_left, margin_right = 0.02, 0.02
    margin_top, margin_bottom = 0.05, 0.02
    row_gap, col_gap_frac = 0.04, 0.008
    available_width = 1.0 - margin_left - margin_right
    available_height = 1.0 - margin_top - margin_bottom
    row_height = (available_height - row_gap * (n_src - 1)) / n_src

    fig = plt.figure(figsize=(24, 2.5 * n_src + 0.5))
    for i, src in enumerate(sources):
        sl = all_lengths[i]
        row_w = (sl / max_len) * available_width
        cell_w = (row_w - col_gap_frac * (n_cols - 1)) / n_cols
        y_bot = 1.0 - margin_top - (i + 1) * row_height - i * row_gap

        all_in_row = [all_specs[i]['orig']] + all_specs[i]['cfs']
        vmin = min(s.min().item() for s in all_in_row)
        vmax = max(s.max().item() for s in all_in_row)

        for j in range(n_cols):
            x_left = margin_left + j * (cell_w + col_gap_frac)
            ax = fig.add_axes([x_left, y_bot, cell_w, row_height])
            if j == 0:
                title = f"Orig {src['age']:.0f}d (L={sl})"
            else:
                ta = target_ages[j - 1]
                d = ">" if ta >= src['age'] else "<"
                title = f"{d}{ta:.0f}d"
            ax.imshow(all_in_row[j].numpy(), aspect='auto', origin='lower',
                      cmap='magma', vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"OT Data-to-Data Flow — Epoch {epoch+1}", fontsize=12, y=0.99)
    out = Path(output_path) / f'viz_epoch_{epoch+1:03d}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


# ========================== Trajectory Variance Analysis ==========================

@torch.no_grad()
def compute_trajectory_variance(model, latents_norm, ages, age_min, age_max,
                                device, num_steps=50, num_target_ages=7,
                                batch_size=512, age_margin=0.0):
    """Compute per-sample trajectory variance using data-to-data flow.

    For each sample z_src at age_src, generate counterfactuals at T fixed target
    ages via direct transport (no noise bottleneck), then compute variance of the
    trajectory endpoints in latent space.

    Var_i = sum_d var_t(z_cf_i(age_t, d))

    Batched for efficiency — processes all samples at once per target age.
    """
    model.eval()
    N = len(latents_norm)
    _m = age_margin * (age_max - age_min)
    norm_min, norm_max = age_min - _m, age_max + _m
    target_ages_days = torch.linspace(age_min, age_max, num_target_ages)
    target_ages_norm = (target_ages_days - norm_min) / (norm_max - norm_min)
    ages_norm = (ages - norm_min) / (norm_max - norm_min)

    trajectories = []  # will be (T, N, D)

    for ta_idx, ta_norm in enumerate(target_ages_norm):
        ta_days = target_ages_days[ta_idx].item()
        print(f"  Target age {ta_idx+1}/{num_target_ages} ({ta_days:.0f}d)...")
        all_z_cf = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            z_src = latents_norm[start:end].to(device)
            a_src = ages_norm[start:end].to(device)
            B = z_src.shape[0]

            a_tgt = torch.full((B,), ta_norm.item(), device=device)
            c = torch.stack([a_src, a_tgt], dim=1)  # (B, 2)

            z_cf = model.generate(z_src, c, steps=num_steps, cfg_scale=1.0, solver='heun')
            all_z_cf.append(z_cf.cpu())

        all_z_cf = torch.cat(all_z_cf, dim=0)  # (N, D)
        trajectories.append(all_z_cf)

    trajectories = torch.stack(trajectories, dim=0)  # (T, N, D)

    # Variance per sample over target-age dimension
    var_per_dim = trajectories.var(dim=0)  # (N, D)
    total_variance = var_per_dim.sum(dim=1)  # (N,) scalar per sample

    return total_variance.numpy(), target_ages_days.numpy(), trajectories


@torch.no_grad()
def compute_trajectory_variance_gaprel(model, latents_norm, ages, age_min, age_max,
                                        device, gap_window=25.0, num_gaps=11,
                                        num_steps=50, batch_size=512, age_margin=0.0):
    """Compute per-sample trajectory variance using GAP-RELATIVE target ages.

    Instead of fixed absolute target ages (which create position-dependent
    gap ranges — edge samples get higher variance by geometry), this uses
    symmetric gaps centered on each sample's source age:

        targets_i = age_src_i + linspace(-gap_window, +gap_window, num_gaps)
        clamped to [age_min, age_max]

    Every sample sees the same range of gaps, so trajectory variance reflects
    HOW the displacement changes with gap (differently for calls vs songs),
    not WHERE the sample sits in the age range.

    Args:
        gap_window: Max gap in days (symmetric). Default 25 = ±25 days.
        num_gaps: Number of gap steps. Default 11 = gaps at -25,-20,...,+20,+25.

    Returns:
        variances: (N,) per-sample trajectory variance
        gaps_days: (num_gaps,) gap values used
        trajectories: (T, N, D) counterfactual trajectories
    """
    model.eval()
    N = len(latents_norm)
    D = latents_norm.shape[1]
    _m = age_margin * (age_max - age_min)
    norm_min, norm_max = age_min - _m, age_max + _m
    norm_range = norm_max - norm_min
    ages_norm = (ages.float() - norm_min) / norm_range

    # Symmetric gaps in days
    gaps_days = torch.linspace(-gap_window, gap_window, num_gaps)
    gaps_norm = gaps_days / norm_range

    print(f'  Gap-relative trajectory variance: ±{gap_window:.0f}d, '
          f'{num_gaps} steps, gaps = {[f"{g:.0f}" for g in gaps_days.tolist()]}')

    trajectories = torch.zeros(num_gaps, N, D)

    for g_idx, g_norm in enumerate(gaps_norm):
        gap_d = gaps_days[g_idx].item()
        print(f"  Gap {g_idx+1}/{num_gaps} ({gap_d:+.0f}d)...")

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            z_src = latents_norm[start:end].to(device)
            a_src = ages_norm[start:end].to(device)
            B = z_src.shape[0]

            # Target age = source age + gap, clamped to valid range
            a_tgt = (a_src + g_norm).clamp(0.0, 1.0)
            c = torch.stack([a_src, a_tgt], dim=1)

            z_cf = model.generate(z_src, c, steps=num_steps, cfg_scale=1.0,
                                  solver='heun')
            trajectories[g_idx, start:end] = z_cf.cpu()

    # Variance per sample over gap dimension
    var_per_dim = trajectories.var(dim=0)  # (N, D)
    total_variance = var_per_dim.sum(dim=1)  # (N,)

    return total_variance.numpy(), gaps_days.numpy(), trajectories


@torch.no_grad()
def compute_transported_latents(model, latents_norm, ages, age_min, age_max,
                                ref_age_days, device, num_steps=50,
                                batch_size=512, age_margin=0.0):
    """Transport all latents to a single reference age.

    Returns transported latents (N, D) in normalized space.
    This removes age as a confound: all samples are projected to the same
    developmental stage, so UMAP clusters reflect vocalization type, not age.
    """
    model.eval()
    N = len(latents_norm)
    _m = age_margin * (age_max - age_min)
    norm_min, norm_max = age_min - _m, age_max + _m
    ages_norm = (ages - norm_min) / (norm_max - norm_min)
    ref_age_norm = (ref_age_days - norm_min) / (norm_max - norm_min)

    print(f"  Transporting {N:,} samples to reference age {ref_age_days:.0f}d...")
    all_z_cf = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        z_src = latents_norm[start:end].to(device)
        a_src = ages_norm[start:end].to(device)
        B = z_src.shape[0]

        a_tgt = torch.full((B,), ref_age_norm, device=device)
        c = torch.stack([a_src, a_tgt], dim=1)  # (B, 2)

        z_cf = model.generate(z_src, c, steps=num_steps, cfg_scale=1.0,
                              solver='heun')
        all_z_cf.append(z_cf.cpu())

        if (start // batch_size) % 50 == 0:
            print(f"    {end:,}/{N:,}")

    transported = torch.cat(all_z_cf, dim=0)  # (N, D)
    print(f"  Done. Shape: {transported.shape}")
    return transported


def plot_variance_analysis(variances, ages, age_min, age_max, target_ages_days, output_path,
                           suffix='', title_extra=''):
    """Plot trajectory variance: overall histogram, vs age, and within-age-bin histograms."""
    ages_np = ages.numpy() if torch.is_tensor(ages) else ages

    # Age bins for within-day analysis
    num_bins = 6
    bin_edges = np.linspace(age_min, age_max, num_bins + 1)

    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, num_bins, hspace=0.35, wspace=0.3)

    # ---- Row 1: Overall histogram + scatter + stats ----
    ax_hist = fig.add_subplot(gs[0, :2])
    ax_hist.hist(variances, bins=60, color='teal', alpha=0.7, edgecolor='white')
    ax_hist.axvline(np.mean(variances), color='red', linestyle='--', alpha=0.7,
                    label=f'Mean={np.mean(variances):.2f}')
    ax_hist.axvline(np.median(variances), color='orange', linestyle='--', alpha=0.7,
                    label=f'Median={np.median(variances):.2f}')
    ax_hist.set_xlabel('Trajectory Variance')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title(f'All Samples (N={len(variances)})')
    ax_hist.legend(fontsize=8)

    ax_scatter = fig.add_subplot(gs[0, 2:4])
    scatter = ax_scatter.scatter(ages_np, variances, c=variances, cmap='viridis',
                                 alpha=0.3, s=3, rasterized=True)
    ax_scatter.set_xlabel('Source Age (days)')
    ax_scatter.set_ylabel('Trajectory Variance')
    ax_scatter.set_title('Variance vs Source Age')
    plt.colorbar(scatter, ax=ax_scatter, label='Variance')

    # Per-bin mean line
    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (ages_np >= lo) & (ages_np < hi) if b < num_bins - 1 \
            else (ages_np >= lo) & (ages_np <= hi)
        if mask.sum() > 0:
            mid = (lo + hi) / 2
            ax_scatter.plot(mid, np.mean(variances[mask]), 'rs', markersize=6)

    ax_stats = fig.add_subplot(gs[0, 4:])
    ax_stats.axis('off')
    cv = np.std(variances) / (np.mean(variances) + 1e-8)
    skew = float(((variances - variances.mean())**3).mean() / (variances.std()**3 + 1e-8))
    stats_text = (
        f"N = {len(variances)}\n"
        f"Mean = {np.mean(variances):.4f}\n"
        f"Std  = {np.std(variances):.4f}\n"
        f"CV   = {cv:.3f}\n"
        f"Min  = {np.min(variances):.4f}\n"
        f"Max  = {np.max(variances):.4f}\n"
        f"Median = {np.median(variances):.4f}\n"
        f"Skewness = {skew:.3f}\n\n"
        f"Target ages: {', '.join(f'{d:.0f}' for d in target_ages_days)}d\n"
    )
    ax_stats.text(0.05, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment='center', fontfamily='monospace')
    ax_stats.set_title('Summary Statistics')

    # ---- Row 2: Within-age-bin histograms ----
    # Shared x-axis range for comparability
    v_max_plot = np.percentile(variances, 99)
    for b in range(num_bins):
        ax = fig.add_subplot(gs[1, b])
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (ages_np >= lo) & (ages_np < hi) if b < num_bins - 1 \
            else (ages_np >= lo) & (ages_np <= hi)
        bin_var = variances[mask]
        if len(bin_var) > 0:
            ax.hist(bin_var, bins=30, color=plt.cm.Set2(b / num_bins),
                    alpha=0.7, edgecolor='white')
            ax.axvline(np.mean(bin_var), color='red', linestyle='--', alpha=0.7)
            ax.set_title(f'{lo:.0f}-{hi:.0f}d  (N={len(bin_var)})', fontsize=9)
            ax.set_xlim(0, v_max_plot)
        else:
            ax.set_title(f'{lo:.0f}-{hi:.0f}d  (empty)', fontsize=9)
        ax.set_xlabel('Variance', fontsize=8)
        if b == 0:
            ax.set_ylabel('Count')

    fig.suptitle(f'OT Flow — Trajectory Variance Analysis (Latent Space){title_extra}', fontsize=14)
    plt.savefig(output_path / f'variance_analysis{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()


# ========================== Training ==========================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae_dir = Path(args.ae_dir)

    print('=' * 60)
    arch_label = 'Direct Transport' if args.arch == 'direct' else 'Flow Matching'
    print(f'Data-to-Data {arch_label} with {args.coupling.upper()} Coupling')
    print('=' * 60)

    # Load AE (frozen)
    ae_config = json.load(open(ae_dir / 'config.json'))
    ae_model_type = ae_config.get('model_type', 'ae')
    AEClass = SpectrogramVAE if ae_model_type == 'vae' else SpectrogramAE
    ae = AEClass(ae_config['n_mels'], ae_config['n_time'],
                 ae_config['latent_dim']).to(device)
    ae.load_state_dict(torch.load(ae_dir / 'best.pt', map_location=device,
                                  weights_only=True))
    ae.eval()
    print(f'  {ae_model_type.upper()} loaded. latent_dim={ae_config["latent_dim"]}')

    # Load latents
    ld = torch.load(ae_dir / 'latents.pt', weights_only=True)
    latents, ages = ld['z'], ld['ages']
    lengths = ld.get('lengths', torch.full((len(latents),), ae_config['n_time']))
    age_min, age_max = ages.min().item(), ages.max().item()
    latent_dim = latents.shape[1]
    print(f'  Latents: {latents.shape}, ages: {age_min:.0f}-{age_max:.0f}d')

    # Normalize latents
    z_mean = latents.mean(0, keepdim=True)
    z_std = latents.std(0, keepdim=True) + 1e-6
    latents_norm = (latents - z_mean) / z_std

    z_mean_d = z_mean.to(device)
    z_std_d = z_std.to(device)

    # Age normalization with margin to avoid boundary effects at age extremes
    # (age_min -> ~0.05 instead of 0.0; age_max -> ~0.95 instead of 1.0)
    age_margin_days = args.age_margin * (age_max - age_min)
    norm_age_min = age_min - age_margin_days
    norm_age_max = age_max + age_margin_days
    norm_range = norm_age_max - norm_age_min
    ages_norm = (ages - norm_age_min) / norm_range
    if args.age_margin > 0:
        print(f'  Age normalization: margin={args.age_margin:.0%} ({age_margin_days:.1f}d), '
              f'norm range [{norm_age_min:.1f}, {norm_age_max:.1f}]d')

    # Train/validation split (90/10)
    N = len(latents_norm)
    split_gen = torch.Generator().manual_seed(42)
    split_perm = torch.randperm(N, generator=split_gen)
    n_val = int(0.1 * N)
    val_indices, train_indices = split_perm[:n_val], split_perm[n_val:]

    latents_train = latents_norm[train_indices]
    ages_norm_train = ages_norm[train_indices]
    ages_train = ages[train_indices]
    latents_val = latents_norm[val_indices]
    ages_norm_val = ages_norm[val_indices]
    ages_val = ages[val_indices]
    N_train, N_val = len(train_indices), len(val_indices)
    print(f'  Train/val split: {N_train} / {N_val} samples')

    # Age-balanced sampling: inverse-frequency weights by age bin
    sample_weights = None
    if args.oversample_young:
        ages_train_np = ages_train.numpy() if torch.is_tensor(ages_train) else np.array(ages_train)
        n_age_bins = 20
        bin_edges = np.linspace(age_min, age_max, n_age_bins + 1)
        bin_idx = np.digitize(ages_train_np, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_age_bins - 1)
        bin_counts = np.bincount(bin_idx, minlength=n_age_bins).astype(np.float64)
        bin_weights = 1.0 / (bin_counts + 1)
        sample_weights = torch.tensor(bin_weights[bin_idx], dtype=torch.float32)
        sample_weights = sample_weights / sample_weights.sum()
        active_bins = bin_counts[bin_counts > 0]
        print(f'  Age-balanced sampling: {n_age_bins} bins, '
              f'counts {active_bins.min():.0f}-{active_bins.max():.0f}, '
              f'weight ratio {bin_weights[bin_counts > 0].max()/bin_weights[bin_counts > 0].min():.1f}x')

    # Model
    use_v2 = args.gap_gate or args.gap_head_weight > 0
    use_hetero = args.heteroscedastic
    if args.arch == 'direct':
        if use_hetero:
            model = HeteroscedasticTransportMLP(
                latent_dim=latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_blocks,
                cond_dim=2,
                embed_dim=64,
            ).to(device)
            arch_name = 'direct'
            print(f'  Architecture: HeteroscedasticTransportMLP (NLL loss, '
                  f'predicts mean + variance)')
            if args.identity_reg_weight > 0:
                print(f'  Identity regularization: weight={args.identity_reg_weight}, '
                      f'tau={args.identity_reg_tau}d')
            if args.nll_warmup_epochs > 0:
                print(f'  NLL warmup: {args.nll_warmup_epochs} epochs of MSE before NLL')
        elif use_v2:
            model = TransportMLPv2(
                latent_dim=latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_blocks,
                cond_dim=2,
                embed_dim=64,
                gap_gate=args.gap_gate,
                gap_head=args.gap_head_weight > 0,
            ).to(device)
            arch_name = 'direct'
            print(f'  Architecture: TransportMLPv2 (gap_gate={args.gap_gate}, '
                  f'gap_head={args.gap_head_weight > 0})')
        else:
            model = TransportMLP(
                latent_dim=latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_blocks,
                cond_dim=2,
                embed_dim=64,
            ).to(device)
            arch_name = 'direct'
            print(f'  Architecture: TransportMLP (direct displacement, no time/ODE)')
    else:
        model = FlowMLP(
            latent_dim=latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_blocks,
            time_dim=64,
            cond_dim=2,
            use_adaln=True,
            cfg_dropout=args.cfg_dropout,
            zero_init=True,
        ).to(device)
        arch_name = 'ot_adaln'
        print(f'  Architecture: FlowMLP (AdaLN, cond_dim=2, CFG dropout={args.cfg_dropout})')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(__file__).parent / 'models' / f'ot_flow_{args.bird}_{timestamp}'
    output_path.mkdir(parents=True, exist_ok=True)

    # Sinkhorn distributional loss (fixes mode collapse)
    sinkhorn_loss_fn = None
    sinkhorn_batch_counter = 0
    if args.sinkhorn_weight > 0:
        try:
            from geomloss import SamplesLoss
            sinkhorn_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.5,
                                           scaling=0.7, backend="tensorized")
            print(f'  Sinkhorn loss: weight={args.sinkhorn_weight}, blur=0.5, '
                  f'every {args.sinkhorn_every} batches')
        except ImportError:
            print('  WARNING: geomloss not installed. pip install geomloss')
            print('  Falling back to no Sinkhorn loss.')
            args.sinkhorn_weight = 0.0

    config = {
        'ae_dir': str(ae_dir),
        'bird': args.bird,
        'arch': arch_name,
        'model_version': 3 if use_hetero else (2 if use_v2 else 1),
        'coupling': args.coupling,
        'cfg_dropout': args.cfg_dropout,
        'latent_dim': latent_dim,
        'hidden_dim': args.hidden_dim,
        'num_blocks': args.num_blocks,
        'cond_dim': 2,
        'age_min': age_min,
        'age_max': age_max,
        'z_mean': z_mean.squeeze(0).tolist(),
        'z_std': z_std.squeeze(0).tolist(),
        'n_time': ae_config['n_time'],
        'n_mels': ae_config['n_mels'],
        'data_mean': ae_config['data_mean'],
        'data_std': ae_config['data_std'],
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'n_samples': N,
        'n_train': N_train,
        'n_val': N_val,
        'knn_k': args.knn_k,
        'min_age_gap': args.min_age_gap,
        'mag_weight': args.mag_weight,
        'std_weight': args.std_weight,
        'mean_weight': args.mean_weight,
        'sinkhorn_weight': args.sinkhorn_weight,
        'age_margin': args.age_margin,
        'norm_age_min': norm_age_min,
        'norm_age_max': norm_age_max,
        'oversample_young': args.oversample_young,
        'gap_gate': args.gap_gate,
        'gap_head': args.gap_head_weight > 0,
        'gap_head_weight': args.gap_head_weight,
        'identity_prob': args.identity_prob,
        'gap_curriculum': args.gap_curriculum,
        'rescale_targets': args.rescale_targets,
        'gap_loss_weight': args.gap_loss_weight,
        'heteroscedastic': args.heteroscedastic,
        'identity_reg_weight': args.identity_reg_weight,
        'identity_reg_tau': args.identity_reg_tau,
        'nll_warmup_epochs': args.nll_warmup_epochs,
    }
    json.dump(config, open(output_path / 'config.json', 'w'), indent=2)

    # kNN coupling is now mini-batch (no global precomputation needed).
    # Convert min_age_gap from days to normalized age units for mini-batch filtering.
    min_age_gap_norm = args.min_age_gap / norm_range if args.min_age_gap > 0 else 0

    # Training
    best_val = float('inf')
    num_batches = max(1, N_train // args.batch_size)
    num_batches_val = max(1, N_val // args.batch_size)

    # Max gap in normalized age space (for target rescaling)
    max_gap_norm = (age_max - age_min) / norm_range  # ~0.91 typically

    # Gap curriculum schedule
    if args.gap_curriculum:
        gap_phases = [
            (0.25, 15.0),   # first 25%: |gap| <= 15d
            (0.50, 30.0),   # next 25%: |gap| <= 30d
            (0.75, 45.0),   # next 25%: |gap| <= 45d
            (1.00, 999.0),  # final 25%: full range
        ]
        print(f'  Gap curriculum: {[(f"{int(f*100)}%", g) for f, g in gap_phases]}')

    print(f'\n  Training for {args.epochs} epochs ({num_batches} batches/epoch)...')
    print(f'  Coupling: {args.coupling}' +
          (f' (mini-batch, k={args.knn_k}, min_age_gap={args.min_age_gap}d)'
           if args.coupling == 'knn' else ''))
    if args.identity_prob > 0:
        print(f'  Identity pair mixing: p={args.identity_prob}')
    if args.gap_head_weight > 0:
        print(f'  Gap-head aux loss: weight={args.gap_head_weight}')
    print(f'  Output: {output_path}')

    epoch_bar = tqdm(range(args.epochs), desc='Training', unit='epoch',
                     dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        losses = []

        batch_bar = tqdm(range(num_batches), desc=f'  Ep {epoch+1}',
                         unit='batch', leave=False, dynamic_ncols=True)
        for _ in batch_bar:
            if sample_weights is not None:
                idx_src = torch.multinomial(sample_weights, args.batch_size, replacement=False)
            else:
                idx_src = torch.randperm(N_train)[:args.batch_size]

            if args.coupling == 'knn':
                # Mini-batch kNN: sample independent targets, then kNN coupling
                idx_tgt = torch.randperm(N_train)[:args.batch_size]
            else:
                # OT: sample independent targets, then Hungarian matching
                idx_tgt = torch.randperm(N_train)[:args.batch_size]

            z_src = latents_train[idx_src].to(device)
            z_tgt = latents_train[idx_tgt].to(device)
            a_src = ages_norm_train[idx_src].to(device)
            a_tgt = ages_norm_train[idx_tgt].to(device)

            if args.coupling == 'ot':
                # OT coupling: reorder targets to minimize transport cost
                perm = ot_coupling(z_src, z_tgt)
                z_tgt = z_tgt[perm]
                a_tgt = a_tgt[perm]
            elif args.coupling == 'knn':
                # kNN coupling: reorder targets so each source is paired
                # with a random one of its k nearest targets
                perm = knn_coupling(z_src, z_tgt, k=args.knn_k,
                                    ages_src=a_src, ages_tgt=a_tgt,
                                    min_age_gap=min_age_gap_norm)
                z_tgt = z_tgt[perm]
                a_tgt = a_tgt[perm]

            # Gap curriculum: filter pairs by max allowed age gap
            if args.gap_curriculum:
                progress = (epoch + 1) / args.epochs
                max_gap_days = 999.0
                for frac, gap_d in gap_phases:
                    if progress <= frac:
                        max_gap_days = gap_d
                        break
                # Convert to normalized gap
                max_gap_norm = max_gap_days / norm_range
                gap_norm = (a_tgt - a_src).abs()
                keep = gap_norm <= max_gap_norm
                if keep.sum() >= 16:  # need enough samples
                    z_src = z_src[keep]
                    z_tgt = z_tgt[keep]
                    a_src = a_src[keep]
                    a_tgt = a_tgt[keep]

            # Identity pair mixing: with probability p, set tgt = src
            if args.identity_prob > 0 and args.arch == 'direct':
                B = z_src.shape[0]
                id_mask = torch.rand(B, device=device) < args.identity_prob
                if id_mask.any():
                    z_tgt = z_tgt.clone()
                    a_tgt = a_tgt.clone()
                    z_tgt[id_mask] = z_src[id_mask]
                    a_tgt[id_mask] = a_src[id_mask]

            # Target displacement
            delta_target = z_tgt - z_src
            delta_target_orig = delta_target  # keep unrescaled for variance head

            # Target rescaling: preserve OT direction, scale magnitude by |gap|
            if args.rescale_targets:
                gap_abs = (a_tgt - a_src).abs()  # (B,)
                scale = gap_abs / (max_gap_norm + 1e-8)  # 0 at gap=0, ~1 at max gap
                delta_norm = delta_target.norm(dim=1, keepdim=True) + 1e-8
                direction = delta_target / delta_norm  # unit vector
                delta_target = direction * delta_norm * scale.unsqueeze(1)

            # Condition: (age_src_norm, age_tgt_norm)
            cond = torch.stack([a_src, a_tgt], dim=1)  # (B, 2)

            if args.arch == 'direct':
                # Direct displacement: predict delta from z_src
                if use_hetero:
                    delta_mean, delta_log_var = model(z_src, cond)
                    delta_pred = delta_mean
                    gap_pred = None
                elif use_v2:
                    delta_pred, gap_pred = model(z_src, cond)
                else:
                    delta_pred = model(z_src, cond)
                    gap_pred = None
                z_pred = z_src + delta_pred  # predicted target latents

                # Build loss
                if use_hetero:
                    # Dual loss for heteroscedastic + rescale_targets:
                    # MSE on rescaled targets (mean head learns gap-proportional displacement)
                    # NLL on ORIGINAL targets with detached mean (variance head learns full coupling noise)
                    # During warmup, use MSE only to let mean head converge first
                    use_nll = (epoch >= args.nll_warmup_epochs)
                    loss_mse = F.mse_loss(delta_mean, delta_target)  # rescaled
                    if use_nll:
                        # Variance head: use original (unrescaled) targets
                        # Detach mean so NLL gradient only flows to variance head
                        residual_sq = (delta_target_orig - delta_mean.detach()).pow(2)
                        inv_var = (-delta_log_var).exp()
                        loss_nll_var = 0.5 * (delta_log_var + residual_sq * inv_var).mean()
                        loss = loss_mse + loss_nll_var
                    else:
                        loss = loss_mse

                elif args.loss_mode == 'sinkhorn_only':
                    # Pure distributional loss — zero MSE but keep grad_fn
                    loss = 0.0 * delta_pred.sum()
                else:
                    # MSE on displacement
                    if args.gap_loss_weight:
                        # Per-sample weighted MSE: down-weight small-gap pairs
                        gap_w = (a_tgt - a_src).abs() / (max_gap_norm + 1e-8)
                        gap_w = gap_w.clamp(min=0.01)  # small floor to keep some gradient
                        per_sample_mse = (delta_pred - delta_target).pow(2).mean(dim=1)
                        loss = (per_sample_mse * gap_w).mean()
                    else:
                        loss = F.mse_loss(delta_pred, delta_target)

                # Soft identity regularization (all loss modes)
                if args.identity_reg_weight > 0:
                    gap_days = (a_tgt - a_src).abs() * norm_range
                    tau = args.identity_reg_tau
                    gauss_weight = torch.exp(-gap_days.pow(2) / (2 * tau * tau))
                    identity_penalty = (delta_pred.pow(2).mean(dim=1) * gauss_weight).mean()
                    loss = loss + args.identity_reg_weight * identity_penalty

                # Gap prediction auxiliary loss
                if gap_pred is not None and args.gap_head_weight > 0:
                    true_gap = (cond[:, 1] - cond[:, 0]).abs().unsqueeze(1)
                    loss_gap = F.mse_loss(gap_pred, true_gap)
                    loss = loss + args.gap_head_weight * loss_gap

                # Sinkhorn distributional loss: match CF distribution to
                # target distribution. Directly prevents mode collapse.
                if sinkhorn_loss_fn is not None:
                    sinkhorn_batch_counter += 1
                    if sinkhorn_batch_counter % args.sinkhorn_every == 0:
                        loss_sink = sinkhorn_loss_fn(z_pred, z_tgt)
                        loss = loss + args.sinkhorn_weight * loss_sink

                # Legacy auxiliary losses
                if args.mag_weight > 0:
                    loss_mag = F.mse_loss(
                        delta_pred.norm(dim=-1), delta_target.norm(dim=-1))
                    loss = loss + args.mag_weight * loss_mag

                if args.std_weight > 0:
                    std_pred = z_pred.std(dim=0)
                    std_tgt = z_tgt.std(dim=0)
                    loss_std = F.mse_loss(std_pred, std_tgt)
                    loss = loss + args.std_weight * loss_std

                if args.mean_weight > 0:
                    mean_pred = z_pred.mean(dim=0)
                    mean_tgt = z_tgt.mean(dim=0)
                    loss_mean = F.mse_loss(mean_pred, mean_tgt)
                    loss = loss + args.mean_weight * loss_mean

                v_pred = delta_pred  # for magnitude monitoring
            else:
                # Flow matching: predict velocity at interpolated z_t
                B = z_src.shape[0]
                t = torch.rand(B, device=device)
                t_expand = t.unsqueeze(1)
                z_t = (1 - t_expand) * z_src + t_expand * z_tgt
                v_pred = model(z_t, t, cond, train_cfg=True)
                loss = F.mse_loss(v_pred, delta_target)

            v_target = delta_target

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

            # Velocity magnitude check
            with torch.no_grad():
                v_norm_pred = v_pred.norm(dim=1).mean().item()
                v_norm_tgt = v_target.norm(dim=1).mean().item()
                if use_hetero:
                    avg_log_var_val = delta_log_var.mean().item()
                    if 'avg_log_var' not in locals():
                        avg_log_var = avg_log_var_val
                    else:
                        avg_log_var = 0.95 * avg_log_var + 0.05 * avg_log_var_val

            # Simple moving average for monitoring
            if 'avg_v_pred' not in locals():
                avg_v_pred, avg_v_tgt = v_norm_pred, v_norm_tgt
            else:
                avg_v_pred = 0.95 * avg_v_pred + 0.05 * v_norm_pred
                avg_v_tgt = 0.95 * avg_v_tgt + 0.05 * v_norm_tgt

        scheduler.step()
        avg_loss = np.mean(losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(num_batches_val):
                idx_src_v = torch.randperm(N_val)[:args.batch_size]

                if args.coupling == 'knn':
                    idx_tgt_v = torch.randperm(N_val)[:args.batch_size]
                else:
                    idx_tgt_v = torch.randperm(N_val)[:args.batch_size]

                z_src_v = latents_val[idx_src_v].to(device)
                z_tgt_v = latents_val[idx_tgt_v].to(device)
                a_src_v = ages_norm_val[idx_src_v].to(device)
                a_tgt_v = ages_norm_val[idx_tgt_v].to(device)

                if args.coupling == 'ot':
                    perm_v = ot_coupling(z_src_v, z_tgt_v)
                    z_tgt_v = z_tgt_v[perm_v]
                    a_tgt_v = a_tgt_v[perm_v]
                elif args.coupling == 'knn':
                    perm_v = knn_coupling(z_src_v, z_tgt_v, k=args.knn_k,
                                          ages_src=a_src_v, ages_tgt=a_tgt_v,
                                          min_age_gap=min_age_gap_norm)
                    z_tgt_v = z_tgt_v[perm_v]
                    a_tgt_v = a_tgt_v[perm_v]

                delta_target_v = z_tgt_v - z_src_v
                delta_target_orig_v = delta_target_v  # keep unrescaled

                # Target rescaling (validation)
                if args.rescale_targets:
                    gap_abs_v = (a_tgt_v - a_src_v).abs()
                    scale_v = gap_abs_v / (max_gap_norm + 1e-8)
                    dn_v = delta_target_v.norm(dim=1, keepdim=True) + 1e-8
                    dir_v = delta_target_v / dn_v
                    delta_target_v = dir_v * dn_v * scale_v.unsqueeze(1)

                cond_v = torch.stack([a_src_v, a_tgt_v], dim=1)

                if args.arch == 'direct':
                    if use_hetero:
                        delta_mean_v, delta_log_var_v = model(z_src_v, cond_v)
                        delta_pred_v = delta_mean_v
                        gap_pred_v = None
                    elif use_v2:
                        delta_pred_v, gap_pred_v = model(z_src_v, cond_v)
                    else:
                        delta_pred_v = model(z_src_v, cond_v)
                        gap_pred_v = None
                    z_pred_v = z_src_v + delta_pred_v

                    if use_hetero:
                        # Dual loss for validation (consistent with training)
                        loss_mse_v = F.mse_loss(delta_mean_v, delta_target_v)
                        residual_sq_v = (delta_target_orig_v - delta_mean_v).pow(2)
                        inv_var_v = (-delta_log_var_v).exp()
                        loss_nll_v = 0.5 * (delta_log_var_v + residual_sq_v * inv_var_v).mean()
                        v_loss = loss_mse_v + loss_nll_v
                    elif args.loss_mode == 'sinkhorn_only':
                        v_loss = 0.0 * delta_pred_v.sum()
                    else:
                        v_loss = F.mse_loss(delta_pred_v, delta_target_v)
                    if args.identity_reg_weight > 0:
                        gap_days_v = (a_tgt_v - a_src_v).abs() * norm_range
                        tau = args.identity_reg_tau
                        gw_v = torch.exp(-gap_days_v.pow(2) / (2 * tau * tau))
                        id_pen_v = (delta_pred_v.pow(2).mean(dim=1) * gw_v).mean()
                        v_loss = v_loss + args.identity_reg_weight * id_pen_v
                    if gap_pred_v is not None and args.gap_head_weight > 0:
                        true_gap_v = (cond_v[:, 1] - cond_v[:, 0]).abs().unsqueeze(1)
                        v_loss = v_loss + args.gap_head_weight * F.mse_loss(
                            gap_pred_v, true_gap_v)
                    if sinkhorn_loss_fn is not None:
                        v_loss = v_loss + args.sinkhorn_weight * sinkhorn_loss_fn(
                            z_pred_v, z_tgt_v)
                    if args.mag_weight > 0:
                        v_loss = v_loss + args.mag_weight * F.mse_loss(
                            delta_pred_v.norm(dim=-1), delta_target_v.norm(dim=-1))
                    if args.std_weight > 0:
                        v_loss = v_loss + args.std_weight * F.mse_loss(
                            z_pred_v.std(dim=0), z_tgt_v.std(dim=0))
                    if args.mean_weight > 0:
                        v_loss = v_loss + args.mean_weight * F.mse_loss(
                            z_pred_v.mean(dim=0), z_tgt_v.mean(dim=0))
                else:
                    B_v = z_src_v.shape[0]
                    t_v = torch.rand(B_v, device=device)
                    z_t_v = (1 - t_v.unsqueeze(1)) * z_src_v + t_v.unsqueeze(1) * z_tgt_v
                    v_pred_v = model(z_t_v, t_v, cond_v, train_cfg=False)
                    v_loss = F.mse_loss(v_pred_v, delta_target_v)

                val_losses.append(v_loss.item())

        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), output_path / 'best.pt')

        # Update progress bar with live stats
        epoch_bar.set_postfix(
            train=f'{avg_loss:.4f}',
            val=f'{avg_val_loss:.4f}',
            best=f'{best_val:.4f}',
            vm=f'{avg_v_pred:.2f}/{avg_v_tgt:.2f}',
            lr=f'{scheduler.get_last_lr()[0]:.1e}',
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            extra = ''
            if use_hetero and 'avg_log_var' in locals():
                extra = f' log_var={avg_log_var:.2f}'
            tqdm.write(f'  Epoch {epoch+1:3d}/{args.epochs}: train={avg_loss:.6f} val={avg_val_loss:.6f} '
                       f'vm_pred={avg_v_pred:.4f} vm_tgt={avg_v_tgt:.4f} '
                       f'(best_val={best_val:.6f}) lr={scheduler.get_last_lr()[0]:.2e}{extra}')

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_path / f'epoch_{epoch+1:03d}.pt')

        if (epoch + 1) % args.viz_every == 0 or epoch == 0:
            visualize(model, ae, latents_norm, ages, lengths, z_mean_d, z_std_d,
                      age_min, age_max, ae_config, device, output_path, epoch,
                      num_samples=4, num_steps=args.num_steps,
                      age_margin=args.age_margin)

    torch.save(model.state_dict(), output_path / 'final.pt')

    # ========================== Post-Training Analysis ==========================
    run_variance_analysis(model, latents_norm, ages, age_min, age_max,
                          device, output_path, args.num_steps, args.num_target_ages,
                          age_margin=args.age_margin)

    # Heteroscedastic diagnostics
    if use_hetero:
        run_heteroscedastic_diagnostics(model, latents_norm, ages, age_min, age_max,
                                        norm_age_min, norm_age_max, device, output_path)

    print(f'\nDone! Best val loss: {best_val:.6f}')
    print(f'Output: {output_path}')


def run_variance_analysis(model, latents_norm, ages, age_min, age_max,
                          device, output_path, num_steps=50, num_target_ages=7,
                          age_margin=0.0):
    """Run trajectory variance analysis: both absolute and gap-relative."""
    # --- 1. Original absolute target ages ---
    print('\n' + '=' * 60)
    print('Trajectory Variance Analysis — Absolute Target Ages')
    print('=' * 60)

    variances, target_ages_days, trajectories = compute_trajectory_variance(
        model, latents_norm, ages, age_min, age_max, device,
        num_steps=num_steps, num_target_ages=num_target_ages,
        age_margin=age_margin)

    cv = np.std(variances) / (np.mean(variances) + 1e-8)
    print(f'  Variance: mean={variances.mean():.4f}, std={variances.std():.4f}')
    print(f'  CV={cv:.3f}, min={variances.min():.4f}, max={variances.max():.4f}')

    plot_variance_analysis(variances, ages, age_min, age_max,
                           target_ages_days, output_path)

    try:
        torch.save(trajectories, output_path / 'trajectories.pt')
    except RuntimeError as e:
        print(f'  WARNING: Could not save trajectories.pt ({e})')

    results = {
        'variances': variances.tolist(),
        'ages': (ages.numpy() if torch.is_tensor(ages) else ages).tolist(),
        'target_ages_days': target_ages_days.tolist(),
        'variance_mean': float(variances.mean()),
        'variance_std': float(variances.std()),
        'variance_cv': float(cv),
        'n_samples': len(variances),
    }
    json.dump(results, open(output_path / 'variance_results.json', 'w'), indent=2)
    print(f'  Saved to {output_path / "variance_analysis.png"}')

    # --- 2. Gap-relative target ages (position-invariant) ---
    print('\n' + '=' * 60)
    print('Trajectory Variance Analysis — Gap-Relative (position-invariant)')
    print('=' * 60)

    gap_window = min(25.0, (age_max - age_min) / 2.5)  # adapt to age range
    var_gaprel, gaps_days, traj_gaprel = compute_trajectory_variance_gaprel(
        model, latents_norm, ages, age_min, age_max, device,
        gap_window=gap_window, num_gaps=11, num_steps=num_steps,
        age_margin=age_margin)

    cv_gr = np.std(var_gaprel) / (np.mean(var_gaprel) + 1e-8)
    print(f'  Variance: mean={var_gaprel.mean():.4f}, std={var_gaprel.std():.4f}')
    print(f'  CV={cv_gr:.3f}, min={var_gaprel.min():.4f}, max={var_gaprel.max():.4f}')

    plot_variance_analysis(var_gaprel, ages, age_min, age_max,
                           gaps_days, output_path,
                           suffix='_gaprel',
                           title_extra=' (Gap-Relative)')

    try:
        torch.save(traj_gaprel, output_path / 'trajectories_gaprel.pt')
    except RuntimeError as e:
        print(f'  WARNING: Could not save trajectories_gaprel.pt ({e})')

    results_gr = {
        'variances': var_gaprel.tolist(),
        'ages': (ages.numpy() if torch.is_tensor(ages) else ages).tolist(),
        'gaps_days': gaps_days.tolist(),
        'gap_window': gap_window,
        'variance_mean': float(var_gaprel.mean()),
        'variance_std': float(var_gaprel.std()),
        'variance_cv': float(cv_gr),
        'n_samples': len(var_gaprel),
    }
    json.dump(results_gr, open(output_path / 'variance_results_gaprel.json', 'w'), indent=2)
    print(f'  Saved to {output_path / "variance_analysis_gaprel.png"}')


@torch.no_grad()
def run_heteroscedastic_diagnostics(model, latents_norm, ages, age_min, age_max,
                                     norm_age_min, norm_age_max, device, output_path,
                                     batch_size=512):
    """Post-training diagnostics for HeteroscedasticTransportMLP.

    Shows:
    1. Learned variance vs age gap (should grow with gap)
    2. Same-age identity check (delta_mean should be ~0, variance should be large)
    3. Variance ratio: deterministic vs sampled vs real
    """
    model.eval()
    norm_range = norm_age_max - norm_age_min
    N = len(latents_norm)

    print('\n' + '=' * 60)
    print('Heteroscedastic Diagnostics')
    print('=' * 60)

    # 1. Learned variance vs age gap
    print('\n  --- Learned variance vs age gap ---')
    print(f'  {"Gap (days)":>12s} | {"||delta_mean||":>15s} | {"mean(log_var)":>14s} | '
          f'{"mean(std)":>10s}')
    print('  ' + '-' * 60)

    # Use a fixed subset of samples
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(N, min(2000, N), replace=False)
    z_sub = latents_norm[sub_idx].to(device)
    ages_sub = ages[sub_idx].float()
    ages_sub_norm = ((ages_sub - norm_age_min) / norm_range).to(device)

    test_gaps_days = [0, 1, 3, 5, 10, 15, 20, 30, 45]
    test_gaps_days = [g for g in test_gaps_days if g <= (age_max - age_min)]

    for gap_d in test_gaps_days:
        gap_norm = gap_d / norm_range
        a_tgt = (ages_sub_norm + gap_norm).clamp(0, 1)
        c = torch.stack([ages_sub_norm, a_tgt], dim=1)

        delta_mean, delta_log_var = model(z_sub, c)
        d_norm = delta_mean.norm(dim=1).mean().item()
        mean_lv = delta_log_var.mean().item()
        mean_std = (0.5 * delta_log_var).exp().mean().item()

        print(f'  {gap_d:>12d} | {d_norm:>15.4f} | {mean_lv:>14.3f} | '
              f'{mean_std:>10.4f}')

    # 2. Variance ratio comparison: deterministic vs sampled
    print('\n  --- Variance ratio: deterministic vs sampled ---')
    target_ages = np.linspace(age_min, age_max, 7)
    age_bin = (age_max - age_min) / 12
    ages_np = ages.numpy() if torch.is_tensor(ages) else ages

    print(f'  {"Tgt Age":>8s} | {"Var(real)":>10s} | {"Var(det)":>10s} | '
          f'{"Var(samp)":>10s} | {"Ratio det":>10s} | {"Ratio samp":>11s}')
    print('  ' + '-' * 70)

    # Source samples for CF generation
    src_idx = rng.choice(N, min(1000, N), replace=False)

    for ta in target_ages:
        # Real samples at this age
        mask = (ages_np >= ta - age_bin) & (ages_np <= ta + age_bin)
        real_idx = np.where(mask)[0]
        if len(real_idx) < 30:
            continue

        z_real = latents_norm[real_idx].numpy()

        # Generate CFs
        z_src = latents_norm[src_idx].to(device)
        a_src = ages[src_idx].float()
        a_src_norm = ((a_src - norm_age_min) / norm_range).to(device)
        ta_norm = (ta - norm_age_min) / norm_range
        a_tgt = torch.full((len(src_idx),), ta_norm, device=device)
        c = torch.stack([a_src_norm, a_tgt], dim=1)

        z_cf_det = model.generate(z_src, c, sample=False).cpu().numpy()
        z_cf_samp = model.generate(z_src, c, sample=True).cpu().numpy()

        var_real = np.var(z_real, axis=0).sum()
        var_det = np.var(z_cf_det, axis=0).sum()
        var_samp = np.var(z_cf_samp, axis=0).sum()

        print(f'  {ta:>8.0f} | {var_real:>10.2f} | {var_det:>10.2f} | '
              f'{var_samp:>10.2f} | {var_det/(var_real+1e-8):>10.3f} | '
              f'{var_samp/(var_real+1e-8):>11.3f}')

    # Save a summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: ||delta_mean|| and mean(std) vs gap
    gaps_plot, norms_plot, stds_plot = [], [], []
    for gap_d in range(0, int(age_max - age_min) + 1, 2):
        gap_norm = gap_d / norm_range
        a_tgt = (ages_sub_norm + gap_norm).clamp(0, 1)
        c = torch.stack([ages_sub_norm, a_tgt], dim=1)
        dm, dlv = model(z_sub, c)
        gaps_plot.append(gap_d)
        norms_plot.append(dm.norm(dim=1).mean().item())
        stds_plot.append((0.5 * dlv).exp().mean().item())

    ax = axes[0]
    ax.plot(gaps_plot, norms_plot, 'b-o', markersize=3, label='||delta_mean||')
    ax.set_xlabel('Age gap (days)')
    ax.set_ylabel('||delta_mean||', color='b')
    ax2 = ax.twinx()
    ax2.plot(gaps_plot, stds_plot, 'r-s', markersize=3, label='mean(std)')
    ax2.set_ylabel('mean(std)', color='r')
    ax.set_title('Learned displacement & uncertainty vs age gap')
    ax.legend(loc='upper left'); ax2.legend(loc='lower right')

    # Panel 2: Variance ratio across target ages
    ratios_det, ratios_samp, ta_plot = [], [], []
    for ta in np.linspace(age_min, age_max, 10):
        mask = (ages_np >= ta - age_bin) & (ages_np <= ta + age_bin)
        real_idx = np.where(mask)[0]
        if len(real_idx) < 30:
            continue
        z_real = latents_norm[real_idx].numpy()
        z_src = latents_norm[src_idx].to(device)
        a_src = ages[src_idx].float()
        a_src_norm = ((a_src - norm_age_min) / norm_range).to(device)
        ta_norm = (ta - norm_age_min) / norm_range
        a_tgt = torch.full((len(src_idx),), ta_norm, device=device)
        c = torch.stack([a_src_norm, a_tgt], dim=1)
        z_det = model.generate(z_src, c, sample=False).cpu().numpy()
        z_samp = model.generate(z_src, c, sample=True).cpu().numpy()
        vr = np.var(z_real, axis=0).sum()
        ta_plot.append(ta)
        ratios_det.append(np.var(z_det, axis=0).sum() / (vr + 1e-8))
        ratios_samp.append(np.var(z_samp, axis=0).sum() / (vr + 1e-8))

    ax = axes[1]
    ax.plot(ta_plot, ratios_det, 'b-o', markersize=4, label='Deterministic')
    ax.plot(ta_plot, ratios_samp, 'r-s', markersize=4, label='Sampled')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Ideal')
    ax.set_xlabel('Target age (days)')
    ax.set_ylabel('Variance ratio (CF/real)')
    ax.set_title('Variance ratio across ages')
    ax.legend()
    ax.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig(output_path / 'heteroscedastic_diagnostics.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'\n  Saved diagnostic plot to {output_path / "heteroscedastic_diagnostics.png"}')


def analyze_only(args):
    """Load a trained OT flow model and run variance analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow_dir = Path(args.flow_dir)
    flow_config = json.load(open(flow_dir / 'config.json'))
    ae_dir = Path(flow_config['ae_dir'])

    print('=' * 60)
    print('OT Flow — Standalone Variance Analysis')
    print('=' * 60)

    # Load latents
    ld = torch.load(ae_dir / 'latents.pt', weights_only=True)
    latents, ages = ld['z'], ld['ages']
    age_min, age_max = flow_config['age_min'], flow_config['age_max']

    z_mean = torch.tensor(flow_config['z_mean']).unsqueeze(0)
    z_std = torch.tensor(flow_config['z_std']).unsqueeze(0)
    latents_norm = (latents - z_mean) / z_std

    print(f'  Latents: {latents.shape}, ages: {age_min:.0f}-{age_max:.0f}d')

    # Load model
    from Counterfactual_generation.models.flow import load_transport_model
    model = load_transport_model(flow_config, device, weights_path=flow_dir / 'best.pt')
    print(f'  Model loaded from {flow_dir}')

    age_margin = flow_config.get('age_margin', 0.0)
    run_variance_analysis(model, latents_norm, ages, age_min, age_max,
                          device, flow_dir, args.num_steps, args.num_target_ages,
                          age_margin=age_margin)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Data-to-Data Transport with k-NN Coupling')
    p.add_argument('--ae_dir', type=str, default=None)
    p.add_argument('--bird', type=str, default='R5018')
    p.add_argument('--arch', type=str, default='direct', choices=['direct', 'flow'],
                   help='Architecture: direct (single-pass displacement) or '
                        'flow (velocity field + ODE integration)')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_blocks', type=int, default=6)
    p.add_argument('--cfg_dropout', type=float, default=0.0,
                   help='CFG dropout rate. Use 0.0 for data-to-data transport '
                        '(CFG is harmful when unconditional has no meaning).')
    p.add_argument('--mag_weight', type=float, default=0.0,
                   help='Weight for magnitude-matching loss term (0 to disable)')
    p.add_argument('--std_weight', type=float, default=0.0,
                   help='Weight for batch-wise std-matching loss (0 to disable). '
                        'Prevents regression-to-mean / mode collapse. Try 0.01, 0.1, 0.5.')
    p.add_argument('--mean_weight', type=float, default=0.0,
                   help='Weight for batch-wise mean-matching loss (0 to disable). '
                        'Usually not needed since MSE on deltas handles mean alignment.')
    p.add_argument('--sinkhorn_weight', type=float, default=0.0,
                   help='Weight for Sinkhorn distributional loss. Fixes mode collapse '
                        'by directly matching CF distribution to target distribution. '
                        'When >0, consider disabling mag_weight and std_weight. '
                        'Try 0.5-2.0. Requires: pip install geomloss')
    p.add_argument('--sinkhorn_every', type=int, default=5,
                   help='Apply Sinkhorn loss every N batches (1=every batch, '
                        '4=every 4th). Higher = faster training.')
    p.add_argument('--loss_mode', type=str, default='mse',
                   choices=['mse', 'sinkhorn_blend', 'sinkhorn_only'],
                   help='Loss mode: mse (original, no Sinkhorn), '
                        'sinkhorn_blend (MSE + Sinkhorn), '
                        'sinkhorn_only (pure Sinkhorn, no MSE)')
    p.add_argument('--coupling', type=str, default='ot', choices=['ot', 'knn'],
                   help='Coupling strategy: ot (Hungarian) or knn (local neighbors)')
    p.add_argument('--knn_k', type=int, default=10,
                   help='Number of nearest neighbors for knn coupling (5-50)')
    p.add_argument('--min_age_gap', type=int, default=1,
                   help='Minimum age gap (days) for knn neighbors. '
                        'Prevents identity collapse by forcing cross-age pairing.')
    p.add_argument('--num_steps', type=int, default=50, help='ODE steps for inference/viz')
    p.add_argument('--num_target_ages', type=int, default=7, help='Number of target ages for variance')
    p.add_argument('--age_margin', type=float, default=0.05,
                   help='Age normalization margin (fraction of age range). '
                        'Prevents boundary effects at min/max ages. '
                        '0.05 = pad by 5%% of age range on each side.')
    p.add_argument('--oversample_young', action='store_true',
                   help='Inverse-frequency age sampling to boost early-age '
                        'representation in training batches.')
    # v2 improvements (all off by default for backward compatibility)
    p.add_argument('--gap_gate', action='store_true',
                   help='Enable gap-aware per-dimension gating (TransportMLPv2). '
                        'Multiplies displacement by alpha(|gap|), forcing '
                        'near-zero displacement at zero age gap.')
    p.add_argument('--gap_head_weight', type=float, default=0.0,
                   help='Weight for gap-prediction auxiliary loss (0=off, try 0.1). '
                        'Adds a head predicting |age_gap| from displacement.')
    p.add_argument('--identity_prob', type=float, default=0.0,
                   help='Probability of replacing targets with sources per sample '
                        '(identity pair mixing). Try 0.1.')
    p.add_argument('--gap_curriculum', action='store_true',
                   help='Enable gap curriculum: train on small gaps first, '
                        'progressively increasing max gap.')
    p.add_argument('--rescale_targets', action='store_true',
                   help='Rescale OT displacement targets: keep direction, '
                        'scale magnitude by |gap|/max_gap. Forces displacement '
                        'to be proportional to age gap. This is the primary fix '
                        'for same-age displacement != 0.')
    p.add_argument('--gap_loss_weight', action='store_true',
                   help='Weight MSE loss by |gap|/max_gap, down-weighting '
                        'noisy small-gap pairs.')
    # Heteroscedastic transport (v3)
    p.add_argument('--heteroscedastic', action='store_true',
                   help='Use HeteroscedasticTransportMLP: predict displacement '
                        'mean AND per-dimension variance. Trained with Gaussian '
                        'NLL instead of MSE. Fixes variance ratio collapse '
                        'without rescale_targets.')
    p.add_argument('--identity_reg_weight', type=float, default=0.0,
                   help='Weight for soft identity regularization. Penalizes '
                        '||delta_mean||^2 smoothly at small age gaps using '
                        'Gaussian weighting exp(-gap^2/(2*tau^2)). '
                        'Replaces rescale_targets for same-age displacement. '
                        'Try 0.5-2.0.')
    p.add_argument('--identity_reg_tau', type=float, default=0.0,
                   help='Width (in days) of the identity regularization zone. '
                        'Controls how smoothly displacement ramps up from zero '
                        'at gap=0. Tau=7 means the penalty is strong for gaps '
                        '< ~7 days and vanishes beyond ~14 days.')
    p.add_argument('--nll_warmup_epochs', type=int, default=0,
                   help='Number of initial epochs using MSE before switching '
                        'to NLL. Helps the mean head converge before the '
                        'variance head starts adapting. Try 10-20 if training '
                        'is unstable.')
    p.add_argument('--save_every', type=int, default=50)
    p.add_argument('--viz_every', type=int, default=10)
    p.add_argument('--analyze_only', action='store_true',
                   help='Skip training, just run variance analysis on existing model')
    p.add_argument('--flow_dir', type=str, default=None,
                   help='Path to trained OT flow model (required for --analyze_only)')
    args = p.parse_args()

    if args.analyze_only:
        assert args.flow_dir, "--flow_dir required with --analyze_only"
        analyze_only(args)
    else:
        assert args.ae_dir, "--ae_dir required for training"
        train(args)
