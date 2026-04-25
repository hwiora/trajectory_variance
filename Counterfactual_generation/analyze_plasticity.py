"""
Analyze whether trajectory variance captures vocalization identity (plasticity)
or is confounded by duration.

Two critical tests:
  1. DURATION CONFOUND: Is variance just a proxy for length?
     → Partial correlation, within-duration-bin bimodality, variance-per-frame
  2. ACOUSTIC FEATURE CORRELATION: Does variance correlate with spectral
     complexity beyond what duration explains?
     → Spectral entropy, bandwidth, temporal modulation — all computable at scale

Usage:
    python Counterfactual_generation/analyze_plasticity.py \
        --flow_dir Counterfactual_generation/models/ot_flow_R4951_20260214_131853

    # All birds
    python Counterfactual_generation/analyze_plasticity.py --all_birds
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats as scipy_stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .train_ae import SpectrogramAE
from .utils import DATA_ROOT


# ========================== Acoustic Features ==========================

def _compute_features_for_batch(specs_batch, lengths_batch, start_idx, features):
    """Compute acoustic features for a batch of spectrograms in-place."""
    for j in range(len(specs_batch)):
        i = start_idx + j
        spec = specs_batch[j]  # (n_mels, n_time)
        L = int(lengths_batch[j].item())
        if L < 2:
            L = 2

        active = spec[:, :L].numpy()  # (n_mels, L)

        features['duration'][i] = L

        # Convert to pseudo-power (relative features are scale-invariant)
        power = np.exp(active * 5)
        power = np.maximum(power, 1e-10)

        features['mean_energy'][i] = power.mean()

        # Per-frame spectral distribution
        col_sums = power.sum(axis=0, keepdims=True) + 1e-10
        p_spec = power / col_sums

        # Spectral entropy
        frame_entropy = -np.sum(p_spec * np.log(p_spec + 1e-10), axis=0)
        features['spectral_entropy'][i] = frame_entropy.mean()

        # Spectral centroid
        freq_bins = np.arange(power.shape[0]).reshape(-1, 1)
        centroid_per_frame = (p_spec * freq_bins).sum(axis=0)
        features['spectral_centroid'][i] = centroid_per_frame.mean()

        # Bandwidth
        bw_per_frame = np.sqrt((p_spec * (freq_bins - centroid_per_frame[None, :])**2).sum(axis=0))
        features['bandwidth'][i] = bw_per_frame.mean()

        # Spectral flatness
        log_mean = np.mean(np.log(power + 1e-10), axis=0)
        arith_mean = np.mean(power, axis=0) + 1e-10
        flatness_per_frame = np.exp(log_mean) / arith_mean
        features['spectral_flatness'][i] = flatness_per_frame.mean()

        # Temporal entropy
        envelope = power.sum(axis=0)
        env_sum = envelope.sum() + 1e-10
        p_temp = envelope / env_sum
        features['temporal_entropy'][i] = -np.sum(p_temp * np.log(p_temp + 1e-10))


def compute_acoustic_features_streaming(bird, N=None):
    """
    Compute acoustic features by streaming day-files from disk.
    Never holds all spectrograms in memory — processes one day at a time.

    If N is given and the on-disk count differs, we use the on-disk count
    (with a warning) because latents.pt may have been encoded from a
    slightly different snapshot of the spectrogram files.
    """
    spec_dir = DATA_ROOT / bird / "Preprocess" / "Spectrograms_PadRight"
    metadata = torch.load(spec_dir / f"{bird}_metadata.pt", weights_only=True)
    g_min, g_max = metadata['global_min'], metadata['global_max']

    # First pass: count total samples on disk
    total_on_disk = 0
    for day_file in metadata['day_files']:
        day_data = torch.load(spec_dir / day_file, weights_only=True)
        total_on_disk += len(day_data['spectrograms'])

    actual_N = total_on_disk
    if N is not None and total_on_disk != N:
        print(f"  WARNING: spectrogram day-files have {total_on_disk} samples "
              f"but latents.pt has {N}. Using min({total_on_disk}, {N}).")
        actual_N = min(total_on_disk, N)

    features = {k: np.zeros(actual_N) for k in [
        'duration', 'spectral_entropy', 'spectral_centroid', 'bandwidth',
        'temporal_entropy', 'mean_energy', 'spectral_flatness',
    ]}

    offset = 0
    for day_file in tqdm(metadata['day_files'], desc='Computing acoustic features'):
        day_data = torch.load(spec_dir / day_file, weights_only=True)
        specs = day_data['spectrograms']
        lengths = day_data['lengths']
        # Normalize same as load_spectrograms
        specs = (specs - g_min) / (g_max - g_min + 1e-6)
        n_this = len(specs)
        if offset + n_this > actual_N:
            n_this = actual_N - offset
            specs = specs[:n_this]
            lengths = lengths[:n_this]
        if n_this <= 0:
            break
        _compute_features_for_batch(specs, lengths, offset, features)
        offset += n_this

    return features


def load_spectrograms_for_indices(bird, indices):
    """
    Load specific spectrograms by global index without loading all data.
    Returns dict mapping index -> (spec_tensor, length).
    """
    spec_dir = DATA_ROOT / bird / "Preprocess" / "Spectrograms_PadRight"
    metadata = torch.load(spec_dir / f"{bird}_metadata.pt")
    g_min, g_max = metadata['global_min'], metadata['global_max']

    indices_set = set(int(i) for i in indices)
    result = {}
    offset = 0
    for day_file in metadata['day_files']:
        day_data = torch.load(spec_dir / day_file)
        n = len(day_data['spectrograms'])
        # Check which requested indices fall in this day
        for idx in list(indices_set):
            if offset <= idx < offset + n:
                local_idx = idx - offset
                spec = day_data['spectrograms'][local_idx]
                spec = (spec - g_min) / (g_max - g_min + 1e-6)
                result[idx] = (spec, day_data['lengths'][local_idx])
                indices_set.discard(idx)
        offset += n
        if not indices_set:
            break

    return result


# ========================== Statistical Tests ==========================

def partial_correlation(x, y, z):
    """
    Partial correlation of x and y, controlling for z.
    All inputs are 1-D arrays of the same length.
    Returns (r_partial, p_value).
    """
    # Regress x on z
    slope_xz, intercept_xz, _, _, _ = scipy_stats.linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)

    # Regress y on z
    slope_yz, intercept_yz, _, _, _ = scipy_stats.linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)

    # Correlate residuals
    r, p = scipy_stats.pearsonr(resid_x, resid_y)
    return r, p


def hartigans_dip_test(data, n_boot=1000):
    """
    Hartigan's dip test for unimodality.
    Returns (dip_statistic, p_value).
    Uses bootstrap p-value estimation.
    """
    try:
        import diptest
        dip, pval = diptest.diptest(data)
        return dip, pval
    except ImportError:
        # Fallback: fit 1-vs-2 component GMM and compare BIC
        from sklearn.mixture import GaussianMixture
        data_2d = data.reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, random_state=42).fit(data_2d)
        gmm2 = GaussianMixture(n_components=2, random_state=42).fit(data_2d)
        bic1, bic2 = gmm1.bic(data_2d), gmm2.bic(data_2d)
        # Return BIC difference as proxy (negative = 2-component preferred)
        return bic1 - bic2, None  # No formal p-value without diptest


def select_duration_matched_pairs(durations, variances, n_pairs=10,
                                  max_duration_diff=2,
                                  min_gap_quantile=0.75,
                                  random_seed=42):
    """
    Select low/high variance exemplars that are truly duration-matched.

    Strategy:
      1) For each source duration, search candidates within ±max_duration_diff.
      2) Pair one low-variance item and one high-variance item from that window.
      3) Keep only pairs with variance gap >= global quantile threshold.
      4) Return top-n_pairs by variance gap.
    """
    rng = np.random.RandomState(random_seed)
    N = len(durations)

    global_gap_thresh = np.quantile(variances, min_gap_quantile) - np.quantile(variances, 1.0 - min_gap_quantile)
    if global_gap_thresh <= 0:
        global_gap_thresh = np.std(variances) * 0.5

    # Candidate source indices (subsample for efficiency on very large N)
    max_sources = min(25000, N)
    src_idx = rng.choice(N, max_sources, replace=False)

    pairs = []
    for idx in src_idx:
        d = durations[idx]
        mask = (durations >= d - max_duration_diff) & (durations <= d + max_duration_diff)
        cand = np.where(mask)[0]
        if len(cand) < 40:
            continue

        cand_var = variances[cand]
        low_thresh = np.quantile(cand_var, 0.10)
        high_thresh = np.quantile(cand_var, 0.90)

        low_cand = cand[cand_var <= low_thresh]
        high_cand = cand[cand_var >= high_thresh]
        if len(low_cand) == 0 or len(high_cand) == 0:
            continue

        low_idx = low_cand[np.argmin(variances[low_cand])]
        high_idx = high_cand[np.argmax(variances[high_cand])]
        gap = variances[high_idx] - variances[low_idx]
        dur_diff = abs(durations[high_idx] - durations[low_idx])

        if gap < global_gap_thresh:
            continue

        pairs.append((int(low_idx), int(high_idx), float(gap), float(dur_diff)))

    if not pairs:
        return []

    # Remove duplicates and keep best gap for each low/high anchor
    pairs.sort(key=lambda x: x[2], reverse=True)
    used_low = set()
    used_high = set()
    selected = []
    for low_idx, high_idx, gap, dur_diff in pairs:
        if low_idx in used_low or high_idx in used_high:
            continue
        selected.append((low_idx, high_idx, gap, dur_diff))
        used_low.add(low_idx)
        used_high.add(high_idx)
        if len(selected) >= n_pairs:
            break

    return selected


# ========================== Main Analysis ==========================

def analyze_bird(flow_dir, output_dir=None):
    """Run full plasticity analysis for one bird."""
    flow_dir = Path(flow_dir)
    config = json.load(open(flow_dir / 'config.json'))
    ae_dir = Path(config['ae_dir'])
    ae_config = json.load(open(ae_dir / 'config.json'))
    bird = config['bird']

    print(f'\n{"="*70}')
    print(f'PLASTICITY ANALYSIS: {bird}')
    print(f'{"="*70}')

    # Output directory
    if output_dir is None:
        output_dir = flow_dir / 'plasticity_analysis'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load variance results ---
    var_path = flow_dir / 'variance_results.json'
    if not var_path.exists():
        print(f"ERROR: {var_path} not found. Run train_ot_flow.py --analyze_only first.")
        return None
    vr = json.load(open(var_path))
    variances = np.array(vr['variances'])
    ages = np.array(vr['ages'])
    N = len(variances)
    print(f'  Loaded {N:,} variance scores')

    # --- Load latents (for lengths) ---
    ld = torch.load(ae_dir / 'latents.pt', weights_only=True)
    lengths = ld['lengths'].numpy()
    assert len(lengths) == N, f"Length mismatch: {len(lengths)} vs {N}"

    # --- Compute acoustic features (streaming, low memory) ---
    print('\n  Computing acoustic features (streaming per day-file)...')
    features = compute_acoustic_features_streaming(bird, N)

    # --- Load AE for decoding exemplars ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = SpectrogramAE(ae_config['n_mels'], ae_config['n_time'],
                       ae_config['latent_dim']).to(device)
    ae.load_state_dict(torch.load(ae_dir / 'best.pt', map_location=device,
                                  weights_only=True))
    ae.eval()

    latents = ld['z']
    z_mean = torch.tensor(config['z_mean']).unsqueeze(0).to(device)
    z_std = torch.tensor(config['z_std']).unsqueeze(0).to(device)
    latents_norm = (latents - z_mean.cpu()) / z_std.cpu()
    age_min, age_max = config['age_min'], config['age_max']

    # --- Load transport model for exemplar generation ---
    from Counterfactual_generation.models.flow import load_transport_model
    model = load_transport_model(config, device, weights_path=flow_dir / 'best.pt')

    # ================================================================
    # TEST 1: DURATION CONFOUND
    # ================================================================
    print(f'\n{"─"*50}')
    print('TEST 1: Duration Confound Analysis')
    print(f'{"─"*50}')

    durations = features['duration']

    # Raw correlation: variance vs duration
    r_raw, p_raw = scipy_stats.pearsonr(variances, durations)
    rho_raw, p_rho = scipy_stats.spearmanr(variances, durations)
    print(f'  Variance ~ Duration:')
    print(f'    Pearson  r = {r_raw:.3f} (p = {p_raw:.2e})')
    print(f'    Spearman ρ = {rho_raw:.3f} (p = {p_rho:.2e})')

    # Variance per frame (normalize out duration)
    var_per_frame = variances / (durations + 1e-6)

    # Within-duration-bin analysis
    dur_bins = np.percentile(durations, [0, 25, 50, 75, 100])
    dur_bin_labels = []
    dur_bin_variances = []
    dur_bin_var_per_frame = []
    for i in range(len(dur_bins) - 1):
        lo, hi = dur_bins[i], dur_bins[i + 1]
        if i < len(dur_bins) - 2:
            mask = (durations >= lo) & (durations < hi)
        else:
            mask = (durations >= lo) & (durations <= hi)
        label = f'{lo:.0f}-{hi:.0f}'
        dur_bin_labels.append(label)
        dur_bin_variances.append(variances[mask])
        dur_bin_var_per_frame.append(var_per_frame[mask])
        print(f'  Duration bin {label} frames (N={mask.sum():,}): '
              f'var mean={variances[mask].mean():.3f}, std={variances[mask].std():.3f}, '
              f'CV={variances[mask].std()/(variances[mask].mean()+1e-8):.3f}')

    # Bimodality tests within duration bins
    print(f'\n  Bimodality tests (within duration bins):')
    for label, bv in zip(dur_bin_labels, dur_bin_variances):
        if len(bv) < 50:
            print(f'    Bin {label}: too few samples ({len(bv)})')
            continue
        dip_stat, dip_p = hartigans_dip_test(bv)
        if dip_p is not None:
            print(f'    Bin {label}: dip={dip_stat:.4f}, p={dip_p:.4f} '
                  f'{"** BIMODAL **" if dip_p < 0.05 else "(unimodal)"}')
        else:
            print(f'    Bin {label}: ΔBIC(1-vs-2 GMM)={dip_stat:.1f} '
                  f'{"** 2-component preferred **" if dip_stat > 10 else "(1-component sufficient)"}')

    # Overall bimodality test
    print(f'\n  Overall bimodality test:')
    dip_all, dip_p_all = hartigans_dip_test(variances)
    if dip_p_all is not None:
        print(f'    dip={dip_all:.4f}, p={dip_p_all:.4f}')
    else:
        print(f'    ΔBIC(1-vs-2 GMM)={dip_all:.1f}')

    # Bimodality test on variance-per-frame (duration-normalized)
    print(f'\n  Bimodality test on variance/frame (duration-normalized):')
    dip_vpf, dip_p_vpf = hartigans_dip_test(var_per_frame)
    if dip_p_vpf is not None:
        print(f'    dip={dip_vpf:.4f}, p={dip_p_vpf:.4f}')
    else:
        print(f'    ΔBIC(1-vs-2 GMM)={dip_vpf:.1f}')

    # ================================================================
    # TEST 2: ACOUSTIC FEATURE CORRELATIONS
    # ================================================================
    print(f'\n{"─"*50}')
    print('TEST 2: Acoustic Feature Correlations')
    print(f'{"─"*50}')

    feature_names = ['spectral_entropy', 'spectral_centroid', 'bandwidth',
                     'temporal_entropy', 'mean_energy', 'spectral_flatness']

    corr_results = {}
    print(f'\n  {"Feature":<22} {"r(var)":<10} {"ρ(var)":<10} '
          f'{"r_partial":<12} {"p_partial":<12}')
    print(f'  {"─"*66}')

    for fname in feature_names:
        fvals = features[fname]

        # Raw correlations with variance
        r, p = scipy_stats.pearsonr(variances, fvals)
        rho, p_rho = scipy_stats.spearmanr(variances, fvals)

        # Partial correlation controlling for duration
        r_part, p_part = partial_correlation(variances, fvals, durations)

        corr_results[fname] = {
            'pearson_r': float(r), 'pearson_p': float(p),
            'spearman_rho': float(rho), 'spearman_p': float(p_rho),
            'partial_r': float(r_part), 'partial_p': float(p_part),
        }
        sig = '***' if p_part < 0.001 else '**' if p_part < 0.01 else '*' if p_part < 0.05 else ''
        print(f'  {fname:<22} {r:>+.3f}     {rho:>+.3f}     '
              f'{r_part:>+.3f}       {p_part:.2e} {sig}')

    # Variance explained by duration alone vs duration + features
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X_dur = durations.reshape(-1, 1)
    reg_dur = LinearRegression().fit(X_dur, variances)
    r2_dur = r2_score(variances, reg_dur.predict(X_dur))

    X_all = np.column_stack([durations] + [features[f] for f in feature_names])
    reg_all = LinearRegression().fit(X_all, variances)
    r2_all = r2_score(variances, reg_all.predict(X_all))

    print(f'\n  Variance explained (R²):')
    print(f'    Duration only:              {r2_dur:.3f}')
    print(f'    Duration + acoustic feats:  {r2_all:.3f}')
    print(f'    → Acoustic features add:    {r2_all - r2_dur:.3f} '
          f'({(r2_all - r2_dur) / (1 - r2_dur + 1e-8) * 100:.1f}% of remaining variance)')

    # ================================================================
    # TEST 3: DURATION-MATCHED EXEMPLARS
    # ================================================================
    print(f'\n{"─"*50}')
    print('TEST 3: Duration-Matched Exemplar Comparison')
    print(f'{"─"*50}')

    # Strict duration-matched pairs with enforced variance gap
    matched_pairs = select_duration_matched_pairs(
        durations, variances,
        n_pairs=10,
        max_duration_diff=2,
        min_gap_quantile=0.75,
        random_seed=42,
    )
    if len(matched_pairs) < 5:
        print('  WARNING: Few strict matched pairs found; relaxing duration tolerance to ±3 frames.')
        matched_pairs = select_duration_matched_pairs(
            durations, variances,
            n_pairs=10,
            max_duration_diff=3,
            min_gap_quantile=0.70,
            random_seed=42,
        )

    if not matched_pairs:
        raise RuntimeError('No duration-matched pairs found with meaningful variance gap.')

    low_var_idx = np.array([p[0] for p in matched_pairs], dtype=int)
    high_var_idx = np.array([p[1] for p in matched_pairs], dtype=int)
    pair_gaps = np.array([p[2] for p in matched_pairs], dtype=float)
    pair_diffs = np.array([p[3] for p in matched_pairs], dtype=float)

    print(f'  Matched pairs found: {len(matched_pairs)}')
    print(f'  Duration difference in pairs: mean={pair_diffs.mean():.2f} frames, '
          f'max={pair_diffs.max():.0f} frames')
    print(f'  Variance gap in pairs: mean={pair_gaps.mean():.3f}, '
          f'min={pair_gaps.min():.3f}, max={pair_gaps.max():.3f}')

    print(f'\n  Duration-matched exemplar pairs (low vs high):')
    for k, (li, hi, gap, dd) in enumerate(matched_pairs[:5], 1):
        print(f'    Pair {k}: '
              f'low idx={li} (dur={durations[li]:.0f}, var={variances[li]:.4f}, age={ages[li]:.0f}d)  |  '
              f'high idx={hi} (dur={durations[hi]:.0f}, var={variances[hi]:.4f}, age={ages[hi]:.0f}d)  '
              f'[Δdur={dd:.0f}, Δvar={gap:.3f}]')

    # ================================================================
    # PLOTTING
    # ================================================================
    print(f'\n  Generating figures...')

    # Compute residual variance (duration regressed out) for better visualization
    slope_vd, intercept_vd, _, _, _ = scipy_stats.linregress(durations, variances)
    resid_variance = variances - (slope_vd * durations + intercept_vd)

    # --- Figure 1: Duration confound ---
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    # 1a: Variance vs Duration (2D density, not scatter)
    ax = fig.add_subplot(gs[0, 0])
    ax.hexbin(durations, variances, gridsize=60, cmap='YlGnBu', mincnt=1,
              rasterized=True)
    # Regression line
    dur_range = np.linspace(durations.min(), durations.max(), 100)
    ax.plot(dur_range, slope_vd * dur_range + intercept_vd, 'r-', lw=2,
            label=f'r={r_raw:.2f}')
    ax.set_xlabel('Duration (frames)')
    ax.set_ylabel('Trajectory Variance')
    ax.set_title(f'Variance vs Duration\nr={r_raw:.3f}, ρ={rho_raw:.3f}')
    ax.legend(fontsize=8)

    # 1b: Variance histogram (overall)
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(variances, bins=80, color='teal', alpha=0.7, edgecolor='none',
            density=True)
    ax.axvline(np.median(variances), color='red', ls='--',
               label=f'Median={np.median(variances):.2f}')
    ax.set_xlabel('Trajectory Variance')
    ax.set_ylabel('Density')
    ax.set_title(f'Overall Distribution\n(dip p={dip_p_all:.3f})')
    ax.legend(fontsize=8)

    # 1c: Residual variance histogram (duration regressed out)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(resid_variance, bins=80, color='coral', alpha=0.7, edgecolor='none',
            density=True)
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Residual Variance\n(duration regressed out)')
    ax.set_ylabel('Density')
    ax.set_title('Variance After Removing\nDuration Effect')

    # 1d: R² comparison
    ax = fig.add_subplot(gs[0, 3])
    bars = ax.bar(['Duration\nonly', 'Duration +\nAcoustic'], [r2_dur, r2_all],
                  color=['gray', 'teal'], alpha=0.8, edgecolor='none')
    ax.set_ylabel('R² (variance explained)')
    ax.set_title('What Explains Variance?')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, [r2_dur, r2_all]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Row 2: Within-duration-bin histograms (key evidence against confound)
    for b in range(min(4, len(dur_bin_labels))):
        ax = fig.add_subplot(gs[1, b])
        bv = dur_bin_variances[b]
        if len(bv) > 10:
            ax.hist(bv, bins=40, color=plt.cm.viridis(b / 4), alpha=0.7,
                    edgecolor='none', density=True)
            ax.axvline(np.mean(bv), color='red', ls='--', alpha=0.7)
            # Show within-bin bimodality
            dip_b, dip_p_b = hartigans_dip_test(bv)
            bm_label = f'dip p={dip_p_b:.3f}' if dip_p_b is not None else f'ΔBIC={dip_b:.0f}'
            ax.text(0.95, 0.95, bm_label, transform=ax.transAxes, fontsize=7,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_xlabel('Variance')
        if b == 0:
            ax.set_ylabel('Density')
        ax.set_title(f'Duration {dur_bin_labels[b]}f\n(N={len(bv):,})', fontsize=9)

    # Row 3: Feature correlations (top 4, as 2D density plots)
    partial_rs_sorted = [(fname, abs(corr_results[fname]['partial_r']))
                         for fname in feature_names]
    partial_rs_sorted.sort(key=lambda x: x[1], reverse=True)
    top_features = [p[0] for p in partial_rs_sorted[:4]]

    # Nicer labels mapping
    feat_display = {
        'spectral_flatness': 'Spectral Flatness\n(≈ Wiener Entropy)',
        'spectral_entropy': 'Spectral Entropy',
        'spectral_centroid': 'Spectral Centroid\n(mean freq bin)',
        'temporal_entropy': 'Temporal Entropy',
        'bandwidth': 'Bandwidth',
        'mean_energy': 'Mean Energy',
    }

    for i, fname in enumerate(top_features):
        ax = fig.add_subplot(gs[2, i])
        fvals = features[fname]
        ax.hexbin(fvals, variances, gridsize=60, cmap='YlOrRd', mincnt=1,
                  rasterized=True)
        r_part = corr_results[fname]['partial_r']
        ax.set_xlabel(feat_display.get(fname, fname))
        ax.set_ylabel('Variance' if i == 0 else '')
        ax.set_title(f'r_partial={r_part:+.3f}\n(controlling for duration)',
                     fontsize=9)

    fig.suptitle(f'{bird} — Plasticity vs Duration Confound Analysis',
                 fontsize=14)
    fig1_path = output_dir / 'duration_confound_analysis.png'
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fig1_path}')

    # --- Figure 2: Duration-matched exemplars ---
    # Load only the specific spectrograms we need for exemplars
    all_exemplar_idx = np.unique(np.concatenate([low_var_idx, high_var_idx]))
    exemplar_specs = load_spectrograms_for_indices(bird, all_exemplar_idx)
    _plot_exemplars(fig_path=output_dir / 'duration_matched_exemplars.png',
                    low_idx=low_var_idx, high_idx=high_var_idx,
                    pair_gaps=pair_gaps, pair_diffs=pair_diffs,
                    exemplar_specs=exemplar_specs,
                    lengths_arr=durations, variances=variances,
                    ages=ages, features=features, bird=bird,
                    # For counterfactual generation
                    model=model, ae=ae, latents_norm=latents_norm,
                    z_mean=z_mean, z_std=z_std,
                    age_min=age_min, age_max=age_max,
                    ae_config=ae_config, device=device)

    # --- Save results ---
    results = {
        'bird': bird,
        'n_samples': int(N),
        'duration_confound': {
            'pearson_r': float(r_raw),
            'spearman_rho': float(rho_raw),
            'r2_duration_only': float(r2_dur),
            'r2_duration_plus_acoustic': float(r2_all),
            'r2_improvement': float(r2_all - r2_dur),
        },
        'bimodality': {
            'overall_dip': float(dip_all),
            'overall_dip_p': float(dip_p_all) if dip_p_all is not None else None,
            'var_per_frame_dip': float(dip_vpf),
            'var_per_frame_dip_p': float(dip_p_vpf) if dip_p_vpf is not None else None,
        },
        'feature_correlations': corr_results,
        'duration_matched_analysis': {
            'n_pairs': int(len(matched_pairs)),
            'pair_duration_diff_mean': float(pair_diffs.mean()),
            'pair_duration_diff_max': float(pair_diffs.max()),
            'pair_variance_gap_mean': float(pair_gaps.mean()),
            'pair_variance_gap_min': float(pair_gaps.min()),
            'pair_variance_gap_max': float(pair_gaps.max()),
        },
    }
    results_path = output_dir / 'plasticity_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved: {results_path}')

    # --- Console summary ---
    print(f'\n{"="*70}')
    print(f'SUMMARY: {bird}')
    print(f'{"="*70}')
    print(f'  Duration explains {r2_dur*100:.1f}% of variance (R²={r2_dur:.3f})')
    print(f'  Duration + acoustics explains {r2_all*100:.1f}% (R²={r2_all:.3f})')
    print(f'  → Acoustic features add {(r2_all-r2_dur)*100:.1f}% beyond duration')
    if r2_dur > 0.7:
        print(f'  ⚠ WARNING: Duration dominates. Variance may be a duration proxy.')
    elif r2_dur > 0.4:
        print(f'  ⚠ CAUTION: Duration explains substantial variance. '
              f'Control for it in all analyses.')
    else:
        print(f'  ✓ Duration is NOT the primary driver of variance.')

    pcorr_sig = [(fname, corr_results[fname]['partial_r'])
                 for fname in feature_names
                 if corr_results[fname]['partial_p'] < 0.001]
    if pcorr_sig:
        print(f'  Significant partial correlations (p<0.001):')
        for fname, r_part in sorted(pcorr_sig, key=lambda x: abs(x[1]), reverse=True):
            print(f'    {fname}: r_partial = {r_part:+.3f}')
    else:
        print(f'  No significant partial correlations after controlling for duration.')

    return results


@torch.no_grad()
def _plot_exemplars(fig_path, low_idx, high_idx, pair_gaps, pair_diffs, exemplar_specs, lengths_arr,
                    variances, ages, features, bird, model, ae, latents_norm,
                    z_mean, z_std, age_min, age_max, ae_config, device):
    """
    Plot duration-matched exemplars: 5 low-variance and 5 high-variance,
    each with source spectrogram + counterfactuals at 3 ages.
    Shows that variance captures vocalization identity, not just duration.
    """
    n_exemplars = 5
    target_ages_days = np.array([40, 70, 100])
    low_sel = low_idx[:n_exemplars]
    high_sel = high_idx[-n_exemplars:][::-1]  # highest first

    fig, axes = plt.subplots(2 * n_exemplars, 1 + len(target_ages_days),
                             figsize=(14, 2.2 * 2 * n_exemplars))

    data_mean = ae_config['data_mean']
    data_std = ae_config['data_std']

    for group_i, (group_idx, group_label) in enumerate([
        (low_sel, 'LOW variance (age-invariant)'),
        (high_sel, 'HIGH variance (age-variant)')
    ]):
        for ei, idx in enumerate(group_idx):
            row = group_i * n_exemplars + ei

            # Source spectrogram (from loaded exemplar specs)
            src_spec, src_len = exemplar_specs[idx]
            src_spec_np = src_spec.numpy()
            L = int(src_len.item()) if torch.is_tensor(src_len) else int(src_len)
            src_crop = src_spec_np[:, :L]

            ax = axes[row, 0]
            ax.imshow(src_crop, aspect='auto', origin='lower', cmap='magma')
            pair_i = ei
            pair_tag = ''
            if pair_i < len(pair_gaps):
                pair_tag = f', Δdur={pair_diffs[pair_i]:.0f}, Δvar={pair_gaps[pair_i]:.2f}'
            ax.set_title(f'Source (age={ages[idx]:.0f}d, L={L})\n'
                         f'var={variances[idx]:.3f}{pair_tag}, '
                         f'SE={features["spectral_entropy"][idx]:.2f}',
                         fontsize=7)
            ax.set_yticks([])
            if row == 0:
                ax.set_ylabel(group_label, fontsize=8, fontweight='bold')
            elif row == n_exemplars:
                ax.set_ylabel(group_label, fontsize=8, fontweight='bold')

            # Counterfactuals at target ages
            z_src = latents_norm[idx].unsqueeze(0).to(device)
            # Age normalization with margin (backward compat)
            _m = config.get('age_margin', 0.0) * (age_max - age_min)
            _norm_min, _norm_max = age_min - _m, age_max + _m
            age_src_norm = (ages[idx] - _norm_min) / (_norm_max - _norm_min)

            for ti, ta in enumerate(target_ages_days):
                age_tgt_norm = (ta - _norm_min) / (_norm_max - _norm_min)
                c = torch.tensor([[age_src_norm, age_tgt_norm]],
                                 device=device, dtype=torch.float32)
                z_cf = model.generate(z_src, c)
                spec_cf = ae.decode(z_cf * z_std + z_mean).squeeze(0).cpu()
                spec_cf = (spec_cf * data_std + data_mean).numpy()
                spec_cf_crop = spec_cf[:, :L]

                ax = axes[row, 1 + ti]
                ax.imshow(spec_cf_crop, aspect='auto', origin='lower', cmap='magma')
                marker = '→' if ta != ages[idx] else '='
                ax.set_title(f'{marker} {ta:.0f}d', fontsize=7)
                ax.set_yticks([]); ax.set_xticks([])

    fig.suptitle(f'{bird} — Duration-Matched Exemplars\n'
                 f'Top: age-invariant (calls?), Bottom: age-variant (syllables?)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fig_path}')


# ========================== Multi-bird ==========================

FLOW_DIRS = {
    'R2915': 'Counterfactual_generation/models/ot_flow_R2915_20260214_111329',
    'R4634': 'Counterfactual_generation/models/ot_flow_R4634_20260214_120822',
    'R4951': 'Counterfactual_generation/models/ot_flow_R4951_20260214_131853',
    'R5018': 'Counterfactual_generation/models/ot_flow_R5018_20260214_144103',
}


def analyze_all_birds():
    """Run analysis for all birds and produce summary."""
    all_results = {}
    for bird, flow_dir in FLOW_DIRS.items():
        print(f'\n\n{"#"*70}')
        print(f'# {bird}')
        print(f'{"#"*70}')
        result = analyze_bird(flow_dir)
        if result:
            all_results[bird] = result

    if not all_results:
        print("No results.")
        return

    # Cross-bird summary
    print(f'\n\n{"="*70}')
    print('CROSS-BIRD SUMMARY')
    print(f'{"="*70}')
    print(f'\n  {"Bird":<8} {"R²(dur)":<10} {"R²(all)":<10} {"Δ":<10} '
          f'{"DurConf?":<12} {"Bimodal?":<12}')
    print(f'  {"─"*62}')
    for bird, r in all_results.items():
        dc = r['duration_confound']
        bm = r['bimodality']
        dur_flag = '⚠ YES' if dc['r2_duration_only'] > 0.7 else \
                   '~ partial' if dc['r2_duration_only'] > 0.4 else '✓ NO'
        bm_flag = '✓ YES' if (bm.get('overall_dip_p') is not None and bm['overall_dip_p'] < 0.05) \
                  else '? (check BIC)' if bm.get('overall_dip_p') is None \
                  else '✗ NO'
        print(f'  {bird:<8} {dc["r2_duration_only"]:<10.3f} '
              f'{dc["r2_duration_plus_acoustic"]:<10.3f} '
              f'{dc["r2_improvement"]:<10.3f} {dur_flag:<12} {bm_flag:<12}')

    # Save cross-bird summary
    summary_path = Path('Counterfactual_generation/models/plasticity_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Saved cross-bird summary: {summary_path}')


# ========================== CLI ==========================

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Plasticity Analysis: Duration Confound + Acoustic Features')
    p.add_argument('--flow_dir', type=str, default=None,
                   help='Path to trained OT flow model directory')
    p.add_argument('--all_birds', action='store_true',
                   help='Run analysis for all 4 birds')
    p.add_argument('--output_dir', type=str, default=None)
    args = p.parse_args()

    if args.all_birds:
        analyze_all_birds()
    elif args.flow_dir:
        analyze_bird(args.flow_dir, output_dir=args.output_dir)
    else:
        print("Specify --flow_dir or --all_birds")
