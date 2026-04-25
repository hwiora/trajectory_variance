"""
Label vocalizations as song syllables vs calls using bout structure
+ within-bout latent diversity.

Algorithm:
  1. Within each recording file, sort vocalizations by onset time.
  2. Compute silence gaps between consecutive vocalizations.
  3. Chain consecutive vocalizations into bouts when gap < threshold.
  4. Bouts with >= min_bout_size vocalizations = candidate song bouts.
  5. For each candidate bout, compute within-bout latent diversity
     (mean pairwise L2 distance of autoencoder latent vectors).
  6. Bouts with diversity > threshold = song. Otherwise = call burst.

The diversity filter distinguishes song bouts (A-B-C-D, diverse syllable
types) from call bursts (tet-tet-tet, same call repeated).

Usage:
  # Step 1: Plot gap + diversity distributions (set thresholds empirically)
  python label_song_calls.py --bird R2915 --mode plot --ae_dir models/ae_R2915_20260217_091904

  # Step 2: Label with chosen thresholds
  python label_song_calls.py --bird R2915 --ae_dir models/ae_R2915_20260217_091904

  # Step 3: All birds (provide ae_dir per bird or use auto-detect)
  python label_song_calls.py --bird all
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from .utils import DATA_ROOT


# ═══════════════════════════ Bird configuration ═══════════════════════════

BIRD_CONFIG = {
    "R2915": {"hatch_datenum": 40674},
    "R4634": {"hatch_datenum": 41839},
    "R4951": {"hatch_datenum": 42086},
    "R5018": {"hatch_datenum": 42142},
}

MODELS_DIR = Path(__file__).parent / "models"


# ═══════════════════════════ Auto-detect AE directory ═══════════════════════════

def find_ae_dir(bird: str, ae_dir_arg: str = None, expected_n: int | None = None) -> Path:
    """Find latent directory (AE/VAE) for a bird.

    If ae_dir_arg is provided, use it. Otherwise, search both ae_{bird}_*
    and vae_{bird}_* directories containing latents.pt, and prefer one whose
    latent count matches expected_n when provided.
    """
    if ae_dir_arg:
        p = Path(ae_dir_arg)
        if not (p / "latents.pt").exists():
            raise FileNotFoundError(f"No latents.pt in {p}")
        return p

    candidates = sorted(MODELS_DIR.glob(f"ae_{bird}_*/latents.pt"))
    candidates += sorted(MODELS_DIR.glob(f"vae_{bird}_*/latents.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No ae_{bird}_*/latents.pt or vae_{bird}_*/latents.pt found in {MODELS_DIR}. "
            f"Provide --ae_dir explicitly."
        )

    # Prefer latent file whose sample count matches expected_n
    if expected_n is not None:
        matched = []
        for c in candidates:
            try:
                ld = torch.load(c, weights_only=True)
                if len(ld["z"]) == expected_n:
                    matched.append(c)
            except Exception:
                continue
        if matched:
            return sorted(matched, key=lambda p: p.stat().st_mtime)[-1].parent

    # Fallback: most recent across AE/VAE
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1].parent


# ═══════════════════════════ Data loading ═══════════════════════════

def load_h5_segments(bird: str):
    """Load segment metadata from H5."""
    h5_path = DATA_ROOT / bird / "Processed" / f"{bird}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        filenames = [
            v.decode() if isinstance(v, bytes) else str(v)
            for v in h5["files/filename"][:]
        ]
        data = {
            "segment_id":   h5["segments/segment_id"][:],
            "onset_sec":    h5["segments/onset_sec"][:],
            "duration_sec": h5["segments/duration_sec"][:],
            "file_id":      h5["segments/file_id"][:],
        }
        if "segments/cluster_id" in h5:
            data["cluster_id"] = h5["segments/cluster_id"][:]

    return data, filenames


def load_latents_pipeline_order(ae_dir: Path):
    """Load autoencoder latents in pipeline iteration order."""
    ld = torch.load(ae_dir / "latents.pt", weights_only=True)
    return ld["z"].numpy(), ld["ages"].numpy()


def load_latent_bundle(ae_dir: Path):
    """Load full latents bundle (z/ages/lengths and optional index metadata)."""
    return torch.load(ae_dir / "latents.pt", weights_only=True)


def load_h5_embeddings_by_h5_index(bird: str, data, embedding_key: str = "raw"):
    """Load embeddings from H5 and align them to H5 array index.

    Uses /embeddings/{embedding_key} and /embeddings/segment_id, then maps each
    embedding row to the corresponding H5 segment index via segment_id.

    Returns:
        emb_by_h5idx: (N_h5, D) float32 embeddings aligned to H5 indices.
    """
    h5_path = DATA_ROOT / bird / "Processed" / f"{bird}.h5"
    with h5py.File(h5_path, "r") as h5:
        if "embeddings" not in h5:
            raise KeyError(f"No /embeddings group in {h5_path}")
        if embedding_key not in h5["embeddings"]:
            raise KeyError(
                f"No /embeddings/{embedding_key} in {h5_path}. "
                f"Available: {list(h5['embeddings'].keys())}"
            )
        emb = h5["embeddings"][embedding_key][:]
        emb_sid = h5["embeddings"]["segment_id"][:]

    sid_to_h5idx = {int(data["segment_id"][i]): i for i in range(len(data["segment_id"]))}
    N = len(data["segment_id"])
    D = emb.shape[1]
    emb_by_h5idx = np.zeros((N, D), dtype=np.float32)
    filled = np.zeros(N, dtype=bool)

    for row_idx, sid in enumerate(emb_sid):
        h5_idx = sid_to_h5idx.get(int(sid), None)
        if h5_idx is not None:
            emb_by_h5idx[h5_idx] = emb[row_idx].astype(np.float32)
            filled[h5_idx] = True

    if not filled.all():
        missing = int((~filled).sum())
        raise RuntimeError(
            f"Embedding alignment incomplete: missing {missing} / {N} segments."
        )

    return emb_by_h5idx


def load_h5_umap_by_h5_index(bird: str, data):
    """Load segments/umap from H5 (already aligned to H5 index order)."""
    h5_path = DATA_ROOT / bird / "Processed" / f"{bird}.h5"
    with h5py.File(h5_path, "r") as h5:
        if "segments/umap" not in h5:
            raise KeyError(f"No /segments/umap in {h5_path}")
        umap = h5["segments/umap"][:].astype(np.float32)

    if len(umap) != len(data["segment_id"]):
        raise RuntimeError(
            f"UMAP length ({len(umap)}) != segment count ({len(data['segment_id'])})"
        )
    return umap


def build_h5_to_pipeline_map_from_latents(latent_bundle, data):
    """Build H5 <-> pipeline maps from latents.pt metadata.

    Preferred fields in latents.pt:
      - h5_idx: direct H5 row index per pipeline position
      - segment_id: segment IDs per pipeline position
    """
    if "h5_idx" in latent_bundle:
        h5_idx_arr = latent_bundle["h5_idx"]
        h5_idx_arr = h5_idx_arr.cpu().numpy() if torch.is_tensor(h5_idx_arr) else np.asarray(h5_idx_arr)
        pipeline_to_h5 = {int(pipe_idx): int(h5_idx) for pipe_idx, h5_idx in enumerate(h5_idx_arr)}
        h5_to_pipeline = {int(h5_idx): int(pipe_idx) for pipe_idx, h5_idx in pipeline_to_h5.items()}
        return h5_to_pipeline, pipeline_to_h5

    if "segment_id" in latent_bundle:
        sid_arr = latent_bundle["segment_id"]
        sid_arr = sid_arr.cpu().numpy() if torch.is_tensor(sid_arr) else np.asarray(sid_arr)
        sid_to_h5idx = {int(data["segment_id"][i]): i for i in range(len(data["segment_id"]))}

        pipeline_to_h5 = {}
        h5_to_pipeline = {}
        for pipe_idx, sid in enumerate(sid_arr):
            sid_i = int(sid)
            if sid_i in sid_to_h5idx:
                h5_idx = int(sid_to_h5idx[sid_i])
                pipeline_to_h5[int(pipe_idx)] = h5_idx
                h5_to_pipeline[h5_idx] = int(pipe_idx)

        if len(pipeline_to_h5) != len(sid_arr):
            raise RuntimeError(
                f"Latents segment_id mapping incomplete: mapped {len(pipeline_to_h5)} / {len(sid_arr)}"
            )
        return h5_to_pipeline, pipeline_to_h5

    raise RuntimeError(
        "latents.pt missing index metadata ('h5_idx' or 'segment_id'). "
        "Update latents with update_latents_index.py or re-encode with updated train_ae.py."
    )


# ═══════════════════════════ Gap computation ═══════════════════════════

def compute_within_file_gaps(data):
    """Compute silence gaps between consecutive vocalizations within each file."""
    by_file = defaultdict(list)
    n = len(data["segment_id"])
    for i in range(n):
        by_file[int(data["file_id"][i])].append(i)

    gaps = []
    for fid in sorted(by_file.keys()):
        indices = sorted(by_file[fid], key=lambda i: data["onset_sec"][i])
        for k in range(len(indices) - 1):
            i, j = indices[k], indices[k + 1]
            gap = data["onset_sec"][j] - (data["onset_sec"][i] + data["duration_sec"][i])
            if gap >= 0:
                gaps.append(gap)

    return np.array(gaps)


# ═══════════════════════════ Bout detection ═══════════════════════════

def detect_bouts_temporal(data, gap_threshold=0.2, min_bout_size=3):
    """Chain vocalizations into bouts based on temporal gaps.

    Returns list of bouts, where each bout is a list of H5 array indices.
    """
    by_file = defaultdict(list)
    n = len(data["segment_id"])
    for i in range(n):
        by_file[int(data["file_id"][i])].append(i)

    all_bouts = []

    for fid in sorted(by_file.keys()):
        indices = sorted(by_file[fid], key=lambda i: data["onset_sec"][i])

        current_bout = [indices[0]]
        for k in range(1, len(indices)):
            prev_i, curr_i = indices[k - 1], indices[k]
            gap = data["onset_sec"][curr_i] - (data["onset_sec"][prev_i] + data["duration_sec"][prev_i])

            if 0 <= gap < gap_threshold:
                current_bout.append(curr_i)
            else:
                all_bouts.append(current_bout)
                current_bout = [curr_i]
        all_bouts.append(current_bout)

    return all_bouts


def compute_bout_diversity(bouts, latents, h5_to_pipeline):
    """Compute within-bout latent diversity for each bout.

    Diversity = mean pairwise L2 distance of latent vectors within the bout.
    For singletons, diversity = 0.

    Returns array of diversity scores, one per bout.
    """
    diversities = np.zeros(len(bouts))

    for bi, bout in enumerate(bouts):
        if len(bout) < 2:
            diversities[bi] = 0.0
            continue

        # Get latent vectors for this bout (map H5 idx -> pipeline idx)
        pipe_indices = []
        for h5_idx in bout:
            if h5_idx in h5_to_pipeline:
                pipe_indices.append(h5_to_pipeline[h5_idx])

        if len(pipe_indices) < 2:
            diversities[bi] = 0.0
            continue

        z_bout = latents[pipe_indices]  # (n_bout, latent_dim)

        # Mean pairwise L2 distance
        # For efficiency, use: mean_pairwise = sqrt(2 * var) for isotropic,
        # but exact is fine for small bouts
        n = len(z_bout)
        total_dist = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(z_bout[i] - z_bout[j])
                n_pairs += 1

        diversities[bi] = total_dist / n_pairs if n_pairs > 0 else 0.0

    return diversities


def compute_bout_diversity_h5(bouts, emb_by_h5idx, metric="euclidean"):
    """Compute within-bout diversity from H5-aligned embeddings.

    Args:
        bouts: list of bouts in H5 index space
        emb_by_h5idx: (N_h5, D) embeddings aligned to H5 index
        metric: distance metric to use ('euclidean' or 'cosine')
    """
    diversities = np.zeros(len(bouts), dtype=np.float32)
    from scipy.spatial.distance import cosine

    for bi, bout in enumerate(bouts):
        if len(bout) < 2:
            continue

        z_bout = emb_by_h5idx[np.array(bout, dtype=np.int64)]
        n = len(z_bout)
        total_dist = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if metric == "cosine":
                    dist = cosine(z_bout[i], z_bout[j])
                    if np.isnan(dist): dist = 0.0
                else:
                    dist = np.linalg.norm(z_bout[i] - z_bout[j])
                total_dist += dist
                n_pairs += 1
        diversities[bi] = total_dist / n_pairs if n_pairs > 0 else 0.0

    return diversities


def label_with_diversity(bouts, diversities, min_bout_size=3,
                         diversity_threshold=None):
    """Label bouts as song or call based on size and diversity.

    Args:
        bouts: list of bouts (each bout = list of H5 array indices)
        diversities: array of diversity scores per bout
        min_bout_size: minimum bout size to be a candidate
        diversity_threshold: minimum diversity to qualify as song.
            If None, all bouts meeting min_bout_size are labeled song.

    Returns:
        is_song: dict mapping H5 array index -> bool
        bout_labels: list of (bout_idx, is_song_bout, size, diversity)
    """
    is_song = {}
    h5idx_to_bout_id = {}  # H5 array index -> song bout index (-1 for calls)
    bout_labels = []
    song_bout_counter = 0

    for bi, bout in enumerate(bouts):
        size = len(bout)
        div = diversities[bi]
        is_song_bout = (size >= min_bout_size)

        if is_song_bout and diversity_threshold is not None:
            is_song_bout = (div >= diversity_threshold)

        bout_labels.append((bi, is_song_bout, size, div))
        assigned_id = song_bout_counter if is_song_bout else -1
        if is_song_bout:
            song_bout_counter += 1
        for idx in bout:
            is_song[idx] = is_song_bout
            h5idx_to_bout_id[idx] = assigned_id

    return is_song, h5idx_to_bout_id, bout_labels


# ═══════════════════════════ Plotting ═══════════════════════════

def plot_distributions(bird, data, bouts, diversities, output_dir,
                       min_bout_size=3):
    """Plot gap distribution and bout diversity distribution."""
    if not HAS_PLT:
        print("matplotlib not available")
        return

    gaps = compute_within_file_gaps(data)

    # Separate diversity by bout size
    sizes = np.array([len(b) for b in bouts])
    large_mask = sizes >= min_bout_size
    div_large = diversities[large_mask]
    div_small = diversities[~large_mask & (sizes >= 2)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # ── Row 1: Gap distributions ──
    axes[0, 0].hist(gaps, bins=200, range=(0, 5), color="steelblue", edgecolor="none")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Gap (s)")
    axes[0, 0].set_ylabel("Count (log)")
    axes[0, 0].set_title(f"{bird}: Gap distribution (0-5s)")
    axes[0, 0].axvline(0.2, color="red", ls="--", alpha=0.7, label="200ms")
    axes[0, 0].legend()

    axes[0, 1].hist(gaps[gaps < 1], bins=200, color="steelblue", edgecolor="none")
    axes[0, 1].set_xlabel("Gap (s)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"{bird}: Gaps < 1s")
    axes[0, 1].axvline(0.2, color="red", ls="--", alpha=0.7, label="200ms")
    axes[0, 1].legend()

    axes[0, 2].hist(gaps[gaps < 0.3], bins=100, color="steelblue", edgecolor="none")
    axes[0, 2].set_xlabel("Gap (s)")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title(f"{bird}: Gaps < 300ms")
    axes[0, 2].axvline(0.2, color="red", ls="--", alpha=0.7, label="200ms")
    axes[0, 2].legend()

    # ── Row 2: Diversity distributions ──
    axes[1, 0].hist(div_large, bins=100, color="coral", edgecolor="none", alpha=0.7,
                     label=f"Bouts ≥{min_bout_size} (N={len(div_large)})")
    if len(div_small) > 0:
        axes[1, 0].hist(div_small, bins=100, color="gray", edgecolor="none", alpha=0.5,
                         label=f"Bouts 2 (N={len(div_small)})")
    axes[1, 0].set_xlabel("Mean pairwise L2 distance")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"{bird}: Bout diversity (all)")
    axes[1, 0].legend()

    # Zoomed diversity for large bouts
    axes[1, 1].hist(div_large, bins=100, color="coral", edgecolor="none")
    axes[1, 1].set_xlabel("Mean pairwise L2 distance")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(f"{bird}: Diversity of bouts ≥{min_bout_size}")

    # Diversity vs bout size scatter
    axes[1, 2].scatter(sizes[large_mask], div_large, s=2, alpha=0.3, color="coral")
    axes[1, 2].set_xlabel("Bout size")
    axes[1, 2].set_ylabel("Mean pairwise L2 distance")
    axes[1, 2].set_title(f"{bird}: Diversity vs bout size")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{bird}_distributions.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    # Print diversity percentiles for large bouts
    print(f"\n  Diversity of bouts ≥{min_bout_size} (N={len(div_large)}):")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{pct}: {np.percentile(div_large, pct):.3f}")


# ═══════════════════════════ Cluster comparison ═══════════════════════════

def compare_with_clusters(data, is_song_dict, bird: str):
    """Compare bout-based labels with WhisperSeg cluster labels."""
    if "cluster_id" not in data:
        print(f"\n{bird}: No cluster_id in H5, skipping cluster comparison")
        return

    cluster_id = data["cluster_id"]
    unique_clusters = np.unique(cluster_id)

    if len(unique_clusters) <= 1:
        print(f"\n{bird}: Only one cluster ({unique_clusters}), skipping comparison")
        return

    print(f"\n{'='*60}")
    print(f"{bird}: Bout labels vs WhisperSeg clusters")
    print(f"{'='*60}")
    print(f"  {'Cluster':>8}  {'N':>8}  {'Song%':>8}  {'Call%':>8}")
    print(f"  {'-'*40}")

    n = len(data["segment_id"])
    for c in unique_clusters:
        mask = cluster_id == c
        n_c = mask.sum()
        n_song_c = sum(1 for i in range(n) if mask[i] and is_song_dict.get(i, False))
        print(f"  {c:>8d}  {n_c:>8,}  {n_song_c/n_c*100:>7.1f}%  {(1-n_song_c/n_c)*100:>7.1f}%")


# ═══════════════════════════ Save labels ═══════════════════════════

def save_labels(bird: str, data, filenames, is_song_dict, h5idx_to_bout_id,
                bout_labels, stats, output_dir: Path):
    """Save labels as NPZ and summary as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(data["segment_id"])

    # Convert dicts to arrays in H5 order
    is_song_arr = np.array([is_song_dict.get(i, False) for i in range(n)], dtype=bool)
    bout_id_arr = np.array([h5idx_to_bout_id.get(i, -1) for i in range(n)], dtype=np.int32)

    # NPZ sorted by segment_id
    order = np.argsort(data["segment_id"])
    npz_path = output_dir / f"{bird}_song_labels.npz"
    np.savez(
        npz_path,
        segment_id=data["segment_id"][order],
        is_song=is_song_arr[order],
        bout_id=bout_id_arr[order],
        onset_sec=data["onset_sec"][order],
        duration_sec=data["duration_sec"][order],
        file_id=data["file_id"][order],
    )
    print(f"  Saved: {npz_path}")

    # Pipeline order: sorted day folders -> sorted wav filenames within each day.
    # Reconstruct from H5 data: each segment becomes {recording}_seg{NNN}.wav
    # grouped by day (derived from hatch_datenum + file datenum).
    hatch = BIRD_CONFIG[bird]["hatch_datenum"]

    # Build segment wav names and days from H5
    # Group segments by file, sort by onset within file to get seg index
    by_file = defaultdict(list)
    for i in range(n):
        by_file[int(data["file_id"][i])].append(i)

    seg_info = []  # (day, wav_name, h5_idx)
    for fid in sorted(by_file.keys()):
        fname = filenames[fid]
        stem = fname.rsplit(".", 1)[0] if fname.endswith(".wav") else fname
        # Day = datenum from filename - hatch_datenum
        # Filename format: {bird}_{datenum}_{m}_{d}_{h}_{min}_{s}.wav
        parts = fname.split("_")
        try:
            datenum = float(parts[1])
            day = int(round(datenum - hatch))
        except (IndexError, ValueError):
            day = 0

        indices = sorted(by_file[fid], key=lambda i: data["onset_sec"][i])
        for seg_within_file, h5_idx in enumerate(indices):
            wav_name = f"{stem}_seg{seg_within_file:03d}.wav"
            seg_info.append((day, wav_name, h5_idx))

    # Sort by (day, wav_name) to match pipeline iteration order
    seg_info.sort(key=lambda x: (x[0], x[1]))

    pipeline_is_song = np.array([is_song_arr[si[2]] for si in seg_info], dtype=bool)
    pipeline_bout_id = np.array([bout_id_arr[si[2]] for si in seg_info], dtype=np.int32)

    pipeline_path = output_dir / f"{bird}_song_labels_pipeline_order.npz"
    np.savez(
        pipeline_path,
        is_song=pipeline_is_song,
        bout_id=pipeline_bout_id,
    )
    print(f"  Saved: {pipeline_path}  ({len(pipeline_is_song):,} entries)")

    # JSON summary
    json_path = output_dir / f"{bird}_song_labels_summary.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {json_path}")


# ═══════════════════════════ Main logic ═══════════════════════════

def process_bird(bird: str, mode: str, gap_threshold: float, min_bout_size: int,
                 diversity_threshold: float, ae_dir_arg: str, output_dir: Path,
                 diversity_source: str = "ae", h5_embedding_key: str = "raw",
                 pca_dims: int = None, metric: str = "euclidean"):
    """Process a single bird."""
    print(f"\n{'='*60}")
    print(f"Processing {bird}")
    print(f"{'='*60}")

    data, filenames = load_h5_segments(bird)
    n = len(data["segment_id"])
    print(f"  Loaded {n:,} segments from {len(filenames):,} files")

    # Detect bouts (temporal)
    bouts = detect_bouts_temporal(data, gap_threshold=gap_threshold,
                                  min_bout_size=1)  # get ALL bouts for diversity
    sizes = np.array([len(b) for b in bouts])
    print(f"  Detected {len(bouts):,} temporal groups "
          f"({(sizes >= min_bout_size).sum():,} with ≥{min_bout_size} members)")

    # Compute diversity only if needed
    need_diversity = (diversity_threshold > 0) or (mode == "plot")
    if need_diversity:
        if diversity_source == "h5_embeddings":
            print(f"  Diversity source: H5 embeddings (/embeddings/{h5_embedding_key})")
            emb_by_h5idx = load_h5_embeddings_by_h5_index(
                bird, data, embedding_key=h5_embedding_key)
            if pca_dims is not None and pca_dims > 0:
                print(f"  Applying PCA to reduce embeddings to {pca_dims} dimensions...")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=pca_dims)
                emb_by_h5idx = pca.fit_transform(emb_by_h5idx)
            print(f"  Embeddings: {emb_by_h5idx.shape}")
            print(f"  Computing within-bout embedding diversity (metric={metric})...")
            diversities = compute_bout_diversity_h5(bouts, emb_by_h5idx, metric=metric)
            ae_dir = None
        elif diversity_source == "h5_umap":
            print("  Diversity source: H5 UMAP coordinates (/segments/umap)")
            umap_by_h5idx = load_h5_umap_by_h5_index(bird, data)
            print(f"  UMAP: {umap_by_h5idx.shape}")
            print("  Computing within-bout UMAP diversity...")
            diversities = compute_bout_diversity_h5(bouts, umap_by_h5idx)
            ae_dir = None
        else:
            ae_dir = find_ae_dir(bird, ae_dir_arg, expected_n=n)
            print(f"  Diversity source: AE/VAE latents ({ae_dir})")
            latent_bundle = load_latent_bundle(ae_dir)
            latents = latent_bundle["z"].cpu().numpy() if torch.is_tensor(latent_bundle["z"]) else latent_bundle["z"]
            print(f"  Latents: {latents.shape}")
            h5_to_pipeline, pipeline_to_h5 = build_h5_to_pipeline_map_from_latents(latent_bundle, data)
            print(f"  Mapped {len(h5_to_pipeline):,} / {n:,} segments to pipeline order")
            print("  Computing within-bout latent diversity...")
            diversities = compute_bout_diversity(bouts, latents, h5_to_pipeline)
    else:
        diversities = np.zeros(len(bouts))

    if mode == "plot":
        plot_distributions(bird, data, bouts, diversities, output_dir,
                           min_bout_size=min_bout_size)
        return

    # Label
    is_song_dict, h5idx_to_bout_id, bout_labels = label_with_diversity(
        bouts, diversities,
        min_bout_size=min_bout_size,
        diversity_threshold=diversity_threshold if diversity_threshold > 0 else None,
    )

    # Convert to array for stats
    is_song_arr = np.array([is_song_dict.get(i, False) for i in range(n)])
    n_song = is_song_arr.sum()

    stats = {
        "total_segments": n,
        "song_segments": int(n_song),
        "call_segments": int(n - n_song),
        "song_fraction": float(n_song / n),
        "gap_threshold": gap_threshold,
        "min_bout_size": min_bout_size,
        "diversity_threshold": diversity_threshold,
        "diversity_source": diversity_source,
        "h5_embedding_key": h5_embedding_key if diversity_source == "h5_embeddings" else None,
        "ae_dir": str(ae_dir) if (need_diversity and ae_dir is not None) else "none",
    }

    print(f"\n  Results (gap={gap_threshold}s, min_bout={min_bout_size}, "
          f"div_thresh={diversity_threshold}):")
    print(f"    Song: {n_song:,} ({n_song/n*100:.1f}%)")
    print(f"    Call: {n - n_song:,} ({(n - n_song)/n*100:.1f}%)")

    # Duration analysis
    dur = data["duration_sec"]
    for label, mask in [("Song", is_song_arr), ("Call", ~is_song_arr)]:
        d = dur[mask]
        if len(d) > 0:
            print(f"    {label}: N={len(d):,}  median={np.median(d)*1000:.0f}ms  "
                  f"mean={np.mean(d)*1000:.0f}ms")

    compare_with_clusters(data, is_song_dict, bird)
    save_labels(bird, data, filenames, is_song_dict, h5idx_to_bout_id, bout_labels, stats, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Label song vs calls using bout structure + latent diversity.")
    parser.add_argument("--bird", required=True,
                        help="Bird ID or 'all'")
    parser.add_argument("--mode", default="label", choices=["label", "plot"],
                        help="'plot' to visualize distributions, 'label' to save labels")
    parser.add_argument("--gap_threshold", type=float, default=0.2,
                        help="Max silence (s) to chain into same bout (default: 0.2)")
    parser.add_argument("--min_bout_size", type=int, default=3,
                        help="Min vocalizations per bout (default: 3)")
    parser.add_argument("--diversity_threshold", type=float, default=0.0,
                        help="Min within-bout diversity to qualify as song. "
                             "0 = no diversity filter (default: 0)")
    parser.add_argument("--ae_dir", type=str, default=None,
                        help="Autoencoder directory (auto-detected if not provided)")
    parser.add_argument("--diversity_source", type=str, default="ae",
                        choices=["ae", "h5_embeddings", "h5_umap"],
                        help="Feature source for diversity: ae latents, h5 embeddings, or h5 UMAP")
    parser.add_argument("--h5_embedding_key", type=str, default="raw",
                        help="Key under /embeddings in H5 when --diversity_source h5_embeddings")
    parser.add_argument("--pca_dims", type=int, default=None,
                        help="Reduce H5 embeddings to this many dimensions using PCA before computing diversity")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"],
                        help="Distance metric for within-bout diversity")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: gold_standard_labels/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("gold_standard_labels")
    birds = list(BIRD_CONFIG.keys()) if args.bird == "all" else [args.bird]

    for bird in birds:
        if bird not in BIRD_CONFIG:
            print(f"Unknown bird: {bird}")
            continue
        process_bird(bird, args.mode, args.gap_threshold, args.min_bout_size,
                     args.diversity_threshold, args.ae_dir, output_dir,
                     diversity_source=args.diversity_source,
                     h5_embedding_key=args.h5_embedding_key,
                     pca_dims=args.pca_dims,
                     metric=args.metric)


if __name__ == "__main__":
    main()
