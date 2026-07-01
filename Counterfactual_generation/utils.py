"""Shared data-loading utilities for the trajectory-variance pipeline."""

import os
from pathlib import Path

import torch
from tqdm import tqdm

# Set the TRAJECTORY_VARIANCE_DATA_ROOT environment variable to point at
# your data directory before running any script.  See README for expected layout.
DATA_ROOT = Path(
    os.environ.get("TRAJECTORY_VARIANCE_DATA_ROOT", "data")
)


def load_spectrograms(bird: str) -> tuple:
    """Load precomputed Spectrograms_PadRight tensors for one bird.

    Used by ``train_ae.py --source precomputed`` and by exploratory
    acoustic-feature scripts.

    Returns: (specs, ages, lengths, norm_stats)
    """
    spec_dir = DATA_ROOT / bird / "Preprocess" / "Spectrograms_PadRight"
    metadata_path = spec_dir / f"{bird}_metadata.pt"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Spectrograms not found. Run compute_spectrograms.py first:\n"
            f"  python compute_spectrograms.py --bird {bird}"
        )
    
    metadata = torch.load(metadata_path)
    
    all_specs, all_ages, all_lengths = [], [], []
    
    for day_file in tqdm(metadata['day_files'], desc="Loading"):
        day_data = torch.load(spec_dir / day_file)
        all_specs.append(day_data['spectrograms'])
        ages = torch.full((day_data['n_samples'],), day_data['age'], dtype=torch.float32)
        all_ages.append(ages)
        all_lengths.append(day_data['lengths'])
    
    specs = torch.cat(all_specs)
    ages = torch.cat(all_ages)
    lengths = torch.cat(all_lengths)
    
    # Normalize
    specs = (specs - metadata['global_min']) / (metadata['global_max'] - metadata['global_min'] + 1e-6)
    
    norm_stats = {'global_min': metadata['global_min'], 'global_max': metadata['global_max']}
    
    print(f"Loaded {len(specs)} spectrograms. Age range: {ages.min():.0f}-{ages.max():.0f}")
    
    return specs, ages, lengths, norm_stats
