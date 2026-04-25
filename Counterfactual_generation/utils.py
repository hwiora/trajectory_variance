"""
Utility functions for Counterfactual Generation pipeline.
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ========================== Configuration ==========================
# Set the TRAJECTORY_VARIANCE_DATA_ROOT environment variable to point at
# your data directory before running any script.  See README for expected layout.
DATA_ROOT = Path(
    os.environ.get("TRAJECTORY_VARIANCE_DATA_ROOT", r"F:\FromDina\counterfactual_variance")
)


def load_spectrograms(bird: str) -> tuple:
    """
    Load pre-computed spectrograms from Spectrograms_PadRight.
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


def masked_l1_loss(recon_x, x, lengths):
    """
    L1 Loss masked by sequence length (content left, padding right).
    x: (B, C, T), lengths: (B,)
    """
    mask = torch.arange(x.size(2), device=x.device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(1).float()  # (B, 1, T)
    
    loss = (torch.abs(recon_x - x) * mask).sum() / (mask.sum() + 1e-6)
    return loss


def visualize_recon(model, val_loader, device, output_dir, epoch, num_samples=8):
    """Visualize reconstruction quality for both AE and VAE."""
    model.eval()
    
    val_batch = next(iter(val_loader))
    x_val = val_batch[0][:num_samples].to(device)
    lens_val = val_batch[2][:num_samples].to(device)
    
    with torch.no_grad():
        # Handle both AE and VAE
        output = model(x_val)
        if len(output) == 2:
            recon_raw, z = output
        else:
            recon_raw, z, mu, logvar = output
        
        # Apply mask
        mask = torch.arange(x_val.size(2), device=device)[None, :] < lens_val[:, None]
        mask = mask.unsqueeze(1).float()
        
        recon = recon_raw * mask
        recon_mse = ((recon - x_val)**2 * mask).sum() / (mask.sum() + 1e-6)
    
    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 5))
    
    vmin = min(x_val.min().item(), recon.min().item())
    vmax = max(x_val.max().item(), recon.max().item())
    
    for i in range(num_samples):
        axes[0, i].imshow(x_val[i].cpu().numpy(), aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'Len={lens_val[i].item()}', fontsize=9)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon[i].cpu().numpy(), aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        m_i = mask[i]
        mse_i = ((recon[i] - x_val[i])**2 * m_i).sum() / (m_i.sum() + 1e-6)
        axes[1, i].set_title(f'MSE={mse_i:.4f}', fontsize=9)
        axes[1, i].axis('off')
    
    plt.suptitle(f'Epoch {epoch+1} | Recon MSE: {recon_mse:.4f}', fontsize=12)
    plt.tight_layout()
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    plt.savefig(viz_dir / f'epoch_{epoch+1:03d}.png', dpi=100)
    plt.close()
    
    return recon_mse.item()
