"""
Phase 1: Train a convolutional autoencoder (AE or VAE) on spectrograms.

Supports both standard AE (pure MSE) and VAE (MSE + KL divergence).
After training, encodes ALL spectrograms and saves latent codes to disk.

Data sources:
  --source precomputed  : Load from Spectrograms_PadRight/*.pt (80 mel, 100 frames)
  --source h5           : Load per-segment spectrograms directly from H5 file
                          (123 freq bins, variable time -> padded to --n_time)

Usage:
    python train_ae.py --bird R2915 --source h5
    python train_ae.py --bird R2915 --source h5 --model_type vae --kl_weight 1e-3

Output (in models/{ae|vae}_<bird>_<timestamp>/):
    - best.pt          : best model weights
    - config.json      : training config
    - latents.pt       : {z, ages, lengths} for all spectrograms
    - viz_recon_*.png  : reconstruction quality checks
"""

import argparse
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None
import random
import h5py

from .utils import DATA_ROOT

BIRD_CONFIG = {
    "R2915": {"hatch_datenum": 40674},
    "R4634": {"hatch_datenum": 41839},
    "R4951": {"hatch_datenum": 42086},
    "R5018": {"hatch_datenum": 42142},
}


# ============================== Data Loading ==============================

def load_spectrograms_precomputed(bird):
    """Load from pre-computed Spectrograms_PadRight (legacy pipeline)."""
    from Counterfactual_generation.utils import load_spectrograms
    specs, ages, lengths, _ = load_spectrograms(bird)
    return specs, ages, lengths, None, None


def load_spectrograms_h5(bird, n_time=100):
    """Load per-segment spectrograms directly from H5.

    Each segment is sliced from the full-recording spectrogram using
    onset_sec and duration_sec. The fps is read from H5 parameters
    (sr / hop_length).

    Returns segments in **pipeline order** (sorted day -> sorted filename
    -> sorted onset within file), matching the order produced by the
    precomputed spectrogram pipeline.

    Returns:
        specs:      (N_segments, n_freq, n_time) float32 tensor
        ages:       (N_segments,) float32 tensor (days post hatch)
        lengths:    (N_segments,) long tensor (original time frames before pad/crop)
        h5_idx:     (N_segments,) long tensor (H5 row index in pipeline order)
        segment_id: (N_segments,) long tensor (segment IDs in pipeline order)
    """
    hatch = BIRD_CONFIG[bird]["hatch_datenum"]
    h5_path = DATA_ROOT / bird / "Processed" / f"{bird}.h5"

    with h5py.File(h5_path, "r") as h5:
        n_segs = len(h5["segments/segment_id"][:])
        segment_ids = h5["segments/segment_id"][:]
        onset_sec = h5["segments/onset_sec"][:]
        duration_sec = h5["segments/duration_sec"][:]
        file_ids = h5["segments/file_id"][:]
        filenames = [
            v.decode() if isinstance(v, bytes) else str(v)
            for v in h5["files/filename"][:]
        ]

        # Read spectrogram parameters from H5 attributes
        sr = h5["parameters"].attrs.get("sr_processing",
             h5["parameters"].attrs.get("audio_sr", 32000))
        hop = h5["parameters"].attrs.get("spec_hop_length",
              h5["parameters"].attrs.get("hop_length", 128))
        fps = sr / hop
        n_freq = h5["spectrograms/0"].shape[0]
        print(f"  H5 spectrograms: {n_freq} freq bins, sr={sr}, hop={hop}, fps={fps:.1f}")

        # Map spectrogram index -> file_id (in case keys aren't identity)
        spec_file_ids = h5["spectrograms/file_id"][:]
        fid_to_spec_key = {int(spec_file_ids[i]): str(i)
                           for i in range(len(spec_file_ids))}

        # Group segments by file for efficient reading
        by_file = defaultdict(list)
        for i in range(n_segs):
            by_file[int(file_ids[i])].append(i)

        # Build pipeline order: sorted by (day, filename, onset_within_file)
        seg_order = []
        for fid, seg_indices in by_file.items():
            fname = filenames[fid]
            stem = fname.rsplit(".", 1)[0] if fname.endswith(".wav") else fname
            parts = fname.split("_")
            try:
                datenum = float(parts[1])
                day = int(round(datenum - hatch))
            except (IndexError, ValueError):
                day = 0
            day_str = f"{day:03d}"  # zero-pad for consistent lexicographic sorting
            for idx in seg_indices:
                seg_order.append((day_str, stem, onset_sec[idx], idx))

        seg_order.sort(key=lambda x: (x[0], x[1], x[2]))
        pipeline_indices = [x[3] for x in seg_order]
        pipeline_ages = torch.tensor([float(int(x[0])) for x in seg_order],
                                      dtype=torch.float32)
        pipeline_h5_idx = torch.tensor(pipeline_indices, dtype=torch.long)
        pipeline_segment_ids = torch.tensor(segment_ids[pipeline_indices], dtype=torch.long)

        # Pre-allocate output (in pipeline order)
        all_specs = torch.zeros(n_segs, n_freq, n_time, dtype=torch.float32)
        all_lengths = torch.zeros(n_segs, dtype=torch.long)

        # Build reverse map: h5_idx -> pipeline_pos
        h5_to_pipe = {h5_idx: pipe_pos
                       for pipe_pos, h5_idx in enumerate(pipeline_indices)}

        # Extract per-segment spectrograms
        for fid in tqdm(sorted(by_file.keys()), desc="Loading H5 specs"):
            spec_key = fid_to_spec_key.get(fid)
            if spec_key is None or spec_key not in h5["spectrograms"]:
                continue
            full_spec = h5[f"spectrograms/{spec_key}"][:].astype(np.float32)
            n_frames = full_spec.shape[1]

            for h5_idx in by_file[fid]:
                pipe_pos = h5_to_pipe[h5_idx]

                start_frame = int(round(onset_sec[h5_idx] * fps))
                end_frame = int(round((onset_sec[h5_idx] + duration_sec[h5_idx]) * fps))

                start_frame = max(0, min(start_frame, n_frames - 1))
                end_frame = max(start_frame + 1, min(end_frame, n_frames))
                seg_len = end_frame - start_frame

                seg_spec = full_spec[:, start_frame:end_frame]

                if seg_len >= n_time:
                    all_specs[pipe_pos] = torch.from_numpy(seg_spec[:, :n_time])
                    all_lengths[pipe_pos] = n_time
                else:
                    all_specs[pipe_pos, :, :seg_len] = torch.from_numpy(seg_spec)
                    all_lengths[pipe_pos] = seg_len

    print(f"  Loaded {n_segs:,} segments, freq={n_freq}, n_time={n_time}")
    print(f"  Age range: {pipeline_ages.min():.0f}-{pipeline_ages.max():.0f} dph")
    print(f"  Length stats: median={all_lengths.median():.0f}, "
          f"mean={all_lengths.float().mean():.0f}, "
          f"max={all_lengths.max():.0f}")

    return all_specs, pipeline_ages, all_lengths, pipeline_h5_idx, pipeline_segment_ids


# ============================== Model ==============================

class SpectrogramAE(nn.Module):
    """
    Convolutional autoencoder.
    Input: (B, n_freq, n_time) -> Latent: (B, latent_dim) -> Output: (B, n_freq, n_time).
    Conv1d: frequency bins = channels, time = spatial dimension.
    """

    def __init__(self, n_mels=80, n_time=100, latent_dim=128):
        super().__init__()
        self.n_mels = n_mels
        self.n_time = n_time
        self.latent_dim = latent_dim

        # ---- Encoder ----
        self.enc = nn.Sequential(
            nn.Conv1d(n_mels, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.GELU(),
        )
        # Compute exact encoder output time dimension
        t = n_time
        for _ in range(3):
            t = (t + 2 * 1 - 3) // 2 + 1  # Conv1d with k=3, s=2, p=1
        self.enc_time = t
        self.enc_flat = 512 * self.enc_time

        self.enc_fc = nn.Sequential(
            nn.Linear(self.enc_flat, 512), nn.GELU(),
            nn.Linear(512, latent_dim),
        )

        # ---- Decoder ----
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.GELU(),
            nn.Linear(512, self.enc_flat),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.ConvTranspose1d(128, n_mels, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.reshape(h.shape[0], -1)
        return self.enc_fc(h)

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.reshape(-1, 512, self.enc_time)
        h = self.dec(h)
        if h.shape[-1] != self.n_time:
            h = F.interpolate(h, size=self.n_time, mode='linear', align_corners=False)
        return h

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class SpectrogramVAE(nn.Module):
    """
    Variational autoencoder. Same conv backbone as SpectrogramAE but with
    mu/logvar bottleneck and KL divergence loss.
    """

    def __init__(self, n_mels=80, n_time=100, latent_dim=128):
        super().__init__()
        self.n_mels = n_mels
        self.n_time = n_time
        self.latent_dim = latent_dim

        # ---- Encoder (same as AE) ----
        self.enc = nn.Sequential(
            nn.Conv1d(n_mels, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.GELU(),
        )
        t = n_time
        for _ in range(3):
            t = (t + 2 * 1 - 3) // 2 + 1
        self.enc_time = t
        self.enc_flat = 512 * self.enc_time

        self.enc_fc = nn.Sequential(
            nn.Linear(self.enc_flat, 512), nn.GELU(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # ---- Decoder (same as AE) ----
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.GELU(),
            nn.Linear(512, self.enc_flat),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.ConvTranspose1d(128, n_mels, 4, stride=2, padding=1),
        )

    def encode(self, x):
        """Encode to mu (used at inference / for latents.pt)."""
        h = self.enc(x)
        h = h.reshape(h.shape[0], -1)
        h = self.enc_fc(h)
        return self.fc_mu(h)

    def encode_dist(self, x):
        """Encode to (mu, logvar) for training."""
        h = self.enc(x)
        h = h.reshape(h.shape[0], -1)
        h = self.enc_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.reshape(-1, 512, self.enc_time)
        h = self.dec(h)
        if h.shape[-1] != self.n_time:
            h = F.interpolate(h, size=self.n_time, mode='linear', align_corners=False)
        return h

    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ============================== Utils ==============================

def make_length_mask(lengths, n_time, device):
    """Mask for left-aligned content."""
    time_idx = torch.arange(n_time, device=device).unsqueeze(0)
    end = lengths.unsqueeze(1).to(device)
    mask = time_idx < end
    return mask.unsqueeze(1).float()


def crop_to_length(spec, length, n_time):
    if length >= n_time:
        return spec
    return spec[:, :length]


def visualize_recon(model, specs, lengths, device, config, output_path, epoch, n=8):
    model.eval()
    n_time = config['n_time']
    data_mean = config['data_mean']
    data_std = config['data_std']

    idxs = random.sample(range(len(specs)), min(n, len(specs)))

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    for col, idx in enumerate(idxs):
        x = specs[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            x_hat = out[0]  # works for both AE (2 outputs) and VAE (3 outputs)
        x_hat = x_hat.squeeze(0).cpu()
        x_orig = specs[idx]

        length = int(lengths[idx].item())

        orig = crop_to_length(x_orig * data_std + data_mean, length, n_time)
        recon = crop_to_length(x_hat * data_std + data_mean, length, n_time)

        vmin, vmax = orig.min().item(), orig.max().item()
        axes[0, col].imshow(orig.numpy(), aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        axes[0, col].set_title('Orig', fontsize=8); axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        axes[1, col].imshow(recon.numpy(), aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        axes[1, col].set_title('Recon', fontsize=8); axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    model_type = config.get('model_type', 'ae')
    fig.suptitle(f'Epoch {epoch + 1} | {model_type.upper()} Reconstruction', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path / f'viz_recon_{epoch:03d}.png', dpi=120, bbox_inches='tight')
    plt.close()


# ============================== Training ==============================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load data
    print('=' * 60)
    print(f'Phase 1: Autoencoder Training (source={args.source})')
    print('=' * 60)

    if args.source == 'h5':
        specs, ages, lengths, h5_idx, segment_ids = load_spectrograms_h5(args.bird, n_time=args.n_time)
    else:
        specs, ages, lengths, h5_idx, segment_ids = load_spectrograms_precomputed(args.bird)

    n_freq, n_time = specs.shape[1], specs.shape[2]
    print(f'  Specs: {specs.shape}, ages: {ages.min():.0f}-{ages.max():.0f}')

    # Standardize
    data_mean = specs.mean().item()
    data_std = specs.std().item()
    specs_normed = (specs - data_mean) / (data_std + 1e-6)

    N = len(specs_normed)

    if args.encode_only:
        train_ds = []
        val_ds = []
        train_loader = None
        val_loader = None
        print('  Mode: encode_only (skip training; encode with existing weights)')
    else:
        perm = torch.randperm(N)
        n_val = int(0.1 * N)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        train_ds = TensorDataset(specs_normed[train_idx], lengths[train_idx])
        val_ds = TensorDataset(specs_normed[val_idx], lengths[val_idx])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        print(f'  Train: {len(train_ds)}, Val: {len(val_ds)}')

    # 2. Model
    is_vae = args.model_type == 'vae'
    if is_vae:
        model = SpectrogramVAE(n_freq, n_time, args.latent_dim).to(device)
    else:
        model = SpectrogramAE(n_freq, n_time, args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model_name = 'SpectrogramVAE' if is_vae else 'SpectrogramAE'
    print(f'  Model: {model_name}({n_freq}, {n_time}), {n_params:,} params, '
          f'latent_dim={args.latent_dim}')

    optimizer = None
    scheduler = None
    if not args.encode_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # 3. Output dir
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(__file__).parent / 'models' / f'{args.model_type}_{args.bird}_{ts}'
    output_path.mkdir(parents=True, exist_ok=True)

    config = dict(
        n_mels=n_freq, n_time=n_time, latent_dim=args.latent_dim,
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
        data_mean=data_mean, data_std=data_std, bird=args.bird,
        num_params=n_params, l2_weight=args.l2_weight,
        n_samples=N, n_train=len(train_ds), n_val=len(val_ds),
        age_min=float(ages.min()), age_max=float(ages.max()),
        source=args.source,
        model_type=args.model_type,
        kl_weight=args.kl_weight,
        encode_only=bool(args.encode_only),
        init_from=args.init_from,
    )
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  Output: {output_path}\n')

    # 4. Train or load existing weights
    best_val = float('inf')

    if args.encode_only:
        if not args.init_from:
            raise ValueError('--encode_only requires --init_from <path_to_best.pt>')
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f'init_from checkpoint not found: {init_path}')

        print(f'\nLoading pretrained weights from: {init_path}')
        model.load_state_dict(torch.load(init_path, weights_only=True, map_location=device))
        shutil.copy2(init_path, output_path / 'best.pt')
        print(f'  Copied weights to: {output_path / "best.pt"}')
    else:
        if args.init_from:
            init_path = Path(args.init_from)
            if not init_path.exists():
                raise FileNotFoundError(f'init_from checkpoint not found: {init_path}')
            print(f'\nWarm-starting from: {init_path}')
            model.load_state_dict(torch.load(init_path, weights_only=True, map_location=device))

        for epoch in range(args.epochs):
            model.train()
            losses_train = []
            for x_batch, len_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}', leave=False):
                x_batch = x_batch.to(device)
                len_batch = len_batch.to(device)

                if is_vae:
                    x_hat, mu, logvar = model(x_batch)
                    z = mu  # for l2 penalty
                else:
                    x_hat, z = model(x_batch)

                mask = make_length_mask(len_batch, n_time, device)
                recon_loss = ((x_hat - x_batch) ** 2 * mask).sum() / (mask.sum() * n_freq + 1e-8)
                l2_loss = (z ** 2).mean()
                loss = recon_loss + args.l2_weight * l2_loss

                if is_vae:
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = loss + args.kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses_train.append(recon_loss.item())

            scheduler.step()

            # Validation
            model.eval()
            losses_val = []
            with torch.no_grad():
                for x_batch, len_batch in val_loader:
                    x_batch = x_batch.to(device)
                    len_batch = len_batch.to(device)
                    out = model(x_batch)
                    x_hat = out[0]
                    mask = make_length_mask(len_batch, n_time, device)
                    loss = ((x_hat - x_batch) ** 2 * mask).sum() / (mask.sum() * n_freq + 1e-8)
                    losses_val.append(loss.item())

            t_loss = np.mean(losses_train)
            v_loss = np.mean(losses_val)
            print(f'Epoch {epoch + 1}/{args.epochs} | Train: {t_loss:.5f} | Val: {v_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}')

            if v_loss < best_val:
                best_val = v_loss
                torch.save(model.state_dict(), output_path / 'best.pt')
                print(f'  -> Best (val={v_loss:.5f})')

            if (epoch + 1) % args.viz_every == 0 or epoch == 0:
                visualize_recon(model, specs_normed, lengths, device, config, output_path, epoch)

    # 5. Encode ALL spectrograms and save
    print('\nEncoding all spectrograms...')
    model.load_state_dict(torch.load(output_path / 'best.pt', weights_only=True, map_location=device))
    model.eval()

    all_z = []
    loader_all = DataLoader(TensorDataset(specs_normed), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (batch,) in tqdm(loader_all, desc='Encoding'):
            z = model.encode(batch.to(device))
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)
    print(f'  Latent shape: {all_z.shape}')

    latent_payload = {'z': all_z, 'ages': ages, 'lengths': lengths}
    if h5_idx is not None:
        latent_payload['h5_idx'] = h5_idx
    if segment_ids is not None:
        latent_payload['segment_id'] = segment_ids

    torch.save(latent_payload, output_path / 'latents.pt')
    print(f'  Saved to {output_path / "latents.pt"}')

    print(f'  Latent mean: {all_z.mean():.4f}, std: {all_z.std():.4f}')
    print(f'  Latent range: [{all_z.min():.2f}, {all_z.max():.2f}]')
    if args.encode_only:
        print('\nDone! Encoded with reused AE weights.')
    else:
        print(f'\nDone! Best val MSE: {best_val:.5f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Spectrogram Autoencoder')
    parser.add_argument('--bird', type=str, default='R5018')
    parser.add_argument('--source', type=str, default='h5', choices=['h5', 'precomputed'],
                        help='Data source: h5 (from H5 file) or precomputed (Spectrograms_PadRight)')
    parser.add_argument('--model_type', type=str, default='ae', choices=['ae', 'vae'],
                        help='Model type: ae (autoencoder) or vae (variational)')
    parser.add_argument('--n_time', type=int, default=100,
                        help='Fixed time frames for H5 source (default: 100)')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--l2_weight', type=float, default=1e-4,
                        help='L2 penalty on latent codes (encourages compact representations)')
    parser.add_argument('--kl_weight', type=float, default=1e-3,
                        help='KL divergence weight for VAE (beta parameter)')
    parser.add_argument('--viz_every', type=int, default=10)
    parser.add_argument('--encode_only', action='store_true',
                        help='Skip AE training and only encode with pretrained weights')
    parser.add_argument('--init_from', type=str, default=None,
                        help='Path to AE checkpoint (best.pt) for warm-start or encode-only')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Optional output directory (default: ae_<bird>_<timestamp>)')
    args = parser.parse_args()
    train(args)
