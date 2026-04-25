"""
compute_fad2.py — clean pooled FAD/FAD_inf evaluation.

This script computes ONLY pooled metrics (no per-target-age reports):
  - Full-sample FAD (CF vs real reconstructions)
  - FAD_inf via FAD(n) = FAD_inf + c/n extrapolation
  - Split-half baseline (real vs real)

Supported methods:
  - ot_flow
  - latent_cfm

Example:
  python compute_fad2.py \
      --method ot_flow \
      --ae_dir models/vae_R4951_20260224_035117 \
      --flow_dir models/ot_flow_R4951_20260224_151200 \
      --bird R4951 \
      --total_samples 2000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.linalg import sqrtm
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .train_ae import SpectrogramAE, SpectrogramVAE
from .models.flow import load_transport_model

try:
    from .train_latent_cfm import VelocityNetwork, generate_cf  # optional; not in this repo
    _HAS_LCFM = True
except ImportError:
    _HAS_LCFM = False


SAMPLE_RATE = 32000
HOP_LENGTH = 128


def spec_to_audio(log_spec: np.ndarray, hop_length: int = HOP_LENGTH, n_iter: int = 32) -> np.ndarray:
    mag = np.exp(log_spec) - 1e-6
    mag = np.maximum(mag, 0)
    n_freq = mag.shape[0]
    n_fft = (n_freq - 1) * 2
    return librosa.griffinlim(mag, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft)


def pad_min_duration(audio: np.ndarray, sr: int = SAMPLE_RATE, min_sec: float = 1.0) -> np.ndarray:
    min_len = int(sr * min_sec)
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)), mode="constant")
    return audio.astype(np.float32, copy=False)


def compute_stats(embeddings: np.ndarray, cov_eps: float = 1e-6):
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]], dtype=np.float64)
    sigma = sigma + cov_eps * np.eye(sigma.shape[0], dtype=sigma.dtype)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    val = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    if not np.isfinite(val):
        return float("nan")
    return float(max(val, 0.0))


def extract_vggish_embeddings(audios, verbose=True) -> np.ndarray:
    import torchaudio

    try:
        bundle = torchaudio.pipelines.VGGISH
    except AttributeError:
        from torchaudio.prototype.pipelines import VGGISH
        bundle = VGGISH

    model = bundle.get_model().eval()
    expected_sr = bundle.sample_rate
    input_proc = bundle.get_input_processor() if hasattr(bundle, "get_input_processor") else None

    embeddings = []
    iterator = tqdm(audios, desc="Extracting VGGish", leave=False) if verbose else audios
    for audio in iterator:
        wav = torch.tensor(audio, dtype=torch.float32)
        if SAMPLE_RATE != expected_sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), SAMPLE_RATE, expected_sr).squeeze(0)

        with torch.no_grad():
            if input_proc is not None:
                batch = input_proc(wav)
                emb = model(batch).mean(dim=0)
            else:
                emb = model(wav.unsqueeze(0)).squeeze(0).mean(dim=0)
        embeddings.append(emb.cpu().numpy())

    return np.stack(embeddings)


def compute_fad_inf(emb_a: np.ndarray, emb_b: np.ndarray,
                    n_points: int = 10, n_repeats: int = 5, seed: int = 42):
    rng = np.random.RandomState(seed)
    n_max = min(len(emb_a), len(emb_b))
    emb_dim = emb_a.shape[1]

    n_min = max(emb_dim + 50, 200)
    if n_min >= n_max:
        mu_a, sig_a = compute_stats(emb_a)
        mu_b, sig_b = compute_stats(emb_b)
        raw = frechet_distance(mu_a, sig_a, mu_b, sig_b)
        return {
            "fad_inf": float(raw),
            "slope": 0.0,
            "r_squared": 0.0,
            "n_sizes": [int(n_max)],
            "fad_at_n": [float(raw)],
        }

    n_sizes = np.unique(np.logspace(np.log10(n_min), np.log10(n_max), n_points).astype(int))
    fad_means = []
    for n in n_sizes:
        reps = 1 if n == n_max else n_repeats
        vals = []
        for _ in range(reps):
            idx_a = rng.choice(len(emb_a), size=n, replace=False)
            idx_b = rng.choice(len(emb_b), size=n, replace=False)
            mu_a, sig_a = compute_stats(emb_a[idx_a])
            mu_b, sig_b = compute_stats(emb_b[idx_b])
            vals.append(frechet_distance(mu_a, sig_a, mu_b, sig_b))
        mean_val = float(np.nanmean(vals))
        fad_means.append(mean_val)
        print(f"    n={n:6d}: FAD={mean_val:.4f} (std={np.nanstd(vals):.4f}, reps={reps})")

    fad_means = np.array(fad_means, dtype=np.float64)
    inv_n = 1.0 / n_sizes.astype(np.float64)
    valid = np.isfinite(fad_means)

    if valid.sum() < 2:
        raw = fad_means[valid][0] if valid.any() else 0.0
        return {
            "fad_inf": float(raw),
            "slope": 0.0,
            "r_squared": 0.0,
            "n_sizes": n_sizes[valid].tolist(),
            "fad_at_n": fad_means[valid].tolist(),
        }

    x = inv_n[valid]
    y = fad_means[valid]
    A = np.vstack([x, np.ones(len(x))]).T
    slope, fad_inf = np.linalg.lstsq(A, y, rcond=None)[0]
    fad_inf = max(float(fad_inf), 0.0)

    fitted = slope * x + fad_inf
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "fad_inf": fad_inf,
        "slope": float(slope),
        "r_squared": float(r2),
        "n_sizes": n_sizes[valid].tolist(),
        "fad_at_n": y.tolist(),
    }


def load_ae(ae_dir: Path, device: torch.device):
    ae_config = json.load(open(ae_dir / "config.json"))
    model_type = ae_config.get("model_type", "ae")
    n_freq = ae_config.get("n_mels", ae_config.get("n_freq", 80))
    n_time = ae_config.get("n_time", 100)
    latent_dim = ae_config.get("latent_dim", 128)

    if model_type == "vae":
        ae = SpectrogramVAE(n_freq, n_time, latent_dim).to(device)
    else:
        ae = SpectrogramAE(n_freq, n_time, latent_dim).to(device)

    ae.load_state_dict(torch.load(ae_dir / "best.pt", map_location=device, weights_only=True))
    ae.eval()
    return ae, ae_config


def load_method(method: str, flow_dir: Path, device: torch.device):
    cfg = json.load(open(flow_dir / "config.json"))
    if method == "ot_flow":
        model = load_transport_model(cfg, device, weights_path=flow_dir / "best.pt")
    elif method == "latent_cfm":
        if not _HAS_LCFM:
            raise ImportError(
                "latent_cfm method requires train_latent_cfm.py which is not included "
                "in this repository. Only 'ot_flow' is supported here."
            )
        model = VelocityNetwork(
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_blocks=cfg["num_blocks"],
            time_embed_dim=cfg.get("embed_dim", 64),
            age_embed_dim=cfg.get("embed_dim", 64),
            p_uncond=cfg.get("p_uncond", 0.15),
        ).to(device)
        model.load_state_dict(torch.load(flow_dir / "best.pt", map_location=device, weights_only=True))
        model.eval()
    else:
        raise ValueError(f"Unsupported method: {method}")

    z_mean = torch.tensor(cfg["z_mean"], device=device)
    z_std = torch.tensor(cfg["z_std"], device=device)
    norm_age_min = cfg.get("norm_age_min", cfg["age_min"])
    norm_age_max = cfg.get("norm_age_max", cfg["age_max"])
    return model, cfg, z_mean, z_std, norm_age_min, norm_age_max


def main():
    parser = argparse.ArgumentParser(description="Clean pooled FAD/FAD_inf evaluation")
    parser.add_argument("--method", type=str, required=True, choices=["ot_flow"])
    parser.add_argument("--ae_dir", type=str, required=True)
    parser.add_argument("--flow_dir", type=str, required=True)
    parser.add_argument("--bird", type=str, default="unknown")
    parser.add_argument("--total_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_points", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_dir = Path(args.ae_dir)
    flow_dir = Path(args.flow_dir)

    print("=" * 60)
    print(f"Pooled FAD/FAD_inf ({args.method})")
    print("=" * 60)

    ae, ae_config = load_ae(ae_dir, device)
    ld = torch.load(ae_dir / "latents.pt", weights_only=True)
    latents, ages, lengths = ld["z"], ld["ages"], ld["lengths"]

    model, flow_cfg, z_mean, z_std, norm_age_min, norm_age_max = load_method(
        args.method, flow_dir, device
    )

    N = min(args.total_samples, len(latents))
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(latents), size=N, replace=False)
    age_min, age_max = ages.min().item(), ages.max().item()
    target_ages = rng.uniform(age_min, age_max, size=N)

    data_mean = ae_config["data_mean"]
    data_std = ae_config["data_std"]

    real_audio = []
    cf_audio = []

    print(f"Generating pooled real/CF audio for {N} samples...")
    with torch.no_grad():
        for sample_i, tgt_age in tqdm(zip(idx, target_ages), total=N, leave=False):
            z = latents[sample_i].unsqueeze(0).to(device)
            sl = int(lengths[sample_i].item())

            # Real reconstruction
            spec_real = ae.decode(z).squeeze(0).cpu().numpy()
            spec_real = spec_real * data_std + data_mean
            spec_real = spec_real[:, :sl]
            audio_real = pad_min_duration(spec_to_audio(spec_real))
            real_audio.append(audio_real)

            # Counterfactual latent in normalized space
            z_norm = (latents[sample_i].to(device) - z_mean) / z_std
            z_norm = z_norm.unsqueeze(0)

            a_src = (ages[sample_i].item() - norm_age_min) / (norm_age_max - norm_age_min)
            a_tgt = (tgt_age - norm_age_min) / (norm_age_max - norm_age_min)

            if args.method == "ot_flow":
                c = torch.tensor([[a_src, a_tgt]], device=device, dtype=torch.float32)
                z_cf = model.generate(z_norm, c, steps=None, cfg_scale=None, solver=None)
            else:
                cfg_scale = flow_cfg.get("cfg_scale", 3.0)
                edit_strength = flow_cfg.get("edit_strength", 0.5)
                steps = flow_cfg.get("steps", 50)
                a_tgt_t = torch.tensor([a_tgt], device=device, dtype=torch.float32)
                z_cf = generate_cf(
                    model,
                    z_norm,
                    a_tgt_t,
                    cfg_scale=cfg_scale,
                    edit_strength=edit_strength,
                    steps=steps,
                )

            spec_cf = ae.decode(z_cf * z_std + z_mean).squeeze(0).cpu().numpy()
            spec_cf = spec_cf * data_std + data_mean
            spec_cf = spec_cf[:, :sl]
            audio_cf = pad_min_duration(spec_to_audio(spec_cf))
            cf_audio.append(audio_cf)

    print("Extracting VGGish embeddings (real)...")
    emb_real = extract_vggish_embeddings(real_audio)
    print("Extracting VGGish embeddings (cf)...")
    emb_cf = extract_vggish_embeddings(cf_audio)

    print("\nComputing pooled FAD_inf (CF vs real)...")
    fad_inf_curve = compute_fad_inf(
        emb_real,
        emb_cf,
        n_points=args.n_points,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )

    mu_r, sig_r = compute_stats(emb_real)
    mu_c, sig_c = compute_stats(emb_cf)
    full_fad = frechet_distance(mu_r, sig_r, mu_c, sig_c)

    print("\nComputing split-half baseline...")
    perm = np.random.RandomState(args.seed).permutation(len(emb_real))
    half = len(perm) // 2
    emb_h1 = emb_real[perm[:half]]
    emb_h2 = emb_real[perm[half:2 * half]]

    baseline_curve = compute_fad_inf(
        emb_h1,
        emb_h2,
        n_points=args.n_points,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )

    mu_h1, sig_h1 = compute_stats(emb_h1)
    mu_h2, sig_h2 = compute_stats(emb_h2)
    full_baseline = frechet_distance(mu_h1, sig_h1, mu_h2, sig_h2)

    results = {
        "method": args.method,
        "bird": args.bird,
        "ae_dir": str(ae_dir),
        "flow_dir": str(flow_dir),
        "n_real": int(N),
        "n_cf": int(N),
        "embedding": "vggish",
        "full_fad": float(full_fad),
        "fad_inf": float(fad_inf_curve["fad_inf"]),
        "fad_inf_r2": float(fad_inf_curve["r_squared"]),
        "fad_inf_curve": fad_inf_curve,
        "baseline_full_fad": float(full_baseline),
        "baseline_fad_inf": float(baseline_curve["fad_inf"]),
        "baseline_fad_inf_r2": float(baseline_curve["r_squared"]),
        "baseline_curve": baseline_curve,
        "seed": int(args.seed),
        "total_samples": int(N),
    }

    out_path = flow_dir / "fad2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"FAD_inf (CF vs real):   {results['fad_inf']:.4f} (R²={results['fad_inf_r2']:.3f})")
    print(f"Full FAD (CF vs real):  {results['full_fad']:.4f}")
    print(f"FAD_inf (split-half):   {results['baseline_fad_inf']:.4f} (R²={results['baseline_fad_inf_r2']:.3f})")
    print(f"Full FAD (split-half):  {results['baseline_full_fad']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
