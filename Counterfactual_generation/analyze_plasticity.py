"""Compute per-vocalization spectral flatness from the H5 spectrograms.

Each segment is sliced from its recording's spectrogram by ``[onset, onset+duration]``
and normalized to ``[0, 1]`` with the per-bird global int8 range, then its spectral
flatness (geometric-over-arithmetic mean of power, averaged over frames) is taken.

The output is aligned to segment order (the latent / pipeline order used throughout
the evaluation). An earlier version streamed a separate ``Spectrograms_PadRight``
snapshot whose row order did not match the segment order for some birds; computing
directly from the H5 by segment removes that misalignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from .utils import DATA_ROOT


def _spectral_flatness(active: np.ndarray) -> float:
    """Spectral flatness of one vocalization.

    ``active`` is an ``(F, L)`` spectrogram slice normalized to ~[0, 1]. Power is
    recovered as ``exp(active * 5)``; flatness is the geometric-over-arithmetic mean
    across frequency, averaged over the ``L`` frames.
    """
    power = np.exp(active * 5.0)
    power = np.maximum(power, 1e-10)
    log_mean = np.mean(np.log(power + 1e-10), axis=0)
    arith_mean = np.mean(power, axis=0) + 1e-10
    return float((np.exp(log_mean) / arith_mean).mean())


def compute_acoustic_features_streaming(bird: str, N: int | None = None) -> dict[str, np.ndarray]:
    """Spectral flatness per vocalization, segment-aligned, computed from the H5 file.

    Returns ``{"spectral_flatness": array}`` of length ``N`` (or the segment count if
    ``N`` is None), aligned to segment order. Positions whose recording lacks a stored
    spectrogram are left as NaN.
    """
    h5_path = DATA_ROOT / bird / "Processed" / f"{bird}.h5"
    with h5py.File(str(h5_path), "r") as h5:
        fps = h5["parameters"].attrs["audio_sr"] / h5["parameters"].attrs["hop_length"]
        onset = h5["segments/onset_sec"][:]
        duration = h5["segments/duration_sec"][:]
        seg_file_id = h5["segments/file_id"][:]
        spec_file_ids = h5["spectrograms/file_id"][:]
        fid_to_key = {int(spec_file_ids[i]): str(i) for i in range(len(spec_file_ids))}

        n_seg = len(onset)
        out_len = n_seg if N is None else int(N)
        n_fill = min(out_len, n_seg)
        flatness = np.full(out_len, np.nan, dtype=np.float64)

        # per-bird global int8 range -> [0, 1] normalization (matches training)
        g_min, g_max = np.inf, -np.inf
        for key in fid_to_key.values():
            spec = h5[f"spectrograms/{key}"]
            g_min = min(g_min, float(spec[:].min()))
            g_max = max(g_max, float(spec[:].max()))
        span = (g_max - g_min) + 1e-6

        rows_by_file: dict[int, list[int]] = {}
        for row in range(n_fill):
            rows_by_file.setdefault(int(seg_file_id[row]), []).append(row)

        for file_id, rows in rows_by_file.items():
            key = fid_to_key.get(file_id)
            if key is None:
                continue
            spec = h5[f"spectrograms/{key}"][:].astype(np.float32)
            n_frames = spec.shape[1]
            for row in rows:
                start = max(0, min(int(round(onset[row] * fps)), n_frames - 1))
                end = max(start + 1, min(int(round((onset[row] + duration[row]) * fps)), n_frames))
                flatness[row] = _spectral_flatness((spec[:, start:end] - g_min) / span)

    return {"spectral_flatness": flatness}


def main() -> None:
    """Rebuild a spectral-flatness cache from the H5 file."""
    parser = argparse.ArgumentParser(description="Compute segment-aligned spectral flatness from H5")
    parser.add_argument("--bird", required=True, help="Bird ID, e.g. R4634")
    parser.add_argument("--N", type=int, default=None, help="Optional latent count")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output NPZ. Defaults to Counterfactual_generation/models/spectral_flatness_<bird>.npz",
    )
    args = parser.parse_args()

    spectral_flatness = compute_acoustic_features_streaming(args.bird, args.N)["spectral_flatness"]

    out_path = (
        Path(args.output)
        if args.output
        else Path(__file__).parent / "models" / f"spectral_flatness_{args.bird}.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, spectral_flatness=spectral_flatness)
    n_valid = int(np.sum(~np.isnan(spectral_flatness)))
    print(f"Saved: {out_path}  ({n_valid}/{len(spectral_flatness)} valid)")


if __name__ == "__main__":
    main()
