"""
Figure 1: Trajectory variance distributions (Songs vs Calls).

Loads fig1_data.npz (saved by baseline_comparison.py) and renders:
    - Raw variance distributions (Songs vs Calls)
    - Optional residualized panel if requested

Usage:
  python plot_fig1.py --npz models/ot_flow_R2915_20260224_134859/fig1_data.npz
  python plot_fig1.py --flow_dir models/ot_flow_R2915_20260224_134859
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


SONG_COLOR = "#2F6C8F"
CALL_COLOR = "#C6A56B"


def set_pub_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.25)
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1 = x.var(ddof=1)
    v2 = y.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(1, (n1 + n2 - 2)))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def regress_out_duration(y: np.ndarray, duration: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    duration = np.asarray(duration, dtype=np.float64)
    if np.allclose(duration.std(), 0.0):
        return y - y.mean()
    slope, intercept = np.polyfit(duration, y, deg=1)
    y_hat = slope * duration + intercept
    return y - y_hat


def annotate_distribution(ax, song_vals: np.ndarray, call_vals: np.ndarray, title: str) -> None:
    med_song = float(np.median(song_vals))
    med_call = float(np.median(call_vals))
    d_val = cohen_d(song_vals, call_vals)

    ax.axvline(med_song, color=SONG_COLOR, lw=1.8, ls="--", alpha=0.9)
    ax.axvline(med_call, color=CALL_COLOR, lw=1.8, ls="--", alpha=0.9)

    txt = (
        f"Median (Song): {med_song:.3f}\n"
        f"Median (Call): {med_call:.3f}\n"
        f"Cohen's d: {d_val:.3f}"
    )
    ax.text(
        0.98,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.85"},
    )
    ax.set_title(title, pad=8)


def resolve_npz_path(npz: str | None, flow_dir: str | None) -> Path:
    if npz:
        p = Path(npz)
    elif flow_dir:
        p = Path(flow_dir) / "fig1_data.npz"
    else:
        raise ValueError("Provide --npz or --flow_dir")

    if not p.exists():
        raise FileNotFoundError(f"fig1_data not found: {p}")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 1: variance distributions (Songs vs Calls)")
    parser.add_argument("--npz", type=str, default=None, help="Path to fig1_data.npz")
    parser.add_argument("--flow_dir", type=str, default=None, help="Flow directory containing fig1_data.npz")
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "both"],
                        help="raw: duration-uncorrected only; both: raw + residualized panel")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (PDF or PNG). Default: <flow_dir>/fig1_trajectory_variance.pdf")
    args = parser.parse_args()

    set_pub_style()
    npz_path = resolve_npz_path(args.npz, args.flow_dir)

    dat = np.load(npz_path)
    variances = dat["variances"].astype(np.float64)
    is_song = dat["is_song"].astype(bool)
    durations = dat["durations"].astype(np.float64)

    if len(variances) != len(is_song) or len(variances) != len(durations):
        raise ValueError("Array length mismatch in fig1_data.npz")

    var_song = variances[is_song]
    var_call = variances[~is_song]

    if args.mode == "both":
        residuals = regress_out_duration(variances, durations)
        res_song = residuals[is_song]
        res_call = residuals[~is_song]
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
        ax0, ax1 = axes[0], axes[1]
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(6.0, 4.8), constrained_layout=True)
        ax1 = None

    sns.kdeplot(var_song, fill=True, alpha=0.35, color=SONG_COLOR, lw=2.0, label="Songs", ax=ax0)
    sns.kdeplot(var_call, fill=True, alpha=0.35, color=CALL_COLOR, lw=2.0, label="Calls", ax=ax0)
    ax0.set_xlabel("Neural Trajectory Variance")
    ax0.set_ylabel("Density")
    annotate_distribution(ax0, var_song, var_call, "Raw Variance")
    ax0.legend(loc="upper left")

    if ax1 is not None:
        sns.kdeplot(res_song, fill=True, alpha=0.35, color=SONG_COLOR, lw=2.0, label="Songs", ax=ax1)
        sns.kdeplot(res_call, fill=True, alpha=0.35, color=CALL_COLOR, lw=2.0, label="Calls", ax=ax1)
        ax1.set_xlabel("Duration-Residual Variance ($y - \\hat{y}$)")
        ax1.set_ylabel("Density")
        annotate_distribution(ax1, res_song, res_call, "Residualized Variance")
        ax1.legend(loc="upper left")

    if args.mode == "both":
        fig.suptitle("Songs Exhibit Higher Developmental Trajectory Plasticity than Calls", y=1.03, fontsize=13)
    else:
        fig.suptitle("Songs Exhibit Higher Developmental Trajectory Plasticity than Calls (Raw)", y=1.03, fontsize=13)

    if args.output is not None:
        out = Path(args.output)
    else:
        out = npz_path.parent / "fig1_trajectory_variance.pdf"

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out.with_suffix('.pdf')}")
    print(f"Saved: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
