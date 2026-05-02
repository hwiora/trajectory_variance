"""Reproduce Figure 2: KDE of duration-residualized trajectory variance
for song vs call vocalizations, per bird.

KDE shape uses the 10K-sample visualization subset stored in
fig1_data_<bird>.npz; the d_r and AUC labels are pulled from the 3K-sample
evaluation in paper_eval_results.json (Displacement / OT coupling), matching
how Figure 2 is annotated in the paper.

Usage:
    python -m Counterfactual_generation.plot_fig2 --output paper/figures/fig2.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

BIRDS = ["R4634", "R4951", "R5018"]
LABELS = {"R4634": "Bird A", "R4951": "Bird B", "R5018": "Bird C"}
SONG_COLOR = "#E67E22"   # orange
CALL_COLOR = "#3498DB"   # blue


def residualize(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(durations.astype(float), values, 1)
    return values - (slope * durations.astype(float) + intercept)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="fig2.pdf",
                        help="Output figure path (PDF/PNG/SVG)")
    parser.add_argument("--models_dir", type=str,
                        default=str(Path(__file__).parent / "models"))
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    eval_results = json.load(open(models_dir / "paper_eval_results.json"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)

    for ax, bird in zip(axes, BIRDS):
        data = np.load(models_dir / f"fig1_data_{bird}.npz")
        var, is_song, dur = data["variances"], data["is_song"].astype(bool), data["durations"]

        resid = residualize(var, dur)
        s, c = resid[is_song], resid[~is_song]

        # KDE
        x_lo, x_hi = np.percentile(resid, [0.5, 99.5])
        xs = np.linspace(x_lo, x_hi, 400)
        kde_s = gaussian_kde(s)(xs)
        kde_c = gaussian_kde(c)(xs)
        ax.fill_between(xs, kde_s, alpha=0.55, color=SONG_COLOR, label="song")
        ax.fill_between(xs, kde_c, alpha=0.55, color=CALL_COLOR, label="call")
        ax.axvline(s.mean(), color=SONG_COLOR, linestyle="--", lw=1.2)
        ax.axvline(c.mean(), color=CALL_COLOR, linestyle="--", lw=1.2)

        # Annotation: dr and AUC from the 3K eval (matches Table 2 / paper Fig 2)
        t2 = eval_results[f"{bird}_ot"]["table2"]["Displacement"]
        d_r = t2["cohens_d_resid"]
        auc = t2["auc_resid"]
        ax.text(0.97, 0.95, f"$d_r$ = {d_r:.2f}\nAUC = {auc:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, family="serif")

        ax.set_title(LABELS[bird], fontsize=12)
        ax.set_xlabel("Duration-residualized\ntrajectory variance", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Density", fontsize=11)
    axes[1].legend(loc="upper left", frameon=False, fontsize=10)
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
