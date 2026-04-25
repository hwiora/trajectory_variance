# Trajectory Variance: An Unsupervised Measure of Developmental Vocal Plasticity

Code accompanying the Interspeech 2026 paper.

📄 **Paper:** [Submitted version (PDF)](paper/Trajectory_Variance_Interspeech2026_submitted.pdf)
<!-- TODO: replace with arXiv / ISCA Archive link once published -->

## Overview

Given a vocalization, how much would it change if produced at a different developmental age? We learn a displacement model that predicts age-conditioned shifts in a VAE latent space, then define **trajectory variance** as the variance of those predicted shifts across target ages. On three zebra finches (40–100 days post-hatch, 183K–274K vocalizations each), trajectory variance separates learned song syllables from innate calls (Cohen's d = 0.29–0.57, AUC = 0.58–0.67 after controlling for duration), without any vocalization-type labels.

## Repository structure

```
trajectory_variance/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── .gitignore
├── paper/
│   └── Trajectory_Variance_Interspeech2026_submitted.pdf
└── Counterfactual_generation/           # Python package
    ├── train_ae.py                      # Spectrogram VAE
    ├── train_ot_flow.py                 # Displacement model (OT-coupled)
    ├── label_song_calls.py              # Bout-based song/call labeling
    ├── baseline_comparison.py           # All baseline transports + variance computation
    ├── analyze_plasticity.py            # Acoustic-feature streaming
    ├── run_evaluations.py               # → models/evaluation_results.json (Tables 1+2)
    ├── compute_fad2.py, run_fad2_all.py # FAD evaluation (Discussion)
    ├── utils.py                         # DATA_ROOT, helpers
    └── models/
        ├── flow.py                      # Displacement-model architecture & loader
        ├── evaluation_results.json      # Reference output for Tables 1+2
        ├── fad2_summary.json            # Reference output for FAD numbers
        └── fig1_data_R{4634,4951,5018}.npz  # Pre-computed per-vocalization variances + labels
```

## Setup

**Requirements:** Python ≥ 3.9, CUDA-capable GPU (training scripts only; plotting and evaluation work on CPU).

**Conda (recommended — matches the environment used to produce the paper):**

> Note: `environment.yml` was exported on Windows. On macOS/Linux use `pip install -e .` instead.

```bash
git clone https://github.com/hwiora/trajectory_variance
cd trajectory_variance
conda env create -f environment.yml        # creates the 'preprocess' environment
conda activate preprocess
pip install -e .                           # installs the package in editable mode
```

**pip only (cross-platform):**

```bash
pip install -e .
```

**Data root — set before running any script:**

```bash
export TRAJECTORY_VARIANCE_DATA_ROOT=/path/to/your/data   # Linux / macOS
set TRAJECTORY_VARIANCE_DATA_ROOT=C:\path\to\your\data    # Windows
```

See the [Data section](#data) below for the expected directory layout.

## Data

The raw H5 files are not included here — they come from a pre-existing data pipeline and are large (~3.5–4 GB per bird). The paper uses three zebra finch datasets (R4634, R4951, R5018; recorded 40–100 dph; 183K–274K vocalizations each). Please contact the author for access if you want to reproduce end-to-end.

The expected layout under `DATA_ROOT` (used by the default `--source h5` flag) is:

```
DATA_ROOT/
└── <bird_id>/
    └── Processed/
        └── <bird_id>.h5
```

## Reproducing the paper

All scripts are run as **modules from the repo root** (`python -m Counterfactual_generation.<script>`).
The package also exposes command-line entry points (e.g. `tv-evaluate`) after `pip install -e .`.

The full pipeline has four stages; each writes outputs consumed by the next.

### 1. Train a VAE per bird

```bash
python -m Counterfactual_generation.train_ae --bird R4634 --source h5 --model_type vae
python -m Counterfactual_generation.train_ae --bird R4951 --source h5 --model_type vae
python -m Counterfactual_generation.train_ae --bird R5018 --source h5 --model_type vae
```

Writes `Counterfactual_generation/models/vae_<bird>_<timestamp>/` (`best.pt`, `config.json`, `latents.pt`).

### 2. Train the displacement model per bird

```bash
python -m Counterfactual_generation.train_ot_flow \
    --ae_dir Counterfactual_generation/models/vae_R4634_<timestamp> --arch direct --coupling ot
# … and similarly for R4951, R5018
```

Writes `Counterfactual_generation/models/ot_flow_<bird>_<timestamp>/` (`best.pt`, `config.json`).

### 3. Label song vs call

```bash
python -m Counterfactual_generation.label_song_calls --bird R4634
# … and similarly for R4951, R5018
```

Writes `gold_standard_labels/<bird>_song_labels_pipeline_order.npz`.

### 4. Tables 1 and 2

```bash
python -m Counterfactual_generation.run_evaluations
# or: tv-evaluate
```

Output: `Counterfactual_generation/models/evaluation_results.json`. A **reference copy is committed** to this repo — diff your re-run against it to verify you reproduced the exact paper numbers.

### Figures

The committed `Counterfactual_generation/models/fig1_data_R{4634,4951,5018}.npz` files contain the pre-computed per-vocalization variances and labels used to produce Figure 2 (KDE panels). Each file has three arrays: `variances`, `is_song` (bool), and `durations`.

Figure 1 (pipeline schematic) is a hand-composed diagram included in the submitted PDF.

### FAD (Discussion section)

```bash
python -m Counterfactual_generation.run_fad2_all
# or: tv-fad
```

Reproduces `Counterfactual_generation/models/fad2_summary.json` (the 0.01–0.06 vs. 0.002–0.007 numbers in Section 5.2). Requires trained model checkpoints and access to the raw data.

## Citation

```bibtex
@unpublished{lee2026trajectory,
  title  = {Trajectory Variance: An Unsupervised Measure of Developmental Vocal Plasticity},
  author = {Lee, Kanghwi},
  year   = {2026},
  note   = {Under review at Interspeech 2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

Kanghwi Lee — kanlee@ini.ethz.ch
