# Trajectory Variance: An Unsupervised Measure of Developmental Vocal Plasticity in Birdsong

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
    ├── run_evaluations.py               # → models/paper_eval_results.json (Tables 1+2)
    ├── compute_fad2.py, run_fad2_all.py # FAD evaluation (Discussion)
    ├── utils.py                         # DATA_ROOT, helpers
    └── models/
        ├── flow.py                      # Displacement-model architecture & loader
        ├── paper_eval_results.json      # Reference output for Tables 1+2
        ├── fad2_summary.json            # Reference output for FAD numbers
        ├── fig1_data_R{4634,4951,5018}.npz  # Pre-computed per-vocalization variances + labels
        ├── spectral_flatness_R{4634,4951,5018}.npz  # Cached spectral flatness per vocalization
        ├── vae_R{4634,4951,5018}/       # Trained VAE checkpoints (best.pt + config.json)
        └── ot_flow_R{4634,4951,5018}_{ot,knn}/  # Trained displacement-model checkpoints
```

`vae_*/latents.pt` (~95–142 MB per bird) is not included in the GitHub repo due to size; download from the accompanying data archive (link in the data section below) and place each `latents.pt` next to its corresponding `best.pt` to skip re-encoding.

## Setup

**Requirements:** Python ≥ 3.9, CUDA-capable GPU (training scripts only; plotting and evaluation work on CPU).

**Conda (recommended — matches the environment used to produce the paper):**

> Note: `environment.yml` was exported on Windows. On macOS/Linux use `pip install -e .` instead.

```bash
git clone <repository-url>
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

The paper uses three zebra finch datasets (R4634 / R4951 / R5018; recorded 40–100 dph; 183K–274K vocalizations each). What's available where:

| Item | Location | Notes |
|------|----------|-------|
| Trained model weights (VAE + displacement) | this repo, under `Counterfactual_generation/models/` | `best.pt` + `config.json` per bird |
| Cached spectral flatness | this repo, `models/spectral_flatness_<bird>.npz` | One feature per vocalization, NaN-padded; see note in §"Reproducing the paper" |
| Song/call gold-standard labels | this repo, `gold_standard_labels/` | Bout-based heuristic |
| Per-vocalization VAE latents (`latents.pt`) | Zenodo (anonymous link below) | ~95–142 MB per bird; over GitHub's file-size limit |
| Raw H5 spectrograms | not publicly released | Schema documented below for users training from their own data |
| Raw audio (WAV/FLAC) | not yet released | Will accompany the camera-ready release |

**Anonymous Zenodo deposit (latents):** https://zenodo.org/records/19922014?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZlYzYzMjNkLTJmZDQtNDFiYy05MDJkLTFiODBkMzJjNTk2YyIsImRhdGEiOnt9LCJyYW5kb20iOiIwNjkwNTdkNTU4YjQyMTNmNjNhYTExMDIyNDFmYjVjMyJ9.xtASLFgKdxTS8DkcVPBBLFcuuC26kGnbFmng6yKp0ilm1OksBHZugqZXqjJ8ccQ28ZXUDu-FP2HdKooKgcOXNg

After downloading, place each `latents.pt` next to its corresponding `best.pt`:

```
Counterfactual_generation/models/vae_<bird>/
├── best.pt        # in this repo
├── config.json    # in this repo
└── latents.pt     # downloaded from Zenodo
```

If you supply your own H5 files (matching the schema below), the training scripts will encode latents from scratch — you do not need to download `latents.pt`. The expected layout is:

```
DATA_ROOT/
└── <bird_id>/
    └── Processed/
        └── <bird_id>.h5
```

set via `TRAJECTORY_VARIANCE_DATA_ROOT` (see Setup).

## H5 file format (for users training from their own data)

The `train_ae.py` and `train_ot_flow.py` scripts read from per-bird HDF5 files at `DATA_ROOT/<bird>/Processed/<bird>.h5`. The expected schema:

### `/spectrograms`
- `/spectrograms/<file_id>` — `(123, N_frames)` int8. Linear-frequency spectrogram of a full recording session: 123 frequency bins (312–8000 Hz), 250 fps (sr=32000, hop=128), int8 quantized.
- `/spectrograms/file_id` — `(N_files,)` int32. Maps spectrogram dataset keys to file IDs.

### `/segments`
- `/segments/segment_id` — `(N,)` int32. Unique vocalization identifier.
- `/segments/file_id` — `(N,)` int32. Links to `/files`.
- `/segments/onset_sec` — `(N,)` float32. Onset within the recording file.
- `/segments/duration_sec` — `(N,)` float32. Vocalization duration.

Arrays are in H5 storage order; the training code re-sorts to pipeline order (sorted by day-post-hatch then filename then onset).

### `/files`
- `/files/file_id` — `(N_files,)` int32.
- `/files/filename` — `(N_files,)` bytes. Format `<bird_id>_<datenum>.<ms_since_midnight>_<month>_<day>_<hour>_<minute>_<second>.wav`. The training code subtracts `parameters/hatch_datenum` from `<datenum>` and rounds to the nearest integer to obtain days post-hatch.

### `/parameters` (HDF5 group with attributes only)
- `audio_sr` (32000), `hop_length` (128), `spec_n_fft` (512), `spec_min_freq` (312), `spec_max_freq` (8000)
- `hatch_datenum` (per-bird Excel/OLE serial date number of the hatch date)
- `spec_global_min`, `spec_global_max` (per-bird floats; pre-quantization range, useful for recovering normalized values)

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

Output: `Counterfactual_generation/models/paper_eval_results.json`. A **reference copy is committed** to this repo — diff your re-run against it to verify you reproduced the exact paper numbers.

**Note on the spectral flatness cache:** A small fraction of segments (~0.4% across all birds) carry NaN values because they were added to the dataset after the spectrogram preprocessing snapshot used to compute these features was generated. The evaluation code excludes these from the analysis.

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
@unpublished{anonymous2026trajectory,
  title  = {Trajectory Variance: An Unsupervised Measure of Developmental Vocal Plasticity in Birdsong},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at Interspeech 2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
