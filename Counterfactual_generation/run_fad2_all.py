"""
Run pooled FAD2 for all birds/methods and aggregate results.

This wrapper executes compute_fad2.py for each selected (bird, method), then
collects per-run JSON outputs into one summary JSON + CSV.

Default method:  ot_flow
Default birds:   R4634, R4951, R5018

Example:
  python -m Counterfactual_generation.run_fad2_all --total_samples 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

BIRDS = ["R4634", "R4951", "R5018"]
METHODS = ["ot_flow"]

AE_DIRS = {
    "R4634": SCRIPT_DIR / "models" / "vae_R4634",
    "R4951": SCRIPT_DIR / "models" / "vae_R4951",
    "R5018": SCRIPT_DIR / "models" / "vae_R5018",
}

FLOW_DIRS = {
    "ot_flow": {
        "R4634": SCRIPT_DIR / "models" / "ot_flow_R4634_ot",
        "R4951": SCRIPT_DIR / "models" / "ot_flow_R4951_ot",
        "R5018": SCRIPT_DIR / "models" / "ot_flow_R5018_ot",
    },
}


def run_one(method: str, bird: str, total_samples: int, seed: int, n_points: int, n_repeats: int):
    ae_dir = AE_DIRS[bird]
    flow_dir = FLOW_DIRS[method][bird]

    cmd = [
        sys.executable,
        "-m",
        "Counterfactual_generation.compute_fad2",
        "--method", method,
        "--ae_dir", str(ae_dir),
        "--flow_dir", str(flow_dir),
        "--bird", bird,
        "--total_samples", str(total_samples),
        "--seed", str(seed),
        "--n_points", str(n_points),
        "--n_repeats", str(n_repeats),
    ]

    print("\n" + "=" * 80)
    print(f"Running: method={method}, bird={bird}")
    print("=" * 80)
    subprocess.run(cmd, check=True)

    result_path = flow_dir / "fad2_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected output not found: {result_path}")

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run pooled FAD2 for all birds/methods")
    parser.add_argument("--birds", nargs="+", default=BIRDS, choices=BIRDS)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--total_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_points", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--out_json", type=str, default="models/fad2_summary_check.json")
    parser.add_argument("--out_csv", type=str, default="models/fad2_summary_check.csv")
    args = parser.parse_args()

    repo_root = SCRIPT_DIR.parent

    def _rel(p):
        if not p:
            return p
        try:
            return str(Path(p).resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p)

    rows = []
    failures = []

    for method in args.methods:
        for bird in args.birds:
            try:
                r = run_one(method, bird, args.total_samples, args.seed, args.n_points, args.n_repeats)
                rows.append({
                    "method": method,
                    "bird": bird,
                    "fad_inf": r.get("fad_inf"),
                    "fad_inf_r2": r.get("fad_inf_r2"),
                    "full_fad": r.get("full_fad"),
                    "baseline_fad_inf": r.get("baseline_fad_inf"),
                    "baseline_fad_inf_r2": r.get("baseline_fad_inf_r2"),
                    "baseline_full_fad": r.get("baseline_full_fad"),
                    "n_real": r.get("n_real"),
                    "n_cf": r.get("n_cf"),
                    "flow_dir": _rel(r.get("flow_dir")),
                    "ae_dir": _rel(r.get("ae_dir")),
                })
            except Exception as exc:
                failures.append({"method": method, "bird": bird, "error": str(exc)})
                print(f"FAILED method={method} bird={bird}: {exc}")

    out_json = SCRIPT_DIR / args.out_json
    out_csv = SCRIPT_DIR / args.out_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": rows,
        "failures": failures,
        "settings": {
            "birds": args.birds,
            "methods": args.methods,
            "total_samples": args.total_samples,
            "seed": args.seed,
            "n_points": args.n_points,
            "n_repeats": args.n_repeats,
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        "method", "bird", "fad_inf", "fad_inf_r2", "full_fad",
        "baseline_fad_inf", "baseline_fad_inf_r2", "baseline_full_fad",
        "n_real", "n_cf", "flow_dir", "ae_dir",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "#" * 80)
    print(f"Saved summary JSON: {out_json}")
    print(f"Saved summary CSV:  {out_csv}")
    if failures:
        print(f"Completed with {len(failures)} failures.")
    else:
        print("Completed successfully.")


if __name__ == "__main__":
    main()
