#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run learner.py over all datasets in data/converted_mlregtest/*

Folder layout assumed, per dataset folder (e.g., data/converted_mlregtest/SL2.1.0):
  - <anything>LearningData*.txt
  - <anything>TestingData*.txt
  - (optional) <anything>Features*.txt   # only used if structure=nonlocal

Outputs:
  - result/mlregtest/<dataset>/judgment_struc-...txt
  - result/mlregtest/<dataset>/matrix_struc-...txt
"""

import argparse
import datetime
import glob
import os
import sys

# --- import your learner ---
try:
    from learner import Phonotactics  # if your file is named learner.py
except ModuleNotFoundError:
    # fallback if the file is named differently (e.g., learner_segment_based.py)
    from learner_segment_based import Phonotactics


def find_file(dirpath, patterns):
    """Return the first file in dirpath matching any glob pattern in patterns (case-insensitive)."""
    # Build case-insensitive match by checking lowercase names
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(dirpath, pat)))
    # If nothing matched, try a looser fallback
    if not candidates:
        lower = {f.lower(): f for f in glob.glob(os.path.join(dirpath, "*")) if os.path.isfile(f)}
        for key in list(lower.keys()):
            for pat in patterns:
                # crude ci match: replace wildcard and check substring
                if pat.replace("*", "").lower() in key:
                    return lower[key]
        return None
    return sorted(candidates)[0]


def run_one_dataset(root, dataset, args):
    ds_dir = os.path.join(root, dataset)
    if not os.path.isdir(ds_dir):
        return False, f"skip: not a directory → {ds_dir}"

    # Find files inside the dataset folder
    training = find_file(ds_dir, ["*LearningData*.txt", "*Train*.txt"])
    testing  = find_file(ds_dir, ["*TestingData*.txt", "*Test*.txt"])
    features = find_file(ds_dir, ["*Features*.txt", "*Feature*.txt"])  # optional for structure='local'

    if training is None or testing is None:
        return False, f"missing training/testing in {ds_dir}"

    # Result paths
    out_dir = os.path.join("result", "mlregtest", dataset)
    os.makedirs(out_dir, exist_ok=True)

    stamp = f"struc-{args.structure}_thr-{args.threshold}_pen-{args.weight}_model-{args.model}"
    judgment = os.path.join(out_dir, f"judgment_{stamp}.txt")
    matrix   = os.path.join(out_dir, f"matrix_{stamp}.txt")

    # Configure & run
    ph = Phonotactics()
    ph.language = dataset                # keep dataset label (e.g., SL2.1.0)
    ph.structure = args.structure        # 'local' or 'nonlocal'
    ph.threshold = args.threshold        # max threshold
    ph.penalty_weight = args.weight
    ph.model = args.model                # 'filtering' or 'gross'
    ph.filter = True
    ph.padding = False

    # If nonlocal structure but no feature file, warn and fall back to local behavior
    feature_path = features if features is not None else ""
    if args.structure == "nonlocal" and not features:
        print(f"[warn] {dataset}: structure=nonlocal but no Features*.txt found — proceeding (it will behave like 'local').")

    try:
        ph.main(training, feature_path, judgment, testing, matrix)
        return True, f"✓ {dataset}: wrote\n  - {judgment}\n  - {matrix}"
    except Exception as e:
        return False, f"✗ {dataset}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Batch-run learner.py on mlregtest datasets.")
    parser.add_argument("--root", default="data/converted_mlregtest",
                        help="Root folder containing dataset subfolders (default: data/converted_mlregtest).")
    parser.add_argument("--structure", choices=["local", "nonlocal"], default="local",
                        help="Structure type for Phonotactics (default: local).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Maximum threshold (default: 0.5).")
    parser.add_argument("--weight", type=float, default=10.0,
                        help="Penalty weight (default: 10).")
    parser.add_argument("--model", choices=["filtering", "gross"], default="filtering",
                        help="Model type (default: filtering).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional list of dataset folder names to run (e.g., SL2.1.0 SP2.1.1).")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Root not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    # Discover datasets
    all_entries = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))])
    if args.only:
        datasets = [d for d in all_entries if d in set(args.only)]
    else:
        datasets = all_entries

    if not datasets:
        print("No dataset folders found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(datasets)} dataset(s) under {args.root}: {', '.join(datasets)}\n")

    ok = 0
    for ds in datasets:
        success, msg = run_one_dataset(args.root, ds, args)
        print(msg)
        ok += int(success)

    print(f"\nDone: {ok}/{len(datasets)} succeeded.")


if __name__ == "__main__":
    main()
