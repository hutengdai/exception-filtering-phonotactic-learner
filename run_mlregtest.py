#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
import importlib

def find_file(dirpath, patterns):
    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(dirpath, pat)))
        if matches:
            return matches[0]
    # very loose fallback: substring match, case-insensitive
    lower = {f.lower(): f for f in glob.glob(os.path.join(dirpath, "*")) if os.path.isfile(f)}
    for key, val in lower.items():
        for pat in patterns:
            needle = pat.replace("*", "").lower()
            if needle and needle in key:
                return val
    return None

def canonical_language_label(dataset_folder: str) -> str:
    # turn "SL.2.1.0" -> "SL2.1.0", "SP.2.1.1" -> "SP2.1.1"
    if dataset_folder.startswith("SL.") or dataset_folder.startswith("SP."):
        return dataset_folder.replace(".", "", 1)
    return dataset_folder

def import_phonotactics(lang_label: str):
    """
    Import the learner module and set its module-global `language`
    so Phonotactics.__init__ can read it.
    """
    for modname in ("learner", "learner_segment_based"):
        try:
            mod = importlib.import_module(modname)
            # IMPORTANT: set the module-global `language` *before* instantiation
            setattr(mod, "language", lang_label)
            return mod.Phonotactics
        except ModuleNotFoundError:
            continue
    print("Could not import `learner` or `learner_segment_based`.", file=sys.stderr)
    sys.exit(1)

def run_one_dataset(root, dataset, args):
    ds_dir = os.path.join(root, dataset)
    if not os.path.isdir(ds_dir):
        return False, f"skip: not a directory → {ds_dir}"

    training = find_file(ds_dir, ["*LearningData*.txt", "*Train*.txt"])
    testing  = find_file(ds_dir, ["*TestingData*.txt", "*Test*.txt"])
    features = find_file(ds_dir, ["*Features*.txt", "*Feature*.txt"])  # only needed for --structure nonlocal

    if training is None or testing is None:
        return False, f"missing training/testing in {ds_dir}"

    # Normalize language label as requested (e.g., SL.2.1.0 → SL2.1.0)
    lang_label = canonical_language_label(dataset)

    # Import the class and ensure the module-global `language` is set
    Phonotactics = import_phonotactics(lang_label)

    # Prepare outputs
    out_dir = os.path.join("result", "mlregtest", dataset)
    os.makedirs(out_dir, exist_ok=True)

    stamp = f"struc-{args.structure}_thr-{args.threshold}_pen-{args.weight}_model-{args.model}"
    judgment = os.path.join(out_dir, f"judgment_{stamp}.txt")
    matrix   = os.path.join(out_dir, f"matrix_{stamp}.txt")

    # Configure & run
    ph = Phonotactics()
    ph.language = lang_label
    ph.structure = args.structure          # 'local' or 'nonlocal'
    ph.threshold = args.threshold
    ph.penalty_weight = args.weight
    ph.model = args.model                  # 'filtering' or 'gross'
    ph.filter = True
    ph.padding = False

    feature_path = features if (args.structure == "nonlocal" and features) else (features or "")
    if args.structure == "nonlocal" and not features:
        print(f"[warn] {dataset}: structure=nonlocal but no Features*.txt found — proceeding anyway.")

    try:
        ph.main(training, feature_path, judgment, testing, matrix)
        return True, f"✓ {dataset}  (language={lang_label})\n  - {judgment}\n  - {matrix}"
    except Exception as e:
        return False, f"✗ {dataset}: {e}"

def main():
    p = argparse.ArgumentParser(description="Batch-run learner over data/converted_mlregtest/*")
    p.add_argument("--root", default="data/converted_mlregtest",
                   help="Root folder with dataset subfolders (default: data/converted_mlregtest)")
    p.add_argument("--structure", choices=["local", "nonlocal"], default="local")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--weight", type=float, default=10.0)
    p.add_argument("--model", choices=["filtering", "gross"], default="filtering")
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only these dataset folder names (space-separated).")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        print(f"Root not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    all_datasets = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))])
    datasets = [d for d in all_datasets if (not args.only or d in set(args.only))]

    print(f"Found {len(datasets)} dataset(s) under {args.root}: {', '.join(datasets)}\n")

    ok = 0
    for ds in datasets:
        success, msg = run_one_dataset(args.root, ds, args)
        print(msg)
        ok += int(success)

    print(f"\nDone: {ok}/{len(datasets)} succeeded.")

if __name__ == "__main__":
    main()
