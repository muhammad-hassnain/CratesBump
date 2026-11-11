#!/usr/bin/env python3
"""
Classify the semver bump type for a specific crate release.

Usage:
    python3 predict.py <crate> <version> [--model <name>]

    python3 predict.py serde 1.0.200
    python3 predict.py serde 1.0.200 --model "Random Forest"

The script has three fast paths, checked in order:
  1. Ground truth known  — version is in crate_csvs/<crate>.csv with a label.
                           Returns immediately; no model inference needed.
  2. Cached prediction   — result was saved to predictions_cache.json on a
                           previous run.  Returns the cached table instantly.
  3. Full inference      — extracts features (if needed), runs all classifiers,
                           saves the result to the cache for next time.
"""

import os
import sys
import json
import argparse
import subprocess
import warnings
import datetime

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

EXTRACT_BIN      = os.path.join("target", "release", "extract")
CRATE_DIR        = "crate_csvs"
PREDICTIONS_CACHE = "predictions_cache.json"


# Predictions cache

def _load_cache() -> dict:
    if os.path.isfile(PREDICTIONS_CACHE):
        try:
            with open(PREDICTIONS_CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    with open(PREDICTIONS_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(crate: str, version: str, model_filter: str | None) -> str:
    v = version.lstrip("v")
    return f"{crate}@{v}" if model_filter is None else f"{crate}@{v}#{model_filter}"


def get_latest_version(crate: str) -> str | None:
    """Fetch the latest published version from crates.io."""
    import urllib.request
    try:
        url = f"https://crates.io/api/v1/crates/{crate}"
        req = urllib.request.Request(url, headers={"User-Agent": "semver-ml-predict/0.1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["crate"]["newest_version"]
    except Exception:
        return None



def local_cargo_version(path: str) -> str | None:
    """Read the [package] version from a local Cargo.toml."""
    import re
    cargo = os.path.join(path, "Cargo.toml")
    if not os.path.isfile(cargo):
        return None
    try:
        text = open(cargo).read()
        m = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        return m.group(1) if m else None
    except Exception:
        return None


# Ground truth lookup

def get_ground_truth(crate: str, version: str) -> str | None:
    """Return the known release_type label if this version is in crate_csvs/."""
    csv_path = os.path.join(CRATE_DIR, f"{crate}.csv")
    if not os.path.isfile(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["version", "release_type"], keep_default_na=False, na_values=[""])
        v_clean = version.lstrip("v")
        match = df[df["version"].astype(str).str.lstrip("v") == v_clean]
        if not match.empty:
            return str(match["release_type"].iloc[0])
    except Exception:
        pass
    return None


# Crate CSV / binary helpers

def find_extract_binary():
    if os.path.isfile(EXTRACT_BIN):
        return EXTRACT_BIN
    import shutil
    return shutil.which("extract")


def ensure_crate_csv(crate: str) -> tuple[str, bool] | tuple[None, bool]:
    """Return (path, was_cached) for crate_csvs/<crate>.csv, extracting if needed."""
    os.makedirs(CRATE_DIR, exist_ok=True)
    csv_path = os.path.join(CRATE_DIR, f"{crate}.csv")

    if os.path.isfile(csv_path):
        print(f"Using cached features: {csv_path}")
        return csv_path, True

    binary = find_extract_binary()
    if binary is None:
        print(f"Error: extract binary not found at {EXTRACT_BIN}")
        print("  Run 'make build' first.")
        return None, False

    print(f"Extracting features for '{crate}' → {csv_path} …")
    try:
        result = subprocess.run(
            [binary, crate, "--output", csv_path],
            capture_output=False, text=True,
        )
        if result.returncode != 0:
            return None, False
    except FileNotFoundError:
        print(f"Error: could not execute {binary}")
        return None, False

    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        print("Extraction produced no output.")
        return None, False

    return csv_path, False


def fetch_missing_version(crate: str, version: str, csv_path: str) -> bool:
    """
    If version is missing from the crate CSV, try extracting features for it again. 
    """
    import tempfile

    binary = find_extract_binary()
    if binary is None:
        print(f"Error: extract binary not found at {EXTRACT_BIN}. Run 'make build' first.")
        return False

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(tmp_fd)

    try:
        print(f"Version '{version}' missing from cached CSV — re-extracting {crate} …")
        result = subprocess.run(
            [binary, crate, "--output", tmp_path],
            capture_output=False, text=True,
        )
        if result.returncode != 0 or not os.path.isfile(tmp_path):
            return False

        new_df = pd.read_csv(tmp_path, keep_default_na=False, na_values=[""])
        v_clean = version.lstrip("v")
        new_df["_vc"] = new_df["version"].astype(str).str.lstrip("v")
        new_row = new_df[new_df["_vc"] == v_clean].drop(columns=["_vc"])

        if new_row.empty:
            print(f"Version '{version}' not found even after re-extraction "
                  f"(version may not exist on crates.io or was skipped due to unparseable semver)."
                  f"Please raise an issue on GitHub.")
            
            return False

        existing_df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
        merged = pd.concat([existing_df, new_row], ignore_index=True)
        merged = merged.drop_duplicates(subset=["package", "version"], keep="last")
        merged.to_csv(csv_path, index=False)
        print(f"Appended {crate} {version} to {csv_path}")
        return True

    finally:
        if os.path.isfile(tmp_path):
            os.unlink(tmp_path)


# Model loading

def load_models(model_dir: str):
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        print(f"Error: {meta_path} not found.")
        print(f"  Run 'make pipeline' first.")
        return None, None

    with open(meta_path) as f:
        meta = json.load(f)

    classifiers = {}
    for name in meta["classifiers"]:
        safe_name = name.replace(" ", "")
        path = os.path.join(model_dir, f"{safe_name}.joblib")
        if not os.path.isfile(path):
            print(f"Warning: model file missing: {path} — skipping")
            continue
        classifiers[name] = joblib.load(path)

    return classifiers, meta


# Output helpers

def _print_table(rows: list[dict], classes: list[str]):
    """Print the per-classifier probability table."""
    w = max(len(r["name"]) for r in rows) + 2
    print("─" * (w + 36))
    print(f"{'Classifier':<{w}} {'Prediction':>12}  {'major':>7} {'minor':>7} {'patch':>7}")
    print("─" * (w + 36))
    for r in rows:
        p = r["probabilities"]
        print(
            f"{r['name']:<{w}} {r['prediction']:>12}"
            f"  {p.get('major', 0):>7.3f}"
            f" {p.get('minor', 0):>7.3f}"
            f" {p.get('patch', 0):>7.3f}"
        )
    print("─" * (w + 36))


# Core prediction 

def predict_version(
    crate: str,
    version: str,
    csv_path: str,
    classifiers: dict,
    meta: dict,
    was_cached: bool,
    model_filter: str | None,
    save_cache: bool = True,
    display_version: str | None = None,
):
    label = display_version if display_version is not None else version
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    v_clean = version.lstrip("v")
    df["version_clean"] = df["version"].astype(str).str.lstrip("v")

    row = df[df["version_clean"] == v_clean]
    if row.empty:
        if not fetch_missing_version(crate, version, csv_path):
            return
        # Reload after append
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
        df["version_clean"] = df["version"].astype(str).str.lstrip("v")
        row = df[df["version_clean"] == v_clean]
        if row.empty:
            return

    feature_cols = meta["feature_cols"]
    classes      = meta["classes"]
    X            = row[feature_cols].values.astype(float)

    if not np.any(X):
        print(f"\nNo differences detected — the local directory is identical to {label}.")
        print("Make some changes first, then re-run to get a prediction.")
        return

    idx = row.index[0]
    prev_info = ""
    if idx > 0 and idx - 1 in df.index:
        prev_info = f" (vs {df.loc[idx - 1, 'version']})"

    ground_truth = row["release_type"].iloc[0] if "release_type" in row.columns else None

    print(f"\nPredicting bump type for  {crate} {label}{prev_info}")
    if ground_truth:
        print(f"Ground truth label:       {ground_truth}")
    if was_cached:
        print(f"⚠  {crate} {label} was part of the training data — "
              f"confidence may be overstated.")

    # Run all classifiers and collect rows
    result_rows = []
    for name, clf in classifiers.items():
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
        else:
            scores = clf.decision_function(X)
            if scores.ndim == 1:
                scores = scores.reshape(1, -1)
            e = np.exp(scores[0] - scores[0].max())
            proba = e / e.sum()

        pred_label = classes[int(np.argmax(proba))]
        result_rows.append({
            "name":          name,
            "prediction":    pred_label,
            "probabilities": {c: float(proba[i]) for i, c in enumerate(classes)},
        })

    _print_table(result_rows, classes)

    # Persist to cache (skipped for local-path predictions)
    if save_cache:
        cache = _load_cache()
        key   = _cache_key(crate, version, model_filter)
        cache[key] = {
            "crate":        crate,
            "version":      v_clean,
            "ground_truth": ground_truth,
            "was_training": was_cached,
            "results":      result_rows,
            "predicted_at": datetime.datetime.now().isoformat(),
        }
        _save_cache(cache)


# main

def main():
    parser = argparse.ArgumentParser(description="Predict semver bump type for a crate release")
    parser.add_argument("crate",              help="Crate name as on crates.io (e.g. serde, tokio)")
    parser.add_argument("version", nargs="?", help="Version to classify (default: latest published)")
    parser.add_argument("--local",            metavar="PATH",
                        help="Local repo path to use as current version instead of a published one")
    parser.add_argument("--prev",             metavar="VERSION",
                        help="Previous version to diff against (with --local). "
                             "If omitted, auto-detected from the local Cargo.toml.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Use a specific classifier. One of: "
            "'XGBoost', 'LightGBM', 'Hist Gradient Boosting', "
            "'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Decision Tree'. "
            "Default: all seven."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip the predictions cache and always run inference.",
    )
    args = parser.parse_args()

    # Local mode
    if args.local:
        if not os.path.isdir(args.local):
            print(f"Error: --local path '{args.local}' does not exist or is not a directory.")
            sys.exit(1)
        if not os.path.isfile(os.path.join(args.local, "Cargo.toml")):
            print(f"Error: '{args.local}' has no Cargo.toml — not a Rust crate.")
            sys.exit(1)

        if not args.prev:
            args.prev = local_cargo_version(args.local)
            if args.prev is None:
                print("Error: could not read version from Cargo.toml in the local path.")
                print("       Specify it manually with --prev <VERSION>.")
                sys.exit(1)
            print(f"Comparing against: {args.prev} (detected from local Cargo.toml)")

        binary = find_extract_binary()
        if binary is None:
            print(f"Error: extract binary not found at {EXTRACT_BIN}. Run 'make build' first.")
            sys.exit(1)

        print(f"Loading models from models/ …")
        classifiers, meta = load_models("models")
        if classifiers is None:
            sys.exit(1)

        if args.model is not None:
            if args.model not in classifiers:
                print(f"Error: unknown model '{args.model}'. Available: {', '.join(classifiers)}")
                sys.exit(1)
            classifiers = {args.model: classifiers[args.model]}

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            print(f"Extracting features for local path '{args.local}' vs {args.prev} …")
            result = subprocess.run(
                [binary, args.crate,
                 "--local", args.local,
                 "--prev-version", args.prev,
                 "--output", tmp_path],
                capture_output=False, text=True,
            )
            if result.returncode != 0 or not os.path.isfile(tmp_path):
                print("Extraction failed.")
                sys.exit(1)

            predict_version(
                args.crate, "local", tmp_path,
                classifiers, meta,
                was_cached=False, model_filter=args.model,
                save_cache=False,
                display_version=f"local (vs {args.prev})",
            )
        finally:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
        return

    # Normal mode
    if args.version is None:
        args.version = get_latest_version(args.crate)
        if args.version is None:
            print(f"Error: could not fetch latest version for '{args.crate}' from crates.io.")
            sys.exit(1)
        print(f"No version specified — using latest: {args.version}")

    v_clean = args.version.lstrip("v")

    # ground truth already known
    if not args.no_cache:
        gt = get_ground_truth(args.crate, args.version)
        if gt is not None:
            print(f"\n📋 Ground truth known — no inference needed.")
            print(f"   {args.crate} {v_clean}  →  {gt}")
            print(f"   (this version is in the training dataset: crate_csvs/{args.crate}.csv)")
            return

    # cached prediction 
    if not args.no_cache:
        cache = _load_cache()
        key   = _cache_key(args.crate, args.version, args.model)
        if key in cache:
            entry = cache[key]
            print(f"\n⚡ Cached result from {entry['predicted_at'][:10]}")
            if entry.get("was_training"):
                print(f"⚠  {args.crate} {v_clean} was part of the training data — "
                      f"confidence may be overstated.")
            gt = entry.get("ground_truth")
            if gt:
                print(f"Ground truth label:  {gt}")
            _print_table(entry["results"], list(entry["results"][0]["probabilities"].keys()))
            return

    # Full inference 
    print(f"Loading models from models/ …")
    classifiers, meta = load_models("models")
    if classifiers is None:
        sys.exit(1)

    if args.model is not None:
        if args.model not in classifiers:
            available = ", ".join(f"'{k}'" for k in classifiers)
            print(f"Error: unknown model '{args.model}'. Available: {available}")
            sys.exit(1)
        classifiers = {args.model: classifiers[args.model]}

    print(f"  Using: {', '.join(classifiers.keys())}")

    csv_path, was_cached = ensure_crate_csv(args.crate)
    if csv_path is None:
        print("Extraction failed — check stderr above for details.")
        sys.exit(1)

    predict_version(args.crate, args.version, csv_path, classifiers, meta, was_cached, args.model)


if __name__ == "__main__":
    main()
