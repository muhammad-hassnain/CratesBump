#!/usr/bin/env python3
"""Permutation ablation on a trained classifier, per feature and per family.

For each feature: shuffle its column in the test matrix and measure the
drop in macro-F1. Repeats n_reps times, reports mean ± sd. Because many
features are correlated (e.g. tcm / tau / nbf), shuffling one column
understates what a whole family carries, so each feature family
(FAMILIES below, the six families of the paper's feature table) is also
shuffled as a block, rows permuted jointly.

The test fold holds out whole source repositories by default
(--group-by repo, see evaluate.py), so no test crate shares a repository,
and hence repository-wide GitHub features, with a training crate.

Reports each feature's (and each feature family's) contribution rather than
one lumped delta per family. In particular, the kw[mpb] and nbf features are
mostly-zero columns of unclear value.

Usage:
    python3 scripts/feature_ablation.py \\
        --data   data/cratesbump.csv.gz \\
        --split  grouped --group-by repo --seed 42 --reps 5 \\
        --out    results/feature_ablation_rebuilt_cargo_rf.json
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402
from evaluate import make_groups  # noqa: E402

FAMILIES = {
    "code (AST, files, comments)": P.FEATURE_COLS[:20],
    "dependencies": ["tcct", "pa", "pd", "pu"],
    "complexity and size": ["acycd", "clcrd", "cycd", "la", "ld"],
    "release cadence": ["rdtd"],
    "GitHub commits": ["tcm", "tau", "nbf", "kwm", "kwp", "kwb", "aml"],
    "GitHub issues and PRs": ["poi", "pci", "popr", "pcpr"],
}
assert sum(len(v) for v in FAMILIES.values()) == len(P.FEATURE_COLS) == 41


def _prep(csv_path: str):
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    for c in P.FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df = df.dropna(subset=P.FEATURE_COLS + ["release_type"])
    df = df[df["release_type"].isin(P.CLASSES)].reset_index(drop=True)
    df[P.FEATURE_COLS] = df[P.FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = df[P.FEATURE_COLS].values.astype(float)
    y = df["release_type"].map({c: i for i, c in enumerate(P.CLASSES)}).values
    return df, X, y


def _split(kind: str, X, y, groups, test_size: float, seed: int):
    if kind == "grouped":
        s = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(s.split(X, y, groups))
    if kind == "stratified":
        tr, te = train_test_split(np.arange(len(X)), test_size=test_size,
                                  stratify=y, random_state=seed, shuffle=True)
        return tr, te
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None,
                    help="Path to a .joblib saved by pipeline.py --save-models. "
                         "If omitted, a Random Forest is trained on the training "
                         "fold with pipeline.DEFAULT_PARAMS + SMOTE, i.e. the same "
                         "recipe and fold as evaluate.py seed 0.")
    ap.add_argument("--split", default="grouped", choices=["grouped", "stratified"])
    ap.add_argument("--group-by", choices=["repo", "crate"], default="repo")
    ap.add_argument("--repos", default="results/crates_all_repos.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42,
                    help="Must match the seed used when training (so the test "
                         "fold is the same held-out slice).")
    ap.add_argument("--reps", type=int, default=5,
                    help="Permutation repetitions per feature.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading {args.data} ...")
    df, X, y = _prep(args.data)
    groups = make_groups(df, args.group_by, args.repos)

    tr, te = _split(args.split, X, y, groups, args.test_size, args.seed)
    X_te, y_te = X[te], y[te]
    print(f"  test set: {len(y_te):,} rows")

    if args.model:
        print(f"Loading model: {args.model}")
        clf = joblib.load(args.model)
    else:
        print("Training Random Forest (DEFAULT_PARAMS + SMOTE) on the training fold ...")
        clf = P.build_classifier("Random Forest", P.DEFAULT_PARAMS["Random Forest"], 42)
        X_fit, y_fit = P.apply_smote(X[tr], y[tr], args.seed)
        clf.fit(X_fit, y_fit)

    y_pred = clf.predict(X_te)
    base = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"  baseline macro-F1: {base:.4f}")

    rng = np.random.default_rng(args.seed)
    rows = []
    for j, feat in enumerate(P.FEATURE_COLS):
        drops = []
        for r in range(args.reps):
            Xp = X_te.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, j] = X_te[perm, j]
            yp = clf.predict(Xp)
            drops.append(base - f1_score(y_te, yp, average="macro", zero_division=0))
        drops = np.array(drops)
        rows.append({
            "feature":   feat,
            "drop_mean": float(drops.mean()),
            "drop_std":  float(drops.std()),
            "drop_min":  float(drops.min()),
            "drop_max":  float(drops.max()),
        })
        print(f"  {feat:>8s}  drop={drops.mean():+.4f} ± {drops.std():.4f}")

    rows.sort(key=lambda r: -r["drop_mean"])

    fam_rows = []
    for fam, cols in FAMILIES.items():
        idx = [P.FEATURE_COLS.index(c) for c in cols]
        drops = []
        for r in range(args.reps):
            Xp = X_te.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idx] = X_te[perm][:, idx]
            drops.append(base - f1_score(y_te, clf.predict(Xp), average="macro", zero_division=0))
        drops = np.array(drops)
        fam_rows.append({"family": fam, "features": cols, "drop_mean": float(drops.mean()),
                         "drop_std": float(drops.std())})
        print(f"  family {fam:>28s}  drop={drops.mean():+.4f} ± {drops.std():.4f}")
    fam_rows.sort(key=lambda r: -r["drop_mean"])

    out = {
        "data":        args.data,
        "model":       args.model or "trained in-script: Random Forest, DEFAULT_PARAMS, SMOTE",
        "split":       args.split,
        "group_by":    args.group_by,
        "family_ablations": fam_rows,
        "test_rows":   int(len(y_te)),
        "baseline_f1": float(base),
        "reps":        args.reps,
        "ablations":   rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.out}")

    print("\nTop 10 most-important features (by F1 drop when shuffled):")
    for r in rows[:10]:
        print(f"  {r['feature']:>8s}  {r['drop_mean']:+.4f}")
    print("\nBottom 5 (candidates for removal):")
    for r in rows[-5:]:
        print(f"  {r['feature']:>8s}  {r['drop_mean']:+.4f}")


if __name__ == "__main__":
    main()
