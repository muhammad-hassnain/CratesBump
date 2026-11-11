#!/usr/bin/env python3
"""Evaluate bump classifiers under multiple split protocols, with variance.

Motivation: a random split over release PAIRS leaks crate identity, because most
crates contribute many correlated pairs. This script compares:
  - stratified : random 80/20 stratified by label (the leaky baseline);
  - grouped    : GroupShuffleSplit by group -> 0% test-group overlap;
  - temporal   : train on earlier releases, test on later (needs a `created_at`
                 column); a realistic prediction setting. Groups are assigned
                 whole: a group is in test only if all its pairs are later.
The group is the crate (`--group-by crate`) or its source repository
(`--group-by repo`, the default). Crates of one repository share their
repository-wide GitHub features, so grouping by crate alone lets a test crate's
repository appear in training. Crates without a repository URL form their own
group.
It reports mean +/- sd of macro-F1, MCC, and macro one-vs-rest ROC-AUC across
seeds, plus the measured test-crate overlap for each protocol. SMOTE is applied
inside the training fold only (via pipeline.train_and_evaluate).

    python3 scripts/evaluate.py data/<file>.csv.gz --splits stratified grouped --seeds 5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402  (reuse the exact models + training code)
from enrich_github import repo_key  # noqa: E402


def make_groups(df, group_by, repos_csv):
    if group_by == "crate":
        return df["package"].values
    if "repository" in df.columns:          # shipped in the headline file
        return np.array([r if isinstance(r, str) and r else f"crate:{p}"
                         for p, r in zip(df["package"], df["repository"])], dtype=object)
    rm = pd.read_csv(repos_csv, keep_default_na=False, na_values=[""])
    key = {p: repo_key(u) for p, u in zip(rm["package"], rm["repository"])}
    return np.array([key.get(p) or f"crate:{p}" for p in df["package"]], dtype=object)


def macro_auc(aucs):
    vals = [v for v in aucs.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def per_class_f1(y_true, y_pred, classes=P.CLASSES):
    """F1 per class (one entry per CLASS, in CLASS order)."""
    from sklearn.metrics import precision_recall_fscore_support
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    return {c: float(v) for c, v in zip(classes, f1)}


def prep(df):
    for c in P.FEATURE_COLS:            # tolerate older CSVs missing new columns
        if c not in df.columns:
            df[c] = 0.0
    df = df.dropna(subset=P.FEATURE_COLS + ["release_type"])
    df = df[df["release_type"].isin(P.CLASSES)].reset_index(drop=True)
    df[P.FEATURE_COLS] = df[P.FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = df[P.FEATURE_COLS].values.astype(float)
    y = df["release_type"].map({c: i for i, c in enumerate(P.CLASSES)}).values
    order = df["created_at"].values if "created_at" in df.columns else None
    return df, X, y, order


def make_split(kind, X, y, groups, order, seed, test_size):
    if kind == "stratified":
        s = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(s.split(X, y))
    if kind == "grouped":
        s = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(s.split(X, y, groups))
    if kind == "temporal":
        # Per-crate temporal cutoff: drop NaT rows, then assign crates whole
        # (a crate is in TEST only if ALL its pairs are later than cutoff).
        # Row-level sort with NaT dumps all unknown-date rows into test;
        # non-grouped temporal leaks within-crate correlation across time.
        if order is None:
            raise ValueError("temporal split needs a 'created_at' column")
        if groups is None:
            raise ValueError("temporal split needs a 'package' column (for grouping)")
        dates = pd.to_datetime(pd.Series(order), errors="coerce").values
        valid = ~pd.isna(dates)
        d_v = dates[valid]
        g_v = groups[valid]
        d_ns = d_v.astype("datetime64[ns]").astype("int64")
        cutoff_ns = int(np.quantile(d_ns, 1 - test_size))
        tmp = pd.DataFrame({"i": np.where(valid)[0], "pkg": g_v, "d_ns": d_ns})
        crate_max = tmp.groupby("pkg")["d_ns"].max()
        crate_min = tmp.groupby("pkg")["d_ns"].min()
        test_crates  = set(crate_min[crate_min >  cutoff_ns].index)
        train_crates = set(crate_max[crate_max <= cutoff_ns].index)
        train_rows = tmp[tmp["pkg"].isin(train_crates)]["i"].values
        test_rows  = tmp[tmp["pkg"].isin(test_crates)]["i"].values
        return train_rows, test_rows
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--splits", nargs="+", default=["stratified", "grouped"],
                    choices=["stratified", "grouped", "temporal"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--sample", type=int, default=0, help="subsample rows for a quick run")
    ap.add_argument("--models", nargs="+", default=None,
                    help="only these models (names as in pipeline.CLASSIFIERS)")
    ap.add_argument("--group-by", choices=["repo", "crate"], default="repo",
                    help="unit held out by the grouped and temporal splits")
    ap.add_argument("--repos", default="results/crates_all_repos.csv",
                    help="package,repository map used by --group-by repo")
    ap.add_argument("--out-json", default=None,
                    help="write per-split, per-model mean±sd results to this JSON "
                         "(make reproduce-section-4 writes the files make headline reads)")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, keep_default_na=False, na_values=[""])
    if a.sample and len(df) > a.sample:
        df = df.sample(a.sample, random_state=42)
    df, X, y, order = prep(df)
    groups = make_groups(df, a.group_by, a.repos)
    crates = df["package"].values
    print(f"{a.csv}: {len(df):,} rows, {len(set(crates)):,} crates, "
          f"{len(set(groups)):,} groups (by {a.group_by})")

    out: dict = {"csv": a.csv, "n_rows": int(len(df)),
                 "n_crates": int(len(set(crates))), "group_by": a.group_by,
                 "n_groups": int(len(set(groups))),
                 "splits": {}}

    for kind in a.splits:
        n_seeds = a.seeds if kind != "temporal" else 1  # temporal is deterministic
        acc: dict = {name: {"f1": [], "mcc": [], "auc": [],
                            "f1_major": [], "f1_minor": [], "f1_patch": []}
                     for name in P.CLASSIFIERS if not a.models or name in a.models}
        overlaps, crate_overlaps, sizes = [], [], []
        for s in range(n_seeds):
            tr, te = make_split(kind, X, y, groups, order, 42 + s, a.test_size)
            test_groups = set(groups[te])
            overlaps.append(len(test_groups & set(groups[tr])) / max(len(test_groups), 1))
            tc = set(crates[te])
            crate_overlaps.append(len(tc & set(crates[tr])) / max(len(tc), 1))
            sizes.append({"train_rows": int(len(tr)), "test_rows": int(len(te)),
                          "train_crates": int(len(set(crates[tr]))), "test_crates": int(len(tc)),
                          "test_groups": int(len(test_groups))})
            res = P.train_and_evaluate(X[tr], X[te], y[tr], y[te], 42 + s, "smote",
                                       models=a.models)
            for r in res:
                acc[r["name"]]["f1"].append(r["macro_f1"])
                acc[r["name"]]["mcc"].append(r["mcc"])
                acc[r["name"]]["auc"].append(macro_auc(r["aucs"]))
                pc = per_class_f1(y[te], r["y_pred"])
                acc[r["name"]]["f1_major"].append(pc["major"])
                acc[r["name"]]["f1_minor"].append(pc["minor"])
                acc[r["name"]]["f1_patch"].append(pc["patch"])

        split_out: dict = {"n_seeds": n_seeds,
                           "test_group_overlap": float(np.mean(overlaps)),
                           "test_crate_overlap": float(np.mean(crate_overlaps)),
                           "sizes": sizes,
                           "models": {}}
        print(f"\n=== {kind} split (n={n_seeds}, by {a.group_by}; test-group overlap "
              f"{100 * np.mean(overlaps):.1f}%, test-crate overlap "
              f"{100 * np.mean(crate_overlaps):.1f}%; sizes {sizes[0]}) ===")
        print(f"{'Model':22s} {'Macro F1':>14s} {'F1 major':>14s} {'F1 minor':>14s} {'F1 patch':>14s} {'MCC':>14s}")
        for name in acc:
            f = acc[name]
            def ms(x): return f"{np.mean(x):.3f}+/-{np.std(x):.3f}"
            print(f"{name:22s} {ms(f['f1']):>14s} {ms(f['f1_major']):>14s} "
                  f"{ms(f['f1_minor']):>14s} {ms(f['f1_patch']):>14s} {ms(f['mcc']):>14s}")
            split_out["models"][name] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "values": [float(x) for x in v]}
                for k, v in f.items()
            }
        out["splits"][kind] = split_out

    if a.out_json:
        import json, os
        os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
        with open(a.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  wrote structured results -> {a.out_json}")


if __name__ == "__main__":
    main()
