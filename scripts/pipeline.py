#!/usr/bin/env python3
"""
Train and evaluate semver classifiers on collected crate data.

Usage:
    python3 pipeline.py                     # auto-loads all crate_csvs/*.csv
    python3 scripts/pipeline.py              # auto-loads all crate_csvs/*.csv
    python3 pipeline.py --synthetic         # smoke-test on synthetic data
    python3 pipeline.py --seed 0 --test-size 0.25
"""

import sys
import os
import json
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit,
)
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

warnings.filterwarnings("ignore")

# Local helper (same directory) for the SemVer-compliance crate filter.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semver_filter import compliant_packages

FEATURE_COLS = [
    # Change-type (20): source = AST + file/comment diff.
    "arf", "mrf", "drf", "anrf", "dnrf", "mnrf",
    "adm", "dem", "mpc", "mpd",
    "mla", "mld", "gva", "gvd",
    "icc", "dcc", "mcc",
    # AST move/rename counters (src/extractor/ast.rs); sparse but real
    # signal (see data/DATA_DICTIONARY.md for their coverage).
    "mom", "mnc", "mlm",
    # Dependencies (4): Cargo.toml diff.
    "tcct", "pa", "pd", "pu",
    # Complexity / code size (5)
    "acycd", "clcrd", "cycd", "la", "ld",
    # Time (1)
    "rdtd",
    # Development activity (3): source = GitHub REST commit crawl.
    "tcm", "tau", "nbf",
    # Textual / commit keywords (4): source = GitHub commit messages
    "kwm", "kwp", "kwb", "aml",
    # Issue/PR activity (4): source = GitHub REST issue/PR crawl.
    "poi", "pci", "popr", "pcpr",
]
# 41 trained features: 20 AST change-type (17 + the 3 move/rename counters
# mom/mnc/mlm, re-included 2026-10-12) + 4 dependency + 5 complexity
# + 1 time + 7 commit activity/keyword + 4 issue/PR.

CLASSES = ["major", "minor", "patch"]

TUNING_RESULTS_PATH = os.path.join("results", "tuning_results.json")


def build_classifier(name: str, params: dict, seed: int):
    """Construct a classifier for `name` using the given hyperparameter dict."""
    p = dict(params)  # shallow copy
    if name == "XGBoost":
        return xgb.XGBClassifier(**p, eval_metric="mlogloss", random_state=seed, n_jobs=-1)
    if name == "LightGBM":
        return lgb.LGBMClassifier(**p, random_state=seed, n_jobs=-1, verbose=-1)
    if name == "Hist Gradient Boosting":
        return HistGradientBoostingClassifier(**p, random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(**p, random_state=seed, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(**p, random_state=seed)
    if name == "Logistic Regression":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(**p, random_state=seed))])
    if name == "Decision Tree":
        return DecisionTreeClassifier(**p, random_state=seed)
    if name == "Majority baseline":
        return DummyClassifier(strategy="most_frequent", random_state=seed)
    if name == "Stratified-random baseline":
        return DummyClassifier(strategy="stratified", random_state=seed)
    raise ValueError(f"Unknown classifier: {name}")


# ── DEFAULT_PARAMS: auto-updated by `make upgrade` ───────────────────────────
DEFAULT_PARAMS = {
    "Majority baseline": {},
    "Stratified-random baseline": {},
    "XGBoost": {"subsample": 0.9, "n_estimators": 100, "min_child_weight": 1, "max_depth": 5, "learning_rate": 0.05, "gamma": 0.1, "colsample_bytree": 0.9},
    "Random Forest": {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "max_depth": 10},
    "LightGBM": {"subsample": 0.6, "num_leaves": 127, "n_estimators": 200, "min_child_samples": 10, "max_depth": 4, "learning_rate": 0.01, "colsample_bytree": 0.8},
    "Hist Gradient Boosting": {"min_samples_leaf": 30, "max_leaf_nodes": None, "max_iter": 300, "max_depth": 5, "learning_rate": 0.01, "l2_regularization": 0.5},
    "Gradient Boosting": {"subsample": 0.6, "n_estimators": 50, "min_samples_leaf": 5, "max_depth": 2, "learning_rate": 0.1},
    "Decision Tree": {"min_samples_split": 5, "min_samples_leaf": 5, "max_features": None, "max_depth": 8, "criterion": "gini"},
    "Logistic Regression": {"solver": "saga", "max_iter": 200, "C": 0.001},
}
# ── /DEFAULT_PARAMS ───────────────────────────────────────────────────────────

# Boosters that need an optional native lib (xgboost/lightgbm need OpenMP, i.e.
# libomp on macOS). If the lib can't be imported we skip just those models — the
# 5 pure-sklearn classifiers always run — so the pipeline never hard-fails on a
# missing system dependency.
_UNAVAILABLE = set()
if xgb is None:
    _UNAVAILABLE.add("XGBoost")
if lgb is None:
    _UNAVAILABLE.add("LightGBM")

# make classifiers with default params
CLASSIFIERS = {
    name: (lambda n: lambda: build_classifier(n, DEFAULT_PARAMS[n], 42))(name)
    for name in DEFAULT_PARAMS if name not in _UNAVAILABLE
}

# we load these if --best-params is passed
def load_best_params(path: str = TUNING_RESULTS_PATH) -> dict:
    """Load best params per model from tuning_results.json.
    Returns {model_name: params_dict}."""
    if not os.path.isfile(path):
        print(f"Error: {path} not found.")
        print("  Run 'make tune' first to generate tuning results.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    return {r["name"]: r["best_params"] for r in data["results"]}

# We apply SMOTE oversampling to the training set to help with class imbalance.
def apply_smote(X, y, seed):
    counts = np.bincount(y)
    k = min(5, int(counts.min()) - 1)
    if k < 1:
        return X, y
    try:
        return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    except Exception:
        return X, y

# Per-class Receiver Operating Characteristic - Area Under Curve (ROC-AUC) with one-vs-rest strategy.
# Receiver Operating Characteristic (ROC) curve plots the true positive rate against the false positive rate, while Area Under the Curve (AUC) summarizes the overall performance of the classifier. 
def roc_auc_ovr(clf, X_test, y_test):
    """Per-class ROC-AUC (one-vs-rest). Returns dict {class: auc}."""
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
    else:
        proba = clf.decision_function(X_test)
        # normalise decision scores to 3-column shape
        if proba.ndim == 1:
            proba = np.column_stack([-proba, proba])

    aucs = {}
    for i, cls in enumerate(CLASSES):
        col = y_bin[:, i]
        if col.sum() == 0 or col.sum() == len(col):
            aucs[cls] = float("nan")
        else:
            aucs[cls] = roc_auc_score(col, proba[:, i])
    return aucs

# Train classifiers and evaluate on the test set, returning a list of results dicts.
# `balance_strategy` controls how class imbalance is handled during training:
#   - "smote"    : synthesise minority examples via SMOTE (the paper's approach)
#   - "weighted" : leave the data alone and pass per-sample weights to fit() so
#                  the loss function up-weights rare classes
def train_and_evaluate(X_train, X_test, y_train, y_test, seed, balance_strategy="smote",
                       models=None):
    results = []
    smoted = None   # SMOTE output depends only on (X_train, y_train, seed): compute once

    for name, factory in CLASSIFIERS.items():
        if models and name not in models:
            continue
        print(f"    [{name}] fitting ...", flush=True)
        clf = factory()

        # Dummy baselines are reference points for the TRUE class distribution,
        # not the resampled one — fit on raw (X_train, y_train) regardless of
        # balance_strategy.
        is_dummy = isinstance(clf, DummyClassifier)
        if is_dummy:
            clf.fit(X_train, y_train)
        elif balance_strategy == "smote":
            if smoted is None:
                smoted = apply_smote(X_train, y_train, seed)
            clf.fit(*smoted)
        elif balance_strategy == "weighted":
            sw = compute_sample_weight("balanced", y_train)
            # Pipeline (used for Logistic Regression) needs the step-prefixed key.
            if isinstance(clf, Pipeline):
                clf.fit(X_train, y_train, clf__sample_weight=sw)
            else:
                clf.fit(X_train, y_train, sample_weight=sw)
        else:
            raise ValueError(f"Unknown balance_strategy: {balance_strategy!r}")

        y_pred   = clf.predict(X_test)
        aucs     = roc_auc_ovr(clf, X_test, y_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        mcc      = matthews_corrcoef(y_test, y_pred)

        results.append({
            "name":     name,
            "clf":      clf,
            "y_pred":   y_pred,
            "aucs":     aucs,
            "macro_f1": macro_f1,
            "mcc":      mcc,
        })
        print(f"    [{name}] done — macro_f1={macro_f1:.4f}  mcc={mcc:.4f}", flush=True)

    return results


def print_results(results, y_test):
    print()
    W = 82
    print("─" * W)
    print(f"{'Classifier':<26} {'Macro F1':>9} {'MCC':>7} {'Maj AUC':>9} {'Min AUC':>9} {'Pat AUC':>9}")
    print("─" * W)
    for r in results:
        a = r["aucs"]
        def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "  n/a"
        print(
            f"{r['name']:<26}"
            f" {fmt(r['macro_f1']):>9}"
            f" {fmt(r['mcc']):>7}"
            f" {fmt(a['major']):>9}"
            f" {fmt(a['minor']):>9}"
            f" {fmt(a['patch']):>9}"
        )
    print("─" * W)


def synthetic_dataset(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for pkg in range(30):
        for _ in range(40):
            rt = rng.choice(CLASSES, p=[0.1, 0.3, 0.6])
            scale = {"major": 3.0, "minor": 1.5, "patch": 0.5}[rt]
            feat = rng.exponential(scale, size=len(FEATURE_COLS))
            rows.append({"release_type": rt, **dict(zip(FEATURE_COLS, feat))})
    return pd.DataFrame(rows)


def _score_color(v: float, lo: float, hi: float, hue: str = "green"):
    """Map a scalar in [lo, hi] to an RGB tuple.

    hue="green"   ->  white (low) to green (high)
    hue="blue"    ->  white (low) to steel-blue (high)
    hue="neutral" ->  light-grey regardless of value
    """
    if np.isnan(v):
        return (0.93, 0.93, 0.93)
    intensity = max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-9)))
    if hue == "green":
        return (1.0 - intensity * 0.45, 0.78 + intensity * 0.18, 1.0 - intensity * 0.45)
    if hue == "blue":
        return (0.78 - intensity * 0.28, 0.88 - intensity * 0.18, 1.0 - intensity * 0.05)
    return (0.93, 0.93, 0.93)


def save_results_image(results, df_stats, output_path, balance_strategy="smote"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    clf_names  = [r["name"] for r in results]
    # Column order: Macro F1 | MCC | Major AUC | Minor AUC | Patch AUC
    col_labels = ["Macro F1", "MCC", "Maj AUC", "Min AUC", "Pat AUC"]

    cell_text, cell_colors = [], []
    for r in results:
        row, colors = [], []

        # Macro F1 
        f1 = r["macro_f1"]
        row.append(f"{f1:.3f}")
        colors.append(_score_color(f1, 0.0, 1.0, "green"))

        # MCC (-1 -> 1, but realistically 0 -> 1 for good models) 
        mcc = r["mcc"]
        row.append(f"{mcc:.3f}")
        colors.append(_score_color(mcc, 0.0, 1.0, "blue"))

        # Per-class AUC (0.5 -> 1)
        for cls in CLASSES:
            v = r["aucs"].get(cls, float("nan"))
            row.append(f"{v:.3f}" if not np.isnan(v) else "n/a")
            colors.append(_score_color(v, 0.5, 1.0, "green"))

        cell_text.append(row)
        cell_colors.append(colors)

    n_rows = len(results)
    fig, ax = plt.subplots(figsize=(10, 1.0 + n_rows * 0.52))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=clf_names,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.15, 1.9)

    # Header row: dark background, white text
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Row labels (classifier names)
    for i in range(len(clf_names)):
        tbl[i + 1, -1].set_facecolor("#dce3ea")
        tbl[i + 1, -1].set_text_props(fontweight="bold")

    strategy_label = {"smote": "SMOTE", "weighted": "class-weighted"}.get(balance_strategy, balance_strategy)
    title = f"Evaluation — 80/20 stratified split  ·  imbalance handling: {strategy_label}"
    subtitle = (
        f"{df_stats['crates']:,} crates · {df_stats['pairs']:,} version pairs"
        f"  ({df_stats['patch']:,} patch / {df_stats['minor']:,} minor / {df_stats['major']:,} major)"
    ) if df_stats else ""

    fig.text(0.5, 0.97, title,    ha="center", va="top", fontsize=12, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.88, subtitle, ha="center", va="top", fontsize=8.5, color="#555555")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Results image  -> {output_path}")


def save_models(results, model_dir, feature_cols, classes):
    os.makedirs(model_dir, exist_ok=True)
    saved = []
    for r in results:
        safe_name = r["name"].replace(" ", "")
        path = os.path.join(model_dir, f"{safe_name}.joblib")
        joblib.dump(r["clf"], path)
        saved.append(path)

    meta = {
        "feature_cols": feature_cols,
        "classes": classes,
        "classifiers": [r["name"] for r in results],
        "saved_at": datetime.datetime.now().isoformat(),
    }
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return saved


def load_crate_csvs(crate_dir: str = "crate_csvs") -> pd.DataFrame:
    """Concatenate all per-crate CSVs in crate_dir into one DataFrame."""
    import glob
    files = sorted(glob.glob(os.path.join(crate_dir, "*.csv")))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, keep_default_na=False, na_values=[""]))
        except Exception as e:
            print(f"  Warning: skipping {f} ({e})")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Train semver classifiers")
    parser.add_argument("csv", nargs="?", help="Explicit CSV path (default: merge all crate_csvs/*.csv)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke-test, ignores CSV)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for testing [0.2]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0,
                        help="Randomly subsample to at most N rows before training "
                             "(0 = use all). Handy for a quick demo (see 'make demo').")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip writing evaluation PNG(s); print the table only.")
    parser.add_argument("--save-models", action="store_true", help="Save trained classifiers to --model-dir")
    parser.add_argument("--model-dir", default="models", help="Directory to save models [models/]")
    parser.add_argument("--results-dir", default="results", help="Directory to save evaluation image [results/]")
    parser.add_argument("--best-params", action="store_true",
                        help=f"Use best hyperparameters from {TUNING_RESULTS_PATH} (produced by tune.py)")
    parser.add_argument("--balance", choices=["both", "smote", "weighted"], default="both",
                        help="Imbalance-handling strategy: 'both' runs SMOTE and class-weighted "
                             "side-by-side and saves two plots (default); 'smote' or 'weighted' "
                             "runs only that one.")
    parser.add_argument("--save-from", choices=["smote", "weighted"], default="smote",
                        help="Which run's models to save when --balance=both (default: smote).")
    parser.add_argument("--compliant-only", action="store_true",
                        help="Restrict to SemVer-compliant crates: those with a stable "
                             "(non-pre-release) release >= 1.0.0. See scripts/semver_filter.py.")
    parser.add_argument("--compliant-strict", action="store_true",
                        help="With --compliant-only, use literal > 1.0.0 instead of >= 1.0.0.")
    parser.add_argument("--split", choices=["grouped", "stratified", "temporal"],
                        default="grouped",
                        help="Train/test split protocol (default: grouped by `package`, "
                             "which prevents within-crate leakage). 'stratified' reproduces "
                             "the LEAKY random split for sanity comparison; 'temporal' trains "
                             "on earlier releases and tests on later (requires a 'created_at' "
                             "column — see scripts/add_created_at.py).")
    args = parser.parse_args()

    if _UNAVAILABLE:
        print(f"Note: skipping unavailable classifier(s): {', '.join(sorted(_UNAVAILABLE))} "
              f"— install the native lib (on macOS: 'brew install libomp').")

    # Build the ordered list of strategies to run this invocation.
    strategies = ["smote", "weighted"] if args.balance == "both" else [args.balance]

    df_stats = None
    if args.synthetic:
        df = synthetic_dataset(args.seed)
    elif args.csv:
        df = pd.read_csv(args.csv, keep_default_na=False, na_values=[""])
        n_crates = df["package"].nunique() if "package" in df.columns else 0
        vc = df["release_type"].value_counts() if "release_type" in df.columns else {}
        df_stats = {
            "crates": n_crates,
            "pairs": len(df),
            "patch": int(vc.get("patch", 0)),
            "minor": int(vc.get("minor", 0)),
            "major": int(vc.get("major", 0)),
        }
    else:
        df = load_crate_csvs()
        if df.empty:
            fallback = os.path.join("data", "cratesbump.csv.gz")
            if os.path.isfile(fallback):
                print(f"No per-crate CSVs in crate_csvs/ — using released dataset {fallback}")
                df = pd.read_csv(fallback, keep_default_na=False, na_values=[""])
            else:
                print("No CSVs in crate_csvs/ and no released dataset at "
                      f"{fallback} — run 'make batch' or 'make extract C=<crate>' first.")
                sys.exit(1)
        n_crates = df["package"].nunique() if "package" in df.columns else 0
        vc = df["release_type"].value_counts() if "release_type" in df.columns else {}
        df_stats = {
            "crates": n_crates,
            "pairs": len(df),
            "patch": int(vc.get("patch", 0)),
            "minor": int(vc.get("minor", 0)),
            "major": int(vc.get("major", 0)),
        }

    # SemVer-compliance filter: keep only crates that reached a stable
    # (non-pre-release) release >= 1.0.0 (or > 1.0.0 with --compliant-strict),
    # so their major/minor/patch bumps carry real SemVer semantics.
    if args.compliant_only and not args.synthetic and "version" in df.columns:
        keep = compliant_packages(df, strict_gt=args.compliant_strict)
        before_c, before_p = df["package"].nunique(), len(df)
        df = df[df["package"].isin(keep)].reset_index(drop=True)
        bound = "> 1.0.0" if args.compliant_strict else ">= 1.0.0"
        print(f"SemVer-compliant filter (stable {bound}): "
              f"kept {df['package'].nunique():,}/{before_c:,} crates, "
              f"{len(df):,}/{before_p:,} pairs")
        vc = df["release_type"].value_counts()
        df_stats = {
            "crates": df["package"].nunique(), "pairs": len(df),
            "patch": int(vc.get("patch", 0)), "minor": int(vc.get("minor", 0)),
            "major": int(vc.get("major", 0)),
        }

    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        vc = df["release_type"].value_counts()
        df_stats = {
            "crates": df["package"].nunique() if "package" in df.columns else 0,
            "pairs": len(df),
            "patch": int(vc.get("patch", 0)), "minor": int(vc.get("minor", 0)),
            "major": int(vc.get("major", 0)),
        }
        print(f"Subsampled to {len(df):,} rows (--sample {args.sample}).")

    for _c in FEATURE_COLS:              # tolerate older CSVs missing new columns
        if _c not in df.columns:
            df[_c] = 0.0
    df = df.dropna(subset=FEATURE_COLS + ["release_type"])
    df = df[df["release_type"].isin(CLASSES)]
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = df[FEATURE_COLS].values.astype(float)
    y = df["release_type"].map({c: i for i, c in enumerate(CLASSES)}).values
    groups = df["package"].values if "package" in df.columns else None

    # Default split: GroupShuffleSplit by `package` so no crate appears in both
    # train and test. The prior stratified random split leaked within-crate
    # correlation and inflated every F1 by ~5 points — kept as an opt-in for
    # the "leaky baseline" sanity row.
    if args.split == "grouped":
        if groups is None:
            raise ValueError("--split grouped requires a 'package' column")
        s = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                              random_state=args.seed)
        tr, te = next(s.split(X, y, groups))
        X_train, X_test, y_train, y_test = X[tr], X[te], y[tr], y[te]
        overlap = len(set(groups[te]) & set(groups[tr])) / max(len(set(groups[te])), 1)
        print(f"Grouped split: {len(tr):,} train rows / {len(te):,} test rows, "
              f"test-crate overlap {overlap * 100:.2f}%")
    elif args.split == "temporal":
        if "created_at" not in df.columns:
            raise ValueError("--split temporal requires a 'created_at' column "
                             "(run scripts/add_created_at.py first)")
        # Rows without a parseable `created_at` cannot participate: dropping
        # them is the only honest choice (the alternative, np.argsort placing
        # NaT at the END, dumps all unknown-date rows into the test fold).
        dates = pd.to_datetime(df["created_at"], errors="coerce")
        valid = dates.notna().values
        n_dropped = int((~valid).sum())
        if n_dropped:
            print(f"Temporal split: dropping {n_dropped:,} rows with missing "
                  f"created_at ({100*n_dropped/len(df):.1f}% of input)")
        X_v, y_v = X[valid], y[valid]
        g_v     = groups[valid] if groups is not None else None
        d_v     = dates.values[valid]

        # Per-crate temporal cutoff: a crate goes to TEST only if ALL its
        # pairs are published AFTER the global cutoff T. This prevents the
        # "same crate, different year" leak that a row-level sort allows.
        if g_v is None:
            raise ValueError("--split temporal requires a 'package' column")
        # Compare in int64-nanoseconds to avoid tz/dtype mismatch between the
        # cutoff and the per-crate min/max datetime values.
        d_ns = d_v.astype("datetime64[ns]").astype("int64")
        cutoff_ns = int(np.quantile(d_ns, 1 - args.test_size))
        df_tmp = pd.DataFrame({"package": g_v, "d_ns": d_ns})
        crate_max = df_tmp.groupby("package")["d_ns"].max()
        crate_min = df_tmp.groupby("package")["d_ns"].min()
        test_crates  = set(crate_min[crate_min >  cutoff_ns].index)
        train_crates = set(crate_max[crate_max <= cutoff_ns].index)
        both         = set(crate_max.index) - test_crates - train_crates
        cutoff = pd.Timestamp(cutoff_ns, unit="ns", tz="UTC")
        # Crates that straddle the cutoff are DROPPED (we cannot assign them
        # cleanly without re-leaking crate identity across the boundary).
        tr = np.where(np.isin(g_v, list(train_crates)))[0]
        te = np.where(np.isin(g_v, list(test_crates)))[0]
        X_train, X_test = X_v[tr], X_v[te]
        y_train, y_test = y_v[tr], y_v[te]
        overlap = len(set(g_v[te]) & set(g_v[tr])) / max(len(set(g_v[te])), 1)
        print(f"Temporal split (per-crate, cutoff={cutoff}): "
              f"train {len(tr):,} / test {len(te):,} rows; "
              f"straddle-dropped {len(both):,} crates; "
              f"test-crate overlap {overlap*100:.2f}%")
    elif args.split == "stratified":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed
        )
        print("Stratified random split (LEAKY: within-crate release pairs appear "
              "in both train and test). Use --split grouped for the headline.")
    else:
        raise ValueError(f"Unknown --split: {args.split!r}")

    if args.best_params:
        best = load_best_params()
        missing = [name for name in CLASSIFIERS if name not in best]
        if missing:
            print(f"Warning: no tuned params found for: {', '.join(missing)} — using defaults.")
        for name, params in best.items():
            if name in CLASSIFIERS:
                CLASSIFIERS[name] = (lambda n, p: lambda: build_classifier(n, p, args.seed))(name, params)

    # readme uses "SMOTE"
    img_name_for = {"smote": "auc_table.png", "weighted": "auc_table_weighted.png"}
    label_for    = {"smote": "SMOTE", "weighted": "class-weighted sample_weight"}

    runs = {} 
    for strategy in strategies:
        print()
        print(f"━━━ Imbalance handling: {label_for[strategy]} ━━━")
        results = train_and_evaluate(X_train, X_test, y_train, y_test, args.seed,
                                     balance_strategy=strategy)
        print_results(results, y_test)

        if not args.no_plot:
            img_path = os.path.join(args.results_dir, img_name_for[strategy])
            save_results_image(results, df_stats, img_path, balance_strategy=strategy)
        runs[strategy] = results

    # If both strategies ran, print a delta table so the comparison is obvious.
    if len(runs) == 2:
        print()
        print("━━━ Δ (weighted − SMOTE) ━━━")
        W = 82
        print("─" * W)
        print(f"{'Classifier':<26} {'Δ F1':>9} {'Δ MCC':>7} {'Δ MajAUC':>9} {'Δ MinAUC':>9} {'Δ PatAUC':>9}")
        print("─" * W)
        smote_by_name    = {r["name"]: r for r in runs["smote"]}
        weighted_by_name = {r["name"]: r for r in runs["weighted"]}
        for name in [r["name"] for r in runs["smote"]]:
            s = smote_by_name[name]; w = weighted_by_name[name]
            def d(a, b):
                if np.isnan(a) or np.isnan(b): return "   n/a"
                v = b - a
                return f"{v:+.3f}"
            print(
                f"{name:<26}"
                f" {d(s['macro_f1'], w['macro_f1']):>9}"
                f" {d(s['mcc'],      w['mcc']):>7}"
                f" {d(s['aucs']['major'], w['aucs']['major']):>9}"
                f" {d(s['aucs']['minor'], w['aucs']['minor']):>9}"
                f" {d(s['aucs']['patch'], w['aucs']['patch']):>9}"
            )
        print("─" * W)

    # Pick which run's models to persist.  When only one strategy ran, that's
    # the one we save; otherwise honour --save-from.
    if args.save_models:
        save_choice = args.save_from if len(runs) == 2 else strategies[0]
        save_models(runs[save_choice], args.model_dir, FEATURE_COLS, CLASSES)
        print()
        print(f"  Models      -> {args.model_dir}/  (from {label_for[save_choice]} run)")


if __name__ == "__main__":
    main()
