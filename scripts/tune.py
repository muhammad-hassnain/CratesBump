#!/usr/bin/env python3
"""
Hyperparameter search across all seven classifiers.

Runs RandomizedSearchCV over each model's hyperparameter space, scored by
macro-F1 (primary; ROC-AUC and MCC also tracked as diagnostics,
one-vs-rest, stratified-grouped 3-fold CV), then evaluates
the winners on a held-out test set and produces:

  results/tuning_results.png   — grouped AUC bar chart + best-params table
  results/tuning_results.json  — machine-readable best params per model

  Run `make upgrade` afterwards to promote the best params into pipeline.py.

Usage:
    python3 tune.py                      # auto-loads all crate_csvs/*.csv
    python3 scripts/tune.py                  # auto-loads all crate_csvs/*.csv
    python3 tune.py --n-iter 50          # more random draws per model [default: 30]
    python3 tune.py --cv 5               # 5-fold CV instead of 3
    python3 tune.py --synthetic          # smoke-test (no CSV needed)
    python3 tune.py --seed 0

SMOTE is applied *inside each CV fold's training partition* (via an imblearn
Pipeline), so CV scores are not inflated by synthetic-sample leakage.
"""

import sys
import os
import json
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
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
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, cross_validate,
    GroupShuffleSplit, StratifiedGroupKFold,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Shared constants — imported from pipeline.py so the two scripts can never
# drift: tune.py searches over exactly the 41 trained FEATURE_COLS.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import FEATURE_COLS, CLASSES, DEFAULT_PARAMS  # noqa: E402
from evaluate import make_groups  # noqa: E402

# Hyperparameter search spaces 

PARAM_GRIDS = {
    "XGBoost": {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 4, 5, 6, 8],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight":[1, 3, 5],
        "gamma":           [0, 0.1, 0.3, 0.5],
    },
    "LightGBM": {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 4, 5, 6, 8, -1],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
        "num_leaves":      [15, 31, 63, 127],
        "min_child_samples":[5, 10, 20, 30],
    },
    "Hist Gradient Boosting": {
        "max_iter":           [100, 200, 300, 500],
        "max_depth":          [3, 4, 5, 6, None],
        "learning_rate":      [0.01, 0.05, 0.1, 0.2],
        "min_samples_leaf":   [10, 20, 30, 50],
        "l2_regularization":  [0.0, 0.1, 0.5, 1.0],
        "max_leaf_nodes":     [15, 31, 63, None],
    },
    "Random Forest": {
        "n_estimators":    [50, 100, 200, 300],
        "max_depth":       [None, 5, 10, 20, 30],
        "min_samples_split":[2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features":    ["sqrt", "log2", 0.3, 0.5],
    },
    "Gradient Boosting": {
        "n_estimators":  [50, 100, 200],
        "max_depth":     [2, 3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample":     [0.6, 0.8, 1.0],
        "min_samples_leaf": [1, 5, 10],
    },
    "Logistic Regression": {
        "C":        [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver":   ["lbfgs", "saga"],
        "max_iter": [200, 500, 1000],
    },
    "Decision Tree": {
        "max_depth":        [None, 3, 5, 8, 15, 20],
        "min_samples_split":[2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion":        ["gini", "entropy"],
        "max_features":     [None, "sqrt", "log2"],
    },
}

# Classifiers
#
# Every model is wrapped in an imblearn Pipeline whose first step is SMOTE.
# This is critical: when RandomizedSearchCV splits the data into CV folds,
# the SMOTE step only runs on each fold's TRAINING partition — never on the
# validation partition. At inference time (predict / predict_proba) the
# SMOTE step is automatically skipped by imblearn.
#
# Applying SMOTE up-front (before the CV split, as we used to do) was leaking
# information: synthetic minority samples generated from real training points
# would end up in validation folds whose "training" neighbours were used to
# create them — yielding wildly inflated CV scores (~0.95 AUC) that vanished
# on the real held-out test set (~0.81 AUC).

# Parallelism lives in ONE place: RandomizedSearchCV runs SEARCH_JOBS fits at
# once and every model is single-threaded (set OMP_NUM_THREADS=1 too, for
# Hist Gradient Boosting), so the two levels never multiply.
SEARCH_JOBS = 6
MODEL_JOBS = 1


def make_classifier(name: str, seed: int):
    """Return an imblearn Pipeline: SMOTE → (optional scaler) → estimator.

    All hyperparameter keys in the search grid will need a `clf__` prefix so
    they target the inner estimator inside this pipeline.
    """
    smote = SMOTE(random_state=seed)

    if name == "XGBoost":
        est = xgb.XGBClassifier(eval_metric="mlogloss", random_state=seed, n_jobs=MODEL_JOBS)
    elif name == "LightGBM":
        est = lgb.LGBMClassifier(random_state=seed, n_jobs=MODEL_JOBS, verbose=-1)
    elif name == "Hist Gradient Boosting":
        est = HistGradientBoostingClassifier(random_state=seed)
    elif name == "Random Forest":
        est = RandomForestClassifier(random_state=seed, n_jobs=MODEL_JOBS)
    elif name == "Gradient Boosting":
        est = GradientBoostingClassifier(random_state=seed)
    elif name == "Decision Tree":
        est = DecisionTreeClassifier(random_state=seed)
    elif name == "Logistic Regression":
        # LR gets an additional scaling step inside the same pipeline.
        return ImbPipeline([
            ("smote",  smote),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(random_state=seed)),
        ])
    else:
        raise ValueError(f"Unknown classifier: {name}")

    return ImbPipeline([("smote", smote), ("clf", est)])


def prefix_params(params: dict, name: str) -> dict:
    """Prefix every search-grid key with 'clf__' so it targets the inner
    estimator inside the SMOTE pipeline returned by make_classifier."""
    return {f"clf__{k}": v for k, v in params.items()}


def strip_prefix(params: dict) -> dict:
    """Strip 'clf__' prefix from best_params_ for display."""
    return {k.replace("clf__", ""): v for k, v in params.items()}

# ROC-AUC helper 

def per_class_auc(clf, X_test, y_test) -> dict:
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    proba = clf.predict_proba(X_test)
    aucs = {}
    for i, cls in enumerate(CLASSES):
        col = y_bin[:, i]
        if col.sum() == 0 or col.sum() == len(col):
            aucs[cls] = float("nan")
        else:
            aucs[cls] = roc_auc_score(col, proba[:, i])
    aucs["macro"] = float(np.nanmean(list(aucs.values())))
    return aucs


# Tuning

def tune_all(X_train, X_test, y_train, y_test, n_iter: int, cv: int, seed: int,
             groups_train=None, group_by="repo", extra_candidates=None) -> list:
    # SMOTE is now a *step* inside each model's pipeline (see make_classifier).
    # RandomizedSearchCV.fit therefore sees the raw X_train/y_train and applies
    # SMOTE separately within every CV fold's training partition. The held-out
    # validation partition never contains synthetic samples — eliminating the
    # data leakage that previously inflated CV AUC from ~0.81 to ~0.95.

    # Score with macro-F1 (primary), ROC-AUC, and MCC in one pass; refit on F1.
    # Macro F1 treats all three classes equally
    # MCC accounts for all cells of the confusion matrix simultaneously.
    # AUC is tracked as a diagnostic but not used for selection.
    scoring = {"f1": "f1_macro", "auc": "roc_auc_ovr", "mcc": "matthews_corrcoef"}

    # Inner CV: if we have `groups_train` (i.e. the outer split is grouped),
    # the inner folds must also respect crate boundaries — otherwise within-crate
    # pairs leak between CV train and CV val and inflate the search score.
    if groups_train is not None:
        cv_strategy = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
        print(f"  (inner CV: StratifiedGroupKFold by {group_by}, {cv} folds)")
    else:
        cv_strategy = cv
        print(f"  (inner CV: stratified random, {cv} folds — LEAKY)")

    results = []
    for name in PARAM_GRIDS:
        print(f"  [{name}]  searching {n_iter} configurations × {cv}-fold CV …")
        clf = make_classifier(name, seed)
        search = RandomizedSearchCV(
            clf,
            param_distributions=prefix_params(PARAM_GRIDS[name], name),
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=scoring,
            refit="f1",
            n_jobs=SEARCH_JOBS,
            random_state=seed,
            error_score=0.0,
        )
        if groups_train is not None:
            search.fit(X_train, y_train, groups=groups_train)
        else:
            search.fit(X_train, y_train)

        best_params  = strip_prefix(search.best_params_)

        # Candidates scored on the SAME inner folds: the search's best, the
        # current defaults (pipeline.DEFAULT_PARAMS), and any extra candidates
        # (e.g. earlier defaults). The one with the highest inner-CV macro-F1
        # is adopted. Test-set scores play no part in the choice.
        def score(params):
            c = make_classifier(name, seed).set_params(**prefix_params(params, name))
            r = cross_validate(c, X_train, y_train, cv=cv_strategy, scoring=scoring,
                               n_jobs=min(SEARCH_JOBS, cv), error_score=0.0,
                               **({"groups": groups_train} if groups_train is not None else {}))
            return float(np.mean(r["test_f1"])), float(np.mean(r["test_mcc"]))
        default_params = DEFAULT_PARAMS.get(name, {})
        default_cv_f1, default_cv_mcc = score(default_params)
        _cvr, _bi = search.cv_results_, search.best_index_
        candidates = {"search_best": {"params": best_params,
                                      "cv_f1": round(float(_cvr["mean_test_f1"][_bi]), 4),
                                      "cv_mcc": round(float(_cvr["mean_test_mcc"][_bi]), 4)},
                      "current_defaults": {"params": default_params,
                                           "cv_f1": round(default_cv_f1, 4),
                                           "cv_mcc": round(default_cv_mcc, 4)}}
        for label, params in (extra_candidates or {}).get(name, {}).items():
            f1_, mcc_ = score(params)
            candidates[label] = {"params": params, "cv_f1": round(f1_, 4), "cv_mcc": round(mcc_, 4)}
            print(f"    {label} CV F1: {f1_:.4f}")
        test_aucs    = per_class_auc(search.best_estimator_, X_test, y_test)
        test_mcc     = matthews_corrcoef(y_test, search.best_estimator_.predict(X_test))

        # Collect every configuration tried with all three scores
        cvr = search.cv_results_
        best_cv_f1  = float(cvr["mean_test_f1"] [search.best_index_])
        best_cv_auc = float(cvr["mean_test_auc"][search.best_index_])
        best_cv_mcc = float(cvr["mean_test_mcc"][search.best_index_])

        all_configs = sorted([
            {
                "params":   strip_prefix(cvr["params"][i]),
                "cv_f1":    round(float(cvr["mean_test_f1"] [i]), 4),
                "cv_auc":   round(float(cvr["mean_test_auc"][i]), 4),
                "cv_mcc":   round(float(cvr["mean_test_mcc"][i]), 4),
                "cv_f1_std":  round(float(cvr["std_test_f1"] [i]), 4),
                "cv_auc_std": round(float(cvr["std_test_auc"][i]), 4),
                "cv_mcc_std": round(float(cvr["std_test_mcc"][i]), 4),
            }
            for i in range(len(cvr["params"]))
        ], key=lambda x: x["cv_f1"], reverse=True)

        print(f"    CV F1: {best_cv_f1:.4f}  CV MCC: {best_cv_mcc:.4f}  CV AUC: {best_cv_auc:.4f}"
              f"  | Test AUC (macro): {test_aucs['macro']:.4f}  Test MCC: {test_mcc:.4f}")
        print(f"    Best params: {best_params}\n")

        winner = max(candidates, key=lambda k: candidates[k]["cv_f1"])
        adopt = winner == "search_best"
        print(f"    current defaults CV F1: {default_cv_f1:.4f} -> adopt {winner}")
        results.append({
            "name":           name,
            "default_params": default_params,
            "default_cv_f1":  round(default_cv_f1, 4),
            "default_cv_mcc": round(default_cv_mcc, 4),
            "candidates":     candidates,
            "adopted":        winner,
            "adopted_params": candidates[winner]["params"],
            "adopt":          bool(adopt),
            "cv_f1":          round(best_cv_f1,  4),
            "cv_auc":         round(best_cv_auc, 4),
            "cv_mcc":         round(best_cv_mcc, 4),
            "test_aucs":      {k: round(v, 4) for k, v in test_aucs.items()},
            "test_mcc":       round(float(test_mcc), 4),
            "best_params":    best_params,
            "all_configs":    all_configs,
        })

    return sorted(results, key=lambda r: r["cv_f1"], reverse=True)


# Visualisation helpers

def fmt_params(p: dict, max_len: int = 55) -> str:
    """Format a param dict as a compact multi-line string."""
    parts = [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items()]
    line, lines = "", []
    for part in parts:
        if line and len(line) + len(part) + 2 > max_len:
            lines.append(line.rstrip(", "))
            line = ""
        line += part + ", "
    if line:
        lines.append(line.rstrip(", "))
    return "\n".join(lines)


# Per-model line plot

def save_model_plot(r: dict, output_path: str):
    """
    Line plot — configurations on the x-axis (ranked best -> worst by F1),
    one line for Macro F1 (blue) and one for MCC (red).  Best point is
    highlighted with a gold star on both lines.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    configs = r["all_configs"]          # already sorted by cv_f1 desc
    n       = len(configs)
    f1s     = [c["cv_f1"] for c in configs]
    mccs    = [c["cv_mcc"] for c in configs]
    x       = np.arange(n)

    # Build compact param labels: one line per k=v pair
    def param_label(p: dict) -> str:
        parts = []
        for k, v in p.items():
            short_k = (k.replace("n_estimators", "n_est")
                        .replace("learning_rate", "lr")
                        .replace("max_depth", "depth")
                        .replace("min_samples_split", "min_split")
                        .replace("min_samples_leaf", "min_leaf")
                        .replace("min_child_weight", "min_cw")
                        .replace("min_child_samples", "min_cs")
                        .replace("colsample_bytree", "col_bt")
                        .replace("max_iter", "iters")
                        .replace("max_leaf_nodes", "leaves")
                        .replace("l2_regularization", "l2")
                        .replace("num_leaves", "n_leaves")
                        .replace("subsample", "sub"))
            short_v = f"{v:g}" if isinstance(v, float) else str(v)
            parts.append(f"{short_k}={short_v}")
        return "\n".join(parts)

    labels = [param_label(c["params"]) for c in configs]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4 + 2), 6))

    # Draw F1 and MCC lines 
    ax.plot(x, f1s,  color="#3498db", linewidth=2.2, marker="o", markersize=6,
            zorder=3, label="CV Macro F1")
    ax.plot(x, mccs, color="#e74c3c", linewidth=2.2, marker="s", markersize=6,
            zorder=3, label="CV MCC")

    # Value labels on every point
    for i, (f1, mcc) in enumerate(zip(f1s, mccs)):
        ax.text(i, f1  + 0.003, f"{f1:.4f}",  ha="center", va="bottom",
                fontsize=7.5, color="#2471a3", fontweight="bold")
        ax.text(i, mcc - 0.005, f"{mcc:.4f}", ha="center", va="top",
                fontsize=7.5, color="#c0392b")

    # Shaded band between the two lines (shows gap between F1 and MCC)
    ax.fill_between(x, f1s, mccs, alpha=0.08, color="#7f8c8d")

    all_vals = f1s + mccs
    lo = max(0.0, min(all_vals) - 0.06)
    hi = min(1.0, max(all_vals) + 0.06)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, ha="center", va="top", linespacing=1.4)
    ax.tick_params(axis="x", pad=6)
    ax.set_xlabel("Hyperparameter configuration  (best -> worst by F1)", fontsize=10,
                  labelpad=8)
    ax.set_ylabel("Score  (cross-validation)", fontsize=10)
    ax.set_title(
        f"{r['name']}  —  {n} hyperparameter configurations tried\n"
        f"shaded area = gap between F1 and MCC",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# Overview bar chart 

def save_overview(results: list, df_stats: dict | None, output_path: str):
    """Grouped bar chart: best-config CV F1, MCC, and ROC-AUC per model."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n     = len(results)
    names = [r["name"] for r in results]
    f1s   = [r["cv_f1"]  for r in results]
    mccs  = [r["cv_mcc"] for r in results]
    aucs  = [r["cv_auc"] for r in results]
    x, w  = np.arange(n), 0.25

    fig, ax = plt.subplots(figsize=(14, 5))

    bars_f1  = ax.bar(x - w,   f1s,  w, label="CV Macro F1  ★ (selection metric)", color="#3498db", alpha=0.88)
    bars_mcc = ax.bar(x,       mccs, w, label="CV MCC",                             color="#e74c3c", alpha=0.88)
    bars_auc = ax.bar(x + w,   aucs, w, label="CV ROC-AUC (diagnostic)",            color="#2ecc71", alpha=0.88)

    for bars, vals, col in [
        (bars_f1,  f1s,  "#2980b9"),
        (bars_mcc, mccs, "#c0392b"),
        (bars_auc, aucs, "#27ae60"),
    ]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                    color=col, fontweight="bold")

    all_vals = f1s + mccs + aucs
    lo = max(0.0, min(all_vals) - 0.05)
    ax.set_ylim(lo, min(1.02, max(all_vals) + 0.07))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9.5)
    ax.set_ylabel("Score", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_title("Best configuration per model — CV Macro F1, MCC, ROC-AUC  (sorted by F1 ↓)",
                 fontsize=12, fontweight="bold", pad=10)

    if df_stats:
        subtitle = (
            f"{df_stats['crates']:,} crates · {df_stats['pairs']:,} version pairs"
            f"  ({df_stats['patch']:,} patch / {df_stats['minor']:,} minor / {df_stats['major']:,} major)"
        )
        fig.text(0.5, -0.04, subtitle, ha="center", fontsize=8, color="#666666")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Overview chart   -> {output_path}")


# Standalone best-results table

def save_summary_table(results: list, output_path: str):
    """
    A single standalone table image — one row per model, showing best CV F1,
    CV ROC-AUC, test AUC, and winning hyperparameters.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n   = len(results)
    fig = plt.figure(figsize=(14, 1.0 + n * 0.72))
    ax  = fig.add_subplot(111)
    ax.axis("off")

    col_labels = ["Classifier", "CV F1 ★", "CV MCC", "CV AUC", "Test MCC", "Test AUC\n(macro)", "Best hyperparameters"]
    palette    = ["#f0f4f8", "#ffffff"]
    cell_text, cell_colors = [], []

    for i, r in enumerate(results):
        bg = "#eafaf1" if i == 0 else palette[i % 2]
        cell_text.append([
            r["name"],
            f"{r['cv_f1']:.4f}",
            f"{r['cv_mcc']:.4f}",
            f"{r['cv_auc']:.4f}",
            f"{r['test_mcc']:.4f}",
            f"{r['test_aucs']['macro']:.4f}",
            fmt_params(r["best_params"]),
        ])
        cell_colors.append([bg] * 7)

    tbl = ax.table(
        cellText=cell_text, cellColours=cell_colors,
        colLabels=col_labels, loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.8)

    for j in range(7):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight the CV F1 column header (selection metric)
    tbl[0, 1].set_facecolor("#1a5276")

    col_widths = [0.16, 0.08, 0.08, 0.08, 0.09, 0.09, 0.42]
    for (row, col), cell in tbl.get_celld().items():
        cell.set_width(col_widths[col])

    ax.set_title("Best hyperparameters per model  (sorted by CV Macro F1 ↓  ·  ★ = selection metric  ·  top row highlighted)",
                 fontsize=11, fontweight="bold", pad=12)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary table    -> {output_path}")


# from pipeline.py
def load_crate_csvs(crate_dir: str = "crate_csvs") -> pd.DataFrame:
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


def synthetic_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(30):
        for _ in range(40):
            rt    = rng.choice(CLASSES, p=[0.1, 0.3, 0.6])
            scale = {"major": 3.0, "minor": 1.5, "patch": 0.5}[rt]
            feat  = rng.exponential(scale, size=len(FEATURE_COLS))
            rows.append({"release_type": rt, **dict(zip(FEATURE_COLS, feat))})
    return pd.DataFrame(rows)


# main

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for semver classifiers")
    parser.add_argument("csv",         nargs="?", help="Explicit CSV path (default: merge crate_csvs/*.csv)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke-test)")
    parser.add_argument("--n-iter",    type=int,   default=30,  help="Random draws per model [30]")
    parser.add_argument("--cv",        type=int,   default=3,   help="CV folds [3]")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction [0.2]")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--results-dir", default="results", help="Output directory [results/]")
    parser.add_argument("--split", choices=["grouped", "stratified"], default="grouped",
                        help="Outer train/test split (default: grouped by `package`). "
                             "Inner CV follows the same scheme (StratifiedGroupKFold "
                             "when grouped, StratifiedKFold when stratified).")
    parser.add_argument("--group-by", choices=["repo", "crate"], default="repo",
                        help="Grouping unit for the outer split and the inner CV: the "
                             "source repository (`repository` column; crates without one "
                             "are their own group) or the crate.")
    parser.add_argument("--sample-rows", type=int, default=0,
                        help="Tune on a random sample of whole groups totalling about this "
                             "many rows (0 = all rows).")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Only tune these models (names as in PARAM_GRIDS).")
    parser.add_argument("--repos", default="results/crates_all_repos.csv",
                        help="package,repository map, used only if the CSV has no `repository` column")
    parser.add_argument("--candidates", default=None,
                        help="JSON {'candidates': {model: {label: params}}} scored on the same "
                             "inner folds as the search best and the current defaults")
    parser.add_argument("--search-jobs", type=int, default=6,
                        help="parallel fits in RandomizedSearchCV (models are single-threaded)")
    args = parser.parse_args()
    global SEARCH_JOBS
    SEARCH_JOBS = args.search_jobs
    if args.models:
        for k in [k for k in PARAM_GRIDS if k not in args.models]:
            del PARAM_GRIDS[k]

    # Load data
    df_stats = None
    if args.synthetic:
        print("Synthetic mode — generating random data\n")
        df = synthetic_dataset(args.seed)
    elif args.csv:
        print(f"Loading {args.csv} …")
        df = pd.read_csv(args.csv, keep_default_na=False, na_values=[""])
    else:
        print("Loading all crate_csvs/*.csv …")
        df = load_crate_csvs()
        if df.empty:
            print("  No CSVs found — run 'make batch' or 'make extract C=<crate>' first.")
            print("  To smoke-test: python3 tune.py --synthetic")
            sys.exit(1)

    if not args.synthetic:
        n_crates = df["package"].nunique() if "package" in df.columns else 0
        vc = df["release_type"].value_counts() if "release_type" in df.columns else {}
        df_stats = {
            "crates": n_crates,
            "pairs":  len(df),
            "patch":  int(vc.get("patch", 0)),
            "minor":  int(vc.get("minor", 0)),
            "major":  int(vc.get("major", 0)),
        }
        print(f"  {len(df):,} version pairs across {n_crates:,} crates\n")

    df = df.dropna(subset=FEATURE_COLS + ["release_type"])
    df = df[df["release_type"].isin(CLASSES)]
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = df[FEATURE_COLS].values.astype(float)
    y = df["release_type"].map({c: i for i, c in enumerate(CLASSES)}).values
    df = df.reset_index(drop=True)
    groups = make_groups(df, args.group_by, args.repos) if "package" in df.columns else None
    sample_info = {"sample_rows": None, "sample_groups": None}
    if args.sample_rows and groups is not None and len(df) > args.sample_rows:
        rng = np.random.default_rng(args.seed)
        sizes = pd.Series(groups).value_counts()
        order = rng.permutation(sizes.index.values)
        keep = set(order[:int(np.searchsorted(sizes.loc[order].cumsum().values, args.sample_rows)) + 1])
        m = np.array([g in keep for g in groups])
        X, y, groups = X[m], y[m], groups[m]
        sample_info = {"sample_rows": int(m.sum()), "sample_groups": len(keep),
                       "sample_crates": int(df.loc[m, "package"].nunique())}
        print(f"  sampled {m.sum():,} rows from {len(keep):,} whole groups (by {args.group_by})")

    # Grouped outer split: no crate appears in both train and test. Prevents
    # the within-crate leakage that previously inflated CV F1 by ~5 points.
    if args.split == "grouped" and groups is not None:
        s = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                              random_state=args.seed)
        tr, te = next(s.split(X, y, groups))
        X_train, X_test, y_train, y_test = X[tr], X[te], y[tr], y[te]
        groups_train = groups[tr]
        overlap = len(set(groups[te]) & set(groups[tr])) / max(len(set(groups[te])), 1)
        print(f"Grouped outer split: {len(tr):,} train / {len(te):,} test, "
              f"test-group overlap {overlap*100:.2f}%")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=args.seed,
        )
        groups_train = None
        print("Stratified random outer split (LEAKY).")

    for i, cls in enumerate(CLASSES):
        print(f"  {cls:6s}: {(y_test==i).sum():4d} test samples")

    # Run search
    print(f"\nSearching ({args.n_iter} iterations × {args.cv}-fold CV per model) …\n")
    extra = json.load(open(args.candidates))["candidates"] if args.candidates else None
    results = tune_all(X_train, X_test, y_train, y_test, args.n_iter, args.cv,
                       args.seed, groups_train=groups_train, group_by=args.group_by,
                       extra_candidates=extra)

    # Print summary 
    print("\n" + "═" * 88)
    print(f"{'Model':<26} {'CV F1':>7} {'CV MCC':>7} {'CV AUC':>7} {'Test MCC':>9} {'Test AUC':>9} {'major':>7} {'minor':>7} {'patch':>7}")
    print("─" * 88)
    for r in results:
        t = r["test_aucs"]
        def f(v): return f"{v:.4f}" if not np.isnan(v) else "  n/a"
        print(f"{r['name']:<26} {r['cv_f1']:>7.4f} {r['cv_mcc']:>7.4f} {r['cv_auc']:>7.4f} "
              f"{f(r['test_mcc']):>9} {f(t['macro']):>9} {f(t['major']):>7} {f(t['minor']):>7} {f(t['patch']):>7}")
    print("═" * 88)

    print("\nBest hyperparameters to use in pipeline.py:\n")
    for r in results:
        print(f"  {r['name']}")
        for k, v in r["best_params"].items():
            print(f"    {k} = {v}")
        print()

    # Save outputs
    os.makedirs(args.results_dir, exist_ok=True)
    overview_path = os.path.join(args.results_dir, "tuning_results.png")
    json_path     = os.path.join(args.results_dir, "tuning_results.json")

    print("\nSaving plots …")
    save_overview(results, df_stats, overview_path)
    save_summary_table(results, os.path.join(args.results_dir, "tuning_table.png"))

    for r in results:
        safe = r["name"].replace(" ", "_")
        path = os.path.join(args.results_dir, f"tuning_{safe}.png")
        save_model_plot(r, path)
        print(f"  {r['name']:<26} -> {path}")

    meta = {
        "searched_at": datetime.datetime.now().isoformat(),
        "csv": args.csv,
        "n_rows": int(len(df)),
        "n_crates": int(df["package"].nunique()) if "package" in df.columns else None,
        "split": args.split,
        "test_size": args.test_size,
        "n_iter": args.n_iter,
        "cv": args.cv,
        "cv_strategy": "StratifiedGroupKFold" if args.split == "grouped" else "StratifiedKFold",
        "group_by": args.group_by,
        "seed": args.seed,
        **sample_info,
        "train_rows": int(len(y_train)),
        "search_jobs": SEARCH_JOBS, "model_jobs": MODEL_JOBS,
        "selection": "per model, the candidate (search best, current pipeline.DEFAULT_PARAMS, "
                     "and any --candidates) with the highest inner-CV macro-F1 on the same "
                     "folds is adopted (fields `candidates`, `adopted`, `adopted_params`)",
        "candidates_file": args.candidates,
        "feature_cols": list(FEATURE_COLS),
        "n_features": len(FEATURE_COLS),
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Results JSON  -> {json_path}")


if __name__ == "__main__":
    main()
