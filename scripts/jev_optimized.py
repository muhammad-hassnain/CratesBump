#!/usr/bin/env python3
"""
Jev-optimized SemVer-bump pipeline: a few-shot LLM hybrid with the ML models.

A vanilla zero-shot setup asks the LLM one Choice question per held-out
release pair. This script keeps the LLM call in the loop but surrounds it with
four optimizations and reports a head-to-head against the ML baselines on the
SAME grouped-by-crate split (results: docs/METHODOLOGY_v2.md §1.1):

  (1) Rule gate            : deterministic SemVer heuristics short-circuit the
                             obvious cases (public API deletions -> major,
                             pure doc/comment churn -> patch, ...), so Jev is
                             only asked about the ambiguous middle.
  (2) Semantic narration   : raw numeric features are rendered as a bucketed
                             natural-language brief grouped by Abdalkareem's
                             six feature dimensions. Bucket cutoffs come from
                             per-feature training quantiles.
  (3) kNN few-shot         : per-query retrieval of class-balanced hard
                             examples from training (standardised L2), so the
                             prompt always carries decision-boundary cases.
  (4) Prior correction     : Jev probabilities are re-normalised with the
                             training class prior to counter the LLM's
                             uniform-ish zero-shot bias.

The final stack is a convex-combination of ML and (prior-corrected) Jev
posteriors; the mixing weight alpha is tuned on a held-out validation slice.

Usage:
    # dry run (no API key) with a fake Jev that follows the rule gate
    python3 scripts/jev_optimized.py data/compliant_dataset_v2_cargo.csv.gz \
        --split grouped --test-sample 2000 --tune-sample 800 --mock-jev

    # the published runs (read jev_cache/v2cargo.json; the API is only called,
    # and TYPESAFE_API_KEY only needed, on a cache miss)
    python3 scripts/jev_optimized.py data/compliant_dataset_v2_cargo.csv.gz \
        --split grouped --test-sample 2000 --tune-sample 800 --few-shot 6 \
        --enable-gate --enable-stack --cache jev_cache/v2cargo.json

Outputs:
    results/jev_optimized_report.json  machine-readable per-model metrics
    results/jev_optimized_table.png    visual summary table
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, matthews_corrcoef, accuracy_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P                          # noqa: E402
from evaluate import make_split               # noqa: E402

CLASSES = P.CLASSES                           # ["major", "minor", "patch"]
IDX = {c: i for i, c in enumerate(CLASSES)}

# -------------------------------------------------------------------- feature groups
# These groupings follow Abdalkareem et al.'s six dimensions and are what we
# narrate to Jev. Keeping them explicit also keeps the rule gate honest.
BREAKING_FEATS = ["dem", "mpd", "mpc", "drf"]        # deletions / arity changes
ADDITION_FEATS = ["adm", "mla", "gva", "arf"]        # new API surface
CHURN_FEATS    = ["mrf", "la", "ld", "clcrd",
                  "icc", "dcc", "mcc"]               # internal / docs
DEP_FEATS      = ["tcct", "pa", "pd", "pu"]
COMPLEX_FEATS  = ["acycd", "cycd"]
TIME_FEATS     = ["rdtd"]
ACTIVITY_FEATS = ["tcm", "tau", "nbf"]
ISSUE_FEATS    = ["poi", "pci", "popr", "pcpr"]
KEYWORD_FEATS  = ["kwm", "kwp", "kwb", "aml"]


# =============================================================================
# 1. Deterministic SemVer rule gate
# =============================================================================

@dataclass
class RuleVerdict:
    label: Optional[str]     # "major" / "minor" / "patch", or None if abstain
    reason: str              # one-line explanation (goes into Jev few-shot too)
    confidence: float        # 0..1 — how sure the rule is


def rule_gate(row: pd.Series) -> RuleVerdict:
    """Short-circuit only the cases where the SemVer verdict is near-tautological.

    The original rule set fired on ~29% of rows but hurt macro-F1 by -3.5
    because `dem` / `adm` count ALL functions (public and private), so a Rust
    refactor that moves a private helper triggered the "deleted methods"
    major rule incorrectly.

    Tightened rules (October 2026):
      * MAJOR  only when the deletion signal is large AND nothing was added
                anywhere (drf, arf, adm, gva all zero). This filters out
                private-helper refactors that just rename or inline things.
      * PATCH  only when NO API surface or dependency moved AND the total
                lines-of-code change is small (<=80), so we do not accidentally
                claim "patch" for a massive internal rewrite.
      * MINOR  dropped. "methods added without deletions" was the leakiest
                rule because adding a private helper is a patch, not a minor.
                Jev handles this class instead.

    Precision over recall. In practice the gate fires on ~1% of rows (not the
    ~10-15% initially targeted); on the earlier dataset versions the branch coverage
    is limited by `tcct` firing on 100% of rows (every release touches
    Cargo.toml), so the "no deps moved" patch rule collapses to a handful of
    cases.
    """
    g = lambda c: float(row.get(c, 0.0) or 0.0)   # noqa: E731

    # --- MAJOR: strong deletion signal with no additions anywhere ------------
    deletions = g("dem") + g("mpd") + g("drf")
    additions = g("adm") + g("arf") + g("gva")
    if deletions >= 3 and additions == 0:
        return RuleVerdict("major",
                           "strong deletion signal (dem+mpd+drf>=3) with no additions",
                           0.90)
    if g("mpd") >= 2 and g("adm") == 0:
        # removing parameters from multiple methods is near-certainly breaking
        return RuleVerdict("major",
                           "parameters removed from 2+ methods, no additions",
                           0.85)

    # --- PATCH: no API or dependency movement AND small LOC change -----------
    api_touched = any(g(c) for c in BREAKING_FEATS + ADDITION_FEATS)
    deps_touched = g("tcct") or g("pa") or g("pd") or g("pu")
    loc_small = (g("la") + g("ld")) <= 80
    if (not api_touched) and (not deps_touched) and loc_small:
        if g("icc") + g("dcc") + g("mcc") + g("la") + g("ld") > 0:
            return RuleVerdict("patch",
                               "docs/internal churn only, <=80 LOC changed",
                               0.85)

    # MINOR is intentionally not covered -- Jev + ML handle it.
    return RuleVerdict(None, "ambiguous", 0.0)


# =============================================================================
# 2. Semantic narration of raw features
# =============================================================================

@dataclass
class Narrator:
    """Turns a feature vector into a short natural-language brief that Jev can
    reason over, by bucketing each feature against training quantiles.

    Bucket labels: "none" (==0), "few" (<q66), "several" (<q90), "many" (>=q90).
    Only non-empty groups are included -- so the resulting prompt is compact
    even though 41 features exist."""
    cutoffs: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, df_train: pd.DataFrame) -> "Narrator":
        cuts: Dict[str, Tuple[float, float]] = {}
        for c in P.FEATURE_COLS:
            vals = df_train[c].astype(float).values
            vals = vals[vals > 0]              # bucketing only among non-zeros
            if len(vals) >= 20:
                cuts[c] = (float(np.quantile(vals, 0.66)),
                           float(np.quantile(vals, 0.90)))
            else:
                cuts[c] = (1.0, 5.0)           # safe defaults for zero-coverage cols
        return cls(cuts)

    def bucket(self, feat: str, v: float) -> str:
        if v <= 0:
            return "none"
        q66, q90 = self.cutoffs.get(feat, (1.0, 5.0))
        if v < q66:
            return "few"
        if v < q90:
            return "several"
        return "many"

    def _group(self, row: pd.Series, feats: List[str]) -> str:
        parts = []
        for c in feats:
            v = float(row.get(c, 0.0) or 0.0)
            b = self.bucket(c, v)
            if b != "none":
                parts.append(f"{c}={b}({v:g})")
        return ", ".join(parts) if parts else "none"

    def narrate(self, row: pd.Series) -> str:
        lines = [
            f"- Breaking signals (deletions, param changes): {self._group(row, BREAKING_FEATS)}",
            f"- Addition signals (new API surface):         {self._group(row, ADDITION_FEATS)}",
            f"- Internal churn (files/LOC/comments):        {self._group(row, CHURN_FEATS)}",
            f"- Dependencies:                               {self._group(row, DEP_FEATS)}",
            f"- Complexity change:                          {self._group(row, COMPLEX_FEATS)}",
            f"- Days since previous release:                {float(row.get('rdtd', 0) or 0):g}",
            f"- Commit activity:                            {self._group(row, ACTIVITY_FEATS)}",
            f"- Issue/PR activity:                          {self._group(row, ISSUE_FEATS)}",
            f"- Commit keywords:                            {self._group(row, KEYWORD_FEATS)}",
        ]
        return "\n".join(lines)


# =============================================================================
# 3. kNN few-shot retrieval
# =============================================================================

class FewShotIndex:
    """Class-balanced nearest-neighbour retriever over training rows.

    For each query we return `k_per_class` nearest training rows of EACH label
    (so Jev always sees at least one example of major/minor/patch). The
    neighbour notion is Euclidean in standardised feature space -- the same
    scaling the ML baselines effectively use."""

    def __init__(self, df_train: pd.DataFrame, k_per_class: int = 2):
        self.k = max(1, k_per_class)
        self.scaler = StandardScaler().fit(df_train[P.FEATURE_COLS].values)
        Xs = self.scaler.transform(df_train[P.FEATURE_COLS].values)
        self.by_class: Dict[str, Tuple[np.ndarray, pd.DataFrame]] = {}
        for cls in CLASSES:
            mask = (df_train["release_type"].values == cls)
            if mask.sum() == 0:
                continue
            self.by_class[cls] = (Xs[mask], df_train.loc[mask].reset_index(drop=True))

    def retrieve(self, row: pd.Series) -> List[Tuple[str, pd.Series]]:
        q = self.scaler.transform(
            np.asarray([float(row.get(c, 0.0) or 0.0) for c in P.FEATURE_COLS])
            .reshape(1, -1)
        )[0]
        picks: List[Tuple[str, pd.Series]] = []
        for cls, (Xs, df) in self.by_class.items():
            d = np.linalg.norm(Xs - q, axis=1)
            idx = np.argpartition(d, min(self.k, len(d) - 1))[: self.k]
            idx = idx[np.argsort(d[idx])]      # closest first within class
            for i in idx:
                picks.append((cls, df.iloc[i]))
        return picks


# =============================================================================
# 4. Jev client wrapper (real + mock)
# =============================================================================

INSTRUCTIONS_HEAD = (
    "You classify the Semantic Versioning bump type for a Rust crate release "
    "based on a short brief describing what changed since the previous release.\n\n"
    "SemVer semantics for Rust/Cargo:\n"
    "  * major -> any BREAKING change to the public API: a removed/renamed "
    "public item, a changed function signature, a changed trait, etc. "
    "For 0.y.z versions the leftmost non-zero field is the breaking boundary.\n"
    "  * minor -> any backward-compatible ADDITION to the public API: new "
    "functions, new modules, new trait impls, new config fields.\n"
    "  * patch -> backward-compatible bug fixes or purely internal changes "
    "(docs, tests, refactors, dependency bumps that do not expose new API).\n\n"
    "Decision procedure (apply in order):\n"
    "  1. If the brief mentions deleted public methods, deleted parameters, "
    "or deleted .rs files, lean strongly toward major.\n"
    "  2. Otherwise, if it mentions added methods, added files, or new "
    "globals, lean toward minor.\n"
    "  3. Otherwise classify as patch.\n"
    "Weigh commit keywords and issue activity as supporting evidence only; "
    "the AST change signals dominate."
)

CRITERIA = {
    "major": "breaking change to the public API (deletions, renames, signature changes)",
    "minor": "backward-compatible feature addition (new public API)",
    "patch": "backward-compatible internal change (bug fix, docs, refactor)",
}


class JevClient:
    """Thin wrapper over `langchain-typesafe` with on-disk caching and retries.

    Pass `mock=True` to run the pipeline without an API key: the mock answers
    use the rule gate when it fires and otherwise a sharp uniform-ish prior,
    so the end-to-end plumbing can be smoke-tested cheaply."""

    def __init__(self, criteria=CRITERIA, instructions=INSTRUCTIONS_HEAD,
                 cache_path="jev_cache/optimized.json", mock=False):
        self.cache_path = cache_path
        self.mock = mock
        self.criteria = criteria
        self.instructions = instructions
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.cache: Dict[str, Dict] = (
            json.load(open(cache_path)) if os.path.isfile(cache_path) else {}
        )
        self._dirty = 0
        self.classifier = None

    def _ensure_client(self):
        """Create the API client on the first cache miss, so a run that is
        fully served from the cache needs neither the key nor the package."""
        if self.classifier is None:
            self._load_key()
            from langchain_typesafe import Choice, TypeSafeClassifier
            self.classifier = TypeSafeClassifier()
            self._Choice = Choice

    @staticmethod
    def _load_key():
        """Make sure TYPESAFE_API_KEY is in env, reading from .typesafe_token
        (project or home) if the env var is not set."""
        if os.environ.get("TYPESAFE_API_KEY"):
            return
        for p in (".typesafe_token", os.path.expanduser("~/.typesafe_token")):
            try:
                t = open(p).read().strip()
                if t:
                    os.environ["TYPESAFE_API_KEY"] = t
                    return
            except FileNotFoundError:
                continue
        sys.exit("ERROR: no TYPESAFE_API_KEY (set the env var or write the "
                 "key into .typesafe_token in the project root).")

    def _question(self, instructions_full: str):
        return {"bump": self._Choice(criteria=self.criteria,
                                     instructions=instructions_full)}

    def _cache_key(self, instructions_full: str, state: str) -> str:
        h = hashlib.sha1()
        h.update(instructions_full.encode())
        h.update(b"||")
        h.update(state.encode())
        return h.hexdigest()

    def _mock_answer(self, state: str) -> Dict:
        """Deterministic stand-in used when --mock-jev is passed. It inspects
        the narrated state for breaking/addition keywords and returns a sharp
        prior; good enough to exercise the ensemble wiring, bad enough that
        the real API still clearly beats it."""
        s = state.lower()
        if "breaking signals" in s and "none" not in s.split("breaking signals")[1].split("\n")[0]:
            probs = {"major": 0.75, "minor": 0.15, "patch": 0.10}
        elif "addition signals" in s and "none" not in s.split("addition signals")[1].split("\n")[0]:
            probs = {"major": 0.10, "minor": 0.70, "patch": 0.20}
        else:
            probs = {"major": 0.05, "minor": 0.20, "patch": 0.75}
        label = max(probs, key=probs.get)
        return {"pred": label, "probs": probs}

    def predict(self, instructions_full: str, state: str) -> Dict:
        key = self._cache_key(instructions_full, state)
        if key in self.cache:
            return self.cache[key]

        if self.mock:
            out = self._mock_answer(state)
            self.cache[key] = out
            self._dirty += 1
            if self._dirty % 100 == 0:
                self.flush()
            return out

        self._ensure_client()
        for attempt in range(5):
            try:
                resp = self.classifier.invoke(
                    {"state": state, "questions": self._question(instructions_full)}
                )
                ans = resp.answers["bump"]
                label = getattr(ans, "choice", None) or getattr(ans, "value", None)
                raw = getattr(ans, "probabilities", None) or {}
                probs = {c: float(raw.get(c, 0.0)) for c in CLASSES}
                s = sum(probs.values()) or 1.0
                probs = {k: v / s for k, v in probs.items()}
                out = {"pred": label if label in CLASSES else max(probs, key=probs.get),
                       "probs": probs}
                self.cache[key] = out
                self._dirty += 1
                if self._dirty % 25 == 0:
                    self.flush()
                return out
            except Exception as e:  # noqa: BLE001
                print(f"  jev error ({e}); retry {attempt+1}/5", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        return {"pred": None, "probs": {c: 1.0 / len(CLASSES) for c in CLASSES}}

    def flush(self):
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)


# =============================================================================
# 5. Prior correction and soft voting
# =============================================================================

def correct_prior(probs: Dict[str, float],
                  llm_prior: Dict[str, float],
                  true_prior: Dict[str, float],
                  eps: float = 1e-6) -> Dict[str, float]:
    """Re-weight LLM probabilities by (train_prior / llm_prior).

    This counters the uniform-ish bias zero-shot LLMs tend to carry. The LLM's
    prior is estimated from its own predictions on the tune split."""
    adj = {c: probs[c] * (true_prior[c] + eps) / (llm_prior[c] + eps)
           for c in CLASSES}
    s = sum(adj.values()) or 1.0
    return {c: v / s for c, v in adj.items()}


def soft_vote(ml: np.ndarray, jev: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * ml + (1.0 - alpha) * jev


def tune_alpha(ml_probs: np.ndarray, jev_probs: np.ndarray,
               y_true: np.ndarray, grid=None) -> Tuple[float, float]:
    """Grid-search alpha in [0,1] to maximise macro-F1 on a held-out slice."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    best_a, best_f1 = 1.0, -1.0
    for a in grid:
        p = soft_vote(ml_probs, jev_probs, a)
        yp = p.argmax(axis=1)
        f1 = f1_score(y_true, yp, average="macro", zero_division=0)
        if f1 > best_f1:
            best_a, best_f1 = float(a), float(f1)
    return best_a, best_f1


def soft_vote3(ml: np.ndarray, jev: np.ndarray, rule: np.ndarray,
               alpha: float, beta: float) -> np.ndarray:
    """Convex combination of three posteriors: alpha*ML + beta*Rule + gamma*Jev,
    where gamma = 1 - alpha - beta. Caller enforces alpha, beta >= 0, alpha+beta <= 1."""
    gamma = 1.0 - alpha - beta
    return alpha * ml + beta * rule + gamma * jev


def tune_alpha_beta(ml_probs: np.ndarray, jev_probs: np.ndarray,
                    rule_probs: np.ndarray, y_true: np.ndarray,
                    step: float = 0.05) -> Tuple[float, float, float]:
    """Joint grid-search over (alpha, beta) with alpha+beta<=1.
    Returns (best_alpha, best_beta, best_macro_f1)."""
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    best = (1.0, 0.0, -1.0)
    for a in grid:
        for b in grid:
            if a + b > 1.0 + 1e-9:
                continue
            p = soft_vote3(ml_probs, jev_probs, rule_probs, a, b)
            yp = p.argmax(axis=1)
            f1 = f1_score(y_true, yp, average="macro", zero_division=0)
            if f1 > best[2]:
                best = (float(a), float(b), float(f1))
    return best


# ---------- Confidence-gated routing ----------------------------------------

def tune_confidence_gate(ml_probs: np.ndarray, jev_probs: np.ndarray,
                         rule_probs: np.ndarray, alpha: float, beta: float,
                         y_true: np.ndarray) -> Tuple[float, float]:
    """Pick an ML max-prob threshold t so that rows with max(P_ML) > t use ML
    directly and the rest use the convex combo. Grid-searched on val for F1."""
    ml_max = ml_probs.max(axis=1)
    combo = soft_vote3(ml_probs, jev_probs, rule_probs, alpha, beta)
    best_t, best_f1 = 1.0, -1.0  # t=1.0 means "never gate", i.e. always use combo
    for t in np.linspace(0.5, 1.0, 26):
        gated = combo.copy()
        mask = ml_max > t
        gated[mask] = ml_probs[mask]
        f1 = f1_score(y_true, gated.argmax(axis=1), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def apply_confidence_gate(ml_probs: np.ndarray, combo_probs: np.ndarray,
                          threshold: float) -> Tuple[np.ndarray, float]:
    """Where ML.max > threshold, use ML; elsewhere use the combo.
    Returns (final_probs, gating_rate_in_[0,1]).
    threshold >= 1.0 disables gating (returns combo unchanged)."""
    if threshold >= 1.0:
        return combo_probs, 0.0
    ml_max = ml_probs.max(axis=1)
    mask = ml_max > threshold
    out = combo_probs.copy()
    out[mask] = ml_probs[mask]
    return out, float(mask.mean())


# ---------- Stacking meta-learner --------------------------------------------

def _stack_features(ml_probs: np.ndarray, jev_probs: np.ndarray,
                    rule_probs: np.ndarray, rule_fired: np.ndarray) -> np.ndarray:
    """Build an 11-D feature matrix for the LogReg combiner:
      [P_ML(3), P_Jev(3), max(P_ML), margin(P_ML), max(P_Jev), rule_fired, max(P_Rule)]"""
    def margin(p):
        s = np.sort(p, axis=1)
        return s[:, -1] - s[:, -2]
    feats = np.hstack([
        ml_probs,
        jev_probs,
        ml_probs.max(axis=1, keepdims=True),
        margin(ml_probs).reshape(-1, 1),
        jev_probs.max(axis=1, keepdims=True),
        rule_fired.reshape(-1, 1).astype(float),
        rule_probs.max(axis=1, keepdims=True),
    ])
    return feats


def fit_stacker(ml_val: np.ndarray, jev_val: np.ndarray, rule_val: np.ndarray,
                rule_val_fired: np.ndarray, y_val: np.ndarray):
    """Train a tiny LogReg combiner on the val slice. Returns the fitted model."""
    from sklearn.linear_model import LogisticRegression
    X = _stack_features(ml_val, jev_val, rule_val, rule_val_fired)
    clf = LogisticRegression(max_iter=1000, C=1.0,
                             solver="lbfgs", class_weight="balanced")
    clf.fit(X, y_val)
    return clf


def predict_stacker(stacker, ml_probs: np.ndarray, jev_probs: np.ndarray,
                    rule_probs: np.ndarray, rule_fired: np.ndarray) -> np.ndarray:
    X = _stack_features(ml_probs, jev_probs, rule_probs, rule_fired)
    return stacker.predict_proba(X)


# =============================================================================
# 6. Few-shot block + end-to-end Jev prediction for one row
# =============================================================================

def few_shot_block(examples: List[Tuple[str, pd.Series]], narrator: Narrator) -> str:
    """Build a few-shot block. Deliberately does NOT include rule-gate verdicts
    so Jev prompts are independent of the rule gate -- that keeps the on-disk
    cache valid across rule-gate iterations and keeps the two optimizations
    cleanly separable in the ablation."""
    if not examples:
        return ""
    lines = ["\n\nWorked examples (nearest neighbours from training):"]
    for cls, ex in examples:
        lines.append(f"\nBrief:\n{narrator.narrate(ex)}\nAnswer: {cls}")
    return "\n".join(lines)


def jev_predict_row(row: pd.Series, narrator: Narrator, index: FewShotIndex,
                    jev: JevClient, few_shot_k: int) -> Dict:
    state = narrator.narrate(row)
    instr = INSTRUCTIONS_HEAD
    if few_shot_k > 0:
        shots = index.retrieve(row)
        instr = INSTRUCTIONS_HEAD + few_shot_block(shots, narrator)
    return jev.predict(instr, state)


def jev_predict_many(df: pd.DataFrame, narrator: Narrator, index: FewShotIndex,
                     jev: JevClient, few_shot_k: int, label="jev") -> np.ndarray:
    out = np.zeros((len(df), len(CLASSES)), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        r = jev_predict_row(row, narrator, index, jev, few_shot_k)
        for j, c in enumerate(CLASSES):
            out[i, j] = r["probs"].get(c, 0.0)
        if (i + 1) % 100 == 0:
            print(f"  [{label}] {i+1}/{len(df)}", file=sys.stderr)
    jev.flush()
    return out


# =============================================================================
# 7. Clean + load
# =============================================================================

def clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in P.FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df = df.dropna(subset=P.FEATURE_COLS + ["release_type"])
    df = df[df["release_type"].isin(CLASSES)].reset_index(drop=True)
    df[P.FEATURE_COLS] = df[P.FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified subsample preserving class proportions.

    Written as an explicit loop rather than `groupby.apply` because pandas 3.0
    excludes the grouping column from the applied result by default, which
    would strip `release_type` from the output."""
    if not n or len(df) <= n:
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = []
    for _, g in df.groupby("release_type"):
        k = min(len(g), max(1, int(round(len(g) * frac))))
        parts.append(g.sample(k, random_state=seed))
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# 8. Reporting
# =============================================================================

def metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Headline + per-class metrics (macro-F1, MCC, accuracy, and per-class
    precision/recall/F1)."""
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASSES), average=None, zero_division=0)
    for i, c in enumerate(CLASSES):
        out[f"precision_{c}"] = float(prec[i])
        out[f"recall_{c}"]    = float(rec[i])
        out[f"f1_{c}"]        = float(f1[i])
    return out


def paired_bootstrap_ci(y_true: np.ndarray, y_pred_a: np.ndarray,
                        y_pred_b: np.ndarray, n_boot: int = 2000,
                        seed: int = 42) -> Dict:
    """Paired bootstrap over the test rows for ΔF1 = F1(b) − F1(a), so a
    single-seed difference comes with a confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fa = f1_score(y_true[idx], y_pred_a[idx], average="macro", zero_division=0)
        fb = f1_score(y_true[idx], y_pred_b[idx], average="macro", zero_division=0)
        deltas[i] = fb - fa
    lo, hi = float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
    p_null = float((deltas <= 0).mean())        # one-sided p against H0: Δ ≤ 0
    return {"delta_mean": float(deltas.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "p_value_one_sided": p_null,
            "n_boot": int(n_boot)}


def row_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                     n_boot: int = 2000, seed: int = 42) -> Dict:
    """Per-row bootstrap: resample the test rows n_boot times, compute macro-F1
    on each resample, return mean/95% CI/std. Lets every row of the output
    table carry its own sampling-variance CI instead of a single-point number."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1s[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    return {"f1_mean": float(f1s.mean()),
            "f1_std":  float(f1s.std()),
            "f1_ci_lo": float(np.quantile(f1s, 0.025)),
            "f1_ci_hi": float(np.quantile(f1s, 0.975))}


def save_report_table(rows: List[Dict], output_path: str, header: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 1.0 + 0.55 * len(rows)))
    ax.axis("off")
    col_labels = ["Macro F1", "MCC", "Accuracy"]
    cell_text, cell_colors = [], []
    for r in rows:
        cell_text.append([f"{r['macro_f1']:.3f}", f"{r['mcc']:.3f}", f"{r['accuracy']:.3f}"])
        shade = min(1.0, max(0.0, r["macro_f1"]))
        cell_colors.append([
            (1 - 0.45 * shade, 0.78 + 0.18 * shade, 1 - 0.45 * shade),
            (0.78 - 0.28 * r["mcc"] if r["mcc"] > 0 else 0.95,
             0.88 - 0.18 * r["mcc"] if r["mcc"] > 0 else 0.95,
             0.98),
            (0.93, 0.93, 0.93),
        ])
    tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
                   rowLabels=[r["name"] for r in rows],
                   colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.15, 1.9)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        tbl[i + 1, -1].set_facecolor("#dce3ea")
        tbl[i + 1, -1].set_text_props(fontweight="bold")
    fig.text(0.5, 0.97, header, ha="center", va="top", fontsize=12, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  table -> {output_path}")


# =============================================================================
# 9. Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--split", default="grouped", choices=["stratified", "grouped", "temporal"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--test-sample", type=int, default=2000,
                    help="cap the number of test rows evaluated with Jev (cost control)")
    ap.add_argument("--tune-sample", type=int, default=800,
                    help="held-out slice from train used to tune alpha and the LLM prior")
    ap.add_argument("--few-shot", type=int, default=2,
                    help="k nearest-neighbour training examples PER CLASS in the prompt")
    ap.add_argument("--ml-model", default="XGBoost",
                    help="which trained model to ensemble with (name from pipeline.CLASSIFIERS)")
    ap.add_argument("--cache", default="jev_cache/optimized.json")
    ap.add_argument("--mock-jev", action="store_true",
                    help="run without an API key using a deterministic mock answerer")
    ap.add_argument("--report-json", default="results/jev_optimized_report.json")
    ap.add_argument("--report-image", default="results/jev_optimized_table.png")
    ap.add_argument("--enable-gate", action="store_true",
                    help="also report a confidence-gated row: skip Jev where ML max-prob > tuned threshold")
    ap.add_argument("--enable-stack", action="store_true",
                    help="also report a stacking meta-learner row (LogReg over ML/Jev/Rule posteriors)")
    ap.add_argument("--llm-provider", default="typesafe",
                    choices=["typesafe", "openai", "anthropic", "ollama", "mock"],
                    help="LLM provider for the Jev-slot of the pipeline. "
                         "Non-default values use scripts/llm_clients.py and ask "
                         "for a strict-JSON response.")
    ap.add_argument("--llm-model", default="default",
                    help="Model identifier for the chosen provider. For "
                         "typesafe: ignored. OpenAI: 'gpt-4o-mini'. "
                         "Anthropic: 'claude-haiku-5-5'. Ollama: 'llama3.1:8b', "
                         "'qwen2.5:7b', 'phi3.5:3.8b', etc.")
    ap.add_argument("--no-prior-correction", action="store_true",
                    help="Skip the (train_prior/llm_prior) rescaling of LLM "
                         "posteriors. Needed for a fair LLM-to-LLM comparison: "
                         "the correction can invert a sharp LLM's predictions "
                         "when its class prior is far from the training prior.")
    args = ap.parse_args()

    # ---- Load + split ----
    df = clean(pd.read_csv(args.csv, keep_default_na=False, na_values=[""]))
    X = df[P.FEATURE_COLS].values.astype(float)
    y = df["release_type"].map(IDX).values
    groups = df["package"].values
    order = df["created_at"].values if "created_at" in df.columns else None
    tr_idx, te_idx = make_split(args.split, X, y, groups, order, args.seed, args.test_size)

    df_train_full = df.iloc[tr_idx].reset_index(drop=True)
    df_test_full = df.iloc[te_idx].reset_index(drop=True)

    # carve a validation slice from training for alpha + prior calibration
    rng = np.random.default_rng(args.seed)
    val_idx = rng.choice(len(df_train_full),
                         size=min(args.tune_sample, len(df_train_full)),
                         replace=False)
    val_mask = np.zeros(len(df_train_full), dtype=bool)
    val_mask[val_idx] = True
    df_val = df_train_full[val_mask].reset_index(drop=True)
    df_train = df_train_full[~val_mask].reset_index(drop=True)

    df_test = balanced_sample(df_test_full, args.test_sample, args.seed)

    print(f"split={args.split}  "
          f"train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    # ---- ML baseline ----
    if args.ml_model not in P.CLASSIFIERS:
        sys.exit(f"ERROR: --ml-model {args.ml_model!r} not in "
                 f"{sorted(P.CLASSIFIERS)}")
    print(f"\n[1/4] Training ML baseline: {args.ml_model}")
    Xtr = df_train[P.FEATURE_COLS].values.astype(float)
    ytr = df_train["release_type"].map(IDX).values
    Xtr_s, ytr_s = P.apply_smote(Xtr, ytr, args.seed)
    ml = P.CLASSIFIERS[args.ml_model]()
    ml.fit(Xtr_s, ytr_s)

    def ml_probs(df_: pd.DataFrame) -> np.ndarray:
        Xq = df_[P.FEATURE_COLS].values.astype(float)
        if hasattr(ml, "predict_proba"):
            return ml.predict_proba(Xq)
        s = ml.decision_function(Xq)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    ml_val = ml_probs(df_val)
    ml_test = ml_probs(df_test)

    # ---- Rule gate (as a probability distribution per row) ----
    print("[2/4] Running rule gate on val+test")
    def gate_probs(df_: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rule_probs, fired_mask). On rows where the gate abstains,
        rule_probs is uniform (1/|C|) so including it in the convex combo
        imparts no bias; on rows where it fires, rule_probs is sharp."""
        probs = np.full((len(df_), len(CLASSES)), 1.0 / len(CLASSES))
        fired = np.zeros(len(df_), dtype=bool)
        for i, (_, row) in enumerate(df_.iterrows()):
            v = rule_gate(row)
            if v.label is None:
                continue
            fired[i] = True
            p = np.full(len(CLASSES), (1 - v.confidence) / (len(CLASSES) - 1))
            p[IDX[v.label]] = v.confidence
            probs[i] = p
        return probs, fired

    gate_val, gate_val_fire = gate_probs(df_val)
    gate_test, gate_test_fire = gate_probs(df_test)
    print(f"  gate fires on {gate_val_fire.mean()*100:.1f}% of val, "
          f"{gate_test_fire.mean()*100:.1f}% of test")

    # ---- Jev / Alt-LLM ----
    print(f"[3/4] {args.llm_provider}/{args.llm_model} predictions "
          f"(narration + kNN few-shot)")
    narrator = Narrator.fit(df_train)
    index = FewShotIndex(df_train, k_per_class=max(1, args.few_shot // len(CLASSES) or 1))

    # Provider plumbing: default is still the TypeSafe Jev client, but
    # --llm-provider lets us swap in OpenAI, Anthropic, or Ollama for the
    # alternative-LLM sweeps.
    if args.llm_provider == "typesafe" and not args.mock_jev:
        jev = JevClient(cache_path=args.cache, mock=False)
    elif args.mock_jev:
        jev = JevClient(cache_path=args.cache, mock=True)
    else:
        from llm_clients import make_client
        _llm = make_client(args.llm_provider, args.llm_model, args.cache,
                           **({"criteria": CRITERIA,
                               "default_instructions": INSTRUCTIONS_HEAD}
                              if args.llm_provider == "typesafe" else {}))
        # Adapter: give the alt-LLM client the same .predict(instructions,state)
        # interface as JevClient so the rest of the pipeline is unchanged.
        class _Adapter:
            def __init__(self, cli):
                self.cli = cli
            def predict(self, instructions, state):
                return self.cli.predict(instructions, state)
            def flush(self):
                self.cli.flush()
        jev = _Adapter(_llm)

    jev_val_raw = jev_predict_many(df_val, narrator, index, jev,
                                   few_shot_k=args.few_shot, label="val")
    # LLM prior from its own predictions on val
    llm_prior = {c: float(jev_val_raw[:, i].mean()) for i, c in enumerate(CLASSES)}
    true_prior = {c: float((df_train["release_type"] == c).mean()) for c in CLASSES}
    print(f"  llm prior: {llm_prior}")
    print(f"  true prior: {true_prior}")

    def apply_prior(mat: np.ndarray) -> np.ndarray:
        """Rescale per-row probabilities by (train_prior / llm_prior). Caveat:
        this operation can INVERT a
        confidently-sharp LLM's predictions when the LLM's own class prior is
        very different from the training prior (e.g. llama3.1 predicted major
        on ~88% of val, so correction pulls its major predictions down to the
        patch class). Use --no-prior-correction to bypass.
        """
        if args.no_prior_correction:
            return mat
        out = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            d = {c: float(mat[i, j]) for j, c in enumerate(CLASSES)}
            adj = correct_prior(d, llm_prior, true_prior)
            for j, c in enumerate(CLASSES):
                out[i, j] = adj[c]
        return out

    jev_val = apply_prior(jev_val_raw)
    jev_test = apply_prior(
        jev_predict_many(df_test, narrator, index, jev,
                         few_shot_k=args.few_shot, label="test")
    )

    # ---- Tune mixing weights on val ----
    print("[4/4] Tuning convex-combination weights on val")
    y_val = df_val["release_type"].map(IDX).values
    alpha_mj, val_f1_mj = tune_alpha(ml_val, jev_val, y_val)
    a3, b3, val_f1_3 = tune_alpha_beta(ml_val, jev_val, gate_val, y_val)
    print(f"  ML+Jev           : alpha={alpha_mj:.2f}  val-F1={val_f1_mj:.3f}")
    print(f"  ML+Jev+Rule      : alpha={a3:.2f}  beta={b3:.2f}  "
          f"gamma={1-a3-b3:.2f}  val-F1={val_f1_3:.3f}")

    # ---- Evaluate everything on test ----
    y_true = df_test["release_type"].tolist()

    def preds_from(probs: np.ndarray) -> List[str]:
        return [CLASSES[i] for i in probs.argmax(axis=1)]

    rows = []
    rows.append({"name": f"ML only ({args.ml_model})",
                 **metrics(y_true, preds_from(ml_test))})

    rows.append({"name": "Jev only (optimized)",
                 **metrics(y_true, preds_from(jev_test))})

    rows.append({"name": f"ML + Jev soft vote (alpha={alpha_mj:.2f})",
                 **metrics(y_true, preds_from(soft_vote(ml_test, jev_test, alpha_mj)))})

    rows.append({"name": f"ML + Jev + Rule (a={a3:.2f}, b={b3:.2f})",
                 **metrics(y_true, preds_from(
                     soft_vote3(ml_test, jev_test, gate_test, a3, b3)))})

    gate_threshold = 1.0
    gate_rate = 0.0
    if args.enable_gate:
        gate_threshold, gate_val_f1 = tune_confidence_gate(
            ml_val, jev_val, gate_val, a3, b3,
            df_val["release_type"].map(IDX).values)
        combo_test = soft_vote3(ml_test, jev_test, gate_test, a3, b3)
        gated_test, gate_rate = apply_confidence_gate(ml_test, combo_test, gate_threshold)
        rows.append({"name": f"+ Confidence gate (t={gate_threshold:.2f}, skipped {gate_rate*100:.0f}%)",
                     **metrics(y_true, preds_from(gated_test))})

    stacker_val_f1 = None
    if args.enable_stack:
        y_val_idx = df_val["release_type"].map(IDX).values
        stacker = fit_stacker(ml_val, jev_val, gate_val, gate_val_fire, y_val_idx)
        stack_val_pred = predict_stacker(stacker, ml_val, jev_val, gate_val, gate_val_fire)
        stacker_val_f1 = float(f1_score(y_val_idx, stack_val_pred.argmax(axis=1),
                                        average="macro", zero_division=0))
        stack_test = predict_stacker(stacker, ml_test, jev_test, gate_test, gate_test_fire)
        rows.append({"name": f"Stacking meta-learner (LogReg; val-F1={stacker_val_f1:.3f})",
                     **metrics(y_true, preds_from(stack_test))})

    # ---- Jev-specificity controls -----------------------------------------
    # (a) ML + class-prior: a real control -- adding a class-weighted vector
    #     (not uniform) CAN flip argmax, so if this matches the ML+Jev lift,
    #     the lift was just calibration smoothing.
    # (b) ML + shuffled-Jev: the authoritative control. Shuffle Jev's rows so
    #     the row->prediction mapping is destroyed while the marginal
    #     distribution of Jev's answers is preserved. If ML + shuffled-Jev
    #     matches ML + Jev, the lift did NOT come from Jev's row-level
    #     information. We average over several shuffles to bound variance.
    # NOTE: ML + uniform is a mathematical tautology (same argmax as ML-only
    # for every alpha), so it is kept out of the ablation table: it would be
    # an invalid control.
    prior_vec = np.array([true_prior[c] for c in CLASSES], dtype=float)
    prior_probs = np.tile(prior_vec, (len(df_test), 1))
    rows.append({"name": f"Ablation: ML + class-prior (alpha={alpha_mj:.2f})",
                 **metrics(y_true, preds_from(soft_vote(ml_test, prior_probs, alpha_mj)))})

    # Shuffled-Jev: average every reported metric over 5 permutations.
    rng = np.random.default_rng(args.seed)
    shuf_metrics_list = []
    for _ in range(5):
        perm = rng.permutation(len(jev_test))
        shuf_test = jev_test[perm]
        shuf_metrics_list.append(
            metrics(y_true, preds_from(soft_vote(ml_test, shuf_test, alpha_mj)))
        )
    keys = shuf_metrics_list[0].keys()
    shuf_row = {k: float(np.mean([m[k] for m in shuf_metrics_list])) for k in keys}
    shuf_row["name"] = (f"Ablation: ML + shuffled-Jev "
                        f"(alpha={alpha_mj:.2f}, mean of 5 perms)")
    shuf_row["shuffled_jev_f1_std"] = float(np.std([m["macro_f1"] for m in shuf_metrics_list]))
    rows.append(shuf_row)

    # ablation: hard rule-gate override (a negative result)
    hard = ml_test.copy()
    hard[gate_test_fire] = gate_test[gate_test_fire]
    rows.append({"name": "Ablation: hard rule-gate -> ML",
                 **metrics(y_true, preds_from(hard))})

    # ---- Paired bootstrap CI for the headline lift ------------------------
    y_true_arr = np.asarray(y_true)
    ml_preds   = np.asarray(preds_from(ml_test))
    jev_mix    = np.asarray(preds_from(soft_vote(ml_test, jev_test, alpha_mj)))
    bootstrap = paired_bootstrap_ci(y_true_arr, ml_preds, jev_mix,
                                    n_boot=2000, seed=args.seed)
    print(f"\nBootstrap ΔF1 (ML+Jev − ML): mean={bootstrap['delta_mean']:+.4f}"
          f"  95% CI=[{bootstrap['ci_lo']:+.4f}, {bootstrap['ci_hi']:+.4f}]"
          f"  p={bootstrap['p_value_one_sided']:.4f}")

    # ---- Per-row bootstrap CI on every result row --------------------------
    # Attach a 95% CI on macro-F1 to every row so single-seed numbers are
    # bounded by sampling variance. Each row's preds are re-derived from the
    # posteriors it was built from (or we re-compute from argmax of the row's
    # table-time probabilities). For simplicity we re-run metrics' internals.
    def _row_preds(name: str) -> List[str]:
        """Recover the predictions used to compute a reported row's metrics."""
        if name.startswith("ML only"):           return preds_from(ml_test)
        if name.startswith("Jev only"):          return preds_from(jev_test)
        if name.startswith("ML + Jev soft vote"):return preds_from(soft_vote(ml_test, jev_test, alpha_mj))
        if name.startswith("ML + Jev + Rule"):   return preds_from(soft_vote3(ml_test, jev_test, gate_test, a3, b3))
        if "class-prior" in name:                return preds_from(soft_vote(ml_test, prior_probs, alpha_mj))
        if "hard rule-gate" in name:
            hp = ml_test.copy(); hp[gate_test_fire] = gate_test[gate_test_fire]
            return preds_from(hp)
        if "Confidence gate" in name:
            cg = soft_vote3(ml_test, jev_test, gate_test, a3, b3)
            g, _ = apply_confidence_gate(ml_test, cg, gate_threshold)
            return preds_from(g)
        if "Stacking" in name and args.enable_stack:
            return preds_from(predict_stacker(stacker, ml_test, jev_test, gate_test, gate_test_fire))
        if "shuffled-Jev" in name:               return None   # averaged over perms
        return None

    for r in rows:
        preds = _row_preds(r["name"])
        if preds is None:
            continue
        ci = row_bootstrap_ci(y_true_arr, np.asarray(preds),
                              n_boot=2000, seed=args.seed)
        r.update({"f1_boot_mean": ci["f1_mean"],
                  "f1_boot_std":  ci["f1_std"],
                  "f1_boot_ci_lo": ci["f1_ci_lo"],
                  "f1_boot_ci_hi": ci["f1_ci_hi"]})

    # ---- Print + save ----
    print("\n=== Results (same test rows for every model) ===")
    print(f"{'Model':44s} {'Macro F1':>9s} {'MCC':>7s} {'Acc':>7s}"
          f"  {'P_maj':>6s} {'R_maj':>6s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['name']:44s} {r['macro_f1']:9.3f} {r['mcc']:7.3f} {r['accuracy']:7.3f}"
              f"  {r.get('precision_major', 0):6.3f} {r.get('recall_major', 0):6.3f}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w") as f:
        json.dump({
            "split": args.split,
            "test_size": args.test_size,
            "test_sample": len(df_test),
            "val_sample": len(df_val),
            "few_shot": args.few_shot,
            "ml_model": args.ml_model,
            "alpha_mj": alpha_mj,
            "alpha_mjr": a3,
            "beta_mjr": b3,
            "val_f1_mj": val_f1_mj,
            "val_f1_mjr": val_f1_3,
            "llm_prior": llm_prior,
            "true_prior": true_prior,
            "gate_fire_val": float(gate_val_fire.mean()),
            "gate_fire_test": float(gate_test_fire.mean()),
            "confidence_gate_threshold": gate_threshold,
            "confidence_gate_skipped": gate_rate,
            "stacker_val_f1": stacker_val_f1,
            "bootstrap_ml_vs_mljev": bootstrap,
            "rows": rows,
            "mock_jev": args.mock_jev,
        }, f, indent=2)
    print(f"\n  report  -> {args.report_json}")
    save_report_table(rows, args.report_image,
                      header=f"Jev-optimized stack  ·  {args.split} split  ·  "
                             f"{len(df_test):,} test rows  ·  "
                             f"ML={args.ml_model}  ·  alpha_MJ={alpha_mj:.2f}")


if __name__ == "__main__":
    main()
