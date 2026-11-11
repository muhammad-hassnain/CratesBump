#!/usr/bin/env python3
"""Aggregate per-backbone Jev-optimized reports into one summary table.

Reads results/jev_<BACKBONE>.json for each named backbone and emits:
  * results/jev_summary.json  cross-backbone comparison, same test sample
  * results/jev_summary.png   headline table for the paper

Usage:
    python3 scripts/jev_summarize.py XGBoost LightGBM "Hist Gradient Boosting" "Random Forest"
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_report(name: str, results_dir: str, prefix: str = "jev_") -> dict:
    safe = name.replace(" ", "_")
    path = os.path.join(results_dir, f"{prefix}{safe}.json")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: missing {path} -- run jev_optimized.py for {name} first.")
    with open(path) as f:
        return json.load(f)


def pick_row(report: dict, keyword: str) -> dict:
    """Find the row in report['rows'] whose 'name' contains keyword."""
    for r in report["rows"]:
        if keyword.lower() in r["name"].lower():
            return r
    raise KeyError(f"no row matching {keyword!r} in {report.get('ml_model')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbones", nargs="+",
                    help="ML backbone names (must match --ml-model used earlier)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--prefix", default="jev_",
                    help="filename prefix: looks for <prefix><Backbone>.json (default jev_)")
    ap.add_argument("--out-json", default="results/jev_summary.json")
    ap.add_argument("--out-image", default="results/jev_summary.png")
    ap.add_argument("--extra-variants", nargs="*", default=[],
                    help="additional ablation rows to include, each as 'display:needle'")
    args = ap.parse_args()

    reports = {name: load_report(name, args.results_dir, args.prefix)
               for name in args.backbones}

    # Each cell: [backbone x variant] -> (macro_f1, mcc, acc)
    variants = [
        ("ML only",             "ML only"),
        ("+ Jev (soft vote)",   "soft vote"),
        ("+ Jev + Rule",        "+ Rule"),
    ]
    for ev in args.extra_variants:
        if ":" in ev:
            display, needle = ev.split(":", 1)
            variants.append((display.strip(), needle.strip()))
    cell_text, cell_colors, deltas = [], [], []
    ml_baselines = {}
    for name in args.backbones:
        rep = reports[name]
        row = []
        row_colors = []
        ml_f1 = pick_row(rep, "ML only")["macro_f1"]
        ml_baselines[name] = ml_f1
        for _, needle in variants:
            r = pick_row(rep, needle)
            cell = f"{r['macro_f1']:.3f}"
            # delta-F1 vs backbone's own ML-only
            if needle != "ML only":
                d = r["macro_f1"] - ml_f1
                sign = "+" if d >= 0 else ""
                cell = f"{r['macro_f1']:.3f}  ({sign}{d:.3f})"
            row.append(cell)
            shade = max(0.0, min(1.0, r["macro_f1"]))
            if needle != "ML only":
                d = r["macro_f1"] - ml_f1
                if d > 0.005:
                    row_colors.append((0.80, 0.95, 0.80))   # green: improvement
                elif d < -0.005:
                    row_colors.append((0.97, 0.80, 0.80))   # red: regression
                else:
                    row_colors.append((0.95, 0.95, 0.95))
            else:
                row_colors.append((1 - 0.3 * shade, 0.88 + 0.08 * shade, 1 - 0.3 * shade))
        cell_text.append(row)
        cell_colors.append(row_colors)
        deltas.append({
            "backbone": name,
            "ml_only": ml_f1,
            "ml_plus_jev": pick_row(rep, "soft vote")["macro_f1"],
            "ml_plus_jev_plus_rule": pick_row(rep, "+ Rule")["macro_f1"],
            "jev_only": pick_row(rep, "Jev only")["macro_f1"],
            "alpha_mj": rep.get("alpha_mj"),
            "alpha_mjr": rep.get("alpha_mjr"),
            "beta_mjr": rep.get("beta_mjr"),
        })

    # ---- Render table ----
    col_labels = [label for label, _ in variants]
    fig, ax = plt.subplots(figsize=(12, 1.5 + 0.6 * len(args.backbones)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
                   rowLabels=args.backbones, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.18, 2.0)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(args.backbones)):
        tbl[i + 1, -1].set_facecolor("#dce3ea")
        tbl[i + 1, -1].set_text_props(fontweight="bold")

    # pick a stable sample for the header, using the first backbone's report
    first = reports[args.backbones[0]]
    header = (f"Jev-optimized stack across ML backbones  ·  grouped-crate split  "
              f"·  {first.get('test_sample', '?')} test rows  "
              f"·  macro-F1 (Δ vs ML-only in brackets)")
    fig.text(0.5, 0.97, header, ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.03,
             "Green = Jev lift over ML-only.  Red = regression.  "
             "α tuned per-backbone on 800-row val slice.",
             ha="center", va="bottom", fontsize=9, color="#444")
    os.makedirs(os.path.dirname(args.out_image) or ".", exist_ok=True)
    plt.savefig(args.out_image, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  summary table -> {args.out_image}")

    with open(args.out_json, "w") as f:
        json.dump({
            "test_sample": first.get("test_sample"),
            "val_sample":  first.get("val_sample"),
            "split":       first.get("split"),
            "few_shot":    first.get("few_shot"),
            "backbones":   deltas,
        }, f, indent=2)
    print(f"  summary json  -> {args.out_json}")

    # Terminal pretty-print
    print("\n=== Cross-backbone summary ===")
    print(f"{'Backbone':24s} {'ML only':>10s} {'+Jev':>18s} {'+Jev+Rule':>18s}")
    print("-" * 74)
    for row, name in zip(cell_text, args.backbones):
        print(f"{name:24s} {row[0]:>10s} {row[1]:>18s} {row[2]:>18s}")


if __name__ == "__main__":
    main()
