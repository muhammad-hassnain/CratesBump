#!/usr/bin/env python3
"""Re-derive `release_type` using Cargo's actual SemVer rule.

The extractor's labels come from a naive
`cm>pm → major; cn>pn → minor; cp>pp → patch` comparison (src/extractor/mod.rs).
For 0.y.z versions the Cargo book is explicit: **the leftmost non-zero field
is the breaking boundary**. So:

  * `0.1.0 -> 0.2.0`  is MAJOR (breaking change), not minor.
  * `0.1.0 -> 0.1.1`  is MINOR (new feature-ish), not patch.
  * `0.0.1 -> 0.0.2`  is MAJOR (every change in 0.0.z is breaking).
  * `0.9.3 -> 1.0.0`  is MAJOR (stabilisation is always a major boundary).
  * `1.2.3 -> 1.3.0`  is MINOR (normal rules apply once ≥1.0.0).
  * `1.2.3 -> 1.2.4`  is PATCH.

This script reads a labelled CSV, re-derives the label with the correct rule,
and writes a new file. Rows whose version strings don't parse are dropped
(same policy as the Rust extractor).

The released dataset is labelled by scripts/label_pairs_cargo.py, which
imports cargo_bump() from here.

Usage:
    python3 scripts/relabel_cargo.py <in.csv.gz> <out.csv.gz>
"""
from __future__ import annotations
import argparse
import os
import sys

import pandas as pd


CLASSES = ("major", "minor", "patch")


def parse_version(s: str) -> tuple[int, int, int] | None:
    """Strip leading 'v', pre-release ('-alpha'), and build metadata ('+build').
    Return (M, m, p) tuple of non-negative ints, or None if unparseable."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("v"):
        s = s[1:]
    for sep in ("-", "+"):
        if sep in s:
            s = s.split(sep, 1)[0]
    parts = s.split(".")
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def cargo_bump(prev: tuple[int, int, int], curr: tuple[int, int, int]) -> str | None:
    """Cargo's SemVer rule. Returns 'major', 'minor', 'patch', or None if curr
    is not strictly after prev (same precedence, or a backport)."""
    pM, pm, pp = prev
    cM, cm, cp = curr

    # ------- Crossing a major boundary at ≥1.0.0 -------
    if cM > pM:
        return "major"
    if cM < pM:
        return None                                    # out-of-order

    # cM == pM at this point. Branch on 0.x.y rules.
    if pM == 0:
        # 0.0.z world — EVERY change is breaking per Cargo.
        if pm == 0 and cm == 0:
            if cp > pp:
                return "major"
            return None
        # Mixed 0.0.z -> 0.1+.z: pm == 0 and cm > 0 means the leftmost non-zero
        # moved from (none) to minor -> that is a breaking change.
        if pm == 0 and cm > 0:
            return "major"
        if cm > pm:
            return "major"                             # 0.y.z -> 0.(y+1).z
        if cm < pm:
            return None
        # cm == pm > 0, 0.y.z world: patch bump IS a minor-like addition
        # under Cargo's convention (0.y.z→0.y.(z+1) is backward-compatible).
        if cp > pp:
            return "minor"
        return None

    # Both prev and curr at major >= 1 — classic rules.
    if cm > pm:
        return "minor"
    if cm < pm:
        return None
    if cp > pp:
        return "patch"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--prev-col", default="prev_version",
                    help="column name for prev version if present (otherwise we "
                         "re-derive it from grouped order)")
    ap.add_argument("--version-col", default="version")
    ap.add_argument("--package-col", default="package")
    ap.add_argument("--keep-unlabeled", action="store_true",
                    help="keep rows whose relabel is None (otherwise drop)")
    args = ap.parse_args()

    print(f"reading {args.in_csv} ...", file=sys.stderr)
    df = pd.read_csv(args.in_csv, keep_default_na=False, na_values=[""])
    n0 = len(df)
    print(f"  {n0:,} rows, {df[args.package_col].nunique():,} crates", file=sys.stderr)

    # Without a prev column, prev is approximated by chaining the dataset's own
    # rows in precedence order. This does NOT match the extractor (plan_pairs
    # walks all published versions in publish order), so it drops each crate's
    # first pair and mislabels rows next to skipped versions. The headline file
    # is built with scripts/label_pairs_cargo.py instead; this mode is kept only
    # to regenerate the archival v3_clean_cargo file.
    if args.prev_col not in df.columns:
        print("  prev_version column absent — reconstructing chain from grouped order",
              file=sys.stderr)
        df["_parsed"] = df[args.version_col].map(parse_version)
        df = df.dropna(subset=["_parsed"]).reset_index(drop=True)
        print(f"  dropped {n0 - len(df):,} unparseable version rows "
              f"({(n0 - len(df)) / max(n0,1) * 100:.1f}%)", file=sys.stderr)

        # per-crate: sort by (parsed_version, original index), chain prev→curr
        new_label = [None] * len(df)
        unlabeled = 0
        for pkg, g in df.groupby(args.package_col, sort=False):
            idx = g.index.tolist()
            idx.sort(key=lambda i: (df.at[i, "_parsed"], i))
            prev = None
            for i in idx:
                curr = df.at[i, "_parsed"]
                if prev is None:
                    # first row for the crate: no previous version to pair against,
                    # so no label can be derived -- mark as unlabeled.
                    new_label[i] = None
                    unlabeled += 1
                else:
                    new_label[i] = cargo_bump(prev, curr)
                    if new_label[i] is None:
                        unlabeled += 1
                prev = curr
        df["release_type_cargo"] = new_label
        df = df.drop(columns=["_parsed"])
        print(f"  {unlabeled:,} rows unlabeled under cargo rules "
              f"({unlabeled / max(len(df),1) * 100:.1f}%)", file=sys.stderr)
    else:
        df["release_type_cargo"] = [
            cargo_bump(parse_version(p), parse_version(c))
            if parse_version(p) and parse_version(c) else None
            for p, c in zip(df[args.prev_col], df[args.version_col])
        ]

    # Compare old vs new labelling
    cmp_counts = (
        df.dropna(subset=["release_type_cargo"])
          .groupby(["release_type", "release_type_cargo"])
          .size().unstack(fill_value=0)
    )
    print("\n  cross-tab: original (rows) × Cargo-correct (cols):", file=sys.stderr)
    print(cmp_counts.to_string(), file=sys.stderr)

    # Of the labelled rows that flipped, decompose by WHY they flipped.
    # Pairs where both prev.major >= 1 and curr.major >= 1 are stable-era
    # pairs where Cargo and naive rules agree; a flip there is a chain-
    # pairing artefact (publish-order vs semver-precedence chain), NOT
    # Cargo-rule attributable, so they are left out of the mechanism table.
    if "_parsed" not in df.columns:
        df["_parsed"] = df[args.version_col].map(parse_version)

    def case_of(prev, curr):
        if prev is None or curr is None:
            return "no_prev"
        pM, pm, pp = prev; cM, cm, cp = curr
        if pM >= 1 and cM >= 1:
            return "stable_era_pairing_artefact"
        if cM > pM:
            if pM == 0 and cM >= 1:
                return "0.x_to_1.0_stabilisation"
            return "cross_boundary_other"
        if pM == 0 and cM == 0:
            if pm == 0 and cm == 0:
                return "0.0.z_every_change_major"
            if pm == 0 and cm > 0:
                return "0.0.z_to_0.y.z_major"
            if cm > pm:
                return "0.y.z_minor_becomes_major"
            if cp > pp:
                return "0.y.z_patch_becomes_minor"
        return "edge_other"

    # Reconstruct the prev for each row from the same chain
    prevs: list = [None] * len(df)
    for pkg, g in df.groupby(args.package_col, sort=False):
        idx = g.index.tolist()
        idx.sort(key=lambda i: (df.at[i, "_parsed"], i))
        prev = None
        for i in idx:
            prevs[i] = prev
            prev = df.at[i, "_parsed"]

    case_counts: dict[str, int] = {}
    pairing_artefact_flips = 0
    for i in range(len(df)):
        new = df.at[i, "release_type_cargo"]
        old = df.at[i, "release_type"]
        # pandas represents our None labels as NaN after column assignment;
        # we need pd.isna, not `is None`.
        if pd.isna(new) or new == old:
            continue
        c = case_of(prevs[i], df.at[i, "_parsed"])
        case_counts[c] = case_counts.get(c, 0) + 1
        if c == "stable_era_pairing_artefact":
            pairing_artefact_flips += 1

    # Count only labelled rows (NaN != "patch" is True in pandas, so we must
    # restrict the "changed" count to the Cargo-labelled subset explicitly).
    labelled_mask = df["release_type_cargo"].notna()
    changed_raw = ((df["release_type"] != df["release_type_cargo"])
                   & labelled_mask).sum()
    cargo_flips = changed_raw - pairing_artefact_flips
    print(f"\n  {changed_raw:,} of {len(df):,} labels differ naively "
          f"({changed_raw / max(len(df),1) * 100:.1f}%)", file=sys.stderr)
    print(f"  of which {pairing_artefact_flips:,} are stable-era chain-pairing "
          f"artefacts (NOT Cargo-rule attributable)", file=sys.stderr)
    print(f"  honest Cargo-attributable flip count: {cargo_flips:,}",
          file=sys.stderr)

    print("\n  §5 mechanism table — Cargo-attributable flips by case:",
          file=sys.stderr)
    print(f"    {'Case':40s} {'Count':>8s} {'% of Cargo':>12s}", file=sys.stderr)
    print(f"    {'-' * 65}", file=sys.stderr)
    for c, n in sorted(case_counts.items(), key=lambda x: -x[1]):
        if c == "stable_era_pairing_artefact":
            continue
        pct = n / max(cargo_flips, 1) * 100
        print(f"    {c:40s} {n:>8,} {pct:>11.1f}%", file=sys.stderr)
    print(f"    {'-' * 65}", file=sys.stderr)
    print(f"    {'TOTAL Cargo-attributable':40s} {cargo_flips:>8,}",
          file=sys.stderr)

    # Overwrite release_type with the Cargo-correct value
    df["release_type"] = df["release_type_cargo"]
    df = df.drop(columns=["release_type_cargo"])

    if not args.keep_unlabeled:
        before = len(df)
        df = df.dropna(subset=["release_type"]).reset_index(drop=True)
        print(f"  dropped {before - len(df):,} unlabeled rows", file=sys.stderr)

    vc = df["release_type"].value_counts()
    print("\n  new label distribution:", file=sys.stderr)
    for c in CLASSES:
        pct = vc.get(c, 0) / max(len(df), 1) * 100
        print(f"    {c:5s}: {int(vc.get(c, 0)):>7,} ({pct:5.1f}%)", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    print(f"\n  writing {args.out_csv} ...", file=sys.stderr)
    df.to_csv(args.out_csv, index=False, compression="infer")
    print(f"  done: {len(df):,} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
