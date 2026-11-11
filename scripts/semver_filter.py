#!/usr/bin/env python3
"""
SemVer-compliance filtering for the crate dataset.

A crate is treated as *SemVer-compliant* when it has reached a stable public
API, i.e. it has published at least one non-pre-release version >= 1.0.0.
Crates that never left 0.x, or that only ever published pre-releases
(1.0.0-beta, ...), never promised stability, so their major/minor/patch bumps
do not carry real SemVer semantics and are excluded.

This module is importable (used by pipeline.py) and runnable as a report:

    python3 scripts/semver_filter.py                 # report over crate_csvs/
    python3 scripts/semver_filter.py --strict        # literal > 1.0.0
    python3 scripts/semver_filter.py --write-list results/compliant_crates.txt
"""

import os
import re
import glob
import argparse

# MAJOR.MINOR.PATCH  with optional -prerelease and +build (crates.io SemVer 2.0).
_SEMVER_RE = re.compile(
    r"^\s*v?(\d+)\.(\d+)\.(\d+)"     # core
    r"(?:-([0-9A-Za-z.\-]+))?"       # pre-release (group 4)
    r"(?:\+[0-9A-Za-z.\-]+)?\s*$"    # build metadata (ignored)
)


def parse_version(v):
    """Return (major, minor, patch, is_prerelease) or None if unparseable."""
    m = _SEMVER_RE.match(str(v))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) is not None)


def is_compliant_version(v, strict_gt=False):
    """True if v is a stable (non-pre-release) release at/after 1.0.0.

    strict_gt=False -> core >= (1,0,0)   (reached stable 1.0; recommended)
    strict_gt=True  -> core >  (1,0,0)   (literal "> 1.0.0"; excludes exactly 1.0.0)
    """
    p = parse_version(v)
    if p is None:
        return False
    major, minor, patch, is_pre = p
    if is_pre:
        return False
    core = (major, minor, patch)
    return core > (1, 0, 0) if strict_gt else core >= (1, 0, 0)


def compliant_packages(df, strict_gt=False):
    """Set of package names having >=1 SemVer-compliant version in `df`.

    Requires 'package' and 'version' columns.
    """
    if "package" not in df.columns or "version" not in df.columns:
        return set(df["package"].unique()) if "package" in df.columns else set()
    mask = df["version"].map(lambda v: is_compliant_version(v, strict_gt))
    return set(df.loc[mask, "package"].unique())


# ── report ────────────────────────────────────────────────────────────────────
def _dist(df):
    total = len(df)
    vc = df["release_type"].value_counts() if "release_type" in df.columns else {}
    parts = []
    for c in ("patch", "minor", "major"):
        n = int(vc.get(c, 0))
        pct = (100.0 * n / total) if total else 0.0
        parts.append(f"{c} {n:,} ({pct:.1f}%)")
    return "  ".join(parts)


def _load(crate_dir):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(crate_dir, "*.csv")))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, keep_default_na=False, na_values=[""]))
        except Exception as e:
            print(f"  warning: skipping {f} ({e})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Report SemVer-compliance impact on the dataset.")
    ap.add_argument("--crate-dir", default="crate_csvs")
    ap.add_argument("--strict", action="store_true",
                    help="Use literal > 1.0.0 (exclude crates whose only stable release is exactly 1.0.0).")
    ap.add_argument("--write-list", metavar="PATH",
                    help="Write the compliant crate names (one per line) to PATH.")
    args = ap.parse_args()

    df = _load(args.crate_dir)
    if df.empty:
        print(f"No CSVs found in {args.crate_dir}/.")
        return

    n_crates = df["package"].nunique()
    n_pairs = len(df)
    print("=== SemVer-compliance report ===")
    print(f"source: {args.crate_dir}/")
    print(f"total : {n_crates:,} crates, {n_pairs:,} pairs")
    print(f"labels: {_dist(df)}")
    print()

    # Crate-level filter (what to keep): crates with >=1 stable release.
    for strict in (False, True):
        keep = compliant_packages(df, strict_gt=strict)
        kept = df[df["package"].isin(keep)]
        tag = "> 1.0.0 (strict)" if strict else ">= 1.0.0"
        print(f"crate-level filter, stable {tag}:")
        print(f"  crates kept: {len(keep):,} / {n_crates:,} "
              f"({100.0*len(keep)/n_crates:.1f}%)")
        print(f"  pairs kept : {len(kept):,} / {n_pairs:,} "
              f"({100.0*len(kept)/n_pairs:.1f}%)")
        print(f"  labels     : {_dist(kept)}")
        print()

    # Pair-level view (stronger): only pairs whose 'to' version is stable >=1.0.0.
    pair_mask = df["version"].map(lambda v: is_compliant_version(v, strict_gt=args.strict))
    pl = df[pair_mask]
    print(f"pair-level view (drop pairs whose 'to' version is 0.x / pre-release):")
    print(f"  pairs kept : {len(pl):,} / {n_pairs:,} ({100.0*len(pl)/n_pairs:.1f}%)")
    print(f"  labels     : {_dist(pl)}")

    if args.write_list:
        keep = sorted(compliant_packages(df, strict_gt=args.strict))
        os.makedirs(os.path.dirname(os.path.abspath(args.write_list)), exist_ok=True)
        with open(args.write_list, "w") as fh:
            fh.write("\n".join(keep) + "\n")
        print()
        print(f"wrote {len(keep):,} compliant crate names -> {args.write_list}")


if __name__ == "__main__":
    main()
