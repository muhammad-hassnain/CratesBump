#!/usr/bin/env python3
"""
Cap over-represented crates/families in a dataset file.

A handful of auto-generated or monorepo families dominate the raw corpus
(e.g. aws-sdk-*, swc*, google-*, spider*). Left uncapped they bias any model
toward their release patterns. This script writes a capped variant by:

  1. capping the number of pairs kept per crate    (--per-crate,  default 100)
  2. then capping the number kept per family        (--per-family, default 5000)

Rows are sampled uniformly at random within each group with a fixed seed, so
the output is deterministic and reproducible. The family key is the token
before the first '-' or '_' in the crate name (aws-sdk-ec2, aws-config -> "aws";
swc, swc_core -> "swc"; spider, spider_cli -> "spider").

Usage:
    python3 scripts/cap_dataset.py \
        --input  data/compliant_dataset.csv.gz \
        --output data/compliant_dataset_capped.csv.gz \
        --per-crate 100 --per-family 5000 --seed 42
"""
import argparse
import re

import pandas as pd


def family_key(name: str) -> str:
    parts = re.split(r"[-_]", str(name))
    return parts[0] if parts and parts[0] else str(name)


def cap(df: pd.DataFrame, per_crate: int, per_family: int, seed: int) -> pd.DataFrame:
    # Shuffle once, then take the first `cap` rows per group: this is a uniform
    # random sample of up to `cap` rows per group, and avoids version-specific
    # groupby.apply behaviour (works on any pandas >= 2.0).
    # Stage 1 — per-crate cap (tames mega-version crates like spider / swc_core).
    if per_crate and per_crate > 0:
        df = (df.sample(frac=1, random_state=seed)
                .groupby("package", group_keys=False).head(per_crate))
    # Stage 2 — per-family cap (tames generated families like aws-sdk / google).
    if per_family and per_family > 0:
        df = df.assign(_fam=df["package"].map(family_key))
        df = (df.sample(frac=1, random_state=seed)
                .groupby("_fam", group_keys=False).head(per_family)
                .drop(columns="_fam"))
    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame, label: str) -> None:
    total = len(df)
    vc = df["release_type"].value_counts(normalize=True)
    fam = df["package"].map(family_key).value_counts().head(6)
    print(f"\n{label}: {df['package'].nunique():,} crates, {total:,} pairs")
    print("  labels:", {k: f"{100 * vc.get(k, 0):.1f}%" for k in ("patch", "minor", "major")})
    print("  top families (pairs):", {k: int(v) for k, v in fam.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description="Cap over-represented crates/families.")
    ap.add_argument("--input", default="data/compliant_dataset.csv.gz")
    ap.add_argument("--output", default="data/compliant_dataset_capped.csv.gz")
    ap.add_argument("--per-crate", type=int, default=100)
    ap.add_argument("--per-family", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input, keep_default_na=False, na_values=[""])
    summarize(df, "BEFORE")
    capped = cap(df, args.per_crate, args.per_family, args.seed)
    summarize(capped, "AFTER ")
    # Deterministic row order (also restores compressibility lost by shuffling).
    capped = capped.sort_values(["package", "version"]).reset_index(drop=True)
    capped.to_csv(args.output, index=False, compression="gzip")
    print(f"\nWrote {args.output}  ({len(capped):,} rows)")


if __name__ == "__main__":
    main()
