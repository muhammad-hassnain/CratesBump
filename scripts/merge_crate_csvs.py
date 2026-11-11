#!/usr/bin/env python3
"""Merge the extractor's per-crate CSVs into one naive-labelled file.

Keeps package, version, the extractor's naive release_type and the 41
trained features (scripts/pipeline.py FEATURE_COLS); adds an empty
created_at, which label_pairs_cargo.py fills from the crates.io dump. The
extractor's extra columns (fa, fd, edc) are not part of the dataset schema.

    python3 scripts/merge_crate_csvs.py crate_csvs_full/ data/cratesbump_extracted.csv.gz
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(src, "*.csv")))
    frames, empty = [], 0
    for f in files:
        d = pd.read_csv(f, keep_default_na=False, na_values=[""], dtype={"version": str, "package": str})
        if d.empty:
            empty += 1
            continue
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in P.FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df["created_at"] = None
    df = df[["package", "version", "release_type"] + P.FEATURE_COLS + ["created_at"]]
    df = df.drop_duplicates(["package", "version"])
    df.to_csv(out, index=False, compression="infer")
    print(f"{len(files):,} crate files ({empty:,} empty) -> {len(df):,} rows, "
          f"{df['package'].nunique():,} crates -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
