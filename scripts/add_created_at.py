#!/usr/bin/env python3
"""Attach a `created_at` ISO-timestamp column to a dataset CSV that lacks one.

Dates come from `enrich_cache/versions/<pkg>.json`, which the GitHub-enrichment
crawler already populated as a side-effect:

    {curr_version: [t_prev_iso, t_curr_iso]}

`created_at` is the SECOND element (the publish timestamp of the row's version).

The temporal split in scripts/pipeline.py / scripts/evaluate.py needs this
column. The released data/cratesbump.csv.gz already has it (written by
scripts/label_pairs_cargo.py); this script is for older CSVs.

Usage:
    python3 scripts/add_created_at.py <in.csv.gz> <out.csv.gz>
"""
import argparse
import json
import os
import urllib.parse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--cache-dir", default="enrich_cache",
                    help="Directory holding versions/<pkg>.json "
                         "(default: enrich_cache)")
    args = ap.parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input, keep_default_na=False, na_values=[""])
    n_rows = len(df)
    print(f"  {n_rows:,} rows, {df['package'].nunique():,} crates")

    cache = {}

    def _windows_for(pkg: str) -> dict:
        if pkg in cache:
            return cache[pkg]
        p = os.path.join(args.cache_dir, "versions",
                         f"{urllib.parse.quote(pkg, safe='')}.json")
        try:
            with open(p) as fh:
                cache[pkg] = json.load(fh) or {}
        except (OSError, json.JSONDecodeError):
            cache[pkg] = {}
        return cache[pkg]

    created = []
    hits = misses = 0
    for pkg, ver in zip(df["package"], df["version"]):
        windows = _windows_for(str(pkg))
        tup = windows.get(str(ver))
        if tup and len(tup) == 2 and tup[1]:
            created.append(tup[1])
            hits += 1
        else:
            created.append("")
            misses += 1

    df["created_at"] = created
    print(f"  attached created_at for {hits:,}/{n_rows:,} rows "
          f"({100*hits/n_rows:.1f}%), {misses:,} unknown")

    df.to_csv(args.output, index=False, compression="gzip")
    print(f"Wrote {args.output}")

    valid = df[df["created_at"] != ""]["created_at"]
    if len(valid):
        ts = pd.to_datetime(valid, errors="coerce").dropna()
        print(f"  date range: {ts.min()} .. {ts.max()}")


if __name__ == "__main__":
    main()
