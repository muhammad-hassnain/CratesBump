#!/usr/bin/env python3
"""Cheap internal-consistency sanity check on the shipped labels.

This is NOT a ground-truth audit — it uses the extractor's OWN features to
flag suspicious labels where internal signals contradict the label. A
cargo-semver-checks-based external audit on a sampled subset lives in
`scripts/label_audit_cargo_semver_checks.py`.

Specifically, for each row (DELETION_COLS = {dem, gvd, drf, mpd, mpc}):
  - A `patch` label is suspicious if any of DELETION_COLS > 0
    (methods/globals/rust-files deleted or method params/signatures
    changed — all are breaking changes per Cargo).
  - A `minor` label is suspicious if any of DELETION_COLS > 0
    (same reason — any deletion is breaking regardless of label).
  - A `major` label is "deletion-less" if NO deletion signal fired
    (`dem == gvd == drf == mpd == mpc == 0`). Deletions are the only
    mandated-breaking changes under SemVer; a major without any is
    either a 0.y.z Cargo-rule major (preventive, no actual break) or
    an author-elective major (dependency bump, release-note convention,
    coordinated breakage without a syntactic trace). We flag these so
    the reader can see the Cargo-rule-only major share.

The output is a per-row flag CSV and a summary printed to stdout. These
flags are NOT dropped from the dataset — they are reported transparently
as an internal consistency probe (make reproduce-section-6).

Usage:
    python3 scripts/label_sanity_check.py \\
        data/cratesbump.csv.gz \\
        results/label_sanity_rebuilt_cargo.csv \\
        --summary results/label_sanity_rebuilt_cargo_summary.json
"""
import argparse
import json
import sys

import pandas as pd


DELETION_COLS = ("dem", "gvd", "drf", "mpd", "mpc")
# Any change in these = detectable code activity (used for the "major with
# no code change at all" branch, which basically never fires because la/ld
# are 100%-populated — kept for completeness).
CHANGE_COLS = ("adm", "dem", "gvd", "arf", "drf", "mpc", "mla", "mld", "la", "ld")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    print(f"Reading {args.input} ...", file=sys.stderr)
    df = pd.read_csv(args.input, keep_default_na=False, na_values=[""])
    print(f"  {len(df):,} rows", file=sys.stderr)

    # ensure numeric
    for c in set(DELETION_COLS + CHANGE_COLS):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0

    deletions = (df[list(DELETION_COLS)].sum(axis=1) > 0)

    flag = []
    flag_reason = []
    for i, row in df.iterrows():
        rt = row["release_type"]
        if rt in ("patch", "minor") and deletions.iloc[i]:
            flag.append(True)
            flag_reason.append(f"{rt}_with_deletions")
        elif rt == "major" and not deletions.iloc[i]:
            # Major label with no deletion signal → the "breakage" is either
            # a 0.y.z Cargo-rule major, a dependency/behaviour change, or an
            # author-elective major without a syntactic deletion trace.
            flag.append(True)
            flag_reason.append("major_without_deletions")
        else:
            flag.append(False)
            flag_reason.append("")

    df_out = pd.DataFrame({
        "package":      df["package"],
        "version":      df["version"],
        "release_type": df["release_type"],
        "flag":         flag,
        "reason":       flag_reason,
    })

    # Only persist the flagged rows to keep the file small.
    flagged = df_out[df_out["flag"]].copy()
    flagged.to_csv(args.output, index=False)

    # Summary breakdown
    counts = {
        "n_rows":          int(len(df)),
        "n_flagged":       int(flagged.shape[0]),
        "pct_flagged":     float(100.0 * flagged.shape[0] / max(len(df), 1)),
        "by_reason":       flagged["reason"].value_counts().to_dict(),
        "by_label_and_reason": {
            rt: flagged[flagged["release_type"] == rt]["reason"].value_counts().to_dict()
            for rt in ("major", "minor", "patch")
        },
    }
    print(json.dumps(counts, indent=2), file=sys.stderr)
    if args.summary:
        with open(args.summary, "w") as fh:
            json.dump(counts, fh, indent=2)
        print(f"Wrote {args.summary}", file=sys.stderr)
    print(f"Wrote flagged rows -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
