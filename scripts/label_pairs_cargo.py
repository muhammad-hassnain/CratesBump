#!/usr/bin/env python3
"""Attach the true predecessor to every extracted release pair and label it
under Cargo's SemVer rule.

Replaces the earlier two-step chain (an extractor-artefact filter, then
relabel_cargo.py) used for the earlier dataset versions. That chain rebuilt each row's
predecessor by sorting the dataset's own `version` values by precedence. That differs from the extractor
(src/extractor/mod.rs::plan_pairs), which walks all non-yanked published
versions in publish order and pairs each new precedence maximum with the
previous one. The mismatch mislabelled ~0.4% of rows and dropped every
crate's first pair (whose predecessor, the crate's first release, is never a
row of its own), removing ~5k single-pair crates entirely.

Here the predecessor is rebuilt from the crates.io dump with the extractor's
exact rule, and a row is kept only if that predecessor reproduces the naive
label the extractor itself wrote (so features and label describe the same
pair). Output adds `prev_version` and `release_type_naive` columns;
`release_type` holds the Cargo label.

Versions are ordered by full SemVer precedence, exactly as in the extractor
(scripts/semver_order.py): 1.0.0 published after 1.0.0-rc.1 forms no pair (same
MAJOR.MINOR.PATCH, no bump class) but becomes the baseline, so the next release
(say 1.0.1) is paired with 1.0.0.

Optionally (--max-date, --cap-per-crate, --cap-per-family) the output is cut
at a single snapshot instant and the per-crate/per-family caps of
cap_dataset.py are re-applied, so both hold uniformly for every crate.

Usage:
    python3 scripts/label_pairs_cargo.py \\
        data/cratesbump_extracted.csv.gz \\
        .dump_cache/2026-10-12-020016 \\
        data/cratesbump.csv.gz \\
        --max-date 2026-09-14 --cap-per-crate 100 --cap-per-family 5000 \\
        --report results/cargo_label_report.json
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downstream_impact import naive_bump, parse_version  # noqa: E402
from relabel_cargo import cargo_bump  # noqa: E402
from cap_dataset import cap  # noqa: E402
from semver_order import plan_prev  # noqa: E402
from enrich_github import repo_key  # noqa: E402


def flip_case(prev, curr):
    pM, pm, pp = prev
    cM, cm, cp = curr
    if pM >= 1:
        return "stable_era"
    if cM >= 1:
        return "0.x_to_1.0_stabilisation"
    if pm == 0 and cm == 0:
        return "0.0.z_every_change_major"
    if pm == 0 and cm > 0:
        return "0.0.z_to_0.y.z"
    if cm > pm:
        return "0.y.z_minor_becomes_major"
    return "0.y.z_patch_becomes_minor"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_csv", help="naive-labelled extractor output")
    ap.add_argument("dump_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--report", default="results/cargo_label_report.json")
    ap.add_argument("--add-from", help="extra naive-labelled CSV; rows absent from in_csv "
                                       "are appended (then verified like all others)")
    ap.add_argument("--exclude-crates", help="file of crate names whose --add-from rows "
                                             "must not be used (e.g. pre-fix backport crates)")
    ap.add_argument("--max-date", help="keep only pairs whose release was published before "
                                       "this UTC instant (e.g. 2026-09-14)")
    ap.add_argument("--cap-per-crate", type=int, default=0,
                    help="at most N pairs per crate, sampled uniformly (0 = no cap)")
    ap.add_argument("--cap-per-family", type=int, default=0,
                    help="at most N pairs per name family, see cap_dataset.family_key")
    ap.add_argument("--cap-seed", type=int, default=42)
    ap.add_argument("--repos", help="package,repository CSV; adds a `repository` column "
                                    "(normalised, e.g. github.com/owner/repo; empty if none) "
                                    "so users can hold out whole repositories")
    ap.add_argument("--check-rdtd", action="store_true",
                    help="also require the extractor's rdtd to match the gap to prev_version "
                         "(within 1e-3 days), proving the features describe that pair")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, keep_default_na=False, na_values=[""])
    n_added = 0
    if args.add_from:
        extra = pd.read_csv(args.add_from, keep_default_na=False, na_values=[""])
        if args.exclude_crates:
            excluded = set(open(args.exclude_crates).read().split())
            extra = extra[~extra["package"].isin(excluded)]
        have = set(zip(df["package"], df["version"]))
        extra = extra[[k not in have for k in zip(extra["package"], extra["version"])]]
        n_added = len(extra)
        df = pd.concat([df, extra[df.columns]], ignore_index=True)
    n_in, crates_in = len(df), df["package"].nunique()
    print(f"input: {n_in:,} rows ({n_added:,} from --add-from), {crates_in:,} crates",
          file=sys.stderr)
    crates = set(df["package"])

    data = os.path.join(args.dump_dir, "data")
    csv.field_size_limit(1 << 30)
    id2name = {}
    with open(os.path.join(data, "crates.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["name"] in crates:
                id2name[r["id"]] = r["name"]
    versions = defaultdict(list)
    published = {}
    with open(os.path.join(data, "versions.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            name = id2name.get(r["crate_id"])
            if name and r.get("yanked", "f") not in ("t", "true", "True", "1"):
                versions[name].append((r["created_at"], r["num"]))
                published[(name, r["num"])] = r["created_at"]

    prev_of = {}
    for name, vs in versions.items():
        vs.sort()
        for curr, prev in plan_prev([num for _, num in vs]).items():
            prev_of[(name, curr)] = prev

    def gap_days(pkg, a, b):
        ta = pd.Timestamp(published[(pkg, a)]); tb = pd.Timestamp(published[(pkg, b)])
        return (tb - ta).total_seconds() / 86400

    rdtd = (pd.to_numeric(df["rdtd"], errors="coerce") if args.check_rdtd
            else pd.Series([None] * len(df)))
    prevs, status = [], Counter()
    # Record every check's count, including zeros, so the report shows what ran.
    status["no_predecessor_in_dump"] = 0
    status["predecessor_disagrees_with_extractor"] = 0
    if args.check_rdtd:
        status["rdtd_disagrees_with_predecessor"] = 0
    for pkg, ver, lab, rd in zip(df["package"], df["version"], df["release_type"], rdtd):
        prev = prev_of.get((pkg, ver))
        if prev is None:
            status["no_predecessor_in_dump"] += 1
            prevs.append(None)
        elif naive_bump(parse_version(prev), parse_version(ver)) != lab:
            status["predecessor_disagrees_with_extractor"] += 1
            prevs.append(None)
        elif args.check_rdtd and not (abs(gap_days(pkg, prev, ver) - rd) <= 1e-3):
            # The extractor's rdtd (days since the predecessor it diffed
            # against) must equal the gap to the predecessor we record; a
            # mismatch means the features describe a different pair.
            status["rdtd_disagrees_with_predecessor"] += 1
            prevs.append(None)
        else:
            status["kept"] += 1
            prevs.append(prev)
    df["prev_version"] = prevs
    df = df[df["prev_version"].notna()].copy()
    df["release_type_naive"] = df["release_type"]
    df["release_type"] = [cargo_bump(parse_version(p), parse_version(c))
                          for p, c in zip(df["prev_version"], df["version"])]
    status["cargo_unlabelable"] = int(df["release_type"].isna().sum())
    df = df[df["release_type"].notna()].reset_index(drop=True)

    # created_at = publish time of `version`; fill gaps (e.g. rows merged from
    # the backport re-extraction) from the same dump, in the file's ISO format.
    dump_ts = pd.to_datetime(pd.Series([published.get(k) for k in zip(df["package"], df["version"])],
                                       index=df.index), utc=True, errors="coerce")
    have = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    both = have.notna() & dump_ts.notna()
    ts_agree = int(((have[both] - dump_ts[both]).abs() < pd.Timedelta(seconds=1)).sum())
    fill = have.isna() & dump_ts.notna()
    df["created_at"] = df["created_at"].astype(object)   # may load as all-NaN float
    df.loc[fill, "created_at"] = dump_ts[fill].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    status["created_at_filled_from_dump"] = int(fill.sum())
    status["created_at_existing_agree_with_dump"] = f"{ts_agree}/{int(both.sum())}"
    status["created_at_still_missing"] = int(df["created_at"].isna().sum())

    # Snapshot cut. Each crate's release list was read when it was extracted
    # (mid-September 2026 for almost all crates), so later releases are only
    # present for crates re-extracted afterwards. Cutting every crate at the
    # same instant makes the file a uniform snapshot.
    if args.max_date:
        ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        keep = ts < pd.Timestamp(args.max_date, tz="UTC")
        status["dropped_after_max_date"] = int((~keep).sum())
        df = df[keep].reset_index(drop=True)

    # Per-crate / per-family caps (scripts/cap_dataset.py), re-applied to the
    # final file so they hold for every crate, including crates merged in by
    # the backport re-extraction (which were extracted uncapped).
    if args.cap_per_crate or args.cap_per_family:
        before = len(df)
        df = cap(df, args.cap_per_crate, args.cap_per_family, args.cap_seed)
        df = df.sort_values(["package", "created_at"]).reset_index(drop=True)
        status["dropped_by_cap"] = before - len(df)

    if args.repos:
        rm = pd.read_csv(args.repos, keep_default_na=False, na_values=[""])
        key = {p: repo_key(u) for p, u in zip(rm["package"], rm["repository"])}
        df["repository"] = [key.get(p) or "" for p in df["package"]]
        status["rows_with_repository"] = int((df["repository"] != "").sum())
        status["distinct_repositories"] = int(df.loc[df["repository"] != "", "repository"].nunique())

    flips = df[df["release_type"] != df["release_type_naive"]]
    cases = Counter(flip_case(parse_version(p), parse_version(c))
                    for p, c in zip(flips["prev_version"], flips["version"]))
    crosstab = (df.groupby(["release_type_naive", "release_type"]).size()
                .unstack(fill_value=0).to_dict())
    dist = df["release_type"].value_counts()
    naive_dist = df["release_type_naive"].value_counts()
    report = {
        "input_rows": n_in, "input_rows_added_from_extra": n_added, "input_crates": crates_in,
        "checks": {"predecessor_reproduces_naive_label": True,
                   "rdtd_matches_predecessor_gap": bool(args.check_rdtd),
                   "rdtd_tolerance_days": 1e-3 if args.check_rdtd else None},
        "status": dict(status),
        "output_rows": len(df), "output_crates": int(df["package"].nunique()),
        "cargo_label_pct": {k: round(100 * dist.get(k, 0) / len(df), 2) for k in ("major", "minor", "patch")},
        "naive_label_pct": {k: round(100 * naive_dist.get(k, 0) / len(df), 2) for k in ("major", "minor", "patch")},
        "flips": len(flips), "flips_pct": round(100 * len(flips) / len(df), 2),
        "flip_cases": {k: {"n": v, "pct_of_flips": round(100 * v / max(1, len(flips)), 1)}
                       for k, v in cases.most_common()},
        "crosstab_naive_rows_cargo_cols": crosstab,
    }
    cols = ["package", "prev_version", "version"] + [c for c in df.columns
                                                     if c not in ("package", "prev_version", "version")]
    df = df[cols]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False, compression="infer")
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
