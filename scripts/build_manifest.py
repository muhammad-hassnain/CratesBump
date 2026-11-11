#!/usr/bin/env python3
"""Build the SemVer-compliance crate list and non-yanked version manifest from a
crates.io database dump.

Inputs (from the dump's `data/` dir):
  - crates.csv    : crate id -> name
  - versions.csv  : per-version crate_id, num, yanked, created_at

A crate is *SemVer-compliant* if it has >= 1 non-yanked, stable (no pre-release)
release with major >= 1 (i.e. a real >= 1.0.0 version). For every compliant
crate we emit all of its non-yanked, parseable versions to the manifest; the
Rust extractor forms consecutive version pairs from these and bypasses the
crates.io API entirely (CDN-only, no rate limit).

Usage:
  python3 scripts/build_manifest.py <dump_dir> \
      [--out-manifest version_manifest.csv] \
      [--out-crates results/compliant_crates_all.txt]

<dump_dir> is the extracted dump root that contains `data/crates.csv` and
`data/versions.csv` (the dump tars to `crates.io-db-dump-YYYY-MM-DD-.../`).
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict


def to_rfc3339(ts: str) -> str:
    """Convert a Postgres dump timestamp ('2026-03-30 23:43:39.025802+00') to
    strict RFC3339 ('2026-03-30T23:43:39.025802+00:00') so the Rust extractor's
    `DateTime::parse_from_rfc3339` accepts it (else the rdtd feature is zeroed)."""
    ts = ts.strip()
    if not ts:
        return ts
    ts = ts.replace(" ", "T", 1)
    m = re.search(r"([+-])(\d{2})(\d{2})?$", ts)
    if m and ts[m.start():].count(":") == 0:
        mins = m.group(3) or "00"
        ts = ts[: m.start()] + m.group(1) + m.group(2) + ":" + mins
    return ts


def parse_semver(num: str):
    """Return (major, minor, patch, is_prerelease) or None if unparseable.

    Build metadata (`+meta`) does NOT make a version a pre-release — it has no
    effect on precedence or stability (semver spec). Only a `-` suffix does.
    """
    s = num.strip().lstrip("v")
    s = s.split("+", 1)[0]        # strip build metadata first
    pre = "-" in s               # pre-release iff a hyphen suffix remains
    core = s.split("-", 1)[0]
    parts = core.split(".")
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), pre
    except ValueError:
        return None


def truthy(v: str) -> bool:
    return v.strip().lower() in ("t", "true", "1", "yes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", help="extracted dump dir containing data/crates.csv and data/versions.csv")
    ap.add_argument("--out-manifest", default="version_manifest.csv")
    ap.add_argument("--out-crates", default="results/compliant_crates_all.txt")
    ap.add_argument("--out-repos", default="results/compliant_repos.csv",
                    help="package,repository map for compliant crates (feeds the GitHub phase)")
    args = ap.parse_args()

    data_dir = os.path.join(args.dump_dir, "data")
    crates_csv = os.path.join(data_dir, "crates.csv")
    versions_csv = os.path.join(data_dir, "versions.csv")
    for p in (crates_csv, versions_csv):
        if not os.path.isfile(p):
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 1

    # crate id -> name, id -> repository
    csv.field_size_limit(1 << 30)
    id2name = {}
    id2repo = {}
    with open(crates_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            id2name[r["id"]] = r["name"]
            id2repo[r["id"]] = (r.get("repository") or "").strip()
    print(f"crates: {len(id2name):,}")

    # collect non-yanked parseable versions per crate; track compliance
    versions = defaultdict(list)          # crate_id -> [(num, created_at)]
    compliant_ids = set()                 # crate_id with >=1 stable >=1.0.0
    n_rows = n_nonyanked = 0
    with open(versions_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n_rows += 1
            if truthy(r.get("yanked", "f")):
                continue
            parsed = parse_semver(r["num"])
            if parsed is None:
                continue
            n_nonyanked += 1
            cid = r["crate_id"]
            versions[cid].append((r["num"], r["created_at"]))
            major, _, _, is_pre = parsed
            if major >= 1 and not is_pre:
                compliant_ids.add(cid)

    print(f"version rows: {n_rows:,}  non-yanked parseable: {n_nonyanked:,}")
    print(f"compliant crates (>=1 stable >=1.0.0): {len(compliant_ids):,}")

    # emit manifest for compliant crates only
    os.makedirs(os.path.dirname(args.out_crates) or ".", exist_ok=True)
    n_manifest = 0
    with open(args.out_manifest, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["package", "version", "created_at"])
        names = []
        for cid in compliant_ids:
            name = id2name.get(cid)
            if not name:
                continue
            names.append(name)
            for num, created in versions[cid]:
                w.writerow([name, num, to_rfc3339(created)])
                n_manifest += 1
    with open(args.out_crates, "w", encoding="utf-8") as cf:
        for name in sorted(names):
            cf.write(name + "\n")

    # package,repository map for compliant crates (for the GitHub phase)
    n_repos = 0
    with open(args.out_repos, "w", newline="", encoding="utf-8") as rf:
        w = csv.writer(rf)
        w.writerow(["package", "repository"])
        for cid in compliant_ids:
            name = id2name.get(cid)
            if not name:
                continue
            w.writerow([name, id2repo.get(cid, "")])
            n_repos += 1

    print(f"manifest rows: {n_manifest:,} -> {args.out_manifest}")
    print(f"compliant crate list: {len(names):,} -> {args.out_crates}")
    print(f"compliant repos: {n_repos:,} -> {args.out_repos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
