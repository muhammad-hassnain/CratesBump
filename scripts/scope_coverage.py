#!/usr/bin/env python3
"""What share of crates.io the dataset's crate scope covers.

Scope = crates with at least two non-yanked, parseable releases published
before the snapshot (results/crates_all.txt, scripts/build_crate_list.py).
Reports the share of crates and of all-time downloads (crate_downloads.csv),
and how many in-scope crates never had a stable >= 1.0.0 release, i.e. the
crates that the earlier, 1.0-only release of the dataset left out.

    python3 scripts/scope_coverage.py <dump_dir> results/crates_all.txt \\
        --out results/scope_coverage.json
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import parse_semver, truthy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir")
    ap.add_argument("crate_list")
    ap.add_argument("--out", default="results/scope_coverage.json")
    args = ap.parse_args()
    data = os.path.join(args.dump_dir, "data")
    csv.field_size_limit(1 << 30)
    scope = set(open(args.crate_list).read().split())
    id2name = {}
    for r in csv.DictReader(open(os.path.join(data, "crates.csv"), newline="", encoding="utf-8")):
        id2name[r["id"]] = r["name"]
    dl = {}
    for r in csv.DictReader(open(os.path.join(data, "crate_downloads.csv"), newline="", encoding="utf-8")):
        dl[r["crate_id"]] = int(r.get("downloads") or 0)
    stable = set()
    for r in csv.DictReader(open(os.path.join(data, "versions.csv"), newline="", encoding="utf-8")):
        if truthy(r.get("yanked", "f")):
            continue
        p = parse_semver(r["num"])
        if p and p[0] >= 1 and not p[3]:
            stable.add(r["crate_id"])
    in_scope = {cid for cid, n in id2name.items() if n in scope}
    total_dl = sum(dl.values())
    scope_dl = sum(dl.get(c, 0) for c in in_scope)
    never_1 = in_scope - stable
    out = {
        "n_crates_total": len(id2name),
        "n_crates_in_scope": len(in_scope),
        "pct_crates_in_scope": round(100 * len(in_scope) / len(id2name), 2),
        "downloads_total": total_dl,
        "downloads_in_scope": scope_dl,
        "pct_downloads_in_scope": round(100 * scope_dl / total_dl, 2),
        "n_in_scope_never_stable_1_0": len(never_1),
        "pct_in_scope_never_stable_1_0": round(100 * len(never_1) / len(in_scope), 2),
        "downloads_in_scope_never_stable_1_0": sum(dl.get(c, 0) for c in never_1),
        "pct_downloads_in_scope_never_stable_1_0":
            round(100 * sum(dl.get(c, 0) for c in never_1) / total_dl, 2),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
