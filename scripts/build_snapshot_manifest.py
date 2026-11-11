#!/usr/bin/env python3
"""Version manifest for a dataset snapshot.

Writes package,version,created_at for every non-yanked, parseable version of
the crates in CRATE_LIST that was published before MAX_DATE, read from a
crates.io dump. The batch extractor (--version-manifest) then pairs
versions from exactly this list instead of the live crates.io API, so the
extraction matches label_pairs_cargo.py (same dump, same non-yanked rule)
and ignores releases after the snapshot.

    python3 scripts/build_snapshot_manifest.py <dump_dir> results/crates_all.txt \\
        2026-09-14 .dump_cache/version_manifest_all_cut20260914.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import parse_semver, to_rfc3339, truthy  # noqa: E402


def main() -> int:
    dump_dir, crate_list, max_date, out = sys.argv[1:5]
    data = os.path.join(dump_dir, "data")
    csv.field_size_limit(1 << 30)
    want = set(open(crate_list).read().split())
    id2name = {}
    with open(os.path.join(data, "crates.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["name"] in want:
                id2name[r["id"]] = r["name"]
    n, crates = 0, set()
    with open(os.path.join(data, "versions.csv"), newline="", encoding="utf-8") as fh, \
            open(out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["package", "version", "created_at"])
        for r in csv.DictReader(fh):
            name = id2name.get(r["crate_id"])
            if not name or truthy(r.get("yanked", "")) or parse_semver(r["num"]) is None:
                continue
            if r["created_at"] >= max_date:      # dump timestamps sort as text
                continue
            w.writerow([name, r["num"], to_rfc3339(r["created_at"])])
            n += 1
            crates.add(name)
    print(f"{n:,} versions of {len(crates):,} crates ({len(want):,} listed, "
          f"{len(want) - len(id2name):,} not in the dump) -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
