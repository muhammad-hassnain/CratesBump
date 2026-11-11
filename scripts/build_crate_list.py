#!/usr/bin/env python3
"""Crate list and crate->repository map for the full-ecosystem snapshot.

Selects every crate in a crates.io dump with at least two non-yanked,
parseable versions published before MAX_DATE: every crate that can form a
release pair. No stability (>= 1.0) filter: Cargo's rule treats 0.y.z
releases as versioned too, so 0.x-only crates (libc, log, futures, ...) are
in scope.

Writes the crate list (one name per line) and package,repository (the dump's
repository field, empty if none) for those crates; build_snapshot_manifest.py
then lists their versions for the extractor, and the GitHub phase reads the map.

    python3 scripts/build_crate_list.py <dump_dir> 2026-09-14 \\
        results/crates_all.txt results/crates_all_repos.csv
"""
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_manifest import parse_semver, truthy  # noqa: E402


def main() -> int:
    dump_dir, max_date, out_list, out_repos = sys.argv[1:5]
    data = os.path.join(dump_dir, "data")
    csv.field_size_limit(1 << 30)
    id2name, id2repo = {}, {}
    with open(os.path.join(data, "crates.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            id2name[r["id"]] = r["name"]
            id2repo[r["id"]] = (r.get("repository") or "").strip()
    n_versions = Counter()
    with open(os.path.join(data, "versions.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (truthy(r.get("yanked", "")) or parse_semver(r["num"]) is None
                    or r["created_at"] >= max_date):      # dump timestamps sort as text
                continue
            n_versions[r["crate_id"]] += 1
    keep = sorted((cid for cid, n in n_versions.items() if n >= 2), key=lambda c: id2name[c])
    with open(out_list, "w") as fh:
        fh.writelines(id2name[c] + "\n" for c in keep)
    with open(out_repos, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["package", "repository"])
        for c in keep:
            w.writerow([id2name[c], id2repo.get(c, "")])
    with_repo = sum(1 for c in keep if id2repo.get(c))
    print(f"{len(id2name):,} crates in the dump; {len(n_versions):,} with >=1 and "
          f"{len(keep):,} with >=2 eligible versions before {max_date}; "
          f"{with_repo:,} of those list a repository -> {out_list}, {out_repos}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
