#!/usr/bin/env python3
"""Which in-scope crates the extractor produced pairs for, and why not.

Compares the snapshot version list (scripts/build_snapshot_manifest.py) with
the per-crate extractor output (crate_csvs_full/): crates whose versions form
no pair under plan_pairs (scripts/semver_order.py), crates extracted, and
crates that should have pairs but produced no file. For the last group it
checks (HTTP HEAD, up to the first ten versions) which tarball the crates.io
CDN no longer serves.

    python3 scripts/extraction_coverage.py .dump_cache/version_manifest_all_cut20260914.csv \\
        crate_csvs_full/ --out results/extraction_report.json
"""
import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semver_order import plan_prev  # noqa: E402


def cdn_status(name, version):
    url = f"https://static.crates.io/crates/{name}/{name}-{version}.crate"
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "semver-ml extraction check"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("crate_dir")
    ap.add_argument("--out", default="results/extraction_report.json")
    args = ap.parse_args()
    vs = defaultdict(list)
    for r in csv.DictReader(open(args.manifest)):
        vs[r["package"]].append((r["created_at"], r["version"]))
    done = {f[:-4] for f in os.listdir(args.crate_dir) if f.endswith(".csv")}
    no_pair, missing = [], []
    for p, v in vs.items():
        if not plan_prev([x for _, x in sorted(v)]):
            no_pair.append(p)
        elif p not in done:
            missing.append((p, [x for _, x in sorted(v)]))
    status = {}
    for p, versions in sorted(missing):
        # First version of the crate that the CDN does not serve, if any.
        status[p] = {"first_unavailable_version": None, "http_status": 200}
        for ver in versions[:10]:
            code = cdn_status(p, ver)
            if code != 200:
                status[p] = {"first_unavailable_version": ver, "http_status": code}
                break
    out = {
        "crates_in_scope": len(vs),
        "crates_with_no_pair_under_plan_pairs": len(no_pair),
        "crates_with_pairs": len(vs) - len(no_pair),
        "crates_extracted": len(done & set(vs)),
        "crates_with_pairs_not_extracted": len(missing),
        "not_extracted_by_cdn_status": dict(Counter(str(s["http_status"]) for s in status.values())),
        "not_extracted": status,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "not_extracted"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
