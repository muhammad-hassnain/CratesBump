#!/usr/bin/env python3
"""Downstream exposure of pre-1.0 Cargo-major bumps.

Question: when a crate ships a release that reads as "minor" under the naive
three-part reading but is "major" (breaking) under Cargo's pre-1.0 rule
(0.y.z -> 0.(y+1).z), which crates that depended on it at that moment had a
declared version range that would let `cargo update` move them onto the
breaking release?

For each such problem pair (prev -> curr of crate P) and each other crate D
that declares a normal dependency on P, we take D's latest non-yanked version
published before curr and classify its `req` on P:

  exposed    req admits prev AND curr  -> would silently resolve to curr
  protected  req admits prev, not curr -> Cargo's range stops at the boundary
  other      req admits neither prev nor curr (D tracks a different line of P)

crates.io dump schema (verified against serde_json 1.0.100 -> serde):
  dependencies.version_id = the DEPENDENT's version (the crate declaring the dep)
  dependencies.crate_id   = the crate being DEPENDED ON

An earlier version of this script had these two columns swapped, so it
measured whether a crate's own requirements on its dependencies happened to
include its new version number; its 54%/60% headline was meaningless.

Also reports the shape distribution of every normal-kind `req` string in the
dump (independent of the column semantics).

Usage:
    python3 scripts/downstream_impact.py <dump_dir> \\
        data/cratesbump.csv.gz \\
        --out results/downstream_impact.json
    python3 scripts/downstream_impact.py --self-test
"""
import argparse
import bisect
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semver_order import plan_prev  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relabel_cargo import cargo_bump  # noqa: E402

INF = (float("inf"),) * 3


def parse_version(s: str):
    s = (s or "").strip().lstrip("v")
    s = re.split(r"[-+]", s, maxsplit=1)[0]
    p = s.split(".")
    if len(p) < 3:
        return None
    try:
        return int(p[0]), int(p[1]), int(p[2])
    except ValueError:
        return None


def naive_bump(prev, curr):
    if curr[0] > prev[0]:
        return "major"
    if curr[0] == prev[0] and curr[1] > prev[1]:
        return "minor"
    if curr[:2] == prev[:2] and curr[2] > prev[2]:
        return "patch"
    return None


# ---------------------------------------------------------------------------
# Cargo VersionReq matcher (semver crate semantics, pre-release tags ignored).
# Each comparator becomes a half-open interval [lo, hi) over (major, minor, patch).
# ---------------------------------------------------------------------------
_COMP_RE = re.compile(
    r"^\s*(=|>=|<=|>|<|~|\^)?\s*v?"
    r"(\d+|\*|x|X)(?:\.(\d+|\*|x|X))?(?:\.(\d+|\*|x|X))?"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\s*$")
_WILD = {"*", "x", "X"}


def _interval(op, parts):
    """Return [lo, hi) for one comparator, or None if unparseable."""
    nums = []
    for p in parts:
        if p is None or p in _WILD:
            break
        nums.append(int(p))
    wildcard = any(p in _WILD for p in parts if p is not None)
    n = len(nums)
    if n == 0:
        return (0, 0, 0), INF                                     # *, x
    if wildcard:                                                # I.*, I.J.*
        if n == 1:
            return (nums[0], 0, 0), (nums[0] + 1, 0, 0)
        return (nums[0], nums[1], 0), (nums[0], nums[1] + 1, 0)
    I = nums[0]
    J = nums[1] if n > 1 else 0
    K = nums[2] if n > 2 else 0
    op = op or "^"
    if op == "=":
        if n == 3:
            return (I, J, K), (I, J, K + 1)
        if n == 2:
            return (I, J, 0), (I, J + 1, 0)
        return (I, 0, 0), (I + 1, 0, 0)
    if op == ">":
        lo = (I, J, K + 1) if n == 3 else ((I, J + 1, 0) if n == 2 else (I + 1, 0, 0))
        return lo, INF
    if op == ">=":
        return (I, J, K), INF
    if op == "<":
        return (0, 0, 0), (I, J, K)
    if op == "<=":
        hi = (I, J, K + 1) if n == 3 else ((I, J + 1, 0) if n == 2 else (I + 1, 0, 0))
        return (0, 0, 0), hi
    if op == "~":
        if n == 1:
            return (I, 0, 0), (I + 1, 0, 0)
        return (I, J, K), (I, J + 1, 0)
    # caret
    if I > 0:
        return (I, J, K), (I + 1, 0, 0)
    if n == 1:                                                    # ^0
        return (0, 0, 0), (1, 0, 0)
    if J > 0:                                                     # ^0.J[.K]
        return (0, J, K), (0, J + 1, 0)
    if n == 2:                                                    # ^0.0
        return (0, 0, 0), (0, 1, 0)
    return (0, 0, K), (0, 0, K + 1)                               # ^0.0.K


def parse_req(req: str):
    """List of [lo, hi) intervals (AND-ed), or None if any part is unparseable."""
    req = (req or "").strip()
    if req in ("", "*"):
        return [((0, 0, 0), INF)]
    out = []
    for part in req.split(","):
        m = _COMP_RE.match(part)
        if not m:
            return None
        out.append(_interval(m.group(1), m.groups()[1:]))
    return out


def req_matches(intervals, ver) -> bool:
    return all(lo <= ver < hi for lo, hi in intervals)


def classify_shape(req: str) -> str:
    r = (req or "").strip()
    if r in ("", "*"):
        return "wildcard_any"
    if "," in r:
        return "compound"
    if re.match(r"^=\s*\d+\.\d+\.\d+", r):
        return "exact_pin"
    if re.search(r"\.[*xX]", r):
        return "wildcard_minor_or_patch"
    if r.startswith("~"):
        return "tilde"
    if re.match(r"^(>=?|<=?)", r):
        return "comparator"
    if re.match(r"^\^?\s*\d+\.\d+\.\d+", r):
        return "caret_full"
    if re.match(r"^\^?\s*\d+(\.\d+)?\s*$", r):
        return "caret_partial"
    return "other"


def self_test() -> int:
    cases = [
        ("^1.2.3", (1, 3, 0), True), ("^1.2.3", (2, 0, 0), False),
        ("1.2.3", (1, 9, 9), True), ("^0.4.5", (0, 4, 9), True),
        ("^0.4.5", (0, 5, 0), False), ("0.4", (0, 4, 7), True),
        ("0.4", (0, 5, 0), False), ("^0.0.5", (0, 0, 5), True),
        ("^0.0.5", (0, 0, 6), False), ("^0.0", (0, 0, 9), True),
        ("^0.0", (0, 1, 0), False), ("^0", (0, 9, 0), True),
        ("0", (1, 0, 0), False), ("=0.4.5", (0, 4, 5), True),
        ("=0.4.5", (0, 4, 6), False), ("~0.4.5", (0, 4, 9), True),
        ("~0.4.5", (0, 5, 0), False), ("~1", (1, 9, 0), True),
        ("0.*", (0, 7, 0), True), ("0.4.*", (0, 5, 0), False),
        (">=0.4", (0, 9, 0), True), (">=0.4, <0.6", (0, 5, 1), True),
        (">=0.4, <0.5", (0, 5, 0), False), ("*", (9, 9, 9), True),
        ("<=0.4", (0, 4, 9), True), ("<=0.4", (0, 5, 0), False),
        (">0.4", (0, 4, 9), False), (">0.4", (0, 5, 0), True),
    ]
    bad = 0
    for req, ver, exp in cases:
        got = req_matches(parse_req(req), ver)
        if got != exp:
            bad += 1
            print(f"FAIL {req!r} {ver} got={got} expected={exp}")
    print(f"{len(cases) - bad}/{len(cases)} pass")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir", nargs="?")
    ap.add_argument("cargo_csv", nargs="?")
    ap.add_argument("--out", default="results/downstream_impact.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    data = os.path.join(args.dump_dir, "data")
    csv.field_size_limit(1 << 30)

    head = pd.read_csv(args.cargo_csv, nrows=0).columns
    cols = ["package", "version", "release_type"] + (["prev_version"] if "prev_version" in head else [])
    df = pd.read_csv(args.cargo_csv, keep_default_na=False, na_values=[""], usecols=cols)
    shipped = {(r.package, r.version): r.release_type for r in df.itertuples()}
    # The file's own prev_version is the pair the label describes; older files
    # without it fall back to the extractor's rule (scripts/semver_order.py).
    file_prev = ({(p, v): pv for p, v, pv in zip(df["package"], df["version"], df["prev_version"])}
                 if "prev_version" in cols else None)
    dataset_crates = set(df["package"])
    print(f"dataset: {len(df):,} pairs over {len(dataset_crates):,} crates", file=sys.stderr)

    print("Reading crates.csv ...", file=sys.stderr)
    id2name = {}
    with open(os.path.join(data, "crates.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            id2name[r["id"]] = r["name"]
    name2id = {v: k for k, v in id2name.items()}

    print("Reading versions.csv ...", file=sys.stderr)
    vinfo = {}                                   # version_id -> (crate_id, num, created_at)
    by_crate = defaultdict(list)                 # dataset crate name -> [(created_at, num)]
    with open(os.path.join(data, "versions.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("yanked", "f") in ("t", "true", "True", "1"):
                continue
            vinfo[r["id"]] = (r["crate_id"], r["num"], r["created_at"])
            name = id2name.get(r["crate_id"])
            if name in dataset_crates:
                by_crate[name].append((r["created_at"], r["num"]))

    # Take each dataset pair's predecessor (from the file, else rebuilt with the
    # extractor's rule) and keep pairs that are naive-minor, Cargo-major, pre-1.0.
    problems = []
    label_check = Counter()
    for name, vs in by_crate.items():
        vs.sort()
        created = {n: t for t, n in vs}
        prev_of = (None if file_prev is not None else plan_prev([n for _, n in vs]))
        for _, num in vs:
            lab = shipped.get((name, num))
            if lab is None:
                continue
            prev = file_prev.get((name, num)) if file_prev is not None else prev_of.get(num)
            if not isinstance(prev, str) or prev not in created:
                label_check["predecessor_missing"] += 1
                continue
            pv, cv = parse_version(prev), parse_version(num)
            cargo = cargo_bump(pv, cv)
            label_check["match" if cargo == lab else "mismatch"] += 1
            if lab == "major" and naive_bump(pv, cv) == "minor" and cv[0] == 0:
                problems.append({"package": name, "prev": prev, "curr": num,
                                 "prev_v": pv, "curr_v": cv, "curr_time": created[num]})
    print(f"shipped-label vs recomputed Cargo label: {dict(label_check)}", file=sys.stderr)
    print(f"naive-minor, Cargo-major pre-1.0 pairs: {len(problems):,}", file=sys.stderr)

    targets = {name2id[p["package"]] for p in problems if p["package"] in name2id}

    print("Scanning dependencies.csv ...", file=sys.stderr)
    shape_counts = Counter()
    # target crate id -> dependent crate id -> [(created_at, req)]
    deps = defaultdict(lambda: defaultdict(list))
    n_rows = 0
    with open(os.path.join(data, "dependencies.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n_rows += 1
            if r.get("kind") != "0":
                continue
            shape_counts[classify_shape(r.get("req", ""))] += 1
            if r["crate_id"] not in targets:
                continue
            dv = vinfo.get(r["version_id"])          # dependent's version (non-yanked only)
            if dv is None:
                continue
            dep_crate, _, dep_time = dv
            if dep_crate == r["crate_id"]:
                continue
            deps[r["crate_id"]][dep_crate].append((dep_time, r.get("req", "")))
    print(f"  dep rows scanned: {n_rows:,}", file=sys.stderr)

    # Publish times of every non-yanked version of every dependent crate, so a
    # dependent is judged by its latest version before the release, not by its
    # latest version that still declared the dependency.
    dep_crates = {d for m in deps.values() for d in m}
    all_times = defaultdict(list)
    for crate_id, _, t in vinfo.values():
        if crate_id in dep_crates:
            all_times[crate_id].append(t)
    for v in all_times.values():
        v.sort()

    totals = Counter()
    exposed_shapes = Counter()
    exposed_reqs = Counter()
    per_pair = []
    unparsed = 0
    for p in problems:
        tid = name2id.get(p["package"])
        counts = Counter()
        for dep_crate, rows in deps.get(tid, {}).items():
            times = all_times[dep_crate]
            k = bisect.bisect_left(times, p["curr_time"])
            if k == 0:
                continue                     # dependent did not exist yet
            latest_t = times[k - 1]          # its latest version before the release
            reqs = [req for t, req in rows if t == latest_t]
            if not reqs:
                counts["dropped"] += 1       # that version no longer depends on it
                continue
            ivs = [parse_req(q) for q in reqs]
            if any(iv is None for iv in ivs):
                unparsed += 1
                counts["unparsed"] += 1
                continue
            a_prev = any(req_matches(iv, p["prev_v"]) for iv in ivs)
            a_curr = any(req_matches(iv, p["curr_v"]) for iv in ivs)
            if a_prev and a_curr:
                counts["exposed"] += 1
                exposed_shapes.update({classify_shape(q) for q in reqs})
                exposed_reqs.update({q.strip() for q in reqs})
            elif a_prev:
                counts["protected"] += 1
            else:
                counts["other"] += 1
        totals.update(counts)
        per_pair.append({"package": p["package"], "prev": p["prev"], "curr": p["curr"],
                         **{k: counts[k] for k in ("exposed", "protected", "other", "unparsed", "dropped")}})

    with_dependents = [x for x in per_pair if x["exposed"] + x["protected"] > 0]
    with_exposed = [x for x in per_pair if x["exposed"] > 0]
    tracking = totals["exposed"] + totals["protected"]
    n_shape = sum(shape_counts.values())
    out = {
        "definition": ("problem pair = shipped Cargo label major, naive label minor, "
                       "curr < 1.0.0. For each other crate that ever declared a normal dependency "
                       "on the package, take its latest non-yanked version published before "
                       "curr; dropped = that version no longer declares it; exposed = its "
                       "req admits prev and curr; protected = admits prev, not curr."),
        "n_problem_pairs": len(problems),
        "label_recompute_check": dict(label_check),
        "pairs_with_tracking_dependents": len(with_dependents),
        "pairs_with_exposed_dependent": len(with_exposed),
        "pct_pairs_with_exposed_among_all": round(100 * len(with_exposed) / max(1, len(problems)), 2),
        "pct_pairs_with_exposed_among_tracked": round(100 * len(with_exposed) / max(1, len(with_dependents)), 2),
        "dependent_relations": {**{k: totals[k] for k in ("exposed", "protected", "other", "unparsed", "dropped")},
                                "pct_exposed_of_tracking": round(100 * totals["exposed"] / max(1, tracking), 2)},
        "exposed_relations_by_req_shape": dict(exposed_shapes.most_common()),
        "exposed_top_reqs": dict(exposed_reqs.most_common(40)),
        "top_30_by_exposed": sorted(per_pair, key=lambda x: -x["exposed"])[:30],
        "req_population": {
            "total_normal_dep_rows": n_shape,
            "by_shape": {k: v for k, v in shape_counts.most_common()},
            "pct_by_shape": {k: round(100 * v / n_shape, 2) for k, v in shape_counts.most_common()},
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "top_30_by_exposed"}, indent=2),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
