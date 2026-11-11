#!/usr/bin/env python3
"""Fill the 11 GitHub features of the headline file.

Uses enrich_github.py's crawl, cache and feature code unchanged: commit
features from a clone of each crate's repository, issue/PR features from its
issue list (REST, with GraphQL for repositories too large for REST paging).

Window: (publish time of prev_version, publish time of version], i.e. the
pair whose code features were extracted.

Rows of the crates in --crates (a file of names, or ALL) are filled. Rows
whose crate has no GitHub repository, or whose repository can no longer be
cloned, keep 0; the report counts them.

Usage:
    GITHUB_TOKEN=... ENRICH_CLONE_FILTER=tree:0 python3 scripts/fill_github_features.py \\
        data/cratesbump.csv.gz <dump_dir> --crates ALL \\
        --report results/github_fill_report.json
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_github as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="headline file; rewritten in place unless --out")
    ap.add_argument("dump_dir")
    ap.add_argument("--crates", required=True,
                    help="file of crate names to fill, or ALL for every crate in the file")
    ap.add_argument("--repos", default="results/crates_all_repos.csv")
    ap.add_argument("--cache-dir", default="enrich_cache")
    ap.add_argument("--work-dir", default="enrich_work")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default="results/github_fill_report.json")
    args = ap.parse_args()

    tokens = E.Tokens(os.environ.get("GITHUB_TOKEN"))
    for sub in ("commits", "issues"):
        os.makedirs(os.path.join(args.cache_dir, sub), exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    df = pd.read_csv(args.csv, keep_default_na=False, na_values=[""])
    if args.crates == "ALL":
        targets = set(df["package"])
    else:
        targets = set(open(args.crates).read().split()) & set(df["package"])
    rows = df.index[df["package"].isin(targets)]
    E.log(f"{len(rows):,} rows from {len(targets):,} crates to back-fill")

    # Publish times of every version of the target crates (for prev_version).
    data = os.path.join(args.dump_dir, "data")
    csv.field_size_limit(1 << 30)
    id2name = {}
    with open(os.path.join(data, "crates.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["name"] in targets:
                id2name[r["id"]] = r["name"]
    published = {}
    with open(os.path.join(data, "versions.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            name = id2name.get(r["crate_id"])
            if name:
                published[(name, r["num"])] = r["created_at"]

    repo_map = pd.read_csv(args.repos, keep_default_na=False, na_values=[""])
    pkg2repo = {p: E.owner_repo(str(u)) for p, u in zip(repo_map["package"], repo_map["repository"])}
    by_repo = defaultdict(list)
    no_repo = 0
    for i in rows:
        orp = pkg2repo.get(df.at[i, "package"])
        if orp:
            by_repo[orp].append(i)
        else:
            no_repo += 1
    E.log(f"{len(by_repo):,} GitHub repos; {no_repo:,} rows have no GitHub repo")
    # Small repositories' issue lists in bulk (one GraphQL query per 25 repos);
    # the rest go through the per-repo REST crawl below.
    c, nf, left = E.issues_batch_prefetch(sorted(by_repo), tokens, args.cache_dir)
    E.log(f"GraphQL prefetch: {c:,} issue lists cached, {nf:,} not found there, "
          f"{left:,} left for the per-repo crawl")

    def work(orp):
        idxs = by_repo[orp]
        wins = {}
        for i in idxs:
            pkg = df.at[i, "package"]
            wins[i] = (published.get((pkg, df.at[i, "prev_version"])), df.at[i, "created_at"])
        until = max((w[1] for w in wins.values() if w[1]), default=None)
        commits = E.repo_commits(orp[0], orp[1], tokens, args.cache_dir, args.work_dir)
        issues = E.repo_issues(orp[0], orp[1], tokens, args.cache_dir, until)
        return orp, len(idxs), bool(commits), bool(issues), [
            (i, E.window_features(commits, issues, E.parse_iso(tp), E.parse_iso(tc)))
            for i, (tp, tc) in wins.items()]

    for c in E.GH_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    filled = repos_no_commits = repos_no_issues = missing_prev = rows_no_commits = 0
    n = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(work, orp) for orp in by_repo]):
            orp, n_rows, got_c, got_i, out = fut.result()
            repos_no_commits += not got_c
            rows_no_commits += 0 if got_c else n_rows
            repos_no_issues += not got_i
            for i, feats in out:
                for k, v in feats.items():
                    df.at[i, k] = v
                filled += 1
            n += 1
            if n % 50 == 0 or n == len(by_repo):
                E.log(f"  [{n}/{len(by_repo)}] repos; {filled:,} rows filled")
    for i in rows:
        if published.get((df.at[i, "package"], df.at[i, "prev_version"])) is None:
            missing_prev += 1

    gh = df.loc[rows, E.GH_FEATURES]
    report = {
        "rows_targeted": int(len(rows)), "crates": len(targets), "github_repos": len(by_repo),
        "rows_without_github_repo": no_repo, "rows_computed": filled,
        "rows_in_repos_without_commits": rows_no_commits,
        "repos_without_commits": repos_no_commits, "repos_without_issues": repos_no_issues,
        "rows_missing_prev_timestamp": missing_prev,
        "pct_targeted_rows_nonzero_after": round(100 * float((gh != 0).any(axis=1).mean()), 2),
        "pct_all_rows_nonzero_after": round(100 * float((df[E.GH_FEATURES] != 0).any(axis=1).mean()), 2),
        "window": "(publish time of prev_version, publish time of version]",
    }
    out = args.out or args.csv
    df.to_csv(out, index=False, compression="infer")
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=2)
    print(json.dumps(report, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
