#!/usr/bin/env python3
"""Numbers in the paper that no other script writes, recomputed from files in
the repository: the GitHub-zero breakdown and the two known GitHub-step bugs
(Section 3.2), and the audit disclosures (Section 5.5).

    python3 scripts/paper_numbers.py                      # -> results/paper_numbers.json
    python3 scripts/paper_numbers.py --dump-dir <dump>    # also splits the audit's
                                                          # no-library pairs (dump: has_lib)

Inputs: data/cratesbump.csv.gz, results/crates_all_repos.csv,
results/github_fill_report.json, results/label_audit/{per_pair.jsonl,
hand_review.csv, summary.json, logs/}. Without --dump-dir, the binary/macro
split is carried over from an existing output file.

Some parts of the GitHub breakdown need the crawl caches (enrich_cache/),
which are not in the repository; the output says which values are exact and
which are approximations from the committed files.
"""
import argparse
import csv
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_github as E  # noqa: E402  (owner_repo, GH_FEATURES)

CHECKED = ("agree", "shipped_higher", "shipped_lower")


def pct(k, n, d=1):
    return round(100 * k / n, d) if n else None


def lenient_owner_repo(url):
    """GitHub owner/repo, accepting what owner_repo() misses: an upper-case
    host and doubled slashes."""
    if not isinstance(url, str) or "github.com" not in url.lower():
        return None
    u = re.sub(r"\.git$", "", url.strip().lower())
    u = re.sub(r"(?<!:)/{2,}", "/", u)
    m = re.search(r"github\.com[:/]+([^/]+)/([^/#?]+)", u)
    return (m.group(1), m.group(2)) if m else None


def github_numbers(args):
    gh = E.GH_FEATURES
    df = pd.read_csv(args.data, keep_default_na=False, na_values=[""],
                     usecols=["package", "created_at", "rdtd", "repository", *gh],
                     dtype={"package": str, "created_at": str, "repository": str})
    n = len(df)
    repos = pd.read_csv(args.repos, keep_default_na=False, na_values=[""], dtype=str)
    raw = dict(zip(repos["package"], repos["repository"]))
    orp = {p: E.owner_repo(u) if isinstance(u, str) else None for p, u in raw.items()}

    zero = (df[gh] == 0).all(axis=1)
    repo = df["repository"]
    no_repo = repo.isna()
    other_host = ~no_repo & ~repo.fillna("").str.startswith("github.com/")
    has_gh = df["package"].map(lambda p: orp.get(p) is not None)
    gh_no_repo = ~no_repo & ~other_host & ~has_gh       # GitHub URL fill_github_features skips
    fill = json.load(open(args.fill_report))

    # Known bug 1: owner_repo() misses an upper-case host or doubled slashes.
    bug_pk = sorted(p for p, u in raw.items()
                    if orp.get(p) is None and lenient_owner_repo(u) is not None)
    bug_rows = df["package"].isin(bug_pk)

    # Rows of GitHub repositories with all 11 features zero. The split between
    # "could not be cloned" and "nothing in the window" needs the per-repository
    # commit caches; github_fill_report.json has the rows of repositories
    # without commits (all of them, zero or not).
    gh_zero = int((zero & has_gh).sum())
    no_commit_rows = int(fill["rows_in_repos_without_commits"])
    rest = gh_zero - no_commit_rows

    # Approximation of the rest from the CSV alone: a zero row of a repository
    # whose clone has commits lies inside the span of that repository's windows
    # with commits (nothing happened: "quick re-release") or outside it (the
    # clone's history misses the release).
    d = df[has_gh].copy()
    d["key"] = d["package"].map(lambda p: "/".join(orp[p]).lower())
    d["t"] = pd.to_datetime(d["created_at"], utc=True)
    d["tp"] = d["t"] - pd.to_timedelta(d["rdtd"].astype(float), unit="D")
    act = d[d["tcm"] > 0].groupby("key").agg(first=("tp", "min"), last=("t", "max"))
    z = d[(d[gh] == 0).all(axis=1)].join(act, on="key")
    with_commits = z["first"].notna()
    inside = with_commits & (z["tp"] >= z["first"]) & (z["t"] <= z["last"])
    outside = with_commits & ~inside

    return {
        "rows": n,
        "all_github_features_zero": {"rows": int(zero.sum()), "pct": pct(zero.sum(), n),
                                     "nonzero_pct": pct(n - zero.sum(), n, 2)},
        "no_repository": {"rows": int(no_repo.sum()), "pct": pct(no_repo.sum(), n)},
        "other_host": {"rows": int(other_host.sum()), "pct": pct(other_host.sum(), n)},
        "github_url_without_owner_repo": {
            "rows": int(gh_no_repo.sum()),
            "note": "repository looks like GitHub but owner_repo() finds no owner/repo "
                    "(e.g. github.com/<user>); includes known bug 1"},
        "github_repo_all_zero": {"rows": gh_zero, "pct": pct(gh_zero, n)},
        "repo_not_cloned": {
            "rows": no_commit_rows, "pct": pct(no_commit_rows, n),
            "source": "github_fill_report.json rows_in_repos_without_commits "
                      f"({fill['repos_without_commits']:,} repositories; includes the few "
                      "rows of these repositories with issue/PR activity)"},
        "no_activity_in_window": {
            "rows": rest, "pct": pct(rest, n),
            "source": "github_repo_all_zero minus repo_not_cloned"},
        "no_activity_split_approx": {
            "note": "CSV-only approximation; the paper's split (1.4% / 1.5%) uses the "
                    "commit dates in enrich_cache/, which is not in the repository",
            "inside_commit_span_rows": int(inside.sum()),
            "inside_commit_span_pct": pct(inside.sum(), n),
            "inside_median_gap_hours": round(float(z.loc[inside, "rdtd"].median()) * 24, 1),
            "outside_commit_span_rows": int(outside.sum()),
            "outside_commit_span_pct": pct(outside.sum(), n)},
        "known_bug_unusual_github_url": {
            "rows": int(bug_rows.sum()), "crates": len(bug_pk), "packages": bug_pk,
            "rows_all_zero": int((bug_rows & zero).sum())},
        "known_bug_unchecked_git_log": {
            "repos": None,
            "note": "49 in the paper and docs: repositories with an empty cached history and no "
                    "logged clone error; needs enrich_cache/ and the crawl logs, which are not "
                    "in the repository"},
    }


FEATURE_CONFLICT = [
    # compile_error! guards that reject a feature combination
    re.compile(r"compile_error!?\(?.*(feature|exclusive|enabled|without)", re.I),
    re.compile(r"^error: .*(mutually exclusive|are exclusive|must be enabled|"
               r"without \w+ is not)", re.I | re.M),
    # the same item defined twice / conflicting impls: two exclusive features on
    re.compile(r"error\[E0(428|252)\]"),
    # nightly-only APIs behind a cargo feature
    re.compile(r"unresolved imports? `std::simd::"),
]


def audit_numbers(args, previous):
    A = args.audit_dir
    rows = [json.loads(l) for l in open(os.path.join(A, "per_pair.jsonl"))]
    a = pd.DataFrame(rows)
    hr = pd.read_csv(os.path.join(A, "hand_review.csv"), keep_default_na=False, dtype=str)
    summ = json.load(open(os.path.join(A, "summary.json")))
    ok = a["outcome"].isin(CHECKED)
    nolib = a["error_kind"].eq("no_lib_target")
    out = {"sampled": len(a), "checked": int(ok.sum()), "unchecked": int((~ok).sum()),
           "no_library_target": int(nolib.sum()),
           "no_longer_build_resolve_or_download": int((~ok & ~nolib).sum())}

    # 46 command-line-only vs 25 macro crates (the dump's versions.has_lib).
    if args.dump_dir:
        want = {(r["package"], v) for r in rows if r.get("error_kind") == "no_lib_target"
                for v in (r["prev"], r["curr"])}
        names = {p for p, _ in want}
        data = os.path.join(args.dump_dir, "data")
        csv.field_size_limit(1 << 30)
        ids = {}
        with open(os.path.join(data, "crates.csv"), newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["name"] in names:
                    ids[r["id"]] = r["name"]
        has_lib = {}
        with open(os.path.join(data, "versions.csv"), newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                p = ids.get(r["crate_id"])
                if p and (p, r["num"]) in want:
                    has_lib[(p, r["num"])] = r["has_lib"] == "t"
        macro = binary = 0
        for r in rows:
            if r.get("error_kind") != "no_lib_target":
                continue
            # A macro crate has a (proc-macro) library in both versions, which
            # cargo-semver-checks does not select; otherwise a version has no library.
            if has_lib.get((r["package"], r["prev"])) and has_lib.get((r["package"], r["curr"])):
                macro += 1
            else:
                binary += 1
        out["no_library_split"] = {"command_line_only": binary, "macro_crates": macro,
                                   "source": "crates.io dump versions.has_lib of both versions"}
    elif previous and "no_library_split" in previous.get("audit", {}):
        out["no_library_split"] = previous["audit"]["no_library_split"]

    # Build failures caused by the feature set the tool turns on (log heuristic).
    conflict = []
    for r in rows:
        if r["outcome"] in CHECKED or r.get("error_kind") == "no_lib_target":
            continue
        f = os.path.join(A, "logs", f"{r['package']}__{r['prev']}__{r['curr']}.patch.txt")
        text = open(f, errors="replace").read() if os.path.isfile(f) else (r.get("log_tail") or "")
        if any(p.search(text) for p in FEATURE_CONFLICT):
            conflict.append(f"{r['package']} {r['curr']}")
    out["feature_conflict_build_failures"] = {
        "count": len(conflict), "pairs": conflict,
        "note": "log heuristic: compile_error! guards on features, items defined twice "
                "(E0428/E0252) or nightly-only std::simd behind a feature; the paper says "
                "'about nine'"}

    # Checkable share of library releases by release year (created_at of version).
    df = pd.read_csv(args.data, keep_default_na=False, na_values=[""],
                     usecols=["package", "prev_version", "version", "created_at"], dtype=str)
    m = a.merge(df, left_on=["package", "prev", "curr"],
                right_on=["package", "prev_version", "version"], how="left")
    m["year"] = m["created_at"].str[:4].astype(int)
    lib = m[~m["error_kind"].eq("no_lib_target")].assign(ok=lambda x: x["outcome"].isin(CHECKED))
    by_year = {int(y): {"checked": int(g["ok"].sum()), "n": len(g), "pct": pct(g["ok"].sum(), len(g))}
               for y, g in lib.groupby("year")}
    pre, post = lib[lib["year"] < 2020], lib[lib["year"] >= 2020]
    out["checkable_library_releases_by_year"] = {
        "before_2020": {"checked": int(pre["ok"].sum()), "n": len(pre),
                        "pct": pct(pre["ok"].sum(), len(pre))},
        "from_2020": {"checked": int(post["ok"].sum()), "n": len(post),
                      "pct": pct(post["ok"].sum(), len(post))},
        "by_year": by_year}

    # Checked minor releases that are 0.y.z; confirmed breaks among them.
    ev = a[ok]
    minor = ev[ev["shipped"] == "minor"]
    gen = hr[hr["verdict"] == "genuine"]
    out["checked_minor_0yz"] = {"n_minor": len(minor),
                                "zero_major": int(minor["curr"].str.startswith("0.").sum()),
                                "breaks": int((gen["shipped"] == "minor").sum()),
                                "breaks_zero_major": int(((gen["shipped"] == "minor")
                                                          & gen["curr"].str.startswith("0.")).sum())}

    # Rates without pairs that have a pre-release at either end.
    def is_pre(df_):
        return df_["prev"].str.contains("-") | df_["curr"].str.contains("-")

    nopre = {}
    for cls in ("minor", "patch"):
        e = ev[ev["shipped"] == cls]
        g = gen[gen["shipped"] == cls]
        k, nn = len(g) - int(is_pre(g).sum()), len(e) - int(is_pre(e).sum())
        nopre[cls] = {"k": k, "n": nn, "pct": pct(k, nn),
                      "prerelease_breaks": gen[(gen["shipped"] == cls) & is_pre(gen)]
                      .apply(lambda r: f"{r['package']} {r['prev']} -> {r['curr']}", axis=1).tolist()}
    out["confirmed_without_prereleases"] = nopre

    g = summ["confirmed"]["genuine"]["minor_plus_patch"]
    out["confirmed_minor_plus_patch"] = {"k": g["k"], "n": g["n"], "pct": pct(g["k"], g["n"])}
    l22 = summ["confirmed"]["genuine_with_0_22_lints"]
    out["confirmed_with_0_22_lints"] = {c: {"k": l22[c]["k"], "n": l22[c]["n"],
                                            "pct": pct(l22[c]["k"], l22[c]["n"])}
                                        for c in ("minor", "patch")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/cratesbump.csv.gz")
    ap.add_argument("--repos", default="results/crates_all_repos.csv")
    ap.add_argument("--fill-report", default="results/github_fill_report.json")
    ap.add_argument("--audit-dir", default="results/label_audit")
    ap.add_argument("--dump-dir", default=None,
                    help="crates.io dump (for the binary/macro split of the audit)")
    ap.add_argument("--out", default="results/paper_numbers.json")
    args = ap.parse_args()
    previous = json.load(open(args.out)) if os.path.isfile(args.out) else None
    res = {"github_zeros": github_numbers(args), "audit": audit_numbers(args, previous)}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
