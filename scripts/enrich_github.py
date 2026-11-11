#!/usr/bin/env python3
"""
Back-fill the 11 GitHub-derived feature columns.

The released dataset (data/cratesbump.csv.gz) is filled by
scripts/fill_github_features.py, which reuses this module's crawl, cache and
feature code; the command-line entry point below is the standalone variant
used for the earlier dataset versions.

Design: work PER REPOSITORY, not per release pair — this is what makes the job
finish in days rather than weeks.

  * commit features (tcm, tau, nbf, kwm, kwp, kwb, aml) come from ONE blobless
    bare clone of each repo + a single `git log` — no API calls at all;
  * issue/PR features (poi, pci, popr, pcpr) come from fetching each repo's
    issues+PRs ONCE via the REST API and bucketing them into release windows
    locally.

Release windows come from each crate's version publish dates (fetched from the
crates.io API and cached, or read from a --manifest CSV of
`package,version,created_at`). Every network result is cached under
--cache-dir, so the run is fully RESUMABLE: re-running skips anything already
done.

One GitHub token (~5,000 req/hr) is enough; set several comma-separated in
GITHUB_TOKEN to rotate them.

Usage:
    export GITHUB_TOKEN=<token>
    # smoke-test on 3 repos first:
    python3 scripts/enrich_github.py --limit-repos 3 --output smoke.csv.gz
    # then the real run:
    python3 scripts/enrich_github.py \
        --input  data/compliant_dataset.csv.gz \
        --repos  results/crates_all_repos.csv \
        --output data/compliant_dataset_v2.csv.gz
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

import pandas as pd

GH_FEATURES = ["tcm", "tau", "nbf", "kwm", "kwp", "kwb", "aml",
               "poi", "pci", "popr", "pcpr"]
COMMIT_FEATURES = ["tcm", "tau", "nbf", "kwm", "kwp", "kwb", "aml"]
ISSUE_PR_FEATURES = ["poi", "pci", "popr", "pcpr"]
BUGFIX_RE = re.compile(r"\b(fix|bug|defect|error|issue)", re.I)
UA = "semver-ml-dataset-builder (+https://github.com/muhammad-hassnain/CratesBump)"
UNIT = "\x1f"  # ASCII unit separator, used to split git-log fields


class NetError(Exception):
    """Raised when a request fails for network reasons after all retries, so
    callers can skip caching an incomplete result as if it were real."""


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


# ── HTTP with rate-limit handling ────────────────────────────────────────────
class Tokens:
    def __init__(self, env):
        self.toks = [t.strip() for t in (env or "").split(",") if t.strip()]
        self.i = 0

    def cur(self):
        return self.toks[self.i] if self.toks else None

    def rotate(self):
        if self.toks:
            self.i = (self.i + 1) % len(self.toks)


def http_json(url, tokens=None, tries=6):
    """GET a JSON URL. Adds a GitHub bearer token when `tokens` is given and
    honours X-RateLimit-Reset / Retry-After. Returns parsed JSON or None."""
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={
            "User-Agent": UA, "Accept": "application/vnd.github+json"})
        if tokens and tokens.cur():
            req.add_header("Authorization", f"Bearer {tokens.cur()}")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                retry = e.headers.get("Retry-After")
                reset = e.headers.get("X-RateLimit-Reset")
                if retry:
                    wait = int(retry)
                elif reset:
                    wait = max(1, int(reset) - int(time.time()) + 2)
                else:
                    wait = 60 * (attempt + 1)
                log(f"  rate-limited ({e.code}); sleeping {min(wait, 3900)}s")
                if tokens:
                    tokens.rotate()
                time.sleep(min(wait, 3900))
                continue
            if e.code == 404:
                return None
            log(f"  HTTP {e.code} for {url}")
            time.sleep(3 * (attempt + 1))
        except Exception as ex:
            log(f"  net error ({ex}) for {url}")
            time.sleep(3 * (attempt + 1))
    raise NetError(url)


# ── crates.io version publish dates -> release windows ───────────────────────
_last_crates_call = [0.0]


def crate_windows(pkg, cache_dir):
    """Return {curr_version: (t_prev_iso, t_curr_iso)} from non-yanked versions,
    ordered by publish date. Cached per crate."""
    cf = os.path.join(cache_dir, "versions", f"{urllib.parse.quote(pkg, safe='')}.json")
    if os.path.isfile(cf):
        try:
            return json.load(open(cf))
        except Exception:
            pass
    dt = time.time() - _last_crates_call[0]
    if dt < 1.0:
        time.sleep(1.0 - dt)  # be polite to crates.io
    _last_crates_call[0] = time.time()
    url = f"https://crates.io/api/v1/crates/{urllib.parse.quote(pkg, safe='')}/versions"
    try:
        data = http_json(url)  # no GitHub token; crates.io is unauthenticated
    except NetError:
        return {}  # transient network failure — don't cache; retry next run
    windows = {}
    if data and isinstance(data.get("versions"), list):
        vs = [(v["num"], v["created_at"]) for v in data["versions"]
              if not v.get("yanked") and v.get("created_at")]
        vs.sort(key=lambda x: x[1])  # publish order (ascending)
        for i in range(1, len(vs)):
            windows[vs[i][0]] = (vs[i - 1][1], vs[i][1])
    os.makedirs(os.path.dirname(cf), exist_ok=True)
    json.dump(windows, open(cf, "w"))
    return windows


def load_manifest_windows(path):
    """Build {package: {curr_version: (t_prev, t_curr)}} from a manifest CSV
    (package,version,created_at). Lets you skip crates.io entirely."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    out = {}
    for pkg, g in df.sort_values("created_at").groupby("package"):
        rows = list(zip(g["version"], g["created_at"]))
        w = {rows[i][0]: (rows[i - 1][1], rows[i][1]) for i in range(1, len(rows))}
        out[pkg] = w
    return out


# ── per-repo commit + issue/PR crawl (cached) ────────────────────────────────
def owner_repo(url):
    if not url or "github.com" not in url:
        return None
    u = re.sub(r"\.git$", "", url.strip())
    m = re.search(r"github\.com[:/]+([^/]+)/([^/#?]+)", u)
    return (m.group(1), m.group(2)) if m else None


def repo_key(url):
    """Normalised source-repository identity (e.g. github.com/owner/repo), or
    None. Used to group crates that share a repository."""
    if not isinstance(url, str) or not url.strip():
        return None
    gh = owner_repo(url)
    if gh:
        return f"github.com/{gh[0]}/{gh[1]}".lower()
    u = re.sub(r"^[a-z+]+://", "", url.strip().lower()).split("#")[0].split("?")[0]
    u = re.sub(r"\.git$", "", u.rstrip("/"))
    parts = u.split("/")
    return "/".join(parts[:3]) if len(parts) >= 3 else u


def repo_commits(owner, repo, tokens, cache_dir, work_dir):
    cf = os.path.join(cache_dir, "commits", f"{owner}__{repo}.json")
    if os.path.isfile(cf):
        try:
            return json.load(open(cf))
        except Exception:
            pass
    dest = os.path.join(work_dir, f"{owner}__{repo}.git")
    tok = tokens.cur()
    auth = f"x-access-token:{tok}@" if tok else ""
    url = f"https://{auth}github.com/{owner}/{repo}.git"
    commits, cacheable = [], True
    # Only commit metadata is read, so tree:0 (no trees, no blobs) gives the
    # same `git log` output as blob:none and is much faster on monorepos.
    clone_filter = os.environ.get("ENRICH_CLONE_FILTER", "blob:none")
    shutil.rmtree(dest, ignore_errors=True)   # leftover from an interrupted run
    try:
        subprocess.run(["git", "clone", "--bare", f"--filter={clone_filter}", "--quiet",
                        url, dest], check=True, timeout=1800,
                       capture_output=True, text=True)
        out = subprocess.run(
            ["git", "-C", dest, "log", "--all", "--no-merges",
             f"--pretty=format:%aI{UNIT}%an{UNIT}%s"],
            capture_output=True, text=True, errors="replace",  # non-UTF-8 author names
            timeout=900).stdout
        for line in out.splitlines():
            p = line.split(UNIT)
            if len(p) == 3:
                commits.append(p)  # [iso_date, author, subject]
    except subprocess.CalledProcessError as ex:
        err = (ex.stderr or "").lower()
        if "not found" in err or "repository not found" in err:
            log(f"  clone failed (private/gone?): {owner}/{repo}")
        else:  # transient (network, timeout): do not cache an empty history
            cacheable = False
            log(f"  clone error {owner}/{repo}: {err.strip()[:120]}")
    except Exception as ex:
        cacheable = False
        log(f"  clone/log error {owner}/{repo}: {ex}")
    finally:
        shutil.rmtree(dest, ignore_errors=True)
    if not cacheable:
        return commits
    os.makedirs(os.path.dirname(cf), exist_ok=True)
    json.dump(commits, open(cf, "w"))
    return commits


_GQL = """query($owner:String!,$name:String!,$cursor:String){
  repository(owner:$owner,name:$name){
    %s(first:100, after:$cursor, orderBy:{field:CREATED_AT,direction:ASC}){
      pageInfo{hasNextPage endCursor} nodes{createdAt closedAt}}}}"""


def _sleep_until_graphql_reset(tokens):
    """Sleep until the GraphQL rate-limit window resets (GET /rate_limit is free)."""
    wait = 600
    try:
        rl = http_json("https://api.github.com/rate_limit", tokens)
        wait = max(1, int(rl["resources"]["graphql"]["reset"]) - int(time.time()) + 2)
    except Exception:
        pass
    log(f"  graphql rate limit; sleeping {min(wait, 3900)}s")
    time.sleep(min(wait, 3900))


def _graphql(query, variables, tokens, tries=6):
    body = json.dumps({"query": query, "variables": variables}).encode()
    for attempt in range(tries):
        req = urllib.request.Request("https://api.github.com/graphql", data=body, headers={
            "User-Agent": UA, "Content-Type": "application/json",
            "Authorization": f"Bearer {tokens.cur()}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read().decode())
            if out.get("errors"):
                if any(e.get("type") == "RATE_LIMIT" for e in out["errors"]):
                    _sleep_until_graphql_reset(tokens)
                    continue
                raise NetError(str(out["errors"])[:200])
            return out["data"]
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 502, 503):
                time.sleep(min(60 * (attempt + 1), 600))
                continue
            raise NetError(f"graphql HTTP {e.code}")
        except NetError:
            raise
        except Exception as ex:
            log(f"  graphql net error ({ex})")
            time.sleep(3 * (attempt + 1))
    raise NetError("graphql retries exhausted")


def repo_issues_graphql(owner, repo, tokens, until=None):
    """Issues + PRs via GraphQL cursor paging. The REST list endpoint refuses
    pages beyond 100 (HTTP 422), i.e. it can only return the oldest ~9,900
    items of very active repos; GraphQL has no such cap."""
    items = []
    for kind, is_pr in (("issues", False), ("pullRequests", True)):
        cursor = None
        while True:
            d = _graphql(_GQL % kind, {"owner": owner, "name": repo, "cursor": cursor}, tokens)
            conn = d["repository"][kind]
            for nd in conn["nodes"]:
                items.append({"pr": is_pr, "created": nd["createdAt"], "closed": nd["closedAt"]})
            if not conn["pageInfo"]["hasNextPage"]:
                break
            # No early stop at `until`: the result is cached per repo and other
            # crates of the same repo may need later windows.
            cursor = conn["pageInfo"]["endCursor"]
    items.sort(key=lambda it: it["created"] or "")
    return items


_GQL_REPO = ("r{i}: repository(owner: $o{i}, name: $n{i}) {{"
             " issues(first: 100) {{ pageInfo {{ hasNextPage }} nodes {{ createdAt closedAt }} }}"
             " pullRequests(first: 100) {{ pageInfo {{ hasNextPage }} nodes {{ createdAt closedAt }} }} }}")


def _issues_batch(part, tokens, cache_dir):
    """One GraphQL query for up to ~25 repositories; returns (cached, not_found)."""
    decl = ", ".join(f"$o{i}: String!, $n{i}: String!" for i in range(len(part)))
    query = "query(%s) { %s }" % (decl, " ".join(_GQL_REPO.format(i=i) for i in range(len(part))))
    variables = {}
    for i, (o, n) in enumerate(part):
        variables[f"o{i}"], variables[f"n{i}"] = o, n
    body = json.dumps({"query": query, "variables": variables}).encode()
    out = None
    for attempt in range(6):
        req = urllib.request.Request("https://api.github.com/graphql", data=body, headers={
            "User-Agent": UA, "Content-Type": "application/json",
            "Authorization": f"Bearer {tokens.cur()}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as e:
            wait = (max(1, int(e.headers.get("X-RateLimit-Reset", "0")) - int(time.time()) + 2)
                    if e.headers.get("X-RateLimit-Remaining") == "0" else 30 * (attempt + 1))
            log(f"  graphql batch HTTP {e.code}; sleeping {min(wait, 3900)}s")
            time.sleep(min(wait, 3900))
        except Exception as ex:
            log(f"  graphql batch net error ({ex})")
            time.sleep(5 * (attempt + 1))
    if out is None:
        return 0, 0
    gone = {e["path"][0] for e in out.get("errors") or []
            if e.get("type") == "NOT_FOUND" and e.get("path")}
    data = out.get("data") or {}
    cached = not_found = 0
    for i, (o, n) in enumerate(part):
        node = data.get(f"r{i}")
        if node is None:
            not_found += f"r{i}" in gone   # left for the REST path (follows renames)
            continue
        conns = (node.get("issues"), node.get("pullRequests"))
        if (any(c is None or c["pageInfo"]["hasNextPage"] for c in conns)
                or any(x is None for c in conns for x in c["nodes"])):
            continue   # large, or partly unreadable via GraphQL: per-repo path
        items = ([{"pr": False, "created": x["createdAt"], "closed": x["closedAt"]}
                  for x in node["issues"]["nodes"]]
                 + [{"pr": True, "created": x["createdAt"], "closed": x["closedAt"]}
                    for x in node["pullRequests"]["nodes"]])
        items.sort(key=lambda it: it["created"] or "")
        json.dump(items, open(os.path.join(cache_dir, "issues", f"{o}__{n}.json"), "w"))
        cached += 1
    return cached, not_found


def issues_batch_prefetch(repos, tokens, cache_dir, batch=25, workers=6):
    """Cache the issue/PR lists of many small repositories, `batch` per GraphQL
    query (aliases; one query costs 1 rate-limit point), so the per-repo REST
    crawl is not limited to 5,000 repos an hour. A repository with at most 100
    issues and 100 PRs is complete in one query and cached here; a larger one
    is left uncached for the per-repo path (repo_issues). So is a repository
    GraphQL reports as not found, because GraphQL does not follow renamed
    repositories and REST does; and so is anything that failed (rate limit,
    timeout). Returns (cached, not_found, left)."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [o for o in repos
            if not os.path.isfile(os.path.join(cache_dir, "issues", f"{o[0]}__{o[1]}.json"))]
    os.makedirs(os.path.join(cache_dir, "issues"), exist_ok=True)
    parts = [todo[s:s + batch] for s in range(0, len(todo), batch)]
    cached = not_found = 0
    with ThreadPoolExecutor(workers) as pool:
        for k, (c, nf) in enumerate(pool.map(lambda p: _issues_batch(p, tokens, cache_dir), parts), 1):
            cached += c
            not_found += nf
            if k % 200 == 0 or k == len(parts):
                log(f"  graphql batch prefetch: {min(k * batch, len(todo)):,}/{len(todo):,} repos")
    return cached, not_found, len(todo) - cached


def repo_issues(owner, repo, tokens, cache_dir, until_iso=None):
    cf = os.path.join(cache_dir, "issues", f"{owner}__{repo}.json")
    if os.path.isfile(cf):
        try:
            return json.load(open(cf))
        except Exception:
            pass
    until = parse_iso(until_iso)
    items, page, complete = [], 1, True
    while page <= 300:  # hard cap ~30k issues+PRs per repo
        url = (f"https://api.github.com/repos/{owner}/{repo}/issues"
               f"?state=all&per_page=100&sort=created&direction=asc&page={page}")
        try:
            data = http_json(url, tokens)
        except NetError:
            complete = False  # network failure — don't cache a partial result
            if page >= 100 and tokens and tokens.cur():
                # REST stops at page 100 for very active repos; fetch the
                # complete list through GraphQL instead.
                try:
                    items = repo_issues_graphql(owner, repo, tokens, until)
                    complete = True
                    log(f"  {owner}/{repo}: {len(items):,} issues+PRs via GraphQL")
                except NetError as ex:
                    log(f"  {owner}/{repo}: GraphQL fallback failed ({ex}); partial issue list")
            break
        if data is None or not isinstance(data, list):
            break
        for it in data:
            items.append({"pr": "pull_request" in it,
                          "created": it.get("created_at"),
                          "closed": it.get("closed_at")})
        if len(data) < 100:
            break
        if until is not None:
            last = parse_iso(data[-1].get("created_at"))
            if last is not None and last > until:
                break  # everything past our windows; stop paging early
        page += 1
    if complete:  # only persist a fully-fetched result, so an outage can't poison the cache
        os.makedirs(os.path.dirname(cf), exist_ok=True)
        json.dump(items, open(cf, "w"))
    return items


# ── feature computation for one release window ───────────────────────────────
def window_features(commits, issues, t_prev, t_curr):
    f = dict.fromkeys(GH_FEATURES, 0)
    if t_curr is None:
        return f
    authors, lens, tcm = set(), [], 0
    nbf = kwm = kwp = kwb = 0
    for iso, author, subj in commits:
        d = parse_iso(iso)
        if d is None or d > t_curr or (t_prev is not None and d <= t_prev):
            continue
        tcm += 1
        authors.add(author)
        lens.append(len(subj))
        s = subj.lower()
        if BUGFIX_RE.search(subj):
            nbf += 1
        if "major" in s:
            kwm += 1
        if "patch" in s:
            kwp += 1
        if "break" in s:
            kwb += 1
    f.update(tcm=tcm, tau=len(authors), nbf=nbf, kwm=kwm, kwp=kwp, kwb=kwb,
             aml=round(sum(lens) / len(lens), 2) if lens else 0)

    poi = pci = popr = pcpr = 0
    for it in issues:
        c, cl = parse_iso(it["created"]), parse_iso(it["closed"])
        opened = c is not None and c <= t_curr and (t_prev is None or c > t_prev)
        closed = cl is not None and cl <= t_curr and (t_prev is None or cl > t_prev)
        if it["pr"]:
            popr += opened
            pcpr += closed
        else:
            poi += opened
            pci += closed
    f.update(poi=poi, pci=pci, popr=popr, pcpr=pcpr)
    return f


def write_output(df, path):
    """Sort deterministically and write atomically (tmp + rename) so a kill
    mid-write can never leave a corrupt file."""
    out = df.sort_values(["package", "version"]).reset_index(drop=True)
    tmp = f"{path}.tmp.{os.getpid()}"  # PID-unique so concurrent runs can't collide on rename
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    out.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    return out


def main():
    ap = argparse.ArgumentParser(description="Back-fill GitHub features (v2).")
    ap.add_argument("--input", default="data/compliant_dataset.csv.gz")
    ap.add_argument("--repos", default="results/crates_all_repos.csv")
    ap.add_argument("--output", default="data/compliant_dataset_v2.csv.gz")
    ap.add_argument("--cache-dir", default="enrich_cache")
    ap.add_argument("--work-dir", default="enrich_work")
    ap.add_argument("--manifest", default=None,
                    help="Optional package,version,created_at CSV; skips crates.io.")
    ap.add_argument("--limit-repos", type=int, default=0,
                    help="Only process the first N repos (smoke test).")
    ap.add_argument("--partial-every", type=int, default=0,
                    help="Also write --output every N repos (atomic), so a kill "
                         "leaves an up-to-date result on the persistent volume.")
    ap.add_argument("--no-issues", action="store_true",
                    help="Skip the issue/PR REST crawl entirely (needs no GitHub "
                         "token); populates only the 7 commit-based features.")
    ap.add_argument("--issues-only", action="store_true",
                    help="Skip the (slow) git-clone commit crawl entirely and ONLY "
                         "fetch issue/PR features via REST. The 7 commit-based "
                         "features (tcm, tau, nbf, kwm, kwp, kwb, aml) are "
                         "preserved from the input CSV. Fast: no git clone, "
                         "only REST calls limited by the token's 5000/h rate.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel repo workers. For --issues-only "
                         "this parallelises the REST crawl against the per-token "
                         "5000/h rate limit; useful values are 4-8. For the "
                         "full crawl the git-clone bottleneck dominates and "
                         "parallelism provides smaller gains.")
    args = ap.parse_args()

    # Token comes from $GITHUB_TOKEN, else a gitignored .token file (same
    # convention as the Rust extractor). The value is never printed.
    tok_env = os.environ.get("GITHUB_TOKEN")
    if not tok_env:
        for p in (".token", os.path.expanduser("~/.token")):
            try:
                with open(p) as fh:
                    tok_env = fh.read().strip()
                if tok_env:
                    log(f"Using GitHub token from {p}")
                    break
            except FileNotFoundError:
                continue
    tokens = Tokens(tok_env)
    if not tokens.toks and not args.no_issues:
        log("WARNING: no GITHUB_TOKEN set — issue/PR calls will hit the 60 req/hr "
            "unauthenticated limit and clones may throttle. Set one, or pass "
            "--no-issues to collect only the commit-based features.")
    for sub in ("versions", "commits", "issues"):
        os.makedirs(os.path.join(args.cache_dir, sub), exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    # keep_default_na=False: a crate literally named "null"/"nan" must stay a string
    df = pd.read_csv(args.input, keep_default_na=False, na_values=[""])
    log(f"dataset: {len(df):,} rows, {df['package'].nunique():,} crates")

    repo_map = pd.read_csv(args.repos, keep_default_na=False, na_values=[""])
    pkg2repo = {}
    for pkg, url in zip(repo_map["package"], repo_map["repository"]):
        orp = owner_repo(str(url))
        if orp:
            pkg2repo[pkg] = orp

    manifest_windows = load_manifest_windows(args.manifest) if args.manifest else None

    # Which repos do we actually need (those referenced by dataset crates)?
    pkgs = set(df["package"].unique())
    repos_needed = {}
    for pkg in pkgs:
        if pkg in pkg2repo:
            repos_needed.setdefault(pkg2repo[pkg], []).append(pkg)
    repo_list = sorted(repos_needed)
    if args.limit_repos:
        repo_list = repo_list[:args.limit_repos]
        keep = {p for r in repo_list for p in repos_needed[r]}
        df = df[df["package"].isin(keep)].reset_index(drop=True)
    log(f"repos to crawl: {len(repo_list):,}")

    # Release windows per crate (from manifest or crates.io, cached).
    def windows_for(pkg):
        if manifest_windows is not None:
            return manifest_windows.get(pkg, {})
        return crate_windows(pkg, args.cache_dir)

    # Which feature columns do we overwrite? With --issues-only we preserve
    # the 7 commit features that are already populated in the input CSV, and
    # only touch the 4 issue/PR features. Without the flag, every GH feature
    # is zeroed and populated.
    overwrite_cols = ISSUE_PR_FEATURES if args.issues_only else GH_FEATURES
    for c in overwrite_cols:
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0).astype(float)
        if not args.issues_only:
            df[c] = 0.0
    # Still make sure columns exist for the commit features (idempotent).
    for c in GH_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    col_loc = {k: df.columns.get_loc(k) for k in GH_FEATURES}
    idx = {(p, v): i for i, (p, v) in enumerate(zip(df["package"], df["version"]))}

    def process_one_repo(orp):
        """Fetch per-repo data and compute per-crate rows. Returns a list of
        (row_index, {feature: value}) tuples; the caller writes them in the
        main thread so we don't race on `df`."""
        owner, repo = orp
        crates = repos_needed[orp]
        all_w = {pkg: windows_for(pkg) for pkg in crates}
        ends = [w[1] for wm in all_w.values() for w in wm.values()]
        until = max(ends) if ends else None
        if args.issues_only:
            commits = {}                             # skip the slow git clone
        else:
            commits = repo_commits(owner, repo, tokens, args.cache_dir, args.work_dir)
        issues = [] if args.no_issues else repo_issues(owner, repo, tokens, args.cache_dir, until)
        out = []
        for pkg in crates:
            for ver, (tp, tc) in all_w[pkg].items():
                i = idx.get((pkg, ver))
                if i is None:
                    continue
                feats = window_features(commits, issues, parse_iso(tp), parse_iso(tc))
                # In --issues-only mode we only emit the issue/PR features so
                # the commit features from the input CSV are preserved.
                if args.issues_only:
                    feats = {k: v for k, v in feats.items() if k in ISSUE_PR_FEATURES}
                out.append((i, feats))
        return orp, out

    done_rows = 0
    n = 0
    if args.workers and args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one_repo, orp): orp for orp in repo_list}
            for fut in as_completed(futures):
                _, rows = fut.result()
                for i, feats in rows:
                    for k, val in feats.items():
                        df.iat[i, col_loc[k]] = val
                    done_rows += 1
                n += 1
                if n % 100 == 0 or n == len(repo_list):
                    log(f"  [{n}/{len(repo_list)}] repos done; {done_rows:,} rows filled")
                if args.partial_every and n % args.partial_every == 0:
                    write_output(df, args.output)
                    log(f"  checkpoint: wrote partial {args.output} after {n} repos")
    else:
        for orp in repo_list:
            _, rows = process_one_repo(orp)
            for i, feats in rows:
                for k, val in feats.items():
                    df.iat[i, col_loc[k]] = val
                done_rows += 1
            n += 1
            if n % 100 == 0 or n == len(repo_list):
                log(f"  [{n}/{len(repo_list)}] repos done; {done_rows:,} rows filled")
            if args.partial_every and n % args.partial_every == 0:
                write_output(df, args.output)
                log(f"  checkpoint: wrote partial {args.output} after {n} repos")

    # Deterministic order + report coverage.
    df = write_output(df, args.output)
    nz = int((df[["tcm", "poi", "popr"]].sum(axis=1) > 0).sum())
    log(f"\nWrote {args.output} ({len(df):,} rows). "
        f"Rows with any GitHub signal: {nz:,} ({100*nz/max(len(df),1):.1f}%).")


if __name__ == "__main__":
    main()
