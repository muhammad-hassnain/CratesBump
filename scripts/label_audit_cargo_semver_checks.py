#!/usr/bin/env python3
"""External audit of shipped Cargo labels with `cargo-semver-checks`.

For a stratified sample of release pairs (at most one pair per crate per
label, so pair rates are also crate rates), downloads both `.crate` tarballs
and runs `cargo semver-checks check-release --release-type patch`: this makes
every major- and minor-level lint run, so the tool reports the bump the
public-API diff *requires*, which we compare to the shipped (declared) label.

Comparison semantics: the tool sees only public-API changes, so
  shipped > required  is expected and benign (new features, behaviour changes,
                      or lints the tool does not have);
  shipped < required  is a FLAG: a candidate SemVer violation by the author.
Flags are not violations until checked by hand: some are harmless in
practice (discriminant-only changes, items in __private / doc(hidden)
modules, binary crates whose library is incidental, generated crates,
crates only used by =-pinned workspace siblings), and some are tool false
positives. Each flagged pair is therefore re-run with --default-features
(the tool's default enables almost all features, which misfires on crates
with mutually exclusive features), its full tool output is saved under
<out-dir>/logs/, and the hand-review verdicts in <out-dir>/hand_review.csv
(package, prev, curr, verdict = genuine | harmless | not_a_break,
compile_break = yes | no, reason) give the confirmed rates in summary.json.

Each pair's predecessor is the file's prev_version column; the sampled
pairs are checked against the extractor's rule rebuilt from the crates.io
dump (src/extractor/mod.rs::plan_pairs, scripts/semver_order.py), and the
run stops if any disagrees.

Sandbox: building a crate runs its build scripts and proc macros, so every
cargo command runs in a throwaway container built from
docker/audit.Dockerfile (non-root, no capabilities, no network during the
build, only the pair's scratch directory mounted). Build the image first:
    docker build -t cratesbump-audit:0.51.0 -f docker/audit.Dockerfile docker/

Usage:
    python3 scripts/label_audit_cargo_semver_checks.py \\
        data/cratesbump.csv.gz \\
        .dump_cache/2026-10-12-020016 \\
        --n 150 --seed 42 --workers 4 \\
        --out-dir results/label_audit
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

RANK = {"patch": 0, "minor": 1, "major": 2}
LOG_DIR = None   # set in main(): <out-dir>/logs


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


from semver_order import plan_prev  # noqa: E402  (mirror of extractor plan_pairs)


def load_predecessors(dump_dir: str, crates: set) -> dict:
    data = os.path.join(dump_dir, "data")
    csv.field_size_limit(1 << 30)
    name_of = {}
    with open(os.path.join(data, "crates.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["name"] in crates:
                name_of[r["id"]] = r["name"]
    vers = defaultdict(list)
    with open(os.path.join(data, "versions.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            name = name_of.get(r["crate_id"])
            if name is None or r.get("yanked", "f") in ("t", "true", "True", "1"):
                continue
            vers[name].append((r["created_at"], r["num"]))
    return {name: plan_prev([n for _, n in sorted(v)]) for name, v in vers.items()}


def download_crate(name: str, version: str, dest: str):
    url = f"https://static.crates.io/crates/{name}/{name}-{version}.crate"
    tar_path = os.path.join(dest, f"{name}-{version}.tar.gz")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tar_path, "wb") as fh:
            fh.write(resp.read())
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(dest, filter="data")
    except Exception as e:  # network / tar errors are recorded, not fatal
        return None, str(e)
    return os.path.join(dest, f"{name}-{version}"), None


class Sandbox:
    """Runs cargo inside a throwaway Docker container (see docker/audit.Dockerfile).

    Two steps per pair:
      fetch : `cargo fetch` for both versions, WITH network. Fetch only
              downloads and unpacks; it never runs build scripts.
      check : `cargo semver-checks`, with --network none and
              CARGO_NET_OFFLINE, so build scripts and proc macros run with
              no network access.
    The container is non-root, has no capabilities, and sees only the pair's
    scratch directory plus a Docker-managed cargo cache volume (no host
    files). With image=None the commands run directly on the host instead.
    """

    def __init__(self, image, cache_volume, cpus, memory):
        self.image, self.cache, self.cpus, self.memory = image, cache_volume, cpus, memory

    def _cache_mount(self, pair_dir):
        # "per-pair": a fresh, empty CARGO_HOME inside the pair's scratch dir,
        # so offline resolution only sees what this pair fetched and results
        # do not depend on which pairs ran before. Otherwise a named volume
        # shared by all pairs (faster, but order-dependent).
        if self.cache == "per-pair":
            path = os.path.join(os.path.abspath(pair_dir), ".cargo-home")
            os.makedirs(path, exist_ok=True)
            return path
        return self.cache

    def _docker(self, pair_dir, cmd, network, timeout):
        name = f"cratesbump-audit-{uuid.uuid4().hex[:12]}"
        full = ["docker", "run", "--rm", "--name", name,
                "--network", "bridge" if network else "none",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--cpus", str(self.cpus), "--memory", self.memory,
                "--pids-limit", "4096",
                "-v", f"{os.path.abspath(pair_dir)}:/work",
                "-v", f"{self._cache_mount(pair_dir)}:/cargo-home",
                "-e", "CARGO_HOME=/cargo-home", "-e", "CARGO_TERM_COLOR=never"]
        if not network:
            full += ["-e", "CARGO_NET_OFFLINE=true"]
        full += [self.image, "timeout", str(timeout)] + cmd
        try:
            res = subprocess.run(full, capture_output=True, text=True, timeout=timeout + 120)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return None, "timeout"
        if res.returncode == 124:
            return None, "timeout"
        return res, res.stdout + "\n" + res.stderr

    def _host(self, pair_dir, cmd, timeout):
        cmd = [c.replace("/work", os.path.abspath(pair_dir)) for c in cmd]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                 env={**os.environ, "CARGO_TERM_COLOR": "never"})
        except subprocess.TimeoutExpired:
            return None, "timeout"
        return res, res.stdout + "\n" + res.stderr

    def run(self, pair_dir, cmd, network, timeout):
        if self.image is None:
            return self._host(pair_dir, cmd, timeout)
        return self._docker(pair_dir, cmd, network, timeout)


def fetch(sb: Sandbox, pair_dir, sub, timeout, fresh: bool):
    """Download dependencies (network on, no build scripts run).

    fresh=False: `cargo fetch` honours a Cargo.lock shipped inside the .crate,
      so the offline build later prefers the dependency versions the crate
      was released with.
    fresh=True: re-resolve to the newest compatible versions first (what
      cargo-semver-checks itself asks for when the crate ships no lockfile).
    """
    m = f"/work/{sub}/Cargo.toml"
    cmd = f"cargo fetch --manifest-path {m}"
    if fresh:
        cmd = f"cargo generate-lockfile --manifest-path {m} && " + cmd
    return sb.run(pair_dir, ["sh", "-c", cmd], network=True, timeout=timeout)


def run_tool(sb: Sandbox, pair_dir, curr_sub, prev_sub, release_type, timeout,
             default_features=False):
    cmd = ["cargo", "semver-checks", "check-release",
           "--manifest-path", f"/work/{curr_sub}/Cargo.toml",
           "--baseline-root", f"/work/{prev_sub}"]
    if release_type:
        cmd += ["--release-type", release_type]
    if default_features:
        cmd += ["--default-features"]
    return sb.run(pair_dir, cmd, network=False, timeout=timeout)


def parse_failures(log: str) -> dict:
    """{lint: [item lines]} from the tool's "--- failure <lint>: ... ---"
    blocks; item lines are the indented lines under "Failed in:"."""
    out, lint, in_items = {}, None, False
    for line in (log or "").splitlines():
        m = re.match(r"--- (failure|warning) ([A-Za-z0-9_]+):", line)
        if m:
            lint = m.group(2) if m.group(1) == "failure" else None
            in_items = False
            if lint:
                out.setdefault(lint, [])
            continue
        if lint and line.strip() == "Failed in:":
            in_items = True
            continue
        if in_items:
            if line.startswith("  ") and line.strip():
                out[lint].append(line.strip())
            elif not line.strip():
                in_items = False
    return out


def log_name(rec: dict, tag: str) -> str:
    safe = lambda x: re.sub(r"[^A-Za-z0-9_.+-]", "_", x)
    return f"{safe(rec['package'])}__{safe(rec['prev'])}__{safe(rec['curr'])}.{tag}.txt"


def classify_error(log: str) -> str:
    """Coarse reason a tool run produced no verdict."""
    if log == "timeout":
        return "timeout"
    low = log.lower()
    if "library targets" in low or "no lib target" in low or "is not a library" in low:
        return "no_lib_target"
    if "offline" in low and ("failed to" in low or "no matching package" in low):
        return "offline_resolution"
    if "failed to build rustdoc" in low or "could not compile" in low or "error[e" in low:
        return "build_failed"
    if "rustdoc json" in low and "format" in low:
        return "rustdoc_format"
    return "other"


def parse_required(log: str):
    if re.search(r"Summary\s+no semver update required", log):
        return "patch"
    m = re.search(r"Summary\s+semver requires new (major|minor) version", log)
    if m:
        return m.group(1)
    return None


def audit_pair(rec: dict, work_root: str, timeout: int, sb: Sandbox) -> dict:
    """Two attempts: dependencies as locked by the crate, then (only if the
    first attempt produced no verdict for a library crate) freshly resolved."""
    out = _attempt(rec, work_root, timeout, sb, fresh=False)
    if out["outcome"] == "tool_error" and out.get("error_kind") != "no_lib_target":
        first = out.get("error_kind")
        out = _attempt(rec, work_root, timeout, sb, fresh=True)
        out["first_attempt_error"] = first
    return out


def _attempt(rec: dict, work_root: str, timeout: int, sb: Sandbox, fresh: bool) -> dict:
    wd = tempfile.mkdtemp(dir=work_root)
    t0 = time.time()
    base = {**rec, "resolution": "fresh" if fresh else "crate_lockfile"}
    try:
        prev_dir, err = download_crate(rec["package"], rec["prev"], wd)
        if not prev_dir:
            return {**base, "outcome": "download_failed", "detail": err}
        curr_dir, err = download_crate(rec["package"], rec["curr"], wd)
        if not curr_dir:
            return {**base, "outcome": "download_failed", "detail": err}
        prev_sub, curr_sub = os.path.basename(prev_dir), os.path.basename(curr_dir)

        for sub in (prev_sub, curr_sub):
            res, log = fetch(sb, wd, sub, timeout, fresh)
            if res is None or res.returncode != 0:
                errs = [l.strip() for l in (log or "").splitlines() if l.strip().startswith("error")]
                return {**base, "outcome": "tool_error", "error_kind": "fetch_failed",
                        "detail": (errs[0] if errs else (log or "")[-300:])}

        res2, log2 = run_tool(sb, wd, curr_sub, prev_sub, "patch", timeout)
        required = parse_required(log2 or "")
        out = {**base, "tool_required": required, "seconds": round(time.time() - t0, 1)}
        if LOG_DIR:
            with open(os.path.join(LOG_DIR, log_name(rec, "patch")), "w") as fh:
                fh.write(log2 or "")
        if required is None:
            errs = [l.strip() for l in (log2 or "").splitlines() if l.strip().startswith("error")]
            out.update(outcome="tool_error", error_kind=classify_error(log2 or ""),
                       detail=(errs[0] if errs else (log2 or "")[-300:]),
                       log_tail=(log2 or "")[-1500:])
        else:
            d = RANK[rec["shipped"]] - RANK[required]
            out.update(outcome="agree" if d == 0 else ("shipped_higher" if d > 0 else "shipped_lower"),
                       detail="")
            # Failure blocks look like "--- failure function_missing: <title> ---".
            fails = parse_failures(log2)
            if fails:
                out["failed_lints"] = sorted(fails)
                out["failed_items"] = fails
            if out["outcome"] == "shipped_lower":
                # Re-check with the crate's default features only.
                res3, log3 = run_tool(sb, wd, curr_sub, prev_sub, "patch", timeout,
                                      default_features=True)
                req3 = parse_required(log3 or "")
                out["tool_required_default_features"] = req3
                out["failed_lints_default_features"] = sorted(parse_failures(log3))
                if req3 is not None:
                    out["outcome_default_features"] = (
                        "agree" if RANK[rec["shipped"]] == RANK[req3] else
                        ("shipped_higher" if RANK[rec["shipped"]] > RANK[req3] else "shipped_lower"))
                if LOG_DIR:
                    with open(os.path.join(LOG_DIR, log_name(rec, "default_features")), "w") as fh:
                        fh.write(log3 or "")
        return out
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = (z / den) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round(max(0.0, c - m), 4), round(min(1.0, c + m), 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("dump_dir")
    ap.add_argument("--n", type=int, default=40, help="pairs per shipped class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--work-dir", default=".audit_work")
    ap.add_argument("--out-dir", default="results/label_audit")
    ap.add_argument("--image", default="cratesbump-audit:0.51.0",
                    help="Docker image built from docker/audit.Dockerfile. Pass "
                         "'none' to run on the host (NOT recommended: the audit "
                         "executes third-party build scripts).")
    ap.add_argument("--cache-volume", default="per-pair",
                    help="'per-pair' (default: fresh CARGO_HOME per attempt, order-"
                         "independent) or the name of a Docker volume shared by all pairs")
    ap.add_argument("--cpus", type=float, default=3)
    ap.add_argument("--memory", default="6g")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run the tool errors in <out-dir>/per_pair.jsonl, then re-summarise")
    ap.add_argument("--rerun-outcomes", nargs="*", default=[],
                    help="with --retry-errors, also re-run pairs with these outcomes "
                         "(e.g. shipped_lower, to record which lints failed)")
    ap.add_argument("--packages", nargs="*", default=[],
                    help="restrict --retry-errors / --rerun-outcomes to these crates")
    ap.add_argument("--summarize-only", action="store_true",
                    help="only rebuild <out-dir>/summary.json from per_pair.jsonl")
    args = ap.parse_args()
    sb = Sandbox(None if args.image == "none" else args.image,
                 args.cache_volume, args.cpus, args.memory)
    if args.summarize_only:
        detail_path = os.path.join(args.out_dir, "per_pair.jsonl")
        write_summary([json.loads(l) for l in open(detail_path)], args, sb)
        return 0
    global LOG_DIR
    LOG_DIR = os.path.join(args.out_dir, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    if sb.image and sb.cache != "per-pair":
        # Docker creates named volumes root-owned; hand it to the audit user.
        subprocess.run(["docker", "run", "--rm", "--user", "0", "-v",
                        f"{args.cache_volume}:/cargo-home", sb.image,
                        "chown", f"{os.getuid()}:{os.getgid()}", "/cargo-home"],
                       check=True, capture_output=True)
    elif not sb.image:
        print("WARNING: running crate builds on the host, unsandboxed", file=sys.stderr)
    if args.retry_errors or args.rerun_outcomes:
        os.makedirs(args.work_dir, exist_ok=True)
        write_summary(retry_errors(args, sb), args, sb)
        return 0

    head = pd.read_csv(args.csv, nrows=0).columns
    cols = ["package", "version", "release_type"] + (["prev_version"] if "prev_version" in head else [])
    df = pd.read_csv(args.csv, keep_default_na=False, na_values=[""], usecols=cols)
    # At most one pair per crate per label, so a pair rate is also a crate rate.
    sample = pd.concat([
        df[df["release_type"] == c].groupby("package").sample(n=1, random_state=args.seed)
          .sample(n=args.n, random_state=args.seed)
        for c in ("major", "minor", "patch")
    ])
    print(f"sampled {len(sample)} pairs from {sample['package'].nunique()} crates", file=sys.stderr)

    # The file's own prev_version is the pair the features and label describe;
    # older files without it fall back to the same rule rebuilt from the dump.
    prev_maps = ({} if "prev_version" in cols
                 else load_predecessors(args.dump_dir, set(sample["package"])))
    if "prev_version" in cols:
        # Every sampled pair must be the extractor's pair rebuilt from the dump.
        dump_prev = load_predecessors(args.dump_dir, set(sample["package"]))
        bad = [(r.package, r.prev_version, r.version, dump_prev.get(r.package, {}).get(r.version))
               for r in sample.itertuples()
               if dump_prev.get(r.package, {}).get(r.version) != r.prev_version]
        if bad:
            print(f"ABORT: {len(bad)} sampled pairs disagree with the dump's pairing: {bad[:5]}",
                  file=sys.stderr)
            return 2
        print("all sampled pairs match the extractor's pairing rebuilt from the dump",
              file=sys.stderr)
    todo, skipped = [], []
    for _, r in sample.iterrows():
        prev = (r["prev_version"] if "prev_version" in cols
                else prev_maps.get(r["package"], {}).get(r["version"]))
        rec = {"package": r["package"], "curr": r["version"], "shipped": r["release_type"],
               "prev": prev if isinstance(prev, str) and prev else None}
        (todo if rec["prev"] else skipped).append(rec)
    for rec in skipped:
        rec.update(outcome="no_predecessor", detail="")
    print(f"{len(todo)} with predecessor, {len(skipped)} without", file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    detail_path = os.path.join(args.out_dir, "per_pair.jsonl")
    results = list(skipped)
    with open(detail_path, "w") as fh, ThreadPoolExecutor(args.workers) as pool:
        for rec in skipped:
            fh.write(json.dumps(rec) + "\n")
        futs = [pool.submit(audit_pair, rec, args.work_dir, args.timeout, sb) for rec in todo]
        for i, f in enumerate(as_completed(futs), 1):
            out = f.result()
            results.append(out)
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            print(f"  [{i}/{len(todo)}] {out['package']} {out['prev']}->{out['curr']} "
                  f"shipped={out['shipped']} required={out.get('tool_required')} "
                  f"-> {out['outcome']}", file=sys.stderr)

    write_summary(results, args, sb)
    return 0


def retry_errors(args, sb) -> list:
    """Re-run pairs whose earlier run ended in a tool error that may be
    transient (e.g. two sandboxes racing on the shared cargo cache).
    Binary-only crates are not retried: the tool cannot check them."""
    detail_path = os.path.join(args.out_dir, "per_pair.jsonl")
    results = [json.loads(l) for l in open(detail_path)]
    for r in results:
        if r["outcome"] == "tool_error" and r.get("error_kind") in (None, "other"):
            r["error_kind"] = classify_error(r.get("log_tail") or r.get("detail", ""))
    idx = [i for i, r in enumerate(results)
           if (not args.packages or r["package"] in args.packages) and (
           (args.retry_errors and r["outcome"] == "tool_error"
               and r["error_kind"] != "no_lib_target")
           or r["outcome"] in args.rerun_outcomes)]
    print(f"re-running {len(idx)} pairs", file=sys.stderr)
    base_keys = ("package", "curr", "shipped", "prev")
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(audit_pair, {k: results[i][k] for k in base_keys},
                            args.work_dir, args.timeout, sb): i for i in idx}
        for f in as_completed(futs):
            i = futs[f]
            out = f.result()
            out["retried"] = True
            results[i] = out
            print(f"  retry {out['package']} {out['prev']}->{out['curr']} -> {out['outcome']}"
                  f" {out.get('error_kind', '')}", file=sys.stderr)
    with open(detail_path, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    return results


def _rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def write_summary(results, args, sb):
    for r in results:
        if r["outcome"] == "tool_error" and r.get("error_kind") in (None, "other"):
            r["error_kind"] = classify_error(r.get("log_tail") or r.get("detail", ""))
    evaluated = [r for r in results if r["outcome"] in ("agree", "shipped_higher", "shipped_lower")]
    by_class = {}
    for c in ("major", "minor", "patch"):
        ev = [r for r in evaluated if r["shipped"] == c]
        by_class[c] = {"evaluated": len(ev), **Counter(r["outcome"] for r in ev)}
    n_ev = len(evaluated)
    oc = Counter(r["outcome"] for r in evaluated)
    mp = {c: [r for r in evaluated if r["shipped"] == c] for c in ("minor", "patch")}
    flagged = {c: [r for r in mp[c] if r["outcome"] == "shipped_lower"] for c in mp}

    # Like-for-like with Nowak & Gruevski (2023): only lints that existed in
    # cargo-semver-checks 0.22.0 (results/label_audit_lints_0.22.0.csv).
    old_lints = set()
    p022 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                        "label_audit_lints_0.22.0.csv")
    if os.path.isfile(p022):
        old_lints = {row["lint_0_51_0"] for row in csv.DictReader(open(p022))}
    def old_flag(r):
        return any(l in old_lints for l in r.get("failed_lints", []))

    # 0.y.z releases whose Cargo label is minor because only the PATCH number moved.
    def zero_y_patch_number(r):
        p, c = parse_version(r["prev"]), parse_version(r["curr"])
        return c[0] == 0 and c[1] == p[1] and c[1] != 0

    summary = {
        "dataset": args.csv,
        "n_sampled": len(results),
        "sampling": (f"{len(results)} pairs from {len({r['package'] for r in results})} distinct "
                     f"crates" + ("; at most one pair per crate per Cargo label"
                                  if max(Counter((r["package"], r["shipped"]) for r in results).values()) == 1
                                  else "")),
        "outcomes": dict(Counter(r["outcome"] for r in results)),
        "n_evaluated": n_ev,
        "evaluated_rate": round(n_ev / len(results), 4) if results else None,
        "declared_vs_required": {
            "agree": oc["agree"], "shipped_higher": oc["shipped_higher"],
            "shipped_lower_flags": oc["shipped_lower"],
        },
        "by_shipped_class": by_class,
        "flags": {
            "minor": _rate(len(flagged["minor"]), len(mp["minor"])),
            "patch": _rate(len(flagged["patch"]), len(mp["patch"])),
            "minor_plus_patch": _rate(len(flagged["minor"]) + len(flagged["patch"]),
                                      len(mp["minor"]) + len(mp["patch"])),
            "flagged_minor_that_are_0y_patch_number_bumps":
                sum(zero_y_patch_number(r) for r in flagged["minor"]),
            "still_flagged_with_default_features": sum(
                r.get("outcome_default_features") == "shipped_lower"
                for c in flagged for r in flagged[c]),
            "no_verdict_with_default_features": sum(
                r.get("tool_required_default_features") is None
                for c in flagged for r in flagged[c]),
            "lints_0_22_only": {
                c: _rate(sum(old_flag(r) for r in flagged[c]), len(mp[c])) for c in mp},
        },
        "tool_error_kinds": dict(Counter(r.get("error_kind", "unknown")
                                         for r in results if r["outcome"] == "tool_error")),
        "by_version_range": {
            rng: dict(Counter(r["outcome"] for r in evaluated
                              if (parse_version(r["curr"])[0] == 0) == (rng == "pre_1.0")))
            for rng in ("pre_1.0", "stable")
        },
        "sandbox": {"image": sb.image, "cpus": args.cpus, "memory": args.memory,
                    "network_during_build": "none" if sb.image else "host"},
    }

    # Hand review of every flagged pair (reading both versions' source).
    hr_path = os.path.join(args.out_dir, "hand_review.csv")
    if os.path.isfile(hr_path):
        hr = {(r["package"], r["prev"], r["curr"]): r for r in csv.DictReader(open(hr_path))}
        conf, strict, missing, reasons = {}, {}, 0, Counter()
        for c in mp:
            g = s_ = 0
            for r in flagged[c]:
                v = hr.get((r["package"], r["prev"], r["curr"]))
                if v is None:
                    missing += 1
                    continue
                g += v["verdict"] == "genuine"
                s_ += v["verdict"] == "genuine" and v.get("compile_break") == "yes"
                if v["verdict"] != "genuine":
                    reasons[f'{v["verdict"]}: {v.get("reason_kind") or v.get("reason", "")[:40]}'] += 1
            conf[c], strict[c] = _rate(g, len(mp[c])), _rate(s_, len(mp[c]))
        tot_g = conf["minor"]["k"] + conf["patch"]["k"]
        tot_s = strict["minor"]["k"] + strict["patch"]["k"]
        n_mp = len(mp["minor"]) + len(mp["patch"])
        crates_mp = {r["package"] for c in mp for r in mp[c]}
        crates_g = {r["package"] for c in mp for r in flagged[c]
                    if hr.get((r["package"], r["prev"], r["curr"]), {}).get("verdict") == "genuine"}
        strict_known = all(v.get("compile_break") in ("yes", "no")
                           for v in hr.values() if v["verdict"] == "genuine")
        summary["confirmed"] = {
            "hand_reviewed": len(hr), "flagged_without_review": missing,
            "genuine": {**conf, "minor_plus_patch": _rate(tot_g, n_mp)},
            "stops_code_compiling": ({**strict, "minor_plus_patch": _rate(tot_s, n_mp)}
                                     if strict_known else "not recorded in hand_review.csv"),
            "per_crate_minor_plus_patch": _rate(len(crates_g), len(crates_mp)),
            "genuine_with_0_22_lints": {
                c: _rate(sum(old_flag(r) for r in flagged[c]
                             if hr.get((r["package"], r["prev"], r["curr"]), {}).get("verdict") == "genuine"),
                         len(mp[c])) for c in mp},
            "genuine_minor_that_are_0y_patch_number_bumps": sum(
                zero_y_patch_number(r) for r in flagged["minor"]
                if hr.get((r["package"], r["prev"], r["curr"]), {}).get("verdict") == "genuine"),
            "not_genuine_reasons": dict(reasons),
        }
    summary["note"] = ("Tool reads only the public API. shipped_higher is expected (features, "
                       "behaviour changes, unlinted changes). shipped_lower is a flag; "
                       "'confirmed' counts flags that a hand review of both versions' source "
                       "found to be genuine breaks.")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
