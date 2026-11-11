#!/usr/bin/env python3
"""
Promote the best hyperparameters from the last tuning run into pipeline.py.

Reads  results/tuning_results.json  and rewrites the DEFAULT_PARAMS block
in pipeline.py so that every future `make pipeline` run uses those params
by default — without needing to pass BEST_PARAMS=1.

A model's tuned params are promoted only if the tuning run marked it
`adopt` (its best inner-CV macro-F1 beat the current defaults on the same
folds); other models, and entries the run did not tune (the trivial
baselines), keep their current params. Results without an `adopt` field
(older runs) are promoted as before.

Usage:
    python3 upgrade_defaults.py            # apply the upgrade
    python3 upgrade_defaults.py --dry-run  # preview the new block, don't write
"""

import argparse
import json
import os
import re
import sys

PIPELINE_PATH = "scripts/pipeline.py"
JSON_PATH     = os.path.join("results", "tuning_results.json")

# The upgrade script finds the section between these two comment lines and
# replaces its content.  Do NOT rename them in pipeline.py.
MARKER_START = "# ── DEFAULT_PARAMS: auto-updated by `make upgrade` ───────────────────────────"
MARKER_END   = "# ── /DEFAULT_PARAMS ───────────────────────────────────────────────────────────"


def fv(v) -> str:
    """Format a single param value as a Python literal."""
    if isinstance(v, str):  return f'"{v}"'
    if v is None:           return "None"
    if isinstance(v, bool): return str(v)
    return repr(v)


def current_defaults(pipeline_path: str) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.abspath(pipeline_path)))
    import pipeline  # noqa: E402
    return dict(pipeline.DEFAULT_PARAMS)


def generate_block(results: list, current: dict) -> str:
    """Produce the new DEFAULT_PARAMS block as a string, in the current order."""
    tuned = {r["name"]: r for r in results}
    lines = [MARKER_START + "\n", "DEFAULT_PARAMS = {\n"]
    for name in list(current) + [n for n in tuned if n not in current]:
        r = tuned.get(name)
        if r is not None and "adopted_params" in r:
            params = r["adopted_params"]
        elif r is not None and r.get("adopt", True):
            params = r["best_params"]
        else:
            params = current.get(name, {})
        kvs = ", ".join(f'"{k}": {fv(v)}' for k, v in params.items())
        lines.append(f'    "{name}": {{{kvs}}},\n')
    lines.append("}\n")
    lines.append(MARKER_END + "\n")
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Promote tuned hyperparameters into pipeline.py DEFAULT_PARAMS"
    )
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print the new block without modifying pipeline.py")
    parser.add_argument("--json",     default=JSON_PATH,
                        help=f"Path to tuning results JSON  [{JSON_PATH}]")
    parser.add_argument("--pipeline", default=PIPELINE_PATH,
                        help=f"Path to pipeline.py  [{PIPELINE_PATH}]")
    args = parser.parse_args()

    # Load tuning results
    if not os.path.isfile(args.json):
        print(f"Error: {args.json} not found.")
        print("  Run 'make tune' first to generate tuning results.")
        sys.exit(1)

    with open(args.json) as f:
        data = json.load(f)

    results     = data["results"]
    searched_at = data.get("searched_at", "unknown")
    n_iter      = data.get("n_iter", "?")
    cv          = data.get("cv", "?")

    # Build the new block
    new_block = generate_block(results, current_defaults(args.pipeline))

    if args.dry_run:
        print(f"── Dry run — would write this block to {args.pipeline} ──\n")
        print(new_block)
        print("── Scores from this tuning run ──")
        for r in results:
            print(f"  {r['name']:<26}  CV F1: {r['cv_f1']:.4f}  "
                  f"CV MCC: {r.get('cv_mcc', 'n/a')}")
        return

    # Read pipeline.py and replace the block
    if not os.path.isfile(args.pipeline):
        print(f"Error: {args.pipeline} not found.")
        sys.exit(1)

    with open(args.pipeline) as f:
        src = f.read()

    # Match everything from the start marker line to the end marker line
    # (inclusive of both marker lines).
    pattern = re.compile(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
        re.DOTALL,
    )
    new_src, n_subs = pattern.subn(new_block.rstrip("\n"), src)

    if n_subs == 0:
        print(f"Error: DEFAULT_PARAMS markers not found in {args.pipeline}.")
        print(f"  Expected:  {MARKER_START}")
        sys.exit(1)

    with open(args.pipeline, "w") as f:
        f.write(new_src)

    # Print summary 
    print(f"Updated DEFAULT_PARAMS in {args.pipeline}")
    print(f"  Tuned at : {searched_at}")
    print(f"  Settings : {n_iter} iterations × {cv}-fold CV")
    print()
    print(f"  {'Model':<26}  {'CV F1':>7}  {'CV MCC':>7}  Best params")
    print("  " + "─" * 80)
    for r in results:
        param_str = ", ".join(f"{k}={v}" for k, v in r["best_params"].items())
        mcc_str   = f"{r['cv_mcc']:.4f}" if "cv_mcc" in r else "  n/a"
        verdict   = (f"adopted {r['adopted']}" if "adopted" in r else
                     "adopted" if r.get("adopt", True) else
                     f"kept defaults (defaults CV F1 {r.get('default_cv_f1')})")
        print(f"  {r['name']:<26}  {r['cv_f1']:>7.4f}  {mcc_str:>7}  {verdict}: {param_str}")

    print()
    print("These are now the default params for every future `make pipeline` run.")
    print("To fall back to the old defaults, use git to restore pipeline.py.")


if __name__ == "__main__":
    main()
