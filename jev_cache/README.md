# LLM response cache

This directory holds the LLM responses behind the few-shot LLM hybrid on an
earlier version of the dataset (`docs/METHODOLOGY_v2.md` §1.1), keyed on
`sha1(instructions || state)`. With these files present,
`scripts/jev_optimized.py` reproduces the published runs without calling any
LLM API.

## What's tracked

| File | Runs it serves | Size |
|---|---|---|
| `v2cargo.json` | the hosted model behind `langchain-typesafe` (`results/jev_v2cargo_<backbone>.json`, `results/jev_v2cargo_summary.json`): Cargo labels, 2,000-row test, 800-row validation slice | ~330 KB |
| `ollama_<model>.json` (deepseek-r1-8b, gemma2-9b, llama31-8b, mistral-nemo-12b, qwen25-7b) | the local-LLM runs (`results/jev_v2cargo_ollama-<model>_*.json`) | ~320 KB each |

Each cache entry is a JSON object with `pred` (string label) and `probs`
(three-class probability dict) keyed by the SHA-1 of the full instructions +
state. Keys are stable across ML backbones because the prompt does not
depend on which ML model it is paired with, so one cache file serves all four
backbones.

## How to use the cache

```bash
# Reproduce the XGBoost run (reads the cache; no API calls)
.venv/bin/python3 scripts/jev_optimized.py \
    data/compliant_dataset_v2_cargo.csv.gz \
    --split grouped --test-sample 2000 --tune-sample 800 \
    --few-shot 6 --ml-model XGBoost --enable-gate --enable-stack \
    --cache jev_cache/v2cargo.json \
    --report-json results/jev_v2cargo_XGBoost.json \
    --report-image results/jev_v2cargo_XGBoost.png
```

`make jev-run` and `make jev-sweep` run the same commands (all four
backbones for the sweep). If the cache is complete, no `TYPESAFE_API_KEY` is
required. On a cache miss (for example, a different `--seed` changes the
few-shot neighbours) the client calls the API and needs `TYPESAFE_API_KEY`.

Smoke-test and ad-hoc caches are gitignored; `make jev-smoke` runs the
pipeline with a mock LLM and needs no cache.
