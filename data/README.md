# CratesBump dataset

Labelled version-bump dataset for the Rust / crates.io ecosystem. Each row is one release pair
of a crate (`prev_version → version`), described by **41 trained features** (see
[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md); types and keys in [`SCHEMA.md`](SCHEMA.md)) and
labelled with the bump type (`major` / `minor` / `patch`) under Cargo's rule, with the literal
MAJOR.MINOR.PATCH label kept alongside.

## Files

Pairs = release pairs (rows). Crates = unique package count.

| File | Pairs | Crates | Description |
|---|---:|---:|---|
| **`cratesbump.csv.gz`** | **2,079,790** | **219,955** | **Headline dataset.** 48 columns: the 41 features in `scripts/pipeline.py` `FEATURE_COLS` + `package`, `prev_version`, `version`, `release_type` (Cargo label), `release_type_naive` (literal MAJOR.MINOR.PATCH label), `created_at`, `repository`. Built by `make rebuild-full DUMP_DIR=<dump>`; build reports in `results/cargo_label_report.json` and `results/github_fill_report.json`. |
| `cratesbump_extracted.csv.gz` | 2,421,308 | 221,027 | Uncapped extractor output (one extractor version for every crate), naive labels, no GitHub features. Input to `scripts/label_pairs_cargo.py`. |
| `compliant_dataset_v3_rebuilt_cargo.csv.gz` | 495,602 | 35,354 | Earlier version: crates that reached 1.0 only, mostly built by an older extractor. |
| `compliant_dataset.csv.gz` | 701,349 | 35,569 | Uncapped extraction of an earlier version (September 2026 extractor), retained for provenance. |
| `compliant_dataset_v2_cargo.csv.gz` | 470,456 | 30,431 | Earlier version; input to the LLM-hybrid experiment (`docs/METHODOLOGY_v2.md` §1.1). |

The three `compliant_dataset*.csv.gz` files are earlier versions. Their scope, labels and
pairings predate the fixes described below; do not use them for benchmarks.

All files are gzipped CSVs.

## Label distribution (headline file)

| Label | patch | minor | major |
|---|---:|---:|---:|
| **Cargo (`release_type`)** | **9.32%** | **55.23%** | **35.45%** |
| Literal MAJOR.MINOR.PATCH (`release_type_naive`) | 64.26% | 32.95% | 2.79% |

Cargo treats the leftmost non-zero component as the breaking one, so `0.1.0 → 0.2.0` is
`major` and `0.1.0 → 0.1.1` is `minor`; for `0.0.z` every change is `major`. The two labels
disagree on **1,712,585 pairs (82.34%)**, which is exactly the set of pairs whose new version is
below 1.0.0:

| Mechanism | Pairs | % of disagreements |
|---|---:|---:|
| `0.y.z` patch-position bump → Cargo `minor` | 1,033,349 | 60.3% |
| `0.y.z` minor-position bump → Cargo `major` | 555,772 | 32.5% |
| `0.0.z` (every change) → Cargo `major` | 109,239 | 6.4% |
| `0.0.z → 0.y.z` → Cargo `major` | 14,225 | 0.8% |

58,026 pairs are `major` under both labels; 569,997 pre-1.0 pairs are `minor` under the literal
label and `major` under Cargo's.

## Provenance

- **Source**: the crates.io database dump of **2026-10-12** (349,979 crates; local copy
  `.dump_cache/2026-10-12-020016`), used for crate selection, release lists, predecessors,
  publish timestamps and the dependency graph.
- **Crate selection** (`scripts/build_crate_list.py`): every crate with at least two non-yanked,
  parseable releases published before 2026-09-14: **225,799 crates (64.5%)**, which account for
  **99.4%** of all crates.io downloads (`results/scope_coverage.json`). There is no stability
  filter: 84% of these crates never had a stable 1.0 release. 219,955 contribute ≥ 1 pair; the
  rest have no forward bump after pairing, have a tarball the crates.io CDN no longer serves (139, HTTP 403; `results/extraction_report.json`), or lost all
  pairs to the family cap (1,072).
- **Feature extraction**: the Rust extractor (`src/bin/batch.rs` + `src/extractor/*`) downloads
  each version's published `.crate` tarball from the crates.io CDN and computes file/comment
  diffs, a `syn` AST diff, a `Cargo.toml` dependency diff, and the inter-release time gap.
  The release list comes from the dump (`scripts/build_snapshot_manifest.py`).
- **Predecessor rule** (`src/extractor/mod.rs::plan_pairs`, mirrored in
  `scripts/semver_order.py`): walk a crate's non-yanked versions in publish order, ordered by
  full SemVer precedence. A version above the running maximum with a higher MAJOR.MINOR.PATCH is
  paired with that maximum; one with the same MAJOR.MINOR.PATCH (`1.0.0` after `1.0.0-rc.1`)
  forms no pair but becomes the new maximum, so `1.0.1` is paired with `1.0.0`; backports are
  skipped. 2.58% of pairs have a pre-release endpoint.
- **Verification against the dump**: `scripts/label_pairs_cargo.py --check-rdtd` rebuilds each
  predecessor from the dump with the same rule and keeps a row only if that predecessor
  reproduces the naive label the extractor wrote and the gap to it equals the extractor's `rdtd`.
  All 2,421,308 extracted rows pass. It writes both labels, `created_at` (UTC publish time of
  `version`) and `repository` for every row.
- **Date cut**: the file keeps only releases published before **2026-09-14T00:00Z** (i.e.
  through 13 September 2026); the version list handed to the extractor is cut at that instant.
- **Caps**: at most 100 pairs per crate and 5,000 per name family (family = crate name up to its
  first `-` or `_`; see `scripts/cap_dataset.py`), uniform random sample with seed 42 (341,516
  rows dropped). 1,738 crates sit at the 100-pair cap.
- **GitHub enrichment**: `scripts/fill_github_features.py` computes the 11 GitHub features
  (`tcm, tau, nbf, aml, kwm, kwp, kwb, poi, pci, popr, pcpr`) over the window (publish time of
  `prev_version`, publish time of `version`]: commit features from a clone of the crate's
  repository, issue/PR features from the GitHub API (batched GraphQL for small repositories,
  REST otherwise), cached under `enrich_cache/`. Pairs without GitHub data get zeros (see [`SCHEMA.md`](SCHEMA.md)).

## Loading

```python
import pandas as pd
df = pd.read_csv("data/cratesbump.csv.gz", keep_default_na=False, na_values=[""])
id_cols = ["package", "prev_version", "version", "release_type", "release_type_naive",
           "created_at", "repository"]
X = df.drop(columns=id_cols)          # the 41 features
y = df["release_type"]                # Cargo label; use release_type_naive for the literal label
groups = df["repository"].fillna("crate:" + df["package"])   # hold out whole repositories
```

## Known limitations

1. **GitHub coverage**: 84.0% of pairs have at least one non-zero GitHub feature. The other
   16.0% are all zero: the crate lists no repository (5.2%), its repository is outside GitHub
   (3.9%), its GitHub repository could not be cloned, mostly because it was deleted or made
   private (4.0%), the repository's history does not cover the release, e.g. because the crate
   moved to another repository (1.4%), or nothing happened in the window, mostly quick
   re-releases (1.5%). A zero therefore means "no data" or "no activity"; `repository` separates
   only the first two cases. GitHub features reflect each repository as crawled in
   October–November 2026. Two known bugs, to be fixed in the next release: GitHub URLs with
   an upper-case host or doubled slashes are not recognised (14 pairs of 2 crates get zeros),
   and a failed `git log` is cached as an empty history (49 repositories may be affected).
2. **Shared repositories**: crates of one repository share repository-wide GitHub features and
   release dates; split by `repository`, not by crate, or scores are inflated.
3. **Date cut**: releases published on or after 2026-09-14 are not in the file, although the
   dump is dated 2026-10-12.
4. **Caps**: the per-crate and per-family caps subsample very active crates and large crate
   families, so per-crate statistics are not population counts.

## License

The dataset files in this directory are released under **CC BY 4.0** — see [`LICENSE`](LICENSE).
The extraction/training code in the repository root is under the MIT License. Please cite the
artifact via [`CITATION.cff`](../CITATION.cff).

## Reproducing the dataset

1. Download a crates.io database dump (<https://crates.io/data-access>).
2. `make build`, then `make rebuild-full DUMP_DIR=<dump>` (needs `GITHUB_TOKEN`; about two
   days). It selects the crates (`scripts/build_crate_list.py`), writes the snapshot version list
   (`scripts/build_snapshot_manifest.py`), extracts every crate against it
   (`batch --version-manifest ... --tarball-only`, then a low-concurrency retry pass), merges
   the per-crate files (`scripts/merge_crate_csvs.py`), and runs `make reproduce-section-5`:
   `scripts/label_pairs_cargo.py ... --check-rdtd --max-date 2026-09-14 --cap-per-crate 100
   --cap-per-family 5000 --repos results/crates_all_repos.csv` (verifies predecessors against
   the dump, writes both labels, `created_at` and `repository`, applies the date cut and the
   caps, writes `results/cargo_label_report.json`), then `scripts/fill_github_features.py
   --crates ALL` (writes `results/github_fill_report.json`).
