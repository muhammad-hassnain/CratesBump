# Data dictionary — CratesBump release dataset

Each **row** is one *release pair* of a crate (`prev_version → version`),
labelled with the SemVer bump type of that transition. The CSV starts with
`package`, `prev_version`, `version`, `release_type`; the 41 feature
columns follow in the order of `scripts/pipeline.py` `FEATURE_COLS`;
`created_at`, `release_type_naive` and `repository` come last. `data/SCHEMA.md` gives
types, ranges and join keys.

## Headline dataset — `data/cratesbump.csv.gz`

- **2,079,790 release pairs** across **219,955 crates**, **48 columns**:
  41 trained features (`scripts/pipeline.py` `FEATURE_COLS`) + `package`,
  `prev_version`, `version`, `release_type`, `release_type_naive`,
  `created_at`, `repository`.
- **Scope**: every crate in the crates.io dump of 2026-10-12 with at least
  two non-yanked releases published before **2026-09-14T00:00Z**: 225,799
  of 349,979 crates (64.5%), 99.4% of all crates.io downloads. No stability
  filter: 84% of these crates never had a stable 1.0 release
  (`results/scope_coverage.json`).
- **41 trained features**: 20 AST change-type (including the move/rename
  counters `mom`, `mnc`, `mlm`) + 4 dependency + 5 complexity + 1 time +
  7 GitHub commit activity/keyword + 4 GitHub issue/PR.
- **Labels**: `release_type` is the Cargo label; `release_type_naive` is the
  literal MAJOR.MINOR.PATCH comparison. They disagree on 1,712,585 pairs
  (82.3%), exactly the pairs whose new version is below 1.0.0.
- At most 100 pairs per crate and 5,000 per name family.

The dataset is the output of this pipeline (`make rebuild-full`):

1. `scripts/build_crate_list.py` selects the crates (above) and writes
   their repository URLs; `scripts/build_snapshot_manifest.py` lists their
   non-yanked versions published before 2026-09-14.
2. `src/bin/batch.rs` + `src/extractor/*` (one extractor version for every
   crate) downloads each crate's `.crate` tarballs from the crates.io CDN,
   pairs that version list with `plan_pairs` (below), diffs each pair with
   `syn`, and emits the AST / dependency / complexity / time features and
   the naive label. `scripts/merge_crate_csvs.py` merges the per-crate
   files into `data/cratesbump_extracted.csv.gz` (2,421,308 pairs from
   221,027 crates; for 139 more crates the crates.io CDN no longer serves a needed tarball, see `results/extraction_report.json`).
3. `scripts/label_pairs_cargo.py --check-rdtd` rebuilds every pair's
   predecessor from the dump with the same rule and keeps a row only if
   that predecessor reproduces the extractor's naive label and the gap to
   it equals the extractor's `rdtd`. All 2,421,308 extracted rows pass. It
   writes both labels, `created_at` and `repository`, applies the date cut
   and the caps (341,516 rows dropped by the caps), and writes
   `results/cargo_label_report.json`.
4. `scripts/fill_github_features.py` computes the 11 GitHub features for
   every row with a GitHub repository: commit features from a clone of the
   repository, issue/PR features from the GitHub API (batched GraphQL for
   small repositories, REST otherwise, GraphQL paging for very large ones),
   cached under `enrich_cache/`; report in `results/github_fill_report.json`.

## Identifier, label and repository columns

| Column | Type | Description |
|---|---|---|
| `package`            | string | Crate name on crates.io (`crates.name` in the dump). |
| `prev_version`       | string | Predecessor version of the pair (`versions.num`). Chosen by `src/extractor/mod.rs::plan_pairs`: walk the crate's non-yanked versions in publish order, ordered by full SemVer precedence (pre-release tags included); a version above the running maximum with a higher MAJOR.MINOR.PATCH is paired with that maximum; one with the same MAJOR.MINOR.PATCH (`1.0.0` after `1.0.0-rc.1`) forms no pair but becomes the new maximum, so `1.0.1` pairs with `1.0.0`; backports are skipped. 2.58% of pairs have a pre-release endpoint. |
| `version`            | string | Newer version of the pair, e.g. `1.4.0`. |
| `release_type`       | string | **Cargo label** `major` / `minor` / `patch`: the leftmost non-zero component is the breaking one, so for `0.y.z` a `y` bump is `major` and a `z` bump is `minor`, and for `0.0.z` every change is `major`. Distribution 35.45% / 55.23% / 9.32%. |
| `release_type_naive` | string | Literal MAJOR.MINOR.PATCH label (which component changed). Distribution 2.79% / 32.95% / 64.26%. |
| `created_at`         | string | ISO-8601 UTC publish timestamp of `version`, from the crates.io dump. Used by `--split temporal`. |
| `repository`         | string | Normalised source repository from the crate's crates.io metadata (`github.com/owner/repo`, or host/path for other hosts); empty (94.77% non-empty) if the crate lists none. Crates of one repository share repository-wide GitHub features, so hold out whole repositories when splitting (`scripts/evaluate.py --group-by repo`). |

## Feature columns (41, all trained)

Source key: **AST** = `syn` AST + file/comment diff of the two crate tarballs
· **DEP** = `Cargo.toml` diff · **TIME** = crates.io publish timestamps
· **GH-C** = clone of the crate's GitHub repository · **GH-I** = GitHub issue/PR lists.

**Coverage** is the share of rows where the value is non-zero in
`data/cratesbump.csv.gz`.

### AST change-type features (20) — source: AST

| Column | Name | Description | Coverage |
|---|---|---|---|
| `arf`  | rust files added    | Rust source files added           | 15.9% |
| `mrf`  | rust files modified | Rust source files modified        | 61.5% |
| `drf`  | rust files deleted  | Rust source files deleted         |  5.5% |
| `anrf` | non-rust added      | Non-Rust files added              |  9.5% |
| `mnrf` | non-rust modified   | Non-Rust files modified (every release changes `Cargo.toml`) | 100%  |
| `dnrf` | non-rust deleted    | Non-Rust files deleted            |  3.4% |
| `adm`  | methods added       | Functions/methods added           | 30.1% |
| `dem`  | methods deleted     | Functions/methods deleted         | 13.8% |
| `mom`  | methods moved       | Functions with an identical body that moved to a different module |  5.8% |
| `mnc`  | method names changed| Functions with an identical body renamed within the same module |  2.2% |
| `mpc`  | method params changed| Methods whose parameter list changed |  6.0% |
| `mpd`  | method params deleted| Methods whose parameters were deleted |  2.2% |
| `mla`  | logic added         | Logic blocks added inside methods | 29.8% |
| `mlm`  | logic moved         | Change in logic-block count inside functions kept under the same name (sum of absolute deltas) | 35.9% |
| `mld`  | logic deleted       | Logic blocks deleted inside methods |  5.7% |
| `gva`  | globals added       | Global vars/consts/statics added  |  5.0% |
| `gvd`  | globals deleted     | Global vars/consts/statics deleted|  1.6% |
| `icc`  | comments added      | Code-comment lines added          | 35.8% |
| `dcc`  | comments deleted    | Code-comment lines deleted        | 13.9% |
| `mcc`  | comments modified   | Code-comment lines modified, `min(added, deleted)` | 10.3% |

### Dependency features (4) — source: DEP

| Column | Name | Description | Coverage |
|---|---|---|---|
| `tcct` | dependencies changed | 1 if a dependency section of `Cargo.toml` changed, else 0 (the version bump alone does not count) | 55.1% |
| `pa`   | packages added      | Dependencies added (`[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`) | 12.3% |
| `pd`   | packages deleted    | Dependencies removed (same sections) |  5.3% |
| `pu`   | packages updated    | Dependencies whose version requirement changed | 48.3% |

### Complexity / code-size features (5) — source: AST

| Column | Name | Description | Coverage |
|---|---|---|---|
| `acycd`| avg cyclomatic Δ    | Change in average cyclomatic complexity (\|Δ\|) | 41.9% |
| `cycd` | total cyclomatic Δ  | Change in total cyclomatic complexity (\|Δ\|)   | 42.2% |
| `clcrd`| net LOC Δ           | `la − ld`                          | 66.1% |
| `la`   | lines added         | Lines added                        | 100%  |
| `ld`   | lines deleted       | Lines deleted                      | 100%  |

### Time (1) — source: TIME

| Column | Name | Description | Coverage |
|---|---|---|---|
| `rdtd` | release time delta | Days since `prev_version` was published | 100% |

### GitHub commit features (7) — source: GH-C

Window: after the publish time of `prev_version`, up to and including that of `version`. Zero when there is no GitHub data for the pair or nothing happened in the window (see Known issues).

| Column | Name | Description | Coverage |
|---|---|---|---|
| `tcm` | total commits        | Commits in the release window         | 83.7% |
| `tau` | total authors        | Unique commit authors in the window   | 83.7% |
| `nbf` | bug-fix commits      | Commits whose subject has a word starting with fix/bug/defect/error/issue | 50.3% |
| `aml` | avg subject length   | Average commit-subject (first line) length, chars | 83.7% |
| `kwm` | keyword "major"      | Commits whose subject contains "major"|  1.3% |
| `kwp` | keyword "patch"      | Commits whose subject contains "patch"|  6.6% |
| `kwb` | keyword "break"      | Commits whose subject contains "break"|  4.0% |

### GitHub issue / PR features (4) — source: GH-I

Counts over the same window (not percentages).

| Column | Name | Description | Coverage |
|---|---|---|---|
| `poi`  | opened issues | Issues opened during the window  | 28.0% |
| `pci`  | closed issues | Issues closed during the window  | 26.3% |
| `popr` | opened PRs    | PRs opened during the window     | 46.7% |
| `pcpr` | closed PRs    | PRs closed (merged or not) during the window | 46.8% |

**Rows with at least one non-zero GitHub feature: 84.0% of all pairs, 92.4% of
the pairs whose crate has a GitHub repository.**

## Known issues

- **GitHub coverage**: 16.0% of pairs have all 11 GitHub features at zero.
  The crate lists no repository (5.2%) or one outside GitHub (3.9%); its
  GitHub repository could not be cloned, mostly because it was deleted or
  made private (4.0%, 7,226 repositories); the repository's history does not
  cover the release, e.g. because the crate moved (1.4%); or nothing happened
  in the window, mostly quick re-releases (1.5%). A zero can therefore mean
  "no data" or "no activity". GitHub features reflect each repository as
  crawled (October–November 2026): deleted repositories, rewritten history
  and closed issue trackers change them on a rebuild.
- **Two known bugs in the GitHub step** (to be fixed in the next release):
  `owner_repo()` in `scripts/enrich_github.py` does not recognise a GitHub
  URL with an upper-case host or doubled slashes, so 14 pairs of 2 crates
  have zero GitHub features; and `repo_commits()` does not check whether
  `git log` failed, so a failed read is cached as an empty history (49
  repositories with an empty history and no logged error may be affected).
- **Shared repositories**: crates of one repository (monorepos, generated
  crate families) share repository-wide GitHub features and release dates.
  Splitting by crate instead of by repository inflates scores (see
  `results/HEADLINE.md`).
- **Date cut**: the file stops at releases published before
  2026-09-14T00:00Z. The version list handed to the extractor is cut at that
  instant, so every crate is cut at the same point even though the dump is
  dated 2026-10-12.
- **Caps**: 1,738 crates sit at the 100-pair cap and the largest name
  families at the 5,000-pair cap; capped crates are a uniform random sample
  (seed 42) of their pairs, not a contiguous history. 1,072 crates of capped
  families lost all their pairs to the family cap.

## Dataset files (data/)

| File | Rows | Purpose |
|---|---:|---|
| `cratesbump.csv.gz` | 2,079,790 | **Headline dataset** (this document). |
| `cratesbump_extracted.csv.gz` | 2,421,308 | Uncapped extractor output with naive labels; input to `scripts/label_pairs_cargo.py`. |
| `compliant_dataset_v3_rebuilt_cargo.csv.gz` | 495,602 | Earlier version: crates that reached 1.0 only, built by an older extractor. |
| `compliant_dataset.csv.gz` | 701,349 | Uncapped extraction of an earlier version, kept for provenance. |
| `compliant_dataset_v2_cargo.csv.gz` | 470,456 | Earlier version; input to the LLM-hybrid experiment. |

The three earlier versions (`compliant_dataset*.csv.gz`) have labels and
pairings that predate the fixes above; do not use them for benchmarks.
