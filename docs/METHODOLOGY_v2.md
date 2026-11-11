# Methodology — CratesBump

A concise statement of the data-collection and evaluation protocol used for
the paper. Everything below is reproducible from the repository; the
`Makefile` targets and the referenced scripts form the artifact. Headline
numbers are in `results/HEADLINE.md` (generated from the result files by
`make headline`).

## 1. Scope

A version-bump dataset and classifier benchmark for the Rust / crates.io
ecosystem. The paper reports:

- a reproducible collection pipeline (`scripts/build_crate_list.py`,
  `scripts/build_snapshot_manifest.py`, `src/extractor/*`,
  `scripts/label_pairs_cargo.py`, `scripts/fill_github_features.py`);
- Cargo-rule labels next to the literal MAJOR.MINOR.PATCH labels, with every
  pair's predecessor verified against the crates.io dump;
- ML baselines on held-out repositories and on a temporal split
  (`scripts/pipeline.py`, `scripts/evaluate.py`);
- a feature-family ablation (`scripts/feature_ablation.py`), a measurement of
  how pre-1.0 breaking releases reach dependents
  (`scripts/downstream_impact.py`), an internal label-consistency probe
  (`scripts/label_sanity_check.py`) and an external API audit checked by
  hand (`scripts/label_audit_cargo_semver_checks.py`,
  `scripts/audit_flag_context.py`).

### 1.1 A few-shot LLM hybrid (earlier version)

On an earlier version of the dataset (`data/compliant_dataset_v2_cargo.csv.gz`,
crates that reached 1.0 only; 2,000 test / 800 validation pairs, grouped by
crate), a hybrid `p = α · p_ML + (1 − α) · p_LLM` of an ML model and a
few-shot LLM (six retrieved examples; the hosted model behind
`langchain-typesafe` plus `deepseek-r1:8b`, `gemma2:9b`, `llama3.1:8b`,
`mistral-nemo:12b`, `qwen2.5:7b` via Ollama; four ML backbones; 24 sweeps)
added at most +0.034 macro-F1 (hosted LLM with Hist GB, 95% CI
[0.014, 0.053]; with XGBoost +0.033, [0.014, 0.052]), gave the LLM zero weight
in 10 of 24 sweeps, and never helped the Random Forest (0.641 alone on that
release). The LLM alone reached at most 0.553. The experiment was not rerun on
the current release. Pipeline, prompts and cached responses:
`scripts/jev_optimized.py`, `jev_cache/`, `results/jev_v2cargo_*.json`.

## 2. Dataset

### 2.1 Source and scope

A **crates.io database dump** of 2026-10-12 (local copy
`.dump_cache/2026-10-12-020016`), used for crate selection, release lists,
predecessors, publish timestamps and the dependency graph. The snapshot
instant is **2026-09-14T00:00Z**.

`scripts/build_crate_list.py` selects every crate with at least two
non-yanked, parseable releases published before the snapshot: **225,799 of
349,979 crates (64.5%)**, accounting for **99.4% of all crates.io
downloads**. There is no stability filter, because Cargo's rule versions
`0.y.z` releases too: 84% of these crates never had a stable 1.0 release and
they carry 62% of all downloads (`results/scope_coverage.json`). The
previous version of the dataset kept only crates with a stable ≥ 1.0.0
release; choosing crates by "1.0+ is SemVer" while labelling by Cargo's rule
was inconsistent and excluded crates such as `libc`, `log` and `futures`.

### 2.2 Release pairs

`scripts/build_snapshot_manifest.py` lists each crate's non-yanked versions
before the snapshot; `src/bin/batch.rs` downloads every `.crate` tarball from
the crates.io CDN and `src/extractor/mod.rs::plan_pairs` forms the pairs:
walk the versions in publish order, ordered by **full SemVer precedence**
(pre-release tags included). A version above the running maximum with a
higher MAJOR.MINOR.PATCH is paired with that maximum; a version above the
maximum with the same MAJOR.MINOR.PATCH (`1.0.0` after `1.0.0-rc.1`) forms no
pair, since it has no bump class, but becomes the new maximum, so `1.0.1` is
compared with `1.0.0`; backports (below the maximum) are skipped.
`scripts/semver_order.py` is the Python mirror used by labelling, the audit
and the dependents analysis. 2.58% of pairs have a pre-release endpoint.

An earlier rule ignored pre-release tags when ordering, so `1.0.1` was
compared with `1.0.0-rc.1` and the diff spanned two releases (10,629 pairs of
the previous version, 2.1%); one of the previous audit's flagged
"violations" was such a pair.

### 2.3 Features

41 features per pair: 20 AST change-type counts (`syn` diff of every `.rs`
file, including the move/rename counters `mom`, `mnc`, `mlm`), 4 dependency
(`Cargo.toml` diff), 5 complexity/size, 1 time (days since `prev_version`),
7 GitHub commit and 4 GitHub issue/PR features. The feature design follows
Abdalkareem et al. (2022) for npm, adapted to Rust. Every crate is extracted
with one extractor version (`make rebuild-full`; the extractor is
deterministic: two runs on the same input give identical rows).

GitHub features (`scripts/fill_github_features.py`) are counted in the window
(publish time of `prev_version`, publish time of `version`]: commits from a
clone of the crate's repository, issues and pull requests from the GitHub
API (batched GraphQL for small repositories, REST otherwise, GraphQL paging
for very large ones). They reflect the repositories as crawled in
October–November 2026. 84.0% of pairs have at least one non-zero GitHub feature
(92.4% of pairs whose crate has a GitHub repository).

### 2.4 Verification, labels, caps

`scripts/label_pairs_cargo.py --check-rdtd` rebuilds each pair's
predecessor from the dump with the same rule and keeps a row only if that
predecessor reproduces the naive label the extractor wrote **and** the gap to
it equals the extractor's `rdtd`, so the features provably describe that
pair. All 2,421,308 extracted pairs pass. It writes both labels,
`created_at`, and `repository` (the normalised source repository, for
grouped splits), cuts at the snapshot, and caps the file at 100 pairs per
crate and 5,000 per name family (uniform sample, seed 42; 341,516 rows
dropped; 1,738 crates at the per-crate cap).

**Headline file**: `data/cratesbump.csv.gz`, **2,079,790 pairs from 219,955
crates**, 48 columns. Build report: `results/cargo_label_report.json`.

## 3. Labels

- `release_type_naive`: the literal MAJOR.MINOR.PATCH comparison.
  Distribution 2.79% major / 32.95% minor / 64.26% patch.
- `release_type`: the **Cargo label** (the leftmost non-zero component is
  breaking: for `0.y.z` a `y` bump is major and a `z` bump minor; for `0.0.z`
  every change is major). Distribution 35.45% / 55.23% / 9.32%.

The two labels differ on **1,712,585 pairs (82.3%)**, exactly the pairs whose
new version is below 1.0.0: this is a property of the ecosystem (most
releases are pre-1.0), not a finding. Mechanisms: `0.y.z` patch-position
bump → Cargo minor 1,033,349 (60.3%); `0.y.z` minor-position bump → Cargo
major 555,772 (32.5%); `0.0.z` → major 109,239 (6.4%); `0.0.z → 0.y.z`
14,225 (0.8%).

### 3.1 Construct validity

Both labels are *declared* bumps, read from version strings.

- **Internal probe** (`scripts/label_sanity_check.py`): 157,284 pairs (7.6%)
  are minor or patch even though the extractor saw a deletion signal
  (`dem, gvd, drf, mpd, mpc`), and 521,681 are majors without one (expected:
  most are pre-1.0 Cargo majors). Source:
  `results/label_sanity_rebuilt_cargo_summary.json`.
- **External audit** (`scripts/label_audit_cargo_semver_checks.py`): 450
  pairs (150 per Cargo label, at most one per crate, seed 42; every sampled
  pair matches the dump's pairing), `cargo-semver-checks` 0.51.0 with
  `--release-type patch`, so its 215 deny-level lints (of 260;
  `results/label_audit_lints_0.51.0.csv`) report the bump the public-API
  change requires. Every cargo command runs in a Docker sandbox
  (`docker/audit.Dockerfile`: non-root, no capabilities, no network during
  builds, a fresh cargo cache per attempt). The tool checked 305 pairs. Of
  the other 145, 71 have no library target it can read (46 contain only
  command-line programs; 25 are macro crates, which it does not support), and
  74 no longer build, resolve their dependencies or, in one case, download;
  eight of these fail only because the tool turns on conflicting
  features. The unchecked pairs lean older: 59% (17/29) of library releases
  from before 2020 could be checked, 82% (288/350) from 2020 on
  (`results/paper_numbers.json`, `make paper-numbers`). It flagged 21 of
  101 minor and 10 of 103 patch releases; all stay flagged with
  `--default-features`. **Every flag was checked by reading both versions'
  source** (`results/label_audit/hand_review.csv`, context in
  `flag_context.csv`, full tool output in `logs/`): 24 genuine breaks, 6
  harmless in practice (generated bindings, libraries of CLI tools with no
  dependents, an extension trait, a `Copy` derive, inferred generic
  parameters), 1 not a break (a cross-crate re-export the tool cannot see).
  **Confirmed: 15/101 minor (14.9%, Wilson 95% CI 9.2–23.1%) and 9/103
  patch (8.7%, 4.7–15.8%)**; in 20 of the 24 ordinary downstream code stops
  compiling. 92 of the 101 checked minor releases (and 13 of the 15 breaks)
  are `0.y.z` releases that only moved the PATCH number, so the minor rate
  mostly describes pre-1.0 crates. One break is a pre-release
  (`tokmd-settings 1.8.0-rc.1`), which SemVer exempts; without pairs that
  have a pre-release end the rates are 14.3% (14/98) and 9.0% (9/100).
  Minor and patch combined: 24/204 = 11.8%. Nowak and Gruevski (2023) found
  violations in 3.2% of minor and patch releases of the 1,000
  most-downloaded library crates, using an early version of
  cargo-semver-checks (the post names no version; 0.22.0, with 45 lints, was
  current in mid-2023); restricted to the 0.22.0 lints
  (`results/label_audit_lints_0.22.0.csv`) our confirmed rates are 12.9%
  (minor) and 6.8% (patch), so the newer lints explain little of the gap.
  Their study also used default features only, confirmed each break with
  failing code, and checked every release of popular crates; we have not
  tested which difference matters. The previous version's audit, with its own hand review, is in
  `results/label_audit_prev_release/` (confirmed 13/123 = 10.6%; its flag
  rate of 19.5% overstated the confirmed one about twofold).

## 4. Evaluation protocol

### 4.1 Splits

- **Held-out repositories (headline)**: `GroupShuffleSplit(test_size=0.2)`
  over source repositories (`repository` column; crates without one are
  their own group), 5 seeds (42–46). No test crate shares a repository with
  a training crate.
- **Held-out crates** (caution only): the same split over crates. Crates of
  one repository share repository-wide GitHub features and release dates,
  so this split lets a model recognise a repository it has seen; it
  inflates scores, the more so the more a model can memorise: the tuned
  Random Forest scores 0.461 vs 0.448 held-out repositories, with unbounded
  tree depth (the earlier defaults; `results/original_params/`) 0.612 vs 0.455.
- **Temporal**: cutoff at the 80th percentile of release times (22 June
  2026); a repository goes to training if all its pairs are at or before the
  cutoff and to testing if all are after it; repositories that straddle it
  are dropped (47% of pairs). 927,421 training and 167,154 test pairs; one
  deterministic run.

### 4.2 Models and metrics

Majority and stratified-random baselines (`DummyClassifier`), XGBoost,
LightGBM, Random Forest, Hist Gradient Boosting, Gradient Boosting, Decision
Tree, Logistic Regression. SMOTE on the training fold only (computed once
per fold). Metrics: macro-F1, MCC, per-class F1, one-vs-rest macro AUC.
Hyperparameters are `scripts/pipeline.py` `DEFAULT_PARAMS`, from
`scripts/tune.py`: 30 random configurations per model, 3-fold
`StratifiedGroupKFold` grouped by repository, SMOTE inside each fold, on a
sample of 500,016 pairs from 32,760 whole repositories (seed 42; the search
used the 397,792-pair training part). Per model, three candidates were
scored on the same inner folds: the search's best, the settings from an
earlier 160,002-pair search, and the original May 2026 defaults
(`results/tuning_candidates_original_defaults.json`). The one with the
highest inner-CV macro-F1 was adopted (`results/tuning_results.json`,
fields `candidates`, `adopted`): the search's best for every model, equal to
the earlier search's settings for Random Forest and Decision Tree. Test
scores played no part in the choice. The tuning sample overlaps the test
repositories of the evaluation splits, but the best tuned model scores no
higher than the best untuned one (both 0.459 macro-F1: Hist GB tuned,
XGBoost with the original settings; `results/original_params/`). The search runs five single-threaded
fits at a time (5 h 18 min).

### 4.3 Results

- Held-out repositories (5 seeds): tree ensembles 0.447–0.459 macro-F1 (Hist GB 0.459 ± 0.005, MCC 0.241; XGBoost 0.457; Gradient Boosting 0.449; Random Forest 0.448 ± 0.007; LightGBM 0.447); Decision Tree 0.437, Logistic Regression 0.361; majority baseline 0.236, stratified-random 0.333. Patch releases (9% of pairs) are the hardest class (F1 about 0.17). With the original May 2026 settings the ensembles scored 0.455–0.459, with the 160,002-pair search's 0.443–0.451.
- Temporal: tree models 0.399–0.423 (Random Forest 0.423), Logistic
  Regression 0.278; majority baseline 0.243.
- Feature families (Random Forest, first held-out-repository fold, baseline
  macro-F1 0.451, five block permutations): code change types +0.029,
  dependencies +0.014, complexity and size +0.012, days since the previous
  release +0.011, GitHub commits +0.008, GitHub issues/PRs +0.004. With
  unbounded tree depth the GitHub families counted for more (+0.046,
  +0.027; `results/original_params/`).

Sources: `results/section_4_grouped_repo_5seed.json`,
`results/section_4_grouped_crate_5seed.json`,
`results/section_4_temporal_repo.json`,
`results/feature_ablation_rebuilt_cargo_rf.json`.

## 5. Dependents of pre-1.0 breaking releases

`scripts/downstream_impact.py`: 94.9% of the dump's 33.6M normal dependency
declarations are caret requirements, which stop at the pre-1.0 breaking
boundary by construction. For each of the 569,997 releases that read as minor
but are major under Cargo, each crate whose latest release before it
declared a normal dependency on the package is checked with a full Cargo
requirement matcher: 26,558 relations would have resolved to the breaking
release vs 1,174,070 that stayed on the previous one (2.2%); 8,525 releases
(1.5%) exposed any dependent. Exposure comes from requirements that opt out
of the caret default: those admitting every `0.y.z` (`^0`, `~0`, `0.*`; 58%),
open-ended comparisons (24%) and `*` (15%). The most-exposed releases are
successive `k8s-openapi` versions (400–460 dependents each). The measurement
is static (no builds; `Cargo.lock` delays upgrades further). Source:
`results/downstream_impact.json`.

## 6. Threats to validity

- **Construct**: labels are declared bumps; the audit shows some minor and
  patch releases break the public API (§3.1), and the tool cannot see
  behavioural changes.
- **Internal**: release pairs of one repository are correlated; the headline
  split holds out whole repositories. Name families spanning several
  repositories can still split across folds. Hyperparameters were tuned
  on a 500,016-pair sample of whole repositories.
- **External**: the dataset covers crates with at least two releases before
  2026-09-14 (99.4% of downloads); capped crates contribute a sample of their
  history. GitHub features exist only for crates with a reachable GitHub
  repository (16.0% of pairs have none non-zero) and depend on the
  repositories' state when crawled.
- **Reproducibility**: `make rebuild-full DUMP_DIR=<dump>` (about two days),
  then `make reproduce-section-4`, `reproduce-section-6 DUMP_DIR=<dump>` and
  `make headline`. The dataset will be deposited at Zenodo (DOI pending).
