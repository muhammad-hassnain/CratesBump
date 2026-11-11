# CratesBump: A Version-Bump Dataset for the Rust Ecosystem — paper summary

**One-sentence pitch.** A reproducible dataset of 2,079,790 consecutive
release pairs from 219,955 crates, covering every crates.io crate with at
least two releases (99.4% of downloads), with 41 features, the verified
predecessor and source repository of each pair, and both the literal and the
Cargo label, plus the tool that rebuilds it.

## Contributions

1. **Dataset** (`data/cratesbump.csv.gz`, 48 columns): every crate in the
   2026-10-12 crates.io dump with ≥ 2 non-yanked releases before
   2026-09-14T00:00Z; one extractor version; pairs formed by full SemVer
   precedence; every pair's predecessor and days-since-release checked against
   the dump (all 2,421,308 extracted pairs pass); capped at 100 pairs per crate
   and 5,000 per name family. 84% of the crates never had a stable 1.0
   release, so the Cargo and literal labels differ on 82.3% of pairs (the
   pre-1.0 ones).
2. **Tool**: `make rebuild-full DUMP_DIR=<dump>` rebuilds the dataset from a
   dump and the GitHub API (about two days).
3. **Reference usage**: a benchmark on held-out repositories, a temporal
   split, a feature-family ablation, how pre-1.0 breaking releases reach
   dependents, and an external API audit checked by hand.

## Headline numbers

All generated into `results/HEADLINE.md` by `make headline`.

- **Held-out repositories, 5 seeds** (`results/section_4_grouped_repo_5seed.json`):
  tree ensembles 0.447–0.459 macro-F1 (Hist GB 0.459 ± 0.005, MCC 0.241; XGBoost 0.457; Gradient Boosting 0.449; Random Forest 0.448 ± 0.007; LightGBM 0.447); Decision Tree 0.437, Logistic Regression 0.361; majority baseline 0.236, stratified-random 0.333. Patch releases (9% of pairs) are the hardest class (F1 about 0.17).
- **Caution — held-out crates** (`results/section_4_grouped_crate_5seed.json`):
  tuned Random Forest 0.461 ± 0.003 vs 0.448 on held-out repositories; with
  unbounded tree depth (the earlier defaults, `results/original_params/`) 0.612 vs 0.455. Crates of one
  repository share its GitHub features and release dates, so a crate-level
  split overstates accuracy, the more so the more a model can memorise;
  split by `repository`.
- **Temporal split** (cutoff 22 June 2026, by repository; 927,421 train /
  167,154 test pairs): tree models 0.399–0.423 (Random Forest 0.423).
- **Feature families** (Random Forest, baseline 0.451): code change types
  +0.029, dependencies +0.014, complexity and size +0.012, release cadence
  +0.011, GitHub commits +0.008, issues/PRs +0.004.
- **Dependents**: of 569,997 pre-1.0 releases that read as minor but are
  major under Cargo, 26,558 dependency relations would have resolved to the
  breaking release vs 1,174,070 that stayed put (2.2%); exposure comes from
  requirements that opt out of the caret default (`^0`/`~0`/`0.*` 58%, open
  `>=` 24%, `*` 15%).
- **API audit** (450 pairs, `cargo-semver-checks` 0.51.0, every flag checked
  by hand): confirmed public-API breaks in 15 of 101 checked minor (14.9%,
  95% CI 9.2–23.1%) and 9 of 103 checked patch releases (8.7%, 4.7–15.8%);
  combined 11.8%. 92 of the 101 checked minor releases (and 13 of the 15
  breaks) are `0.y.z` patch-number bumps. Without pre-releases: 14.3% / 9.0%.
  Restricted to the 45 lints of version 0.22.0 (current in mid-2023): 12.9% /
  6.8% (Nowak & Gruevski: 3.2% on the top 1,000 library crates, with a
  different setup; we have not tested which difference explains the gap).
  Unchecked: 71 pairs without a library the tool can read (46 command-line
  only, 25 macro crates) and 74 that no longer build, resolve or download
  (eight only because the tool turns on conflicting features); 59% of
  library releases before 2020 were checkable, 82% from 2020 on
  (`results/paper_numbers.json`).
- **Internal probe**: 157,284 pairs (7.6%) are minor or patch despite a
  deletion signal.

Hyperparameters (`scripts/pipeline.py` `DEFAULT_PARAMS`) come from
`scripts/tune.py`: 30 random configurations per model, 3-fold
cross-validation grouped by repository, on a sample of 500,016 pairs from
32,760 whole repositories (seed 42; `results/tuning_results.json`). Per
model, three candidates were scored on the same folds (the search's best,
the settings from an earlier 160,002-pair search, and the original May
2026 defaults) and the one with the highest inner-CV macro-F1 was adopted;
test scores played no part. The search's best won for every model (for
Random Forest and Decision Tree it equals the earlier search's settings).
The tuning sample overlaps the test repositories, but the best tuned model
scores no higher than the best untuned one (both 0.459).

## LLM hybrid (earlier version)

On an earlier version (crates that reached 1.0 only), mixing a few-shot LLM
into the ML models added at most +0.034 macro-F1 (Hist GB; +0.033 with XGBoost) and never helped the Random
Forest (`docs/METHODOLOGY_v2.md` §1.1). Not rerun on this release.

## Reproducibility

- `make reproduce-section-3 DUMP_DIR=<dump>` — crate list and scope coverage
- `make rebuild-full DUMP_DIR=<dump>` — extract every crate and rebuild the
  headline file
- `make reproduce-section-4` — held-out repositories, held-out crates,
  temporal split
- `make reproduce-section-5 DUMP_DIR=<dump>` — predecessors, labels, caps,
  `repository`, GitHub features
- `make reproduce-section-6 DUMP_DIR=<dump>` — ablation, internal probe,
  dependents, API audit (hand-review verdicts in
  `results/label_audit/hand_review.csv`)
- `make paper-numbers` — GitHub-zero breakdown, known bugs, audit disclosures
  (`results/paper_numbers.json`)
- `make headline` — regenerate `results/HEADLINE.md`

## Known limitations

- Hyperparameters were tuned on a 500,016-pair sample of whole
  repositories, not on the full file.
- 16.0% of pairs have no GitHub signal (no repository 5.2%, other host 3.9%,
  deleted or private GitHub repository 4.0%, nothing in the window 2.9%);
  GitHub features reflect the repositories as crawled in October–November
  2026. Two known bugs, to be fixed in the next release, add zeros: unusual
  GitHub URLs (14 pairs) and unchecked `git log` failures (at most 49
  repositories).
- Capped crates contribute a sample of their history; releases after
  2026-09-14 are not included.
- The dependents measurement is static; the audit covers library crates that
  still build and sees only public-API changes.
