SHELL := /bin/bash

# Paths 
VENV        := .venv
PYTHON      := $(VENV)/bin/python3
PIP         := $(VENV)/bin/pip3
RUST_BIN    := target/release

# Tuneable defaults (override on the command line) 
COUNT       ?= 1000
WORKERS     ?= 6
MODEL_DIR   ?= models
DATA        ?= data/cratesbump.csv.gz
N_ITER      ?= 30
TUNE_CV     ?= 3
BEST_PARAMS ?= 0

.DEFAULT_GOAL := help

# Help 
.PHONY: help
help:
	@echo ""
	@echo "CratesBump — usage"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  make setup          Full setup: Rust toolchain + Python venv + token"
	@echo "  make build          Compile Rust binaries (extract, batch)"
	@echo ""
	@echo "  make batch          Collect features for the top \$$COUNT crates by downloads"
	@echo "                        (a custom sample; the released dataset is built by"
	@echo "                        make rebuild-full)  COUNT=\$$COUNT  WORKERS=\$$WORKERS"
	@echo "  make extract C=serde  Extract features for a single crate"
	@echo ""
	@echo "  Paper reproduction (see README.md, 'Reproducing the paper'):"
	@echo "  make reproduce-section-3 DUMP_DIR=<dump>  Crate list and scope coverage"
	@echo "  make rebuild-full DUMP_DIR=<dump>         Rebuild data/cratesbump.csv.gz (~2 days)"
	@echo "  make reproduce-section-4                  Held-out repositories/crates, temporal split"
	@echo "  make reproduce-section-5 DUMP_DIR=<dump>  Predecessors, labels, caps, GitHub features"
	@echo "  make reproduce-section-6 DUMP_DIR=<dump>  Ablation, label probe, dependents, API audit"
	@echo "  make reproduce-all DUMP_DIR=<dump>        Sections 4-6, paper-numbers, then make headline"
	@echo "  make paper-numbers [DUMP_DIR=<dump>]      GitHub-zero breakdown, known bugs, audit"
	@echo "                                              disclosures -> results/paper_numbers.json"
	@echo "  make headline                             Regenerate results/HEADLINE.md"
	@echo ""
	@echo "  make demo           Quick 5-min demo: train on a 25k-row sample of the"
	@echo "                        released dataset and print the results table"
	@echo ""
	@echo "  make pipeline       Train on the released dataset (\$$DATA), save models"
	@echo "                        DATA=$(DATA)"
	@echo "                        MODEL_DIR=\$$MODEL_DIR"
	@echo "                        BEST_PARAMS=1      use tuned params from make tune"
	@echo ""
	@echo "  make tune           Hyperparameter search across all seven classifiers"
	@echo "                        N_ITER=\$$N_ITER  TUNE_CV=\$$TUNE_CV"
	@echo "                        Writes results/tuning_results.png + .json"
	@echo ""
	@echo "  make upgrade        Promote best params from make tune → pipeline.py defaults"
	@echo "                        Rewrites DEFAULT_PARAMS so every future make pipeline"
	@echo "                        uses the tuned values without needing BEST_PARAMS=1"
	@echo ""
	@echo "  make predict        Predict bump type for a specific crate+version"
	@echo "    C=<crate>           Crate name (required)"
	@echo "    V=<version>         Version to classify (default: latest published)"
	@echo ""
	@echo "  make clean          Remove target/ (cargo clean)"
	@echo ""
	@echo "  First time?  Run:  make setup && make build"
	@echo ""

# ── Setup (all-in-one) ─────────────────────────────────────────────────────────
.PHONY: setup
setup: setup-rust setup-python setup-token
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo " Setup complete."
	@echo ""
	@echo " Activate the Python environment in your shell:"
	@echo "   source $(VENV)/bin/activate"
	@echo ""
	@echo " Then build the Rust binaries:"
	@echo "   make build"
	@echo "═══════════════════════════════════════════════════════════════"

# ── Rust toolchain ─────────────────────────────────────────────────────────────
.PHONY: setup-rust
setup-rust:
	@echo ""
	@echo "── Rust toolchain ────────────────────────────────────────────"
	@command -v rustup >/dev/null 2>&1 || { \
		echo ""; \
		echo "  rustup not found. Install it with:"; \
		echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; \
		echo ""; \
		exit 1; \
	}
	@echo "  rustup found: $$(rustup --version)"
	rustup toolchain install stable --no-self-update
	rustup override set stable
	@echo "  Active toolchain: $$(rustc --version)"

# Python virtual environment 
.PHONY: setup-python
setup-python:
	@echo ""
	@echo "── Python virtual environment ────────────────────────────────"
	@command -v python3 >/dev/null 2>&1 || { \
		echo "  python3 not found. Install Python 3.11+ first."; \
		exit 1; \
	}
	@echo "  python3 found: $$(python3 --version)"
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements.txt
	@echo "  Packages installed into $(VENV)/"
	@echo ""
	@echo "  To activate in your current shell, run:"
	@echo "    source $(VENV)/bin/activate"

# GitHub token 
.PHONY: setup-token
setup-token:
	@echo ""
	@echo "── GitHub personal access token ──────────────────────────────"
	@echo "  A token is optional but enables issue/PR features and raises"
	@echo "  the rate limit from 60 to 5000 requests/hour."
	@echo ""
	@echo "  How to create one:"
	@echo "    1. Go to github.com → Settings → Developer settings"
	@echo "    2. Personal access tokens → Fine-grained tokens"
	@echo "    3. Grant 'Public Repositories (read-only)' — nothing else needed"
	@echo ""
	@if [ -f .token ] && [ -s .token ]; then \
		echo "  Token already present in .token — skipping."; \
	else \
		printf "  Paste your token (or press Enter to skip): "; \
		read -r token; \
		if [ -n "$$token" ]; then \
			echo "$$token" > .token; \
			echo "  Saved to .token"; \
		else \
			echo "  Skipped. Run 'make setup-token' again when ready."; \
		fi; \
	fi

# Build Rust binaries 
.PHONY: build
build:
	@echo ""
	@echo "── Building Rust binaries ────────────────────────────────────"
	cargo build --release
	@echo ""
	@echo "  Binaries:"
	@echo "    $(RUST_BIN)/extract    — single-crate feature extractor"
	@echo "    $(RUST_BIN)/batch      — bulk extractor (top-N crates)"

# Data collection 
.PHONY: batch
batch: $(RUST_BIN)/batch
	@printf "How many crates to collect? [default: $(COUNT)]: "; \
	read -r input; \
	count=$${input:-$(COUNT)}; \
	echo "  Collecting $$count crates (workers=$(WORKERS), all versions)…"; \
	$(RUST_BIN)/batch \
		--count   $$count \
		--workers $(WORKERS) \
		--crate-dir crate_csvs/

.PHONY: extract
extract: $(RUST_BIN)/extract
	@[ -n "$(C)" ] || { echo "Usage: make extract C=<crate-name>"; exit 1; }
	$(RUST_BIN)/extract $(C) --output crate_csvs/$(C).csv

$(RUST_BIN)/batch $(RUST_BIN)/extract: $(shell find src -name '*.rs') Cargo.toml
	cargo build --release

# Python ML pipeline 
.PHONY: pipeline
pipeline: $(PYTHON)
	$(PYTHON) scripts/pipeline.py $(DATA) --save-models --model-dir $(MODEL_DIR) \
		$(if $(filter 1,$(BEST_PARAMS)),--best-params,)

# Quick 5-minute demo: train on a subsample of the released dataset, print the
# evaluation table to the terminal, and write nothing to disk.
.PHONY: demo
demo: $(PYTHON)
	@echo "Quick demo — 25k-row sample of $(DATA), SMOTE, no files written:"
	$(PYTHON) scripts/pipeline.py $(DATA) --balance smote --sample 25000 --no-plot --seed 42

# Hyperparameter tuning 
.PHONY: tune
TUNE_SAMPLE ?= 500000
tune: $(PYTHON)
	@# Repository-grouped search on a sample of whole repositories; six
	@# single-threaded fits at a time (OMP_NUM_THREADS=1 for Hist GB).
	OMP_NUM_THREADS=1 $(PYTHON) scripts/tune.py $(DATA) --n-iter $(N_ITER) --cv $(TUNE_CV) \
	    --group-by repo --sample-rows $(TUNE_SAMPLE) --seed 42 --search-jobs 5 \
	    --candidates results/tuning_candidates_original_defaults.json

# Promote best hyperparameters -> pipeline.py defaults 
.PHONY: upgrade
upgrade: $(PYTHON)
	@[ -f results/tuning_results.json ] || { \
		echo ""; \
		echo "  Error: results/tuning_results.json not found."; \
		echo "  Run 'make tune' first to generate tuning results."; \
		echo ""; \
		exit 1; \
	}
	$(PYTHON) scripts/upgrade_defaults.py

# Prediction pipeline
.PHONY: predict
predict: $(PYTHON) $(RUST_BIN)/extract
	@[ -n "$(C)" ] || { echo "Usage: make predict C=<crate-name> [V=<version>]"; exit 1; }
	$(PYTHON) scripts/predict.py $(C) $(V)

# ── Few-shot LLM hybrid on an earlier version (docs/METHODOLOGY_v2.md §1.1) ────
# Runs on data/compliant_dataset_v2_cargo.csv.gz with the cached responses in
# jev_cache/v2cargo.json, so no API key is needed while the cache is complete.
JEV_DATA         ?= data/compliant_dataset_v2_cargo.csv.gz
JEV_SPLIT        ?= grouped
JEV_TEST_SAMPLE  ?= 2000
JEV_TUNE_SAMPLE  ?= 800
JEV_FEW_SHOT     ?= 6
JEV_ML_MODEL     ?= XGBoost
JEV_CACHE        ?= jev_cache/v2cargo.json
JEV_PREFIX       ?= jev_v2cargo_

.PHONY: jev-smoke
jev-smoke: $(PYTHON)
	$(PYTHON) scripts/jev_optimized.py $(JEV_DATA) \
		--split $(JEV_SPLIT) --test-sample 400 --tune-sample 300 \
		--few-shot 3 --ml-model $(JEV_ML_MODEL) --mock-jev

.PHONY: jev-run
jev-run: $(PYTHON)
	@[ -n "$$TYPESAFE_API_KEY" ] || [ -f .typesafe_token ] || \
		echo "Note: no TYPESAFE_API_KEY; every LLM response must come from $(JEV_CACHE)."
	SAFE=$$(echo "$(JEV_ML_MODEL)" | tr ' ' '_'); \
	$(PYTHON) scripts/jev_optimized.py $(JEV_DATA) \
		--split $(JEV_SPLIT) \
		--test-sample $(JEV_TEST_SAMPLE) --tune-sample $(JEV_TUNE_SAMPLE) \
		--few-shot $(JEV_FEW_SHOT) --ml-model "$(JEV_ML_MODEL)" \
		--enable-gate --enable-stack \
		--cache $(JEV_CACHE) \
		--report-json  results/$(JEV_PREFIX)$$SAFE.json \
		--report-image results/$(JEV_PREFIX)$$SAFE.png

# ── Paper reproduction targets (one command per headline section) ────────────
# ── §3 Dataset build (dump -> compliant -> extract -> enrich -> clean) ────────
.PHONY: reproduce-section-3
reproduce-section-3: $(PYTHON)
	@echo ""
	@echo "  §3 — Dataset coverage figures from the dump."
	@echo ""
	@[ -n "$(DUMP_DIR)" ] || { echo "  Pass DUMP_DIR=/path/to/crates.io-db-dump-YYYY-MM-DD/"; exit 1; }
	$(PYTHON) scripts/build_crate_list.py $(DUMP_DIR) 2026-09-14 \
	    results/crates_all.txt results/crates_all_repos.csv
	$(PYTHON) scripts/scope_coverage.py $(DUMP_DIR) results/crates_all.txt \
	    --out results/scope_coverage.json

# ── Full rebuild: every crate with >= 2 releases, one extractor version ──────
.PHONY: rebuild-full
rebuild-full: $(RUST_BIN)/batch $(PYTHON)
	@echo ""
	@echo "  Full rebuild of the headline file from a crates.io dump: every crate with"
	@echo "  >= 2 releases before the snapshot, one extractor version for every crate."
	@echo "  Needs DUMP_DIR and GITHUB_TOKEN; ~2 days."
	@echo ""
	@[ -n "$(DUMP_DIR)" ] || { echo "  Pass DUMP_DIR=/path/to/crates.io-db-dump/"; exit 1; }
	$(PYTHON) scripts/build_crate_list.py $(DUMP_DIR) 2026-09-14 \
	    results/crates_all.txt results/crates_all_repos.csv
	$(PYTHON) scripts/build_snapshot_manifest.py $(DUMP_DIR) \
	    results/crates_all.txt 2026-09-14 .dump_cache/version_manifest_all_cut20260914.csv
	$(RUST_BIN)/batch --crate-list results/crates_all.txt \
	    --version-manifest .dump_cache/version_manifest_all_cut20260914.csv \
	    --count 0 --workers 16 --tarball-only --crate-dir crate_csvs_full/
	@# Second pass at low concurrency: finished crates are skipped, so this only
	@# retries crates that failed transiently (e.g. static.crates.io rate limits).
	$(RUST_BIN)/batch --crate-list results/crates_all.txt \
	    --version-manifest .dump_cache/version_manifest_all_cut20260914.csv \
	    --count 0 --workers 4 --tarball-only --crate-dir crate_csvs_full/
	$(PYTHON) scripts/merge_crate_csvs.py crate_csvs_full/ data/cratesbump_extracted.csv.gz
	$(PYTHON) scripts/extraction_coverage.py .dump_cache/version_manifest_all_cut20260914.csv \
	    crate_csvs_full/ --out results/extraction_report.json
	$(MAKE) reproduce-section-5 DUMP_DIR=$(DUMP_DIR)

# ── §4 Headline ML numbers (held-out repositories, held-out crates, temporal) ─
HEADLINE_DATA = data/cratesbump.csv.gz

.PHONY: reproduce-section-4
reproduce-section-4: $(PYTHON)
	@echo ""
	@echo "  §4 — Headline ML numbers on the Cargo-labelled file: held-out"
	@echo "         repositories (5 seeds, Table III), held-out crates (5 seeds, to show"
	@echo "         the leakage through shared repositories), and the temporal split."
	@echo ""
	@mkdir -p results
	$(PYTHON) scripts/evaluate.py $(HEADLINE_DATA) \
	    --splits grouped --seeds 5 --group-by repo \
	    --out-json results/section_4_grouped_repo_5seed.json
	$(PYTHON) scripts/evaluate.py $(HEADLINE_DATA) \
	    --splits grouped --seeds 5 --group-by crate \
	    --models "Majority baseline" "Random Forest" \
	    --out-json results/section_4_grouped_crate_5seed.json
	$(PYTHON) scripts/evaluate.py $(HEADLINE_DATA) \
	    --splits temporal --group-by repo \
	    --out-json results/section_4_temporal_repo.json

.PHONY: reproduce-section-5
reproduce-section-5: $(PYTHON)
	@echo ""
	@echo "  §5 — Predecessors, Cargo labels, snapshot cut and caps from the dump,"
	@echo "         then the GitHub features. Writes results/cargo_label_report.json."
	@echo ""
	@[ -n "$(DUMP_DIR)" ] || { echo "  Pass DUMP_DIR=/path/to/crates.io-db-dump/"; exit 1; }
	$(PYTHON) scripts/label_pairs_cargo.py \
	    data/cratesbump_extracted.csv.gz $(DUMP_DIR) $(HEADLINE_DATA) \
	    --check-rdtd --max-date 2026-09-14 --cap-per-crate 100 --cap-per-family 5000 \
	    --repos results/crates_all_repos.csv --report results/cargo_label_report.json
	ENRICH_CLONE_FILTER=tree:0 $(PYTHON) scripts/fill_github_features.py \
	    $(HEADLINE_DATA) $(DUMP_DIR) --crates ALL \
	    --report results/github_fill_report.json

# ── §6 Validity: feature ablation + internal label consistency + external audit
.PHONY: reproduce-section-6
reproduce-section-6: $(PYTHON)
	@echo ""
	@echo "  §6 — Validity probes: per-feature ablation + label sanity check +"
	@echo "         external cargo-semver-checks audit."
	@echo ""
	@mkdir -p results
	@# Trains the Table III Random Forest on the seed-42 grouped fold, then
	@# permutes one feature at a time on that fold's test set.
	$(PYTHON) scripts/feature_ablation.py \
	    --data   $(HEADLINE_DATA) \
	    --split  grouped --group-by repo --seed 42 --reps 5 \
	    --out    results/feature_ablation_rebuilt_cargo_rf.json
	$(PYTHON) scripts/label_sanity_check.py \
	    $(HEADLINE_DATA) \
	    results/label_sanity_rebuilt_cargo.csv \
	    --summary results/label_sanity_rebuilt_cargo_summary.json
	@[ -n "$(DUMP_DIR)" ] || { echo "  Pass DUMP_DIR=/path/to/crates.io-db-dump-YYYY-MM-DD/"; exit 1; }
	$(PYTHON) scripts/downstream_impact.py $(DUMP_DIR) $(HEADLINE_DATA) \
	    --out results/downstream_impact.json
	@echo "  running external audit (cargo-semver-checks inside the Docker sandbox;"
	@echo "  build it first: docker build -t cratesbump-audit:0.51.0 -f docker/audit.Dockerfile docker/)"
	$(PYTHON) scripts/label_audit_cargo_semver_checks.py \
	    $(HEADLINE_DATA) $(DUMP_DIR) \
	    --n 150 --seed 42 --workers 4 --timeout 900 \
	    --out-dir results/label_audit
	$(PYTHON) scripts/label_audit_cargo_semver_checks.py \
	    $(HEADLINE_DATA) $(DUMP_DIR) \
	    --retry-errors --workers 4 --timeout 900 --out-dir results/label_audit
	@# Context for the hand review of every flagged pair (sources unpacked, never
	@# built); verdicts go into results/label_audit/hand_review.csv, then:
	$(PYTHON) scripts/audit_flag_context.py results/label_audit $(DUMP_DIR)
	$(PYTHON) scripts/label_audit_cargo_semver_checks.py x x --summarize-only \
	    --out-dir results/label_audit

# Paper numbers no other script writes (GitHub-zero breakdown, known bugs, audit
# disclosures), from committed files; DUMP_DIR (optional) adds the audit's
# binary/macro split, which is otherwise kept from results/paper_numbers.json.
.PHONY: paper-numbers
paper-numbers: $(PYTHON)
	$(PYTHON) scripts/paper_numbers.py $(if $(DUMP_DIR),--dump-dir $(DUMP_DIR),)

.PHONY: headline
headline: $(PYTHON)
	$(PYTHON) scripts/write_headline.py

.PHONY: reproduce-all
reproduce-all: reproduce-section-4 reproduce-section-5 reproduce-section-6 paper-numbers headline

# Run the hybrid across four ML backbones (the first run fills the LLM cache;
# the rest read it) and write one summary table.
.PHONY: jev-sweep
jev-sweep: $(PYTHON)
	@[ -n "$$TYPESAFE_API_KEY" ] || [ -f .typesafe_token ] || \
		echo "Note: no TYPESAFE_API_KEY; every LLM response must come from $(JEV_CACHE)."
	@for M in XGBoost LightGBM "Hist Gradient Boosting" "Random Forest"; do \
		echo "=== $$M ==="; \
		SAFE=$$(echo "$$M" | tr ' ' '_'); \
		$(PYTHON) scripts/jev_optimized.py $(JEV_DATA) \
			--split $(JEV_SPLIT) \
			--test-sample $(JEV_TEST_SAMPLE) --tune-sample $(JEV_TUNE_SAMPLE) \
			--few-shot $(JEV_FEW_SHOT) --ml-model "$$M" \
			--enable-gate --enable-stack \
			--cache $(JEV_CACHE) \
			--report-json  results/$(JEV_PREFIX)$$SAFE.json \
			--report-image results/$(JEV_PREFIX)$$SAFE.png || exit $$?; \
	done
	$(PYTHON) scripts/jev_summarize.py XGBoost LightGBM "Hist Gradient Boosting" "Random Forest" \
		--prefix $(JEV_PREFIX) \
		--out-json  results/$(JEV_PREFIX)summary.json \
		--out-image results/$(JEV_PREFIX)summary.png \
		--extra-variants "+ Confidence gate:Confidence gate" "+ Stacking:Stacking"

$(PYTHON):
	@echo "Python venv not found. Run 'make setup-python' first."; exit 1

# Clean 
.PHONY: clean
clean:
	cargo clean
	@echo "Removed target/"
