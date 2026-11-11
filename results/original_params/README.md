# Results with the original model settings

Section 4 and ablation results on `data/cratesbump.csv.gz` with the
hyperparameters that were `scripts/pipeline.py` `DEFAULT_PARAMS` before the
re-tuning on this release (a May 2026 random search; the parameter sets are in
`results/tuning_candidates_original_defaults.json`). Produced at commit
b3e5ba7 by the same commands as `make reproduce-section-4` and the ablation
step of `make reproduce-section-6`.

The paper and `results/HEADLINE.md` cite these files for the original
settings: held-out repositories (ensembles 0.455–0.459 macro-F1), held-out
crates (Random Forest 0.612, against 0.455 held out by repository), and the
feature-family ablation (GitHub commits +0.046, issues/PRs +0.027).

To regenerate them, copy the parameter sets from
`results/tuning_candidates_original_defaults.json` into `DEFAULT_PARAMS` and
rerun those targets.
