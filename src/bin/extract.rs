/// Single-crate feature extractor.
/// Extracts per-release features for one crate and writes them to a CSV in
/// crate_csvs/.  Use `make extract C=<crate>` or invoke the binary directly.
/// The GitHub token is read from `.token` in the project root (run `make setup-token`).
///
/// Each version's source is downloaded from crates.io as a `.crate` tarball —
/// no git clone is performed.
///
/// Usage:
///   extract <CRATE_NAME> [OPTIONS]
///
/// Options:
///   --output <FILE>          Output CSV path [default: crate_csvs/<crate>.csv]
///   --local <PATH>           Local source tree to use as current version
///   --prev-version <V>       crates.io version to diff against (required with --local)
///   --tarball-only           Skip the GitHub features (no GitHub token or GitHub API calls);
///                            the released dataset computes them separately with
///                            scripts/fill_github_features.py
///
/// Local mode (--local):
///   Diffs the source tree at <PATH> against the crates.io tarball for
///   <prev-version>.  Writes a single-row CSV with version = "local" and no
///   release_type label.
///
/// Examples:
///   extract serde
///   extract tokio --output crate_csvs/tokio.csv
///   extract serde --local /path/to/serde --prev-version 1.0.199 --output local.csv
use semver_ml::data::{ReleaseRecord, ReleaseType, write_csv};
use semver_ml::extractor::{ExtractConfig, extract_crate, extract_local};
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print_usage();
        return;
    }

    let crate_name = args[0].clone();
    let mut output: Option<String> = None;
    let mut local_path: Option<std::path::PathBuf> = None;
    let mut prev_version: Option<String> = None;
    let mut no_github = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                output = args.get(i).cloned();
            }
            "--local" => {
                i += 1;
                local_path = args.get(i).map(std::path::PathBuf::from);
            }
            "--prev-version" => {
                i += 1;
                prev_version = args.get(i).cloned();
            }
            "--tarball-only" | "--no-github" => {
                no_github = true;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }

    let output_path = output.unwrap_or_else(|| format!("crate_csvs/{}.csv", crate_name));

    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("cannot create output directory");
        }
    }

    eprintln!("=== semver-ml feature extractor ===");
    eprintln!("Crate  : {}", crate_name);
    eprintln!("Output : {}", output_path);
    eprintln!();

    // ── Local mode ────────────────────────────────────────────────────────────
    if let Some(ref local) = local_path {
        let prev = match prev_version {
            Some(ref v) => v.clone(),
            None => {
                eprintln!("Error: --prev-version <V> is required when --local is given.");
                process::exit(1);
            }
        };

        eprintln!("Local mode: diffing working tree against {}", prev);

        match extract_local(local, &prev, &crate_name, semver_ml::token::load().as_deref()) {
            Ok(features) => {
                // Wrap in a ReleaseRecord — no label since this is unreleased code
                let record = ReleaseRecord {
                    package: crate_name.clone(),
                    version: "local".to_string(),
                    release_type: ReleaseType::Patch, // placeholder; ignored by predict
                    features,
                };
                match write_csv(&output_path, &[record]) {
                    Ok(()) => eprintln!("\nWrote local features to {}", output_path),
                    Err(e) => { eprintln!("Error writing CSV: {}", e); process::exit(1); }
                }
            }
            Err(e) => { eprintln!("Extraction failed: {}", e); process::exit(1); }
        }
        return;
    }

    // ── Normal mode ───────────────────────────────────────────────────────────
    let cfg = ExtractConfig {
        crate_name: crate_name.clone(),
        github_token: if no_github { None } else { semver_ml::token::load() },
        no_github,
        versions: None,
    };

    match extract_crate(&cfg) {
        Ok(records) => {
            if records.is_empty() {
                eprintln!("No records extracted — check that the crate is on crates.io.");
                process::exit(1);
            }
            match write_csv(&output_path, &records) {
                Ok(()) => {
                    eprintln!("\nWrote {} records to {}", records.len(), output_path);
                    eprintln!("Train or retrain the ML pipeline with:  make pipeline");
                }
                Err(e) => {
                    eprintln!("Error writing CSV: {}", e);
                    process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Extraction failed: {}", e);
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
        r#"extract — Rust crate feature extractor for semver-ml

USAGE:
  extract <CRATE_NAME> [OPTIONS]

OPTIONS:
  --output <FILE>          Output CSV  [default: crate_csvs/<crate>.csv]
  --local <PATH>           Diff source tree at <PATH> vs --prev-version
  --prev-version <V>       Previous version to diff against (required with --local)
  --tarball-only           Skip the GitHub features (no GitHub token or GitHub API calls)

  GitHub token is read automatically from .token (run `make setup-token`).

EXAMPLES:
  extract serde
  extract tokio --output crate_csvs/tokio.csv
  extract serde --tarball-only --output crate_csvs/serde.csv
  extract serde --local /path/to/serde --prev-version 1.0.199 --output local.csv

OUTPUT:
  CSV with columns: package, version, release_type, arf, mrf, …, aml
  Train on the collected crate_csvs/*.csv with:  python3 scripts/pipeline.py
"#
    );
}
