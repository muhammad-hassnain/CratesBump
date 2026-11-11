/// Batch feature extractor.
/// Fetches the top N most-downloaded Rust crates from crates.io and writes
/// one CSV per crate into crate_csvs/.  Crates whose CSV already exists are
/// skipped, so interrupted runs can be safely resumed.
///
/// Usage:
///   batch [OPTIONS]
///
/// Options:
///   --count <N>            Number of top crates to process [default: 1000]
///   --workers <N>          Parallel workers [default: 6]
///   --crate-dir <DIR>      Output directory for per-crate CSVs [default: crate_csvs/]
///   --crate-list <FILE>    Read crate names from FILE (one per line) instead of
///                          fetching the top-N by downloads. --count still caps
///                          how many are taken (0 = all).
///   --version-manifest <FILE>
///                          package,version,created_at CSV (from a crates.io
///                          dump, see scripts/build_snapshot_manifest.py); each
///                          crate's versions come from it instead of the
///                          crates.io API.
///   --tarball-only         Skip all GitHub features (repo lookup + issue/PR +
///                          commit). Only tarball + AST features are computed.
///                          Recommended for large runs (no GitHub rate limit).
use semver_ml::data::write_csv;
use semver_ml::extractor::api::CrateVersion;
use semver_ml::extractor::{extract_crate, ExtractConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

const USAGE: &str = "\
batch: extract features for many crates, one CSV per crate (resumable)

Usage: batch [OPTIONS]

Options:
  --count <N>                 Number of top crates to process [default: 1000; 0 = all]
  --workers <N>               Parallel workers [default: 6]
  --crate-dir <DIR>           Output directory for per-crate CSVs [default: crate_csvs/]
  --crate-list <FILE>         Read crate names from FILE (one per line) instead of
                              fetching the top-N by downloads
  --version-manifest <FILE>   package,version,created_at CSV from a crates.io dump
                              (scripts/build_snapshot_manifest.py)
  --tarball-only              Skip GitHub features (tarball + AST features only)
  -h, --help                  Print this help
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut count: usize = 1000;
    let mut workers: usize = 6;
    let mut crate_dir = PathBuf::from("crate_csvs");
    let mut crate_list: Option<PathBuf> = None;
    let mut version_manifest: Option<PathBuf> = None;
    let mut no_github = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--version-manifest" => {
                i += 1;
                version_manifest = Some(PathBuf::from(&args[i]));
            }
            "--count" => {
                i += 1;
                count = args[i].parse().unwrap_or(1000);
            }
            "--workers" => {
                i += 1;
                workers = args[i].parse().unwrap_or(6);
            }
            "--crate-dir" => {
                i += 1;
                crate_dir = PathBuf::from(&args[i]);
            }
            "--crate-list" => {
                i += 1;
                crate_list = Some(PathBuf::from(&args[i]));
            }
            "--tarball-only" | "--no-github" => {
                no_github = true;
            }
            "-h" | "--help" => {
                print!("{}", USAGE);
                return;
            }
            other => {
                eprintln!("batch: unknown option '{}'\n\n{}", other, USAGE);
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let github_token = if no_github { None } else { semver_ml::token::load() };

    std::fs::create_dir_all(&crate_dir).expect("cannot create crate-dir");

    eprintln!("=== semver-ml batch extractor ===");
    if no_github {
        eprintln!("mode: tarball-only (GitHub features skipped)");
    }

    let crate_names = if let Some(ref path) = crate_list {
        eprintln!("Reading crate names from {} …", path.display());
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to read crate list {}: {}", path.display(), e);
                std::process::exit(1);
            }
        };
        let mut names: Vec<String> = text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        // --count 0 means "all"; otherwise cap.
        if count > 0 && names.len() > count {
            names.truncate(count);
        }
        names
    } else {
        eprintln!("Fetching top {} crate names from crates.io …", count);
        match fetch_top_crates(count) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to fetch crate list: {}", e);
                std::process::exit(1);
            }
        }
    };
    eprintln!("Got {} crate names", crate_names.len());

    // Work queue
    // Optional version manifest (package,version,created_at) from the DB dump.
    // When present, workers use it instead of calling the crates.io API — this
    // bypasses the API rate limit entirely (only the tarball CDN is hit).
    let manifest: Arc<Option<HashMap<String, Vec<CrateVersion>>>> = Arc::new(
        version_manifest.as_ref().map(|path| {
            eprintln!("Loading version manifest from {} …", path.display());
            let m = load_manifest(path);
            eprintln!("Manifest: {} crates", m.len());
            m
        }),
    );

    let total = crate_names.len();
    let queue: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(crate_names));
    let done = Arc::new(Mutex::new(0usize));

    let mut handles = Vec::new();

    for worker_id in 0..workers {
        let queue = Arc::clone(&queue);
        let done = Arc::clone(&done);
        let crate_dir = crate_dir.clone();
        let token = github_token.clone();
        let manifest = Arc::clone(&manifest);

        // Deeply nested generated sources (e.g. instruction-set models) overflow
        // the default 2 MiB thread stack while syn parses them; a stack overflow
        // aborts the whole process, so give each worker a large stack.
        let handle = thread::Builder::new().stack_size(256 << 20).spawn(move || loop {
            let crate_name = {
                let mut q = queue.lock().unwrap();
                if q.is_empty() {
                    break;
                }
                q.remove(0)
            };

            let out_path = crate_dir.join(format!("{}.csv", crate_name));
            if out_path.exists() {
                let mut d = done.lock().unwrap();
                *d += 1;
                eprintln!("[{}/{}] {} skipped (already done)", *d, total, crate_name);
                continue;
            }

            let versions = manifest
                .as_ref()
                .as_ref()
                .and_then(|m| m.get(&crate_name).cloned());

            let cfg = ExtractConfig {
                crate_name: crate_name.clone(),
                github_token: token.clone(),
                no_github,
                versions,
            };

            match extract_crate(&cfg) {
                Ok(records) if !records.is_empty() => match write_csv(&out_path, &records) {
                    Ok(()) => {
                        let mut d = done.lock().unwrap();
                        *d += 1;
                        eprintln!(
                            "[w{}] [{}/{}] {} → {} records ✓",
                            worker_id,
                            *d,
                            total,
                            crate_name,
                            records.len()
                        );
                    }
                    Err(e) => eprintln!("[w{}] {} CSV write failed: {}", worker_id, crate_name, e),
                },
                Ok(_) => {
                    let mut d = done.lock().unwrap();
                    *d += 1;
                    eprintln!(
                        "[w{}] [{}/{}] {} → 0 records (no tags?)",
                        worker_id, *d, total, crate_name
                    );
                }
                Err(e) => {
                    let mut d = done.lock().unwrap();
                    *d += 1;
                    eprintln!(
                        "[w{}] [{}/{}] {} FAILED: {}",
                        worker_id, *d, total, crate_name, e
                    );
                }
            }
        }).expect("spawn worker thread");
        handles.push(handle);
    }

    for h in handles {
        h.join().ok();
    }
}

// Version manifest (package,version,created_at) from the crates.io DB dump.

/// Parse a CSV with a `package,version,created_at` header into a map of
/// crate name -> versions. Order within a crate is preserved as-is; the
/// extractor re-sorts by `created_at`.
fn load_manifest(path: &PathBuf) -> HashMap<String, Vec<CrateVersion>> {
    let mut map: HashMap<String, Vec<CrateVersion>> = HashMap::new();
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to read manifest {}: {}", path.display(), e);
            std::process::exit(1);
        }
    };
    let mut lines = text.lines();
    // Locate the columns from the header (order-independent).
    let header = lines.next().unwrap_or("");
    let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
    let idx = |name: &str| cols.iter().position(|c| *c == name);
    let (ip, iv, ic) = match (idx("package"), idx("version"), idx("created_at")) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => {
            eprintln!("Manifest {} must have package,version,created_at columns", path.display());
            std::process::exit(1);
        }
    };
    let maxi = ip.max(iv).max(ic);
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() <= maxi {
            continue;
        }
        map.entry(f[ip].to_string()).or_default().push(CrateVersion {
            num: f[iv].to_string(),
            created_at: f[ic].to_string(),
        });
    }
    map
}

// crates.io top-crates fetch

fn fetch_top_crates(n: usize) -> Result<Vec<String>, String> {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Page {
        crates: Vec<CrateInfo>,
    }
    #[derive(Deserialize)]
    struct CrateInfo {
        name: String,
    }

    let per_page = 100;
    let pages = (n + per_page - 1) / per_page;
    let mut names = Vec::with_capacity(n);

    for page in 1..=pages {
        let url = format!(
            "https://crates.io/api/v1/crates?sort=downloads&per_page={}&page={}",
            per_page, page
        );
        let resp: Page = ureq::get(&url)
            .set("User-Agent", "semver-ml-batch/0.1.0")
            .call()
            .map_err(|e| e.to_string())?
            .into_json()
            .map_err(|e| e.to_string())?;

        if resp.crates.is_empty() {
            break;
        }
        names.extend(resp.crates.into_iter().map(|c| c.name));

        // rate limiting
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    names.truncate(n);
    Ok(names)
}
