pub mod api;
pub mod ast;
pub mod diff;

use std::path::{Path, PathBuf};

use crate::data::{ReleaseRecord, ReleaseType};
use crate::features::ReleaseFeatures;

pub struct ExtractConfig {
    /// Crate name as it appears on crates.io (e.g. "serde", "tokio").
    pub crate_name: String,
    /// GitHub personal access token (optional — only used for issue/PR features).
    pub github_token: Option<String>,
    /// Skip the GitHub REST path entirely (repo lookup + issue/PR + commit
    /// features). The commit/issue features are left at their default zero.
    /// Use this for large-scale runs where the GitHub rate limit (5k/hr with a
    /// token, 60/hr without) is the bottleneck and only the tarball + AST
    /// features are wanted.
    pub no_github: bool,
    /// Pre-supplied version list (num + created_at), e.g. from the crates.io DB
    /// dump. When `Some`, `api::get_versions` is NOT called — this bypasses the
    /// crates.io API rate limit entirely (only the tarball CDN is used).
    pub versions: Option<Vec<api::CrateVersion>>,
}

// Main entry point

/// Extract all features for every consecutive version pair of a crate and
/// return them as `ReleaseRecord`s.  Each version's source is downloaded from
/// crates.io as a `.crate` tarball 
pub fn extract_crate(cfg: &ExtractConfig) -> Result<Vec<ReleaseRecord>, String> {
    let versions = match &cfg.versions {
        Some(v) => {
            // Pre-supplied (e.g. from the DB dump) — sort chronologically and use
            // directly, no crates.io API call.
            let mut v = v.clone();
            v.sort_by(|a, b| a.created_at.cmp(&b.created_at));
            eprintln!("'{}': {} versions (from manifest)", cfg.crate_name, v.len());
            v
        }
        None => {
            eprint!("Fetching version list for '{}' … ", cfg.crate_name);
            let v = api::get_versions(&cfg.crate_name)?;
            eprintln!("{} versions found", v.len());
            v
        }
    };

    if versions.len() < 2 {
        return Err("Need at least 2 versions to form a release pair".to_string());
    }

    // GitHub repo (used only for issue/PR + commit features).
    // In --tarball-only mode we skip the lookup entirely so no api.github.com
    // request is ever made (avoids the 60/hr unauthenticated rate limit).
    let gh: Option<(String, String)> = if cfg.no_github {
        eprintln!("tarball-only mode — skipping GitHub features");
        None
    } else {
        let g = api::get_repo_url(&cfg.crate_name)
            .ok()
            .flatten()
            .and_then(|url| api::parse_github_owner_repo(&url));
        if let Some((ref owner, ref repo)) = g {
            eprintln!("GitHub repository (for issue/PR features): {}/{}", owner, repo);
        } else {
            eprintln!("No GitHub repository found — issue/PR features will be zero");
        }
        g
    };

    let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let nums: Vec<&str> = versions.iter().map(|v| v.num.as_str()).collect();
    let plan = plan_pairs(&nums);

    let mut records = Vec::new();
    // The plan is chained (plan[k].1 == plan[k+1].0), so each version's tree is
    // downloaded once: keep the previous pair's `curr` tree as this pair's
    // `prev`. `cached_idx` is the version index that `prev_dir` holds.
    let mut prev_dir: Option<PathBuf> = None;
    let mut cached_idx: Option<usize> = None;

    for (k, &(pi, ci)) in plan.iter().enumerate() {
        let (prev_v, curr_v) = (&versions[pi], &versions[ci]);

        eprint!(
            "[{}/{}] {} → {} … ",
            k + 1, plan.len(), prev_v.num, curr_v.num
        );

        // `plan_pairs` already guaranteed curr is a forward bump over prev, so
        // semver_type is Some here; skip defensively (without disturbing the
        // cache) if it somehow is not.
        let release_type = match semver_type(&prev_v.num, &curr_v.num) {
            Some(rt) => rt,
            None => {
                eprintln!("(no forward bump, skipped)");
                continue;
            }
        };

        // Reuse the cached tree when it belongs to this pair's prev; otherwise
        // drop whatever is cached and download prev fresh.
        let pd = match prev_dir.take() {
            Some(d) if cached_idx == Some(pi) => d,
            other => {
                if let Some(d) = other {
                    let _ = std::fs::remove_dir_all(&d);
                }
                match download_prev_source(&cfg.crate_name, &prev_v.num, tmp.path()) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("(failed to download {}: {})", prev_v.num, e);
                        cached_idx = None;
                        continue;
                    }
                }
            }
        };

        let cd = match download_prev_source(&cfg.crate_name, &curr_v.num, tmp.path()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("(failed to download {}: {})", curr_v.num, e);
                let _ = std::fs::remove_dir_all(&pd);
                cached_idx = None;
                continue;
            }
        };

        let features = compute_features_fs(
            &pd,
            &cd,
            Some(&prev_v.created_at),
            Some(&curr_v.created_at),
            gh.as_ref().map(|(o, r)| (o.as_str(), r.as_str())),
            cfg.github_token.as_deref(),
        );

        eprintln!("{}", release_type);

        records.push(ReleaseRecord {
            package: cfg.crate_name.clone(),
            version: curr_v.num.clone(),
            release_type,
            features,
        });

        // Free prev's tree; keep curr's as the next pair's prev (plan is chained),
        // so each version is downloaded at most once and the tempdir stays small.
        let _ = std::fs::remove_dir_all(&pd);
        prev_dir = Some(cd);
        cached_idx = Some(ci);
    }

    eprintln!("Extracted {} release records", records.len());
    Ok(records)
}

/// Compute features for a single version pair from two extracted source trees.
///
/// `prev_date` / `curr_date` are crates.io publication timestamps (RFC 3339).
/// If absent, the time-based feature `rdtd` is zero.
/// `gh` is `(owner, repo)` on GitHub; if absent (e.g. `--tarball-only`), all
/// eleven GitHub features are zero, and without a token the issue/PR features
/// are zero. The commit features here come from the GitHub commits API; the
/// released dataset instead computes all eleven GitHub features with
/// scripts/fill_github_features.py (commits from repository clones, issues and
/// pull requests from the GitHub API).
fn compute_features_fs(
    prev_dir: &Path,
    curr_dir: &Path,
    prev_date: Option<&str>,
    curr_date: Option<&str>,
    gh: Option<(&str, &str)>,
    github_token: Option<&str>,
) -> ReleaseFeatures {
    let mut f = ReleaseFeatures::default();

    // File/line-count diff stats
    let ds = diff::diff_stats_dirs(prev_dir, curr_dir);
    f.arf   = ds.rs_added as f64;
    f.mrf   = ds.rs_modified as f64;
    f.drf   = ds.rs_deleted as f64;
    f.anrf  = ds.other_added as f64;
    f.mnrf  = ds.other_modified as f64;
    f.dnrf  = ds.other_deleted as f64;
    f.la    = ds.lines_added as f64;
    f.ld    = ds.lines_deleted as f64;
    f.clcrd = (ds.lines_added as i64 - ds.lines_deleted as i64) as f64;

    // Comment line diff
    let cd = diff::comment_diff_dirs(prev_dir, curr_dir);
    f.icc = cd.added as f64;
    f.dcc = cd.deleted as f64;
    f.mcc = cd.added.min(cd.deleted) as f64;

    // Cargo.toml dependency / feature / edition diff
    let prev_cargo = std::fs::read_to_string(prev_dir.join("Cargo.toml")).unwrap_or_default();
    let curr_cargo = std::fs::read_to_string(curr_dir.join("Cargo.toml")).unwrap_or_default();
    let dep_diff = ast::diff_cargo_toml(&prev_cargo, &curr_cargo);
    f.pa = dep_diff.added as f64;
    f.pd = dep_diff.deleted as f64;
    f.pu = dep_diff.updated as f64;
    // tcct redefined (fix): 1 iff a dependency section actually changed, not
    // merely that the (always-version-bumped) Cargo.toml file was touched.
    f.tcct = if dep_diff.deps_changed { 1.0 } else { 0.0 };
    // New Rust-specific Cargo signals.
    f.fa = dep_diff.features_added as f64;
    f.fd = dep_diff.features_removed as f64;
    f.edc = if dep_diff.edition_changed { 1.0 } else { 0.0 };

    // Time between releases
    if let (Some(p), Some(c)) = (prev_date, curr_date) {
        f.rdtd = date_diff_days(p, c).unwrap_or(0.0);
    }

    // GitHub features (issues, PRs, commits)
    if let (Some((owner, repo)), Some(p), Some(c)) = (gh, prev_date, curr_date) {
        let since = iso_min(p, c);
        let until = iso_max(p, c);

        // Issues/PRs: only meaningful with a token (60 req/hr without is too low).
        if github_token.is_some() {
            if let Ok((oi, ci, opr, cpr)) =
                api::count_issues_prs(owner, repo, &since, &until, github_token)
            {
                f.poi  = oi as f64;
                f.pci  = ci as f64;
                f.popr = opr as f64;
                f.pcpr = cpr as f64;
            }
        }

        // Commits: public endpoint, works with or without a token (a token just
        // raises the rate limit from 60 to 5000 req/hr).
        if let Ok(cs) = api::commit_stats(owner, repo, &since, &until, github_token) {
            f.tcm = cs.total as f64;
            f.tau = cs.unique_authors as f64;
            f.nbf = cs.bug_fixes as f64;
            f.kwm = cs.kw_major as f64;
            f.kwp = cs.kw_patch as f64;
            f.kwb = cs.kw_break as f64;
            f.aml = cs.avg_msg_len;
        }
    }

    // AST diff
    let snap_prev = ast::snapshot_dir(prev_dir);
    let snap_curr = ast::snapshot_dir(curr_dir);
    let d = ast::diff(&snap_prev, &snap_curr);
    f.adm   = d.added_fns as f64;
    f.dem   = d.deleted_fns as f64;
    f.mpc   = d.param_changed_fns as f64;
    f.mpd   = d.param_deleted_fns as f64;
    f.gva   = d.added_globals as f64;
    f.gvd   = d.deleted_globals as f64;
    f.mla   = d.logic_added as f64;
    f.mld   = d.logic_deleted as f64;
    f.cycd  = d.cc_delta.unsigned_abs() as f64;
    f.acycd = d.avg_cc_delta.abs();
    // Move/rename metrics (body-fingerprint matching in ast::diff).
    f.mom   = d.methods_moved as f64;
    f.mnc   = d.method_name_changes as f64;
    f.mlm   = d.logic_moved as f64;

    f
}

// Semver helpers

/// Parse a version string into its `(major, minor, patch)` precedence tuple.
/// Pre-release (`-alpha`) and build metadata (`+20140915`) after PATCH are both
/// stripped; without dropping `+build`, versions like "0.1.0+20140915" (common
/// in google-* crates) would fail to parse. Returns `None` if unparseable.
fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.trim_start_matches('v');
    let s = s.split(|c| c == '-' || c == '+').next().unwrap_or(s);
    let p: Vec<&str> = s.split('.').collect();
    if p.len() < 3 {
        return None;
    }
    Some((p[0].parse().ok()?, p[1].parse().ok()?, p[2].parse().ok()?))
}

fn semver_type(prev: &str, curr: &str) -> Option<ReleaseType> {
    let (pm, pn, pp) = parse_version(prev)?;
    let (cm, cn, cp) = parse_version(curr)?;

    if cm > pm {
        Some(ReleaseType::Major)
    } else if cn > pn {
        Some(ReleaseType::Minor)
    } else if cp > pp {
        Some(ReleaseType::Patch)
    } else {
        None
    }
}

/// One pre-release identifier. Numeric identifiers rank below alphanumeric
/// ones and compare numerically; alphanumeric ones compare in ASCII order
/// (SemVer 2.0.0, rule 11). The derived `Ord` follows the variant order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PreId {
    Num(u64),
    Alpha(String),
}

/// Full SemVer precedence key: `(major, minor, patch)`, then 0 for a
/// pre-release / 1 for a release (a release ranks above any pre-release of the
/// same core), then the pre-release identifiers. Build metadata is ignored.
type Precedence = ((u64, u64, u64), u8, Vec<PreId>);

fn precedence(s: &str) -> Option<Precedence> {
    let s = s.trim_start_matches('v');
    let s = s.split('+').next().unwrap_or(s);
    let (core, pre) = match s.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (s, None),
    };
    let p: Vec<&str> = core.split('.').collect();
    if p.len() != 3 {
        return None;
    }
    let core = (p[0].parse().ok()?, p[1].parse().ok()?, p[2].parse().ok()?);
    Some(match pre {
        None => (core, 1, Vec::new()),
        Some(pre) => (
            core,
            0,
            pre.split('.')
                .map(|id| match id.parse::<u64>() {
                    Ok(n) if id.bytes().all(|b| b.is_ascii_digit()) => PreId::Num(n),
                    _ => PreId::Alpha(id.to_string()),
                })
                .collect(),
        ),
    })
}

/// Decide which `(prev_idx, curr_idx)` version pairs to extract, given versions
/// in PUBLISH order. Versions are ordered by full SemVer precedence, pre-release
/// tags included, and the running maximum is tracked:
///   - a version below or equal to the maximum (a backport published out of
///     order, a duplicate, a late pre-release) is skipped and leaves the
///     maximum alone;
///   - a version above the maximum with a higher `(major, minor, patch)` forms
///     a pair with the maximum and becomes the new maximum;
///   - a version above the maximum with the SAME `(major, minor, patch)`
///     (`1.0.0-rc.1` -> `1.0.0-rc.2` -> `1.0.0`) forms no pair, because it
///     has no bump class, but becomes the new maximum, so the next release
///     (`1.0.1`) is diffed against `1.0.0` and not against `1.0.0-rc.1`.
/// Pairs are usually chained (`pairs[k].1 == pairs[k+1].0`); the extraction
/// loop re-downloads `prev` when they are not.
fn plan_pairs(nums: &[&str]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut last: Option<(usize, Precedence)> = None;
    for (i, num) in nums.iter().enumerate() {
        let Some(ck) = precedence(num) else { continue };
        match &last {
            None => last = Some((i, ck)),
            Some((p, pk)) => {
                if ck > *pk {
                    if ck.0 > pk.0 {
                        pairs.push((*p, i));
                    }
                    last = Some((i, ck));
                }
            }
        }
    }
    pairs
}

#[cfg(test)]
mod plan_pairs_tests {
    use super::plan_pairs;

    #[test]
    fn normal_sequence() {
        assert_eq!(
            plan_pairs(&["1.0.0", "1.0.1", "1.1.0", "2.0.0"]),
            vec![(0, 1), (1, 2), (2, 3)]
        );
    }

    #[test]
    fn midlist_backport_excluded_and_rechained() {
        // 0.9.5 published between 1.0.1 and 1.0.2 must be dropped, and 1.0.2 must
        // pair against 1.0.1 (not the backport 0.9.5).
        assert_eq!(
            plan_pairs(&["1.0.0", "1.0.1", "0.9.5", "1.0.2"]),
            vec![(0, 1), (1, 3)]
        );
    }

    #[test]
    fn trailing_backport_excluded() {
        assert_eq!(plan_pairs(&["1.0.0", "1.0.1", "0.11.0"]), vec![(0, 1)]);
    }

    #[test]
    fn consecutive_backports_excluded() {
        assert_eq!(
            plan_pairs(&["1.2.0", "1.1.5", "1.1.6", "1.3.0"]),
            vec![(0, 3)]
        );
    }

    #[test]
    fn duplicates_and_prerelease_skipped() {
        // equal precedence after stripping build/pre-release is not a forward bump
        assert_eq!(
            plan_pairs(&["1.0.0", "1.0.0+build", "1.0.0-rc.1", "1.0.1"]),
            vec![(0, 3)]
        );
    }

    #[test]
    fn final_release_replaces_its_prerelease_as_baseline() {
        // 1.0.0 forms no pair with 1.0.0-rc.1 (same core) but becomes the
        // baseline, so 1.0.1 is diffed against 1.0.0.
        assert_eq!(
            plan_pairs(&["0.9.0", "1.0.0-rc.1", "1.0.0", "1.0.1"]),
            vec![(0, 1), (2, 3)]
        );
    }

    #[test]
    fn prerelease_chain_before_minor() {
        assert_eq!(
            plan_pairs(&["1.0.0", "1.1.0-beta.1", "1.1.0-beta.2", "1.1.0", "1.2.0"]),
            vec![(0, 1), (3, 4)]
        );
    }

    #[test]
    fn prerelease_identifiers_compare_numerically() {
        // alpha.10 > alpha.2; neither forms a pair; 2.0.0 pairs with 1.0.0.
        assert_eq!(
            plan_pairs(&["1.0.0-alpha.2", "1.0.0-alpha.10", "1.0.0", "2.0.0"]),
            vec![(2, 3)]
        );
        assert!(super::precedence("1.0.0-alpha.10") > super::precedence("1.0.0-alpha.2"));
        assert!(super::precedence("1.0.0-alpha") < super::precedence("1.0.0-alpha.1"));
        assert!(super::precedence("1.0.0-1") < super::precedence("1.0.0-alpha"));
        assert!(super::precedence("1.0.0-rc.1") < super::precedence("1.0.0"));
        assert_eq!(super::precedence("1.0.0+build"), super::precedence("1.0.0"));
    }

    #[test]
    fn single_version_no_pairs() {
        assert_eq!(plan_pairs(&["1.0.0"]), Vec::<(usize, usize)>::new());
    }
}

// Date helpers

fn date_diff_days(from: &str, to: &str) -> Option<f64> {
    use chrono::DateTime;
    let f: DateTime<chrono::FixedOffset> = DateTime::parse_from_rfc3339(from).ok()?;
    let t: DateTime<chrono::FixedOffset> = DateTime::parse_from_rfc3339(to).ok()?;
    let dur = t.signed_duration_since(f);
    Some(dur.num_seconds().abs() as f64 / 86_400.0)
}

fn iso_min<'a>(a: &'a str, b: &'a str) -> String {
    if a <= b {
        a.to_string()
    } else {
        b.to_string()
    }
}
fn iso_max<'a>(a: &'a str, b: &'a str) -> String {
    if a >= b {
        a.to_string()
    } else {
        b.to_string()
    }
}

// Local-path extraction 

/// Extract features for a local source tree diffed against a crates.io
/// version. 
///
/// Differences from `extract_crate`:
///   - `curr_dir` is the user's local source tree, not a downloaded tarball.
///   - Date and GitHub issue/PR features are left at zero (no published date known).
///   - Commit-log features are also zero (same reason as `extract_crate`).
pub fn extract_local(
    local_repo: &Path,
    prev_version: &str,
    crate_name: &str,
    github_token: Option<&str>,
) -> Result<ReleaseFeatures, String> {
    if !local_repo.exists() {
        return Err(format!("local path does not exist: {}", local_repo.display()));
    }
    if !local_repo.is_dir() {
        return Err(format!("local path is not a directory: {}", local_repo.display()));
    }
    // A Rust crate must have a Cargo.toml at its root.
    if !local_repo.join("Cargo.toml").is_file() {
        return Err(format!(
            "no Cargo.toml found in {} — is this actually a Rust crate?",
            local_repo.display()
        ));
    }

    // Exit if the local path looks like a workspace root rather than a crate root.
    if let Ok(toml_text) = std::fs::read_to_string(local_repo.join("Cargo.toml")) {
        let has_package = toml_text.contains("[package]");
        let has_workspace = toml_text.contains("[workspace]");
        if has_workspace && !has_package {
            println!();
            println!("WARNING: {} appears to be a Cargo workspace root (no [package] section).", local_repo.display());
            println!("   Point --local at the inner crate directory instead, e.g.:");
            println!("       --local {}/{}", local_repo.display(), crate_name);
            println!();
            return Err("local path looks like a workspace root, not a crate root".to_string());
        }
    }

    eprintln!("Downloading {} {} from crates.io for comparison …", crate_name, prev_version);
    let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let prev_dir = download_prev_source(crate_name, prev_version, tmp.path())?;
    extract_local_fs(&prev_dir, local_repo, github_token)
}

/// Filesystem-based local extraction: compares two directories directly.
/// Commit-based features (tcm, tau, nbf, …) stay at 0 since there is no history.
fn extract_local_fs(
    prev_dir: &Path,
    curr_dir: &Path,
    github_token: Option<&str>,
) -> Result<ReleaseFeatures, String> {
    let _ = github_token;
    // Local mode has no publication dates and no GitHub repo to query.
    Ok(compute_features_fs(prev_dir, curr_dir, None, None, None, None))
}

/// Download a crate from crates.io and extract it, returning the path to the
/// extracted source directory (`<dest>/<crate>-<version>/`).
fn download_prev_source(crate_name: &str, version: &str, dest: &Path) -> Result<PathBuf, String> {
    use std::io::Read;

    let url = format!(
        "https://static.crates.io/crates/{}/{}-{}.crate",
        crate_name, crate_name, version
    );
    eprintln!("Downloading {} …", url);

    // Retry on 429: the CDN throttles under sustained parallel load. Without
    // this, a single 429 skips the pair and busy crates yield 0 records.
    const MAX_RETRIES: u32 = 5;
    let mut attempt: u32 = 0;
    let resp = loop {
        match ureq::get(&url).set("User-Agent", "semver-ml/0.1.0").call() {
            Ok(r) => break r,
            Err(ureq::Error::Status(429, r)) => {
                attempt += 1;
                if attempt > MAX_RETRIES {
                    return Err(format!("HTTP 429 fetching {} (gave up after {} retries)", url, MAX_RETRIES));
                }
                let wait = r
                    .header("retry-after")
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(std::time::Duration::from_secs)
                    .unwrap_or_else(|| std::time::Duration::from_millis(500u64.saturating_mul(1 << attempt)));
                std::thread::sleep(wait);
            }
            Err(e) => return Err(format!("HTTP error fetching {}: {}", url, e)),
        }
    };

    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Failed to read response body: {}", e))?;

    std::fs::create_dir_all(dest).map_err(|e| e.to_string())?;

    // A .crate file is a .tar.gz — pipe through `tar -xz`
    let mut child = std::process::Command::new("tar")
        .args(["-xz", "-C", dest.to_str().unwrap()])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn tar: {}", e))?;

    {
        use std::io::Write;
        child.stdin.as_mut().unwrap()
            .write_all(&bytes)
            .map_err(|e| format!("Failed to pipe to tar: {}", e))?;
    }
    child.wait().map_err(|e| e.to_string())?;

    let extracted = dest.join(format!("{}-{}", crate_name, version));
    if extracted.is_dir() {
        Ok(extracted)
    } else {
        Err(format!(
            "Expected extracted directory {} not found after tar",
            extracted.display()
        ))
    }
}
