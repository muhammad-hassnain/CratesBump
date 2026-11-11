/// HTTP clients for crates.io and the GitHub REST API.
use serde::Deserialize;

const CRATES_IO: &str = "https://crates.io/api/v1";
const GITHUB_API: &str = "https://api.github.com";
const UA: &str = "semver-ml-extractor/0.1.0 (+https://github.com/muhammad-hassnain/CratesBump)";

// crates.io 
/// One published version as returned by crates.io.
#[derive(Debug, Clone, Deserialize)]
pub struct CrateVersion {
    /// Semver string, e.g. "1.2.3"
    pub num: String,
    /// ISO-8601 publish date, e.g. "2023-04-01T12:00:00.000000+00:00"
    pub created_at: String,
}

#[derive(Deserialize)]
struct VersionsResp {
    versions: Vec<CrateVersion>,
}

#[derive(Deserialize)]
struct CrateResp {
    #[serde(rename = "crate")]
    info: CrateInfo,
}

#[derive(Deserialize)]
struct CrateInfo {
    repository: Option<String>,
}

/// Return all published versions for `crate_name`, oldest first.
pub fn get_versions(crate_name: &str) -> Result<Vec<CrateVersion>, String> {
    let url = format!("{}/crates/{}/versions", CRATES_IO, crate_name);
    let mut versions: Vec<CrateVersion> = get_json(&url, None)?
        .into_json::<VersionsResp>()
        .map_err(|e| e.to_string())?
        .versions;
    // crates.io returns newest-first; reverse to chronological order
    versions.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    Ok(versions)
}

/// Return the GitHub repo URL listed in the crate's metadata
pub fn get_repo_url(crate_name: &str) -> Result<Option<String>, String> {
    let url = format!("{}/crates/{}", CRATES_IO, crate_name);
    let resp: CrateInfo = get_json(&url, None)?
        .into_json::<CrateResp>()
        .map_err(|e| e.to_string())?
        .info;
    Ok(resp.repository)
}

// GitHub 

/// GitHub issue/PR representation 
#[derive(Debug, Deserialize)]
struct GhItem {
    state: String,
    created_at: String,
    closed_at: Option<String>,
    /// Present only on pull-request items.
    pull_request: Option<serde_json::Value>,
}

/// Count issues and PRs whose activity falls inside `[since, until]`.
/// Returns `(open_issues, closed_issues, open_prs, closed_prs)`.
pub fn count_issues_prs(
    owner: &str,
    repo: &str,
    since: &str, // ISO-8601 lower bound (inclusive)
    until: &str, // ISO-8601 upper bound (inclusive)
    token: Option<&str>,
) -> Result<(usize, usize, usize, usize), String> {
    let mut open_issues: usize = 0;
    let mut closed_issues: usize = 0;
    let mut open_prs: usize = 0;
    let mut closed_prs: usize = 0;
    let mut page: usize = 1;

    loop {
        let url = format!(
            "{}/repos/{}/{}/issues?state=all&since={}&per_page=100&page={}",
            GITHUB_API, owner, repo, since, page
        );
        let items: Vec<GhItem> = get_json(&url, token)?
            .into_json()
            .map_err(|e| e.to_string())?;

        if items.is_empty() {
            break;
        }

        for item in &items {
            let is_pr = item.pull_request.is_some();
            let created_in = item.created_at.as_str() >= since && item.created_at.as_str() <= until;
            let closed_in = item
                .closed_at
                .as_deref()
                .map(|c| c >= since && c <= until)
                .unwrap_or(false);

            match (is_pr, item.state.as_str(), created_in, closed_in) {
                (false, "open", true, _) => open_issues += 1,
                (false, "closed", _, true) => closed_issues += 1,
                (true, "open", true, _) => open_prs += 1,
                (true, "closed", _, true) => closed_prs += 1,
                _ => {}
            }
        }

        if items.len() < 100 { 
            break;
        }
        page += 1;
    }

    Ok((open_issues, closed_issues, open_prs, closed_prs))
}

// Commits between two dates

#[derive(Debug, Deserialize)]
struct GhCommit {
    commit: GhCommitInner,
}

#[derive(Debug, Deserialize)]
struct GhCommitInner {
    author: Option<GhCommitAuthor>,
    message: String,
}

#[derive(Debug, Deserialize)]
struct GhCommitAuthor {
    email: Option<String>,
}

/// Aggregate statistics over the commits in a release window.
#[derive(Debug, Default, Clone)]
pub struct CommitStats {
    /// Total commit count.
    pub total: usize,
    /// Number of unique author emails.
    pub unique_authors: usize,
    /// Commits whose subject mentions a bug-fix keyword.
    pub bug_fixes: usize,
    /// Commits whose subject mentions "major".
    pub kw_major: usize,
    /// Commits whose subject mentions "patch".
    pub kw_patch: usize,
    /// Commits whose subject mentions "break".
    pub kw_break: usize,
    /// Average full-message length, in characters.
    pub avg_msg_len: f64,
}

/// Fetch commits in `[since, until]` via the GitHub REST API and aggregate
/// them into the commit-log features used by the model.
pub fn commit_stats(
    owner: &str,
    repo: &str,
    since: &str,
    until: &str,
    token: Option<&str>,
) -> Result<CommitStats, String> {
    let bug_keywords = ["fix", "bug", "defect", "error", "issue", "patch"];
    let mut stats = CommitStats::default();
    let mut authors: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut total_msg_len: usize = 0;
    let mut page: usize = 1;

    loop {
        let url = format!(
            "{}/repos/{}/{}/commits?since={}&until={}&per_page=100&page={}",
            GITHUB_API, owner, repo, since, until, page
        );
        let items: Vec<GhCommit> = get_json(&url, token)?
            .into_json()
            .map_err(|e| e.to_string())?;

        if items.is_empty() {
            break;
        }

        for c in &items {
            stats.total += 1;
            if let Some(ref a) = c.commit.author {
                if let Some(ref email) = a.email {
                    authors.insert(email.clone());
                }
            }
            let subj = c.commit.message
                .lines()
                .next()
                .unwrap_or("")
                .to_lowercase();
            if bug_keywords.iter().any(|kw| subj.contains(kw)) { stats.bug_fixes += 1; }
            if subj.contains("major") { stats.kw_major += 1; }
            if subj.contains("patch") { stats.kw_patch += 1; }
            if subj.contains("break") { stats.kw_break += 1; }
            total_msg_len += c.commit.message.len();
        }

        if items.len() < 100 {
            break;
        }
        page += 1;
    }

    stats.unique_authors = authors.len();
    stats.avg_msg_len = if stats.total == 0 {
        0.0
    } else {
        total_msg_len as f64 / stats.total as f64
    };
    Ok(stats)
}

// HTTP helper

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Minimum spacing between crates.io/GitHub API requests, applied *globally*
/// across all worker threads (crates.io rate-limits per source IP, so a
/// per-worker delay would still burst). Override with SEMVER_ML_API_DELAY_MS.
fn api_min_interval() -> Duration {
    static D: OnceLock<Duration> = OnceLock::new();
    *D.get_or_init(|| {
        let ms = std::env::var("SEMVER_ML_API_DELAY_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(200);
        Duration::from_millis(ms)
    })
}

/// Block until at least `api_min_interval()` has elapsed since the previous
/// API call from any thread. Holding the mutex across the sleep serializes
/// callers so the aggregate request rate stays under the limit.
fn throttle() {
    static LAST: OnceLock<Mutex<Instant>> = OnceLock::new();
    let lock = LAST.get_or_init(|| Mutex::new(Instant::now() - Duration::from_secs(3600)));
    let mut last = lock.lock().unwrap();
    let interval = api_min_interval();
    let elapsed = last.elapsed();
    if elapsed < interval {
        std::thread::sleep(interval - elapsed);
    }
    *last = Instant::now();
}

fn get_json(url: &str, token: Option<&str>) -> Result<ureq::Response, String> {
    const MAX_RETRIES: u32 = 5;
    let mut attempt: u32 = 0;
    loop {
        throttle();
        let mut req = ureq::get(url)
            .set("User-Agent", UA)
            .set("Accept", "application/json");
        if let Some(t) = token {
            req = req.set("Authorization", &format!("token {}", t));
        }
        match req.call() {
            Ok(resp) => return Ok(resp),
            // 429 Too Many Requests: back off and retry (respect Retry-After).
            Err(ureq::Error::Status(429, resp)) => {
                attempt += 1;
                if attempt > MAX_RETRIES {
                    return Err(format!("status code 429 (gave up after {} retries)", MAX_RETRIES));
                }
                let wait = resp
                    .header("retry-after")
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(Duration::from_secs)
                    .unwrap_or_else(|| Duration::from_millis(500u64.saturating_mul(1 << attempt)));
                eprintln!("(429 on {}; backing off {:?}, retry {}/{})", url, wait, attempt, MAX_RETRIES);
                std::thread::sleep(wait);
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

//  GitHub URL parsing 

/// Extract `(owner, repo)` from a GitHub URL such as
/// `https://github.com/tokio-rs/tokio` or `git@github.com:tokio-rs/tokio`.
pub fn parse_github_owner_repo(url: &str) -> Option<(String, String)> {
    // Strip trailing .git
    let url = url.trim_end_matches(".git");

    // HTTPS form
    if let Some(rest) = url.strip_prefix("https://github.com/") {
        let parts: Vec<&str> = rest.splitn(2, '/').collect();
        if parts.len() == 2 {
            return Some((parts[0].to_string(), parts[1].to_string()));
        }
    }
    // SSH form
    if let Some(rest) = url.strip_prefix("git@github.com:") {
        let parts: Vec<&str> = rest.splitn(2, '/').collect();
        if parts.len() == 2 {
            return Some((parts[0].to_string(), parts[1].to_string()));
        }
    }
    None
}
