/// Filesystem-based diff helpers — 
///
/// All git-CLI functions (clone, diff, log, etc.) were removed when the
/// pipeline switched to comparing crates.io tarballs directly.  Only the
/// two public entry-points (`diff_stats_dirs`, `comment_diff_dirs`) and
/// their private helpers remain.
use std::path::Path;

// Public types

pub struct DiffStats {
    pub rs_added: usize,
    pub rs_modified: usize,
    pub rs_deleted: usize,
    pub other_added: usize,
    pub other_modified: usize,
    pub other_deleted: usize,
    pub lines_added: usize,
    pub lines_deleted: usize,
}

pub struct CommentDiff {
    pub added: usize,
    pub deleted: usize,
}

// Public API

/// Compare two source trees on the filesystem and return file/line-count
/// statistics — same fields as the old git-based `diff_stats`.
pub fn diff_stats_dirs(prev_dir: &Path, curr_dir: &Path) -> DiffStats {
    let mut s = DiffStats {
        rs_added: 0, rs_modified: 0, rs_deleted: 0,
        other_added: 0, other_modified: 0, other_deleted: 0,
        lines_added: 0, lines_deleted: 0,
    };

    let prev_files = collect_files(prev_dir);
    let curr_files = collect_files(curr_dir);

    let prev_paths: std::collections::HashSet<&String> = prev_files.keys().collect();
    let curr_paths: std::collections::HashSet<&String> = curr_files.keys().collect();

    // Files added in curr
    for path in curr_paths.difference(&prev_paths) {
        let is_rs    = path.ends_with(".rs");
        s.lines_added += curr_files[*path].lines().count();
        if is_rs { s.rs_added += 1; } else { s.other_added += 1; }
    }

    // Files deleted in curr
    for path in prev_paths.difference(&curr_paths) {
        let is_rs    = path.ends_with(".rs");
        s.lines_deleted += prev_files[*path].lines().count();
        if is_rs { s.rs_deleted += 1; } else { s.other_deleted += 1; }
    }

    // Files present in both — check for modifications
    for path in prev_paths.intersection(&curr_paths) {
        if prev_files[*path] == curr_files[*path] { continue; }
        let is_rs    = path.ends_with(".rs");
        let (a, d) = line_diff_count(&prev_files[*path], &curr_files[*path]);
        s.lines_added   += a;
        s.lines_deleted += d;
        if is_rs { s.rs_modified += 1; } else { s.other_modified += 1; }
    }

    s
}

/// Count comment lines added/deleted in `.rs` files between two source trees.
pub fn comment_diff_dirs(prev_dir: &Path, curr_dir: &Path) -> CommentDiff {
    let prev_files = collect_files(prev_dir);
    let curr_files = collect_files(curr_dir);

    // Use the single block-comment-aware counter from `ast` (counts both `//`
    // line comments and `/* … */` block comments) so the comment features are
    // consistent with it — the old `//`-only closure here undercounted block
    // comments and disagreed with the AST snapshot's counter.
    let comment_count = |contents: &str| -> usize {
        crate::extractor::ast::count_comment_lines(contents)
    };

    let prev_cc: std::collections::HashMap<String, usize> = prev_files.iter()
        .filter(|(k, _)| k.ends_with(".rs"))
        .map(|(k, v)| (k.clone(), comment_count(v)))
        .collect();

    let curr_cc: std::collections::HashMap<String, usize> = curr_files.iter()
        .filter(|(k, _)| k.ends_with(".rs"))
        .map(|(k, v)| (k.clone(), comment_count(v)))
        .collect();

    let mut added   = 0usize;
    let mut deleted = 0usize;
    let all_keys: std::collections::HashSet<&String> =
        prev_cc.keys().chain(curr_cc.keys()).collect();
    for key in all_keys {
        let pc = *prev_cc.get(key).unwrap_or(&0);
        let cc = *curr_cc.get(key).unwrap_or(&0);
        if cc > pc { added   += cc - pc; }
        if pc > cc { deleted += pc - cc; }
    }
    CommentDiff { added, deleted }
}

// Private helpers

/// Directories that exist in a working copy but not in a published `.crate`
/// tarball — skipping them keeps the filesystem diff focused on source code.
fn is_noise_dir(name: &str) -> bool {
    matches!(
        name,
        ".git" | "target" | ".github" | ".idea" | ".vscode" | "node_modules"
    )
}

/// Walk `dir` and return a map of relative-path → file-contents, skipping
/// noise directories and crates.io-specific metadata files.
fn collect_files(dir: &Path) -> std::collections::HashMap<String, String> {
    use walkdir::WalkDir;
    let mut map = std::collections::HashMap::new();
    let walker = WalkDir::new(dir).sort_by_file_name().into_iter().filter_entry(|e| {
        if e.depth() == 0 { return true; }
        let name = e.file_name().to_string_lossy();
        if e.file_type().is_dir() && is_noise_dir(&name) {
            return false;
        }
        true
    });
    for entry in walker.filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() { continue; }
        let rel = entry.path()
            .strip_prefix(dir).ok()
            .and_then(|p| p.to_str())
            .map(|s| s.replace('\\', "/"))
            .unwrap_or_default();
        if rel.is_empty() { continue; }
        // Files that crates.io adds to the tarball but are absent from a
        // checkout (or vice versa) — skip so they don't appear as spurious diffs.
        if rel == ".cargo_vcs_info.json"
            || rel == "Cargo.toml.orig"
            || rel == "Cargo.lock"
            || rel == ".gitignore"
        {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(entry.path()) {
            map.insert(rel, content);
        }
    }
    map
}

/// Count lines added and deleted between two file contents using a
/// bag-of-lines approach (fast, order-insensitive approximation).
fn line_diff_count(prev: &str, curr: &str) -> (usize, usize) {
    let mut counts: std::collections::HashMap<&str, i64> = std::collections::HashMap::new();
    for line in prev.lines() { *counts.entry(line).or_default() -= 1; }
    for line in curr.lines() { *counts.entry(line).or_default() += 1; }
    let (mut added, mut deleted) = (0usize, 0usize);
    for v in counts.values() {
        if *v > 0      { added   += *v as usize; }
        else if *v < 0 { deleted += (-v) as usize; }
    }
    (added, deleted)
}
