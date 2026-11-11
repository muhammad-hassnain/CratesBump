/// All 44 release-level features: the 41 from the paper plus three
/// Rust-specific Cargo signals (fa/fd/edc).
/// We have six dimensions: change types, dependency, complexity/code, time,
/// development activities, and textual.
///
/// API-surface note: the change-type features that describe the public API
/// (adm, dem, mpc, mpd, gva, gvd) are computed over `pub` items ONLY — private
/// items cannot affect a crate's SemVer contract. Line/comment/complexity
/// features are whole-file. See `extractor::ast` for the exact keying rules.
///
/// CSV column order is ALPHABETICAL by field name (see `data::write_csv`), not
/// struct-field order, so these fields may be declared in any order.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ReleaseFeatures {
    // Change-type features (20)
    /// Rust files added between two releases
    pub arf: f64,
    /// Rust files modified between two releases
    pub mrf: f64,
    /// Rust files deleted between two releases
    pub drf: f64,
    /// Non-Rust files added
    pub anrf: f64,
    /// Non-Rust files deleted
    pub dnrf: f64,
    /// Non-Rust files modified
    pub mnrf: f64,
    /// Public API items added (public fns/methods + public re-exports +
    /// exported macros). Column name kept for schema stability.
    pub adm: f64,
    /// Public API items deleted (see `adm`).
    pub dem: f64,
    /// Methods moved
    pub mom: f64,
    /// Methods whose names changed
    pub mnc: f64,
    /// Public methods/functions whose parameter count changed
    pub mpc: f64,
    /// Public methods/functions whose parameter count decreased
    pub mpd: f64,
    /// Logic blocks added inside methods
    pub mla: f64,
    /// Logic blocks moved inside methods
    pub mlm: f64,
    /// Logic blocks deleted inside methods
    pub mld: f64,
    /// Public "globals" added. Redefined: public item-level `const`/`static`
    /// AND public associated `const`s; function-local consts are excluded.
    /// Column name kept for schema stability.
    pub gva: f64,
    /// Public "globals" deleted (see `gva`).
    pub gvd: f64,
    /// Code comments added
    pub icc: f64,
    /// Code comments deleted
    pub dcc: f64,
    /// Code comments modified
    pub mcc: f64,

    // Dependency features (7)
    /// Dependency sections changed (0/1). Redefined: 1 iff a
    /// [dependencies]/[dev-]/[build-] entry was actually added/removed/updated.
    /// (Every published .crate bumps the Cargo.toml version line, so the old
    /// "Cargo.toml touched" signal was ~always 1.) Column name kept.
    pub tcct: f64,
    /// Packages added to dependencies
    pub pa: f64,
    /// Packages removed from dependencies
    pub pd: f64,
    /// Packages whose pinned versions changed
    pub pu: f64,
    /// Feature flags added to the [features] table
    pub fa: f64,
    /// Feature flags removed from the [features] table
    pub fd: f64,
    /// Edition changed between releases (0 or 1)
    pub edc: f64,

    // Complexity / code features (5) 
    /// change in average cyclomatic complexity
    pub acycd: f64,
    /// change in lines of code (CLOC diff)
    pub clcrd: f64,
    /// change in total cyclomatic complexity
    pub cycd: f64,
    /// Lines of code added
    pub la: f64,
    /// Lines of code deleted
    pub ld: f64,

    //  Time feature (1) 
    /// Days since the previous release
    pub rdtd: f64,

    // Development-activity features (7) 
    /// Total commits in this release window
    pub tcm: f64,
    /// Unique authors in this release window
    pub tau: f64,
    /// Open issues during this release window
    pub poi: f64,
    /// Closed issues during this release window
    pub pci: f64,
    /// Closed PRs during this release window
    pub pcpr: f64,
    /// Open PRs during this release window
    pub popr: f64,
    /// Bug-fixing commits (keyword heuristic)
    pub nbf: f64,

    //  Textual features (4) 
    /// Commits containing "major"
    pub kwm: f64,
    /// Commits containing "patch"
    pub kwp: f64,
    /// Commits containing "break"
    pub kwb: f64,
    /// Average commit-message length (characters)
    pub aml: f64,
}

