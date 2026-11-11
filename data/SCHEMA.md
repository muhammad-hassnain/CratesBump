# CratesBump schema

Storage format, keys, column types and companion files of the headline file
`data/cratesbump.csv.gz`. Feature definitions are in
[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md); this file covers how the data is
stored and how it links to other data. Ranges and counts are from the
current file (2,079,790 rows, 219,955 crates). Re-run the checks in
[Constraints](#6-constraints) on every new release.

## 1. Storage

| Property | Value |
|---|---|
| Container | One file, gzip-compressed (`.csv.gz`); no database, no sharding |
| Format | CSV (RFC 4180 subset): comma separator, one header row, `\n` line endings, fields quoted only where needed (5 `repository` values of the crate `dummy-test-xss` contain `"`), no BOM |
| Encoding | UTF-8 (ASCII except 4 `repository` values) |
| Grain | One row per release pair `(prev_version → version)` of one crate |
| Size | 2,079,790 rows × 48 columns; 91.5 MB compressed |
| Missing values | Only `repository` can be empty (108,733 rows: the crate lists no repository). Measurements that could not be taken are `0` (see §4) |
| Numbers | All 41 features are written as decimals (`3.0`, `0.0204…`); some `acycd`/`rdtd` values use exponent notation (`1e-05`), so use a real CSV/float parser |
| Column order | `package, prev_version, version, release_type`, then the 41 features in the order of `scripts/pipeline.py` `FEATURE_COLS`, then `created_at, release_type_naive, repository` |
| License | CC BY 4.0 ([`LICENSE`](LICENSE)) |

```python
import pandas as pd
# keep_default_na=False: crates named "NULL" and "nan" exist; na_values=[""] keeps
# an empty repository as missing.
df = pd.read_csv("data/cratesbump.csv.gz", keep_default_na=False, na_values=[""])
```

## 2. Keys

- **Primary key: `(package, version)`.** Each release appears at most once as
  the newer end of a pair.
- `(package, prev_version)` is also unique: the pairs of a crate form a chain
  (each release has at most one successor), thinned by the per-crate cap.
- **Foreign keys into the crates.io database dump** (dump of 2026-10-12):
  - `package` = `crates.name` (exact string, case preserved).
  - `version` and `prev_version` = `versions.num` of that crate, verbatim,
    including any pre-release (`-rc.1`) and build-metadata (`+20150414`)
    suffix. `(crates.id, versions.num)` is unique in the dump.
  - From `versions.id` you reach `dependencies.version_id` (dependencies
    declared by that release), `version_downloads.version_id`, etc.
- **Grouping key: `repository`.** Crates of one repository share its GitHub
  features; hold out whole repositories when splitting (rows with an empty
  `repository` are their own group per crate).

## 3. Columns

### 3.1 Identifier, label, time and repository columns (7)

| Column | Type | Allowed values / range |
|---|---|---|
| `package` | string | crates.io crate name, `^[A-Za-z][A-Za-z0-9_-]*$`, ≤ 64 chars; 219,955 distinct; 1–100 rows per crate |
| `prev_version` | string (SemVer 2.0) | Predecessor chosen by `src/extractor/mod.rs::plan_pairs` (full SemVer precedence). All values parse; 19,810 carry a pre-release suffix, 7,048 build metadata |
| `version` | string (SemVer 2.0) | Newer release. All values parse; 48,823 carry a pre-release suffix, 7,240 build metadata. 2.58% of rows have a pre-release at either end |
| `release_type` | categorical | `major` / `minor` / `patch` under Cargo's rule (leftmost non-zero component is breaking). 737,262 / 1,148,632 / 193,896 |
| `release_type_naive` | categorical | `major` / `minor` / `patch` by the leftmost changed MAJOR.MINOR.PATCH component. 58,026 / 685,280 / 1,336,484 |
| `created_at` | timestamp (ISO 8601, UTC) | Publish time of `version`, copied from `versions.created_at`. Format `YYYY-MM-DDTHH:MM:SS.ffffffZ`. 2014-11-11T06:02:14Z … 2026-09-13T23:57:47Z |
| `repository` | string, may be empty | Normalised `crates.repository`: `github.com/owner/repo` (1,890,744 rows), host/path for other hosts (80,313 rows), empty (108,733 rows); 112,921 distinct |

### 3.2 Feature columns (41)

Type is the logical type; every feature is stored as a decimal. "Non-zero" is
the share of rows with a value ≠ 0. All features are ≥ 0 except `clcrd`.

**AST change-type (20).** Source: diff of the two unpacked `.crate`
tarballs and their `syn` ASTs.

| Column | Type | Unit | Range | Non-zero |
|---|---|---|---|---:|
| `arf` | int | `.rs` files added | 0–24,386 | 15.9% |
| `mrf` | int | `.rs` files modified | 0–24,387 | 61.5% |
| `drf` | int | `.rs` files deleted | 0–14,954 | 5.5% |
| `anrf` | int | non-`.rs` files added | 0–55,970 | 9.5% |
| `mnrf` | int | non-`.rs` files modified | 0–6,893 | 100% |
| `dnrf` | int | non-`.rs` files deleted | 0–80,779 | 3.4% |
| `adm` | int | functions/methods added | 0–593,343 | 30.1% |
| `dem` | int | functions/methods deleted | 0–428,301 | 13.8% |
| `mom` | int | functions moved to another module (identical body) | 0–365,975 | 5.8% |
| `mnc` | int | functions renamed in the same module (identical body) | 0–7,130 | 2.2% |
| `mpc` | int | functions whose parameter list changed | 0–4,467 | 6.0% |
| `mpd` | int | functions whose parameters were deleted | 0–4,054 | 2.2% |
| `mla` | int | logic blocks added | 0–438,440 | 29.8% |
| `mlm` | int | sum of absolute logic-block-count changes in functions kept under the same name | 0–139,859 | 35.9% |
| `mld` | int | logic blocks deleted | 0–716,699 | 5.7% |
| `gva` | int | global `static`/`const` items added | 0–186,582 | 5.0% |
| `gvd` | int | global `static`/`const` items deleted | 0–40,470 | 1.6% |
| `icc` | int | comment lines added (`.rs` files) | 0–780,843 | 35.8% |
| `dcc` | int | comment lines deleted (`.rs` files) | 0–725,953 | 13.9% |
| `mcc` | int | comment lines modified, `min(icc, dcc)` | 0–410,403 | 10.3% |

**Dependency (4).** Source: `Cargo.toml` diff, across `[dependencies]`,
`[dev-dependencies]` and `[build-dependencies]`
(`src/extractor/ast.rs::diff_cargo_toml`).

| Column | Type | Unit | Range | Non-zero |
|---|---|---|---|---:|
| `tcct` | flag | 1 if a dependency section changed, else 0 | 0–1 | 55.1% |
| `pa` | int | dependencies added | 0–380 | 12.3% |
| `pd` | int | dependencies removed | 0–135 | 5.3% |
| `pu` | int | dependencies whose version requirement changed | 0–356 | 48.3% |

**Complexity and size (5).** Source: tarball diff and AST.

| Column | Type | Unit | Range | Non-zero |
|---|---|---|---|---:|
| `acycd` | float | \|Δ mean cyclomatic complexity per function\| | 0–32,768 | 41.9% |
| `cycd` | int | \|Δ total cyclomatic complexity\| | 0–885,763 | 42.2% |
| `clcrd` | int (signed) | lines, `la − ld` | −27,196,794–21,000,132 | 66.1% |
| `la` | int | lines added, all text files of the tarball | 0–21,000,202 | 100% |
| `ld` | int | lines deleted, all text files of the tarball | 0–27,196,821 | 100% |

**Time (1).**

| Column | Type | Unit | Range | Non-zero |
|---|---|---|---|---:|
| `rdtd` | float | days from the publish time of `prev_version` to that of `version` | 0–3,897.81 (median 3.21) | 100% |

**GitHub commit activity (7).** Source: clone of the crate's repository,
commits in the window (publish time of `prev_version`, publish time of
`version`].

| Column | Type | Unit | Range | Non-zero |
|---|---|---|---|---:|
| `tcm` | int | commits | 0–122,659 | 83.7% |
| `tau` | int | distinct commit authors | 0–4,687 | 83.7% |
| `nbf` | int | commits whose subject matches `\b(fix\|bug\|defect\|error\|issue)` | 0–19,662 | 50.3% |
| `aml` | float | mean commit-subject length, characters (2 decimals) | 0–668,951.25 | 83.7% |
| `kwm` | int | commits whose lower-cased subject contains `major` | 0–116 | 1.3% |
| `kwp` | int | … contains `patch` | 0–429 | 6.6% |
| `kwb` | int | … contains `break` | 0–334 | 4.0% |

**GitHub issue/PR activity (4).** Source: the repository's issue and
pull-request lists (GitHub GraphQL and REST APIs), same window.

| Column | Type | Unit | Range | Non-zero |
|---|---|---|---|---:|
| `poi` | int | issues opened | 0–27,331 | 28.0% |
| `pci` | int | issues closed | 0–21,935 | 26.3% |
| `popr` | int | pull requests opened | 0–48,791 | 46.7% |
| `pcpr` | int | pull requests closed (merged or not) | 0–48,296 | 46.8% |

## 4. Missing values

Only `repository` can be empty. Values that could not be measured are `0`:

- **GitHub features (11).** All 0 in 333,309 rows (16.0%): the crate lists
  no repository (108,733 rows) or one outside GitHub (80,313); the GitHub URL
  names no repository, e.g. `github.com/<user>` (515); the GitHub repository
  could not be cloned, mostly because it was deleted or made private
  (82,909); the clone's history does not reach the release window, e.g.
  because the crate moved to another repository (29,934); or nothing happened
  in the window, mostly quick re-releases with a median gap of 1.4 hours
  (30,905). A 0 therefore means "none observed", not "none happened";
  `repository` separates only the first two cases (189,046 rows). 83.97% of
  all rows have at least one non-zero GitHub feature.
- `rdtd` is 0 in 33 rows where the two releases were published less than a
  second apart.
- `aml` is 0 in 47 rows with `tcm > 0` (commits with empty subjects).

## 5. Joining to the crates.io dump

Download and extract a dump from <https://crates.io/data-access>; the paper
uses the dump of 2026-10-12. Join on `(crates.name, versions.num)`; both
ends of every pair resolve.

```python
import pandas as pd

DUMP = "<dump>/data"  # data/ folder of the extracted db-dump.tar.gz
df = pd.read_csv("data/cratesbump.csv.gz", keep_default_na=False, na_values=[""])

crates = (pd.read_csv(f"{DUMP}/crates.csv", usecols=["id", "name"], keep_default_na=False)
            .rename(columns={"id": "crate_id", "name": "package"}))
versions = (pd.read_csv(f"{DUMP}/versions.csv", keep_default_na=False,
                        usecols=["id", "crate_id", "num", "license", "crate_size"])
              .merge(crates, on="crate_id"))

curr = versions.rename(columns={"num": "version", "id": "version_id"})
prev = versions[["package", "num", "id"]].rename(
    columns={"num": "prev_version", "id": "prev_version_id"})
df = (df.merge(curr, on=["package", "version"], how="left", validate="one_to_one")
        .merge(prev, on=["package", "prev_version"], how="left", validate="one_to_one"))
assert df["version_id"].notna().all() and df["prev_version_id"].notna().all()

# Example: dependencies declared by the newer release of each pair
# (dependencies.kind: 0 = normal, 1 = build, 2 = dev)
deps = pd.read_csv(f"{DUMP}/dependencies.csv", keep_default_na=False,
                   usecols=["version_id", "crate_id", "req", "kind", "optional"])
deps = deps.merge(crates.rename(columns={"package": "dependency"}), on="crate_id")
new_deps = df[["package", "version", "version_id"]].merge(deps, on="version_id")
```

Notes:

- A later dump also works for the join, but `versions.yanked` can change
  after the snapshot, and the date cut (releases before 2026-09-14T00:00Z) is
  already applied in the file.
- To compare versions, use a SemVer library (or `scripts/semver_order.py`),
  not string order.

## 6. Constraints

Checked on the current file (`scripts/label_pairs_cargo.py` report and a
separate verification pass).

| # | Constraint | Result |
|---|---|---|
| 1 | `(package, version)` unique | PASS |
| 2 | `(package, prev_version)` unique | PASS |
| 3 | Every `prev_version` and `version` parses as SemVer 2.0 | PASS |
| 4 | `prev_version < version` under SemVer precedence; MAJOR.MINOR.PATCH strictly increases | PASS |
| 5 | `release_type`, `release_type_naive` ∈ {`major`, `minor`, `patch`} | PASS |
| 6 | `release_type_naive` and `release_type` equal the naive and Cargo rules recomputed from the two versions | PASS (0 mismatches each) |
| 7 | `release_type ≠ release_type_naive` exactly when `version < 1.0.0` | PASS (1,712,585 rows) |
| 8 | `package` is in `crates.name`; `(package, version)` and `(package, prev_version)` are in `versions` | PASS |
| 9 | `created_at` equals `versions.created_at` of `version` | PASS (filled from the dump) |
| 10 | `created_at < 2026-09-14T00:00Z` | PASS |
| 11 | Neither end is yanked in the 2026-10-12 dump | PASS |
| 12 | `prev_version` is the `plan_pairs` predecessor of `version` | PASS (2,079,790) |
| 13 | `rdtd` equals the gap between the publish times of `prev_version` and `version` (±10⁻³ days) | PASS (checked on all 2,421,308 extracted rows) |
| 14 | ≤ 100 rows per crate; ≤ 5,000 per name family (name up to the first `-`/`_`) | PASS (1,738 crates at 100; 24 families at 5,000) |
| 15 | Every feature is numeric; all features ≥ 0 except `clcrd` | PASS |
| 16 | Every feature except `acycd`, `aml`, `rdtd` is integer-valued | PASS |
| 17 | `clcrd = la − ld`; `mcc = min(icc, dcc)` | PASS |
| 18 | `tau ≤ tcm`; `nbf`, `kwm`, `kwp`, `kwb ≤ tcm`; `tau > 0` exactly when `tcm > 0` | PASS |
| 19 | `tcct` ∈ {0, 1}, and `tcct = 1` exactly when `pa + pd + pu > 0` | PASS |
| 20 | All 11 GitHub features are 0 when `repository` is not a GitHub repository | PASS (189,046 rows) |
| 21 | Every `package` is in `results/crates_all.txt` and `results/crates_all_repos.csv` | PASS |
| 22 | `aml > 0` exactly when `tcm > 0` | **FAIL: 47 rows** (§4) |

## 7. Companion files

All are committed in the repository. Keys refer to headline columns.

| File | Format | Rows | Columns / fields | Key and link |
|---|---|---:|---|---|
| `results/crates_all.txt` | text, one crate name per line | 225,799 | crate name | Crate-selection frame: crates with ≥ 2 non-yanked releases before the snapshot. Every headline `package` is in it |
| `results/crates_all_repos.csv` | CSV, header | 225,799 | `package`, `repository` (`crates.repository` from the dump, trimmed; may be empty) | PK `package`; the headline `repository` column is this URL normalised |
| `results/scope_coverage.json` | JSON | – | crates and downloads in scope, share that never reached 1.0 | none |
| `results/cargo_label_report.json` | JSON | – | `input_rows`, `status` (rows kept or dropped per check, by the caps; repository counts), `output_rows`, `output_crates`, label shares, `flips`, `flip_cases`, `crosstab_naive_rows_cargo_cols` | none (build report) |
| `results/github_fill_report.json` | JSON | – | rows and repositories with GitHub data, repositories without commits or issues, coverage | none (build report) |
| `results/label_audit/per_pair.jsonl` | JSON Lines | 450 | see below | `(package, prev, curr)` = `(package, prev_version, version)` |
| `results/label_audit/hand_review.csv` | CSV, header | 31 | `package, prev, curr, shipped, verdict, compile_break, reason_kind, reason` | the flagged pairs of `per_pair.jsonl` |
| `results/label_audit/flag_context.csv` | CSV, header | 31 | per flagged pair: binary targets, private / doc(hidden) items, generated code, dependents, `=`-pinned share | same key |
| `results/label_audit/logs/*.txt` | text | – | full `cargo-semver-checks` output per checked pair | file name `<package>__<prev>__<curr>.<run>.txt` |
| `results/label_audit/summary.json` | JSON | – | outcomes, flag and confirmed rates with Wilson CIs, sandbox settings | none (summary) |
| `results/label_audit_lints_0.51.0.csv`, `results/label_audit_lints_0.22.0.csv` | CSV | 260, 45 | lint id, required update, default level / name in 0.51.0 | lint tables of the tool versions |
| `results/downstream_impact.json` | JSON | – | `definition`, counts, `exposed_relations_by_req_shape`, `top_30_by_exposed`, `req_population` | `top_30_by_exposed[*].(package, prev, curr)` = headline `(package, prev_version, version)` |
| `results/label_sanity_rebuilt_cargo.csv` | CSV, header | – | `package`, `version`, `release_type`, `flag`, `reason` | `(package, version)`; only flagged rows are listed |
| `results/paper_numbers.json` | JSON | – | `github_zeros` (the §4 breakdown, the two known bugs), `audit` (unchecked pairs, checkable share by year, `0.y.z` share, rates without pre-releases) | none (summary; `make paper-numbers`) |

`results/label_audit/per_pair.jsonl` fields (one object per audited pair).
Optional keys are absent unless noted:

| Field | Type | Meaning |
|---|---|---|
| `package`, `prev`, `curr` | string | pair key |
| `shipped` | categorical | the pair's `release_type` |
| `outcome` | categorical | `agree`, `shipped_higher`, `shipped_lower` (flag: the tool says a bigger bump was needed), `tool_error`, `download_failed` |
| `resolution` | categorical | `crate_lockfile` (Cargo.lock shipped in the `.crate`) or `fresh` (re-resolved) |
| `tool_required` | categorical; null on `tool_error` | bump the public-API diff requires (deny-level lints, all features except unstable ones) |
| `tool_required_default_features`, `outcome_default_features`, `failed_lints_default_features` | optional, flagged pairs only | the same check with the crate's default features only |
| `failed_lints` | list of strings, optional | lints that fired |
| `failed_items` | object, optional | lint → list of failing item paths with source locations |
| `seconds` | float, optional | wall time of the check |
| `detail` | string | short error detail (empty when none) |
| `first_attempt_error` | string, optional | error of the lockfile attempt before the `fresh` retry |
| `error_kind` | categorical, optional | for `tool_error`: `no_lib_target`, `build_failed`, `fetch_failed`, `offline_resolution`, `other` |
| `log_tail` | string, optional | end of the tool log (errors only; full logs under `logs/`) |
