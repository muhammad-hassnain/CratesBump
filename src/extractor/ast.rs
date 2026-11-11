/// Rust source analysis using `syn`.
///
/// For each release we build a `Snapshot` containing:
///   - The PUBLIC API surface: public functions/methods (with parameter
///     counts), public consts/statics/associated-consts, public re-exports
///     and exported macros.
///   - Aggregate cyclomatic complexity and logic-block count (WHOLE FILE —
///     these are code-complexity metrics, not API-surface metrics).
///
/// Diffing two snapshots yields most of the Change-type and Complexity
/// feature columns from Table 3 of arXiv:2204.05929.
///
/// SemVer/public-API framing: the API-surface features (adm/dem/mpc/mpd/
/// gva/gvd) are restricted to `pub` items only, because private items cannot
/// affect a crate's public SemVer contract. `pub(crate)` / `pub(super)` /
/// `pub(in …)` are Visibility::Restricted and are deliberately NOT counted —
/// they are not part of the external API. Complexity/comment/line features are
/// left whole-file.
///
/// Keying is SYNTACTIC (paths are not name-resolved): a method is keyed by
/// module path + trait name (for trait impls) + Self type WITH generic
/// arguments + method name. This distinguishes e.g. `impl Display for Foo`
/// from `impl Debug for Foo` (both `fn fmt`) and `Foo<A>` from `Foo<B>`. Two
/// residual approximations remain and are accepted: (a) between two versions,
/// an import-style change (`fmt::Display` vs `std::fmt::Display`) or a
/// generic-parameter rename (`impl<T>` vs `impl<U>`) can re-key an otherwise
/// unchanged item; (b) items that still collide on a key are counted by
/// MULTIPLICITY (parameter-count multiset) rather than silently overwritten,
/// so no item is ever lost.
use std::collections::{HashMap, HashSet};
use syn::visit::{self, Visit};
use walkdir::WalkDir;

// Public snapshot type

#[derive(Debug, Default, Clone)]
pub struct Snapshot {
    /// Public functions/methods keyed by a fully-qualified signature key,
    /// mapped to the MULTISET of their parameter counts. A `Vec` (not a
    /// scalar) so that residual key collisions are counted rather than
    /// overwritten. Feeds adm/dem/mpc/mpd.
    pub fns: HashMap<String, Vec<usize>>,
    /// Public item-level `const`/`static` and public associated `const`
    /// (module/impl-qualified). Feeds gva/gvd. Function-local consts excluded.
    pub globals: HashSet<String>,
    /// Other public API items that add to the added/deleted counts but have no
    /// parameters: `pub use` re-exports and exported (`#[macro_export]`)
    /// macros. Folded into adm/dem.
    pub api_items: HashSet<String>,
    /// Sum of cyclomatic complexity over all functions (WHOLE FILE — includes
    /// private fns; complexity is not an API-surface metric).
    pub total_cc: usize,
    /// Total number of functions (whole file).
    pub fn_count: usize,
    /// Total logic-branch nodes (if / while / for / loop / match arms), whole file.
    pub logic_blocks: usize,
    /// For each PUBLIC function/method we record a body fingerprint (hash of
    /// the Debug representation of the body `Block`) together with the
    /// containing (module_prefix, name). This supports move/rename detection
    /// across snapshots (mom/mnc/mlm). The map value is a `Vec` so repeated
    /// identical bodies (e.g. generated code) can be matched by multiset.
    pub fn_bodies: HashMap<u64, Vec<FnBody>>,
}

/// Body-fingerprint record attached to a function. Used by `diff` to detect
/// moves (mom), renames (mnc), and logic rearrangements (mlm).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FnBody {
    /// Module path prefix the function lives under (`a::b::`, empty at root).
    pub module_prefix: String,
    /// Final name of the function/method.
    pub name: String,
    /// Logic-block count inside this specific function (if / for / while / match arms).
    pub logic_count: usize,
}

impl Snapshot {
    pub fn avg_cc(&self) -> f64 {
        if self.fn_count == 0 {
            0.0
        } else {
            self.total_cc as f64 / self.fn_count as f64
        }
    }
}

/// Build a `Snapshot` by walking every `.rs` file under `root`.
pub fn snapshot_dir(root: &std::path::Path) -> Snapshot {
    let mut snap = Snapshot::default();
    // Sorted walk: file order feeds the multiset matching below, so it must
    // not depend on the filesystem's directory order.
    for entry in WalkDir::new(root).sort_by_file_name().into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(src) = std::fs::read_to_string(entry.path()) {
                merge(&mut snap, &analyse_source(&src));
            }
        }
    }
    snap
}

// for merging per-file snapshots into a whole-release snapshot
fn merge(dst: &mut Snapshot, src: &Snapshot) {
    // Extend (don't overwrite) so multiplicity is preserved across files.
    for (k, v) in &src.fns {
        dst.fns.entry(k.clone()).or_default().extend_from_slice(v);
    }
    for g in &src.globals {
        dst.globals.insert(g.clone());
    }
    for a in &src.api_items {
        dst.api_items.insert(a.clone());
    }
    dst.total_cc += src.total_cc;
    dst.fn_count += src.fn_count;
    dst.logic_blocks += src.logic_blocks;
    for (hash, bodies) in &src.fn_bodies {
        dst.fn_bodies
            .entry(*hash)
            .or_default()
            .extend_from_slice(bodies);
    }
}

/// Hash the Debug representation of a syn::Block to produce a stable
/// fingerprint for move/rename matching. Two bodies with identical token
/// structure produce the same hash; whitespace and comments are stripped by
/// syn's parser so they do not affect the fingerprint.
fn body_fingerprint(block: &syn::Block) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    // Debug on syn::Block walks the full AST deterministically.
    format!("{:?}", block).hash(&mut hasher);
    hasher.finish()
}

/// Count logic branches (if / for / while / loop / match arms, || and &&)
/// inside a single Block. Used to populate `FnBody.logic_count` so we can
/// detect "logic moved within a function" (mlm).
fn block_logic_count(block: &syn::Block) -> usize {
    use syn::visit::Visit;
    struct LogicCounter(usize);
    impl<'ast> Visit<'ast> for LogicCounter {
        fn visit_expr_if(&mut self, n: &'ast syn::ExprIf) {
            self.0 += 1;
            syn::visit::visit_expr_if(self, n);
        }
        fn visit_expr_for_loop(&mut self, n: &'ast syn::ExprForLoop) {
            self.0 += 1;
            syn::visit::visit_expr_for_loop(self, n);
        }
        fn visit_expr_while(&mut self, n: &'ast syn::ExprWhile) {
            self.0 += 1;
            syn::visit::visit_expr_while(self, n);
        }
        fn visit_expr_loop(&mut self, n: &'ast syn::ExprLoop) {
            self.0 += 1;
            syn::visit::visit_expr_loop(self, n);
        }
        fn visit_arm(&mut self, n: &'ast syn::Arm) {
            self.0 += 1;
            syn::visit::visit_arm(self, n);
        }
    }
    let mut c = LogicCounter(0);
    c.visit_block(block);
    c.0
}

// Diff - all metrics are from Table 3 of paper
pub struct AstDiff {
    /// Public API items present in `after` but not `before` (ADM). Includes
    /// added function/method occurrences (by multiset) plus added re-exports
    /// and exported macros.
    pub added_fns: usize,
    /// Public API items present in `before` but not `after` (DEM).
    pub deleted_fns: usize,
    /// Functions where parameter count changed (MPC).
    pub param_changed_fns: usize,
    /// Subset of `param_changed_fns` where count decreased (MPD).
    pub param_deleted_fns: usize,
    /// Globals added (GVA).
    pub added_globals: usize,
    /// Globals deleted (GVD).
    pub deleted_globals: usize,
    /// Net change in logic blocks (positive means added).
    pub logic_added: usize,
    pub logic_deleted: usize,
    /// Change in total cyclomatic complexity (CYCD).
    pub cc_delta: i64,
    /// Change in average cyclomatic complexity (ACYCD).
    pub avg_cc_delta: f64,
    /// Methods Moved (MOM): count of functions whose body fingerprint is
    /// identical between snapshots but whose containing module path changed
    /// (same body, different file/module location).
    pub methods_moved: usize,
    /// Method Name Changes (MNC): count of functions whose body fingerprint is
    /// identical between snapshots and module path is the same, but final
    /// name differs. Pure renames.
    pub method_name_changes: usize,
    /// Logic blocks Moved (MLM): for functions matched by (module_prefix, name)
    /// across snapshots whose body fingerprint differs, the sum of absolute
    /// deltas in per-function logic-block counts. Measures logic rearrangement
    /// inside preserved methods. Heuristic — documented in the data card.
    pub logic_moved: usize,
}

pub fn diff(before: &Snapshot, after: &Snapshot) -> AstDiff {
    // Functions/methods compared as a multiset keyed by qualified signature.
    // In the common case (multiplicity 1 per key) this reduces exactly to the
    // old "present-in-after-not-before" logic; when a key holds several items
    // (residual collision) the extra occurrences count as added/deleted and
    // the overlapping ones are matched by sorted parameter count.
    let mut added_fns = 0usize;
    let mut deleted_fns = 0usize;
    let mut param_changed = 0usize;
    let mut param_deleted = 0usize;

    let empty: Vec<usize> = Vec::new();
    let keys: HashSet<&String> = before.fns.keys().chain(after.fns.keys()).collect();
    for k in keys {
        let mut pb = before.fns.get(k).unwrap_or(&empty).clone();
        let mut pa = after.fns.get(k).unwrap_or(&empty).clone();
        pb.sort_unstable();
        pa.sort_unstable();
        let overlap = pb.len().min(pa.len());
        if pa.len() > pb.len() {
            added_fns += pa.len() - pb.len();
        }
        if pb.len() > pa.len() {
            deleted_fns += pb.len() - pa.len();
        }
        for i in 0..overlap {
            if pa[i] != pb[i] {
                param_changed += 1;
                if pa[i] < pb[i] {
                    param_deleted += 1;
                }
            }
        }
    }

    // Other public API items (re-exports, exported macros) fold into the
    // added/deleted API-item counts. They carry no parameters, so they never
    // contribute to mpc/mpd.
    added_fns += after.api_items.difference(&before.api_items).count();
    deleted_fns += before.api_items.difference(&after.api_items).count();

    let added_globals = after.globals.difference(&before.globals).count();
    let deleted_globals = before.globals.difference(&after.globals).count();

    let logic_added = after.logic_blocks.saturating_sub(before.logic_blocks);
    let logic_deleted = before.logic_blocks.saturating_sub(after.logic_blocks);

    let cc_delta = after.total_cc as i64 - before.total_cc as i64;

    // --- MOM / MNC / MLM via body-fingerprint matching -------------------
    // Pass 1: for each body-fingerprint that appears in BOTH snapshots,
    // match multiset instances greedily. For each matched pair, classify:
    //   same (module, name)                 → unchanged, ignore
    //   same name, different module         → move     (mom++)
    //   same module, different name         → rename   (mnc++)
    //   different module AND different name → treat as mom (dominant change)
    let mut methods_moved = 0usize;
    let mut method_name_changes = 0usize;
    let body_keys: HashSet<&u64> =
        before.fn_bodies.keys().chain(after.fn_bodies.keys()).collect();
    for h in body_keys {
        let empty: Vec<FnBody> = Vec::new();
        let bs = before.fn_bodies.get(h).unwrap_or(&empty);
        let as_ = after.fn_bodies.get(h).unwrap_or(&empty);
        let overlap = bs.len().min(as_.len());
        if overlap == 0 {
            continue;
        }
        // Greedy matching: for each `after` body, find the first `before`
        // body with the same (module, name) if possible, else first unmatched.
        let mut taken = vec![false; bs.len()];
        for ab in as_.iter().take(overlap) {
            // Prefer unchanged matches (same module+name) so they don't
            // steal match slots from genuine moves/renames.
            let mut idx = bs
                .iter()
                .enumerate()
                .position(|(i, bb)| !taken[i] && bb.module_prefix == ab.module_prefix && bb.name == ab.name);
            if idx.is_none() {
                idx = bs.iter().enumerate().position(|(i, _)| !taken[i]);
            }
            let Some(i) = idx else { continue };
            taken[i] = true;
            let bb = &bs[i];
            if bb.module_prefix == ab.module_prefix && bb.name == ab.name {
                continue; // unchanged
            } else if bb.module_prefix != ab.module_prefix {
                methods_moved += 1;
            } else {
                method_name_changes += 1;
            }
        }
    }

    // --- MLM: logic blocks that moved inside preserved methods ------------
    // For functions matched by (module_prefix, name) across snapshots where
    // the body fingerprint differs AND the per-function logic count differs,
    // accumulate the absolute delta. We index each snapshot's bodies by
    // (module_prefix, name) → earliest occurrence.
    // Several functions can share (module_prefix, name), e.g. `fmt` in many
    // trait impls of one module. Pick one per location deterministically
    // (smallest (fingerprint, logic_count)); iterating the HashMap directly
    // made the choice, and so `mlm`, vary from run to run.
    let mut before_entries: Vec<((String, String), u64, usize)> = before
        .fn_bodies
        .iter()
        .flat_map(|(h, bodies)| {
            bodies
                .iter()
                .map(move |b| ((b.module_prefix.clone(), b.name.clone()), *h, b.logic_count))
        })
        .collect();
    before_entries.sort();
    let mut before_by_loc: HashMap<(String, String), (u64, usize)> = HashMap::new();
    for (key, h, lc) in before_entries {
        before_by_loc.entry(key).or_insert((h, lc));
    }
    let mut logic_moved = 0usize;
    for (h_after, bodies) in &after.fn_bodies {
        for b in bodies {
            let key = (b.module_prefix.clone(), b.name.clone());
            if let Some((h_before, lc_before)) = before_by_loc.get(&key) {
                if h_before != h_after && *lc_before != b.logic_count {
                    logic_moved += (b.logic_count as isize - *lc_before as isize).unsigned_abs();
                }
            }
        }
    }

    AstDiff {
        added_fns,
        deleted_fns,
        param_changed_fns: param_changed,
        param_deleted_fns: param_deleted,
        added_globals,
        deleted_globals,
        logic_added,
        logic_deleted,
        cc_delta,
        avg_cc_delta: after.avg_cc() - before.avg_cc(),
        methods_moved,
        method_name_changes,
        logic_moved,
    }
}

//  Per-file analysis

fn analyse_source(src: &str) -> Snapshot {
    let mut snap = Snapshot::default();

    if let Ok(ast) = syn::parse_file(src) {
        let mut v = AstVisitor {
            snap: &mut snap,
            mod_path: Vec::new(),
            impl_ctx: None,
            fn_depth: 0,
        };
        v.visit_file(&ast);
    }
    snap
}

// This struct for walking the AST and populating the Snapshot.

struct AstVisitor<'a> {
    snap: &'a mut Snapshot,
    /// Idents of the enclosing inline `mod`s, so items in different modules of
    /// the same file get distinct keys.
    mod_path: Vec<String>,
    /// Current impl context (Self type rendered with generics, plus the trait
    /// name when this is a trait impl). `None` when not inside an impl block.
    impl_ctx: Option<ImplCtx>,
    /// Depth of enclosing fn/method bodies. `> 0` means "inside a function":
    /// used to exclude function-local `const`/`static`/`fn` from the API
    /// surface (they are not public API).
    fn_depth: usize,
}

struct ImplCtx {
    /// Self type rendered with generic args, e.g. `Foo<T>`.
    self_ty: String,
    /// Trait name (path with generics) if this is a trait impl.
    trait_name: Option<String>,
}

impl<'a> AstVisitor<'a> {
    /// `a::b::` prefix for the current module path (empty at crate root).
    fn mod_prefix(&self) -> String {
        if self.mod_path.is_empty() {
            String::new()
        } else {
            format!("{}::", self.mod_path.join("::"))
        }
    }

    /// Whether the current impl block is a trait impl (its methods/consts are
    /// public API regardless of an explicit `pub`).
    fn in_trait_impl(&self) -> bool {
        self.impl_ctx
            .as_ref()
            .map(|c| c.trait_name.is_some())
            .unwrap_or(false)
    }

    /// Key for an associated item (method or assoc const) in the current impl,
    /// e.g. `a::b::<Display>Foo<T>::fmt` or `a::Foo<T>::new`.
    fn assoc_key(&self, name: &str) -> String {
        let prefix = self.mod_prefix();
        match &self.impl_ctx {
            Some(ctx) => match &ctx.trait_name {
                Some(tr) => format!("{}<{}>{}::{}", prefix, tr, ctx.self_ty, name),
                None => format!("{}{}::{}", prefix, ctx.self_ty, name),
            },
            None => format!("{}{}", prefix, name),
        }
    }
}

impl<'a, 'ast> Visit<'ast> for AstVisitor<'a> {
    // Inline modules — track the path so keys are module-qualified.
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        self.mod_path.push(node.ident.to_string());
        visit::visit_item_mod(self, node);
        self.mod_path.pop();
    }

    //  Free functions (not in impl blocks)
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        // Whole-file complexity: count every function, public or not.
        let cc = cyclomatic_complexity(&node.block);
        self.snap.total_cc += cc;
        self.snap.fn_count += 1;

        // API surface: only module-level `pub` free functions.
        if self.fn_depth == 0 && is_pub(&node.vis) {
            let key = format!("{}{}", self.mod_prefix(), node.sig.ident);
            self.snap
                .fns
                .entry(key)
                .or_default()
                .push(node.sig.inputs.len());

            // Body fingerprint for move/rename detection.
            let fp = body_fingerprint(&node.block);
            let lc = block_logic_count(&node.block);
            let module_prefix = self.mod_prefix();
            let fn_name = node.sig.ident.to_string();
            self.snap.fn_bodies.entry(fp).or_default().push(FnBody {
                module_prefix,
                name: fn_name,
                logic_count: lc,
            });
        }

        self.fn_depth += 1;
        visit::visit_item_fn(self, node);
        self.fn_depth -= 1;
    }

    // Methods in impl blocks
    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let cc = cyclomatic_complexity(&node.block);
        self.snap.total_cc += cc;
        self.snap.fn_count += 1;

        // Trait-impl methods are public API even without `pub`; inherent-impl
        // methods must be `pub`.
        if self.fn_depth == 0 && (self.in_trait_impl() || is_pub(&node.vis)) {
            let key = self.assoc_key(&node.sig.ident.to_string());
            self.snap
                .fns
                .entry(key)
                .or_default()
                .push(node.sig.inputs.len());

            // Body fingerprint for move/rename detection. For impl-methods
            // the "module_prefix" we track is mod_prefix + impl Self-ty, so
            // that moving a method between two different impls on different
            // types counts as a move.
            let fp = body_fingerprint(&node.block);
            let lc = block_logic_count(&node.block);
            let module_prefix = {
                let mp = self.mod_prefix();
                match &self.impl_ctx {
                    Some(ctx) => format!("{}{}::", mp, ctx.self_ty),
                    None => mp,
                }
            };
            let fn_name = node.sig.ident.to_string();
            self.snap.fn_bodies.entry(fp).or_default().push(FnBody {
                module_prefix,
                name: fn_name,
                logic_count: lc,
            });
        }

        self.fn_depth += 1;
        visit::visit_impl_item_fn(self, node);
        self.fn_depth -= 1;
    }

    // impl blocks — record the Self type (with generics) and trait name.
    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let self_ty = type_key(&node.self_ty);
        let trait_name = node.trait_.as_ref().map(|(_, path, _)| path_key(path));
        let prev = self.impl_ctx.take();
        self.impl_ctx = Some(ImplCtx {
            self_ty,
            trait_name,
        });
        visit::visit_item_impl(self, node);
        self.impl_ctx = prev;
    }

    //  Module-level public statics and consts (NOT function-local ones).
    fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
        if self.fn_depth == 0 && is_pub(&node.vis) {
            self.snap
                .globals
                .insert(format!("{}{}", self.mod_prefix(), node.ident));
        }
        visit::visit_item_static(self, node);
    }
    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        // Only module-level (`fn_depth == 0`) public consts count as globals.
        // Function-local `const`s are NOT globals (they were mis-counted before).
        if self.fn_depth == 0 && is_pub(&node.vis) {
            self.snap
                .globals
                .insert(format!("{}{}", self.mod_prefix(), node.ident));
        }
        visit::visit_item_const(self, node);
    }

    // Associated consts (`impl … { const X … }`) — public ones count as globals.
    fn visit_impl_item_const(&mut self, node: &'ast syn::ImplItemConst) {
        if self.fn_depth == 0 && (self.in_trait_impl() || is_pub(&node.vis)) {
            let key = self.assoc_key(&node.ident.to_string());
            self.snap.globals.insert(key);
        }
        visit::visit_impl_item_const(self, node);
    }

    // Public re-exports (`pub use …`) are public API surface.
    fn visit_item_use(&mut self, node: &'ast syn::ItemUse) {
        if self.fn_depth == 0 && is_pub(&node.vis) {
            let prefix = self.mod_prefix();
            collect_use_leaves(&node.tree, &prefix, &mut self.snap.api_items);
        }
        visit::visit_item_use(self, node);
    }

    // Exported macros (`#[macro_export] macro_rules! …`) are public API.
    // `macro` 2.0 declarative macros are rare/unstable and parse as verbatim,
    // so they are not separately counted (documented approximation).
    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        if self.fn_depth == 0 {
            if let Some(ident) = &node.ident {
                if has_macro_export(&node.attrs) {
                    self.snap.api_items.insert(format!("macro:{}", ident));
                }
            }
        }
        visit::visit_item_macro(self, node);
    }

    // Logic blocks (whole file, all depths).
    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        self.snap.logic_blocks += 1;
        visit::visit_expr_if(self, node);
    }
    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        self.snap.logic_blocks += 1;
        visit::visit_expr_while(self, node);
    }
    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        self.snap.logic_blocks += 1;
        visit::visit_expr_for_loop(self, node);
    }
    fn visit_expr_loop(&mut self, node: &'ast syn::ExprLoop) {
        self.snap.logic_blocks += 1;
        visit::visit_expr_loop(self, node);
    }
    fn visit_arm(&mut self, node: &'ast syn::Arm) {
        self.snap.logic_blocks += 1;
        visit::visit_arm(self, node);
    }
}

// Cyclomatic complexity for a single function body

fn cyclomatic_complexity(block: &syn::Block) -> usize {
    let mut v = CcVisitor { count: 1 };
    v.visit_block(block);
    v.count
}

struct CcVisitor {
    count: usize,
}

impl<'ast> Visit<'ast> for CcVisitor {
    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        self.count += 1;
        visit::visit_expr_if(self, node);
    }
    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        self.count += 1;
        visit::visit_expr_while(self, node);
    }
    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        self.count += 1;
        visit::visit_expr_for_loop(self, node);
    }
    fn visit_expr_loop(&mut self, node: &'ast syn::ExprLoop) {
        self.count += 1;
        visit::visit_expr_loop(self, node);
    }
    fn visit_arm(&mut self, node: &'ast syn::Arm) {
        // Each match arm is a decision point. Then recurse normally so that
        // control flow inside the arm's GUARD and BODY (nested if/for/while/
        // match/&&/||) is counted too. The default `visit_expr_match` already
        // dispatches to `visit_arm` exactly once per arm, so recursing here
        // does not double-count. (Previously this visited an empty
        // `Expr::Verbatim` on a throwaway visitor — dead code that dropped all
        // control flow inside every match arm body.)
        self.count += 1;
        visit::visit_arm(self, node);
    }
    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        if matches!(node.op, syn::BinOp::And(_) | syn::BinOp::Or(_)) {
            self.count += 1;
        }
        visit::visit_expr_binary(self, node);
    }
}

// helpers

/// Whether a visibility is fully public (`pub`). `pub(crate)`/`pub(super)`/
/// `pub(in …)` are `Restricted` and are NOT part of the external API.
fn is_pub(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

/// Whether an attribute list contains `#[macro_export]`.
fn has_macro_export(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("macro_export"))
}

/// Render a type as a stable string key INCLUDING generic arguments, so that
/// `Foo<A>` and `Foo<B>` (and `a::Foo` vs `b::Foo`) get distinct keys. This is
/// a pragmatic renderer covering the type forms that appear as impl Self types
/// and generic arguments; anything exotic falls back to a short placeholder
/// (collisions are then absorbed by parameter-count multiplicity in `diff`).
fn type_key(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(p) => {
            let mut s = String::new();
            if p.qself.is_some() {
                s.push_str("<q>");
            }
            s.push_str(&path_key(&p.path));
            s
        }
        syn::Type::Reference(r) => {
            let mut s = String::from("&");
            if r.mutability.is_some() {
                s.push_str("mut ");
            }
            s.push_str(&type_key(&r.elem));
            s
        }
        syn::Type::Slice(sl) => format!("[{}]", type_key(&sl.elem)),
        syn::Type::Array(a) => format!("[{};_]", type_key(&a.elem)),
        syn::Type::Tuple(t) => {
            let inner: Vec<String> = t.elems.iter().map(type_key).collect();
            format!("({})", inner.join(","))
        }
        syn::Type::Ptr(p) => {
            let m = if p.mutability.is_some() {
                "*mut "
            } else {
                "*const "
            };
            format!("{}{}", m, type_key(&p.elem))
        }
        syn::Type::Group(g) => type_key(&g.elem),
        syn::Type::Paren(p) => type_key(&p.elem),
        syn::Type::TraitObject(t) => {
            let parts: Vec<String> = t
                .bounds
                .iter()
                .filter_map(|b| match b {
                    syn::TypeParamBound::Trait(tb) => Some(path_key(&tb.path)),
                    _ => None,
                })
                .collect();
            format!("dyn {}", parts.join("+"))
        }
        syn::Type::ImplTrait(_) => "impl".to_string(),
        syn::Type::Never(_) => "!".to_string(),
        _ => "_".to_string(),
    }
}

/// Render a path (with per-segment generic arguments) as a stable key.
fn path_key(path: &syn::Path) -> String {
    let mut parts = Vec::new();
    for seg in &path.segments {
        let mut s = seg.ident.to_string();
        match &seg.arguments {
            syn::PathArguments::None => {}
            syn::PathArguments::AngleBracketed(ab) => {
                let args: Vec<String> = ab.args.iter().map(generic_arg_key).collect();
                s.push('<');
                s.push_str(&args.join(","));
                s.push('>');
            }
            syn::PathArguments::Parenthesized(pa) => {
                let inputs: Vec<String> = pa.inputs.iter().map(type_key).collect();
                s.push('(');
                s.push_str(&inputs.join(","));
                s.push(')');
            }
        }
        parts.push(s);
    }
    parts.join("::")
}

fn generic_arg_key(arg: &syn::GenericArgument) -> String {
    match arg {
        syn::GenericArgument::Type(t) => type_key(t),
        syn::GenericArgument::Lifetime(l) => format!("'{}", l.ident),
        syn::GenericArgument::Const(_) => "const".to_string(),
        syn::GenericArgument::AssocType(a) => format!("{}={}", a.ident, type_key(&a.ty)),
        _ => "_".to_string(),
    }
}

/// Walk a `use` tree and insert one `api_items` entry per re-exported leaf.
/// Keyed by the source path + the EXPORTED name (rename target, or `*` for a
/// glob), namespaced with a `use:` tag so it can't collide with fn/global keys.
fn collect_use_leaves(tree: &syn::UseTree, mod_prefix: &str, out: &mut HashSet<String>) {
    fn walk(tree: &syn::UseTree, path: &mut Vec<String>, mod_prefix: &str, out: &mut HashSet<String>) {
        match tree {
            syn::UseTree::Path(p) => {
                path.push(p.ident.to_string());
                walk(&p.tree, path, mod_prefix, out);
                path.pop();
            }
            syn::UseTree::Name(n) => {
                out.insert(format!("{}use:{}::{}", mod_prefix, path.join("::"), n.ident));
            }
            syn::UseTree::Rename(r) => {
                out.insert(format!(
                    "{}use:{}::{} as {}",
                    mod_prefix,
                    path.join("::"),
                    r.ident,
                    r.rename
                ));
            }
            syn::UseTree::Glob(_) => {
                out.insert(format!("{}use:{}::*", mod_prefix, path.join("::")));
            }
            syn::UseTree::Group(g) => {
                for t in &g.items {
                    walk(t, path, mod_prefix, out);
                }
            }
        }
    }
    let mut path = Vec::new();
    walk(tree, &mut path, mod_prefix, out);
}

/// Count comment lines in a source file, block-comment aware (`//` line
/// comments AND `/* … */` block comments). `syn` strips comments, so this is a
/// text-based pass. Made `pub` so the comment-diff path reuses this single
/// definition rather than a cruder `//`-only counter.
pub fn count_comment_lines(src: &str) -> usize {
    let mut count = 0usize;
    let mut in_block = false;

    for line in src.lines() {
        let trimmed = line.trim();
        if in_block {
            count += 1;
            if trimmed.contains("*/") {
                in_block = false;
            }
        } else if trimmed.starts_with("//") {
            count += 1;
        } else if trimmed.starts_with("/*") {
            count += 1;
            if !trimmed.contains("*/") {
                in_block = true;
            }
        }
    }
    count
}

//  Cargo.toml dependency diff

pub struct DepDiff {
    /// Dependencies added (across [dependencies]/[dev-]/[build-]).
    pub added: usize,
    /// Dependencies removed.
    pub deleted: usize,
    /// Dependencies whose pinned version changed.
    pub updated: usize,
    /// Feature flags added to the `[features]` table.
    pub features_added: usize,
    /// Feature flags removed from the `[features]` table.
    pub features_removed: usize,
    /// Whether the crate edition changed.
    pub edition_changed: bool,
    /// Whether ANY dependency section actually changed (added/deleted/updated).
    /// Used to redefine `tcct`: every published `.crate` ships a
    /// cargo-normalized Cargo.toml whose version line always changes, so a raw
    /// file-touched signal is ~always 1 and useless. This is the meaningful
    /// signal instead.
    pub deps_changed: bool,
}

/// One crate manifest's relevant contents.
struct Manifest {
    /// Dependency name -> version requirement (across the three dep sections).
    deps: HashMap<String, String>,
    /// Feature-flag names from the `[features]` table.
    features: HashSet<String>,
    /// `edition` from `[package]`, if declared as a concrete string.
    edition: Option<String>,
}

fn parse_manifest(cargo_toml: &str) -> Manifest {
    let mut m = Manifest {
        deps: HashMap::new(),
        features: HashSet::new(),
        edition: None,
    };
    let Ok(value) = cargo_toml.parse::<toml::Value>() else {
        return m;
    };

    for section in &["dependencies", "dev-dependencies", "build-dependencies"] {
        let Some(toml::Value::Table(tbl)) = value.get(section) else {
            continue;
        };
        for (name, spec) in tbl {
            let version = match spec {
                toml::Value::String(s) => s.clone(),
                toml::Value::Table(t) => t
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("*")
                    .to_string(),
                _ => "*".to_string(),
            };
            m.deps.insert(name.clone(), version);
        }
    }

    // [features] table: each key is a feature-flag name.
    if let Some(toml::Value::Table(feats)) = value.get("features") {
        for name in feats.keys() {
            m.features.insert(name.clone());
        }
    }

    // edition lives under [package]. Published `.crate` manifests are
    // normalized and usually carry a concrete edition string; workspace-
    // inherited (`edition.workspace = true`) or absent editions leave this
    // None and are treated as the language default ("2015") when diffing.
    if let Some(toml::Value::Table(pkg)) = value.get("package") {
        if let Some(toml::Value::String(ed)) = pkg.get("edition") {
            m.edition = Some(ed.clone());
        }
    }

    m
}

/// Compare dependency/feature/edition sections of two `Cargo.toml` strings.
pub fn diff_cargo_toml(before: &str, after: &str) -> DepDiff {
    let b = parse_manifest(before);
    let a = parse_manifest(after);

    let added = a.deps.keys().filter(|k| !b.deps.contains_key(*k)).count();
    let deleted = b.deps.keys().filter(|k| !a.deps.contains_key(*k)).count();
    let updated = a
        .deps
        .iter()
        .filter(|(k, v)| b.deps.get(*k).map(|bv| bv != *v).unwrap_or(false))
        .count();

    let features_added = a.features.difference(&b.features).count();
    let features_removed = b.features.difference(&a.features).count();

    // Absent edition -> language default "2015", so an old (implicit-2015)
    // crate moving to an explicit newer edition is detected, while a normalized
    // manifest merely ADDING an explicit `edition = "2015"` is not a change.
    let b_ed = b.edition.as_deref().unwrap_or("2015");
    let a_ed = a.edition.as_deref().unwrap_or("2015");
    let edition_changed = b_ed != a_ed;

    let deps_changed = added > 0 || deleted > 0 || updated > 0;

    DepDiff {
        added,
        deleted,
        updated,
        features_added,
        features_removed,
        edition_changed,
        deps_changed,
    }
}
