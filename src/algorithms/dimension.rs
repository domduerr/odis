//! Algorithms for computing the order dimension of a poset.
//!
//! The order dimension of a partially ordered set (poset) is the minimum number
//! of linear extensions whose intersection equals the original partial order.
//!
//! This module provides implementations of [`crate::traits::DimensionAlgorithm`]:
//!
//! - [`SatReduction`] — SAT reduction (Heitz 2022)
//! - [`HypergraphColoring`] — Chromatic number of the alternating cycle hypergraph (Trotter 1992)
//! - [`GraphColoring`] — Incompatibility graph coloring with iterative extension (Yáñez & Montero 1999)
//! - [`GraphColoringHeuristic`] — Same as GraphColoring but uses DSatur heuristic instead of exact backtracking
//! - [`Hybrid`] — Lazy amplification combining DSatur-guided graph coloring with hypergraph constraints

use crate::data_structures::poset::Poset;
use crate::traits::DimensionAlgorithm;
use std::collections::HashSet;
use std::cmp;

// ─── SAT Reduction ────────────────────────────────────────────────────────────

/// SAT-based dimension computation.
///
/// Reduces the order dimension problem to SAT: for each k = 1, 2, ...,
/// encodes whether k linear extensions can form a realizer. Variables
/// represent orientations of incomparable pairs in each extension;
/// transitivity and completeness constraints ensure validity.
pub struct SatReduction;

impl<T: Clone> DimensionAlgorithm<T> for SatReduction {
    fn dimension(&self, poset: &Poset<T>) -> usize {
        sat_dimension(poset)
    }

    fn realizer(&self, poset: &Poset<T>) -> Vec<Vec<usize>> {
        sat_realizer(poset)
    }
}

// ─── Hypergraph Coloring (Trotter) ──────────────────────────────────────────

/// Hypergraph coloring approach (Felsner & Trotter 2000).
///
/// Computes the dimension via the chromatic number of the strict alternating
/// cycle hypergraph K_s(P). Vertices are critical pairs; hyperedges are
/// strict alternating cycles. Per Felsner & Trotter (2000), dim(P) = χ(K_s(P)).
pub struct HypergraphColoring;

impl<T: Clone> DimensionAlgorithm<T> for HypergraphColoring {
    fn dimension(&self, poset: &Poset<T>) -> usize {
        hypergraph_coloring_dimension(poset)
    }

    fn realizer(&self, poset: &Poset<T>) -> Vec<Vec<usize>> {
        hypergraph_coloring_realizer(poset)
    }
}

// ─── Graph Coloring (Yáñez & Montero) ───────────────────────────────────────

/// Graph coloring approach (Yáñez & Montero 1999).
///
/// Builds the incompatibility graph G*(P) on critical pairs (Definition 3.2),
/// computes its chromatic number, constructs a minimum realizer, and checks
/// for circuits. If circuits are found in any linear extension, edges are
/// added to G*(P) (extension procedure, Section 5) and the process iterates.
///
/// Uses Brown's backtracking algorithm for exact coloring.
pub struct GraphColoring;

/// Heuristic graph coloring approach (Yáñez & Montero 1999, Section 6).
///
/// Same algorithm as [`GraphColoring`] but uses DSatur heuristic instead of
/// Brown's exact backtracking. Computes an upper bound on the dimension
/// with drastically reduced computation time.
pub struct GraphColoringHeuristic;

impl<T: Clone> DimensionAlgorithm<T> for GraphColoring {
    fn dimension(&self, poset: &Poset<T>) -> usize {
        graph_coloring_dimension(poset)
    }

    fn realizer(&self, poset: &Poset<T>) -> Vec<Vec<usize>> {
        graph_coloring_realizer(poset)
    }
}

impl<T: Clone> DimensionAlgorithm<T> for GraphColoringHeuristic {
    fn dimension(&self, poset: &Poset<T>) -> usize {
        graph_coloring_heuristic_dimension(poset)
    }

    fn realizer(&self, poset: &Poset<T>) -> Vec<Vec<usize>> {
        graph_coloring_heuristic_realizer(poset)
    }
}

// ─── Hybrid (DSatur-Guided Selective Pruning) ─────────────────────────────

/// Hybrid approach combining DSatur-guided graph coloring with selective
/// hypergraph constraint enforcement.
///
/// Algorithm:
/// 1. Build the incompatibility graph G*(P) and alternating cycle adjacency
/// 2. Run DSatur for upper bound, vertex ordering, and color hints
/// 3. For k = 2, 3, ..., k_ub: DSatur-guided backtracking with:
///    a. Graph adjacency pruning (fast O(degree) — from GC)
///    b. Selective strict alternating cycle detection (precise — from HG):
///       only triggered when a vertex has alternating cycle adjacency to
///       another same-colored vertex; skipped otherwise
///    c. DSatur color hints: try the most promising color first
///
/// Key innovations:
/// - Selective pruning: cycle detection is expensive, so we only invoke it
///   when the cheap O(degree) check confirms same-colored hg neighbors exist
/// - DSatur ordering: processes most constrained vertices first, pruning
///   the search tree as early as possible
/// - Pre-allocated buffers: zero allocations in the hot backtracking loop
pub struct Hybrid;

impl<T: Clone> DimensionAlgorithm<T> for Hybrid {
    fn dimension(&self, poset: &Poset<T>) -> usize {
        hybrid_dimension(poset)
    }

    fn realizer(&self, poset: &Poset<T>) -> Vec<Vec<usize>> {
        hybrid_realizer(poset)
    }
}

// ─── Default dimension on Poset ──────────────────────────────────────────────

impl<T: Clone> Poset<T> {
    /// Computes the order dimension of this poset using the default algorithm
    /// (SAT reduction).
    pub fn dimension(&self) -> usize {
        sat_dimension(self)
    }

    /// Returns a realizer using the default algorithm (SAT reduction).
    pub fn realizer(&self) -> Vec<Vec<usize>> {
        sat_realizer(self)
    }
}

// ─── SAT Reduction Implementation ───────────────────────────────────────────

/// Computes the order dimension using SAT reduction.
fn sat_dimension<T: Clone>(poset: &Poset<T>) -> usize {
    let n = poset.nodes.len();
    let n_edges = poset.transitive_edges.len();
    
    // Handle trivial math edge cases correctly
    if n == 0 { return 0; }
    if n == 1 { return 1; }

    // Check for chain (Dimension 1) - O(1) mathematical check
    if 2 * n_edges == n * (n - 1) {
        return 1;
    }

    // Check for antichain (Dimension 2)
    if n_edges == 0 {
        return 2;
    }

    // Since it's not a chain, dimension is at least 2.
    // The maximum possible dimension is n.
    let setup = sat_setup(poset);
    for k in 2..=n {
        if solve_for_k(&setup.base_clauses, setup.n_inc, k).is_some() {
            return k;
        }
    }

    unreachable!("Dimension search must terminate within theoretical bounds (k <= n)")
}

/// Computes a realizer (minimal set of linear extensions) using SAT reduction.
/// Shared SAT infrastructure: computes incomparable pairs, variable mapping,
/// base transitivity clauses. Used by both `sat_dimension`/`sat_realizer`
/// (iterating k) and `sat_realizer_for_k` (fixed k from coloring).
struct SatSetup {
    incomparable: Vec<(u32, u32)>,
    to_var: std::collections::HashMap<(u32, u32), usize>,
    n_inc: usize,
    base_clauses: Vec<Vec<i32>>,
}

fn sat_setup<T: Clone>(poset: &Poset<T>) -> SatSetup {
    let incomparable = poset.incomparable_pairs();
    let to_var: std::collections::HashMap<(u32, u32), usize> = incomparable
        .iter()
        .enumerate()
        .map(|(i, &pair)| (pair, i + 1))
        .collect();
    let n_inc = incomparable.len();
    let base_clauses = build_transitivity_clauses(poset, &incomparable, &to_var);
    SatSetup { incomparable, to_var, n_inc, base_clauses }
}

fn sat_realizer<T: Clone>(poset: &Poset<T>) -> Vec<Vec<usize>> {
    let n = poset.nodes.len();
    if n <= 1 || poset.incomparable_pairs().is_empty() {
        let le = poset.linear_extension().unwrap_or_else(|| (0..n).collect());
        return vec![le];
    }

    let setup = sat_setup(poset);
    for k in 1.. {
        if let Some(model) = solve_for_k(&setup.base_clauses, setup.n_inc, k) {
            return decode_sat_realizer(poset, k, &model, &setup.incomparable, setup.n_inc, &setup.to_var);
        }
    }

    unreachable!("dimension search must terminate")
}

/// Decodes a SAT model into k linear extensions (a realizer).
fn decode_sat_realizer<T: Clone>(
    poset: &Poset<T>,
    k: usize,
    model: &[i32],
    incomparable: &[(u32, u32)],
    n_inc: usize,
    to_var: &std::collections::HashMap<(u32, u32), usize>,
) -> Vec<Vec<usize>> {
    use std::collections::HashSet as StdSet;
    let n = poset.nodes.len();
    let model_set: StdSet<i32> = model.iter().copied().collect();
    let mut realizer = Vec::with_capacity(k);

    for ext_idx in 0..k {
        let mut edges: StdSet<(usize, usize)> = StdSet::new();
        // Add all poset order relations (transitive closure)
        for &(u, v) in &poset.transitive_edges {
            edges.insert((u as usize, v as usize));
        }
        // Orient incomparable pairs according to the SAT model
        for &(a, b) in incomparable {
            let var_num = to_var[&(a, b)] + ext_idx * n_inc;
            if model_set.contains(&(var_num as i32)) {
                edges.insert((a as usize, b as usize));
            } else {
                edges.insert((b as usize, a as usize));
            }
        }
        let le = Poset::<T>::linear_extension_static(n, &edges)
            .unwrap_or_else(|| (0..n).collect());
        realizer.push(le);
    }

    realizer
}

/// Gets the SAT variable for orientation (a, b) in extension i.
fn get_var(
    a: u32,
    b: u32,
    to_var: &std::collections::HashMap<(u32, u32), usize>,
    n_inc: usize,
    extension_index: usize,
) -> Option<i32> {
    if let Some(&v) = to_var.get(&(a, b)) {
        Some((v + extension_index * n_inc) as i32)
    } else if let Some(&v) = to_var.get(&(b, a)) {
        Some(-((v + extension_index * n_inc) as i32))
    } else {
        None
    }
}

/// Builds transitivity clauses for the SAT encoding.
fn build_transitivity_clauses<T: Clone>(
    poset: &Poset<T>,
    incomparable: &[(u32, u32)],
    to_var: &std::collections::HashMap<(u32, u32), usize>,
) -> Vec<Vec<i32>> {
    let n = poset.nodes.len();
    let mut clauses = Vec::new();
    let elements: Vec<u32> = (0..n as u32).collect();

    for &(a, c) in incomparable {
        let ac_var = to_var[&(a, c)] as i32;

        for &b in &elements {
            let ab_var = get_var(a, b, to_var, incomparable.len(), 0);
            let bc_var = get_var(b, c, to_var, incomparable.len(), 0);

            match (ab_var, bc_var) {
                (Some(ab), Some(bc)) => {
                    clauses.push(vec![-ab, -bc, ac_var]);
                }
                (None, Some(bc)) => {
                    if poset.is_leq(b, a) {
                        continue;
                    }
                    clauses.push(vec![-bc, ac_var]);
                }
                (Some(ab), None) => {
                    if poset.is_leq(c, b) {
                        continue;
                    }
                    clauses.push(vec![-ab, ac_var]);
                }
                (None, None) => {
                    if poset.is_leq(b, a) || poset.is_leq(c, b) {
                        continue;
                    }
                }
            }
        }
    }

    // Anti-transitivity clauses for comparable pairs:
    // If c < a in the poset, then a→b→c (both incomparable) would imply a < c,
    // contradicting c < a. Add clause: NOT (a→b AND b→c) = [-ab_var, -bc_var].
    for &(u, v) in &poset.transitive_edges {
        // u < v in poset. Prevent paths from v back to u through incomparable pairs.
        for &b in &elements {
            let vb_var = get_var(v, b, to_var, incomparable.len(), 0);
            let bu_var = get_var(b, u, to_var, incomparable.len(), 0);
            if let (Some(vb), Some(bu)) = (vb_var, bu_var) {
                clauses.push(vec![-vb, -bu]);
            }
        }
    }

    clauses
}

/// Solves the SAT instance for a given k (number of linear extensions).
/// Optimized to eliminate dynamic reallocation overhead.
fn solve_for_k(base_clauses: &[Vec<i32>], n_inc: usize, k: usize) -> Option<Vec<i32>> {
    // 1. Calculate exact capacities to prevent any runtime reallocations
    let num_base = base_clauses.len();
    let total_clauses = (num_base * k) + (2 * n_inc);
    
    // Allocate the outer vector exactly once
    let mut clauses: Vec<Vec<i32>> = Vec::with_capacity(total_clauses);

    // 2. Process base clauses
    for i in 0..k {
        let offset = (i * n_inc) as i32;
        
        for clause in base_clauses {
            // Allocate the inner vector exactly once per clause
            let mut shifted_clause = Vec::with_capacity(clause.len());
            
            for &v in clause {
                let sign = if v > 0 { v + offset } else { v - offset };
                shifted_clause.push(sign);
            }
            clauses.push(shifted_clause);
        }
    }

    // 3. Process the linking clauses (incomparable pairs)
    for var in 1..=n_inc {
        let mut pos_clause = Vec::with_capacity(k);
        let mut neg_clause = Vec::with_capacity(k);
        
        for i in 0..k {
            let v = (var + i * n_inc) as i32;
            pos_clause.push(v);
            neg_clause.push(-v);
        }
        
        clauses.push(pos_clause);
        clauses.push(neg_clause);
    }

    // 4. Invoke the solver
    match splr::Certificate::try_from(clauses) {
        Ok(splr::Certificate::SAT(model)) => Some(model),
        Ok(splr::Certificate::UNSAT) => None,
        _ => None,
    }
}

// ─── Hypergraph Coloring Implementation (Felsner & Trotter) ────────────────────

/// Computes the dimension using hypergraph coloring (Felsner & Trotter 2000).
///
/// Pre-computes all minimal strict alternating cycles (hyperedges of K_s(P))
/// among critical pairs, then finds the chromatic number of the hypergraph
/// via backtracking.
///
/// Per Felsner & Trotter (2000), dim(P) = χ(K_s(P)).
fn hypergraph_coloring_dimension<T: Clone>(poset: &Poset<T>) -> usize {
    let n = poset.nodes.len();
    let n_edges = poset.transitive_edges.len();

    if n <= 1 {
        return 1;
    }

    // Total order (chain): all (n choose 2) comparable pairs
    if 2 * n_edges == n * (n - 1) {
        return 1;
    }

    // Antichain: no comparable pairs
    if n_edges == 0 {
        return 2;
    }

    let critical_pairs = poset.critical_pairs();

    let hyperedges = find_minimal_hyperedges(&critical_pairs, poset);

    // Hiraguchi's Theorem: dim(P) <= n / 2 for n >= 4. 
    // We use max(2, n / 2) to safely handle small posets (n=2, n=3).
    let max_possible_dim = std::cmp::max(2, n / 2);

    for k in 2..=max_possible_dim {
        if hypergraph_coloring_solve(critical_pairs.len(), &hyperedges, k).is_some() {
            return k;
        }
    }

    unreachable!("Dimension exceeded Hiraguchi's theoretical upper bound ({}), indicating a bug in hyperedge generation or solver.", max_possible_dim);
}

/// Builds the directed adjacency for alternating cycles among pairs.
///
/// Edge from pair i=(a_i, b_i) to pair j=(a_j, b_j) if a_i ≤ b_j,
/// following Trotter (1992) Definition 2.2: an alternating cycle is a
/// sequence (x_1,y_1),..., (x_s,y_s) with a_i ≤ b_{(i mod s)+1}.
fn build_alternating_cycle_adj<T: Clone>(
    pairs: &[(u32, u32)],
    poset: &Poset<T>,
) -> Vec<Vec<usize>> {
    let n = pairs.len();
    (0..n)
        .map(|i| {
            let (a_i, _) = pairs[i];
            (0..n)
                .filter(|&j| j != i && poset.is_leq(a_i, pairs[j].1))
                .collect()
        })
        .collect()
}

/// Finds all minimal strict alternating cycles (hyperedges of K_s(P)).
///
/// A strict alternating cycle is a sequence of critical pairs
/// (x₁,y₁),...,(xₛ,yₛ) with s ≥ 2 such that x_i ≤ y_j holds
/// if and only if j is the successor of i in the cycle.
/// All x values and all y values in the cycle must be distinct.
///
/// After enumeration, filters to minimal hyperedges: no hyperedge
/// is a proper subset of another.
fn find_minimal_hyperedges<T: Clone>(
    critical_pairs: &[(u32, u32)],
    poset: &Poset<T>,
) -> Vec<HashSet<usize>> {
    let m = critical_pairs.len();
    if m < 2 {
        return Vec::new();
    }

    let adj = build_alternating_cycle_adj(critical_pairs, poset);

    let mut hyperedges: Vec<HashSet<usize>> = Vec::new();

    for start in 0..m {
        let x_init = critical_pairs[start].0;
        let y_init = critical_pairs[start].1;
        let mut path = vec![start];
        let mut visited_xs = HashSet::from([x_init]);
        let mut visited_ys = HashSet::from([y_init]);
        let mut in_path = vec![false; m];
        in_path[start] = true;

        dfs_find_cycles(
            start, start, &mut path, &mut visited_xs, &mut visited_ys,
            &mut in_path, &adj, critical_pairs, poset, &mut hyperedges,
        );
    }

    // Filter to minimal hyperedges (no proper subset exists)
    let mut minimal: Vec<HashSet<usize>> = Vec::new();
    for h in &hyperedges {
        let is_minimal = !hyperedges.iter().any(|other| other != h && other.is_subset(h));
        if is_minimal && !minimal.iter().any(|existing| existing == h) {
            minimal.push(h.clone());
        }
    }

    minimal
}

/// DFS to find all strict alternating cycles starting from start.
fn dfs_find_cycles<T: Clone>(
    start: usize,
    current: usize,
    path: &mut Vec<usize>,
    visited_xs: &mut HashSet<u32>,
    visited_ys: &mut HashSet<u32>,
    in_path: &mut [bool],
    adj: &[Vec<usize>],
    critical_pairs: &[(u32, u32)],
    poset: &Poset<T>,
    hyperedges: &mut Vec<HashSet<usize>>,
) {
    for &neighbor in &adj[current] {
        if neighbor == start && path.len() >= 2 {
            if is_strict_cycle(path, critical_pairs, poset) {
                hyperedges.push(path.iter().copied().collect());
            }
            continue;
        }

        if in_path[neighbor] {
            continue;
        }

        let next_x = critical_pairs[neighbor].0;
        let next_y = critical_pairs[neighbor].1;

        // Trotter condition: x and y values must be globally distinct in the cycle
        if !visited_xs.contains(&next_x) && !visited_ys.contains(&next_y) {
            visited_xs.insert(next_x);
            visited_ys.insert(next_y);
            path.push(neighbor);
            in_path[neighbor] = true;

            dfs_find_cycles(
                start, neighbor, path, visited_xs, visited_ys,
                in_path, adj, critical_pairs, poset, hyperedges,
            );

            path.pop();
            in_path[neighbor] = false;
            visited_xs.remove(&next_x);
            visited_ys.remove(&next_y);
        }
    }
}

/// Checks whether a cycle is a strict alternating cycle.
///
/// For all pairs (i, j) in the cycle, verifies that x_i ≤ y_j holds
/// if and only if j is the successor of i.
fn is_strict_cycle<T: Clone>(
    cycle: &[usize],
    critical_pairs: &[(u32, u32)],
    poset: &Poset<T>,
) -> bool {
    let s = cycle.len();
    if s < 2 {
        return false;
    }

    for i in 0..s {
        let x_k = critical_pairs[cycle[i]].0;
        let target_j = (i + 1) % s;
        for j in 0..s {
            let y_l = critical_pairs[cycle[j]].1;
            if poset.is_leq(x_k, y_l) != (j == target_j) {
                return false;
            }
        }
    }
    true
}

/// Backtracking hypergraph coloring solver.
///
/// Assigns colors to critical pairs such that no hyperedge is monochromatic.
/// Uses early pruning: after each assignment, checks if any fully-assigned
/// hyperedge has become monochromatic.
fn hypergraph_coloring_solve(
    num_pairs: usize,
    hyperedges: &[HashSet<usize>],
    k: usize,
) -> Option<Vec<usize>> {
    if num_pairs == 0 {
        return Some(Vec::new());
    }
    if k == 0 {
        return None;
    }

    let mut assignment: Vec<i32> = vec![-1; num_pairs];

    if coloring_backtrack(0, &mut assignment, num_pairs, hyperedges, k) {
        Some(assignment.iter().map(|&c| c as usize).collect())
    } else {
        None
    }
}

/// Backtracking helper for hypergraph coloring.
fn coloring_backtrack(
    current_pair: usize,
    assignment: &mut [i32],
    num_pairs: usize,
    hyperedges: &[HashSet<usize>],
    k: usize,
) -> bool {
    if current_pair == num_pairs {
        for edge in hyperedges {
            let first_color = assignment[*edge.iter().next().unwrap()];
            if edge.iter().all(|&idx| assignment[idx] == first_color) {
                return false;
            }
        }
        return true;
    }

    for color in 0..k {
        assignment[current_pair] = color as i32;

        // Pruning: check if any fully-assigned hyperedge is monochromatic
        let mut legal = true;
        for edge in hyperedges {
            if edge.iter().all(|&idx| assignment[idx] != -1) {
                let first_color = assignment[*edge.iter().next().unwrap()];
                if edge.iter().all(|&idx| assignment[idx] == first_color) {
                    legal = false;
                    break;
                }
            }
        }

        if legal && coloring_backtrack(current_pair + 1, assignment, num_pairs, hyperedges, k) {
            return true;
        }

        assignment[current_pair] = -1;
    }

    false
}


/// Checks whether a cycle (given as a path of vertex indices) is a strict
/// alternating cycle per Trotter (1992), Definition 2.2, with the
/// additional condition that x values and y values must be distinct.
///
/// An alternating cycle (x₁,y₁), ..., (xₛ,yₛ) requires:
/// 1. x₁,...,xₛ are distinct (no two pairs share a first element)
/// 2. y₁,...,yₛ are distinct (no two pairs share a second element)
/// 3. xᵢ ≤ y_{(i mod s)+1} for each i (successor edges in adjacency)
/// 4. xᵢ ≰ yⱼ for all j ≠ (i+1) mod s (strictness condition)
///
/// These strict alternating cycles are the hyperedges of K_s(P). Per Trotter (1992),
/// dim(P) = chi(K_s(P)), so coloring K_s(P) yields the exact dimension.
fn is_strict_alternating_cycle<T: Clone>(
    cycle: &[usize],
    pairs: &[(u32, u32)],
    poset: &Poset<T>,
) -> bool {
    let s = cycle.len();
    if s < 2 {
        return false;
    }

    // Condition 1: all first elements (x values) must be distinct
    let mut seen_x = HashSet::new();
    for &idx in cycle {
        let (x, _) = pairs[idx];
        if !seen_x.insert(x) {
            return false;
        }
    }

    // Condition 2: all second elements (y values) must be distinct
    let mut seen_y = HashSet::new();
    for &idx in cycle {
        let (_, y) = pairs[idx];
        if !seen_y.insert(y) {
            return false;
        }
    }

    // Condition 4: strictness — for each vertex v_i, verify that
    // a_{v_i} is NOT comparable to b_{v_j} for any j that is NOT the
    // successor of i in the cycle.
    for i in 0..s {
        let (a_i, _) = pairs[cycle[i]];
        let succ_i = (i + 1) % s;
        for j in 0..s {
            if j == succ_i {
                continue;
            }
            let (_, b_j) = pairs[cycle[j]];
            if poset.is_leq(a_i, b_j) {
                return false;
            }
        }
    }
    true
}

// ─── Graph Coloring Implementation (Yáñez & Montero) ────────────────────────

/// Computes the dimension using the Yáñez & Montero algorithm.
///
/// Algorithm (Yáñez & Montero, Section 6):
/// 1. Build consistency digraph G(P) (Definition 2.2)
/// 2. Build incompatibility graph G*(P) (Definition 3.2)
/// 3. Color G*(P) using exact graph coloring
/// 4. Construct minimum realizer from coloring
/// 5. Check for circuits; if found, add edges to G*(P) and go to 3
///
/// When K_s(P) = graph K_s(P) (no hyperedges beyond edges), this gives
/// the exact dimension. Otherwise, the extension procedure adds edges
/// and may give an upper bound.
fn graph_coloring_dimension<T: Clone>(poset: &Poset<T>) -> usize {
    let n = poset.nodes.len();
    let n_edges = poset.transitive_edges.len();
    
    // Check for single node
    if n <= 1 {
        return 1;
    }

    // Check for chain
    if 2*n_edges == n * (n - 1) {
        return 1
    }

    // Check for antichain
    if n_edges == 0 {
        return 2;
    }

    let critical_pairs = poset.critical_pairs();

    // Build initial incompatibility graph (Definition 3.2)
    let mut incompat_edges = build_incompatibility_graph(&critical_pairs, poset);

    // Extension procedure (Section 5): iteratively add edges from
    // hyperedges found as circuits in the realizer
    loop {
        // Color the incompatibility graph using exact backtracking
        let (chromatic_num, coloring) = chromatic_number_exact(
            critical_pairs.len(),
            &incompat_edges,
        );

        // Build the partial orders L_i from the coloring and check for circuits
        let circuits = find_circuits_in_realizer(
            &critical_pairs,
            &coloring,
            chromatic_num,
            poset,
        );

        if circuits.is_empty() {
            return chromatic_num;
        }

        // Add edges for found circuits (Proposition 5.1)
        for (i, j) in circuits {
            let (min_ij, max_ij) = (i.min(j), i.max(j));
            if !incompat_edges.contains(&(min_ij, max_ij)) {
                incompat_edges.push((min_ij, max_ij));
            }
        }
    }
}

/// Builds the incompatibility graph G*(P) (Definition 3.2).
///
/// Two critical pairs a = (x1,y1) and b = (x2,y2) in V*(P) are incompatible if:
/// 1. They are opposite pairs: {x1,y1} = {x2,y2} (always incompatible)
/// 2. There exists a successor g of a in G(P) such that g^(-1) is a successor of b
///    (Definition 3.1)
///
/// Condition 1 ensures opposite pairs like (a,b) and (b,a) are always connected,
/// which is required for correctness even when G(P) has no arcs (e.g., antichains).
fn build_incompatibility_graph<T: Clone>(
    critical_pairs: &[(u32, u32)],
    poset: &Poset<T>,
) -> Vec<(usize, usize)> {
    let n = critical_pairs.len();

    // Build successors in the consistency digraph G(P)
    // Successors of (x,y) = {(z,w) ∈ inc(P) : z ≤ x and y ≤ w}
    let succs: Vec<HashSet<(u32, u32)>> = critical_pairs
        .iter()
        .map(|&(x, y)| {
            let mut succ_set = HashSet::new();
            for &(z, w) in critical_pairs {
                if (z, w) != (x, y) && poset.is_leq(z, x) && poset.is_leq(y, w) {
                    succ_set.insert((z, w));
                }
            }
            succ_set
        })
        .collect();

    let mut edges = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let (x1, y1) = critical_pairs[i];
            let (x2, y2) = critical_pairs[j];

            // Condition 1: opposite pairs are always incompatible
            // (a,b) and (b,a) can never be in the same linear extension
            if x1 == y2 && y1 == x2 {
                edges.push((i, j));
                continue;
            }

            // Condition 2: Definition 3.1 — successor-based incompatibility
            let incompatible = succs[i].iter().any(|&(z, w)| succs[j].contains(&(w, z)));
            if incompatible {
                edges.push((i, j));
            }
        }
    }

    edges
}

/// Helper: Linear-time O(V+E) cycle detection using White/Gray/Black node states.
fn find_cycle_dfs(
    u: usize,
    adj: &[Vec<usize>],
    state: &mut [u8],
    parent: &mut [usize],
) -> Option<Vec<usize>> {
    state[u] = 1;

    for &v in &adj[u] {
        if state[v] == 0 {
            parent[v] = u;
            if let Some(cycle) = find_cycle_dfs(v, adj, state, parent) {
                return Some(cycle);
            }
        } else if state[v] == 1 {
            let mut cycle = vec![v];
            let mut curr = u;
            while curr != v && curr != usize::MAX {
                cycle.push(curr);
                curr = parent[curr];
            }
            cycle.reverse();
            return Some(cycle);
        }
    }

    state[u] = 2;
    None
}

/// Checks for circuits in the realizer constructed from a coloring.
fn find_circuits_in_realizer<T: Clone>(
    critical_pairs: &[(u32, u32)],
    coloring: &[usize],
    k: usize,
    poset: &Poset<T>,
) -> Vec<(usize, usize)> {
    let mut circuits = Vec::new();

    // For each color class, collect the critical pairs assigned to it
    for color in 0..k {
        let pairs_in_class: Vec<usize> = coloring
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c == color)
            .map(|(i, _)| i)
            .collect();

        let m = pairs_in_class.len();
        if m < 2 {
            continue;
        }

        let class_pairs: Vec<(u32, u32)> = pairs_in_class
            .iter()
            .map(|&i| critical_pairs[i])
            .collect();

        // Build adjacency for alternating cycle detection within this class
        let adj: Vec<Vec<usize>> = (0..m)
            .map(|i| {
                let (a_i, _) = class_pairs[i];
                (0..m)
                    .filter(|&j| j != i && poset.is_leq(a_i, class_pairs[j].1))
                    .collect()
            })
            .collect();

        let mut state = vec![0u8; m];
        let mut parent = vec![usize::MAX; m];
        let mut cycle_found = None;

        for start in 0..m {
            if state[start] == 0 {
                if let Some(cycle) = find_cycle_dfs(start, &adj, &mut state, &mut parent) {
                    cycle_found = Some(cycle);
                    break;
                }
            }
        }

        if let Some(cycle) = cycle_found {
            for w in 0..cycle.len() {
                let next_w = (w + 1) % cycle.len();
                let pi = pairs_in_class[cycle[w]];
                let pj = pairs_in_class[cycle[next_w]];
                circuits.push((pi.min(pj), pi.max(pj)));
            }
        }
    }

    circuits
}

/// Exact graph coloring using backtracking (Brown's algorithm).
///
/// Returns the chromatic number and an optimal coloring as a vector of
/// color assignments (0-based color indices).
fn chromatic_number_exact(n: usize, edges: &[(usize, usize)]) -> (usize, Vec<usize>) {
    if n == 0 {
        return (0, Vec::new());
    }

    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u < n && v < n {
            adj[u].push(v);
            adj[v].push(u);
        }
    }

    // Try increasing number of colors
    for k in 1..=n {
        let mut colors = vec![0usize; n];
        if backtrack_color(0, k, &mut colors, &adj) {
            return (k, colors);
        }
    }

    // Worst case: each vertex gets its own color
    let colors: Vec<usize> = (0..n).collect();
    (n, colors)
}

/// Backtracking graph coloring.
fn backtrack_color(
    node: usize,
    k: usize,
    colors: &mut [usize],
    adj: &[Vec<usize>],
) -> bool {
    if node == colors.len() {
        return true;
    }

    for c in 0..k {
        if adj[node].iter().all(|&neighbor| {
            neighbor >= node || colors[neighbor] != c
        }) {
            colors[node] = c;
            if backtrack_color(node + 1, k, colors, adj) {
                return true;
            }
        }
    }

    false
}

/// Heuristic graph coloring using DSatur (Brélaz 1979).
///
/// DSatur colors vertices greedily, always selecting the uncolored vertex
/// with the highest saturation degree (number of distinct colors used by
/// its neighbors). Ties are broken by choosing the vertex with the highest
/// degree. This gives an upper bound on the chromatic number.
///
/// Yáñez & Montero (1999) explicitly recommend replacing Brown's exact
/// backtracking with DSatur for large instances: "the exact minimal-coloration
/// procedure can be changed by an approximate procedure which allows the
/// computation of an upper bound for the chromatic number with a drastic
/// reduction of computation time."


/// Heuristic graph coloring using DSatur (Brélaz 1979).
fn chromatic_number_heuristic(n: usize, edges: &[(usize, usize)]) -> (usize, Vec<usize>) {
    if n == 0 {
        return (0, Vec::new());
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u < n && v < n {
            adj[u].push(v);
            adj[v].push(u);
        }
    }

    let mut degrees = vec![0usize; n];
    for i in 0..n {
        degrees[i] = adj[i].len();
    }

    let mut colors = vec![0usize; n];
    let mut colored = vec![false; n];

    let mut sat_counts = vec![0usize; n];
    
    let mut neighbor_colors = vec![false; n * n];

    for _ in 0..n {
        let next = (0..n)
            .filter(|&v| !colored[v])
            .max_by(|&a, &b| {
                sat_counts[a].cmp(&sat_counts[b])
                    .then_with(|| degrees[a].cmp(&degrees[b]))
            })
            .unwrap();

        let mut c = 0;
        let row_offset = next * n;
        while neighbor_colors[row_offset + c] {
            c += 1;
        }

        colors[next] = c;
        colored[next] = true;

        for &neighbor in &adj[next] {
            if !colored[neighbor] {
                let neighbor_idx = neighbor * n + c;
                // Wenn der Nachbar diese Farbe noch nicht gesehen hat:
                if !neighbor_colors[neighbor_idx] {
                    neighbor_colors[neighbor_idx] = true;
                    sat_counts[neighbor] += 1;
                }
            }
        }
    }

    let chromatic_num = colors.iter().copied().max().unwrap_or(0) + 1;
    (chromatic_num, colors)
}

/// Computes the dimension using the heuristic Yáñez & Montero algorithm.
///
/// Same as [`graph_coloring_dimension`] but uses DSatur heuristic coloring
/// instead of Brown's exact backtracking. Returns an upper bound on the
/// dimension, not necessarily the exact value.
///
/// The incompatibility graph is amplified with 2-alternating cycle edges
/// before DSatur runs, giving the heuristic strictly more constraints and
/// improving accuracy on sparse instances.
pub fn graph_coloring_heuristic_dimension<T: Clone>(poset: &Poset<T>) -> usize {
    let n = poset.nodes.len();
    let n_edges = poset.transitive_edges.len();

    // Check for single node
    if n <= 1 {
        return 1;
    }

    // Check for chain
    if 2 * n_edges == n * (n - 1) {
        return 1;
    }

    // Check for antichain
    if n_edges == 0 {
        return 2;
    }

    let critical_pairs = poset.critical_pairs();

    // Build alternating cycle adjacency for 2-cycle amplification
    let hg_adj = build_alternating_cycle_adj(&critical_pairs, poset);

    // Build initial incompatibility graph (Definition 3.2)
    let initial_edges = build_incompatibility_graph(&critical_pairs, poset);

    // Amplify with 2-alternating cycle edges: mutual hg_adj adjacency means
    // two critical pairs form a 2-alternating cycle and MUST have different colors.
    let mut incompat_edges_set: HashSet<(usize, usize)> = initial_edges.into_iter().collect();

    let m = critical_pairs.len();
    for i in 0..m {
        for &j in &hg_adj[i] {
            if j > i && hg_adj[j].binary_search(&i).is_ok() {
                incompat_edges_set.insert((i, j));
            }
        }
    }

    // FIX 2 & 3: Limits setzen
    const MAX_ITERATIONS: usize = 50;
    let mut iter_count = 0;
    
    // Hirschel's Theorem / Dushnik-Miller: Dim(P) <= n/2 für n >= 4
    // Safe Fallback value, if heuristics fail
    let safe_upper_bound = cmp::max(2, n / 2);

    loop {
        iter_count += 1;

        let edges_vec: Vec<(usize, usize)> = incompat_edges_set.iter().copied().collect();

        // Color the incompatibility graph using DSatur heuristic
        let (chromatic_num, coloring) = chromatic_number_heuristic(
            critical_pairs.len(),
            &edges_vec,
        );

        // Build the partial orders L_i from the coloring and check for circuits
        let circuits = find_circuits_in_realizer(
            &critical_pairs,
            &coloring,
            chromatic_num,
            poset,
        );

        if circuits.is_empty() {
            return chromatic_num;
        }

        if iter_count >= MAX_ITERATIONS {
            return cmp::max(chromatic_num, safe_upper_bound);
        }

        // Add edges for found circuits (Proposition 5.1)
        let mut added_new_edges = false;
        
        for (i, j) in circuits {
            let edge = (i.min(j), i.max(j));
            if incompat_edges_set.insert(edge) {
                added_new_edges = true;
            }
        }

        if !added_new_edges {
            return cmp::max(chromatic_num, safe_upper_bound);
        }
    }
}

// ─── Hybrid (DSatur-Guided Selective Pruning) Implementation ─────────────

/// Pre-allocated buffers for cycle detection in the hot loop.
struct CycleBufs {
    path: Vec<usize>,
    visited: Vec<bool>,
    seen_x: Vec<bool>,
    seen_y: Vec<bool>,
}

/// Computes the dimension using DSatur-guided backtracking with selective
/// hypergraph pruning.
fn hybrid_dimension<T: Clone>(poset: &Poset<T>) -> usize {
    let n = poset.nodes.len();
    let n_edges = poset.transitive_edges.len();

    if n <= 1 { return 1; }
    if 2 * n_edges == n * (n - 1) { return 1; }
    if n_edges == 0 { return 2; }

    let critical_pairs = poset.critical_pairs();
    if critical_pairs.is_empty() { return 1; }

    let m = critical_pairs.len();

    // Build alternating cycle adjacency (from HG) FIRST — needed for amplification
    let hg_adj = build_alternating_cycle_adj(&critical_pairs, poset);

    // Build amplified incompatibility graph: start with G*(P) from GC,
    // then add 2-alternating cycle edges from HG. A 2-alternating cycle is
    // two critical pairs (a₁,b₁) and (a₂,b₂) with a₁ ≤ b₂ AND a₂ ≤ b₁.
    // These pairs MUST have different colors, so adding them as graph edges
    // is correct and drastically reduces the backtracking search space.
    let incompat_edges = build_incompatibility_graph(&critical_pairs, poset);
    let mut edge_set: HashSet<(usize, usize)> = incompat_edges.iter()
        .map(|&(u, v)| (u.min(v), u.max(v)))
        .collect();

    for i in 0..m {
        for &j in &hg_adj[i] {
            if j > i && hg_adj[j].binary_search(&i).is_ok() {
                // Mutual adjacency: 2-alternating cycle → must have different colors
                edge_set.insert((i, j));
            }
        }
    }

    let mut graph_adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for &(u, v) in &edge_set {
        graph_adj[u].push(v);
        graph_adj[v].push(u);
    }

    // DSatur on the amplified graph: ordering + color hints
    let amplified_edges: Vec<(usize, usize)> = edge_set.iter().copied().collect();
    let (_, dsatur_colors, dsatur_order) = dsatur_with_ordering(m, &amplified_edges);

    // Hiraguchi's bound: dim(P) <= n/2 for n >= 4
    let max_dim = cmp::max(2, n / 2);

    // Pre-allocate cycle search buffers (zero alloc in hot loop)
    let mut bufs = CycleBufs {
        path: Vec::with_capacity(m),
        visited: vec![false; m],
        seen_x: vec![false; n],
        seen_y: vec![false; n],
    };

    for k in 2..=max_dim {
        let mut colors = vec![0usize; m];
        let mut colored = vec![false; m];

        if hybrid_backtrack(
            0, k, &mut colors, &mut colored,
            &graph_adj, &hg_adj, &dsatur_order, &dsatur_colors,
            &critical_pairs, poset, &mut bufs,
        ) {
            return k;
        }
    }

    max_dim
}

/// DSatur-guided backtracking with selective hypergraph pruning.
///
/// For each vertex (in DSatur order) and each color candidate:
/// 1. Graph prune: skip if any graph neighbor already has this color (O(degree))
/// 2. Selective hypergraph prune: only check for monochromatic strict
///    alternating cycles if vertex has a same-colored hg neighbor
/// 3. Forward check: skip if any uncolored graph neighbor has all k colors blocked
fn hybrid_backtrack<T: Clone>(
    step: usize,
    k: usize,
    colors: &mut [usize],
    colored: &mut [bool],
    graph_adj: &[Vec<usize>],
    hg_adj: &[Vec<usize>],
    ordering: &[usize],
    hints: &[usize],
    pairs: &[(u32, u32)],
    poset: &Poset<T>,
    bufs: &mut CycleBufs,
) -> bool {
    if step == ordering.len() {
        return true;
    }

    let vertex = ordering[step];
    let hint = if hints[vertex] < k { hints[vertex] } else { 0 };

    for color_idx in 0..k {
        // Color order: hint first, then 0..k excluding hint
        let c = if color_idx == 0 {
            hint
        } else {
            let mut c = color_idx - 1;
            if c >= hint { c += 1; }
            c
        };

        // Fast prune: graph adjacency
        let graph_conflict = graph_adj[vertex].iter().any(|&nb| colored[nb] && colors[nb] == c);
        if graph_conflict { continue; }

        colors[vertex] = c;
        colored[vertex] = true;

        // Selective prune: only check for strict alternating cycles
        // if vertex has a same-colored neighbor in the hg adjacency.
        // Most assignments skip this entirely.
        let needs_check = hg_adj[vertex].iter().any(|&nb| colored[nb] && colors[nb] == c);
        let cycle_found = if needs_check {
            check_cycle_involving(vertex, c, colors, colored, hg_adj, pairs, poset, bufs)
        } else {
            false
        };

        if !cycle_found {
            if hybrid_backtrack(
                step + 1, k, colors, colored,
                graph_adj, hg_adj, ordering, hints,
                pairs, poset, bufs,
            ) {
                return true;
            }
        }

        colored[vertex] = false;
    }

    false
}

/// Checks if assigning color `c` to `vertex` creates a monochromatic strict
/// alternating cycle involving `vertex`. Uses pre-allocated buffers.
fn check_cycle_involving<T: Clone>(
    vertex: usize,
    color: usize,
    colors: &[usize],
    colored: &[bool],
    adj: &[Vec<usize>],
    pairs: &[(u32, u32)],
    poset: &Poset<T>,
    bufs: &mut CycleBufs,
) -> bool {
    bufs.path.clear();
    bufs.path.push(vertex);
    for v in &mut bufs.visited { *v = false; }
    bufs.visited[vertex] = true;
    for v in &mut bufs.seen_x { *v = false; }
    bufs.seen_x[pairs[vertex].0 as usize] = true;
    for v in &mut bufs.seen_y { *v = false; }
    bufs.seen_y[pairs[vertex].1 as usize] = true;

    dfs_check_cycle(vertex, vertex, color, colors, colored, adj, pairs, poset, bufs)
}

/// DFS for strict alternating cycles involving start, restricted to same-colored
/// vertices. Returns true if a monochromatic strict alternating cycle is found.
fn dfs_check_cycle<T: Clone>(
    start: usize,
    current: usize,
    color: usize,
    colors: &[usize],
    colored: &[bool],
    adj: &[Vec<usize>],
    pairs: &[(u32, u32)],
    poset: &Poset<T>,
    bufs: &mut CycleBufs,
) -> bool {
    for &next in &adj[current] {
        if !colored[next] || colors[next] != color { continue; }

        if next == start && bufs.path.len() >= 2 {
            if is_strict_alternating_cycle(&bufs.path, pairs, poset) {
                return true;
            }
            continue;
        }

        if bufs.visited[next] { continue; }

        let xn = pairs[next].0 as usize;
        let yn = pairs[next].1 as usize;
        if bufs.seen_x[xn] || bufs.seen_y[yn] { continue; }

        bufs.visited[next] = true;
        bufs.seen_x[xn] = true;
        bufs.seen_y[yn] = true;
        bufs.path.push(next);

        if dfs_check_cycle(start, next, color, colors, colored, adj, pairs, poset, bufs) {
            return true;
        }

        bufs.path.pop();
        bufs.visited[next] = false;
        bufs.seen_x[xn] = false;
        bufs.seen_y[yn] = false;
    }

    false
}

/// Runs DSatur on the incompatibility graph and returns the chromatic number,
/// coloring, and vertex ordering.
fn dsatur_with_ordering(
    n: usize,
    edges: &[(usize, usize)],
) -> (usize, Vec<usize>, Vec<usize>) {
    if n == 0 { return (0, Vec::new(), Vec::new()); }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        if u < n && v < n {
            adj[u].push(v);
            adj[v].push(u);
        }
    }

    let degrees: Vec<usize> = (0..n).map(|i| adj[i].len()).collect();
    let mut colors = vec![0usize; n];
    let mut colored = vec![false; n];
    let mut ordering = Vec::with_capacity(n);
    let mut sat_counts = vec![0usize; n];
    let mut neighbor_colors = vec![false; n * n];

    for _ in 0..n {
        let next = (0..n)
            .filter(|&v| !colored[v])
            .max_by(|&a, &b| {
                sat_counts[a].cmp(&sat_counts[b])
                    .then_with(|| degrees[a].cmp(&degrees[b]))
            })
            .unwrap();

        let mut c = 0;
        let row_offset = next * n;
        while neighbor_colors[row_offset + c] { c += 1; }

        colors[next] = c;
        colored[next] = true;
        ordering.push(next);

        for &neighbor in &adj[next] {
            if !colored[neighbor] {
                let idx = neighbor * n + c;
                if !neighbor_colors[idx] {
                    neighbor_colors[idx] = true;
                    sat_counts[neighbor] += 1;
                }
            }
        }
    }

    let chromatic_num = colors.iter().copied().max().unwrap_or(0) + 1;
    (chromatic_num, colors, ordering)
}

/// Constructs a realizer from the Hybrid coloring.
fn hybrid_realizer<T: Clone>(poset: &Poset<T>) -> Vec<Vec<usize>> {
    let n = poset.nodes.len();
    if n <= 1 || poset.critical_pairs().is_empty() {
        let le = poset.linear_extension().unwrap_or_else(|| (0..n).collect());
        return vec![le];
    }

    let critical_pairs = poset.critical_pairs();
    let m = critical_pairs.len();
    let hg_adj = build_alternating_cycle_adj(&critical_pairs, poset);

    let incompat_edges = build_incompatibility_graph(&critical_pairs, poset);
    let mut edge_set: HashSet<(usize, usize)> = incompat_edges.iter()
        .map(|&(u, v)| (u.min(v), u.max(v)))
        .collect();
    for i in 0..m {
        for &j in &hg_adj[i] {
            if j > i && hg_adj[j].binary_search(&i).is_ok() {
                edge_set.insert((i, j));
            }
        }
    }
    let mut graph_adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for &(u, v) in &edge_set {
        graph_adj[u].push(v);
        graph_adj[v].push(u);
    }

    let amplified_edges: Vec<(usize, usize)> = edge_set.iter().copied().collect();
    let (_, dsatur_colors, dsatur_order) = dsatur_with_ordering(m, &amplified_edges);

    let max_dim = cmp::max(2, n / 2);

    let mut bufs = CycleBufs {
        path: Vec::with_capacity(m),
        visited: vec![false; m],
        seen_x: vec![false; n],
        seen_y: vec![false; n],
    };

    for k in 2..=max_dim {
        let mut colors = vec![0usize; m];
        let mut colored = vec![false; m];

        if hybrid_backtrack(
            0, k, &mut colors, &mut colored,
            &graph_adj, &hg_adj, &dsatur_order, &dsatur_colors,
            &critical_pairs, poset, &mut bufs,
        ) {
            return realize_with_iterative_extension(poset, &critical_pairs, &colors, k);
        }
    }

    unreachable!()
}

// ─── Realizer Functions ────────────────────────────────────────────────────

/// Converts a coloring of critical pairs into a realizer via direct construction.
///
/// Two-phase approach:
/// 1. Add reverse edges for critical pairs (oriented in their class, reversed
///    in another class) — checking acyclicity before each addition.
/// 2. Add reverse edges for any remaining incomparable pairs that are ordered
///    the same way in all extensions — again checking acyclicity.
///
/// Adding (b,a) for a critical pair (a,b) to extension c ≠ class(a,b) is
/// safe as long as no path a→...→b exists in extension c's edge set. We
/// verify this by attempting topological sort after each addition.
fn coloring_to_realizer_direct<T: Clone>(
    poset: &Poset<T>,
    critical_pairs: &[(u32, u32)],
    coloring: &[usize],
    k: usize,
) -> Vec<Vec<usize>> {
    let n = poset.nodes.len();
    if k == 1 {
        let le = poset.linear_extension().unwrap_or_else(|| (0..n).collect());
        return vec![le];
    }

    // Build initial edge sets: poset edges + critical pair orientations per class
    let mut edge_sets: Vec<HashSet<(usize, usize)>> = Vec::with_capacity(k);
    for c in 0..k {
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        for &(u, v) in &poset.transitive_edges {
            edges.insert((u as usize, v as usize));
        }
        for (i, &color) in coloring.iter().enumerate() {
            if color == c {
                let (a, b) = critical_pairs[i];
                edges.insert((a as usize, b as usize));
            }
        }
        edge_sets.push(edges);
    }

    // Phase 1: add reverse edges for critical pairs.
    // For each critical pair (a,b) in class c, we need (b,a) in some extension i ≠ c.
    // Try extensions in order (c+1)%k, (c+2)%k, etc., checking acyclicity.
    for (i, &color) in coloring.iter().enumerate() {
        let (a, b) = critical_pairs[i];
        let (a_u, b_u) = (a as usize, b as usize);
        for offset in 1..=k {
            let c = (color + offset) % k;
            edge_sets[c].insert((b_u, a_u));
            if Poset::<T>::linear_extension_static(n, &edge_sets[c]).is_some() {
                break; // safe to keep
            }
            edge_sets[c].remove(&(b_u, a_u));
        }
    }

    // Compute linear extensions after critical pair reversals
    let mut realizer: Vec<Vec<usize>> = edge_sets.iter()
        .map(|edges| Poset::<T>::linear_extension_static(n, edges).expect("DAG after critical reversals"))
        .collect();

    // Phase 2: fix any remaining incomparable pairs that are ordered the same
    // way in all extensions. Add reverse edges (cycle-checking each).
    let incomparable = poset.incomparable_pairs();
    let mut changed = true;
    while changed {
        changed = false;
        for &(a, b) in &incomparable {
            let (a_u, b_u) = (a as usize, b as usize);
            let a_before_b_in_all = realizer.iter().all(|le| {
                le.iter().position(|&x| x == a_u).unwrap() <
                le.iter().position(|&x| x == b_u).unwrap()
            });
            let b_before_a_in_all = realizer.iter().all(|le| {
                le.iter().position(|&x| x == b_u).unwrap() <
                le.iter().position(|&x| x == a_u).unwrap()
            });
            if a_before_b_in_all || b_before_a_in_all {
                let (from, to) = if a_before_b_in_all { (b_u, a_u) } else { (a_u, b_u) };
                for c in 0..k {
                    edge_sets[c].insert((from, to));
                    if let Some(le) = Poset::<T>::linear_extension_static(n, &edge_sets[c]) {
                        realizer[c] = le;
                        changed = true;
                        break;
                    }
                    edge_sets[c].remove(&(from, to));
                }
            }
        }
    }

    realizer
}

/// Converts an HG/Hybrid coloring into a realizer using iterative extension.
///
/// A valid coloring of K_s(P) guarantees no monochromatic strict alternating
/// cycle, but non-strict alternating cycles can still create cycles in the
/// direct construction. This function resolves them by repeatedly:
/// 1. Detecting cycles in color classes via find_circuits_in_realizer,
/// 2. Adding incompatibility edges between consecutive pairs in each cycle,
/// 3. Re-coloring with the exact graph coloring algorithm.
/// Once no cycles remain, converts to realizer via coloring_to_realizer_direct.
fn realize_with_iterative_extension<T: Clone>(
    poset: &Poset<T>,
    critical_pairs: &[(u32, u32)],
    initial_coloring: &[usize],
    k: usize,
) -> Vec<Vec<usize>> {
    let mut incompat_edges = build_incompatibility_graph(critical_pairs, poset);
    let mut coloring = initial_coloring.to_vec();

    loop {
        let circuits = find_circuits_in_realizer(critical_pairs, &coloring, k, poset);
        if circuits.is_empty() {
            return coloring_to_realizer_direct(poset, critical_pairs, &coloring, k);
        }
        for (i, j) in circuits {
            let (min_ij, max_ij) = (i.min(j), i.max(j));
            if !incompat_edges.contains(&(min_ij, max_ij)) {
                incompat_edges.push((min_ij, max_ij));
            }
        }
        let (_, new_coloring) = chromatic_number_exact(critical_pairs.len(), &incompat_edges);
        coloring = new_coloring;
    }
}

fn hypergraph_coloring_realizer<T: Clone>(poset: &Poset<T>) -> Vec<Vec<usize>> {
    let n = poset.nodes.len();
    if n <= 1 || poset.critical_pairs().is_empty() {
        let le = poset.linear_extension().unwrap_or_else(|| (0..n).collect());
        return vec![le];
    }
    let critical_pairs = poset.critical_pairs();
    let hyperedges = find_minimal_hyperedges(&critical_pairs, poset);
    for k in 2..=critical_pairs.len() {
        if let Some(coloring) = hypergraph_coloring_solve(critical_pairs.len(), &hyperedges, k) {
            return realize_with_iterative_extension(poset, &critical_pairs, &coloring, k);
        }
    }
    unreachable!()
}

fn graph_coloring_realizer<T: Clone>(poset: &Poset<T>) -> Vec<Vec<usize>> {
    let n = poset.nodes.len();
    if n <= 1 || poset.critical_pairs().is_empty() {
        let le = poset.linear_extension().unwrap_or_else(|| (0..n).collect());
        return vec![le];
    }
    let critical_pairs = poset.critical_pairs();
    let mut incompat_edges = build_incompatibility_graph(&critical_pairs, poset);

    loop {
        let (chromatic_num, coloring) = chromatic_number_exact(critical_pairs.len(), &incompat_edges);
        let circuits = find_circuits_in_realizer(&critical_pairs, &coloring, chromatic_num, poset);

        if circuits.is_empty() {
            return coloring_to_realizer_direct(poset, &critical_pairs, &coloring, chromatic_num);
        }

        for (i, j) in circuits {
            let (min_ij, max_ij) = (i.min(j), i.max(j));
            if !incompat_edges.contains(&(min_ij, max_ij)) {
                incompat_edges.push((min_ij, max_ij));
            }
        }
    }
}

fn graph_coloring_heuristic_realizer<T: Clone>(poset: &Poset<T>) -> Vec<Vec<usize>> {
    let n = poset.nodes.len();
    if n <= 1 || poset.critical_pairs().is_empty() {
        let le = poset.linear_extension().unwrap_or_else(|| (0..n).collect());
        return vec![le];
    }
    let critical_pairs = poset.critical_pairs();
    let mut incompat_edges = build_incompatibility_graph(&critical_pairs, poset);

    loop {
        let (chromatic_num, coloring) = chromatic_number_heuristic(critical_pairs.len(), &incompat_edges);
        let circuits = find_circuits_in_realizer(&critical_pairs, &coloring, chromatic_num, poset);

        if circuits.is_empty() {
            return coloring_to_realizer_direct(poset, &critical_pairs, &coloring, chromatic_num);
        }

        for (i, j) in circuits {
            let (min_ij, max_ij) = (i.min(j), i.max(j));
            if !incompat_edges.contains(&(min_ij, max_ij)) {
                incompat_edges.push((min_ij, max_ij));
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn chain_poset() -> Poset<u32> {
        Poset::from_covering_relation(vec![0u32, 1, 2], vec![(0, 1), (1, 2)]).unwrap()
    }

    fn antichain_poset() -> Poset<u32> {
        Poset::from_covering_relation(vec![0u32, 1, 2], vec![]).unwrap()
    }

    fn antichain_poset_n(n: usize) -> Poset<u32> {
        let nodes: Vec<u32> = (0..n as u32).collect();
        Poset::from_covering_relation(nodes, vec![]).unwrap()
    }

    fn diamond_poset() -> Poset<u32> {
        Poset::from_covering_relation(
            vec![0u32, 1, 2, 3],
            vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        ).unwrap()
    }

    fn standard_example_s3() -> Poset<u32> {
        Poset::from_covering_relation(
            vec![0u32, 1, 2, 3, 4, 5],
            vec![(0, 4), (0, 5), (1, 3), (1, 5), (2, 3), (2, 4)],
        ).unwrap()
    }

    fn single_element() -> Poset<u32> {
        Poset::from_covering_relation(vec![0u32], vec![]).unwrap()
    }

    fn n_shaped() -> Poset<u32> {
        Poset::from_covering_relation(
            vec![0u32, 1, 2, 3], vec![(0, 1), (2, 3)],
        ).unwrap()
    }

    // ─── SAT Reduction tests ─────────────────────────────────────────────

    #[test]
    fn test_sat_chain() { assert_eq!(sat_dimension(&chain_poset()), 1); }
    #[test]
    fn test_sat_antichain() { assert_eq!(sat_dimension(&antichain_poset()), 2); }
    #[test]
    fn test_sat_single() { assert_eq!(sat_dimension(&single_element()), 1); }
    #[test]
    fn test_sat_diamond() { assert_eq!(sat_dimension(&diamond_poset()), 2); }
    #[test]
    fn test_sat_n_shaped() { assert_eq!(sat_dimension(&n_shaped()), 2); }
    #[test]
    fn test_sat_s3() { assert_eq!(sat_dimension(&standard_example_s3()), 3); }
    #[test]
    fn test_sat_trait() { assert_eq!(SatReduction.dimension(&chain_poset()), 1); }

    // ─── Hypergraph Coloring tests ──────────────────────────────────────

    #[test]
    fn test_hg_chain() { assert_eq!(HypergraphColoring.dimension(&chain_poset()), 1); }
    #[test]
    fn test_hg_antichain() { assert_eq!(HypergraphColoring.dimension(&antichain_poset()), 2); }
    #[test]
    fn test_hg_antichain_5() {
        let poset = antichain_poset_n(5);
        assert_eq!(HypergraphColoring.dimension(&poset), 2);
        assert_eq!(Hybrid.dimension(&poset), 2);
        assert_eq!(SatReduction.dimension(&poset), 2);
        assert_eq!(GraphColoring.dimension(&poset), 2);
        assert_eq!(GraphColoringHeuristic.dimension(&poset), 2);
    }
    #[test]
    fn test_antichain_6() {
        let poset = antichain_poset_n(6);
        assert_eq!(SatReduction.dimension(&poset), 2);
        assert_eq!(GraphColoring.dimension(&poset), 2);
        assert_eq!(GraphColoringHeuristic.dimension(&poset), 2);
    }
    #[test]
    fn test_hg_diamond() { assert_eq!(HypergraphColoring.dimension(&diamond_poset()), 2); }
    #[test]
    fn test_hg_s3() { assert_eq!(HypergraphColoring.dimension(&standard_example_s3()), 3); }

    // ─── Graph Coloring tests (Yáñez & Montero) ─────────────────────────

    #[test]
    fn test_gc_chain() { assert_eq!(GraphColoring.dimension(&chain_poset()), 1); }
    #[test]
    fn test_gc_antichain() { assert_eq!(GraphColoring.dimension(&antichain_poset()), 2); }
    #[test]
    fn test_gc_diamond() { assert_eq!(GraphColoring.dimension(&diamond_poset()), 2); }
    #[test]
    fn test_gc_s3() { assert_eq!(GraphColoring.dimension(&standard_example_s3()), 3); }

    // ─── Graph Coloring Heuristic tests (DSatur) ───────────────────────

    #[test]
    fn test_gch_chain() { assert_eq!(GraphColoringHeuristic.dimension(&chain_poset()), 1); }
    #[test]
    fn test_gch_antichain() { assert_eq!(GraphColoringHeuristic.dimension(&antichain_poset()), 2); }
    #[test]
    fn test_gch_diamond() { assert_eq!(GraphColoringHeuristic.dimension(&diamond_poset()), 2); }
    #[test]
    fn test_gch_s3() { assert_eq!(GraphColoringHeuristic.dimension(&standard_example_s3()), 3); }

    #[test]
    fn test_gch_is_upper_bound() {
        // Heuristic must give >= exact dimension on small posets
        let posets: Vec<(&str, Poset<u32>)> = vec![
            ("chain", chain_poset()),
            ("antichain", antichain_poset()),
            ("diamond", diamond_poset()),
            ("s3", standard_example_s3()),
            ("n-shaped", n_shaped()),
        ];
        for (name, poset) in &posets {
            let sat = SatReduction.dimension(poset);
            let gch = GraphColoringHeuristic.dimension(poset);
            assert!(gch >= sat, "GCH ({}) < SAT ({}) on {}", gch, sat, name);
        }
    }

    // ─── Hybrid tests ──────────────────────────────────────────────────

    #[test]
    fn test_hybrid_chain() { assert_eq!(Hybrid.dimension(&chain_poset()), 1); }
    #[test]
    fn test_hybrid_antichain() { assert_eq!(Hybrid.dimension(&antichain_poset()), 2); }
    #[test]
    fn test_hybrid_diamond() { assert_eq!(Hybrid.dimension(&diamond_poset()), 2); }
    #[test]
    fn test_hybrid_s3() { assert_eq!(Hybrid.dimension(&standard_example_s3()), 3); }

    // ─── Cross-algorithm consistency ─────────────────────────────────────

    #[test]
    fn test_all_algorithms_agree() {
        let posets: Vec<(&str, Poset<u32>)> = vec![
            ("chain", chain_poset()),
            ("antichain", antichain_poset()),
            ("diamond", diamond_poset()),
            ("n-shaped", n_shaped()),
            ("s3", standard_example_s3()),
        ];

        for (name, poset) in &posets {
            let sat = SatReduction.dimension(poset);
            let hg = HypergraphColoring.dimension(poset);
            let gc = GraphColoring.dimension(poset);
            let hybrid = Hybrid.dimension(poset);
            let gch = GraphColoringHeuristic.dimension(poset);
            assert_eq!(sat, hg, "SAT vs HG disagree on {}", name);
            assert_eq!(sat, hybrid, "SAT vs Hybrid disagree on {}", name);
            assert!(gc <= sat, "GC ({}) > SAT ({}) on {}", gc, sat, name);
            assert!(gch >= sat, "GCH ({}) < SAT ({}) on {}", gch, sat, name);
        }
    }

    #[test]
    fn test_dimension_method() {
        assert_eq!(chain_poset().dimension(), 1);
        assert_eq!(antichain_poset().dimension(), 2);
        assert_eq!(diamond_poset().dimension(), 2);
    }

    // ─── Incompatibility graph tests ────────────────────────────────────

    #[test]
    fn test_incompatibility_graph_diamond() {
        let p = diamond_poset();
        let cp = p.critical_pairs();
        let edges = build_incompatibility_graph(&cp, &p);
        // Diamond has 2 critical pairs (1,2) and (2,1), which are opposite
        // and thus incompatible
        assert!(!edges.is_empty(), "diamond should have incompatibility edges");
    }

    #[test]
    fn test_incompatibility_graph_antichain() {
        // Antichain of 3 elements: 6 critical pairs, all opposite pairs
        // are incompatible
        let p = antichain_poset();
        let cp = p.critical_pairs();
        assert_eq!(cp.len(), 6, "antichain should have 6 critical pairs");
        let edges = build_incompatibility_graph(&cp, &p);
        // 3 opposite pairs: (0,1)-(1,0), (0,2)-(2,0), (1,2)-(2,1)
        assert!(edges.len() >= 3, "antichain should have at least 3 edges (opposite pairs)");
    }

    #[test]
    fn test_incompatibility_graph_s3() {
        // Standard example S₃: 3 critical pairs, no 2-alternating cycles
        let p = standard_example_s3();
        let cp = p.critical_pairs();
        assert_eq!(cp.len(), 3, "S₃ should have 3 critical pairs");
        let edges = build_incompatibility_graph(&cp, &p);
        // The 3 critical pairs don't form 2-alternating cycles with each other,
        // so the incompatibility graph should have no edges (besides opposite pairs,
        // but S₃ has no opposite pairs among critical pairs)
        assert_eq!(edges.len(), 0, "S₃ critical pairs have no incompatibility edges");
    }

    // ─── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_two_element_chain() {
        let p = Poset::from_covering_relation(vec![0u32, 1], vec![(0, 1)]).unwrap();
        assert_eq!(sat_dimension(&p), 1);
        assert_eq!(HypergraphColoring.dimension(&p), 1);
        assert_eq!(GraphColoring.dimension(&p), 1);
        assert_eq!(Hybrid.dimension(&p), 1);
    }

    #[test]
    fn test_two_element_antichain() {
        let p = Poset::from_covering_relation(vec![0u32, 1], vec![]).unwrap();
        assert_eq!(sat_dimension(&p), 2);
        assert_eq!(HypergraphColoring.dimension(&p), 2);
        assert_eq!(GraphColoring.dimension(&p), 2);
        assert_eq!(Hybrid.dimension(&p), 2);
    }

    #[test]
    fn test_four_element_antichain() {
        let p = Poset::from_covering_relation(vec![0u32, 1, 2, 3], vec![]).unwrap();
        assert_eq!(sat_dimension(&p), 2);
        assert_eq!(HypergraphColoring.dimension(&p), 2);
        assert_eq!(GraphColoring.dimension(&p), 2);
        assert_eq!(Hybrid.dimension(&p), 2);
    }

    #[test]
    fn test_standard_example_s4() {
        // Standard example S₄: dimension 4
        // Elements: 0,1,2,3 (minimal), 4,5,6,7 (maximal)
        // a_i < b_j iff i != j
        let p = Poset::from_covering_relation(
            vec![0u32, 1, 2, 3, 4, 5, 6, 7],
            vec![
                (0, 5), (0, 6), (0, 7),  // a_0 < b_1, b_2, b_3
                (1, 4), (1, 6), (1, 7),  // a_1 < b_0, b_2, b_3
                (2, 4), (2, 5), (2, 7),  // a_2 < b_0, b_1, b_3
                (3, 4), (3, 5), (3, 6),  // a_3 < b_0, b_1, b_2
            ],
        ).unwrap();
        assert_eq!(sat_dimension(&p), 4);
        assert_eq!(HypergraphColoring.dimension(&p), 4);
        assert_eq!(GraphColoring.dimension(&p), 4);
        assert_eq!(Hybrid.dimension(&p), 4);
    }

    #[test]
    fn test_crown_3() {
        // Crown poset (standard example S₃): 3 minimal {0,1,2}, 3 maximal {3,4,5}
        // a_i < b_j for all j ≠ i (i.e., a_i is incomparable to b_i only)
        // Covering: 0<4, 0<5, 1<3, 1<5, 2<3, 2<4
        // Dimension = 3 (standard example)
        let p = Poset::from_covering_relation(
            vec![0u32, 1, 2, 3, 4, 5],
            vec![(0, 4), (0, 5), (1, 3), (1, 5), (2, 3), (2, 4)],
        ).unwrap();
        let dim = sat_dimension(&p);
        // Crown S₃ is the standard example — dimension = 3
        assert_eq!(dim, 3, "crown S₃ (standard example) has dimension 3");
        assert_eq!(HypergraphColoring.dimension(&p), 3);
        assert_eq!(GraphColoring.dimension(&p), 3);
        assert_eq!(Hybrid.dimension(&p), 3);
    }

    // ─── Cross-algorithm consistency with more posets ────────────────────

    #[test]
    fn test_all_algorithms_extended() {
        // Additional posets beyond the basic test
        let posets: Vec<(&str, Poset<u32>)> = vec![
            ("single", single_element()),
            ("2-chain", Poset::from_covering_relation(vec![0u32, 1], vec![(0, 1)]).unwrap()),
            ("2-antichain", Poset::from_covering_relation(vec![0u32, 1], vec![]).unwrap()),
            ("4-antichain", Poset::from_covering_relation(vec![0u32, 1, 2, 3], vec![]).unwrap()),
            ("n-shaped", n_shaped()),
            ("s3", standard_example_s3()),
        ];

        for (name, poset) in &posets {
            let sat = SatReduction.dimension(poset);
            let hg = HypergraphColoring.dimension(poset);
            let gc = GraphColoring.dimension(poset);
            let hybrid = Hybrid.dimension(poset);
            let gch = GraphColoringHeuristic.dimension(poset);
            assert_eq!(sat, hg, "SAT vs HG disagree on {}", name);
            assert_eq!(sat, hybrid, "SAT vs Hybrid disagree on {}", name);
            assert!(gc <= sat, "GC ({}) > SAT ({}) on {}", gc, sat, name);
            assert!(gch >= sat, "GCH ({}) < SAT ({}) on {}", gch, sat, name);
        }
    }

    // ─── Graph coloring internal tests ────────────────────────────────────

    #[test]
    fn test_chromatic_number_empty_graph() {
        let (k, colors) = chromatic_number_exact(0, &[]);
        assert_eq!(k, 0);
        assert!(colors.is_empty());
    }

    #[test]
    fn test_chromatic_number_single_vertex() {
        let (k, colors) = chromatic_number_exact(1, &[]);
        assert_eq!(k, 1);
        assert_eq!(colors.len(), 1);
    }

    #[test]
    fn test_chromatic_number_complete_graph() {
        // K₃ (triangle): chromatic number = 3
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let (k, _) = chromatic_number_exact(3, &edges);
        assert_eq!(k, 3);
    }

    #[test]
    fn test_chromatic_number_path() {
        // P₃ (path graph): chromatic number = 2
        let edges = vec![(0, 1), (1, 2)];
        let (k, _) = chromatic_number_exact(3, &edges);
        assert_eq!(k, 2);
    }

    // ─── DSatur heuristic tests ───────────────────────────────────────

    #[test]
    fn test_dsatur_empty_graph() {
        let (k, colors) = chromatic_number_heuristic(0, &[]);
        assert_eq!(k, 0);
        assert!(colors.is_empty());
    }

    #[test]
    fn test_dsatur_single_vertex() {
        let (k, colors) = chromatic_number_heuristic(1, &[]);
        assert_eq!(k, 1);
        assert_eq!(colors.len(), 1);
    }

    #[test]
    fn test_dsatur_complete_graph() {
        // K₃: DSatur gives exact answer on complete graphs
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let (k, _) = chromatic_number_heuristic(3, &edges);
        assert_eq!(k, 3);
    }

    #[test]
    fn test_dsatur_path() {
        // P₃: DSatur gives exact answer on bipartite graphs
        let edges = vec![(0, 1), (1, 2)];
        let (k, _) = chromatic_number_heuristic(3, &edges);
        assert_eq!(k, 2);
    }

    #[test]
    fn test_dsatur_upper_bound() {
        // DSatur result must be >= exact chromatic number
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let (exact_k, _) = chromatic_number_exact(3, &edges);
        let (heur_k, _) = chromatic_number_heuristic(3, &edges);
        assert!(heur_k >= exact_k);
    }

    #[test]
    fn test_dsatur_cycle_5() {
        // C₅ (5-cycle): chromatic number = 3
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
        let (exact_k, _) = chromatic_number_exact(5, &edges);
        let (heur_k, _) = chromatic_number_heuristic(5, &edges);
        assert_eq!(exact_k, 3);
        assert!(heur_k >= exact_k, "DSatur ({}) < exact ({})", heur_k, exact_k);
    }

    // ─── Yáñez & Montero iterative extension test ─────────────────────

    #[test]
    fn test_gc_iterative_extension_s3() {
        // S₃ requires the extension procedure: initial incompatibility graph
        // has no edges, but circuits are found and edges are added iteratively
        let p = standard_example_s3();
        // GraphColoring should find dimension 3 through iterative extension
        assert_eq!(GraphColoring.dimension(&p), 3);
    }

    #[test]
    fn test_gc_iterative_extension_antichain() {
        // Antichain requires iterative extension for opposite pairs
        let p = antichain_poset();
        assert_eq!(GraphColoring.dimension(&p), 2);
    }

    // ─── Critical pairs tests ───────────────────────────────────────────

    #[test]
    fn test_critical_pairs_s3() {
        let p = standard_example_s3();
        let cp = p.critical_pairs();
        // S₃ has exactly 3 critical pairs: (0,3), (1,4), (2,5)
        assert_eq!(cp.len(), 3);
        // Check that each is a valid critical pair
        for (a, b) in &cp {
            assert!(!p.is_leq(*a, *b), "critical pair ({}, {}) should be incomparable", a, b);
            assert!(!p.is_leq(*b, *a), "critical pair ({}, {}) should be incomparable (reverse)", a, b);
        }
    }

    #[test]
    fn test_critical_pairs_diamond_count() {
        let p = diamond_poset();
        let cp = p.critical_pairs();
        // Diamond has 2 critical pairs: (1,2) and (2,1)
        assert_eq!(cp.len(), 2);
    }

    // ─── Concept lattice (FCA) dimension tests ────────────────────────────

    fn load_concept_lattice_poset(filename: &str) -> Poset<(bit_set::BitSet, bit_set::BitSet)> {
        use std::fs;
        let path = format!("test_data/{}", filename);
        let bytes = fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));
        let ctx = crate::FormalContext::<String>::from(&bytes).unwrap();
        let lattice = ctx.concept_lattice().unwrap();
        lattice.poset
    }

    #[test]
    fn test_concept_lattice_famous_animals() {
        let poset = load_concept_lattice_poset("famous_animals_en.cxt");
        let sat_dim = SatReduction.dimension(&poset);
        let hg_dim = HypergraphColoring.dimension(&poset);
        let gc_dim = GraphColoring.dimension(&poset);
        assert_eq!(sat_dim, 3, "famous_animals: SAT dimension should be 3");
        assert!(hg_dim >= sat_dim, "famous_animals: HG should be >= SAT dimension (upper bound)");
        assert!(gc_dim >= sat_dim, "famous_animals: GC should be >= SAT dimension (upper bound)");
    }

    #[test]
    fn test_concept_lattice_triangles() {
        let poset = load_concept_lattice_poset("triangles.cxt");
        let sat_dim = SatReduction.dimension(&poset);
        let hg_dim = HypergraphColoring.dimension(&poset);
        assert_eq!(sat_dim, 3, "triangles: SAT dimension should be 3");
        assert!(hg_dim >= sat_dim, "triangles: HG should be >= SAT dimension (upper bound)");
    }

    #[test]
    fn test_concept_lattice_livingbeings() {
        let poset = load_concept_lattice_poset("livingbeings_en.cxt");
        let sat_dim = SatReduction.dimension(&poset);
        let hg_dim = HypergraphColoring.dimension(&poset);
        assert_eq!(sat_dim, 3, "livingbeings: SAT dimension should be 3");
        assert!(hg_dim >= sat_dim, "livingbeings: HG should be >= SAT dimension (upper bound)");
    }

    #[test]
    fn test_concept_lattice_planets() {
        let poset = load_concept_lattice_poset("planets_en.cxt");
        let sat_dim = SatReduction.dimension(&poset);
        let hg_dim = HypergraphColoring.dimension(&poset);
        assert_eq!(sat_dim, 2, "planets: SAT dimension should be 2");
        assert!(hg_dim >= sat_dim, "planets: HG should be >= SAT dimension (upper bound)");
    }

    #[test]
    fn test_concept_lattice_missmarple() {
        let poset = load_concept_lattice_poset("missmarple_en.cxt");
        let sat_dim = SatReduction.dimension(&poset);
        let hg_dim = HypergraphColoring.dimension(&poset);
        assert_eq!(sat_dim, 3, "missmarple: SAT dimension should be 3");
        assert!(hg_dim >= sat_dim, "missmarple: HG should be >= SAT dimension (upper bound)");
    }

    #[test]
    fn test_concept_lattice_forum_romanum() {
        let poset = load_concept_lattice_poset("forum_romanum_en.cxt");
        let sat_dim = SatReduction.dimension(&poset);
        let hg_dim = HypergraphColoring.dimension(&poset);
        assert_eq!(sat_dim, 3, "forum_romanum: SAT dimension should be 3");
        assert!(hg_dim >= sat_dim, "forum_romanum: HG should be >= SAT dimension (upper bound)");
    }

    // ─── Realizer tests ─────────────────────────────────────────────────────

    fn verify_realizer<T: Clone>(poset: &Poset<T>, realizer: &[Vec<usize>]) {
        let n = poset.nodes.len();
        // Each extension must have n elements
        for le in realizer {
            assert_eq!(le.len(), n, "linear extension must have n elements");
        }
        // Each extension must respect the poset order
        for le in realizer {
            let pos: Vec<usize> = (0..n).map(|node| {
                le.iter().position(|&x| x == node).unwrap()
            }).collect();
            for &(u, v) in &poset.transitive_edges {
                assert!(pos[u as usize] < pos[v as usize],
                        "extension violates poset: {} < {} but positions are {}, {}",
                        u, v, pos[u as usize], pos[v as usize]);
            }
        }
        // Intersection property: if a < b in ALL extensions, then a ≤ b in the poset
        for a in 0..n {
            for b in 0..n {
                if a == b { continue; }
                if poset.is_leq(a as u32, b as u32) { continue; }
                // a and b are incomparable or b < a
                let a_before_b_in_all = realizer.iter().all(|le| {
                    let pa = le.iter().position(|&x| x == a).unwrap();
                    let pb = le.iter().position(|&x| x == b).unwrap();
                    pa < pb
                });
                assert!(!a_before_b_in_all,
                        "a={} < b={} in all extensions but not in poset", a, b);
            }
        }
    }

    #[test]
    fn test_sat_realizer_chain() {
        let p = chain_poset();
        let realizer = SatReduction.realizer(&p);
        assert_eq!(realizer.len(), 1);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_sat_realizer_antichain() {
        let p = antichain_poset();
        let realizer = SatReduction.realizer(&p);
        assert_eq!(realizer.len(), 2);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_sat_realizer_diamond() {
        let p = diamond_poset();
        let realizer = SatReduction.realizer(&p);
        assert_eq!(realizer.len(), 2);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_sat_realizer_s3() {
        let p = standard_example_s3();
        let realizer = SatReduction.realizer(&p);
        assert_eq!(realizer.len(), 3);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_gc_realizer_diamond() {
        let p = diamond_poset();
        let realizer = GraphColoring.realizer(&p);
        assert_eq!(realizer.len(), 2);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_gc_realizer_s3() {
        let p = standard_example_s3();
        let realizer = GraphColoring.realizer(&p);
        assert_eq!(realizer.len(), 3);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_gch_realizer_diamond() {
        let p = diamond_poset();
        let realizer = GraphColoringHeuristic.realizer(&p);
        // GCH is an upper bound, so len >= exact dimension
        assert!(realizer.len() >= 2);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_hg_realizer_diamond() {
        let p = diamond_poset();
        let realizer = HypergraphColoring.realizer(&p);
        assert_eq!(realizer.len(), 2);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_hg_realizer_s3() {
        let p = standard_example_s3();
        let realizer = HypergraphColoring.realizer(&p);
        assert_eq!(realizer.len(), 3);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_hybrid_realizer_s3() {
        let p = standard_example_s3();
        let realizer = Hybrid.realizer(&p);
        assert_eq!(realizer.len(), 3);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_hybrid_matches_sat_on_concept_lattices() {
        let contexts = [
            ("famous_animals", "famous_animals_en.cxt"),
            ("triangles", "triangles.cxt"),
            ("planets", "planets_en.cxt"),
        ];
        for (name, file) in &contexts {
            let poset = load_concept_lattice_poset(file);
            let sat_dim = SatReduction.dimension(&poset);
            let hybrid_dim = Hybrid.dimension(&poset);
            assert!(hybrid_dim >= sat_dim, "{}: Hybrid should be >= SAT", name);
        }
    }

    #[test]
    fn test_poset_realizer_convenience() {
        let p = diamond_poset();
        let realizer = p.realizer();
        assert_eq!(realizer.len(), 2);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_sat_realizer_fca_famous_animals() {
        let poset = load_concept_lattice_poset("famous_animals_en.cxt");
        let realizer = SatReduction.realizer(&poset);
        verify_realizer(&poset, &realizer);
    }

    #[test]
    fn test_sat_realizer_antichain_4() {
        // Antichain with 4 elements: dim = 2, but all pairs must split
        let p = Poset::from_covering_relation(
            vec![0u32, 1, 2, 3], vec![],
        ).unwrap();
        let realizer = SatReduction.realizer(&p);
        assert_eq!(realizer.len(), 2);
        verify_realizer(&p, &realizer);
    }

    #[test]
    fn test_standard_example_s4_realizer() {
        // S4: elements {0,1,2,3,4,5,6,7} with a_i < b_j iff i != j
        // a_i = i, b_j = j+4, covering edges: (a_i, b_j) for i != j
        let p = Poset::from_covering_relation(
            vec![0u32, 1, 2, 3, 4, 5, 6, 7],
            vec![(0,5),(0,6),(0,7),(1,4),(1,6),(1,7),(2,4),(2,5),(2,7),(3,4),(3,5),(3,6)],
        ).unwrap();
        let realizer = SatReduction.realizer(&p);
        assert_eq!(realizer.len(), 4);
        verify_realizer(&p, &realizer);
    }
}