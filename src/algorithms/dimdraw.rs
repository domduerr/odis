//! Branch-and-bound DimDraw lattice drawing algorithm.
//!
//! Picks the two linear extensions of the order whose *agreements* are as few
//! as possible — an agreement being a pair the order leaves incomparable but
//! both extensions put the same way round — and reads the drawing off their
//! ranks. Fewer agreements mean fewer pairs forced into a needless vertical
//! relationship in the diagram.
//!
//! Three things carry the search.
//!
//! **The incumbent comes first.** Every pair of linear extensions is a
//! feasible solution, and swapping two adjacent incomparable elements in one
//! of them flips the cost by exactly one, which makes both the move filter and
//! the cost update O(1). Simulated annealing over that neighbourhood finds a
//! near-optimal bound in milliseconds, so the branch-and-bound prunes hard
//! from the start instead of working its way down from a weak heuristic.
//!
//! **The orders live in a flat bit matrix**, mutated in place and rewound from
//! an undo trail, with the cost carried along incrementally. There is no
//! memoization: nodes are cheap enough that it would not pay for itself, and
//! its absence keeps memory proportional to the search depth rather than to
//! the number of explored nodes.
//!
//! **Symmetries of the order are broken away.** Every automorphism carries a
//! solution to an equally good one, and so does swapping the roles of the two
//! extensions, so a plain search walks `2 |Aut(P)|` copies of each solution
//! class. A complete solution reads as one letter per incomparable pair
//! `(u, v)`, `u < v`: `0` if u comes first in the one extension and v in the
//! other, `1` the other way round, `2` if u comes first in both, `3` if v
//! does. The group acts on these words, and only the lexicographically
//! smallest word of each orbit is admitted: for every symmetry `g` the search
//! requires `w <= w∘g`. The orbit's minimum satisfies all of those at once, so
//! no solution class is lost — and requiring it for only *some* of the group
//! is equally safe, which is what makes the cap on collected automorphisms
//! harmless.
//!
//! The comparison runs incrementally as pairs get decided. Position `i` can
//! only be judged once both `i` and the position `g` draws it from are
//! settled, so the pairs are ordered with each orbit contiguous, which is what
//! makes those two moments coincide. Reordering costs nothing: since the
//! annealed incumbent is already optimal on every context measured, the search
//! is a pure optimality proof and every unpruned node has to be visited
//! whatever the order.
//!
//! # What that adds up to
//!
//! On the convex-ordinal scale — 37 concepts, 369 incomparable pairs, 16
//! automorphisms — the optimality proof takes 176 931 039 nodes and runs in
//! tens of seconds at a few megabytes. Symmetry breaking accounts for a factor
//! of 8.2 in nodes against a ceiling of 32.

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use crate::algorithms::search_budget::SearchBudget;
use crate::data_structures::{drawing::Drawing, lattice::Lattice, poset::Poset};
use crate::traits::DrawingAlgorithm;

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

/// Annealing restarts, and steps per restart relative to the node count.
const ANNEAL_RESTARTS: usize = 8;
const ANNEAL_STEPS_PER_NODE_SQUARED: u64 = 200;

/// Exact branch-and-bound lattice drawing algorithm.
///
/// The search is time-bounded by its [`SearchBudget`], which
/// [`DimDraw::default`] sets to
/// [`DEFAULT_SEARCH_BUDGET_MS`](crate::algorithms::DEFAULT_SEARCH_BUDGET_MS).
/// Running out of budget does not count as failure: the best valid solution
/// found so far is returned, so a truncated run still yields a usable drawing.
///
/// [`SearchBudget::Unbounded`] exhausts the search and so returns a proven
/// optimum, at a cost that climbs steeply with the size of the lattice.
///
/// # Examples
///
/// ```
/// use odis::algorithms::{DimDraw, SearchBudget};
///
/// let quick = DimDraw::default();
/// let patient = DimDraw { budget: SearchBudget::Milliseconds(5_000) };
/// let exact = DimDraw { budget: SearchBudget::Unbounded };
/// # let _ = (quick, patient, exact);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct DimDraw {
    /// How long the search may run.
    pub budget: SearchBudget,
}

#[allow(dead_code)] // fields only read in #[cfg(test)] code
pub(crate) struct SearchOutcome {
    pub(crate) drawing: Drawing,
    pub(crate) best_cost: usize,
    pub(crate) baseline_cost: usize,
    pub(crate) explored_nodes: usize,
    pub(crate) timed_out: bool,
}

/// Deterministic xorshift64* generator — the annealing must be reproducible.
struct Rng(u64);

impl Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Both orders as successor bit matrices in one allocation: bit `v` of row `u`
/// of matrix `m` is set exactly if `u < v` in extension `m`.
///
/// Word `j` of row `u` of matrix `m` lives at `m * stride + u * words + j`.
struct Solver {
    node_count: usize,
    words: usize,
    stride: usize,
    matrices: Vec<u64>,
    /// Row `u` has bit `v` set exactly if `v` is *not* above `u` in the
    /// original order — the pairs an agreement is counted for.
    not_orig: Vec<u64>,
    /// Scratch buffer for the successor mask an edge propagates.
    mask: Vec<u64>,
    cost: u32,
    best_cost: u32,
    best: Vec<u64>,
    /// Words overwritten since the current branch started, newest last.
    trail: Vec<(u32, u64)>,
    /// Letters of the solution word for every pair the search has passed.
    word: Vec<u8>,
    /// How much of `word` is filled in.
    known: usize,
    symmetries: Vec<Symmetry>,
    /// Per symmetry: how far its comparison has got, and whether it already
    /// came out strictly in this branch's favour and needs no more looking at.
    lex_at: Vec<u32>,
    lex_done: Vec<bool>,
    lex_trail: Vec<(u32, u32, bool)>,
    nodes_explored: usize,
    timed_out: bool,
    /// The deadline in milliseconds from the start, or `None` if unbounded.
    limit_ms: Option<u64>,
    #[cfg(not(target_arch = "wasm32"))]
    start_time: Instant,
    #[cfg(target_arch = "wasm32")]
    start_ms: f64,
}

impl Solver {
    #[inline(always)]
    fn bit(&self, matrix: usize, u: usize, v: usize) -> bool {
        self.matrices[matrix * self.stride + u * self.words + v / 64] >> (v % 64) & 1 == 1
    }

    #[inline(always)]
    fn has_timed_out(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.limit_ms
                .is_some_and(|ms| self.start_time.elapsed() >= Duration::from_millis(ms))
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.limit_ms
                .is_some_and(|ms| (now_ms() - self.start_ms) >= ms as f64)
        }
    }

    /// Adds `u < v` to one extension, propagating transitively and updating the
    /// cost. Returns `false` if the edge would close a cycle; the trail is left
    /// consistent either way, so the caller rewinds as usual.
    fn add_edge(&mut self, matrix: usize, u: usize, v: usize) -> bool {
        if self.bit(matrix, u, v) {
            return true;
        }
        if self.bit(matrix, v, u) {
            return false;
        }

        let words = self.words;
        let base = matrix * self.stride;
        let other = (1 - matrix) * self.stride;

        // Everything at or above v moves above every predecessor of u. v itself
        // is never a predecessor of u (that would be the cycle rejected above),
        // so the mask stays valid while the rows are rewritten.

        // Orders of up to 64 elements are the common case and worth their own
        // path: with one word per row the inner loop and all index arithmetic
        // collapse, which measures out at roughly half again the throughput.
        if words == 1 {
            let mask = self.matrices[base + v] | 1u64 << v;
            for r in 0..self.node_count {
                let old = self.matrices[base + r];
                if r != u && old >> u & 1 == 0 {
                    continue;
                }
                let new = old | mask;
                if new == old {
                    continue;
                }
                self.trail.push(((base + r) as u32, old));
                self.matrices[base + r] = new;
                let counted = self.matrices[other + r] & self.not_orig[r];
                self.cost += (new & counted).count_ones() - (old & counted).count_ones();
            }
            return true;
        }

        let v_row = base + v * words;
        self.mask.copy_from_slice(&self.matrices[v_row..v_row + words]);
        self.mask[v / 64] |= 1u64 << (v % 64);

        for r in 0..self.node_count {
            if r != u && !self.bit(matrix, r, u) {
                continue;
            }
            let (row, other_row, orig_row) = (base + r * words, other + r * words, r * words);
            for j in 0..words {
                let old = self.matrices[row + j];
                let new = old | self.mask[j];
                if new == old {
                    continue;
                }
                self.trail.push(((row + j) as u32, old));
                self.matrices[row + j] = new;
                // An agreement is a pair both extensions order the same way and
                // the original order leaves incomparable.
                let counted = self.matrices[other_row + j] & self.not_orig[orig_row + j];
                self.cost += (new & counted).count_ones() - (old & counted).count_ones();
            }
        }
        true
    }

    /// The letter of a pair both extensions have already ordered.
    #[inline]
    fn letter(&self, u: usize, v: usize) -> u8 {
        match (self.bit(0, u, v), self.bit(1, u, v)) {
            (true, false) => 0,
            (false, true) => 1,
            (true, true) => 2,
            (false, false) => 3,
        }
    }

    /// Advances every symmetry's comparison as far as the known prefix allows.
    /// Returns true once some symmetry carries this branch to a
    /// lexicographically smaller word, meaning its representative is reached
    /// elsewhere in the tree.
    fn outranked_by_a_symmetry(&mut self) -> bool {
        for g in 0..self.symmetries.len() {
            while !self.lex_done[g] {
                let at = self.lex_at[g] as usize;
                if at >= self.word.len() {
                    break; // the two words agree throughout
                }
                let source = self.symmetries[g].source[at] as usize;
                if at >= self.known || source >= self.known {
                    break; // one of the two sides is still undecided
                }
                let ours = self.word[at];
                let theirs = self.symmetries[g].rewrite[at][self.word[source] as usize];
                self.lex_trail.push((g as u32, self.lex_at[g], self.lex_done[g]));
                if ours < theirs {
                    self.lex_done[g] = true;
                } else if ours > theirs {
                    return true;
                } else {
                    self.lex_at[g] += 1;
                }
            }
        }
        false
    }

    fn rewind_symmetries(&mut self, mark: usize) {
        while self.lex_trail.len() > mark {
            let (g, at, done) = self.lex_trail.pop().unwrap();
            self.lex_at[g as usize] = at;
            self.lex_done[g as usize] = done;
        }
    }

    fn rewind(&mut self, mark: usize) {
        let Self { trail, matrices, .. } = self;
        // Newest first, so that a word written several times in this branch
        // ends up back at the value it had when the branch started.
        for &(index, previous) in trail[mark..].iter().rev() {
            matrices[index as usize] = previous;
        }
        trail.truncate(mark);
    }

    /// Branch over the still-undecided incomparable pairs.
    ///
    /// Swapping the two extensions is one of the symmetries, so unlike in
    /// [`crate::algorithms::DimDraw2`] it needs no special case here.
    fn search(&mut self, pair_index: usize, pairs: &[(usize, usize)]) {
        self.nodes_explored += 1;

        // Reading the clock costs about as much as visiting a node, so sample
        // it periodically; overshooting the budget by a fraction of a
        // millisecond does not matter.
        if self.nodes_explored & 1023 == 0 && self.has_timed_out() {
            self.timed_out = true;
            return;
        }
        if self.cost >= self.best_cost {
            return;
        }

        let mut index = pair_index;
        while index < pairs.len() {
            let (u, v) = pairs[index];
            let decided_in_both = (self.bit(0, u, v) || self.bit(0, v, u))
                && (self.bit(1, u, v) || self.bit(1, v, u));
            if decided_in_both {
                index += 1;
            } else {
                break;
            }
        }

        // Extend the known prefix of the solution word, then let every symmetry
        // compare as far as the prefix allows.
        let saved_known = self.known;
        let lex_mark = self.lex_trail.len();
        while self.known < index {
            let (a, b) = pairs[self.known];
            self.word[self.known] = self.letter(a, b);
            self.known += 1;
        }

        if !self.outranked_by_a_symmetry() {
            if index == pairs.len() {
                self.best_cost = self.cost;
                self.best.copy_from_slice(&self.matrices);
            } else {
                self.branch(index, pairs);
            }
        }

        self.rewind_symmetries(lex_mark);
        self.known = saved_known;
    }

    fn branch(&mut self, index: usize, pairs: &[(usize, usize)]) {
        let (u, v) = pairs[index];
        // Opposing orientations first (they cost nothing), then the two
        // agreements. This ordering drives the incumbent down quickly.
        let choices = [(u, v, v, u), (v, u, u, v), (u, v, u, v), (v, u, v, u)];

        for (choice, &(u1, v1, u2, v2)) in choices.iter().enumerate() {
            // The last two choices make u and v agree, so any completion below
            // them costs at least one more than this node. Once that reaches
            // the incumbent the child would be cut on entry, so skip straight
            // past the propagation.
            if choice >= 2 && self.cost + 1 >= self.best_cost {
                break;
            }
            let (mark, saved_cost) = (self.trail.len(), self.cost);
            if self.add_edge(0, u1, v1) && self.add_edge(1, u2, v2) {
                self.search(index + 1, pairs);
            }
            self.rewind(mark);
            self.cost = saved_cost;

            if self.timed_out {
                break; // the caller still has to rewind its symmetry state
            }
        }
    }
}

/// Transitive closure of the covering relation as a bit matrix, or `None` if
/// the covering edges are cyclic.
fn transitive_closure(node_count: usize, words: usize, edges: &[(u32, u32)]) -> Option<Vec<u64>> {
    let mut leq = vec![false; node_count * node_count];
    for &(u, v) in edges {
        let (u, v) = (u as usize, v as usize);
        if u >= node_count || v >= node_count {
            return None;
        }
        leq[u * node_count + v] = true;
    }
    for k in 0..node_count {
        for i in 0..node_count {
            if leq[i * node_count + k] {
                for j in 0..node_count {
                    if leq[k * node_count + j] {
                        leq[i * node_count + j] = true;
                    }
                }
            }
        }
    }
    if (0..node_count).any(|i| leq[i * node_count + i]) {
        return None;
    }

    let mut closure = vec![0u64; node_count * words];
    for i in 0..node_count {
        for j in 0..node_count {
            if leq[i * node_count + j] {
                closure[i * words + j / 64] |= 1u64 << (j % 64);
            }
        }
    }
    Some(closure)
}

/// Incomparable pairs, most constrained first.
fn incomparable_pairs(node_count: usize, words: usize, closure: &[u64]) -> Vec<(usize, usize)> {
    let above = |u: usize, v: usize| closure[u * words + v / 64] >> (v % 64) & 1 == 1;

    let mut degrees = vec![0u32; node_count];
    let mut pairs = Vec::new();
    for u in 0..node_count {
        for v in (u + 1)..node_count {
            if !above(u, v) && !above(v, u) {
                pairs.push((u, v));
                degrees[u] += 1;
                degrees[v] += 1;
            }
        }
    }
    pairs.sort_by_key(|&(u, v)| std::cmp::Reverse(degrees[u] as u64 * degrees[v] as u64));
    pairs
}

/// A uniformly drawn linear extension of the order.
fn random_linear_extension(
    node_count: usize,
    words: usize,
    closure: &[u64],
    rng: &mut Rng,
) -> Vec<usize> {
    let above = |u: usize, v: usize| closure[u * words + v / 64] >> (v % 64) & 1 == 1;

    let mut remaining = vec![0usize; node_count];
    for u in 0..node_count {
        for (v, count) in remaining.iter_mut().enumerate() {
            if above(u, v) {
                *count += 1;
            }
        }
    }

    let mut ready: Vec<usize> = (0..node_count).filter(|&x| remaining[x] == 0).collect();
    let mut order = Vec::with_capacity(node_count);
    while !ready.is_empty() {
        let picked = ready.swap_remove(rng.below(ready.len()));
        order.push(picked);
        for (v, count) in remaining.iter_mut().enumerate() {
            if above(picked, v) {
                *count -= 1;
                if *count == 0 {
                    ready.push(v);
                }
            }
        }
    }
    order
}

/// Simulated annealing over pairs of linear extensions.
///
/// Returns the two orders realizing the best cost found. A swap of adjacent
/// incomparable elements changes the cost by exactly one, which makes both the
/// move filter and the cost update O(1).
fn anneal(
    node_count: usize,
    words: usize,
    closure: &[u64],
    pairs: &[(usize, usize)],
    deadline: &dyn Fn() -> bool,
) -> (Vec<usize>, Vec<usize>, u32) {
    let above = |u: usize, v: usize| closure[u * words + v / 64] >> (v % 64) & 1 == 1;
    let comparable = |u: usize, v: usize| above(u, v) || above(v, u);

    let cost_of = |p1: &[usize], p2: &[usize]| -> u32 {
        pairs
            .iter()
            .filter(|&&(u, v)| (p1[u] < p1[v]) == (p2[u] < p2[v]))
            .count() as u32
    };

    let steps_per_restart = ANNEAL_STEPS_PER_NODE_SQUARED * (node_count * node_count) as u64;
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut best = (Vec::new(), Vec::new(), u32::MAX);

    for _ in 0..ANNEAL_RESTARTS {
        let mut orders = [
            random_linear_extension(node_count, words, closure, &mut rng),
            random_linear_extension(node_count, words, closure, &mut rng),
        ];
        let mut positions = [vec![0usize; node_count], vec![0usize; node_count]];
        for side in 0..2 {
            for (i, &x) in orders[side].iter().enumerate() {
                positions[side][x] = i;
            }
        }
        let mut cost = cost_of(&positions[0], &positions[1]);
        if cost < best.2 {
            best = (orders[0].clone(), orders[1].clone(), cost);
        }

        for step in 0..steps_per_restart {
            if step % 4096 == 0 && deadline() {
                break;
            }

            let side = (rng.next_u64() & 1) as usize;
            let at = rng.below(node_count - 1);
            let (u, v) = (orders[side][at], orders[side][at + 1]);
            if comparable(u, v) {
                continue;
            }

            // Swapping flips exactly this pair's agreement.
            let other = &positions[1 - side];
            let agreed = (positions[side][u] < positions[side][v]) == (other[u] < other[v]);
            if !agreed {
                // Uphill: accept on a geometrically cooling Metropolis schedule.
                let progress = step as f64 / steps_per_restart as f64;
                let temperature = 1.2 * (0.0005f64 / 1.2).powf(progress);
                let draw = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                if draw >= (-1.0 / temperature).exp() {
                    continue;
                }
            }

            orders[side].swap(at, at + 1);
            positions[side][u] = at + 1;
            positions[side][v] = at;
            cost = if agreed { cost - 1 } else { cost + 1 };

            if cost < best.2 {
                best = (orders[0].clone(), orders[1].clone(), cost);
            }
        }

        if deadline() {
            break;
        }
    }

    best
}

/// Turns a linear order into a successor bit matrix row block.
fn linear_order_matrix(order: &[usize], node_count: usize, words: usize) -> Vec<u64> {
    let mut matrix = vec![0u64; node_count * words];
    for (i, &u) in order.iter().enumerate() {
        for &v in &order[i + 1..] {
            matrix[u * words + v / 64] |= 1u64 << (v % 64);
        }
    }
    matrix
}

/// Height of each node: how many nodes sit above it in the extension.
fn positions_from_matrix(matrix: &[u64], node_count: usize, words: usize) -> Vec<f64> {
    (0..node_count)
        .map(|u| {
            let above: u32 = matrix[u * words..(u + 1) * words]
                .iter()
                .map(|word| word.count_ones())
                .sum();
            (node_count.saturating_sub(1 + above as usize)) as f64
        })
        .collect()
}

impl DimDraw {
    /// Core solver operating directly on a `Poset`, so that iceberg posets
    /// (which may lack a unique top or bottom) can be drawn too.
    pub(crate) fn solve_from_poset<T>(&self, poset: &Poset<T>) -> Option<SearchOutcome> {
        self.solve_with_incumbent(poset, true)
    }

    /// Runs the search without seeding it from the annealing, so that it has
    /// to find the optimum on its own. Symmetry breaking that cut a solution
    /// class away shows up only this way: with the optimal incumbent already
    /// in hand, an over-eager cut simply leaves it standing.
    #[cfg(test)]
    pub(crate) fn solve_blind<T>(&self, poset: &Poset<T>) -> Option<SearchOutcome> {
        self.solve_with_incumbent(poset, false)
    }

    fn solve_with_incumbent<T>(
        &self,
        poset: &Poset<T>,
        seed_from_annealing: bool,
    ) -> Option<SearchOutcome> {
        let node_count = poset.nodes.len();
        if node_count == 0 {
            return None;
        }
        if node_count == 1 {
            return Some(SearchOutcome {
                drawing: Drawing::new(vec![(0.0, 0.0)]),
                best_cost: 0,
                baseline_cost: 0,
                explored_nodes: 1,
                timed_out: false,
            });
        }

        let words = node_count.div_ceil(64);
        let closure = transitive_closure(node_count, words, &poset.covering_edges)?;
        let mut pairs = incomparable_pairs(node_count, words, &closure);
        let maps = automorphisms(node_count, words, &closure);
        if !maps.is_empty() {
            group_pairs_by_orbit(&mut pairs, node_count, &maps);
        }
        let symmetries = compile_symmetries(&pairs, node_count, &maps);

        #[cfg(not(target_arch = "wasm32"))]
        let start_time = Instant::now();
        #[cfg(target_arch = "wasm32")]
        let start_ms = now_ms();

        let limit_ms = self.budget.limit_ms();
        let expired = move || {
            #[cfg(not(target_arch = "wasm32"))]
            {
                limit_ms.is_some_and(|ms| start_time.elapsed() >= Duration::from_millis(ms))
            }
            #[cfg(target_arch = "wasm32")]
            {
                limit_ms.is_some_and(|ms| (now_ms() - start_ms) >= ms as f64)
            }
        };

        let (order1, order2, baseline_cost) =
            anneal(node_count, words, &closure, &pairs, &expired);

        let stride = node_count * words;
        let mut incumbent = linear_order_matrix(&order1, node_count, words);
        incumbent.extend_from_slice(&linear_order_matrix(&order2, node_count, words));

        // The original order sits in both extensions at the root; the pairs it
        // leaves incomparable are the ones an agreement is charged for.
        let mut matrices = Vec::with_capacity(2 * stride);
        matrices.extend_from_slice(&closure);
        matrices.extend_from_slice(&closure);
        let mut not_orig = vec![0u64; stride];
        for u in 0..node_count {
            for j in 0..words {
                let in_range = if (j + 1) * 64 <= node_count {
                    u64::MAX
                } else {
                    (1u64 << (node_count % 64)) - 1
                };
                not_orig[u * words + j] = in_range & !closure[u * words + j];
            }
        }

        let mut solver = Solver {
            node_count,
            words,
            stride,
            matrices,
            not_orig,
            mask: vec![0; words],
            word: vec![0u8; pairs.len()],
            known: 0,
            lex_at: vec![0u32; symmetries.len()],
            lex_done: vec![false; symmetries.len()],
            lex_trail: Vec::new(),
            symmetries,
            cost: 0,
            best_cost: if seed_from_annealing {
                baseline_cost
            } else {
                pairs.len() as u32 + 1
            },
            best: incumbent,
            trail: Vec::new(),
            nodes_explored: 0,
            timed_out: false,
            limit_ms,
            #[cfg(not(target_arch = "wasm32"))]
            start_time,
            #[cfg(target_arch = "wasm32")]
            start_ms,
        };
        solver.search(0, &pairs);

        let p1 = positions_from_matrix(&solver.best[..stride], node_count, words);
        let p2 = positions_from_matrix(&solver.best[stride..], node_count, words);
        let coordinates: Vec<(f64, f64)> = (0..node_count)
            .map(|i| (p1[i] - p2[i], -(p1[i] + p2[i])))
            .collect();
        if !coordinates.iter().all(|(x, y)| x.is_finite() && y.is_finite()) {
            return None;
        }

        Some(SearchOutcome {
            drawing: Drawing::new(coordinates),
            best_cost: solver.best_cost as usize,
            baseline_cost: baseline_cost as usize,
            explored_nodes: solver.nodes_explored,
            timed_out: solver.timed_out,
        })
    }

    pub(crate) fn solve_with_stats<T>(&self, lattice: &Lattice<T>) -> Option<SearchOutcome> {
        self.solve_from_poset(&lattice.poset)
    }
}

impl DrawingAlgorithm for DimDraw {
    fn draw<T>(&self, lattice: &Lattice<T>) -> Option<Drawing> {
        self.solve_with_stats(lattice).map(|outcome| outcome.drawing)
    }

    fn draw_poset<T: Clone>(&self, poset: &Poset<T>) -> Option<Drawing> {
        self.solve_from_poset(poset).map(|outcome| outcome.drawing)
    }
}

// ---------------------------------------------------------------------------
// Symmetry breaking
// ---------------------------------------------------------------------------

/// At most this many automorphisms are collected. An antichain has n! of them,
/// so the enumeration always needs a ceiling; using only some of the group is
/// still sound, it just breaks less symmetry.
const AUTOMORPHISM_CAP: usize = 32;

/// A complete solution reads as one letter per incomparable pair `(u, v)`,
/// `u < v`: 0 = u first in the one extension and v first in the other, 1 the
/// other way round, 2 = u first in both, 3 = v first in both.
///
/// Relabelling the pair swaps the roles of u and v, mirroring the two
/// extensions swaps which one is "the one".
const RELABEL: [u8; 4] = [1, 0, 3, 2];
const MIRROR: [u8; 4] = [1, 0, 2, 3];

/// One symmetry, compiled into what the lexicographic test needs: for every
/// pair position, which position of the original word feeds it and how that
/// letter is rewritten.
struct Symmetry {
    source: Vec<u32>,
    rewrite: Vec<[u8; 4]>,
}

fn reaches(closure: &[u64], words: usize, u: usize, v: usize) -> bool {
    closure[u * words + v / 64] >> (v % 64) & 1 == 1
}

/// Backtracking search for order automorphisms, matching elements only where
/// the number of elements above and below them agrees.
struct AutomorphismSearch<'a> {
    node_count: usize,
    words: usize,
    closure: &'a [u64],
    up: Vec<u32>,
    down: Vec<u32>,
    map: Vec<usize>,
    used: Vec<bool>,
    found: Vec<Vec<usize>>,
}

impl AutomorphismSearch<'_> {
    fn extend(&mut self, at: usize) {
        if self.found.len() >= AUTOMORPHISM_CAP {
            return;
        }
        if at == self.node_count {
            if (0..self.node_count).any(|i| self.map[i] != i) {
                self.found.push(self.map.clone());
            }
            return;
        }
        for candidate in 0..self.node_count {
            if self.used[candidate]
                || self.up[candidate] != self.up[at]
                || self.down[candidate] != self.down[at]
            {
                continue;
            }
            let consistent = (0..at).all(|earlier| {
                let image = self.map[earlier];
                reaches(self.closure, self.words, earlier, at)
                    == reaches(self.closure, self.words, image, candidate)
                    && reaches(self.closure, self.words, at, earlier)
                        == reaches(self.closure, self.words, candidate, image)
            });
            if !consistent {
                continue;
            }
            self.map[at] = candidate;
            self.used[candidate] = true;
            self.extend(at + 1);
            self.used[candidate] = false;
        }
    }
}

/// Non-identity automorphisms of the order, up to [`AUTOMORPHISM_CAP`].
fn automorphisms(node_count: usize, words: usize, closure: &[u64]) -> Vec<Vec<usize>> {
    let mut search = AutomorphismSearch {
        node_count,
        words,
        closure,
        up: (0..node_count)
            .map(|u| (0..node_count).filter(|&v| reaches(closure, words, u, v)).count() as u32)
            .collect(),
        down: (0..node_count)
            .map(|u| (0..node_count).filter(|&v| reaches(closure, words, v, u)).count() as u32)
            .collect(),
        map: vec![usize::MAX; node_count],
        used: vec![false; node_count],
        found: Vec::new(),
    };
    search.extend(0);
    search.found
}

/// Reorders the pairs so that each orbit under the automorphisms is
/// contiguous, keeping the incoming order inside an orbit.
///
/// The lexicographic test compares position `i` against the position an
/// automorphism sends it to, and can only do so once both are decided. Keeping
/// orbits together is what makes those two moments coincide; the branching
/// order is otherwise free, because for a pure optimality proof every unpruned
/// node has to be visited anyway.
fn group_pairs_by_orbit(
    pairs: &mut Vec<(usize, usize)>,
    node_count: usize,
    maps: &[Vec<usize>],
) {
    let mut index_of = vec![usize::MAX; node_count * node_count];
    for (i, &(u, v)) in pairs.iter().enumerate() {
        index_of[u * node_count + v] = i;
    }

    let mut orbit = vec![usize::MAX; pairs.len()];
    let mut next_orbit = 0;
    for start in 0..pairs.len() {
        if orbit[start] != usize::MAX {
            continue;
        }
        let mut queue = vec![start];
        orbit[start] = next_orbit;
        while let Some(i) = queue.pop() {
            let (u, v) = pairs[i];
            for map in maps {
                let (a, b) = (map[u], map[v]);
                let j = index_of[a.min(b) * node_count + a.max(b)];
                if orbit[j] == usize::MAX {
                    orbit[j] = next_orbit;
                    queue.push(j);
                }
            }
        }
        next_orbit += 1;
    }

    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&i| (orbit[i], i));
    *pairs = order.into_iter().map(|i| pairs[i]).collect();
}

/// Compiles every automorphism, with and without mirroring the two extensions,
/// into a [`Symmetry`]. The identity without mirroring is left out; it would
/// only demand that the word not exceed itself.
fn compile_symmetries(
    pairs: &[(usize, usize)],
    node_count: usize,
    maps: &[Vec<usize>],
) -> Vec<Symmetry> {
    let mut index_of = vec![usize::MAX; node_count * node_count];
    for (i, &(u, v)) in pairs.iter().enumerate() {
        index_of[u * node_count + v] = i;
    }

    let identity: Vec<usize> = (0..node_count).collect();
    let mut out = Vec::new();
    for map in std::iter::once(&identity).chain(maps) {
        let relabels_identically = map.iter().enumerate().all(|(i, &m)| i == m);
        for mirrored in [false, true] {
            if relabels_identically && !mirrored {
                continue;
            }
            let mut source = vec![0u32; pairs.len()];
            let mut rewrite = vec![[0u8, 1, 2, 3]; pairs.len()];
            for (i, &(u, v)) in pairs.iter().enumerate() {
                let (a, b) = (map[u], map[v]);
                let target = index_of[a.min(b) * node_count + a.max(b)];
                source[target] = i as u32;
                let mut table = [0u8, 1, 2, 3];
                if a > b {
                    table = table.map(|l| RELABEL[l as usize]);
                }
                if mirrored {
                    table = table.map(|l| MIRROR[l as usize]);
                }
                rewrite[target] = table;
            }
            out.push(Symmetry { source, rewrite });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{Duration, Instant};

    use super::DimDraw;
    use crate::algorithms::SearchBudget;
    use crate::data_structures::lattice::Lattice;
    use crate::traits::DrawingAlgorithm;
    use crate::FormalContext;

    type ConceptLattice = Lattice<(bit_set::BitSet, bit_set::BitSet)>;

    fn lattice(name: &str) -> ConceptLattice {
        let bytes = fs::read(format!("test_data/{name}.cxt")).unwrap();
        FormalContext::<String>::from(&bytes)
            .unwrap()
            .concept_lattice()
            .expect("concept_lattice returned None")
    }

    /// Proven optima. Five of these orders have a non-trivial automorphism
    /// group — `nominal` has 120 of them, `b3` and `fm3` six — so they do
    /// exercise the symmetry breaking.
    const KNOWN_OPTIMA: [(&str, usize); 7] = [
        ("b3", 1),
        ("nominal", 0),
        ("triangles", 1),
        ("data_from_paper", 3),
        ("eu", 5),
        ("living_beings_and_water", 5),
        ("fm3", 16),
    ];

    #[test]
    fn test_dimdraw_reaches_the_known_optima() {
        for (name, optimum) in KNOWN_OPTIMA {
            let outcome = DimDraw { budget: SearchBudget::Unbounded }
                .solve_with_stats(&lattice(name))
                .expect("DimDraw should return an outcome");
            assert_eq!(outcome.best_cost, optimum, "{name}");
        }
    }

    /// The test that can catch an over-eager symmetry cut: without the annealed
    /// incumbent the search has to reach the optimum by itself, so a lost
    /// solution class raises the reported cost. With the optimum already in
    /// hand an unsound cut would simply leave it standing, unnoticed.
    #[test]
    fn test_dimdraw_reaches_the_optima_without_an_incumbent() {
        for (name, optimum) in KNOWN_OPTIMA {
            let outcome = DimDraw { budget: SearchBudget::Unbounded }
                .solve_blind(&lattice(name).poset)
                .expect("DimDraw should return an outcome");
            assert_eq!(outcome.best_cost, optimum, "{name}, searched blind");
        }
    }

    /// Every collected map must really be an automorphism of the order.
    #[test]
    fn test_collected_maps_preserve_the_order() {
        for (name, _) in KNOWN_OPTIMA {
            let lattice = lattice(name);
            let n = lattice.poset.nodes.len();
            let words = n.div_ceil(64);
            let closure = super::transitive_closure(n, words, &lattice.poset.covering_edges)
                .expect("covering edges should be acyclic");
            let reaches = |u: usize, v: usize| closure[u * words + v / 64] >> (v % 64) & 1 == 1;

            for map in super::automorphisms(n, words, &closure) {
                let mut seen = vec![false; n];
                for &image in &map {
                    assert!(!seen[image], "{name}: map is not a bijection");
                    seen[image] = true;
                }
                for u in 0..n {
                    for v in 0..n {
                        assert_eq!(
                            reaches(u, v),
                            reaches(map[u], map[v]),
                            "{name}: map does not preserve the order at ({u}, {v})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_dimdraw_result_is_never_worse_than_the_incumbent() {
        for (name, _) in KNOWN_OPTIMA {
            let outcome = DimDraw { budget: SearchBudget::Unbounded }
                .solve_with_stats(&lattice(name))
                .expect("DimDraw should return an outcome");
            assert!(
                outcome.best_cost <= outcome.baseline_cost,
                "{name}: result {} worse than incumbent {}",
                outcome.best_cost,
                outcome.baseline_cost
            );
        }
    }

    #[test]
    fn test_dimdraw_timeout_returns_coordinate_per_node() {
        let lattice = lattice("living_beings_and_water");
        let outcome = DimDraw { budget: SearchBudget::Milliseconds(1) }
            .solve_with_stats(&lattice)
            .expect("DimDraw should return an outcome");

        assert_eq!(outcome.drawing.coordinates.len(), lattice.poset.nodes.len());
        assert!(outcome
            .drawing
            .coordinates
            .iter()
            .all(|(x, y)| x.is_finite() && y.is_finite()));
    }

    #[test]
    fn test_dimdraw_larger_budget_explores_at_least_as_much() {
        let lattice = lattice("fm3");
        let short = DimDraw { budget: SearchBudget::Milliseconds(1) }
            .solve_with_stats(&lattice)
            .expect("short run should produce outcome");
        let long = DimDraw { budget: SearchBudget::Milliseconds(200) }
            .solve_with_stats(&lattice)
            .expect("long run should produce outcome");

        assert!(
            long.explored_nodes >= short.explored_nodes,
            "expected longer budget to explore at least as much (short={}, long={})",
            short.explored_nodes,
            long.explored_nodes
        );
    }

    #[test]
    fn test_dimdraw_draw_returns_some_on_valid_lattice() {
        assert!(DimDraw { budget: SearchBudget::Milliseconds(10) }
            .draw(&lattice("living_beings_and_water"))
            .is_some());
    }

    /// The hard instance: 37 concepts, 369 incomparable pairs, 16 automorphisms.
    #[test]
    #[ignore = "runs for tens of seconds; run manually with --ignored --nocapture"]
    fn proof_dimdraw_convex_ordinal_scale_is_exact() {
        let lattice = lattice("convex_ordinal_scale");
        let started = Instant::now();
        let outcome = DimDraw { budget: SearchBudget::Unbounded }
            .solve_with_stats(&lattice)
            .expect("DimDraw should return an outcome");
        let elapsed = started.elapsed();

        eprintln!(
            "convex_ordinal_scale: elapsed={:?}, nodes={}, incumbent={}, optimum={}",
            elapsed, outcome.explored_nodes, outcome.baseline_cost, outcome.best_cost
        );
        assert!(!outcome.timed_out, "search did not exhaust its tree");
        assert_eq!(outcome.best_cost, 51);
        assert!(elapsed <= Duration::from_secs(600), "unexpectedly slow: {elapsed:?}");
    }
}
