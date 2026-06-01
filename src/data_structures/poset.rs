use bit_set::BitSet;
use std::collections::{HashSet, VecDeque};

/// Error returned when constructing a `Poset` from invalid input.
#[derive(Debug, PartialEq)]
pub enum PosetError {
    /// The edge set contains a directed cycle, so no valid partial order exists.
    Cycle,
}

/// A partially ordered set (poset).
///
/// Stores both the covering relation (order diagram edges) and the
/// full transitive relation. Exactly one is provided by the caller;
/// the other is computed at construction time.
#[derive(Debug)]
pub struct Poset<T> {
    /// The nodes of the poset, indexed by position (0-based).
    pub nodes: Vec<T>,
    /// The covering relation (order diagram / Hasse edges): (u, v) means u ≺ v.
    pub covering_edges: Vec<(u32, u32)>,
    /// All comparable pairs (u, v) with u < v (strict, reflexivity excluded).
    pub transitive_edges: Vec<(u32, u32)>,
}

impl<T: Clone> Poset<T> {
    /// Construct a poset from its covering relation (order diagram edges).
    ///
    /// Detects cycles via Kahn's algorithm; returns `Err(PosetError::Cycle)`
    /// if the edge set is not a DAG. On success, computes the transitive
    /// closure via BitSet DP in reverse topological order.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Poset;
    ///
    /// // Chain: 0 ≺ 1 ≺ 2
    /// let p = Poset::from_covering_relation(
    ///     vec!["a", "b", "c"],
    ///     vec![(0, 1), (1, 2)],
    /// ).unwrap();
    /// assert_eq!(p.nodes.len(), 3);
    /// assert!(p.is_leq(0, 2)); // a ≤ c via transitivity
    /// ```
    pub fn from_covering_relation(
        nodes: Vec<T>,
        edges: Vec<(u32, u32)>,
    ) -> Result<Self, PosetError> {
        let n = nodes.len();
        let topo = kahn_topo_sort(n, &edges).ok_or(PosetError::Cycle)?;
        let transitive_edges = transitive_closure(n, &edges, &topo);
        Ok(Poset {
            nodes,
            covering_edges: edges,
            transitive_edges,
        })
    }

    /// Construct a poset from a relation (not necessarily transitively closed).
    ///
    /// Computes the full transitive closure, then derives the covering relation
    /// via BitSet transitive reduction. Detects cycles via Kahn's algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Poset;
    ///
    /// let p = Poset::from_transitive_relation(
    ///     vec!["a", "b", "c"],
    ///     vec![(0, 1), (1, 2)],
    /// ).unwrap();
    /// assert!(p.is_leq(0, 2));
    /// assert_eq!(p.covering_edges.len(), 2);
    /// ```
    pub fn from_transitive_relation(
        nodes: Vec<T>,
        edges: Vec<(u32, u32)>,
    ) -> Result<Self, PosetError> {
        let n = nodes.len();
        let topo = kahn_topo_sort(n, &edges).ok_or(PosetError::Cycle)?;
        let transitive_edges = transitive_closure(n, &edges, &topo);
        let covering_edges = transitive_reduction(n, &transitive_edges, &{
            let mut reach: Vec<BitSet> = vec![BitSet::with_capacity(n); n];
            for &(u, v) in &transitive_edges {
                reach[u as usize].insert(v as usize);
            }
            reach
        });
        Ok(Poset {
            nodes,
            covering_edges,
            transitive_edges,
        })
    }

    /// Returns `true` if `a ≤ b` in this poset (a is below b or equal).
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Poset;
    ///
    /// let p = Poset::from_covering_relation(vec![0u32, 1, 2], vec![(0, 1), (1, 2)]).unwrap();
    /// assert!(p.is_leq(0, 0)); // reflexive
    /// assert!(p.is_leq(0, 2)); // transitive
    /// assert!(!p.is_leq(2, 0)); // not symmetric
    /// ```
    pub fn is_leq(&self, a: u32, b: u32) -> bool {
        if a == b {
            return true;
        }
        self.transitive_edges.contains(&(a, b))
    }

    /// Returns `true` if `a` is directly covered by `b` (a ≺ b, no element strictly between them).
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Poset;
    ///
    /// let p = Poset::from_covering_relation(vec![0u32, 1, 2], vec![(0, 1), (1, 2)]).unwrap();
    /// assert!(p.covers(0, 1));   // direct cover
    /// assert!(!p.covers(0, 2)); // not a direct cover (1 is between them)
    /// ```
    pub fn covers(&self, a: u32, b: u32) -> bool {
        self.covering_edges.contains(&(a, b))
    }

    /// Returns a linear extension of this poset (topological sort of covering edges).
    ///
    /// Returns `None` only if the covering edges contain a cycle (invariant:
    /// should not happen for a valid `Poset`).
    pub fn linear_extension(&self) -> Option<Vec<usize>> {
        kahn_topo_sort(self.nodes.len(), &self.covering_edges)
    }

    /// Static variant: computes a topological linear extension from a raw edge set.
    pub fn linear_extension_static(
        n: usize,
        edges: &HashSet<(usize, usize)>,
    ) -> Option<Vec<usize>> {
        let edges_u32: Vec<(u32, u32)> = edges.iter().map(|&(u, v)| (u as u32, v as u32)).collect();
        kahn_topo_sort(n, &edges_u32)
    }


    /// Returns the set of elements strictly above `a` (strict upset).
    ///
    /// The strict upset of `a` is { b | a < b }, i.e., all elements
    /// that are strictly greater than `a` in the partial order.
    pub fn strict_upset(&self, a: u32) -> HashSet<u32> {
        self.transitive_edges
            .iter()
            .filter(|&&(x, _)| x == a)
            .map(|&(_, y)| y)
            .collect()
    }

    /// Returns the set of elements strictly below `a` (strict downset).
    ///
    /// The strict downset of `a` is { b | b < a }, i.e., all elements
    /// that are strictly less than `a` in the partial order.
    pub fn strict_downset(&self, a: u32) -> HashSet<u32> {
        self.transitive_edges
            .iter()
            .filter(|&&(_, y)| y == a)
            .map(|&(x, _)| x)
            .collect()
    }

    /// Returns all incomparable pairs (a, b) with a < b (by index).
    ///
    /// Two elements are incomparable if neither a ≤ b nor b ≤ a.
    /// Only pairs with a < b (by index) are returned to avoid duplicates.
    pub fn incomparable_pairs(&self) -> Vec<(u32, u32)> {
        let n = self.nodes.len();
        let mut result = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                if !self.is_leq(a as u32, b as u32) && !self.is_leq(b as u32, a as u32) {
                    result.push((a as u32, b as u32));
                }
            }
        }
        result
    }

    /// Constructs the standard example Sₙ — a poset with dimension exactly n.
    ///
    /// Sₙ has 2n elements: n minimal elements a₀, …, aₙ₋₁ and n maximal elements
    /// b₀, …, bₙ₋₁, where aᵢ < bⱼ iff i ≠ j. This is the canonical poset
    /// with order dimension n (Trotter 1992).
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Poset;
    ///
    /// let s3 = Poset::<u32>::standard_example(3);
    /// assert_eq!(s3.nodes.len(), 6);
    /// // aᵢ < bⱼ for all i ≠ j
    /// assert!(s3.is_leq(0, 5)); // a₀ < b₂
    /// assert!(!s3.is_leq(0, 3)); // a₀ ≮ b₀
    /// ```
    pub fn standard_example(n: usize) -> Self
    where
        T: From<u32>,
    {
        assert!(n > 0, "standard example requires n >= 1");
        let nodes: Vec<T> = (0..(2 * n) as u32).map(T::from).collect();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    edges.push((i as u32, (n + j) as u32));
                }
            }
        }
        Self::from_covering_relation(nodes, edges).unwrap()
    }

    /// Constructs a chain of n elements: 0 < 1 < … < n-1. Dimension 1.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Poset;
    ///
    /// let c = Poset::<u32>::chain(3);
    /// assert_eq!(c.nodes.len(), 3);
    /// assert!(c.is_leq(0, 2)); // 0 < 2 via transitivity
    /// assert!(!c.is_leq(2, 0));
    /// ```
    pub fn chain(n: usize) -> Self
    where
        T: From<u32>,
    {
        assert!(n > 0, "chain requires n >= 1");
        let nodes: Vec<T> = (0..n as u32).map(T::from).collect();
        let edges: Vec<(u32, u32)> = (0..n - 1).map(|i| (i as u32, i as u32 + 1)).collect();
        Self::from_covering_relation(nodes, edges).unwrap()
    }

    /// Constructs an antichain of n elements (no two are comparable). Dimension 2 for n >= 2.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Poset;
    ///
    /// let a = Poset::<u32>::antichain(4);
    /// assert_eq!(a.nodes.len(), 4);
    /// assert!(!a.is_leq(0, 1)); // no element is comparable to another
    /// ```
    pub fn antichain(n: usize) -> Self
    where
        T: From<u32>,
    {
        assert!(n > 0, "antichain requires n >= 1");
        let nodes: Vec<T> = (0..n as u32).map(T::from).collect();
        Self::from_covering_relation(nodes, vec![]).unwrap()
    }

    /// Returns all critical pairs of this poset.
    ///
    /// A critical pair (a, b) is an incomparable pair where:
    /// - downset(a) ⊆ downset(b) (a's lower bounds are a subset of b's)
    /// - upset(b) ⊆ upset(a) (b's upper bounds are a subset of a's)
    ///
    /// Critical pairs are the minimal constraints that must be
    /// resolved by linear extensions in a realizer.
    pub fn critical_pairs(&self) -> Vec<(u32, u32)> {
        let incomparables = self.incomparable_pairs();
        let mut criticals = Vec::new();
        for &(a, b) in &incomparables {
            // Check both orientations: (a,b) and (b,a)
            let down_a = self.strict_downset(a);
            let down_b = self.strict_downset(b);
            let up_a = self.strict_upset(a);
            let up_b = self.strict_upset(b);

            if down_a.is_subset(&down_b) && up_b.is_subset(&up_a) {
                criticals.push((a, b));
            }
            if down_b.is_subset(&down_a) && up_a.is_subset(&up_b) {
                criticals.push((b, a));
            }
        }
        criticals
    }
}

// ─── Internal graph algorithms ───────────────────────────────────────────────

/// Kahn's topological sort. Returns `None` if the graph has a cycle.
/// Input: `n` nodes (0..n), directed edges `(u, v)` meaning u → v.
pub(crate) fn kahn_topo_sort(n: usize, edges: &[(u32, u32)]) -> Option<Vec<usize>> {
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];

    for &(u, v) in edges {
        let (u, v) = (u as usize, v as usize);
        adj[u].push(v);
        in_degree[v] += 1;
    }

    // Sort adjacency for determinism (important for consistent layout output)
    for neighbors in adj.iter_mut() {
        neighbors.sort();
    }

    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if order.len() == n { Some(order) } else { None }
}

/// Compute the transitive closure of a DAG via BitSet DP in reverse topological order.
/// Returns the set of all strict comparable pairs (u, v) with u < v (u strictly below v).
fn transitive_closure(n: usize, edges: &[(u32, u32)], topo: &[usize]) -> Vec<(u32, u32)> {
    // reach[u] = set of all nodes reachable from u (excluding u itself)
    let mut reach: Vec<BitSet> = vec![BitSet::with_capacity(n); n];

    // Build direct-successor sets
    let mut direct: Vec<BitSet> = vec![BitSet::with_capacity(n); n];
    for &(u, v) in edges {
        direct[u as usize].insert(v as usize);
    }

    // Process in reverse topological order so reach[v] is already complete
    // when we process u
    for &u in topo.iter().rev() {
        reach[u] = direct[u].clone();
        // Collect the direct successors first (avoid borrow conflict)
        let succs: Vec<usize> = direct[u].iter().collect();
        for v in succs {
            let reach_v = reach[v].clone();
            reach[u].union_with(&reach_v);
        }
    }

    let mut result = Vec::new();
    for (u, reach_u) in reach.iter().enumerate().take(n) {
        for v in reach_u.iter() {
            result.push((u as u32, v as u32));
        }
    }
    result
}

/// Compute the transitive reduction (covering relation) from a full transitive
/// relation represented as `reach[]` BitSets (node u → reach[u]).
/// Uses BitSet set-difference: a direct edge u→v exists iff v ∈ reach[u]
/// but v is NOT reachable from u via any intermediate w ∈ reach[u].
fn transitive_reduction(n: usize, edges: &[(u32, u32)], reach: &[BitSet]) -> Vec<(u32, u32)> {
    // Build adjacency from transitive edges
    let mut succ: Vec<BitSet> = vec![BitSet::with_capacity(n); n];
    for &(u, v) in edges {
        succ[u as usize].insert(v as usize);
    }

    let mut result = Vec::new();
    for (u, succ_u) in succ.iter().enumerate().take(n) {
        // reachable_via_intermediate = union of reach[w] for w ∈ succ[u]
        // A covering edge u→v exists iff v ∈ succ[u] and v ∉ reachable_via_intermediate
        let mut indirect = BitSet::with_capacity(n);
        for w in succ_u.iter() {
            indirect.union_with(&reach[w]);
        }
        // covering successors = succ[u] - indirect
        let mut covering = succ_u.clone();
        covering.difference_with(&indirect);
        for v in covering.iter() {
            result.push((u as u32, v as u32));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_covering_chain() {
        // a(0) ≤ b(1) ≤ c(2)
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2],
            vec![(0, 1), (1, 2)],
        )
        .unwrap();
        assert_eq!(p.covering_edges.len(), 2);
        // Transitive: (0,1), (1,2), (0,2)
        assert!(p.transitive_edges.contains(&(0, 1)));
        assert!(p.transitive_edges.contains(&(1, 2)));
        assert!(p.transitive_edges.contains(&(0, 2)));
        assert_eq!(p.transitive_edges.len(), 3);
    }

    #[test]
    fn test_from_covering_diamond() {
        // 0 ≤ 1, 0 ≤ 2, 1 ≤ 3, 2 ≤ 3
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2, 3],
            vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        )
        .unwrap();
        assert!(p.transitive_edges.contains(&(0, 3)));
        assert!(p.transitive_edges.contains(&(0, 1)));
        assert!(p.transitive_edges.contains(&(0, 2)));
        assert!(p.transitive_edges.contains(&(1, 3)));
        assert!(p.transitive_edges.contains(&(2, 3)));
    }

    #[test]
    fn test_from_transitive_diamond() {
        // All comparable pairs provided
        let transitive = vec![(0u32, 1u32), (0, 2), (1, 3), (2, 3), (0, 3)];
        let p = Poset::from_transitive_relation(vec![0usize, 1, 2, 3], transitive).unwrap();
        // Covering edges should be exactly the 4 direct cover pairs
        let mut cov = p.covering_edges.clone();
        cov.sort();
        assert_eq!(cov, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn test_cycle_detection() {
        let result = Poset::from_covering_relation(vec![0usize, 1], vec![(0, 1), (1, 0)]);
        assert_eq!(result.unwrap_err(), PosetError::Cycle);
    }

    #[test]
    fn test_is_leq() {
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2, 3],
            vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        )
        .unwrap();
        assert!(p.is_leq(0, 3));
        assert!(p.is_leq(0, 0)); // reflexivity
        assert!(!p.is_leq(1, 2)); // incomparable
        assert!(!p.is_leq(3, 0)); // reversed direction
    }

    #[test]
    fn test_covers() {
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2, 3],
            vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        )
        .unwrap();
        assert!(p.covers(0, 1));
        assert!(!p.covers(0, 3)); // not a direct cover
        assert!(!p.covers(1, 2)); // incomparable
    }

    #[test]
    fn test_linear_extension() {
        // Chain: 0 ≤ 1 ≤ 2
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2],
            vec![(0, 1), (1, 2)],
        )
        .unwrap();
        let le = p.linear_extension().unwrap();
        assert_eq!(le, vec![0, 1, 2]);
    }

    // ── Critical pairs tests (Inkrement 1) ───────────────────────────────────

    #[test]
    fn test_strict_upset_downset_chain() {
        // Chain: 0 < 1 < 2
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2],
            vec![(0, 1), (1, 2)],
        ).unwrap();
        assert_eq!(p.strict_upset(0), HashSet::from([1, 2]));
        assert_eq!(p.strict_upset(1), HashSet::from([2]));
        assert_eq!(p.strict_upset(2), HashSet::new());
        assert_eq!(p.strict_downset(2), HashSet::from([0, 1]));
        assert_eq!(p.strict_downset(1), HashSet::from([0]));
        assert_eq!(p.strict_downset(0), HashSet::new());
    }

    #[test]
    fn test_incomparable_pairs_chain() {
        // Chain: no incomparable pairs
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2],
            vec![(0, 1), (1, 2)],
        ).unwrap();
        assert_eq!(p.incomparable_pairs(), Vec::<(u32, u32)>::new());
    }

    #[test]
    fn test_incomparable_pairs_antichain() {
        // Antichain of 3 elements: all pairs are incomparable
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2],
            vec![],
        ).unwrap();
        let mut pairs = p.incomparable_pairs();
        pairs.sort();
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn test_incomparable_pairs_diamond() {
        // Diamond: 0 < 1, 0 < 2, 1 < 3, 2 < 3
        // 1 and 2 are incomparable
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2, 3],
            vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        ).unwrap();
        let mut pairs = p.incomparable_pairs();
        pairs.sort();
        assert_eq!(pairs, vec![(1, 2)]);
    }

    #[test]
    fn test_critical_pairs_diamond() {
        // Diamond: 0 < 1, 0 < 2, 1 < 3, 2 < 3
        // downset(1) = {0}, downset(2) = {0}, upset(1) = {3}, upset(2) = {3}
        // Both (1,2) and (2,1) are critical pairs
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2, 3],
            vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        ).unwrap();
        let mut pairs = p.critical_pairs();
        pairs.sort();
        assert_eq!(pairs, vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn test_standard_example_s2() {
        let s2 = Poset::<u32>::standard_example(2);
        assert_eq!(s2.nodes.len(), 4);
        // a₀=0, a₁=1, b₀=2, b₁=3
        // a₀ < b₁, a₁ < b₀
        assert!(s2.is_leq(0, 3));
        assert!(s2.is_leq(1, 2));
        assert!(!s2.is_leq(0, 2)); // a₀ ≮ b₀
        assert!(!s2.is_leq(1, 3)); // a₁ ≮ b₁
    }

    #[test]
    fn test_standard_example_s3() {
        let s3 = Poset::<u32>::standard_example(3);
        assert_eq!(s3.nodes.len(), 6);
        // a₀=0, a₁=1, a₂=2, b₀=3, b₁=4, b₂=5
        assert!(s3.is_leq(0, 4)); // a₀ < b₁
        assert!(s3.is_leq(0, 5)); // a₀ < b₂
        assert!(!s3.is_leq(0, 3)); // a₀ ≮ b₀
        assert!(!s3.is_leq(1, 4)); // a₁ ≮ b₁
    }

    #[test]
    fn test_standard_example_dimension() {
        use crate::DimensionAlgorithm;
        use crate::algorithms::dimension::SatReduction;
        // S_n has dimension n for n >= 2; S_1 is an antichain of 2 (dimension 2)
        for n in 2..=5 {
            let sn = Poset::<u32>::standard_example(n);
            assert_eq!(SatReduction.dimension(&sn), n, "S_{} should have dimension {}", n, n);
        }
    }

    #[test]
    fn test_chain() {
        let c = Poset::<u32>::chain(1);
        assert_eq!(c.nodes.len(), 1);

        let c = Poset::<u32>::chain(4);
        assert_eq!(c.nodes.len(), 4);
        assert!(c.is_leq(0, 3));
        assert!(!c.is_leq(3, 0));
        assert_eq!(c.covering_edges.len(), 3);
    }

    #[test]
    fn test_antichain() {
        let a = Poset::<u32>::antichain(1);
        assert_eq!(a.nodes.len(), 1);

        let a = Poset::<u32>::antichain(5);
        assert_eq!(a.nodes.len(), 5);
        assert!(!a.is_leq(0, 1));
        assert!(a.covering_edges.is_empty());
        assert!(a.transitive_edges.is_empty());
    }

    #[test]
    fn test_critical_pairs_n_poset() {
        // N-shaped poset (standard example for dimension 2):
        // 0 < 1, 2 < 3, 0 incomparable to 2 and 3, 1 incomparable to 2 and 3
        // Covering: 0 < 1, 2 < 3
        let p = Poset::from_covering_relation(
            vec![0usize, 1, 2, 3],
            vec![(0, 1), (2, 3)],
        ).unwrap();
        let mut pairs = p.critical_pairs();
        pairs.sort();
        // Critical pairs: (0,3), (2,1) — and their reverses if applicable
        // downset(0)={}, upset(3)={}, so (0,3) is critical (both conditions trivially hold)
        // downset(2)={}, upset(1)={}, so (2,1) is critical
        // But also (1,0) etc? Let's check:
        // (0,2): downset(0)={}, upset(2)={3}, downset(2)={}, upset(0)={1}
        //         down_a ⊆ down_b? {} ⊆ {} ✓; up_b ⊆ up_a? {3} ⊆ {1} ✗ → NOT critical
        // (0,3): down_a={} ⊆ down_b={} ✓; up_b={} ⊆ up_a={1} ✓ → critical
        // (1,2): down_a={0} ⊆ down_b={} ✗ → NOT critical
        // (2,1): down_a={} ⊆ down_b={0}? {} ⊆ {0} ✓; up_b={} ⊆ up_a={3}? {} ⊆ {3} ✓ → critical
        // (1,3): down_a={0} ⊆ down_b={}? ✗ → NOT critical
        // (3,1): down_a={0,2} ⊆ down_b={0}? ✗ → NOT critical
        assert!(pairs.contains(&(0, 3)));
        assert!(pairs.contains(&(2, 1)));
    }
}
