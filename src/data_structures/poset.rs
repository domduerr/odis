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
    /// The covering relation (order diagram): (u, v) means u ≺ v.
    pub covering_edges: Vec<(u32, u32)>,
    /// All comparable pairs (u, v) with u < v (strict, reflexivity excluded).
    pub transitive_edges: Vec<(u32, u32)>,
    /// The same relation as a bit matrix: bit `v` of row `u` is set exactly if
    /// `u < v`. `transitive_edges` is the readable form of this and is what
    /// callers iterate; this is what the order queries index into, so that
    /// [`Poset::is_leq`] is a bit test rather than a scan of every comparable
    /// pair in the order.
    pub(crate) reach: Vec<BitSet>,
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
        let reach = reachability(n, &edges, &topo);
        let transitive_edges = pairs_of(&reach);
        Ok(Poset {
            nodes,
            covering_edges: edges,
            transitive_edges,
            reach,
        })
    }

    /// Construct a poset from its full transitive relation.
    ///
    /// Detects cycles (irreflexive strict order must be a DAG); computes the
    /// covering relation via BitSet transitive reduction.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Poset;
    ///
    /// // Same chain as above, but given as full transitive closure
    /// let p = Poset::from_transitive_relation(
    ///     vec!["a", "b", "c"],
    ///     vec![(0, 1), (1, 2), (0, 2)], // all comparable pairs
    /// ).unwrap();
    /// // covering relation should reduce to (0,1) and (1,2)
    /// assert_eq!(p.covering_edges.len(), 2);
    /// ```
    pub fn from_transitive_relation(
        nodes: Vec<T>,
        edges: Vec<(u32, u32)>,
    ) -> Result<Self, PosetError> {
        let n = nodes.len();
        // Validate: topo-sort on the edge set must succeed
        let _ = kahn_topo_sort(n, &edges).ok_or(PosetError::Cycle)?;
        // Build reach[] BitSets from the provided transitive edges
        let mut reach: Vec<BitSet> = vec![BitSet::with_capacity(n); n];
        for &(u, v) in &edges {
            reach[u as usize].insert(v as usize);
        }
        let covering_edges = transitive_reduction(n, &edges, &reach);
        Ok(Poset {
            nodes,
            covering_edges,
            transitive_edges: edges,
            reach,
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
        self.reach
            .get(a as usize)
            .is_some_and(|above| above.contains(b as usize))
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
fn reachability(n: usize, edges: &[(u32, u32)], topo: &[usize]) -> Vec<BitSet> {
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

    reach
}

/// Flattens a reachability matrix into the pair list callers iterate over.
fn pairs_of(reach: &[BitSet]) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    for (u, above) in reach.iter().enumerate() {
        for v in above.iter() {
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
}
