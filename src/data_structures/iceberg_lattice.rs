use bit_set::BitSet;

use crate::data_structures::poset::Poset;

/// An iceberg concept lattice: a concept lattice restricted to concepts whose
/// extent size (support) meets a minimum threshold.
///
/// Nodes are ordered by subset-inclusion of extents: `i < j` if `extent(i) ⊊ extent(j)`.
pub struct IcebergLattice {
    /// The Diagram of the iceberg concepts.
    /// Each node value is `(extent, intent)` as a pair of `BitSet`s.
    pub poset: Poset<(BitSet, BitSet)>,
    /// Absolute support count for each node (parallel to `poset.nodes`).
    pub support: Vec<u32>,
    /// Total number of objects in the underlying formal context.
    pub total_objects: u32,
}

impl IcebergLattice {
    /// Construct an `IcebergLattice` from its components.
    ///
    /// # Panics (debug)
    /// Panics in debug builds if `support.len() != poset.nodes.len()`.
    pub fn new(poset: Poset<(BitSet, BitSet)>, support: Vec<u32>, total_objects: u32) -> Self {
        debug_assert_eq!(support.len(), poset.nodes.len());
        IcebergLattice { poset, support, total_objects }
    }

    /// Construct an empty `IcebergLattice` (no concepts).
    pub fn empty(total_objects: u32) -> Self {
        let poset = Poset::from_covering_relation(vec![], vec![])
            .expect("empty poset is always valid");
        IcebergLattice { poset, support: Vec::new(), total_objects }
    }

    /// Relative support (fraction of objects) for a node, in `[0.0, 1.0]`.
    /// Returns `0.0` if `total_objects == 0`.
    pub fn relative_support(&self, node_idx: usize) -> f64 {
        if self.total_objects == 0 {
            return 0.0;
        }
        self.support[node_idx] as f64 / self.total_objects as f64
    }
}
