//! Trait for algorithms that compute the order dimension of a poset.

use crate::data_structures::poset::Poset;

/// Trait for algorithms that compute the order dimension of a poset.
///
/// The order dimension of a poset is the minimum number of linear extensions
/// whose intersection equals the original partial order.
///
/// # Implementors
///
/// - [`crate::algorithms::dimension::SatReduction`] — SAT reduction (Heitz 2022)
/// - [`crate::algorithms::dimension::HypergraphColoring`] — Hypergraph coloring (Trotter 1992)
/// - [`crate::algorithms::dimension::GraphColoring`] — Graph coloring with iterative extension (Yáñez & Montero 1999)
/// - [`crate::algorithms::dimension::GraphColoringHeuristic`] — DSatur heuristic (Yáñez & Montero 1999, Section 6)
/// - [`crate::algorithms::dimension::Hybrid`] — Combined graph and hypergraph coloring
///
/// # Examples
///
/// ```ignore
/// use odis::{Poset, DimensionAlgorithm};
/// use odis::algorithms::dimension::SatReduction;
///
/// let p = Poset::from_covering_relation(
///     vec![0u32, 1, 2],
///     vec![(0, 1), (1, 2)],
/// ).unwrap();
/// let dim = SatReduction.dimension(&p);
/// assert_eq!(dim, 1);
/// ```
pub trait DimensionAlgorithm<T> {
    /// Computes the order dimension of the given poset.
    fn dimension(&self, poset: &Poset<T>) -> usize;

    /// Returns a realizer: the minimal set of linear extensions whose
    /// intersection equals the original partial order.
    ///
    /// Each linear extension is a permutation of node indices (0..n),
    /// where position indicates rank (element at index 0 is the minimum).
    fn realizer(&self, poset: &Poset<T>) -> Vec<Vec<usize>>;
}