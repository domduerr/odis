//! Trait for lattice drawing algorithms.

use crate::data_structures::{drawing::Drawing, lattice::Lattice, poset::Poset};

/// A layout algorithm that computes a 2D drawing from a `Lattice` or `Poset`.
///
/// Being generic in the node type, an implementor sees nothing of a node but
/// its place in the order. Layouts that also need to know which objects and
/// attributes a node stands for implement
/// [`crate::traits::ConceptDrawingAlgorithm`] instead.
///
/// # Preconditions
/// - The lattice/poset must be non-empty (at least one node).
///
/// # Output guarantees
/// - If `Some(drawing)` is returned, `drawing.coordinates.len()` equals
///   the number of nodes in the lattice (`lattice.poset.nodes.len()`).
/// - All coordinates are finite `f64` values in the algorithm's native space.
///   Viewport scaling is the consumer's responsibility.
///
/// # Complexity
/// - `DimDraw`: annealed incumbent plus branch-and-bound over one
///   representative per symmetry class; bounded by its
///   [`crate::algorithms::SearchBudget`], and a proven optimum only when that
///   budget is `Unbounded`.
/// - `Sugiyama`: O((V + E) log V) via the `rust_sugiyama` library.
///   Returns `None` only for an empty lattice.
pub trait DrawingAlgorithm {
    /// Compute a 2D layout from the given `Lattice`.
    ///
    /// Returns `None` only if the lattice is empty or the algorithm cannot
    /// produce a valid drawing within its budget.
    fn draw<T>(&self, lattice: &Lattice<T>) -> Option<Drawing>;

    /// Draw a poset (not necessarily a complete lattice).
    ///
    /// Promotes the poset to a `Lattice` and calls `draw`.
    /// Algorithms that support disconnected posets (e.g. `Sugiyama`) override this.
    fn draw_poset<T: Clone>(&self, poset: &Poset<T>) -> Option<Drawing> {
        let poset_owned =
            Poset::from_covering_relation(poset.nodes.clone(), poset.covering_edges.clone())
                .ok()?;
        self.draw(&Lattice::from_poset(poset_owned)?)
    }
}
