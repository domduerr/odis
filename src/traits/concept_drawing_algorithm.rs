//! Trait for drawing algorithms that need the concepts, not just the order.

use bit_set::BitSet;

use crate::data_structures::{
    drawing::Drawing, formal_context::FormalContext, iceberg_lattice::IcebergLattice,
    lattice::Lattice, poset::Poset,
};

/// A layout algorithm that reads the objects and attributes each node stands
/// for.
///
/// [`crate::traits::DrawingAlgorithm`] is generic in the node type and so sees
/// nothing of a node but its place in the order. That is all most layouts need.
/// It is not all every layout needs: an additive diagram puts a node at the sum
/// of one vector per object in its extent and per attribute in its intent, so
/// there the labelling *is* the input and the order only says which nodes to
/// join by an edge. Algorithms of that kind implement this trait instead — they
/// cannot implement the other one, whatever node type it is handed.
///
/// Implementors write [`ConceptDrawingAlgorithm::draw_concepts`]. The entry
/// points for a formal context, a concept lattice and an iceberg lattice are
/// derived from it, and may be overridden by an implementor that can do better
/// for one of them.
///
/// # Preconditions
/// - Each node carries its `(extent, intent)` as index sets over the objects
///   and attributes of one formal context.
/// - `object_count` and `attribute_count` are the dimensions of that context.
///   Extent and intent bits outside those ranges are ignored.
/// - The order need not be bounded, or a lattice: an iceberg concept lattice is
///   an order filter and usually has no bottom element.
///
/// # Output guarantees
/// - If `Some(drawing)` is returned, `drawing.coordinates.len()` equals the
///   number of nodes, indexed in node order.
/// - All coordinates are finite `f64` values in the algorithm's native space,
///   in which the top of the order has the *smallest* `y`. Viewport scaling is
///   the consumer's responsibility.
pub trait ConceptDrawingAlgorithm {
    /// Compute a 2D layout for an order of formal concepts.
    ///
    /// Returns `None` if the order is empty, or if the algorithm cannot produce
    /// a valid drawing within its budget.
    fn draw_concepts(
        &self,
        concepts: &Poset<(BitSet, BitSet)>,
        object_count: usize,
        attribute_count: usize,
    ) -> Option<Drawing>;

    /// Draw the concept lattice of `context`.
    ///
    /// Returns `None` if the context has no concepts.
    fn draw_context<T: Clone>(&self, context: &FormalContext<T>) -> Option<Drawing> {
        let lattice = context.concept_lattice()?;
        self.draw_concepts(
            &lattice.poset,
            context.objects.len(),
            context.attributes.len(),
        )
    }

    /// Draw a concept lattice whose nodes carry their `(extent, intent)` pair,
    /// as produced by [`FormalContext::concept_lattice`].
    ///
    /// `object_count` and `attribute_count` are the dimensions of the context
    /// the lattice came from.
    fn draw_lattice(
        &self,
        lattice: &Lattice<(BitSet, BitSet)>,
        object_count: usize,
        attribute_count: usize,
    ) -> Option<Drawing> {
        self.draw_concepts(&lattice.poset, object_count, attribute_count)
    }

    /// Draw an iceberg concept lattice.
    ///
    /// The number of objects is taken from `iceberg.total_objects`; only the
    /// number of attributes has to be supplied. Returns `None` for an empty
    /// iceberg.
    fn draw_iceberg(&self, iceberg: &IcebergLattice, attribute_count: usize) -> Option<Drawing> {
        self.draw_concepts(
            &iceberg.poset,
            iceberg.total_objects as usize,
            attribute_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs;

    use super::*;
    use crate::algorithms::Titanic;
    use crate::traits::IcebergConceptEnumerator;

    /// Records what the provided methods pass down to `draw_concepts`.
    #[derive(Default)]
    struct Recorder {
        seen: RefCell<Vec<(usize, usize, usize)>>,
    }

    impl ConceptDrawingAlgorithm for Recorder {
        fn draw_concepts(
            &self,
            concepts: &Poset<(BitSet, BitSet)>,
            object_count: usize,
            attribute_count: usize,
        ) -> Option<Drawing> {
            self.seen
                .borrow_mut()
                .push((concepts.nodes.len(), object_count, attribute_count));
            Some(Drawing::new(vec![(0.0, 0.0); concepts.nodes.len()]))
        }
    }

    fn living_beings() -> FormalContext<String> {
        let bytes = fs::read("test_data/living_beings_and_water.cxt").unwrap();
        FormalContext::<String>::from(&bytes).unwrap()
    }

    #[test]
    fn test_the_entry_points_pass_down_the_context_dimensions() {
        let context = living_beings();
        let lattice = context.concept_lattice().unwrap();
        let recorder = Recorder::default();

        recorder.draw_context(&context).unwrap();
        recorder
            .draw_lattice(&lattice, context.objects.len(), context.attributes.len())
            .unwrap();

        let expected = (
            lattice.poset.nodes.len(),
            context.objects.len(),
            context.attributes.len(),
        );
        assert_eq!(*recorder.seen.borrow(), vec![expected, expected]);
    }

    /// The object count of an iceberg is its own, not that of the concepts it
    /// kept: the filtered-out concepts still stand for objects, and an element
    /// vector is needed for every one of them.
    #[test]
    fn test_the_iceberg_entry_point_uses_the_total_object_count() {
        let context = living_beings();
        let iceberg = Titanic.enumerate(&context, 3);
        assert!(
            iceberg.poset.nodes.len() < context.concept_lattice().unwrap().poset.nodes.len(),
            "the threshold should actually filter something out"
        );

        let recorder = Recorder::default();
        recorder
            .draw_iceberg(&iceberg, context.attributes.len())
            .unwrap();

        assert_eq!(
            *recorder.seen.borrow(),
            vec![(
                iceberg.poset.nodes.len(),
                context.objects.len(),
                context.attributes.len(),
            )]
        );
    }
}
