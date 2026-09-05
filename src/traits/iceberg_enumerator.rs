//! Trait for iceberg concept enumeration algorithms.

use std::collections::HashSet;

use crate::data_structures::{formal_context::FormalContext, iceberg_lattice::IcebergLattice};

/// An algorithm that enumerates iceberg concepts from a formal context.
///
/// An iceberg concept is a formal concept whose extent (set of objects) has at least
/// `min_support` elements. The set of iceberg concepts forms a sub-lattice of the
/// full concept lattice (the "iceberg concept lattice").
pub trait IcebergConceptEnumerator {
    /// Enumerate all concepts with support ≥ `min_support`.
    ///
    /// Returns a poset ordered by extent inclusion. The result is empty if
    /// `min_support > ctx.objects.len()`.
    fn enumerate<T>(&self, ctx: &FormalContext<T>, min_support: u32) -> IcebergLattice;

    /// Named variant of `enumerate`; translates index pairs to `(HashSet<T>, HashSet<T>)` pairs.
    ///
    /// Returns an empty `Vec` if no concepts meet the support threshold.
    fn enumerate_named_concepts<T>(
        &self,
        ctx: &FormalContext<T>,
        min_support: u32,
    ) -> Vec<(HashSet<T>, HashSet<T>)>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        self.enumerate(ctx, min_support)
            .poset
            .nodes
            .iter()
            .map(|(extent_bits, intent_bits)| {
                let extent = extent_bits.iter().map(|i| ctx.objects[i].clone()).collect();
                let intent = intent_bits.iter().map(|i| ctx.attributes[i].clone()).collect();
                (extent, intent)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{fs};

    use crate::{algorithms::Titanic, traits::IcebergConceptEnumerator, FormalContext};

    fn load_living_beings() -> FormalContext<String> {
        FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_enumerate_named_concepts_count_matches_enumerate() {
        let ctx = load_living_beings();
        let min_support = 3;
        let lattice = Titanic.enumerate(&ctx, min_support);
        let named = Titanic.enumerate_named_concepts(&ctx, min_support);
        assert_eq!(
            named.len(),
            lattice.poset.nodes.len(),
            "enumerate_named_concepts count must match enumerate node count"
        );
    }

    #[test]
    fn test_enumerate_named_concepts_min_support_exceeds_objects_is_empty() {
        let ctx = load_living_beings();
        // min_support larger than the total number of objects → no concepts
        let min_support = ctx.objects.len() as u32 + 1;
        let named = Titanic.enumerate_named_concepts(&ctx, min_support);
        assert!(
            named.is_empty(),
            "enumerate_named_concepts should return empty vec when min_support > object count"
        );
    }

    #[test]
    fn test_enumerate_named_concepts_empty_context_is_empty() {
        let ctx = FormalContext::<String>::new();
        let named = Titanic.enumerate_named_concepts(&ctx, 0);
        assert!(
            named.is_empty(),
            "enumerate_named_concepts on empty context should return empty vec"
        );
    }
}
