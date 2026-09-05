//! Trait for formal concept enumeration algorithms.

use std::collections::HashSet;

use bit_set::BitSet;

use crate::FormalContext;

/// Trait for algorithms that enumerate all formal concepts of a formal context.
///
/// A formal concept is a maximal pair `(A, B)` where `A ⊆ G` is an extent,
/// `B ⊆ M` is an intent, and the Galois connection gives `A' = B`, `B' = A`.
///
/// # Preconditions
/// - `context` must be a valid `FormalContext<T>` whose incidence dimensions match
///   `objects.len() × attributes.len()`.
///
/// # Output guarantees
/// - **Completeness**: every formal concept is yielded exactly once.
/// - **No duplicates**: no `(A, B)` pair appears more than once.
/// - **Correctness**: for every `(A, B)` yielded, `A'' = A` and `B'' = B`.
///
/// # Complexity
/// - Worst-case `O(|G| × |M| × |L|)` where `|L|` is the number of formal concepts.
pub trait ConceptEnumerator<T> {
    /// Enumerate all formal concepts of `context`.
    ///
    /// Each yielded `(extent, intent)` is index-based: `extent` is a `BitSet` over
    /// `0..context.objects.len()` and `intent` over `0..context.attributes.len()`.
    fn enumerate_concepts<'ctx>(
        &self,
        context: &'ctx FormalContext<T>,
    ) -> impl Iterator<Item = (BitSet, BitSet)> + 'ctx;

    /// Named variant of `enumerate_concepts`; collects into a `Vec`.
    fn enumerate_named_concepts(&self, context: &FormalContext<T>) -> Vec<(HashSet<T>, HashSet<T>)>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        self.enumerate_concepts(context)
            .map(|(g_bits, m_bits)| {
                let extent = g_bits.iter().map(|i| context.objects[i].clone()).collect();
                let intent = m_bits.iter().map(|i| context.attributes[i].clone()).collect();
                (extent, intent)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs};

    use crate::{
        algorithms::next_closure::NextClosure,
        traits::ConceptEnumerator,
        FormalContext,
    };

    #[test]
    fn test_enumerate_named_concepts_matches_context_concepts() {
        let ctx = FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap();

        let via_trait: Vec<(HashSet<String>, HashSet<String>)> =
            NextClosure.enumerate_named_concepts(&ctx);
        let via_method: Vec<(HashSet<String>, HashSet<String>)> = ctx.next_closure_concepts().collect();

        assert_eq!(via_trait.len(), via_method.len());
        for pair in &via_trait {
            assert!(via_method.contains(pair), "concept {:?} missing from context.concepts()", pair);
        }
    }
}
