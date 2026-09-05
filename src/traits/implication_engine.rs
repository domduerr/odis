//! Trait for implication basis computation algorithms.

use std::collections::HashSet;

use bit_set::BitSet;

use crate::FormalContext;

/// Trait for algorithms that compute an implication basis of a formal context.
///
/// Returns the Duquenne–Guigues canonical implication basis. An implication
/// `A → B` holds in `context` iff every object sharing all attributes of `A`
/// also shares every attribute in `B`.
///
/// # Preconditions
/// - `context` must be a valid `FormalContext<T>`.
///
/// # Output guarantees
/// - **Soundness**: every returned implication is valid in `context`.
/// - **Completeness**: every valid implication follows from the result under
///   Armstrong's axioms.
/// - **Minimality**: no strict subset of the result is still complete.
/// - **Canonicity**: premise sets are pseudo-closed (Duquenne–Guigues basis).
///
/// # Complexity
/// - Worst-case exponential in `|M|` (number of attributes).
pub trait ImplicationEngine<T> {
    /// Compute the Duquenne–Guigues canonical implication basis of `context`.
    ///
    /// Each returned `(premise, conclusion)` is index-based: sets over
    /// `0..context.attributes.len()`.
    fn compute_basis(&self, context: &FormalContext<T>) -> Vec<(BitSet, BitSet)>;

    /// Named variant of `compute_basis`; translates attribute indices to `T` values.
    fn compute_named_basis(&self, context: &FormalContext<T>) -> Vec<(HashSet<T>, HashSet<T>)>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        self.compute_basis(context)
            .into_iter()
            .map(|(premise_bits, conclusion_bits)| {
                let premise = premise_bits.iter().map(|i| context.attributes[i].clone()).collect();
                let conclusion = conclusion_bits.iter().map(|i| context.attributes[i].clone()).collect();
                (premise, conclusion)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs};

    use crate::{algorithms::CanonicalBasis, traits::ImplicationEngine, FormalContext};

    #[test]
    fn test_compute_named_basis_matches_context_canonical_basis() {
        let ctx = FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap();

        let via_trait: Vec<(HashSet<String>, HashSet<String>)> =
            CanonicalBasis.compute_named_basis(&ctx);
        let via_method: Vec<(HashSet<String>, HashSet<String>)> = ctx.canonical_basis();

        assert_eq!(via_trait.len(), via_method.len());
        for pair in &via_trait {
            assert!(
                via_method.contains(pair),
                "implication {:?} missing from context.canonical_basis()",
                pair
            );
        }
    }
}
