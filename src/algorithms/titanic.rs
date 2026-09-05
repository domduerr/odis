//! Titanic iceberg concept enumeration algorithm.

use std::collections::HashMap;

use bit_set::BitSet;

use crate::data_structures::{formal_context::FormalContext, iceberg_lattice::IcebergLattice, poset::Poset};
use crate::traits::IcebergConceptEnumerator;

/// Implements the TITANIC algorithm for iceberg concept lattice computation.
///
/// Reference: Stumme, Taouil, Bastide, Pasquier, Lakhal (2002):
/// "Computing iceberg concept lattices with TITANIC", Data & Knowledge Engineering.
pub struct Titanic;

impl IcebergConceptEnumerator for Titanic {
    fn enumerate<T>(&self, ctx: &FormalContext<T>, min_support: u32) -> IcebergLattice {
        titanic_enumerate(ctx, min_support)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the formal closure of a given extent:
///   closure(E) = { m | E ⊆ col[m] }
/// i.e. every attribute shared by all objects in the extent.
fn closure_from_extent<T>(extent: &BitSet, num_attrs: usize, ctx: &FormalContext<T>) -> BitSet {
    let mut closure = BitSet::with_capacity(num_attrs);
    for m in 0..num_attrs {
        if extent.is_subset(&ctx.atomic_attribute_derivations[m]) {
            closure.insert(m);
        }
    }
    closure
}

/// Generate (k+1)-candidates from sorted k-key-sets using the Apriori join step.
///
/// Joins pairs `(a, b)` that share the same first (k-1) elements and differ only
/// in the last. The resulting candidate is `a ∪ {last(b)}`, sorted.
fn apriori_gen(sorted_keys: &[Vec<u16>]) -> Vec<Vec<u16>> {
    let mut candidates = Vec::new();
    let n = sorted_keys.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &sorted_keys[i];
            let b = &sorted_keys[j];
            let k = a.len();
            // Same (k-1)-prefix and last element of b strictly greater than last of a.
            let same_prefix = if k == 0 {
                true
            } else {
                a[..k - 1] == b[..k - 1] && a[k - 1] < b[k - 1]
            };
            if same_prefix {
                let mut new = a.clone();
                new.push(*b.last().unwrap());
                candidates.push(new);
            }
        }
    }
    candidates
}

/// Return `false` if any k-1 subset of `candidate` (formed by removing elements
/// at indices `0..len-2`) is absent from `key_set`.
///
/// The last two subsets (removing the last two elements) are guaranteed to be in
/// `key_set` by the Apriori join construction and need not be re-checked.
fn all_k_minus_1_subsets_in_keys(candidate: &[u16], key_set: &std::collections::HashSet<Vec<u16>>) -> bool {
    let len = candidate.len();
    for i in 0..len.saturating_sub(2) {
        let subset: Vec<u16> = candidate[..i].iter().chain(&candidate[i + 1..]).copied().collect();
        if !key_set.contains(&subset) {
            return false;
        }
    }
    true
}

/// Minimum support across all (len-1)-subsets of `candidate`, looked up in `support_map`.
fn min_support_of_subsets(candidate: &[u16], support_map: &HashMap<Vec<u16>, u32>) -> u32 {
    let len = candidate.len();
    let mut min_sup = u32::MAX;
    for i in 0..len {
        let subset: Vec<u16> = candidate[..i].iter().chain(&candidate[i + 1..]).copied().collect();
        if let Some(&s) = support_map.get(&subset) {
            min_sup = min_sup.min(s);
        }
    }
    min_sup
}

/// Convert a `BitSet` to a sorted `Vec<usize>` for use as a HashMap key.
#[inline]
fn bitset_key(b: &BitSet) -> Vec<usize> {
    b.iter().collect()
}

/// Construct an `IcebergLattice` from a flat collection of `(extent, intent, support)` triples.
///
/// Builds the transitive relation (extent inclusion) and delegates to
/// `Poset::from_transitive_relation` for reduction.
fn build_iceberg_lattice(
    concepts: Vec<(BitSet, BitSet, u32)>,
    total_objects: u32,
) -> IcebergLattice {
    if concepts.is_empty() {
        return IcebergLattice::empty(total_objects);
    }

    let n = concepts.len();

    // edge (i, j): extent[i] ⊊ extent[j]  →  concept i is below concept j
    let mut trans_edges: Vec<(u32, u32)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let (ext_i, _, _) = &concepts[i];
                let (ext_j, _, _) = &concepts[j];
                if ext_i != ext_j && ext_i.is_subset(ext_j) {
                    trans_edges.push((i as u32, j as u32));
                }
            }
        }
    }

    let support: Vec<u32> = concepts.iter().map(|(_, _, s)| *s).collect();
    let nodes: Vec<(BitSet, BitSet)> = concepts.into_iter().map(|(ext, int, _)| (ext, int)).collect();

    let poset = Poset::from_transitive_relation(nodes, trans_edges)
        .expect("iceberg concepts form a valid DAG");

    IcebergLattice::new(poset, support, total_objects)
}

// ── Core algorithm ────────────────────────────────────────────────────────────

fn titanic_enumerate<T>(ctx: &FormalContext<T>, min_support: u32) -> IcebergLattice {
    let num_objects = ctx.objects.len();
    let num_attrs = ctx.attributes.len();
    let total_objects = num_objects as u32;

    if num_objects == 0 || min_support > total_objects {
        return IcebergLattice::empty(total_objects);
    }

    // ── Level 0: empty key set ∅, support = |G| ──────────────────────────────
    let mut all_objects = BitSet::with_capacity(num_objects);
    for i in 0..num_objects {
        all_objects.insert(i);
    }

    let empty_closure = closure_from_extent(&all_objects, num_attrs, ctx);

    // Deduplication: intent (sorted attrs) → already seen?
    let mut seen_intents: HashMap<Vec<usize>, ()> = HashMap::new();
    let mut concepts: Vec<(BitSet, BitSet, u32)> = Vec::new();

    seen_intents.insert(bitset_key(&empty_closure), ());
    concepts.push((all_objects, empty_closure.clone(), total_objects));

    // Support map for key sets found so far: sorted itemset → support.
    // Seed with ∅ → |G| for the ps computation at level 1.
    let mut support_map: HashMap<Vec<u16>, u32> = HashMap::new();
    support_map.insert(vec![], total_objects);

    // ── Level 1: 1-attribute key sets ────────────────────────────────────────
    let mut prev_key_sets: Vec<Vec<u16>> = Vec::new();

    for m in 0..num_attrs {
        // Attributes already in G'' have support = |G|, so they are not key sets
        // (their 1-set support equals ps(∅) = |G|).
        if empty_closure.contains(m) {
            continue;
        }

        let extent = ctx.atomic_attribute_derivations[m].clone();
        let support = extent.len() as u32;

        if support < min_support {
            continue;
        }

        let closure = closure_from_extent(&extent, num_attrs, ctx);
        let key = bitset_key(&closure);
        if seen_intents.insert(key, ()).is_none() {
            concepts.push((extent, closure, support));
        }

        let itemset: Vec<u16> = vec![m as u16];
        support_map.insert(itemset.clone(), support);
        prev_key_sets.push(itemset);
    }

    // ── Levels k ≥ 2 ─────────────────────────────────────────────────────────
    loop {
        if prev_key_sets.len() < 2 {
            break;
        }

        let candidates = apriori_gen(&prev_key_sets);
        if candidates.is_empty() {
            break;
        }

        let prev_set: std::collections::HashSet<Vec<u16>> =
            prev_key_sets.iter().cloned().collect();

        let mut next_key_sets: Vec<Vec<u16>> = Vec::new();

        for candidate in candidates {
            // Apriori prune: all non-guaranteed (k-1)-subsets must be key sets.
            if !all_k_minus_1_subsets_in_keys(&candidate, &prev_set) {
                continue;
            }

            // Compute extent = column-intersection of all candidate attributes.
            let mut extent = ctx.atomic_attribute_derivations[candidate[0] as usize].clone();
            for &m in &candidate[1..] {
                extent.intersect_with(&ctx.atomic_attribute_derivations[m as usize]);
            }
            let support = extent.len() as u32;

            if support < min_support {
                continue;
            }

            // Key-set check: X is a key set iff supp(X) < min support of (|X|-1)-subsets.
            let ps = min_support_of_subsets(&candidate, &support_map);
            if support >= ps {
                continue;
            }

            let closure = closure_from_extent(&extent, num_attrs, ctx);
            let key = bitset_key(&closure);
            if seen_intents.insert(key, ()).is_none() {
                concepts.push((extent, closure, support));
            }

            support_map.insert(candidate.clone(), support);
            next_key_sets.push(candidate);
        }

        if next_key_sets.is_empty() {
            break;
        }
        prev_key_sets = next_key_sets;
    }

    build_iceberg_lattice(concepts, total_objects)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::algorithms::SearchBudget;
    use crate::traits::IcebergConceptEnumerator;
    use crate::FormalContext;

    fn living_beings_ctx() -> FormalContext<String> {
        FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap()
    }

    fn eu_ctx() -> FormalContext<String> {
        FormalContext::<String>::from(&fs::read("test_data/eu.cxt").unwrap()).unwrap()
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn empty_context_returns_empty_lattice() {
        let ctx = FormalContext::<String>::new();
        let ice = Titanic.enumerate(&ctx, 0);
        assert_eq!(ice.poset.nodes.len(), 0);
    }

    #[test]
    fn min_support_above_object_count_returns_empty() {
        let ctx = living_beings_ctx();
        let total = ctx.objects.len() as u32;
        let ice = Titanic.enumerate(&ctx, total + 1);
        assert_eq!(ice.poset.nodes.len(), 0);
    }

    #[test]
    fn min_support_zero_returns_all_concepts() {
        let ctx = living_beings_ctx();
        let ice = Titanic.enumerate(&ctx, 0);
        let full = ctx.concept_lattice().expect("concept lattice");
        assert_eq!(ice.poset.nodes.len(), full.poset.nodes.len());
    }

    #[test]
    fn min_support_equals_object_count_returns_one_concept() {
        let ctx = living_beings_ctx();
        let total = ctx.objects.len() as u32;
        let ice = Titanic.enumerate(&ctx, total);
        // Only the top concept (∅' = G, G'' = most-general intent) has support = |G|.
        assert_eq!(ice.poset.nodes.len(), 1);
    }

    // ── Regression ───────────────────────────────────────────────────────────

    #[test]
    fn living_beings_min_support_4() {
        let ctx = living_beings_ctx();
        let ice = Titanic.enumerate(&ctx, 4);
        // All returned concepts must have support ≥ 4.
        for (i, s) in ice.support.iter().enumerate() {
            assert!(*s >= 4, "concept {i} has support {s} < 4");
        }
        // The top concept (support = |G| = 8) must be present.
        assert!(ice.support.contains(&8));
    }

    #[test]
    fn living_beings_support_values_match_extents() {
        let ctx = living_beings_ctx();
        let ice = Titanic.enumerate(&ctx, 1);
        for (i, &s) in ice.support.iter().enumerate() {
            let (extent, _intent) = &ice.poset.nodes[i];
            assert_eq!(s, extent.len() as u32, "support mismatch at node {i}");
        }
    }

    #[test]
    fn covering_edges_satisfy_transitivity() {
        let ctx = living_beings_ctx();
        let ice = Titanic.enumerate(&ctx, 1);
        // Every covering edge (u, v) must satisfy extent[u] ⊊ extent[v].
        for &(u, v) in &ice.poset.covering_edges {
            let (ext_u, _) = &ice.poset.nodes[u as usize];
            let (ext_v, _) = &ice.poset.nodes[v as usize];
            assert!(
                ext_u.is_subset(ext_v) && ext_u != ext_v,
                "covering edge ({u}, {v}) does not satisfy extent inclusion"
            );
        }
    }

    #[test]
    fn eu_context_min_support_5() {
        let ctx = eu_ctx();
        let ice = Titanic.enumerate(&ctx, 5);
        for &s in &ice.support {
            assert!(s >= 5, "concept has support {s} < 5");
        }
    }

    /// Validates the quickstart/integration scenario: enumerate + draw_poset round-trip.
    #[test]
    fn quickstart_living_beings_draw_poset() {
        use crate::algorithms::dimdraw::DimDraw;
        use crate::traits::DrawingAlgorithm;

        let ctx = living_beings_ctx();
        let ice = Titanic.enumerate(&ctx, 4);
        assert!(!ice.poset.nodes.is_empty(), "expected at least one concept at min_support=4");

        // draw_poset must return a drawing with one coordinate per node
        let drawing = DimDraw { budget: SearchBudget::Milliseconds(5000) }
            .draw_poset(&ice.poset)
            .expect("draw_poset should succeed on non-empty poset");
        assert_eq!(
            drawing.coordinates.len(),
            ice.poset.nodes.len(),
            "coordinate count must equal node count"
        );
    }
}
