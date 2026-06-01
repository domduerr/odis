//! Algorithms for Formal Concept Analysis.
//!
//! ## Concept enumeration
//!
//! Two algorithms are provided; both implement [`crate::traits::ConceptEnumerator`]:
//!
//! - [`NextClosure`] — lectic-order enumeration; simple, low memory overhead.
//! - [`Fcbo`] — Fast Close-by-One; faster on dense contexts.
//!
//! Convenience wrappers on [`crate::FormalContext`] (`index_concepts`,
//! `index_next_closure_concepts`, `index_fcbo_concepts`) delegate to `NextClosure` by default.
//!
//! ## Implication bases
//!
//! - [`CanonicalBasis`] — computes the Duquenne–Guigues canonical implication basis
//!   (implements [`crate::traits::ImplicationEngine`]).
//!
//! ## Lattice drawing
//!
//! Both algorithms implement [`crate::traits::DrawingAlgorithm`]:
//!
//! - [`DimDraw`] — branch-and-bound drawing; time-bounded via `timeout_ms`.
//! - [`Sugiyama`] — hierarchical layout via the `rust-sugiyama` library.
//!
//! ## Iceberg lattices
//!
//! - [`Titanic`] — enumerates concepts above a minimum support threshold
//!   (implements [`crate::traits::IcebergConceptEnumerator`]).
//!
//! ## Attribute exploration
//!
//! - [`ExplorationMachine`] — drives interactive attribute exploration over a growing context.

use std::collections::HashSet;

use bit_set::BitSet;

use crate::data_structures::lattice::Lattice;
use crate::FormalContext;

pub(crate) mod attribute_exploration;
pub mod canonical_basis;
pub mod dimension;
pub mod exploration_machine;
pub mod dimdraw;
pub mod fcbo;
pub mod next_closure;
pub mod sugiyama;
pub mod titanic;
pub mod upper_neighbor;

pub use canonical_basis::CanonicalBasis;
pub use exploration_machine::{ExplorationInput, ExplorationMachine, ExplorationState};
pub use dimdraw::DimDraw;
pub use fcbo::Fcbo;
pub use next_closure::NextClosure;
pub use sugiyama::Sugiyama;
pub use titanic::Titanic;

impl<T> FormalContext<T> {
    /// Returns an iterator over all formal concepts using the Next Closure algorithm.
    /// See also [`FormalContext::next_closure_concepts`] and [`FormalContext::index_fcbo_concepts`].
    pub fn index_next_closure_concepts(&self) -> impl Iterator<Item = (BitSet, BitSet)> + '_ {
        next_closure::index_next_closure_concepts(self)
    }

    /// Returns an iterator over all formal concepts using the default algorithm (Next Closure).
    ///
    /// Delegates to [`FormalContext::index_next_closure_concepts`]; use that directly for explicit
    /// algorithm control (e.g., to switch to [`FormalContext::index_fcbo_concepts`]).
    pub fn index_concepts(&self) -> impl Iterator<Item = (BitSet, BitSet)> + '_ {
        self.index_next_closure_concepts()
    }

    /// Returns an iterator over all formal concepts using the Next Closure algorithm,
    /// with each concept expressed as named object and attribute sets.
    /// See also [`FormalContext::index_next_closure_concepts`] and [`FormalContext::fcbo_concepts`].
    pub fn next_closure_concepts(&self) -> impl Iterator<Item = (HashSet<T>, HashSet<T>)> + '_
    where
        T: Eq + std::hash::Hash + Clone,
    {
        self.index_next_closure_concepts().map(|(g_bits, m_bits)| {
            let extent = g_bits.iter().map(|i| self.objects[i].clone()).collect();
            let intent = m_bits.iter().map(|i| self.attributes[i].clone()).collect();
            (extent, intent)
        })
    }

    /// Returns an iterator over all formal concepts using the default algorithm (Next Closure),
    /// with each concept expressed as named object and attribute sets.
    ///
    /// Delegates to [`FormalContext::next_closure_concepts`]; use that directly for explicit
    /// algorithm control (e.g., to switch to [`FormalContext::fcbo_concepts`]).
    pub fn concepts(&self) -> impl Iterator<Item = (HashSet<T>, HashSet<T>)> + '_
    where
        T: Eq + std::hash::Hash + Clone,
    {
        self.next_closure_concepts()
    }

    /// Returns an iterator over all formal concepts using the FCBO algorithm.
    /// See also [`FormalContext::fcbo_concepts`].
    pub fn index_fcbo_concepts(&self) -> impl Iterator<Item = (BitSet, BitSet)> + '_ {
        fcbo::index_fcbo_concepts(self)
    }

    /// Returns an iterator over all formal concepts using the FCBO algorithm,
    /// with each concept expressed as named object and attribute sets.
    /// See also [`FormalContext::index_fcbo_concepts`].
    pub fn fcbo_concepts(&self) -> impl Iterator<Item = (HashSet<T>, HashSet<T>)> + '_
    where
        T: Eq + std::hash::Hash + Clone,
    {
        self.index_fcbo_concepts().map(|(g_bits, m_bits)| {
            let extent = g_bits.iter().map(|i| self.objects[i].clone()).collect();
            let intent = m_bits.iter().map(|i| self.attributes[i].clone()).collect();
            (extent, intent)
        })
    }

    /// Computes the canonical basis of implications using indices.
    pub fn index_canonical_basis(&self) -> Vec<(BitSet, BitSet)> {
        canonical_basis::index_canonical_basis(self)
    }

    /// Computes the canonical basis of implications with attribute names.
    pub fn canonical_basis(&self) -> Vec<(HashSet<T>, HashSet<T>)>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        self.index_canonical_basis()
            .into_iter()
            .map(|(premise_bits, conclusion_bits)| {
                let premise = premise_bits.iter().map(|i| self.attributes[i].clone()).collect();
                let conclusion = conclusion_bits.iter().map(|i| self.attributes[i].clone()).collect();
                (premise, conclusion)
            })
            .collect()
    }

    /// Computes the canonical basis of implications using indices and an optimized approach.
    pub fn index_canonical_basis_optimised(&self) -> Vec<(BitSet, BitSet)> {
        canonical_basis::index_canonical_basis_optimised(self)
    }

    /// Computes the canonical basis of implications with attribute names using an optimized approach.
    pub fn canonical_basis_optimised(&self) -> Vec<(HashSet<T>, HashSet<T>)>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        self.index_canonical_basis_optimised()
            .into_iter()
            .map(|(premise_bits, conclusion_bits)| {
                let premise = premise_bits.iter().map(|i| self.attributes[i].clone()).collect();
                let conclusion = conclusion_bits.iter().map(|i| self.attributes[i].clone()).collect();
                (premise, conclusion)
            })
            .collect()
    }

    /// Computes the upper neighbor of a concept extent using indices.
    pub fn index_upper_neighbor(&self, input: &BitSet) -> BitSet {
        upper_neighbor::index_upper_neighbor(input, self)
    }

    /// Computes the upper neighbor of a concept extent, accepting and returning named object sets.
    ///
    /// Names in `extent` that are not found in the context are silently ignored.
    pub fn upper_neighbor(&self, extent: &HashSet<T>) -> HashSet<T>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        let bits: BitSet = extent
            .iter()
            .filter_map(|name| self.objects.iter().position(|o| o == name))
            .collect();
        let result_bits = upper_neighbor::index_upper_neighbor(&bits, self);
        result_bits.iter().map(|i| self.objects[i].clone()).collect()
    }

    /// Computes the next preclosure in lectic order using attribute indices.
    ///
    /// Delegates entirely to [`canonical_basis::index_next_preclosure`].
    /// See also [`FormalContext::next_preclosure`].
    pub fn index_next_preclosure(
        &self,
        implications: &[(BitSet, BitSet)],
        input: &BitSet,
    ) -> BitSet {
        canonical_basis::index_next_preclosure(self, implications, input)
    }

    /// Computes the next preclosure in lectic order, accepting and returning attribute names.
    ///
    /// Translates names to indices (unknown names are silently dropped), calls
    /// [`FormalContext::index_next_preclosure`], and translates the result back to names.
    /// See also [`FormalContext::index_next_preclosure`].
    pub fn next_preclosure(
        &self,
        implications: &[(HashSet<T>, HashSet<T>)],
        input: &HashSet<T>,
    ) -> HashSet<T>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        let impl_bits: Vec<(BitSet, BitSet)> = implications
            .iter()
            .map(|(premise, conclusion)| {
                let premise_bits = premise
                    .iter()
                    .filter_map(|name| self.attributes.iter().position(|a| a == name))
                    .collect();
                let conclusion_bits = conclusion
                    .iter()
                    .filter_map(|name| self.attributes.iter().position(|a| a == name))
                    .collect();
                (premise_bits, conclusion_bits)
            })
            .collect();
        let input_bits: BitSet = input
            .iter()
            .filter_map(|name| self.attributes.iter().position(|a| a == name))
            .collect();
        let result_bits = self.index_next_preclosure(&impl_bits, &input_bits);
        result_bits.iter().map(|i| self.attributes[i].clone()).collect()
    }
}

impl FormalContext<String> {
    /// Performs interactive attribute exploration using indices.
    pub fn index_attribute_exploration(&mut self) -> Vec<(BitSet, BitSet)> {
        attribute_exploration::index_attribute_exploration(self)
    }

    /// Performs interactive attribute exploration, returning the final implication basis
    /// with attribute names instead of indices.
    pub fn attribute_exploration(&mut self) -> Vec<(HashSet<String>, HashSet<String>)> {
        self.index_attribute_exploration()
            .into_iter()
            .map(|(premise_bits, conclusion_bits)| {
                let premise = premise_bits.iter().map(|i| self.attributes[i].clone()).collect();
                let conclusion = conclusion_bits.iter().map(|i| self.attributes[i].clone()).collect();
                (premise, conclusion)
            })
            .collect()
    }

    /// Callback-based attribute exploration (index space).
    /// The callback receives `(premise_bits, conclusion_bits)` and returns:
    /// - `None` → accept the implication
    /// - `Some((object_name, attribute_bits))` → reject; add counterexample object
    pub fn index_attribute_exploration_with_callback<F>(
        &mut self,
        callback: F,
    ) -> Vec<(BitSet, BitSet)>
    where
        F: FnMut(&BitSet, &BitSet) -> Option<(String, BitSet)>,
    {
        attribute_exploration::index_attribute_exploration_with_callback(self, callback)
    }

    /// Callback-based attribute exploration (named).
    /// The callback receives `(premise_names, conclusion_names)` and returns:
    /// - `None` → accept the implication
    /// - `Some((object_name, attribute_names))` → reject; add counterexample object
    pub fn attribute_exploration_with_callback<F>(
        &mut self,
        mut callback: F,
    ) -> Vec<(HashSet<String>, HashSet<String>)>
    where
        F: FnMut(&HashSet<String>, &HashSet<String>) -> Option<(String, HashSet<String>)>,
    {
        // Clone attribute list upfront to avoid borrowing `self` inside the closure
        // while also borrowing it mutably for exploration.
        let attributes_snap = self.attributes.clone();
        let wrapped = |premise_bits: &BitSet, conclusion_bits: &BitSet| -> Option<(String, BitSet)> {
            let premise: HashSet<String> = premise_bits.iter().map(|i| attributes_snap[i].clone()).collect();
            let conclusion: HashSet<String> = conclusion_bits.iter().map(|i| attributes_snap[i].clone()).collect();
            callback(&premise, &conclusion).map(|(name, attr_names)| {
                let attr_bits: BitSet = attr_names
                    .iter()
                    .filter_map(|a| attributes_snap.iter().position(|x| x == a))
                    .collect();
                (name, attr_bits)
            })
        };
        let attr_snap2 = self.attributes.clone();
        self.index_attribute_exploration_with_callback(wrapped)
            .into_iter()
            .map(|(premise_bits, conclusion_bits)| {
                let premise = premise_bits.iter().map(|i| attr_snap2[i].clone()).collect();
                let conclusion = conclusion_bits.iter().map(|i| attr_snap2[i].clone()).collect();
                (premise, conclusion)
            })
            .collect()
    }
}

impl<T: Clone> FormalContext<T> {
    /// Derives the concept lattice from this formal context.
    ///
    /// Per the Hauptsatz der FCA, the concept lattice is a complete lattice
    /// ordered by the subconcept relation: concept A ≤ concept B iff
    /// extent(A) ⊆ extent(B). `join` and `meet` correspond to the Galois
    /// connection closure of extent union / intent union respectively.
    ///
    /// Returns `None` if the context has no formal concepts.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n2\n2\n\ncat\ndog\nx\ny\nXX\nX.\n").unwrap();
    /// let lattice = ctx.concept_lattice().unwrap();
    /// // 2 objects + 2 attributes generate at least 2 concepts
    /// assert!(lattice.poset.nodes.len() >= 2);
    /// ```
    pub fn concept_lattice(&self) -> Option<crate::data_structures::lattice::Lattice<(BitSet, BitSet)>> {
        use crate::data_structures::{lattice::Lattice, poset::Poset};

        let concepts: Vec<(BitSet, BitSet)> = self.index_concepts().collect();
        let n = concepts.len();
        if n == 0 {
            return None;
        }

        // Step 1: find all strict-subconcept pairs: i is a strict subconcept of j
        // when extent(i) ⊊ extent(j).
        let extents: Vec<&BitSet> = concepts.iter().map(|c| &c.0).collect();
        let mut is_strict_sub = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j && extents[i].is_subset(extents[j]) {
                    is_strict_sub[i][j] = true;
                }
            }
        }

        // Step 2: transitive reduction → covering edges (i, j) meaning i ≺ j
        // (concept i is directly covered by concept j; no concept k strictly between them).
        // Convention: (lower, upper) = (more specific, more general).
        let mut covering_edges: Vec<(u32, u32)> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if is_strict_sub[i][j] {
                    let is_direct = !(0..n).any(|k| is_strict_sub[i][k] && is_strict_sub[k][j]);
                    if is_direct {
                        covering_edges.push((i as u32, j as u32));
                    }
                }
            }
        }

        let poset = Poset::from_covering_relation(concepts, covering_edges).ok()?;
        Lattice::from_poset(poset)
    }
}

impl FormalContext<String> {
    /// Compute reduced concept labels for each node in a concept lattice.
    ///
    /// For each concept node, returns the objects first introduced at that
    /// concept (reduced object labelling) and the attributes first introduced
    /// at that concept (reduced attribute labelling).  Most nodes will have
    /// empty lists; objects/attributes that do not generate a unique concept
    /// are not listed at any node.
    ///
    /// The returned `Vec` is parallel to `lattice.poset.nodes`: index `i`
    /// corresponds to the concept at `lattice.poset.nodes[i]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n2\n2\n\ncat\ndog\nx\ny\nXX\nX.\n").unwrap();
    /// let lattice = ctx.concept_lattice().unwrap();
    /// let labels = ctx.reduced_labels(&lattice);
    /// // Total labelled objects == number of objects
    /// let total_objects: usize = labels.iter().map(|(objs, _)| objs.len()).sum();
    /// assert_eq!(total_objects, ctx.objects.len());
    /// ```
    pub fn reduced_labels(
        &self,
        lattice: &Lattice<(BitSet, BitSet)>,
    ) -> Vec<(Vec<String>, Vec<String>)> {
        let node_count = lattice.poset.nodes.len();
        let mut object_labels: Vec<Vec<String>> = vec![Vec::new(); node_count];
        let mut attribute_labels: Vec<Vec<String>> = vec![Vec::new(); node_count];

        // Reduced object labelling: place each object g at the concept generated by {g}''.
        for obj_idx in 0..self.objects.len() {
            let mut g = BitSet::new();
            g.insert(obj_idx);
            let hull = self.index_object_hull(&g);
            if let Some(node_idx) = lattice
                .poset
                .nodes
                .iter()
                .position(|(extent, _)| *extent == hull)
            {
                object_labels[node_idx].push(self.objects[obj_idx].clone());
            }
        }

        // Reduced attribute labelling: place each attribute m at the concept generated by {m}'.
        for attr_idx in 0..self.attributes.len() {
            let mut m = BitSet::new();
            m.insert(attr_idx);
            let extent = self.index_extent(&m);
            if let Some(node_idx) = lattice
                .poset
                .nodes
                .iter()
                .position(|(node_extent, _)| *node_extent == extent)
            {
                attribute_labels[node_idx].push(self.attributes[attr_idx].clone());
            }
        }

        (0..node_count)
            .map(|i| (object_labels[i].clone(), attribute_labels[i].clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs};

    use crate::FormalContext;

    fn load_living_beings() -> FormalContext<String> {
        FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_concepts_named_matches_index() {
        let ctx = load_living_beings();
        let expected: Vec<(HashSet<String>, HashSet<String>)> = ctx
            .index_next_closure_concepts()
            .map(|(g, m)| {
                let extent = g.iter().map(|i| ctx.objects[i].clone()).collect();
                let intent = m.iter().map(|i| ctx.attributes[i].clone()).collect();
                (extent, intent)
            })
            .collect();
        let named: Vec<_> = ctx.next_closure_concepts().collect();
        assert_eq!(named.len(), expected.len());
        for pair in &named {
            assert!(expected.contains(pair), "concept {:?} not found in index results", pair);
        }
    }

    #[test]
    fn test_fcbo_concepts_named_matches_index() {
        let ctx = load_living_beings();
        let expected: Vec<(HashSet<String>, HashSet<String>)> = ctx
            .index_fcbo_concepts()
            .map(|(g, m)| {
                let extent = g.iter().map(|i| ctx.objects[i].clone()).collect();
                let intent = m.iter().map(|i| ctx.attributes[i].clone()).collect();
                (extent, intent)
            })
            .collect();
        let named: Vec<_> = ctx.fcbo_concepts().collect();
        assert_eq!(named.len(), expected.len());
        for pair in &named {
            assert!(expected.contains(pair), "concept {:?} not found in fcbo index results", pair);
        }
    }

    #[test]
    fn test_canonical_basis_named_matches_index() {
        let ctx = load_living_beings();
        // Both BitSets in the canonical basis are attribute indices
        let expected: Vec<(HashSet<String>, HashSet<String>)> = ctx
            .index_canonical_basis()
            .into_iter()
            .map(|(premise_bits, conclusion_bits)| {
                let premise = premise_bits.iter().map(|i| ctx.attributes[i].clone()).collect();
                let conclusion = conclusion_bits.iter().map(|i| ctx.attributes[i].clone()).collect();
                (premise, conclusion)
            })
            .collect();
        let named = ctx.canonical_basis();
        assert_eq!(named.len(), expected.len());
        for pair in &named {
            assert!(expected.contains(pair), "implication {:?} not in index basis", pair);
        }
    }

    #[test]
    fn test_canonical_basis_optimised_named_matches_index() {
        let ctx = load_living_beings();
        let expected: Vec<(HashSet<String>, HashSet<String>)> = ctx
            .index_canonical_basis_optimised()
            .into_iter()
            .map(|(premise_bits, conclusion_bits)| {
                let premise = premise_bits.iter().map(|i| ctx.attributes[i].clone()).collect();
                let conclusion = conclusion_bits.iter().map(|i| ctx.attributes[i].clone()).collect();
                (premise, conclusion)
            })
            .collect();
        let named = ctx.canonical_basis_optimised();
        assert_eq!(named.len(), expected.len());
        for pair in &named {
            assert!(expected.contains(pair), "implication {:?} not in optimised index basis", pair);
        }
    }

    #[test]
    fn test_upper_neighbor_named_matches_index() {
        let ctx = load_living_beings();
        // Use the empty extent — its upper neighbor is well-defined
        let empty: HashSet<String> = HashSet::new();
        let named = ctx.upper_neighbor(&empty);
        let index_result = ctx.index_upper_neighbor(&bit_set::BitSet::new());
        let expected: HashSet<String> = index_result.iter().map(|i| ctx.objects[i].clone()).collect();
        assert_eq!(named, expected);
    }

    /// This test is ignored in normal runs because `attribute_exploration` reads from stdin
    /// interactively. Run explicitly with `cargo test -- --ignored` when stdin is available.
    #[test]
    #[ignore]
    fn test_attribute_exploration_named_returns_strings() {
        // Use a small hand-built context to avoid interactive stdin in tests
        use bit_set::BitSet;
        let mut ctx = FormalContext::<String>::new();
        ctx.add_attribute("a".to_string(), &BitSet::new());
        ctx.add_attribute("b".to_string(), &BitSet::new());
        let full: BitSet = [0, 1].into_iter().collect();
        ctx.add_object("obj1".to_string(), &full);

        let basis = ctx.attribute_exploration();
        // All premises and conclusions must contain only strings from attributes
        let valid_attrs: HashSet<&str> = ["a", "b"].into_iter().collect();
        for (premise, conclusion) in &basis {
            for name in premise.iter().chain(conclusion.iter()) {
                assert!(valid_attrs.contains(name.as_str()), "unexpected attribute name: {}", name);
            }
        }
    }

    #[test]
    fn test_index_concepts_default_matches_next_closure() {
        let ctx = load_living_beings();
        let shortcut: Vec<_> = ctx.index_concepts().collect();
        let explicit: Vec<_> = ctx.index_next_closure_concepts().collect();
        assert_eq!(shortcut.len(), explicit.len());
        for pair in &shortcut {
            assert!(explicit.contains(pair), "index_concepts default result {:?} missing from index_next_closure_concepts", pair);
        }
    }

    #[test]
    fn test_concepts_default_matches_next_closure() {
        let ctx = load_living_beings();
        let shortcut: Vec<_> = ctx.concepts().collect();
        let explicit: Vec<_> = ctx.next_closure_concepts().collect();
        assert_eq!(shortcut.len(), explicit.len());
        for pair in &shortcut {
            assert!(explicit.contains(pair), "concepts() default result {:?} missing from next_closure_concepts", pair);
        }
    }

    // ── concept_lattice() tests (US2) ─────────────────────────────────────────

    #[test]
    fn test_concept_lattice_node_count() {
        let ctx = load_living_beings();
        let concept_count = ctx.index_concepts().count();
        let lattice = ctx.concept_lattice().expect("concept_lattice returned None");
        assert_eq!(lattice.poset.nodes.len(), concept_count);
    }

    #[test]
    fn test_concept_lattice_top_has_full_extent() {
        let ctx = load_living_beings();
        let lattice = ctx.concept_lattice().unwrap();
        // The top concept (most general) has all objects in its extent.
        let top_extent = &lattice.poset.nodes[lattice.top as usize].0;
        let all_objects: bit_set::BitSet = (0..ctx.objects.len()).collect();
        assert_eq!(*top_extent, all_objects, "top concept should have full extent");
    }

    #[test]
    fn test_concept_lattice_bottom_has_empty_extent() {
        let ctx = load_living_beings();
        let lattice = ctx.concept_lattice().unwrap();
        // The bottom concept (most specific) has the smallest extent.
        // It should have empty extent if no object has all attributes.
        let bottom_extent = &lattice.poset.nodes[lattice.bottom as usize].0;
        // All other extents must be supersets of the bottom extent
        for (i, node) in lattice.poset.nodes.iter().enumerate() {
            if i != lattice.bottom as usize {
                assert!(
                    bottom_extent.is_subset(&node.0),
                    "bottom extent should be ⊆ every other extent"
                );
            }
        }
    }

    #[test]
    fn test_concept_lattice_join_correctness() {
        let ctx = load_living_beings();
        let lattice = ctx.concept_lattice().unwrap();
        let n = lattice.poset.nodes.len();
        if n < 2 {
            return;
        }
        // Check join for a few pairs: join(a, b) must be the ≤-minimum upper bound.
        for a in 0..std::cmp::min(n, 3) {
            for b in (a + 1)..std::cmp::min(n, 4) {
                let j = lattice.join(a as u32, b as u32);
                // j must be an upper bound of both a and b
                assert!(lattice.poset.is_leq(a as u32, j), "join({},{})={} is not ≥ {}", a, b, j, a);
                assert!(lattice.poset.is_leq(b as u32, j), "join({},{})={} is not ≥ {}", a, b, j, b);
                // j must be ≤ every other upper bound
                for c in 0..n {
                    if lattice.poset.is_leq(a as u32, c as u32) && lattice.poset.is_leq(b as u32, c as u32) {
                        assert!(
                            lattice.poset.is_leq(j, c as u32),
                            "join({},{})={} is not ≤ upper bound {}", a, b, j, c
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_concept_lattice_empty_context() {
        // An empty context has exactly one concept (∅, ∅); the lattice has 1 node.
        let ctx = FormalContext::<String>::new();
        let lattice = ctx.concept_lattice();
        assert!(lattice.is_some(), "empty context has the trivial 1-concept lattice");
        assert_eq!(lattice.unwrap().poset.nodes.len(), 1);
    }

    // ── Drawing algorithm tests (US3) ─────────────────────────────────────────

    #[test]
    fn test_dimdraw_coordinate_count() {
        use crate::algorithms::dimdraw::DimDraw;
        use crate::traits::DrawingAlgorithm;
        let ctx = load_living_beings();
        let lattice = ctx.concept_lattice().expect("concept_lattice returned None");
        let concept_count = lattice.poset.nodes.len();
        let drawing = DimDraw { timeout_ms: 0 }
            .draw(&lattice)
            .expect("DimDraw returned None");
        assert_eq!(drawing.coordinates.len(), concept_count);
    }

    #[test]
    fn test_sugiyama_coordinate_count() {
        use crate::algorithms::sugiyama::Sugiyama;
        use crate::traits::DrawingAlgorithm;
        let ctx = load_living_beings();
        let lattice = ctx.concept_lattice().expect("concept_lattice returned None");
        let concept_count = lattice.poset.nodes.len();
        let drawing = Sugiyama { vertex_spacing: 1 }
            .draw(&lattice)
            .expect("Sugiyama returned None");
        assert_eq!(drawing.coordinates.len(), concept_count);
    }

    // ── index_next_preclosure / next_preclosure tests (T003, T005) ────────────

    #[test]
    fn test_index_next_preclosure_method_matches_free_fn() {
        use crate::algorithms::canonical_basis;
        let ctx = load_living_beings();
        let implications = ctx.index_canonical_basis();
        let input = bit_set::BitSet::new();
        let via_method = ctx.index_next_preclosure(&implications, &input);
        let via_free = canonical_basis::index_next_preclosure(&ctx, &implications, &input);
        assert_eq!(via_method, via_free);
    }

    #[test]
    fn test_index_next_preclosure_full_attribute_set_returns_full() {
        let ctx = load_living_beings();
        let implications = ctx.index_canonical_basis();
        let all_attrs: bit_set::BitSet = (0..ctx.attributes.len()).collect();
        let result = ctx.index_next_preclosure(&implications, &all_attrs);
        // The full attribute set is the terminal element: next preclosure of it is the full set.
        assert_eq!(result, all_attrs);
    }

    #[test]
    fn test_next_preclosure_named_matches_index() {
        let ctx = load_living_beings();
        let index_implications = ctx.index_canonical_basis();
        let named_implications = ctx.canonical_basis();
        let input_index = bit_set::BitSet::new();
        let input_named: HashSet<String> = HashSet::new();

        let via_index = ctx.index_next_preclosure(&index_implications, &input_index);
        let expected_named: HashSet<String> =
            via_index.iter().map(|i| ctx.attributes[i].clone()).collect();
        let via_named = ctx.next_preclosure(&named_implications, &input_named);
        assert_eq!(via_named, expected_named);
    }

    #[test]
    fn test_next_preclosure_unknown_names_silently_dropped() {
        let ctx = load_living_beings();
        let named_implications = ctx.canonical_basis();
        // Input with one valid attribute and one fabricated name.
        let mut input_named: HashSet<String> = HashSet::new();
        input_named.insert("nonexistent_attribute_xyz".to_string());
        // Should not panic; unknown name is dropped, result = next_preclosure({}).
        let input_empty: HashSet<String> = HashSet::new();
        let result_unknown = ctx.next_preclosure(&named_implications, &input_named);
        let result_empty = ctx.next_preclosure(&named_implications, &input_empty);
        assert_eq!(result_unknown, result_empty,
            "unknown names should be silently dropped, result equals next_preclosure(empty)");
    }

    #[test]
    fn test_next_preclosure_empty_implications_empty_input() {
        let ctx = load_living_beings();
        // No implications: closure of empty set under no rules = empty set.
        // next_preclosure of {} under {} is the first lecticaly non-empty set,
        // which is the singleton {last attribute} if it is closed (no implications to fire).
        let empty_implications: Vec<(HashSet<String>, HashSet<String>)> = Vec::new();
        let input_named: HashSet<String> = HashSet::new();
        let result = ctx.next_preclosure(&empty_implications, &input_named);
        // Verify consistency with index variant
        let index_implications: Vec<(bit_set::BitSet, bit_set::BitSet)> = Vec::new();
        let index_input = bit_set::BitSet::new();
        let index_result = ctx.index_next_preclosure(&index_implications, &index_input);
        let expected: HashSet<String> = index_result.iter().map(|i| ctx.attributes[i].clone()).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dimdraw_trivial_single_node() {
        use crate::algorithms::dimdraw::DimDraw;
        use crate::data_structures::{lattice::Lattice, poset::Poset};
        use crate::traits::DrawingAlgorithm;
        let poset = Poset::from_covering_relation(vec![0usize], vec![]).unwrap();
        let lattice = Lattice::from_poset(poset).unwrap();
        let drawing = DimDraw { timeout_ms: 0 }.draw(&lattice);
        assert!(drawing.is_some());
        assert_eq!(drawing.unwrap().coordinates.len(), 1);
    }

    #[test]
    fn test_sugiyama_trivial_single_node() {
        use crate::algorithms::sugiyama::Sugiyama;
        use crate::data_structures::{lattice::Lattice, poset::Poset};
        use crate::traits::DrawingAlgorithm;
        let poset = Poset::from_covering_relation(vec![0usize], vec![]).unwrap();
        let lattice = Lattice::from_poset(poset).unwrap();
        let drawing = Sugiyama { vertex_spacing: 1 }.draw(&lattice);
        assert!(drawing.is_some());
        assert_eq!(drawing.unwrap().coordinates.len(), 1);
    }

    #[test]
    fn test_dimdraw_chain_axis_aligned() {
        use crate::algorithms::dimdraw::DimDraw;
        use crate::data_structures::{lattice::Lattice, poset::Poset};
        use crate::traits::DrawingAlgorithm;

        // 0 < 1 < 2
        let poset = Poset::from_covering_relation(vec![0usize, 1usize, 2usize], vec![(0, 1), (1, 2)]).unwrap();
        let lattice = Lattice::from_poset(poset).unwrap();
        let drawing = DimDraw { timeout_ms: 0 }
            .draw(&lattice)
            .expect("DimDraw returned None");

        assert_eq!(drawing.coordinates.len(), 3);

        let x0 = drawing.coordinates[0].0;
        let y0 = drawing.coordinates[0].1;
        let all_x_same = drawing.coordinates.iter().all(|(x, _)| (*x - x0).abs() < f64::EPSILON);
        let all_y_same = drawing.coordinates.iter().all(|(_, y)| (*y - y0).abs() < f64::EPSILON);

        assert!(
            all_x_same || all_y_same,
            "expected chain coordinates to be axis-aligned, got {:?}",
            drawing.coordinates
        );
    }

    #[test]
    fn test_sugiyama_chain_orientation_matches_dimdraw() {
        use crate::algorithms::dimdraw::DimDraw;
        use crate::algorithms::sugiyama::Sugiyama;
        use crate::data_structures::{lattice::Lattice, poset::Poset};
        use crate::traits::DrawingAlgorithm;

        // 0 < 1 < 2 where 2 is top and 0 is bottom.
        let poset = Poset::from_covering_relation(vec![0usize, 1usize, 2usize], vec![(0, 1), (1, 2)]).unwrap();
        let lattice = Lattice::from_poset(poset).unwrap();

        let dimdraw = DimDraw { timeout_ms: 0 }
            .draw(&lattice)
            .expect("DimDraw returned None");
        let sugiyama = Sugiyama { vertex_spacing: 1 }
            .draw(&lattice)
            .expect("Sugiyama returned None");

        assert_eq!(dimdraw.coordinates.len(), 3);
        assert_eq!(sugiyama.coordinates.len(), 3);

        let top = lattice.top as usize;
        let bottom = lattice.bottom as usize;

        let dy_dimdraw = dimdraw.coordinates[top].1 - dimdraw.coordinates[bottom].1;
        let dy_sugiyama = sugiyama.coordinates[top].1 - sugiyama.coordinates[bottom].1;

        assert!(
            dy_dimdraw.abs() > f64::EPSILON,
            "DimDraw top/bottom collapsed: {:?}",
            dimdraw.coordinates
        );
        assert!(
            dy_sugiyama.abs() > f64::EPSILON,
            "Sugiyama top/bottom collapsed: {:?}",
            sugiyama.coordinates
        );
        assert!(
            dy_dimdraw.signum() == dy_sugiyama.signum(),
            "orientation mismatch: dimdraw={:?}, sugiyama={:?}",
            dimdraw.coordinates,
            sugiyama.coordinates
        );
    }

    #[test]
    fn test_dimdraw_stops_after_timeout_budget() {
        use crate::algorithms::dimdraw::DimDraw;

        let ctx = load_living_beings();
        let lattice = ctx.concept_lattice().expect("concept_lattice returned None");

        let solver = DimDraw { timeout_ms: 1 };
        let started = std::time::Instant::now();
        let out = solver
            .solve_with_stats(&lattice)
            .expect("DimDraw should return an outcome");
        let elapsed = started.elapsed();

        assert!(
            elapsed <= std::time::Duration::from_millis(800),
            "expected bounded run to finish quickly, elapsed={:?}",
            elapsed
        );
        assert!(out.explored_nodes > 0, "expected at least one explored node");
    }

    #[test]
    fn test_dimdraw_early_timeout_uses_heuristic_baseline() {
        use crate::algorithms::dimdraw::DimDraw;

        let ctx = load_living_beings();
        let lattice = ctx.concept_lattice().expect("concept_lattice returned None");

        let out = DimDraw { timeout_ms: 1 }
            .solve_with_stats(&lattice)
            .expect("DimDraw should return an outcome");

        assert!(
            out.best_cost <= out.baseline_cost,
            "best cost {} should be <= baseline {}",
            out.best_cost,
            out.baseline_cost
        );
        assert_eq!(out.drawing.coordinates.len(), lattice.poset.nodes.len());
    }

    #[test]
    fn test_dimdraw_zero_timeout_runs_unbounded_mode() {
        use crate::algorithms::dimdraw::DimDraw;

        let ctx = load_living_beings();
        let lattice = ctx.concept_lattice().expect("concept_lattice returned None");

        let out = DimDraw { timeout_ms: 0 }
            .solve_with_stats(&lattice)
            .expect("DimDraw should return an outcome");

        assert!(
            !out.timed_out,
            "timeout_ms=0 should keep search in unbounded mode"
        );
    }

    #[test]
    fn test_dimdraw_addition_keeps_dimdraw_and_sugiyama_compatible() {
        use crate::algorithms::dimdraw::DimDraw;
        use crate::algorithms::sugiyama::Sugiyama;
        use crate::traits::DrawingAlgorithm;

        let ctx = load_living_beings();
        let lattice = ctx.concept_lattice().expect("concept_lattice returned None");
        let concept_count = lattice.poset.nodes.len();

        let sugiyama = Sugiyama { vertex_spacing: 1 }
            .draw(&lattice)
            .expect("Sugiyama returned None");
        let fast = DimDraw { timeout_ms: 10 }
            .draw(&lattice)
            .expect("DimDraw returned None");

        assert_eq!(sugiyama.coordinates.len(), concept_count);
        assert_eq!(fast.coordinates.len(), concept_count);
    }

    #[test]
    fn test_attribute_exploration_with_callback_always_accept_equals_canonical_basis() {
        let mut ctx = load_living_beings();

        // Always-accept callback: every implication is accepted as-is.
        let callback_basis = ctx.index_attribute_exploration_with_callback(|_, _| None);
        let canonical_basis = {
            let ctx2 = load_living_beings();
            ctx2.index_canonical_basis()
        };

        // Sort both for stable comparison (exploration order may differ).
        let mut cb_sorted: Vec<(bit_set::BitSet, bit_set::BitSet)> = callback_basis;
        let mut ca_sorted: Vec<(bit_set::BitSet, bit_set::BitSet)> = canonical_basis;
        let key = |p: &(bit_set::BitSet, bit_set::BitSet)| -> Vec<usize> {
            let mut v: Vec<usize> = p.0.iter().chain(p.1.iter()).collect();
            v.sort();
            v
        };
        cb_sorted.sort_by(|a, b| key(a).cmp(&key(b)));
        ca_sorted.sort_by(|a, b| key(a).cmp(&key(b)));
        assert_eq!(cb_sorted, ca_sorted, "always-accept callback should produce the canonical basis");
    }
}

