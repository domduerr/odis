//! # odis — Formal Concept Analysis in Rust
//!
//! `odis` is a library for [Formal Concept Analysis](https://en.wikipedia.org/wiki/Formal_concept_analysis).
//! It operates on formal contexts (objects × attributes incidence tables) and provides
//! concept enumeration, implication bases, lattice drawing, and attribute exploration.
//!
//! ## Quick start
//!
//! ```
//! use odis::{FormalContext, algorithms::NextClosure};
//! use odis::traits::ConceptEnumerator;
//!
//! // Burmeister (.cxt) format: 2 objects, 2 attributes
//! let bytes = b"B\n\n2\n2\n\nbird\nfish\nflies\nlives_in_water\nX.\n.X\n";
//! let ctx = FormalContext::<String>::from(bytes).unwrap();
//!
//! let concepts: Vec<_> = NextClosure.enumerate_concepts(&ctx).collect();
//! assert_eq!(concepts.len(), 4); // ∅, {bird}, {fish}, {bird, fish}
//! ```
//!
//! ## Modules
//!
//! - [`algorithms`] — Concept enumeration, implication bases, lattice drawing, exploration
//! - [`traits`] — Extension traits: [`traits::ConceptEnumerator`], [`traits::DrawingAlgorithm`],
//!   [`traits::ImplicationEngine`], [`traits::IcebergConceptEnumerator`]
//!
//! ## Key types
//!
//! - [`FormalContext`] — objects, attributes, incidence relation
//! - [`Lattice`] / [`Poset`] — concept lattice and partially ordered set
//! - [`Drawing`] — 2-D coordinate output from a layout algorithm
//! - [`IcebergLattice`] — concept lattice filtered by minimum support

#![warn(missing_docs)]

pub mod algorithms;
mod data_structures;
pub mod traits;

pub use data_structures::digraph::{DiNode, Digraph};
pub use data_structures::drawing::Drawing;
pub use data_structures::formal_context::{FormalContext, FormatError};
pub use data_structures::iceberg_lattice::IcebergLattice;
pub use data_structures::lattice::Lattice;
pub use data_structures::poset::{Poset, PosetError};
pub use traits::{ConceptEnumerator, DimensionAlgorithm, DrawingAlgorithm, IcebergConceptEnumerator, ImplicationEngine};
