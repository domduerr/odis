//! Traits for the algorithm types in [`crate::algorithms`].
//!
//! | Trait | Implementors |
//! |---|---|
//! | [`ConceptEnumerator`] | [`crate::algorithms::NextClosure`], [`crate::algorithms::Fcbo`] |
//! | [`DimensionAlgorithm`] | [`crate::algorithms::dimension::SatReduction`], [`crate::algorithms::dimension::HypergraphColoring`], [`crate::algorithms::dimension::GraphColoring`], [`crate::algorithms::dimension::GraphColoringHeuristic`], [`crate::algorithms::dimension::Hybrid`] |
//! | [`DrawingAlgorithm`] | [`crate::algorithms::DimDraw`], [`crate::algorithms::Sugiyama`] |
//! | [`ImplicationEngine`] | [`crate::algorithms::CanonicalBasis`] |
//! | [`IcebergConceptEnumerator`] | [`crate::algorithms::Titanic`] |

pub mod concept_enumerator;
pub mod dimension_algorithm;
pub mod drawing_algorithm;
pub mod iceberg_enumerator;
pub mod implication_engine;

pub use concept_enumerator::ConceptEnumerator;
pub use dimension_algorithm::DimensionAlgorithm;
pub use drawing_algorithm::DrawingAlgorithm;
pub use iceberg_enumerator::IcebergConceptEnumerator;
pub use implication_engine::ImplicationEngine;
