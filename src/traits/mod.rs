//! Traits for the algorithm types in [`crate::algorithms`].
//!
//! | Trait | Implementors |
//! |---|---|
//! | [`ConceptEnumerator`] | [`crate::algorithms::NextClosure`], [`crate::algorithms::Fcbo`] |
//! | [`DrawingAlgorithm`] | [`crate::algorithms::DimDraw`], [`crate::algorithms::Sugiyama`] |
//! | [`ConceptDrawingAlgorithm`] | [`crate::algorithms::DimFlux`] |
//! | [`ImplicationEngine`] | [`crate::algorithms::CanonicalBasis`] |
//! | [`IcebergConceptEnumerator`] | [`crate::algorithms::Titanic`] |

pub mod concept_drawing_algorithm;
pub mod concept_enumerator;
pub mod drawing_algorithm;
pub mod iceberg_enumerator;
pub mod implication_engine;

pub use concept_drawing_algorithm::ConceptDrawingAlgorithm;
pub use concept_enumerator::ConceptEnumerator;
pub use drawing_algorithm::DrawingAlgorithm;
pub use iceberg_enumerator::IcebergConceptEnumerator;
pub use implication_engine::ImplicationEngine;
