//! Sugiyama hierarchical graph layout algorithm.

use crate::data_structures::{drawing::Drawing, lattice::Lattice, poset::Poset};
use crate::traits::DrawingAlgorithm;

/// Configuration for the Sugiyama hierarchical layout algorithm.
///
/// Delegates to the `rust_sugiyama` crate.
pub struct Sugiyama {
    /// Minimum horizontal spacing between vertices in the same layer.
    pub vertex_spacing: usize,
}

impl Sugiyama {
    fn layout_from_edges(&self, edges: &[(u32, u32)], n: usize) -> Option<Drawing> {
        if n == 0 {
            return None;
        }

        if edges.is_empty() {
            // All nodes are isolated — place them in a row.
            let coords: Vec<(f64, f64)> = (0..n)
                .map(|i| (i as f64 * self.vertex_spacing as f64, 0.0))
                .collect();
            return Some(Drawing::new(coords));
        }

        let layouts = rust_sugiyama::from_edges(edges)
            .vertex_spacing(self.vertex_spacing)
            .build();

        let mut coords = vec![(0.0f64, 0.0f64); n];
        let mut x_offset = 0.0f64;

        for (layout_points, width, _height) in layouts {
            if layout_points.is_empty() {
                continue;
            }
            for (node_id, (x, y)) in &layout_points {
                let idx = *node_id;
                if idx < n {
                    coords[idx] = (x_offset + *x as f64, *y as f64);
                }
            }
            x_offset += width as f64 + self.vertex_spacing as f64;
        }

        Some(Drawing::new(coords))
    }
}

impl DrawingAlgorithm for Sugiyama {
    fn draw<T>(&self, lattice: &Lattice<T>) -> Option<Drawing> {
        self.layout_from_edges(&lattice.poset.covering_edges, lattice.poset.nodes.len())
    }

    /// Draw a poset, including disconnected ones (multiple components are placed
    /// side-by-side with a horizontal gap equal to `vertex_spacing`).
    fn draw_poset<T: Clone>(&self, poset: &Poset<T>) -> Option<Drawing> {
        self.layout_from_edges(&poset.covering_edges, poset.nodes.len())
    }
}
