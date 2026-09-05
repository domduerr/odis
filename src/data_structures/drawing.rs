/// A positional layout: maps each node (by Vec index) to a 2D coordinate.
///
/// Coordinates are in the algorithm's native space (not scaled to viewport).
/// Viewport scaling is the consumer's responsibility (e.g., `odis-web`).
///
/// `coordinates[i]` is the position of the node with ID `i`.
pub struct Drawing {
    /// Coordinates for each node, indexed by node ID.
    pub coordinates: Vec<(f64, f64)>,
}

impl Drawing {
    /// Creates a new `Drawing` from a vector of `(x, y)` coordinates.
    ///
    /// The index of each pair corresponds to a node ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Drawing;
    ///
    /// let d = Drawing::new(vec![(0.0, 0.0), (1.0, 2.0)]);
    /// assert_eq!(d.coordinates.len(), 2);
    /// ```
    pub fn new(coordinates: Vec<(f64, f64)>) -> Self {
        Drawing { coordinates }
    }

    /// Linearly scale the raw algorithm coordinates to fit a viewport of the given
    /// dimensions. Returns a parallel `Vec<(f64, f64)>` where each pair is the
    /// scaled `(x, y)` position of the corresponding node in pixels.
    ///
    /// Coordinates are mapped so that the graph fills `[margin, dim-margin]` on
    /// each axis. When all source points are identical (degenerate case) the
    /// result places every node at the canvas centre.
    ///
    /// Positions that land on the same pixel are nudged apart by 20 px per
    /// collision so that nodes are not drawn on top of each other. The nudge
    /// stays inside the viewport: enough coincident nodes exhaust the room for
    /// it and end up overlapping again, which is the lesser of the two evils
    /// against pushing them off the canvas.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::Drawing;
    ///
    /// let d = Drawing::new(vec![(0.0, 0.0), (10.0, 10.0)]);
    /// let scaled = d.scale_to_viewport(100.0, 100.0, 10.0);
    /// assert_eq!(scaled.len(), 2);
    /// // Both points fall within [margin, dim-margin]
    /// for &(x, y) in &scaled {
    ///     assert!(x >= 10.0 && x <= 90.0);
    ///     assert!(y >= 10.0 && y <= 90.0);
    /// }
    /// ```
    pub fn scale_to_viewport(&self, width: f64, height: f64, margin: f64) -> Vec<(f64, f64)> {
        let coords = &self.coordinates;
        if coords.is_empty() {
            return Vec::new();
        }

        let min_x = coords.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
        let max_x = coords.iter().map(|c| c.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = coords.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        let max_y = coords.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);

        let graph_width = max_x - min_x;
        let graph_height = max_y - min_y;

        let available_width = (width - 2.0 * margin).max(1.0);
        let available_height = (height - 2.0 * margin).max(1.0);

        let x_coef = if graph_width > 0.0 {
            available_width / graph_width
        } else {
            1.0
        };
        let y_coef = if graph_height > 0.0 {
            available_height / graph_height
        } else {
            1.0
        };

        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;
        let canvas_center_x = width / 2.0;
        let canvas_center_y = height / 2.0;

        let mut position_counts: std::collections::HashMap<(i32, i32), usize> =
            std::collections::HashMap::new();

        let max_x = (width - margin).max(margin);
        let max_y = (height - margin).max(margin);

        coords
            .iter()
            .map(|&(x, y)| {
                let scaled_x = (x - center_x) * x_coef + canvas_center_x;
                let scaled_y = (y - center_y) * y_coef + canvas_center_y;
                let pos_key = (scaled_x as i32, scaled_y as i32);
                let collisions = position_counts.entry(pos_key).or_insert(0);
                let offset = *collisions as f64 * 20.0;
                *collisions += 1;
                (
                    (scaled_x + offset).clamp(margin, max_x),
                    (scaled_y + offset).clamp(margin, max_y),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scaled coordinates are documented to land inside
    /// `[margin, dim - margin]`. The collision nudge used to be unbounded, so a
    /// handful of coincident nodes walked straight off the canvas.
    #[test]
    fn test_the_collision_nudge_stays_inside_the_viewport() {
        let mut points = vec![(0.0, 0.0)];
        points.extend(std::iter::repeat_n((10.0, 10.0), 8));
        let scaled = Drawing::new(points).scale_to_viewport(100.0, 100.0, 10.0);

        for &(x, y) in &scaled {
            assert!(
                (10.0..=90.0).contains(&x) && (10.0..=90.0).contains(&y),
                "({x}, {y}) is outside the documented box"
            );
        }
    }

    /// Nudging must still do its job where there is room for it.
    #[test]
    fn test_coincident_nodes_are_pulled_apart_when_there_is_room() {
        let scaled = Drawing::new(vec![(0.0, 0.0), (0.0, 0.0), (100.0, 100.0)])
            .scale_to_viewport(400.0, 400.0, 10.0);
        assert_ne!(scaled[0], scaled[1]);
    }
}
