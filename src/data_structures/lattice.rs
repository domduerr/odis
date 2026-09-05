use crate::data_structures::poset::Poset;

/// A complete lattice: a poset with a unique top (⊤) and bottom (⊥) element,
/// closed under binary join (∨) and meet (∧).
pub struct Lattice<T> {
    /// The underlying poset (covering relation, transitive closure, and node list).
    pub poset: Poset<T>,
    /// Index of the top element (greatest element, ⊤).
    pub top: u32,
    /// Index of the bottom element (least element, ⊥).
    pub bottom: u32,
}

impl<T: Clone> Lattice<T> {
    /// Construct a `Lattice` from a `Poset`.
    ///
    /// Returns `None` if the poset does not have a unique top and unique bottom
    /// element (i.e., it is not a complete lattice).
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::{Poset, Lattice};
    ///
    /// // Diamond: bottom(0) ≺ left(1), bottom(0) ≺ right(2), left(1) ≺ top(3), right(2) ≺ top(3)
    /// let poset = Poset::from_covering_relation(
    ///     vec!["bot", "l", "r", "top"],
    ///     vec![(0, 1), (0, 2), (1, 3), (2, 3)],
    /// ).unwrap();
    /// let lattice = Lattice::from_poset(poset).unwrap();
    /// assert_eq!(lattice.bottom, 0);
    /// assert_eq!(lattice.top, 3);
    /// ```
    pub fn from_poset(poset: Poset<T>) -> Option<Self> {
        let n = poset.nodes.len();
        if n == 0 {
            return None;
        }

        // top = the one node with nothing strictly above it, bottom = the one
        // with nothing strictly below. Read off the reachability index: a node
        // has something above it exactly if its row is non-empty, and something
        // below it exactly if it appears in some row.
        let mut has_below = bit_set::BitSet::with_capacity(n);
        for above in &poset.reach {
            has_below.union_with(above);
        }
        let top_candidates: Vec<u32> = (0..n as u32)
            .filter(|&u| poset.reach[u as usize].is_empty())
            .collect();
        let bottom_candidates: Vec<u32> = (0..n as u32)
            .filter(|&u| !has_below.contains(u as usize))
            .collect();

        if top_candidates.len() != 1 || bottom_candidates.len() != 1 {
            return None;
        }

        Some(Lattice {
            poset,
            top: top_candidates[0],
            bottom: bottom_candidates[0],
        })
    }

    /// Returns the least upper bound (join, ∨) of nodes `a` and `b`.
    ///
    /// Collects all nodes that are ≥ both `a` and `b`, then returns the minimum
    /// among them (the node that is ≤ all other common upper bounds).
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::{Poset, Lattice};
    ///
    /// let poset = Poset::from_covering_relation(
    ///     vec!["bot", "l", "r", "top"],
    ///     vec![(0, 1), (0, 2), (1, 3), (2, 3)],
    /// ).unwrap();
    /// let lattice = Lattice::from_poset(poset).unwrap();
    /// // join of the two incomparable middle nodes is the top
    /// assert_eq!(lattice.join(1, 2), 3);
    /// // join of a node with itself is itself
    /// assert_eq!(lattice.join(0, 0), 0);
    /// ```
    pub fn join(&self, a: u32, b: u32) -> u32 {
        if a == b {
            return a;
        }
        // upper bounds of a: all v such that a ≤ v
        // upper bounds of b: all v such that b ≤ v
        // common upper bounds = intersection; join = the minimum one
        let is_upper_bound = |v: u32| self.poset.is_leq(a, v) && self.poset.is_leq(b, v);
        let n = self.poset.nodes.len() as u32;
        let upper_bounds: Vec<u32> = (0..n).filter(|&v| is_upper_bound(v)).collect();
        // The join is the unique lower bound of all upper bounds
        upper_bounds
            .iter()
            .cloned()
            .find(|&v| upper_bounds.iter().all(|&w| self.poset.is_leq(v, w)))
            .unwrap_or(self.top)
    }

    /// Returns the greatest lower bound (meet, ∧) of nodes `a` and `b`.
    ///
    /// Collects all nodes that are ≤ both `a` and `b`, then returns the maximum
    /// among them (the node that is ≥ all other common lower bounds).
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::{Poset, Lattice};
    ///
    /// let poset = Poset::from_covering_relation(
    ///     vec!["bot", "l", "r", "top"],
    ///     vec![(0, 1), (0, 2), (1, 3), (2, 3)],
    /// ).unwrap();
    /// let lattice = Lattice::from_poset(poset).unwrap();
    /// // meet of the two incomparable middle nodes is the bottom
    /// assert_eq!(lattice.meet(1, 2), 0);
    /// // meet of a node with itself is itself
    /// assert_eq!(lattice.meet(3, 3), 3);
    /// ```
    pub fn meet(&self, a: u32, b: u32) -> u32 {
        if a == b {
            return a;
        }
        let is_lower_bound = |v: u32| self.poset.is_leq(v, a) && self.poset.is_leq(v, b);
        let n = self.poset.nodes.len() as u32;
        let lower_bounds: Vec<u32> = (0..n).filter(|&v| is_lower_bound(v)).collect();
        lower_bounds
            .iter()
            .cloned()
            .find(|&v| lower_bounds.iter().all(|&w| self.poset.is_leq(w, v)))
            .unwrap_or(self.bottom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_structures::poset::Poset;

    fn diamond_lattice() -> Lattice<usize> {
        // bottom(0) ≺ left(1), bottom(0) ≺ right(2), left(1) ≺ top(3), right(2) ≺ top(3)
        let poset = Poset::from_covering_relation(
            vec![0usize, 1, 2, 3],
            vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        )
        .unwrap();
        Lattice::from_poset(poset).unwrap()
    }

    #[test]
    fn test_from_poset_diamond() {
        let l = diamond_lattice();
        assert_eq!(l.bottom, 0);
        assert_eq!(l.top, 3);
    }

    #[test]
    fn test_join_diamond() {
        let l = diamond_lattice();
        assert_eq!(l.join(1, 2), 3); // left ∨ right = top
        assert_eq!(l.join(0, 1), 1); // bottom ∨ left = left
        assert_eq!(l.join(0, 0), 0); // idempotent
        assert_eq!(l.join(0, 3), 3); // bottom ∨ top = top
    }

    #[test]
    fn test_meet_diamond() {
        let l = diamond_lattice();
        assert_eq!(l.meet(1, 2), 0); // left ∧ right = bottom
        assert_eq!(l.meet(1, 3), 1); // left ∧ top = left
        assert_eq!(l.meet(0, 3), 0); // bottom ∧ top = bottom
    }

    #[test]
    fn test_trivial_single_node() {
        let poset = Poset::from_covering_relation(vec![42usize], vec![]).unwrap();
        let l = Lattice::from_poset(poset).unwrap();
        assert_eq!(l.top, 0);
        assert_eq!(l.bottom, 0);
        assert_eq!(l.join(0, 0), 0);
        assert_eq!(l.meet(0, 0), 0);
    }

    #[test]
    fn test_non_lattice_returns_none() {
        // Two-element antichain: no edges, two nodes → two maxima, two minima
        let poset = Poset::from_covering_relation(vec![0usize, 1], vec![]).unwrap();
        assert!(Lattice::from_poset(poset).is_none());
    }
}
