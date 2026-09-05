/// A node in a directed graph.
#[derive(PartialEq, Debug, Clone)]
pub struct DiNode<T> {
    /// Position index of the node in the parent `Digraph.nodes` vector.
    pub id: usize,
    /// Data associated with this node.
    pub label: T,
}

/// A directed graph.
///
/// Nodes are indexed by their position in `nodes` (0-based).
/// Edges are directed: `(u, v)` means u → v.
pub struct Digraph<T> {
    /// All nodes in the graph, indexed by position (0-based).
    pub nodes: Vec<DiNode<T>>,
    /// All directed edges as `(u, v)` index pairs (u → v).
    pub edges: Vec<(u32, u32)>,
}

impl<T: Clone> Digraph<T> {
    /// Creates a new `Digraph` with the given nodes and directed edges.
    ///
    /// Edges are stored as `(u, v)` pairs where `u` and `v` are node indices
    /// into the `nodes` vector (0-based).
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::{Digraph, DiNode};
    ///
    /// let nodes = vec![
    ///     DiNode { id: 0, label: "a" },
    ///     DiNode { id: 1, label: "b" },
    /// ];
    /// let g = Digraph::new(nodes, vec![(0, 1)]);
    /// assert_eq!(g.node_count(), 2);
    /// assert_eq!(g.edge_count(), 1);
    /// ```
    pub fn new(nodes: Vec<DiNode<T>>, edges: Vec<(u32, u32)>) -> Self {
        Digraph { nodes, edges }
    }

    /// Returns an iterator over the IDs of all direct successors of `node_id`.
    ///
    /// A successor of `u` is any node `v` such that the edge `(u, v)` exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::{Digraph, DiNode};
    ///
    /// let nodes = vec![
    ///     DiNode { id: 0, label: "a" },
    ///     DiNode { id: 1, label: "b" },
    ///     DiNode { id: 2, label: "c" },
    /// ];
    /// let g = Digraph::new(nodes, vec![(0, 1), (0, 2)]);
    /// let mut succs: Vec<u32> = g.successors(0).collect();
    /// succs.sort();
    /// assert_eq!(succs, vec![1, 2]);
    /// ```
    pub fn successors(&self, node_id: u32) -> impl Iterator<Item = u32> + '_ {
        self.edges
            .iter()
            .filter(move |&&(u, _)| u == node_id)
            .map(|&(_, v)| v)
    }

    /// Returns an iterator over the IDs of all direct predecessors of `node_id`.
    ///
    /// A predecessor of `v` is any node `u` such that the edge `(u, v)` exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::{Digraph, DiNode};
    ///
    /// let nodes = vec![
    ///     DiNode { id: 0, label: "a" },
    ///     DiNode { id: 1, label: "b" },
    /// ];
    /// let g = Digraph::new(nodes, vec![(0, 1)]);
    /// let preds: Vec<u32> = g.predecessors(1).collect();
    /// assert_eq!(preds, vec![0]);
    /// ```
    pub fn predecessors(&self, node_id: u32) -> impl Iterator<Item = u32> + '_ {
        self.edges
            .iter()
            .filter(move |&&(_, v)| v == node_id)
            .map(|&(u, _)| u)
    }

    /// Returns the number of nodes in this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::{Digraph, DiNode};
    ///
    /// let g: Digraph<&str> = Digraph::new(
    ///     vec![DiNode { id: 0, label: "x" }],
    ///     vec![],
    /// );
    /// assert_eq!(g.node_count(), 1);
    /// ```
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of directed edges in this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::{Digraph, DiNode};
    ///
    /// let nodes = vec![
    ///     DiNode { id: 0, label: "a" },
    ///     DiNode { id: 1, label: "b" },
    /// ];
    /// let g = Digraph::new(nodes, vec![(0, 1)]);
    /// assert_eq!(g.edge_count(), 1);
    /// ```
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: usize) -> DiNode<usize> {
        DiNode { id, label: id }
    }

    #[test]
    fn test_empty_graph() {
        let g: Digraph<usize> = Digraph::new(vec![], vec![]);
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.successors(0).count(), 0);
        assert_eq!(g.predecessors(0).count(), 0);
    }

    #[test]
    fn test_chain_graph() {
        // a(0) → b(1) → c(2)
        let g = Digraph::new(
            vec![make_node(0), make_node(1), make_node(2)],
            vec![(0, 1), (1, 2)],
        );
        let succ_a: Vec<u32> = g.successors(0).collect();
        let succ_b: Vec<u32> = g.successors(1).collect();
        let succ_c: Vec<u32> = g.successors(2).collect();
        assert_eq!(succ_a, vec![1]);
        assert_eq!(succ_b, vec![2]);
        assert_eq!(succ_c, vec![]);

        let pred_a: Vec<u32> = g.predecessors(0).collect();
        let pred_b: Vec<u32> = g.predecessors(1).collect();
        assert_eq!(pred_a, vec![]);
        assert_eq!(pred_b, vec![0]);
    }

    #[test]
    fn test_diamond_graph() {
        // a(0) → b(1), a(0) → c(2), b(1) → d(3), c(2) → d(3)
        let g = Digraph::new(
            vec![make_node(0), make_node(1), make_node(2), make_node(3)],
            vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        );
        let mut succ_a: Vec<u32> = g.successors(0).collect();
        succ_a.sort();
        assert_eq!(succ_a, vec![1, 2]);

        let mut pred_d: Vec<u32> = g.predecessors(3).collect();
        pred_d.sort();
        assert_eq!(pred_d, vec![1, 2]);

        assert_eq!(g.predecessors(0).count(), 0);
        assert_eq!(g.successors(3).count(), 0);
    }
}
