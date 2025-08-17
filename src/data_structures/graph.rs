use bit_set::BitSet;

use crate::FormalContext;

/// Graphs are important
pub struct Graph<T> {
    pub width: usize,
    pub height: usize,
    pub edges: Vec<(u32, u32)>,
    pub nodes: Vec<Node<T>>,
}

#[derive(PartialEq)]
pub struct Node<T> {
    pub id: usize,
    pub x: usize,
    pub y: usize,
    pub label: (Option<Vec<T>>, Option<Vec<T>>),
}

struct Task<'a> {
    set_index: usize,
    set: &'a BitSet,
}

impl<'a> PartialEq for Task<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.set == other.set
    }
}

impl<T: Clone> Graph<T> {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Graph {
            width: 0,
            height: 0,
            edges: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Creates a Graph from a set of concepts and their context.
    pub fn from_concepts(
        concepts: &Vec<(BitSet, BitSet)>,
        context: &FormalContext<T>,
    ) -> Option<Self> {
        let concepts: Vec<BitSet> = concepts.iter().map(|x| x.0.clone()).collect();

        let mut edges: Vec<(u32, u32)> = Vec::new();
        let mut queue: Vec<Task> = Vec::new();
        let mut root_index = concepts.len() - 1;

        if root_index == 0 {
            return None;
        }

        'a: loop {
            let lenght;
            if queue.len() > 0 {
                lenght = queue.len()
            } else {
                lenght = 1;
            }

            for _ in 0..lenght {
                let obj_list = FormalContext::upper_neighbor(&context, &concepts[root_index]);
                for n in &obj_list {
                    let mut set_n = BitSet::new();
                    set_n.insert(n);

                    let concept = FormalContext::index_object_hull(
                        context,
                        &set_n.union(&concepts[root_index]).collect(),
                    );

                    let set_index = concepts.iter().position(|x| *x == concept).unwrap();

                    let new_task = Task {
                        set_index: set_index,
                        set: &concepts[set_index],
                    };

                    if !queue.contains(&new_task) {
                        queue.push(new_task);
                        edges.push((set_index as u32, root_index as u32));
                    }
                }
                if queue.len() != 0 {
                    let task = queue.pop().unwrap();
                    root_index = task.set_index;
                } else {
                    break 'a;
                }
            }
        }

        let mut obj_labels = Vec::new();

        for obj in 0..context.objects.len() {
            let mut g = BitSet::new();
            g.insert(obj);
            let index = concepts
                .iter()
                .position(|x| *x == context.index_object_hull(&g))
                .unwrap();
            obj_labels.push((index, obj));
        }

        let mut attr_labels = Vec::new();

        for attr in 0..context.attributes.len() {
            let mut m = BitSet::new();
            m.insert(attr);
            let index = concepts
                .iter()
                .position(|x| *x == context.index_attribute_derivation(&m))
                .unwrap();
            attr_labels.push((index, attr));
        }

        let (points, width, height) = rust_sugiyama::from_edges(&edges)
            .vertex_spacing(1)
            .vertex_spacing(1)
            .build()
            .remove(0);

        let mut nodes: Vec<Node<T>> = points
            .iter()
            .map(|point| Node {
                id: point.0,
                x: point.1 .0.abs() as usize,
                y: point.1 .1.abs() as usize,
                label: (None, None),
            })
            .collect();

        for (concept_index, obj) in obj_labels {
            let index = nodes
                .iter()
                .position(|node| node.id == concept_index)
                .unwrap();
            if let Some(ref mut obj_vec) = nodes[index].label.0 {
                obj_vec.push(context.objects[obj].clone());
            } else {
                nodes[index].label.0 = Some(vec![context.objects[obj].clone()]);
            }
        }

        for (concept_index, attr) in attr_labels {
            let index = nodes
                .iter()
                .position(|node| node.id == concept_index)
                .unwrap();
            if let Some(ref mut attr_vec) = nodes[index].label.1 {
                attr_vec.push(context.attributes[attr].clone());
            } else {
                nodes[index].label.1 = Some(vec![context.attributes[attr].clone()]);
            }
        }

        let graph = Graph {
            width: width,
            height: height,
            edges: edges,
            nodes: nodes,
        };

        Some(graph)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use bit_set::BitSet;

    use crate::{
        data_structures::graph::{Graph, Node},
        FormalContext,
    };

    #[test]
    fn graph_from_concepts() {
        let context: FormalContext<String> =
            FormalContext::<String>::from(&fs::read("test_data/triangles.cxt").unwrap()).unwrap();
        let concepts: Vec<(BitSet, BitSet)> = context.fcbo_index_concepts().collect();

        let graph = Graph::from_concepts(&concepts, &context).unwrap();

        let mut nodes = Vec::new();
        nodes.push(Node {
            id: 0,
            x: 1,
            y: 0,
            label: (None, None),
        });
        nodes.push(Node {
            id: 1,
            x: 3,
            y: 3,
            label: (Some(vec!["3".to_string()]), Some(vec!["0".to_string()])),
        });
        nodes.push(Node {
            id: 2,
            x: 1,
            y: 1,
            label: (None, Some(vec!["1".to_string()])),
        });
        nodes.push(Node {
            id: 3,
            x: 3,
            y: 1,
            label: (Some(vec!["2".to_string()]), Some(vec!["2".to_string()])),
        });
        nodes.push(Node {
            id: 4,
            x: 0,
            y: 1,
            label: (Some(vec!["4".to_string()]), Some(vec!["3".to_string()])),
        });
        nodes.push(Node {
            id: 5,
            x: 2,
            y: 1,
            label: (Some(vec!["6".to_string()]), Some(vec!["4".to_string()])),
        });
        nodes.push(Node {
            id: 6,
            x: 3,
            y: 2,
            label: (Some(vec!["5".to_string()]), None),
        });
        nodes.push(Node {
            id: 7,
            x: 0,
            y: 2,
            label: (Some(vec!["0".to_string()]), None),
        });
        nodes.push(Node {
            id: 8,
            x: 1,
            y: 2,
            label: (Some(vec!["1".to_string()]), None),
        });
        nodes.push(Node {
            id: 9,
            x: 1,
            y: 4,
            label: (None, None),
        });

        assert!(graph.width == 4);
        assert!(graph.height == 5);
        assert!(graph.nodes.len() == nodes.len());
        for node in &graph.nodes {
            assert!(nodes.contains(&node));
        }

        // println!("Width: {}", graph.width);
        // println!("Height: {}", graph.height);
        // for x in graph.nodes {
        //     println!("Name: {}, ( {}, {} )", x.id, x.x, x.y);
        //     if let Some(n) = x.label.0 {
        //         println!("   Obj_Label: {}", n);
        //     }
        //     if let Some(n) = x.label.1 {
        //         println!("   Attr_Label: {}", n);
        //     }
        // }
    }
}
