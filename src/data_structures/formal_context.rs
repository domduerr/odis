use std::{
    collections::HashSet,
    io::{BufRead, Error},
    num::ParseIntError,
};

use bit_set::{self, BitSet};

/// Error returned when parsing a formal context from bytes fails.
#[derive(Debug)]
pub enum FormatError {
    /// Underlying I/O error while reading the context file.
    IoError(Error),
    /// An integer field in the header could not be parsed.
    ParseError(ParseIntError),
    /// The file does not conform to the expected Burmeister (.cxt) format.
    InvalidFormat,
}

impl From<Error> for FormatError {
    fn from(err: Error) -> FormatError {
        FormatError::IoError(err)
    }
}

impl From<ParseIntError> for FormatError {
    fn from(err: ParseIntError) -> FormatError {
        FormatError::ParseError(err)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Formal context: a set of *objects*, a set of *attributes*, and a binary incidence
/// relation. Operations (extent, intent, closure) are available in labelled form
/// (on `T` values) and index-based form (on [`bit_set::BitSet`] indices).
///
/// # Examples
///
/// ```
/// use odis::FormalContext;
///
/// // Build a small context from Burmeister bytes: objects {a, b}, attributes {x, y}
/// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n2\n2\n\na\nb\nx\ny\nXX\nX.\n").unwrap();
/// assert_eq!(ctx.objects.len(), 2);
/// assert_eq!(ctx.attributes.len(), 2);
/// ```
pub struct FormalContext<T> {
    /// The name of the formal context (Burmeister format line 2).
    pub name: String,
    /// The objects (rows) of the context table.
    pub objects: Vec<T>,
    /// The attributes (columns) of the context table.
    pub attributes: Vec<T>,
    /// The incidence relation: `(g, m)` is present iff object `g` has attribute `m` (index-based).
    pub incidence: HashSet<(usize, usize)>,
    // Precomputed derivations for single objects/attributes for performance
    pub(crate) atomic_object_derivations: Vec<BitSet>,
    pub(crate) atomic_attribute_derivations: Vec<BitSet>,
}

impl<T> Default for FormalContext<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FormalContext<T> {
    fn construct(name: String, objects: Vec<T>, attributes: Vec<T>, incidence: HashSet<(usize, usize)>) -> Self {
        let mut atomic_object_derivations =
            vec![BitSet::with_capacity(attributes.len()); objects.len()];
        let mut atomic_attribute_derivations =
            vec![BitSet::with_capacity(objects.len()); attributes.len()];
        for &(g, m) in incidence.iter() {
            atomic_object_derivations[g].insert(m);
            atomic_attribute_derivations[m].insert(g);
        }

        FormalContext {
            name,
            objects,
            attributes,
            incidence,
            atomic_object_derivations,
            atomic_attribute_derivations,
        }
    }

    /// Creates an empty formal context.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    ///
    /// let ctx = FormalContext::<String>::new();
    /// assert_eq!(ctx.objects.len(), 0);
    /// assert_eq!(ctx.attributes.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self::construct(String::new(), Vec::new(), Vec::new(), HashSet::new())
    }

    /// Reads a formal context in Burmeister format (.cxt).
    ///
    /// Returns `Err(FormatError)` if the byte slice is not valid Burmeister format.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    ///
    /// let cxt = b"B\n\n2\n2\n\ncat\ndog\nlegs\nfur\nXX\nX.\n";
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(cxt).unwrap();
    /// assert!(ctx.objects.contains(&"cat".to_string()));
    /// assert!(ctx.attributes.contains(&"legs".to_string()));
    /// ```
    pub fn from(contents: &[u8]) -> Result<FormalContext<String>, FormatError> {
        let mut lines = contents.lines();

        if lines.next().ok_or(FormatError::InvalidFormat)?? != "B" {
            return Err(FormatError::InvalidFormat);
        }

        let name = lines.next().ok_or(FormatError::InvalidFormat)??;

        let object_count: usize = lines.next().ok_or(FormatError::InvalidFormat)??.parse()?;
        let attribute_count: usize = lines.next().ok_or(FormatError::InvalidFormat)??.parse()?;

        lines.next().ok_or(FormatError::InvalidFormat)??;

        let mut objects: Vec<String> = Vec::with_capacity(object_count);
        for _ in 0..object_count {
            objects.push(lines.next().ok_or(FormatError::InvalidFormat)??);
        }

        let mut attributes: Vec<String> = Vec::with_capacity(attribute_count);
        for _ in 0..attribute_count {
            attributes.push(lines.next().ok_or(FormatError::InvalidFormat)??);
        }

        let mut incidence: HashSet<(usize, usize)> = HashSet::new();
        for g in 0..object_count {
            let line = lines.next().ok_or(FormatError::InvalidFormat)??;
            for (m, x) in line.chars().enumerate() {
                if x == 'X' || x == 'x' {
                    incidence.insert((g, m));
                }
            }
        }

        Ok(FormalContext::construct(name, objects, attributes, incidence))
    }

    // --- Index Methods ---

    /// Computes the extent (set of objects) for a given set of attribute indices.
    ///
    /// Index-based variant of [`FormalContext::extent`]. Operates on
    /// [`bit_set::BitSet`] indices directly, bypassing label translation.
    /// Prefer [`FormalContext::extent`] unless working in a performance-critical
    /// inner loop.
    ///
    /// See also: [`FormalContext::extent`]
    pub fn index_extent(&self, attributes: &BitSet) -> BitSet {
        match attributes.len() {
            0 => (0..self.objects.len()).collect(),
            1 => self.atomic_attribute_derivations[attributes.iter().next().unwrap()].clone(),
            _ => {
                let mut iter = attributes.iter();
                let mut result = self.atomic_attribute_derivations[iter.next().unwrap()].clone();
                for n in iter {
                    result.intersect_with(&self.atomic_attribute_derivations[n]);
                }
                result
            }
        }
    }

    /// Computes the intent (set of attributes) for a given set of object indices.
    ///
    /// Index-based variant of [`FormalContext::intent`]. Prefer
    /// [`FormalContext::intent`] unless working in a performance-critical inner loop.
    ///
    /// See also: [`FormalContext::intent`]
    pub fn index_intent(&self, objects: &BitSet) -> BitSet {
        match objects.len() {
            0 => (0..self.attributes.len()).collect(),
            1 => self.atomic_object_derivations[objects.iter().next().unwrap()].clone(),
            _ => {
                let mut iter = objects.iter();
                let mut result = self.atomic_object_derivations[iter.next().unwrap()].clone();
                for n in iter {
                    result.intersect_with(&self.atomic_object_derivations[n]);
                }
                result
            }
        }
    }

    /// Computes the attribute hull (closure) for a given set of attribute indices.
    ///
    /// Index-based variant of [`FormalContext::attribute_hull`].
    ///
    /// See also: [`FormalContext::attribute_hull`]
    pub fn index_attribute_hull(&self, attributes: &BitSet) -> BitSet {
        let extent = self.index_extent(attributes);
        self.index_intent(&extent)
    }

    /// Computes the object hull (closure) for a given set of object indices.
    ///
    /// Index-based variant of [`FormalContext::object_hull`].
    ///
    /// See also: [`FormalContext::object_hull`]
    pub fn index_object_hull(&self, objects: &BitSet) -> BitSet {
        let intent = self.index_intent(objects);
        self.index_extent(&intent)
    }

    // --- High-Level Methods (Sets of T) ---

    /// Computes the extent for a given set of attributes.
    ///
    /// Returns the set of all objects that possess every attribute in `attributes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    /// use std::collections::HashSet;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n2\n2\n\ncat\ndog\nx\ny\nXX\nX.\n").unwrap();
    /// let extent = ctx.extent(&HashSet::from(["x".to_string()]));
    /// assert!(extent.contains("cat"));
    /// assert!(extent.contains("dog"));
    /// ```
    pub fn extent(&self, attributes: &HashSet<T>) -> HashSet<T>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        let mut bits = BitSet::new();
        for attr in attributes {
            if let Some(idx) = self.attributes.iter().position(|a| a == attr) {
                bits.insert(idx);
            }
        }
        let result_bits = self.index_extent(&bits);
        result_bits
            .iter()
            .map(|idx| self.objects[idx].clone())
            .collect()
    }

    /// Computes the intent for a given set of objects.
    ///
    /// Returns the set of all attributes that every object in `objects` possesses.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    /// use std::collections::HashSet;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n2\n2\n\ncat\ndog\nx\ny\nXX\nX.\n").unwrap();
    /// let intent = ctx.intent(&HashSet::from(["cat".to_string(), "dog".to_string()]));
    /// // Both cat and dog have attribute x; only cat has y
    /// assert!(intent.contains("x"));
    /// assert!(!intent.contains("y"));
    /// ```
    pub fn intent(&self, objects: &HashSet<T>) -> HashSet<T>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        let mut bits = BitSet::new();
        for obj in objects {
            if let Some(idx) = self.objects.iter().position(|o| o == obj) {
                bits.insert(idx);
            }
        }
        let result_bits = self.index_intent(&bits);
        result_bits
            .iter()
            .map(|idx| self.attributes[idx].clone())
            .collect()
    }

    /// Computes the closure (hull) of a set of attributes under the Galois connection.
    ///
    /// Returns the smallest concept intent that contains all given attributes.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    /// use std::collections::HashSet;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n2\n2\n\ncat\ndog\nx\ny\nXX\nX.\n").unwrap();
    /// // Hull of {x} is {x, y} because every object with x also has y (only cat does)
    /// let hull = ctx.attribute_hull(&HashSet::from(["x".to_string()]));
    /// // hull contains at least x
    /// assert!(hull.contains("x"));
    /// ```
    pub fn attribute_hull(&self, attributes: &HashSet<T>) -> HashSet<T>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        let mut bits = BitSet::new();
        for attr in attributes {
            if let Some(idx) = self.attributes.iter().position(|a| a == attr) {
                bits.insert(idx);
            }
        }
        let result_bits = self.index_attribute_hull(&bits);
        result_bits
            .iter()
            .map(|idx| self.attributes[idx].clone())
            .collect()
    }

    /// Computes the closure (hull) of a set of objects under the Galois connection.
    ///
    /// Returns the smallest concept extent that contains all given objects.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    /// use std::collections::HashSet;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n2\n2\n\ncat\ndog\nx\ny\nXX\nX.\n").unwrap();
    /// let hull = ctx.object_hull(&HashSet::from(["cat".to_string()]));
    /// assert!(hull.contains("cat"));
    /// ```
    pub fn object_hull(&self, objects: &HashSet<T>) -> HashSet<T>
    where
        T: Eq + std::hash::Hash + Clone,
    {
        let mut bits = BitSet::new();
        for obj in objects {
            if let Some(idx) = self.objects.iter().position(|o| o == obj) {
                bits.insert(idx);
            }
        }
        let result_bits = self.index_object_hull(&bits);
        result_bits
            .iter()
            .map(|idx| self.objects[idx].clone())
            .collect()
    }

    /// Adds a new object with its corresponding attributes to the existing FormalContext.
    ///
    /// The `attributes` BitSet uses attribute indices (positions in `self.attributes`).
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    /// use bit_set::BitSet;
    ///
    /// let mut ctx = FormalContext::<String>::new();
    /// ctx.add_object("cat".to_string(), &BitSet::new()); // no attributes yet
    /// assert_eq!(ctx.objects.len(), 1);
    /// ```
    pub fn add_object(&mut self, new_object: T, attributes: &BitSet) {
        self.objects.push(new_object);
        let object_index = self.objects.len() - 1;
        self.atomic_object_derivations.push(BitSet::new());

        for attribute in attributes.iter() {
            self.incidence.insert((object_index, attribute));
            self.atomic_object_derivations[object_index].insert(attribute);
            self.atomic_attribute_derivations[attribute].insert(object_index);
        }
    }

    /// Adds a new attribute with its corresponding objects to the existing FormalContext.
    ///
    /// The `objects` BitSet uses object indices (positions in `self.objects`).
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    /// use bit_set::BitSet;
    ///
    /// let mut ctx = FormalContext::<String>::new();
    /// ctx.add_attribute("legs".to_string(), &BitSet::new());
    /// assert_eq!(ctx.attributes.len(), 1);
    /// ```
    pub fn add_attribute(&mut self, new_attribute: T, objects: &BitSet) {
        self.attributes.push(new_attribute);
        let attribute_index = self.attributes.len() - 1;
        self.atomic_attribute_derivations.push(BitSet::new());

        for object in objects.iter() {
            self.incidence.insert((object, attribute_index));
            self.atomic_object_derivations[object].insert(attribute_index);
            self.atomic_attribute_derivations[attribute_index].insert(object);
        }
    }

    /// Sets or clears the incidence cross at object index `g` and attribute index `m`,
    /// keeping the precomputed derivation caches in sync.
    pub fn set_cross(&mut self, g: usize, m: usize, value: bool) {
        if value {
            self.incidence.insert((g, m));
            self.atomic_object_derivations[g].insert(m);
            self.atomic_attribute_derivations[m].insert(g);
        } else {
            self.incidence.remove(&(g, m));
            self.atomic_object_derivations[g].remove(m);
            self.atomic_attribute_derivations[m].remove(g);
        }
    }

    /// Removes the object at the specified index from the existing FormalContext.
    ///
    /// All incidence entries involving the removed object are deleted.
    /// Indices of subsequent objects are decremented to remain contiguous.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    ///
    /// let mut ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n2\n1\n\ncat\ndog\nlegs\nX\nX\n").unwrap();
    /// assert_eq!(ctx.objects.len(), 2);
    /// ctx.remove_object(0); // remove "cat" (index 0)
    /// assert_eq!(ctx.objects.len(), 1);
    /// assert_eq!(ctx.objects[0], "dog");
    /// ```
    pub fn remove_object(&mut self, index: usize) {
        for n in 0..self.attributes.len() {
            self.incidence.remove(&(index, n));
        }

        self.incidence = self
            .incidence
            .iter()
            .map(|x| if x.0 > index { (x.0 - 1, x.1) } else { *x })
            .collect();

        for n in 0..self.attributes.len() {
            self.atomic_attribute_derivations[n].remove(index);
        }

        for n in 0..self.attributes.len() {
            self.atomic_attribute_derivations[n] = self.atomic_attribute_derivations[n]
                .iter()
                .map(|x| if x > index { x - 1 } else { x })
                .collect();
        }

        self.atomic_object_derivations.remove(index);
        self.objects.remove(index);
    }

    /// Removes the attribute at the specified index from the existing FormalContext.
    ///
    /// All incidence entries involving the removed attribute are deleted.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    ///
    /// let mut ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n1\n2\n\ncat\nlegs\nfur\nXX\n").unwrap();
    /// assert_eq!(ctx.attributes.len(), 2);
    /// ctx.remove_attribute(1); // remove "fur" (index 1)
    /// assert_eq!(ctx.attributes.len(), 1);
    /// assert_eq!(ctx.attributes[0], "legs");
    /// ```
    pub fn remove_attribute(&mut self, index: usize) {
        for n in 0..self.objects.len() {
            self.incidence.remove(&(n, index));
        }

        self.incidence = self
            .incidence
            .iter()
            .map(|x| if x.1 > index { (x.0, x.1 - 1) } else { *x })
            .collect();

        for n in 0..self.objects.len() {
            self.atomic_object_derivations[n].remove(index);
        }

        for n in 0..self.objects.len() {
            self.atomic_object_derivations[n] = self.atomic_object_derivations[n]
                .iter()
                .map(|x| if x > index { x - 1 } else { x })
                .collect();
        }

        self.atomic_attribute_derivations.remove(index);
        self.attributes.remove(index);
    }

    /// Changes the name of the object at the specified index.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    ///
    /// let mut ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n1\n1\n\ncat\nlegs\nX\n").unwrap();
    /// ctx.change_object_name("feline".to_string(), 0);
    /// assert_eq!(ctx.objects[0], "feline");
    /// ```
    pub fn change_object_name(&mut self, name: T, index: usize) {
        self.objects[index] = name;
    }

    /// Changes the name of the attribute at the specified index.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    ///
    /// let mut ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n1\n1\n\ncat\nlegs\nX\n").unwrap();
    /// ctx.change_attribute_name("limbs".to_string(), 0);
    /// assert_eq!(ctx.attributes[0], "limbs");
    /// ```
    pub fn change_attribute_name(&mut self, name: T, index: usize) {
        self.attributes[index] = name;
    }

    /// In place sorts the concepts in lectic order using indices.
    ///
    /// Index-based variant of [`FormalContext::sort_lectic_order`].
    ///
    /// See also: [`FormalContext::sort_lectic_order`]
    pub fn index_sort_lectic_order(&self, concepts: &mut [(BitSet, BitSet)]) {
        let length = self.attributes.len();
        let weight: Vec<usize> = (1..=length).map(|x| 2_usize.pow(x as u32)).collect();

        let mut order: Vec<(usize, usize)> = Vec::new();

        for (index, (_, set)) in concepts.iter().enumerate() {
            let mut sum = 0;
            for n in set {
                if length > n {
                    sum += weight[length - 1 - n];
                }
            }
            order.push((index, sum));
        }

        order.sort_by_key(|x| x.1);

        for index in 0..(order.len() - 1) {
            let swap = order[index].0;
            if index != swap {
                concepts.swap(index, swap);
                let correction = order.iter().position(|x| x.0 == index).unwrap();
                order[index] = (0, 0);
                order[correction] = (swap, 0);
            }
        }
    }

    /// In place sorts named concept pairs in lectic order.
    ///
    /// Attribute names in a concept intent that are not found in `context.attributes` are
    /// treated as weight 0 (silently ignored in the sort-key computation).
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    /// use std::collections::HashSet;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n2\n2\n\ncat\ndog\nx\ny\nXX\nX.\n").unwrap();
    /// let mut concepts: Vec<(HashSet<String>, HashSet<String>)> = vec![
    ///     (HashSet::from(["cat".to_string()]), HashSet::from(["x".to_string(), "y".to_string()])),
    ///     (HashSet::from(["cat".to_string(), "dog".to_string()]), HashSet::from(["x".to_string()])),
    /// ];
    /// ctx.sort_lectic_order(&mut concepts);
    /// // After sorting, intents appear in lectic (canonical) order
    /// assert_eq!(concepts.len(), 2);
    /// ```
    pub fn sort_lectic_order(&self, concepts: &mut [(HashSet<T>, HashSet<T>)])
    where
        T: Eq + std::hash::Hash + Clone,
    {
        let mut indexed: Vec<(BitSet, BitSet)> = concepts
            .iter()
            .map(|(g_set, m_set)| {
                let g_bits = g_set
                    .iter()
                    .filter_map(|name| self.objects.iter().position(|o| o == name))
                    .collect();
                let m_bits = m_set
                    .iter()
                    .filter_map(|name| self.attributes.iter().position(|a| a == name))
                    .collect();
                (g_bits, m_bits)
            })
            .collect();
        self.index_sort_lectic_order(&mut indexed);
        for (i, (g_bits, m_bits)) in indexed.into_iter().enumerate() {
            concepts[i] = (
                g_bits.iter().map(|idx| self.objects[idx].clone()).collect(),
                m_bits.iter().map(|idx| self.attributes[idx].clone()).collect(),
            );
        }
    }

    /// Filter a concept iterator to only keep concepts whose extent is at least `min_extent` objects.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    /// use odis::ConceptEnumerator;
    /// use bit_set::BitSet;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n3\n2\n\na\nb\nc\nx\ny\nXX\nX.\n.X\n").unwrap();
    /// let all: Vec<_> = ctx.index_concepts().collect();
    /// let filtered = FormalContext::<String>::filter_concepts_by_min_extent(all.into_iter(), 2);
    /// assert!(filtered.iter().all(|(ext, _)| ext.len() >= 2));
    /// ```
    pub fn filter_concepts_by_min_extent(
        concepts: impl Iterator<Item = (BitSet, BitSet)>,
        min_extent: usize,
    ) -> Vec<(BitSet, BitSet)> {
        concepts
            .filter(|(extent, _)| extent.len() >= min_extent)
            .collect()
    }

    /// Filter a concept iterator to keep concepts whose extent is at least `min_extent_percent` of all objects.
    ///
    /// The threshold is rounded up: `(total_objects × min_extent_percent / 100).ceil()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    /// use odis::ConceptEnumerator;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n4\n2\n\na\nb\nc\nd\nx\ny\nXX\nXX\nX.\n..\n").unwrap();
    /// let all: Vec<_> = ctx.index_concepts().collect();
    /// // Keep only concepts covering at least 50% of objects (i.e., ≥ 2 out of 4)
    /// let filtered = FormalContext::<String>::filter_concepts_by_min_extent_percent(
    ///     all.into_iter(), 50.0, 4
    /// );
    /// assert!(filtered.iter().all(|(ext, _)| ext.len() >= 2));
    /// ```
    pub fn filter_concepts_by_min_extent_percent(
        concepts: impl Iterator<Item = (BitSet, BitSet)>,
        min_extent_percent: f64,
        total_objects: usize,
    ) -> Vec<(BitSet, BitSet)> {
        let min_extent = ((total_objects as f64) * min_extent_percent / 100.0).ceil() as usize;
        Self::filter_concepts_by_min_extent(concepts, min_extent)
    }
}

impl FormalContext<String> {
    /// Serializes this context to Burmeister (.cxt) format bytes.
    ///
    /// The returned bytes can be written to a file or passed to
    /// [`FormalContext::from`] to round-trip a context.
    ///
    /// # Examples
    ///
    /// ```
    /// use odis::FormalContext;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n1\n1\n\ncat\nlegs\nX\n").unwrap();
    /// let bytes = ctx.to_cxt_bytes();
    /// let ctx2: FormalContext<String> = FormalContext::<String>::from(&bytes).unwrap();
    /// assert_eq!(ctx2.objects, ctx.objects);
    /// ```
    pub fn to_cxt_bytes(&self) -> Vec<u8> {
        let mut out = String::new();
        out.push_str("B\n");
        out.push_str(&self.name);
        out.push('\n');
        out.push_str(&self.objects.len().to_string());
        out.push('\n');
        out.push_str(&self.attributes.len().to_string());
        out.push_str("\n\n");
        for obj in &self.objects {
            out.push_str(obj.as_str());
            out.push('\n');
        }
        for attr in &self.attributes {
            out.push_str(attr.as_str());
            out.push('\n');
        }
        for (g, _) in self.objects.iter().enumerate() {
            for m in 0..self.attributes.len() {
                if self.incidence.contains(&(g, m)) {
                    out.push('X');
                } else {
                    out.push('.');
                }
            }
            out.push('\n');
        }
        out.into_bytes()
    }

    /// Writes this context to a `.cxt` file at the given path.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use odis::FormalContext;
    ///
    /// let ctx: FormalContext<String> = FormalContext::<String>::from(b"B\n\n1\n1\n\ncat\nlegs\nX\n").unwrap();
    /// ctx.to_file("/tmp/output.cxt").unwrap();
    /// ```
    pub fn to_file(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_cxt_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::FormalContext;
    use bit_set::BitSet;
    use itertools::Itertools;
    use std::{collections::HashSet, fs};

    #[test]
    fn test_read_context() {
        let context =
            FormalContext::<String>::from(&fs::read("test_data/eu.cxt").unwrap()).unwrap();
        assert_eq!(context.objects.len(), 48);
        assert_eq!(context.objects[0], "Albanien");
        assert_eq!(context.objects[47], "Zypern");
        assert_eq!(context.attributes.len(), 7);
        assert_eq!(context.attributes[0], "EU");
        assert_eq!(context.attributes[6], "Europarat");
        assert_eq!(context.incidence.len(), 201);
        assert!(!context.incidence.contains(&(1, 0)));
        assert!(context.incidence.contains(&(1, 1)));
        assert!(!context.incidence.contains(&(1, 2)));
        assert!(!context.incidence.contains(&(1, 3)));
        assert!(!context.incidence.contains(&(1, 4)));
        assert!(context.incidence.contains(&(1, 5)));
        assert!(context.incidence.contains(&(1, 6)));
    }

    #[test]
    fn text_index_derivations() {
        let context =
            FormalContext::<String>::from(&fs::read("test_data/eu.cxt").unwrap()).unwrap();

        assert_eq!(context.index_extent(&BitSet::new()).len(), 48);
        assert_eq!(context.index_intent(&BitSet::new()).len(), 7);
        let context = FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap();
        assert_eq!(
            context.index_extent(&BitSet::from_bytes(&[0b10000000])),
            BitSet::from_bytes(&[0b11111111])
        );
    }

    #[test]
    fn test_high_level_methods() {
        let context = FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap();

        let mut attrs = HashSet::new();
        attrs.insert("needs water to live".to_string());

        let objects = context.extent(&attrs);
        assert_eq!(objects.len(), 8);
        assert!(objects.contains("fish leech"));

        let mut objects_set = HashSet::new();
        objects_set.insert("frog".to_string());
        let intent = context.intent(&objects_set);
        assert!(intent.contains("lives in water"));
        assert!(intent.contains("lives on land"));
    }

    #[test]
    fn text_index_hulls() {
        let context = FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap();
        assert_eq!(
            context.index_attribute_hull(&BitSet::from_bytes(&[0b00000000])),
            BitSet::from_bytes(&[0b10000000])
        );
        for gs in (0..context.objects.len()).powerset() {
            let sub: BitSet = gs.into_iter().collect();
            let hull = context.index_object_hull(&sub);
            assert!(sub.is_subset(&hull));
        }
        for ms in (0..context.attributes.len()).powerset() {
            let sub: BitSet = ms.into_iter().collect();
            let hull = context.index_attribute_hull(&sub);
            assert!(sub.is_subset(&hull));
        }
    }

    #[test]
    fn lectic_sort() {
        let context =
            FormalContext::<String>::from(&fs::read("test_data/living_beings_and_water.cxt").unwrap()).unwrap();

        let mut concepts_unsorted: Vec<(BitSet, BitSet)> = context.index_fcbo_concepts().collect();
        let concepts_sorted: Vec<(BitSet, BitSet)> = context.index_next_closure_concepts().collect();

        assert!(concepts_sorted != concepts_unsorted);

        context.index_sort_lectic_order(&mut concepts_unsorted);

        assert!(concepts_sorted == concepts_unsorted);
    }

    #[test]
    fn test_to_file_round_trip() {
        let ctx = FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap();

        let tmp = std::env::temp_dir().join("odis_round_trip_test.cxt");
        ctx.to_file(tmp.to_str().unwrap()).expect("to_file should succeed");

        let reloaded = FormalContext::<String>::from(&fs::read(&tmp).unwrap()).unwrap();
        assert_eq!(ctx.name, reloaded.name);
        assert_eq!(ctx.objects, reloaded.objects);
        assert_eq!(ctx.attributes, reloaded.attributes);
        assert_eq!(ctx.incidence, reloaded.incidence);

        let _ = fs::remove_file(tmp);
    }
}

