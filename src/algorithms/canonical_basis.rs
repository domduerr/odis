//! Duquenne–Guigues canonical implication basis algorithm.

use bit_set::BitSet;

use crate::FormalContext;

/// Checks if `min` is smaller than any element in `input_set`.
/// Returns true if `input_set` contains no elements smaller than `min`.
fn is_smallest_num(min: usize, input_set: &BitSet) -> bool {
    for n in 0..min {
        if input_set.contains(n) {
            return false;
        }
    }
    true
}

/// Creates a BitSet containing all integers from 0 to `n`.
fn set_upto(n: usize) -> BitSet {
    let mut b = BitSet::with_capacity(n + 1);
    for i in 0..=n {
        b.insert(i);
    }
    b
}

/// Returns a subset of `input_set` containing only elements <= `max`.
fn retain_eq_less(max: usize, input_set: &BitSet) -> BitSet {
    input_set.iter().filter(|&x| x <= max).collect()
}

/// Computes the closure of a set of attributes `input` under a set of implications.
fn index_implication_closure(implications: &[(BitSet, BitSet)], input: &BitSet) -> BitSet {
    let mut active_implications: Vec<&(BitSet, BitSet)> = implications.iter().collect();
    let mut output = input.clone();

    loop {
        let mut changed = false;
        let mut to_remove = Vec::new();

        for (i, impl_ref) in active_implications.iter().enumerate() {
            let (premise, conclusion) = impl_ref;
            if premise.is_subset(&output) {
                if !conclusion.is_subset(&output) {
                    output.union_with(conclusion);
                    changed = true;
                }
                to_remove.push(i);
            }
        }

        if !changed && to_remove.is_empty() {
            break;
        }

        if !to_remove.is_empty() {
            let mut next_active = Vec::new();
            let mut remove_iter = to_remove.iter();
            let mut next_remove = remove_iter.next();
            for (i, imp) in active_implications.iter().enumerate() {
                if let Some(r_idx) = next_remove {
                    if *r_idx == i {
                        next_remove = remove_iter.next();
                        continue;
                    }
                }
                next_active.push(*imp);
            }
            active_implications = next_active;

            if !changed {
                break;
            }
        }
    }
    output
}

/// Computes the next preclosure in lectic order.
pub fn index_next_preclosure<T>(
    context: &FormalContext<T>,
    implications: &[(BitSet, BitSet)],
    input: &BitSet,
) -> BitSet {
    let mut temp_set = input.clone();

    for m in (0..context.attributes.len()).rev() {
        if temp_set.contains(m) {
            temp_set.remove(m);
        } else {
            temp_set.insert(m);
            let output = index_implication_closure(implications, &temp_set);

            let diff: BitSet = output.difference(&temp_set).collect();
            if is_smallest_num(m, &diff) {
                return output;
            }
            temp_set.remove(m);
        }
    }
    (0..context.attributes.len()).collect()
}

/// Computes the canonical basis of implications for the formal context.
pub fn index_canonical_basis<T>(context: &FormalContext<T>) -> Vec<(BitSet, BitSet)> {
    let mut temp_set = BitSet::new();
    let mut implications: Vec<(BitSet, BitSet)> = Vec::new();
    let all_attributes: BitSet = (0..context.attributes.len()).collect();

    while temp_set != all_attributes {
        let temp_set_hull = context.index_attribute_hull(&temp_set);
        if temp_set != temp_set_hull {
            implications.push((temp_set.clone(), temp_set_hull));
        }
        temp_set = index_next_preclosure(context, &implications, &temp_set);
    }
    implications
}

/// Computes the canonical basis using an optimized NextClosure approach.
pub fn index_canonical_basis_optimised<T>(context: &FormalContext<T>) -> Vec<(BitSet, BitSet)> {
    let mut temp_set = context.index_attribute_hull(&BitSet::new());
    let mut implications: Vec<(BitSet, BitSet)> = Vec::new();

    if !temp_set.is_empty() {
        implications.push((BitSet::new(), temp_set.clone()));
    }

    let mut i = context.attributes.len().saturating_sub(1);
    let all_attributes = set_upto(context.attributes.len().saturating_sub(1));

    while temp_set != all_attributes {
        for j in (0..=i).rev() {
            if temp_set.contains(j) {
                temp_set.remove(j);
            } else {
                temp_set.insert(j);
                let b = index_implication_closure(&implications, &temp_set);
                temp_set.remove(j);

                let diff: BitSet = b.difference(&temp_set).collect();
                if is_smallest_num(j, &diff) {
                    temp_set = b;
                    i = j;
                    break;
                }
            }
        }

        let temp_set_hull = context.index_attribute_hull(&temp_set);

        if temp_set != temp_set_hull {
            implications.push((temp_set.clone(), temp_set_hull.clone()));
        }

        let diff: BitSet = temp_set_hull.difference(&temp_set).collect();
        if is_smallest_num(i, &diff) {
            temp_set = temp_set_hull;
            i = context.attributes.len().saturating_sub(1);
        } else {
            temp_set = retain_eq_less(i, &temp_set);
        }
    }
    implications
}

/// Zero-size handle for the canonical implication basis algorithm.
pub struct CanonicalBasis;

impl<T> crate::traits::ImplicationEngine<T> for CanonicalBasis {
    fn compute_basis(&self, context: &crate::FormalContext<T>) -> Vec<(BitSet, BitSet)> {
        index_canonical_basis(context)
    }
}

#[cfg(test)]
mod tests {
    use crate::algorithms::{
        canonical_basis::{
            index_canonical_basis, index_implication_closure, index_next_preclosure,
        },
        FormalContext,
    };
    use bit_set::BitSet;
    use std::fs;

    #[test]
    fn canonical_basis_test() {
        let context =
            FormalContext::<String>::from(&fs::read("test_data/triangles.cxt").unwrap()).unwrap();

        let output = index_canonical_basis(&context);

        let canonical_basis = vec![
            (
                BitSet::from_bytes(&[0b00011000]),
                BitSet::from_bytes(&[0b11111000]),
            ),
            (
                BitSet::from_bytes(&[0b00101000]),
                BitSet::from_bytes(&[0b11111000]),
            ),
            (
                BitSet::from_bytes(&[0b00110000]),
                BitSet::from_bytes(&[0b11111000]),
            ),
            (
                BitSet::from_bytes(&[0b10000000]),
                BitSet::from_bytes(&[0b11100000]),
            ),
        ];

        assert_eq!(output, canonical_basis);
    }

    #[test]
    fn next_closure_test() {
        let context =
            FormalContext::<String>::from(&fs::read("test_data/triangles.cxt").unwrap()).unwrap();

        let mut canonical_basis = Vec::new();

        let input = BitSet::new();
        let output = index_next_preclosure(&context, &canonical_basis, &input);
        assert_eq!(output, BitSet::from_bytes(&[0b00001000]));

        let input = BitSet::from_bytes(&[0b00001000]);
        let output = index_next_preclosure(&context, &canonical_basis, &input);
        assert_eq!(output, BitSet::from_bytes(&[0b00010000]));

        let input = BitSet::from_bytes(&[0b00010000]);
        let output = index_next_preclosure(&context, &canonical_basis, &input);
        assert_eq!(output, BitSet::from_bytes(&[0b00011000]));

        canonical_basis.push((
            BitSet::from_bytes(&[0b00011000]),
            BitSet::from_bytes(&[0b11111000]),
        ));
        let input = BitSet::from_bytes(&[0b00011000]);
        let output = index_next_preclosure(&context, &canonical_basis, &input);
        assert_eq!(output, BitSet::from_bytes(&[0b00100000]));
    }

    #[test]
    fn implication_closure_test() {
        let implications = vec![
            (
                BitSet::from_bytes(&[0b01000000]),
                BitSet::from_bytes(&[0b00110000]),
            ),
            (
                BitSet::from_bytes(&[0b00001100]),
                BitSet::from_bytes(&[0b01111100]),
            ),
            (
                BitSet::from_bytes(&[0b00010100]),
                BitSet::from_bytes(&[0b01111100]),
            ),
            (
                BitSet::from_bytes(&[0b00011000]),
                BitSet::from_bytes(&[0b01111100]),
            ),
        ];

        let mut input = BitSet::new();
        input.insert(1);
        assert_eq!(
            index_implication_closure(&implications, &input),
            BitSet::from_bytes(&[0b01110000])
        );

        let mut input = BitSet::new();
        input.insert(4);
        input.insert(5);
        assert_eq!(
            index_implication_closure(&implications, &input),
            BitSet::from_bytes(&[0b01111100])
        );

        let mut input = BitSet::new();
        input.insert(3);
        input.insert(5);
        assert_eq!(
            index_implication_closure(&implications, &input),
            BitSet::from_bytes(&[0b01111100])
        );

        let mut input = BitSet::new();
        input.insert(3);
        input.insert(4);
        assert_eq!(
            index_implication_closure(&implications, &input),
            BitSet::from_bytes(&[0b01111100])
        );
    }

    fn is_sound_implication<T>(ctx: &FormalContext<T>, premise: &BitSet, conclusion: &BitSet) -> bool {
        for g in 0..ctx.objects.len() {
            let obj_attrs = &ctx.atomic_object_derivations[g];
            if premise.is_subset(obj_attrs) && !conclusion.is_subset(obj_attrs) {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_basis_zero_attributes() {
        // 3 objects, 0 attributes: no implications possible
        let mut ctx = FormalContext::<usize>::new();
        for g in 0..3 {
            ctx.add_object(g, &BitSet::new());
        }
        let basis = index_canonical_basis(&ctx);
        assert!(basis.is_empty(), "zero-attribute context must have empty basis");
    }

    #[test]
    fn test_basis_zero_objects() {
        // 0 objects, 5 attributes: implies ∅ → M (top of concept lattice)
        let mut ctx = FormalContext::<usize>::new();
        for m in 0..5 {
            ctx.add_attribute(m, &BitSet::new());
        }
        let basis = index_canonical_basis(&ctx);
        assert_eq!(basis.len(), 1);
        let (premise, conclusion) = &basis[0];
        assert!(premise.is_empty(), "premise of the unique implication must be ∅");
        assert_eq!(*conclusion, (0..5).collect::<BitSet>(), "conclusion must be M");
    }

    #[test]
    fn test_basis_full_context() {
        // 3×3 full context: every object has every attribute → basis is [∅ → M]
        let mut ctx = FormalContext::<usize>::new();
        for m in 0..3 {
            ctx.add_attribute(m, &BitSet::new());
        }
        for g in 0..3 {
            ctx.add_object(g, &(0..3).collect::<BitSet>());
        }
        let basis = index_canonical_basis(&ctx);
        // All returned implications must be sound
        for (premise, conclusion) in &basis {
            assert!(is_sound_implication(&ctx, premise, conclusion));
        }
        // Exactly one implication: ∅ → M
        assert_eq!(basis.len(), 1);
        assert!(basis[0].0.is_empty());
        assert_eq!(basis[0].1, (0..3).collect::<BitSet>());
    }

    #[test]
    fn test_basis_unique_attribute_sets() {
        // 3 objects × 3 attributes, each object has exactly one unique attribute
        let mut ctx = FormalContext::<usize>::new();
        for m in 0..3 {
            ctx.add_attribute(m, &BitSet::new());
        }
        for g in 0..3 {
            let mut intent = BitSet::new();
            intent.insert(g); // object g has only attribute g
            ctx.add_object(g, &intent);
        }
        let basis = index_canonical_basis(&ctx);
        assert!(!basis.is_empty(), "unique-attribute context must have non-empty basis");
        for (premise, conclusion) in &basis {
            assert!(is_sound_implication(&ctx, premise, conclusion),
                "implication {:?} → {:?} must be sound", premise, conclusion);
        }
    }
}
