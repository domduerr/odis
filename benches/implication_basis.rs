use bit_set::BitSet;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use odis::{algorithms::canonical_basis::index_canonical_basis, FormalContext};
use rand::{Rng, SeedableRng};
use std::fs;

fn small_context() -> FormalContext<String> {
    FormalContext::<String>::from(&fs::read("test_data/living_beings_and_water.cxt").unwrap())
        .unwrap()
}

fn random_context_20x20() -> FormalContext<usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let n_objects = 20;
    let n_attrs = 20;
    let mut ctx = FormalContext::<usize>::new();
    for m in 0..n_attrs {
        ctx.add_attribute(m, &BitSet::new());
    }
    for g in 0..n_objects {
        let mut intent = BitSet::new();
        for m in 0..n_attrs {
            if rng.gen::<bool>() {
                intent.insert(m);
            }
        }
        ctx.add_object(g, &intent);
    }
    ctx
}

fn bench_canonical_basis(c: &mut Criterion) {
    let small = small_context();
    let large = random_context_20x20();

    c.bench_function("canonical_basis/small", |b| {
        b.iter(|| index_canonical_basis(black_box(&small)))
    });
    c.bench_function("canonical_basis/20x20_random", |b| {
        b.iter(|| index_canonical_basis(black_box(&large)))
    });
}

criterion_group!(benches, bench_canonical_basis);
criterion_main!(benches);
