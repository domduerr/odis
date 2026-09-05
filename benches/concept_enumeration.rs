use bit_set::BitSet;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use odis::{
    algorithms::{fcbo::index_fcbo_concepts, next_closure::index_next_closure_concepts},
    FormalContext,
};
use rand::{Rng, SeedableRng};
use std::fs;

fn small_context() -> FormalContext<String> {
    FormalContext::<String>::from(&fs::read("test_data/living_beings_and_water.cxt").unwrap())
        .unwrap()
}

fn random_context_50x50() -> FormalContext<usize> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let n_objects = 50;
    let n_attrs = 50;
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

fn bench_next_closure(c: &mut Criterion) {
    let small = small_context();
    let large = random_context_50x50();

    c.bench_function("next_closure/small", |b| {
        b.iter(|| index_next_closure_concepts(black_box(&small)).count())
    });
    c.bench_function("next_closure/50x50_random", |b| {
        b.iter(|| index_next_closure_concepts(black_box(&large)).count())
    });
}

fn bench_fcbo(c: &mut Criterion) {
    let small = small_context();
    let large = random_context_50x50();

    c.bench_function("fcbo/small", |b| {
        b.iter(|| index_fcbo_concepts(black_box(&small)).count())
    });
    c.bench_function("fcbo/50x50_random", |b| {
        b.iter(|| index_fcbo_concepts(black_box(&large)).count())
    });
}

criterion_group!(benches, bench_next_closure, bench_fcbo);
criterion_main!(benches);
