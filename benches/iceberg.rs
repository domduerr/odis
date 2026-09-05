use criterion::{black_box, criterion_group, criterion_main, Criterion};
use odis::{algorithms::titanic::Titanic, traits::IcebergConceptEnumerator, FormalContext};
use std::fs;

fn living_beings_ctx() -> FormalContext<String> {
    FormalContext::<String>::from(
        &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
    )
    .unwrap()
}

fn eu_ctx() -> FormalContext<String> {
    FormalContext::<String>::from(&fs::read("test_data/eu.cxt").unwrap()).unwrap()
}

fn bench_titanic_living_beings(c: &mut Criterion) {
    let ctx = living_beings_ctx();
    c.bench_function("titanic/living_beings_min1", |b| {
        b.iter(|| Titanic.enumerate(black_box(&ctx), 1))
    });
}

fn bench_titanic_eu(c: &mut Criterion) {
    let ctx = eu_ctx();
    c.bench_function("titanic/eu_min3", |b| {
        b.iter(|| Titanic.enumerate(black_box(&ctx), 3))
    });
}

criterion_group!(benches, bench_titanic_living_beings, bench_titanic_eu);
criterion_main!(benches);
