use criterion::{black_box, criterion_group, criterion_main, Criterion};
use odis::{
    algorithms::{dimdraw::DimDraw, sugiyama::Sugiyama, SearchBudget},
    traits::DrawingAlgorithm,
    FormalContext, Lattice,
};
use std::fs;

fn living_beings_lattice() -> Lattice<(bit_set::BitSet, bit_set::BitSet)> {
    let ctx = FormalContext::<String>::from(
        &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
    )
    .unwrap();
    ctx.concept_lattice().expect("concept_lattice returned None")
}

fn fm3_lattice() -> Lattice<(bit_set::BitSet, bit_set::BitSet)> {
    let ctx =
        FormalContext::<String>::from(&fs::read("test_data/fm3.cxt").unwrap()).unwrap();
    ctx.concept_lattice().expect("concept_lattice returned None")
}

fn bench_sugiyama(c: &mut Criterion) {
    let lattice = living_beings_lattice();
    c.bench_function("sugiyama/living_beings", |b| {
        b.iter(|| Sugiyama { vertex_spacing: 1 }.draw(black_box(&lattice)))
    });
}

fn bench_dimdraw_bounded(c: &mut Criterion) {
    let lattice = living_beings_lattice();
    c.bench_function("dimdraw/living_beings_timeout_10ms", |b| {
        b.iter(|| DimDraw { budget: SearchBudget::Milliseconds(10) }.draw(black_box(&lattice)))
    });
}

fn bench_dimdraw_unbounded(c: &mut Criterion) {
    let lattice = living_beings_lattice();
    c.bench_function("dimdraw/living_beings_unbounded", |b| {
        b.iter(|| DimDraw { budget: SearchBudget::Unbounded }.draw(black_box(&lattice)))
    });
}

fn bench_dimdraw_fm3_unbounded(c: &mut Criterion) {
    let lattice = fm3_lattice();
    c.bench_function("dimdraw/fm3_unbounded", |b| {
        b.iter(|| DimDraw { budget: SearchBudget::Unbounded }.draw(black_box(&lattice)))
    });
}

criterion_group!(
    benches,
    bench_sugiyama,
    bench_dimdraw_bounded,
    bench_dimdraw_unbounded,
    bench_dimdraw_fm3_unbounded
);
criterion_main!(benches);
