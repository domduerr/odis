//! Scalability benchmarks for dimension algorithms.
//!
//! Excludes trivial posets (chains, antichains) that are recognized in O(1).
//! Uses parameterized inputs (Criterion bench_with_input) to generate
//! proper scaling curves in the Criterion HTML report.

use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use odis::{
    algorithms::dimension::{GraphColoring, GraphColoringHeuristic, HypergraphColoring, Hybrid, SatReduction},
    DimensionAlgorithm, Poset,
};

use rand::{Rng, SeedableRng};
use std::time::Duration;

// ─── Poset Generators ───────────────────────────────────────────────────

fn standard_example(n: usize) -> Poset<u32> {
    let nodes: Vec<u32> = (0..(2 * n) as u32).collect();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                edges.push((i as u32, (n + j) as u32));
            }
        }
    }
    Poset::from_covering_relation(nodes, edges).unwrap()
}

/// Random poset: n elements, edges i < j with probability p.
fn random_poset(n: usize, edge_prob: f64, seed: u64) -> Poset<u32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let nodes: Vec<u32> = (0..n as u32).collect();
    let mut edges = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if rng.gen_bool(edge_prob) {
                edges.push((i as u32, j as u32));
            }
        }
    }
    Poset::from_transitive_relation(nodes, edges).unwrap()
}

// ─── Scaling Benchmarks ──────────────────────────────────────────────────

fn bench_scaling_standard_examples(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_standard_examples");
    group.measurement_time(Duration::from_secs(3));
    
    // Set the plot to a logarithmic scale
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for n in 2..=10usize {
        let poset = standard_example(n);

        group.bench_with_input(BenchmarkId::new("SAT", n), &n, |b, _| {
            b.iter(|| SatReduction.dimension(black_box(&poset)))
        });
        group.bench_with_input(BenchmarkId::new("GCH", n), &n, |b, _| {
            b.iter(|| GraphColoringHeuristic.dimension(black_box(&poset)))
        });
        group.bench_with_input(BenchmarkId::new("Hypergraph", n), &n, |b, _| {
            b.iter(|| HypergraphColoring.dimension(black_box(&poset)))
        });
        group.bench_with_input(BenchmarkId::new("Graph", n), &n, |b, _| {
            b.iter(|| GraphColoring.dimension(black_box(&poset)))
        });
        group.bench_with_input(BenchmarkId::new("Hybrid", n), &n, |b, _| {
            b.iter(|| Hybrid.dimension(black_box(&poset)))
        });
    }
    group.finish();
}


fn bench_scaling_random_posets(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_random_posets");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);

    // Set the plot to a logarithmic scale
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let sizes = [10, 20, 30, 40, 50, 60];
    let density = 0.3;

    for size in sizes {
        let poset = random_poset(size, density, 42);

        group.bench_with_input(BenchmarkId::new("SAT", size), &size, |b, _| {
            b.iter(|| SatReduction.dimension(black_box(&poset)))
        });

        if size < 30 {
            group.bench_with_input(BenchmarkId::new("Graph Coloring", size), &size, |b, _| {
                b.iter(|| GraphColoring.dimension(black_box(&poset)))
            });
            group.bench_with_input(BenchmarkId::new("Hypergraph", size), &size, |b, _| {
                b.iter(|| HypergraphColoring.dimension(black_box(&poset)))
            });
            group.bench_with_input(BenchmarkId::new("Hybrid", size), &size, |b, _| {
                b.iter(|| Hybrid.dimension(black_box(&poset)))
            });
        }

        group.bench_with_input(BenchmarkId::new("GCH", size), &size, |b, _| {
            b.iter(|| GraphColoringHeuristic.dimension(black_box(&poset)))
        });
    }
    group.finish();
}

fn bench_scaling_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_density");
    group.measurement_time(Duration::from_secs(2));

    // Set the plot to a logarithmic scale
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let size = 20;
    let densities = [0.1, 0.3, 0.5, 0.7, 0.9];

    for p in densities {
        let poset = random_poset(size, p, 1337);

        let p_label = (p * 100.0) as u32;

        group.bench_with_input(BenchmarkId::new("SAT", p_label), &p_label, |b, _| {
            b.iter(|| SatReduction.dimension(black_box(&poset)))
        });
        group.bench_with_input(BenchmarkId::new("GCH", p_label), &p_label, |b, _| {
            b.iter(|| GraphColoringHeuristic.dimension(black_box(&poset)))
        });
        group.bench_with_input(BenchmarkId::new("Hybrid", p_label), &p_label, |b, _| {
            b.iter(|| Hybrid.dimension(black_box(&poset)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_scaling_standard_examples,
    bench_scaling_random_posets,
    bench_scaling_density,
);
criterion_main!(benches);