# odis — Formal Concept Analysis in Rust

`odis` is a Rust library for [Formal Concept Analysis](https://en.wikipedia.org/wiki/Formal_concept_analysis).
It works on formal contexts (objects × attributes incidence tables) and currently provides
concept enumeration, implication bases, lattice drawing, and attribute exploration.

## Installation

```toml
[dependencies]
odis = "2026.3.0"
```

## Quick start

```rust
use odis::{FormalContext, algorithms::NextClosure};
use odis::traits::ConceptEnumerator;

// Parse a Burmeister (.cxt) file
let bytes = std::fs::read("context.cxt").unwrap();
let ctx = FormalContext::<String>::from(&bytes).unwrap();

// Enumerate all formal concepts
for (extent, intent) in NextClosure.enumerate_concepts(&ctx) {
    println!("extent: {:?}  intent: {:?}", extent, intent);
}

// Or use the built-in shortcut (defaults to NextClosure)
let concepts: Vec<_> = ctx.concepts().collect();
```

## Algorithms

### Concept enumeration

[`ConceptEnumerator`](https://docs.rs/odis/latest/odis/traits/trait.ConceptEnumerator.html) is implemented by:

| Struct | Algorithm |
|---|---|
| `NextClosure` | Lectic-order enumeration |
| `Fcbo` | Fast Close-by-One |

### Implication basis

```rust
use odis::algorithms::CanonicalBasis;
use odis::traits::ImplicationEngine;

let basis = CanonicalBasis.compute_named_basis(&ctx);
for (premise, conclusion) in &basis {
    println!("{:?} → {:?}", premise, conclusion);
}
```

### Lattice drawing

[`DrawingAlgorithm`](https://docs.rs/odis/latest/odis/traits/trait.DrawingAlgorithm.html) is implemented by:

| Struct | Algorithm |
|---|---|
| `DimDraw { timeout_ms }` | Realizer-based drawing algorithm; `timeout_ms = 0` for unbounded |
| `Sugiyama` | Hierarchical layout via `rust-sugiyama` |

```rust
use odis::algorithms::{DimDraw, NextClosure};
use odis::traits::{ConceptEnumerator, DrawingAlgorithm};

let lattice = ctx.concept_lattice().unwrap();
let drawing = DimDraw { timeout_ms: 500 }.draw(&lattice).unwrap();
```

### Iceberg concept lattices

```rust
use odis::algorithms::Titanic;
use odis::traits::IcebergConceptEnumerator;

// Concepts whose extent has at least 3 objects
let iceberg = Titanic.enumerate_named_concepts(&ctx, 3);
```

### Attribute exploration

`ExplorationMachine` drives interactive attribute exploration over a growing context.
The state machine separates exploration logic from I/O.

### Poset Dimension

```rust
use odis::algorithms::dimension;
use odis::Poset;

let p = Poset::<u32>::standard_example(3);

// For smaller Posets
let exact_dim_hybrid = dimension::Hybrid.dimension(p);

// For larger Posets
let exact_dim_sat = dimension::SatReduction.dimension(p);

// for an upper bound (For larger Posets)
let dim = dimension::GraphColoringHeuristic.dimension(p);
```

## Formal context format

Contexts are parsed from the [Burmeister `.cxt` format](https://fc-bug-search.uni-wuppertal.de/cxt-file-format):

```
B

2
2

bird
fish
flies
lives_in_water
X.
.X
```

## API documentation

```bash
cargo doc -p odis --no-deps --open
```

## License

[AGPL-3.0-only](LICENSE)
