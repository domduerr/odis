# odis — Formal Concept Analysis in Rust

`odis` is a Rust library for [Formal Concept Analysis](https://en.wikipedia.org/wiki/Formal_concept_analysis).
It works on formal contexts (objects × attributes incidence tables) and currently provides
concept enumeration, implication bases, lattice drawing, and attribute exploration.

## Installation

```toml
[dependencies]
odis = "2026.9.1"
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
| `DimDraw { budget }` | Realizer-based drawing algorithm; `budget` defaults to one second, `SearchBudget::Unbounded` searches to a proven optimum |
| `Sugiyama` | Hierarchical layout via `rust-sugiyama` |

```rust
use odis::algorithms::{DimDraw, NextClosure};
use odis::traits::{ConceptEnumerator, DrawingAlgorithm};

let lattice = ctx.concept_lattice().unwrap();
let drawing = DimDraw::default().draw(&lattice).unwrap();
```

[`ConceptDrawingAlgorithm`](https://docs.rs/odis/latest/odis/traits/trait.ConceptDrawingAlgorithm.html)
is for layouts that need to know which objects and attributes each node stands
for, which a node type generic in `T` cannot tell them:

| Struct | Algorithm |
|---|---|
| `DimFlux { budget, iterations }` | DimDraw layout projected into the space of additive diagrams, then refined by a force-directed model that pushes concept nodes away from the edges they are not part of |

```rust
use odis::{algorithms::DimFlux, traits::ConceptDrawingAlgorithm};

let drawing = DimFlux::default().draw_context(&ctx).unwrap();
```

The same trait draws iceberg lattices, which need no bottom element of their own:

```rust
use odis::{algorithms::{DimFlux, Titanic}, traits::{ConceptDrawingAlgorithm, IcebergConceptEnumerator}};

let iceberg = Titanic.enumerate(&ctx, 3);
let drawing = DimFlux::default().draw_iceberg(&iceberg, ctx.attributes.len()).unwrap();
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

## FCA repository

Contexts published in the FCA literature can be loaded straight from the
[FCA repository](https://fcarepository.org/):

```rust
use odis::repository;

for entry in repository::fetch_catalog().await? {
    println!("{} — {} ({:?} objects)", entry.filename, entry.title, entry.objects);
}

let ctx = repository::fetch_context("livingbeings_en.cxt").await?;
```

The API is async because it also compiles for `wasm32`, where the only available
transport is the browser's fetch API, which has no blocking form. The same code runs
on all targets. `repository::parse_catalog` and `repository::context_url` are public
too, for a caller that already has the bytes.

## API documentation

```bash
cargo doc -p odis --no-deps --open
```

## License

[AGPL-3.0-only](LICENSE)
