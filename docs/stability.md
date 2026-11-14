# Stability promise

## Versioning

kit follows Semantic Versioning. Crates are `0.x` now and may break between minor versions. From `1.0.0` no breaking change before `2.0.0`.

## What is public API

Every `pub` item reachable from a crate root is public API. `kit-core` is covered by `crates/kit-core/tests/public_api.rs`. `arraykit` and `frame` are covered only for their root re-exports (`DType`, `NdArray`, `Shape`; `DataFrame`, `Series`). The other crates are not yet covered.

## Minimum Rust version

`1.99.1`, the version in `rust-toolchain.toml`. Raising it is a minor change.

## How a breaking change is made

Change the item, update `crates/kit-core/tests/public_api.rs` in the same pull request, and add a line under `Unreleased` in `CHANGELOG.md`.
