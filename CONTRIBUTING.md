# Contributing

Issue work goes through the `aw` CLI (`aw claim`, the work, `aw report`), as [AGENTS.md](AGENTS.md) says.

## Checks

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo deny check
cargo test --workspace
cargo test --workspace --all-features
```

CI runs the same commands with `--locked` (see `.github/workflows/ci.yml`).

A public API change also updates `crates/kit-core/tests/public_api.rs` and `CHANGELOG.md` (see [docs/stability.md](docs/stability.md)).
