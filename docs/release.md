# Release checklist

1. Bump `version` in `[workspace.package]` of `Cargo.toml`.
2. Move the `CHANGELOG.md` items from `Unreleased` to the new version heading.
3. Check that `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, `cargo test --workspace --locked` and `cargo deny check` are green on `main`.
4. Run `cargo publish --dry-run --locked -p kit-core`, then `cargo package --list --locked -p <crate>` for the other crates.
5. Create the annotated tag with `git tag -a vX.Y.Z -m "kit X.Y.Z"` and push it.
6. Publish with `cargo publish --locked -p <crate>` in this dependency order: `kit-core`, `kit-http`, `arraykit`, `frame`, `scikit`, `plotkit`, `agent-derive`, `agent`.

A dry-run publish of a crate that depends on an unpublished sibling cannot pass. So only `kit-core` has a dry-run publish. Each later crate gets its dry-run when its dependencies are on crates.io. Sibling `path` dependencies need a `version` key before the real publish.

This file does not publish anything.
