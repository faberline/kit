# Kit repository instructions

Kit is a Cargo workspace of pure Rust libraries for data analysis, machine learning and agents.

## Output style

- Write in the user's language. Keep paths, commands and exact output contracts intact.
- Use ASD-STE100 Simplified Technical English: active voice, short sentences, one idea in each sentence.
- Report what changed, what passed, what is unknown and the next action.
- Offer at most two choices and recommend one.

## Scope

- Crates are plain Rust libraries. Do not add `pyo3`, a Python binding or a Python file.
- Keep the "Not there yet" list in `README.md` true. Update it when a feature lands.
- Do not write a secret, a token, a private host name, an e-mail address or a personal path in this repo. It is public.

## Issue work

- Work on a claimed issue only through the [`aw` CLI](https://github.com/faberline/aw): `aw claim`, the work, `aw report`.
  `aw` does the branch, push, pull request, merge and signal comment.
- Edit labels only in the plan step of the `aw-worker` skill.

## Rust

- Layout: [Rust layout standard](https://github.com/faberline/aw/blob/main/docs/standards/rust-layout.md).
- Before a commit, run what CI runs: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`,
  `cargo test --workspace --all-features`.
- Do not delete or weaken a test to make it pass.
