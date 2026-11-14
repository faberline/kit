# kit

Rust libraries for data analysis, machine learning and agents.

`kit` is a Cargo workspace of plain Rust crates. They have no Python binding, no
`pyo3` and no runtime dependency outside crates.io.

## Status

The crates are experimental.

Results are not yet checked against reference libraries (NumPy, pandas, SciPy).

A quality check found wrong results. Open issues track them: #3 (frame), #4 (arraykit and scikit), #5 (agent), and #1, #7 to #11 (CI and clippy).

See [docs/stability.md](docs/stability.md), [docs/limits.md](docs/limits.md) and [CHANGELOG.md](CHANGELOG.md).

## Crates

| Crate | What it is |
| --- | --- |
| `arraykit` | N-dimensional arrays (inspired by NumPy): shapes, broadcasting, linear algebra basics. |
| `frame` | DataFrame and Series (inspired by pandas): grouping, joins, CSV and JSON I/O. |
| `scikit` | Scientific computing (inspired by SciPy), by feature: `stats`, `fft`, `signal`, `interpolate`, `optimize`, `ts`, `spatial`, `sparse`, `integrate`. |
| `plotkit` | Data visualization (inspired by Matplotlib) chart model and rendering. |
| `agent` | LLM agent framework: agents, tools, LLM providers, MCP, structured output, integrations. |
| `agent-derive` | `AgentSchema` derive macro for typed agent input and output. |
| `agent-graph` (package `agentkit-graph`) | Stateful agent graph runtime. |
| `agent-tools` (package `agentkit-tools`) | MCP client and server tools for agents. |
| `kit-http` | HTTP client and server-side route model. `agent` uses it. |
| `kit-core` | Small shared error, error-sanitizing and HTTP scalar types. |

## Not there yet

- No autograd, neural network layers, optimizers, GPU support or model file formats.
- No machine-learning estimators.
- `frame` is row-oriented, not columnar.
- In `agent`, only the Claude provider streams. Other providers do not.
- No embeddings and no vector store.
- The MCP support has no stdio transport and no SSE transport.

## Origin and renames

These crates were copied from a private project. The history is not imported. `mambalibs-http` became `kit-http` and `mamba-core` became `kit-core`.

The package names `agentkit-graph` and `agentkit-tools` do not match their directories.

## Known gaps in checks

- `cargo clippy --workspace --all-targets -- -D warnings` does not pass yet. The copied
  code has about 100 clippy warnings and a few deny-level lints in tests (`approx_constant`
  in `frame` and `scikit` tests). CI runs `cargo clippy --workspace` on library targets only.
- `cargo deny check` was not run.

## Develop

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

License: MIT, see `LICENSE`.
