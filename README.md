# kit

Rust libraries for data analysis, machine learning and agents.

`kit` is a Cargo workspace of plain Rust crates. They have no Python binding, no
`pyo3` and no runtime dependency outside crates.io.

## Crates

| Crate | What it is |
| --- | --- |
| `arraykit` | N-dimensional arrays (NumPy-like): shapes, broadcasting, linear algebra basics. |
| `frame` | DataFrame and Series (pandas-like): grouping, joins, CSV and JSON I/O. |
| `scikit` | Scientific computing (SciPy-like), by feature: `stats`, `fft`, `signal`, `interpolate`, `optimize`, `ts`, `spatial`, `sparse`, `integrate`. |
| `plotkit` | Data visualization (Matplotlib-like) chart model and rendering. |
| `agent` | LLM agent framework: agents, tools, LLM providers, MCP, structured output, integrations. |
| `agent-derive` | `AgentSchema` derive macro for typed agent input and output. |
| `agent-graph` (package `agentkit-graph`) | Stateful agent graph runtime. |
| `agent-tools` (package `agentkit-tools`) | Reusable agent tool implementations. |
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

These crates were copied from the `mambalibs` kits of a private project. The history
is not imported. Renamed: `mambalibs-http` to `kit-http`, `mamba-core` to `kit-core`.

Not renamed (not mechanical, or still in use as names inside the code):

- The `agent` crate still names its error type `NovaError` and uses the User-Agent
  `cclab-nova`. Several doc comments and test fixtures still say `cclab`.
- `kit-core` still names its error type `DataBridgeError`.
- `kit-http` still emits the OpenAPI extension key `x-mamba-dependencies` and the
  default title `Mamba API`.
- Package names `agentkit-graph` and `agentkit-tools` do not match their directories.
- Some crate READMEs (`frame`, `kit-core`) still say "Cclab".

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
