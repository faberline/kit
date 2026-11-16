# Architecture

## Dependency graph

```text
frame -> arraykit
scikit -> arraykit   (optional dependency)
scikit -> frame      (optional dependency)
agent -> kit-http
agent -> agent-derive
kit-http -> kit-core
agent-graph -> agent
agent-tools -> agent
agent-derive -> agent (dev only, for tests)
```

`plotkit`, `arraykit` and `kit-core` have no kit dependency.

## Crates

- `arraykit`: N-dimensional arrays (inspired by NumPy): shapes, broadcasting, linear algebra basics.
- `frame`: DataFrame and Series (inspired by pandas): grouping, joins, CSV and JSON I/O.
- `scikit`: Scientific computing (inspired by SciPy), by feature: `stats`, `fft`, `signal`, `interpolate`, `optimize`, `ts`, `spatial`, `sparse`, `integrate`.
- `plotkit`: Data visualization (inspired by Matplotlib) chart model and rendering.
- `agent`: LLM agent framework: agents, tools, LLM providers, MCP, structured output, integrations.
- `agent-derive`: `AgentSchema` derive macro for typed agent input and output.
- `agent-graph` (package `agentkit-graph`): Stateful agent graph runtime.
- `agent-tools` (package `agentkit-tools`): MCP client and server tools for agents.
- `kit-http`: HTTP client and server-side route model. `agent` uses it.
- `kit-core`: Small shared error, error-sanitizing and HTTP scalar types.

## Why kit-core and kit-http are separate

`kit-core` holds the small shared error, error-sanitizing and HTTP scalar types and has no dependency. `kit-http` builds the client and route model on it. A crate that needs only the types does not pull in an HTTP stack.

## Exceptions

- Only `arraykit`, `frame` and `kit-core` have a `tests/public_api.rs` (three standalone tests). The other crates have none.
- The package names `agentkit-graph` and `agentkit-tools` do not match their directories `agent-graph` and `agent-tools`.
- `ddd.toml` is not added until the `rust-arch` checker is ported.
