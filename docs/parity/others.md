# Parity for the other crates

The reference libraries are only a guide. The states are `has`, `partial` and `missing`.

- `has`: the feature exists. The evidence path is a file in this repository.
- `partial`: some of the reference API exists, or the row is not yet checked against the whole reference. The phrase after the state says what is missing or what is not checked.
- `missing`: kit has no such feature.

Each row names one public module. Where no reference fits, the reference column says `none (kit only)`.

## plotkit (Matplotlib)

| Module | Reference | State | Evidence |
|---|---|---|---|
| `chart` | `matplotlib.pyplot.figure`, `Axes.set_title`, `set_xlabel`, `set_ylabel` | has | `crates/plotkit/src/viz/chart.rs` |
| `series_data` | `plot`, `bar`, `scatter`, `hist`, `boxplot`, `imshow`, `pie` | has | `crates/plotkit/src/viz/series_data.rs` |
| `subplot` | `matplotlib.pyplot.subplots` | has | `crates/plotkit/src/viz/subplot.rs` |
| `render/svg` | `savefig` to SVG | has | `crates/plotkit/src/viz/render/svg.rs` |
| `export` | `savefig` for other formats | partial; only the formats in `ExportFormat` | `crates/plotkit/src/viz/export.rs` |
| `annotation` | `text`, `annotate` | partial; not all of the reference API is checked | `crates/plotkit/src/viz/annotation.rs` |
| `theme` | `matplotlib.style.use` | partial; named themes only | `crates/plotkit/src/viz/theme.rs` |
| `style` | `rcParams` | partial; not all of the reference API is checked | `crates/plotkit/src/viz/style.rs` |
| `palette` | colormaps (`matplotlib.cm`) | partial; categorical, sequential blue and diverging only | `crates/plotkit/src/viz/palette.rs` |
| `axis` | `matplotlib.ticker` | partial; not all of the reference API is checked | `crates/plotkit/src/viz/axis.rs` |
| `dataframe` | `DataFrame.plot` (pandas) | partial; not all of the reference API is checked | `crates/plotkit/src/viz/dataframe.rs` |
| `error` | none (kit only) | partial; not yet checked | `crates/plotkit/src/viz/error.rs` |

## agent, agent-graph, agent-tools, agent-derive (LangChain-like frameworks)

| Module | Reference | State | Evidence |
|---|---|---|---|
| `agent` | LangChain agent executor | partial; not all of the reference API is checked | `crates/agent/src/agent.rs` |
| `cancellation_timeout` | `asyncio.timeout` | partial; not yet checked | `crates/agent/src/cancellation_timeout.rs` |
| `context` | none (kit only) | partial; not yet checked | `crates/agent/src/context.rs` |
| `error` | none (kit only) | partial; not yet checked | `crates/agent/src/error.rs` |
| `eval` | LangSmith evaluators | partial; not all of the reference API is checked | `crates/agent/src/eval.rs` |
| `events` | LangChain callbacks | partial; not all of the reference API is checked | `crates/agent/src/events.rs` |
| `integrations` | LangChain document loaders | partial; GitHub, GitLab and Jira only | `crates/agent/src/integrations/github.rs` |
| `llm` | LangChain chat models | partial; only the Claude provider streams | `crates/agent/src/llm/claude.rs` |
| `observability` | LangSmith tracing | partial; not all of the reference API is checked | `crates/agent/src/observability.rs` |
| `protocols` | none (kit only) | partial; not yet checked | `crates/agent/src/protocols/spec.rs` |
| `schema` | pydantic JSON schema | partial; not all of the reference API is checked | `crates/agent/src/schema.rs` |
| `security` | none (kit only) | partial; not yet checked | `crates/agent/src/security.rs` |
| `storage` | LangChain chat memory | partial; file and in-memory storage only | `crates/agent/src/storage/file.rs` |
| `stream` | LangChain streaming | partial; only the Claude provider streams | `crates/agent/src/stream.rs` |
| `structured` | `with_structured_output` | partial; not all of the reference API is checked | `crates/agent/src/structured.rs` |
| `sync_adapter` | LangChain document loaders | partial; Confluence, Google Docs, GitHub, GitLab and Jira only | `crates/agent/src/sync_adapter/confluence.rs` |
| `tokenizer` | `tiktoken` | has; OpenAI model tokenizers through tiktoken-rs | `crates/agent/src/tokenizer.rs` |
| `tools` | LangChain `Tool` | partial; not all of the reference API is checked | `crates/agent/src/tools/tool.rs` |
| `types` | none (kit only) | partial; not yet checked | `crates/agent/src/types.rs` |
| `agent-graph` `graph` | LangGraph `StateGraph` | partial; not all of the reference API is checked | `crates/agent-graph/src/graph.rs` |
| `agent-graph` `node` | LangGraph nodes | partial; not all of the reference API is checked | `crates/agent-graph/src/node.rs` |
| `agent-graph` `checkpoint` | LangGraph checkpointers | partial; not all of the reference API is checked | `crates/agent-graph/src/checkpoint.rs` |
| `agent-graph` `events` | LangGraph stream events | partial; not all of the reference API is checked | `crates/agent-graph/src/events.rs` |
| `agent-tools` `mcp_client` | MCP client SDK | partial; no stdio and no SSE transport | `crates/agent-tools/src/mcp_client.rs` |
| `agent-tools` `mcp_server` | MCP server SDK | partial; not all of the reference API is checked | `crates/agent-tools/src/mcp_server.rs` |
| `agent-derive` | pydantic models | partial; not all of the reference API is checked | `crates/agent-derive/src/lib.rs` |

## kit-core and kit-http (requests-like)

| Module | Reference | State | Evidence |
|---|---|---|---|
| `kit-core` `error` | `requests.exceptions` | partial; not all of the reference API is checked | `crates/kit-core/src/error.rs` |
| `kit-core` `error_utils` | none (kit only) | partial; not yet checked | `crates/kit-core/src/error_utils.rs` |
| `kit-core` `http` | none (kit only) | partial; not yet checked | `crates/kit-core/src/http.rs` |
| `kit-core` `utils` | none (kit only) | partial; not yet checked | `crates/kit-core/src/utils.rs` |
| `kit-http` `client` | `requests.Session` | partial; not all of the reference API is checked | `crates/kit-http/src/client/client.rs` |
| `kit-http` `request_response` | `requests.Request`, `requests.Response` | partial; not all of the reference API is checked | `crates/kit-http/src/request_response.rs` |
| `kit-http` `http` | `requests` auth and cookies | partial; not all of the reference API is checked | `crates/kit-http/src/http/auth.rs` |
| `kit-http` `http_exception` | `requests.HTTPError` | partial; not all of the reference API is checked | `crates/kit-http/src/http_exception.rs` |
| `kit-http` `app` | none (kit only) | partial; not yet checked | `crates/kit-http/src/app.rs` |
| `kit-http` `health` | none (kit only) | partial; not yet checked | `crates/kit-http/src/health.rs` |
| `kit-http` `host` | none (kit only) | partial; not yet checked | `crates/kit-http/src/host.rs` |
| `kit-http` `protocol` | none (kit only) | partial; not yet checked | `crates/kit-http/src/protocol.rs` |

## Classic machine learning (scikit-learn)

| Feature | Reference | State | Later issue |
|---|---|---|---|
| `LinearRegression` | `sklearn.linear_model.LinearRegression` | missing | ML: add LinearRegression |
| `Ridge` | `sklearn.linear_model.Ridge` | missing | ML: add Ridge |
| `LogisticRegression` | `sklearn.linear_model.LogisticRegression` | missing | ML: add LogisticRegression |
| `KMeans` | `sklearn.cluster.KMeans` | missing | ML: add KMeans |
| `DecisionTreeClassifier` | `sklearn.tree.DecisionTreeClassifier` | missing | ML: add DecisionTreeClassifier |
| `RandomForest` | `sklearn.ensemble.RandomForestClassifier` | missing | ML: add RandomForest |
| `StandardScaler` | `sklearn.preprocessing.StandardScaler` | missing | ML: add StandardScaler |
| `train_test_split` | `sklearn.model_selection.train_test_split` | missing | ML: add train_test_split |
| `cross_val_score` | `sklearn.model_selection.cross_val_score` | missing | ML: add cross_val_score |
| `PCA` | `sklearn.decomposition.PCA` | missing | ML: add PCA |

## Deep learning (PyTorch)

| Feature | Reference | State | Later issue |
|---|---|---|---|
| Tensor with autograd | `torch.Tensor` with `backward()` | missing | DL: add Tensor with autograd |
| Linear layer | `torch.nn.Linear` | missing | DL: add Linear layer |
| Conv2d | `torch.nn.Conv2d` | missing | DL: add Conv2d |
| ReLU and softmax | `torch.nn.ReLU`, `torch.nn.Softmax` | missing | DL: add ReLU and softmax |
| SGD and Adam optimizers | `torch.optim.SGD`, `torch.optim.Adam` | missing | DL: add SGD and Adam optimizers |
| Loss functions | `torch.nn.CrossEntropyLoss`, `torch.nn.MSELoss` | missing | DL: add Loss functions |
| DataLoader | `torch.utils.data.DataLoader` | missing | DL: add DataLoader |
| Model save and load | `torch.save`, `torch.load` | missing | DL: add Model save and load |
| GPU backend | `torch.device("cuda")` | missing | DL: add GPU backend |
