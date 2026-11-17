# Threat model

This page covers the agent tools, the HTTP client, the MCP code and secret handling in this repository. It names the code that enforces each control. It also lists the gaps that the code has today. It is a review aid, not a promise: a control applies only where the named code path runs.

## 1. Agent bash tool

### Assets

- The host shell and the files it can reach.
- Output the model reads back, which can hold secrets from the host.

### Threats

- Command injection from model output. The shell runs `/bin/bash -c` on the whole command string.
- A command that runs forever.
- A command with very large output.
- A path escape through the working directory.

### Controls (code path)

- `SecurityPolicy::check_command` in `crates/agent/src/security.rs` runs before every shell command (`BashTool` in `crates/agent/src/tools/bash.rs`).
- The default policy is deny by default: `SecurityPolicy::default` allows no command. `SecurityPolicy::permissive()` is the explicit opt-out.
- A blocked pattern is refused in every case.
- The timeout kills the child process, and `kill_on_drop` is set (issue #68).
- Output is cut to `max_output_size` on a character boundary with `truncate_at_char_boundary`.
- `bash` is in the default `approval_required` list.

### Gaps

- `check_command` checks only the first word of the command. The shell then runs the whole string, so `ls && rm ...` passes when `ls` is allowed.
- A blocked pattern is a plain substring match on the lowercased command. Two spaces in `rm  -rf /` avoid the match.
- The working directory is not a sandbox. A command can read or write any path that the host user can reach.
- No issue yet.

## 2. Agent file tools

### Assets

- Files in and around the working directory.
- Files outside it that the host user can read or write.

### Threats

- Path traversal with `..`.
- A symlink that points outside the allowed root.
- A write outside the work directory.
- A very large file read or written.

### Controls (code path)

- `SecurityPolicy::check_path` in `crates/agent/src/security.rs` runs in `ReadFileTool`, `WriteFileTool` and `EditFileTool` (`crates/agent/src/tools/file.rs`). It refuses any `..` component.
- The policy allows a path only under `allowed_paths`, unless `allow_all_paths` is set. `blocked_paths` refuses a listed directory in every case.
- An absolute path outside the working directory is refused unless `allow_absolute_paths` is set.
- `max_file_bytes` limits reads and writes. The default is 1 MiB.
- `write_file` requires approval by default.

### Gaps

- `GlobTool` and `GrepTool` do not hold a `SecurityPolicy`. They do not call `check_path`, so they can list and search any path that they are given.
- `check_path` resolves symlinks with `canonicalize` only when the path exists. For a new file, the path is checked as written. A new file under a symlinked directory is therefore not checked against its real target.
- No issue yet.

## 3. web_fetch tool

### Assets

- The internal network that the host can reach.
- The host's memory and time.

### Threats

- Server-side request forgery (SSRF) to a private or local address.
- A redirect to a host that the policy blocks.
- A very large response body.

### Controls (code path)

- `SecurityPolicy::check_url` in `crates/agent/src/security.rs` allows only the `http` and `https` schemes.
- The default allows no host. A host must be in `allowed_hosts`, or `allow_all_hosts` must be set.
- `WebFetchTool` (`crates/agent/src/tools/analysis.rs`) builds its client with `follow_redirects(false)`. It follows redirects itself, up to `MAX_REDIRECTS` (3), and it checks each hop with `check_url`. A blocked redirect target is returned as data and is not requested.
- The body is cut to the smaller of `max_fetch_bytes` and the tool's content limit.

### Gaps

- `check_url` compares the host name as text. It does not resolve the name. A host name that resolves to a private address is not blocked when the name is in `allowed_hosts`.
- `allow_all_hosts` removes the host check entirely.
- No issue yet.

## 4. HTTP client and LLM providers

### Assets

- The API keys for the LLM providers.
- The request and response data sent to each provider.

### Threats

- A plain `http://` base URL sends the key without encryption.
- A redirect sends the key to another host.
- A retry storm against a provider.
- A request that hangs.

### Controls (code path)

- `HttpClientConfig` in `crates/kit-http/src/client/config.rs` sets a total timeout of 30 seconds, a connect timeout of 10 seconds, and a maximum of 10 redirects.
- `RetryConfig` in the same file retries responses with status 429 and 5xx, with backoff. `middleware.rs` applies it.
- TLS uses `rustls` with the system roots. The workspace `Cargo.toml` enables `rustls-tls-native-roots` in `reqwest`. The crates do not use `native-tls`.
- `ClaudeProvider`, `OpenAIProvider` and `GeminiProvider` in `crates/agent/src/llm/` each take a base URL with `with_base_url`. The default base URL is the provider's `https` endpoint.

### Gaps

- `follow_redirects` defaults to `true` in `HttpClientConfig`, so a provider client follows a redirect. Whether the key header is sent to the new host was not checked.
- `with_base_url` accepts any URL. It does not refuse `http://`. The scheme check exists only in `check_url`, which the providers do not call.
- The `max_body_bytes` field in `crates/kit-http/src/protocol.rs` (16 MiB by default) has no reader in the crates. No response size limit was found in the HTTP client.
- No issue yet.

## 5. MCP client and server

### Assets

- The local tools that the MCP server exposes.
- The messages that the client sends and receives.

### Threats

- A hostile server sends a huge or malformed message.
- A server answers a request with the wrong id, or floods the client with unrelated messages.
- A caller runs a server tool with no check of who the caller is.
- The transport has no encryption.

### Controls (code path)

- `McpClient` in `crates/agent-tools/src/mcp_client.rs` checks the `jsonrpc` version of each reply. An error reply becomes a `ToolError`.
- The client stops after 100 unrelated messages (`too many unrelated messages`) and after too many result pages (`too many pages`).
- `McpServer` in `crates/agent-tools/src/mcp_server.rs` maps a handler error to a JSON-RPC internal error. A notification gets no reply.
- `McpTransport` is a trait with `send` and `recv`. The caller supplies the transport.

### Gaps

- The client has no size limit on one message.
- The protocol layer has no authentication. `McpServer` runs any registered tool for any caller.
- `register` replaces an existing tool with the same name, without a warning.
- The transport encryption is the caller's job. No code in the crates sets it.
- Open issues #149 (stdio transport for client and server) and #150 (bridge MCP tools into `Agent` tools) cover the missing parts of the transport and the bridge.

## 6. Secret handling

### Assets

- The API keys for the LLM providers.

### Threats

- A key appears in an error message, a log line or a `Debug` print.
- A key appears in a URL, where a proxy or a log can record it.

### Controls (code path)

- `redact` in `crates/agent/src/llm/claude.rs` replaces the key in the error text of a failed Claude request.
- `api_error` in `crates/agent/src/llm/openai.rs` replaces the key in the body of a failed OpenAI request. Its test `api_error_redacts_the_key` checks the replacement.
- `redact_key` in `crates/agent/src/llm/gemini.rs` replaces the key in the transport error and the error body. The Gemini key goes in the `x-goog-api-key` header, not in the URL. The test `crates/agent/tests/it/gemini_key_header.rs` checks the header.
- `ClaudeProvider`, `OpenAIProvider` and `GeminiProvider` do not derive `Debug`. A `{:?}` print of a provider cannot show `api_key`.

### Gaps

- The key is a `String`, not a type that clears its memory on drop.
- The redaction is a text replace. It works only when the error text holds the key as the same string.
- The `redact` in `claude.rs` covers the error body. Other error paths were not checked for the key.
- The key is sent as a header value, for example `Bearer` in `openai.rs`. A header value is not in a URL. Do not add it to a URL or a log.

## Review

Reviewed on 2026-11-17.
