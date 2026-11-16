# Duplicate-code check (S4)

## Method

- Kind of check: manual comparison by role. Date of the run: 2026-11-16 (UTC).
- The installed `aw` has an `aw standards check` command. It needs a `faberline/core` checkout next to the repository, and this run has none, so the command could not run (it exits with a usage error, code 2). This check does not use it.
- The `faberline/core` source is not readable from this repository. Each verdict uses only the role table "Which crate does what" in `docs/standards/core-reuse.md` of `faberline/aw`.
- `unknown` means that the role table does not decide the case: a candidate crate matches only part of the module's job.
- `duplicate` needs a core crate with the same job, and the module's own doc text must show that job.

## kit-core

| Module | Job | Core crate with the same job | Verdict | Reason |
|---|---|---|---|---|
| `lib` | Crate root: shared functionality used by all kit crates | none | not a duplicate | Crate root only; the role table has no shared-utility crate. |
| `error` | Error types for kit | none | not a duplicate | The role table has no crate for kit error types. |
| `error_utils` | Error sanitization utilities shared across kit crates | none | not a duplicate | Sanitization is not the `{error, message}` envelope of `service-http`. |
| `http` | Shared HTTP types for the kit ecosystem | none | not a duplicate | The role table has no crate for shared HTTP types. |
| `utils` | String validation utilities | none | not a duplicate | The role table has no string-utility crate. |

## kit-http

| Module | Job | Core crate with the same job | Verdict | Reason |
|---|---|---|---|---|
| `lib` | Crate root: app and route metadata, host contract, wire types | none | not a duplicate | Crate root only. |
| `app` | Core API application model: route registry | none | not a duplicate | The role table has no route-registry crate. |
| `app/validation` | JSON schema value validation for app inputs | none | not a duplicate | The role table has no schema-validation crate. |
| `health` | Aggregate health status of a component | `service-http` | unknown | `service-http` covers probe routes and readiness wiring. The module is only a status type, so the job matches in part. |
| `host` | Native host contract: TCP, keepalive, HTTP/1.1, HTTP/2, shutdown, backpressure | `server-http` | unknown | `server-http` covers HTTP/1.1 and h2c serving. The module also owns TCP, shutdown and backpressure, which other roles cover; the match is in part. |
| `http_exception` | HTTP exception type with a validated status code | `service-http` | unknown | `service-http` owns the `{error, message}` envelope. The module is an exception type, so the match is in part. |
| `protocol` | Native protocol model normalizing HTTP/1.1 and HTTP/2 for host implementations | `server-http` | unknown | `server-http` serves HTTP/1.1 and h2c. The module is a model type only, so the match is in part. |
| `request_response` | Cookie type with validated name | none | not a duplicate | No role table row covers this type. |
| `http` (module root) | Unified HTTP wire-level types shared by server and client | none | not a duplicate | The role table has no shared wire-types crate. |
| `http/auth` | Authentication carried on the request side | `service-auth` | unknown | `service-auth` is request authentication middleware. The module is an auth value type, so the match is in part. |
| `http/body` | Request body model: tagged union of wire representations | none | not a duplicate | The role table has no body-model crate. |
| `http/cookie` | HTTP cookie type, shared by server and client | none | not a duplicate | The role table has no cookie crate. |
| `http/request` | Unified HTTP request type | none | not a duplicate | The role table has no request-type crate. |
| `http/response` | Unified HTTP response type | none | not a duplicate | The role table has no response-type crate. |
| `client` (module root) | Connection-pooled async HTTP client | `transport-h2c` | unknown | `transport-h2c` covers HTTP/2 cleartext client and pooling. The module is generic over HTTP versions, so the match is in part. |
| `client/client` | HTTP client with connection pooling and async operations | `transport-h2c` | unknown | Pooling matches `transport-h2c`. The client is not limited to HTTP/2 cleartext, so the match is in part. |
| `client/config` | HTTP client configuration | none | not a duplicate | `transport-h2c` covers the client and pooling only, not its configuration. |
| `client/error` | HTTP client error types | none | not a duplicate | `transport-h2c` does not cover client error types. |
| `client/middleware` | Middleware trait and chain for request and response processing | none | not a duplicate | The role table has no middleware-chain crate; `service-auth` is one authentication middleware only. |
| `client/request` | Client-side request adapter | none | not a duplicate | Adapter over the shared `http` request type; no role table row. |
| `client/response` | Client-side response adapter | none | not a duplicate | Adapter over the shared `http` response type; no role table row. |

## Real duplicates

None found.
