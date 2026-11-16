# kit-core

Small shared Rust support crate for the kit crates: one error type, error sanitizing and HTTP scalar types.

- `KitError` and `Result`: shared error and result types. `is_retryable` and `is_constraint_violation` classify an error.
- `sanitize_error`, `sanitize_error_message`, `categorize_error` and `ErrorCategory`: remove connection strings, credentials, internal IP addresses and tokens from an error text before logging.
- `HttpMethod`, `HttpStatus`, `HttpRequestLike` and `HttpResponseLike`: HTTP method and status types, and the request and response traits that higher-level kit crates use.

## Example

```rust
use kit_core::{sanitize_error, HttpMethod};

let _text = sanitize_error("connect failed: password=hunter2", false);
let _method = HttpMethod::Get;
```

## Test

```bash
cargo test -p kit-core
```
