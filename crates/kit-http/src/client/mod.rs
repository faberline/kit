//! `kit_http::client` — high-performance async HTTP client.
//!
//! Connection-pooled async HTTP client with automatic latency measurement,
//! request building, and middleware. The native API toolkit owns both
//! server-side route registration and client-side request execution.
//!
//! Mamba bindings are exposed through the sibling `kit-http-binding`
//! crate under `mambalibs.http`.

pub mod client;
pub mod config;
pub mod error;
pub mod middleware;
pub mod request;
pub mod response;

pub use client::HttpClient;
pub use config::{HttpClientConfig, HttpProtocolPolicy};
pub use error::{HttpError, HttpResult};
pub use middleware::{
    AuthMiddleware, LoggingMiddleware, Middleware, MiddlewareChain, MiddlewareFuture,
    MiddlewareResult, Next, RetryConfig, RetryMiddleware,
};
// `RequestBuilder` is now a type alias for the unified `kit_http::http::Request`;
// `HttpResponse` for `kit_http::http::Response`. Old imports still work.
pub use request::{HttpMethod, RequestBuilder};
pub use response::HttpResponse;

pub use kit_core::http::{HttpRequestLike, HttpResponseLike, HttpStatus};
