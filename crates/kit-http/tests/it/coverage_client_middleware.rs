use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use kit_http::client::middleware::{
    AuthMiddleware, LoggingMiddleware, Middleware, MiddlewareChain, MiddlewareFuture, Next,
    RetryConfig, RetryMiddleware,
};
use kit_http::client::request::{ExtractedAuth, ExtractedBody, ExtractedRequest};
use kit_http::client::{HttpClient, HttpClientConfig, HttpError, HttpMethod};
use kit_http::http::Response;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn extracted(url: &str) -> ExtractedRequest {
    ExtractedRequest {
        method: HttpMethod::Get,
        url: url.to_string(),
        headers: Vec::new(),
        query_params: Vec::new(),
        body: ExtractedBody::None,
        auth: ExtractedAuth::None,
        timeout_ms: None,
    }
}

/// A `next` that records the request headers and answers 200.
fn recording_next(captured: Arc<Mutex<Vec<(String, String)>>>) -> Next {
    Arc::new(move |r: ExtractedRequest| {
        let captured = captured.clone();
        Box::pin(async move {
            captured.lock().unwrap().extend(r.headers);
            Ok(Response::new(200))
        }) as MiddlewareFuture
    })
}

/// A `next` that counts its calls and always fails with a network error.
fn failing_next(calls: Arc<AtomicUsize>) -> Next {
    Arc::new(move |_r: ExtractedRequest| {
        calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Err(HttpError::Network("boom".into())) }) as MiddlewareFuture
    })
}

/// A `next` that sends the request through a real client (no retry configured).
fn via_client(client: HttpClient) -> Next {
    Arc::new(move |r: ExtractedRequest| {
        let client = client.clone();
        Box::pin(async move { client.execute(r).await }) as MiddlewareFuture
    })
}

fn fast_retry(max_retries: u32, status_codes: Vec<u16>) -> RetryConfig {
    RetryConfig {
        max_retries,
        initial_backoff_ms: 1,
        backoff_multiplier: 1.0,
        max_backoff_ms: 5,
        retry_status_codes: status_codes,
    }
}

#[tokio::test]
async fn chain_with_boxed_adds_middleware() {
    let chain = MiddlewareChain::new().with_boxed(Arc::new(AuthMiddleware::bearer("t")));

    assert_eq!(chain.len(), 1);
    assert!(!chain.is_empty());
    assert_eq!(chain.names(), vec!["AuthMiddleware"]);
    assert!(MiddlewareChain::new().is_empty());
}

#[tokio::test]
async fn auth_middleware_bearer_adds_authorization_header() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let result = AuthMiddleware::bearer("tok")
        .handle(
            extracted("http://127.0.0.1/x"),
            recording_next(captured.clone()),
        )
        .await;

    assert_eq!(result.unwrap().status_code, 200);
    assert_eq!(
        *captured.lock().unwrap(),
        vec![("authorization".to_string(), "Bearer tok".to_string())]
    );
}

#[tokio::test]
async fn auth_middleware_basic_encodes_credentials() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let result = AuthMiddleware::basic("user", "pass")
        .handle(
            extracted("http://127.0.0.1/x"),
            recording_next(captured.clone()),
        )
        .await;

    assert!(result.is_ok());
    assert_eq!(
        *captured.lock().unwrap(),
        vec![(
            "authorization".to_string(),
            "Basic dXNlcjpwYXNz".to_string()
        )]
    );
}

#[tokio::test]
async fn auth_middleware_custom_uses_value_as_is() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let result = AuthMiddleware::custom("Token abc")
        .handle(
            extracted("http://127.0.0.1/x"),
            recording_next(captured.clone()),
        )
        .await;

    assert!(result.is_ok());
    assert_eq!(
        *captured.lock().unwrap(),
        vec![("authorization".to_string(), "Token abc".to_string())]
    );
}

#[test]
fn auth_middleware_name() {
    assert_eq!(AuthMiddleware::bearer("x").name(), "AuthMiddleware");
}

#[tokio::test]
async fn logging_middleware_passes_through_error() {
    let calls = Arc::new(AtomicUsize::new(0));
    let result = LoggingMiddleware
        .handle(extracted("http://127.0.0.1/x"), failing_next(calls.clone()))
        .await;

    let Err(err) = result else {
        panic!("expected the error of next to pass through");
    };
    assert!(err.to_string().contains("boom"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn logging_middleware_passes_through_ok() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let result = LoggingMiddleware
        .handle(extracted("http://127.0.0.1/x"), recording_next(captured))
        .await;

    assert_eq!(result.unwrap().status_code, 200);
}

#[test]
fn logging_middleware_name() {
    assert_eq!(LoggingMiddleware.name(), "LoggingMiddleware");
}

#[tokio::test]
async fn retry_middleware_retries_status_then_returns_response() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let client = HttpClient::new(HttpClientConfig::new()).unwrap();
    let middleware = RetryMiddleware::new(fast_retry(3, vec![503]));
    let result = middleware
        .handle(
            extracted(&format!("{}/x", server.uri())),
            via_client(client),
        )
        .await;

    assert_eq!(result.unwrap().status_code, 200);
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
async fn retry_middleware_gives_up_after_max_retries() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let client = HttpClient::new(HttpClientConfig::new()).unwrap();
    let middleware = RetryMiddleware::new(fast_retry(2, vec![503]));
    let result = middleware
        .handle(
            extracted(&format!("{}/x", server.uri())),
            via_client(client),
        )
        .await;

    assert_eq!(result.unwrap().status_code, 503);
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
async fn retry_middleware_retries_errors() {
    let calls = Arc::new(AtomicUsize::new(0));
    let middleware = RetryMiddleware::new(fast_retry(2, vec![]));
    let result = middleware
        .handle(extracted("http://127.0.0.1/x"), failing_next(calls.clone()))
        .await;

    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn retry_middleware_does_not_retry_other_status() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let client = HttpClient::new(HttpClientConfig::new()).unwrap();
    let middleware = RetryMiddleware::new(fast_retry(3, vec![503]));
    let result = middleware
        .handle(
            extracted(&format!("{}/x", server.uri())),
            via_client(client),
        )
        .await;

    assert_eq!(result.unwrap().status_code, 404);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[test]
fn retry_middleware_name() {
    assert_eq!(RetryMiddleware::default().name(), "RetryMiddleware");
}

#[test]
fn retry_middleware_default_config_is_applied() {
    let config = RetryConfig::default();
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.initial_backoff_ms, 100);
    assert_eq!(config.max_backoff_ms, 10_000);
    assert!(config.retry_status_codes.contains(&503));
}
