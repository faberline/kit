use std::time::Duration;

use futures::StreamExt;
use kit_http::client::config::RetryConfig;
use kit_http::client::{HttpClient, HttpClientConfig, HttpError, HttpMethod};
use kit_http::http::Request;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn server_answering(http_method: &str, status: u16) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method(http_method))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(status))
        .mount(&server)
        .await;
    server
}

fn client_for(server: &MockServer) -> HttpClient {
    HttpClient::new(HttpClientConfig::new().base_url(server.uri())).unwrap()
}

/// Retry config with short delays; `Retry-After` of 1s is ignored for 500.
fn quick_retry() -> RetryConfig {
    RetryConfig {
        max_retries: 2,
        initial_delay: Duration::from_millis(10),
        retry_on_status: vec![429, 500, 503],
        ..RetryConfig::default()
    }
}

#[tokio::test]
async fn client_get_returns_status() {
    let server = server_answering("GET", 200).await;
    let response = client_for(&server).get("/x").await.unwrap();
    assert_eq!(response.status_code, 200);
}

#[tokio::test]
async fn client_post_sends_json_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(201))
        .mount(&server)
        .await;

    let response = client_for(&server)
        .post("/x", serde_json::json!({"k": "v"}))
        .await
        .unwrap();
    assert_eq!(response.status_code, 201);

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].body, br#"{"k":"v"}"#.to_vec());
}

#[tokio::test]
async fn client_put_returns_status() {
    let server = server_answering("PUT", 200).await;
    let response = client_for(&server)
        .put("/x", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(response.status_code, 200);
}

#[tokio::test]
async fn client_patch_returns_status() {
    let server = server_answering("PATCH", 200).await;
    let response = client_for(&server)
        .patch("/x", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(response.status_code, 200);
}

#[tokio::test]
async fn client_delete_returns_status() {
    let server = server_answering("DELETE", 204).await;
    let response = client_for(&server).delete("/x").await.unwrap();
    assert_eq!(response.status_code, 204);
}

#[tokio::test]
async fn client_head_returns_status() {
    let server = server_answering("HEAD", 200).await;
    let response = client_for(&server).head("/x").await.unwrap();
    assert_eq!(response.status_code, 200);
}

#[tokio::test]
async fn client_options_returns_status() {
    let server = server_answering("OPTIONS", 200).await;
    let response = client_for(&server).options("/x").await.unwrap();
    assert_eq!(response.status_code, 200);
}

#[tokio::test]
async fn default_client_builds_without_base_url() {
    let client = HttpClient::default_client().unwrap();
    assert!(client.base_url().is_none());
}

#[tokio::test]
async fn base_url_returns_configured() {
    let server = MockServer::start().await;
    let client = client_for(&server);
    assert_eq!(client.base_url(), Some(server.uri().as_str()));
}

#[tokio::test]
async fn debug_format_shows_base_url_and_timeout() {
    let server = MockServer::start().await;
    let client = client_for(&server);
    let text = format!("{client:?}");
    assert!(text.contains("HttpClient"));
    assert!(text.contains(&server.uri()));
    assert!(text.contains("timeout"));
}

#[tokio::test]
async fn request_builder_helper_sets_method() {
    let client = HttpClient::default_client().unwrap();
    let builder = client.request(HttpMethod::Put, "/x");
    assert_eq!(builder.method, HttpMethod::Put);
    assert_eq!(builder.url, "/x");
}

#[tokio::test]
async fn new_with_redirects_disabled_returns_302() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/r"))
        .respond_with(ResponseTemplate::new(302).insert_header("Location", "/target"))
        .mount(&server)
        .await;

    let client = HttpClient::new(
        HttpClientConfig::new()
            .base_url(server.uri())
            .follow_redirects(false),
    )
    .unwrap();
    let response = client.get("/r").await.unwrap();
    assert_eq!(response.status_code, 302);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn new_with_invalid_certs_flags_builds() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let mut config = HttpClientConfig::new()
        .base_url(server.uri())
        .danger_accept_invalid_certs(true);
    config.danger_accept_invalid_hostnames = true;
    let client = HttpClient::new(config).unwrap();

    assert_eq!(client.get("/x").await.unwrap().status_code, 200);
}

#[tokio::test]
async fn execute_stream_returns_chunks() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200).set_body_string("hello world"))
        .mount(&server)
        .await;

    let client = client_for(&server);
    let extracted = kit_http::client::request::ExtractedRequest::from(Request::get("/x"));
    let mut stream = client.execute_stream(extracted).await.unwrap();

    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(out, b"hello world");
}

#[tokio::test]
async fn send_stream_returns_chunks() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200).set_body_string("abc"))
        .mount(&server)
        .await;

    let mut stream = client_for(&server)
        .send_stream(Request::get("/x"))
        .await
        .unwrap();

    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(out, b"abc");
}

#[tokio::test]
async fn execute_builder_stream_returns_chunks() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200).set_body_string("xyz"))
        .mount(&server)
        .await;

    let mut stream = client_for(&server)
        .execute_builder_stream(Request::get("/x"))
        .await
        .unwrap();

    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(out, b"xyz");
}

#[tokio::test]
async fn execute_builder_sends_request() {
    let server = server_answering("GET", 200).await;
    let response = client_for(&server)
        .execute_builder(Request::get("/x"))
        .await
        .unwrap();
    assert_eq!(response.status_code, 200);
}

#[tokio::test]
async fn execute_stream_error_status_is_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(500).set_body_string("bad"))
        .mount(&server)
        .await;

    let result = client_for(&server).send_stream(Request::get("/x")).await;
    let Err(err) = result else {
        panic!("expected a status error");
    };
    assert!(matches!(
        err,
        HttpError::Status { code: 500, ref message } if message == "bad"
    ));
}

#[tokio::test]
async fn retry_after_header_seconds_is_used() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "0"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let client = HttpClient::new(HttpClientConfig::new().retry(quick_retry())).unwrap();
    let response = client
        .send(Request::get(format!("{}/x", server.uri())))
        .await
        .unwrap();

    assert_eq!(response.status_code, 200);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn retry_after_on_success_is_ignored() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200).insert_header("Retry-After", "0"))
        .mount(&server)
        .await;

    let client = HttpClient::new(HttpClientConfig::new().retry(quick_retry())).unwrap();
    let response = client
        .send(Request::get(format!("{}/x", server.uri())))
        .await
        .unwrap();

    assert_eq!(response.status_code, 200);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn retry_after_invalid_value_falls_back_to_config_delay() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "abc"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let client = HttpClient::new(HttpClientConfig::new().retry(quick_retry())).unwrap();
    let response = client
        .send(Request::get(format!("{}/x", server.uri())))
        .await
        .unwrap();

    assert_eq!(response.status_code, 200);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn retry_after_on_500_is_ignored_and_config_delay_used() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(500).insert_header("Retry-After", "1"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let client = HttpClient::new(HttpClientConfig::new().retry(quick_retry())).unwrap();
    let started = std::time::Instant::now();
    let response = client
        .send(Request::get(format!("{}/x", server.uri())))
        .await
        .unwrap();

    assert_eq!(response.status_code, 200);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
    // The config delay (10 ms) applies; the 1 s Retry-After is not used.
    assert!(started.elapsed() < Duration::from_millis(900));
}

#[tokio::test]
async fn retry_503_without_retry_after_uses_config_delay() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/x"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let client = HttpClient::new(HttpClientConfig::new().retry(quick_retry())).unwrap();
    let response = client
        .send(Request::get(format!("{}/x", server.uri())))
        .await
        .unwrap();

    assert_eq!(response.status_code, 200);
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

/// A raw server that speaks only HTTP/1.1 on 127.0.0.1 (no h2c preface).
async fn start_http1_only_server() -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 1024];
            loop {
                match stream.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await;
            let _ = stream.shutdown().await;
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn require_http2_error_is_connection_error() {
    let base = start_http1_only_server().await;
    let client = HttpClient::new(HttpClientConfig::new().base_url(base).require_http2()).unwrap();

    let result = client.get("/x").await;
    let Err(err) = result else {
        panic!("expected HTTP/2 to be required and refused by an HTTP/1.1 server");
    };
    match err {
        HttpError::Connection(msg) => assert!(msg.contains("HTTP/2 required"), "message: {msg}"),
        other => panic!("expected Connection error, got {other:?}"),
    }
}

#[tokio::test]
async fn send_to_closed_port_is_error() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let client = HttpClient::new(HttpClientConfig::new()).unwrap();
    let result = client
        .send(Request::get(format!("http://127.0.0.1:{port}/x")))
        .await;
    assert!(result.is_err());
}
