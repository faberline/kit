//! `HttpClient` retries 429 and 5xx responses when `HttpClientConfig::retry`
//! is set. The server is a small loopback fixture that replies with a
//! scripted status per connection.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use kit_http::client::config::RetryConfig;
use kit_http::client::{HttpClient, HttpClientConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// One reply per connection: (status, optional Retry-After value).
/// Connection `i` gets `script[i]`; past the end, the last reply repeats.
type Script = Vec<(u16, Option<&'static str>)>;

/// Start the loopback server. Returns its base URL and a counter of the
/// requests it has received.
async fn start_server(script: Script) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let seen = count.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let i = seen.fetch_add(1, Ordering::SeqCst);
            let (status, retry_after) = script[i.min(script.len() - 1)];

            // Read the request head up to the blank line.
            let mut head = Vec::new();
            let mut buf = [0u8; 1024];
            while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                match stream.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => head.extend_from_slice(&buf[..n]),
                }
            }

            let mut response =
                format!("HTTP/1.1 {status} Scripted\r\nContent-Length: 0\r\nConnection: close\r\n");
            if let Some(value) = retry_after {
                response.push_str(&format!("Retry-After: {value}\r\n"));
            }
            response.push_str("\r\n");
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });
    (format!("http://{addr}"), count)
}

fn retrying_client() -> HttpClient {
    HttpClient::new(HttpClientConfig::new().retry(RetryConfig {
        max_retries: 2,
        initial_delay: Duration::from_millis(10),
        ..RetryConfig::default()
    }))
    .unwrap()
}

#[tokio::test]
async fn retries_429_then_ok() {
    let (base, count) = start_server(vec![(429, None), (200, None)]).await;

    let response = retrying_client().get(&format!("{base}/x")).await.unwrap();

    assert_eq!(response.status_code, 200);
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn retry_after_header_is_honored() {
    let (base, count) = start_server(vec![(429, Some("1")), (200, None)]).await;

    let started = Instant::now();
    let response = retrying_client().get(&format!("{base}/x")).await.unwrap();
    let elapsed = started.elapsed();

    assert_eq!(response.status_code, 200);
    assert_eq!(count.load(Ordering::SeqCst), 2);
    // The backoff alone is 10 ms, so this fails if Retry-After is ignored.
    assert!(
        elapsed >= Duration::from_millis(900),
        "expected to wait about 1 s for Retry-After, waited {elapsed:?}"
    );
}

#[tokio::test]
async fn permanent_500_stops_at_the_bound() {
    let (base, count) = start_server(vec![(500, None)]).await;

    let response = retrying_client().get(&format!("{base}/x")).await.unwrap();

    // The 5xx response is returned, not an error, after 1 try + 2 retries.
    assert_eq!(response.status_code, 500);
    assert_eq!(count.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn status_400_is_not_retried() {
    let (base, count) = start_server(vec![(400, None), (200, None)]).await;

    let response = retrying_client().get(&format!("{base}/x")).await.unwrap();

    assert_eq!(response.status_code, 400);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn no_retry_without_config() {
    let (base, count) = start_server(vec![(429, None), (200, None)]).await;
    let client = HttpClient::new(HttpClientConfig::new()).unwrap();

    let response = client.get(&format!("{base}/x")).await.unwrap();

    assert_eq!(response.status_code, 429);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
