use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use kit_http::client::config::RetryConfig;
use kit_http::client::{HttpClient, HttpClientConfig};
use kit_http::http::Request;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Starts a raw HTTP/1.1 server on 127.0.0.1. Request `i` gets `responses[i]`
/// (status, optional Retry-After value). Past the end, the last entry repeats.
async fn start_server(responses: Vec<(u16, Option<&'static str>)>) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    let served = counter.clone();

    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 1024];
            loop {
                let n = match stream.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                buf.extend_from_slice(&chunk[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }

            let i = served.fetch_add(1, Ordering::SeqCst);
            let (status, retry_after) = responses[i.min(responses.len() - 1)];
            let mut head =
                format!("HTTP/1.1 {status} X\r\nContent-Length: 0\r\nConnection: close\r\n");
            if let Some(value) = retry_after {
                head.push_str(&format!("Retry-After: {value}\r\n"));
            }
            head.push_str("\r\n");
            let _ = stream.write_all(head.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });

    (format!("http://{addr}"), counter)
}

fn retry_client() -> HttpClient {
    let retry = RetryConfig {
        max_retries: 2,
        initial_delay: Duration::from_millis(10),
        ..RetryConfig::default()
    };
    HttpClient::new(HttpClientConfig::new().retry(retry)).unwrap()
}

#[tokio::test]
async fn retries_429_then_ok() {
    let (base, counter) = start_server(vec![(429, None), (200, None)]).await;

    let response = retry_client()
        .send(Request::get(format!("{base}/x")))
        .await
        .unwrap();

    assert_eq!(response.status_code, 200);
    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn retry_after_header_is_honored() {
    let (base, _) = start_server(vec![(429, Some("1")), (200, None)]).await;

    let started = Instant::now();
    let response = retry_client()
        .send(Request::get(format!("{base}/x")))
        .await
        .unwrap();

    assert_eq!(response.status_code, 200);
    assert!(started.elapsed() >= Duration::from_millis(900));
}

#[tokio::test]
async fn permanent_500_stops_at_the_bound() {
    let (base, counter) = start_server(vec![(500, None)]).await;

    let response = retry_client()
        .send(Request::get(format!("{base}/x")))
        .await
        .unwrap();

    assert_eq!(response.status_code, 500);
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn status_400_is_not_retried() {
    let (base, counter) = start_server(vec![(400, None), (200, None)]).await;

    let response = retry_client()
        .send(Request::get(format!("{base}/x")))
        .await
        .unwrap();

    assert_eq!(response.status_code, 400);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn no_retry_without_config() {
    let (base, counter) = start_server(vec![(429, None), (200, None)]).await;

    let client = HttpClient::new(HttpClientConfig::new()).unwrap();
    let response = client
        .send(Request::get(format!("{base}/x")))
        .await
        .unwrap();

    assert_eq!(response.status_code, 429);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}
