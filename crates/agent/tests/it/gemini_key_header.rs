//! The Gemini API key must travel in the `x-goog-api-key` header, never in the URL,
//! and must not reach an error string. Uses a loopback server; no network.

use agent::llm::{CompletionRequest, GeminiProvider, LLMProvider};
use agent::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Accept one connection, send the raw request text through the returned receiver,
/// then answer with `status_line` and `body`. Returns the bound port.
async fn serve_once(
    status_line: &'static str,
    body: &'static str,
) -> (u16, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut raw: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];

        // Read until the end of the headers.
        let header_end = loop {
            if let Some(pos) = find_subsequence(&raw, b"\r\n\r\n") {
                break pos + 4;
            }
            let n = stream.read(&mut chunk).await.unwrap();
            if n == 0 {
                break raw.len();
            }
            raw.extend_from_slice(&chunk[..n]);
        };

        // Read the body, using Content-Length.
        let headers = String::from_utf8_lossy(&raw[..header_end]).to_string();
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.trim().eq_ignore_ascii_case("content-length") {
                    value.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);
        while raw.len() < header_end + content_length {
            let n = stream.read(&mut chunk).await.unwrap();
            if n == 0 {
                break;
            }
            raw.extend_from_slice(&chunk[..n]);
        }

        let _ = tx.send(String::from_utf8_lossy(&raw).to_string());

        let response = format!(
            "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes()).await;
        let _ = stream.shutdown().await;
    });

    (port, rx)
}

#[tokio::test]
async fn api_key_is_sent_in_header_not_in_url() {
    let (port, rx) = serve_once(
        "HTTP/1.1 200 OK",
        r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}"#,
    )
    .await;

    let provider =
        GeminiProvider::with_base_url("sk-test-key-123", format!("http://127.0.0.1:{port}"))
            .unwrap();
    let request = CompletionRequest::new(vec![Message::user("hi")], "gemini-2.0-flash");

    let response = provider
        .complete(request)
        .await
        .expect("complete must succeed");
    assert_eq!(response.content, "ok");

    let raw = rx.await.expect("the server must capture the request");
    let first_line = raw.lines().next().unwrap_or_default();
    assert!(
        first_line.starts_with("POST /v1beta/models/gemini-2.0-flash:generateContent HTTP/1.1"),
        "request line: {first_line}"
    );
    assert!(!first_line.contains("key="), "request line: {first_line}");
    assert!(
        !first_line.contains("sk-test-key-123"),
        "request line: {first_line}"
    );
    assert!(
        raw.to_lowercase()
            .contains("x-goog-api-key: sk-test-key-123"),
        "raw request: {raw}"
    );
}

#[tokio::test]
async fn error_string_does_not_contain_the_key() {
    let (port, _rx) = serve_once(
        "HTTP/1.1 400 Bad Request",
        r#"{"error":{"message":"bad key sk-test-key-123"}}"#,
    )
    .await;

    let provider =
        GeminiProvider::with_base_url("sk-test-key-123", format!("http://127.0.0.1:{port}"))
            .unwrap();
    let request = CompletionRequest::new(vec![Message::user("hi")], "gemini-2.0-flash");

    let err = match provider.complete(request).await {
        Err(err) => err,
        Ok(_) => panic!("complete must fail on a 400 response"),
    };
    let text = err.to_string();
    assert!(!text.contains("sk-test-key-123"), "error: {text}");
    assert!(text.contains("400"), "error: {text}");
}
