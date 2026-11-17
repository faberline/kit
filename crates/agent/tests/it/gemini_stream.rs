//! Gemini streaming against a loopback server: the request shape, the text, the
//! function call, the final stop reason and key redaction.
//! Uses a loopback server; no network, no paid API.

use agent::llm::{CompletionRequest, GeminiProvider, LLMProvider};
use agent::{AgentError, Message};
use futures::StreamExt;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Read one request (headers and a Content-Length body) from the connection.
async fn read_request(stream: &mut TcpStream) -> String {
    let mut raw: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];

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

    String::from_utf8_lossy(&raw).to_string()
}

/// Accept one connection, send the raw request text through the receiver, then answer
/// with `status_line` and a close-delimited body written in `pieces`, one write each,
/// with a short pause between writes so that the stream arrives in parts.
async fn serve(
    status_line: &'static str,
    content_type: &'static str,
    pieces: Vec<&'static str>,
) -> (u16, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_request(&mut stream).await;
        let _ = tx.send(request);

        let head =
            format!("{status_line}\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n");
        let _ = stream.write_all(head.as_bytes()).await;
        for piece in pieces {
            let _ = stream.write_all(piece.as_bytes()).await;
            let _ = stream.flush().await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let _ = stream.shutdown().await;
    });

    (port, rx)
}

/// Three events: text "Hel", text "lo" (split across two writes), then a function
/// call with a finish reason of STOP.
fn text_and_call_pieces() -> Vec<&'static str> {
    vec![
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hel\"}]}}]}\r\n\r\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"te",
        "xt\":\"lo\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"q\":\"x\"}}}]},\"finishReason\":\"STOP\"}]}\n\n",
    ]
}

#[tokio::test]
async fn stream_yields_text_function_call_and_stop() {
    let (port, _rx) = serve(
        "HTTP/1.1 200 OK",
        "text/event-stream",
        text_and_call_pieces(),
    )
    .await;

    let provider =
        GeminiProvider::with_base_url("sk-test-key-123", format!("http://127.0.0.1:{port}"))
            .unwrap();
    let request = CompletionRequest::new(vec![Message::user("hi")], "gemini-2.0-flash");

    let stream = provider
        .complete_stream(request)
        .await
        .expect("complete_stream must start");
    let items: Vec<_> = stream.collect().await;

    assert!(
        items.iter().all(|item| item.is_ok()),
        "no item may be an error"
    );
    let chunks: Vec<_> = items.into_iter().map(|c| c.unwrap()).collect();

    let text: String = chunks.iter().map(|c| c.content.as_str()).collect();
    assert_eq!(text, "Hello");

    let calls: Vec<_> = chunks
        .iter()
        .filter_map(|c| c.tool_calls.as_ref())
        .flatten()
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "lookup");
    assert_eq!(calls[0].arguments, serde_json::json!({"q": "x"}));

    let finals: Vec<_> = chunks.iter().filter(|c| c.is_final).collect();
    assert_eq!(finals.len(), 1, "exactly one final chunk");
    let last = chunks.last().unwrap();
    assert!(last.is_final);
    assert_eq!(last.finish_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn stream_request_uses_sse_url_and_key_header() {
    let (port, rx) = serve(
        "HTTP/1.1 200 OK",
        "text/event-stream",
        text_and_call_pieces(),
    )
    .await;

    let provider =
        GeminiProvider::with_base_url("sk-test-key-123", format!("http://127.0.0.1:{port}"))
            .unwrap();
    let request = CompletionRequest::new(vec![Message::user("hi")], "gemini-2.0-flash");
    let stream = provider
        .complete_stream(request)
        .await
        .expect("complete_stream must start");
    let _: Vec<_> = stream.collect().await;

    let raw = rx.await.expect("the server must capture the request");
    let first_line = raw.lines().next().unwrap_or_default();
    assert!(
        first_line.starts_with(
            "POST /v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse HTTP/1.1"
        ),
        "request line: {first_line}"
    );
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
async fn stream_error_status_does_not_leak_key() {
    let (port, _rx) = serve(
        "HTTP/1.1 400 Bad Request",
        "application/json",
        vec![r#"{"error":{"message":"bad key sk-test-key-123"}}"#],
    )
    .await;

    let provider =
        GeminiProvider::with_base_url("sk-test-key-123", format!("http://127.0.0.1:{port}"))
            .unwrap();
    let request = CompletionRequest::new(vec![Message::user("hi")], "gemini-2.0-flash");

    let err = match provider.complete_stream(request).await {
        Err(err) => err,
        Ok(_) => panic!("complete_stream must fail on a 400 response"),
    };
    assert!(
        matches!(err, AgentError::HttpError(_)),
        "unexpected error kind"
    );
    let text = err.to_string();
    assert!(!text.contains("sk-test-key-123"), "error: {text}");
    assert!(text.contains("400"), "error: {text}");
}
