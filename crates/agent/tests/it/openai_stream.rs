//! OpenAI streaming against a loopback server: the request shape, the assembled text
//! and tool call, the stop reason, key redaction and the idle timeout.
//! Uses a loopback server; no network, no paid API.

use agent::llm::{CompletionRequest, LLMProvider, OpenAIProvider, ToolDefinition};
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
/// with `status_line` and a close-delimited body written in `pieces`, one write each.
/// With `hold`, keep the connection open silently for that long after the pieces.
async fn serve(
    status_line: &'static str,
    pieces: Vec<&'static str>,
    hold: Option<Duration>,
) -> (u16, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_request(&mut stream).await;
        let _ = tx.send(request);

        let head = format!(
            "{status_line}\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(head.as_bytes()).await;
        for piece in pieces {
            let _ = stream.write_all(piece.as_bytes()).await;
            let _ = stream.flush().await;
        }
        if let Some(d) = hold {
            tokio::time::sleep(d).await;
        }
        let _ = stream.shutdown().await;
    });

    (port, rx)
}

/// Two text deltas, one tool call whose arguments arrive in two fragments, a stop
/// reason and `[DONE]`. The second text event is split across two writes.
fn text_and_tool_pieces() -> Vec<&'static str> {
    vec![
        r#"data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"content":"lo"},"#,
        r#""finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "\n\n",
        "data: [DONE]\n\n",
    ]
}

fn weather_request() -> CompletionRequest {
    let tool = ToolDefinition {
        name: "get_weather".to_string(),
        description: "Weather for a city".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } }
        }),
    };
    CompletionRequest::new(vec![Message::user("hi")], "gpt-4o").with_tools(vec![tool])
}

#[tokio::test]
async fn streams_text_tool_call_and_stop_reason() {
    let (port, _rx) = serve("HTTP/1.1 200 OK", text_and_tool_pieces(), None).await;
    let provider =
        OpenAIProvider::with_base_url("test-key", format!("http://127.0.0.1:{port}")).unwrap();

    let mut stream = provider.complete_stream(weather_request()).await.unwrap();

    let mut text = String::new();
    let mut calls = Vec::new();
    let mut final_reasons = Vec::new();
    while let Some(item) = stream.next().await {
        let chunk = item.unwrap();
        text.push_str(&chunk.content);
        if let Some(c) = chunk.tool_calls {
            calls.extend(c);
        }
        if chunk.is_final {
            final_reasons.push(chunk.finish_reason.clone());
        }
    }

    assert_eq!(text, "Hello");
    assert_eq!(calls.len(), 1, "one assembled tool call");
    assert_eq!(calls[0].id, "call_1");
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments, serde_json::json!({"city": "Paris"}));
    assert_eq!(final_reasons, vec![Some("tool_calls".to_string())]);
}

#[tokio::test]
async fn request_sets_stream_true_and_bearer_header() {
    let (port, rx) = serve("HTTP/1.1 200 OK", text_and_tool_pieces(), None).await;
    let provider =
        OpenAIProvider::with_base_url("test-key", format!("http://127.0.0.1:{port}")).unwrap();

    let mut stream = provider.complete_stream(weather_request()).await.unwrap();
    while stream.next().await.is_some() {}

    let raw = rx.await.unwrap();
    assert!(raw.contains(r#""stream":true"#), "request: {raw}");
    assert!(
        raw.to_lowercase()
            .contains("authorization: bearer test-key"),
        "request: {raw}"
    );
}

#[tokio::test]
async fn error_status_does_not_leak_key() {
    let (port, _rx) = serve(
        "HTTP/1.1 401 Unauthorized",
        vec![r#"{"error":{"message":"bad key test-key"}}"#],
        None,
    )
    .await;
    let provider =
        OpenAIProvider::with_base_url("test-key", format!("http://127.0.0.1:{port}")).unwrap();

    let err = match provider.complete_stream(weather_request()).await {
        Ok(_) => panic!("expected an error for status 401"),
        Err(e) => e,
    };
    let text = err.to_string();
    assert!(!text.contains("test-key"), "key leaked: {text}");
}

#[tokio::test]
async fn idle_timeout_ends_stream_with_error() {
    let (port, _rx) = serve(
        "HTTP/1.1 200 OK",
        vec![
            r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#,
            "\n\n",
        ],
        Some(Duration::from_secs(3)),
    )
    .await;
    let provider = OpenAIProvider::with_base_url("test-key", format!("http://127.0.0.1:{port}"))
        .unwrap()
        .with_stream_idle_timeout(Duration::from_millis(200));

    let mut stream = provider.complete_stream(weather_request()).await.unwrap();

    let first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("first chunk in time")
        .expect("a first item")
        .expect("first chunk is Ok");
    assert_eq!(first.content, "Hi");

    let second = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("idle error in time")
        .expect("a second item");
    match second {
        Err(AgentError::StreamingError(m)) => assert!(m.contains("idle"), "message: {m}"),
        _ => panic!("expected an idle StreamingError"),
    }

    let after = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("stream ends in time");
    assert!(after.is_none(), "stream must end after the idle error");
}
