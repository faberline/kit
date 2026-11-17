//! Streaming for the Gemini provider: the SSE parser and the idle timeout.

use super::gemini::{redact_key, GeminiPart, GeminiResponse, BLOCKED_FINISH_REASONS};
use super::provider::StreamChunk;
use crate::error::{AgentError, AgentResult};
use crate::types::ToolCall;
use futures::StreamExt;

/// Wrap a stream so that silence longer than `idle` ends it with one error.
pub(super) fn with_idle_timeout<S>(
    stream: S,
    idle: std::time::Duration,
) -> impl futures::Stream<Item = AgentResult<StreamChunk>> + Send
where
    S: futures::Stream<Item = AgentResult<StreamChunk>> + Send + Unpin + 'static,
{
    futures::stream::unfold((stream, false), move |(mut stream, done)| async move {
        if done {
            return None;
        }
        match tokio::time::timeout(idle, stream.next()).await {
            Ok(Some(item)) => Some((item, (stream, false))),
            Ok(None) => None,
            Err(_) => Some((
                Err(AgentError::StreamingError(format!(
                    "stream idle for {} s",
                    idle.as_secs()
                ))),
                (stream, true),
            )),
        }
    })
}

pub(super) fn parse_gemini_sse_stream(
    byte_stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<bytes::Bytes, kit_http::client::HttpError>> + Send>,
    >,
    api_key: String,
) -> impl futures::Stream<Item = AgentResult<StreamChunk>> + Send {
    let mut parser = GeminiSseParser::new();

    byte_stream.flat_map(move |result| {
        let chunks: Vec<AgentResult<StreamChunk>> = match result {
            Ok(bytes_data) => parser.push(&bytes_data),
            Err(e) => vec![Err(AgentError::StreamingError(redact_key(
                &e.to_string(),
                &api_key,
            )))],
        };

        futures::stream::iter(chunks)
    })
}

/// Stateful parser for the Gemini SSE stream.
///
/// Network chunks can split an event, or a UTF-8 character, at any byte.
/// The parser keeps the raw bytes until an event separator arrives.
struct GeminiSseParser {
    /// Raw bytes that do not yet form a complete event.
    buf: Vec<u8>,
    /// Number of function calls so far, for unique ids over the whole stream.
    call_count: usize,
    /// Set after the final chunk or a blocked candidate.
    finished: bool,
}

impl GeminiSseParser {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            call_count: 0,
            finished: false,
        }
    }

    /// Append `bytes` and return every chunk or error from the events that are now complete.
    fn push(&mut self, bytes: &[u8]) -> Vec<AgentResult<StreamChunk>> {
        self.buf.extend_from_slice(bytes);

        let mut out = Vec::new();
        while let Some((pos, sep_len)) = self.next_separator() {
            let raw: Vec<u8> = self.buf.drain(..pos + sep_len).collect();
            match std::str::from_utf8(&raw[..pos]) {
                Ok(event) => out.extend(self.handle_event(event)),
                Err(_) => out.push(Err(AgentError::StreamingError(
                    "invalid UTF-8 in stream".to_string(),
                ))),
            }
        }
        out
    }

    /// The earliest event separator in the buffer: its start and its length.
    fn next_separator(&self) -> Option<(usize, usize)> {
        let crlf = find_bytes(&self.buf, b"\r\n\r\n").map(|pos| (pos, 4));
        let lf = find_bytes(&self.buf, b"\n\n").map(|pos| (pos, 2));
        match (crlf, lf) {
            (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
            (a, b) => a.or(b),
        }
    }

    fn handle_event(&mut self, event: &str) -> Vec<AgentResult<StreamChunk>> {
        if self.finished {
            return Vec::new();
        }

        // A Gemini event has only `data:` lines. Join them into one payload.
        let mut payload: Option<String> = None;
        for line in event.lines() {
            if let Some(value) = line.strip_prefix("data:") {
                let value = value
                    .strip_prefix(' ')
                    .unwrap_or(value)
                    .trim_end_matches('\r');
                match payload.as_mut() {
                    Some(text) => {
                        text.push('\n');
                        text.push_str(value);
                    }
                    None => payload = Some(value.to_string()),
                }
            }
        }

        let Some(payload) = payload.filter(|text| !text.trim().is_empty()) else {
            return Vec::new();
        };

        let response: GeminiResponse = match serde_json::from_str(&payload) {
            Ok(response) => response,
            Err(e) => {
                return vec![Err(AgentError::StreamingError(format!(
                    "invalid Gemini stream event: {e}"
                )))]
            }
        };

        let Some(candidate) = response.candidates.into_iter().next() else {
            return match response.prompt_feedback.and_then(|f| f.block_reason) {
                Some(reason) => {
                    self.finished = true;
                    vec![Err(AgentError::ApiError(format!(
                        "Gemini blocked the prompt: blockReason={reason}"
                    )))]
                }
                None => Vec::new(),
            };
        };

        if let Some(reason) = candidate.finish_reason.as_deref() {
            if BLOCKED_FINISH_REASONS.contains(&reason) {
                self.finished = true;
                return vec![Err(AgentError::ApiError(format!(
                    "Gemini blocked the candidate: finishReason={reason}"
                )))];
            }
        }

        let mut out = Vec::new();
        let parts = candidate
            .content
            .map(|content| content.parts)
            .unwrap_or_default();

        for part in parts {
            match part {
                GeminiPart::Text { text } => {
                    if !text.is_empty() {
                        out.push(Ok(StreamChunk {
                            content: text,
                            tool_calls: None,
                            finish_reason: None,
                            is_final: false,
                        }));
                    }
                }
                GeminiPart::FunctionCall { function_call } => {
                    // Gemini sends a whole function call in one part.
                    let id = format!("call_{}", self.call_count);
                    self.call_count += 1;
                    out.push(Ok(StreamChunk {
                        content: String::new(),
                        tool_calls: Some(vec![ToolCall {
                            id,
                            name: function_call.name,
                            arguments: function_call.args,
                        }]),
                        finish_reason: None,
                        is_final: false,
                    }));
                }
                _ => {}
            }
        }

        if let Some(reason) = candidate.finish_reason.as_deref() {
            let mapped = match reason {
                "STOP" => "stop",
                "MAX_TOKENS" => "length",
                _ => "unknown",
            };
            out.push(Ok(StreamChunk {
                content: String::new(),
                tool_calls: None,
                finish_reason: Some(mapped.to_string()),
                is_final: true,
            }));
            self.finished = true;
        }

        out
    }
}

/// Position of the first occurrence of `needle` in `haystack`.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(json: &str) -> String {
        format!("data: {json}\r\n\r\n")
    }

    #[test]
    fn gemini_sse_text_chunks_then_final() {
        let mut parser = GeminiSseParser::new();
        let mut out = parser.push(
            sse(r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]}}]}"#)
                .as_bytes(),
        );
        out.extend(parser.push(
            sse(r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"finishReason":"STOP"}]}"#)
                .as_bytes(),
        ));
        let chunks: Vec<StreamChunk> = out.into_iter().map(|c| c.unwrap()).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content, "Hel");
        assert!(!chunks[0].is_final);
        assert_eq!(chunks[1].content, "lo");
        assert!(!chunks[1].is_final);
        assert_eq!(chunks[2].content, "");
        assert!(chunks[2].is_final);
        assert_eq!(chunks[2].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn gemini_sse_function_call_is_one_chunk() {
        let mut parser = GeminiSseParser::new();
        let out = parser.push(
            sse(r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}]},"finishReason":"STOP"}]}"#)
                .as_bytes(),
        );
        let chunks: Vec<StreamChunk> = out.into_iter().map(|c| c.unwrap()).collect();
        assert_eq!(chunks.len(), 2);
        let calls = chunks[0].tool_calls.as_ref().expect("a tool call chunk");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, serde_json::json!({"city": "Paris"}));
        assert!(chunks[1].is_final);
    }

    #[test]
    fn gemini_sse_utf8_split_across_pushes() {
        let mut parser = GeminiSseParser::new();
        let event = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"é\"}]}}]}\n\n";
        let bytes = event.as_bytes();
        let split = bytes.iter().position(|b| *b == 0xC3).unwrap() + 1;
        assert!(parser.push(&bytes[..split]).is_empty());
        let out = parser.push(&bytes[split..]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap().content, "é");
    }

    #[test]
    fn gemini_sse_crlf_and_lf_separators() {
        let mut parser = GeminiSseParser::new();
        let input = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\"}]}}]}\r\n\r\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"b\"}]}}]}\n\n",
        );
        let out = parser.push(input.as_bytes());
        let texts: Vec<String> = out.into_iter().map(|c| c.unwrap().content).collect();
        assert_eq!(texts, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn gemini_sse_blocked_candidate_is_err() {
        let mut parser = GeminiSseParser::new();
        let out = parser.push(sse(r#"{"candidates":[{"finishReason":"SAFETY"}]}"#).as_bytes());
        assert_eq!(out.len(), 1);
        match &out[0] {
            Err(AgentError::ApiError(m)) => assert!(m.contains("finishReason=SAFETY"), "{m}"),
            _ => panic!("expected ApiError"),
        }
    }

    #[test]
    fn gemini_sse_blocked_prompt_is_err() {
        let mut parser = GeminiSseParser::new();
        let out = parser
            .push(sse(r#"{"candidates":[],"promptFeedback":{"blockReason":"OTHER"}}"#).as_bytes());
        assert_eq!(out.len(), 1);
        match &out[0] {
            Err(AgentError::ApiError(m)) => assert!(m.contains("blockReason=OTHER"), "{m}"),
            _ => panic!("expected ApiError"),
        }
    }

    #[test]
    fn gemini_sse_usage_only_event_gives_nothing() {
        let mut parser = GeminiSseParser::new();
        let out = parser
            .push(sse(r#"{"candidates":[],"usageMetadata":{"promptTokenCount":1}}"#).as_bytes());
        assert!(out.is_empty());
    }

    #[test]
    fn gemini_sse_invalid_json_is_streaming_error() {
        let mut parser = GeminiSseParser::new();
        let out = parser.push(b"data: {not json\n\n");
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], Err(AgentError::StreamingError(_))));
    }

    #[tokio::test(start_paused = true)]
    async fn gemini_idle_timeout_ends_stream_with_error() {
        let inner = Box::pin(futures::stream::unfold(0u8, |state| async move {
            if state == 0 {
                Some((
                    Ok(StreamChunk {
                        content: "a".to_string(),
                        tool_calls: None,
                        finish_reason: None,
                        is_final: false,
                    }),
                    1u8,
                ))
            } else {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
                None
            }
        }));
        let mut stream = Box::pin(with_idle_timeout(inner, std::time::Duration::from_secs(5)));

        assert!(matches!(stream.next().await, Some(Ok(_))));
        match stream.next().await {
            Some(Err(e)) => assert!(e.to_string().contains("idle")),
            _ => panic!("expected an idle error"),
        }
        assert!(stream.next().await.is_none());
    }
}
