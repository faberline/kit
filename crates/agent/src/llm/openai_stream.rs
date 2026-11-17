//! Parser for the OpenAI chat completion SSE stream.

use super::provider::StreamChunk;
use crate::error::{AgentError, AgentResult};
use crate::types::ToolCall;
use futures::{Stream, StreamExt};
use std::collections::BTreeMap;
use std::pin::Pin;

/// Turn the byte stream of an OpenAI streaming response into chunks.
pub(super) fn parse_openai_sse_stream(
    byte_stream: Pin<
        Box<dyn Stream<Item = Result<bytes::Bytes, kit_http::client::HttpError>> + Send>,
    >,
) -> impl Stream<Item = AgentResult<StreamChunk>> + Send {
    let mut parser = OpenAiSseParser::new();

    byte_stream.flat_map(move |result| {
        let chunks: Vec<AgentResult<StreamChunk>> = match result {
            Ok(bytes_data) => parser.push(&bytes_data),
            Err(e) => vec![Err(AgentError::StreamingError(e.to_string()))],
        };

        futures::stream::iter(chunks)
    })
}

/// Stateful parser for the OpenAI SSE stream.
///
/// Network chunks can split an event, or a UTF-8 character, at any byte.
/// The parser keeps the raw bytes until an event separator arrives.
struct OpenAiSseParser {
    /// Raw bytes that do not yet form a complete event.
    buf: Vec<u8>,
    /// Streamed tool calls by index: (id, name, partial JSON arguments).
    tools: BTreeMap<usize, (String, String, String)>,
    /// The first finish reason; it arrives before `[DONE]`.
    finish_reason: Option<String>,
    /// Set after `[DONE]`: later events are ignored.
    done: bool,
}

impl OpenAiSseParser {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            tools: BTreeMap::new(),
            finish_reason: None,
            done: false,
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
        if self.done {
            return Vec::new();
        }

        // OpenAI sends no `event:` line; comments start with `:`.
        let data: Vec<&str> = event
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim)
            .collect();
        if data.is_empty() {
            return Vec::new();
        }
        let data = data.join("\n");

        if data == "[DONE]" {
            return self.finish();
        }

        // Text that is not JSON is ignored, as the Claude parser does.
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
            return Vec::new();
        };

        if json.get("error").is_some() {
            let message = json["error"]["message"].as_str().unwrap_or("unknown error");
            return vec![Err(AgentError::StreamingError(format!(
                "OpenAI stream error: {message}"
            )))];
        }

        let mut out = Vec::new();
        let choice = &json["choices"][0];

        if let Some(text) = choice["delta"]["content"].as_str() {
            if !text.is_empty() {
                out.push(Ok(StreamChunk {
                    content: text.to_string(),
                    tool_calls: None,
                    finish_reason: None,
                    is_final: false,
                }));
            }
        }

        if let Some(items) = choice["delta"]["tool_calls"].as_array() {
            for item in items {
                let index = item["index"].as_u64().unwrap_or(0) as usize;
                let entry = self.tools.entry(index).or_default();
                if let Some(id) = item["id"].as_str() {
                    entry.0 = id.to_string();
                }
                if let Some(name) = item["function"]["name"].as_str() {
                    entry.1 = name.to_string();
                }
                if let Some(fragment) = item["function"]["arguments"].as_str() {
                    entry.2.push_str(fragment);
                }
            }
        }

        if let Some(reason) = choice["finish_reason"].as_str() {
            if self.finish_reason.is_none() {
                self.finish_reason = Some(reason.to_string());
            }
            out.extend(self.emit_tool_calls());
        }

        out
    }

    /// `[DONE]`: flush any tool call still open, then send the final chunk once.
    fn finish(&mut self) -> Vec<AgentResult<StreamChunk>> {
        self.done = true;
        let mut out = self.emit_tool_calls();
        out.push(Ok(StreamChunk {
            content: String::new(),
            tool_calls: None,
            finish_reason: Some(
                self.finish_reason
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
            ),
            is_final: true,
        }));
        out
    }

    /// Emit every assembled tool call in index order, and clear the buffer.
    fn emit_tool_calls(&mut self) -> Vec<AgentResult<StreamChunk>> {
        std::mem::take(&mut self.tools)
            .into_values()
            .map(|(id, name, partial)| {
                let parsed = if partial.is_empty() {
                    Ok(serde_json::json!({}))
                } else {
                    serde_json::from_str::<serde_json::Value>(&partial)
                };
                match parsed {
                    Ok(arguments) => Ok(StreamChunk {
                        content: String::new(),
                        tool_calls: Some(vec![ToolCall {
                            id,
                            name,
                            arguments,
                        }]),
                        finish_reason: None,
                        is_final: false,
                    }),
                    Err(e) => Err(AgentError::StreamingError(format!(
                        "invalid tool call arguments: {e}"
                    ))),
                }
            })
            .collect()
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

    fn ev(data: &str) -> Vec<u8> {
        format!("data: {data}\n\n").into_bytes()
    }

    fn text_event(text: &str) -> String {
        serde_json::json!({"choices": [{"index": 0, "delta": {"content": text}}]}).to_string()
    }

    fn tool_event(index: u64, id: Option<&str>, name: Option<&str>, args: &str) -> String {
        let mut function = serde_json::json!({ "arguments": args });
        if let Some(n) = name {
            function["name"] = serde_json::json!(n);
        }
        let mut call = serde_json::json!({ "index": index, "function": function });
        if let Some(i) = id {
            call["id"] = serde_json::json!(i);
        }
        serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}).to_string()
    }

    fn finish_event(reason: &str) -> String {
        serde_json::json!({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
            .to_string()
    }

    fn feed(parser: &mut OpenAiSseParser, events: &[Vec<u8>]) -> Vec<AgentResult<StreamChunk>> {
        events.iter().flat_map(|e| parser.push(e)).collect()
    }

    fn ok_chunks(items: Vec<AgentResult<StreamChunk>>) -> Vec<StreamChunk> {
        items.into_iter().map(|r| r.expect("chunk is Ok")).collect()
    }

    #[test]
    fn stream_text_deltas() {
        let mut p = OpenAiSseParser::new();
        let events = vec![
            ev(&text_event("Hel")),
            ev(&text_event("lo")),
            ev(&text_event(" there")),
        ];
        let chunks = ok_chunks(feed(&mut p, &events));
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(texts, vec!["Hel", "lo", " there"]);
        assert!(chunks.iter().all(|c| !c.is_final));
    }

    #[test]
    fn stream_tool_call_assembled() {
        let mut p = OpenAiSseParser::new();
        let events = vec![
            ev(&tool_event(0, Some("call_1"), Some("get_weather"), "")),
            ev(&tool_event(0, None, None, "{\"city\":")),
            ev(&tool_event(0, None, None, "\"Paris\"}")),
            ev(&finish_event("tool_calls")),
            ev("[DONE]"),
        ];
        let chunks = ok_chunks(feed(&mut p, &events));

        let calls: Vec<ToolCall> = chunks
            .iter()
            .filter_map(|c| c.tool_calls.clone())
            .flatten()
            .collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, serde_json::json!({"city": "Paris"}));

        let last = chunks.last().unwrap();
        assert!(last.is_final);
        assert_eq!(last.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn stream_two_tool_calls_by_index() {
        let mut p = OpenAiSseParser::new();
        let events = vec![
            ev(&tool_event(1, Some("call_b"), Some("second"), "{\"n\":2}")),
            ev(&tool_event(0, Some("call_a"), Some("first"), "{\"n\":")),
            ev(&tool_event(0, None, None, "1}")),
            ev(&finish_event("tool_calls")),
            ev("[DONE]"),
        ];
        let chunks = ok_chunks(feed(&mut p, &events));
        let calls: Vec<ToolCall> = chunks
            .iter()
            .filter_map(|c| c.tool_calls.clone())
            .flatten()
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "first");
        assert_eq!(calls[0].arguments, serde_json::json!({"n": 1}));
        assert_eq!(calls[1].name, "second");
        assert_eq!(calls[1].arguments, serde_json::json!({"n": 2}));
    }

    #[test]
    fn stream_stop_reason_kept() {
        let mut p = OpenAiSseParser::new();
        let events = vec![
            ev(&text_event("cut")),
            ev(&finish_event("length")),
            ev("[DONE]"),
        ];
        let chunks = ok_chunks(feed(&mut p, &events));
        let last = chunks.last().unwrap();
        assert!(last.is_final);
        assert_eq!(last.finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn done_without_finish_reason_is_unknown() {
        let mut p = OpenAiSseParser::new();
        let chunks = ok_chunks(feed(&mut p, &[ev("[DONE]")]));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_final);
        assert_eq!(chunks[0].finish_reason.as_deref(), Some("unknown"));
    }

    #[test]
    fn stream_error_event_is_err() {
        let mut p = OpenAiSseParser::new();
        let items = p.push(b"data: {\"error\":{\"message\":\"overloaded\"}}\n\n");
        assert_eq!(items.len(), 1);
        match &items[0] {
            Err(AgentError::StreamingError(m)) => assert!(m.contains("overloaded"), "{m}"),
            _ => panic!("expected a StreamingError"),
        }
    }

    #[test]
    fn stream_utf8_split_across_chunks() {
        let mut p = OpenAiSseParser::new();
        let event = ev(&text_event("café"));
        // Split inside the two-byte "é" (the last two bytes before the separator).
        let split = event.len() - 5;
        let mut items = p.push(&event[..split]);
        items.extend(p.push(&event[split..]));
        let chunks = ok_chunks(items);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "café");
    }

    #[test]
    fn stream_crlf_separator() {
        let mut p = OpenAiSseParser::new();
        let raw = format!(
            "data: {}\r\n\r\ndata: {}\r\n\r\n",
            text_event("a"),
            text_event("b")
        );
        let chunks = ok_chunks(p.push(raw.as_bytes()));
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(texts, vec!["a", "b"]);
    }

    #[test]
    fn stream_empty_choices_ignored() {
        let mut p = OpenAiSseParser::new();
        let items = p.push(b"data: {\"choices\":[]}\n\n");
        assert!(items.is_empty());
    }

    #[test]
    fn stream_invalid_tool_arguments_is_err() {
        let mut p = OpenAiSseParser::new();
        let events = vec![
            ev(&tool_event(0, Some("call_1"), Some("broken"), "{bad")),
            ev(&finish_event("tool_calls")),
        ];
        let items = feed(&mut p, &events);
        assert!(items
            .iter()
            .any(|r| matches!(r, Err(AgentError::StreamingError(_)))));
    }

    #[test]
    fn events_after_done_are_ignored() {
        let mut p = OpenAiSseParser::new();
        let events = vec![ev("[DONE]"), ev(&text_event("late"))];
        let chunks = ok_chunks(feed(&mut p, &events));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_final);
    }
}
