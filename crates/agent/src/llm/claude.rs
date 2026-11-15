use super::provider::{
    CompletionRequest, CompletionResponse, LLMProvider, StreamChunk, StreamResponse, ToolDefinition,
};
use crate::error::{AgentError, AgentResult};
use crate::types::{Message, Role, TokenUsage, ToolCall};
use async_trait::async_trait;
use futures::StreamExt;
use kit_http::client::{HttpClient, HttpClientConfig, HttpMethod, RequestBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tracing::debug;

/// Anthropic Claude LLM provider
pub struct ClaudeProvider {
    client: Arc<HttpClient>,
    api_key: String,
    default_model: String,
}

impl ClaudeProvider {
    pub fn new(api_key: impl Into<String>) -> AgentResult<Self> {
        Self::with_base_url(api_key, "https://api.anthropic.com")
    }

    /// Create with a custom base URL (e.g., internal gateway).
    pub fn with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> AgentResult<Self> {
        let base_url: String = base_url.into();
        let config = HttpClientConfig::new()
            .base_url(base_url)
            .timeout_secs(120.0);

        let client = HttpClient::new(config)?;

        Ok(Self {
            client: Arc::new(client),
            api_key: api_key.into(),
            default_model: "claude-sonnet-4-20250514".to_string(),
        })
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    fn convert_message(&self, msg: &Message) -> AgentResult<ClaudeMessage> {
        match msg.role {
            Role::System => Err(AgentError::InvalidRequest(
                "System messages should be passed as system parameter".to_string(),
            )),
            Role::User => Ok(ClaudeMessage {
                role: "user".to_string(),
                content: vec![ClaudeContent::Text {
                    text: msg.content.clone(),
                }],
            }),
            Role::Assistant => {
                let mut content = vec![];

                if !msg.content.is_empty() {
                    content.push(ClaudeContent::Text {
                        text: msg.content.clone(),
                    });
                }

                if let Some(tool_calls) = &msg.tool_calls {
                    for tool_call in tool_calls {
                        content.push(ClaudeContent::ToolUse {
                            id: tool_call.id.clone(),
                            name: tool_call.name.clone(),
                            input: tool_call.arguments.clone(),
                        });
                    }
                }

                Ok(ClaudeMessage {
                    role: "assistant".to_string(),
                    content,
                })
            }
            Role::Tool => Ok(ClaudeMessage {
                role: "user".to_string(),
                content: vec![ClaudeContent::ToolResult {
                    tool_use_id: msg
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string()),
                    content: msg.content.clone(),
                }],
            }),
        }
    }

    fn convert_tool(&self, tool: &ToolDefinition) -> ClaudeTool {
        ClaudeTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.parameters.clone(),
        }
    }

    fn build_request(&self, url: &str, body: serde_json::Value) -> RequestBuilder {
        RequestBuilder::new(HttpMethod::Post, url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json_value(body)
    }
}

#[async_trait]
impl LLMProvider for ClaudeProvider {
    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "claude-sonnet-4-20250514".to_string(),
            "claude-opus-4-20250514".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-sonnet-20240620".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-sonnet-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ]
    }

    async fn complete(&self, request: CompletionRequest) -> AgentResult<CompletionResponse> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        debug!("Claude completion request: model={}", model);

        let mut system_content = String::new();
        let mut claude_messages = Vec::new();

        for msg in &request.messages {
            if msg.role == Role::System {
                if !system_content.is_empty() {
                    system_content.push_str("\n\n");
                }
                system_content.push_str(&msg.content);
            } else {
                claude_messages.push(self.convert_message(msg)?);
            }
        }

        // Build tools list; optionally inject __structured_output for schema requests.
        let mut all_tools: Vec<ClaudeTool> = request
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(|t| self.convert_tool(t)).collect())
            .unwrap_or_default();

        // Detect if structured output is requested
        let is_structured = request.response_schema.is_some()
            || all_tools.iter().any(|t| t.name == "__structured_output");

        // If response_schema set and tool not yet injected, inject it now.
        if let Some(ref schema) = request.response_schema {
            if !all_tools.iter().any(|t| t.name == "__structured_output") {
                all_tools.push(ClaudeTool {
                    name: "__structured_output".to_string(),
                    description: "Extract structured data matching the required JSON schema. \
                        You MUST call this tool with all extracted data."
                        .to_string(),
                    input_schema: schema.clone(),
                });
            }
        }

        let mut body = serde_json::json!({
            "model": model,
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "messages": claude_messages,
        });

        if !system_content.is_empty() {
            body["system"] = serde_json::Value::String(system_content);
        }

        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        if !all_tools.is_empty() {
            body["tools"] = serde_json::to_value(&all_tools)?;
        }

        // Force the structured output tool when structured output is requested
        if is_structured {
            body["tool_choice"] = serde_json::json!({
                "type": "tool",
                "name": "__structured_output"
            });
        }

        let builder = self.build_request("/v1/messages", body);
        let response = self.client.execute_builder(builder).await?;

        if !response.is_success() {
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::ApiError(format!(
                "Claude API error ({}): {}",
                response.status_code, error_text
            )));
        }

        let response_text = response.text()?;
        let claude_response: ClaudeResponse = serde_json::from_str(&response_text)?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for content_block in claude_response.content {
            match content_block {
                ClaudeContent::Text { text } => {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(&text);
                }
                ClaudeContent::ToolUse { id, name, input } => {
                    if is_structured && name == "__structured_output" {
                        // Structured output: serialise the tool input as content
                        content = serde_json::to_string(&input).unwrap_or_default();
                    } else {
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments: input,
                        });
                    }
                }
                _ => {}
            }
        }

        let finish_reason = match claude_response.stop_reason.as_deref() {
            Some("end_turn") => "stop",
            Some("max_tokens") => "length",
            Some("tool_use") => "tool_calls",
            Some("stop_sequence") => "stop",
            _ => "unknown",
        }
        .to_string();

        Ok(CompletionResponse {
            content,
            finish_reason,
            model: claude_response.model,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            usage: TokenUsage {
                prompt_tokens: claude_response.usage.input_tokens as u32,
                completion_tokens: claude_response.usage.output_tokens as u32,
                total_tokens: (claude_response.usage.input_tokens
                    + claude_response.usage.output_tokens) as u32,
            },
            metadata: HashMap::new(),
        })
    }

    async fn complete_stream(&self, request: CompletionRequest) -> AgentResult<StreamResponse> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        debug!("Claude streaming request: model={}", model);

        let mut system_content = String::new();
        let mut claude_messages = Vec::new();

        for msg in &request.messages {
            if msg.role == Role::System {
                if !system_content.is_empty() {
                    system_content.push_str("\n\n");
                }
                system_content.push_str(&msg.content);
            } else {
                claude_messages.push(self.convert_message(msg)?);
            }
        }

        let tools: Option<Vec<ClaudeTool>> = request
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(|t| self.convert_tool(t)).collect());

        let mut body = serde_json::json!({
            "model": model,
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "messages": claude_messages,
            "stream": true,
        });

        if !system_content.is_empty() {
            body["system"] = serde_json::Value::String(system_content);
        }

        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        if let Some(tools) = tools {
            body["tools"] = serde_json::to_value(tools)?;
        }

        let builder = self.build_request("/v1/messages", body);
        let byte_stream = self.client.execute_builder_stream(builder).await?;

        let chunk_stream = parse_claude_sse_stream(byte_stream);

        Ok(Box::pin(chunk_stream))
    }
}

fn parse_claude_sse_stream(
    byte_stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<bytes::Bytes, kit_http::client::HttpError>> + Send>,
    >,
) -> impl futures::Stream<Item = AgentResult<StreamChunk>> + Send {
    let mut parser = SseParser::new();

    byte_stream.flat_map(move |result| {
        let chunks: Vec<AgentResult<StreamChunk>> = match result {
            Ok(bytes_data) => parser.push(&bytes_data),
            Err(e) => vec![Err(AgentError::StreamingError(e.to_string()))],
        };

        futures::stream::iter(chunks)
    })
}

/// Stateful parser for the Claude Messages SSE stream.
///
/// Bytes are kept until a complete event arrives, so a character split
/// across network chunks is decoded only once. Streamed tool arguments are
/// collected per content block index and emitted when the block stops.
struct SseParser {
    buf: Vec<u8>,
    /// Content block index -> (tool id, tool name, partial JSON text).
    tools: BTreeMap<usize, (String, String, String)>,
    /// First stop reason from `message_delta`; a `null` never overwrites it.
    stop_reason: Option<String>,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            tools: BTreeMap::new(),
            stop_reason: None,
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

    /// Earliest event separator in the buffer: (start, length).
    fn next_separator(&self) -> Option<(usize, usize)> {
        let crlf = find_bytes(&self.buf, b"\r\n\r\n").map(|p| (p, 4));
        let lf = find_bytes(&self.buf, b"\n\n").map(|p| (p, 2));
        match (crlf, lf) {
            (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
            (a, b) => a.or(b),
        }
    }

    fn handle_event(&mut self, event: &str) -> Vec<AgentResult<StreamChunk>> {
        let mut event_name = None;
        let mut data = None;

        for line in event.lines() {
            if let Some(value) = line.strip_prefix("event:") {
                event_name = Some(value.trim());
            } else if let Some(value) = line.strip_prefix("data:") {
                data = Some(value.trim());
            }
        }

        let (Some(event_name), Some(data)) = (event_name, data) else {
            return Vec::new();
        };
        let json: serde_json::Value = serde_json::from_str(data).unwrap_or(serde_json::Value::Null);

        match event_name {
            "content_block_start" => {
                let block = &json["content_block"];
                if block["type"].as_str() == Some("tool_use") {
                    self.tools.insert(
                        index_of(&json),
                        (
                            block["id"].as_str().unwrap_or_default().to_string(),
                            block["name"].as_str().unwrap_or_default().to_string(),
                            String::new(),
                        ),
                    );
                }
                Vec::new()
            }
            "content_block_delta" => {
                let delta = &json["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => vec![Ok(StreamChunk {
                        content: delta["text"].as_str().unwrap_or_default().to_string(),
                        tool_calls: None,
                        finish_reason: None,
                        is_final: false,
                    })],
                    Some("input_json_delta") => {
                        if let Some(entry) = self.tools.get_mut(&index_of(&json)) {
                            entry
                                .2
                                .push_str(delta["partial_json"].as_str().unwrap_or_default());
                        }
                        Vec::new()
                    }
                    _ => Vec::new(),
                }
            }
            "content_block_stop" => match self.tools.remove(&index_of(&json)) {
                None => Vec::new(),
                Some((id, name, partial)) => {
                    let arguments = if partial.is_empty() {
                        Ok(serde_json::json!({}))
                    } else {
                        serde_json::from_str(&partial).map_err(|e| {
                            AgentError::StreamingError(format!("invalid tool call arguments: {e}"))
                        })
                    };
                    match arguments {
                        Ok(arguments) => vec![Ok(StreamChunk {
                            content: String::new(),
                            tool_calls: Some(vec![ToolCall {
                                id,
                                name,
                                arguments,
                            }]),
                            finish_reason: None,
                            is_final: false,
                        })],
                        Err(e) => vec![Err(e)],
                    }
                }
            },
            "message_delta" => {
                if self.stop_reason.is_none() {
                    if let Some(reason) = json["delta"]["stop_reason"].as_str() {
                        self.stop_reason = Some(reason.to_string());
                    }
                }
                Vec::new()
            }
            "message_stop" => {
                let finish_reason = match self.stop_reason.as_deref() {
                    Some("end_turn") => "stop",
                    Some("max_tokens") => "length",
                    Some("tool_use") => "tool_calls",
                    Some("stop_sequence") => "stop",
                    _ => "unknown",
                };
                vec![Ok(StreamChunk {
                    content: String::new(),
                    tool_calls: None,
                    finish_reason: Some(finish_reason.to_string()),
                    is_final: true,
                })]
            }
            "error" => {
                let message = json["error"]["message"].as_str().unwrap_or("unknown error");
                vec![Err(AgentError::StreamingError(format!(
                    "Claude stream error: {message}"
                )))]
            }
            _ => Vec::new(),
        }
    }
}

fn index_of(json: &serde_json::Value) -> usize {
    json["index"].as_u64().unwrap_or(0) as usize
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[derive(Debug, Serialize, Deserialize)]
struct ClaudeMessage {
    role: String,
    content: Vec<ClaudeContent>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClaudeContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct ClaudeTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    #[allow(dead_code)]
    id: String,
    model: String,
    #[allow(dead_code)]
    #[serde(rename = "type")]
    response_type: String,
    #[allow(dead_code)]
    role: String,
    content: Vec<ClaudeContent>,
    stop_reason: Option<String>,
    #[allow(dead_code)]
    stop_sequence: Option<String>,
    usage: ClaudeUsage,
}

#[derive(Debug, Deserialize)]
struct ClaudeUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider = ClaudeProvider::new("test-key").unwrap();
        assert_eq!(provider.provider_name(), "anthropic");
    }

    #[test]
    fn test_supported_models() {
        let provider = ClaudeProvider::new("test-key").unwrap();
        let models = provider.supported_models();
        assert!(models.contains(&"claude-sonnet-4-20250514".to_string()));
    }

    fn ev(name: &str, data: &str) -> Vec<u8> {
        format!("event: {name}\ndata: {data}\n\n").into_bytes()
    }

    #[test]
    fn stream_text_delta_unchanged() {
        let mut parser = SseParser::new();
        let out = parser.push(&ev(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        ));
        assert_eq!(out.len(), 1);
        let chunk = out.into_iter().next().unwrap().unwrap();
        assert_eq!(chunk.content, "Hello");
        assert!(!chunk.is_final);
    }

    #[test]
    fn stream_tool_call_assembled() {
        let mut bytes = Vec::new();
        bytes.extend(ev(
            "content_block_start",
            r#"{"index":1,"content_block":{"type":"tool_use","id":"t1","name":"get_weather","input":{}}}"#,
        ));
        bytes.extend(ev(
            "content_block_delta",
            r#"{"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
        ));
        bytes.extend(ev(
            "content_block_delta",
            r#"{"index":1,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}"#,
        ));
        bytes.extend(ev("content_block_stop", r#"{"index":1}"#));

        let mut parser = SseParser::new();
        let out = parser.push(&bytes);
        assert_eq!(out.len(), 1);
        let chunk = out.into_iter().next().unwrap().unwrap();
        let calls = chunk.tool_calls.expect("tool_calls must be Some");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "t1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, serde_json::json!({"city": "Paris"}));
        assert!(!chunk.is_final);
    }

    #[test]
    fn stream_stop_reason_kept() {
        let mut bytes = Vec::new();
        bytes.extend(ev(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        ));
        bytes.extend(ev(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":null}}"#,
        ));
        bytes.extend(ev("message_stop", r#"{"type":"message_stop"}"#));

        let mut parser = SseParser::new();
        let out = parser.push(&bytes);
        let chunks: Vec<StreamChunk> = out.into_iter().map(|c| c.unwrap()).collect();

        assert_eq!(chunks.iter().filter(|c| c.is_final).count(), 1);
        let last = chunks.last().unwrap();
        assert!(last.is_final);
        assert_eq!(last.finish_reason, Some("tool_calls".to_string()));
    }

    #[test]
    fn stream_error_event_is_err() {
        let mut parser = SseParser::new();
        let out = parser.push(&ev(
            "error",
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        ));
        assert_eq!(out.len(), 1);
        let err = out.into_iter().next().unwrap().unwrap_err();
        assert!(err.to_string().contains("Overloaded"));
    }

    #[test]
    fn stream_utf8_split_across_chunks() {
        let bytes = ev(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"text_delta","text":"héllo"}}"#,
        );
        // Cut right after the first byte of the two-byte character `é`.
        let cut = bytes.iter().position(|&b| b == 0xC3).unwrap() + 1;

        let mut parser = SseParser::new();
        let first = parser.push(&bytes[..cut]);
        assert!(first.is_empty());

        let second = parser.push(&bytes[cut..]);
        assert_eq!(second.len(), 1);
        let chunk = second.into_iter().next().unwrap().unwrap();
        assert_eq!(chunk.content, "héllo");
    }

    #[test]
    fn stream_crlf_separator() {
        let mut parser = SseParser::new();
        let out = parser.push(
            b"event: content_block_delta\r\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\r\n\r\n",
        );
        assert_eq!(out.len(), 1);
        let chunk = out.into_iter().next().unwrap().unwrap();
        assert_eq!(chunk.content, "Hi");
    }
}
