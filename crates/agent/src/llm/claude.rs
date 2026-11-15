use super::provider::{
    CompletionRequest, CompletionResponse, LLMProvider, StreamChunk, StreamResponse, ToolDefinition,
};
use crate::error::{AgentError, AgentResult};
use crate::types::{Message, Role, TokenUsage, ToolCall};
use async_trait::async_trait;
use futures::StreamExt;
use kit_http::client::{HttpClient, HttpClientConfig, HttpMethod, RequestBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// Anthropic Claude LLM provider
pub struct ClaudeProvider {
    client: Arc<HttpClient>,
    /// Client for streams: a long ceiling, because the idle timeout is the real limit.
    stream_client: Arc<HttpClient>,
    api_key: String,
    default_model: String,
    stream_idle_timeout: std::time::Duration,
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
            .base_url(base_url.clone())
            .timeout_secs(120.0);
        let stream_config = HttpClientConfig::new()
            .base_url(base_url)
            .timeout_secs(3600.0);

        let client = HttpClient::new(config)?;
        let stream_client = HttpClient::new(stream_config)?;

        Ok(Self {
            client: Arc::new(client),
            stream_client: Arc::new(stream_client),
            api_key: api_key.into(),
            default_model: "claude-sonnet-4-20250514".to_string(),
            stream_idle_timeout: std::time::Duration::from_secs(60),
        })
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Set the longest silence allowed between two chunks of a stream.
    pub fn with_stream_idle_timeout(mut self, d: std::time::Duration) -> Self {
        self.stream_idle_timeout = d;
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

    fn build_completion_body(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> AgentResult<(serde_json::Value, bool)> {
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
            // Force the schema tool only when it is the only tool; otherwise let
            // the model call the agent's tools too.
            let only_structured =
                all_tools.len() == 1 && all_tools[0].name == "__structured_output";
            body["tool_choice"] = if only_structured {
                serde_json::json!({"type": "tool", "name": "__structured_output"})
            } else {
                serde_json::json!({"type": "auto"})
            };
        }

        Ok((body, is_structured))
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

        let (body, is_structured) = self.build_completion_body(&request, &model)?;

        let builder = self.build_request("/v1/messages", body);
        let response = self.client.execute_builder(builder).await?;

        if !response.is_success() {
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            let error_text = redact(&error_text, &self.api_key);
            return Err(AgentError::ApiError(format!(
                "Claude API error ({}): {}",
                response.status_code, error_text
            )));
        }

        let response_text = response.text()?;
        let claude_response: ClaudeResponse = serde_json::from_str(&response_text)?;

        Ok(map_response(claude_response, is_structured))
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
        let byte_stream = self.stream_client.execute_builder_stream(builder).await?;

        let chunk_stream = Box::pin(parse_claude_sse_stream(byte_stream));
        let chunk_stream = with_idle_timeout(chunk_stream, self.stream_idle_timeout);

        Ok(Box::pin(chunk_stream))
    }
}

fn map_response(claude_response: ClaudeResponse, is_structured: bool) -> CompletionResponse {
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

    CompletionResponse {
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
            total_tokens: (claude_response.usage.input_tokens + claude_response.usage.output_tokens)
                as u32,
        },
        metadata: HashMap::new(),
    }
}

/// Replace every occurrence of `secret` in `text` (an empty secret changes nothing).
fn redact(text: &str, secret: &str) -> String {
    if secret.is_empty() {
        return text.to_string();
    }
    text.replace(secret, "[redacted]")
}

/// Wrap a stream so that silence longer than `idle` ends it with one error.
fn with_idle_timeout<S>(
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

fn parse_claude_sse_stream(
    byte_stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<bytes::Bytes, kit_http::client::HttpError>> + Send>,
    >,
) -> impl futures::Stream<Item = AgentResult<StreamChunk>> + Send {
    let mut buffer = String::new();

    byte_stream.flat_map(move |result| {
        let chunks: Vec<AgentResult<StreamChunk>> = match result {
            Ok(bytes_data) => {
                buffer.push_str(&String::from_utf8_lossy(&bytes_data));

                let mut parsed_chunks = Vec::new();

                while let Some(event_end) = buffer.find("\n\n") {
                    let event = buffer[..event_end].to_string();
                    buffer = buffer[event_end + 2..].to_string();

                    if let Some(chunk) = parse_sse_event(&event) {
                        parsed_chunks.push(Ok(chunk));
                    }
                }

                parsed_chunks
            }
            Err(e) => vec![Err(AgentError::StreamingError(e.to_string()))],
        };

        futures::stream::iter(chunks)
    })
}

fn parse_sse_event(event: &str) -> Option<StreamChunk> {
    let mut event_type = None;
    let mut data = None;

    for line in event.lines() {
        if let Some(stripped) = line.strip_prefix("event: ") {
            event_type = Some(stripped.trim());
        } else if let Some(stripped) = line.strip_prefix("data: ") {
            data = Some(stripped.trim());
        }
    }

    let event_type = event_type?;
    let data = data?;

    match event_type {
        "content_block_delta" => {
            if let Ok(delta) = serde_json::from_str::<ClaudeStreamDelta>(data) {
                if let Some(text) = delta.delta.and_then(|d| d.text) {
                    return Some(StreamChunk {
                        content: text,
                        tool_calls: None,
                        finish_reason: None,
                        is_final: false,
                    });
                }
            }
        }
        "message_delta" => {
            if let Ok(delta) = serde_json::from_str::<ClaudeMessageDelta>(data) {
                return Some(StreamChunk {
                    content: String::new(),
                    tool_calls: None,
                    finish_reason: delta.delta.and_then(|d| d.stop_reason),
                    is_final: true,
                });
            }
        }
        "message_stop" => {
            return Some(StreamChunk {
                content: String::new(),
                tool_calls: None,
                finish_reason: Some("stop".to_string()),
                is_final: true,
            });
        }
        _ => {}
    }

    None
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
    /// Any other block type (for example `thinking`). Built only when reading.
    #[serde(other)]
    #[allow(dead_code)]
    Unknown,
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

#[derive(Debug, Deserialize)]
struct ClaudeStreamDelta {
    delta: Option<ClaudeTextDelta>,
}

#[derive(Debug, Deserialize)]
struct ClaudeTextDelta {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ClaudeMessageDelta {
    delta: Option<ClaudeStopDelta>,
}

#[derive(Debug, Deserialize)]
struct ClaudeStopDelta {
    stop_reason: Option<String>,
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

    #[test]
    fn test_parse_sse_content_delta() {
        let event = r#"event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;

        let chunk = parse_sse_event(event);
        assert!(chunk.is_some());
        let chunk = chunk.unwrap();
        assert_eq!(chunk.content, "Hello");
        assert!(!chunk.is_final);
    }

    const TEST_MODEL: &str = "claude-sonnet-4-20250514";

    fn request(with_tool: bool, with_schema: bool) -> CompletionRequest {
        let mut req = CompletionRequest::new(vec![Message::user("hi")], TEST_MODEL);
        if with_tool {
            req = req.with_tools(vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: "Weather".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }]);
        }
        if with_schema {
            req = req.with_response_schema(serde_json::json!({"type": "object"}));
        }
        req
    }

    #[test]
    fn tool_choice_forced_when_only_schema() {
        let provider = ClaudeProvider::new("test-key").unwrap();
        let (body, is_structured) = provider
            .build_completion_body(&request(false, true), TEST_MODEL)
            .unwrap();
        assert!(is_structured);
        assert_eq!(
            body["tool_choice"],
            serde_json::json!({"type": "tool", "name": "__structured_output"})
        );
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn tool_choice_auto_with_tools_and_schema() {
        let provider = ClaudeProvider::new("test-key").unwrap();
        let (body, is_structured) = provider
            .build_completion_body(&request(true, true), TEST_MODEL)
            .unwrap();
        assert!(is_structured);
        assert_eq!(body["tool_choice"], serde_json::json!({"type": "auto"}));
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert!(tools.iter().any(|t| t["name"] == "__structured_output"));
    }

    #[test]
    fn no_tool_choice_without_schema() {
        let provider = ClaudeProvider::new("test-key").unwrap();
        let (body, is_structured) = provider
            .build_completion_body(&request(true, false), TEST_MODEL)
            .unwrap();
        assert!(!is_structured);
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn unknown_content_block_is_skipped() {
        let raw = r#"{"id":"m","type":"message","role":"assistant","model":"x","content":[{"type":"thinking","thinking":"..."},{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}"#;
        let resp: ClaudeResponse = serde_json::from_str(raw).unwrap();
        let out = map_response(resp, false);
        assert_eq!(out.content, "hi");
        assert_eq!(out.finish_reason, "stop");
        assert_eq!(out.usage.total_tokens, 3);
    }

    fn text_chunk(content: &str) -> StreamChunk {
        StreamChunk {
            content: content.to_string(),
            tool_calls: None,
            finish_reason: None,
            is_final: false,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn idle_timeout_ends_stream_with_error() {
        use futures::StreamExt;

        let inner = Box::pin(futures::stream::unfold(0u8, |state| async move {
            if state == 0 {
                Some((Ok(text_chunk("a")), 1u8))
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

    #[tokio::test(start_paused = true)]
    async fn fast_stream_passes_through() {
        use futures::StreamExt;

        let inner = futures::stream::iter(vec![Ok(text_chunk("a")), Ok(text_chunk("b"))]);
        let stream = with_idle_timeout(inner, std::time::Duration::from_secs(5));

        let items: Vec<AgentResult<StreamChunk>> = stream.collect().await;
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_ref().unwrap().content, "a");
        assert_eq!(items[1].as_ref().unwrap().content, "b");
    }

    #[test]
    fn stream_idle_timeout_default_and_builder() {
        let provider = ClaudeProvider::new("test-key").unwrap();
        assert_eq!(
            provider.stream_idle_timeout,
            std::time::Duration::from_secs(60)
        );
        let provider = provider.with_stream_idle_timeout(std::time::Duration::from_secs(7));
        assert_eq!(
            provider.stream_idle_timeout,
            std::time::Duration::from_secs(7)
        );
    }

    #[test]
    fn redact_removes_key() {
        assert_eq!(
            redact("bad key sk-test-123 here", "sk-test-123"),
            "bad key [redacted] here"
        );
        assert_eq!(redact("abc", ""), "abc");
    }
}
