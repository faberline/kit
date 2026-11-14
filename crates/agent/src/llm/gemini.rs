use super::provider::{
    CompletionRequest, CompletionResponse, LLMProvider, StreamResponse, ToolDefinition,
};
use crate::error::{AgentError, AgentResult};
use crate::types::{Message, Role, TokenUsage, ToolCall};
use async_trait::async_trait;
use kit_http::client::{HttpClient, HttpClientConfig, HttpMethod, RequestBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// Google Gemini LLM provider
pub struct GeminiProvider {
    client: Arc<HttpClient>,
    api_key: String,
    default_model: String,
    _base_url: String,
}

impl GeminiProvider {
    pub fn new(api_key: impl Into<String>) -> AgentResult<Self> {
        Self::with_base_url(api_key, "https://generativelanguage.googleapis.com")
    }

    /// Create with a custom base URL (e.g., internal gateway).
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use agent::llm::GeminiProvider;
    ///
    /// let provider = GeminiProvider::with_base_url(
    ///     "test-key",
    ///     "https://gateway.example.invalid/v1",
    /// )?;
    /// # let _ = provider;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> AgentResult<Self> {
        let base = base_url.into();
        let config = HttpClientConfig::new().base_url(&base).timeout_secs(60.0);

        let client = HttpClient::new(config)?;

        Ok(Self {
            client: Arc::new(client),
            api_key: api_key.into(),
            default_model: "gemini-2.0-flash".to_string(),
            _base_url: base,
        })
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    fn convert_messages(
        &self,
        messages: &[Message],
    ) -> (Option<GeminiSystemInstruction>, Vec<GeminiContent>) {
        let mut system_parts: Vec<GeminiPart> = Vec::new();
        let mut contents = Vec::new();
        // Function name of each tool call, by call id, for functionResponse.name.
        let mut call_names: HashMap<String, String> = HashMap::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    system_parts.push(GeminiPart::Text {
                        text: msg.content.clone(),
                    });
                }
                Role::User => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::Text {
                            text: msg.content.clone(),
                        }],
                    });
                }
                Role::Assistant => {
                    let mut parts = vec![];

                    if !msg.content.is_empty() {
                        parts.push(GeminiPart::Text {
                            text: msg.content.clone(),
                        });
                    }

                    if let Some(tool_calls) = &msg.tool_calls {
                        for tool_call in tool_calls {
                            call_names.insert(tool_call.id.clone(), tool_call.name.clone());
                            parts.push(GeminiPart::FunctionCall {
                                function_call: GeminiFunctionCall {
                                    name: tool_call.name.clone(),
                                    args: tool_call.arguments.clone(),
                                },
                            });
                        }
                    }

                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts,
                    });
                }
                Role::Tool => {
                    let id = msg.tool_call_id.clone().unwrap_or_default();
                    let name = call_names
                        .get(&id)
                        .cloned()
                        .or_else(|| msg.name.clone())
                        .unwrap_or_else(|| id.clone());
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::FunctionResponse {
                            function_response: GeminiFunctionResponse {
                                name,
                                response: serde_json::json!({ "result": msg.content }),
                            },
                        }],
                    });
                }
            }
        }

        let system_instruction = if system_parts.is_empty() {
            None
        } else {
            Some(GeminiSystemInstruction {
                parts: system_parts,
            })
        };

        (system_instruction, contents)
    }

    fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<GeminiTool> {
        vec![GeminiTool {
            function_declarations: tools
                .iter()
                .map(|t| GeminiFunctionDeclaration {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                })
                .collect(),
        }]
    }

    /// Build the JSON body of a generateContent request.
    fn build_body(&self, request: &CompletionRequest) -> AgentResult<serde_json::Value> {
        let (system_instruction, contents) = self.convert_messages(&request.messages);

        let mut body = serde_json::json!({
            "contents": contents,
        });

        if let Some(system) = system_instruction {
            body["systemInstruction"] = serde_json::to_value(system)?;
        }

        let mut generation_config = serde_json::Map::new();

        if let Some(temp) = request.temperature {
            generation_config.insert("temperature".to_string(), serde_json::json!(temp));
        }

        if let Some(max_tokens) = request.max_tokens {
            generation_config.insert("maxOutputTokens".to_string(), serde_json::json!(max_tokens));
        }

        // Structured output: map response_schema → responseMimeType + responseSchema
        if let Some(schema) = &request.response_schema {
            generation_config.insert(
                "responseMimeType".to_string(),
                serde_json::json!("application/json"),
            );
            generation_config.insert("responseSchema".to_string(), schema.clone());
        } else {
            // Fallback: check extras injected by complete_structured helper
            if let Some(mime_type) = request.extras.get("response_mime_type") {
                generation_config.insert("responseMimeType".to_string(), mime_type.clone());
            }
            if let Some(resp_schema) = request.extras.get("response_schema") {
                generation_config.insert("responseSchema".to_string(), resp_schema.clone());
            }
        }

        if !generation_config.is_empty() {
            body["generationConfig"] = serde_json::Value::Object(generation_config);
        }

        if let Some(tools) = &request.tools {
            body["tools"] = serde_json::to_value(self.convert_tools(tools))?;
        }

        Ok(body)
    }
}

/// Replace every occurrence of `key` in `text` with `[redacted]`.
fn redact_key(text: &str, key: &str) -> String {
    if key.is_empty() {
        text.to_string()
    } else {
        text.replace(key, "[redacted]")
    }
}

/// Finish reasons that mean the candidate was blocked.
const BLOCKED_FINISH_REASONS: [&str; 5] = [
    "SAFETY",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "RECITATION",
];

/// Parse a generateContent response body.
fn parse_response(response_text: &str, model: String) -> AgentResult<CompletionResponse> {
    let gemini_response: GeminiResponse = serde_json::from_str(response_text)?;

    let candidate = match gemini_response.candidates.into_iter().next() {
        Some(candidate) => candidate,
        None => {
            return Err(
                match gemini_response
                    .prompt_feedback
                    .and_then(|feedback| feedback.block_reason)
                {
                    Some(reason) => AgentError::ApiError(format!(
                        "Gemini blocked the prompt: blockReason={reason}"
                    )),
                    None => AgentError::ApiError("No candidates in response".to_string()),
                },
            );
        }
    };

    if let Some(reason) = candidate.finish_reason.as_deref() {
        if BLOCKED_FINISH_REASONS.contains(&reason) {
            return Err(AgentError::ApiError(format!(
                "Gemini blocked the candidate: finishReason={reason}"
            )));
        }
    }

    let parts = candidate
        .content
        .map(|content| content.parts)
        .unwrap_or_default();

    let mut content = String::new();
    let mut tool_calls = Vec::new();

    for part in parts {
        match part {
            GeminiPart::Text { text } => {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&text);
            }
            GeminiPart::FunctionCall { function_call } => {
                tool_calls.push(ToolCall {
                    id: format!("call_{}", tool_calls.len()),
                    name: function_call.name,
                    arguments: function_call.args,
                });
            }
            _ => {}
        }
    }

    let finish_reason = match candidate.finish_reason.as_deref() {
        Some("STOP") => "stop",
        Some("MAX_TOKENS") => "length",
        Some("SAFETY") => "content_filter",
        _ => "unknown",
    }
    .to_string();

    let usage = gemini_response.usage_metadata.unwrap_or_default();

    Ok(CompletionResponse {
        content,
        finish_reason,
        model,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        usage: TokenUsage {
            prompt_tokens: usage.prompt_token_count,
            completion_tokens: usage.candidates_token_count,
            total_tokens: usage.total_token_count,
        },
        metadata: HashMap::new(),
    })
}

#[async_trait]
impl LLMProvider for GeminiProvider {
    fn provider_name(&self) -> &str {
        "google"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "gemini-2.0-flash".to_string(),
            "gemini-2.0-flash-lite".to_string(),
            "gemini-1.5-pro".to_string(),
            "gemini-1.5-flash".to_string(),
        ]
    }

    async fn complete(&self, request: CompletionRequest) -> AgentResult<CompletionResponse> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        debug!("Gemini completion request: model={}", model);

        let body = self.build_body(&request)?;

        // The key goes in a header, never in the URL, so it cannot reach an error string.
        let url = format!("/v1beta/models/{}:generateContent", model);

        let builder = RequestBuilder::new(HttpMethod::Post, &url)
            .header("x-goog-api-key", self.api_key.clone())
            .header("Content-Type", "application/json")
            .json_value(body);

        let response = self
            .client
            .execute_builder(builder)
            .await
            .map_err(|e| AgentError::HttpError(redact_key(&e.to_string(), &self.api_key)))?;

        if !response.is_success() {
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::ApiError(redact_key(
                &format!(
                    "Gemini API error ({}): {}",
                    response.status_code, error_text
                ),
                &self.api_key,
            )));
        }

        parse_response(&response.text()?, model)
    }

    async fn complete_stream(&self, _request: CompletionRequest) -> AgentResult<StreamResponse> {
        Err(AgentError::NotSupported(
            "Gemini streaming is not supported yet; use complete()".to_string(),
        ))
    }
}

#[derive(Debug, Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
    /// Any other part (for example `inlineData`). Skipped in responses.
    /// Keep this variant last: untagged parsing tries variants in order.
    Other(serde_json::Value),
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(rename = "promptFeedback", default)]
    prompt_feedback: Option<GeminiPromptFeedback>,
}

#[derive(Debug, Deserialize)]
struct GeminiPromptFeedback {
    #[serde(rename = "blockReason")]
    block_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiCandidateContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidateContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize, Default)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
    #[serde(rename = "totalTokenCount", default)]
    total_token_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Response with a function call, an inline image and text, used by several tests.
    const MIXED_PARTS_RESPONSE: &str = r#"{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"AAAA"}},{"text":"hello"},{"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}"#;

    #[test]
    fn test_provider_name() {
        let provider = GeminiProvider::new("test-key").unwrap();
        assert_eq!(provider.provider_name(), "google");
    }

    #[test]
    fn test_supported_models() {
        let provider = GeminiProvider::new("test-key").unwrap();
        let models = provider.supported_models();
        assert!(models.contains(&"gemini-2.0-flash".to_string()));
    }

    #[test]
    fn tool_response_name_is_the_function_name() {
        let provider = GeminiProvider::new("k").unwrap();
        let messages = vec![
            Message::user("w?"),
            Message::assistant("").with_tool_calls(vec![ToolCall {
                id: "call_0".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            }]),
            Message::tool("call_0", "18C"),
        ];
        let (_, contents) = provider.convert_messages(&messages);
        let value = serde_json::to_value(&contents).unwrap();
        assert_eq!(
            value[2]["parts"][0]["functionResponse"]["name"],
            "get_weather"
        );

        let (_, alone) =
            provider.convert_messages(&[Message::tool("zzz", "x").with_name("lookup")]);
        let value = serde_json::to_value(&alone).unwrap();
        assert_eq!(value[0]["parts"][0]["functionResponse"]["name"], "lookup");

        let (_, bare) = provider.convert_messages(&[Message::tool("zzz", "x")]);
        let value = serde_json::to_value(&bare).unwrap();
        assert_eq!(value[0]["parts"][0]["functionResponse"]["name"], "zzz");
    }

    #[test]
    fn two_system_messages_are_both_sent() {
        let provider = GeminiProvider::new("k").unwrap();
        let request = CompletionRequest::new(
            vec![
                Message::system("A"),
                Message::user("hi"),
                Message::system("B"),
            ],
            "gemini-2.0-flash",
        );
        let body = provider.build_body(&request).unwrap();
        assert_eq!(
            body["systemInstruction"]["parts"],
            serde_json::json!([{"text": "A"}, {"text": "B"}])
        );
        assert_eq!(body["contents"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn blocked_candidate_returns_error_with_reason() {
        match parse_response(r#"{"candidates":[{"finishReason":"SAFETY"}]}"#, "m".into()) {
            Err(AgentError::ApiError(m)) => assert!(m.contains("SAFETY"), "message: {m}"),
            _ => panic!("expected ApiError"),
        }
    }

    #[test]
    fn blocked_prompt_returns_error_with_reason() {
        match parse_response(
            r#"{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"}}"#,
            "m".into(),
        ) {
            Err(AgentError::ApiError(m)) => {
                assert!(m.contains("PROHIBITED_CONTENT"), "message: {m}")
            }
            _ => panic!("expected ApiError"),
        }
        match parse_response(r#"{"candidates":[]}"#, "m".into()) {
            Err(AgentError::ApiError(m)) => assert!(m.contains("No candidates"), "message: {m}"),
            _ => panic!("expected ApiError"),
        }
    }

    #[test]
    fn non_text_parts_are_skipped_and_function_calls_kept() {
        let response = parse_response(MIXED_PARTS_RESPONSE, "gemini-2.0-flash".into())
            .expect("mixed parts must parse");
        assert_eq!(response.content, "hello");
        let calls = response.tool_calls.expect("one tool call expected");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, serde_json::json!({"city": "Paris"}));
        assert_eq!(response.finish_reason, "stop");
        assert_eq!(response.usage.total_tokens, 3);
    }

    #[test]
    fn tool_round_trip_request_and_response() {
        let provider = GeminiProvider::new("k").unwrap();
        let tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "d".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let request =
            CompletionRequest::new(vec![Message::user("w?")], "gemini-2.0-flash").with_tools(tools);
        let body = provider.build_body(&request).unwrap();
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["name"],
            "get_weather"
        );

        let response = parse_response(MIXED_PARTS_RESPONSE, "gemini-2.0-flash".into()).unwrap();
        let call = response.tool_calls.expect("tool call expected").remove(0);
        let call_id = call.id.clone();
        let messages = vec![
            Message::assistant("").with_tool_calls(vec![call]),
            Message::tool(call_id, "18C"),
        ];
        let (_, contents) = provider.convert_messages(&messages);
        let value = serde_json::to_value(&contents).unwrap();
        assert_eq!(
            value[1]["parts"][0]["functionResponse"]["name"],
            "get_weather"
        );
        assert_eq!(
            value[1]["parts"][0]["functionResponse"]["response"],
            serde_json::json!({"result": "18C"})
        );
    }

    #[test]
    fn redact_key_replaces_every_occurrence() {
        assert_eq!(redact_key("a K b K", "K"), "a [redacted] b [redacted]");
        assert_eq!(redact_key("a K b", ""), "a K b");
    }

    #[tokio::test]
    async fn complete_stream_returns_not_supported() {
        let provider = GeminiProvider::new("k").unwrap();
        let request = CompletionRequest::new(vec![Message::user("hi")], "gemini-2.0-flash");
        match provider.complete_stream(request).await {
            Err(AgentError::NotSupported(m)) => {
                assert!(m.contains("not supported"), "message: {m}")
            }
            _ => panic!("expected NotSupported"),
        }
    }
}
