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

/// OpenAI LLM provider
pub struct OpenAIProvider {
    client: Arc<HttpClient>,
    api_key: String,
    default_model: String,
}

impl OpenAIProvider {
    pub fn new(api_key: impl Into<String>) -> AgentResult<Self> {
        Self::with_base_url(api_key, "https://api.openai.com")
    }

    /// Create with a custom base URL (e.g., internal gateway, Azure OpenAI).
    pub fn with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> AgentResult<Self> {
        let base_url: String = base_url.into();
        let config = HttpClientConfig::new()
            .base_url(base_url)
            .timeout_secs(60.0);

        let client = HttpClient::new(config)?;

        Ok(Self {
            client: Arc::new(client),
            api_key: api_key.into(),
            default_model: "gpt-4o".to_string(),
        })
    }

    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    fn convert_message(&self, msg: &Message) -> OpenAIMessage {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
        .to_string();

        OpenAIMessage {
            role,
            content: Some(msg.content.clone()),
            name: msg.name.clone(),
            tool_calls: msg.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .map(|c| OpenAIToolCall {
                        id: c.id.clone(),
                        r#type: "function".to_string(),
                        function: OpenAIFunctionCall {
                            name: c.name.clone(),
                            arguments: c.arguments.to_string(),
                        },
                    })
                    .collect()
            }),
            tool_call_id: msg.tool_call_id.clone(),
        }
    }

    fn convert_tool(&self, tool: &ToolDefinition) -> OpenAITool {
        OpenAITool {
            r#type: "function".to_string(),
            function: OpenAIFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            },
        }
    }

    fn build_request(&self, url: &str, body: serde_json::Value) -> RequestBuilder {
        RequestBuilder::new(HttpMethod::Post, url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json_value(body)
    }

    /// Build the JSON body of a chat completion request.
    fn build_body(
        &self,
        request: &CompletionRequest,
        model: &str,
    ) -> AgentResult<serde_json::Value> {
        let messages: Vec<OpenAIMessage> = request
            .messages
            .iter()
            .map(|m| self.convert_message(m))
            .collect();

        let tools: Option<Vec<OpenAITool>> = request
            .tools
            .as_ref()
            .map(|tools| tools.iter().map(|t| self.convert_tool(t)).collect());

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
        });

        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        if let Some(max_tokens) = request.max_tokens {
            let key = if uses_max_completion_tokens(model) {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            body[key] = serde_json::json!(max_tokens);
        }

        if let Some(tools) = tools {
            body["tools"] = serde_json::to_value(tools)?;
        }

        // Structured output: map response_schema → response_format (native) or extras fallback.
        // A schema that cannot meet strict mode is sent unchanged with strict: false.
        if let Some(schema) = &request.response_schema {
            let (strict, schema_value) = match strictify_schema(schema) {
                Some(strict_schema) => (true, strict_schema),
                None => (false, schema.clone()),
            };
            body["response_format"] = serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": strict,
                    "schema": schema_value
                }
            });
        } else if let Some(rf) = request.extras.get("response_format") {
            body["response_format"] = rf.clone();
        }

        Ok(body)
    }
}

/// True when the model needs `max_completion_tokens` instead of `max_tokens`
/// (the o-series and gpt-5 names).
fn uses_max_completion_tokens(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    if matches!(m.as_str(), "o1" | "o3" | "o4") {
        return true;
    }
    ["o1-", "o3-", "o4-", "o1.", "o3.", "o4.", "gpt-5"]
        .iter()
        .any(|prefix| m.starts_with(prefix))
}

/// Return a copy of `schema` that meets OpenAI strict mode, or `None` when it cannot.
fn strictify_schema(schema: &serde_json::Value) -> Option<serde_json::Value> {
    let map = schema.as_object()?;
    let mut out = map.clone();

    let is_object = map.get("type").and_then(|t| t.as_str()) == Some("object")
        || map.contains_key("properties");
    if is_object {
        let properties = map.get("properties")?.as_object()?;
        let required: Vec<&str> = match map.get("required") {
            None => Vec::new(),
            Some(value) => value
                .as_array()?
                .iter()
                .map(|v| v.as_str())
                .collect::<Option<Vec<&str>>>()?,
        };
        // Every property must be required; making an optional one required would change its meaning.
        if properties
            .keys()
            .any(|key| !required.contains(&key.as_str()))
        {
            return None;
        }

        let mut strict_properties = serde_json::Map::new();
        for (key, value) in properties {
            strict_properties.insert(key.clone(), strictify_schema(value)?);
        }
        out.insert(
            "properties".to_string(),
            serde_json::Value::Object(strict_properties),
        );
        out.insert(
            "additionalProperties".to_string(),
            serde_json::Value::Bool(false),
        );
    }

    if let Some(items) = map.get("items") {
        out.insert("items".to_string(), strictify_schema(items)?);
    }

    if let Some(variants) = map.get("anyOf").and_then(|v| v.as_array()) {
        let mut strict_variants = Vec::with_capacity(variants.len());
        for variant in variants {
            strict_variants.push(strictify_schema(variant)?);
        }
        out.insert(
            "anyOf".to_string(),
            serde_json::Value::Array(strict_variants),
        );
    }

    Some(serde_json::Value::Object(out))
}

/// Build the API error. The API key never appears in the message.
fn api_error(status: u16, body: &str, api_key: &str) -> AgentError {
    let text = if api_key.is_empty() {
        body.to_string()
    } else {
        body.replace(api_key, "[redacted]")
    };
    AgentError::ApiError(format!("OpenAI API error ({status}): {text}"))
}

/// Parse a chat completion response body.
fn parse_response(response_text: &str) -> AgentResult<CompletionResponse> {
    let openai_response: OpenAIResponse = serde_json::from_str(response_text)?;

    let choice = openai_response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| AgentError::ApiError("No choices in response".to_string()))?;

    let tool_calls: Option<Vec<ToolCall>> = match choice.message.tool_calls {
        None => None,
        Some(calls) => {
            let mut parsed = Vec::with_capacity(calls.len());
            for c in calls {
                let arguments = if c.function.arguments.trim().is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::from_str(&c.function.arguments).map_err(|_| {
                        AgentError::MalformedLLMResponse(format!(
                            "OpenAI tool call '{}' (id {}) has arguments that are not valid JSON: {}",
                            c.function.name, c.id, c.function.arguments
                        ))
                    })?
                };
                parsed.push(ToolCall {
                    id: c.id,
                    name: c.function.name,
                    arguments,
                });
            }
            Some(parsed)
        }
    };

    Ok(CompletionResponse {
        content: choice.message.content.unwrap_or_default(),
        finish_reason: choice.finish_reason,
        model: openai_response.model,
        tool_calls,
        usage: TokenUsage {
            prompt_tokens: openai_response.usage.prompt_tokens,
            completion_tokens: openai_response.usage.completion_tokens,
            total_tokens: openai_response.usage.total_tokens,
        },
        metadata: HashMap::new(),
    })
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn provider_name(&self) -> &str {
        "openai"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "gpt-4o".to_string(),
            "gpt-4o-mini".to_string(),
            "gpt-4-turbo".to_string(),
            "gpt-4".to_string(),
            "gpt-3.5-turbo".to_string(),
            "o1".to_string(),
            "o1-mini".to_string(),
        ]
    }

    async fn complete(&self, request: CompletionRequest) -> AgentResult<CompletionResponse> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        debug!("OpenAI completion request: model={}", model);

        let body = self.build_body(&request, &model)?;

        let builder = self.build_request("/v1/chat/completions", body);
        let response = self.client.execute_builder(builder).await?;

        if !response.is_success() {
            let error_text = response
                .text()
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(api_error(response.status_code, &error_text, &self.api_key));
        }

        parse_response(&response.text()?)
    }

    async fn complete_stream(&self, _request: CompletionRequest) -> AgentResult<StreamResponse> {
        Err(AgentError::NotSupported(
            "OpenAI streaming is not supported yet; use complete()".to_string(),
        ))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    r#type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAITool {
    r#type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    #[allow(dead_code)]
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Schema;

    /// Optional property `b`: cannot meet strict mode.
    fn optional_property_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"]
        })
    }

    fn tool_call_response(arguments: &str) -> String {
        serde_json::json!({
            "id": "x",
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": arguments}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        })
        .to_string()
    }

    #[test]
    fn test_provider_name() {
        let provider = OpenAIProvider::new("test-key").unwrap();
        assert_eq!(provider.provider_name(), "openai");
    }

    #[test]
    fn test_supported_models() {
        let provider = OpenAIProvider::new("test-key").unwrap();
        let models = provider.supported_models();
        assert!(models.contains(&"gpt-4o".to_string()));
    }

    #[test]
    fn uses_max_completion_tokens_matches_o_series_and_gpt5() {
        for model in [
            "o1",
            "o1-mini",
            "o3-mini",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "O3-mini",
        ] {
            assert!(
                uses_max_completion_tokens(model),
                "expected true for {model}"
            );
        }
        for model in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "ft:gpt-4o:x",
            "omni",
            "",
        ] {
            assert!(
                !uses_max_completion_tokens(model),
                "expected false for {model}"
            );
        }
    }

    #[test]
    fn build_body_uses_max_completion_tokens_for_o3() {
        let provider = OpenAIProvider::new("k").unwrap();
        let request =
            CompletionRequest::new(vec![Message::user("hi")], "o3-mini").with_max_tokens(100);
        let body = provider.build_body(&request, "o3-mini").unwrap();
        assert_eq!(body["max_completion_tokens"], 100);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn build_body_uses_max_tokens_for_gpt4o() {
        let provider = OpenAIProvider::new("k").unwrap();
        let request =
            CompletionRequest::new(vec![Message::user("hi")], "gpt-4o").with_max_tokens(100);
        let body = provider.build_body(&request, "gpt-4o").unwrap();
        assert_eq!(body["max_tokens"], 100);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn strictify_adds_additional_properties_false_to_nested_objects() {
        let input = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            },
            "required": ["name", "address"]
        });
        let out = strictify_schema(&input).expect("schema must be strict-compatible");
        assert_eq!(out["additionalProperties"], serde_json::Value::Bool(false));
        assert_eq!(
            out["properties"]["address"]["additionalProperties"],
            serde_json::Value::Bool(false)
        );
        assert_eq!(
            out["properties"]["address"]["required"],
            serde_json::json!(["city"])
        );
        let mut root_required: Vec<String> = out["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        root_required.sort();
        assert_eq!(
            root_required,
            vec!["address".to_string(), "name".to_string()]
        );
    }

    #[test]
    fn strictify_handles_array_items() {
        let input = serde_json::json!({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"k": {"type": "string"}},
                        "required": ["k"]
                    }
                }
            },
            "required": ["tags"]
        });
        let out = strictify_schema(&input).expect("schema must be strict-compatible");
        assert_eq!(
            out["properties"]["tags"]["items"]["additionalProperties"],
            serde_json::Value::Bool(false)
        );
    }

    #[test]
    fn strictify_returns_none_for_optional_property() {
        assert!(strictify_schema(&optional_property_schema()).is_none());
        assert!(strictify_schema(&serde_json::json!({"type": "object"})).is_none());
    }

    #[test]
    fn build_body_strict_schema_request_shape() {
        let provider = OpenAIProvider::new("k").unwrap();
        let schema = Schema::object()
            .field("name", Schema::string())
            .required(&["name"])
            .build()
            .to_json_schema();
        let request = CompletionRequest::new(vec![Message::user("hi")], "gpt-4o")
            .with_response_schema(schema);
        let body = provider.build_body(&request, "gpt-4o").unwrap();
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["additionalProperties"],
            serde_json::Value::Bool(false)
        );
    }

    #[test]
    fn build_body_non_strict_schema_falls_back() {
        let provider = OpenAIProvider::new("k").unwrap();
        let schema = optional_property_schema();
        let request = CompletionRequest::new(vec![Message::user("hi")], "gpt-4o")
            .with_response_schema(schema.clone());
        let body = provider.build_body(&request, "gpt-4o").unwrap();
        assert_eq!(body["response_format"]["json_schema"]["strict"], false);
        assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
    }

    #[test]
    fn bad_tool_arguments_return_error_with_tool_name_and_raw_text() {
        let text = tool_call_response("{\"city\": Paris");
        match parse_response(&text) {
            Err(AgentError::MalformedLLMResponse(m)) => {
                assert!(m.contains("get_weather"), "message: {m}");
                assert!(m.contains("call_1"), "message: {m}");
                assert!(m.contains("{\"city\": Paris"), "message: {m}");
            }
            _ => panic!("expected MalformedLLMResponse"),
        }
    }

    #[test]
    fn empty_tool_arguments_become_empty_object() {
        let text = tool_call_response("");
        let response = parse_response(&text).expect("empty arguments must parse");
        let calls = response.tool_calls.expect("tool calls expected");
        assert_eq!(calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn tool_round_trip_request_and_response() {
        let provider = OpenAIProvider::new("k").unwrap();
        let messages = vec![
            Message::user("weather in Paris?"),
            Message::assistant("").with_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "Paris"}),
            }]),
            Message::tool("call_1", "18C"),
        ];
        let tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "d".into(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let request = CompletionRequest::new(messages, "gpt-4o").with_tools(tools);
        let body = provider.build_body(&request, "gpt-4o").unwrap();
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_1");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");

        let text = tool_call_response("{\"city\":\"Paris\"}");
        let response = parse_response(&text).expect("valid tool call must parse");
        let calls = response.tool_calls.expect("tool calls expected");
        assert_eq!(calls[0].arguments, serde_json::json!({"city": "Paris"}));
        assert_eq!(response.finish_reason, "tool_calls");
        assert_eq!(response.usage.total_tokens, 3);
        assert_eq!(response.content, "");
    }

    #[test]
    fn api_error_redacts_the_key() {
        match api_error(401, "bad key sk-secret-123 given", "sk-secret-123") {
            AgentError::ApiError(m) => {
                assert!(!m.contains("sk-secret-123"), "message: {m}");
                assert!(m.contains("[redacted]"), "message: {m}");
                assert!(m.contains("401"), "message: {m}");
            }
            _ => panic!("expected ApiError"),
        }
        match api_error(500, "x", "") {
            AgentError::ApiError(m) => assert!(m.contains('x'), "message: {m}"),
            _ => panic!("expected ApiError"),
        }
    }

    #[tokio::test]
    async fn complete_stream_returns_not_supported() {
        let provider = OpenAIProvider::new("k").unwrap();
        let request = CompletionRequest::new(vec![Message::user("hi")], "gpt-4o");
        match provider.complete_stream(request).await {
            Err(AgentError::NotSupported(m)) => {
                assert!(m.contains("not supported"), "message: {m}")
            }
            _ => panic!("expected NotSupported"),
        }
    }
}
