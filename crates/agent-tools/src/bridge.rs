//! Use the tools of a remote MCP server as agent tools (#150).
//!
//! [`McpBridgeTool`] wraps one [`McpTool`] of an [`McpClient`] and implements
//! [`agent::Tool`], so an `Agent` can call it like any other tool. Register it
//! with `AgentBuilder::tool_impl`.

use std::sync::Arc;

use agent::{AgentError, AgentResult, Tool, ToolParameter};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::mcp_client::{McpClient, McpTool};

/// One tool of an MCP server, callable from an `Agent`.
pub struct McpBridgeTool {
    client: Arc<McpClient>,
    spec: McpTool,
}

impl McpBridgeTool {
    /// Wrap `spec`, a tool that `client`'s server provides.
    pub fn new(client: Arc<McpClient>, spec: McpTool) -> Self {
        Self { client, spec }
    }

    /// One bridged tool for each tool that the server lists. The caller runs
    /// `initialize` first.
    pub async fn from_client(client: Arc<McpClient>) -> AgentResult<Vec<McpBridgeTool>> {
        let specs = client.list_tools().await?;
        Ok(specs
            .into_iter()
            .map(|spec| McpBridgeTool::new(Arc::clone(&client), spec))
            .collect())
    }
}

#[async_trait]
impl Tool for McpBridgeTool {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn description(&self) -> &str {
        self.spec.description.as_deref().unwrap_or("")
    }

    fn parameters(&self) -> Vec<ToolParameter> {
        let schema = &self.spec.input_schema;
        let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
            return Vec::new();
        };
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let mut params: Vec<ToolParameter> = properties
            .iter()
            .map(|(name, prop)| ToolParameter {
                name: name.clone(),
                description: prop
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                required: required.contains(&name.as_str()),
                parameter_type: prop
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("string")
                    .to_string(),
            })
            .collect();
        params.sort_by(|a, b| a.name.cmp(&b.name));
        params
    }

    async fn execute(&self, arguments: Value) -> AgentResult<Value> {
        let result = self.client.call_tool(&self.spec.name, arguments).await?;
        if result.is_error {
            let texts: Vec<&str> = result
                .content
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect();
            let message = if texts.is_empty() {
                "MCP tool returned an error".to_string()
            } else {
                texts.join("\n")
            };
            return Err(AgentError::ToolError(message));
        }
        Ok(json!({ "content": result.content }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_client::{McpToolCallResult, McpTransport};
    use crate::mcp_server::McpServer;
    use agent::{
        Agent, CompletionRequest, CompletionResponse, LLMProvider, Schema, StreamResponse,
        TokenUsage, ToolCall,
    };
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex as StdMutex;

    fn echo_spec() -> McpTool {
        McpTool {
            name: "echo".into(),
            description: Some("echo args verbatim".into()),
            input_schema: json!({
                "type": "object",
                "properties": { "msg": { "type": "string", "description": "text" } },
                "required": ["msg"],
            }),
        }
    }

    /// Feeds each server reply back to the client, like a pipe.
    struct LoopbackTransport {
        server: Arc<McpServer>,
        pending: StdMutex<Option<Value>>,
    }

    #[async_trait]
    impl McpTransport for LoopbackTransport {
        async fn send(&self, message: Value) -> AgentResult<()> {
            if let Some(response) = self.server.handle_message(message).await {
                *self.pending.lock().unwrap() = Some(response);
            }
            Ok(())
        }
        async fn recv(&self) -> AgentResult<Value> {
            self.pending
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| AgentError::ConfigError("no pending response".into()))
        }
    }

    fn client_for(server: McpServer) -> Arc<McpClient> {
        Arc::new(McpClient::new(LoopbackTransport {
            server: Arc::new(server),
            pending: StdMutex::new(None),
        }))
    }

    #[derive(Clone)]
    struct ScriptedProvider {
        replies: Arc<StdMutex<Vec<CompletionResponse>>>,
    }

    impl ScriptedProvider {
        fn new(replies: Vec<CompletionResponse>) -> Self {
            Self {
                replies: Arc::new(StdMutex::new(replies)),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for ScriptedProvider {
        fn provider_name(&self) -> &str {
            "scripted"
        }
        fn supported_models(&self) -> Vec<String> {
            vec!["test-model".into()]
        }
        async fn complete(&self, _req: CompletionRequest) -> AgentResult<CompletionResponse> {
            let mut q = self.replies.lock().unwrap();
            if q.is_empty() {
                return Err(AgentError::LLMError("no scripted reply left".into()));
            }
            Ok(q.remove(0))
        }
        async fn complete_stream(&self, _req: CompletionRequest) -> AgentResult<StreamResponse> {
            Err(AgentError::NotSupported("stream".into()))
        }
        fn validate_model(&self, _model: &str) -> AgentResult<()> {
            Ok(())
        }
    }

    fn reply(content: &str) -> CompletionResponse {
        CompletionResponse {
            content: content.to_string(),
            tool_calls: None,
            finish_reason: "stop".into(),
            usage: TokenUsage::default(),
            model: "test-model".into(),
            metadata: Default::default(),
        }
    }

    fn reply_with_tool(name: &str, args: Value, id: &str) -> CompletionResponse {
        CompletionResponse {
            content: String::new(),
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: args,
            }]),
            finish_reason: "tool_use".into(),
            usage: TokenUsage::default(),
            model: "test-model".into(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn bridge_lists_parameters_from_schema() {
        let client = client_for(McpServer::new("agentkit", "1.0.0"));
        let spec = McpTool {
            name: "calc".into(),
            description: None,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "a": { "type": "integer" },
                    "b": { "type": "string", "description": "bee" },
                },
                "required": ["a"],
            }),
        };
        let tool = McpBridgeTool::new(client.clone(), spec);
        let params = tool.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "a");
        assert!(params[0].required);
        assert_eq!(params[0].parameter_type, "integer");
        assert_eq!(params[0].description, "");
        assert_eq!(params[1].name, "b");
        assert!(!params[1].required);
        assert_eq!(params[1].parameter_type, "string");
        assert_eq!(params[1].description, "bee");
        assert_eq!(tool.description(), "");

        let bare = McpBridgeTool::new(
            client,
            McpTool {
                name: "bare".into(),
                description: None,
                input_schema: json!({ "type": "object" }),
            },
        );
        assert!(bare.parameters().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn bridge_execute_calls_server() {
        let mut server = McpServer::new("agentkit", "1.0.0");
        server.register_fn(echo_spec(), |args| async move {
            Ok(McpToolCallResult {
                content: vec![json!({ "type": "text", "text": args["msg"].clone() })],
                is_error: false,
            })
        });
        let client = client_for(server);
        client.initialize().await.unwrap();

        let tools = McpBridgeTool::from_client(client.clone()).await.unwrap();
        assert_eq!(tools.len(), 1);
        let out = tools[0].execute(json!({ "msg": "hi" })).await.unwrap();
        assert_eq!(
            out,
            json!({ "content": [{ "type": "text", "text": "hi" }] })
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn bridge_maps_is_error_to_tool_error() {
        let mut server = McpServer::new("agentkit", "1.0.0");
        server.register_fn(echo_spec(), |_args| async move {
            Ok(McpToolCallResult {
                content: vec![json!({ "type": "text", "text": "boom" })],
                is_error: true,
            })
        });
        let client = client_for(server);
        client.initialize().await.unwrap();
        let tools = McpBridgeTool::from_client(client).await.unwrap();
        match tools[0].execute(json!({ "msg": "x" })).await {
            Err(AgentError::ToolError(m)) => assert_eq!(m, "boom"),
            other => panic!("expected ToolError, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn agent_calls_bridged_tool() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut server = McpServer::new("agentkit", "1.0.0");
        let counter = Arc::clone(&calls);
        server.register_fn(echo_spec(), move |args| {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(McpToolCallResult {
                    content: vec![json!({ "type": "text", "text": args["msg"].clone() })],
                    is_error: false,
                })
            }
        });
        let client = client_for(server);
        client.initialize().await.unwrap();
        let tool = McpBridgeTool::from_client(client)
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let provider = Arc::new(ScriptedProvider::new(vec![
            reply_with_tool("echo", json!({ "msg": "yo" }), "call_1"),
            reply(r#"{"text":"done"}"#),
        ]));
        let schema = Schema::object()
            .field("text", Schema::string())
            .required(&["text"])
            .build();
        let agent: Agent<(), Value> = Agent::builder()
            .provider(provider)
            .model("test-model")
            .deps(())
            .tool_impl(Arc::new(tool))
            .output_schema(schema)
            .build()
            .unwrap();

        let out = agent.run("say it").await.unwrap();
        assert_eq!(out["text"], "done");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn agent_rejects_missing_required_argument() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut server = McpServer::new("agentkit", "1.0.0");
        let counter = Arc::clone(&calls);
        server.register_fn(echo_spec(), move |_args| {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(McpToolCallResult {
                    content: vec![],
                    is_error: false,
                })
            }
        });
        let client = client_for(server);
        client.initialize().await.unwrap();
        let tool = McpBridgeTool::from_client(client)
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert!(tool.validate_arguments(&json!({})).is_err());

        let provider = Arc::new(ScriptedProvider::new(vec![
            reply_with_tool("echo", json!({}), "call_1"),
            reply(r#"{"text":"done"}"#),
        ]));
        let schema = Schema::object()
            .field("text", Schema::string())
            .required(&["text"])
            .build();
        let agent: Agent<(), Value> = Agent::builder()
            .provider(provider)
            .model("test-model")
            .deps(())
            .tool_impl(Arc::new(tool))
            .output_schema(schema)
            .build()
            .unwrap();

        let _ = agent.run("say it").await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "the server must not see a call without its required argument"
        );
    }
}
