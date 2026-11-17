//! One tool system: a tool given once to the agent is described to the model and run
//! when the model asks for it. Uses a scripted provider; no network, no paid API.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent::tools::{Tool, ToolParameter, ToolRegistry};
use agent::{
    Agent, AgentError, AgentResult, CompletionRequest, CompletionResponse, LLMProvider, Role,
    StreamResponse, ToolCall, ToolSpec,
};
use async_trait::async_trait;

/// Replies in order, and records every request the agent sends.
#[derive(Clone)]
struct RecordingProvider {
    replies: Arc<Mutex<Vec<CompletionResponse>>>,
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

#[async_trait]
impl LLMProvider for RecordingProvider {
    fn provider_name(&self) -> &str {
        "recording"
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["test-model".into()]
    }
    async fn complete(&self, req: CompletionRequest) -> AgentResult<CompletionResponse> {
        self.requests.lock().unwrap().push(req);
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

fn provider(
    replies: Vec<CompletionResponse>,
) -> (Arc<RecordingProvider>, Arc<Mutex<Vec<CompletionRequest>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = Arc::new(RecordingProvider {
        replies: Arc::new(Mutex::new(replies)),
        requests: requests.clone(),
    });
    (provider, requests)
}

fn reply(content: &str) -> CompletionResponse {
    CompletionResponse {
        content: content.to_string(),
        tool_calls: None,
        finish_reason: "stop".into(),
        usage: Default::default(),
        model: "test-model".into(),
        metadata: Default::default(),
    }
}

fn reply_with_tool(name: &str, args: serde_json::Value, id: &str) -> CompletionResponse {
    CompletionResponse {
        content: String::new(),
        tool_calls: Some(vec![ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: args,
        }]),
        finish_reason: "tool_use".into(),
        usage: Default::default(),
        model: "test-model".into(),
        metadata: Default::default(),
    }
}

/// The tool message with `call_id` in a recorded request, as parsed JSON.
fn tool_message_json(req: &CompletionRequest, call_id: &str) -> serde_json::Value {
    let msg = req
        .messages
        .iter()
        .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some(call_id))
        .unwrap_or_else(|| panic!("no tool message for {call_id}"));
    serde_json::from_str(&msg.content).expect("tool message is JSON")
}

/// Names of the tools that a recorded request lists for the model, sorted.
fn listed_tool_names(req: &CompletionRequest) -> Vec<String> {
    let mut names: Vec<String> = req
        .tools
        .as_ref()
        .expect("the request lists the tools")
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    names.sort();
    names
}

fn echo_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Message to echo"}
        },
        "required": ["message"]
    })
}

struct EchoTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echo the message"
    }

    fn parameters(&self) -> Vec<ToolParameter> {
        vec![ToolParameter {
            name: "message".to_string(),
            description: "Message to echo".to_string(),
            required: true,
            parameter_type: "string".to_string(),
        }]
    }

    async fn execute(&self, arguments: serde_json::Value) -> AgentResult<serde_json::Value> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(serde_json::json!({"echoed": arguments["message"]}))
    }
}

struct UpperTool;

#[async_trait]
impl Tool for UpperTool {
    fn name(&self) -> &str {
        "upper"
    }

    fn description(&self) -> &str {
        "Upper-case the message"
    }

    fn parameters(&self) -> Vec<ToolParameter> {
        vec![]
    }

    async fn execute(&self, _arguments: serde_json::Value) -> AgentResult<serde_json::Value> {
        Ok(serde_json::json!({}))
    }
}

#[tokio::test]
async fn tool_registered_once_is_described_and_executed() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (provider, requests) = provider(vec![
        reply_with_tool("echo", serde_json::json!({"message": "hi"}), "call_1"),
        reply("done"),
    ]);

    let agent: Agent<(), String> = Agent::builder()
        .provider(provider)
        .model("test-model")
        .deps(())
        .tool_impl(Arc::new(EchoTool {
            calls: calls.clone(),
        }))
        .build()
        .unwrap();

    let out = agent.run("go").await.unwrap();
    assert_eq!(out, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);

    let listed = requests[0]
        .tools
        .as_ref()
        .expect("the first request lists the tools");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].name, "echo");
    assert_eq!(listed[0].parameters, echo_schema());

    assert_eq!(
        tool_message_json(&requests[1], "call_1"),
        serde_json::json!({"echoed": "hi"})
    );
}

#[tokio::test]
async fn tool_registry_registers_every_tool() {
    let registry = ToolRegistry::new();
    registry
        .register(Arc::new(EchoTool {
            calls: Arc::new(AtomicUsize::new(0)),
        }))
        .unwrap();
    registry.register(Arc::new(UpperTool)).unwrap();

    let (provider, requests) = provider(vec![reply("ok")]);
    let agent: Agent<(), String> = Agent::builder()
        .provider(provider)
        .model("test-model")
        .deps(())
        .tool_registry(&registry)
        .build()
        .unwrap();

    agent.run("go").await.unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(
        listed_tool_names(&requests[0]),
        vec!["echo".to_string(), "upper".to_string()]
    );
}

#[tokio::test]
async fn missing_required_argument_is_returned_to_the_model() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (provider, requests) = provider(vec![
        reply_with_tool("echo", serde_json::json!({}), "call_1"),
        reply("done"),
    ]);

    let agent: Agent<(), String> = Agent::builder()
        .provider(provider)
        .model("test-model")
        .deps(())
        .tool_impl(Arc::new(EchoTool {
            calls: calls.clone(),
        }))
        .build()
        .unwrap();

    let out = agent.run("go").await.unwrap();
    assert_eq!(out, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 0, "execute must not run");

    let requests = requests.lock().unwrap();
    let error = tool_message_json(&requests[1], "call_1");
    assert!(
        error["error"]
            .as_str()
            .unwrap()
            .contains("Missing required parameter: message"),
        "tool message: {error}"
    );
}

#[tokio::test]
async fn tool_impl_and_tool_spec_can_be_mixed() {
    let calls = Arc::new(AtomicUsize::new(0));
    let ping = ToolSpec::new(
        "ping",
        "Ping the host",
        serde_json::json!({"type": "object"}),
        |_: (), _args: serde_json::Value| async move { Ok(serde_json::json!({"pong": true})) },
    );

    let (provider, requests) = provider(vec![reply("ok")]);
    let agent: Agent<(), String> = Agent::builder()
        .provider(provider)
        .model("test-model")
        .deps(())
        .tool_impl(Arc::new(EchoTool { calls }))
        .tool(ping)
        .build()
        .unwrap();

    agent.run("go").await.unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(
        listed_tool_names(&requests[0]),
        vec!["echo".to_string(), "ping".to_string()]
    );
}
