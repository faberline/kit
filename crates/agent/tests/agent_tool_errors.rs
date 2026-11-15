//! Tool errors go back to the model, and a plain text reply is a valid
//! result when the agent has no output schema (#64).

use std::sync::{Arc, Mutex};

use agent::{
    Agent, AgentError, AgentResult, CompletionRequest, CompletionResponse, LLMProvider, Role,
    Schema, StreamResponse, ToolCall, ToolSpec,
};
use async_trait::async_trait;
use serde::Deserialize;

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

fn failing_tool() -> ToolSpec<()> {
    ToolSpec::new(
        "boom",
        "Always fails",
        serde_json::json!({"type": "object"}),
        |_: (), _args: serde_json::Value| async move {
            Err(AgentError::ToolError("kaboom".to_string()))
        },
    )
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

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct WeatherReport {
    city: String,
    temp_c: i32,
}

fn weather_schema() -> Schema {
    Schema::object()
        .field("city", Schema::string())
        .field("temp_c", Schema::integer())
        .required(&["city", "temp_c"])
        .build()
}

#[tokio::test]
async fn failing_tool_error_goes_back_to_model() {
    let (provider, requests) = provider(vec![
        reply_with_tool("boom", serde_json::json!({}), "c1"),
        reply("all done"),
    ]);

    let agent: Agent<(), String> = Agent::builder()
        .provider(provider)
        .model("test-model")
        .deps(())
        .tool(failing_tool())
        .build()
        .unwrap();

    let out = agent.run("go").await.unwrap();
    assert_eq!(out, "all done");

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let error = tool_message_json(&requests[1], "c1");
    assert!(error["error"].as_str().unwrap().contains("kaboom"));
}

#[tokio::test]
async fn unknown_tool_error_goes_back_to_model() {
    let (provider, requests) = provider(vec![
        reply_with_tool("nope", serde_json::json!({}), "c2"),
        reply("recovered"),
    ]);

    let agent: Agent<(), String> = Agent::builder()
        .provider(provider)
        .model("test-model")
        .deps(())
        .build()
        .unwrap();

    let out = agent.run("go").await.unwrap();
    assert_eq!(out, "recovered");

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let error = tool_message_json(&requests[1], "c2");
    let text = error["error"].as_str().unwrap();
    assert!(text.contains("Tool not found"));
    assert!(text.contains("nope"));
}

#[tokio::test]
async fn keeps_failing_ends_with_max_turns() {
    let (provider, _requests) = provider(vec![
        reply_with_tool("boom", serde_json::json!({}), "c1"),
        reply_with_tool("boom", serde_json::json!({}), "c2"),
        reply_with_tool("boom", serde_json::json!({}), "c3"),
    ]);

    let agent: Agent<(), String> = Agent::builder()
        .provider(provider)
        .model("test-model")
        .deps(())
        .tool(failing_tool())
        .max_turns(2)
        .build()
        .unwrap();

    let err = agent.run("go").await.unwrap_err();
    assert!(matches!(err, AgentError::MaxTurnsReached(2)));
}

#[tokio::test]
async fn plain_text_final_reply_without_schema() {
    let (provider, _requests) = provider(vec![reply("Sunny in Taipei")]);

    let agent: Agent<(), String> = Agent::builder()
        .provider(provider)
        .model("test-model")
        .deps(())
        .build()
        .unwrap();

    let out = agent.run("weather?").await.unwrap();
    assert_eq!(out, "Sunny in Taipei");
}

#[tokio::test]
async fn schema_still_requires_json() {
    let (provider, requests) = provider(vec![
        reply("not json"),
        reply(r#"{"city":"Taipei","temp_c":24}"#),
    ]);

    let agent: Agent<(), WeatherReport> = Agent::builder()
        .provider(provider)
        .model("test-model")
        .deps(())
        .output_schema(weather_schema())
        .max_revisions(1)
        .build()
        .unwrap();

    let report = agent.run("weather?").await.unwrap();
    assert_eq!(report.city, "Taipei");
    assert_eq!(report.temp_c, 24);
    assert_eq!(requests.lock().unwrap().len(), 2);
}
