//! Integration tests for the public MCP client and server API.
//!
//! These tests use only the crate's public items. They run in memory: no
//! network and no child process.

use agent::{AgentError, AgentResult};
use agentkit_tools::{
    McpClient, McpServer, McpTool, McpToolCallResult, McpTransport, ERR_INTERNAL,
    ERR_INVALID_PARAMS, ERR_METHOD_NOT_FOUND, JSONRPC_VERSION, MCP_PROTOCOL_VERSION,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn echo_tool() -> McpTool {
    McpTool {
        name: "echo".into(),
        description: Some("echo".into()),
        input_schema: json!({"type": "object"}),
    }
}

fn req(id: u64, method: &str, params: Value) -> Value {
    json!({"jsonrpc": JSONRPC_VERSION, "id": id, "method": method, "params": params})
}

fn ok(id: u64, result: Value) -> Value {
    json!({"jsonrpc": JSONRPC_VERSION, "id": id, "result": result})
}

fn echo_server() -> McpServer {
    let mut server = McpServer::new("n", "1");
    server.register_fn(echo_tool(), |args: Value| async move {
        Ok(McpToolCallResult {
            content: vec![json!({"type": "text", "text": args["msg"].clone()})],
            is_error: false,
        })
    });
    server
}

fn error_message(resp: &Value) -> String {
    resp["error"]["message"]
        .as_str()
        .expect("error.message is a string")
        .to_string()
}

/// Scripted transport: `send` records the envelope, `recv` pops the next reply.
struct FakeTransport {
    sent: Arc<Mutex<Vec<Value>>>,
    replies: Mutex<VecDeque<AgentResult<Value>>>,
    send_error: bool,
}

fn fake(
    replies: Vec<AgentResult<Value>>,
    send_error: bool,
) -> (FakeTransport, Arc<Mutex<Vec<Value>>>) {
    let sent = Arc::new(Mutex::new(Vec::new()));
    let transport = FakeTransport {
        sent: Arc::clone(&sent),
        replies: Mutex::new(replies.into_iter().collect()),
        send_error,
    };
    (transport, sent)
}

#[async_trait]
impl McpTransport for FakeTransport {
    async fn send(&self, message: Value) -> AgentResult<()> {
        if self.send_error {
            return Err(AgentError::ConfigError("send failed".into()));
        }
        self.sent.lock().unwrap().push(message);
        Ok(())
    }

    async fn recv(&self) -> AgentResult<Value> {
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err(AgentError::ConfigError("no reply".into())))
    }
}

/// Transport that hands each request straight to an in-memory server.
struct LoopbackTransport {
    server: Arc<McpServer>,
    pending: Mutex<Option<Value>>,
}

#[async_trait]
impl McpTransport for LoopbackTransport {
    async fn send(&self, message: Value) -> AgentResult<()> {
        let response = self.server.handle_request(message).await;
        *self.pending.lock().unwrap() = Some(response);
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

// ---------------------------------------------------------------------------
// Server tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn server_initialize_tools_list_and_call() {
    let server = echo_server();

    let init = server.handle_request(req(1, "initialize", json!({}))).await;
    assert_eq!(init["jsonrpc"], JSONRPC_VERSION);
    assert_eq!(init["id"], 1);
    assert_eq!(init["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);
    assert_eq!(init["result"]["serverInfo"]["name"], "n");
    assert_eq!(init["result"]["serverInfo"]["version"], "1");

    let list = server.handle_request(req(2, "tools/list", json!({}))).await;
    assert_eq!(list["jsonrpc"], JSONRPC_VERSION);
    assert_eq!(list["id"], 2);
    let tools = list["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "echo");

    let call = server
        .handle_request(req(
            3,
            "tools/call",
            json!({"name": "echo", "arguments": {"msg": "hi"}}),
        ))
        .await;
    assert_eq!(call["jsonrpc"], JSONRPC_VERSION);
    assert_eq!(call["id"], 3);
    assert_eq!(call["result"]["content"][0]["text"], "hi");
    assert_eq!(call["result"]["isError"], false);
}

#[tokio::test(flavor = "current_thread")]
async fn server_unknown_method_is_32601() {
    let server = echo_server();
    let resp = server.handle_request(req(5, "nope", json!({}))).await;

    assert_eq!(resp["error"]["code"], ERR_METHOD_NOT_FOUND);
    assert!(error_message(&resp).contains("nope"));
}

#[tokio::test(flavor = "current_thread")]
async fn server_unknown_tool_is_error_object() {
    let server = echo_server();
    let resp = server
        .handle_request(req(
            6,
            "tools/call",
            json!({"name": "missing", "arguments": {}}),
        ))
        .await;

    assert!(resp.get("result").is_none());
    assert!(resp.get("error").is_some());
    assert_eq!(resp["error"]["code"], ERR_METHOD_NOT_FOUND);
}

#[tokio::test(flavor = "current_thread")]
async fn server_request_without_method_is_invalid_params() {
    let server = echo_server();
    let resp = server
        .handle_request(json!({"jsonrpc": "2.0", "id": 9}))
        .await;

    assert_eq!(resp["error"]["code"], ERR_INVALID_PARAMS);
    assert_eq!(resp["id"], 9);
}

#[tokio::test(flavor = "current_thread")]
async fn server_handler_error_is_internal_and_keeps_id() {
    let mut server = McpServer::new("n", "1");
    server.register_fn(
        McpTool {
            name: "boom".into(),
            description: None,
            input_schema: json!({"type": "object"}),
        },
        |_args: Value| async move { Err(AgentError::ToolError("boom".into())) },
    );

    let resp = server
        .handle_request(req(
            11,
            "tools/call",
            json!({"name": "boom", "arguments": {}}),
        ))
        .await;

    assert_eq!(resp["error"]["code"], ERR_INTERNAL);
    assert!(error_message(&resp).contains("boom"));
    assert_eq!(resp["id"], 11);
}

#[tokio::test(flavor = "current_thread")]
async fn server_tool_is_error_flag_is_kept() {
    let mut server = McpServer::new("n", "1");
    server.register_fn(
        McpTool {
            name: "bad".into(),
            description: None,
            input_schema: json!({"type": "object"}),
        },
        |_args: Value| async move {
            Ok(McpToolCallResult {
                content: vec![],
                is_error: true,
            })
        },
    );

    let resp = server
        .handle_request(req(
            12,
            "tools/call",
            json!({"name": "bad", "arguments": {}}),
        ))
        .await;

    assert_eq!(resp["result"]["isError"], true);
}

// ---------------------------------------------------------------------------
// Client tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn client_handshake_list_and_call() {
    let (transport, sent) = fake(
        vec![
            Ok(ok(1, json!({"protocolVersion": MCP_PROTOCOL_VERSION}))),
            Ok(ok(
                2,
                json!({"tools": [{"name": "echo", "description": "d", "inputSchema": {"type": "object"}}]}),
            )),
            Ok(ok(
                3,
                json!({"content": [{"type": "text", "text": "x"}], "isError": false}),
            )),
        ],
        false,
    );
    let client = McpClient::new(transport);

    assert_eq!(client.initialize().await.unwrap(), MCP_PROTOCOL_VERSION);

    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");

    let result = client.call_tool("echo", json!({"msg": "x"})).await.unwrap();
    assert_eq!(result.content[0]["text"], "x");
    assert!(!result.is_error);

    let sent = sent.lock().unwrap();
    assert_eq!(sent.len(), 3);
    assert_eq!(sent[0]["method"], "initialize");
    assert_eq!(sent[0]["id"], 1);
    assert_eq!(sent[1]["method"], "tools/list");
    assert_eq!(sent[1]["id"], 2);
    assert_eq!(sent[2]["method"], "tools/call");
    assert_eq!(sent[2]["id"], 3);
    assert_eq!(sent[2]["params"]["name"], "echo");
    assert_eq!(sent[2]["params"]["arguments"]["msg"], "x");
}

#[tokio::test(flavor = "current_thread")]
async fn client_returns_err_when_transport_recv_fails() {
    let (transport, _sent) = fake(vec![], false);
    let client = McpClient::new(transport);

    assert!(client.initialize().await.is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn client_returns_err_when_transport_send_fails() {
    let (transport, _sent) = fake(vec![], true);
    let client = McpClient::new(transport);

    assert!(client.initialize().await.is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn client_returns_err_when_transport_reply_is_err() {
    let (transport, _sent) = fake(vec![Err(AgentError::ConfigError("down".into()))], false);
    let client = McpClient::new(transport);

    assert!(client.list_tools().await.is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn client_maps_jsonrpc_error_to_tool_error() {
    let (transport, _sent) = fake(
        vec![Ok(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "nope"}
        }))],
        false,
    );
    let client = McpClient::new(transport);

    match client.initialize().await {
        Err(AgentError::ToolError(message)) => {
            assert!(message.contains("-32601"), "message was: {message}");
            assert!(message.contains("nope"), "message was: {message}");
        }
        other => panic!("expected ToolError, got {other:?}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn client_rejects_wrong_id_and_missing_jsonrpc() {
    // A reply with the wrong id.
    let (transport, _sent) = fake(vec![Ok(ok(99, json!({"protocolVersion": "x"})))], false);
    let client = McpClient::new(transport);
    assert!(matches!(
        client.initialize().await,
        Err(AgentError::InvalidRequest(_))
    ));

    // A reply without the jsonrpc field.
    let (transport, _sent) = fake(
        vec![Ok(json!({"id": 1, "result": {"protocolVersion": "x"}}))],
        false,
    );
    let client = McpClient::new(transport);
    assert!(matches!(
        client.initialize().await,
        Err(AgentError::InvalidRequest(_))
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn client_rejects_malformed_results() {
    // initialize with an empty result has no protocolVersion.
    let (transport, _sent) = fake(vec![Ok(ok(1, json!({})))], false);
    let client = McpClient::new(transport);
    assert!(matches!(
        client.initialize().await,
        Err(AgentError::MalformedLLMResponse(_))
    ));

    // tools/list with an empty result has no tools array.
    let (transport, _sent) = fake(vec![Ok(ok(1, json!({})))], false);
    let client = McpClient::new(transport);
    assert!(matches!(
        client.list_tools().await,
        Err(AgentError::MalformedLLMResponse(_))
    ));

    // tools/call with content that is not an array.
    let (transport, _sent) = fake(vec![Ok(ok(1, json!({"content": "not an array"})))], false);
    let client = McpClient::new(transport);
    assert!(matches!(
        client.call_tool("echo", json!({})).await,
        Err(AgentError::MalformedLLMResponse(_))
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn client_and_server_round_trip() {
    let transport = LoopbackTransport {
        server: Arc::new(echo_server()),
        pending: Mutex::new(None),
    };
    let client = McpClient::new(transport);

    assert_eq!(client.initialize().await.unwrap(), MCP_PROTOCOL_VERSION);

    let tools = client.list_tools().await.unwrap();
    assert!(tools.iter().any(|t| t.name == "echo"));

    let result = client
        .call_tool("echo", json!({"msg": "yo"}))
        .await
        .unwrap();
    assert_eq!(result.content[0]["text"], "yo");

    match client.call_tool("missing", json!({})).await {
        Err(AgentError::ToolError(message)) => {
            assert!(message.contains("-32601"), "message was: {message}");
        }
        other => panic!("expected ToolError, got {other:?}"),
    }
}
