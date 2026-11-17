//! Newline-delimited JSON over stdio for the MCP client and server (#149).
//!
//! One JSON-RPC message is one line: the JSON text has no embedded newline,
//! and each message ends with `\n`. [`StdioTransport`] plugs into
//! [`crate::mcp_client::McpClient`]; [`serve_stdio`] drives an
//! [`crate::mcp_server::McpServer`] over any reader and writer pair.

use std::process::Stdio;

use agent::{AgentError, AgentResult};
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

use crate::mcp_client::McpTransport;
use crate::mcp_server::McpServer;

/// A line-based MCP transport over any async reader and writer.
pub struct StdioTransport {
    reader: Mutex<BufReader<Box<dyn AsyncRead + Unpin + Send>>>,
    writer: Mutex<Box<dyn AsyncWrite + Unpin + Send>>,
}

impl StdioTransport {
    /// Build a transport that reads messages from `reader` and writes them to `writer`.
    pub fn new<R, W>(reader: R, writer: W) -> Self
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let reader: Box<dyn AsyncRead + Unpin + Send> = Box::new(reader);
        let writer: Box<dyn AsyncWrite + Unpin + Send> = Box::new(writer);
        Self {
            reader: Mutex::new(BufReader::new(reader)),
            writer: Mutex::new(writer),
        }
    }
}

/// Map an I/O failure on the pipe to the crate's transport error.
fn io_err(e: std::io::Error) -> AgentError {
    AgentError::ConfigError(format!("MCP stdio: {e}"))
}

/// The error text that [`StdioTransport::recv`] returns at EOF.
const CLOSED: &str = "MCP stdio: closed";

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&self, message: Value) -> AgentResult<()> {
        let line = serde_json::to_string(&message)
            .map_err(|e| AgentError::InvalidRequest(e.to_string()))?;
        let mut writer = self.writer.lock().await;
        writer.write_all(line.as_bytes()).await.map_err(io_err)?;
        writer.write_all(b"\n").await.map_err(io_err)?;
        writer.flush().await.map_err(io_err)?;
        Ok(())
    }

    async fn recv(&self) -> AgentResult<Value> {
        let mut reader = self.reader.lock().await;
        loop {
            let mut line = String::new();
            let read = reader.read_line(&mut line).await.map_err(io_err)?;
            if read == 0 {
                return Err(AgentError::ConfigError(CLOSED.into()));
            }
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            return serde_json::from_str::<Value>(text).map_err(|e| {
                AgentError::MalformedLLMResponse(format!("MCP stdio: invalid JSON: {e}"))
            });
        }
    }
}

/// Serve MCP requests from `reader` and write the replies to `writer`, one
/// line each, until `reader` reaches EOF.
pub async fn serve_stdio<R, W>(server: &McpServer, reader: R, writer: W) -> AgentResult<()>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let transport = StdioTransport::new(reader, writer);
    loop {
        let message = match transport.recv().await {
            Ok(message) => message,
            Err(AgentError::ConfigError(m)) if m == CLOSED => return Ok(()),
            Err(e) => return Err(e),
        };
        if let Some(reply) = server.handle_message(message).await {
            transport.send(reply).await?;
        }
    }
}

/// Start `command` with piped stdin and stdout and return a client transport
/// on those pipes, with the child process.
pub fn spawn_stdio_client(
    command: &str,
    args: &[&str],
) -> AgentResult<(StdioTransport, tokio::process::Child)> {
    let mut child = tokio::process::Command::new(command)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| AgentError::ConfigError(format!("MCP stdio: cannot start {command}: {e}")))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| AgentError::ConfigError("MCP stdio: no stdin pipe".into()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AgentError::ConfigError("MCP stdio: no stdout pipe".into()))?;
    Ok((StdioTransport::new(stdout, stdin), child))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_client::{McpClient, McpTool, McpToolCallResult, MCP_PROTOCOL_VERSION};
    use serde_json::json;
    use std::sync::Arc;

    fn echo_tool() -> McpTool {
        McpTool {
            name: "echo".into(),
            description: Some("echo args verbatim".into()),
            input_schema: json!({ "type": "object" }),
        }
    }

    fn echo_server() -> McpServer {
        let mut server = McpServer::new("agentkit", "1.0.0");
        server.register_fn(echo_tool(), |args| async move {
            Ok(McpToolCallResult {
                content: vec![json!({ "type": "text", "text": args["msg"].clone() })],
                is_error: false,
            })
        });
        server
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stdio_round_trip_client_server() {
        let (c_to_s_w, c_to_s_r) = tokio::io::duplex(4096);
        let (s_to_c_w, s_to_c_r) = tokio::io::duplex(4096);
        let server = Arc::new(echo_server());
        let server_task = {
            let server = Arc::clone(&server);
            tokio::spawn(async move { serve_stdio(&server, c_to_s_r, s_to_c_w).await })
        };

        let client = McpClient::new(StdioTransport::new(s_to_c_r, c_to_s_w));

        let v = client.initialize().await.unwrap();
        assert_eq!(v, MCP_PROTOCOL_VERSION);

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools[0].name, "echo");

        let r = client
            .call_tool("echo", json!({ "msg": "yo" }))
            .await
            .unwrap();
        assert_eq!(r.content[0]["text"], "yo");

        // The client still holds the writer, so the server has not reached
        // EOF yet; the task is left running and dropped with the runtime.
        drop(server_task);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn recv_skips_blank_lines() {
        let (mut w, r) = tokio::io::duplex(4096);
        let (sink_w, _sink_r) = tokio::io::duplex(4096);
        use tokio::io::AsyncWriteExt as _;
        w.write_all(b"\n  \n{\"a\":1}\n").await.unwrap();
        let t = StdioTransport::new(r, sink_w);
        let v = t.recv().await.unwrap();
        assert_eq!(v, json!({ "a": 1 }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn recv_at_eof_is_an_error() {
        let (w, r) = tokio::io::duplex(4096);
        let (sink_w, _sink_r) = tokio::io::duplex(4096);
        drop(w);
        let t = StdioTransport::new(r, sink_w);
        match t.recv().await {
            Err(AgentError::ConfigError(m)) => assert_eq!(m, "MCP stdio: closed"),
            other => panic!("expected closed error, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn recv_invalid_json_is_an_error() {
        let (mut w, r) = tokio::io::duplex(4096);
        let (sink_w, _sink_r) = tokio::io::duplex(4096);
        use tokio::io::AsyncWriteExt as _;
        w.write_all(b"not json\n").await.unwrap();
        let t = StdioTransport::new(r, sink_w);
        assert!(matches!(
            t.recv().await,
            Err(AgentError::MalformedLLMResponse(_))
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn send_writes_one_line() {
        use tokio::io::AsyncReadExt as _;
        let (_in_w, in_r) = tokio::io::duplex(4096);
        let (out_w, mut out_r) = tokio::io::duplex(4096);
        let t = StdioTransport::new(in_r, out_w);
        t.send(json!({ "x": 1 })).await.unwrap();
        drop(t);
        let mut out = String::new();
        out_r.read_to_string(&mut out).await.unwrap();
        assert_eq!(out, "{\"x\":1}\n");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn serve_stdio_ignores_notifications() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let server = echo_server();
        let (mut in_w, in_r) = tokio::io::duplex(4096);
        let (out_w, mut out_r) = tokio::io::duplex(4096);
        in_w.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
            .await
            .unwrap();
        drop(in_w);
        serve_stdio(&server, in_r, out_w).await.unwrap();
        let mut out = String::new();
        out_r.read_to_string(&mut out).await.unwrap();
        assert!(
            out.is_empty(),
            "a notification must get no reply, got {out:?}"
        );
    }
}
