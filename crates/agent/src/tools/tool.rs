use crate::error::{AgentError, AgentResult};
use crate::tools::registry::ToolRegistry;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, error, warn};

/// Tool parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub parameter_type: String,
}

/// Tool definition with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
}

/// Tool trait - all tools must implement this
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Vec<ToolParameter>;
    async fn execute(&self, arguments: serde_json::Value) -> AgentResult<serde_json::Value>;

    /// True when calling the tool again with the same arguments is safe, so
    /// `ToolExecutor` may retry it after an error. Default: false.
    fn is_idempotent(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
        }
    }

    fn validate_arguments(&self, arguments: &serde_json::Value) -> AgentResult<()> {
        let args = arguments
            .as_object()
            .ok_or_else(|| AgentError::InvalidArguments("Arguments must be an object".into()))?;

        for param in self.parameters() {
            if param.required && !args.contains_key(&param.name) {
                return Err(AgentError::InvalidArguments(format!(
                    "Missing required parameter: {}",
                    param.name
                )));
            }
        }

        Ok(())
    }
}

/// Tool executor with timeout and retry support
pub struct ToolExecutor {
    registry: Arc<ToolRegistry>,
    timeout_duration: Duration,
    max_retries: u32,
}

impl ToolExecutor {
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            timeout_duration: Duration::from_secs(30),
            max_retries: 1,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_duration = timeout;
        self
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub async fn execute(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> AgentResult<serde_json::Value> {
        debug!("Executing tool: {} with args: {:?}", tool_name, arguments);

        let tool = self
            .registry
            .get(tool_name)
            .ok_or_else(|| AgentError::ToolNotFound(tool_name.to_string()))?;

        let mut attempt = 0;
        loop {
            let result = self
                .execute_with_timeout(tool.clone(), arguments.clone())
                .await;

            match result {
                Ok(output) => {
                    debug!("Tool {} executed successfully", tool_name);
                    return Ok(output);
                }
                Err(e) if tool.is_idempotent() && attempt < self.max_retries => {
                    warn!(
                        "Tool {} execution failed (attempt {}): {}. Retrying...",
                        tool_name,
                        attempt + 1,
                        e
                    );
                    attempt += 1;
                }
                Err(e) => {
                    error!("Tool {} execution failed: {}", tool_name, e);
                    return Err(e);
                }
            }
        }
    }

    async fn execute_with_timeout(
        &self,
        tool: Arc<dyn Tool>,
        arguments: serde_json::Value,
    ) -> AgentResult<serde_json::Value> {
        timeout(self.timeout_duration, tool.execute(arguments))
            .await
            .map_err(|_| AgentError::Timeout(self.timeout_duration.as_secs()))?
    }
}

impl Clone for ToolExecutor {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            timeout_duration: self.timeout_duration,
            max_retries: self.max_retries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echo the input"
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
            Ok(arguments)
        }
    }

    #[tokio::test]
    async fn test_tool_executor() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(EchoTool)).unwrap();

        let executor = ToolExecutor::new(registry);
        let result = executor
            .execute("echo", serde_json::json!({"message": "hello"}))
            .await
            .unwrap();

        assert_eq!(result["message"], "hello");
    }

    #[tokio::test]
    async fn test_tool_not_found() {
        let registry = Arc::new(ToolRegistry::new());
        let executor = ToolExecutor::new(registry);

        let result = executor.execute("nonexistent", serde_json::json!({})).await;
        assert!(result.is_err());
    }

    struct CountingTool {
        calls: Arc<AtomicUsize>,
        idempotent: bool,
        fail_first_only: bool,
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            "counting"
        }

        fn description(&self) -> &str {
            "Count calls and fail"
        }

        fn parameters(&self) -> Vec<ToolParameter> {
            Vec::new()
        }

        fn is_idempotent(&self) -> bool {
            self.idempotent
        }

        async fn execute(&self, _arguments: serde_json::Value) -> AgentResult<serde_json::Value> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if !self.fail_first_only || n == 0 {
                Err(AgentError::ToolError("boom".into()))
            } else {
                Ok(serde_json::json!({"ok": true}))
            }
        }
    }

    fn counting_executor(tool: CountingTool) -> ToolExecutor {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(tool)).unwrap();
        ToolExecutor::new(registry)
    }

    #[tokio::test]
    async fn non_idempotent_tool_runs_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = counting_executor(CountingTool {
            calls: calls.clone(),
            idempotent: false,
            fail_first_only: false,
        });

        let result = executor.execute("counting", serde_json::json!({})).await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn idempotent_tool_is_retried_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = counting_executor(CountingTool {
            calls: calls.clone(),
            idempotent: true,
            fail_first_only: false,
        });

        let result = executor.execute("counting", serde_json::json!({})).await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn idempotent_tool_recovers_after_one_failure() {
        let calls = Arc::new(AtomicUsize::new(0));
        let executor = counting_executor(CountingTool {
            calls: calls.clone(),
            idempotent: true,
            fail_first_only: true,
        });

        let result = executor.execute("counting", serde_json::json!({})).await;
        assert!(result.is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn file_tools_declare_idempotency() {
        use crate::tools::bash::BashTool;
        use crate::tools::file::{EditFileTool, GlobTool, GrepTool, ReadFileTool, WriteFileTool};

        assert!(!WriteFileTool::new().is_idempotent());
        assert!(!EditFileTool::new().is_idempotent());
        assert!(ReadFileTool::new().is_idempotent());
        assert!(GlobTool::new().is_idempotent());
        assert!(GrepTool::new().is_idempotent());
        assert!(!BashTool::new().is_idempotent());
    }
}
