use crate::error::{AgentError, AgentResult};
use crate::security::SecurityPolicy;
use crate::tools::text::truncate_at_char_boundary;
use crate::tools::tool::{Tool, ToolParameter};
use async_trait::async_trait;
use serde::Deserialize;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, warn};

/// Tool for executing shell commands
pub struct BashTool {
    default_timeout: Duration,
    max_output_size: usize,
    shell: String,
    working_dir: Option<String>,
    policy: Arc<SecurityPolicy>,
}

impl BashTool {
    pub fn new() -> Self {
        Self {
            default_timeout: Duration::from_secs(120),
            max_output_size: 100_000,
            shell: "/bin/bash".to_string(),
            working_dir: None,
            policy: Arc::new(SecurityPolicy::default()),
        }
    }

    pub fn with_policy(mut self, policy: Arc<SecurityPolicy>) -> Self {
        self.policy = policy;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    pub fn with_max_output_size(mut self, size: usize) -> Self {
        self.max_output_size = size;
        self
    }

    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    async fn execute_impl(&self, args: BashArgs) -> AgentResult<serde_json::Value> {
        self.policy.check_command(&args.command)?;

        let timeout_ms = args
            .timeout
            .unwrap_or(self.default_timeout.as_millis() as u64);
        let timeout_duration = Duration::from_millis(timeout_ms);

        debug!("Executing command: {}", args.command);

        let mut cmd = Command::new(&self.shell);
        cmd.arg("-c").arg(&args.command);

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        // A child that is still alive when the handle drops is killed, not left running.
        cmd.kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| AgentError::CommandFailed(format!("Failed to spawn command: {}", e)))?;

        let mut stdout_pipe = child.stdout.take().unwrap();
        let mut stderr_pipe = child.stderr.take().unwrap();
        let mut out_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            let _ = stdout_pipe.read_to_end(&mut buf).await;
            buf
        });
        let mut err_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            let _ = stderr_pipe.read_to_end(&mut buf).await;
            buf
        });

        let (status, stdout_bytes, stderr_bytes) = match timeout(timeout_duration, async {
            tokio::join!(child.wait(), &mut out_task, &mut err_task)
        })
        .await
        {
            Ok((status, out, err)) => (status, out.unwrap_or_default(), err.unwrap_or_default()),
            Err(_) => {
                // Kill and reap the shell. A grandchild may still hold the pipes,
                // so the readers are aborted, not waited for.
                let _ = child.kill().await;
                out_task.abort();
                err_task.abort();
                return Err(AgentError::CommandTimeout(timeout_ms / 1000));
            }
        };
        let status = status
            .map_err(|e| AgentError::CommandFailed(format!("Command execution failed: {}", e)))?;

        let exit_code = status.code().unwrap_or(-1);
        let mut stdout = String::from_utf8_lossy(&stdout_bytes).to_string();
        let mut stderr = String::from_utf8_lossy(&stderr_bytes).to_string();

        let stdout_truncated = stdout.len() > self.max_output_size;
        let stderr_truncated = stderr.len() > self.max_output_size;

        if stdout_truncated {
            truncate_at_char_boundary(&mut stdout, self.max_output_size);
            stdout.push_str("\n... [output truncated]");
        }

        if stderr_truncated {
            truncate_at_char_boundary(&mut stderr, self.max_output_size);
            stderr.push_str("\n... [output truncated]");
        }

        Ok(serde_json::json!({
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "success": exit_code == 0,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated
        }))
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
    timeout: Option<u64>,
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command and return its output."
    }

    fn parameters(&self) -> Vec<ToolParameter> {
        vec![
            ToolParameter {
                name: "command".to_string(),
                description: "The bash command to execute".to_string(),
                required: true,
                parameter_type: "string".to_string(),
            },
            ToolParameter {
                name: "timeout".to_string(),
                description: "Timeout in milliseconds".to_string(),
                required: false,
                parameter_type: "integer".to_string(),
            },
        ]
    }

    async fn execute(&self, arguments: serde_json::Value) -> AgentResult<serde_json::Value> {
        let args: BashArgs = serde_json::from_value(arguments)?;
        self.execute_impl(args).await
    }
}

/// Tool for executing bash commands with streaming output.
///
/// Does not check a `SecurityPolicy` yet.
pub struct StreamingBashTool {
    inner: BashTool,
}

impl StreamingBashTool {
    pub fn new() -> Self {
        Self {
            inner: BashTool::new(),
        }
    }

    pub async fn execute_streaming<F>(
        &self,
        command: &str,
        timeout_ms: Option<u64>,
        mut on_output: F,
    ) -> AgentResult<i32>
    where
        F: FnMut(StreamOutput) + Send,
    {
        let timeout_duration = Duration::from_millis(
            timeout_ms.unwrap_or(self.inner.default_timeout.as_millis() as u64),
        );

        let mut cmd = Command::new(&self.inner.shell);
        cmd.arg("-c").arg(command);

        if let Some(ref dir) = self.inner.working_dir {
            cmd.current_dir(dir);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| AgentError::CommandFailed(format!("Failed to spawn command: {}", e)))?;

        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        let read_task = async {
            loop {
                tokio::select! {
                    result = stdout_reader.next_line() => {
                        match result {
                            Ok(Some(line)) => on_output(StreamOutput::Stdout(line)),
                            Ok(None) => break,
                            Err(e) => {
                                warn!("Error reading stdout: {}", e);
                                break;
                            }
                        }
                    }
                    result = stderr_reader.next_line() => {
                        match result {
                            Ok(Some(line)) => on_output(StreamOutput::Stderr(line)),
                            Ok(None) => {}
                            Err(e) => {
                                warn!("Error reading stderr: {}", e);
                            }
                        }
                    }
                }
            }
        };

        let (_, status) = match timeout(timeout_duration, async {
            tokio::join!(read_task, child.wait())
        })
        .await
        {
            Ok(done) => done,
            Err(_) => {
                let _ = child.kill().await;
                return Err(AgentError::CommandTimeout(timeout_duration.as_secs()));
            }
        };

        let status = status
            .map_err(|e| AgentError::CommandFailed(format!("Failed to wait for command: {}", e)))?;

        Ok(status.code().unwrap_or(-1))
    }
}

impl Default for StreamingBashTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Output from streaming bash execution
#[derive(Debug, Clone)]
pub enum StreamOutput {
    Stdout(String),
    Stderr(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A policy that allows only the given first words of a command.
    fn allow_commands(commands: &[&str]) -> Arc<SecurityPolicy> {
        Arc::new(
            SecurityPolicy::builder()
                .with_allowed_commands(commands.iter().map(|c| c.to_string()).collect())
                .build(),
        )
    }

    #[tokio::test]
    async fn test_bash_echo() {
        let tool = BashTool::new().with_policy(allow_commands(&["echo", "exit"]));
        let result = tool
            .execute(serde_json::json!({
                "command": "echo 'Hello, World!'"
            }))
            .await
            .unwrap();

        assert!(result["success"].as_bool().unwrap());
        assert!(result["stdout"].as_str().unwrap().contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_bash_exit_code() {
        let tool = BashTool::new().with_policy(allow_commands(&["echo", "exit"]));
        let result = tool
            .execute(serde_json::json!({
                "command": "exit 42"
            }))
            .await
            .unwrap();

        assert_eq!(result["exit_code"], 42);
        assert!(!result["success"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_streaming_bash() {
        let tool = StreamingBashTool::new();
        let mut lines = Vec::new();

        let exit_code = tool
            .execute_streaming(
                "echo 'line1' && echo 'line2' && echo 'line3'",
                Some(5000),
                |output| {
                    if let StreamOutput::Stdout(line) = output {
                        lines.push(line);
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(exit_code, 0);
        assert_eq!(lines.len(), 3);
    }

    /// Wait up to 2 s for `pid` to disappear. Returns true when it is gone.
    #[cfg(unix)]
    async fn pid_gone_within_2s(pid: &str) -> bool {
        for _ in 0..20 {
            let alive = std::process::Command::new("kill")
                .args(["-0", pid])
                .status()
                .unwrap()
                .success();
            if !alive {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        false
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_the_child() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("pid");
        let command = format!("echo $$ > {}; sleep 30", pid_file.display());

        let started = std::time::Instant::now();
        let err = BashTool::new()
            .with_policy(allow_commands(&["echo"]))
            .execute(serde_json::json!({ "command": command, "timeout": 1000 }))
            .await
            .unwrap_err();

        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timeout took {:?}",
            started.elapsed()
        );
        assert!(
            matches!(err, AgentError::CommandTimeout(_)),
            "expected CommandTimeout, got {err:?}"
        );
        let pid = std::fs::read_to_string(&pid_file).unwrap();
        assert!(
            pid_gone_within_2s(pid.trim()).await,
            "child {} is still running after the timeout",
            pid.trim()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn streaming_timeout_kills_the_child() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("pid");
        let command = format!("echo $$ > {}; sleep 30", pid_file.display());

        let started = std::time::Instant::now();
        let err = StreamingBashTool::new()
            .execute_streaming(&command, Some(500), |_| {})
            .await
            .unwrap_err();

        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timeout took {:?}",
            started.elapsed()
        );
        assert!(
            matches!(err, AgentError::CommandTimeout(_)),
            "expected CommandTimeout, got {err:?}"
        );
        let pid = std::fs::read_to_string(&pid_file).unwrap();
        assert!(
            pid_gone_within_2s(pid.trim()).await,
            "child {} is still running after the timeout",
            pid.trim()
        );
    }

    /// The stderr end of the read loop must be read while stdout is still
    /// open. The child closes stderr at once, prints one line to stdout and
    /// keeps stdout open for 0.3 s. Without this test, the stderr arm and its
    /// end-of-stream arm (lines 247 to 249) were hit only by chance: `select!`
    /// picks at random between two ready branches, and no other test writes to
    /// stderr. The coverage of those lines then changed between runs.
    #[cfg(unix)]
    #[tokio::test]
    async fn stderr_end_is_read_while_stdout_is_open() {
        let mut lines = Vec::new();
        let exit_code = StreamingBashTool::new()
            .execute_streaming("exec 2>&-; echo out; sleep 0.3", Some(5000), |output| {
                lines.push(output);
            })
            .await
            .unwrap();

        assert_eq!(exit_code, 0);
        assert_eq!(lines.len(), 1, "got {lines:?}");
        assert!(
            matches!(&lines[0], StreamOutput::Stdout(line) if line == "out"),
            "got {lines:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn output_cut_inside_a_multibyte_char_does_not_panic() {
        // 10 000 two-byte characters = 20 000 bytes, cut at 101 bytes, which
        // falls inside a character.
        let tool = BashTool::new()
            .with_policy(allow_commands(&["yes"]))
            .with_max_output_size(101);
        let result = tool
            .execute(serde_json::json!({
                "command": "yes é | head -n 10000 | tr -d '\\n'"
            }))
            .await
            .unwrap();

        assert_eq!(result["stdout_truncated"], true);
        assert!(result["stdout"]
            .as_str()
            .unwrap()
            .starts_with(&"é".repeat(50)));
    }

    #[tokio::test]
    async fn bash_default_policy_refuses() {
        let result = BashTool::new()
            .execute(serde_json::json!({ "command": "echo hi" }))
            .await;

        assert!(matches!(result, Err(AgentError::CommandNotAllowed(_))));
    }

    #[tokio::test]
    async fn bash_blocked_pattern_refused_even_when_listed() {
        let result = BashTool::new()
            .with_policy(allow_commands(&["rm"]))
            .execute(serde_json::json!({ "command": "rm -rf /" }))
            .await;

        assert!(matches!(result, Err(AgentError::CommandNotAllowed(_))));
    }
}
