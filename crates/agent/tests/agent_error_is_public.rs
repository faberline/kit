//! The public error type of the `agent` crate is `AgentError` and its result
//! alias is `AgentResult`, both re-exported at the crate root.

#[test]
fn agent_error_is_public() {
    let e: agent::AgentError = agent::AgentError::ToolError("x".to_string());
    assert_eq!(e.to_string(), "Tool execution failed: x");
    let r: agent::AgentResult<u32> = Ok(1);
    assert!(matches!(r, Ok(1)));
}
