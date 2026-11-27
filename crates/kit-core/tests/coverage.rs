//! Tests that cover the default trait methods and the uncovered branches of kit-core.

use kit_core::{
    categorize_error, sanitize_error, sanitize_error_message, ErrorCategory, HttpResponseLike,
};
use std::collections::HashMap;

struct FakeResponse {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl HttpResponseLike for FakeResponse {
    fn status_code(&self) -> u16 {
        self.status
    }

    fn headers(&self) -> &HashMap<String, String> {
        &self.headers
    }

    fn body_bytes(&self) -> &[u8] {
        &self.body
    }
}

fn fake(status: u16) -> FakeResponse {
    FakeResponse {
        status,
        headers: HashMap::new(),
        body: Vec::new(),
    }
}

#[test]
fn is_client_error_default_method() {
    assert!(fake(404).is_client_error());
    assert!(!fake(503).is_client_error());
    assert!(!fake(200).is_client_error());
}

#[test]
fn is_server_error_default_method() {
    assert!(fake(503).is_server_error());
    assert!(!fake(404).is_server_error());
    assert!(!fake(200).is_server_error());
}

#[test]
fn sanitize_error_message_redacts_like_production_mode() {
    let input = "login failed: bad password";
    assert_eq!(
        sanitize_error_message(input),
        "login failed: bad [PASSWORD_REDACTED]"
    );
    assert_eq!(sanitize_error_message(input), sanitize_error(input, false));
}

#[test]
fn sanitize_error_message_keeps_plain_text() {
    assert_eq!(sanitize_error_message("plain failure"), "plain failure");
}

#[test]
fn categorize_error_query_is_operation() {
    assert_eq!(categorize_error("query failed"), ErrorCategory::Operation);
    assert_eq!(
        categorize_error("operation failed"),
        ErrorCategory::Operation
    );
}
