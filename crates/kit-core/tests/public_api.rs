//! Public API stability checks for kit-core.
//!
//! These tests use only the crate root paths. A removed or changed public
//! item makes this file fail to compile, or an assertion fail.

use kit_core::{
    categorize_error, sanitize_error, sanitize_error_message, ErrorCategory, HttpMethod,
    HttpRequestLike, HttpResponseLike, HttpStatus, KitError, Result,
};
use std::collections::HashMap;
use std::str::FromStr;

/// Exhaustive match with no wildcard arm: a new or removed variant is a compile error.
fn kit_error_label(e: &KitError) -> &'static str {
    match e {
        KitError::MongoDB(_) => "mongodb",
        KitError::Database(_) => "database",
        KitError::Serialization(_) => "serialization",
        KitError::Deserialization(_) => "deserialization",
        KitError::Connection(_) => "connection",
        KitError::Query(_) => "query",
        KitError::Validation(_) => "validation",
        KitError::Internal(_) => "internal",
        KitError::Conflict(_) => "conflict",
        KitError::ForeignKey(_) => "foreign_key",
        KitError::Deadlock(_) => "deadlock",
        KitError::Timeout(_) => "timeout",
        KitError::Transient(_) => "transient",
    }
}

/// Exhaustive match with no wildcard arm.
fn category_label(c: ErrorCategory) -> &'static str {
    match c {
        ErrorCategory::Connection => "connection",
        ErrorCategory::Authentication => "authentication",
        ErrorCategory::Timeout => "timeout",
        ErrorCategory::Validation => "validation",
        ErrorCategory::Operation => "operation",
        ErrorCategory::Unknown => "unknown",
    }
}

/// Exhaustive match with no wildcard arm.
fn method_label(m: HttpMethod) -> &'static str {
    match m {
        HttpMethod::Get => "get",
        HttpMethod::Post => "post",
        HttpMethod::Put => "put",
        HttpMethod::Patch => "patch",
        HttpMethod::Delete => "delete",
        HttpMethod::Head => "head",
        HttpMethod::Options => "options",
    }
}

struct Resp {
    code: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl HttpResponseLike for Resp {
    fn status_code(&self) -> u16 {
        self.code
    }
    fn headers(&self) -> &HashMap<String, String> {
        &self.headers
    }
    fn body_bytes(&self) -> &[u8] {
        &self.body
    }
}

struct Req {
    url: String,
    headers: HashMap<String, String>,
    body: Option<Vec<u8>>,
}

impl HttpRequestLike for Req {
    fn method(&self) -> HttpMethod {
        HttpMethod::Post
    }
    fn url(&self) -> &str {
        &self.url
    }
    fn headers(&self) -> &HashMap<String, String> {
        &self.headers
    }
    fn body_bytes(&self) -> Option<&[u8]> {
        self.body.as_deref()
    }
}

#[test]
fn kit_core_public_api_is_stable() {
    // Function signatures, checked by coercion to typed fn pointers.
    let _: fn(&str, bool) -> String = sanitize_error;
    let _: fn(&str) -> String = sanitize_error_message;
    let _: fn(&str) -> ErrorCategory = categorize_error;
    let _: fn(&str) -> bool = kit_core::utils::is_alphanumeric;
    let _: fn(&KitError) -> bool = KitError::is_retryable;
    let _: fn(&KitError) -> bool = KitError::is_constraint_violation;
    let _: fn(&HttpMethod) -> &'static str = HttpMethod::as_str;
    let _: fn(&HttpStatus) -> u16 = HttpStatus::code;
    let _: fn(&HttpStatus) -> bool = HttpStatus::is_success;
    let _: fn(&HttpStatus) -> bool = HttpStatus::is_client_error;
    let _: fn(&HttpStatus) -> bool = HttpStatus::is_server_error;
    let _: fn(&HttpStatus) -> bool = HttpStatus::is_redirect;
    let _: fn(&HttpStatus) -> bool = HttpStatus::is_informational;

    // The Result alias.
    let _: Result<u8> = Ok(1);
    let _: std::result::Result<u8, KitError> = Result::<u8>::Ok(1);

    // Every variant of each public enum has its label.
    let errors = [
        (KitError::MongoDB("x".to_string()), "mongodb"),
        (KitError::Database("x".to_string()), "database"),
        (KitError::Serialization("x".to_string()), "serialization"),
        (
            KitError::Deserialization("x".to_string()),
            "deserialization",
        ),
        (KitError::Connection("x".to_string()), "connection"),
        (KitError::Query("x".to_string()), "query"),
        (KitError::Validation("x".to_string()), "validation"),
        (KitError::Internal("x".to_string()), "internal"),
        (KitError::Conflict("x".to_string()), "conflict"),
        (KitError::ForeignKey("x".to_string()), "foreign_key"),
        (KitError::Deadlock("x".to_string()), "deadlock"),
        (KitError::Timeout("x".to_string()), "timeout"),
        (KitError::Transient("x".to_string()), "transient"),
    ];
    for (err, label) in &errors {
        assert_eq!(kit_error_label(err), *label);
    }

    let categories = [
        (ErrorCategory::Connection, "connection"),
        (ErrorCategory::Authentication, "authentication"),
        (ErrorCategory::Timeout, "timeout"),
        (ErrorCategory::Validation, "validation"),
        (ErrorCategory::Operation, "operation"),
        (ErrorCategory::Unknown, "unknown"),
    ];
    for (category, label) in categories {
        assert_eq!(category_label(category), label);
    }

    let methods = [
        (HttpMethod::Get, "get"),
        (HttpMethod::Post, "post"),
        (HttpMethod::Put, "put"),
        (HttpMethod::Patch, "patch"),
        (HttpMethod::Delete, "delete"),
        (HttpMethod::Head, "head"),
        (HttpMethod::Options, "options"),
    ];
    for (method, label) in methods {
        assert_eq!(method_label(method), label);
    }

    // HttpStatus constants and conversions.
    assert_eq!(HttpStatus::OK.code(), 200);
    assert_eq!(HttpStatus::CREATED.code(), 201);
    assert_eq!(HttpStatus::NO_CONTENT.code(), 204);
    assert_eq!(HttpStatus::BAD_REQUEST.code(), 400);
    assert_eq!(HttpStatus::UNAUTHORIZED.code(), 401);
    assert_eq!(HttpStatus::FORBIDDEN.code(), 403);
    assert_eq!(HttpStatus::NOT_FOUND.code(), 404);
    assert_eq!(HttpStatus::METHOD_NOT_ALLOWED.code(), 405);
    assert_eq!(HttpStatus::CONFLICT.code(), 409);
    assert_eq!(HttpStatus::UNPROCESSABLE_ENTITY.code(), 422);
    assert_eq!(HttpStatus::INTERNAL_SERVER_ERROR.code(), 500);
    assert_eq!(HttpStatus::BAD_GATEWAY.code(), 502);
    assert_eq!(HttpStatus::SERVICE_UNAVAILABLE.code(), 503);
    assert_eq!(HttpStatus(204).0, 204);
    assert_eq!(HttpStatus::from(404u16), HttpStatus::NOT_FOUND);
    assert_eq!(u16::from(HttpStatus::OK), 200);

    // Behavior that is part of the promise.
    assert_eq!(HttpMethod::from_str("get"), Ok(HttpMethod::Get));
    assert_eq!(
        HttpMethod::from_str("x"),
        Err("Invalid HTTP method: x".to_string())
    );
    assert_eq!(HttpMethod::Post.to_string(), "POST");
    assert!(KitError::Deadlock("x".to_string()).is_retryable());
    assert!(!KitError::Query("x".to_string()).is_retryable());
    assert!(KitError::Conflict("x".to_string()).is_constraint_violation());
    assert_eq!(
        KitError::Validation("bad".to_string()).to_string(),
        "Validation error: bad"
    );

    // Trait shape: only the required methods are implemented; defaults must work.
    let mut resp_headers = HashMap::new();
    resp_headers.insert("content-type".to_string(), "text/plain".to_string());
    resp_headers.insert("content-length".to_string(), "5".to_string());
    let resp = Resp {
        code: 204,
        headers: resp_headers,
        body: b"hello".to_vec(),
    };
    assert!(resp.is_success());
    assert_eq!(resp.header("Content-Type"), Some("text/plain"));
    assert_eq!(resp.content_type(), Some("text/plain"));
    assert_eq!(resp.content_length(), Some(5));
    assert_eq!(resp.body_bytes(), b"hello");

    let mut req_headers = HashMap::new();
    req_headers.insert("content-type".to_string(), "application/json".to_string());
    let req = Req {
        url: "https://example.invalid/items".to_string(),
        headers: req_headers,
        body: None,
    };
    assert_eq!(req.method(), HttpMethod::Post);
    assert_eq!(req.url(), "https://example.invalid/items");
    assert_eq!(req.content_type(), Some("application/json"));
    assert_eq!(req.body_bytes(), None);
}

#[test]
fn docs_exist() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let read = |path: &str| {
        std::fs::read_to_string(format!("{root}/{path}"))
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
    };
    assert!(read("docs/stability.md").contains("# Stability promise"));
    assert!(read("docs/limits.md").contains("# Limits"));
    assert!(read("CHANGELOG.md").contains("## Unreleased"));
    assert!(read("README.md").contains("docs/stability.md"));
}
