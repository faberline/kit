use std::collections::HashMap;
use std::time::Duration;

use kit_http::client::HttpError;
use kit_http::http::{HttpResponseLike, HttpStatus, Response};

fn with_body(status: u16, body: &[u8]) -> Response {
    let mut resp = Response::new(status);
    resp.body = body.to_vec();
    resp
}

fn with_header(name: &str, value: &str) -> Response {
    let mut resp = Response::new(200);
    resp.headers.insert(name.to_string(), value.to_string());
    resp
}

#[test]
fn response_new_defaults() {
    let resp = Response::new(201);
    assert_eq!(resp.status_code, 201);
    assert!(resp.headers.is_empty());
    assert!(resp.body.is_empty());
    assert!(resp.cookies.is_empty());
    assert_eq!(resp.media_type, "");
    assert_eq!(resp.latency_ms, 0);
    assert_eq!(resp.final_url, "");
    assert_eq!(resp.version, "");
}

#[test]
fn response_status_wraps_code() {
    assert_eq!(Response::new(404).status(), HttpStatus(404));
}

#[test]
fn response_status_class_predicates() {
    // (code, is_success, is_redirect, is_client_error, is_server_error)
    let table: [(u16, [bool; 4]); 9] = [
        (199, [false, false, false, false]),
        (200, [true, false, false, false]),
        (299, [true, false, false, false]),
        (300, [false, true, false, false]),
        (399, [false, true, false, false]),
        (400, [false, false, true, false]),
        (499, [false, false, true, false]),
        (500, [false, false, false, true]),
        (599, [false, false, false, true]),
    ];

    for (code, [success, redirect, client, server]) in table {
        let resp = Response::new(code);
        assert_eq!(resp.is_success(), success, "is_success for {code}");
        assert_eq!(resp.is_redirect(), redirect, "is_redirect for {code}");
        assert_eq!(resp.is_client_error(), client, "is_client_error for {code}");
        assert_eq!(resp.is_server_error(), server, "is_server_error for {code}");
    }
}

#[test]
fn response_text_ok_for_utf8_body() {
    assert_eq!(with_body(200, b"hi").text().unwrap(), "hi");
}

#[test]
fn response_text_invalid_utf8_is_error() {
    match with_body(200, &[0xff, 0xfe]).text() {
        Err(HttpError::ResponseError(msg)) => {
            assert!(
                msg.starts_with("Invalid UTF-8 in response"),
                "message was: {msg}"
            );
        }
        other => panic!("expected ResponseError, got {other:?}"),
    }
}

#[test]
fn response_json_ok_for_valid_body() {
    let value = with_body(200, br#"{"a":1}"#).json().unwrap();
    assert_eq!(value, serde_json::json!({"a": 1}));
}

#[test]
fn response_json_invalid_body_is_error() {
    match with_body(200, b"not json").json() {
        Err(HttpError::Json(msg)) => {
            assert!(
                msg.starts_with("Failed to parse JSON"),
                "message was: {msg}"
            );
        }
        other => panic!("expected Json error, got {other:?}"),
    }
}

#[test]
fn response_json_as_ok_for_matching_type() {
    let map: HashMap<String, i64> = with_body(200, br#"{"a":1}"#).json_as().unwrap();
    assert_eq!(map.get("a"), Some(&1));
}

#[test]
fn response_json_as_wrong_shape_is_error() {
    let result: Result<HashMap<String, i64>, HttpError> = with_body(200, b"[1]").json_as();
    match result {
        Err(HttpError::Json(msg)) => {
            assert!(
                msg.starts_with("Failed to deserialize JSON"),
                "message was: {msg}"
            );
        }
        other => panic!("expected Json error, got {other:?}"),
    }
}

#[test]
fn response_bytes_and_content_length() {
    let resp = with_body(200, &[1, 2, 3]);
    assert_eq!(resp.bytes(), &[1, 2, 3]);
    assert_eq!(resp.content_length(), 3);
}

#[test]
fn response_latency_is_millis() {
    let mut resp = Response::new(200);
    resp.latency_ms = 250;
    assert_eq!(resp.latency(), Duration::from_millis(250));
}

#[test]
fn response_protocol_http2_is_detected() {
    let mut resp = Response::new(200);
    resp.version = "HTTP/2".to_string();
    assert!(resp.is_http2());
    assert_eq!(resp.protocol_version(), "HTTP/2");
}

#[test]
fn response_protocol_http11_is_not_http2() {
    let mut resp = Response::new(200);
    resp.version = "HTTP/1.1".to_string();
    assert!(!resp.is_http2());
    assert_eq!(resp.protocol_version(), "HTTP/1.1");
}

#[test]
fn response_header_is_case_insensitive() {
    let resp = with_header("Content-Type", "application/json");
    assert_eq!(resp.header("content-type"), Some("application/json"));
    assert_eq!(resp.header("CONTENT-TYPE"), Some("application/json"));
}

#[test]
fn response_header_missing_is_none() {
    assert_eq!(with_header("X-A", "1").header("X-B"), None);
}

#[test]
fn response_content_type_reads_header() {
    assert_eq!(
        with_header("Content-Type", "text/plain").content_type(),
        Some("text/plain")
    );
    assert_eq!(Response::new(200).content_type(), None);
}

#[test]
fn response_is_json_true_for_json_with_charset() {
    let resp = with_header("Content-Type", "application/json; charset=utf-8");
    assert!(resp.is_json());
}

#[test]
fn response_is_json_false_for_text() {
    let resp = with_header("Content-Type", "text/plain");
    assert!(!resp.is_json());
}

#[test]
fn response_is_json_false_without_header() {
    assert!(!Response::new(200).is_json());
}

#[test]
fn response_like_trait_accessors_return_fields() {
    let mut resp = with_body(202, b"body");
    resp.headers.insert("X-A".into(), "1".into());

    assert_eq!(HttpResponseLike::status_code(&resp), 202);
    assert_eq!(
        HttpResponseLike::headers(&resp)
            .get("X-A")
            .map(String::as_str),
        Some("1")
    );
    assert_eq!(HttpResponseLike::body_bytes(&resp), b"body");
}
