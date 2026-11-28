use std::collections::HashMap;

use kit_http::http_exception::{status_phrase, HTTPException};

#[test]
fn status_phrase_table() {
    let table: &[(u16, &str)] = &[
        (100, "Continue"),
        (101, "Switching Protocols"),
        (102, "Processing"),
        (103, "Early Hints"),
        (200, "OK"),
        (201, "Created"),
        (202, "Accepted"),
        (203, "Non-Authoritative Information"),
        (204, "No Content"),
        (205, "Reset Content"),
        (206, "Partial Content"),
        (207, "Multi-Status"),
        (208, "Already Reported"),
        (226, "IM Used"),
        (300, "Multiple Choices"),
        (301, "Moved Permanently"),
        (302, "Found"),
        (303, "See Other"),
        (304, "Not Modified"),
        (305, "Use Proxy"),
        (307, "Temporary Redirect"),
        (308, "Permanent Redirect"),
        (400, "Bad Request"),
        (401, "Unauthorized"),
        (402, "Payment Required"),
        (403, "Forbidden"),
        (404, "Not Found"),
        (405, "Method Not Allowed"),
        (406, "Not Acceptable"),
        (407, "Proxy Authentication Required"),
        (408, "Request Timeout"),
        (409, "Conflict"),
        (410, "Gone"),
        (411, "Length Required"),
        (412, "Precondition Failed"),
        (413, "Content Too Large"),
        (414, "URI Too Long"),
        (415, "Unsupported Media Type"),
        (416, "Range Not Satisfiable"),
        (417, "Expectation Failed"),
        (418, "I'm a Teapot"),
        (421, "Misdirected Request"),
        (422, "Unprocessable Entity"),
        (423, "Locked"),
        (424, "Failed Dependency"),
        (425, "Too Early"),
        (426, "Upgrade Required"),
        (428, "Precondition Required"),
        (429, "Too Many Requests"),
        (431, "Request Header Fields Too Large"),
        (451, "Unavailable For Legal Reasons"),
        (500, "Internal Server Error"),
        (501, "Not Implemented"),
        (502, "Bad Gateway"),
        (503, "Service Unavailable"),
        (504, "Gateway Timeout"),
        (505, "HTTP Version Not Supported"),
        (506, "Variant Also Negotiates"),
        (507, "Insufficient Storage"),
        (508, "Loop Detected"),
        (510, "Not Extended"),
        (511, "Network Authentication Required"),
    ];

    for (code, phrase) in table {
        assert_eq!(status_phrase(*code), *phrase, "phrase for {code}");
    }
}

#[test]
fn status_phrase_unlisted_codes_are_unknown() {
    assert_eq!(status_phrase(299), "Unknown");
    assert_eq!(status_phrase(600), "Unknown");
}

#[test]
fn exception_new_keeps_headers() {
    let headers: HashMap<String, String> = HashMap::from([("X".to_string(), "1".to_string())]);
    let exc = HTTPException::new(404, Some("x".into()), headers.clone()).unwrap();
    assert_eq!(exc.status_code, 404);
    assert_eq!(exc.detail, "x");
    assert_eq!(exc.headers, Some(headers));
}

#[test]
fn exception_new_detail_defaults_to_status_phrase() {
    let exc = HTTPException::new(404, None, HashMap::new()).unwrap();
    assert_eq!(exc.detail, "Not Found");
}

#[test]
fn exception_boundaries() {
    assert!(HTTPException::new(100, None, HashMap::new()).is_ok());
    assert!(HTTPException::new(599, None, HashMap::new()).is_ok());
    assert_eq!(
        HTTPException::new(99, None, HashMap::new()),
        Err("status_code must be in [100, 599]".to_string())
    );
    assert_eq!(
        HTTPException::new(600, None, HashMap::new()),
        Err("status_code must be in [100, 599]".to_string())
    );
}

#[test]
fn exception_serde_roundtrip() {
    let exc = HTTPException::new(409, Some("taken".into()), HashMap::new()).unwrap();
    let text = serde_json::to_string(&exc).unwrap();
    let back: HTTPException = serde_json::from_str(&text).unwrap();
    assert_eq!(back, exc);
}

#[test]
fn exception_deserialize_defaults_headers_to_none() {
    let exc: HTTPException = serde_json::from_str(r#"{"status_code":404,"detail":"x"}"#).unwrap();
    assert_eq!(exc.status_code, 404);
    assert_eq!(exc.detail, "x");
    assert_eq!(exc.headers, None);
}
