use std::collections::HashMap;
use std::time::Duration;

use kit_http::client::HttpError;
use kit_http::http::{Auth, HttpMethod, MultipartField, Request, RequestBody};

fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn request_new_has_empty_defaults() {
    let req = Request::new(HttpMethod::Get, "/items");
    assert_eq!(req.method, HttpMethod::Get);
    assert_eq!(req.url, "/items");
    assert!(req.headers.is_empty());
    assert!(req.query_params.is_empty());
    assert!(req.cookies.is_empty());
    assert!(req.path_params.is_empty());
    assert!(matches!(req.body, RequestBody::None));
    assert!(matches!(req.auth, Auth::None));
    assert_eq!(req.timeout, None);
}

#[test]
fn request_get_sets_method() {
    assert_eq!(Request::get("/a").method, HttpMethod::Get);
}

#[test]
fn request_post_sets_method() {
    assert_eq!(Request::post("/a").method, HttpMethod::Post);
}

#[test]
fn request_put_sets_method() {
    assert_eq!(Request::put("/a").method, HttpMethod::Put);
}

#[test]
fn request_patch_sets_method() {
    assert_eq!(Request::patch("/a").method, HttpMethod::Patch);
}

#[test]
fn request_delete_sets_method() {
    assert_eq!(Request::delete("/a").method, HttpMethod::Delete);
}

#[test]
fn request_head_sets_method() {
    assert_eq!(Request::head("/a").method, HttpMethod::Head);
}

#[test]
fn request_options_sets_method() {
    assert_eq!(Request::options("/a").method, HttpMethod::Options);
}

#[test]
fn request_header_and_headers_merge() {
    let req = Request::get("/a")
        .header("A", "1")
        .headers(map(&[("B", "2"), ("A", "3")]));
    assert_eq!(req.headers.get("A").map(String::as_str), Some("3"));
    assert_eq!(req.headers.get("B").map(String::as_str), Some("2"));
}

#[test]
fn request_query_and_query_params_merge() {
    let req = Request::get("/a")
        .query("page", "1")
        .query_params(map(&[("size", "20"), ("page", "2")]));
    assert_eq!(req.query_params.get("page").map(String::as_str), Some("2"));
    assert_eq!(req.query_params.get("size").map(String::as_str), Some("20"));
}

#[test]
fn request_cookie_pushes_in_order() {
    let req = Request::get("/a")
        .cookie(kit_http::http::Cookie::new("a", "1"))
        .cookie(kit_http::http::Cookie::new("b", "2"));
    let names: Vec<&str> = req.cookies.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn request_json_sets_body_and_content_type() {
    let req = Request::post("/a")
        .json(&serde_json::json!({"k": 1}))
        .unwrap();
    assert!(matches!(req.body, RequestBody::Json(ref v) if *v == serde_json::json!({"k": 1})));
    assert_eq!(
        req.headers.get("Content-Type").map(String::as_str),
        Some("application/json")
    );
}

#[test]
fn request_json_serialize_error() {
    // A map with a non-string key cannot be encoded as a JSON object.
    let mut bad: HashMap<Vec<u8>, u8> = HashMap::new();
    bad.insert(vec![1], 1);

    match Request::post("/a").json(&bad) {
        Err(HttpError::Json(msg)) => {
            assert!(
                msg.starts_with("Failed to serialize JSON"),
                "message was: {msg}"
            );
        }
        Err(other) => panic!("expected Json error, got {other:?}"),
        Ok(_) => panic!("expected a serialization error"),
    }
}

#[test]
fn request_json_value_sets_body_and_content_type() {
    let req = Request::post("/a").json_value(serde_json::json!([1, 2]));
    assert!(matches!(req.body, RequestBody::Json(ref v) if *v == serde_json::json!([1, 2])));
    assert_eq!(
        req.headers.get("Content-Type").map(String::as_str),
        Some("application/json")
    );
}

#[test]
fn request_form_sets_body_and_content_type() {
    let req = Request::post("/a").form(map(&[("name", "x")]));
    assert!(matches!(
        req.body,
        RequestBody::Form(ref m) if m.get("name").map(String::as_str) == Some("x")
    ));
    assert_eq!(
        req.headers.get("Content-Type").map(String::as_str),
        Some("application/x-www-form-urlencoded")
    );
}

#[test]
fn request_text_sets_body_and_content_type() {
    let req = Request::post("/a").text("hi");
    assert!(matches!(req.body, RequestBody::Text(ref t) if t == "hi"));
    assert_eq!(
        req.headers.get("Content-Type").map(String::as_str),
        Some("text/plain")
    );
}

#[test]
fn request_bytes_sets_body_without_content_type() {
    let req = Request::post("/a").bytes(vec![1, 2]);
    assert!(matches!(req.body, RequestBody::Bytes(ref b) if *b == vec![1, 2]));
    assert!(!req.headers.contains_key("Content-Type"));
}

#[test]
fn request_multipart_sets_body_with_fields() {
    let field = MultipartField {
        name: "file".into(),
        filename: Some("a.txt".into()),
        mime_type: Some("text/plain".into()),
        data: b"hi".to_vec(),
    };
    let req = Request::post("/a").multipart(vec![field]);
    assert!(matches!(
        req.body,
        RequestBody::Multipart(ref fields)
            if fields.len() == 1 && fields[0].name == "file" && fields[0].data == b"hi"
    ));
}

#[test]
fn request_auth_basic_and_bearer() {
    let basic = Request::get("/a").basic_auth("u", "p");
    assert!(matches!(
        basic.auth,
        Auth::Basic { ref username, ref password } if username == "u" && password == "p"
    ));

    let bearer = Request::get("/a").bearer_auth("t");
    assert!(matches!(bearer.auth, Auth::Bearer(ref t) if t == "t"));
}

#[test]
fn request_timeout_and_timeout_secs() {
    let by_duration = Request::get("/a").timeout(Duration::from_secs(2));
    assert_eq!(by_duration.timeout, Some(Duration::from_secs(2)));

    let by_secs = Request::get("/a").timeout_secs(1.5);
    assert_eq!(by_secs.timeout, Some(Duration::from_millis(1500)));
}
