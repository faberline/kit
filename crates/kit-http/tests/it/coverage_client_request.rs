use std::collections::HashMap;
use std::time::Duration;

use kit_http::client::request::{ExtractedAuth, ExtractedBody, ExtractedRequest};
use kit_http::client::{HttpError, HttpMethod};
use kit_http::http::{Cookie, MultipartField, Request};

fn extracted(url: &str) -> ExtractedRequest {
    ExtractedRequest {
        method: HttpMethod::Get,
        url: url.to_string(),
        headers: Vec::new(),
        query_params: Vec::new(),
        body: ExtractedBody::None,
        auth: ExtractedAuth::None,
        timeout_ms: None,
    }
}

fn header(req: &reqwest::Request, name: &str) -> Option<String> {
    req.headers()
        .get(name)
        .map(|v| v.to_str().unwrap().to_string())
}

#[test]
fn from_request_joins_cookies_into_one_cookie_header() {
    let req = Request::get("/p")
        .cookie(Cookie::new("a", "1"))
        .cookie(Cookie::new("b", "2"));
    let ex: ExtractedRequest = req.into();

    assert!(ex
        .headers
        .contains(&("cookie".to_string(), "a=1; b=2".to_string())));
}

#[test]
fn from_request_without_cookies_adds_no_cookie_header() {
    let ex: ExtractedRequest = Request::get("/p").into();
    assert!(ex.headers.iter().all(|(name, _)| name != "cookie"));
}

#[test]
fn from_request_body_none_maps_to_none() {
    let ex: ExtractedRequest = Request::get("/p").into();
    assert!(matches!(ex.body, ExtractedBody::None));
}

#[test]
fn from_request_body_json_maps_to_json() {
    let ex: ExtractedRequest = Request::post("/p")
        .json_value(serde_json::json!({"k": 1}))
        .into();
    assert!(matches!(ex.body, ExtractedBody::Json(ref v) if v["k"] == 1));
}

#[test]
fn from_request_body_form_maps_to_form() {
    let mut data = HashMap::new();
    data.insert("name".to_string(), "value".to_string());
    let ex: ExtractedRequest = Request::post("/p").form(data).into();
    assert!(matches!(
        ex.body,
        ExtractedBody::Form(ref pairs)
            if pairs == &vec![("name".to_string(), "value".to_string())]
    ));
}

#[test]
fn from_request_body_multipart_maps_to_multipart() {
    let field = MultipartField {
        name: "file".into(),
        filename: Some("a.txt".into()),
        mime_type: Some("text/plain".into()),
        data: b"hi".to_vec(),
    };
    let ex: ExtractedRequest = Request::post("/p").multipart(vec![field]).into();
    assert!(matches!(ex.body, ExtractedBody::Multipart(ref f) if f.len() == 1));
}

#[test]
fn from_request_body_bytes_maps_to_bytes() {
    let ex: ExtractedRequest = Request::post("/p").bytes(vec![1, 2, 3]).into();
    assert!(matches!(ex.body, ExtractedBody::Bytes(ref b) if b == &vec![1, 2, 3]));
}

#[test]
fn from_request_body_text_maps_to_text() {
    let ex: ExtractedRequest = Request::post("/p").text("hello").into();
    assert!(matches!(ex.body, ExtractedBody::Text(ref t) if t == "hello"));
}

#[test]
fn from_request_auth_none_maps_to_none() {
    let ex: ExtractedRequest = Request::get("/p").into();
    assert!(matches!(ex.auth, ExtractedAuth::None));
}

#[test]
fn from_request_auth_basic_maps_to_basic() {
    let ex: ExtractedRequest = Request::get("/p").basic_auth("u", "p").into();
    assert!(matches!(
        ex.auth,
        ExtractedAuth::Basic { ref username, ref password } if username == "u" && password == "p"
    ));
}

#[test]
fn from_request_auth_bearer_maps_to_bearer() {
    let ex: ExtractedRequest = Request::get("/p").bearer_auth("tok").into();
    assert!(matches!(ex.auth, ExtractedAuth::Bearer(ref t) if t == "tok"));
}

#[test]
fn from_request_timeout_becomes_millis() {
    let ex: ExtractedRequest = Request::get("/p")
        .timeout(Duration::from_millis(1500))
        .into();
    assert_eq!(ex.timeout_ms, Some(1500));
}

#[test]
fn build_reqwest_joins_relative_url_to_base() {
    let client = reqwest::Client::new();
    let built = extracted("/p")
        .build_reqwest(&client, Some("http://127.0.0.1:1"))
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(built.url().as_str(), "http://127.0.0.1:1/p");
}

#[test]
fn build_reqwest_adds_slash_when_url_has_none() {
    let client = reqwest::Client::new();
    let built = extracted("p")
        .build_reqwest(&client, Some("http://127.0.0.1:1"))
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(built.url().as_str(), "http://127.0.0.1:1/p");
}

#[test]
fn build_reqwest_absolute_url_ignores_base() {
    let client = reqwest::Client::new();
    let built = extracted("https://127.0.0.2:9/x")
        .build_reqwest(&client, Some("http://127.0.0.1:1"))
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(built.url().as_str(), "https://127.0.0.2:9/x");

    let built = extracted("http://127.0.0.1:2/y")
        .build_reqwest(&client, Some("http://127.0.0.1:1"))
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(built.url().as_str(), "http://127.0.0.1:2/y");
}

#[test]
fn build_reqwest_trims_trailing_slash_of_base() {
    let client = reqwest::Client::new();
    let built = extracted("/p")
        .build_reqwest(&client, Some("http://127.0.0.1:1/"))
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(built.url().as_str(), "http://127.0.0.1:1/p");
}

#[test]
fn build_reqwest_without_base_keeps_url() {
    let client = reqwest::Client::new();
    let built = extracted("http://127.0.0.1:1/q")
        .build_reqwest(&client, None)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(built.url().as_str(), "http://127.0.0.1:1/q");
}

#[test]
fn build_reqwest_method_get() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.method = HttpMethod::Get;
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.method(), reqwest::Method::GET);
}

#[test]
fn build_reqwest_method_post() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.method = HttpMethod::Post;
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.method(), reqwest::Method::POST);
}

#[test]
fn build_reqwest_method_put() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.method = HttpMethod::Put;
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.method(), reqwest::Method::PUT);
}

#[test]
fn build_reqwest_method_patch() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.method = HttpMethod::Patch;
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.method(), reqwest::Method::PATCH);
}

#[test]
fn build_reqwest_method_delete() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.method = HttpMethod::Delete;
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.method(), reqwest::Method::DELETE);
}

#[test]
fn build_reqwest_method_head() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.method = HttpMethod::Head;
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.method(), reqwest::Method::HEAD);
}

#[test]
fn build_reqwest_method_options() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.method = HttpMethod::Options;
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.method(), reqwest::Method::OPTIONS);
}

#[test]
fn build_reqwest_query_params_appear_in_url() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/s");
    ex.query_params = vec![("a".into(), "1".into())];
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.url().as_str(), "http://127.0.0.1:1/s?a=1");
}

#[test]
fn build_reqwest_json_sets_content_type() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.body = ExtractedBody::Json(serde_json::json!({"k": "v"}));
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(
        header(&built, "content-type").as_deref(),
        Some("application/json")
    );
}

#[test]
fn build_reqwest_form_sets_content_type() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.body = ExtractedBody::Form(vec![("a".into(), "1".into())]);
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(
        header(&built, "content-type").as_deref(),
        Some("application/x-www-form-urlencoded")
    );
}

#[test]
fn build_reqwest_multipart_sets_multipart_content_type() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.body = ExtractedBody::Multipart(vec![MultipartField {
        name: "file".into(),
        filename: Some("a.txt".into()),
        mime_type: Some("text/plain".into()),
        data: b"hi".to_vec(),
    }]);
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert!(header(&built, "content-type")
        .unwrap()
        .starts_with("multipart/form-data"));
}

#[test]
fn build_reqwest_multipart_without_filename_or_mime_is_built() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.body = ExtractedBody::Multipart(vec![MultipartField {
        name: "field".into(),
        filename: None,
        mime_type: None,
        data: b"x".to_vec(),
    }]);
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert!(header(&built, "content-type")
        .unwrap()
        .starts_with("multipart/form-data"));
}

#[test]
fn build_reqwest_bytes_body_is_present() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.body = ExtractedBody::Bytes(vec![9, 8, 7]);
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.body().unwrap().as_bytes().unwrap(), &[9, 8, 7]);
}

#[test]
fn build_reqwest_text_body_is_present() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.body = ExtractedBody::Text("plain".into());
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.body().unwrap().as_bytes().unwrap(), b"plain");
}

#[test]
fn build_reqwest_basic_auth_sets_authorization() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.auth = ExtractedAuth::Basic {
        username: "user".into(),
        password: "pass".into(),
    };
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(
        header(&built, "authorization").as_deref(),
        Some("Basic dXNlcjpwYXNz")
    );
}

#[test]
fn build_reqwest_bearer_auth_sets_authorization() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.auth = ExtractedAuth::Bearer("tok".into());
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(
        header(&built, "authorization").as_deref(),
        Some("Bearer tok")
    );
}

#[test]
fn build_reqwest_timeout_ms_sets_timeout() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.timeout_ms = Some(250);
    let built = ex.build_reqwest(&client, None).unwrap().build().unwrap();
    assert_eq!(built.timeout(), Some(&Duration::from_millis(250)));
}

#[test]
fn build_reqwest_invalid_multipart_mime_is_error() {
    let client = reqwest::Client::new();
    let mut ex = extracted("http://127.0.0.1:1/");
    ex.body = ExtractedBody::Multipart(vec![MultipartField {
        name: "file".into(),
        filename: None,
        mime_type: Some("not a mime".into()),
        data: b"x".to_vec(),
    }]);

    match ex.build_reqwest(&client, None) {
        Err(HttpError::InvalidRequest(msg)) => {
            assert!(msg.starts_with("invalid MIME type"), "message was: {msg}");
        }
        Err(other) => panic!("expected InvalidRequest, got {other:?}"),
        Ok(_) => panic!("expected an error for an invalid MIME type"),
    }
}
