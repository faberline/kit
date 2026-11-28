use kit_http::http::Cookie;

#[test]
fn cookie_new_has_defaults() {
    let cookie = Cookie::new("a", "b");
    assert_eq!(cookie.name, "a");
    assert_eq!(cookie.value, "b");
    assert_eq!(cookie.path, None);
    assert_eq!(cookie.domain, None);
    assert!(!cookie.secure);
    assert!(!cookie.http_only);
    assert_eq!(cookie.max_age, None);
}

#[test]
fn cookie_builder_sets_every_attribute() {
    let cookie = Cookie::new("sid", "v")
        .path("/x")
        .domain("example.test")
        .secure(true)
        .http_only(true)
        .max_age(60);

    assert_eq!(cookie.path.as_deref(), Some("/x"));
    assert_eq!(cookie.domain.as_deref(), Some("example.test"));
    assert!(cookie.secure);
    assert!(cookie.http_only);
    assert_eq!(cookie.max_age, Some(60));
}

#[test]
fn cookie_builder_flags_can_be_cleared() {
    let cookie = Cookie::new("sid", "v").secure(true).http_only(true);
    let cleared = cookie.secure(false).http_only(false);
    assert!(!cleared.secure);
    assert!(!cleared.http_only);
}

#[test]
fn cookie_serde_roundtrip_is_equal() {
    let cookie = Cookie::new("sid", "v")
        .path("/")
        .domain("example.test")
        .secure(true)
        .http_only(true)
        .max_age(-1);

    let text = serde_json::to_string(&cookie).unwrap();
    let back: Cookie = serde_json::from_str(&text).unwrap();
    assert_eq!(back, cookie);
    assert!(text.contains(r#""max_age":-1"#));
}

#[test]
fn cookie_deserialize_applies_defaults() {
    let cookie: Cookie = serde_json::from_str(r#"{"name":"n","value":"v"}"#).unwrap();
    assert_eq!(cookie, Cookie::new("n", "v"));
}
