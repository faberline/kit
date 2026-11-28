use std::time::Duration;

use kit_http::client::config::{
    BackoffStrategy, HttpClientConfig, HttpProtocolPolicy, RetryConfig,
};

#[test]
fn config_builder_base_url_sets_field() {
    let cfg = HttpClientConfig::new().base_url("http://127.0.0.1:9");
    assert_eq!(cfg.base_url.as_deref(), Some("http://127.0.0.1:9"));
}

#[test]
fn config_builder_timeout_sets_field() {
    let cfg = HttpClientConfig::new().timeout(Duration::from_secs(7));
    assert_eq!(cfg.timeout, Duration::from_secs(7));
}

#[test]
fn config_builder_timeout_secs_sets_field() {
    let cfg = HttpClientConfig::new().timeout_secs(1.5);
    assert_eq!(cfg.timeout, Duration::from_millis(1500));
}

#[test]
fn config_builder_connect_timeout_sets_field() {
    let cfg = HttpClientConfig::new().connect_timeout(Duration::from_secs(3));
    assert_eq!(cfg.connect_timeout, Duration::from_secs(3));
}

#[test]
fn config_builder_connect_timeout_secs_sets_field() {
    let cfg = HttpClientConfig::new().connect_timeout_secs(2.25);
    assert_eq!(cfg.connect_timeout, Duration::from_millis(2250));
}

#[test]
fn config_builder_read_timeout_sets_field() {
    let cfg = HttpClientConfig::new().read_timeout(Duration::from_secs(4));
    assert_eq!(cfg.read_timeout, Some(Duration::from_secs(4)));
}

#[test]
fn config_builder_pool_max_idle_per_host_sets_field() {
    let cfg = HttpClientConfig::new().pool_max_idle_per_host(5);
    assert_eq!(cfg.pool_max_idle_per_host, 5);
}

#[test]
fn config_builder_pool_idle_timeout_sets_field() {
    let cfg = HttpClientConfig::new().pool_idle_timeout(Duration::from_secs(11));
    assert_eq!(cfg.pool_idle_timeout, Duration::from_secs(11));
}

#[test]
fn config_builder_follow_redirects_sets_field() {
    let cfg = HttpClientConfig::new().follow_redirects(false);
    assert!(!cfg.follow_redirects);
}

#[test]
fn config_builder_max_redirects_sets_field() {
    let cfg = HttpClientConfig::new().max_redirects(4);
    assert_eq!(cfg.max_redirects, 4);
}

#[test]
fn config_builder_user_agent_sets_field() {
    let cfg = HttpClientConfig::new().user_agent("probe/1.0");
    assert_eq!(cfg.user_agent, "probe/1.0");
}

#[test]
fn config_builder_danger_accept_invalid_certs_sets_field() {
    let cfg = HttpClientConfig::new().danger_accept_invalid_certs(true);
    assert!(cfg.danger_accept_invalid_certs);
}

#[test]
fn config_builder_gzip_sets_field() {
    let cfg = HttpClientConfig::new().gzip(false);
    assert!(!cfg.gzip);
}

#[test]
fn config_builder_brotli_sets_field() {
    let cfg = HttpClientConfig::new().brotli(false);
    assert!(!cfg.brotli);
}

#[test]
fn config_builder_protocol_policy_sets_field() {
    let cfg = HttpClientConfig::new().protocol_policy(HttpProtocolPolicy::RequireHttp2);
    assert_eq!(cfg.protocol_policy, HttpProtocolPolicy::RequireHttp2);
}

#[test]
fn config_builder_retry_sets_field() {
    let cfg = HttpClientConfig::new().retry(RetryConfig {
        max_retries: 7,
        ..RetryConfig::default()
    });
    assert_eq!(cfg.retry.as_ref().map(|r| r.max_retries), Some(7));
}

#[test]
fn config_builder_auto_http_protocol_sets_field() {
    let cfg = HttpClientConfig::new().require_http2().auto_http_protocol();
    assert_eq!(cfg.protocol_policy, HttpProtocolPolicy::Auto);
}

#[test]
fn config_builder_require_http2_sets_field() {
    let cfg = HttpClientConfig::new().require_http2();
    assert_eq!(cfg.protocol_policy, HttpProtocolPolicy::RequireHttp2);
}

#[test]
fn config_default_hostname_flag_is_off() {
    let cfg = HttpClientConfig::default();
    assert!(!cfg.danger_accept_invalid_hostnames);
    assert!(cfg.read_timeout.is_none());
}

#[test]
fn retry_delay_for_constant_uses_base_then_caps_at_max() {
    let cfg = RetryConfig {
        initial_delay: Duration::from_millis(100),
        max_delay: Duration::from_millis(250),
        backoff: BackoffStrategy::Constant,
        ..RetryConfig::default()
    };
    assert_eq!(cfg.delay_for(0, None), Duration::from_millis(100));
    assert_eq!(cfg.delay_for(9, None), Duration::from_millis(100));
    // Retry-After wins over the backoff and is capped at max_delay.
    assert_eq!(
        cfg.delay_for(0, Some(Duration::from_millis(40))),
        Duration::from_millis(40)
    );
    assert_eq!(
        cfg.delay_for(0, Some(Duration::from_secs(9))),
        Duration::from_millis(250)
    );
}

#[test]
fn retry_delay_for_linear_grows_by_attempt() {
    let cfg = RetryConfig {
        initial_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(5),
        backoff: BackoffStrategy::Linear,
        ..RetryConfig::default()
    };
    assert_eq!(cfg.delay_for(0, None), Duration::ZERO);
    assert_eq!(cfg.delay_for(3, None), Duration::from_millis(300));
}

#[test]
fn retry_delay_for_exponential_caps_at_max() {
    let cfg = RetryConfig {
        initial_delay: Duration::from_millis(100),
        max_delay: Duration::from_millis(500),
        backoff: BackoffStrategy::Exponential,
        ..RetryConfig::default()
    };
    assert_eq!(cfg.delay_for(1, None), Duration::from_millis(200));
    assert_eq!(cfg.delay_for(3, None), Duration::from_millis(500));
}

#[test]
fn backoff_constant_returns_base() {
    let base = Duration::from_millis(100);
    let max = Duration::from_secs(10);
    assert_eq!(BackoffStrategy::Constant.delay(base, 0, max), base);
    assert_eq!(BackoffStrategy::Constant.delay(base, 7, max), base);
}

#[test]
fn backoff_constant_is_capped_by_max() {
    let base = Duration::from_millis(100);
    let max = Duration::from_millis(50);
    assert_eq!(BackoffStrategy::Constant.delay(base, 2, max), max);
}

#[test]
fn backoff_linear_is_base_times_attempt() {
    let base = Duration::from_millis(100);
    let max = Duration::from_secs(10);
    assert_eq!(BackoffStrategy::Linear.delay(base, 0, max), Duration::ZERO);
    assert_eq!(
        BackoffStrategy::Linear.delay(base, 3, max),
        Duration::from_millis(300)
    );
}

#[test]
fn backoff_linear_is_capped_by_max() {
    let base = Duration::from_millis(100);
    let max = Duration::from_millis(200);
    assert_eq!(BackoffStrategy::Linear.delay(base, 3, max), max);
}

#[test]
fn backoff_exponential_is_base_times_power_of_two_capped() {
    let base = Duration::from_millis(100);
    let max = Duration::from_secs(10);
    assert_eq!(
        BackoffStrategy::Exponential.delay(base, 3, max),
        Duration::from_millis(800)
    );
    assert_eq!(
        BackoffStrategy::Exponential.delay(base, 3, Duration::from_millis(500)),
        Duration::from_millis(500)
    );
}
