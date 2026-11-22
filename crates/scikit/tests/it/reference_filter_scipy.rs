//! Reference-value tests for `lfilter_zi` and the initial-state form of
//! `lfilter` (`lfilter_with_state`), against `scipy.signal` 1.17.1.
//!
//! The expected values are in `fixtures/filter_scipy_reference.tsv`, written
//! by `tools/reference/scikit/generate.py`. The input is the integer formula
//! of `sig` below, so Rust and Python build the same values.

use scikit::signal::{lfilter, lfilter_with_state, lfilter_zi};

const FIXTURE: &str = include_str!("../fixtures/filter_scipy_reference.tsv");

/// The value column of the fixture line `key`. Panics when the key is missing.
fn raw(key: &str) -> &'static str {
    let line = FIXTURE
        .lines()
        .find(|l| l.split('\t').next() == Some(key))
        .unwrap_or_else(|| panic!("fixture key `{key}` not found"));
    line.split('\t')
        .nth(1)
        .unwrap_or_else(|| panic!("fixture key `{key}` has no value column"))
}

/// The comma-separated values of the fixture line `key`.
fn reference_list(key: &str) -> Vec<f64> {
    raw(key)
        .split(',')
        .map(|v| {
            v.parse::<f64>()
                .unwrap_or_else(|e| panic!("bad value in `{key}`: {e}"))
        })
        .collect()
}

/// Asserts `actual` is within `1e-9` relative to `expected`, with an absolute
/// floor of `1e-12` for results that are zero or round-off.
fn close(actual: f64, expected: f64, label: &str) {
    let bound = f64::max(1e-9 * expected.abs(), 1e-12);
    assert!(
        (actual - expected).abs() <= bound,
        "{label}: actual {actual}, expected {expected}"
    );
}

/// Checks the length first, then each element with `close`.
fn close_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "`{key}`: length");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        close(*a, *e, &format!("`{key}`[{i}]"));
    }
}

/// The generator's input `x[i] = ((i * 37 + 11) % 101) / 4.0`, for 20 samples.
fn sig() -> Vec<f64> {
    (0..20)
        .map(|i| ((i * 37 + 11) % 101) as f64 / 4.0)
        .collect()
}

/// The filters `f1` to `f5` of the generator (`F6` has no row: SciPy raises).
// `-0.5772` is a filter coefficient that must match the generator's text, not Euler's constant.
#[allow(clippy::approx_constant)]
fn filters() -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
    vec![
        ("f1", vec![0.2, 0.3, 0.2], vec![1.0, -0.5]),
        (
            "f2",
            vec![0.0985, 0.2956, 0.2956, 0.0985],
            vec![1.0, -0.5772, 0.4218, -0.0563],
        ),
        ("f3", vec![2.0, 1.0], vec![2.0, -0.5]),
        ("f4", vec![0.25, 0.25, 0.25, 0.25], vec![1.0]),
        ("f5", vec![1.0, 0.5], vec![1.0, -0.3, 0.1]),
    ]
}

#[test]
fn lfilter_zi_matches_scipy() {
    for (name, b, a) in filters() {
        let zi = lfilter_zi(&b, &a);
        close_list(&zi, &format!("lfilter_zi.{name}"));
    }
}

#[test]
fn lfilter_zi_single_tap_is_empty() {
    assert!(lfilter_zi(&[1.0], &[1.0]).is_empty());
}

#[test]
fn lfilter_with_state_matches_scipy() {
    let x = sig();
    for (name, b, a) in filters() {
        let zi: Vec<f64> = lfilter_zi(&b, &a).iter().map(|v| v * x[0]).collect();
        let (y, zf) = lfilter_with_state(&b, &a, &x, &zi);
        close_list(&y, &format!("lfilter_state.{name}.y"));
        close_list(&zf, &format!("lfilter_state.{name}.zf"));
    }
}

#[test]
fn lfilter_with_state_zero_state_matches_scipy() {
    let x = sig();
    let (b, a) = (vec![0.2, 0.3, 0.2], vec![1.0, -0.5]);
    let (y, zf) = lfilter_with_state(&b, &a, &x, &[0.0, 0.0]);
    close_list(&y, "lfilter_state.f1.y_zero");
    close_list(&zf, "lfilter_state.f1.zf_zero");
}

#[test]
fn lfilter_with_state_zero_state_equals_lfilter() {
    let x = sig();
    let (b, a) = (vec![0.2, 0.3, 0.2], vec![1.0, -0.5]);
    let (y, _) = lfilter_with_state(&b, &a, &x, &[0.0, 0.0]);
    assert_eq!(y, lfilter(&b, &a, &x));
}

#[test]
#[should_panic(expected = "zi length")]
fn lfilter_with_state_rejects_wrong_zi_length() {
    // F1 needs max(3, 2) - 1 = 2 entries; one is wrong.
    let _ = lfilter_with_state(&[0.2, 0.3, 0.2], &[1.0, -0.5], &sig(), &[0.0]);
}

#[test]
#[should_panic(expected = "a[0] must be nonzero")]
fn lfilter_zi_rejects_zero_leading_coefficient() {
    let _ = lfilter_zi(&[1.0, 1.0], &[0.0, 1.0]);
}

#[test]
#[should_panic(expected = "singular")]
fn lfilter_zi_rejects_singular_system() {
    // A pole at z = 1 makes I - C^T singular.
    let _ = lfilter_zi(&[1.0, 1.0], &[1.0, -1.0]);
}
