//! Reference-value tests for `scikit::signal` filters with an initial state.
//!
//! Expected values come from SciPy 1.17.1 with NumPy 2.4.6. The expression of
//! each value is in the `source` column of
//! `tests/fixtures/filter_scipy_reference.tsv`. No Python runs in CI: the
//! fixture is the only source of the expected values.

use scikit::signal::{lfilter, lfilter_with_state, lfilter_zi};

const FIXTURE: &str = include_str!("../fixtures/filter_scipy_reference.tsv");

/// Return the value column stored under `key` in the fixture, as raw text.
fn raw(key: &str) -> &'static str {
    for line in FIXTURE.lines().skip(1) {
        let mut fields = line.split('\t');
        if fields.next() == Some(key) {
            return fields
                .next()
                .unwrap_or_else(|| panic!("fixture key {key}: no value column"));
        }
    }
    panic!("fixture key {key} is missing");
}

/// Return the comma-separated list stored under `key` in the fixture.
fn reference_list(key: &str) -> Vec<f64> {
    let text = raw(key);
    if text.is_empty() {
        return Vec::new();
    }
    text.split(',')
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("fixture key {key}: bad number {v}"))
        })
        .collect()
}

/// Asserts that `actual` is within 1e-9 relative of `expected`, with a
/// 1e-12 absolute floor for results that are zero or round-off. A NaN
/// expected value needs a NaN result.
fn close(actual: f64, expected: f64, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}: got {actual}, expected NaN");
        return;
    }
    let tol = f64::max(1e-9 * expected.abs(), 1e-12);
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: got {actual}, expected {expected} (tol {tol})"
    );
}

/// Asserts that `actual` has the length and the values of the fixture list `key`.
fn close_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "length of `{key}`");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        close(*a, *e, &format!("`{key}`[{i}]"));
    }
}

/// The input of the fixture rows: `sig(i) = ((i * 37 + 11) % 101) / 4.0`.
fn sig(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 37 + 11) % 101) as f64 / 4.0).collect()
}

/// The filters F1 to F5 of the fixture rows: (name, b, a).
// The coefficient -0.5772 of F2 is a filter value, not the Euler constant.
#[allow(clippy::approx_constant)]
const FILTERS: [(&str, &[f64], &[f64]); 5] = [
    ("f1", &[0.2, 0.3, 0.2], &[1.0, -0.5]),
    (
        "f2",
        &[0.0985, 0.2956, 0.2956, 0.0985],
        &[1.0, -0.5772, 0.4218, -0.0563],
    ),
    ("f3", &[2.0, 1.0], &[2.0, -0.5]),
    ("f4", &[0.25, 0.25, 0.25, 0.25], &[1.0]),
    ("f5", &[1.0, 0.5], &[1.0, -0.3, 0.1]),
];

#[test]
fn lfilter_zi_matches_scipy() {
    for (name, b, a) in FILTERS {
        close_list(&lfilter_zi(b, a), &format!("lfilter_zi.{name}"));
    }
}

#[test]
fn lfilter_zi_single_tap_is_empty() {
    assert!(lfilter_zi(&[1.0], &[1.0]).is_empty());
}

#[test]
fn lfilter_with_state_matches_scipy() {
    let x = sig(20);
    for (name, b, a) in FILTERS {
        // zi = lfilter_zi(b, a) * x[0], the steady state for a first input of x[0].
        let zi: Vec<f64> = lfilter_zi(b, a).iter().map(|v| v * x[0]).collect();
        let (y, zf) = lfilter_with_state(b, a, &x, &zi);
        close_list(&y, &format!("lfilter_state.{name}.y"));
        close_list(&zf, &format!("lfilter_state.{name}.zf"));
    }
}

#[test]
fn lfilter_with_state_zero_state_matches_scipy() {
    let x = sig(20);
    let (b, a) = (FILTERS[0].1, FILTERS[0].2);
    let (y, zf) = lfilter_with_state(b, a, &x, &[0.0, 0.0]);
    close_list(&y, "lfilter_state.f1.y_zero");
    close_list(&zf, "lfilter_state.f1.zf_zero");
}

#[test]
fn lfilter_with_state_zero_state_equals_lfilter() {
    let x = sig(20);
    let (b, a) = (FILTERS[0].1, FILTERS[0].2);
    assert_eq!(
        lfilter_with_state(b, a, &x, &[0.0, 0.0]).0,
        lfilter(b, a, &x)
    );
}

#[test]
#[should_panic(expected = "zi length")]
fn lfilter_with_state_rejects_wrong_zi_length() {
    // F1 needs max(3, 2) - 1 = 2 entries; one is wrong.
    let (b, a) = (FILTERS[0].1, FILTERS[0].2);
    lfilter_with_state(b, a, &sig(5), &[0.0]);
}

#[test]
#[should_panic]
fn lfilter_zi_rejects_zero_leading_coefficient() {
    lfilter_zi(&[1.0, 1.0], &[0.0, 1.0]);
}

#[test]
#[should_panic(expected = "singular")]
fn lfilter_zi_rejects_singular_system() {
    // A pole at z = 1 makes I - C^T singular.
    lfilter_zi(&[1.0, 1.0], &[1.0, -1.0]);
}
