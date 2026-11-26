//! Coverage tests for the window and cumulative methods of `frame::frame/series`
//! and `frame::frame/ops/window`. Tests only: every expected value is written out
//! by hand from the current behavior of the code.

use frame::frame::Series;

fn floats(s: &Series) -> Vec<Option<f64>> {
    s.values().iter().map(|v| v.as_float()).collect()
}

/// Compares each value; a `None` must be null, and a `NaN` must be `NaN`.
fn assert_series(actual: &[Option<f64>], expected: &[Option<f64>]) {
    assert_eq!(actual.len(), expected.len(), "length differs: {actual:?}");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        match (a, e) {
            (None, None) => {}
            (Some(a), Some(e)) if a.is_nan() && e.is_nan() => {}
            (Some(a), Some(e)) => assert!((a - e).abs() < 1e-9, "index {i}: expected {e}, got {a}"),
            _ => panic!("index {i}: expected {e:?}, got {a:?}"),
        }
    }
}

/// `[1, null, 3, 4]`
fn with_null() -> Series {
    Series::new(vec![Some(1.0), None, Some(3.0), Some(4.0)])
}

fn n() -> Option<f64> {
    None
}

fn v(x: f64) -> Option<f64> {
    Some(x)
}

#[test]
fn cumulative_ops_skip_nulls() {
    let s = with_null();
    assert_series(&floats(&s.cumsum()), &[v(1.0), n(), v(4.0), v(8.0)]);
    assert_series(&floats(&s.cumprod()), &[v(1.0), n(), v(3.0), v(12.0)]);
    assert_series(&floats(&s.cummax()), &[v(1.0), n(), v(3.0), v(4.0)]);
    assert_series(&floats(&s.cummin()), &[v(1.0), n(), v(1.0), v(1.0)]);

    let t = Series::new(vec![5.0, 2.0, 7.0]);
    assert_series(&floats(&t.cummax()), &[v(5.0), v(5.0), v(7.0)]);
    assert_series(&floats(&t.cummin()), &[v(5.0), v(2.0), v(2.0)]);
}

#[test]
fn shift_cases() {
    let s = Series::new(vec![1.0, 2.0, 3.0]);
    assert_series(&floats(&s.shift(1)), &[n(), v(1.0), v(2.0)]);
    assert_series(&floats(&s.shift(0)), &[v(1.0), v(2.0), v(3.0)]);
    assert_series(&floats(&s.shift(3)), &[n(), n(), n()]);
    assert_series(&floats(&s.shift(10)), &[n(), n(), n()]);
    assert_series(&floats(&s.shift(-1)), &[v(2.0), v(3.0), n()]);
    assert_series(&floats(&s.shift(-3)), &[n(), n(), n()]);
    assert_series(&floats(&s.shift(-10)), &[n(), n(), n()]);
}

#[test]
fn diff_and_pct_change_with_nulls() {
    let s = with_null();
    assert_series(&floats(&s.diff(1)), &[n(), n(), n(), v(1.0)]);
    assert_series(&floats(&s.diff(2)), &[n(), n(), v(2.0), n()]);
    assert_series(&floats(&s.pct_change(1)), &[n(), n(), n(), v(1.0 / 3.0)]);
    assert_series(&floats(&s.pct_change(2)), &[n(), n(), v(2.0), n()]);
}

#[test]
fn pct_change_zero_previous_is_null() {
    let s = Series::new(vec![0.0, 5.0, 10.0]);
    assert_series(&floats(&s.pct_change(1)), &[n(), n(), v(1.0)]);
}

#[test]
fn diff_period_longer_than_series() {
    let s = Series::new(vec![1.0, 2.0]);
    assert_series(&floats(&s.diff(5)), &[n(), n()]);
    assert_series(&floats(&s.pct_change(5)), &[n(), n()]);
}

#[test]
fn rolling_all_functions() {
    let s = Series::new(vec![1.0, 2.0, 4.0, 8.0]);
    let r = s.rolling(2);
    assert_series(&floats(&r.sum()), &[n(), v(3.0), v(6.0), v(12.0)]);
    assert_series(&floats(&r.mean()), &[n(), v(1.5), v(3.0), v(6.0)]);
    assert_series(&floats(&r.min()), &[n(), v(1.0), v(2.0), v(4.0)]);
    assert_series(&floats(&r.max()), &[n(), v(2.0), v(4.0), v(8.0)]);
    assert_series(&floats(&r.var()), &[n(), v(0.5), v(2.0), v(8.0)]);
    assert_series(
        &floats(&r.std()),
        &[
            n(),
            v(std::f64::consts::FRAC_1_SQRT_2),
            v(std::f64::consts::SQRT_2),
            v(2.8284271247461903),
        ],
    );
}

#[test]
fn rolling_window_one_std_var_is_nan() {
    let s = Series::new(vec![3.0, 5.0]);
    for value in s.rolling(1).std().values() {
        assert!(value.as_float().unwrap().is_nan());
    }
    for value in s.rolling(1).var().values() {
        assert!(value.as_float().unwrap().is_nan());
    }
}

#[test]
fn rolling_null_in_window() {
    let s = with_null();
    assert_series(&floats(&s.rolling(2).sum()), &[n(), n(), n(), v(7.0)]);
    assert_series(
        &floats(&s.rolling(2).min_periods(1).sum()),
        &[v(1.0), v(1.0), v(3.0), v(7.0)],
    );
}

#[test]
fn expanding_all_functions() {
    let s = Series::new(vec![3.0, 1.0, 4.0]);
    let e = s.expanding();
    assert_series(&floats(&e.sum()), &[v(3.0), v(4.0), v(8.0)]);
    assert_series(&floats(&e.mean()), &[v(3.0), v(2.0), v(8.0 / 3.0)]);
    assert_series(&floats(&e.min()), &[v(3.0), v(1.0), v(1.0)]);
    assert_series(&floats(&e.max()), &[v(3.0), v(3.0), v(4.0)]);
    assert_series(
        &floats(&e.var()),
        &[v(f64::NAN), v(2.0), v(2.333333333333333)],
    );
    assert_series(
        &floats(&e.std()),
        &[
            v(f64::NAN),
            v(std::f64::consts::SQRT_2),
            v(1.5275252316519465),
        ],
    );
}

#[test]
fn expanding_skips_nulls_and_min_periods() {
    let s = with_null();
    assert_series(
        &floats(&s.expanding().sum()),
        &[v(1.0), v(1.0), v(4.0), v(8.0)],
    );
    assert_series(
        &floats(&s.expanding().min_periods(2).sum()),
        &[n(), n(), v(4.0), v(8.0)],
    );
}

#[test]
fn ewm_mean_adjusted_with_null() {
    let s = with_null();
    assert_series(
        &floats(&s.ewm_alpha(0.5).mean()),
        &[v(1.0), n(), v(2.3333333333333335), v(3.2857142857142856)],
    );
    assert_series(
        &floats(&s.ewm_alpha(0.5).min_periods(2).mean()),
        &[n(), n(), v(2.3333333333333335), v(3.2857142857142856)],
    );
}

#[test]
fn ewm_mean_non_adjusted_with_null_and_min_periods() {
    let s = with_null();
    assert_series(
        &floats(&s.ewm_alpha(0.5).adjust(false).mean()),
        &[v(1.0), n(), v(2.0), v(3.0)],
    );
    assert_series(
        &floats(&s.ewm_alpha(0.5).adjust(false).min_periods(2).mean()),
        &[n(), n(), v(2.0), v(3.0)],
    );
}

#[test]
fn ewm_alpha_is_clamped() {
    let s = Series::new(vec![1.0, 2.0, 3.0]);
    assert_series(&floats(&s.ewm_alpha(2.0).mean()), &[v(1.0), v(2.0), v(3.0)]);
    assert_series(
        &floats(&s.ewm_alpha(2.0).adjust(false).mean()),
        &[v(1.0), v(2.0), v(3.0)],
    );
    assert_series(
        &floats(&s.ewm_alpha(-1.0).adjust(false).mean()),
        &[v(1.0), v(1.0), v(1.0)],
    );
}

#[test]
fn ewm_var_std_adjusted_and_not() {
    let s = Series::new(vec![1.0, 2.0, 4.0]);
    assert_series(&floats(&s.ewm_alpha(0.5).var()), &[n(), v(0.5), v(2.5)]);
    assert_series(
        &floats(&s.ewm_alpha(0.5).std()),
        &[
            n(),
            v(std::f64::consts::FRAC_1_SQRT_2),
            v(1.5811388300841898),
        ],
    );
    assert_series(
        &floats(&s.ewm_alpha(0.5).adjust(false).var()),
        &[n(), v(0.5), v(2.7)],
    );
    assert_series(
        &floats(&s.ewm_alpha(0.5).adjust(false).std()),
        &[
            n(),
            v(std::f64::consts::FRAC_1_SQRT_2),
            v(1.6431676725154984),
        ],
    );

    let t = Series::new(vec![1.0, 3.0]);
    assert_series(&floats(&t.ewm_alpha(0.5).var()), &[n(), v(2.0)]);
}

#[test]
fn ewm_var_alpha_zero() {
    let s = Series::new(vec![1.0, 2.0, 3.0]);
    assert_series(&floats(&s.ewm_alpha(0.0).var()), &[n(), v(0.5), v(1.0)]);
}

#[test]
fn ewm_var_zero_denominator_is_null() {
    let s = Series::new(vec![1.0, 2.0, 3.0]);
    assert_series(
        &floats(&s.ewm_alpha(1.0).adjust(false).var()),
        &[n(), n(), n()],
    );
    assert_series(&floats(&s.ewm_alpha(1.0).var()), &[n(), n(), n()]);
}

#[test]
fn ewm_var_null_input_and_min_periods() {
    let s = with_null();
    assert_series(
        &floats(&s.ewm_span(3.0).var()),
        &[n(), n(), v(2.0), v(1.857142857142857)],
    );
    assert_series(
        &floats(&s.ewm_span(3.0).min_periods(3).var()),
        &[n(), n(), n(), v(1.857142857142857)],
    );
    assert_series(
        &floats(&s.ewm_span(3.0).std()),
        &[n(), n(), v(std::f64::consts::SQRT_2), v(1.3627702877384937)],
    );
}

#[test]
fn ewm_halflife_matches_alpha() {
    let s = Series::new(vec![1.0, 2.0, 3.0]);
    assert_series(
        &floats(&s.ewm_halflife(1.0).mean()),
        &[v(1.0), v(1.6666666666666667), v(2.4285714285714284)],
    );
    assert_series(
        &floats(&s.ewm_halflife(1.0).mean()),
        &floats(&s.ewm_alpha(0.5).mean()),
    );
}

#[test]
fn ewm_span_equals_alpha() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0]);
    assert_series(
        &floats(&s.ewm_span(3.0).mean()),
        &floats(&s.ewm_alpha(0.5).mean()),
    );
}
