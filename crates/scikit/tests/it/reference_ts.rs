//! Reference-value tests for the `ts` rows of `docs/reference-coverage.md`.
//!
//! The expected values are in `fixtures/ts_reference.tsv`. The generator
//! `tools/reference/scikit/generate.py` writes them from NumPy 2.4.6, SciPy
//! 1.17.1, pandas 3.1.1 and statsmodels 0.15.0; its `source` column names the
//! expression of each value. The inputs come from the integer formulas below,
//! so Rust and Python build the same values.

use std::f64::consts::PI;

use frame::frame::{Series, Value};
use scikit::ts::acf::{acf, ljung_box, pacf, pacf_from_acf, LjungBoxResult};
use scikit::ts::arima::{difference, ArimaModel, ArimaOrder};
use scikit::ts::decompose::{seasonal_decompose, DecomposeModel};
use scikit::ts::ops::{
    ewma, ewma_halflife, ewma_span, expanding_max, expanding_mean, expanding_min, expanding_std,
    expanding_sum, lag, lead,
};
use scikit::ts::resample::{downsample, resample_to_len, upsample, AggMethod, InterpMethod};
use scikit::ts::smoothing::{holt, holt_winters, ses, HoltWintersParams, HwSeasonal};

const FIXTURE: &str = include_str!("../fixtures/ts_reference.tsv");

/// Relative tolerance of every reference value (the table's `1e-9 rel`).
const REL_TOL: f64 = 1e-9;

/// Absolute part of every tolerance (the table's `1e-12 abs`), so that an
/// expected value of `0.0` has a bound.
const ABS_FLOOR: f64 = 1e-12;

/// The values of the fixture line `key`. Panics when the key is missing.
fn reference(key: &str) -> Vec<f64> {
    let line = FIXTURE
        .lines()
        .find(|l| l.split('\t').next() == Some(key))
        .unwrap_or_else(|| panic!("fixture key `{key}` not found"));
    line.split('\t')
        .nth(1)
        .unwrap_or_else(|| panic!("fixture key `{key}` has no value column"))
        .split(',')
        .map(|v| {
            v.parse::<f64>()
                .unwrap_or_else(|e| panic!("bad value in `{key}`: {e}"))
        })
        .collect()
}

/// The single value of the fixture line `key`.
fn scalar(key: &str) -> f64 {
    let v = reference(key);
    assert_eq!(v.len(), 1, "`{key}` is not a scalar");
    v[0]
}

/// The generator's input `sample(n)`: `x[i] = ((i * 37 + 11) % 101) / 4.0`.
fn sample(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 37 + 11) % 101) as f64 / 4.0).collect()
}

/// The generator's `ts_y(n)`: a trend, a period-12 cycle and an integer ripple.
fn ts_y(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            10.0 + 0.05 * i as f64
                + 3.0 * (2.0 * PI * i as f64 / 12.0).sin()
                + ((i * 37 + 11) % 101) as f64 / 50.0
        })
        .collect()
}

/// The generator's `ts_q(n)`: integer-built values for the Ljung-Box p-values.
fn ts_q(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * i * 31 + i * 17 + 7) % 103) as f64 / 10.0)
        .collect()
}

/// The generator's `ts_b(n)`: the ARIMA input.
fn ts_b(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            5.0 + 2.0 * (2.0 * PI * i as f64 / 7.0).sin()
                + ((i * 37 + 11) % 101) as f64 / 25.0
                + (i % 5) as f64 * 0.3
        })
        .collect()
}

/// Every element of `actual` is within `REL_TOL * |e| + ABS_FLOOR` of
/// `expected`. A NaN expected value needs a NaN actual value.
fn assert_close_list(actual: &[f64], expected: &[f64], key: &str) {
    assert_eq!(actual.len(), expected.len(), "length of `{key}`");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "`{key}`[{i}] = {a}, expected NaN");
            continue;
        }
        let tol = REL_TOL * e.abs() + ABS_FLOOR;
        assert!(
            (a - e).abs() <= tol,
            "`{key}`[{i}] = {a}, expected {e} (tol {tol})"
        );
    }
}

/// A list of at most 64 values is compared element by element. A longer list
/// is compared by its first 5 values, its last 5 values, the sum of absolute
/// values and its length, as the generator's `vec_rows` writes it.
fn check_vec(actual: &[f64], prefix: &str) {
    if actual.len() <= 64 {
        assert_close_list(actual, &reference(prefix), prefix);
        return;
    }
    let n = actual.len();
    assert_close_list(&actual[..5], &reference(&format!("{prefix}_head")), prefix);
    assert_close_list(
        &actual[n - 5..],
        &reference(&format!("{prefix}_tail")),
        prefix,
    );
    let abs_sum: f64 = actual.iter().map(|v| v.abs()).sum();
    let want = scalar(&format!("{prefix}_abs_sum"));
    assert!(
        (abs_sum - want).abs() <= REL_TOL * want.abs() + ABS_FLOOR,
        "`{prefix}_abs_sum` = {abs_sum}, expected {want}"
    );
    assert_eq!(
        n as f64,
        scalar(&format!("{prefix}_len")),
        "length of `{prefix}`"
    );
}

/// A series with NaN at both ends, as the generator's `nan_vec_rows` writes
/// it: the valid range, 5 values at each end and the sum of absolute values.
fn check_nan_vec(actual: &[f64], prefix: &str) {
    let valid = reference(&format!("{prefix}_valid"));
    let (first, last, count) = (valid[0] as usize, valid[1] as usize, valid[2] as usize);
    assert!(
        actual[..first].iter().all(|v| v.is_nan()),
        "`{prefix}`: values before the valid range must be NaN"
    );
    assert!(
        actual[last + 1..].iter().all(|v| v.is_nan()),
        "`{prefix}`: values after the valid range must be NaN"
    );
    let inner = &actual[first..=last];
    assert!(
        inner.iter().all(|v| v.is_finite()),
        "`{prefix}`: a value inside the valid range is not finite"
    );
    assert_eq!(inner.len(), count, "valid count of `{prefix}`");
    assert_close_list(
        &actual[first..first + 5],
        &reference(&format!("{prefix}_head")),
        prefix,
    );
    assert_close_list(
        &actual[last - 4..=last],
        &reference(&format!("{prefix}_tail")),
        prefix,
    );
    let abs_sum: f64 = inner.iter().map(|v| v.abs()).sum();
    let want = scalar(&format!("{prefix}_abs_sum"));
    assert!(
        (abs_sum - want).abs() <= REL_TOL * want.abs() + ABS_FLOOR,
        "`{prefix}_abs_sum` = {abs_sum}, expected {want}"
    );
}

/// The values of a `Series` as floats, with `Null` as NaN.
fn series_floats(s: &Series) -> Vec<f64> {
    s.values()
        .iter()
        .map(|v| match v {
            Value::Float(f) => *f,
            Value::Null => f64::NAN,
            other => panic!("unexpected series value {other:?}"),
        })
        .collect()
}

// ============================================================================
// acf (5 rows)
// ============================================================================

#[test]
fn acf_matches_statsmodels() {
    let y = ts_y(500);
    let out = acf(&y, 24).unwrap();
    check_vec(&out, "acf_y500");
}

#[test]
fn pacf_matches_statsmodels() {
    let y = ts_y(500);
    let out = pacf(&y, 24).unwrap();
    check_vec(&out, "pacf_y500");
}

#[test]
fn pacf_from_acf_matches_statsmodels() {
    // The input is the acf of the series, computed by `acf` (checked by
    // `acf_matches_statsmodels`). The fixture keeps 12 significant digits of
    // the acf, and Durbin-Levinson amplifies that rounding at lag 16 to about
    // 1e-8 relative, which is a fixture precision effect and not a defect.
    let out = pacf_from_acf(&acf(&ts_y(500), 24).unwrap()).unwrap();
    check_vec(&out, "pacf_y500");
}

#[test]
fn ljung_box_matches_statsmodels() {
    let cases: [(&str, Vec<f64>, usize); 3] = [
        ("lb_q500_lag5", ts_q(500), 5),
        ("lb_q500_lag20", ts_q(500), 20),
        ("lb_y500_lag10", ts_y(500), 10),
    ];
    for (key, x, lags) in cases {
        let r: LjungBoxResult = ljung_box(&x, lags).unwrap();
        assert_eq!(r.lags, lags, "lags of `{key}`");
        assert_close_list(&[r.statistic, r.p_value], &reference(key), key);
    }
}

#[test]
fn is_significant_follows_the_pvalue_rule() {
    let p_values = reference("is_significant_p");
    let alphas = reference("is_significant_alpha");
    let expected = reference("is_significant_expected");
    for ((p, alpha), want) in p_values.iter().zip(&alphas).zip(&expected) {
        let r = LjungBoxResult {
            statistic: 0.0,
            p_value: *p,
            lags: 1,
        };
        assert_eq!(
            r.is_significant(*alpha),
            *want == 1.0,
            "p = {p}, alpha = {alpha}"
        );
    }
}

// ============================================================================
// arima (5 rows); b = ts_b(120)
// ============================================================================

#[test]
fn difference_matches_numpy() {
    let b = ts_b(120);
    for d in 1..=3 {
        check_vec(&difference(&b, d), &format!("arima_diff{d}"));
    }
    assert_eq!(difference(&b, 0), b, "difference of order 0 is the input");
}

/// The four AR orders (p, d, q) of the AR-part rows.
const AR_ORDERS: [(usize, usize, usize); 4] = [(2, 0, 0), (1, 1, 0), (2, 1, 0), (1, 0, 1)];

#[test]
fn fit_ar_matches_statsmodels() {
    let b = ts_b(120);
    for (p, d, q) in AR_ORDERS {
        let key = format!("arima_fit_ar_{p}{d}{q}");
        let m = ArimaModel::fit(&b, ArimaOrder::new(p, d, q)).unwrap();
        let mut actual = m.ar_coeffs.clone();
        actual.push(m.intercept);
        assert_close_list(&actual, &reference(&key), &key);
    }
}

#[test]
fn fit_ma_matches_css_optimum() {
    let b = ts_b(120);
    // The CSS search has a step of 1e-3, so its error is at most 5e-4.
    const MA_TOL: f64 = 2e-3;
    for (p, d, q) in [(0, 0, 1), (0, 1, 1), (1, 0, 1)] {
        let key = format!("arima_fit_ma_{p}{d}{q}");
        let m = ArimaModel::fit(&b, ArimaOrder::new(p, d, q)).unwrap();
        assert_eq!(m.ma_coeffs.len(), q, "MA order of `{key}`");
        let want = scalar(&key);
        let got = m.ma_coeffs[0];
        assert!(
            (got - want).abs() <= MA_TOL,
            "`{key}` = {got}, expected {want} (tol {MA_TOL})"
        );
    }
}

#[test]
fn fitted_values_matches_statsmodels() {
    let b = ts_b(120);
    let m = ArimaModel::fit(&b, ArimaOrder::new(2, 0, 0)).unwrap();
    // The first max(p, q) = 2 values are the start of the state; skip them.
    check_vec(&m.fitted_values(&b)[2..], "arima_fitted_200");
}

#[test]
fn fitted_values_d1_matches_statsmodels() {
    let b = ts_b(120);
    for p in [1usize, 2] {
        let key = format!("arima_fitted_{p}10");
        let m = ArimaModel::fit(&b, ArimaOrder::new(p, 1, 0)).unwrap();
        // The first p + d values are the start of the state; skip them.
        check_vec(&m.fitted_values(&b)[p + 1..], &key);
    }
}

#[test]
fn forecast_matches_statsmodels() {
    let b = ts_b(120);
    for (p, d) in [(2, 0), (1, 1), (2, 1)] {
        let key = format!("arima_forecast_{p}{d}0");
        let m = ArimaModel::fit(&b, ArimaOrder::new(p, d, 0)).unwrap();
        check_vec(&m.forecast(&b, 5), &key);
    }
}

#[test]
fn aic_matches_statsmodels_residuals() {
    let b = ts_b(120);
    for (p, d) in [(2, 0), (1, 1), (2, 1)] {
        let key = format!("arima_aic_{p}{d}0");
        let m = ArimaModel::fit(&b, ArimaOrder::new(p, d, 0)).unwrap();
        let got = m.aic(&b);
        let want = scalar(&key);
        assert!(
            (got - want).abs() <= REL_TOL * want.abs() + ABS_FLOOR,
            "`{key}` = {got}, expected {want}"
        );
    }
}

// ============================================================================
// decompose (3 rows); y = ts_y(500)
// ============================================================================

/// The four cases of the decompose rows: (period, model, key tag).
const DECOMPOSE_CASES: [(usize, DecomposeModel, &str); 4] = [
    (12, DecomposeModel::Additive, "add12"),
    (12, DecomposeModel::Multiplicative, "mul12"),
    (7, DecomposeModel::Additive, "add7"),
    (7, DecomposeModel::Multiplicative, "mul7"),
];

#[test]
fn seasonal_decompose_matches_statsmodels() {
    let y = ts_y(500);
    for (period, model, tag) in DECOMPOSE_CASES {
        let r = seasonal_decompose(&y, period, model).unwrap();
        assert_eq!(r.period, period, "period of `{tag}`");
        assert_eq!(r.observed, y, "observed of `{tag}`");
        check_vec(&r.seasonal[..period], &format!("dec_{tag}_seasonal"));
        check_nan_vec(&r.trend, &format!("dec_{tag}_trend"));
        check_nan_vec(&r.residual, &format!("dec_{tag}_resid"));
    }
}

#[test]
fn trend_strength_matches_closed_form() {
    let y = ts_y(500);
    for (period, model, tag) in DECOMPOSE_CASES {
        let r = seasonal_decompose(&y, period, model).unwrap();
        let key = format!("dec_{tag}_strength");
        let want = reference(&key)[0];
        let got = r.trend_strength();
        assert!(
            (got - want).abs() <= REL_TOL * want.abs() + ABS_FLOOR,
            "`{key}`[0] = {got}, expected {want}"
        );
    }
}

#[test]
fn seasonal_strength_matches_closed_form() {
    let y = ts_y(500);
    for (period, model, tag) in DECOMPOSE_CASES {
        let r = seasonal_decompose(&y, period, model).unwrap();
        let key = format!("dec_{tag}_strength");
        let want = reference(&key)[1];
        let got = r.seasonal_strength();
        assert!(
            (got - want).abs() <= REL_TOL * want.abs() + ABS_FLOOR,
            "`{key}`[1] = {got}, expected {want}"
        );
    }
}

// ============================================================================
// ops (10 rows); x = sample(60)
// ============================================================================

#[test]
fn ewma_matches_pandas() {
    let x = sample(60);
    check_vec(&ewma(&x, 0.3), "ewma_a0p3");
}

#[test]
fn ewma_span_matches_pandas() {
    let x = sample(60);
    check_vec(&ewma_span(&x, 10.0), "ewma_span10");
}

#[test]
fn ewma_halflife_matches_pandas() {
    let x = sample(60);
    check_vec(&ewma_halflife(&x, 3.0), "ewma_halflife3");
}

#[test]
fn expanding_mean_matches_pandas() {
    assert_close_list(
        &expanding_mean(&sample(60)),
        &reference("expanding_mean"),
        "expanding_mean",
    );
}

#[test]
fn expanding_sum_matches_pandas() {
    assert_close_list(
        &expanding_sum(&sample(60)),
        &reference("expanding_sum"),
        "expanding_sum",
    );
}

#[test]
fn expanding_min_matches_pandas() {
    assert_close_list(
        &expanding_min(&sample(60)),
        &reference("expanding_min"),
        "expanding_min",
    );
}

#[test]
fn expanding_max_matches_pandas() {
    assert_close_list(
        &expanding_max(&sample(60)),
        &reference("expanding_max"),
        "expanding_max",
    );
}

#[test]
fn expanding_std_matches_pandas() {
    assert_close_list(
        &expanding_std(&sample(60)),
        &reference("expanding_std"),
        "expanding_std",
    );
}

#[test]
fn lag_matches_pandas() {
    let s = Series::new(sample(12));
    let got = series_floats(&lag(&s, 3));
    assert_eq!(
        got.iter().filter(|v| v.is_nan()).count(),
        3,
        "NaN count of lag"
    );
    assert_close_list(&got, &reference("lag3"), "lag3");
}

#[test]
fn lead_matches_pandas() {
    let s = Series::new(sample(12));
    let got = series_floats(&lead(&s, 2));
    assert_eq!(
        got.iter().filter(|v| v.is_nan()).count(),
        2,
        "NaN count of lead"
    );
    assert_close_list(&got, &reference("lead2"), "lead2");
}

// ============================================================================
// resample (3 rows)
// ============================================================================

#[test]
fn downsample_matches_pandas() {
    let x = sample(58);
    let cases = [
        (AggMethod::Mean, "downsample_mean"),
        (AggMethod::Sum, "downsample_sum"),
        (AggMethod::Min, "downsample_min"),
        (AggMethod::Max, "downsample_max"),
        (AggMethod::First, "downsample_first"),
        (AggMethod::Last, "downsample_last"),
        (AggMethod::Median, "downsample_median"),
    ];
    for (method, key) in cases {
        let got = downsample(&x, 4, method).unwrap();
        assert_close_list(&got, &reference(key), key);
    }
}

#[test]
fn upsample_matches_numpy_pandas() {
    let x = sample(5);
    let cases = [
        (InterpMethod::Linear, "upsample_linear"),
        (InterpMethod::ForwardFill, "upsample_ffill"),
        (InterpMethod::BackwardFill, "upsample_bfill"),
        (InterpMethod::Nearest, "upsample_nearest"),
    ];
    for (method, key) in cases {
        let got = upsample(&x, 3, method).unwrap();
        check_vec(&got, key);
    }
}

#[test]
fn resample_to_len_matches_numpy() {
    let x = sample(9);
    for k in [14, 5, 9] {
        let key = format!("resample_to_len_{k}");
        let got = resample_to_len(&x, k).unwrap();
        check_vec(&got, &key);
    }
}

// ============================================================================
// smoothing (6 rows); y = ts_y(60)
// ============================================================================

#[test]
fn ses_matches_statsmodels() {
    let y = ts_y(60);
    let r = ses(&y, 0.3).unwrap();
    assert!((r.alpha - 0.3).abs() <= ABS_FLOOR, "alpha of ses");
    check_vec(&r.fitted, "ses_a0p3_fitted");
    assert_close_list(&[r.level], &[scalar("ses_a0p3_level")], "ses_a0p3_level");
}

#[test]
fn ses_forecast_matches_statsmodels() {
    let y = ts_y(60);
    let r = ses(&y, 0.3).unwrap();
    check_vec(&r.forecast(3), "ses_a0p3_forecast3");
}

#[test]
fn holt_matches_statsmodels() {
    let y = ts_y(60);
    let r = holt(&y, 0.5, 0.2).unwrap();
    check_vec(&r.fitted, "holt_a0p5_b0p2_fitted");
    assert_close_list(
        &[r.level, r.trend],
        &reference("holt_a0p5_b0p2_state"),
        "holt_a0p5_b0p2_state",
    );
}

#[test]
fn holt_forecast_matches_statsmodels() {
    let y = ts_y(60);
    let r = holt(&y, 0.5, 0.2).unwrap();
    check_vec(&r.forecast(5), "holt_a0p5_b0p2_forecast5");
}

#[test]
fn holt_winters_matches_statsmodels() {
    let y = ts_y(60);
    let cases = [
        (HwSeasonal::Additive, "hw_add"),
        (HwSeasonal::Multiplicative, "hw_mul"),
    ];
    for (seasonal, tag) in cases {
        let params = HoltWintersParams {
            alpha: 0.4,
            beta: 0.1,
            gamma: 0.3,
            period: 12,
            seasonal,
        };
        let r = holt_winters(&y, &params).unwrap();
        check_vec(&r.fitted, &format!("{tag}_fitted"));
        assert_close_list(
            &[r.level, r.trend],
            &reference(&format!("{tag}_state")),
            &format!("{tag}_state"),
        );
    }
}

#[test]
fn holt_winters_forecast_matches_statsmodels() {
    let y = ts_y(60);
    let cases = [
        (HwSeasonal::Additive, "hw_add"),
        (HwSeasonal::Multiplicative, "hw_mul"),
    ];
    for (seasonal, tag) in cases {
        let params = HoltWintersParams {
            alpha: 0.4,
            beta: 0.1,
            gamma: 0.3,
            period: 12,
            seasonal,
        };
        let r = holt_winters(&y, &params).unwrap();
        check_vec(&r.forecast(14), &format!("{tag}_forecast14"));
    }
}
