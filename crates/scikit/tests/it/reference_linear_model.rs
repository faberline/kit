//! Reference-value tests for the `linear_model` rows of `docs/reference-coverage.md`.
//!
//! The expected values are in `fixtures/linear_model_reference.tsv`. The generator
//! `tools/reference/scikit/generate.py` writes them from NumPy 2.4.6; its `source`
//! column names the NumPy expression of each value. The inputs come from the
//! integer formulas below, so Rust and Python build the same values.

use arraykit::NdArray;
use scikit::linear_model::{LinearRegression, Ridge};

const FIXTURE: &str = include_str!("../fixtures/linear_model_reference.tsv");

/// Relative tolerance of the fixture values (the table's `1e-9 rel`).
const REL_TOL: f64 = 1e-9;

/// Relative tolerance for the ill-conditioned case (condition number about 1e8).
const ILL_REL_TOL: f64 = 1e-6;

/// Absolute part of every tolerance.
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

/// Every element of `actual` is within `rel * |e| + ABS_FLOOR` of the fixture
/// value `e` of `key`, and the lengths are equal.
fn assert_close(actual: &[f64], key: &str, rel: f64) {
    let expected = reference(key);
    assert_eq!(actual.len(), expected.len(), "length of `{key}`");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = rel * e.abs() + ABS_FLOOR;
        assert!(
            (a - e).abs() <= tol,
            "`{key}`[{i}] = {a}, expected {e} (tol {tol})"
        );
    }
}

/// The generator's noise term `e(i)`.
fn noise(i: usize) -> f64 {
    (((i * 7919) % 101) as f64 - 50.0) / 250.0
}

/// Case A feature row `i` (5 features).
fn xa_row(i: usize) -> [f64; 5] {
    let mut r = [0.0; 5];
    for (j, v) in r.iter_mut().enumerate() {
        *v = ((i * 31 + j * 17 + i * j) % 23) as f64 / 7.0;
    }
    r
}

/// Case A: `(x, y, x_new)` with 1000 training rows and 7 prediction rows (i in 1000..1007).
fn case_a() -> (NdArray<f64>, NdArray<f64>, NdArray<f64>) {
    let mut x = Vec::with_capacity(1000 * 5);
    let mut y = Vec::with_capacity(1000);
    for i in 0..1000 {
        let r = xa_row(i);
        x.extend_from_slice(&r);
        y.push(1.5 + 2.0 * r[0] - 1.25 * r[1] + 0.5 * r[2] + 3.0 * r[3] - 0.75 * r[4] + noise(i));
    }
    let mut xn = Vec::with_capacity(7 * 5);
    for i in 1000..1007 {
        xn.extend_from_slice(&xa_row(i));
    }
    (
        NdArray::new(x, vec![1000, 5]).unwrap(),
        NdArray::new(y, vec![1000]).unwrap(),
        NdArray::new(xn, vec![7, 5]).unwrap(),
    )
}

/// Case B feature row `i` (3 features, nearly collinear first two).
fn xb_row(i: usize) -> [f64; 3] {
    let b0 = ((i * 13 + 5) % 29) as f64 / 3.0;
    let b1 = b0 + (((i * 7 + 3) % 11) as f64 - 5.0) * 1e-4;
    let b2 = ((i * 5 + 2) % 17) as f64 / 4.0;
    [b0, b1, b2]
}

/// Case B: `(x, y, x_new)` with 1000 training rows and 7 prediction rows.
fn case_b() -> (NdArray<f64>, NdArray<f64>, NdArray<f64>) {
    let mut x = Vec::with_capacity(1000 * 3);
    let mut y = Vec::with_capacity(1000);
    for i in 0..1000 {
        let b = xb_row(i);
        x.extend_from_slice(&b);
        y.push(0.5 + 1.0 * b[0] - 0.5 * b[1] + 2.0 * b[2] + noise(i));
    }
    let mut xn = Vec::with_capacity(7 * 3);
    for i in 1000..1007 {
        xn.extend_from_slice(&xb_row(i));
    }
    (
        NdArray::new(x, vec![1000, 3]).unwrap(),
        NdArray::new(y, vec![1000]).unwrap(),
        NdArray::new(xn, vec![7, 3]).unwrap(),
    )
}

#[test]
fn ols_matches_lstsq() {
    let (x, y, xn) = case_a();
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    assert_close(model.coef(), "ols.coef", REL_TOL);
    assert_close(&[model.intercept()], "ols.intercept", REL_TOL);
    let pred = model.predict(&xn).unwrap();
    assert_close(pred.data(), "ols.predict", REL_TOL);
}

/// Fits `Ridge::new(alpha)` on case A and compares with the closed form
/// stored under `prefix`.
fn check_ridge_case_a(alpha: f64, prefix: &str) {
    let (x, y, xn) = case_a();
    let mut model = Ridge::new(alpha);
    model.fit(&x, &y).unwrap();
    assert_eq!(model.alpha(), alpha);
    assert_close(model.coef(), &format!("{prefix}.coef"), REL_TOL);
    assert_close(
        &[model.intercept()],
        &format!("{prefix}.intercept"),
        REL_TOL,
    );
    let pred = model.predict(&xn).unwrap();
    assert_close(pred.data(), &format!("{prefix}.predict"), REL_TOL);
}

#[test]
fn ridge_alpha_0p1_matches_closed_form() {
    check_ridge_case_a(0.1, "ridge_a0p1");
}

#[test]
fn ridge_alpha_1_matches_closed_form() {
    check_ridge_case_a(1.0, "ridge_a1");
}

#[test]
fn ridge_alpha_100_matches_closed_form() {
    check_ridge_case_a(100.0, "ridge_a100");
}

#[test]
fn ridge_ill_conditioned_matches_closed_form() {
    let (x, y, xn) = case_b();
    let mut model = Ridge::new(1e-4);
    model.fit(&x, &y).unwrap();
    assert_close(model.coef(), "ridge_ill.coef", ILL_REL_TOL);
    assert_close(&[model.intercept()], "ridge_ill.intercept", ILL_REL_TOL);
    let pred = model.predict(&xn).unwrap();
    assert_close(pred.data(), "ridge_ill.predict", ILL_REL_TOL);
}

/// Negative test: a `Ridge` that ignored `alpha` would equal OLS and fail here.
#[test]
fn ridge_penalty_changes_the_result() {
    let (x, y, _) = case_a();
    let mut ols = LinearRegression::new();
    ols.fit(&x, &y).unwrap();
    let mut ridge = Ridge::new(100.0);
    ridge.fit(&x, &y).unwrap();
    let moved = ols
        .coef()
        .iter()
        .zip(ridge.coef())
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(moved, "alpha = 100 must move at least one coefficient");
}
