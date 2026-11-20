//! Reference-value tests for fft, signal, integrate and ts.
//!
//! The reference values were made with NumPy 2.4.6, SciPy 1.17.1 and
//! statsmodels (ARIMA) from the same environment. The expression of each
//! value is in the `source` column of `fixtures/numeric_reference.tsv`.

use scikit::fft::{fft, ifft, rfft, Complex};
use scikit::integrate::{fixed_quad, solve_ivp, OdeSolver};
use scikit::signal::{butter, sosfilt, FilterType};
use scikit::ts::arima::{ArimaModel, ArimaOrder};

const FIXTURE: &str = include_str!("../fixtures/numeric_reference.tsv");

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

/// Every element of `actual` is within `rel_tol * max(|e|, 1.0)` of the fixture.
fn assert_vec_close(actual: &[f64], key: &str, rel_tol: f64) {
    let expected = reference(key);
    assert_eq!(actual.len(), expected.len(), "length of `{key}`");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = rel_tol * e.abs().max(1.0);
        assert!(
            (a - e).abs() <= tol,
            "`{key}`[{i}] = {a}, expected {e} (tol {tol})"
        );
    }
}

#[test]
fn fft_len3_constant_matches_numpy() {
    let x = vec![Complex::new(1.0, 0.0); 3];
    let y = fft(&x);
    assert_eq!(y.len(), 3);
    // numpy.fft.fft([1,1,1]) = [3, 0, 0]
    let want_re = [3.0, 0.0, 0.0];
    let want_im = [0.0, 0.0, 0.0];
    for (k, (yk, (wre, wim))) in y.iter().zip(want_re.iter().zip(want_im.iter())).enumerate() {
        assert!(
            (yk.re - wre).abs() <= 1e-12 && (yk.im - wim).abs() <= 1e-12,
            "X[{k}] = {yk:?}, expected ({wre}, {wim})"
        );
    }
}

#[test]
fn ifft_of_fft_is_identity_for_odd_and_even_lengths() {
    for n in [3usize, 5, 6, 7] {
        let x: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i % 4) as f64 * 1.5 - 1.0 + 0.25 * i as f64, 0.0))
            .collect();
        let back = ifft(&fft(&x));
        assert_eq!(back.len(), n, "length for n = {n}");
        for (i, (b, xi)) in back.iter().zip(x.iter()).enumerate() {
            assert!(
                (b.re - xi.re).abs() <= 1e-9 && (b.im - xi.im).abs() <= 1e-9,
                "n = {n}, i = {i}: got {b:?}, expected {xi:?}"
            );
        }
    }
}

#[test]
fn rfft_len5_matches_numpy() {
    let y = rfft(&[1.0, 2.0, 0.5, -1.0, 3.0]);
    assert_eq!(y.len(), 3);
    let re: Vec<f64> = y.iter().map(|c| c.re).collect();
    let im: Vec<f64> = y.iter().map(|c| c.im).collect();
    assert_vec_close(&re, "rfft_re", 1e-9);
    assert_vec_close(&im, "rfft_im", 1e-9);
}

#[test]
fn butter_lowpass_step_response_matches_scipy() {
    // Compare the filtered output, not the sections: the section layout
    // differs between kit and SciPy.
    for (order, key) in [
        (1usize, "butter1_step"),
        (2, "butter2_step"),
        (3, "butter3_step"),
        (5, "butter5_step"),
    ] {
        let sos = butter(order, 0.3, FilterType::Lowpass);
        let y = sosfilt(&sos, &[1.0; 12]);
        assert_vec_close(&y, key, 1e-9);
    }
}

#[test]
fn fixed_quad_orders() {
    let q = fixed_quad(|x| x.powi(4), 0.0, 1.0, 5).expect("n = 5 is supported");
    let want = reference("fixed_quad_x4_n5")[0];
    assert!(
        (q - want).abs() <= 1e-12,
        "fixed_quad = {q}, expected {want}"
    );
    assert!(fixed_quad(|x| x, 0.0, 1.0, 0).is_err(), "n = 0 must be Err");
    assert!(fixed_quad(|x| x, 0.0, 1.0, 6).is_err(), "n = 6 must be Err");
}

#[test]
fn rk45_decay_matches_exact_and_scipy() {
    let r = solve_ivp(
        |_t, y| vec![-y[0]],
        (0.0, 1.0),
        &[1.0],
        OdeSolver::RK45,
        0.1,
        1e-6,
    );
    assert!(r.success, "RK45 did not succeed");
    let y = r.y.last().expect("at least one output point")[0];
    let exact = reference("exp_minus_1")[0];
    let scipy = reference("rk45_scipy")[0];
    assert!(
        (y - exact).abs() <= 1e-6 * exact.abs(),
        "y(1) = {y}, expected exp(-1) = {exact}"
    );
    assert!(
        (y - scipy).abs() <= 1e-6 * scipy.abs(),
        "y(1) = {y}, expected SciPy solve_ivp = {scipy}"
    );
}

#[test]
fn arima_011_forecast_matches_statsmodels() {
    let a = reference("arima_a_series");
    let model = ArimaModel::fit(&a, ArimaOrder::new(0, 1, 1)).expect("fit ARIMA(0,1,1)");
    let forecast = model.forecast(&a, 5);
    // Tolerance 1e-3: kit estimates the MA term by conditional sum of squares,
    // statsmodels by exact maximum likelihood. This is a stated estimator
    // difference, not a defect. Measured largest relative error on main: 3e-4.
    assert_vec_close(&forecast, "arima_a_forecast_011", 1e-3);
}

#[test]
fn arima_120_forecast_matches_statsmodels() {
    let b = reference("arima_b_series");
    let model = ArimaModel::fit(&b, ArimaOrder::new(1, 2, 0)).expect("fit ARIMA(1,2,0)");
    let forecast = model.forecast(&b, 5);
    // Tolerance 1e-3: kit uses Yule-Walker, statsmodels uses maximum likelihood,
    // a stated estimator difference, not a defect. Measured largest relative
    // error on main: 3e-4. A forecast that ignores the second difference fails.
    assert_vec_close(&forecast, "arima_b_forecast_120", 1e-3);
}
