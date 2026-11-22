//! Reference-value tests for the `interpolate`, `optimize` and `integrate` rows
//! of `docs/reference-coverage.md`.
//!
//! The expected values are in `fixtures/calc_reference.tsv`. The generator
//! `tools/reference/scikit/generate.py` writes them from NumPy 2.4.6 and SciPy
//! 1.17.1; its `source` column names the expression of each value. The inputs
//! come from the integer formulas below, so Rust and Python build the same
//! values.

use scikit::integrate::{
    cumulative_trapezoid, euler, fixed_quad, quad, rk4, simps, solve_ivp, trapezoid, OdeSolver,
};
use scikit::interpolate::{interp1d, interp2d, CubicSpline, InterpKind};
use scikit::optimize::{bfgs, bisect, brentq, curve_fit, linprog, nelder_mead, newton};
use std::f64::consts::PI;

const FIXTURE: &str = include_str!("../fixtures/calc_reference.tsv");

/// Relative tolerance of a reference value with `|e| >= 1`; below 1 the same
/// number is an absolute tolerance (the table rows say which one applies).
const REL_TOL: f64 = 1e-9;

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

/// Every element of `actual` is within `tol * max(|e|, 1)` of the fixture
/// values of `key`.
fn check(actual: &[f64], key: &str, tol: f64) {
    let expected = reference(key);
    assert_eq!(actual.len(), expected.len(), "length of `{key}`");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let bound = tol * e.abs().max(1.0);
        assert!(
            (a - e).abs() <= bound,
            "`{key}`[{i}] = {a}, expected {e} (tol {bound})"
        );
    }
}

/// The generator's input `y[i] = ((i * 37 + 11) % 101) / 4.0`.
fn data(i: usize) -> f64 {
    ((i * 37 + 11) % 101) as f64 / 4.0
}

/// The interpolation grid `xs[i] = i * 0.5`, `ys[i] = data(i)` for 20 points.
fn interp_grid() -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let ys: Vec<f64> = (0..20).map(data).collect();
    (xs, ys)
}

/// The query points `xq[k] = (k * 9 + 3) / 20.0` for 21 points.
fn interp_queries() -> Vec<f64> {
    (0..21).map(|k| ((k * 9 + 3) as f64) / 20.0).collect()
}

/// The 2-D grid `xg[i] = i * 0.8`, `yg[j] = j * 0.5` and the values
/// `z[i][j] = ((i * 7 + j * 13 + 3) % 17) / 2.0`, row-major with `ny = 7`.
fn grid_2d() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let xg: Vec<f64> = (0..6).map(|i| i as f64 * 0.8).collect();
    let yg: Vec<f64> = (0..7).map(|j| j as f64 * 0.5).collect();
    let mut values = Vec::with_capacity(6 * 7);
    for i in 0..6 {
        for j in 0..7 {
            values.push(((i * 7 + j * 13 + 3) % 17) as f64 / 2.0);
        }
    }
    (xg, yg, values)
}

/// The 2-D query points `px[k] = ((k*13+2) % 39) / 10.0`, `py[k] = ((k*7+1) % 29) / 10.0`.
fn points_2d() -> Vec<(f64, f64)> {
    (0..12)
        .map(|k| {
            (
                ((k * 13 + 2) % 39) as f64 / 10.0,
                ((k * 7 + 1) % 29) as f64 / 10.0,
            )
        })
        .collect()
}

/// The fit data `x[i] = i * 0.25`, `y = 2.5 exp(-1.3 x) + 0.5 + 0.01 (((i*37+11)%101) - 50) / 50`.
fn fit_data() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.25).collect();
    let y: Vec<f64> = (0..40)
        .map(|i| {
            let xi = x[i];
            2.5 * (-1.3 * xi).exp() + 0.5 + 0.01 * (((i * 37 + 11) % 101) as f64 - 50.0) / 50.0
        })
        .collect();
    (x, y)
}

/// Rosenbrock function and its gradient.
fn rosen(x: &[f64]) -> f64 {
    100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
}

fn rosen_grad(x: &[f64]) -> Vec<f64> {
    vec![
        -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
        200.0 * (x[1] - x[0] * x[0]),
    ]
}

/// A convex bowl with its minimum at about (1.548, -2.194), and its gradient.
fn bowl(p: &[f64]) -> f64 {
    (p[0] - 1.0).powi(2) + 2.0 * (p[1] + 2.0).powi(2) + 0.5 * p[0] * p[1]
}

fn bowl_grad(p: &[f64]) -> Vec<f64> {
    vec![
        2.0 * (p[0] - 1.0) + 0.5 * p[1],
        4.0 * (p[1] + 2.0) + 0.5 * p[0],
    ]
}

fn cubic(x: f64) -> f64 {
    x * x * x - 2.0 * x - 5.0
}

fn cubic_der(x: f64) -> f64 {
    3.0 * x * x - 2.0
}

fn cos_minus_x(x: f64) -> f64 {
    x.cos() - x
}

fn cos_minus_x_der(x: f64) -> f64 {
    -x.sin() - 1.0
}

/// The system f(t, y) = [y1, -y0 + t] of the one-step tests.
fn step_system(t: f64, y: &[f64]) -> Vec<f64> {
    vec![y[1], -y[0] + t]
}

/// The oscillator y0' = y1, y1' = -y0 of the `solve_ivp` tests.
fn oscillator(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![y[1], -y[0]]
}

#[test]
fn interp1d_matches_scipy() {
    let (xs, ys) = interp_grid();
    let xq = interp_queries();
    check(
        &interp1d(&xs, &ys, &xq, InterpKind::Linear),
        "interp1d_linear",
        REL_TOL,
    );
    check(
        &interp1d(&xs, &ys, &xq, InterpKind::CubicSpline),
        "interp1d_cubic_natural",
        REL_TOL,
    );
}

#[test]
fn eval_matches_scipy() {
    let (xs, ys) = interp_grid();
    let spline = CubicSpline::new(&xs, &ys);
    let got: Vec<f64> = interp_queries()
        .into_iter()
        .map(|x| spline.eval(x))
        .collect();
    check(&got, "interp1d_cubic_natural", REL_TOL);
}

#[test]
fn interp2d_matches_scipy() {
    let (xg, yg, values) = grid_2d();
    let got: Vec<f64> = points_2d()
        .into_iter()
        .map(|(px, py)| interp2d(&xg, &yg, &values, px, py).unwrap())
        .collect();
    check(&got, "interp2d_linear", REL_TOL);
}

#[test]
fn curve_fit_matches_scipy() {
    let (x, y) = fit_data();
    let fit = curve_fit(
        |xi, p| p[0] * (-p[1] * xi).exp() + p[2],
        &x,
        &y,
        &[2.0, 1.0, 0.4],
        200,
        1e-12,
    )
    .unwrap();
    check(&fit.params, "curve_fit_params", 1e-6);
}

#[test]
fn linprog_matches_scipy() {
    let a = linprog(
        &[-1.0, -2.0],
        &[1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        &[4.0, 3.0, 3.0],
    )
    .unwrap();
    assert!(a.success);
    check(&a.x, "linprog_a_x", REL_TOL);
    check(&[a.fun], "linprog_a_fun", REL_TOL);

    let b = linprog(
        &[-5.0, -4.0, -3.0],
        &[2.0, 3.0, 1.0, 4.0, 1.0, 2.0, 3.0, 4.0, 2.0],
        &[5.0, 11.0, 8.0],
    )
    .unwrap();
    assert!(b.success);
    check(&b.x, "linprog_b_x", REL_TOL);
    check(&[b.fun], "linprog_b_fun", REL_TOL);
}

#[test]
fn bfgs_matches_scipy() {
    let r = bfgs(rosen, rosen_grad, &[-1.2, 1.0], 1000, 1e-10).unwrap();
    check(&r.x, "rosen_bfgs_x", 1e-6);
    check(&[r.fun], "rosen_bfgs_fun", 1e-6);

    let b = bfgs(bowl, bowl_grad, &[0.0, 0.0], 1000, 1e-10).unwrap();
    check(&b.x, "bowl_bfgs_x", 1e-6);
}

#[test]
fn nelder_mead_matches_scipy() {
    let r = nelder_mead(rosen, &[-1.2, 1.0], 20000, 1e-12).unwrap();
    check(&r.x, "rosen_nm_x", 1e-6);
    check(&[r.fun], "rosen_nm_fun", 1e-6);

    let b = nelder_mead(bowl, &[0.0, 0.0], 20000, 1e-12).unwrap();
    check(&b.x, "bowl_nm_x", 1e-6);
}

#[test]
fn brentq_matches_scipy() {
    let cubic_root = brentq(cubic, 2.0, 3.0, 1e-14, 200).unwrap();
    check(&[cubic_root], "root_cubic_brentq", REL_TOL);
    let cos_root = brentq(cos_minus_x, 0.0, 1.0, 1e-14, 200).unwrap();
    check(&[cos_root], "root_cos_brentq", REL_TOL);
}

#[test]
fn bisect_matches_scipy() {
    let cubic_root = bisect(cubic, 2.0, 3.0, 1e-14, 200).unwrap();
    check(&[cubic_root], "root_cubic_bisect", REL_TOL);
    let cos_root = bisect(cos_minus_x, 0.0, 1.0, 1e-14, 200).unwrap();
    check(&[cos_root], "root_cos_bisect", REL_TOL);
}

#[test]
fn newton_matches_scipy() {
    let cubic_root = newton(cubic, cubic_der, 2.0, 1e-14, 100).unwrap();
    check(&[cubic_root], "root_cubic_newton", REL_TOL);
    let cos_root = newton(cos_minus_x, cos_minus_x_der, 0.5, 1e-14, 100).unwrap();
    check(&[cos_root], "root_cos_newton", REL_TOL);
}

#[test]
fn quad_matches_scipy() {
    let cases = [
        ("quad_sin", quad(f64::sin, 0.0, PI, 1e-12).value),
        (
            "quad_gauss",
            quad(|x| (-x * x).exp(), 0.0, 2.0, 1e-12).value,
        ),
        (
            "quad_lorentz",
            quad(|x| 1.0 / (1.0 + x * x), 0.0, 1.0, 1e-12).value,
        ),
        (
            "quad_xlog",
            quad(|x| x * (x + 1.0).ln(), 0.0, 2.0, 1e-12).value,
        ),
        (
            "quad_damped",
            quad(|x| (-x).exp() * x.sin(), 0.0, 10.0, 1e-12).value,
        ),
    ];
    for (key, value) in cases {
        check(&[value], key, REL_TOL);
    }
}

#[test]
fn fixed_quad_matches_scipy() {
    for n in 1..=5usize {
        let exp = fixed_quad(f64::exp, 0.5, 2.5, n).unwrap();
        check(&[exp], &format!("fixed_quad_exp_n{n}"), REL_TOL);
        let cos = fixed_quad(|x| (3.0 * x).cos(), 0.5, 2.5, n).unwrap();
        check(&[cos], &format!("fixed_quad_cos_n{n}"), REL_TOL);
    }
}

/// The data `y[i] = data(i)` for 41 points (40 intervals, an even count).
fn integral_data() -> Vec<f64> {
    (0..41).map(data).collect()
}

#[test]
fn trapezoid_matches_scipy() {
    check(
        &[trapezoid(&integral_data(), 0.25)],
        "trapezoid_data",
        REL_TOL,
    );
}

#[test]
fn simps_matches_scipy() {
    check(&[simps(&integral_data(), 0.25)], "simpson_data", REL_TOL);
}

#[test]
fn cumulative_trapezoid_matches_scipy() {
    let got = cumulative_trapezoid(&integral_data(), 0.25);
    check(&got, "cumtrapz_data", REL_TOL);
}

#[test]
fn euler_matches_formula() {
    let got = euler(&step_system, 0.3, &[1.0, 0.5], 0.1);
    check(&got, "euler_step", 1e-12);
}

#[test]
fn rk4_matches_formula() {
    let got = rk4(&step_system, 0.3, &[1.0, 0.5], 0.1);
    check(&got, "rk4_step", 1e-12);
}

#[test]
fn solve_ivp_matches_scipy() {
    let exp_rhs = |_t: f64, y: &[f64]| vec![-y[0]];

    // Each entry: solver, max step, rtol, tolerance of the table row.
    let exp_checks = [
        (OdeSolver::Euler, 0.001, 1e-6, 5e-3),
        (OdeSolver::RK4, 0.01, 1e-6, 1e-8),
        (OdeSolver::RK45, 0.1, 1e-6, 1e-5),
    ];
    for (solver, max_step, rtol, tol) in exp_checks {
        let r = solve_ivp(exp_rhs, (0.0, 1.0), &[1.0], solver, max_step, rtol);
        assert!(r.success, "solve_ivp exp did not succeed");
        assert!((r.t.last().unwrap() - 1.0).abs() < 1e-9, "final t");
        check(r.y.last().unwrap(), "ivp_exp", tol);
    }

    for (solver, max_step, rtol, tol) in exp_checks {
        let r = solve_ivp(oscillator, (0.0, 1.0), &[1.0, 0.0], solver, max_step, rtol);
        assert!(r.success, "solve_ivp oscillator did not succeed");
        assert!((r.t.last().unwrap() - 1.0).abs() < 1e-9, "final t");
        check(r.y.last().unwrap(), "ivp_osc", tol);
    }
}
