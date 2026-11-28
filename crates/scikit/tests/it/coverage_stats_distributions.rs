//! Coverage tests for `stats::distributions` (edge branches of the distributions and special functions).

use scikit::stats::{
    gamma_fn, incomplete_beta, ln_gamma, lower_incomplete_gamma, Beta, ContinuousDistribution,
    Exponential, FDistribution, Normal, StudentT,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b}");
}

#[test]
fn default_std_is_sqrt_of_variance() {
    assert_eq!(Normal::new(0.0, 2.0).std(), 2.0);
}

#[test]
fn exponential_negative_x_is_zero() {
    let d = Exponential::new(2.0);
    assert_eq!(d.pdf(-1.0), 0.0);
    assert_eq!(d.cdf(-1.0), 0.0);
}

#[test]
fn beta_pdf_outside_unit_interval_is_zero() {
    let d = Beta::new(2.0, 3.0);
    assert_eq!(d.pdf(1.5), 0.0);
    assert_eq!(d.pdf(-0.1), 0.0);
}

#[test]
fn student_t_moments_small_df() {
    assert!(StudentT::new(1.0).mean().is_nan());
    assert_eq!(StudentT::new(1.5).variance(), f64::INFINITY);
    assert!(StudentT::new(1.0).variance().is_nan());
    assert_eq!(StudentT::new(3.0).mean(), 0.0);
}

#[test]
fn f_distribution_moments_small_d2() {
    assert!(FDistribution::new(3.0, 2.0).mean().is_nan());
    assert!(FDistribution::new(3.0, 4.0).variance().is_nan());
}

#[test]
fn gamma_fn_pole_is_infinity() {
    assert_eq!(gamma_fn(0.0), f64::INFINITY);
    assert_eq!(gamma_fn(-2.0), f64::INFINITY);
}

#[test]
fn gamma_fn_negative_non_integer_uses_reflection() {
    close(gamma_fn(-0.5), -2.0 * std::f64::consts::PI.sqrt(), 1e-12);
}

#[test]
fn ln_gamma_below_half_uses_reflection() {
    close(ln_gamma(0.25), gamma_fn(0.25).ln(), 1e-12);
}

#[test]
fn lower_incomplete_gamma_non_positive_x_is_zero() {
    assert_eq!(lower_incomplete_gamma(1.0, 0.0), 0.0);
    assert_eq!(lower_incomplete_gamma(1.0, -1.0), 0.0);
}

#[test]
fn lower_incomplete_gamma_not_converged_is_nan() {
    assert!(lower_incomplete_gamma(1.0, 20000.0).is_nan());
}

#[test]
fn incomplete_beta_tiny_x_huge_b_is_one() {
    // a = 5, b = 1e30, x = 1e-20: b * x = 1e10, far above a, so the true value is 1
    // (1 - I = I_{1-x}(b, a) < exp(-1e10)). Reaches the three `1e-30` guards that
    // `incomplete_beta` can reach: the first `d` guard, and the `d` and `c` guards
    // of the second half of the loop in `beta_cf`.
    close(incomplete_beta(5.0, 1e30, 1e-20), 1.0, 1e-12);
}
