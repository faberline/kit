//! Reference-value tests for `scikit::stats`.
//!
//! Expected values come from SciPy 1.17.1 with NumPy 2.4.6. The expression of
//! each value is in the `source` column of `tests/fixtures/stats_reference.tsv`.
//! No Python runs in CI: the fixture is the only source of the expected values.

use scikit::stats::{
    chisquare, erf, f_oneway, ln_gamma, mannwhitneyu, pearsonr, spearmanr, ttest_1samp, ttest_ind,
    ChiSquared, ContinuousDistribution, Normal, StudentT,
};

const FIXTURE: &str = include_str!("../fixtures/stats_reference.tsv");

/// Return the expected value stored under `key` in the fixture.
fn reference(key: &str) -> f64 {
    for line in FIXTURE.lines().skip(1) {
        let mut fields = line.split('\t');
        if fields.next() == Some(key) {
            let value = fields
                .next()
                .unwrap_or_else(|| panic!("fixture key {key}: no value column"));
            return value
                .parse()
                .unwrap_or_else(|_| panic!("fixture key {key}: bad number {value}"));
        }
    }
    panic!("fixture key {key} is missing");
}

/// Check `actual` against the fixture value `key` with a relative tolerance.
fn assert_close(actual: f64, key: &str, rel_tol: f64) {
    let expected = reference(key);
    assert!(
        (actual - expected).abs() <= rel_tol * expected.abs().max(1e-300),
        "{key}: got {actual}, expected {expected}"
    );
}

/// The 400-value sample of `ttest_1samp n400` (same integer arithmetic as the fixture).
fn sample_400() -> Vec<f64> {
    (0..400)
        .map(|i| (((i % 20) * 10 + (i % 7) * 5) as f64) / 100.0)
        .collect()
}

const S8: [f64; 8] = [2.1, 3.4, 1.9, 5.6, 4.4, 3.8, 2.7, 4.9];
const B7: [f64; 7] = [1.2, 2.8, 2.2, 3.1, 2.5, 3.3, 1.8];

#[test]
fn ln_gamma_matches_scipy() {
    assert_close(ln_gamma(200.0), "lgamma_200", 1e-12);
    assert_close(ln_gamma(1000.0), "lgamma_1000", 1e-12);
    assert!(ln_gamma(200.0).is_finite());
    assert!(ln_gamma(1000.0).is_finite());
}

#[test]
fn erf_matches_scipy() {
    assert_close(erf(0.5), "erf_0p5", 1e-12);
    assert_close(erf(2.0), "erf_2", 1e-12);
}

#[test]
fn normal_cdf_matches_scipy() {
    assert_close(Normal::standard().cdf(1.96), "norm_cdf_1p96", 1e-10);
    assert_close(Normal::new(2.0, 3.0).cdf(5.5), "norm_cdf_5p5_mu2_s3", 1e-10);
}

#[test]
fn student_t_cdf_matches_scipy() {
    assert_close(StudentT::new(5.0).cdf(2.0), "t_cdf_2_df5", 1e-10);
    assert_close(StudentT::new(10.0).cdf(-1.5), "t_cdf_m1p5_df10", 1e-10);
}

#[test]
fn chi_squared_cdf_matches_scipy() {
    assert_close(ChiSquared::new(4.0).cdf(3.0), "chi2_cdf_3_df4", 1e-10);
    assert_close(ChiSquared::new(7.0).cdf(10.0), "chi2_cdf_10_df7", 1e-10);
}

#[test]
fn ttest_1samp_n400_is_finite_and_matches_scipy() {
    let x = sample_400();
    let result = ttest_1samp(&x, 1.0);
    assert!(result.statistic.is_finite(), "statistic is not finite");
    assert!(result.pvalue.is_finite(), "pvalue is not finite");
    assert_close(result.statistic, "t1_400_stat", 1e-8);
    assert_close(result.pvalue, "t1_400_p", 1e-8);
}

#[test]
fn ttest_1samp_small_matches_scipy() {
    let result = ttest_1samp(&S8, 3.0);
    assert_close(result.statistic, "t1_8_stat", 1e-8);
    assert_close(result.pvalue, "t1_8_p", 1e-8);
}

#[test]
fn ttest_ind_matches_scipy() {
    let eq = ttest_ind(&S8, &B7, true);
    assert_close(eq.statistic, "ind_eq_stat", 1e-8);
    assert_close(eq.pvalue, "ind_eq_p", 1e-8);

    let welch = ttest_ind(&S8, &B7, false);
    assert_close(welch.statistic, "ind_welch_stat", 1e-8);
    assert_close(welch.pvalue, "ind_welch_p", 1e-8);
}

#[test]
fn pearsonr_matches_scipy() {
    let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let ys = [2.1, 2.9, 4.2, 3.8, 6.1, 5.9, 7.7, 8.4];
    let result = pearsonr(&xs, &ys);
    assert_close(result.statistic, "pearson_r", 1e-8);
    assert_close(result.pvalue, "pearson_p", 1e-8);
}

#[test]
fn f_oneway_matches_scipy() {
    let g1 = [6.0, 8.0, 4.0, 5.0, 3.0, 4.0];
    let g2 = [8.0, 12.0, 9.0, 11.0, 6.0, 8.0];
    let g3 = [13.0, 9.0, 11.0, 8.0, 7.0, 12.0];
    let result = f_oneway(&[&g1[..], &g2[..], &g3[..]]);
    assert_close(result.statistic, "anova_f", 1e-8);
    assert_close(result.pvalue, "anova_p", 1e-8);
}

#[test]
fn chisquare_matches_scipy() {
    let observed = [16.0, 18.0, 16.0, 14.0, 12.0, 12.0];
    let expected = [16.0, 16.0, 16.0, 16.0, 16.0, 8.0];
    let result = chisquare(&observed, &expected);
    assert_close(result.statistic, "chisq_stat", 1e-8);
    assert_close(result.pvalue, "chisq_p", 1e-8);
}

#[test]
fn spearmanr_with_ties_matches_scipy() {
    let result = spearmanr(&[1.0, 1.0, 1.0, 2.0], &[1.0, 2.0, 3.0, 4.0]);
    assert_close(result.statistic, "spearman_rho", 1e-10);
}

#[test]
fn mannwhitneyu_with_ties_matches_scipy() {
    let x = [1.0, 2.0, 2.0, 3.0, 5.0, 6.0];
    let y = [2.0, 3.0, 4.0, 4.0, 7.0, 8.0, 9.0];
    let result = mannwhitneyu(&x, &y);
    assert_close(result.statistic, "mw_u", 1e-8);
    assert_close(result.pvalue, "mw_p", 1e-8);
}

#[test]
fn ttest_ind_tiny_input_does_not_panic() {
    assert!(ttest_ind(&[], &[], true).statistic.is_nan());
    assert!(ttest_ind(&[1.0], &[2.0], true).statistic.is_nan());
    assert!(ttest_ind(&[1.0], &[2.0], false).statistic.is_nan());
}

#[test]
fn chisquare_tiny_input_does_not_panic() {
    assert!(chisquare(&[], &[]).pvalue.is_nan());
    assert!(chisquare(&[1.0], &[1.0]).pvalue.is_nan());
}
