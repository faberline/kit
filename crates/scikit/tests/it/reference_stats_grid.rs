//! Reference tests for the functions of `scikit::stats` that the grid fixture covers.
//! Each expected value is read from `fixtures/stats_grid_reference.tsv`, which
//! `tools/reference/scikit/generate.py` writes from SciPy 1.17.1 and NumPy 2.4.6.
//! The inputs below are the same integer formulas as in the generator.

use scikit::stats::{
    beta_fn, binomial_coef, chi2_contingency, chisquare, describe, erf, f_oneway, factorial,
    gamma_fn, geometric_mean, harmonic_mean, incomplete_beta, iqr, kstest_normal, kurtosis,
    ln_gamma, lower_incomplete_gamma, mannwhitneyu, median, median_abs_deviation, mode, moment,
    pearsonr, percentile, raw_moment, sem, skew, spearmanr, trim_mean, ttest_1samp, ttest_ind,
    ttest_rel, variation, zscore, Beta, Binomial, ChiSquared, ContinuousDistribution,
    DiscreteDistribution, Exponential, FDistribution, Gamma, Geometric, Normal, Poisson, StudentT,
    TestResult, Uniform,
};

const FIXTURE: &str = include_str!("../fixtures/stats_grid_reference.tsv");

const GRID_SYM: [f64; 9] = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0];
const GRID_POS: [f64; 9] = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
const GRID_UNIT: [f64; 9] = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0];
const GRID_UNI: [f64; 9] = [-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.99, 5.0, 6.0];
const GRID_K: [u64; 7] = [0, 1, 2, 5, 10, 20, 30];

/// The text of the value stored under `key`.
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

/// The expected value stored under `key`.
fn reference(key: &str) -> f64 {
    let value = raw(key);
    value
        .parse()
        .unwrap_or_else(|_| panic!("fixture key {key}: bad number {value}"))
}

/// The expected list stored under `key`: comma separated, a NaN written as `nan`.
fn reference_list(key: &str) -> Vec<f64> {
    raw(key)
        .split(',')
        .map(|item| {
            if item == "nan" {
                f64::NAN
            } else {
                item.parse()
                    .unwrap_or_else(|_| panic!("fixture key {key}: bad number {item}"))
            }
        })
        .collect()
}

/// Check `actual` against the fixture value `key` with a relative tolerance.
fn assert_close(actual: f64, key: &str, rel_tol: f64) {
    let expected = reference(key);
    assert!(
        (actual - expected).abs() <= rel_tol * expected.abs().max(1e-300),
        "{key}: got {actual}, expected {expected}"
    );
}

/// Check `actual` against the fixture value `key` with a relative and an absolute tolerance.
fn assert_close_abs(actual: f64, key: &str, rel_tol: f64, abs_tol: f64) {
    let expected = reference(key);
    assert!(
        (actual - expected).abs() <= rel_tol * expected.abs() + abs_tol,
        "{key}: got {actual}, expected {expected}"
    );
}

/// Check each value of `actual` against the fixture list `key` (relative tolerance 1e-9).
fn assert_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "{key}: length");
    for (pos, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let same = if e.is_nan() {
            a.is_nan()
        } else {
            (a - e).abs() <= 1e-9 * e.abs().max(1e-300)
        };
        assert!(same, "{key}: position {pos}: got {a}, expected {e}");
    }
}

/// The 500-value sample X: values 1.0 to 26.0, with ties.
fn sample_x() -> Vec<f64> {
    (0..500i64)
        .map(|i| ((i * 37 + 11) % 101) as f64 / 4.0 + 1.0)
        .collect()
}

/// The 500-value sample Y.
fn sample_y() -> Vec<f64> {
    (0..500i64)
        .map(|i| ((i * 53 + 7) % 97) as f64 / 4.0 + 2.0)
        .collect()
}

/// The 500-value sample Z.
fn sample_z() -> Vec<f64> {
    (0..500i64)
        .map(|i| ((i * 29 + 3) % 89) as f64 / 4.0 + 1.5)
        .collect()
}

/// Sample X with six more values of 13.5, so that 13.5 is the unique mode.
fn sample_m() -> Vec<f64> {
    let mut m = sample_x();
    m.extend([13.5; 6]);
    m
}

/// The 30 observed counts of the chi-square goodness-of-fit test.
fn observed_o() -> Vec<f64> {
    (0..30i64)
        .map(|i| ((i * 7 + 3) % 23) as f64 + 10.0)
        .collect()
}

/// The 3 by 4 table of the contingency test, row by row.
fn table_3x4() -> Vec<f64> {
    (0..3i64)
        .flat_map(|r| (0..4i64).map(move |c| (((r * 5 + c * 3) % 11) + 8) as f64))
        .collect()
}

/// The 2 by 2 table of the contingency test, row by row.
fn table_2x2() -> Vec<f64> {
    vec![12.0, 5.0, 7.0, 19.0]
}

/// Check a continuous distribution on a grid: cdf and pdf at each point, mean and variance.
fn check_continuous<D: ContinuousDistribution>(dist: &D, name: &str, grid: &[f64]) {
    for (i, &x) in grid.iter().enumerate() {
        assert_close_abs(dist.cdf(x), &format!("{name}_cdf_{i}"), 1e-9, 1e-12);
        assert_close_abs(dist.pdf(x), &format!("{name}_pdf_{i}"), 1e-9, 1e-12);
    }
    assert_close_abs(dist.mean(), &format!("{name}_mean"), 1e-9, 1e-12);
    assert_close_abs(dist.variance(), &format!("{name}_var"), 1e-9, 1e-12);
}

/// Check a discrete distribution on GRID_K: pmf and cdf at each point, mean and variance.
fn check_discrete<D: DiscreteDistribution>(dist: &D, name: &str) {
    for (i, &k) in GRID_K.iter().enumerate() {
        assert_close_abs(dist.pmf(k), &format!("{name}_pmf_{i}"), 1e-9, 1e-12);
        assert_close_abs(dist.cdf(k), &format!("{name}_cdf_{i}"), 1e-9, 1e-12);
    }
    assert_close_abs(dist.mean(), &format!("{name}_mean"), 1e-9, 1e-12);
    assert_close_abs(dist.variance(), &format!("{name}_var"), 1e-9, 1e-12);
}

// ============================================================================
// A. Descriptive functions
// ============================================================================

#[test]
fn skew_matches_scipy() {
    assert_close(skew(&sample_x()), "skew", 1e-9);
}

#[test]
fn kurtosis_matches_scipy() {
    assert_close(kurtosis(&sample_x()), "kurtosis", 1e-9);
}

#[test]
fn zscore_matches_scipy() {
    let z = zscore(&sample_x());
    assert_eq!(z.len(), 500);
    assert_list(&z[..5], "zscore_first5");
    assert_list(&z[495..], "zscore_last5");
    let sumsq: f64 = z.iter().map(|v| v * v).sum();
    assert_close(sumsq, "zscore_sumsq", 1e-9);
}

#[test]
fn moment_matches_scipy() {
    let x = sample_x();
    for k in [2, 3, 4] {
        assert_close(moment(&x, k), &format!("moment_{k}"), 1e-9);
    }
}

#[test]
fn raw_moment_matches_numpy() {
    let x = sample_x();
    for k in [2, 3] {
        assert_close(raw_moment(&x, k), &format!("raw_moment_{k}"), 1e-9);
    }
}

#[test]
fn mode_matches_scipy() {
    // 13.5 has at least 10 copies; every other value has at most 5.
    assert_eq!(mode(&sample_m()), Some(13.5));
    assert_close(mode(&sample_m()).unwrap(), "mode", 1e-9);
}

#[test]
fn median_matches_numpy() {
    assert_close(median(&sample_x()).unwrap(), "median", 1e-9);
}

#[test]
fn iqr_matches_scipy() {
    assert_close(iqr(&sample_x()).unwrap(), "iqr", 1e-9);
}

#[test]
fn percentile_matches_numpy() {
    let x = sample_x();
    for p in [0, 10, 25, 50, 90, 100] {
        let actual = percentile(&x, p as f64).unwrap();
        assert_close(actual, &format!("percentile_{p}"), 1e-9);
    }
}

#[test]
fn geometric_mean_matches_scipy() {
    assert_close(geometric_mean(&sample_x()), "gmean", 1e-9);
}

#[test]
fn harmonic_mean_matches_scipy() {
    assert_close(harmonic_mean(&sample_x()), "hmean", 1e-9);
}

#[test]
fn trim_mean_matches_scipy() {
    let x = sample_x();
    assert_close(trim_mean(&x, 0.1), "trim_0p1", 1e-9);
    assert_close(trim_mean(&x, 0.25), "trim_0p25", 1e-9);
}

#[test]
fn sem_matches_scipy() {
    assert_close(sem(&sample_x()), "sem", 1e-9);
}

#[test]
fn variation_matches_scipy() {
    assert_close(variation(&sample_x()), "variation", 1e-9);
}

#[test]
fn median_abs_deviation_matches_scipy() {
    assert_close(median_abs_deviation(&sample_x()), "mad", 1e-9);
}

#[test]
fn describe_matches_numpy_scipy() {
    let d = describe(&sample_x());
    assert_eq!(d.count, 500);
    assert_close(d.mean, "describe_mean", 1e-9);
    assert_close(d.std, "describe_std", 1e-9);
    assert_close(d.min, "describe_min", 1e-9);
    assert_close(d.q1, "describe_q1", 1e-9);
    assert_close(d.median, "describe_median", 1e-9);
    assert_close(d.q3, "describe_q3", 1e-9);
    assert_close(d.max, "describe_max", 1e-9);
    assert_close(d.skewness, "describe_skew", 1e-9);
    assert_close(d.kurtosis, "describe_kurt", 1e-9);
}

// ============================================================================
// B. Special functions
// ============================================================================

#[test]
fn erf_matches_scipy() {
    for (i, x) in [-3.0, -1.0, -0.1, 0.1, 1.0, 3.0, 5.0]
        .into_iter()
        .enumerate()
    {
        assert_close_abs(erf(x), &format!("erf_{i}"), 1e-12, 1e-15);
    }
}

#[test]
fn gamma_fn_matches_scipy() {
    for (i, x) in [0.5, 1.5, 2.5, 5.5, 10.5, 20.5].into_iter().enumerate() {
        assert_close(gamma_fn(x), &format!("gamma_fn_{i}"), 1e-9);
    }
}

#[test]
fn ln_gamma_matches_scipy() {
    for (i, x) in [0.5, 1.5, 10.5, 100.5, 500.5].into_iter().enumerate() {
        assert_close(ln_gamma(x), &format!("ln_gamma_{i}"), 1e-12);
    }
}

#[test]
fn beta_fn_matches_scipy() {
    for (i, (a, b)) in [(0.5, 0.5), (2.0, 3.0), (10.5, 3.5), (50.0, 60.0)]
        .into_iter()
        .enumerate()
    {
        assert_close(beta_fn(a, b), &format!("beta_fn_{i}"), 1e-9);
    }
}

#[test]
fn lower_incomplete_gamma_matches_scipy() {
    // Unregularized: SciPy's gammainc(s, x) * gamma(s).
    let cases = [
        (0.5, 0.1),
        (2.0, 1.0),
        (5.0, 3.0),
        (5.0, 12.0),
        (20.0, 15.0),
    ];
    for (i, (s, x)) in cases.into_iter().enumerate() {
        assert_close(
            lower_incomplete_gamma(s, x),
            &format!("lower_inc_gamma_{i}"),
            1e-9,
        );
    }
}

#[test]
fn incomplete_beta_matches_scipy() {
    let cases = [
        (2.0, 3.0, 0.4),
        (0.5, 0.5, 0.3),
        (10.0, 5.0, 0.7),
        (1.0, 1.0, 0.9),
        (30.0, 40.0, 0.45),
    ];
    for (i, (a, b, x)) in cases.into_iter().enumerate() {
        assert_close(
            incomplete_beta(a, b, x),
            &format!("incomplete_beta_{i}"),
            1e-9,
        );
    }
}

#[test]
fn factorial_matches_python() {
    for (i, n) in [0u64, 1, 5, 10, 20, 25].into_iter().enumerate() {
        assert_close(factorial(n), &format!("factorial_{i}"), 1e-15);
    }
}

#[test]
fn binomial_coef_matches_python() {
    let cases = [(5u64, 2u64), (10, 0), (10, 10), (20, 10), (50, 25), (6, 7)];
    for (i, (n, k)) in cases.into_iter().enumerate() {
        assert_close(binomial_coef(n, k), &format!("binomial_coef_{i}"), 1e-12);
    }
}

// ============================================================================
// C. Distributions (cdf, pdf, mean and variance on a grid with both tails)
// ============================================================================

#[test]
fn normal_matches_scipy() {
    check_continuous(&Normal::new(1.5, 2.0), "normal", &GRID_SYM);
}

#[test]
fn normal_standard_matches_scipy() {
    check_continuous(&Normal::standard(), "stdnormal", &GRID_SYM);
}

#[test]
fn uniform_matches_scipy() {
    check_continuous(&Uniform::new(-3.0, 5.0), "uniform", &GRID_UNI);
}

#[test]
fn exponential_matches_scipy() {
    check_continuous(&Exponential::new(0.5), "expon", &GRID_POS);
}

#[test]
fn gamma_matches_scipy() {
    check_continuous(&Gamma::new(2.5, 1.5), "gamma", &GRID_POS);
}

#[test]
fn beta_matches_scipy() {
    check_continuous(&Beta::new(2.0, 5.0), "beta", &GRID_UNIT);
}

#[test]
fn student_t_matches_scipy() {
    check_continuous(&StudentT::new(5.0), "t5", &GRID_SYM);
    check_continuous(&StudentT::new(30.0), "t30", &GRID_SYM);
}

#[test]
fn chi_squared_matches_scipy() {
    check_continuous(&ChiSquared::new(4.0), "chi2", &GRID_POS);
}

#[test]
fn f_distribution_matches_scipy() {
    check_continuous(&FDistribution::new(5.0, 10.0), "f", &GRID_POS);
}

#[test]
fn binomial_matches_scipy() {
    check_discrete(&Binomial::new(20, 0.3), "binom");
}

#[test]
fn poisson_matches_scipy() {
    check_discrete(&Poisson::new(4.5), "poisson");
}

#[test]
fn geometric_matches_scipy() {
    check_discrete(&Geometric::new(0.2), "geom");
}

// ============================================================================
// D. Hypothesis tests on 500-value samples with ties
// ============================================================================

#[test]
fn ttest_1samp_matches_scipy() {
    let r = ttest_1samp(&sample_x(), 13.0);
    assert_close(r.statistic, "ttest1_stat", 1e-9);
    assert_close(r.pvalue, "ttest1_p", 1e-9);
}

#[test]
fn ttest_ind_matches_scipy() {
    let (x, y) = (sample_x(), sample_y());
    let eq = ttest_ind(&x, &y, true);
    assert_close(eq.statistic, "ttest_ind_eq_stat", 1e-9);
    assert_close(eq.pvalue, "ttest_ind_eq_p", 1e-9);
    let welch = ttest_ind(&x, &y, false);
    assert_close(welch.statistic, "ttest_ind_welch_stat", 1e-9);
    assert_close(welch.pvalue, "ttest_ind_welch_p", 1e-9);
}

#[test]
fn ttest_rel_matches_scipy() {
    let r = ttest_rel(&sample_x(), &sample_y());
    assert_close(r.statistic, "ttest_rel_stat", 1e-9);
    assert_close(r.pvalue, "ttest_rel_p", 1e-9);
}

#[test]
fn f_oneway_matches_scipy() {
    let (x, y, z) = (sample_x(), sample_y(), sample_z());
    let r = f_oneway(&[&x[..], &y[..], &z[..]]);
    assert_close(r.statistic, "anova_stat", 1e-9);
    assert_close(r.pvalue, "anova_p", 1e-9);
}

#[test]
fn chisquare_matches_scipy() {
    let o = observed_o();
    let expected = vec![o.iter().sum::<f64>() / 30.0; 30];
    let r = chisquare(&o, &expected);
    assert_close(r.statistic, "chisq_stat", 1e-9);
    assert_close(r.pvalue, "chisq_p", 1e-9);
}

#[test]
fn chi2_contingency_matches_scipy() {
    // scikit has no Yates correction: SciPy is called with correction=False.
    let r = chi2_contingency(&table_3x4(), 3, 4);
    assert_close(r.statistic, "chi2c_3x4_stat", 1e-9);
    assert_close(r.pvalue, "chi2c_3x4_p", 1e-9);
    let r = chi2_contingency(&table_2x2(), 2, 2);
    assert_close(r.statistic, "chi2c_2x2_stat", 1e-9);
    assert_close(r.pvalue, "chi2c_2x2_p", 1e-9);
}

#[test]
fn pearsonr_matches_scipy() {
    let r = pearsonr(&sample_x(), &sample_y());
    assert_close(r.statistic, "pearson_stat", 1e-9);
    assert_close(r.pvalue, "pearson_p", 1e-9);
}

#[test]
fn spearmanr_matches_scipy() {
    let r = spearmanr(&sample_x(), &sample_y());
    assert_close(r.statistic, "spearman_stat", 1e-9);
    assert_close(r.pvalue, "spearman_p", 1e-9);
}

#[test]
fn mannwhitneyu_matches_scipy() {
    // scikit returns min(U1, U2), the same value as the fixture's mw_stat.
    let r = mannwhitneyu(&sample_x(), &sample_y());
    assert_close(r.statistic, "mw_stat", 1e-9);
    assert_close(r.pvalue, "mw_p", 1e-9);
}

#[test]
fn kstest_normal_matches_scipy() {
    let r = kstest_normal(&sample_x());
    assert_close(r.statistic, "ks_stat", 1e-9);
    // scikit keeps only the first term of the Kolmogorov series; SciPy sums all terms.
    // On this sample the difference is about 1.7e-7 (absolute), so the check is 1e-6.
    assert_close_abs(r.pvalue, "ks_p_asym", 0.0, 1e-6);
}

#[test]
fn is_significant_follows_the_pvalue_rule() {
    let p = reference("ttest1_p");
    for alpha in [0.001, 0.01, 0.05, 0.1, 0.5] {
        assert_eq!(TestResult::new(0.0, p).is_significant(alpha), p < alpha);
    }
    assert!(!TestResult::new(0.0, 0.05).is_significant(0.05));
}
