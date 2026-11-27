//! Tests for the degenerate-input guards of `stats::descriptive` and `stats::hypothesis`.

use scikit::stats::{
    describe, f_oneway, geometric_mean, harmonic_mean, kurtosis, median, median_abs_deviation,
    mode, moment, pearsonr, raw_moment, sem, skew, spearmanr, trim_mean, ttest_ind, variation,
    zscore, TestResult,
};

fn both_nan(r: TestResult) -> bool {
    r.statistic.is_nan() && r.pvalue.is_nan()
}

#[test]
fn skew_short_input_is_nan() {
    assert!(skew(&[1.0, 2.0]).is_nan());
}

#[test]
fn skew_zero_variance_is_zero() {
    assert_eq!(skew(&[1.0, 1.0, 1.0]), 0.0);
}

#[test]
fn kurtosis_short_input_is_nan() {
    assert!(kurtosis(&[1.0, 2.0, 3.0]).is_nan());
}

#[test]
fn kurtosis_zero_variance_is_zero() {
    assert_eq!(kurtosis(&[2.0, 2.0, 2.0, 2.0]), 0.0);
}

#[test]
fn zscore_single_value_is_copied() {
    assert_eq!(zscore(&[7.0]), vec![7.0]);
}

#[test]
fn zscore_zero_std_is_all_zero() {
    assert_eq!(zscore(&[2.0, 2.0, 2.0]), vec![0.0, 0.0, 0.0]);
}

#[test]
fn moment_empty_is_nan() {
    assert!(moment(&[], 2).is_nan());
}

#[test]
fn raw_moment_empty_is_nan() {
    assert!(raw_moment(&[], 2).is_nan());
}

#[test]
fn mode_empty_is_none() {
    assert_eq!(mode(&[]), None);
}

#[test]
fn median_empty_is_none() {
    assert_eq!(median(&[]), None);
}

#[test]
fn geometric_mean_empty_is_nan() {
    assert!(geometric_mean(&[]).is_nan());
}

#[test]
fn geometric_mean_non_positive_is_nan() {
    assert!(geometric_mean(&[1.0, 0.0, 4.0]).is_nan());
}

#[test]
fn harmonic_mean_empty_is_nan() {
    assert!(harmonic_mean(&[]).is_nan());
}

#[test]
fn harmonic_mean_non_positive_is_nan() {
    assert!(harmonic_mean(&[1.0, -2.0]).is_nan());
}

#[test]
fn trim_mean_empty_is_nan() {
    assert!(trim_mean(&[], 0.1).is_nan());
}

#[test]
fn trim_mean_cut_all_falls_back_to_median() {
    assert_eq!(trim_mean(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
}

#[test]
fn sem_single_value_is_nan() {
    assert!(sem(&[1.0]).is_nan());
}

#[test]
fn variation_empty_is_nan() {
    assert!(variation(&[]).is_nan());
}

#[test]
fn variation_zero_mean_is_nan() {
    assert!(variation(&[-1.0, 1.0]).is_nan());
}

#[test]
fn median_abs_deviation_empty_is_nan() {
    assert!(median_abs_deviation(&[]).is_nan());
}

#[test]
fn describe_empty() {
    let d = describe(&[]);
    assert_eq!(d.count, 0);
    assert!(d.mean.is_nan());
    assert!(d.std.is_nan());
    assert!(d.min.is_nan());
    assert!(d.q1.is_nan());
    assert!(d.median.is_nan());
    assert!(d.q3.is_nan());
    assert!(d.max.is_nan());
    assert!(d.skewness.is_nan());
    assert!(d.kurtosis.is_nan());
}

#[test]
fn describe_single_value() {
    let d = describe(&[5.0]);
    assert_eq!(d.count, 1);
    assert_eq!(d.mean, 5.0);
    assert_eq!(d.min, 5.0);
    assert_eq!(d.q1, 5.0);
    assert_eq!(d.median, 5.0);
    assert_eq!(d.q3, 5.0);
    assert_eq!(d.max, 5.0);
    assert!(d.std.is_nan());
    assert!(d.skewness.is_nan());
    assert!(d.kurtosis.is_nan());
}

#[test]
fn ttest_ind_welch_underflow_is_nan() {
    assert!(both_nan(ttest_ind(&[0.0, 2e-100], &[0.0, 2e-100], false)));
}

#[test]
fn f_oneway_single_group_is_nan() {
    assert!(both_nan(f_oneway(&[&[1.0, 2.0, 3.0]])));
}

#[test]
fn f_oneway_group_with_one_element_is_nan() {
    assert!(both_nan(f_oneway(&[&[1.0, 2.0], &[3.0]])));
}

#[test]
fn pearsonr_zero_variance_is_nan() {
    assert!(both_nan(pearsonr(&[1.0, 2.0, 3.0], &[5.0, 5.0, 5.0])));
}

#[test]
fn pearsonr_length_mismatch_is_nan() {
    assert!(both_nan(pearsonr(&[1.0, 2.0, 3.0], &[1.0, 2.0])));
}

#[test]
fn spearmanr_short_input_is_nan() {
    assert!(both_nan(spearmanr(&[1.0, 2.0], &[2.0, 1.0])));
}
