//! Coverage tests for stats.rs and math.rs of arraykit.

use arraykit::array::{ArrayError, FloatOps, NdArray, StatOps, VarOps};

/// The error of a result that must fail. Panics when the result is Ok.
fn expect_err<T>(result: Result<T, ArrayError>) -> ArrayError {
    match result {
        Err(error) => error,
        Ok(_) => panic!("expected an error, got Ok"),
    }
}

fn empty() -> NdArray<f64> {
    NdArray::<f64>::new(Vec::new(), vec![0]).unwrap()
}

fn v(xs: &[f64]) -> NdArray<f64> {
    NdArray::new(xs.to_vec(), vec![xs.len()]).unwrap()
}

fn m() -> NdArray<f64> {
    // [[1, 2, 3], [4, 5, 6]]
    NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

// ---- trait impls ----

#[test]
fn stat_ops_scalar_conversions() {
    assert_eq!(<f32 as StatOps>::from_usize(3), 3.0f32);
    assert_eq!(<f64 as StatOps>::to_f64(2.5), 2.5);
    assert_eq!(<i64 as StatOps>::from_usize(7), 7i64);
}

#[test]
fn var_ops_f32_sqrt() {
    assert_eq!(<f32 as VarOps>::sqrt_val(9.0f32), 3.0f32);
}

#[test]
fn float_ops_f32_all_methods() {
    assert_eq!(<f32 as FloatOps>::sqrt_val(4.0), 2.0);
    assert_eq!(<f32 as FloatOps>::exp_val(0.0), 1.0);
    assert_eq!(<f32 as FloatOps>::ln_val(1.0), 0.0);
    assert_eq!(<f32 as FloatOps>::abs_val(-2.5), 2.5);
    assert_eq!(<f32 as FloatOps>::from_f64(1.5), 1.5f32);
    assert_eq!(<f32 as FloatOps>::to_f64(1.5f32), 1.5f64);
}

// ---- math.rs ----

#[test]
fn linspace_zero_points_is_empty() {
    let points = NdArray::<f64>::linspace(0.0, 1.0, 0);
    assert_eq!(points.dims(), &[0]);
    assert!(points.data().is_empty());
}

#[test]
fn where_cond_length_mismatch() {
    let error = expect_err(v(&[1.0, 2.0, 3.0]).where_cond(&[true, false], &v(&[9.0, 9.0, 9.0])));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![3],
            got: vec![2]
        }
    );
}

#[test]
fn abs_int_i64() {
    let a = NdArray::new(vec![-3i64, 0, 4], vec![3]).unwrap();
    assert_eq!(a.abs_int().data(), &[3, 0, 4]);
}

// ---- axis errors and edge shapes ----

#[test]
fn sum_axis_out_of_range() {
    let error = expect_err(m().sum_axis(2));
    assert_eq!(
        error,
        ArrayError::IndexOutOfBounds {
            index: vec![2],
            shape: vec![2, 3]
        }
    );
}

#[test]
fn mean_axis_out_of_range() {
    let error = expect_err(m().mean_axis(2));
    assert_eq!(
        error,
        ArrayError::IndexOutOfBounds {
            index: vec![2],
            shape: vec![2, 3]
        }
    );
}

#[test]
fn var_axis_out_of_range() {
    let error = expect_err(m().var_axis(2));
    assert_eq!(
        error,
        ArrayError::IndexOutOfBounds {
            index: vec![2],
            shape: vec![2, 3]
        }
    );
}

#[test]
fn sum_axis_empty_output() {
    let zero_rows = NdArray::<f64>::new(Vec::new(), vec![0, 3]).unwrap();
    let sums = zero_rows.sum_axis(1).unwrap();
    assert_eq!(sums.dims(), &[0]);
    assert!(sums.data().is_empty());
}

#[test]
fn var_axis_single_element_slices() {
    let column = NdArray::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let vars = column.var_axis(1).unwrap();
    assert_eq!(vars.dims(), &[2]);
    assert_eq!(vars.data(), &[0.0, 0.0]);
}

// ---- empty and tiny input ----

#[test]
fn argmin_empty_is_none() {
    assert_eq!(empty().argmin(), None);
}

#[test]
fn argmax_empty_is_none() {
    assert_eq!(empty().argmax(), None);
}

#[test]
fn median_empty_is_none() {
    assert_eq!(empty().median(), None);
}

#[test]
fn mode_empty_is_none() {
    assert_eq!(empty().mode(), None);
}

#[test]
fn skew_needs_three_values() {
    assert!(v(&[1.0, 2.0]).skew().is_nan());
}

#[test]
fn skew_constant_is_zero() {
    assert_eq!(v(&[3.0, 3.0, 3.0]).skew(), 0.0);
}

#[test]
fn kurtosis_needs_four_values() {
    assert!(v(&[1.0, 2.0, 3.0]).kurtosis().is_nan());
}

#[test]
fn kurtosis_constant_is_zero() {
    assert_eq!(v(&[3.0, 3.0, 3.0, 3.0]).kurtosis(), 0.0);
}

#[test]
fn zscore_single_value_is_clone() {
    assert_eq!(v(&[5.0]).zscore().data(), &[5.0]);
}

#[test]
fn zscore_constant_is_zero() {
    assert_eq!(v(&[2.0, 2.0, 2.0]).zscore().data(), &[0.0, 0.0, 0.0]);
}

#[test]
fn moment_empty_is_nan() {
    assert!(empty().moment(2).is_nan());
}

#[test]
fn geometric_mean_empty_and_non_positive() {
    assert!(empty().geometric_mean().is_nan());
    assert!(v(&[1.0, 0.0, 4.0]).geometric_mean().is_nan());
}

#[test]
fn harmonic_mean_empty_and_non_positive() {
    assert!(empty().harmonic_mean().is_nan());
    assert!(v(&[1.0, -2.0]).harmonic_mean().is_nan());
}

#[test]
fn trim_mean_empty_is_nan() {
    assert!(empty().trim_mean(0.1).is_nan());
}

#[test]
fn trim_mean_over_half_falls_back_to_median() {
    // k = 2 values are cut from each end; 2k >= 4, so the median of 1, 2, 3, 4 is used.
    assert_eq!(v(&[1.0, 2.0, 3.0, 4.0]).trim_mean(0.5), 2.5);
}

#[test]
fn sem_needs_two_values() {
    assert!(v(&[1.0]).sem().is_nan());
}

#[test]
fn coefficient_of_variation_empty_and_zero_mean() {
    assert!(empty().coefficient_of_variation().is_nan());
    assert!(v(&[-1.0, 1.0]).coefficient_of_variation().is_nan());
}

#[test]
fn percentile_interpolates() {
    // Position 1.5 between 2 and 3: 2 * 0.5 + 3 * 0.5.
    let data = v(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(data.percentile(50.0), Some(2.5));
    assert_eq!(data.quantile(0.5), Some(2.5));
    assert_eq!(data.percentile(120.0), None);
}

#[test]
fn corrcoef_requires_2d() {
    let error = expect_err(v(&[1.0, 2.0, 3.0]).corrcoef());
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![2],
            got: vec![1]
        }
    );
}

#[test]
fn cov_requires_2d() {
    let error = expect_err(v(&[1.0, 2.0, 3.0]).cov());
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![2],
            got: vec![1]
        }
    );
}

#[test]
fn corrcoef_needs_two_observations() {
    let one_observation = NdArray::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let error = expect_err(one_observation.corrcoef());
    assert_eq!(
        error,
        ArrayError::InvalidOperation("need at least 2 observations".into())
    );
}

#[test]
fn cov_needs_two_observations() {
    let one_observation = NdArray::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let error = expect_err(one_observation.cov());
    assert_eq!(
        error,
        ArrayError::InvalidOperation("need at least 2 observations".into())
    );
}

#[test]
fn corrcoef_constant_row() {
    // Row 0 has zero variance: its off-diagonal is 0 and its diagonal is 1.
    let rows = NdArray::new(vec![1.0, 1.0, 1.0, 1.0, 2.0, 3.0], vec![2, 3]).unwrap();
    let corr = rows.corrcoef().unwrap();
    assert_eq!(corr.dims(), &[2, 2]);
    let expected = [1.0, 0.0, 0.0, 1.0];
    for (got, want) in corr.data().iter().zip(expected.iter()) {
        assert!(close(*got, *want), "got {got}, want {want}");
    }
}

#[test]
fn histogram_empty_or_zero_bins() {
    assert_eq!(empty().histogram(3), (vec![], vec![]));
    assert_eq!(v(&[1.0, 2.0]).histogram(0), (vec![], vec![]));
}

#[test]
fn histogram_constant_data() {
    assert_eq!(v(&[2.0, 2.0, 2.0]).histogram(3), (vec![3], vec![2.0, 2.0]));
}
