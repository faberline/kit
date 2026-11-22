//! Reference-value tests. Expected values come from NumPy 1.26; the expression
//! that made each value is in the comment above it. CI does not run Python.
//!
//! The linear algebra reference tests (values from NumPy 2.4.6) are in
//! `reference_linalg.rs`.

use arraykit::array::{ArrayError, KitRng, NdArray};

/// Absolute-or-relative check with a 1e-9 scale.
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{a} != {b}");
}

// numpy 1.26: np.arange(0, 5, 0) raises ZeroDivisionError, so the call must be an error.
// numpy 1.26: len(np.arange(0.0, 1.0, 0.1)) = 10
// numpy 1.26: np.arange(0.0, 1.0, 0.1)[3] = 0 + 3 * 0.1 = 0.30000000000000004
#[test]
fn arange_zero_step_is_error_and_float_length() {
    assert!(NdArray::arange(0i32, 5, 0).is_err());

    let a = NdArray::arange(0.0f64, 1.0, 0.1).unwrap();
    assert_eq!(a.data().len(), 10);
    close(a.data()[3], 0.30000000000000004);
}

// numpy 1.26: np.linalg.eig([[0., -1.], [1., 0.]]) gives eigenvalues +1j and -1j.
// The kit returns real eigenvalues only, so eig must return an error, not real values.
#[test]
fn eig_rotation_matrix_is_an_error() {
    let a = NdArray::new(vec![0.0f64, -1.0, 1.0, 0.0], vec![2, 2]).unwrap();
    assert!(a.eig().is_err());
}

// numpy 1.26: np.linalg.svd([[nan, 1.], [1., 1.]]) raises LinAlgError, so the kit must
// return an error. It must not panic.
#[test]
fn svd_with_nan_is_an_error_not_a_panic() {
    let a = NdArray::new(vec![f64::NAN, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
    assert!(a.svd().is_err());
}

// These calls must return an error for bad arguments, not panic.
#[test]
fn no_panic_on_bad_arguments() {
    let a = NdArray::new(vec![1.0f64, 2.0], vec![2]).unwrap();
    let b = NdArray::new(vec![3.0f64, 4.0], vec![2]).unwrap();

    assert!(a.rolling(0).is_err());
    assert!(KitRng::new(1).choice(0).is_err());
    assert!(matches!(
        NdArray::stack(&[&a, &b], 5),
        Err(ArrayError::InvalidAxis { .. })
    ));
}
