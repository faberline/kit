//! Coverage tests for the search and manipulation functions of `NdArray`.
//!
//! Each test targets a branch in `src/array/search.rs` or `src/array/manip.rs`
//! that the existing tests do not reach. Expected values are computed by hand.

use arraykit::array::{ArrayError, NdArray};

fn arr(data: Vec<f64>, shape: Vec<usize>) -> NdArray<f64> {
    NdArray::new(data, shape).unwrap()
}

// ============================================================================
// concatenate
// ============================================================================

#[test]
fn concatenate_empty_list_is_invalid_operation() {
    let arrays: &[&NdArray<f64>] = &[];
    let err = NdArray::concatenate(arrays, 0).expect_err("error");
    assert!(matches!(err, ArrayError::InvalidOperation(_)));
}

#[test]
fn concatenate_axis_out_of_bounds() {
    // A 1D array has only axis 0. Axis 1 is out of bounds.
    let a = arr(vec![1.0, 2.0], vec![2]);
    let err = NdArray::concatenate(&[&a], 1).expect_err("error");
    match err {
        ArrayError::IndexOutOfBounds { index, shape } => {
            assert_eq!(index, vec![1]);
            assert_eq!(shape, vec![2]);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn concatenate_ndim_mismatch() {
    // A 1D array and a 2D array cannot be concatenated.
    let a = arr(vec![1.0, 2.0], vec![2]);
    let b = arr(vec![1.0, 2.0], vec![1, 2]);
    let err = NdArray::concatenate(&[&a, &b], 0).expect_err("error");
    assert!(matches!(err, ArrayError::ShapeMismatch { .. }));
}

#[test]
fn concatenate_other_axis_mismatch() {
    // Along axis 0, the second axis must match: 2 is not 3.
    let a = arr(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let err = NdArray::concatenate(&[&a, &b], 0).expect_err("error");
    match err {
        ArrayError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2, 2]);
            assert_eq!(got, vec![2, 3]);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// ============================================================================
// stack
// ============================================================================

#[test]
fn stack_empty_list_is_invalid_operation() {
    let arrays: &[&NdArray<f64>] = &[];
    let err = NdArray::stack(arrays, 0).expect_err("error");
    assert!(matches!(err, ArrayError::InvalidOperation(_)));
}

#[test]
fn stack_shape_mismatch() {
    let a = arr(vec![1.0, 2.0], vec![2]);
    let b = arr(vec![1.0, 2.0, 3.0], vec![3]);
    let err = NdArray::stack(&[&a, &b], 0).expect_err("error");
    match err {
        ArrayError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2]);
            assert_eq!(got, vec![3]);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// ============================================================================
// hstack
// ============================================================================

#[test]
fn hstack_empty_list_is_invalid_operation() {
    let arrays: &[&NdArray<f64>] = &[];
    let err = NdArray::hstack(arrays).expect_err("error");
    assert!(matches!(err, ArrayError::InvalidOperation(_)));
}

#[test]
fn hstack_1d_joins_along_axis_0() {
    let a = arr(vec![1.0, 2.0], vec![2]);
    let b = arr(vec![3.0], vec![1]);
    let out = NdArray::hstack(&[&a, &b]).unwrap();
    assert_eq!(out.dims(), &[3]);
    assert_eq!(out.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn hstack_2d_joins_along_axis_1() {
    // Row 0: 1 | 3 4  -> 1 3 4
    // Row 1: 2 | 5 6  -> 2 5 6
    let a = arr(vec![1.0, 2.0], vec![2, 1]);
    let b = arr(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
    let out = NdArray::hstack(&[&a, &b]).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    assert_eq!(out.data(), &[1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
}

// ============================================================================
// split
// ============================================================================

#[test]
fn split_axis_out_of_bounds() {
    let a = arr(vec![1.0, 2.0], vec![2]);
    let err = a.split(&[1], 1).expect_err("error");
    assert!(matches!(err, ArrayError::IndexOutOfBounds { .. }));
}

// ============================================================================
// tile
// ============================================================================

#[test]
fn tile_empty_reps_returns_clone() {
    let a = arr(vec![1.0, 2.0, 3.0], vec![3]);
    let out = a.tile(&[]).unwrap();
    assert_eq!(out.dims(), a.dims());
    assert_eq!(out.data(), a.data());
}

#[test]
fn tile_more_reps_than_dims() {
    // A 1D array [1, 2] with reps [2, 2] becomes shape [1, 2] first, then [2, 4].
    let a = arr(vec![1.0, 2.0], vec![2]);
    let out = a.tile(&[2, 2]).unwrap();
    assert_eq!(out.dims(), &[2, 4]);
    assert_eq!(out.data(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

// ============================================================================
// allclose
// ============================================================================

#[test]
fn allclose_different_shapes_is_false() {
    let a = arr(vec![1.0, 2.0], vec![2]);
    let b = arr(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(!a.allclose(&b, 1e-5, 1e-8));
}

// ============================================================================
// searchsorted
// ============================================================================

#[test]
fn searchsorted_requires_1d() {
    let a = arr(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let err = a.searchsorted(&[1.0], "left").expect_err("error");
    assert!(matches!(err, ArrayError::InvalidOperation(_)));
}

// ============================================================================
// sort_axis and argsort_axis
// ============================================================================

#[test]
fn sort_axis_invalid_axis() {
    let a = arr(vec![3.0, 1.0, 2.0], vec![3]);
    let err = a.sort_axis(1).expect_err("error");
    assert_eq!(err, ArrayError::InvalidAxis { axis: 1, ndim: 1 });
}

#[test]
fn argsort_axis_invalid_axis() {
    let a = arr(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let err = a.argsort_axis(2).expect_err("error");
    assert_eq!(err, ArrayError::InvalidAxis { axis: 2, ndim: 2 });
}

#[test]
fn argsort_axis_1d() {
    // Values 3, 1, 2 sort to positions 1, 2, 0.
    let a = arr(vec![3.0, 1.0, 2.0], vec![3]);
    let out = a.argsort_axis(0).unwrap();
    assert_eq!(out.data(), &[1.0, 2.0, 0.0]);
}

#[test]
fn argsort_axis_2d_axis0() {
    // Shape [2, 3]: [[3, 1, 2], [0, 5, 4]].
    // Column 0: 3, 0 -> indices 1, 0.
    // Column 1: 1, 5 -> indices 0, 1.
    // Column 2: 2, 4 -> indices 0, 1.
    // Result rows: [1, 0, 0] and [0, 1, 1].
    let a = arr(vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0], vec![2, 3]);
    let out = a.argsort_axis(0).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    assert_eq!(out.data(), &[1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn argsort_axis_2d_axis1() {
    // Shape [2, 3]: [[3, 1, 2], [0, 5, 4]].
    // Row 0: 3, 1, 2 -> indices 1, 2, 0.
    // Row 1: 0, 5, 4 -> indices 0, 2, 1.
    let a = arr(vec![3.0, 1.0, 2.0, 0.0, 5.0, 4.0], vec![2, 3]);
    let out = a.argsort_axis(1).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    assert_eq!(out.data(), &[1.0, 2.0, 0.0, 0.0, 2.0, 1.0]);
}

#[test]
fn argsort_axis_3d_with_nan() {
    // Shape [2, 2, 2]. Lines along axis 2 are:
    // (0,0): 4, NaN -> 0, 1   (NaN sorts last)
    // (0,1): 1, 3   -> 0, 1
    // (1,0): 2, 7   -> 0, 1
    // (1,1): 6, 5   -> 1, 0
    let a = arr(
        vec![4.0, f64::NAN, 1.0, 3.0, 2.0, 7.0, 6.0, 5.0],
        vec![2, 2, 2],
    );
    let out = a.argsort_axis(2).unwrap();
    assert_eq!(out.dims(), &[2, 2, 2]);
    assert_eq!(out.data(), &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
}

// ============================================================================
// where_select
// ============================================================================

#[test]
fn where_select_shape_mismatch() {
    let cond = NdArray::new(vec![true, false], vec![2]).unwrap();
    let x = arr(vec![1.0, 2.0, 3.0], vec![3]);
    let y = arr(vec![4.0, 5.0, 6.0], vec![3]);
    let err = NdArray::where_select(&cond, &x, &y).expect_err("error");
    match err {
        ArrayError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2]);
            assert_eq!(got, vec![3]);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
