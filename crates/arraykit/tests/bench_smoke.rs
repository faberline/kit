//! Smoke tests for the calls that the criterion benchmarks use.
//!
//! The benchmark file is not part of `cargo test`. These tests run the same
//! calls on tiny inputs, so a broken bench input or shape fails here.

use arraykit::NdArray;

#[test]
fn matmul_smoke() {
    let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
    assert_eq!(c.dims(), &[2, 2]);
}

#[test]
fn sum_smoke() {
    let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    assert_eq!(a.sum(), 10.0);
}

#[test]
fn sort_smoke() {
    let a = NdArray::new(vec![3i64, 1, 2, 1], vec![4]).unwrap();
    assert_eq!(a.sort().data(), &[1, 1, 2, 3]);
}

#[test]
fn matmul_shape_mismatch_is_error() {
    let a = NdArray::new(vec![0.0; 6], vec![2, 3]).unwrap();
    let b = NdArray::new(vec![0.0; 6], vec![2, 3]).unwrap();
    assert!(a.matmul(&b).is_err());
}
