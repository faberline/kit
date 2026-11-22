//! `NdArray::split` must return the same pieces as `numpy.split`.
//!
//! numpy slices `a[start:end]` for each gap, so a repeated or decreasing index
//! gives an empty piece and an index past the end is clamped.

use arraykit::array::{ArrayError, NdArray};

fn six() -> NdArray<f64> {
    NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap()
}

#[test]
fn split_repeated_index_gives_empty_piece() {
    // numpy.split(a, [2, 2]) -> [1, 2], [], [3, 4, 5, 6]
    let pieces = six().split(&[2, 2], 0).unwrap();
    assert_eq!(pieces.len(), 3);
    assert_eq!(pieces[0].dims(), &[2]);
    assert_eq!(pieces[0].data(), &[1.0, 2.0]);
    assert_eq!(pieces[1].dims(), &[0]);
    assert!(pieces[1].data().is_empty());
    assert_eq!(pieces[2].dims(), &[4]);
    assert_eq!(pieces[2].data(), &[3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn split_index_past_end_is_clamped() {
    // numpy.split(a, [9]) -> all six values, then an empty piece
    let pieces = six().split(&[9], 0).unwrap();
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[0].dims(), &[6]);
    assert_eq!(pieces[0].data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(pieces[1].dims(), &[0]);
    assert!(pieces[1].data().is_empty());
}

#[test]
fn split_decreasing_index_gives_empty_piece() {
    // numpy.split(a, [4, 2]) -> a[0:4], a[4:2] (empty), a[2:6]
    let pieces = six().split(&[4, 2], 0).unwrap();
    assert_eq!(pieces.len(), 3);
    assert_eq!(pieces[0].dims(), &[4]);
    assert_eq!(pieces[0].data(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(pieces[1].dims(), &[0]);
    assert!(pieces[1].data().is_empty());
    assert_eq!(pieces[2].dims(), &[4]);
    assert_eq!(pieces[2].data(), &[3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn split_2d_axis_1() {
    // Normal path: shape [2, 3], split at column 1.
    let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let pieces = a.split(&[1], 1).unwrap();
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[0].dims(), &[2, 1]);
    assert_eq!(pieces[0].data(), &[1.0, 4.0]);
    assert_eq!(pieces[1].dims(), &[2, 2]);
    assert_eq!(pieces[1].data(), &[2.0, 3.0, 5.0, 6.0]);
}

#[test]
fn split_axis_out_of_bounds() {
    let err = six().split(&[1], 1).unwrap_err();
    assert!(matches!(err, ArrayError::IndexOutOfBounds { .. }));
}
