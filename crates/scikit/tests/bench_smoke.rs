//! Smoke tests for the calls that `benches/scikit.rs` measures.
//! Each test runs one benched call on a tiny input and checks the result.

#![cfg(all(feature = "fft", feature = "stats"))]

use scikit::fft::{fft, Complex};
use scikit::stats::{describe, median, pearsonr};

#[test]
fn fft_smoke() {
    let input: Vec<Complex> = [1.0, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .collect();
    let out = fft(&input);
    assert_eq!(out.len(), 4);
    assert!((out[0].re - 10.0).abs() < 1e-9);
    assert!(out[0].im.abs() < 1e-9);
}

#[test]
fn describe_smoke() {
    let d = describe(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(d.count, 5);
    assert!((d.mean - 3.0).abs() < 1e-9);
    assert_eq!(d.min, 1.0);
    assert_eq!(d.max, 5.0);
    assert!((d.median - 3.0).abs() < 1e-9);
}

#[test]
fn median_smoke() {
    assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
}

#[test]
fn pearsonr_smoke() {
    let r = pearsonr(&[1.0, 2.0, 3.0, 4.0], &[2.0, 4.0, 6.0, 8.0]);
    assert!((r.statistic - 1.0).abs() < 1e-9);
}

#[test]
fn pearsonr_length_mismatch_is_nan() {
    assert!(pearsonr(&[1.0, 2.0, 3.0], &[1.0, 2.0]).statistic.is_nan());
}
