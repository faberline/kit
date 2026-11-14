//! Integration tests for the fft module: exact DFT for any length, and irfft for odd n.

#![cfg(feature = "fft")]

use scikit::fft::{fft, ifft, irfft, rfft, Complex};
use std::f64::consts::PI;

/// True when both parts of `a` are within `tol` of `re` and `im`.
fn close(a: Complex, re: f64, im: f64, tol: f64) -> bool {
    (a.re - re).abs() <= tol && (a.im - im).abs() <= tol
}

#[test]
fn fft_len3_constant() {
    let x = vec![Complex::new(1.0, 0.0); 3];
    let y = fft(&x);
    assert_eq!(y.len(), 3);
    assert!(close(y[0], 3.0, 0.0, 1e-12), "X[0] = {:?}", y[0]);
    assert!(close(y[1], 0.0, 0.0, 1e-12), "X[1] = {:?}", y[1]);
    assert!(close(y[2], 0.0, 0.0, 1e-12), "X[2] = {:?}", y[2]);
}

#[test]
fn fft_len5_matches_definition() {
    let xs = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let x: Vec<Complex> = xs.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let y = fft(&x);
    let n = xs.len();
    for (k, &yk) in y.iter().enumerate() {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &v) in xs.iter().enumerate() {
            let theta = 2.0 * PI * (j * k) as f64 / n as f64;
            re += v * theta.cos();
            im -= v * theta.sin();
        }
        assert!(
            close(yk, re, im, 1e-10),
            "X[{k}] = {:?}, expected ({re}, {im})",
            yk
        );
    }
}

#[test]
fn ifft_fft_roundtrip_lengths() {
    for n in [1_usize, 3, 5, 6, 7, 8] {
        let x: Vec<Complex> = (0..n)
            .map(|j| Complex::new((j as f64 * 0.7).sin() + 0.1, 0.0))
            .collect();
        let back = ifft(&fft(&x));
        assert_eq!(back.len(), n, "length for n = {n}");
        for (j, (a, b)) in x.iter().zip(back.iter()).enumerate() {
            assert!(
                close(*b, a.re, a.im, 1e-12),
                "n = {n}, j = {j}: got {:?}, want {:?}",
                b,
                a
            );
        }
    }
}

#[test]
fn fft_len1() {
    let c = Complex::new(2.5, -1.0);
    let y = fft(&[c]);
    assert_eq!(y.len(), 1);
    assert!(close(y[0], 2.5, -1.0, 1e-12), "got {:?}", y[0]);
}

#[test]
fn fft_len8_unchanged() {
    let xs = [1.0_f64, 2.0, 0.0, -1.0, 3.0, 0.5, -2.0, 4.0];
    let x: Vec<Complex> = xs.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let y = fft(&x);
    let n = xs.len();
    for (k, &yk) in y.iter().enumerate() {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &v) in xs.iter().enumerate() {
            let theta = 2.0 * PI * (j * k) as f64 / n as f64;
            re += v * theta.cos();
            im -= v * theta.sin();
        }
        assert!(
            close(yk, re, im, 1e-10),
            "X[{k}] = {:?}, expected ({re}, {im})",
            yk
        );
    }
}

#[test]
fn irfft_rfft_roundtrip() {
    for n in [4_usize, 5, 6, 7] {
        let x: Vec<f64> = (0..n).map(|j| (j as f64 * 0.3).cos() * 2.0).collect();
        let back = irfft(&rfft(&x), n);
        assert_eq!(back.len(), n, "length for n = {n}");
        for (j, (a, b)) in x.iter().zip(back.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12,
                "n = {n}, j = {j}: got {b}, want {a}"
            );
        }
    }
}
