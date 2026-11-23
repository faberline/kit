//! Reference-value tests for `scikit::signal::stft` and `stft_with`.
//!
//! Expected values come from SciPy 1.17.1 with NumPy 2.4.6. The expression of
//! each value is in the `source` column of
//! `tests/fixtures/stft_scipy_reference.tsv`. No Python runs in CI: the
//! fixture is the only source of the expected values.

use scikit::fft::Complex;
use scikit::signal::{stft, stft_with, StftOptions, StftResult, StftScaling, StftWindow};

const FIXTURE: &str = include_str!("../fixtures/stft_scipy_reference.tsv");

/// Return the value column stored under `key` in the fixture, as raw text.
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

/// Return the comma-separated list stored under `key` in the fixture.
fn reference_list(key: &str) -> Vec<f64> {
    let text = raw(key);
    if text.is_empty() {
        return Vec::new();
    }
    text.split(',')
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("fixture key {key}: bad number {v}"))
        })
        .collect()
}

/// Return the single number stored under `key` in the fixture.
fn reference_scalar(key: &str) -> f64 {
    let values = reference_list(key);
    assert_eq!(values.len(), 1, "fixture key {key} must hold one number");
    values[0]
}

/// Asserts that `actual` is within 1e-9 relative of `expected`, with a
/// 1e-12 absolute floor for results that are zero or round-off. A NaN
/// expected value needs a NaN result.
fn close(actual: f64, expected: f64, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}: got {actual}, expected NaN");
        return;
    }
    let tol = f64::max(1e-9 * expected.abs(), 1e-12);
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: got {actual}, expected {expected} (tol {tol})"
    );
}

/// Asserts that `actual` has the length and the values of the fixture list `key`.
fn close_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "length of `{key}`");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        close(*a, *e, &format!("`{key}`[{i}]"));
    }
}

/// The input of the fixture rows: `sig(i) = ((i * 37 + 11) % 101) / 4.0`.
fn sig(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 37 + 11) % 101) as f64 / 4.0).collect()
}

/// Asserts that `result` equals the fixture case `case`: the frame count, the
/// frame times, the frequencies and the real and imaginary parts of the
/// spectrogram (frame first, then frequency bin).
fn check_case(case: &str, result: &StftResult) {
    let prefix = format!("stft.{case}");
    let frames = result.spectrogram.len();
    assert_eq!(
        frames as f64,
        reference_scalar(&format!("{prefix}.n_frames")),
        "frames of `{prefix}`"
    );
    assert_eq!(result.times.len(), frames, "times of `{prefix}`");
    let inner = result.frequencies.len();
    for (t, frame) in result.spectrogram.iter().enumerate() {
        assert_eq!(frame.len(), inner, "bins of frame {t} of `{prefix}`");
    }

    close_list(&result.times, &format!("{prefix}.times"));
    close_list(&result.frequencies, &format!("{prefix}.freqs"));
    let flat: Vec<&Complex> = result.spectrogram.iter().flatten().collect();
    let re: Vec<f64> = flat.iter().map(|c| c.re).collect();
    let im: Vec<f64> = flat.iter().map(|c| c.im).collect();
    close_list(&re, &format!("{prefix}.re"));
    close_list(&im, &format!("{prefix}.im"));
}

#[test]
fn stft_matches_scipy_nobound() {
    let x = sig(256);
    let result = stft(&x, 64, 32, 100.0);
    check_case("nobound", &result);
}

#[test]
fn stft_with_matches_scipy_default() {
    let x = sig(256);
    let result = stft_with(&x, 100.0, &StftOptions::new(64));
    check_case("default", &result);
}

#[test]
fn stft_with_matches_scipy_density() {
    let x = sig(256);
    let options = StftOptions {
        scaling: StftScaling::Density,
        boundary: false,
        padded: false,
        noverlap: Some(32),
        ..StftOptions::new(64)
    };
    let result = stft_with(&x, 100.0, &options);
    check_case("density", &result);
}

#[test]
fn stft_with_matches_scipy_hamming_odd() {
    let x = sig(250);
    let options = StftOptions {
        window: StftWindow::Hamming,
        noverlap: Some(16),
        ..StftOptions::new(33)
    };
    let result = stft_with(&x, 100.0, &options);
    check_case("hamming_odd", &result);
}

#[test]
fn stft_with_matches_scipy_blackman_pad() {
    let x = sig(256);
    let options = StftOptions {
        window: StftWindow::Blackman,
        noverlap: Some(10),
        boundary: false,
        ..StftOptions::new(40)
    };
    let result = stft_with(&x, 100.0, &options);
    check_case("blackman_pad", &result);
}

#[test]
#[should_panic(expected = "noverlap")]
fn stft_with_rejects_noverlap_not_below_nperseg() {
    let x = sig(256);
    let options = StftOptions {
        noverlap: Some(64),
        ..StftOptions::new(64)
    };
    let _ = stft_with(&x, 100.0, &options);
}

#[test]
fn stft_with_default_options_equal_scipy_defaults() {
    let options = StftOptions::new(64);
    assert_eq!(options.noverlap, None);
    assert_eq!(options.window, StftWindow::Hann);
    assert!(options.boundary);
    assert!(options.padded);
    assert_eq!(options.scaling, StftScaling::Spectrum);
}
