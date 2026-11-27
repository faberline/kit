//! Reference-value tests for `scikit::signal::stft` and `stft_with`.
//!
//! Expected values come from `scipy.signal.stft` (SciPy 1.17.1 with NumPy
//! 2.4.6). The `source` column of `tests/fixtures/stft_scipy_reference.tsv`
//! names the SciPy call of each case. No Python runs in CI.

use scikit::fft::Complex;
use scikit::signal::{stft, stft_with, StftOptions, StftResult, StftScaling, StftWindow};

const FIXTURE: &str = include_str!("../fixtures/stft_scipy_reference.tsv");

/// Relative tolerance of every comparison in this file.
const REL_TOL: f64 = 1e-9;
/// Absolute floor of every comparison, for results that are zero or round-off.
const ZERO_ABS_TOL: f64 = 1e-12;

/// The shared input: x[i] = ((i * 37 + 11) % 101) / 4.0 (the same values as the generator).
fn sample(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 37 + 11) % 101) as f64 / 4.0).collect()
}

/// Return the value column of the fixture row `key` as a list of numbers.
fn reference_list(key: &str) -> Vec<f64> {
    let line = FIXTURE
        .lines()
        .find(|l| l.split('\t').next() == Some(key))
        .unwrap_or_else(|| panic!("fixture key `{key}` not found"));
    line.split('\t')
        .nth(1)
        .unwrap_or_else(|| panic!("fixture key `{key}` has no value column"))
        .split(',')
        .map(|v| {
            v.parse::<f64>()
                .unwrap_or_else(|e| panic!("bad value in `{key}`: {e}"))
        })
        .collect()
}

/// Return the scalar value of the fixture row `key`.
fn reference_scalar(key: &str) -> f64 {
    let v = reference_list(key);
    assert_eq!(v.len(), 1, "`{key}` is not a scalar");
    v[0]
}

/// Check one value with the relative bound `REL_TOL` and the absolute floor `ZERO_ABS_TOL`.
fn close(actual: f64, expected: f64, label: &str) {
    let tol = f64::max(REL_TOL * expected.abs(), ZERO_ABS_TOL);
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: {actual} vs expected {expected} (tol {tol})"
    );
}

/// Check a list against the fixture row `key`: the length first, then each value.
fn close_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "length of `{key}`");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        close(*a, *e, &format!("`{key}`[{i}]"));
    }
}

/// Check a whole result against the fixture rows `stft.<case>.*`: the frame and
/// bin counts, the times, the frequencies, and the real and imaginary parts of
/// every frame, frame first.
fn check_case(case: &str, result: &StftResult) {
    let prefix = format!("stft.{case}");
    assert_eq!(
        result.times.len() as f64,
        reference_scalar(&format!("{prefix}.n_frames")),
        "frames of `{case}`"
    );
    assert_eq!(
        result.spectrogram.len(),
        result.times.len(),
        "rows of `{case}`"
    );
    assert_eq!(
        result.frequencies.len() as f64,
        reference_scalar(&format!("{prefix}.n_freqs")),
        "bins of `{case}`"
    );
    for frame in &result.spectrogram {
        assert_eq!(
            frame.len(),
            result.frequencies.len(),
            "bins per frame of `{case}`"
        );
    }
    close_list(&result.times, &format!("{prefix}.times"));
    close_list(&result.frequencies, &format!("{prefix}.freqs"));
    let flat: Vec<Complex> = result.spectrogram.iter().flatten().cloned().collect();
    let re: Vec<f64> = flat.iter().map(|c| c.re).collect();
    let im: Vec<f64> = flat.iter().map(|c| c.im).collect();
    close_list(&re, &format!("{prefix}.re"));
    close_list(&im, &format!("{prefix}.im"));
}

#[test]
fn stft_matches_scipy_nobound() {
    let x = sample(256);
    let result = stft(&x, 64, 32, 100.0);
    check_case("nobound", &result);
}

#[test]
fn stft_matches_scipy_nobound_w100_fs1000() {
    let x = sample(256);
    let result = stft(&x, 100, 25, 1000.0);
    check_case("nobound_w100", &result);
}

#[test]
fn stft_with_matches_scipy_default() {
    let x = sample(256);
    let result = stft_with(&x, 100.0, &StftOptions::new(64));
    check_case("default", &result);
}

#[test]
fn stft_with_matches_scipy_density() {
    let x = sample(256);
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
    let x = sample(250);
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
    let x = sample(256);
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
    let x = sample(256);
    let options = StftOptions {
        noverlap: Some(64),
        ..StftOptions::new(64)
    };
    let _ = stft_with(&x, 100.0, &options);
}

#[test]
fn stft_with_default_options_equal_scipy_defaults() {
    let o = StftOptions::new(64);
    assert_eq!(o.nperseg, 64);
    assert_eq!(o.noverlap, None);
    assert_eq!(o.window, StftWindow::Hann);
    assert!(o.boundary);
    assert!(o.padded);
    assert_eq!(o.scaling, StftScaling::Spectrum);
}
