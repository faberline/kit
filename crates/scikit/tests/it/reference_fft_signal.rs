//! Reference-value tests for the `fft` and `signal` rows of
//! `docs/reference-coverage.md`.
//!
//! The expected values are in `fixtures/fft_signal_reference.tsv`. The
//! generator `tools/reference/scikit/generate.py` writes them from NumPy 2.4.6
//! and SciPy 1.17.1; its `source` column names the expression of each value.
//! The inputs come from the integer formula of `sample` below, so Rust and
//! Python build the same values.

use scikit::fft::{fft, fftfreq, ifft, irfft, rfft, rfftfreq, Complex};
use scikit::signal::{
    blackman, butter, convolve, correlate, filtfilt, find_peaks, hamming, hann, kaiser, lfilter,
    sosfilt, stft, ConvolveMode, FilterType, PeakOptions, SosSection,
};

const FIXTURE: &str = include_str!("../fixtures/fft_signal_reference.tsv");

/// The values of the fixture line `key`. Panics when the key is missing.
fn reference(key: &str) -> Vec<f64> {
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

/// The single value of the fixture line `key`.
fn scalar(key: &str) -> f64 {
    let v = reference(key);
    assert_eq!(v.len(), 1, "`{key}` is not a scalar");
    v[0]
}

/// The generator's input `x[i] = ((i * 37 + 11) % 101) / 4.0`.
fn sample(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 37 + 11) % 101) as f64 / 4.0).collect()
}

fn complex(v: &[f64]) -> Vec<Complex> {
    v.iter().map(|&x| Complex::new(x, 0.0)).collect()
}

/// Every element of `actual` is within `1e-9 * max(|e|, 1)` of `expected`.
fn assert_close_list(actual: &[f64], expected: &[f64], key: &str) {
    assert_eq!(actual.len(), expected.len(), "length of `{key}`");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = 1e-9 * e.abs().max(1.0);
        assert!(
            (a - e).abs() <= tol,
            "`{key}`[{i}] = {a}, expected {e} (tol {tol})"
        );
    }
}

/// A list of at most 64 values is compared element by element. A longer list
/// is compared by its first 5 values, its last 5 values, the sum of absolute
/// values and its length, as the generator's `vec_rows` writes it.
fn check_vec(actual: &[f64], prefix: &str) {
    if actual.len() <= 64 {
        assert_close_list(actual, &reference(prefix), prefix);
        return;
    }
    let n = actual.len();
    assert_close_list(&actual[..5], &reference(&format!("{prefix}_head")), prefix);
    assert_close_list(
        &actual[n - 5..],
        &reference(&format!("{prefix}_tail")),
        prefix,
    );
    let abs_sum: f64 = actual.iter().map(|v| v.abs()).sum();
    let want = scalar(&format!("{prefix}_abs_sum"));
    assert!(
        (abs_sum - want).abs() <= 1e-9 * want.abs().max(1.0),
        "`{prefix}_abs_sum` = {abs_sum}, expected {want}"
    );
    assert_eq!(
        n as f64,
        scalar(&format!("{prefix}_len")),
        "length of `{prefix}`"
    );
}

/// The real part under `prefix_re` and the imaginary part under `prefix_im`.
fn check_complex(z: &[Complex], prefix: &str) {
    let re: Vec<f64> = z.iter().map(|c| c.re).collect();
    let im: Vec<f64> = z.iter().map(|c| c.im).collect();
    check_vec(&re, &format!("{prefix}_re"));
    check_vec(&im, &format!("{prefix}_im"));
}

/// The sections of a fixture row with 6 values per section (b0 b1 b2 a0 a1 a2).
fn sections(key: &str) -> Vec<SosSection> {
    reference(key)
        .chunks(6)
        .map(|c| SosSection {
            b: [c[0], c[1], c[2]],
            a: [c[3], c[4], c[5]],
        })
        .collect()
}

#[test]
fn fft_matches_numpy() {
    for n in [5usize, 8, 1000, 1024] {
        check_complex(&fft(&complex(&sample(n))), &format!("fft_n{n}"));
    }
}

#[test]
fn ifft_matches_numpy() {
    for n in [5usize, 8, 1000, 1024] {
        check_complex(&ifft(&complex(&sample(n))), &format!("ifft_n{n}"));
    }
}

#[test]
fn rfft_matches_numpy() {
    for n in [5usize, 8, 1000, 1024] {
        check_complex(&rfft(&sample(n)), &format!("rfft_n{n}"));
    }
}

#[test]
fn irfft_matches_numpy() {
    for n in [5usize, 8, 1000, 1024] {
        let y = irfft(&rfft(&sample(n)), n);
        check_vec(&y, &format!("irfft_n{n}"));
    }
}

#[test]
fn conj_matches_numpy() {
    let z: Vec<Complex> = fft(&complex(&sample(8)))
        .iter()
        .map(Complex::conj)
        .collect();
    check_complex(&z, "conj_n8");
}

#[test]
fn norm_matches_numpy() {
    let m: Vec<f64> = fft(&complex(&sample(8)))
        .iter()
        .map(Complex::norm)
        .collect();
    check_vec(&m, "norm_n8");
}

#[test]
fn fftfreq_matches_numpy() {
    check_vec(&fftfreq(8, 0.5), "fftfreq_n8_d0p5");
    check_vec(&fftfreq(5, 0.1), "fftfreq_n5_d0p1");
    check_vec(&fftfreq(1000, 0.001), "fftfreq_n1000_d0p001");
}

#[test]
fn rfftfreq_matches_numpy() {
    check_vec(&rfftfreq(8, 0.5), "rfftfreq_n8_d0p5");
    check_vec(&rfftfreq(5, 0.1), "rfftfreq_n5_d0p1");
    check_vec(&rfftfreq(1000, 0.001), "rfftfreq_n1000_d0p001");
}

#[test]
fn hann_matches_scipy() {
    for n in [9usize, 64] {
        check_vec(&hann(n), &format!("hann_n{n}"));
    }
}

#[test]
fn hamming_matches_scipy() {
    for n in [9usize, 64] {
        check_vec(&hamming(n), &format!("hamming_n{n}"));
    }
}

#[test]
fn blackman_matches_scipy() {
    for n in [9usize, 64] {
        check_vec(&blackman(n), &format!("blackman_n{n}"));
    }
}

#[test]
fn kaiser_matches_scipy() {
    for n in [9usize, 64] {
        check_vec(&kaiser(n, 5.0), &format!("kaiser_n{n}_beta5p0"));
        check_vec(&kaiser(n, 8.6), &format!("kaiser_n{n}_beta8p6"));
    }
}

const MODES: [(&str, ConvolveMode); 3] = [
    ("full", ConvolveMode::Full),
    ("same", ConvolveMode::Same),
    ("valid", ConvolveMode::Valid),
];

#[test]
fn convolve_matches_scipy() {
    let cases = [
        ("small", sample(7), vec![0.5, -1.0, 2.0]),
        ("long", sample(100), vec![0.25, 0.5, 0.25]),
    ];
    for (size, a, b) in &cases {
        for &(mode_name, mode) in MODES.iter() {
            check_vec(
                &convolve(a, b, mode),
                &format!("convolve_{mode_name}_{size}"),
            );
        }
    }
}

#[test]
fn correlate_matches_scipy() {
    let cases = [
        ("small", sample(7), vec![0.5, -1.0, 2.0]),
        ("long", sample(100), vec![0.25, 0.5, 0.25]),
    ];
    for (size, a, b) in &cases {
        for &(mode_name, mode) in MODES.iter() {
            check_vec(
                &correlate(a, b, mode),
                &format!("correlate_{mode_name}_{size}"),
            );
        }
    }
}

#[test]
fn lfilter_matches_scipy() {
    let b = [0.2, 0.3, 0.2];
    let a = [1.0, -0.5];
    check_vec(&lfilter(&b, &a, &sample(64)), "lfilter_iir_64");
    check_vec(&lfilter(&b, &a, &sample(1000)), "lfilter_iir_1000");
    check_vec(
        &lfilter(&[0.25; 4], &[1.0], &sample(100)),
        "lfilter_fir_100",
    );
    check_vec(
        &lfilter(&[1.0, 2.0], &[2.0, -1.0], &sample(64)),
        "lfilter_a0_2",
    );
}

#[test]
fn filtfilt_matches_scipy_lfilter() {
    let b = [0.2, 0.3, 0.2];
    let a = [1.0, -0.5];
    check_vec(&filtfilt(&b, &a, &sample(64)), "filtfilt_iir_64");
    check_vec(
        &filtfilt(&[0.25; 4], &[1.0], &sample(100)),
        "filtfilt_fir_100",
    );
}

#[test]
fn sosfilt_matches_scipy() {
    let x = sample(64);
    check_vec(
        &sosfilt(&sections("sos_o4_wn0p3_low"), &x),
        "sosfilt_o4_wn0p3_low",
    );
    check_vec(
        &sosfilt(&sections("sos_o3_wn0p2_high"), &x),
        "sosfilt_o3_wn0p2_high",
    );
}

#[test]
fn butter_matches_scipy() {
    let cases = [
        (2usize, 0.3, FilterType::Lowpass, "butter_o2_wn0p3_lowpass"),
        (4, 0.3, FilterType::Lowpass, "butter_o4_wn0p3_lowpass"),
        (2, 0.2, FilterType::Highpass, "butter_o2_wn0p2_highpass"),
        (3, 0.2, FilterType::Highpass, "butter_o3_wn0p2_highpass"),
    ];
    for (order, wn, ftype, key) in cases {
        check_vec(&sosfilt(&butter(order, wn, ftype), &sample(64)), key);
    }
}

#[test]
fn find_peaks_matches_scipy() {
    let data = sample(40);
    let cases = [
        ("all", PeakOptions::default()),
        (
            "height",
            PeakOptions {
                height: Some(10.0),
                ..Default::default()
            },
        ),
        (
            "distance",
            PeakOptions {
                distance: Some(5),
                ..Default::default()
            },
        ),
        (
            "prominence",
            PeakOptions {
                prominence: Some(8.0),
                ..Default::default()
            },
        ),
    ];
    for (name, opts) in cases {
        let peaks = find_peaks(&data, &opts);
        let index: Vec<f64> = peaks.iter().map(|p| p.index as f64).collect();
        let height: Vec<f64> = peaks.iter().map(|p| p.height).collect();
        let prominence: Vec<f64> = peaks.iter().map(|p| p.prominence).collect();
        check_vec(&index, &format!("peaks_{name}_index"));
        check_vec(&height, &format!("peaks_{name}_height"));
        check_vec(&prominence, &format!("peaks_{name}_prominence"));
    }
}

#[test]
fn stft_matches_numpy_hann() {
    let x = sample(256);
    for (w, h, fs) in [(64usize, 32usize, 100.0), (100, 25, 1000.0)] {
        let prefix = format!("stft_w{w}_h{h}");
        let res = stft(&x, w, h, fs);
        check_vec(&res.frequencies, &format!("{prefix}_freqs"));
        check_vec(&res.times, &format!("{prefix}_times"));
        assert_eq!(
            res.spectrogram.len() as f64,
            scalar(&format!("{prefix}_n_frames")),
            "frames of `{prefix}`"
        );
        let first = &res.spectrogram[0];
        let last = res.spectrogram.last().expect("at least one frame");
        check_complex(first, &format!("{prefix}_frame0"));
        check_complex(last, &format!("{prefix}_last"));
        let all: Vec<Complex> = res.spectrogram.iter().flatten().cloned().collect();
        check_complex(&all, &format!("{prefix}_all"));
    }
}
