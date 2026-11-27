//! Short-Time Fourier Transform (STFT), as `scipy.signal.stft`.

use super::windows::{blackman, hamming, hann};
use crate::fft::{fft, Complex};

/// Result of the Short-Time Fourier Transform.
#[derive(Debug, Clone)]
pub struct StftResult {
    /// Frequency bins (Hz). Length = window_size / 2 + 1.
    pub frequencies: Vec<f64>,
    /// Time centers of each frame (seconds), as SciPy gives them. With a
    /// boundary extension the times are measured from the start of the signal.
    pub times: Vec<f64>,
    /// Spectrogram: `spectrogram[t][f]` is the complex value for time frame `t`
    /// and frequency bin `f`. Outer length = `times.len()`, inner = `frequencies.len()`.
    pub spectrogram: Vec<Vec<Complex>>,
}

/// The window of an STFT. Each one is periodic, as SciPy's `get_window` with
/// `fftbins=True` builds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StftWindow {
    /// Periodic Hann window.
    Hann,
    /// Periodic Hamming window.
    Hamming,
    /// Periodic Blackman window.
    Blackman,
}

/// The scaling of the spectrum of an STFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StftScaling {
    /// Each bin is divided by the sum of the window (SciPy `scaling='spectrum'`).
    Spectrum,
    /// Each bin is divided by `sqrt(fs * sum(window^2))` (SciPy `scaling='psd'`).
    Density,
}

/// The options of [`stft_with`]. [`StftOptions::new`] gives the defaults of
/// `scipy.signal.stft`.
#[derive(Debug, Clone, PartialEq)]
pub struct StftOptions {
    /// Samples per segment (`nperseg`).
    pub nperseg: usize,
    /// Overlap in samples. `None` means `nperseg / 2`.
    pub noverlap: Option<usize>,
    /// The window.
    pub window: StftWindow,
    /// Extend the signal with `nperseg / 2` zeros at both ends (SciPy `boundary='zeros'`).
    pub boundary: bool,
    /// Extend the end with zeros so that the last segment is complete (SciPy `padded=True`).
    pub padded: bool,
    /// The scaling of the spectrum.
    pub scaling: StftScaling,
}

impl StftOptions {
    /// Options with the SciPy defaults for segments of `nperseg` samples:
    /// `noverlap = nperseg / 2`, a Hann window, `boundary` and `padded` on,
    /// and spectrum scaling.
    pub fn new(nperseg: usize) -> Self {
        Self {
            nperseg,
            noverlap: None,
            window: StftWindow::Hann,
            boundary: true,
            padded: true,
            scaling: StftScaling::Spectrum,
        }
    }
}

/// The settings of one computation. `nstep` is the hop between segments.
struct Plan {
    nperseg: usize,
    nstep: usize,
    window: StftWindow,
    boundary: bool,
    padded: bool,
    scaling: StftScaling,
}

/// Compute the Short-Time Fourier Transform with a Hann window, without a
/// boundary extension and without padding.
///
/// The result equals `scipy.signal.stft(x, fs, window='hann', nperseg=window_size,
/// noverlap=window_size - hop_size, boundary=None, padded=False)`.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `window_size` - Number of samples per window (must be >= 2)
/// * `hop_size` - Number of samples between successive windows (must be >= 1)
/// * `fs` - Sampling frequency in Hz
///
/// # Panics
///
/// Panics if `window_size < 2` or `hop_size < 1` or `x` is shorter than `window_size`.
///
/// # Example
///
/// ```
/// use scikit::signal::stft;
///
/// let fs = 100.0;
/// let x: Vec<f64> = (0..256).map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin()).collect();
/// let result = stft(&x, 64, 32, fs);
/// assert!(!result.times.is_empty());
/// assert_eq!(result.frequencies.len(), 33); // 64/2 + 1
/// ```
pub fn stft(x: &[f64], window_size: usize, hop_size: usize, fs: f64) -> StftResult {
    let plan = Plan {
        nperseg: window_size,
        nstep: hop_size,
        window: StftWindow::Hann,
        boundary: false,
        padded: false,
        scaling: StftScaling::Spectrum,
    };
    stft_core(x, fs, &plan)
}

/// Compute the Short-Time Fourier Transform with the options of SciPy's
/// `scipy.signal.stft`: window, overlap, boundary extension, padding and scaling.
///
/// Build the options with [`StftOptions::new`] and change the fields that differ.
///
/// # Panics
///
/// Panics if `nperseg < 2`, if `noverlap` is not less than `nperseg`, or if `x` is
/// shorter than `nperseg`.
///
/// # Example
///
/// ```
/// use scikit::signal::{stft_with, StftOptions, StftWindow};
///
/// let x: Vec<f64> = (0..256).map(|i| (i as f64 * 0.3).sin()).collect();
/// let options = StftOptions { window: StftWindow::Hamming, ..StftOptions::new(64) };
/// let result = stft_with(&x, 100.0, &options);
/// assert_eq!(result.frequencies.len(), 33); // 64/2 + 1
/// ```
pub fn stft_with(x: &[f64], fs: f64, options: &StftOptions) -> StftResult {
    let nperseg = options.nperseg;
    assert!(nperseg >= 2, "nperseg must be >= 2");
    let noverlap = options.noverlap.unwrap_or(nperseg / 2);
    assert!(
        noverlap < nperseg,
        "noverlap ({noverlap}) must be less than nperseg ({nperseg})"
    );
    let plan = Plan {
        nperseg,
        nstep: nperseg - noverlap,
        window: options.window,
        boundary: options.boundary,
        padded: options.padded,
        scaling: options.scaling,
    };
    stft_core(x, fs, &plan)
}

/// The periodic window of length `n`: the symmetric window of length `n + 1`
/// without its last sample, as SciPy's `get_window` with `fftbins=True` builds it.
fn periodic_window(kind: StftWindow, n: usize) -> Vec<f64> {
    let mut w = match kind {
        StftWindow::Hann => hann(n + 1),
        StftWindow::Hamming => hamming(n + 1),
        StftWindow::Blackman => blackman(n + 1),
    };
    w.truncate(n);
    w
}

fn stft_core(x: &[f64], fs: f64, plan: &Plan) -> StftResult {
    let nperseg = plan.nperseg;
    let nstep = plan.nstep;
    assert!(nperseg >= 2, "window_size must be >= 2");
    assert!(nstep >= 1, "hop_size must be >= 1");
    assert!(
        x.len() >= nperseg,
        "signal length ({}) must be >= window_size ({nperseg})",
        x.len()
    );

    let win = periodic_window(plan.window, nperseg);
    let scale = match plan.scaling {
        StftScaling::Spectrum => 1.0 / win.iter().sum::<f64>(),
        StftScaling::Density => 1.0 / (fs * win.iter().map(|w| w * w).sum::<f64>()).sqrt(),
    };

    // The boundary adds nperseg / 2 zeros at both ends; the padding then
    // completes the last segment, as SciPy's `_spectral_helper` does.
    let mut data: Vec<f64> = if plan.boundary {
        let half = nperseg / 2;
        let mut extended = vec![0.0; half];
        extended.extend_from_slice(x);
        extended.resize(extended.len() + half, 0.0);
        extended
    } else {
        x.to_vec()
    };
    if plan.padded {
        let nadd = (nstep - (data.len() - nperseg) % nstep) % nstep % nperseg;
        data.resize(data.len() + nadd, 0.0);
    }

    let n_freqs = nperseg / 2 + 1;
    let frequencies: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * fs / nperseg as f64)
        .collect();

    // SciPy measures the frame centers in the extended data, then moves them
    // back by nperseg / 2 samples when the boundary was added.
    let half = nperseg as f64 / 2.0;
    let shift = if plan.boundary { half / fs } else { 0.0 };

    let mut times = Vec::new();
    let mut spectrogram = Vec::new();

    let mut start = 0;
    while start + nperseg <= data.len() {
        times.push((half + start as f64) / fs - shift);

        let windowed: Vec<Complex> = (0..nperseg)
            .map(|i| Complex::new(data[start + i] * win[i], 0.0))
            .collect();
        let spectrum = fft(&windowed);

        // Keep only the non-negative frequencies and scale each bin.
        let row: Vec<Complex> = spectrum[..n_freqs]
            .iter()
            .map(|c| Complex::new(c.re * scale, c.im * scale))
            .collect();
        spectrogram.push(row);

        start += nstep;
    }

    StftResult {
        frequencies,
        times,
        spectrogram,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_stft_dimensions() {
        let fs = 100.0;
        let n = 256;
        let window_size = 64;
        let hop_size = 32;

        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / fs).sin())
            .collect();

        let result = stft(&x, window_size, hop_size, fs);

        // Number of frequency bins
        assert_eq!(result.frequencies.len(), window_size / 2 + 1);

        // Number of time frames: floor((n - window_size) / hop_size) + 1
        let expected_frames = (n - window_size) / hop_size + 1;
        assert_eq!(result.times.len(), expected_frames);
        assert_eq!(result.spectrogram.len(), expected_frames);

        // Each spectrum has the right length
        for spectrum in &result.spectrogram {
            assert_eq!(spectrum.len(), window_size / 2 + 1);
        }
    }

    #[test]
    fn test_stft_detects_frequency() {
        let fs = 256.0;
        let n = 512;
        let window_size = 64;
        let hop_size = 32;
        let freq = 32.0; // Hz

        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();

        let result = stft(&x, window_size, hop_size, fs);

        // The frequency resolution is fs / window_size = 4 Hz
        // 32 Hz should appear at bin index 32 / 4 = 8
        let target_bin = (freq / (fs / window_size as f64)).round() as usize;

        // Check a middle frame (avoid edges) to see dominant frequency
        let mid_frame = result.spectrogram.len() / 2;
        let magnitudes: Vec<f64> = result.spectrogram[mid_frame]
            .iter()
            .map(|c| c.norm())
            .collect();

        // The target bin should have the largest magnitude (skip DC)
        let peak_bin = (1..magnitudes.len())
            .max_by(|&a, &b| magnitudes[a].partial_cmp(&magnitudes[b]).unwrap())
            .unwrap();

        assert_eq!(
            peak_bin, target_bin,
            "expected peak at bin {target_bin} ({freq} Hz), got bin {peak_bin}"
        );
    }

    #[test]
    fn test_stft_frequency_values() {
        let fs = 100.0;
        let window_size = 16;
        let hop_size = 8;
        let x: Vec<f64> = (0..32).map(|i| i as f64).collect();

        let result = stft(&x, window_size, hop_size, fs);

        // First frequency should be 0 Hz (DC)
        assert!((result.frequencies[0] - 0.0).abs() < 1e-10);

        // Last frequency should be fs/2 (Nyquist)
        let nyquist = fs / 2.0;
        assert!(
            (result.frequencies[result.frequencies.len() - 1] - nyquist).abs() < 1e-10,
            "expected Nyquist = {nyquist}, got {}",
            result.frequencies[result.frequencies.len() - 1]
        );

        // Frequency spacing should be fs / window_size
        let df = fs / window_size as f64;
        for i in 0..result.frequencies.len() {
            let expected = i as f64 * df;
            assert!(
                (result.frequencies[i] - expected).abs() < 1e-10,
                "freq[{i}]: expected {expected}, got {}",
                result.frequencies[i]
            );
        }
    }
}
