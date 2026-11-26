//! Zero-phase forward-backward digital filtering, as `scipy.signal.filtfilt`
//! with `method='pad'`: the signal is extended at both ends, filtered forward
//! and backward, and the extension is removed.

use super::filter::{lfilter_with_state, lfilter_zi};

/// The extension at each end of the signal (SciPy `padtype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiltfiltPad {
    /// Odd extension: the signal mirrored through its end value (SciPy `'odd'`).
    Odd,
    /// Even extension: the signal mirrored at its end value (SciPy `'even'`).
    Even,
    /// Constant extension: the end value repeated (SciPy `'constant'`).
    Constant,
    /// No extension (SciPy `None`). `padlen` is then ignored.
    None,
}

/// Options of [`filtfilt_with`], the keyword arguments of `scipy.signal.filtfilt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiltfiltOptions {
    /// The extension at each end of the signal.
    pub padtype: FiltfiltPad,
    /// The length of each extension. `None` uses `3 * max(a.len(), b.len())`, as SciPy does.
    pub padlen: Option<usize>,
}

impl Default for FiltfiltOptions {
    fn default() -> Self {
        Self {
            padtype: FiltfiltPad::Odd,
            padlen: None,
        }
    }
}

/// Apply zero-phase forward-backward filtering, with the SciPy defaults
/// (`padtype='odd'`, `padlen=None`, `method='pad'`).
///
/// The result has the length of `x`. The signal is extended at both ends
/// (odd extension, `3 * max(a.len(), b.len())` values), filtered forward and
/// backward from the steady state of the filter, and the extension is removed.
/// This eliminates the phase distortion of a single-pass IIR filter.
///
/// `method='gust'`, `irlen` and `axis` of SciPy are not supported.
///
/// # Example
///
/// ```
/// use scikit::signal::filtfilt;
///
/// let b = vec![0.5, 0.5]; // simple 2-tap FIR
/// let a = vec![1.0];
/// let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
/// let y = filtfilt(&b, &a, &x);
/// assert_eq!(y.len(), x.len());
/// ```
///
/// # Panics
///
/// Panics when the extension is not shorter than `x`; see [`filtfilt_with`].
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    filtfilt_with(b, a, x, &FiltfiltOptions::default())
}

/// Apply zero-phase forward-backward filtering with the options `padtype` and
/// `padlen` of `scipy.signal.filtfilt` (`method='pad'`).
///
/// An empty `x` gives an empty result. `padlen` is ignored for
/// [`FiltfiltPad::None`], as in SciPy. `method='gust'`, `irlen` and `axis` of
/// SciPy are not supported.
///
/// # Panics
///
/// Panics when the extension is used and `x.len()` is not greater than the
/// extension length (the message names `padlen`).
pub fn filtfilt_with(b: &[f64], a: &[f64], x: &[f64], options: &FiltfiltOptions) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }

    let ntaps = a.len().max(b.len());
    let edge = match options.padtype {
        FiltfiltPad::None => 0,
        _ => options.padlen.unwrap_or(3 * ntaps),
    };
    assert!(
        edge == 0 || x.len() > edge,
        "filtfilt: the length of x ({}) must be greater than padlen ({})",
        x.len(),
        edge
    );

    let n = x.len();
    let ext: Vec<f64> = if edge > 0 {
        let mut ext = Vec::with_capacity(n + 2 * edge);
        match options.padtype {
            FiltfiltPad::Odd => {
                ext.extend((0..edge).map(|k| 2.0 * x[0] - x[edge - k]));
                ext.extend_from_slice(x);
                ext.extend((0..edge).map(|k| 2.0 * x[n - 1] - x[n - 2 - k]));
            }
            FiltfiltPad::Even => {
                ext.extend((0..edge).map(|k| x[edge - k]));
                ext.extend_from_slice(x);
                ext.extend((0..edge).map(|k| x[n - 2 - k]));
            }
            FiltfiltPad::Constant => {
                ext.extend(std::iter::repeat_n(x[0], edge));
                ext.extend_from_slice(x);
                ext.extend(std::iter::repeat_n(x[n - 1], edge));
            }
            FiltfiltPad::None => unreachable!("edge is 0 for FiltfiltPad::None"),
        }
        ext
    } else {
        x.to_vec()
    };

    // Steady state of the filter, scaled by the first value: no start transient.
    let zi = lfilter_zi(b, a);
    let zi_start: Vec<f64> = zi.iter().map(|&v| v * ext[0]).collect();
    let (forward, _) = lfilter_with_state(b, a, &ext, &zi_start);

    // Backward pass, started from the steady state scaled by the last forward value.
    let y0 = *forward.last().expect("the extended signal is not empty");
    let reversed: Vec<f64> = forward.into_iter().rev().collect();
    let zi_end: Vec<f64> = zi.iter().map(|&v| v * y0).collect();
    let (backward, _) = lfilter_with_state(b, a, &reversed, &zi_end);

    let y: Vec<f64> = backward.into_iter().rev().collect();
    if edge > 0 {
        y[edge..y.len() - edge].to_vec()
    } else {
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_filtfilt_preserves_length() {
        let b = vec![0.25, 0.25, 0.25, 0.25];
        let a = vec![1.0];
        let x: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / 100.0).sin())
            .collect();
        let y = filtfilt(&b, &a, &x);
        assert_eq!(y.len(), x.len());
    }

    #[test]
    fn test_filtfilt_zero_phase() {
        // For a symmetric signal, filtfilt should preserve symmetry better
        // than a single-pass filter (which introduces phase shift).
        let n = 64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).sin())
            .collect();

        let b = vec![0.5, 0.5];
        let a = vec![1.0];

        let y = filtfilt(&b, &a, &x);

        // Zero-phase: peak of output should align with peak of input.
        // Find the index of maximum in mid-section to avoid edge effects.
        let start = n / 4;
        let end = 3 * n / 4;
        let x_peak = (start..end)
            .max_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap())
            .unwrap();
        let y_peak = (start..end)
            .max_by(|&i, &j| y[i].partial_cmp(&y[j]).unwrap())
            .unwrap();

        // Peaks should be at the same index (zero phase shift)
        assert_eq!(x_peak, y_peak, "filtfilt should not shift peaks");
    }

    #[test]
    fn test_filtfilt_identity() {
        // Identity filter b=[1], a=[1] should return input unchanged
        let b = vec![1.0];
        let a = vec![1.0];
        let x = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let y = filtfilt(&b, &a, &x);
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert!(
                (xi - yi).abs() < 1e-10,
                "identity filter: expected {xi}, got {yi}"
            );
        }
    }

    #[test]
    fn test_filtfilt_empty() {
        let b = vec![1.0, 0.5];
        let a = vec![1.0];
        let y = filtfilt(&b, &a, &[]);
        assert!(y.is_empty());
    }
}
