//! Digital filtering: FIR/IIR filters and Butterworth design.

use std::f64::consts::PI;

/// Filter type for design functions.
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    Lowpass,
    Highpass,
}

/// A second-order section (biquad) filter: (b0, b1, b2, a0, a1, a2).
#[derive(Debug, Clone, Copy)]
pub struct SosSection {
    pub b: [f64; 3],
    pub a: [f64; 3],
}

/// Apply a transfer-function filter (IIR/FIR) to data.
///
/// `b` = numerator coefficients, `a` = denominator coefficients.
/// Uses direct form II transposed.
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let n = b.len().max(a.len());
    lfilter_with_state(b, a, x, &vec![0.0; n.saturating_sub(1)]).0
}

/// Apply a transfer-function filter with an initial state (SciPy `lfilter(b, a, x, zi=zi)`).
///
/// `zi` is the delay line at the start, with `max(b.len(), a.len()) - 1` values.
/// Returns `(y, zf)`: the output and the final delay line.
///
/// # Panics
/// Panics when `a[0]` is zero, or when `zi` has the wrong length.
pub fn lfilter_with_state(b: &[f64], a: &[f64], x: &[f64], zi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    assert!(!a.is_empty() && a[0].abs() > 1e-14, "a[0] must be nonzero");
    let a0 = a[0];

    let n = b.len().max(a.len());
    assert!(
        zi.len() == n - 1,
        "zi length must be {}, got {}",
        n - 1,
        zi.len()
    );
    // The delay line: the first n - 1 entries start from zi, the last is 0.
    let mut d = vec![0.0; n];
    d[..n - 1].copy_from_slice(zi);
    let mut y = Vec::with_capacity(x.len());

    for &xi in x {
        let yi = b.first().copied().unwrap_or(0.0) / a0 * xi + d[0];
        y.push(yi);

        for j in 0..n - 1 {
            let bj1 = if j + 1 < b.len() { b[j + 1] } else { 0.0 };
            let aj1 = if j + 1 < a.len() { a[j + 1] } else { 0.0 };
            d[j] = bj1 / a0 * xi - aj1 / a0 * yi + d[j + 1];
        }
        d[n - 1] = 0.0;
    }

    (y, d[..n - 1].to_vec())
}

/// Initial state of a filter for a step input of 1 (SciPy `lfilter_zi(b, a)`).
///
/// Multiply the result by the first input value to start the filter at steady
/// state. SciPy raises an error for a single tap (`b` and `a` of length 1);
/// this function returns an empty state for it.
///
/// # Panics
/// Panics when `a[0]` is zero, or when the steady-state system is singular
/// (a pole at z = 1).
///
/// ```
/// use scikit::signal::lfilter_zi;
/// let zi = lfilter_zi(&[1.0], &[1.0]);
/// assert!(zi.is_empty());
/// ```
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Vec<f64> {
    assert!(!a.is_empty() && a[0].abs() > 1e-14, "a[0] must be nonzero");
    let a0 = a[0];

    let n = b.len().max(a.len());
    if n == 1 {
        return Vec::new();
    }
    // Normalise by a[0] and pad the shorter of b and a with zeros.
    let mut bn: Vec<f64> = b.iter().map(|v| v / a0).collect();
    let mut an: Vec<f64> = a.iter().map(|v| v / a0).collect();
    bn.resize(n, 0.0);
    an.resize(n, 0.0);

    let size = n - 1;
    // Companion matrix C of a: first row -a[1..], subdiagonal ones.
    let mut c = vec![vec![0.0; size]; size];
    for (j, cell) in c[0].iter_mut().enumerate() {
        *cell = -an[j + 1];
    }
    for i in 1..size {
        c[i][i - 1] = 1.0;
    }
    // m = I - C^T, rhs[k] = b[k + 1] - a[k + 1] * b[0]
    let mut m = vec![vec![0.0; size]; size];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let identity = if i == j { 1.0 } else { 0.0 };
            *cell = identity - c[j][i];
        }
    }
    let rhs: Vec<f64> = (0..size).map(|k| bn[k + 1] - an[k + 1] * bn[0]).collect();
    solve_linear(m, rhs)
}

/// Solve `m x = rhs` by Gaussian elimination with partial pivoting.
///
/// # Panics
/// Panics with a message that contains `singular` when a pivot has an
/// absolute value below 1e-14.
fn solve_linear(mut m: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
            .unwrap_or(col);
        assert!(
            m[pivot][col].abs() >= 1e-14,
            "singular system: a pole at z = 1 (pivot {} in column {col})",
            m[pivot][col]
        );
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        let pivot_row = m[col].clone();
        let pivot_rhs = rhs[col];
        for row in col + 1..n {
            let factor = m[row][col] / pivot_row[col];
            for (cell, p) in m[row].iter_mut().zip(&pivot_row).skip(col) {
                *cell -= factor * p;
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }

    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| m[row][k] * x[k]).sum();
        x[row] = (rhs[row] - tail) / m[row][row];
    }
    x
}

/// Filter data using cascaded second-order sections.
pub fn sosfilt(sos: &[SosSection], x: &[f64]) -> Vec<f64> {
    let mut data = x.to_vec();
    for section in sos {
        data = lfilter(&section.b, &section.a, &data);
    }
    data
}

/// Design a Butterworth filter (analog prototype → digital via bilinear transform).
///
/// Returns `ceil(order / 2)` sections: one second-order section per conjugate
/// pole pair, plus one first-order section for odd order. The first-order
/// section is the last one (like SciPy `sos`); its `b[2]` and `a[2]` are 0.
///
/// * `order` - Filter order (1-8 typical)
/// * `wn` - Normalized cutoff frequency (0 to 1, where 1 = Nyquist)
/// * `ftype` - Lowpass or Highpass
pub fn butter(order: usize, wn: f64, ftype: FilterType) -> Vec<SosSection> {
    assert!(order > 0, "order must be > 0");
    assert!(wn > 0.0 && wn < 1.0, "wn must be in (0, 1)");

    // Pre-warp
    let fs = 2.0;
    let warped = 2.0 * fs * (PI * wn / fs).tan();

    // Analog Butterworth poles
    let mut poles = Vec::new();
    for k in 0..order {
        let theta = PI * (2 * k + order + 1) as f64 / (2 * order) as f64;
        poles.push((warped * theta.cos(), warped * theta.sin()));
    }

    // Convert to second-order sections via bilinear transform
    let mut sections = Vec::new();

    for (k, &(p_re, p_im)) in poles.iter().enumerate() {
        // For odd order the real pole is the middle one: k = (order - 1) / 2.
        // Test `k` instead of `|p_im| < 1e-14`: sin(PI) is not exactly 0 for every wn.
        let is_real = order % 2 == 1 && k == (order - 1) / 2;
        if is_real {
            // Real pole → first-order section (embed in SOS)
            let pd_re = (1.0 + p_re / (2.0 * fs)) / (1.0 - p_re / (2.0 * fs));

            let (b, a) = match ftype {
                FilterType::Lowpass => {
                    let g = (1.0 - pd_re) / 2.0;
                    ([g, g, 0.0], [1.0, -pd_re, 0.0])
                }
                FilterType::Highpass => {
                    let g = (1.0 + pd_re) / 2.0;
                    ([g, -g, 0.0], [1.0, -pd_re, 0.0])
                }
            };
            sections.push(SosSection { b, a });
        } else if p_im > 0.0 {
            // Upper-half-plane pole → second-order section (its conjugate is implied)
            // Bilinear transform: z = (1 + s/(2fs)) / (1 - s/(2fs))
            let c = 2.0 * fs;
            let denom_re = 1.0 - p_re / c;
            let denom_im = -p_im / c;
            let num_re = 1.0 + p_re / c;
            let num_im = p_im / c;
            let d2 = denom_re * denom_re + denom_im * denom_im;
            let pd_re = (num_re * denom_re + num_im * denom_im) / d2;
            let pd_im = (num_im * denom_re - num_re * denom_im) / d2;

            let a1 = -2.0 * pd_re;
            let a2 = pd_re * pd_re + pd_im * pd_im;

            let (b0, b1, b2) = match ftype {
                FilterType::Lowpass => {
                    let g = (1.0 + a1 + a2) / 4.0;
                    (g, 2.0 * g, g)
                }
                FilterType::Highpass => {
                    let g = (1.0 - a1 + a2) / 4.0;
                    (g, -2.0 * g, g)
                }
            };

            sections.push(SosSection {
                b: [b0, b1, b2],
                a: [1.0, a1, a2],
            });
        }
        // p_im < 0: the conjugate pole is already in a section; skip it.
    }

    sections
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn test_lfilter_fir() {
        // Simple moving average (FIR)
        let b = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let a = vec![1.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = lfilter(&b, &a, &x);
        assert_eq!(y.len(), 5);
        assert!((y[2] - 2.0).abs() < 1e-10); // (1+2+3)/3
        assert!((y[3] - 3.0).abs() < 1e-10); // (2+3+4)/3
    }

    #[test]
    fn test_lfilter_iir() {
        // First-order IIR: y[n] = x[n] + 0.5*y[n-1]
        let b = vec![1.0];
        let a = vec![1.0, -0.5];
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let y = lfilter(&b, &a, &x);
        // Impulse response: 1, 0.5, 0.25, 0.125, 0.0625
        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 0.5).abs() < 1e-10);
        assert!((y[2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_butter_lowpass() {
        let sos = butter(2, 0.3, FilterType::Lowpass);
        assert!(!sos.is_empty());
        // Apply to signal: should attenuate high frequencies
        let n = 100;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * std::f64::consts::PI * 2.0 * t).sin()
                    + (2.0 * std::f64::consts::PI * 40.0 * t).sin()
            })
            .collect();
        let y = sosfilt(&sos, &x);
        assert_eq!(y.len(), n);
    }

    /// Magnitude response of cascaded sections at normalized frequency `w` (radians).
    fn sos_response(sos: &[SosSection], w: f64) -> f64 {
        let mut gain = 1.0;
        for s in sos {
            // z^-1 = cos(w) - j sin(w), z^-2 = cos(2w) - j sin(2w)
            let (c1, s1) = (w.cos(), -w.sin());
            let (c2, s2) = ((2.0 * w).cos(), -(2.0 * w).sin());
            let num_re = s.b[0] + s.b[1] * c1 + s.b[2] * c2;
            let num_im = s.b[1] * s1 + s.b[2] * s2;
            let den_re = s.a[0] + s.a[1] * c1 + s.a[2] * c2;
            let den_im = s.a[1] * s1 + s.a[2] * s2;
            gain *= num_re.hypot(num_im) / den_re.hypot(den_im);
        }
        gain
    }

    #[test]
    fn butter_section_count() {
        assert_eq!(butter(1, 0.3, FilterType::Lowpass).len(), 1);
        assert_eq!(butter(2, 0.3, FilterType::Lowpass).len(), 1);
        assert_eq!(butter(3, 0.3, FilterType::Lowpass).len(), 2);
        assert_eq!(butter(4, 0.3, FilterType::Lowpass).len(), 2);
        assert_eq!(butter(5, 0.3, FilterType::Lowpass).len(), 3);
    }

    #[test]
    fn butter_odd_order_dc_gain_and_cutoff() {
        for order in [3, 5] {
            let sos = butter(order, 0.3, FilterType::Lowpass);
            assert!(
                (sos_response(&sos, 0.0) - 1.0).abs() < 1e-9,
                "order {order} DC gain"
            );
            assert!(
                (sos_response(&sos, PI * 0.3) - FRAC_1_SQRT_2).abs() < 1e-6,
                "order {order} -3 dB at cutoff"
            );
        }
    }

    #[test]
    fn butter_even_order_cutoff() {
        for order in [2, 4] {
            let sos = butter(order, 0.3, FilterType::Lowpass);
            assert!(
                (sos_response(&sos, PI * 0.3) - FRAC_1_SQRT_2).abs() < 1e-6,
                "order {order} -3 dB at cutoff"
            );
        }
    }

    #[test]
    fn butter_highpass_odd_order() {
        let sos = butter(3, 0.3, FilterType::Highpass);
        assert!((sos_response(&sos, PI) - 1.0).abs() < 1e-9);
        assert!((sos_response(&sos, PI * 0.3) - FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn butter_odd_order_stable() {
        for order in [1, 3, 5] {
            for s in butter(order, 0.3, FilterType::Lowpass) {
                assert!(s.a[2].abs() < 1.0, "order {order}: |a2| must be < 1");
                assert!(
                    s.a[1].abs() < 1.0 + s.a[2],
                    "order {order}: |a1| must be < 1 + a2"
                );
            }
        }
    }

    #[test]
    fn test_sosfilt_passthrough() {
        // Identity filter: b=[1,0,0], a=[1,0,0]
        let sos = vec![SosSection {
            b: [1.0, 0.0, 0.0],
            a: [1.0, 0.0, 0.0],
        }];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = sosfilt(&sos, &x);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
