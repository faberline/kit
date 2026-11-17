//! Linear models: ordinary least squares and ridge regression.

use arraykit::NdArray;
use thiserror::Error;

/// Errors of the linear models.
#[derive(Debug, Error, PartialEq)]
pub enum LinearModelError {
    /// The shapes of the inputs do not agree.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    /// An input value or parameter is not allowed.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// `predict` was called before `fit`.
    #[error("model is not fitted")]
    NotFitted,
    /// A linear algebra routine failed.
    #[error("linear algebra error: {0}")]
    Linalg(String),
}

/// Result type of the linear models.
pub type Result<T> = std::result::Result<T, LinearModelError>;

fn linalg(e: impl std::fmt::Display) -> LinearModelError {
    LinearModelError::Linalg(e.to_string())
}

/// Checks `x` (2-D) and `y` (1-D) and returns `(n_samples, n_features)`.
fn check_xy(x: &NdArray<f64>, y: &NdArray<f64>) -> Result<(usize, usize)> {
    if x.ndim() != 2 || y.ndim() != 1 {
        return Err(LinearModelError::ShapeMismatch(format!(
            "x must be 2-D and y 1-D, got x with {} dims and y with {} dims",
            x.ndim(),
            y.ndim()
        )));
    }
    let (n, p) = (x.dims()[0], x.dims()[1]);
    if y.dims()[0] != n {
        return Err(LinearModelError::ShapeMismatch(format!(
            "x has {} rows but y has {}",
            n,
            y.dims()[0]
        )));
    }
    if n == 0 || p == 0 {
        return Err(LinearModelError::InvalidInput(
            "x must have at least one row and one column".to_string(),
        ));
    }
    if x.data()
        .iter()
        .chain(y.data().iter())
        .any(|v| !v.is_finite())
    {
        return Err(LinearModelError::InvalidInput(
            "x and y must be finite".to_string(),
        ));
    }
    Ok((n, p))
}

/// Centred `x`, centred `y`, column means of `x` and mean of `y`.
type Centred = (NdArray<f64>, NdArray<f64>, Vec<f64>, f64);

/// Centres `x` by its column means and `y` by its mean.
/// Returns `(x_centred, y_centred, x_mean, y_mean)`.
fn centre(x: &NdArray<f64>, y: &NdArray<f64>, n: usize, p: usize) -> Result<Centred> {
    let xd = x.data();
    let yd = y.data();
    let mut x_mean = vec![0.0; p];
    for i in 0..n {
        for j in 0..p {
            x_mean[j] += xd[i * p + j];
        }
    }
    for m in x_mean.iter_mut() {
        *m /= n as f64;
    }
    let y_mean = yd.iter().sum::<f64>() / n as f64;
    let mut xc = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            xc.push(xd[i * p + j] - x_mean[j]);
        }
    }
    let yc: Vec<f64> = yd.iter().map(|v| v - y_mean).collect();
    let xc = NdArray::new(xc, vec![n, p]).map_err(linalg)?;
    let yc = NdArray::new(yc, vec![n]).map_err(linalg)?;
    Ok((xc, yc, x_mean, y_mean))
}

/// Predicts `x @ coef + intercept` for a fitted model.
fn predict_linear(
    fitted: bool,
    coef: &[f64],
    intercept: f64,
    x: &NdArray<f64>,
) -> Result<NdArray<f64>> {
    if !fitted {
        return Err(LinearModelError::NotFitted);
    }
    if x.ndim() != 2 || x.dims()[1] != coef.len() {
        return Err(LinearModelError::ShapeMismatch(format!(
            "x must be 2-D with {} columns",
            coef.len()
        )));
    }
    let (n, p) = (x.dims()[0], x.dims()[1]);
    let xd = x.data();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = intercept;
        for j in 0..p {
            acc += xd[i * p + j] * coef[j];
        }
        out.push(acc);
    }
    NdArray::new(out, vec![n]).map_err(linalg)
}

/// Ordinary least squares regression.
#[derive(Debug, Clone, Default)]
pub struct LinearRegression {
    coef: Vec<f64>,
    intercept: f64,
    fitted: bool,
}

impl LinearRegression {
    /// Creates an unfitted model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fits the model to `x` (`[n_samples, n_features]`) and `y` (`[n_samples]`).
    pub fn fit(&mut self, x: &NdArray<f64>, y: &NdArray<f64>) -> Result<()> {
        let (n, p) = check_xy(x, y)?;
        let (xc, yc, x_mean, y_mean) = centre(x, y, n, p)?;
        let sol = xc.lstsq(&yc).map_err(linalg)?;
        let coef = sol.x.data().to_vec();
        self.intercept = y_mean - coef.iter().zip(&x_mean).map(|(c, m)| c * m).sum::<f64>();
        self.coef = coef;
        self.fitted = true;
        Ok(())
    }

    /// Predicts `y` for `x` (`[n_samples, n_features]`).
    pub fn predict(&self, x: &NdArray<f64>) -> Result<NdArray<f64>> {
        predict_linear(self.fitted, &self.coef, self.intercept, x)
    }

    /// Fitted coefficients; empty before `fit`.
    pub fn coef(&self) -> &[f64] {
        &self.coef
    }

    /// Fitted intercept; `0.0` before `fit`.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

/// Ridge (L2-penalised) regression.
#[derive(Debug, Clone)]
pub struct Ridge {
    alpha: f64,
    coef: Vec<f64>,
    intercept: f64,
    fitted: bool,
}

impl Ridge {
    /// Creates an unfitted model with penalty `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            coef: Vec::new(),
            intercept: 0.0,
            fitted: false,
        }
    }

    /// Fits the model; the intercept is not penalised.
    pub fn fit(&mut self, x: &NdArray<f64>, y: &NdArray<f64>) -> Result<()> {
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(LinearModelError::InvalidInput(
                "alpha must be finite and not negative".to_string(),
            ));
        }
        let (n, p) = check_xy(x, y)?;
        let (xc, yc, x_mean, y_mean) = centre(x, y, n, p)?;
        let xd = xc.data();
        let yd = yc.data();
        // A = XᵀX + alpha·I and b = Xᵀy, with plain loops.
        let mut a = vec![0.0; p * p];
        let mut b = vec![0.0; p];
        for i in 0..n {
            for j in 0..p {
                let xij = xd[i * p + j];
                b[j] += xij * yd[i];
                for k in 0..p {
                    a[j * p + k] += xij * xd[i * p + k];
                }
            }
        }
        for j in 0..p {
            a[j * p + j] += self.alpha;
        }
        let a = NdArray::new(a, vec![p, p]).map_err(linalg)?;
        let b = NdArray::new(b, vec![p]).map_err(linalg)?;
        let w = a.solve(&b).map_err(linalg)?;
        let coef = w.data().to_vec();
        self.intercept = y_mean - coef.iter().zip(&x_mean).map(|(c, m)| c * m).sum::<f64>();
        self.coef = coef;
        self.fitted = true;
        Ok(())
    }

    /// Predicts `y` for `x` (`[n_samples, n_features]`).
    pub fn predict(&self, x: &NdArray<f64>) -> Result<NdArray<f64>> {
        predict_linear(self.fitted, &self.coef, self.intercept, x)
    }

    /// Fitted coefficients; empty before `fit`.
    pub fn coef(&self) -> &[f64] {
        &self.coef
    }

    /// Fitted intercept; `0.0` before `fit`.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// The penalty.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: Vec<f64>, shape: Vec<usize>) -> NdArray<f64> {
        NdArray::new(data, shape).unwrap()
    }

    fn line_x() -> NdArray<f64> {
        arr(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5, 1])
    }
    fn line_y() -> NdArray<f64> {
        arr(vec![1.0, 3.0, 5.0, 7.0, 9.0], vec![5])
    }
    fn two_x() -> NdArray<f64> {
        arr(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
            vec![5, 2],
        )
    }
    fn two_y() -> NdArray<f64> {
        // y = 3*x0 - 2*x1 + 0.5
        let x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]];
        arr(
            x.iter().map(|r| 3.0 * r[0] - 2.0 * r[1] + 0.5).collect(),
            vec![5],
        )
    }

    #[test]
    fn ols_recovers_line() {
        let mut m = LinearRegression::new();
        m.fit(&line_x(), &line_y()).unwrap();
        assert!((m.coef()[0] - 2.0).abs() < 1e-9);
        assert!((m.intercept() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ols_two_features() {
        let mut m = LinearRegression::new();
        m.fit(&two_x(), &two_y()).unwrap();
        assert!((m.coef()[0] - 3.0).abs() < 1e-9);
        assert!((m.coef()[1] + 2.0).abs() < 1e-9);
        assert!((m.intercept() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn predict_uses_fit() {
        let mut m = LinearRegression::new();
        m.fit(&line_x(), &line_y()).unwrap();
        let p = m.predict(&arr(vec![10.0], vec![1, 1])).unwrap();
        assert!((p.data()[0] - 21.0).abs() < 1e-9);
    }

    #[test]
    fn ridge_alpha_zero_equals_ols() {
        let mut ols = LinearRegression::new();
        ols.fit(&two_x(), &two_y()).unwrap();
        let mut r = Ridge::new(0.0);
        r.fit(&two_x(), &two_y()).unwrap();
        for (a, b) in r.coef().iter().zip(ols.coef()) {
            assert!((a - b).abs() < 1e-9);
        }
        assert!((r.intercept() - ols.intercept()).abs() < 1e-9);
    }

    #[test]
    fn ridge_large_alpha_shrinks() {
        let mut big = Ridge::new(1e9);
        big.fit(&line_x(), &line_y()).unwrap();
        assert!(big.coef()[0].abs() < 1e-3);
        let mut one = Ridge::new(1.0);
        one.fit(&line_x(), &line_y()).unwrap();
        let c = one.coef()[0];
        assert!(c > 0.0 && c < 2.0);
    }

    #[test]
    fn ridge_intercept_is_not_penalised() {
        let mut r = Ridge::new(1e9);
        r.fit(&line_x(), &line_y()).unwrap();
        let p = r.predict(&arr(vec![2.0], vec![1, 1])).unwrap();
        assert!((p.data()[0] - 5.0).abs() < 1e-3);
    }

    #[test]
    fn mismatched_rows_is_error() {
        let y2 = arr(vec![1.0, 2.0], vec![2]);
        let mut m = LinearRegression::new();
        assert!(matches!(
            m.fit(&line_x(), &y2),
            Err(LinearModelError::ShapeMismatch(_))
        ));
        let mut r = Ridge::new(1.0);
        assert!(matches!(
            r.fit(&line_x(), &y2),
            Err(LinearModelError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn wrong_ndim_is_error() {
        let y = arr(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let x = arr(vec![0.0, 1.0, 2.0], vec![3, 1]);
        let mut m = LinearRegression::new();
        assert!(matches!(
            m.fit(&x, &y),
            Err(LinearModelError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn non_finite_is_error() {
        let x = arr(vec![0.0, f64::NAN, 2.0], vec![3, 1]);
        let y = arr(vec![1.0, 2.0, 3.0], vec![3]);
        let mut m = LinearRegression::new();
        assert!(matches!(
            m.fit(&x, &y),
            Err(LinearModelError::InvalidInput(_))
        ));
    }

    #[test]
    fn negative_alpha_is_error() {
        let mut r = Ridge::new(-1.0);
        assert!(matches!(
            r.fit(&line_x(), &line_y()),
            Err(LinearModelError::InvalidInput(_))
        ));
    }

    #[test]
    fn predict_before_fit_is_error() {
        let m = LinearRegression::new();
        assert!(matches!(
            m.predict(&line_x()),
            Err(LinearModelError::NotFitted)
        ));
        let r = Ridge::new(1.0);
        assert!(matches!(
            r.predict(&line_x()),
            Err(LinearModelError::NotFitted)
        ));
    }

    #[test]
    fn predict_wrong_feature_count_is_error() {
        let mut m = LinearRegression::new();
        m.fit(&line_x(), &line_y()).unwrap();
        let bad = arr(vec![1.0, 2.0], vec![1, 2]);
        assert!(matches!(
            m.predict(&bad),
            Err(LinearModelError::ShapeMismatch(_))
        ));
    }
}
