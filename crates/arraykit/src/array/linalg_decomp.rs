//! Matrix decompositions: QR, Cholesky, Eigenvalue, SVD, Least Squares.

use super::error::{ArrayError, Result};
use super::ndarray::NdArray;

/// Result of QR decomposition: A = Q * R.
pub struct QrResult {
    /// Orthogonal matrix Q (m × m).
    pub q: NdArray<f64>,
    /// Upper triangular matrix R (m × n).
    pub r: NdArray<f64>,
}

/// Result of eigenvalue decomposition.
pub struct EigResult {
    /// Eigenvalues (may be complex — real parts stored).
    pub values: Vec<f64>,
    /// Eigenvectors as columns of an n × n matrix.
    pub vectors: NdArray<f64>,
}

/// Result of Singular Value Decomposition: A = U * diag(S) * V^T.
pub struct SvdResult {
    /// Left singular vectors (m × m).
    pub u: NdArray<f64>,
    /// Singular values (min(m,n)).
    pub s: Vec<f64>,
    /// Right singular vectors transposed (n × n).
    pub vt: NdArray<f64>,
}

/// Result of least squares: min ||Ax - b||^2.
pub struct LstsqResult {
    /// Solution vector x.
    pub x: NdArray<f64>,
    /// Sum of squared residuals.
    pub residuals: f64,
    /// Rank of matrix A.
    pub rank: usize,
}

impl NdArray<f64> {
    /// QR decomposition using Householder reflections.
    ///
    /// Returns Q (orthogonal) and R (upper triangular) such that A = Q * R.
    pub fn qr(&self) -> Result<QrResult> {
        if self.ndim() != 2 {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.ndim()],
            });
        }

        let m = self.dims()[0];
        let n = self.dims()[1];

        // Work on a copy for R
        let mut r_data = self.data.clone();
        // Q starts as identity
        let mut q_data = vec![0.0; m * m];
        for i in 0..m {
            q_data[i * m + i] = 1.0;
        }

        let min_mn = m.min(n);

        for k in 0..min_mn {
            // Extract column k below diagonal
            let mut col = vec![0.0; m - k];
            for i in k..m {
                col[i - k] = r_data[i * n + k];
            }

            // Compute Householder vector
            let norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-14 {
                continue;
            }

            let sign = if col[0] >= 0.0 { 1.0 } else { -1.0 };
            col[0] += sign * norm;
            let col_norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if col_norm < 1e-14 {
                continue;
            }
            for v in &mut col {
                *v /= col_norm;
            }

            // Apply H = I - 2*v*v^T to R (columns k..n)
            for j in k..n {
                let mut dot = 0.0;
                for i in 0..col.len() {
                    dot += col[i] * r_data[(k + i) * n + j];
                }
                for i in 0..col.len() {
                    r_data[(k + i) * n + j] -= 2.0 * col[i] * dot;
                }
            }

            // Apply H to Q (all columns)
            for j in 0..m {
                let mut dot = 0.0;
                for i in 0..col.len() {
                    dot += col[i] * q_data[(k + i) * m + j];
                }
                for i in 0..col.len() {
                    q_data[(k + i) * m + j] -= 2.0 * col[i] * dot;
                }
            }
        }

        // Transpose Q (we accumulated H^T, so Q = (H1 H2 ... Hk)^T needs transpose)
        let mut qt = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                qt[i * m + j] = q_data[j * m + i];
            }
        }

        Ok(QrResult {
            q: NdArray::new(qt, vec![m, m])?,
            r: NdArray::new(r_data, vec![m, n])?,
        })
    }

    /// Cholesky decomposition for symmetric positive-definite matrices.
    ///
    /// Returns lower triangular L such that A = L * L^T.
    pub fn cholesky(&self) -> Result<NdArray<f64>> {
        if self.ndim() != 2 {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.ndim()],
            });
        }

        let n = self.dims()[0];
        if self.dims()[1] != n {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![n, n],
                got: self.dims().to_vec(),
            });
        }

        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }

                if i == j {
                    let diag = self.data[i * n + i] - sum;
                    if diag <= 0.0 {
                        return Err(ArrayError::InvalidOperation(
                            "matrix is not positive definite".into(),
                        ));
                    }
                    l[i * n + j] = diag.sqrt();
                } else {
                    l[i * n + j] = (self.data[i * n + j] - sum) / l[j * n + j];
                }
            }
        }

        NdArray::new(l, vec![n, n])
    }

    /// Eigenvalue decomposition using the QR algorithm with shifts.
    ///
    /// Works on square matrices. Returns eigenvalues and eigenvectors.
    /// For non-symmetric matrices, eigenvalues are real approximations.
    pub fn eig(&self) -> Result<EigResult> {
        if self.ndim() != 2 {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.ndim()],
            });
        }

        let n = self.dims()[0];
        if self.dims()[1] != n {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![n, n],
                got: self.dims().to_vec(),
            });
        }

        if n == 0 {
            return Ok(EigResult {
                values: vec![],
                vectors: NdArray::new(vec![], vec![0, 0])?,
            });
        }

        if n == 1 {
            return Ok(EigResult {
                values: vec![self.data[0]],
                vectors: NdArray::new(vec![1.0], vec![1, 1])?,
            });
        }

        // QR iteration: repeatedly do QR decomposition and form RQ
        let max_iter = 200;
        let tol = 1e-10;

        let mut a_data = self.data.clone();
        // Accumulate eigenvectors
        let mut v_data = vec![0.0; n * n];
        for i in 0..n {
            v_data[i * n + i] = 1.0;
        }

        for _ in 0..max_iter {
            // Wilkinson shift: use eigenvalue of bottom-right 2×2 closer to a[n-1][n-1]
            let shift = if n >= 2 {
                wilkinson_shift(
                    a_data[(n - 2) * n + (n - 2)],
                    a_data[(n - 2) * n + (n - 1)],
                    a_data[(n - 1) * n + (n - 2)],
                    a_data[(n - 1) * n + (n - 1)],
                )
            } else {
                0.0
            };

            // Apply shift: A - σI
            for i in 0..n {
                a_data[i * n + i] -= shift;
            }

            // QR decomposition of shifted matrix
            let a_mat = NdArray::new(a_data.clone(), vec![n, n])?;
            let qr = a_mat.qr()?;

            // A = R * Q + σI
            let rq = qr.r.matmul(&qr.q)?;
            a_data = rq.data.clone();
            for i in 0..n {
                a_data[i * n + i] += shift;
            }

            // Accumulate eigenvectors: V = V * Q
            let v_mat = NdArray::new(v_data.clone(), vec![n, n])?;
            let new_v = v_mat.matmul(&qr.q)?;
            v_data = new_v.data;

            // Check convergence: sub-diagonal elements should be near zero
            let mut max_off = 0.0_f64;
            for i in 1..n {
                max_off = max_off.max(a_data[i * n + (i - 1)].abs());
            }
            if max_off < tol {
                break;
            }
        }

        // Extract eigenvalues from diagonal
        let values: Vec<f64> = (0..n).map(|i| a_data[i * n + i]).collect();

        Ok(EigResult {
            values,
            vectors: NdArray::new(v_data, vec![n, n])?,
        })
    }

    /// Symmetric eigenvalue decomposition with the cyclic Jacobi method.
    ///
    /// Returns eigenvalues in ascending order and the eigenvectors as columns.
    /// Returns an error for a non-square, non-symmetric or non-finite matrix,
    /// and when the method does not converge in 100 sweeps.
    #[allow(clippy::needless_range_loop)]
    pub fn eigh(&self) -> Result<EigResult> {
        if self.ndim() != 2 {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.ndim()],
            });
        }

        let n = self.dims()[0];
        if self.dims()[1] != n {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![n, n],
                got: self.dims().to_vec(),
            });
        }

        if self.data.iter().any(|x| !x.is_finite()) {
            return Err(ArrayError::InvalidOperation(
                "matrix contains NaN or infinity".into(),
            ));
        }

        if n == 0 {
            return Ok(EigResult {
                values: vec![],
                vectors: NdArray::new(vec![], vec![0, 0])?,
            });
        }

        if n == 1 {
            return Ok(EigResult {
                values: vec![self.data[0]],
                vectors: NdArray::new(vec![1.0], vec![1, 1])?,
            });
        }

        let scale = self.data.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
        for i in 0..n {
            for j in (i + 1)..n {
                if (self.data[i * n + j] - self.data[j * n + i]).abs() > 1e-12 * scale {
                    return Err(ArrayError::InvalidOperation(
                        "matrix is not symmetric".into(),
                    ));
                }
            }
        }

        let mut a = self.data.clone();
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        let total: f64 = a.iter().map(|x| x * x).sum();
        let off_sq = |a: &[f64]| -> f64 {
            let mut off = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off += a[i * n + j] * a[i * n + j];
                    }
                }
            }
            off
        };

        let mut converged = false;
        for _sweep in 0..100 {
            let off = off_sq(&a);
            if off == 0.0 || off < 1e-30 * total {
                converged = true;
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[p * n + q];
                    if apq == 0.0 {
                        continue;
                    }
                    let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                    let t = if theta == 0.0 {
                        1.0
                    } else {
                        theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                    };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;

                    for k in 0..n {
                        let (akp, akq) = (a[k * n + p], a[k * n + q]);
                        a[k * n + p] = c * akp - s * akq;
                        a[k * n + q] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                        a[p * n + k] = c * apk - s * aqk;
                        a[q * n + k] = s * apk + c * aqk;
                    }
                    for k in 0..n {
                        let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        if !converged {
            let off = off_sq(&a);
            if !(off == 0.0 || off < 1e-30 * total) {
                return Err(ArrayError::InvalidOperation("eigh did not converge".into()));
            }
        }

        let raw: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&x, &y| raw[x].total_cmp(&raw[y]));

        let values: Vec<f64> = order.iter().map(|&i| raw[i]).collect();
        let mut vectors_data = vec![0.0; n * n];
        for row in 0..n {
            for (new_col, &old_col) in order.iter().enumerate() {
                vectors_data[row * n + new_col] = v[row * n + old_col];
            }
        }

        Ok(EigResult {
            values,
            vectors: NdArray::new(vectors_data, vec![n, n])?,
        })
    }

    /// Singular Value Decomposition: A = U * diag(S) * V^T.
    ///
    /// One-sided Jacobi on the columns of A; A^T A is not formed. Returns U
    /// (m x m, orthonormal), the singular values in descending order (length
    /// min(m, n)) and V^T (n x n). Returns an error for a non-finite entry and
    /// when the Jacobi sweeps do not converge.
    pub fn svd(&self) -> Result<SvdResult> {
        if self.ndim() != 2 {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.ndim()],
            });
        }

        if self.data.iter().any(|v| !v.is_finite()) {
            return Err(ArrayError::InvalidOperation(
                "matrix contains NaN or infinity".into(),
            ));
        }

        let m = self.dims()[0];
        let n = self.dims()[1];

        let (u_data, s, vt_data) = if m >= n {
            // A = U S V^T, with the thin U (m x n) from the Jacobi sweeps
            let (u_thin, s, v) = svd_tall(&self.data, m, n)?;
            let mut cols = unit_columns(&u_thin, m, n);
            cols.resize(m, None);
            complete_columns(&mut cols, m)?;
            (columns_to_row_major(&cols, m), s, transpose_square(&v, n))
        } else {
            // A^T = U' S V'^T, so A = V' S U'^T: U = V' and V^T = U'^T
            let mut at = vec![0.0; n * m];
            for i in 0..n {
                for j in 0..m {
                    at[i * m + j] = self.data[j * n + i];
                }
            }
            let (u_thin, s, v) = svd_tall(&at, n, m)?;
            let mut cols = unit_columns(&u_thin, n, m);
            cols.resize(n, None);
            complete_columns(&mut cols, n)?;
            (v, s, transpose_square(&columns_to_row_major(&cols, n), n))
        };

        Ok(SvdResult {
            u: NdArray::new(u_data, vec![m, m])?,
            s,
            vt: NdArray::new(vt_data, vec![n, n])?,
        })
    }

    /// Least squares solution: min ||Ax - b||^2.
    ///
    /// Uses QR decomposition for numerical stability.
    pub fn lstsq(&self, b: &NdArray<f64>) -> Result<LstsqResult> {
        if self.ndim() != 2 || b.ndim() != 1 {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![2, 1],
                got: vec![self.ndim(), b.ndim()],
            });
        }

        let m = self.dims()[0];
        let n = self.dims()[1];
        if b.size() != m {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![m],
                got: vec![b.size()],
            });
        }

        // QR decomposition: A = Q * R
        let qr = self.qr()?;

        // Compute Q^T * b
        let qt_b = qr.q.t()?.dot(b)?;

        // Solve R * x = Q^T * b (back substitution on top n rows)
        let rank = m.min(n);
        let mut x = vec![0.0; n];

        for i in (0..rank).rev() {
            let r_ii = qr.r.data[i * n + i];
            if r_ii.abs() < 1e-12 {
                continue;
            }
            let mut sum = qt_b.data[i];
            for (j, xj) in x.iter().enumerate().take(n).skip(i + 1) {
                sum -= qr.r.data[i * n + j] * *xj;
            }
            x[i] = sum / r_ii;
        }

        // Compute residuals: ||Ax - b||^2
        let x_arr = NdArray::new(x.clone(), vec![n])?;
        let ax = self.dot(&x_arr)?;
        let residuals: f64 = ax
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&a, &bi)| (a - bi).powi(2))
            .sum();

        // Effective rank
        let eff_rank = (0..rank)
            .filter(|&i| qr.r.data[i * n + i].abs() > 1e-10)
            .count();

        Ok(LstsqResult {
            x: NdArray::new(x, vec![n])?,
            residuals,
            rank: eff_rank,
        })
    }

    /// Transpose helper for 2D arrays.
    fn t(&self) -> Result<NdArray<f64>> {
        if self.ndim() != 2 {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.ndim()],
            });
        }
        let m = self.dims()[0];
        let n = self.dims()[1];
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = self.data[i * n + j];
            }
        }
        NdArray::new(result, vec![n, m])
    }
}

/// One-sided (Hestenes) Jacobi SVD of the m x n row-major matrix `data`, m >= n.
/// Returns the thin U (m x n, row-major), the singular values in descending
/// order (length n) and V (n x n, row-major). A zero singular value gives a
/// zero column of U.
fn svd_tall(data: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut w = data.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let mut converged = false;
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let (wp, wq) = (w[i * n + p], w[i * n + q]);
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                if gamma == 0.0 || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                rotate_columns(&mut w, m, n, p, q, c, s);
                rotate_columns(&mut v, n, n, p, q, c, s);
                rotated = true;
            }
        }
        if !rotated {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(ArrayError::InvalidOperation("svd did not converge".into()));
    }

    // The column norms are the singular values. Sort them in descending order.
    let norms: Vec<f64> = (0..n)
        .map(|j| {
            (0..m)
                .map(|i| w[i * n + j] * w[i * n + j])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| norms[b].total_cmp(&norms[a]));
    let s: Vec<f64> = order.iter().map(|&j| norms[j]).collect();
    let s0 = s.first().copied().unwrap_or(0.0);

    let mut u = vec![0.0; m * n];
    let mut v_sorted = vec![0.0; n * n];
    for (new_j, &old_j) in order.iter().enumerate() {
        // Only a clearly non-zero singular value gets a left vector
        if s0 > 0.0 && s[new_j] > 1e-14 * s0 {
            for i in 0..m {
                u[i * n + new_j] = w[i * n + old_j] / s[new_j];
            }
        }
        for i in 0..n {
            v_sorted[i * n + new_j] = v[i * n + old_j];
        }
    }
    Ok((u, s, v_sorted))
}

/// Rotates columns p and q of the `rows` x `cols` row-major matrix `a`:
/// new_p = c * p - s * q and new_q = s * p + c * q.
fn rotate_columns(a: &mut [f64], rows: usize, cols: usize, p: usize, q: usize, c: f64, s: f64) {
    for i in 0..rows {
        let (x, y) = (a[i * cols + p], a[i * cols + q]);
        a[i * cols + p] = c * x - s * y;
        a[i * cols + q] = s * x + c * y;
    }
}

/// The columns of a `rows` x `k` row-major matrix with unit norm. A zero column
/// is None.
fn unit_columns(data: &[f64], rows: usize, k: usize) -> Vec<Option<Vec<f64>>> {
    (0..k)
        .map(|j| {
            let col: Vec<f64> = (0..rows).map(|i| data[i * k + j]).collect();
            let norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            (norm > 0.5).then_some(col)
        })
        .collect()
}

/// Fills each None column with a unit vector orthogonal to all the others. Each
/// vector starts from the first standard basis vector that is not in the span of
/// the columns already set; it is orthogonalized twice (Gram-Schmidt).
fn complete_columns(cols: &mut [Option<Vec<f64>>], rows: usize) -> Result<()> {
    for j in 0..cols.len() {
        if cols[j].is_some() {
            continue;
        }
        let mut found = None;
        for k in 0..rows {
            let mut v = vec![0.0; rows];
            v[k] = 1.0;
            for _ in 0..2 {
                for c in cols.iter().flatten() {
                    let dot: f64 = v.iter().zip(c).map(|(x, y)| x * y).sum();
                    for (x, y) in v.iter_mut().zip(c) {
                        *x -= dot * y;
                    }
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-8 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
                found = Some(v);
                break;
            }
        }
        cols[j] = Some(found.ok_or_else(|| {
            ArrayError::InvalidOperation("svd could not complete the basis".into())
        })?);
    }
    Ok(())
}

/// The `rows` x `cols.len()` row-major matrix with the given columns. A None
/// column is left as zeros.
fn columns_to_row_major(cols: &[Option<Vec<f64>>], rows: usize) -> Vec<f64> {
    let k = cols.len();
    let mut out = vec![0.0; rows * k];
    for (j, col) in cols.iter().enumerate() {
        if let Some(c) = col {
            for (i, &x) in c.iter().enumerate() {
                out[i * k + j] = x;
            }
        }
    }
    out
}

/// Transpose of the n x n row-major matrix `data`.
fn transpose_square(data: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            out[r * n + c] = data[c * n + r];
        }
    }
    out
}

/// Wilkinson shift for QR iteration.
fn wilkinson_shift(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let delta = (a - d) / 2.0;
    let bc = b * c;
    if delta.abs() < 1e-14 {
        d - bc.abs().sqrt()
    } else {
        d - bc / (delta + delta.signum() * (delta * delta + bc).abs().sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_identity() {
        let a = NdArray::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let qr = a.qr().unwrap();
        // Q*R should reconstruct A (signs of Q/R may differ)
        let reconstructed = qr.q.matmul(&qr.r).unwrap();
        for i in 0..4 {
            assert!((reconstructed.data[i] - a.data[i]).abs() < 1e-10);
        }
        // Q should be orthogonal: Q^T * Q = I
        let qt = NdArray::new(
            vec![qr.q.data[0], qr.q.data[2], qr.q.data[1], qr.q.data[3]],
            vec![2, 2],
        )
        .unwrap();
        let qtq = qt.matmul(&qr.q).unwrap();
        for i in 0..2 {
            assert!((qtq.data[i * 2 + i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_qr_reconstruction() {
        let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let qr = a.qr().unwrap();
        let reconstructed = qr.q.matmul(&qr.r).unwrap();
        for i in 0..6 {
            assert!(
                (reconstructed.data[i] - a.data[i]).abs() < 1e-10,
                "mismatch at {i}: {} vs {}",
                reconstructed.data[i],
                a.data[i]
            );
        }
    }

    #[test]
    fn test_cholesky() {
        // Symmetric positive definite: [[4, 2], [2, 3]]
        let a = NdArray::new(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let l = a.cholesky().unwrap();
        // L * L^T should equal A
        let lt =
            NdArray::new(vec![l.data[0], l.data[2], l.data[1], l.data[3]], vec![2, 2]).unwrap();
        let result = l.matmul(&lt).unwrap();
        for i in 0..4 {
            assert!((result.data[i] - a.data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        let a = NdArray::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).unwrap();
        assert!(a.cholesky().is_err());
    }

    #[test]
    fn test_eig_diagonal() {
        // Diagonal matrix: eigenvalues are diagonal elements
        let a = NdArray::new(vec![3.0, 0.0, 0.0, 5.0], vec![2, 2]).unwrap();
        let eig = a.eig().unwrap();
        let mut vals = eig.values.clone();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 3.0).abs() < 1e-6);
        assert!((vals[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_eig_symmetric() {
        // Symmetric: [[2, 1], [1, 2]], eigenvalues = 1, 3
        let a = NdArray::new(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let eig = a.eig().unwrap();
        let mut vals = eig.values.clone();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let svd = a.svd().unwrap();

        // Reconstruct: U * diag(S) * V^T
        let m = 2;
        let n = 2;
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..svd.s.len() {
                    val += svd.u.data[i * m + k] * svd.s[k] * svd.vt.data[k * n + j];
                }
                result[i * n + j] = val;
            }
        }

        for (i, r) in result.iter().enumerate().take(4) {
            assert!(
                (*r - a.data[i]).abs() < 1e-6,
                "SVD reconstruction mismatch at {i}: {} vs {}",
                *r,
                a.data[i]
            );
        }
    }

    #[test]
    fn test_lstsq_exact() {
        // Exact system: 2x2
        let a = NdArray::new(vec![1.0, 1.0, 1.0, -1.0], vec![2, 2]).unwrap();
        let b = NdArray::new(vec![3.0, 1.0], vec![2]).unwrap();
        let result = a.lstsq(&b).unwrap();
        assert!((result.x.data[0] - 2.0).abs() < 1e-6);
        assert!((result.x.data[1] - 1.0).abs() < 1e-6);
        assert!(result.residuals < 1e-10);
    }

    #[test]
    fn test_lstsq_overdetermined() {
        // Overdetermined: 3 equations, 2 unknowns
        // y ≈ x: points (1,1), (2,2), (3,3.1)
        let a = NdArray::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![3, 2]).unwrap();
        let b = NdArray::new(vec![1.0, 2.0, 3.1], vec![3]).unwrap();
        let result = a.lstsq(&b).unwrap();
        // Solution should be close to the mean
        assert!(result.rank == 1 || result.rank == 2);
    }
}
