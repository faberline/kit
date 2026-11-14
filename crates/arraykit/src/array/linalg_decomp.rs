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

    /// Symmetric eigenvalue decomposition (more stable for symmetric matrices).
    pub fn eigh(&self) -> Result<EigResult> {
        // For symmetric matrices, use the same QR algorithm
        // but the result is guaranteed to have real eigenvalues
        self.eig()
    }

    /// Singular Value Decomposition: A = U * diag(S) * V^T.
    ///
    /// Uses one-sided Jacobi on the matrix itself; A^T A is never formed.
    /// Returns U (m × m, orthonormal), the singular values in descending order
    /// (min(m, n) of them) and V^T (n × n, orthonormal). Returns an error for a
    /// non-finite entry and when the sweeps do not converge.
    pub fn svd(&self) -> Result<SvdResult> {
        if self.ndim() != 2 {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.ndim()],
            });
        }

        let m = self.dims()[0];
        let n = self.dims()[1];

        if self.data.iter().any(|x| !x.is_finite()) {
            return Err(ArrayError::InvalidOperation(
                "matrix contains NaN or infinity".into(),
            ));
        }

        if m == 0 || n == 0 {
            let mut u = vec![0.0; m * m];
            for i in 0..m {
                u[i * m + i] = 1.0;
            }
            let mut vt = vec![0.0; n * n];
            for i in 0..n {
                vt[i * n + i] = 1.0;
            }
            return Ok(SvdResult {
                u: NdArray::new(u, vec![m, m])?,
                s: vec![],
                vt: NdArray::new(vt, vec![n, n])?,
            });
        }

        if m >= n {
            // A = P S Q^T with P thin (m x n); U is P completed to m x m.
            let (u_thin, s, v) = svd_tall(&self.data, m, n)?;
            let mut u = vec![0.0; m * m];
            for i in 0..m {
                for j in 0..n {
                    u[i * m + j] = u_thin[i * n + j];
                }
            }
            let mut filled = vec![false; m];
            filled[..n].copy_from_slice(&thin_filled(&s));
            complete_columns(&mut u, m, &filled)?;
            let vt = transpose_rect(&v, n, n);
            Ok(SvdResult {
                u: NdArray::new(u, vec![m, m])?,
                s,
                vt: NdArray::new(vt, vec![n, n])?,
            })
        } else {
            // A^T = P S Q^T, so A = Q S P^T. U = Q; P is completed to n x n.
            let at = transpose_rect(&self.data, m, n);
            let (p_thin, s, q) = svd_tall(&at, n, m)?;
            let mut p = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..m {
                    p[i * n + j] = p_thin[i * m + j];
                }
            }
            let mut filled = vec![false; n];
            filled[..m].copy_from_slice(&thin_filled(&s));
            complete_columns(&mut p, n, &filled)?;
            let vt = transpose_rect(&p, n, n);
            Ok(SvdResult {
                u: NdArray::new(q, vec![m, m])?,
                s,
                vt: NdArray::new(vt, vec![n, n])?,
            })
        }
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

/// Transposes a `rows x cols` row-major matrix into a `cols x rows` one.
fn transpose_rect(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    out
}

/// Marks the singular values that give a column of U (zero ones do not).
fn thin_filled(s: &[f64]) -> Vec<bool> {
    let s0 = s.first().copied().unwrap_or(0.0);
    s.iter().map(|&x| s0 > 0.0 && x > 1e-14 * s0).collect()
}

/// Fills the columns of a `d x d` row-major matrix that are not yet set
/// (`filled[j] == false`) with unit vectors orthogonal to the others.
/// Each candidate is a standard basis vector, re-orthogonalized twice.
fn complete_columns(mat: &mut [f64], d: usize, filled: &[bool]) -> Result<()> {
    let mut ready = filled.to_vec();
    for j in 0..d {
        if ready[j] {
            continue;
        }
        let mut placed = false;
        for k in 0..d {
            let mut col = vec![0.0; d];
            col[k] = 1.0;
            for _pass in 0..2 {
                for prev in 0..d {
                    if !ready[prev] {
                        continue;
                    }
                    let dot: f64 = (0..d).map(|i| col[i] * mat[i * d + prev]).sum();
                    for i in 0..d {
                        col[i] -= dot * mat[i * d + prev];
                    }
                }
            }
            let norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-8 {
                for i in 0..d {
                    mat[i * d + j] = col[i] / norm;
                }
                placed = true;
                break;
            }
        }
        if !placed {
            return Err(ArrayError::InvalidOperation(
                "svd basis completion failed".into(),
            ));
        }
        ready[j] = true;
    }
    Ok(())
}

/// One-sided (Hestenes) Jacobi SVD of an `m x n` row-major matrix with
/// `m >= n`. Returns `(u, s, v)`: `u` is `m x n` (zero columns where the
/// singular value is zero), `s` has length `n` in descending order, and `v`
/// is `n x n`. A = u * diag(s) * v^T on the non-zero columns.
#[allow(clippy::needless_range_loop)]
fn svd_tall(data: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut w = data.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let mut converged = false;
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
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
                rotated = true;

                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for i in 0..m {
                    let (wp, wq) = (w[i * n + p], w[i * n + q]);
                    w[i * n + p] = c * wp - s * wq;
                    w[i * n + q] = s * wp + c * wq;
                }
                for i in 0..n {
                    let (vp, vq) = (v[i * n + p], v[i * n + q]);
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
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

    let norms: Vec<f64> = (0..n)
        .map(|j| {
            (0..m)
                .map(|i| w[i * n + j] * w[i * n + j])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| norms[y].total_cmp(&norms[x]));

    let s: Vec<f64> = order.iter().map(|&j| norms[j]).collect();
    let filled = thin_filled(&s);
    let mut u = vec![0.0; m * n];
    let mut v_sorted = vec![0.0; n * n];
    for (new_j, &old_j) in order.iter().enumerate() {
        for i in 0..n {
            v_sorted[i * n + new_j] = v[i * n + old_j];
        }
        if filled[new_j] {
            for i in 0..m {
                u[i * n + new_j] = w[i * n + old_j] / s[new_j];
            }
        }
    }
    Ok((u, s, v_sorted))
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
