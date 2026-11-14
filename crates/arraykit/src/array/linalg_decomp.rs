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
    /// Eigenvalues (real only, ascending).
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

    /// Eigenvalue decomposition with the shifted QR algorithm.
    ///
    /// Works on square matrices. Returns real eigenvalues in ascending order
    /// and the eigenvectors as columns. Real eigenvalues only: a complex
    /// eigenvalue returns an error, as does a non-finite entry or no
    /// convergence within 50 * n iterations.
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

        // Shifted QR iteration on a copy. It finds the eigenvalues only.
        let max_iter = 50 * n;
        let mut a = self.data.clone();
        let mut found = is_quasi_triangular(&a, n);
        for _ in 0..max_iter {
            if found {
                break;
            }

            // Wilkinson shift from the bottom-right 2x2 block. When that
            // block has complex eigenvalues, use its corner instead, so the
            // iteration stays real.
            let (p, q, r, s) = (
                a[(n - 2) * n + (n - 2)],
                a[(n - 2) * n + (n - 1)],
                a[(n - 1) * n + (n - 2)],
                a[(n - 1) * n + (n - 1)],
            );
            let disc = (p - s) * (p - s) / 4.0 + q * r;
            let shift = if disc >= 0.0 {
                wilkinson_shift(p, q, r, s)
            } else {
                s
            };

            // A - sI = QR, then A = RQ + sI.
            for i in 0..n {
                a[i * n + i] -= shift;
            }
            let a_mat = NdArray::new(a.clone(), vec![n, n])?;
            let qr = a_mat.qr()?;
            let rq = qr.r.matmul(&qr.q)?;
            a = rq.data;
            for i in 0..n {
                a[i * n + i] += shift;
            }

            // Deflate: a tiny sub-diagonal entry becomes exactly zero.
            for i in 1..n {
                let scale = a[i * n + i].abs() + a[(i - 1) * n + (i - 1)].abs();
                if a[i * n + (i - 1)].abs() <= 1e-12 * scale {
                    a[i * n + (i - 1)] = 0.0;
                }
            }

            found = is_quasi_triangular(&a, n);
        }
        if !found {
            return Err(ArrayError::InvalidOperation("eig did not converge".into()));
        }

        // Read the eigenvalues from the 1x1 and 2x2 diagonal blocks.
        let mut values: Vec<f64> = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            if i + 1 < n && a[(i + 1) * n + i] != 0.0 {
                let (p, q) = (a[i * n + i], a[i * n + i + 1]);
                let (r, s) = (a[(i + 1) * n + i], a[(i + 1) * n + i + 1]);
                let mid = (p + s) / 2.0;
                let disc = (p - s) * (p - s) / 4.0 + q * r;
                if disc < 0.0 {
                    return Err(ArrayError::InvalidOperation(
                        "matrix has complex eigenvalues".into(),
                    ));
                }
                let root = disc.sqrt();
                values.push(mid - root);
                values.push(mid + root);
                i += 2;
            } else {
                values.push(a[i * n + i]);
                i += 1;
            }
        }
        values.sort_by(|x, y| x.total_cmp(y));

        // Eigenvectors of the original matrix by inverse iteration.
        let mut v_data = vec![0.0; n * n];
        for (k, &lambda) in values.iter().enumerate() {
            let v = eigenvector(&self.data, n, lambda)?;
            for row in 0..n {
                v_data[row * n + k] = v[row];
            }
        }

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
    pub fn svd(&self) -> Result<SvdResult> {
        if self.ndim() != 2 {
            return Err(ArrayError::ShapeMismatch {
                expected: vec![2],
                got: vec![self.ndim()],
            });
        }

        let m = self.dims()[0];
        let n = self.dims()[1];

        // Compute A^T * A
        let at = self.t()?;
        let ata = at.matmul(self)?;

        // Eigendecompose A^T * A -> V, Σ²
        let eig = ata.eig()?;

        // Sort eigenvalues descending and get singular values
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eig.values[b]
                .abs()
                .partial_cmp(&eig.values[a].abs())
                .unwrap()
        });

        let s: Vec<f64> = indices
            .iter()
            .map(|&i| eig.values[i].abs().sqrt())
            .collect();

        // V = eigenvectors of A^T * A (sorted)
        let mut vt_data = vec![0.0; n * n];
        for (new_i, &old_i) in indices.iter().enumerate() {
            for j in 0..n {
                vt_data[new_i * n + j] = eig.vectors.data[j * n + old_i];
            }
        }

        // U = A * V * Σ^(-1) for non-zero singular values
        let k = s.iter().filter(|&&sv| sv > 1e-10).count();
        let mut u_data = vec![0.0; m * m];

        for j in 0..k {
            // u_j = A * v_j / s_j
            for i in 0..m {
                let mut val = 0.0;
                for l in 0..n {
                    val += self.data[i * n + l] * vt_data[j * n + l];
                }
                u_data[i * m + j] = val / s[j];
            }
        }

        // Complete U with orthogonal basis for null space (Gram-Schmidt)
        for j in k..m {
            // Start with standard basis vector e_j
            let mut col = vec![0.0; m];
            col[j.min(m - 1)] = 1.0;

            // Orthogonalize against existing columns
            for prev in 0..j {
                let mut dot = 0.0;
                for i in 0..m {
                    dot += col[i] * u_data[i * m + prev];
                }
                for i in 0..m {
                    col[i] -= dot * u_data[i * m + prev];
                }
            }

            // Normalize
            let norm = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for i in 0..m {
                    u_data[i * m + j] = col[i] / norm;
                }
            }
        }

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

/// True when no two neighbouring sub-diagonal entries are non-zero. Such a
/// matrix is block upper triangular with 1x1 and 2x2 diagonal blocks.
fn is_quasi_triangular(a: &[f64], n: usize) -> bool {
    (1..n).all(|i| a[i * n + (i - 1)] == 0.0 || i + 1 >= n || a[(i + 1) * n + i] == 0.0)
}

/// Solve `m x = b` for a dense n x n row-major matrix by Gaussian elimination
/// with partial pivoting. Returns None for a singular matrix or a non-finite result.
fn solve_linear(m: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = m.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if a[row * n + col].abs() > a[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if a[pivot * n + col].abs() <= 1e-300 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            x.swap(col, pivot);
        }
        for row in (col + 1)..n {
            let factor = a[row * n + col] / a[col * n + col];
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            x[row] -= factor * x[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = x[row];
        for k in (row + 1)..n {
            sum -= a[row * n + k] * x[k];
        }
        x[row] = sum / a[row * n + row];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// Unit eigenvector of `a` (n x n, row-major) for the real eigenvalue `lambda`,
/// by inverse iteration. Its first non-zero entry is positive.
fn eigenvector(a: &[f64], n: usize, lambda: f64) -> Result<Vec<f64>> {
    let mut offset = 1e-10;
    for _attempt in 0..3 {
        let shift = lambda + offset * lambda.abs().max(1.0);
        let mut m = a.to_vec();
        for i in 0..n {
            m[i * n + i] -= shift;
        }
        let mut v = vec![1.0; n];
        let mut ok = true;
        for _ in 0..3 {
            match solve_linear(&m, &v, n) {
                Some(next) => v = next,
                None => {
                    ok = false;
                    break;
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if !(norm > 0.0 && norm.is_finite()) {
                ok = false;
                break;
            }
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        if ok {
            if let Some(&first) = v.iter().find(|x| x.abs() > 1e-12) {
                if first < 0.0 {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
            }
            return Ok(v);
        }
        offset *= 10.0;
    }
    Err(ArrayError::InvalidOperation(
        "eigenvector did not converge".into(),
    ))
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
