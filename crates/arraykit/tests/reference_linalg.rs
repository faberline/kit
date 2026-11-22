//! Reference-value tests for `arraykit::array/linalg` and `arraykit::array/linalg_decomp`.
//!
//! Expected values come from NumPy 2.4.6. The expression of each value is in
//! the `source` column of `fixtures/linalg_reference.tsv`, which
//! `tools/reference/arraykit/generate.py` writes. No Python runs in CI: the
//! fixture is the only source of the expected values.
//!
//! The inputs are the same integer formulas as in the generator, built with
//! the same loops, so the Rust side holds the same numbers.

use arraykit::array::NdArray;

const FIXTURE: &str = include_str!("fixtures/linalg_reference.tsv");

/// Relative tolerance of every check in this file, except the ill-conditioned cases.
const TOL: f64 = 1e-9;

/// Relative tolerance of the cases with condition number about 1e10. A solve or
/// an inverse of such a matrix loses about 10 digits, so 1e-9 is too strict.
const COND_TOL: f64 = 1e-5;

/// Size of the well-conditioned matrices.
const N: usize = 50;

/// The value column of `key` in the fixture, as text.
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

/// The expected number stored under `key`. `nan` parses to NaN.
fn reference(key: &str) -> f64 {
    let value = raw(key);
    value
        .parse()
        .unwrap_or_else(|_| panic!("fixture key {key}: bad number {value}"))
}

/// The expected list stored under `key`: comma separated numbers.
fn reference_list(key: &str) -> Vec<f64> {
    raw(key)
        .split(',')
        .map(|item| {
            item.parse()
                .unwrap_or_else(|_| panic!("fixture key {key}: bad number {item}"))
        })
        .collect()
}

/// Check one number with a relative tolerance. A zero expected value needs
/// an absolute error of at most 1e-12. A NaN expected value needs a NaN result.
fn check(actual: f64, expected: f64, rel_tol: f64, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}: got {actual}, expected NaN");
    } else if expected == 0.0 {
        assert!(actual.abs() <= 1e-12, "{label}: got {actual}, expected 0");
    } else {
        assert!(
            (actual - expected).abs() <= rel_tol * expected.abs(),
            "{label}: got {actual}, expected {expected}"
        );
    }
}

/// Check `actual` against the fixture value `key`.
fn assert_close(actual: f64, key: &str, rel_tol: f64) {
    check(actual, reference(key), rel_tol, key);
}

/// Check each element of `actual` against the fixture list `key`.
fn assert_close_list(actual: &[f64], key: &str, rel_tol: f64) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "{key}: length");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        check(*a, *e, rel_tol, &format!("{key}[{i}]"));
    }
}

/// Absolute-or-relative check with a 1e-9 scale, for the extra structural checks.
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{a} != {b}");
}

/// The 50 x 50 well-conditioned matrix A (row-major).
fn well_matrix() -> Vec<f64> {
    let mut a = vec![0.0f64; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut x = ((i * 31 + j * 17 + i * j) % 23) as f64 / 7.0;
            if i == j {
                x += 50.0;
            }
            a[i * N + j] = x;
        }
    }
    a
}

/// `a` times `b` for square row-major matrices, summed in index order.
fn plain_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0;
            for t in 0..N {
                acc += a[i * N + t] * b[t * N + j];
            }
            out[i * N + j] = acc;
        }
    }
    out
}

/// The transpose of a square row-major matrix.
fn transpose(a: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; N * N];
    for i in 0..N {
        for j in 0..N {
            out[i * N + j] = a[j * N + i];
        }
    }
    out
}

/// S = A + A^T: symmetric.
fn symmetric_matrix() -> Vec<f64> {
    let a = well_matrix();
    let at = transpose(&a);
    a.iter().zip(&at).map(|(x, y)| x + y).collect()
}

/// P = A A^T: symmetric positive definite.
fn spd_matrix() -> Vec<f64> {
    let a = well_matrix();
    plain_product(&a, &transpose(&a))
}

/// M: 50 x 50 matrix with condition number about 1e10, built as H diag(d) H.
fn cond_matrix() -> Vec<f64> {
    let k = (1e10 - 1.0) / 49.0;
    let d: Vec<f64> = (0..N).map(|i| 1.0 / (1.0 + i as f64 * k)).collect();
    let v: Vec<f64> = (0..N).map(|i| (i + 1) as f64).collect();
    let mut vtv = 0.0;
    for item in &v {
        vtv += item * item;
    }
    let mut h = vec![0.0f64; N * N];
    for i in 0..N {
        for j in 0..N {
            let delta = if i == j { 1.0 } else { 0.0 };
            h[i * N + j] = delta - 2.0 * v[i] * v[j] / vtv;
        }
    }
    let mut out = vec![0.0f64; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0;
            for t in 0..N {
                acc += (h[i * N + t] * d[t]) * h[t * N + j];
            }
            out[i * N + j] = acc;
        }
    }
    out
}

/// b: right-hand side of length 50.
fn rhs_vector() -> Vec<f64> {
    (0..N).map(|i| (i % 9) as f64 - 4.0).collect()
}

/// L: 50 x 20, overdetermined and full rank.
fn tall_matrix() -> Vec<f64> {
    let mut l = vec![0.0f64; N * 20];
    for i in 0..N {
        for j in 0..20 {
            let mut x = ((i * 13 + j * 29 + i * j) % 19) as f64 / 5.0;
            if i == j {
                x += 5.0;
            }
            l[i * 20 + j] = x;
        }
    }
    l
}

fn square(data: Vec<f64>) -> NdArray<f64> {
    NdArray::new(data, vec![N, N]).unwrap()
}

// numpy: np.linalg.solve(a, b) for the well-conditioned A and the right-hand side b
#[test]
fn solve_matches_numpy() {
    let a = square(well_matrix());
    let b = NdArray::new(rhs_vector(), vec![N]).unwrap();
    let x = a.solve(&b).unwrap();
    let data = x.data();
    assert_close_list(&data[..5], "linalg.solve.well.head", TOL);
    assert_close_list(&data[N - 5..], "linalg.solve.well.tail", TOL);
}

// numpy: np.linalg.solve(m, b) for the cond 1e10 matrix M (tolerance: see COND_TOL)
#[test]
fn solve_cond_matches_numpy() {
    let m = square(cond_matrix());
    let b = NdArray::new(rhs_vector(), vec![N]).unwrap();
    let x = m.solve(&b).unwrap();
    let data = x.data();
    assert_close_list(&data[..5], "linalg.solve.cond.head", COND_TOL);
    assert_close_list(&data[N - 5..], "linalg.solve.cond.tail", COND_TOL);
}

// numpy: np.trace(np.linalg.inv(a)) and np.linalg.inv(a)[0, :5]
#[test]
fn inv_matches_numpy() {
    let inv = square(well_matrix()).inv().unwrap();
    assert_close(inv.trace().unwrap(), "linalg.inv.well.trace", TOL);
    assert_close_list(&inv.data()[..5], "linalg.inv.well.row0_head", TOL);
}

// numpy: np.trace(np.linalg.inv(m)) for the cond 1e10 matrix (tolerance: see COND_TOL)
#[test]
fn inv_cond_matches_numpy() {
    let inv = square(cond_matrix()).inv().unwrap();
    assert_close(inv.trace().unwrap(), "linalg.inv.cond.trace", COND_TOL);
}

// numpy: np.linalg.slogdet(a)[1] (the 50 x 50 determinant is about 1e85, so the log is compared)
// numpy: np.linalg.det([[1, 2, 3], [4, 5, 6], [7, 8, 10]]) = -3
#[test]
fn det_matches_numpy() {
    let det = square(well_matrix()).det().unwrap();
    assert_close(det.ln(), "linalg.det.well.logdet", TOL);

    let t3 = NdArray::new(
        vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        vec![3, 3],
    )
    .unwrap();
    assert_close(t3.det().unwrap(), "linalg.det.t3", TOL);
}

// numpy: np.linalg.eigh(s)[0] for the symmetric S = A + A^T (ascending)
#[test]
fn eigh_matches_numpy() {
    let eig = square(symmetric_matrix()).eigh().unwrap();
    assert_close_list(&eig.values, "linalg.eigh.well.values", TOL);
}

// numpy: np.sort(np.linalg.eig(t)[0].real) for the nonsymmetric triangular t
#[test]
fn eig_matches_numpy() {
    let t = NdArray::new(vec![1.0f64, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
    let mut values = t.eig().unwrap().values;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_close_list(&values, "linalg.eig.triangular", TOL);
}

// numpy: np.linalg.svd(a, compute_uv=False) (all singular values, descending)
// numpy: np.linalg.svd(m, compute_uv=False) for the cond 1e10 matrix (tolerance: see COND_TOL)
#[test]
fn svd_matches_numpy() {
    let s = square(well_matrix()).svd().unwrap().s;
    assert_close_list(&s, "linalg.svd.well.s", TOL);

    let s_cond = square(cond_matrix()).svd().unwrap().s;
    assert_close_list(&s_cond, "linalg.svd.cond.s", COND_TOL);
}

// numpy: np.abs(np.diag(np.linalg.qr(a)[1])); the signs of R differ between libraries
#[test]
fn qr_matches_numpy() {
    let r = square(well_matrix()).qr().unwrap().r;
    let diag: Vec<f64> = (0..N).map(|i| r.data()[i * N + i].abs()).collect();
    assert_close_list(&diag, "linalg.qr.well.rdiag_abs", TOL);
}

// numpy: np.diag(np.linalg.cholesky(p)) for the symmetric positive definite P = A A^T
#[test]
fn cholesky_matches_numpy() {
    let l = square(spd_matrix()).cholesky().unwrap();
    let diag: Vec<f64> = (0..N).map(|i| l.data()[i * N + i]).collect();
    assert_close_list(&diag, "linalg.cholesky.well.diag", TOL);
}

// numpy: np.linalg.lstsq(l, b, rcond=None): the solution, the residual sum and the rank
#[test]
fn lstsq_matches_numpy() {
    let l = NdArray::new(tall_matrix(), vec![N, 20]).unwrap();
    let b = NdArray::new(rhs_vector(), vec![N]).unwrap();
    let result = l.lstsq(&b).unwrap();
    let data = result.x.data();
    assert_close_list(&data[..5], "linalg.lstsq.x.head", TOL);
    assert_close_list(&data[data.len() - 5..], "linalg.lstsq.x.tail", TOL);
    assert_close(result.residuals, "linalg.lstsq.residuals", TOL);
    assert_close(result.rank as f64, "linalg.lstsq.rank", TOL);
}

// numpy: np.matmul(p23, q32).ravel() with p23 = [[1, 2, 3], [4, 5, 6]]
//        and q32 = [[7, 8], [9, 10], [11, 12]]
#[test]
fn matmul_matches_numpy() {
    let p23 = NdArray::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let q32 = NdArray::new(vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
    let out = p23.matmul(&q32).unwrap();
    assert_close_list(out.data(), "linalg.matmul.p23_q32", TOL);
}

// numpy: np.dot(x, y), np.dot(p23, q32).ravel()
#[test]
fn dot_matches_numpy() {
    let x = NdArray::new(vec![1.5f64, -2.0, 3.0, 0.5], vec![4]).unwrap();
    let y = NdArray::new(vec![2.0f64, 4.0, -1.0, 8.0], vec![4]).unwrap();
    assert_close(x.dot(&y).unwrap().data()[0], "linalg.dot.xy", TOL);

    let p23 = NdArray::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let q32 = NdArray::new(vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
    let out = p23.dot(&q32).unwrap();
    assert_close_list(out.data(), "linalg.dot.p23_q32", TOL);
}

// numpy: np.inner(x, y) for the vectors x and y of the test above
#[test]
fn inner_matches_numpy() {
    let x = NdArray::new(vec![1.5f64, -2.0, 3.0, 0.5], vec![4]).unwrap();
    let y = NdArray::new(vec![2.0f64, 4.0, -1.0, 8.0], vec![4]).unwrap();
    assert_close(x.inner(&y).unwrap().data()[0], "linalg.inner.xy", TOL);
}

// numpy: np.outer(x, y).ravel() for the vectors x and y of the test above
#[test]
fn outer_matches_numpy() {
    let x = NdArray::new(vec![1.5f64, -2.0, 3.0, 0.5], vec![4]).unwrap();
    let y = NdArray::new(vec![2.0f64, 4.0, -1.0, 8.0], vec![4]).unwrap();
    let out = x.outer(&y).unwrap();
    assert_close_list(out.data(), "linalg.outer.xy", TOL);
}

// numpy: np.trace(t3) for t3 = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
#[test]
fn trace_matches_numpy() {
    let t3 = NdArray::new(
        vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        vec![3, 3],
    )
    .unwrap();
    assert_close(t3.trace().unwrap(), "linalg.trace.t3", TOL);
}

// numpy: np.linalg.norm(x) (ord 2, 1, inf and 3 for the vector) and the Frobenius norm of p23.
// The 2-D orders 1 and inf are not tested: NumPy gives matrix norms there, arraykit flattens.
#[test]
fn norm_matches_numpy() {
    let x = NdArray::new(vec![1.5f64, -2.0, 3.0, 0.5], vec![4]).unwrap();
    assert_close(x.norm(None), "linalg.norm.x.none", TOL);
    assert_close(x.norm(Some(2.0)), "linalg.norm.x.2", TOL);
    assert_close(x.norm(Some(1.0)), "linalg.norm.x.1", TOL);
    assert_close(x.norm(Some(f64::INFINITY)), "linalg.norm.x.inf", TOL);
    assert_close(x.norm(Some(3.0)), "linalg.norm.x.3", TOL);

    let p23 = NdArray::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    assert_close(p23.norm(None), "linalg.norm.p23.none", TOL);
}

// numpy: np.linalg.solve([[3., 1.], [1., 2.]], [[9., 1.], [8., 3.]]) = [[2., -0.2], [3., 1.6]]
#[test]
fn solve_with_2d_right_hand_side() {
    let a = NdArray::new(vec![3.0f64, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![9.0f64, 1.0, 8.0, 3.0], vec![2, 2]).unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    assert_close_list(x.data(), "linalg.solve.small2d", TOL);
}

// numpy: np.linalg.inv([[1e-13]]) = [[1e13]]
// numpy: np.linalg.inv([[0.]]) raises LinAlgError (singular), so the call must be an error.
#[test]
fn inv_of_small_scale_matrix() {
    let small = NdArray::new(vec![1e-13f64], vec![1, 1]).unwrap();
    let inv = small.inv().unwrap();
    assert_close(inv.data()[0], "linalg.inv.small_scale", TOL);

    let zero = NdArray::new(vec![0.0f64], vec![1, 1]).unwrap();
    assert!(zero.inv().is_err());
}

// numpy: np.linalg.det([[1e-7, 0.], [0., 1e-7]]) = 1e-14
#[test]
fn det_of_small_scale_matrix() {
    let a = NdArray::new(vec![1e-7f64, 0.0, 0.0, 1e-7], vec![2, 2]).unwrap();
    assert_close(a.det().unwrap(), "linalg.det.small_scale", TOL);
}

// numpy: np.linalg.eig([[1., 1.], [0., 2.]]) gives eigenvalues [1., 2.]
#[test]
fn eig_nonsymmetric_triangular() {
    let a_data = [1.0f64, 1.0, 0.0, 2.0];
    let a = NdArray::new(a_data.to_vec(), vec![2, 2]).unwrap();
    let eig = a.eig().unwrap();

    // Compare a sorted copy with the fixture; keep `eig` in its own order,
    // because column j of `vectors` belongs to values[j].
    let mut sorted = eig.values.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_close_list(&sorted, "linalg.eig.triangular", TOL);

    // Column j of `vectors` is the eigenvector for values[j]: check A v = λ v.
    let v = eig.vectors.data();
    for j in 0..2 {
        let lambda = eig.values[j];
        for i in 0..2 {
            let av_i = a_data[i * 2] * v[j] + a_data[i * 2 + 1] * v[2 + j];
            close(av_i, lambda * v[i * 2 + j]);
        }
    }
}

// numpy: np.linalg.eigh([[2., 1.], [1., 2.]]) gives eigenvalues [1., 3.]
#[test]
fn eigh_symmetric() {
    let a = NdArray::new(vec![2.0f64, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let eig = a.eigh().unwrap();
    assert_close_list(&eig.values, "linalg.eigh.small", TOL);
}

// numpy: np.linalg.svd([[0., 0.], [1., 0.]], compute_uv=False) gives s = [1., 0.]
#[test]
fn svd_rank_deficient_has_no_zero_u_column() {
    let a_data = [0.0f64, 0.0, 1.0, 0.0];
    let a = NdArray::new(a_data.to_vec(), vec![2, 2]).unwrap();
    let svd = a.svd().unwrap();

    assert_close_list(&svd.s, "linalg.svd.rank_deficient", TOL);

    // Each column of U has norm 1.
    let u = svd.u.data();
    for j in 0..2 {
        let norm = (u[j] * u[j] + u[2 + j] * u[2 + j]).sqrt();
        close(norm, 1.0);
    }

    // U diag(S) Vt rebuilds the input.
    let vt = svd.vt.data();
    for i in 0..2 {
        for k in 0..2 {
            let mut entry = 0.0;
            for j in 0..2 {
                entry += u[i * 2 + j] * svd.s[j] * vt[j * 2 + k];
            }
            close(entry, a_data[i * 2 + k]);
        }
    }
}
