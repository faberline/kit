//! Coverage tests for linalg.rs and linalg_decomp.rs of arraykit.

use arraykit::array::{ArrayError, NdArray};

fn f(data: Vec<f64>, dims: Vec<usize>) -> NdArray<f64> {
    NdArray::new(data, dims).unwrap()
}

fn i(data: Vec<i32>, dims: Vec<usize>) -> NdArray<i32> {
    NdArray::new(data, dims).unwrap()
}

fn shape(expected: &[usize], got: &[usize]) -> ArrayError {
    ArrayError::ShapeMismatch {
        expected: expected.to_vec(),
        got: got.to_vec(),
    }
}

fn invalid(text: &str) -> ArrayError {
    ArrayError::InvalidOperation(text.to_string())
}

// ---- linalg.rs ----

#[test]
fn matmul_rejects_non_2d() {
    let v = i(vec![1, 2, 3], vec![3]);
    let m = i(vec![1, 2, 3, 4], vec![2, 2]);
    assert_eq!(v.matmul(&m).err(), Some(shape(&[2], &[1, 2])));
}

#[test]
fn matmul_rejects_inner_mismatch() {
    let a = i(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let b = i(vec![1, 2, 3, 4], vec![2, 2]);
    assert_eq!(a.matmul(&b).err(), Some(shape(&[2, 3], &[2, 2])));
}

#[test]
fn dot_2d_2d_is_matmul() {
    let a = i(vec![1, 2, 3, 4], vec![2, 2]);
    let b = i(vec![5, 6, 7, 8], vec![2, 2]);
    assert_eq!(a.dot(&b).unwrap().data(), &[19, 22, 43, 50]);
}

#[test]
fn dot_1d_size_mismatch() {
    let a = i(vec![1, 2, 3], vec![3]);
    let b = i(vec![1, 2], vec![2]);
    assert_eq!(a.dot(&b).err(), Some(shape(&[3], &[2])));
}

#[test]
fn dot_matrix_vector_size_mismatch() {
    let a = i(vec![1, 2, 3, 4], vec![2, 2]);
    let b = i(vec![1, 2, 3], vec![3]);
    assert_eq!(a.dot(&b).err(), Some(shape(&[2], &[3])));
}

#[test]
fn dot_other_ndim_is_error() {
    let a = i(vec![1, 2], vec![2]);
    let b = i(vec![1, 2, 3, 4], vec![2, 2]);
    assert_eq!(a.dot(&b).err(), Some(shape(&[1, 2], &[1, 2])));
}

#[test]
fn outer_rejects_non_1d() {
    let a = i(vec![1, 2, 3, 4], vec![2, 2]);
    let b = i(vec![1, 2], vec![2]);
    assert_eq!(a.outer(&b).err(), Some(shape(&[1], &[2, 1])));
}

#[test]
fn inner_1d_is_dot() {
    let a = i(vec![1, 2, 3], vec![3]);
    let b = i(vec![4, 5, 6], vec![3]);
    assert_eq!(a.inner(&b).unwrap().data(), &[32]);
}

#[test]
fn inner_rejects_2d() {
    let a = i(vec![1, 2, 3, 4], vec![2, 2]);
    let b = i(vec![1, 2], vec![2]);
    assert_eq!(a.inner(&b).err(), Some(shape(&[1], &[2, 1])));
}

#[test]
fn trace_rejects_non_2d() {
    let a = i(vec![1, 2, 3], vec![3]);
    assert_eq!(a.trace().err(), Some(shape(&[2], &[1])));
}

#[test]
fn trace_of_rectangular_matrices() {
    // diagonal of 2 x 3 [[1,2,3],[4,5,6]] is 1, 5
    assert_eq!(i(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).trace().unwrap(), 6);
    // diagonal of 3 x 2 [[1,2],[3,4],[5,6]] is 1, 4
    assert_eq!(i(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).trace().unwrap(), 5);
}

#[test]
fn det_rejects_non_2d_and_non_square() {
    let v = f(vec![1.0, 2.0, 3.0], vec![3]);
    assert_eq!(v.det().err(), Some(shape(&[2], &[1])));
    let r = f(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(r.det().err(), Some(shape(&[2, 2], &[2, 3])));
}

#[test]
fn det_1x1_is_the_entry() {
    assert_eq!(f(vec![7.0], vec![1, 1]).det().unwrap(), 7.0);
}

#[test]
fn inv_rejects_non_2d_and_non_square() {
    let v = f(vec![1.0, 2.0, 3.0], vec![3]);
    assert_eq!(v.inv().err(), Some(shape(&[2], &[1])));
    let r = f(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(r.inv().err(), Some(shape(&[2, 2], &[2, 3])));
}

#[test]
fn solve_rejects_non_square_and_bad_rhs_length() {
    let r = f(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = f(vec![1.0, 2.0], vec![2]);
    assert_eq!(r.solve(&b).err(), Some(shape(&[2, 2], &[2, 3])));
    let sq = f(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let long = f(vec![1.0, 2.0, 3.0], vec![3]);
    assert_eq!(sq.solve(&long).err(), Some(shape(&[2], &[3])));
}

#[test]
fn norm_negative_infinity_and_general_order() {
    let a = f(vec![3.0, -4.0], vec![2]);
    assert_eq!(a.norm(Some(f64::NEG_INFINITY)), 3.0);
    // ord = 3: (27 + 64)^(1/3) = 91^(1/3) = 4.497941445275415
    assert!((a.norm(Some(3.0)) - 4.497941445275415).abs() < 1e-9);
    assert_eq!(a.norm(None), 5.0);
}

// ---- linalg_decomp.rs ----

#[test]
fn qr_rejects_non_2d() {
    let v = f(vec![1.0, 2.0], vec![2]);
    assert_eq!(v.qr().err(), Some(shape(&[2], &[1])));
}

#[test]
fn qr_zero_column_is_skipped() {
    // first column is zero: the Householder step is skipped, R keeps the zero column
    let a = f(vec![0.0, 1.0, 0.0, 2.0], vec![2, 2]);
    let qr = a.qr().unwrap();
    let back = qr.q.matmul(&qr.r).unwrap();
    for (x, y) in back.data().iter().zip(a.data()) {
        assert!((x - y).abs() < 1e-12);
    }
}

#[test]
fn cholesky_rejects_bad_shapes() {
    let v = f(vec![1.0, 2.0], vec![2]);
    assert_eq!(v.cholesky().err(), Some(shape(&[2], &[1])));
    let r = f(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(r.cholesky().err(), Some(shape(&[2, 2], &[2, 3])));
}

#[test]
fn eig_rejects_bad_shapes_and_non_finite() {
    let v = f(vec![1.0, 2.0], vec![2]);
    assert_eq!(v.eig().err(), Some(shape(&[2], &[1])));
    let r = f(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(r.eig().err(), Some(shape(&[2, 2], &[2, 3])));
    let nan = f(vec![1.0, f64::NAN, 0.0, 1.0], vec![2, 2]);
    assert_eq!(
        nan.eig().err(),
        Some(invalid("matrix contains NaN or infinity"))
    );
}

#[test]
fn eig_empty_and_1x1() {
    let e = f(vec![], vec![0, 0]).eig().unwrap();
    assert!(e.values.is_empty());
    assert_eq!(e.vectors.dims(), &[0, 0]);
    let one = f(vec![5.0], vec![1, 1]).eig().unwrap();
    assert_eq!(one.values, vec![5.0]);
    assert_eq!(one.vectors.data(), &[1.0]);
}

#[test]
fn eig_zero_matrix() {
    let e = f(vec![0.0; 9], vec![3, 3]).eig().unwrap();
    assert_eq!(e.values, vec![0.0, 0.0, 0.0]);
    assert_eq!(
        e.vectors.data(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn eig_complex_eigenvalues_is_error() {
    // rotation by 90 degrees: eigenvalues +i and -i
    let a = f(vec![0.0, -1.0, 1.0, 0.0], vec![2, 2]);
    assert_eq!(
        a.eig().err(),
        Some(invalid("matrix has complex eigenvalues"))
    );
}

#[test]
fn eigh_rejects_bad_shapes_non_finite_and_asymmetric() {
    let v = f(vec![1.0, 2.0], vec![2]);
    assert_eq!(v.eigh().err(), Some(shape(&[2], &[1])));
    let r = f(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(r.eigh().err(), Some(shape(&[2, 2], &[2, 3])));
    let inf = f(vec![1.0, 0.0, 0.0, f64::INFINITY], vec![2, 2]);
    assert_eq!(
        inf.eigh().err(),
        Some(invalid("matrix contains NaN or infinity"))
    );
    let asym = f(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(asym.eigh().err(), Some(invalid("matrix is not symmetric")));
}

#[test]
fn eigh_empty_and_1x1() {
    let e = f(vec![], vec![0, 0]).eigh().unwrap();
    assert!(e.values.is_empty());
    let one = f(vec![-3.0], vec![1, 1]).eigh().unwrap();
    assert_eq!(one.values, vec![-3.0]);
    assert_eq!(one.vectors.data(), &[1.0]);
}

#[test]
fn eigh_skips_zero_off_diagonal_entry() {
    // [[1,0,2],[0,3,0],[2,0,4]]: the (0,1) entry is zero, so that rotation is skipped
    // the block [[1,2],[2,4]] has trace 5 and determinant 0, so its eigenvalues are 0 and 5
    // eigenvalues: 0, 3, 5
    let a = f(
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0],
        vec![3, 3],
    );
    let e = a.eigh().unwrap();
    let want = [0.0, 3.0, 5.0];
    for (x, y) in e.values.iter().zip(want) {
        assert!((x - y).abs() < 1e-9);
    }
}

#[test]
fn svd_rejects_non_2d_and_non_finite() {
    let v = f(vec![1.0, 2.0], vec![2]);
    assert_eq!(v.svd().err(), Some(shape(&[2], &[1])));
    let nan = f(vec![1.0, f64::NAN, 0.0, 1.0], vec![2, 2]);
    assert_eq!(
        nan.svd().err(),
        Some(invalid("matrix contains NaN or infinity"))
    );
}

#[test]
fn svd_equal_norm_columns() {
    // columns (3,4) and (5,0) have the same norm 5 and are not orthogonal
    let a = f(vec![3.0, 5.0, 4.0, 0.0], vec![2, 2]);
    let s = a.svd().unwrap();
    // A^T A = [[25, 15], [15, 25]] has eigenvalues 40 and 10: singular values sqrt(40), sqrt(10)
    assert!((s.s[0] - 40.0_f64.sqrt()).abs() < 1e-9);
    assert!((s.s[1] - 10.0_f64.sqrt()).abs() < 1e-9);
}

#[test]
fn lstsq_rejects_bad_shapes() {
    let a = f(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let b2 = f(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(a.lstsq(&b2).err(), Some(shape(&[2, 1], &[2, 2])));
    let b3 = f(vec![1.0, 2.0, 3.0], vec![3]);
    assert_eq!(a.lstsq(&b3).err(), Some(shape(&[2], &[3])));
}
