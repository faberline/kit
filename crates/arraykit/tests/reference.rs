//! Reference-value tests. Expected values come from NumPy 1.26; the expression
//! that made each value is in the comment above it. CI does not run Python.

use arraykit::array::{ArrayError, KitRng, NdArray};

/// Absolute-or-relative check with a 1e-9 scale.
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{a} != {b}");
}

/// Strict relative check, for values that are very small.
fn rel_close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-9 * b.abs(), "{a} != {b}");
}

fn close_all(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "length {} != {}", a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        close(*x, *y);
    }
}

// numpy 1.26: np.arange(0, 5, 0) raises ZeroDivisionError, so the call must be an error.
// numpy 1.26: len(np.arange(0.0, 1.0, 0.1)) = 10
// numpy 1.26: np.arange(0.0, 1.0, 0.1)[3] = 0 + 3 * 0.1 = 0.30000000000000004
#[test]
fn arange_zero_step_is_error_and_float_length() {
    assert!(NdArray::arange(0i32, 5, 0).is_err());

    let a = NdArray::arange(0.0f64, 1.0, 0.1).unwrap();
    assert_eq!(a.data().len(), 10);
    close(a.data()[3], 0.30000000000000004);
}

// numpy 1.26: np.linalg.solve([[3., 1.], [1., 2.]], [[9., 1.], [8., 3.]])
//             = [[2., -0.2], [3., 1.6]]
#[test]
fn solve_with_2d_right_hand_side() {
    let a = NdArray::new(vec![3.0f64, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![9.0f64, 1.0, 8.0, 3.0], vec![2, 2]).unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    close_all(x.data(), &[2.0, -0.2, 3.0, 1.6]);
}

// numpy 1.26: np.linalg.inv([[1e-13]]) = [[1e13]]
// numpy 1.26: np.linalg.inv([[0.]]) raises LinAlgError (singular), so the call must be an error.
#[test]
fn inv_of_small_scale_matrix() {
    let small = NdArray::new(vec![1e-13f64], vec![1, 1]).unwrap();
    let inv = small.inv().unwrap();
    rel_close(inv.data()[0], 1e13);

    let zero = NdArray::new(vec![0.0f64], vec![1, 1]).unwrap();
    assert!(zero.inv().is_err());
}

// numpy 1.26: np.linalg.det([[1e-7, 0.], [0., 1e-7]]) = 1e-14
#[test]
fn det_of_small_scale_matrix() {
    let a = NdArray::new(vec![1e-7f64, 0.0, 0.0, 1e-7], vec![2, 2]).unwrap();
    rel_close(a.det().unwrap(), 1e-14);
}

// numpy 1.26: np.linalg.eig([[0., -1.], [1., 0.]]) gives eigenvalues +1j and -1j.
// The kit returns real eigenvalues only, so eig must return an error, not real values.
#[test]
fn eig_rotation_matrix_is_an_error() {
    let a = NdArray::new(vec![0.0f64, -1.0, 1.0, 0.0], vec![2, 2]).unwrap();
    assert!(a.eig().is_err());
}

// numpy 1.26: np.linalg.eig([[1., 1.], [0., 2.]]) gives eigenvalues [1., 2.]
#[test]
fn eig_nonsymmetric_triangular() {
    let a_data = [1.0f64, 1.0, 0.0, 2.0];
    let a = NdArray::new(a_data.to_vec(), vec![2, 2]).unwrap();
    let eig = a.eig().unwrap();

    close_all(&eig.values, &[1.0, 2.0]);

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

// numpy 1.26: np.linalg.eigh([[2., 1.], [1., 2.]]) gives eigenvalues [1., 3.]
#[test]
fn eigh_symmetric() {
    let a = NdArray::new(vec![2.0f64, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let eig = a.eigh().unwrap();
    close_all(&eig.values, &[1.0, 3.0]);
}

// numpy 1.26: np.linalg.svd([[0., 0.], [1., 0.]]) gives s = [1., 0.]
#[test]
fn svd_rank_deficient_has_no_zero_u_column() {
    let a_data = [0.0f64, 0.0, 1.0, 0.0];
    let a = NdArray::new(a_data.to_vec(), vec![2, 2]).unwrap();
    let svd = a.svd().unwrap();

    close_all(&svd.s, &[1.0, 0.0]);

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

// numpy 1.26: np.linalg.svd([[nan, 1.], [1., 1.]]) raises LinAlgError, so the kit must
// return an error. It must not panic.
#[test]
fn svd_with_nan_is_an_error_not_a_panic() {
    let a = NdArray::new(vec![f64::NAN, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
    assert!(a.svd().is_err());
}

// These calls must return an error for bad arguments, not panic.
#[test]
fn no_panic_on_bad_arguments() {
    let a = NdArray::new(vec![1.0f64, 2.0], vec![2]).unwrap();
    let b = NdArray::new(vec![3.0f64, 4.0], vec![2]).unwrap();

    assert!(a.rolling(0).is_err());
    assert!(KitRng::new(1).choice(0).is_err());
    assert!(matches!(
        NdArray::stack(&[&a, &b], 5),
        Err(ArrayError::InvalidAxis { .. })
    ));
}
