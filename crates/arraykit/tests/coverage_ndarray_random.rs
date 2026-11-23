//! Coverage tests for the ndarray and random files of arraykit.

use arraykit::array::{ArrayError, AxisSlice, KitRng, NdArray, RandomExt, Shape, SliceInfo};
use std::fmt::{self, Write as FmtWrite};

/// A writer that refuses every write after `remaining` successful ones.
struct FailAfter {
    remaining: usize,
}

impl FmtWrite for FailAfter {
    fn write_str(&mut self, _text: &str) -> fmt::Result {
        if self.remaining == 0 {
            return Err(fmt::Error);
        }
        self.remaining -= 1;
        Ok(())
    }
}

/// The error of a result that must fail. Panics when the result is Ok.
fn expect_err<T>(result: Result<T, ArrayError>) -> ArrayError {
    match result {
        Err(error) => error,
        Ok(_) => panic!("expected an error, got Ok"),
    }
}

// ---- ndarray.rs ----

#[test]
fn new_rejects_data_length_mismatch() {
    let error = expect_err(NdArray::<i32>::new(vec![1, 2, 3], vec![2, 2]));
    assert_eq!(
        error,
        ArrayError::DataLengthMismatch {
            len: 3,
            shape: vec![2, 2],
            expected: 4,
        }
    );
}

#[test]
fn diag_rejects_three_dimensions() {
    let a = NdArray::<i32>::zeros(vec![2, 2, 2]);
    let error = expect_err(NdArray::diag(&a));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![1],
            got: vec![2, 2, 2],
        }
    );
}

#[test]
fn reshape_rejects_size_change() {
    let a = NdArray::<i32>::zeros(vec![2, 3]);
    let error = expect_err(a.reshape(vec![4, 2]));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![4, 2],
            got: vec![2, 3],
        }
    );
}

#[test]
fn slice_with_fewer_slices_than_axes_keeps_trailing_axes() {
    let a = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let sliced = a.slice(&SliceInfo::new(vec![AxisSlice::Index(1)])).unwrap();
    assert_eq!(sliced.dims(), &[3]);
    assert_eq!(sliced.data(), &[4, 5, 6]);
}

#[test]
fn transpose_axes_rejects_wrong_length() {
    let a = NdArray::<i32>::zeros(vec![2, 3]);
    let error = expect_err(a.transpose_axes(&[0]));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![0],
        }
    );
}

#[test]
fn transpose_axes_rejects_duplicate_axis() {
    let a = NdArray::<i32>::zeros(vec![2, 3]);
    let error = expect_err(a.transpose_axes(&[0, 0]));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![0, 1],
            got: vec![0, 0],
        }
    );
}

#[test]
fn transpose_axes_rejects_axis_out_of_range() {
    let a = NdArray::<i32>::zeros(vec![2, 3]);
    let error = expect_err(a.transpose_axes(&[0, 2]));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![0, 1],
            got: vec![0, 2],
        }
    );
}

#[test]
fn debug_prints_shape_and_dtype() {
    let a = NdArray::<i32>::zeros(vec![2, 3]);
    assert_eq!(format!("{:?}", a), "NdArray(shape=[2, 3], dtype=int32)");
}

#[test]
fn display_prints_one_dimension() {
    let a = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    assert_eq!(a.to_string(), "[1, 2, 3]");
}

#[test]
fn display_prints_two_dimensions() {
    let a = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    assert_eq!(a.to_string(), "[[1, 2],\n [3, 4]]");
}

#[test]
fn display_prints_three_dimensions() {
    let a = NdArray::new(vec![1, 2, 3, 4], vec![2, 1, 2]).unwrap();
    assert_eq!(a.to_string(), "[[[1, 2]],\n [[3, 4]]]");
}

#[test]
fn display_prints_empty_axis() {
    let a = NdArray::<i32>::zeros(vec![0]);
    assert_eq!(a.to_string(), "[]");
}

#[test]
fn get_out_of_bounds_is_error() {
    let a = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let error = expect_err(a.get(&[2, 0]));
    assert_eq!(
        error,
        ArrayError::IndexOutOfBounds {
            index: vec![2, 0],
            shape: vec![2, 3],
        }
    );
}

#[test]
fn set_out_of_bounds_is_error() {
    let mut a = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let error = expect_err(a.set(&[0, 3], 9));
    assert_eq!(
        error,
        ArrayError::IndexOutOfBounds {
            index: vec![0, 3],
            shape: vec![2, 3],
        }
    );
}

#[test]
fn slice_out_of_range_index_is_error() {
    let a = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let error = expect_err(a.slice(&SliceInfo::new(vec![AxisSlice::Index(2)])));
    assert_eq!(
        error,
        ArrayError::InvalidSlice {
            start: 2,
            stop: 2,
            step: 1,
            len: 2,
        }
    );
}

#[test]
fn broadcast_to_fewer_dimensions_is_error() {
    let a = NdArray::<i32>::zeros(vec![2, 3]);
    let error = expect_err(a.broadcast_to(&Shape::new(vec![3])));
    assert_eq!(
        error,
        ArrayError::BroadcastError {
            left: vec![2, 3],
            right: vec![3],
        }
    );
}

#[test]
fn display_stops_when_the_writer_fails() {
    let a = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    // Refuse the first n writes, for n = 0, 1, 2, ..., until the output fits.
    // Every refused write must return an error, which covers each `?` path.
    let mut allowed = 0;
    while write!(FailAfter { remaining: allowed }, "{}", a).is_err() {
        allowed += 1;
    }
    assert!(allowed > 0, "a writer that refuses every write must fail");
}

// ---- random.rs (public API) ----

#[test]
fn with_stream_is_deterministic_and_differs_by_stream() {
    let mut first = KitRng::with_stream(7, 3);
    let mut second = KitRng::with_stream(7, 3);
    let a: Vec<u64> = (0..5).map(|_| first.next_u64()).collect();
    let b: Vec<u64> = (0..5).map(|_| second.next_u64()).collect();
    assert_eq!(a, b);

    let mut other = KitRng::with_stream(7, 4);
    let c: Vec<u64> = (0..5).map(|_| other.next_u64()).collect();
    assert_ne!(a, c, "a different stream should differ");
}

#[test]
fn randint_with_empty_range_returns_low() {
    assert_eq!(KitRng::new(1).randint(5, 5), 5);
    assert_eq!(KitRng::new(1).randint(9, 5), 9);
}

#[test]
fn poisson_non_positive_lambda_is_zero() {
    assert_eq!(KitRng::new(1).poisson(0.0), 0);
    assert_eq!(KitRng::new(1).poisson(-2.0), 0);
}

#[test]
fn poisson_large_lambda_uses_normal_approximation() {
    let mut rng = KitRng::new(42);
    let total: u64 = (0..2000).map(|_| rng.poisson(100.0)).sum();
    let mean = total as f64 / 2000.0;
    assert!(
        (mean - 100.0).abs() < 1.0,
        "mean of Poisson(100) draws was {mean}"
    );

    let draw = KitRng::new(42).poisson(30.0);
    assert!(draw < 200, "Poisson(30) draw was {draw}");
}

#[test]
fn rand_int_fills_shape_with_values_in_range() {
    let a = NdArray::<f64>::rand_int(&mut KitRng::new(3), 2, 7, vec![4, 5]);
    assert_eq!(a.dims(), &[4, 5]);
    assert_eq!(a.size(), 20);
    for &v in a.data() {
        assert_eq!(v.fract(), 0.0);
        assert!((2.0..7.0).contains(&v), "value out of range: {v}");
    }
    let again = NdArray::<f64>::rand_int(&mut KitRng::new(3), 2, 7, vec![4, 5]);
    assert_eq!(a.data(), again.data());
}

#[test]
fn rand_binomial_fills_shape_with_values_in_range() {
    let a = NdArray::<f64>::rand_binomial(&mut KitRng::new(3), 10, 0.5, vec![3, 4]);
    assert_eq!(a.dims(), &[3, 4]);
    for &v in a.data() {
        assert_eq!(v.fract(), 0.0);
        assert!((0.0..=10.0).contains(&v), "value out of range: {v}");
    }
    let again = NdArray::<f64>::rand_binomial(&mut KitRng::new(3), 10, 0.5, vec![3, 4]);
    assert_eq!(a.data(), again.data());
}

#[test]
fn rand_poisson_fills_shape_with_non_negative_integers() {
    let a = NdArray::<f64>::rand_poisson(&mut KitRng::new(3), 4.0, vec![2, 6]);
    assert_eq!(a.dims(), &[2, 6]);
    for &v in a.data() {
        assert_eq!(v.fract(), 0.0);
        assert!(v >= 0.0, "negative Poisson value: {v}");
    }
    let again = NdArray::<f64>::rand_poisson(&mut KitRng::new(3), 4.0, vec![2, 6]);
    assert_eq!(a.data(), again.data());
}
