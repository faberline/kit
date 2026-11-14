//! Integration tests for the array module.

use arraykit::array::{ArrayError, AxisSlice, DType, KitRng, NdArray, Shape, SliceInfo, SvdResult};

// ============================================================================
// DType tests (from dtype.rs)
// ============================================================================

#[test]
fn test_dtype_size() {
    assert_eq!(DType::Float32.size(), 4);
    assert_eq!(DType::Float64.size(), 8);
    assert_eq!(DType::Int32.size(), 4);
    assert_eq!(DType::Int64.size(), 8);
    assert_eq!(DType::Bool.size(), 1);
}

#[test]
fn test_dtype_display() {
    assert_eq!(format!("{}", DType::Float32), "float32");
    assert_eq!(format!("{}", DType::Int64), "int64");
}

// ============================================================================
// Shape tests (from shape.rs)
// ============================================================================

#[test]
fn test_shape_creation() {
    let shape = Shape::new(vec![2, 3, 4]);
    assert_eq!(shape.ndim(), 3);
    assert_eq!(shape.dims(), &[2, 3, 4]);
    assert_eq!(shape.strides(), &[12, 4, 1]);
    assert_eq!(shape.size(), 24);
}

#[test]
fn test_shape_offset() {
    let shape = Shape::new(vec![2, 3]);
    assert_eq!(shape.offset(&[0, 0]).unwrap(), 0);
    assert_eq!(shape.offset(&[0, 1]).unwrap(), 1);
    assert_eq!(shape.offset(&[1, 0]).unwrap(), 3);
    assert_eq!(shape.offset(&[1, 2]).unwrap(), 5);
}

#[test]
fn test_broadcast_same_shape() {
    let a = Shape::new(vec![2, 3]);
    let b = Shape::new(vec![2, 3]);
    let result = Shape::broadcast(&a, &b).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
}

#[test]
fn test_broadcast_scalar() {
    let a = Shape::new(vec![2, 3]);
    let b = Shape::new(vec![1]);
    let result = Shape::broadcast(&a, &b).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
}

#[test]
fn test_broadcast_different_ndim() {
    let a = Shape::new(vec![2, 3]);
    let b = Shape::new(vec![3]);
    let result = Shape::broadcast(&a, &b).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
}

#[test]
fn test_broadcast_error() {
    let a = Shape::new(vec![2, 3]);
    let b = Shape::new(vec![4]);
    let result = Shape::broadcast(&a, &b);
    assert!(result.is_err());
}

// ============================================================================
// Slice tests (from slice.rs)
// ============================================================================

#[test]
fn test_axis_slice_full() {
    let slice = AxisSlice::Full;
    let norm = slice.normalize(5).unwrap();
    assert_eq!(norm.start, 0);
    assert_eq!(norm.stop, 5);
    assert_eq!(norm.step, 1);
    assert_eq!(norm.output_len, 5);
}

#[test]
fn test_axis_slice_range() {
    let slice = AxisSlice::range(1, 4);
    let norm = slice.normalize(5).unwrap();
    assert_eq!(norm.start, 1);
    assert_eq!(norm.stop, 4);
    assert_eq!(norm.step, 1);
    assert_eq!(norm.output_len, 3);
}

#[test]
fn test_axis_slice_negative_index() {
    let slice = AxisSlice::range(-2, -1);
    let norm = slice.normalize(5).unwrap();
    assert_eq!(norm.start, 3); // 5 + (-2) = 3
    assert_eq!(norm.stop, 4); // 5 + (-1) = 4
    assert_eq!(norm.step, 1);
    assert_eq!(norm.output_len, 1);
}

#[test]
fn test_axis_slice_negative_step() {
    // arr[4:1:-1] should give indices 4, 3, 2
    let slice = AxisSlice::range_step(4, 1, -1);
    let norm = slice.normalize(5).unwrap();
    assert_eq!(norm.start, 4);
    assert_eq!(norm.stop, 1);
    assert_eq!(norm.step, -1);
    assert_eq!(norm.output_len, 3);
}

#[test]
fn test_axis_slice_reverse() {
    // arr[::-1] should reverse the array
    let slice = AxisSlice::Slice {
        start: None,
        stop: None,
        step: Some(-1),
    };
    let norm = slice.normalize(5).unwrap();
    assert_eq!(norm.start, 4); // Default start for negative step
    assert_eq!(norm.stop, -1isize); // Before index 0 (isize)
    assert_eq!(norm.step, -1);
    assert_eq!(norm.output_len, 5);
}

#[test]
fn test_slice_info_output_shape() {
    let info = SliceInfo::new(vec![AxisSlice::Full, AxisSlice::Index(1)]);
    let shape = info.output_shape(&[3, 4]).unwrap();
    assert_eq!(shape, vec![3]); // Index reduces dimension
}

#[test]
fn test_slice_column() {
    // Slice to get second column: arr[:, 1]
    let info = SliceInfo::new(vec![AxisSlice::Full, AxisSlice::Index(1)]);
    let shape = info.output_shape(&[2, 2]).unwrap();
    assert_eq!(shape, vec![2]); // 2x2 -> 2 (column vector)
}

// ============================================================================
// NdArray tests (from ndarray.rs)
// ============================================================================

#[test]
fn test_create_2d_array() {
    let arr = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    assert_eq!(arr.dims(), &[2, 2]);
    assert_eq!(arr.get(&[0, 0]).unwrap(), 1);
    assert_eq!(arr.get(&[0, 1]).unwrap(), 2);
    assert_eq!(arr.get(&[1, 0]).unwrap(), 3);
    assert_eq!(arr.get(&[1, 1]).unwrap(), 4);
}

#[test]
fn test_zeros_ones() {
    let zeros = NdArray::<f64>::zeros(vec![2, 3]);
    assert_eq!(zeros.size(), 6);
    assert_eq!(zeros.get(&[0, 0]).unwrap(), 0.0);

    let ones = NdArray::<i32>::ones(vec![3, 2]);
    assert_eq!(ones.get(&[1, 1]).unwrap(), 1);
}

#[test]
fn test_reshape() {
    let arr = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let reshaped = arr.reshape(vec![3, 2]).unwrap();
    assert_eq!(reshaped.dims(), &[3, 2]);
    assert_eq!(reshaped.get(&[0, 0]).unwrap(), 1);
    assert_eq!(reshaped.get(&[2, 1]).unwrap(), 6);
}

#[test]
fn test_ndarray_slice_column() {
    // Test acceptance scenario: slice 2D array to get second column
    let arr = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let info = SliceInfo::new(vec![AxisSlice::Full, AxisSlice::Index(1)]);
    let result = arr.slice(&info).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.data(), &[2, 4]);
}

#[test]
fn test_ndarray_slice_reverse() {
    // Test reverse slicing: arr[::-1]
    let arr = NdArray::new(vec![1, 2, 3, 4, 5], vec![5]).unwrap();
    let reverse_slice = AxisSlice::Slice {
        start: None,
        stop: None,
        step: Some(-1),
    };
    let info = SliceInfo::new(vec![reverse_slice]);
    let result = arr.slice(&info).unwrap();
    assert_eq!(result.dims(), &[5]);
    assert_eq!(result.data(), &[5, 4, 3, 2, 1]);
}

#[test]
fn test_arange_positive_step() {
    let arr = NdArray::arange(0, 5, 1).unwrap();
    assert_eq!(arr.data(), &[0, 1, 2, 3, 4]);
}

#[test]
fn test_arange_negative_step() {
    let arr = NdArray::arange(5, 0, -1).unwrap();
    assert_eq!(arr.data(), &[5, 4, 3, 2, 1]);
}

#[test]
fn test_arange_zero_step() {
    assert!(NdArray::arange(0, 5, 0).is_err()); // step must not be zero
    assert!(NdArray::arange(0.0_f64, 1.0, 0.0).is_err());
}

#[test]
fn test_arange_basic() {
    // np.arange(0, 5, 1) is [0, 1, 2, 3, 4]
    assert_eq!(NdArray::arange(0, 5, 1).unwrap().data(), &[0, 1, 2, 3, 4]);
    // np.arange(5, 0, -2) is [5, 3, 1]
    assert_eq!(NdArray::arange(5, 0, -2).unwrap().data(), &[5, 3, 1]);
    // np.arange(3, 3, 1) is empty
    assert_eq!(NdArray::arange(3, 3, 1).unwrap().size(), 0);
    // np.arange(0, 5, -1) is empty
    assert_eq!(NdArray::arange(0, 5, -1).unwrap().size(), 0);
}

#[test]
fn test_arange_float_len_matches_numpy() {
    // np.arange(0.0, 1.0, 0.1) has 10 elements, the last one is 0.9
    let a = NdArray::arange(0.0_f64, 1.0, 0.1).unwrap();
    assert_eq!(a.size(), 10);
    assert!(f64::abs(a.data()[9] - 0.9) < 1e-12);
}

#[test]
fn test_arange_float_negative_step() {
    // np.arange(1.0, 0.0, -0.25) is [1.0, 0.75, 0.5, 0.25]
    let a = NdArray::arange(1.0_f64, 0.0, -0.25).unwrap();
    assert_eq!(a.data(), &[1.0, 0.75, 0.5, 0.25]);
}

#[test]
fn test_arange_f32() {
    // np.arange(0.0, 1.0, 0.5) is [0.0, 0.5]
    let a = NdArray::arange(0.0_f32, 1.0, 0.5).unwrap();
    assert_eq!(a.data(), &[0.0, 0.5]);
}

#[test]
fn test_arange_i64_large() {
    // The length is computed without overflow: two elements, 0 and i64::MAX / 2 + 1
    let a = NdArray::arange(0_i64, i64::MAX, i64::MAX / 2 + 1).unwrap();
    assert_eq!(a.size(), 2);
}

#[test]
fn test_transpose_2d() {
    // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
    let arr = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let transposed = arr.transpose();
    assert_eq!(transposed.dims(), &[3, 2]);
    assert_eq!(transposed.get(&[0, 0]).unwrap(), 1);
    assert_eq!(transposed.get(&[0, 1]).unwrap(), 4);
    assert_eq!(transposed.get(&[1, 0]).unwrap(), 2);
    assert_eq!(transposed.get(&[1, 1]).unwrap(), 5);
    assert_eq!(transposed.get(&[2, 0]).unwrap(), 3);
    assert_eq!(transposed.get(&[2, 1]).unwrap(), 6);
}

#[test]
fn test_transpose_1d() {
    // 1D transpose should return the same array
    let arr = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let transposed = arr.transpose();
    assert_eq!(transposed.dims(), &[3]);
    assert_eq!(transposed.data(), &[1, 2, 3]);
}

#[test]
fn test_transpose_axes() {
    // 2x3x4 -> permute(2, 0, 1) -> 4x2x3
    let data: Vec<i32> = (0..24).collect();
    let arr = NdArray::new(data, vec![2, 3, 4]).unwrap();
    let transposed = arr.transpose_axes(&[2, 0, 1]).unwrap();
    assert_eq!(transposed.dims(), &[4, 2, 3]);
}

#[test]
fn test_eye() {
    let eye: NdArray<f64> = NdArray::eye(3);
    assert_eq!(eye.dims(), &[3, 3]);
    assert_eq!(eye.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(eye.get(&[1, 1]).unwrap(), 1.0);
    assert_eq!(eye.get(&[2, 2]).unwrap(), 1.0);
    assert_eq!(eye.get(&[0, 1]).unwrap(), 0.0);
    assert_eq!(eye.get(&[1, 0]).unwrap(), 0.0);
}

#[test]
fn test_identity() {
    let id: NdArray<i32> = NdArray::identity(2);
    assert_eq!(id.dims(), &[2, 2]);
    assert_eq!(id.data(), &[1, 0, 0, 1]);
}

#[test]
fn test_diag_from_1d() {
    let vec: NdArray<i32> = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let diag = NdArray::diag(&vec).unwrap();
    assert_eq!(diag.dims(), &[3, 3]);
    assert_eq!(diag.get(&[0, 0]).unwrap(), 1);
    assert_eq!(diag.get(&[1, 1]).unwrap(), 2);
    assert_eq!(diag.get(&[2, 2]).unwrap(), 3);
    assert_eq!(diag.get(&[0, 1]).unwrap(), 0);
}

#[test]
fn test_diag_from_2d() {
    let mat: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]).unwrap();
    let diag = NdArray::diag(&mat).unwrap();
    assert_eq!(diag.dims(), &[3]);
    assert_eq!(diag.data(), &[1, 5, 9]);
}

#[test]
fn test_full() {
    let arr: NdArray<i32> = NdArray::full(vec![2, 3], 5);
    assert_eq!(arr.size(), 6);
    assert_eq!(arr.get(&[0, 0]).unwrap(), 5);
    assert_eq!(arr.get(&[1, 2]).unwrap(), 5);
}

#[test]
fn test_shape() {
    let arr = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let shape = arr.shape();
    assert_eq!(shape.dims(), &[2, 3]);
}

#[test]
fn test_dims_ndim_size() {
    let arr = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    assert_eq!(arr.dims(), &[2, 3]);
    assert_eq!(arr.ndim(), 2);
    assert_eq!(arr.size(), 6);
}

#[test]
fn test_data_data_mut() {
    let mut arr = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    assert_eq!(arr.data(), &[1, 2, 3, 4]);

    arr.data_mut()[0] = 10;
    assert_eq!(arr.data(), &[10, 2, 3, 4]);
}

#[test]
fn test_get_set() {
    let mut arr = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    assert_eq!(arr.get(&[0, 1]).unwrap(), 2);

    arr.set(&[0, 1], 20).unwrap();
    assert_eq!(arr.get(&[0, 1]).unwrap(), 20);
}

#[test]
fn test_flatten() {
    let arr = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let flat = arr.flatten();
    assert_eq!(flat.dims(), &[6]);
    assert_eq!(flat.data(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_broadcast_to() {
    let arr = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let target = Shape::new(vec![2, 3]);
    let broadcasted = arr.broadcast_to(&target).unwrap();
    assert_eq!(broadcasted.dims(), &[2, 3]);
    assert_eq!(broadcasted.get(&[0, 0]).unwrap(), 1);
    assert_eq!(broadcasted.get(&[1, 0]).unwrap(), 1);
}

// ============================================================================
// Ops tests (from ops.rs)
// ============================================================================

#[test]
fn test_add_same_shape() {
    let a = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.data(), &[11, 22, 33, 44]);
}

#[test]
fn test_broadcast_add() {
    // Test acceptance scenario: 1D + 2D broadcasting
    let a = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![10], vec![1]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.data(), &[11, 12, 13, 14]);
}

#[test]
fn test_broadcast_row() {
    let a = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![10, 20], vec![2]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.data(), &[11, 22, 13, 24]);
}

#[test]
fn test_sub() {
    let a = NdArray::new(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let c = a.sub(&b).unwrap();
    assert_eq!(c.data(), &[9, 18, 27, 36]);
}

#[test]
fn test_mul() {
    let a = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![2, 3, 4, 5], vec![2, 2]).unwrap();
    let c = a.mul(&b).unwrap();
    assert_eq!(c.data(), &[2, 6, 12, 20]);
}

#[test]
fn test_div() {
    let a = NdArray::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![2.0, 4.0, 5.0, 8.0], vec![2, 2]).unwrap();
    let c = a.div(&b).unwrap();
    assert_eq!(c.data(), &[5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_scalar_ops() {
    let a = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b = a.add_scalar(10);
    assert_eq!(b.data(), &[11, 12, 13, 14]);

    let c = a.mul_scalar(2);
    assert_eq!(c.data(), &[2, 4, 6, 8]);
}

#[test]
fn test_operator_syntax() {
    let a = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b = NdArray::new(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
    let c = (&a + &b).unwrap();
    assert_eq!(c.data(), &[11, 22, 33, 44]);
}

// ============================================================================
// Linalg tests (from linalg.rs)
// ============================================================================

#[test]
fn test_matmul_2x2() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b: NdArray<i32> = NdArray::new(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.data(), &[19, 22, 43, 50]);
}

#[test]
fn test_matmul_2x3_3x2() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    let b: NdArray<i32> = NdArray::new(vec![7, 8, 9, 10, 11, 12], vec![3, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert_eq!(c.data(), &[58, 64, 139, 154]);
}

#[test]
fn test_dot_1d() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let b: NdArray<i32> = NdArray::new(vec![4, 5, 6], vec![3]).unwrap();
    let c = a.dot(&b).unwrap();
    assert_eq!(c.data(), &[32]); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_dot_matrix_vector() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b: NdArray<i32> = NdArray::new(vec![5, 6], vec![2]).unwrap();
    let c = a.dot(&b).unwrap();
    assert_eq!(c.dims(), &[2]);
    assert_eq!(c.data(), &[17, 39]); // [1*5+2*6, 3*5+4*6]
}

#[test]
fn test_outer() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let b: NdArray<i32> = NdArray::new(vec![4, 5], vec![2]).unwrap();
    let c = a.outer(&b).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    assert_eq!(c.data(), &[4, 5, 8, 10, 12, 15]);
}

#[test]
fn test_trace() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]).unwrap();
    assert_eq!(a.trace().unwrap(), 15); // 1 + 5 + 9
}

#[test]
fn test_det_2x2() {
    let a: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let det = a.det().unwrap();
    assert!(f64::abs(det - (-2.0)) < 1e-10); // 1*4 - 2*3 = -2
}

#[test]
fn test_det_3x3() {
    let a: NdArray<f64> = NdArray::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        vec![3, 3],
    )
    .unwrap();
    let det = a.det().unwrap();
    assert!(f64::abs(det - (-3.0)) < 1e-10);
}

#[test]
fn test_inv_2x2() {
    let a: NdArray<f64> = NdArray::new(vec![4.0, 7.0, 2.0, 6.0], vec![2, 2]).unwrap();
    let inv = a.inv().unwrap();
    // A * A^-1 should be identity
    let product = a.matmul(&inv).unwrap();
    assert!(f64::abs(product.get(&[0, 0]).unwrap() - 1.0) < 1e-10);
    assert!(f64::abs(product.get(&[0, 1]).unwrap()) < 1e-10);
    assert!(f64::abs(product.get(&[1, 0]).unwrap()) < 1e-10);
    assert!(f64::abs(product.get(&[1, 1]).unwrap() - 1.0) < 1e-10);
}

#[test]
fn test_solve() {
    // Solve: 2x + y = 5, x + 3y = 7
    let a: NdArray<f64> = NdArray::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
    let b: NdArray<f64> = NdArray::new(vec![5.0, 7.0], vec![2]).unwrap();
    let x = a.solve(&b).unwrap();
    // x = 8/5 = 1.6, y = 9/5 = 1.8
    assert!(f64::abs(x.data()[0] - 1.6) < 1e-10);
    assert!(f64::abs(x.data()[1] - 1.8) < 1e-10);
}

#[test]
fn test_solve_2d_rhs() {
    // np.linalg.solve([[3, 1], [1, 2]], [[9, 1], [8, 3]]) = [[2, -0.2], [3, 1.6]]
    let a: NdArray<f64> = NdArray::new(vec![3.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let b: NdArray<f64> = NdArray::new(vec![9.0, 1.0, 8.0, 3.0], vec![2, 2]).unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    let expected = [2.0, -0.2, 3.0, 1.6];
    for (got, want) in x.data().iter().zip(expected.iter()) {
        assert!(f64::abs(got - want) < 1e-12, "got {got}, want {want}");
    }
}

#[test]
fn test_solve_1d_rhs_unchanged() {
    // np.linalg.solve([[3, 1], [1, 2]], [9, 8]) = [2, 3]
    let a: NdArray<f64> = NdArray::new(vec![3.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let b: NdArray<f64> = NdArray::new(vec![9.0, 8.0], vec![2]).unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.dims(), &[2]);
    assert!(f64::abs(x.data()[0] - 2.0) < 1e-12);
    assert!(f64::abs(x.data()[1] - 3.0) < 1e-12);
}

#[test]
fn test_solve_needs_pivoting() {
    // Zero on the diagonal: the solver must swap rows.
    // np.linalg.solve([[0, 1], [1, 0]], [2, 3]) = [3, 2]
    let a: NdArray<f64> = NdArray::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap();
    let b: NdArray<f64> = NdArray::new(vec![2.0, 3.0], vec![2]).unwrap();
    let x = a.solve(&b).unwrap();
    assert!(f64::abs(x.data()[0] - 3.0) < 1e-12);
    assert!(f64::abs(x.data()[1] - 2.0) < 1e-12);
}

#[test]
fn test_solve_3x3_2d() {
    let a: NdArray<f64> = NdArray::new(
        vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0],
        vec![3, 3],
    )
    .unwrap();
    let b: NdArray<f64> = NdArray::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.dims(), &[3, 1]);
    let check = a.matmul(&x).unwrap();
    for (got, want) in check.data().iter().zip(b.data().iter()) {
        assert!(f64::abs(got - want) < 1e-10, "got {got}, want {want}");
    }
}

#[test]
fn test_solve_singular_is_error() {
    let a: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let b: NdArray<f64> = NdArray::new(vec![1.0, 2.0], vec![2]).unwrap();
    assert!(a.solve(&b).is_err());
}

#[test]
fn test_solve_bad_shapes() {
    let a: NdArray<f64> = NdArray::new(vec![3.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let short: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert!(a.solve(&short).is_err());
    let rank3: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]).unwrap();
    assert!(a.solve(&rank3).is_err());
}

#[test]
fn test_inv_small_scale() {
    // np.linalg.inv([[1e-13]]) = [[1e13]]
    let a: NdArray<f64> = NdArray::new(vec![1e-13], vec![1, 1]).unwrap();
    let x = a.inv().unwrap().data()[0];
    assert!(f64::abs(x - 1e13) / 1e13 < 1e-9, "got {x}");
}

#[test]
fn test_inv_small_scale_2x2() {
    // np.linalg.inv(1e-7 * np.eye(2)) = 1e7 * np.eye(2)
    let a: NdArray<f64> = NdArray::new(vec![1e-7, 0.0, 0.0, 1e-7], vec![2, 2]).unwrap();
    let inv = a.inv().unwrap();
    assert!(f64::abs(inv.data()[0] - 1e7) / 1e7 < 1e-9);
    assert!(f64::abs(inv.data()[3] - 1e7) / 1e7 < 1e-9);
    assert!(inv.data()[1].abs() < 1e-12);
    assert!(inv.data()[2].abs() < 1e-12);
}

#[test]
fn test_det_small_scale() {
    // np.linalg.det(1e-7 * np.eye(2)) = 1e-14
    let a: NdArray<f64> = NdArray::new(vec![1e-7, 0.0, 0.0, 1e-7], vec![2, 2]).unwrap();
    let det = a.det().unwrap();
    assert!(f64::abs(det - 1e-14) / 1e-14 < 1e-9, "got {det}");
    // 3x3: np.linalg.det(1e-7 * np.eye(3)) = 1e-21
    let b: NdArray<f64> = NdArray::new(
        vec![1e-7, 0.0, 0.0, 0.0, 1e-7, 0.0, 0.0, 0.0, 1e-7],
        vec![3, 3],
    )
    .unwrap();
    let det3 = b.det().unwrap();
    assert!(f64::abs(det3 - 1e-21) / 1e-21 < 1e-9, "got {det3}");
}

#[test]
fn test_inv_det_singular() {
    let a: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
    assert!(matches!(
        a.inv().err(),
        Some(ArrayError::InvalidOperation(_))
    ));
    assert!(matches!(a.det(), Ok(d) if d == 0.0));

    let s: NdArray<f64> = NdArray::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3],
    )
    .unwrap();
    assert!(f64::abs(s.det().unwrap()) < 1e-12);
    assert!(s.inv().is_err());
}

#[test]
fn test_det_zero_matrix() {
    let z: NdArray<f64> = NdArray::new(vec![0.0; 9], vec![3, 3]).unwrap();
    assert!(matches!(z.det(), Ok(d) if d == 0.0));
    assert!(z.inv().is_err());
}

#[test]
fn test_norm() {
    let a: NdArray<f64> = NdArray::new(vec![3.0, 4.0], vec![2]).unwrap();
    assert_eq!(a.norm(Some(2.0)), 5.0); // sqrt(9+16)
    assert_eq!(a.norm(Some(1.0)), 7.0); // |3| + |4|
    assert_eq!(a.norm(Some(f64::INFINITY)), 4.0); // max
}

// ============================================================================
// Math tests (from math.rs)
// ============================================================================

#[test]
fn test_linspace() {
    let arr: NdArray<f64> = NdArray::linspace(0.0, 1.0, 5);
    assert_eq!(arr.size(), 5);
    assert!(f64::abs(arr.data()[0] - 0.0) < 1e-10);
    assert!(f64::abs(arr.data()[4] - 1.0) < 1e-10);
    assert!(f64::abs(arr.data()[2] - 0.5) < 1e-10);
}

#[test]
fn test_linspace_single() {
    let arr: NdArray<f64> = NdArray::linspace(5.0, 5.0, 1);
    assert_eq!(arr.size(), 1);
    assert_eq!(arr.data()[0], 5.0);
}

#[test]
fn test_sqrt() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 4.0, 9.0, 16.0], vec![4]).unwrap();
    let result = arr.sqrt();
    assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_exp_log() {
    let arr: NdArray<f64> = NdArray::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap();
    let exp_arr = arr.exp();
    assert!(f64::abs(exp_arr.data()[0] - 1.0) < 1e-10);
    assert!(f64::abs(exp_arr.data()[1] - std::f64::consts::E) < 1e-10);

    let log_arr = exp_arr.log();
    assert!(f64::abs(log_arr.data()[0] - 0.0) < 1e-10);
    assert!(f64::abs(log_arr.data()[1] - 1.0) < 1e-10);
}

#[test]
fn test_abs() {
    let arr: NdArray<f64> = NdArray::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]).unwrap();
    let result = arr.abs();
    assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_clip() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 5.0, 10.0, 15.0], vec![4]).unwrap();
    let clipped = arr.clip(3.0, 12.0);
    assert_eq!(clipped.data(), &[3.0, 5.0, 10.0, 12.0]);
}

#[test]
fn test_where_cond() {
    let a: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let b: NdArray<f64> = NdArray::new(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
    let cond = vec![true, false, true, false];
    let result = a.where_cond(&cond, &b).unwrap();
    assert_eq!(result.data(), &[1.0, 20.0, 3.0, 40.0]);
}

#[test]
fn test_pow() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let result = arr.pow(2.0);
    assert_eq!(result.data(), &[1.0, 4.0, 9.0]);
}

#[test]
fn test_trig() {
    let arr: NdArray<f64> = NdArray::new(vec![0.0], vec![1]).unwrap();
    assert!(f64::abs(arr.sin().data()[0] - 0.0) < 1e-10);
    assert!(f64::abs(arr.cos().data()[0] - 1.0) < 1e-10);
    assert!(f64::abs(arr.tan().data()[0] - 0.0) < 1e-10);
}

#[test]
fn test_abs_int() {
    let arr: NdArray<i32> = NdArray::new(vec![-1, 2, -3, 4], vec![4]).unwrap();
    let result = arr.abs_int();
    assert_eq!(result.data(), &[1, 2, 3, 4]);
}

// ============================================================================
// Manip tests (from manip.rs)
// ============================================================================

#[test]
fn test_concatenate_1d() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let b: NdArray<i32> = NdArray::new(vec![4, 5, 6], vec![3]).unwrap();
    let c = NdArray::concatenate(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[6]);
    assert_eq!(c.data(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_concatenate_2d_axis0() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b: NdArray<i32> = NdArray::new(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
    let c = NdArray::concatenate(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[4, 2]);
    assert_eq!(c.data(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_stack() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let b: NdArray<i32> = NdArray::new(vec![4, 5, 6], vec![3]).unwrap();
    let c = NdArray::stack(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.data(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_stack_axis_too_large() {
    let a: NdArray<f64> = NdArray::new(vec![1.0, 2.0], vec![2]).unwrap();
    let b: NdArray<f64> = NdArray::new(vec![3.0, 4.0], vec![2]).unwrap();
    assert_eq!(
        NdArray::stack(&[&a, &b], 3).err(),
        Some(ArrayError::InvalidAxis { axis: 3, ndim: 2 })
    );
    assert!(NdArray::stack(&[&a, &b], 2).is_err());
}

#[test]
fn test_stack_last_axis_ok() {
    let a: NdArray<f64> = NdArray::new(vec![1.0, 2.0], vec![2]).unwrap();
    let b: NdArray<f64> = NdArray::new(vec![3.0, 4.0], vec![2]).unwrap();
    let c = NdArray::stack(&[&a, &b], 1).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.data(), &[1.0, 3.0, 2.0, 4.0]);
    let d = NdArray::stack(&[&a, &b], 0).unwrap();
    assert_eq!(d.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_rolling_zero_window_is_error() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert!(arr.rolling(0).is_err());
    assert!(matches!(
        arr.rolling(0).err(),
        Some(ArrayError::InvalidOperation(_))
    ));
}

#[test]
fn test_rolling_window_one_ok() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert!(arr.rolling(1).is_ok());
    assert_eq!(arr.rolling(1).unwrap().sum().data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_choice_zero_is_error() {
    let mut rng = KitRng::new(42);
    assert!(rng.choice(0).is_err());
}

#[test]
fn test_choice_in_range() {
    let mut rng = KitRng::new(42);
    for _ in 0..100 {
        let k = rng.choice(5).unwrap();
        assert!(k < 5);
    }
}

#[test]
fn test_shuffle_still_permutes() {
    let mut rng = KitRng::new(42);
    let mut v = vec![1, 2, 3, 4, 5];
    rng.shuffle(&mut v);
    v.sort();
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
    let mut empty: Vec<i32> = vec![];
    rng.shuffle(&mut empty);
    let mut one = vec![7];
    rng.shuffle(&mut one);
    assert_eq!(one, vec![7]);
}

#[test]
fn test_vstack() {
    let a: NdArray<i32> = NdArray::new(vec![1, 2], vec![2]).unwrap();
    let b: NdArray<i32> = NdArray::new(vec![3, 4], vec![2]).unwrap();
    let c = NdArray::vstack(&[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[4]);
    assert_eq!(c.data(), &[1, 2, 3, 4]);
}

#[test]
fn test_split() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![6]).unwrap();
    let parts = arr.split(&[2, 4], 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].data(), &[1, 2]);
    assert_eq!(parts[1].data(), &[3, 4]);
    assert_eq!(parts[2].data(), &[5, 6]);
}

#[test]
fn test_repeat() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let repeated = arr.repeat(2);
    assert_eq!(repeated.data(), &[1, 1, 2, 2, 3, 3]);
}

#[test]
fn test_tile() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2], vec![2]).unwrap();
    let tiled = arr.tile(&[3]).unwrap();
    assert_eq!(tiled.dims(), &[6]);
    assert_eq!(tiled.data(), &[1, 2, 1, 2, 1, 2]);
}

#[test]
fn test_unique() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 2, 3, 3, 3, 1], vec![7]).unwrap();
    let uniq = arr.unique();
    assert_eq!(uniq.size(), 3);
}

#[test]
fn test_sort() {
    let arr: NdArray<i32> = NdArray::new(vec![3, 1, 4, 1, 5, 9, 2, 6], vec![8]).unwrap();
    let sorted = arr.sort();
    assert_eq!(sorted.data(), &[1, 1, 2, 3, 4, 5, 6, 9]);
}

#[test]
fn test_argsort() {
    let arr: NdArray<i32> = NdArray::new(vec![3, 1, 4], vec![3]).unwrap();
    let indices = arr.argsort();
    assert_eq!(indices, vec![1, 0, 2]); // indices that sort: 1, 3, 4
}

#[test]
fn test_all_any() {
    let arr = NdArray::new(vec![true, true, true], vec![3]).unwrap();
    assert!(arr.all());
    assert!(arr.any());

    let arr2 = NdArray::new(vec![true, false, true], vec![3]).unwrap();
    assert!(!arr2.all());
    assert!(arr2.any());

    let arr3 = NdArray::new(vec![false, false, false], vec![3]).unwrap();
    assert!(!arr3.all());
    assert!(!arr3.any());
}

#[test]
fn test_allclose() {
    let a: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert!(a.allclose(&b, 1e-5, 1e-8));

    let c: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.1], vec![3]).unwrap();
    assert!(!a.allclose(&c, 1e-5, 1e-8));
}

// ============================================================================
// Stats tests (from stats.rs)
// ============================================================================

#[test]
fn test_sum_all() {
    let arr = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    assert_eq!(arr.sum(), 10);
}

#[test]
fn test_sum_axis_0() {
    // [[1, 2], [3, 4]] -> sum along axis 0 -> [4, 6]
    let arr = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let result = arr.sum_axis(0).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.data(), &[4, 6]);
}

#[test]
fn test_sum_axis_1() {
    // [[1, 2], [3, 4]] -> sum along axis 1 -> [3, 7]
    let arr = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let result = arr.sum_axis(1).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.data(), &[3, 7]);
}

#[test]
fn test_sum_axis_3d() {
    // 2x2x3 array
    let arr = NdArray::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], vec![2, 2, 3]).unwrap();

    // Sum along axis 2 (innermost) -> 2x2
    let result = arr.sum_axis(2).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    assert_eq!(result.data(), &[6, 15, 24, 33]); // 1+2+3, 4+5+6, 7+8+9, 10+11+12
}

#[test]
fn test_mean_all() {
    let arr = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    assert_eq!(arr.mean(), 2.5);
}

#[test]
fn test_mean_axis() {
    let arr = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = arr.mean_axis(0).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.data(), &[2.0, 3.0]); // (1+3)/2, (2+4)/2
}

#[test]
fn test_min_max_all() {
    let arr = NdArray::new(vec![3, 1, 4, 1, 5, 9], vec![6]).unwrap();
    assert_eq!(arr.min(), Some(1));
    assert_eq!(arr.max(), Some(9));
}

#[test]
fn test_min_axis() {
    let arr = NdArray::new(vec![3, 1, 4, 2], vec![2, 2]).unwrap();
    let result = arr.min_axis(0).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.data(), &[3, 1]); // min(3,4), min(1,2)
}

#[test]
fn test_max_axis() {
    let arr = NdArray::new(vec![3, 1, 4, 2], vec![2, 2]).unwrap();
    let result = arr.max_axis(1).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.data(), &[3, 4]); // max(3,1), max(4,2)
}

#[test]
fn test_empty_array() {
    let arr = NdArray::<f64>::zeros(vec![0]);
    assert_eq!(arr.sum(), 0.0);
    // NumPy: np.mean([]) is nan (the old value 0.0 was wrong)
    assert!(arr.mean().is_nan());
    assert_eq!(arr.min(), None);
    assert_eq!(arr.max(), None);
}

#[test]
fn test_var_std() {
    let arr = NdArray::new(vec![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], vec![8]).unwrap();
    let variance = arr.var();
    let std_dev = arr.std();
    // Sample variance = 4.571..., std = 2.138...
    assert!(f64::abs(variance - 4.571428571428571) < 1e-10);
    assert!(f64::abs(std_dev - 2.138089935299395) < 1e-10);
}

#[test]
fn test_var_pop() {
    let arr = NdArray::new(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let var_pop = arr.var_pop();
    assert!(f64::abs(var_pop - 2.0) < 1e-10); // Population variance = 2.0
}

#[test]
fn test_mean_var_empty_are_nan() {
    // NumPy: np.mean([]) and np.var([]) are nan
    let a = NdArray::<f64>::new(vec![], vec![0]).unwrap();
    assert!(a.mean().is_nan());
    assert!(a.var().is_nan());
    assert!(a.var_pop().is_nan());
}

#[test]
fn test_var_single_element() {
    // NumPy: np.var([5.0], ddof=1) is nan; np.var([5.0]) is 0.0
    let a = NdArray::new(vec![5.0_f64], vec![1]).unwrap();
    assert!(a.var().is_nan());
    assert_eq!(a.var_pop(), 0.0);
}

#[test]
fn test_var_sample_four() {
    // NumPy: np.var([1.0, 2.0, 3.0, 4.0], ddof=1) is 1.6666666666666667
    let a = NdArray::new(vec![1.0_f64, 2.0, 3.0, 4.0], vec![4]).unwrap();
    assert!((a.var() - 1.6666666666666667).abs() < 1e-12);
}

#[test]
fn test_checked_sum_i32_overflow() {
    // NumPy wraps on overflow; checked_sum must return an error instead
    let a = NdArray::new(vec![i32::MAX, 1], vec![2]).unwrap();
    match a.checked_sum() {
        Err(ArrayError::InvalidOperation(m)) => assert_eq!(m, "integer overflow in sum"),
        other => panic!("expected InvalidOperation, got {other:?}"),
    }
}

#[test]
fn test_empty_mean_per_type() {
    // NumPy: np.mean([], dtype=float32) is nan; integers have no NaN, so zero
    let f = NdArray::<f32>::new(vec![], vec![0]).unwrap();
    assert!(f.mean().is_nan());
    let i = NdArray::<i32>::new(vec![], vec![0]).unwrap();
    assert_eq!(i.mean(), 0);
    let l = NdArray::<i64>::new(vec![], vec![0]).unwrap();
    assert_eq!(l.mean(), 0);
    let s = NdArray::new(vec![0.5_f32, 0.25], vec![2]).unwrap();
    assert_eq!(s.checked_sum().unwrap(), 0.75);
}

#[test]
fn test_checked_sum_ok() {
    let a = NdArray::new(vec![1_i32, 2, 3], vec![3]).unwrap();
    assert_eq!(a.checked_sum().unwrap(), 6);
    let f = NdArray::new(vec![0.5_f64, 0.25], vec![2]).unwrap();
    assert_eq!(f.checked_sum().unwrap(), 0.75);
    let big = NdArray::new(vec![i64::MAX, 1], vec![2]).unwrap();
    match big.checked_sum() {
        Err(ArrayError::InvalidOperation(m)) => assert_eq!(m, "integer overflow in sum"),
        other => panic!("expected InvalidOperation, got {other:?}"),
    }
}

#[test]
fn test_argmin_argmax() {
    let arr = NdArray::new(vec![3, 1, 4, 1, 5, 9, 2, 6], vec![8]).unwrap();
    assert_eq!(arr.argmin(), Some(1)); // First occurrence of min (1)
    assert_eq!(arr.argmax(), Some(5)); // Index of max (9)
}

#[test]
fn test_cumsum() {
    let arr = NdArray::new(vec![1, 2, 3, 4], vec![4]).unwrap();
    let result = arr.cumsum();
    assert_eq!(result.data(), &[1, 3, 6, 10]);
}

#[test]
fn test_cumprod() {
    let arr = NdArray::new(vec![1, 2, 3, 4], vec![4]).unwrap();
    let result = arr.cumprod();
    assert_eq!(result.data(), &[1, 2, 6, 24]);
}

#[test]
fn test_percentile() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    assert_eq!(arr.percentile(0.0), Some(1.0));
    assert_eq!(arr.percentile(50.0), Some(3.0));
    assert_eq!(arr.percentile(100.0), Some(5.0));
}

#[test]
fn percentile_rejects_out_of_range_q() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert_eq!(arr.percentile(-0.1), None);
    assert_eq!(arr.percentile(100.1), None);
}

#[test]
fn percentile_rejects_nan_q() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert_eq!(arr.percentile(f64::NAN), None);
}

#[test]
fn test_quantile() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    assert_eq!(arr.quantile(0.0), Some(1.0));
    assert_eq!(arr.quantile(0.5), Some(3.0));
    assert_eq!(arr.quantile(1.0), Some(5.0));
}

#[test]
fn test_corrcoef() {
    // Two variables, 5 observations each
    // x = [1, 2, 3, 4, 5], y = [2, 4, 6, 8, 10]
    let arr: NdArray<f64> = NdArray::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        vec![2, 5],
    )
    .unwrap();
    let corr = arr.corrcoef().unwrap();
    assert_eq!(corr.dims(), &[2, 2]);
    // Perfect correlation
    assert!(f64::abs(corr.get(&[0, 1]).unwrap() - 1.0) < 1e-10);
    assert!(f64::abs(corr.get(&[1, 0]).unwrap() - 1.0) < 1e-10);
}

#[test]
fn test_cov() {
    let arr: NdArray<f64> = NdArray::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        vec![2, 5],
    )
    .unwrap();
    let cov_mat = arr.cov().unwrap();
    assert_eq!(cov_mat.dims(), &[2, 2]);
    // Variance of [1,2,3,4,5] = 2.5
    assert!(f64::abs(cov_mat.get(&[0, 0]).unwrap() - 2.5) < 1e-10);
}

#[test]
fn test_histogram() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let (counts, edges) = arr.histogram(5);
    assert_eq!(counts.len(), 5);
    assert_eq!(edges.len(), 6);
    assert_eq!(counts.iter().sum::<usize>(), 5);
}

// Phase 2 advanced statistics tests
#[test]
fn test_median() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 3.0, 5.0, 7.0, 9.0], vec![5]).unwrap();
    assert_eq!(arr.median(), Some(5.0));

    let arr2: NdArray<f64> = NdArray::new(vec![1.0, 3.0, 5.0, 7.0], vec![4]).unwrap();
    assert_eq!(arr2.median(), Some(4.0));
}

#[test]
fn test_mode() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0], vec![6]).unwrap();
    assert_eq!(arr.mode(), Some(3.0));
}

#[test]
fn test_skew() {
    // Symmetric distribution should have skewness near 0
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    assert!(arr.skew().abs() < 0.1);

    // Right-skewed distribution
    let arr2: NdArray<f64> = NdArray::new(vec![1.0, 1.0, 1.0, 1.0, 10.0], vec![5]).unwrap();
    assert!(arr2.skew() > 0.0);
}

#[test]
fn test_kurtosis() {
    // Normal-like distribution should have excess kurtosis near 0
    let arr: NdArray<f64> = NdArray::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        vec![10],
    )
    .unwrap();
    assert!(arr.kurtosis().abs() < 2.0);
}

#[test]
fn test_zscore() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let z = arr.zscore();
    // Mean of z-scores should be 0
    assert!(z.mean().abs() < 1e-10);
}

#[test]
fn test_iqr() {
    let arr: NdArray<f64> =
        NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![9]).unwrap();
    let result = arr.iqr().unwrap();
    assert!((result - 4.0).abs() < 0.5);
}

#[test]
fn test_geometric_mean() {
    let arr: NdArray<f64> = NdArray::new(vec![2.0, 8.0], vec![2]).unwrap();
    assert!((arr.geometric_mean() - 4.0).abs() < 1e-10);
}

#[test]
fn test_harmonic_mean() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 4.0], vec![3]).unwrap();
    let hm = arr.harmonic_mean();
    assert!((hm - 12.0 / 7.0).abs() < 1e-10);
}

#[test]
fn test_trim_mean() {
    let arr: NdArray<f64> = NdArray::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        vec![10],
    )
    .unwrap();
    let tm = arr.trim_mean(0.1);
    // Should remove 1 element from each end
    assert!((tm - 5.5).abs() < 0.5);
}

#[test]
fn test_sem() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let s = arr.sem();
    // std = sqrt(2.5), sem = sqrt(2.5) / sqrt(5)
    assert!((s - (2.5_f64.sqrt() / 5.0_f64.sqrt())).abs() < 1e-10);
}

#[test]
fn test_cv() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let cv = arr.coefficient_of_variation();
    // mean = 3, std = sqrt(2.5), cv = sqrt(2.5) / 3
    assert!((cv - 2.5_f64.sqrt() / 3.0).abs() < 1e-10);
}

#[test]
fn test_moment() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    // Second central moment = variance (population)
    let m2 = arr.moment(2);
    assert!((m2 - 2.0).abs() < 1e-10);
}

#[test]
fn test_std_axis() {
    let arr = NdArray::new(vec![1.0_f64, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let result = arr.std_axis(0).unwrap();
    assert_eq!(result.dims(), &[2]);
    // std([1, 2]) and std([3, 4])
    assert!(result.data()[0] > 0.0);
}

#[test]
fn test_var_axis() {
    let arr = NdArray::new(vec![1.0_f64, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let result = arr.var_axis(0).unwrap();
    assert_eq!(result.dims(), &[2]);
    // var([1, 2]) and var([3, 4])
    assert!(result.data()[0] > 0.0);
}

#[test]
fn test_std_pop() {
    let arr = NdArray::new(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let std_pop = arr.std_pop();
    // Population std = sqrt(2.0)
    assert!((std_pop - 2.0_f64.sqrt()).abs() < 1e-10);
}

// ============================================================================
// Window tests (from window.rs)
// ============================================================================

#[test]
fn test_rolling_sum() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let result = arr.rolling(3).unwrap().sum();
    // First 2 elements are partial windows
    assert_eq!(result.data()[2], 6.0); // 1+2+3
    assert_eq!(result.data()[3], 9.0); // 2+3+4
    assert_eq!(result.data()[4], 12.0); // 3+4+5
}

#[test]
fn test_rolling_mean() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let result = arr.rolling(3).unwrap().mean();
    assert_eq!(result.data()[2], 2.0); // (1+2+3)/3
    assert_eq!(result.data()[4], 4.0); // (3+4+5)/3
}

#[test]
fn test_rolling_min_max() {
    let arr: NdArray<f64> = NdArray::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();
    let min_result = arr.rolling(3).unwrap().min();
    let max_result = arr.rolling(3).unwrap().max();
    assert_eq!(min_result.data()[2], 1.0);
    assert_eq!(max_result.data()[2], 4.0);
}

#[test]
fn test_squeeze() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3], vec![1, 3, 1]).unwrap();
    let squeezed = arr.squeeze();
    assert_eq!(squeezed.dims(), &[3]);
}

#[test]
fn test_expand_dims() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let expanded = arr.expand_dims(0);
    assert_eq!(expanded.dims(), &[1, 3]);
}

#[test]
fn test_flip() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5], vec![5]).unwrap();
    let flipped = arr.flip();
    assert_eq!(flipped.data(), &[5, 4, 3, 2, 1]);
}

#[test]
fn test_roll() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4, 5], vec![5]).unwrap();
    let rolled = arr.roll(2);
    assert_eq!(rolled.data(), &[4, 5, 1, 2, 3]);
}

#[test]
fn test_pad() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let padded = arr.pad(2).unwrap();
    assert_eq!(padded.data(), &[0, 0, 1, 2, 3, 0, 0]);
}

#[test]
fn test_diff() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 4.0, 7.0, 11.0], vec![5]).unwrap();
    let diff1 = arr.diff(1).unwrap();
    assert_eq!(diff1.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_gradient() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 4.0, 7.0, 11.0], vec![5]).unwrap();
    let grad = arr.gradient().unwrap();
    assert_eq!(grad.size(), 5);
    assert_eq!(grad.data()[0], 1.0); // Forward: 2-1
    assert_eq!(grad.data()[2], 2.5); // Central: (7-2)/2
}

#[test]
fn test_interp() {
    let xp = vec![0.0, 1.0, 2.0];
    let fp = vec![0.0, 10.0, 20.0];
    let x = vec![0.5, 1.5];
    let result = NdArray::interp(&x, &xp, &fp);
    assert_eq!(result, vec![5.0, 15.0]);
}

#[test]
fn test_rot90() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let rotated = arr.rot90(1).unwrap();
    assert_eq!(rotated.dims(), &[2, 2]);
    assert_eq!(rotated.data(), &[2, 4, 1, 3]);
}

#[test]
fn test_rot90_rejects_3d() {
    let arr: NdArray<i32> = NdArray::new(vec![0, 1, 2, 3, 4, 5, 6, 7], vec![2, 2, 2]).unwrap();
    assert!(matches!(arr.rot90(1), Err(ArrayError::InvalidOperation(_))));
    // The dimension check comes before the k % 4 check.
    assert!(matches!(arr.rot90(0), Err(ArrayError::InvalidOperation(_))));
}

#[test]
fn test_pad_rejects_2d() {
    let arr: NdArray<i32> = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    assert!(matches!(arr.pad(1), Err(ArrayError::InvalidOperation(_))));
}

#[test]
fn test_diff_rejects_2d() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    assert!(matches!(arr.diff(1), Err(ArrayError::InvalidOperation(_))));
}

#[test]
fn test_diff_n_too_large() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert!(matches!(arr.diff(3), Err(ArrayError::InvalidOperation(_))));
    // n == 0 is not an error: nothing to difference.
    assert_eq!(arr.diff(0).unwrap().data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_gradient_rejects_2d() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    assert!(matches!(
        arr.gradient(),
        Err(ArrayError::InvalidOperation(_))
    ));
}

#[test]
fn test_gradient_rejects_single_element() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0], vec![1]).unwrap();
    assert!(matches!(
        arr.gradient(),
        Err(ArrayError::InvalidOperation(_))
    ));
}

#[test]
fn test_rolling_min_periods() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    // With min_periods=1, we get values even for partial windows
    let result = arr.rolling(3).unwrap().min_periods(1).sum();
    assert_eq!(result.data()[0], 1.0); // just first element
    assert_eq!(result.data()[1], 3.0); // 1+2
    assert_eq!(result.data()[2], 6.0); // 1+2+3
}

#[test]
fn test_rolling_count() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let counts = arr.rolling(3).unwrap().count();
    assert_eq!(counts, vec![1, 2, 3, 3, 3]);
}

#[test]
fn test_rolling_std() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let result = arr.rolling(3).unwrap().std();
    // First two are NaN (min_periods=3 by default)
    assert!(result.data()[0].is_nan());
    assert!(result.data()[1].is_nan());
    // std of [1,2,3] = 1.0 (sample std)
    assert!((result.data()[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_rolling_var() {
    let arr: NdArray<f64> = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    let result = arr.rolling(3).unwrap().var();
    // First two are NaN
    assert!(result.data()[0].is_nan());
    assert!(result.data()[1].is_nan());
    // var of [1,2,3] = 1.0 (sample var)
    assert!((result.data()[2] - 1.0).abs() < 1e-10);
}

// ============================================================================
// Issue #41: exact mean_f64, NaN-independent min/max, exact mode keys
// ============================================================================

#[test]
fn test_mean_f64_integer() {
    // np.mean([1, 2]) == 1.5
    let a = NdArray::<i32>::new(vec![1, 2], vec![2]).unwrap();
    assert_eq!(a.mean_f64(), 1.5);
    // The integer mean() still truncates.
    assert_eq!(a.mean(), 1);
}

#[test]
fn test_mean_f64_empty() {
    // np.mean([]) is nan
    let a = NdArray::<f64>::new(vec![], vec![0]).unwrap();
    assert!(a.mean_f64().is_nan());
}

#[test]
fn test_mean_f64_i64_and_f32() {
    // np.mean([1, 2, 4]) == 7/3
    let a = NdArray::<i64>::new(vec![1, 2, 4], vec![3]).unwrap();
    assert!(f64::abs(a.mean_f64() - 7.0 / 3.0) < 1e-12);
    // np.mean([1.0, 2.0]) == 1.5
    let b = NdArray::<f32>::new(vec![1.0, 2.0], vec![2]).unwrap();
    assert_eq!(b.mean_f64(), 1.5);
}

#[test]
fn test_min_nan_order() {
    // np.min([1.0, nan, 0.5]) is nan; the order must not matter
    for data in [
        vec![1.0, f64::NAN, 0.5],
        vec![f64::NAN, 1.0, 0.5],
        vec![0.5, 1.0, f64::NAN],
    ] {
        let a = NdArray::new(data, vec![3]).unwrap();
        assert!(a.min().unwrap().is_nan());
    }
}

#[test]
fn test_max_nan_order() {
    // np.max([nan, 1.0]) is nan; the order must not matter
    for data in [
        vec![f64::NAN, 1.0],
        vec![1.0, f64::NAN],
        vec![1.0, f64::NAN, 2.0],
    ] {
        let n = data.len();
        let a = NdArray::new(data, vec![n]).unwrap();
        assert!(a.max().unwrap().is_nan());
    }
}

#[test]
fn test_min_max_empty_and_plain() {
    let e = NdArray::<f64>::new(vec![], vec![0]).unwrap();
    assert_eq!(e.min(), None);
    assert_eq!(e.max(), None);
    let a = NdArray::new(vec![3.0_f64, 1.0, 2.0], vec![3]).unwrap();
    assert_eq!(a.min(), Some(1.0));
    assert_eq!(a.max(), Some(3.0));
}

#[test]
fn test_min_max_axis_nan() {
    // [[1.0, NaN], [3.0, 4.0]]
    let a = NdArray::new(vec![1.0_f64, f64::NAN, 3.0, 4.0], vec![2, 2]).unwrap();
    let mins = a.min_axis(1).unwrap();
    assert!(mins.data()[0].is_nan());
    assert_eq!(mins.data()[1], 3.0);
    let maxs = a.max_axis(0).unwrap();
    assert_eq!(maxs.data()[0], 3.0);
    assert!(maxs.data()[1].is_nan());
}

#[test]
fn test_mode_large_values() {
    // scipy.stats.mode([1e300, 2e300, 2e300]) is 2e300
    let a = NdArray::new(vec![1e300_f64, 2e300, 2e300], vec![3]).unwrap();
    assert_eq!(a.mode(), Some(2e300));
}

#[test]
fn test_mode_close_values_are_different() {
    // Values that differ by 1e-11 are different keys.
    let a = NdArray::new(
        vec![1.0_f64, 1.0 + 1e-11, 1.0 + 1e-11, 5.0, 5.0, 5.0],
        vec![6],
    )
    .unwrap();
    assert_eq!(a.mode(), Some(5.0));
    let b = NdArray::new(vec![1.0_f64, 1.0 + 1e-11, 1.0 + 1e-11], vec![3]).unwrap();
    assert_eq!(b.mode(), Some(1.0 + 1e-11));
}

#[test]
fn test_mode_tie_smallest_wins() {
    // scipy.stats.mode returns the smallest value on a tie.
    let a = NdArray::new(vec![3.0_f64, 1.0, 2.0, 2.0, 3.0, 1.0], vec![6]).unwrap();
    assert_eq!(a.mode(), Some(1.0));
    let b = NdArray::new(vec![2.0_f64, 3.0, 1.0, 3.0, 1.0, 2.0], vec![6]).unwrap();
    assert_eq!(b.mode(), Some(1.0));
}

#[test]
fn test_mode_zero_and_nan() {
    // -0.0 and 0.0 are one value.
    let a = NdArray::new(vec![-0.0_f64, 0.0, 0.0, 1.0], vec![4]).unwrap();
    assert_eq!(a.mode(), Some(0.0));
    // All NaN are one value.
    let b = NdArray::new(vec![f64::NAN, f64::NAN, 1.0], vec![3]).unwrap();
    assert!(b.mode().unwrap().is_nan());
}

// ============================================================================
// KitRng binomial tests (exact for every n)
// ============================================================================

#[test]
fn test_binomial_n100_p05_moments() {
    let mut rng = KitRng::new(42);
    let draws: Vec<f64> = (0..10_000).map(|_| rng.binomial(100, 0.5) as f64).collect();
    let count = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / count;
    let mean_sq = draws.iter().map(|x| x * x).sum::<f64>() / count;
    let variance = mean_sq - mean * mean;
    assert!((mean - 50.0).abs() < 0.2, "mean was {mean}");
    assert!((variance - 25.0).abs() < 1.5, "variance was {variance}");
}

#[test]
fn test_binomial_small_p() {
    let mut rng = KitRng::new(42);
    let draws: Vec<u64> = (0..10_000).map(|_| rng.binomial(100, 0.01)).collect();
    assert!(draws.iter().all(|&k| k <= 100));
    let mean = draws.iter().sum::<u64>() as f64 / draws.len() as f64;
    assert!((mean - 1.0).abs() < 0.05, "mean was {mean}");
}

#[test]
fn test_binomial_small_p_zero_share() {
    // Exact law: P(0) = 0.99^100 and P(3) = C(100, 3) * 0.01^3 * 0.99^97.
    let mut rng = KitRng::new(42);
    let draws: Vec<u64> = (0..10_000).map(|_| rng.binomial(100, 0.01)).collect();
    let total = draws.len() as f64;
    let share_zero = draws.iter().filter(|&&k| k == 0).count() as f64 / total;
    let share_three = draws.iter().filter(|&&k| k == 3).count() as f64 / total;
    let p_zero = 0.99_f64.powi(100);
    let p_three = 161_700.0 * 0.01_f64.powi(3) * 0.99_f64.powi(97);
    assert!(
        (share_zero - p_zero).abs() < 0.02,
        "share of 0 was {share_zero}, expected about {p_zero}"
    );
    assert!(
        (share_three - p_three).abs() < 0.015,
        "share of 3 was {share_three}, expected about {p_three}"
    );
}

#[test]
fn test_binomial_large_p() {
    let mut rng = KitRng::new(42);
    let draws: Vec<u64> = (0..10_000).map(|_| rng.binomial(100, 0.99)).collect();
    assert!(draws.iter().all(|&k| k <= 100));
    let mean = draws.iter().sum::<u64>() as f64 / draws.len() as f64;
    assert!((mean - 99.0).abs() < 0.2, "mean was {mean}");
}

#[test]
fn test_binomial_edges() {
    let mut rng = KitRng::new(42);
    assert_eq!(rng.binomial(100, 0.0), 0);
    assert_eq!(rng.binomial(100, 1.0), 100);
    assert_eq!(rng.binomial(0, 0.5), 0);
    assert_eq!(rng.binomial(1_000_000, 0.0), 0);
    assert_eq!(rng.binomial(1_000_000, 1.0), 1_000_000);
}

#[test]
fn test_binomial_huge_n_small_p() {
    let mut rng = KitRng::new(42);
    assert!(rng.binomial(1_000_000, 0.000001) <= 1_000_000);
    let draws: Vec<u64> = (0..1_000)
        .map(|_| rng.binomial(1_000_000, 0.000001))
        .collect();
    assert!(draws.iter().all(|&k| k <= 1_000_000));
    let mean = draws.iter().sum::<u64>() as f64 / draws.len() as f64;
    assert!((mean - 1.0).abs() < 0.2, "mean was {mean}");
}

#[test]
fn test_kitrng_reproducible_binomial() {
    let mut rng1 = KitRng::new(7);
    let mut rng2 = KitRng::new(7);
    for _ in 0..100 {
        assert_eq!(rng1.binomial(500, 0.3), rng2.binomial(500, 0.3));
    }
}

// ============================================================================
// eigh tests (issue #49: cyclic Jacobi, ascending eigenvalues)
// ============================================================================

#[test]
fn eigh_ascending_values() {
    // np.linalg.eigh(np.array([[2.,1.],[1.,2.]]))[0] == [1.0, 3.0]
    let a = NdArray::new(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let r = a.eigh().unwrap();
    assert!((r.values[0] - 1.0).abs() < 1e-12);
    assert!((r.values[1] - 3.0).abs() < 1e-12);
}

#[test]
fn eigh_3x3_values() {
    // np.linalg.eigh([[4,1,0],[1,3,1],[0,1,2]])[0]
    // == [1.2679491924311228, 3.0, 4.732050807568877]
    let a = NdArray::new(
        vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
        vec![3, 3],
    )
    .unwrap();
    let r = a.eigh().unwrap();
    assert!((r.values[0] - 1.2679491924311228).abs() < 1e-10);
    assert!((r.values[1] - 3.0).abs() < 1e-10);
    assert!((r.values[2] - 4.732050807568877).abs() < 1e-10);
}

#[test]
fn eigh_vectors_reconstruct() {
    let data = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let a = NdArray::new(data.clone(), vec![3, 3]).unwrap();
    let r = a.eigh().unwrap();
    let v = r.vectors.data();
    let n = 3;
    // A v_i = lambda_i v_i
    for i in 0..n {
        for row in 0..n {
            let av: f64 = (0..n).map(|k| data[row * n + k] * v[k * n + i]).sum();
            let lv = r.values[i] * v[row * n + i];
            assert!(
                (av - lv).abs() < 1e-10,
                "A v_{i} != lambda v_{i} at row {row}"
            );
        }
    }
    // V^T V = I
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n).map(|k| v[k * n + i] * v[k * n + j]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn eigh_vector_sign_free() {
    // NumPy column 0 is [-1/sqrt2, 1/sqrt2] or its negative; sign is free.
    let a = NdArray::new(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let r = a.eigh().unwrap();
    let v = r.vectors.data();
    let s = 0.5_f64.sqrt();
    let (x0, x1) = (v[0], v[2]);
    let same = (x0 - (-s)).abs() < 1e-12 && (x1 - s).abs() < 1e-12;
    let negated = (x0 - s).abs() < 1e-12 && (x1 - (-s)).abs() < 1e-12;
    assert!(same || negated);
}

#[test]
fn eigh_not_symmetric_errors() {
    let a = NdArray::new(vec![1.0, 2.0, 0.0, 1.0], vec![2, 2]).unwrap();
    assert!(a.eigh().is_err());
}

#[test]
fn eigh_nan_errors() {
    let a = NdArray::new(vec![1.0, f64::NAN, f64::NAN, 1.0], vec![2, 2]).unwrap();
    assert!(a.eigh().is_err());
    let b = NdArray::new(vec![f64::INFINITY, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    assert!(b.eigh().is_err());
}

#[test]
fn eigh_empty_and_one() {
    let e = NdArray::<f64>::new(vec![], vec![0, 0]).unwrap();
    let r0 = e.eigh().unwrap();
    assert!(r0.values.is_empty());

    let one = NdArray::new(vec![5.0], vec![1, 1]).unwrap();
    let r1 = one.eigh().unwrap();
    assert_eq!(r1.values, vec![5.0]);
    assert_eq!(r1.vectors.data(), &[1.0]);
}

#[test]
fn eigh_diagonal_sorted() {
    let a = NdArray::new(
        vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
        vec![3, 3],
    )
    .unwrap();
    let r = a.eigh().unwrap();
    assert!((r.values[0] - 1.0).abs() < 1e-12);
    assert!((r.values[1] - 2.0).abs() < 1e-12);
    assert!((r.values[2] - 3.0).abs() < 1e-12);
}

#[test]
fn eigh_not_square_errors() {
    let a = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();
    assert!(a.eigh().is_err());
}

// ============================================================================
// svd tests
// ============================================================================

/// U[:, :k] * diag(s) * Vt[:k, :] with k = s.len(), as a row-major m x n matrix.
fn svd_reconstruct(r: &SvdResult, m: usize, n: usize) -> Vec<f64> {
    let k = r.s.len();
    let u = r.u.data();
    let vt = r.vt.data();
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|c| u[i * m + c] * r.s[c] * vt[c * n + j]).sum();
        }
    }
    out
}

/// Asserts that the n x n row-major matrix `data` has M^T M = I within 1e-12.
fn assert_orthonormal(data: &[f64], n: usize) {
    for p in 0..n {
        for q in 0..n {
            let dot: f64 = (0..n).map(|i| data[i * n + p] * data[i * n + q]).sum();
            let want = if p == q { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-12, "M^T M[{p}][{q}] = {dot}");
        }
    }
}

/// Asserts that every entry of `got` is within `tol` of the entry of `want`.
fn assert_close_all(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < tol, "got {g}, want {w}");
    }
}

#[test]
fn svd_zero_column_u_orthonormal() {
    // np.linalg.svd(np.array([[0.,0.],[1.,0.]]))[1] == [1.0, 0.0]; U = [[0,1],[1,0]] up to sign
    let a = NdArray::new(vec![0.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
    let r = a.svd().unwrap();
    assert!((r.s[0] - 1.0).abs() < 1e-12);
    assert!(r.s[1].abs() < 1e-12);
    assert_orthonormal(r.u.data(), 2);
    assert_orthonormal(r.vt.data(), 2);
    assert_close_all(&svd_reconstruct(&r, 2, 2), a.data(), 1e-12);
}

#[test]
fn svd_2x2_values() {
    // np.linalg.svd(np.array([[1.,2.],[3.,4.]]))[1] == [5.464985704219043, 0.36596619062625746]
    let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let r = a.svd().unwrap();
    assert!((r.s[0] - 5.464985704219043).abs() < 1e-12);
    assert!((r.s[1] - 0.36596619062625746).abs() < 1e-12);
    assert_orthonormal(r.u.data(), 2);
    assert_orthonormal(r.vt.data(), 2);
    assert_close_all(&svd_reconstruct(&r, 2, 2), a.data(), 1e-12);
}

#[test]
fn svd_wide_2x3() {
    // np.linalg.svd(np.array([[1.,2.,3.],[4.,5.,6.]]))[1] == [9.508032000695724, 0.7728696356734838]
    let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let r = a.svd().unwrap();
    assert_eq!(r.s.len(), 2);
    assert!((r.s[0] - 9.508032000695724).abs() < 1e-12);
    assert!((r.s[1] - 0.7728696356734838).abs() < 1e-12);
    assert_orthonormal(r.u.data(), 2);
    assert_orthonormal(r.vt.data(), 3);
    assert_close_all(&svd_reconstruct(&r, 2, 3), a.data(), 1e-12);
}

#[test]
fn svd_tall_3x2() {
    // np.linalg.svd(np.array([[1.,2.],[3.,4.],[5.,6.]]))[1] == [9.525518091565107, 0.5143005806586446]
    let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    let r = a.svd().unwrap();
    assert_eq!(r.s.len(), 2);
    assert!((r.s[0] - 9.525518091565107).abs() < 1e-12);
    assert!((r.s[1] - 0.5143005806586446).abs() < 1e-12);
    assert_orthonormal(r.u.data(), 3);
    assert_orthonormal(r.vt.data(), 2);
    assert_close_all(&svd_reconstruct(&r, 3, 2), a.data(), 1e-12);
}

#[test]
fn svd_ill_conditioned() {
    // np.linalg.svd(np.array([[1.,1.],[1.,1.+1e-9]]))[1][1] is about 5e-10.
    // The old A^T A method loses this value; the smaller one must be within 1%.
    let a = NdArray::new(vec![1.0, 1.0, 1.0, 1.0 + 1e-9], vec![2, 2]).unwrap();
    let r = a.svd().unwrap();
    let small = r.s[1];
    assert!(
        (small - 5e-10).abs() <= 0.01 * 5e-10,
        "small singular value {small}"
    );
    assert_close_all(&svd_reconstruct(&r, 2, 2), a.data(), 1e-12);
}

#[test]
fn svd_nan_errors() {
    // np.linalg.svd with NaN or inf raises LinAlgError
    let a = NdArray::new(vec![1.0, f64::NAN, 0.0, 1.0], vec![2, 2]).unwrap();
    assert!(a.svd().is_err());
    let b = NdArray::new(vec![1.0, f64::INFINITY, 0.0, 1.0], vec![2, 2]).unwrap();
    assert!(b.svd().is_err());
}

#[test]
fn svd_zero_matrix() {
    // np.linalg.svd(np.zeros((2,2)))[1] == [0.0, 0.0]
    let a = NdArray::new(vec![0.0; 4], vec![2, 2]).unwrap();
    let r = a.svd().unwrap();
    assert!(r.s.iter().all(|s| s.abs() < 1e-12));
    assert_orthonormal(r.u.data(), 2);
    assert_orthonormal(r.vt.data(), 2);
    assert_close_all(&svd_reconstruct(&r, 2, 2), a.data(), 1e-12);
}
