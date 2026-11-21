//! Coverage tests for the indexing, dtype, ops, shape, slice, view and window files of arraykit.

use arraykit::array::indexing::AxisIndex;
use arraykit::array::{
    ArrayElement, ArrayError, AxisSlice, DType, NdArray, RangeElement, Shape, SliceInfo,
};

/// The error of a result that must fail. Panics when the result is Ok.
fn expect_err<T>(result: Result<T, ArrayError>) -> ArrayError {
    match result {
        Err(error) => error,
        Ok(_) => panic!("expected an error, got Ok"),
    }
}

fn matrix() -> NdArray<i32> {
    // [[1, 2, 3], [4, 5, 6]]
    NdArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap()
}

// ---- dtype.rs ----

#[test]
fn dtype_numpy_str_all_variants() {
    assert_eq!(DType::Float32.numpy_str(), "float32");
    assert_eq!(DType::Float64.numpy_str(), "float64");
    assert_eq!(DType::Int32.numpy_str(), "int32");
    assert_eq!(DType::Int64.numpy_str(), "int64");
    assert_eq!(DType::Bool.numpy_str(), "bool");
}

#[test]
fn array_element_zero_and_one() {
    assert_eq!(<f32 as ArrayElement>::zero(), 0.0);
    assert_eq!(<f32 as ArrayElement>::one(), 1.0);
    assert_eq!(<i64 as ArrayElement>::one(), 1);
    assert_eq!(<i64 as ArrayElement>::zero(), 0);
    assert!(!<bool as ArrayElement>::zero());
    assert!(<bool as ArrayElement>::one());
}

#[test]
fn range_len_negative_step_ints() {
    // 5, 3, 1
    assert_eq!(<i32 as RangeElement>::range_len(5, 0, -2), 3);
    assert_eq!(<i64 as RangeElement>::range_len(5, 0, -2), 3);
    // wrong direction: no element
    assert_eq!(<i32 as RangeElement>::range_len(0, 5, -1), 0);
    assert_eq!(<i64 as RangeElement>::range_len(0, 5, -1), 0);
}

#[test]
fn range_len_float_wrong_direction() {
    assert_eq!(<f32 as RangeElement>::range_len(0.0, 5.0, -1.0), 0);
    assert_eq!(<f64 as RangeElement>::range_len(0.0, 5.0, -1.0), 0);
}

// ---- ops.rs ----

#[test]
fn ref_sub_operator() {
    let x = NdArray::new(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2]).unwrap();
    let y = NdArray::new(vec![2.0, 4.0, 5.0, 3.0], vec![2, 2]).unwrap();
    assert_eq!((&x - &y).unwrap().data(), &[4.0, 4.0, 5.0, 9.0]);
}

#[test]
fn ref_mul_operator() {
    let x = NdArray::new(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2]).unwrap();
    let y = NdArray::new(vec![2.0, 4.0, 5.0, 3.0], vec![2, 2]).unwrap();
    assert_eq!((&x * &y).unwrap().data(), &[12.0, 32.0, 50.0, 36.0]);
}

#[test]
fn ref_div_operator() {
    let x = NdArray::new(vec![6.0, 8.0, 10.0, 12.0], vec![2, 2]).unwrap();
    let y = NdArray::new(vec![2.0, 4.0, 5.0, 3.0], vec![2, 2]).unwrap();
    assert_eq!((&x / &y).unwrap().data(), &[3.0, 2.0, 2.0, 4.0]);
}

// ---- shape.rs ----

#[test]
fn shape_with_strides_length_mismatch() {
    let error = expect_err(Shape::with_strides(vec![2, 3], vec![1]));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![1]
        }
    );
}

#[test]
fn shape_new_empty_has_no_strides() {
    let shape = Shape::new(Vec::new());
    assert_eq!(shape.ndim(), 0);
    assert!(shape.strides().is_empty());
}

#[test]
fn shape_broadcast_pads_missing_leading_dims() {
    let a = Shape::broadcast(&Shape::new(vec![3]), &Shape::new(vec![2, 3])).unwrap();
    assert_eq!(a.dims(), &[2, 3]);
    let b = Shape::broadcast(&Shape::new(vec![2, 3]), &Shape::new(vec![3])).unwrap();
    assert_eq!(b.dims(), &[2, 3]);
}

#[test]
fn shape_broadcast_takes_other_dim_when_one() {
    let a = Shape::broadcast(&Shape::new(vec![1, 3]), &Shape::new(vec![2, 3])).unwrap();
    assert_eq!(a.dims(), &[2, 3]);
    let b = Shape::broadcast(&Shape::new(vec![2, 1]), &Shape::new(vec![2, 3])).unwrap();
    assert_eq!(b.dims(), &[2, 3]);
}

#[test]
fn shape_broadcast_strides_target_has_fewer_dims() {
    let error = expect_err(Shape::new(vec![2, 3]).broadcast_strides(&Shape::new(vec![3])));
    assert_eq!(
        error,
        ArrayError::BroadcastError {
            left: vec![2, 3],
            right: vec![3]
        }
    );
}

#[test]
fn shape_broadcast_strides_incompatible_dim() {
    let error = expect_err(Shape::new(vec![2]).broadcast_strides(&Shape::new(vec![3])));
    assert_eq!(
        error,
        ArrayError::BroadcastError {
            left: vec![2],
            right: vec![3]
        }
    );
}

#[test]
fn shape_broadcast_strides_ok() {
    // A new leading axis gets stride 0; a length-1 axis gets stride 0.
    assert_eq!(
        Shape::new(vec![3])
            .broadcast_strides(&Shape::new(vec![2, 3]))
            .unwrap(),
        vec![0, 1]
    );
    assert_eq!(
        Shape::new(vec![1, 3])
            .broadcast_strides(&Shape::new(vec![2, 3]))
            .unwrap(),
        vec![0, 1]
    );
}

#[test]
fn shape_from_slice() {
    let shape = Shape::from(&[2usize, 3][..]);
    assert_eq!(shape.dims(), &[2, 3]);
}

// ---- slice.rs ----

#[test]
fn normalize_zero_step_is_error() {
    let error = expect_err(
        AxisSlice::Slice {
            start: None,
            stop: None,
            step: Some(0),
        }
        .normalize(5),
    );
    assert_eq!(
        error,
        ArrayError::InvalidSlice {
            start: 0,
            stop: 5,
            step: 0,
            len: 5
        }
    );
}

#[test]
fn normalize_empty_forward_range() {
    let slice = AxisSlice::range(3, 1).normalize(5).unwrap();
    assert_eq!(slice.output_len, 0);
}

#[test]
fn normalize_empty_backward_range() {
    let slice = AxisSlice::range_step(1, 3, -1).normalize(5).unwrap();
    assert_eq!(slice.output_len, 0);
}

#[test]
fn slice_info_output_shape_keeps_trailing_dims() {
    let info = SliceInfo::new(vec![AxisSlice::range(0, 2)]);
    assert_eq!(info.output_shape(&[3, 4]).unwrap(), vec![2, 4]);
}

#[test]
fn slice_info_normalize_all_pads_full_slices() {
    let info = SliceInfo::new(vec![AxisSlice::range(0, 2)]);
    let normalized = info.normalize_all(&[3, 4]).unwrap();
    assert_eq!(normalized.len(), 2);
    let second = normalized[1];
    assert_eq!(second.start, 0);
    assert_eq!(second.stop, 4);
    assert_eq!(second.step, 1);
    assert_eq!(second.output_len, 4);
    assert!(!second.is_index);
}

// ---- view.rs ----

#[test]
fn view_data_ptr_of_empty_view() {
    let empty = NdArray::<f64>::new(Vec::new(), vec![0]).unwrap();
    let view = empty.view();
    assert_eq!(view.size(), 0);
    assert!(!view.data_ptr().is_null());
}

#[test]
fn view_transpose_axes_duplicate_axis() {
    let a = matrix();
    let error = expect_err(a.view().transpose_axes(&[0, 0]));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![0, 1],
            got: vec![0, 0]
        }
    );
}

// ---- window.rs ----

#[test]
fn squeeze_all_ones_gives_one_element_vector() {
    let a = NdArray::new(vec![7.0], vec![1, 1]).unwrap();
    let squeezed = a.squeeze();
    assert_eq!(squeezed.dims(), &[1]);
    assert_eq!(squeezed.data(), &[7.0]);
}

#[test]
fn roll_empty_array_is_unchanged() {
    let empty = NdArray::<f64>::new(Vec::new(), vec![0]).unwrap();
    let rolled = empty.roll(3);
    assert_eq!(rolled.dims(), &[0]);
    assert!(rolled.data().is_empty());
}

#[test]
fn interp_edges_and_empty_inputs() {
    // Left of the range gives fp[0], right of it gives the last fp, inside it is linear.
    let values = NdArray::<f64>::interp(&[-1.0, 5.0, 1.0], &[0.0, 1.0, 2.0], &[10.0, 20.0, 30.0]);
    assert_eq!(values, vec![10.0, 30.0, 20.0]);
    // No sample points: the one value is NaN.
    let empty = NdArray::<f64>::interp(&[1.0], &[], &[]);
    assert_eq!(empty.len(), 1);
    assert!(empty[0].is_nan());
}

#[test]
fn rot90_zero_and_full_turn_return_clone() {
    let m = NdArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    for k in [0, 4] {
        let turned = m.rot90(k).unwrap();
        assert_eq!(turned.dims(), &[2, 2]);
        assert_eq!(turned.data(), &[1, 2, 3, 4]);
    }
}

// ---- indexing.rs ----

#[test]
fn fancy_index_zero_dim_is_error() {
    // A shape with no dimensions holds one element.
    let scalar = NdArray::new(vec![7.0], Vec::<usize>::new()).unwrap();
    let error = expect_err(scalar.fancy_index(&[0]));
    assert_eq!(
        error,
        ArrayError::InvalidOperation("cannot index 0-dimensional array".into())
    );
}

#[test]
fn bool_index_wrong_mask_length() {
    let a = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let error = expect_err(a.bool_index(&[true, false]));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![3],
            got: vec![2]
        }
    );
}

#[test]
fn bool_index_array_wrong_size() {
    let a = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let mask = NdArray::new(vec![true, false], vec![2]).unwrap();
    let error = expect_err(a.bool_index_array(&mask));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![3],
            got: vec![2]
        }
    );
}

#[test]
fn bool_set_wrong_mask_length() {
    let mut v = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let error = expect_err(v.bool_set(&[true], 9));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![3],
            got: vec![1]
        }
    );
}

#[test]
fn fancy_set_length_mismatch() {
    let mut v = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let error = expect_err(v.fancy_set(&[0, 1], &[9]));
    assert_eq!(
        error,
        ArrayError::InvalidOperation("indices and values must have same length".into())
    );
}

#[test]
fn fancy_set_out_of_bounds() {
    let mut v = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    let error = expect_err(v.fancy_set(&[5], &[9]));
    assert_eq!(
        error,
        ArrayError::IndexOutOfBounds {
            index: vec![5],
            shape: vec![3]
        }
    );
}

#[test]
fn fancy_set_one_dim_ok_and_two_dim_error() {
    let mut v = NdArray::new(vec![1, 2, 3], vec![3]).unwrap();
    v.fancy_set(&[0, 2], &[9, 8]).unwrap();
    assert_eq!(v.data(), &[9, 2, 8]);

    let mut m = matrix();
    let error = expect_err(m.fancy_set(&[0], &[9]));
    assert_eq!(
        error,
        ArrayError::InvalidOperation("fancy_set only supports 1D arrays currently".into())
    );
}

#[test]
fn advanced_index_empty_returns_clone() {
    let a = matrix();
    let copy = a.advanced_index(&[]).unwrap();
    assert_eq!(copy.dims(), &[2, 3]);
    assert_eq!(copy.data(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn advanced_index_too_many_indices() {
    let a = matrix();
    let error = expect_err(a.advanced_index(&[AxisIndex::Full, AxisIndex::Full, AxisIndex::Full]));
    assert_eq!(error, ArrayError::InvalidAxis { axis: 3, ndim: 2 });
}

#[test]
fn advanced_index_fancy_out_of_bounds() {
    let a = matrix();
    let error = expect_err(a.advanced_index(&[AxisIndex::Fancy(vec![2]), AxisIndex::Full]));
    assert_eq!(
        error,
        ArrayError::IndexOutOfBounds {
            index: vec![2],
            shape: vec![2, 3]
        }
    );
}

#[test]
fn advanced_index_bool_wrong_length() {
    let a = matrix();
    let error = expect_err(a.advanced_index(&[AxisIndex::Bool(vec![true])]));
    assert_eq!(
        error,
        ArrayError::ShapeMismatch {
            expected: vec![2],
            got: vec![1]
        }
    );
}

#[test]
fn advanced_index_bool_selects_rows() {
    let a = matrix();
    let rows = a
        .advanced_index(&[AxisIndex::Bool(vec![false, true]), AxisIndex::Full])
        .unwrap();
    assert_eq!(rows.dims(), &[1, 3]);
    assert_eq!(rows.data(), &[4, 5, 6]);
}

#[test]
fn advanced_index_single_out_of_bounds() {
    let a = matrix();
    let error = expect_err(a.advanced_index(&[AxisIndex::Single(2)]));
    assert_eq!(
        error,
        ArrayError::IndexOutOfBounds {
            index: vec![2],
            shape: vec![2, 3]
        }
    );
}

#[test]
fn advanced_index_single_keeps_trailing_axis() {
    let a = matrix();
    let row = a.advanced_index(&[AxisIndex::Single(1)]).unwrap();
    assert_eq!(row.dims(), &[3]);
    assert_eq!(row.data(), &[4, 5, 6]);
}

#[test]
fn advanced_index_all_single_gives_scalar() {
    let a = matrix();
    let scalar = a
        .advanced_index(&[AxisIndex::Single(0), AxisIndex::Single(1)])
        .unwrap();
    assert_eq!(scalar.dims(), &[1]);
    assert_eq!(scalar.data(), &[2]);
}

#[test]
fn advanced_index_fancy_and_full() {
    let a = matrix();
    let picked = a
        .advanced_index(&[AxisIndex::Full, AxisIndex::Fancy(vec![2, 0])])
        .unwrap();
    assert_eq!(picked.dims(), &[2, 2]);
    assert_eq!(picked.data(), &[3, 1, 6, 4]);
}

#[test]
fn take_invalid_axis() {
    let a = matrix();
    let error = expect_err(a.take(&[0], 2));
    assert_eq!(error, ArrayError::InvalidAxis { axis: 2, ndim: 2 });
}

#[test]
fn take_out_of_bounds() {
    let a = matrix();
    let error = expect_err(a.take(&[3], 1));
    assert_eq!(
        error,
        ArrayError::IndexOutOfBounds {
            index: vec![3],
            shape: vec![2, 3]
        }
    );
}

#[test]
fn take_columns() {
    let a = matrix();
    let columns = a.take(&[2, 0], 1).unwrap();
    assert_eq!(columns.dims(), &[2, 2]);
    assert_eq!(columns.data(), &[3, 1, 6, 4]);
}
