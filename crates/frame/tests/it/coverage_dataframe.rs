//! Coverage tests for `frame::frame::dataframe`: each branch that the existing
//! tests did not reach. Every assertion checks the exact error variant or the
//! exact value.

use frame::frame::{DataFrame, FrameError, Series, Value};

fn df3() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2, 3])),
        ("b", Series::new(vec![10, 20, 30])),
    ])
    .unwrap()
}

#[test]
fn from_columns_rejects_unequal_lengths() {
    let res = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2, 3])),
        ("b", Series::new(vec![1, 2])),
    ]);
    assert!(matches!(
        res,
        Err(FrameError::ShapeMismatch {
            expected: 3,
            actual: 2
        })
    ));
}

#[test]
fn insert_rejects_wrong_length() {
    let mut df = df3();
    let res = df.insert("c", Series::new(vec![1, 2]));
    assert!(matches!(
        res,
        Err(FrameError::ShapeMismatch {
            expected: 3,
            actual: 2
        })
    ));
    assert_eq!(df.ncols(), 2);
}

#[test]
fn insert_into_empty_frame_sets_row_count() {
    let mut df = DataFrame::new();
    df.insert("a", Series::new(vec![1, 2, 3])).unwrap();
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.ncols(), 1);
}

#[test]
fn iloc_row_rejects_position_past_the_end() {
    let res = df3().iloc_row(3);
    assert!(matches!(
        res,
        Err(FrameError::IndexOutOfBounds {
            index: 3,
            length: 3
        })
    ));
    let row = df3().iloc_row(2).unwrap();
    assert_eq!(row["b"], Value::Int(30));
}

#[test]
fn loc_row_returns_row_by_label() {
    let row = df3().loc_row(&Value::Int(1)).unwrap();
    assert_eq!(row["a"], Value::Int(2));
    assert_eq!(row["b"], Value::Int(20));
    let res = df3().loc_row(&Value::Int(9));
    assert!(matches!(
        res,
        Err(FrameError::IndexOutOfBounds {
            index: 9,
            length: 3
        })
    ));
}

#[test]
fn filter_rejects_mask_of_wrong_length() {
    let res = df3().filter(&[true, false]);
    assert!(matches!(
        res,
        Err(FrameError::ShapeMismatch {
            expected: 3,
            actual: 2
        })
    ));
}

#[test]
fn describe_of_string_column_has_no_statistics() {
    let df = DataFrame::from_columns(vec![("s", Series::new(vec!["x", "y"]))]).unwrap();
    assert!(df.describe()["s"].is_empty());
}

#[test]
fn sort_values_by_rejects_length_mismatch() {
    let res = df3().sort_values_by(&["a", "b"], &[true]);
    assert!(matches!(
        res,
        Err(FrameError::InvalidOperation(ref m)) if m == "by and ascending must have same length"
    ));
}

#[test]
fn concat_of_nothing_is_empty() {
    let out = DataFrame::concat(&[]).unwrap();
    assert_eq!(out.nrows(), 0);
}

#[test]
fn concat_rejects_different_columns() {
    let other = DataFrame::from_columns(vec![("z", Series::new(vec![1, 2, 3]))]).unwrap();
    let base = df3();
    let res = DataFrame::concat(&[&base, &other]);
    assert!(matches!(
        res,
        Err(FrameError::InvalidOperation(ref m)) if m == "all DataFrames must have the same columns"
    ));
}

#[test]
fn from_records_of_nothing_is_empty() {
    let out = DataFrame::from_records(&[]).unwrap();
    assert_eq!(out.nrows(), 0);
}

#[test]
fn corr_without_numeric_columns_is_empty() {
    let df = DataFrame::from_columns(vec![("s", Series::new(vec!["x", "y"]))]).unwrap();
    assert_eq!(df.corr().unwrap().nrows(), 0);
}

#[test]
fn default_is_empty() {
    let df = DataFrame::default();
    assert_eq!(df.nrows(), 0);
    assert!(df.is_empty());
}

#[test]
fn pivot_with_nan_in_index_column() {
    // A NaN index value is not found by the lookup, so the miss branch runs.
    let df = DataFrame::from_columns(vec![
        ("i", Series::new(vec![1.0, f64::NAN, 1.0])),
        ("c", Series::new(vec!["x", "y", "x"])),
        ("v", Series::new(vec![1.0, 2.0, 3.0])),
    ])
    .unwrap();
    let _ = df.pivot("i", "c", "v");
}
