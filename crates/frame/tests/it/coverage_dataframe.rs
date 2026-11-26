//! Coverage tests for `frame::frame::dataframe` (`crates/frame/src/frame/dataframe.rs`).
//! Tests only: every expected value is written out by hand from the current behavior of the code.

use frame::frame::{DataFrame, FrameError, Series, Value};

/// `a = [1, 2, 3]`, `b = [10, 20, 30]`
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
}

#[test]
fn insert_into_empty_frame_sets_row_count() {
    let mut df = DataFrame::new();
    df.insert("a", Series::new(vec![1, 2, 3])).unwrap();
    assert_eq!(df.nrows(), 3);
}

#[test]
fn iloc_row_rejects_position_past_the_end() {
    assert!(matches!(
        df3().iloc_row(3),
        Err(FrameError::IndexOutOfBounds {
            index: 3,
            length: 3
        })
    ));
    assert_eq!(df3().iloc_row(2).unwrap()["b"], Value::Int(30));
}

#[test]
fn loc_row_returns_row_by_label() {
    let row = df3().loc_row(&Value::Int(1)).unwrap();
    assert_eq!(row["a"], Value::Int(2));
    assert_eq!(row["b"], Value::Int(20));
    assert!(matches!(
        df3().loc_row(&Value::Int(9)),
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
    assert_eq!(DataFrame::concat(&[]).unwrap().nrows(), 0);
}

#[test]
fn concat_rejects_different_columns() {
    let other = DataFrame::from_columns(vec![("z", Series::new(vec![1, 2, 3]))]).unwrap();
    let res = DataFrame::concat(&[&df3(), &other]);
    assert!(matches!(
        res,
        Err(FrameError::InvalidOperation(ref m)) if m == "all DataFrames must have the same columns"
    ));
}

#[test]
fn from_records_of_nothing_is_empty() {
    assert_eq!(DataFrame::from_records(&[]).unwrap().nrows(), 0);
}

#[test]
fn corr_without_numeric_columns_is_empty() {
    let df = DataFrame::from_columns(vec![("s", Series::new(vec!["x", "y"]))]).unwrap();
    assert_eq!(df.corr().unwrap().nrows(), 0);
}

#[test]
fn default_is_empty() {
    assert_eq!(DataFrame::default().nrows(), 0);
}

#[test]
fn pivot_with_nan_in_index_column() {
    // Index keys: 1.0 (rows 0 and 2) and NaN (row 1). Column keys: "x" and "y".
    // Row 2 overwrites row 0 for ("1.0", "x"), so "x" at index 1.0 is 3.0.
    let df = DataFrame::from_columns(vec![
        ("i", Series::new(vec![1.0, f64::NAN, 1.0])),
        ("c", Series::new(vec!["x", "y", "x"])),
        ("v", Series::new(vec![2.0, 5.0, 3.0])),
    ])
    .unwrap();

    let out = df.pivot("i", "c", "v").unwrap();

    assert_eq!(out.columns(), ["i", "x", "y"]);
    assert_eq!(
        out.col("i").unwrap().values(),
        &[Value::Float(1.0), Value::Float(f64::NAN)]
    );
    assert_eq!(
        out.col("x").unwrap().values(),
        &[Value::Float(3.0), Value::Null]
    );
    assert_eq!(
        out.col("y").unwrap().values(),
        &[Value::Null, Value::Float(5.0)]
    );
}
