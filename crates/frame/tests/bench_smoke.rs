//! Smoke tests for the operations that `benches/frame.rs` measures.
//! Each test runs one benched operation on a tiny input and checks the result.

use frame::frame::io::read_csv;
use frame::frame::ops::JoinType;
use frame::frame::{DataFrame, Series};
use std::collections::HashMap;
use std::io::Write;

#[test]
fn groupby_sum_smoke() {
    let df = DataFrame::from_columns(vec![
        ("key", Series::new(vec![1i64, 1, 2])),
        ("val", Series::new(vec![1.0f64, 2.0, 5.0])),
    ])
    .unwrap();
    let mut aggs = HashMap::new();
    aggs.insert("val", "sum");
    let out = df.groupby(&["key"]).unwrap().agg(aggs).unwrap();
    assert_eq!(out.nrows(), 2);
    assert_eq!(out.ncols(), 2);
}

#[test]
fn join_inner_smoke() {
    let left = DataFrame::from_columns(vec![
        ("key", Series::new(vec![1i64, 2, 3])),
        ("val", Series::new(vec![1.0f64, 2.0, 3.0])),
    ])
    .unwrap();
    let right = DataFrame::from_columns(vec![
        ("key", Series::new(vec![2i64, 3, 4])),
        ("rval", Series::new(vec![20.0f64, 30.0, 40.0])),
    ])
    .unwrap();
    let out = left.join(&right, &["key"], JoinType::Inner).unwrap();
    assert_eq!(out.nrows(), 2);
    assert_eq!(out.ncols(), 3);
}

#[test]
fn join_missing_column_is_error() {
    let left = DataFrame::from_columns(vec![("key", Series::new(vec![1i64]))]).unwrap();
    let right = DataFrame::from_columns(vec![("key", Series::new(vec![1i64]))]).unwrap();
    assert!(left.join(&right, &["nope"], JoinType::Inner).is_err());
}

#[test]
fn read_csv_smoke() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(b"a,b,c\n1,2.5,name1\n2,3.5,name2\n")
        .unwrap();
    file.flush().unwrap();
    let out = read_csv(file.path()).unwrap();
    assert_eq!(out.nrows(), 2);
    assert_eq!(out.ncols(), 3);
}
