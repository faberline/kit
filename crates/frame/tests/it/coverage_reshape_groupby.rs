//! Coverage tests for the groupby, pivot and wide-to-long paths of
//! `frame::frame/ops`. Tests only: every expected value was written by hand
//! from the current behavior of the code.

use frame::frame::ops::reshape::{pivot_table, wide_to_long};
use frame::frame::ops::AggFunc;
use frame::frame::{DataFrame, Series, Value};
use std::collections::HashMap;

fn column(df: &DataFrame, name: &str) -> Vec<Value> {
    df.get(name).unwrap().values().to_vec()
}

/// Keys `a, a, b, c, c` with values `1, 3, 5, null, null`.
/// Group `a` has two numbers, group `b` one, group `c` none.
fn sample() -> DataFrame {
    DataFrame::from_columns(vec![
        ("k", Series::new(vec!["a", "a", "b", "c", "c"])),
        (
            "v",
            Series::new(vec![Some(1.0), Some(3.0), Some(5.0), None, None]),
        ),
    ])
    .unwrap()
}

#[test]
fn var_group_with_one_value_is_null() {
    let df = sample();
    let g = df.groupby(&["k"]).unwrap();
    assert_eq!(
        column(&g.var().unwrap(), "v"),
        vec![Value::Float(2.0), Value::Null, Value::Null]
    );
}

#[test]
fn std_group_with_one_value_is_null() {
    let df = sample();
    let g = df.groupby(&["k"]).unwrap();
    assert_eq!(
        column(&g.std().unwrap(), "v"),
        vec![
            Value::Float(std::f64::consts::SQRT_2),
            Value::Null,
            Value::Null
        ]
    );
}

#[test]
fn median_group_without_numbers_is_null() {
    let df = sample();
    let g = df.groupby(&["k"]).unwrap();
    assert_eq!(
        column(&g.median().unwrap(), "v"),
        vec![Value::Float(2.0), Value::Float(5.0), Value::Null]
    );
}

#[test]
fn agg_mean_group_without_numbers_is_null() {
    let df = sample();
    let g = df.groupby(&["k"]).unwrap();
    let mut aggs = HashMap::new();
    aggs.insert("v", "mean");
    let r = g.agg(aggs).unwrap();
    assert_eq!(r.columns(), ["k", "v"]);
    assert_eq!(
        column(&r, "k"),
        vec![Value::from("a"), Value::from("b"), Value::from("c")]
    );
    assert_eq!(
        column(&r, "v"),
        vec![Value::Float(2.0), Value::Float(5.0), Value::Null]
    );
}

#[test]
fn agg_median_by_name() {
    let df = sample();
    let g = df.groupby(&["k"]).unwrap();
    let mut aggs = HashMap::new();
    aggs.insert("v", "median");
    let r = g.agg(aggs).unwrap();
    assert_eq!(
        column(&r, "v"),
        vec![Value::Float(2.0), Value::Float(5.0), Value::Null]
    );
}

#[test]
fn pivot_table_null_value_cell_stays_null() {
    let p = DataFrame::from_columns(vec![
        ("r", Series::new(vec!["x", "x", "y"])),
        ("c", Series::new(vec!["p", "q", "p"])),
        ("v", Series::new(vec![Some(1.0), None, Some(4.0)])),
    ])
    .unwrap();
    let t = pivot_table(&p, "v", "r", "c", AggFunc::Sum).unwrap();
    assert_eq!(t.columns(), ["r", "p", "q"]);
    assert_eq!(column(&t, "r"), vec![Value::from("x"), Value::from("y")]);
    assert_eq!(column(&t, "p"), vec![Value::Float(1.0), Value::Float(4.0)]);
    assert_eq!(column(&t, "q"), vec![Value::Null, Value::Null]);
}

#[test]
fn wide_to_long_no_matching_columns_is_error() {
    let w = DataFrame::from_columns(vec![
        ("id", Series::new(vec![1i64, 2])),
        ("a", Series::new(vec![1.0, 2.0])),
    ])
    .unwrap();
    let err = wide_to_long(&w, &["score"], "id", "t", "_").unwrap_err();
    assert_eq!(
        err.to_string(),
        "invalid operation: no matching columns found for stub names"
    );
}

#[test]
fn wide_to_long_empty_stub_names_is_error() {
    let w = DataFrame::from_columns(vec![
        ("id", Series::new(vec![1i64, 2])),
        ("a", Series::new(vec![1.0, 2.0])),
    ])
    .unwrap();
    let err = wide_to_long(&w, &[], "id", "t", "_").unwrap_err();
    assert_eq!(
        err.to_string(),
        "invalid operation: stub_names must not be empty"
    );
}
