//! Reference-value tests. Expected values come from pandas 2.2 / NumPy 1.26;
//! the expression that made each value is in the comment above it. CI does not run Python.

use frame::frame::io::{read_csv, write_csv};
use frame::frame::ops::JoinType;
use frame::frame::{DataFrame, Series, Value};
use std::collections::HashMap;
use std::io::Write;
use tempfile::NamedTempFile;

/// Maps the values of a series to `Some(i)` for `Value::Int(i)` and `None` for `Value::Null`.
fn ints(s: &Series) -> Vec<Option<i64>> {
    s.values()
        .iter()
        .map(|v| match v {
            Value::Int(i) => Some(*i),
            Value::Null => None,
            other => panic!("expected Int or Null, got {:?}", other),
        })
        .collect()
}

/// Sorts a list of optional ints. `None` sorts before `Some`.
fn sorted(mut v: Vec<Option<i64>>) -> Vec<Option<i64>> {
    v.sort();
    v
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Left and right frames shared by the join tests.
fn join_frames() -> (DataFrame, DataFrame) {
    let l = DataFrame::from_columns(vec![
        ("id", Series::new(vec![1i64, 2, 3])),
        ("lv", Series::new(vec![10i64, 20, 30])),
    ])
    .unwrap();
    let r = DataFrame::from_columns(vec![
        ("id", Series::new(vec![2i64, 3, 3, 4])),
        ("rv", Series::new(vec![200i64, 300, 301, 400])),
    ])
    .unwrap();
    (l, r)
}

// pandas 2.2: l.merge(r, on='id', how='right')
// -> id [2, 3, 3, 4]; lv is NaN for id 4 (no left row).
#[test]
fn join_right_keeps_right_only_key() {
    let (l, r) = join_frames();
    let out = l.join(&r, &["id"], JoinType::Right).unwrap();

    assert_eq!(out.nrows(), 4);
    assert_eq!(
        sorted(ints(out.get("id").unwrap())),
        vec![Some(2), Some(3), Some(3), Some(4)]
    );

    let ids = ints(out.get("id").unwrap());
    let lv = out.get("lv").unwrap().values();
    let row4 = ids.iter().position(|&x| x == Some(4)).unwrap();
    assert_eq!(lv[row4], Value::Null);
}

// pandas 2.2: l.merge(r, on='id', how='outer')
// -> id [1, 2, 3, 3, 4], no null id.
#[test]
fn join_outer_keeps_both_sides() {
    let (l, r) = join_frames();
    let out = l.join(&r, &["id"], JoinType::Outer).unwrap();

    assert_eq!(out.nrows(), 5);
    assert_eq!(
        sorted(ints(out.get("id").unwrap())),
        vec![Some(1), Some(2), Some(3), Some(3), Some(4)]
    );
}

// pandas 2.2: l.merge(r, on='k') with l.k int64 [1, 2, 3] and r.k float64 [2.0, 3.0]
// -> the int keys 2 and 3 match the float keys 2.0 and 3.0; 2 rows.
#[test]
fn join_int_key_matches_float_key() {
    let l = DataFrame::from_columns(vec![("k", Series::new(vec![1i64, 2, 3]))]).unwrap();
    let r = DataFrame::from_columns(vec![
        ("k", Series::new(vec![2.0f64, 3.0])),
        ("rv", Series::new(vec![5i64, 6])),
    ])
    .unwrap();
    let out = l.join(&r, &["k"], JoinType::Inner).unwrap();

    assert_eq!(out.nrows(), 2);
    // Right-side non-key columns get the `_right` suffix in join output.
    assert_eq!(
        sorted(ints(out.get("rv_right").unwrap())),
        vec![Some(5), Some(6)]
    );
}

// pandas 2.2: pd.Series([1, None, 3, 5]).describe()
// -> count 3.0, mean 3.0, std 2.0, min 1.0, 25% 2.0, 50% 3.0, 75% 4.0, max 5.0.
#[test]
fn describe_skips_nulls() {
    let df = DataFrame::from_columns(vec![(
        "x",
        Series::new(vec![
            Value::Float(1.0),
            Value::Null,
            Value::Float(3.0),
            Value::Float(5.0),
        ]),
    )])
    .unwrap();
    let d = df.describe();
    let x = &d["x"];

    let expected = [
        ("count", 3.0),
        ("mean", 3.0),
        ("std", 2.0),
        ("min", 1.0),
        ("25%", 2.0),
        ("50%", 3.0),
        ("75%", 4.0),
        ("max", 5.0),
    ];
    for (key, want) in expected {
        let got = x[key];
        assert!(
            close(got, want),
            "describe {}: got {}, want {}",
            key,
            got,
            want
        );
    }

    // pandas 2.2: pd.Series([1, None, 3, 5]).sum() -> 9.0
    let s = df.get("x").unwrap();
    assert!(close(s.sum().unwrap(), 9.0));
}

// pandas 2.2 (skipna=True): pd.Series([1, None, 5]).sum() -> 6.0, .mean() -> 3.0
#[test]
fn series_sum_and_mean_skip_nulls() {
    let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(5)]);
    assert!(close(s.sum().unwrap(), 6.0));
    assert!(close(s.mean().unwrap(), 3.0));
}

// pandas 2.2: df.groupby('g')['v'].mean() with g = [a, a, b, NaN], v = [1, NaN, 4, 9]
// -> a = 1.0, b = 4.0; the null key row is dropped.
#[test]
fn groupby_mean_skips_nulls_and_drops_null_key() {
    let df = DataFrame::from_columns(vec![
        (
            "g",
            Series::new(vec![
                Value::from("a"),
                Value::from("a"),
                Value::from("b"),
                Value::Null,
            ]),
        ),
        (
            "v",
            Series::new(vec![
                Value::Float(1.0),
                Value::Null,
                Value::Float(4.0),
                Value::Float(9.0),
            ]),
        ),
    ])
    .unwrap();
    let out = df.groupby(&["g"]).unwrap().mean().unwrap();

    assert_eq!(out.nrows(), 2);
    let gs = out.get("g").unwrap().values();
    let vs = out.get("v").unwrap().values();
    let mut means: HashMap<String, f64> = HashMap::new();
    for (g, v) in gs.iter().zip(vs.iter()) {
        means.insert(g.as_str().unwrap().to_string(), v.as_float().unwrap());
    }
    assert!(close(means["a"], 1.0));
    assert!(close(means["b"], 4.0));
}

// pandas 2.2: df.groupby('g')['v'].agg('std') and .agg('median')
// a: [1, 2, 4] -> std 1.5275252316519468, median 2.0
// b: [10, 20]  -> std 7.0710678118654755, median 15.0
#[test]
fn groupby_agg_std_median_and_unknown_name() {
    let df = DataFrame::from_columns(vec![
        ("g", Series::new(vec!["a", "a", "a", "b", "b"])),
        ("v", Series::new(vec![1.0f64, 2.0, 4.0, 10.0, 20.0])),
    ])
    .unwrap();
    let gb = df.groupby(&["g"]).unwrap();

    let std_out = gb.agg(HashMap::from([("v", "std")])).unwrap();
    let median_out = gb.agg(HashMap::from([("v", "median")])).unwrap();

    let key_col = std_out.get("g").unwrap().values().to_vec();
    let std_col = std_out.get("v").unwrap().values().to_vec();
    let med_col = median_out.get("v").unwrap().values().to_vec();
    let a_row = key_col
        .iter()
        .position(|v| v.as_str() == Some("a"))
        .unwrap();
    let b_row = key_col
        .iter()
        .position(|v| v.as_str() == Some("b"))
        .unwrap();

    assert!(close(
        std_col[a_row].as_float().unwrap(),
        1.5275252316519468
    ));
    assert!(close(
        std_col[b_row].as_float().unwrap(),
        7.0710678118654755
    ));
    assert!(close(med_col[a_row].as_float().unwrap(), 2.0));
    assert!(close(med_col[b_row].as_float().unwrap(), 15.0));

    // An unknown aggregation name is an error.
    assert!(gb.agg(HashMap::from([("v", "nope")])).is_err());
}

// pandas 2.2: pd.Series([1, 1, 2]).rank() -> [1.5, 1.5, 3.0] (method='average')
#[test]
fn series_rank_average_ties() {
    let s = Series::new(vec![1i64, 1, 2]);
    assert_eq!(
        s.rank().values(),
        &[Value::Float(1.5), Value::Float(1.5), Value::Float(3.0)]
    );
}

// pandas 2.2: df.sort_values('x') with x = [3, NaN, 1] -> [1, 3, NaN]
// ascending=False keeps NaN last -> [3, 1, NaN]
#[test]
fn sort_values_nulls_last_ascending() {
    let df = DataFrame::from_columns(vec![(
        "x",
        Series::new(vec![Value::Int(3), Value::Null, Value::Int(1)]),
    )])
    .unwrap();

    let asc = df.sort_values("x", true).unwrap();
    assert_eq!(ints(asc.get("x").unwrap()), vec![Some(1), Some(3), None]);

    let desc = df.sort_values("x", false).unwrap();
    assert_eq!(ints(desc.get("x").unwrap()), vec![Some(3), Some(1), None]);
}

// pandas 2.2: pd.read_csv on 'a,b\n"l1\nl2",1\n' -> a = "l1\nl2", b = 1
#[test]
fn read_csv_quoted_newline_round_trips() {
    let mut src = NamedTempFile::new().unwrap();
    src.write_all(b"a,b\n\"l1\nl2\",1\n").unwrap();
    src.flush().unwrap();

    let df = read_csv(src.path()).unwrap();
    assert_eq!(df.nrows(), 1);
    assert_eq!(
        df.get("a").unwrap().values()[0],
        Value::String("l1\nl2".to_string())
    );
    assert_eq!(df.get("b").unwrap().values()[0], Value::Int(1));

    let out = NamedTempFile::new().unwrap();
    write_csv(&df, out.path()).unwrap();
    let again = read_csv(out.path()).unwrap();
    assert_eq!(again.nrows(), 1);
    assert_eq!(
        again.get("a").unwrap().values()[0],
        Value::String("l1\nl2".to_string())
    );
    assert_eq!(again.get("b").unwrap().values()[0], Value::Int(1));
}

// pandas 2.2: pd.read_csv(..., skip_blank_lines=False) on 'n\n1\n\n3\n' -> [1, NaN, 3]
#[test]
fn read_csv_one_column_blank_row_is_null() {
    let mut src = NamedTempFile::new().unwrap();
    src.write_all(b"n\n1\n\n3\n").unwrap();
    src.flush().unwrap();

    let df = read_csv(src.path()).unwrap();
    assert_eq!(ints(df.get("n").unwrap()), vec![Some(1), None, Some(3)]);
}

// pandas 2.2: pd.read_csv(..., encoding='utf-8-sig') on a UTF-8 BOM file -> columns [a, b]
#[test]
fn read_csv_strips_utf8_bom() {
    let mut src = NamedTempFile::new().unwrap();
    src.write_all(b"\xEF\xBB\xBFa,b\n1,2\n").unwrap();
    src.flush().unwrap();

    let df = read_csv(src.path()).unwrap();
    let cols: Vec<&str> = df.columns().iter().map(String::as_str).collect();
    assert_eq!(cols, ["a", "b"]);
}
