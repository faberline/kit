//! Reference tests for the DataFrame methods `head`, `tail`, `filter`, `query`,
//! `select`, `assign`, `insert`, `drop`, `rename`, `fillna`, `concat`, `apply`,
//! `applymap`, `set_index` and `reset_index`.
//! Each expected value is read from `fixtures/dataframe_ops_reference.tsv`,
//! which `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::{DataFrame, Index, Series, Value};
use std::collections::HashMap;

const FIXTURE: &str = include_str!("../fixtures/dataframe_ops_reference.tsv");

/// The column names of the main frame, in order.
const COLS: [&str; 4] = ["k", "a", "b", "c"];

/// The value column of `key` in the fixture, as text.
fn raw(key: &str) -> &'static str {
    for line in FIXTURE.lines().skip(1) {
        let mut fields = line.split('\t');
        if fields.next() == Some(key) {
            return fields
                .next()
                .unwrap_or_else(|| panic!("fixture key {key}: no value column"));
        }
    }
    panic!("fixture key {key} not found");
}

/// The expected number stored under `key`.
fn reference(key: &str) -> f64 {
    let value = raw(key);
    value
        .parse()
        .unwrap_or_else(|_| panic!("fixture key {key}: bad number {value}"))
}

/// The expected list stored under `key`: comma separated numbers, `nan` for a null.
fn reference_list(key: &str) -> Vec<f64> {
    raw(key)
        .split(',')
        .map(|item| {
            item.parse()
                .unwrap_or_else(|_| panic!("fixture key {key}: bad number {item}"))
        })
        .collect()
}

fn assert_close(actual: f64, key: &str) {
    let expected = reference(key);
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-300),
        "{key}: got {actual}, expected {expected}"
    );
}

/// Check each element of `actual` against the fixture list `key`. A NaN
/// expected value needs a NaN result; an expected 0.0 needs |actual| <= 1e-12.
fn assert_close_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "{key}: length");
    for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{key}[{i}]: got {a}, expected NaN");
        } else if e == 0.0 {
            assert!(a.abs() <= 1e-12, "{key}[{i}]: got {a}, expected 0");
        } else {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs(),
                "{key}[{i}]: got {a}, expected {e}"
            );
        }
    }
}

/// The values of a column as numbers: a null becomes NaN.
fn values_f64(values: &[Value]) -> Vec<f64> {
    values
        .iter()
        .map(|v| match v {
            Value::Null => f64::NAN,
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            other => panic!("not a number: {other:?}"),
        })
        .collect()
}

/// The values of the column `name` of `df` as numbers.
fn col_f64(df: &DataFrame, name: &str) -> Vec<f64> {
    values_f64(df.get(name).unwrap().values())
}

/// The checksum sum((pos + 1) * value) over a list in its order.
fn weighted(values: &[f64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(pos, &v)| (pos as f64 + 1.0) * v)
        .sum()
}

/// The numbers that are not null, as pandas skips them in a sum or a count.
fn non_null(values: &[f64]) -> Vec<f64> {
    values.iter().copied().filter(|v| !v.is_nan()).collect()
}

/// The labels of the index of `df`, as numbers.
fn index_labels(df: &DataFrame) -> Vec<f64> {
    df.index()
        .to_values()
        .iter()
        .map(|v| match v {
            Value::Int(i) => *i as f64,
            other => panic!("label is not an int: {other:?}"),
        })
        .collect()
}

fn assert_columns(df: &DataFrame, expected: &[&str]) {
    let actual: Vec<&str> = df.columns().iter().map(String::as_str).collect();
    assert_eq!(actual, expected);
}

/// The checks of one result frame, against the fixture keys under `prefix`:
/// the row count; for each float column in `floats` its sum and count (nulls
/// skipped); for each int column in `ints` its position checksum.
fn check_frame(df: &DataFrame, prefix: &str, floats: &[&str], ints: &[&str]) {
    assert_close(df.nrows() as f64, &format!("{prefix}.nrows"));
    for &name in floats {
        let kept = non_null(&col_f64(df, name));
        let total: f64 = kept.iter().sum();
        assert_close(total, &format!("{prefix}.{name}.sum"));
        assert_close(kept.len() as f64, &format!("{prefix}.{name}.count"));
    }
    for &name in ints {
        assert_close(
            weighted(&col_f64(df, name)),
            &format!("{prefix}.{name}.wsum"),
        );
    }
}

/// The main case: 10,000 rows. Column `k` has keys with duplicates, `a` has a
/// null where i % 7 == 3, `b` has ties, and `c` has no null.
fn main_frame() -> DataFrame {
    let k: Vec<i64> = (0..10000).map(|i| i % 50).collect();
    let a: Vec<Value> = (0..10000)
        .map(|i| {
            if i % 7 == 3 {
                Value::Null
            } else {
                Value::Float(((i * 37 + 11) % 101) as f64 / 4.0)
            }
        })
        .collect();
    let b: Vec<i64> = (0..10000).map(|i| (i * 13 + 5) % 37).collect();
    let c: Vec<f64> = (0..10000)
        .map(|i| ((i * 29 + 7) % 53) as f64 / 8.0)
        .collect();
    DataFrame::from_columns(vec![
        ("k", Series::new(k)),
        ("a", Series::new(a)),
        ("b", Series::new(b)),
        ("c", Series::new(c)),
    ])
    .unwrap()
}

/// The small case: `x` is 1.0, a null, 3.0 and 5.0; `y` is 2, 4, 6 and 8.
fn small_frame() -> DataFrame {
    DataFrame::from_columns(vec![
        (
            "x",
            Series::new(vec![
                Value::Float(1.0),
                Value::Null,
                Value::Float(3.0),
                Value::Float(5.0),
            ]),
        ),
        ("y", Series::new(vec![2i64, 4, 6, 8])),
    ])
    .unwrap()
}

/// The column `b` of the main case, times 2.
fn doubled_b() -> Series {
    Series::new(
        (0..10000i64)
            .map(|i| 2 * ((i * 13 + 5) % 37))
            .collect::<Vec<i64>>(),
    )
}

#[test]
fn head_matches_pandas() {
    let m = main_frame();

    let five = m.head(5).unwrap();
    assert_columns(&five, &COLS);
    check_frame(&five, "dataframe.head.n5", &["a", "c"], &["k", "b"]);

    // More rows than the frame has: all 10,000 rows come back.
    let all = m.head(20000).unwrap();
    assert_columns(&all, &COLS);
    check_frame(&all, "dataframe.head.n20000", &["a", "c"], &["k", "b"]);
}

#[test]
fn tail_matches_pandas() {
    let m = main_frame();

    let five = m.tail(5).unwrap();
    assert_columns(&five, &COLS);
    check_frame(&five, "dataframe.tail.n5", &["a", "c"], &["k", "b"]);

    let all = m.tail(20000).unwrap();
    assert_columns(&all, &COLS);
    check_frame(&all, "dataframe.tail.n20000", &["a", "c"], &["k", "b"]);
}

#[test]
fn filter_matches_pandas() {
    let mask: Vec<bool> = (0..10000).map(|i| i % 3 == 0).collect();
    let kept = main_frame().filter(&mask).unwrap();
    assert_columns(&kept, &COLS);
    check_frame(&kept, "dataframe.filter", &["a", "c"], &["k", "b"]);
}

#[test]
fn query_matches_pandas() {
    // pandas: M.query("k < 10").
    let kept = main_frame()
        .query(|row| matches!(row.get("k"), Some(Value::Int(k)) if *k < 10))
        .unwrap();
    assert_columns(&kept, &COLS);
    check_frame(&kept, "dataframe.query", &["a", "c"], &["k", "b"]);
}

#[test]
fn select_matches_pandas() {
    let picked = main_frame().select(&["c", "k"]).unwrap();
    assert_columns(&picked, &["c", "k"]);
    check_frame(&picked, "dataframe.select", &["c"], &["k"]);
}

#[test]
fn assign_matches_pandas() {
    let m = main_frame();

    // A new column d = b * 2.
    let added = m.assign(vec![("d", doubled_b())]).unwrap();
    assert_columns(&added, &["k", "a", "b", "c", "d"]);
    check_frame(
        &added,
        "dataframe.assign.new",
        &["a", "c"],
        &["k", "b", "d"],
    );

    // An existing column b is replaced by b * 2 and keeps its place.
    let replaced = m.assign(vec![("b", doubled_b())]).unwrap();
    assert_columns(&replaced, &COLS);
    check_frame(
        &replaced,
        "dataframe.assign.replace",
        &["a", "c"],
        &["k", "b"],
    );
}

#[test]
fn insert_matches_pandas() {
    let mut m = main_frame();
    let plus_one: Vec<f64> = col_f64(&m, "c").iter().map(|c| c + 1.0).collect();
    m.insert("e", Series::new(plus_one)).unwrap();
    assert_columns(&m, &["k", "a", "b", "c", "e"]);
    check_frame(&m, "dataframe.insert", &["a", "c", "e"], &["k", "b"]);
}

#[test]
fn drop_matches_pandas() {
    let mut m = main_frame();
    let removed = m.drop("c").unwrap();
    assert_columns(&m, &["k", "a", "b"]);
    check_frame(&m, "dataframe.drop", &["a"], &["k", "b"]);

    // The returned series is the dropped column c.
    let kept = non_null(&values_f64(removed.values()));
    let total: f64 = kept.iter().sum();
    assert_close(total, "dataframe.drop.removed.sum");
    assert_close(kept.len() as f64, "dataframe.drop.removed.count");
}

#[test]
fn rename_matches_pandas() {
    let mut m = main_frame();
    m.rename(HashMap::from([("a", "alpha")]));
    assert_columns(&m, &["k", "alpha", "b", "c"]);
    check_frame(&m, "dataframe.rename", &["alpha", "c"], &["k", "b"]);
}

#[test]
fn fillna_matches_pandas() {
    let filled = main_frame().fillna(Value::Float(-1.5));
    assert_columns(&filled, &COLS);
    check_frame(&filled, "dataframe.fillna", &["a", "c"], &["k", "b"]);
}

#[test]
fn concat_matches_pandas() {
    let m = main_frame();
    let both = DataFrame::concat(&[&m, &m]).unwrap();
    assert_columns(&both, &COLS);
    check_frame(&both, "dataframe.concat", &["a", "c"], &["k", "b"]);
}

#[test]
fn apply_matches_pandas() {
    let m = main_frame();
    let sums = m.apply(|col| col.sum().unwrap_or(0.0));
    let counts = m.apply(|col| col.count());
    for name in COLS {
        assert_close(sums[name], &format!("dataframe.apply.{name}.sum"));
        assert_close(
            counts[name] as f64,
            &format!("dataframe.apply.{name}.count"),
        );
    }
}

#[test]
fn applymap_matches_pandas() {
    // pandas: S.map(lambda v: v * 2). A null stays null.
    let doubled = small_frame().applymap(|v| match v {
        Value::Int(i) => Value::Int(i * 2),
        Value::Float(x) => Value::Float(x * 2.0),
        other => other.clone(),
    });
    assert_columns(&doubled, &["x", "y"]);
    check_frame(&doubled, "dataframe.applymap.small", &["x"], &["y"]);
}

#[test]
fn set_index_matches_pandas() {
    let mut s = small_frame();
    s.set_index(Index::int(vec![10, 20, 30, 40])).unwrap();
    assert_close_list(&index_labels(&s), "dataframe.set_index.small.labels");
}

#[test]
fn set_index_wrong_length_is_error() {
    let mut s = small_frame();
    assert!(s.set_index(Index::int(vec![1, 2])).is_err());
}

#[test]
fn reset_index_matches_pandas() {
    let mut s = small_frame();
    s.set_index(Index::int(vec![10, 20, 30, 40])).unwrap();
    s.reset_index();
    assert_close_list(&index_labels(&s), "dataframe.reset_index.small.labels");
}
