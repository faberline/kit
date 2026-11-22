//! Reference tests for the join methods of `frame::frame/ops/join`.
//! Each expected value is read from `fixtures/join_reference.tsv`, which
//! `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::ops::{join, JoinType};
use frame::frame::{DataFrame, Series, Value};

const FIXTURE: &str = include_str!("../fixtures/join_reference.tsv");

fn reference(key: &str) -> f64 {
    for line in FIXTURE.lines().skip(1) {
        let mut fields = line.split('\t');
        if fields.next() == Some(key) {
            let value = fields
                .next()
                .unwrap_or_else(|| panic!("fixture key {key}: no value column"));
            return value
                .parse()
                .unwrap_or_else(|_| panic!("fixture key {key}: bad number {value}"));
        }
    }
    panic!("fixture key {key} not found");
}

/// A list value of the fixture, comma separated; `nan` is a null.
fn reference_list(key: &str) -> Vec<f64> {
    for line in FIXTURE.lines().skip(1) {
        let mut fields = line.split('\t');
        if fields.next() == Some(key) {
            let value = fields
                .next()
                .unwrap_or_else(|| panic!("fixture key {key}: no value column"));
            return value
                .split(',')
                .map(|item| {
                    if item == "nan" {
                        f64::NAN
                    } else {
                        item.parse()
                            .unwrap_or_else(|_| panic!("fixture key {key}: bad number {item}"))
                    }
                })
                .collect();
        }
    }
    panic!("fixture key {key} not found");
}

/// 1e-9 relative, as the table says.
fn assert_close(actual: f64, key: &str) {
    let expected = reference(key);
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-300),
        "{key}: got {actual}, expected {expected}"
    );
}

/// Each element of a list, in order, against the fixture list `key`; a NaN equals a NaN.
fn assert_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "{key}: length");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{key}[{i}]: got {a}, expected NaN");
        } else {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1e-300),
                "{key}[{i}]: got {a}, expected {e}"
            );
        }
    }
}

/// The values of a column in row order; a null is NaN.
fn floats(out: &DataFrame, column: &str) -> Vec<f64> {
    out.get(column)
        .unwrap_or_else(|_| panic!("output has no column {column}"))
        .values()
        .iter()
        .map(|v| v.as_float().unwrap_or(f64::NAN))
        .collect()
}

/// The `lv` column in the order of `sort_values(['id', 'rv'])` of pandas (NaN last).
fn lv_by_id(out: &DataFrame) -> Vec<f64> {
    let ids = floats(out, "id");
    let rvs = floats(out, "rv_right");
    let lvs = floats(out, "lv");
    let mut rows: Vec<(f64, f64, f64)> = (0..ids.len()).map(|i| (ids[i], rvs[i], lvs[i])).collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    rows.iter().map(|row| row.2).collect()
}

/// The non-null count and the sum of a column; a null is skipped.
fn count_and_sum(out: &DataFrame, column: &str) -> (f64, f64) {
    let values: Vec<f64> = out
        .get(column)
        .unwrap_or_else(|_| panic!("output has no column {column}"))
        .values()
        .iter()
        .filter_map(Value::as_float)
        .collect();
    (values.len() as f64, values.iter().sum())
}

/// Checks the row count and, for each `(output column, fixture name)`, the non-null count
/// and the sum against the fixture keys `<prefix>.rows`, `<prefix>.<name>_count`
/// and `<prefix>.<name>_sum`.
fn assert_frame(out: &DataFrame, prefix: &str, columns: &[(&str, &str)]) {
    assert_close(out.nrows() as f64, &format!("{prefix}.rows"));
    for (column, name) in columns {
        let (count, sum) = count_and_sum(out, column);
        assert_close(count, &format!("{prefix}.{name}_count"));
        assert_close(sum, &format!("{prefix}.{name}_sum"));
    }
}

/// The output columns of the large join: the key `k`, `lv` from the left, `rv_right` from the right.
const LARGE_COLUMNS: [(&str, &str); 3] = [("k", "k"), ("lv", "lv"), ("rv_right", "rv")];
/// The output columns of the small join.
const SMALL_COLUMNS: [(&str, &str); 3] = [("id", "id"), ("lv", "lv"), ("rv_right", "rv")];

/// Left frame: 10,000 rows. The key is null where i % 97 == 5, otherwise i % 50.
/// `lv` is ((i * 37 + 11) % 101) / 4.
fn left_frame() -> DataFrame {
    let keys: Vec<Value> = (0..10_000i64)
        .map(|i| {
            if i % 97 == 5 {
                Value::Null
            } else {
                Value::Int(i % 50)
            }
        })
        .collect();
    let vals: Vec<Value> = (0..10_000i64)
        .map(|i| Value::Float(((i * 37 + 11) % 101) as f64 / 4.0))
        .collect();
    DataFrame::from_columns(vec![("k", Series::new(keys)), ("lv", Series::new(vals))]).unwrap()
}

/// Right frame: 8,000 rows. The key is null where j % 89 == 7, otherwise j % 4000 + 10.
/// `rv` is ((j * 53 + 7) % 103) / 2.
fn right_frame() -> DataFrame {
    let keys: Vec<Value> = (0..8_000i64)
        .map(|j| {
            if j % 89 == 7 {
                Value::Null
            } else {
                Value::Int(j % 4000 + 10)
            }
        })
        .collect();
    let vals: Vec<Value> = (0..8_000i64)
        .map(|j| Value::Float(((j * 53 + 7) % 103) as f64 / 2.0))
        .collect();
    DataFrame::from_columns(vec![("k", Series::new(keys)), ("rv", Series::new(vals))]).unwrap()
}

/// Small frames shared by the three moved tests.
fn small_frames() -> (DataFrame, DataFrame) {
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

const HOWS: [(&str, JoinType); 4] = [
    ("inner", JoinType::Inner),
    ("left", JoinType::Left),
    ("right", JoinType::Right),
    ("outer", JoinType::Outer),
];

#[test]
fn join_matches_pandas() {
    // the free function, for each join type
    let (left, right) = (left_frame(), right_frame());
    for (how, kind) in HOWS {
        let out = join(&left, &right, &["k"], kind).unwrap();
        assert_frame(&out, &format!("join.{how}.large"), &LARGE_COLUMNS);
    }
}

#[test]
fn join_method_matches_pandas() {
    // the `DataFrame::join` method, for each join type
    let (left, right) = (left_frame(), right_frame());
    for (how, kind) in HOWS {
        let out = left.join(&right, &["k"], kind).unwrap();
        assert_frame(&out, &format!("join.{how}.large"), &LARGE_COLUMNS);
    }
}

#[test]
fn merge_matches_pandas() {
    let out = left_frame().merge(&right_frame(), &["k"]).unwrap();
    assert_frame(&out, "join.inner.large", &LARGE_COLUMNS);
}

#[test]
fn left_join_matches_pandas() {
    let out = left_frame().left_join(&right_frame(), &["k"]).unwrap();
    assert_frame(&out, "join.left.large", &LARGE_COLUMNS);
}

#[test]
fn right_join_matches_pandas() {
    let out = left_frame().right_join(&right_frame(), &["k"]).unwrap();
    assert_frame(&out, "join.right.large", &LARGE_COLUMNS);
}

#[test]
fn outer_join_matches_pandas() {
    let out = left_frame().outer_join(&right_frame(), &["k"]).unwrap();
    assert_frame(&out, "join.outer.large", &LARGE_COLUMNS);
}

// pandas: l.merge(r, on='id', how='right'): id [2, 3, 3, 4]; lv is NaN for id 4 (no left row).
#[test]
fn join_right_keeps_right_only_key() {
    let (l, r) = small_frames();
    let out = l.join(&r, &["id"], JoinType::Right).unwrap();
    assert_frame(&out, "join.right.small", &SMALL_COLUMNS);
    let mut ids = floats(&out, "id");
    ids.sort_by(f64::total_cmp);
    assert_list(&ids, "join.right.small.id_sorted");
    assert_list(&lv_by_id(&out), "join.right.small.lv_by_id");
}

// pandas: l.merge(r, on='id', how='outer'): id [1, 2, 3, 3, 4], no null id.
#[test]
fn join_outer_keeps_both_sides() {
    let (l, r) = small_frames();
    let out = l.join(&r, &["id"], JoinType::Outer).unwrap();
    assert_frame(&out, "join.outer.small", &SMALL_COLUMNS);
    let mut ids = floats(&out, "id");
    ids.sort_by(f64::total_cmp);
    assert_list(&ids, "join.outer.small.id_sorted");
    assert_list(&lv_by_id(&out), "join.outer.small.lv_by_id");
}

// pandas: l.merge(r, on='k') with l.k int64 [1, 2, 3] and r.k float64 [2.0, 3.0]:
// the int keys 2 and 3 match the float keys 2.0 and 3.0; 2 rows.
#[test]
fn join_int_key_matches_float_key() {
    let l = DataFrame::from_columns(vec![("k", Series::new(vec![1i64, 2, 3]))]).unwrap();
    let r = DataFrame::from_columns(vec![
        ("k", Series::new(vec![2.0f64, 3.0])),
        ("rv", Series::new(vec![5i64, 6])),
    ])
    .unwrap();
    let out = l.join(&r, &["k"], JoinType::Inner).unwrap();
    assert_frame(&out, "join.intfloat.inner", &[("rv_right", "rv")]);
    let mut rv = floats(&out, "rv_right");
    rv.sort_by(f64::total_cmp);
    assert_list(&rv, "join.intfloat.inner.rv_sorted");
}
