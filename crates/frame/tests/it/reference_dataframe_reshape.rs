//! Reference tests for the DataFrame methods `isna`, `isna_any`, `isna_all`,
//! `melt`, `pivot`, `sample` and `sample_seeded`.
//! Each expected value is read from `fixtures/dataframe_reshape_reference.tsv`,
//! which `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::{DataFrame, Series, Value};
use std::collections::HashSet;

const FIXTURE: &str = include_str!("../fixtures/dataframe_reshape_reference.tsv");

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

fn assert_close(actual: f64, key: &str) {
    let expected = reference(key);
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-300),
        "{key}: got {actual}, expected {expected}"
    );
}

fn assert_columns(df: &DataFrame, expected: &[&str]) {
    let actual: Vec<&str> = df.columns().iter().map(String::as_str).collect();
    assert_eq!(actual, expected);
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

/// The checksum of the flags `flags`, with true as 1 and false as 0.
fn flag_checksum(flags: &[bool]) -> f64 {
    let numbers: Vec<f64> = flags.iter().map(|&f| if f { 1.0 } else { 0.0 }).collect();
    weighted(&numbers)
}

/// The number of true flags.
fn true_count(flags: &[bool]) -> usize {
    flags.iter().filter(|&&f| f).count()
}

/// The value of column `a` at row `i`: a null where i % 7 == 3.
fn a_value(i: i64) -> Value {
    if i % 7 == 3 {
        Value::Null
    } else {
        Value::Float(((i * 37 + 11) % 101) as f64 / 4.0)
    }
}

/// The main case: 10,000 rows. `k` has keys with duplicates, `a` has nulls,
/// `b` has ties and `c` has no null.
fn main_frame() -> DataFrame {
    let k: Vec<i64> = (0..10000).map(|i| i % 50).collect();
    let a: Vec<Value> = (0..10000).map(a_value).collect();
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

/// The null case: `p` is null where i % 7 == 3, `q` where i % 5 == 2, and
/// `r` = i % 9 has no null. A row is all-null in `p` and `q` where i % 35 == 17.
fn null_frame() -> DataFrame {
    let p: Vec<Value> = (0..10000).map(a_value).collect();
    let q: Vec<Value> = (0..10000)
        .map(|i| {
            if i % 5 == 2 {
                Value::Null
            } else {
                Value::Float(((i * 11) % 17) as f64 / 2.0)
            }
        })
        .collect();
    let r: Vec<i64> = (0..10000).map(|i| i % 9).collect();
    DataFrame::from_columns(vec![
        ("p", Series::new(p)),
        ("q", Series::new(q)),
        ("r", Series::new(r)),
    ])
    .unwrap()
}

/// The long case for pivot: 600 rows, `g` = i / 6, `h` = i % 6, and `v` with
/// nulls. Every (g, h) pair is unique, which pivot needs.
fn long_frame() -> DataFrame {
    let g: Vec<i64> = (0..600).map(|i| i / 6).collect();
    let h: Vec<i64> = (0..600).map(|i| i % 6).collect();
    let v: Vec<Value> = (0..600).map(a_value).collect();
    DataFrame::from_columns(vec![
        ("g", Series::new(g)),
        ("h", Series::new(h)),
        ("v", Series::new(v)),
    ])
    .unwrap()
}

/// The frame for sample: 10,000 rows, `id` = i and `a` as in the main case.
fn id_frame() -> DataFrame {
    let id: Vec<i64> = (0..10000).collect();
    let a: Vec<Value> = (0..10000).map(a_value).collect();
    DataFrame::from_columns(vec![("id", Series::new(id)), ("a", Series::new(a))]).unwrap()
}

/// The `id` values of a sampled frame, in their order.
fn ids_of(df: &DataFrame) -> Vec<i64> {
    col_f64(df, "id").iter().map(|&x| x as i64).collect()
}

/// The checks of a melt result against the fixture keys under `prefix`: the
/// row count, the sum and count of `value` (nulls skipped), the checksum of
/// `k`, and the number of rows per variable name in `names`.
fn check_melt(out: &DataFrame, prefix: &str, names: &[&str]) {
    assert_columns(out, &["k", "variable", "value"]);
    assert_close(out.nrows() as f64, &format!("{prefix}.nrows"));
    let kept = non_null(&col_f64(out, "value"));
    let total: f64 = kept.iter().sum();
    assert_close(total, &format!("{prefix}.value.sum"));
    assert_close(kept.len() as f64, &format!("{prefix}.value.count"));
    assert_close(weighted(&col_f64(out, "k")), &format!("{prefix}.k.wsum"));
    let variables = out.get("variable").unwrap().values();
    for &name in names {
        let count = variables
            .iter()
            .filter(|v| matches!(v, Value::String(s) if s.as_str() == name))
            .count();
        assert_close(count as f64, &format!("{prefix}.variable.{name}"));
    }
}

#[test]
fn isna_matches_pandas() {
    let m = null_frame().isna();
    assert_eq!(m.len(), 3);
    for col in ["p", "q", "r"] {
        let flags = &m[col];
        assert_eq!(flags.len(), 10000);
        assert_close(
            true_count(flags) as f64,
            &format!("dataframe.isna.{col}.count"),
        );
        assert_close(flag_checksum(flags), &format!("dataframe.isna.{col}.pos"));
    }
}

#[test]
fn isna_any_matches_pandas() {
    let any = null_frame().isna_any();
    assert_eq!(any.len(), 10000);
    assert_close(true_count(&any) as f64, "dataframe.isna_any.P.count");
    assert_close(flag_checksum(&any), "dataframe.isna_any.P.pos");
}

#[test]
fn isna_all_matches_pandas() {
    // p and q together: all-null rows, where i % 35 == 17.
    let pq = null_frame().select(&["p", "q"]).unwrap().isna_all();
    assert_eq!(pq.len(), 10000);
    assert_close(true_count(&pq) as f64, "dataframe.isna_all.PQ.count");
    assert_close(flag_checksum(&pq), "dataframe.isna_all.PQ.pos");

    // r is never null, so no row of the whole frame is all-null.
    let all = null_frame().isna_all();
    assert_close(true_count(&all) as f64, "dataframe.isna_all.P.count");
}

#[test]
fn melt_matches_pandas() {
    // Case A: value_vars a and b.
    let a = main_frame().melt(&["k"], &["a", "b"]).unwrap();
    check_melt(&a, "dataframe.melt.A", &["a", "b"]);

    // Case B: value_vars left empty, so every other column (a, b, c) is melted.
    let b = main_frame().melt(&["k"], &[]).unwrap();
    check_melt(&b, "dataframe.melt.B", &["a", "b", "c"]);
}

#[test]
fn pivot_matches_pandas() {
    let out = long_frame().pivot("g", "h", "v").unwrap();
    assert_columns(&out, &["g", "0", "1", "2", "3", "4", "5"]);
    assert_close(out.nrows() as f64, "dataframe.pivot.nrows");
    // ncols counts the index column g too; pandas W has 6 value columns.
    assert_close((out.ncols() - 1) as f64, "dataframe.pivot.ncols");
    assert_close(weighted(&col_f64(&out, "g")), "dataframe.pivot.g.wsum");
    for h in 0..6 {
        let kept = non_null(&col_f64(&out, &h.to_string()));
        let total: f64 = kept.iter().sum();
        assert_close(total, &format!("dataframe.pivot.h{h}.sum"));
        assert_close(kept.len() as f64, &format!("dataframe.pivot.h{h}.count"));
    }
}

#[test]
fn sample_matches_pandas_facts() {
    let df = id_frame();

    // 100 rows: 100 rows, all distinct, and each row is the source row of its id.
    let s = df.sample(100).unwrap();
    assert_close(s.nrows() as f64, "dataframe.sample.n100.nrows");
    let ids = ids_of(&s);
    let distinct: HashSet<i64> = ids.iter().copied().collect();
    assert_close(distinct.len() as f64, "dataframe.sample.n100.unique_ids");
    let a = col_f64(&s, "a");
    for (&id, &got) in ids.iter().zip(&a) {
        let expected = values_f64(&[a_value(id)])[0];
        if expected.is_nan() {
            assert!(got.is_nan(), "row id {id}: got {got}, expected null");
        } else {
            assert!(
                (got - expected).abs() <= 1e-12,
                "row id {id}: got {got}, expected {expected}"
            );
        }
    }

    // More rows than the frame has: all 10,000 rows, each id once.
    let all = df.sample(20000).unwrap();
    assert_close(all.nrows() as f64, "dataframe.sample.n20000.nrows");
    let id_sum: f64 = ids_of(&all).iter().map(|&x| x as f64).sum();
    assert_close(id_sum, "dataframe.sample.n20000.id_sum");

    // Zero rows: an empty frame.
    let none = df.sample(0).unwrap();
    assert_close(none.nrows() as f64, "dataframe.sample.n0.nrows");
}

#[test]
fn sample_seeded_matches_pandas_facts() {
    let df = id_frame();

    // The same seed gives the same rows in the same order.
    let first = df.sample_seeded(100, 7).unwrap();
    let again = df.sample_seeded(100, 7).unwrap();
    assert_eq!(ids_of(&first), ids_of(&again));

    // Another seed gives other rows.
    let other = df.sample_seeded(100, 8).unwrap();
    assert_ne!(ids_of(&first), ids_of(&other));

    // sample(n) is sample_seeded(n, 42).
    assert_eq!(
        ids_of(&df.sample(100).unwrap()),
        ids_of(&df.sample_seeded(100, 42).unwrap())
    );

    // Counts and distinct ids as in the sample test.
    assert_close(first.nrows() as f64, "dataframe.sample.n100.nrows");
    let distinct: HashSet<i64> = ids_of(&first).into_iter().collect();
    assert_close(distinct.len() as f64, "dataframe.sample.n100.unique_ids");
}
