//! Reference tests for the DataFrame methods `describe`, `sort_values`,
//! `sort_values_by`, `corr`, `duplicated`, `drop_duplicates`, `dropna`,
//! `dropna_all`, `dropna_subset` and `fillna_column`.
//! Each expected value is read from `fixtures/dataframe_stats_reference.tsv`,
//! which `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::{DataFrame, Series, Value};

const FIXTURE: &str = include_str!("../fixtures/dataframe_stats_reference.tsv");

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
fn col_f64(df: &DataFrame, name: &str) -> Vec<f64> {
    df.get(name)
        .unwrap()
        .values()
        .iter()
        .map(|v| match v {
            Value::Null => f64::NAN,
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            other => panic!("column {name}: not a number: {other:?}"),
        })
        .collect()
}

/// The checksum sum((pos + 1) * value) over a list in its order.
fn weighted(values: &[f64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(pos, &v)| (pos as f64 + 1.0) * v)
        .sum()
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

/// The null case: `p` is null where i % 7 == 3, `q` where i % 5 == 2 (so both
/// are null where i % 35 == 17), and `r` never is.
fn null_frame() -> DataFrame {
    let p: Vec<Value> = (0..10000)
        .map(|i| {
            if i % 7 == 3 {
                Value::Null
            } else {
                Value::Float(((i * 37 + 11) % 101) as f64 / 4.0)
            }
        })
        .collect();
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

/// The small case of `describe`: 1, a null, 3 and 5.
fn small_frame() -> DataFrame {
    DataFrame::from_columns(vec![(
        "x",
        Series::new(vec![
            Value::Float(1.0),
            Value::Null,
            Value::Float(3.0),
            Value::Float(5.0),
        ]),
    )])
    .unwrap()
}

/// The tiny case of `sort_values`: 3, a null and 1.
fn tiny_frame() -> DataFrame {
    DataFrame::from_columns(vec![(
        "x",
        Series::new(vec![Value::Int(3), Value::Null, Value::Int(1)]),
    )])
    .unwrap()
}

/// The first five and the last five values of `a`, and the checksums of `k`
/// and `b`, for a sorted frame. The checksums show that ties keep the input order.
fn check_sorted(df: &DataFrame, base: &str) {
    let a = col_f64(df, "a");
    assert_close_list(&a[..5], &format!("{base}.a.first5"));
    assert_close_list(&a[a.len() - 5..], &format!("{base}.a.last5"));
    assert_close(weighted(&col_f64(df, "k")), &format!("{base}.k.wsum"));
    assert_close(weighted(&col_f64(df, "b")), &format!("{base}.b.wsum"));
}

#[test]
fn describe_matches_pandas() {
    let stats = [
        "count", "mean", "std", "min", "25%", "50%", "75%", "max", "sum",
    ];

    let main = main_frame().describe();
    for col in ["k", "a", "b", "c"] {
        for stat in stats {
            assert_close(main[col][stat], &format!("dataframe.describe.{col}.{stat}"));
        }
    }

    let small = small_frame().describe();
    for stat in stats {
        assert_close(
            small["x"][stat],
            &format!("dataframe.describe.small.{stat}"),
        );
    }
}

#[test]
fn sort_values_matches_pandas() {
    let tiny = tiny_frame();
    let asc = tiny.sort_values("x", true).unwrap();
    assert_close_list(&col_f64(&asc, "x"), "dataframe.sort_values.small.asc");
    let desc = tiny.sort_values("x", false).unwrap();
    assert_close_list(&col_f64(&desc, "x"), "dataframe.sort_values.small.desc");

    let main = main_frame();
    let asc = main.sort_values("a", true).unwrap();
    check_sorted(&asc, "dataframe.sort_values.a.asc");
    let desc = main.sort_values("a", false).unwrap();
    check_sorted(&desc, "dataframe.sort_values.a.desc");
}

#[test]
fn sort_values_by_matches_pandas() {
    let main = main_frame();
    let sorted = main.sort_values_by(&["k", "a"], &[true, false]).unwrap();
    check_sorted(&sorted, "dataframe.sort_values_by.k_asc_a_desc");
    let sorted = main.sort_values_by(&["a", "k"], &[false, true]).unwrap();
    check_sorted(&sorted, "dataframe.sort_values_by.a_desc_k_asc");
}

#[test]
fn corr_matches_pandas() {
    let corr = main_frame().corr().unwrap();
    let names = ["k", "a", "b", "c"];
    let cols: Vec<Vec<f64>> = names.iter().map(|name| col_f64(&corr, name)).collect();
    // Row-major: entry [i][j] is the correlation of column i with column j.
    let flat: Vec<f64> = (0..names.len())
        .flat_map(|i| cols.iter().map(move |col| col[i]))
        .collect();
    assert_close_list(&flat, "dataframe.corr.large");
}

#[test]
fn duplicated_matches_pandas() {
    let main = main_frame();
    let cases: [(&str, Option<&[&str]>, &str); 5] = [
        ("kb_first", Some(&["k", "b"][..]), "first"),
        ("kb_last", Some(&["k", "b"][..]), "last"),
        ("kb_none", Some(&["k", "b"][..]), "none"),
        ("a_first", Some(&["a"][..]), "first"),
        ("all_first", None, "first"),
    ];
    for (name, subset, keep) in cases {
        let mask = main.duplicated(subset, keep).unwrap();
        let count = mask.iter().filter(|&&dup| dup).count();
        let wsum: usize = mask
            .iter()
            .enumerate()
            .filter(|(_, &dup)| dup)
            .map(|(pos, _)| pos + 1)
            .sum();
        assert_close(count as f64, &format!("dataframe.duplicated.{name}.count"));
        assert_close(wsum as f64, &format!("dataframe.duplicated.{name}.wsum"));
    }
}

#[test]
fn drop_duplicates_matches_pandas() {
    let main = main_frame();
    let cases: [(&str, Option<&[&str]>, &str); 5] = [
        ("kb_first", Some(&["k", "b"][..]), "first"),
        ("kb_last", Some(&["k", "b"][..]), "last"),
        ("kb_none", Some(&["k", "b"][..]), "none"),
        ("a_first", Some(&["a"][..]), "first"),
        ("all_first", None, "first"),
    ];
    for (name, subset, keep) in cases {
        let kept = main.drop_duplicates(subset, keep).unwrap();
        assert_close(
            kept.nrows() as f64,
            &format!("dataframe.drop_duplicates.{name}.nrows"),
        );
        assert_close(
            weighted(&col_f64(&kept, "c")),
            &format!("dataframe.drop_duplicates.{name}.c_wsum"),
        );
    }
}

#[test]
fn dropna_matches_pandas() {
    let null = null_frame();
    let kept = null.dropna().unwrap();
    assert_close(kept.nrows() as f64, "dataframe.dropna.nrows");
    assert_close(weighted(&col_f64(&kept, "r")), "dataframe.dropna.r_wsum");

    let pq = null.select(&["p", "q"]).unwrap();
    assert_close(
        pq.dropna().unwrap().nrows() as f64,
        "dataframe.dropna.pq.nrows",
    );
}

#[test]
fn dropna_all_matches_pandas() {
    let null = null_frame();
    let kept = null.dropna_all().unwrap();
    assert_close(kept.nrows() as f64, "dataframe.dropna_all.nrows");
    assert_close(
        weighted(&col_f64(&kept, "r")),
        "dataframe.dropna_all.r_wsum",
    );

    let pq = null.select(&["p", "q"]).unwrap();
    assert_close(
        pq.dropna_all().unwrap().nrows() as f64,
        "dataframe.dropna_all.pq.nrows",
    );
}

#[test]
fn dropna_subset_matches_pandas() {
    let null = null_frame();
    for (name, columns) in [
        ("p", &["p"][..]),
        ("q", &["q"][..]),
        ("pq", &["p", "q"][..]),
    ] {
        let kept = null.dropna_subset(columns).unwrap();
        assert_close(
            kept.nrows() as f64,
            &format!("dataframe.dropna_subset.{name}.nrows"),
        );
        assert_close(
            weighted(&col_f64(&kept, "r")),
            &format!("dataframe.dropna_subset.{name}.r_wsum"),
        );
    }
}

#[test]
fn fillna_column_matches_pandas() {
    let null = null_frame();
    let filled = null.fillna_column("p", Value::Float(-1.5)).unwrap();
    let p = filled.get("p").unwrap();
    assert_close(p.sum().unwrap(), "dataframe.fillna_column.p.sum");
    assert_close(p.count() as f64, "dataframe.fillna_column.p.count");
    assert_close(
        filled.get("q").unwrap().count() as f64,
        "dataframe.fillna_column.q.count",
    );
    assert_close(
        weighted(&col_f64(&filled, "r")),
        "dataframe.fillna_column.r.wsum",
    );
}
