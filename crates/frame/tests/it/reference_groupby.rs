//! Reference tests for the group-by methods of `frame::frame/ops/groupby`.
//! Each expected value is read from `fixtures/groupby_reference.tsv`, which
//! `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::{DataFrame, Series, Value};
use std::collections::{BTreeSet, HashMap};

const FIXTURE: &str = include_str!("../fixtures/groupby_reference.tsv");

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

fn reference(key: &str) -> f64 {
    let value = raw(key);
    value
        .parse()
        .unwrap_or_else(|_| panic!("fixture key {key}: bad number {value}"))
}

/// A scalar from the fixture; a `nan` value stays NaN.
fn assert_value(actual: f64, key: &str) {
    let expected = reference(key);
    let same = if expected.is_nan() {
        actual.is_nan()
    } else {
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-300)
    };
    assert!(same, "{key}: got {actual}, expected {expected}");
}

/// The values of a column; a null becomes NaN.
fn to_vec(s: &Series) -> Vec<f64> {
    s.values()
        .iter()
        .map(|v| v.as_float().unwrap_or(f64::NAN))
        .collect()
}

/// The large frame: 10,000 rows. The key is null where i % 97 == 5, otherwise i % 50.
/// The value is null where i % 7 == 3.
fn large_frame() -> DataFrame {
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
        .map(|i| {
            if i % 7 == 3 {
                Value::Null
            } else {
                Value::Float((((i * 37 + 11) % 101) + 1) as f64 / 4.0)
            }
        })
        .collect();
    DataFrame::from_columns(vec![("g", Series::new(keys)), ("v", Series::new(vals))]).unwrap()
}

/// Checks one group-by output of `large_frame` against the fixture keys under `prefix`:
/// the group count, the keys 0 to 49 in order, the first five values, and the sum.
fn assert_groups(out: &DataFrame, prefix: &str) {
    assert_value(out.nrows() as f64, &format!("{prefix}.len"));
    let keys = out.get("g").unwrap().values();
    assert_eq!(keys.len(), 50, "{prefix}: group count");
    for (pos, key) in keys.iter().enumerate() {
        assert_eq!(*key, Value::Int(pos as i64), "{prefix}: key at {pos}");
    }
    let vals = to_vec(out.get("v").unwrap());
    for (pos, value) in vals.iter().take(5).enumerate() {
        assert_value(*value, &format!("{prefix}.k{pos}"));
    }
    let total: f64 = vals.iter().filter(|x| !x.is_nan()).sum();
    assert_value(total, &format!("{prefix}.sum"));
}

/// The non-null floats of a group.
fn group_floats(series: &Series, indices: &[usize]) -> Vec<f64> {
    indices
        .iter()
        .filter_map(|&i| series.iloc(i).ok()?.as_float())
        .collect()
}

#[test]
fn groupby_sum_matches_pandas() {
    let out = large_frame().groupby(&["g"]).unwrap().sum().unwrap();
    assert_groups(&out, "groupby.sum.large");
}

#[test]
fn groupby_mean_matches_pandas() {
    let out = large_frame().groupby(&["g"]).unwrap().mean().unwrap();
    assert_groups(&out, "groupby.mean.large");
}

#[test]
fn groupby_median_matches_pandas() {
    let out = large_frame().groupby(&["g"]).unwrap().median().unwrap();
    assert_groups(&out, "groupby.median.large");
}

#[test]
fn groupby_min_matches_pandas() {
    let out = large_frame().groupby(&["g"]).unwrap().min().unwrap();
    assert_groups(&out, "groupby.min.large");
}

#[test]
fn groupby_max_matches_pandas() {
    let out = large_frame().groupby(&["g"]).unwrap().max().unwrap();
    assert_groups(&out, "groupby.max.large");
}

#[test]
fn groupby_std_matches_pandas() {
    let out = large_frame().groupby(&["g"]).unwrap().std().unwrap();
    assert_groups(&out, "groupby.std.large");
}

#[test]
fn groupby_var_matches_pandas() {
    let out = large_frame().groupby(&["g"]).unwrap().var().unwrap();
    assert_groups(&out, "groupby.var.large");
}

#[test]
fn groupby_count_matches_pandas() {
    let out = large_frame().groupby(&["g"]).unwrap().count().unwrap();
    assert_groups(&out, "groupby.count.large");
}

#[test]
fn groupby_first_matches_pandas() {
    // pandas `first` takes the first non-null value of each group.
    let out = large_frame().groupby(&["g"]).unwrap().first().unwrap();
    assert_groups(&out, "groupby.first.large");
}

#[test]
fn groupby_last_matches_pandas() {
    // pandas `last` takes the last non-null value of each group.
    let out = large_frame().groupby(&["g"]).unwrap().last().unwrap();
    assert_groups(&out, "groupby.last.large");
}

#[test]
fn groupby_agg_matches_pandas() {
    let df = large_frame();
    let gb = df.groupby(&["g"]).unwrap();
    for name in [
        "sum", "mean", "count", "min", "max", "first", "last", "std", "median",
    ] {
        let out = gb.agg(HashMap::from([("v", name)])).unwrap();
        assert_groups(&out, &format!("groupby.agg.{name}.large"));
    }
}

#[test]
fn groupby_aggregate_matches_pandas() {
    let df = large_frame();
    let gb = df.groupby(&["g"]).unwrap();
    let out = gb
        .aggregate(|series, indices| {
            let vals = group_floats(series, indices);
            if vals.is_empty() {
                return Value::Null;
            }
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            Value::Float(max - min)
        })
        .unwrap();
    assert_groups(&out, "groupby.aggregate.range.large");
}

#[test]
fn groupby_transform_matches_pandas() {
    let df = large_frame();
    let gb = df.groupby(&["g"]).unwrap();
    let out = gb
        .transform(|series, indices| {
            let vals = group_floats(series, indices);
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            indices
                .iter()
                .map(|&i| {
                    series
                        .iloc(i)
                        .ok()
                        .and_then(|v| v.as_float())
                        .map(|x| Value::Float(x / max))
                })
                .collect()
        })
        .unwrap();

    let all = to_vec(out.get("v").unwrap());
    let count = all.iter().filter(|x| !x.is_nan()).count() as f64;
    let sum: f64 = all.iter().filter(|x| !x.is_nan()).sum();
    let wsum: f64 = all
        .iter()
        .enumerate()
        .filter(|(_, x)| !x.is_nan())
        .map(|(i, x)| x * (i as f64 + 1.0))
        .sum();
    assert_value(count, "groupby.transform.div_max.large.count");
    assert_value(sum, "groupby.transform.div_max.large.sum");
    assert_value(wsum, "groupby.transform.div_max.large.wsum");
}

#[test]
fn groupby_filter_groups_matches_pandas() {
    let df = large_frame();
    let gb = df.groupby(&["g"]).unwrap();
    let out = gb
        .filter_groups(|frame, indices| {
            let vals = group_floats(frame.get("v").unwrap(), indices);
            vals.iter().sum::<f64>() / vals.len() as f64 > 12.5
        })
        .unwrap();

    assert_value(out.nrows() as f64, "groupby.filter.mean_gt.large.rows");
    let sum: f64 = to_vec(out.get("v").unwrap())
        .iter()
        .filter(|x| !x.is_nan())
        .sum();
    assert_value(sum, "groupby.filter.mean_gt.large.sum");
    let groups: BTreeSet<i64> = out
        .get("g")
        .unwrap()
        .values()
        .iter()
        .filter_map(|v| v.as_int())
        .collect();
    assert_value(groups.len() as f64, "groupby.filter.mean_gt.large.groups");
}

// Null values are skipped and a null key row is dropped: group a = [1], b = [4].
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

    assert_value(out.nrows() as f64, "groupby.mean.nulls.len");
    let gs = out.get("g").unwrap().values();
    let vs = out.get("v").unwrap().values();
    let mut means: HashMap<String, f64> = HashMap::new();
    for (g, v) in gs.iter().zip(vs.iter()) {
        means.insert(g.as_str().unwrap().to_string(), v.as_float().unwrap());
    }
    assert_value(means["a"], "groupby.mean.nulls.a");
    assert_value(means["b"], "groupby.mean.nulls.b");
}

// a: [1, 2, 4], b: [10, 20]. The std and median values are in the fixture.
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

    assert_value(
        std_col[a_row].as_float().unwrap(),
        "groupby.agg_std.small.a",
    );
    assert_value(
        std_col[b_row].as_float().unwrap(),
        "groupby.agg_std.small.b",
    );
    assert_value(
        med_col[a_row].as_float().unwrap(),
        "groupby.agg_median.small.a",
    );
    assert_value(
        med_col[b_row].as_float().unwrap(),
        "groupby.agg_median.small.b",
    );

    // An unknown aggregation name is an error.
    assert!(gb.agg(HashMap::from([("v", "nope")])).is_err());
}
