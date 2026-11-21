//! Reference tests for the Series methods of `frame::frame/series`.
//! Each expected value is read from `fixtures/series_reference.tsv`, which
//! `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::{Index, Series, Value};
use std::collections::HashMap;

const FIXTURE: &str = include_str!("../fixtures/series_reference.tsv");

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

fn assert_close(actual: f64, key: &str) {
    let expected = reference(key);
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-300),
        "{key}: got {actual}, expected {expected}"
    );
}

fn as_f64(value: &Value) -> f64 {
    value
        .as_float()
        .unwrap_or_else(|| panic!("value {value:?} is not a number"))
}

/// The small case: 8 values with one null, a tie (4, 4) and a negative (-1).
fn small_values() -> Vec<Value> {
    vec![
        Value::Int(3),
        Value::Null,
        Value::Int(-1),
        Value::Int(4),
        Value::Int(4),
        Value::Int(2),
        Value::Int(7),
        Value::Int(0),
    ]
}

fn small() -> Series {
    Series::new(small_values())
}

/// The large case: 10,000 rows, a null where i % 7 == 3.
fn large() -> Series {
    Series::new(
        (0..10_000i64)
            .map(|i| {
                if i % 7 == 3 {
                    Value::Null
                } else {
                    Value::Float(((i * 37 + 11) % 101) as f64 / 4.0)
                }
            })
            .collect::<Vec<Value>>(),
    )
}

/// The key series of the large case: i % 50, with 50 keys of 200 rows each.
fn large_keys() -> Series {
    Series::new((0..10_000i64).map(|i| i % 50).collect::<Vec<i64>>())
}

fn corr_x() -> Series {
    Series::new(
        (0..10_000i64)
            .map(|i| (i * 13 + 5) % 37)
            .collect::<Vec<i64>>(),
    )
}

fn corr_y() -> Series {
    Series::new(
        (0..10_000i64)
            .map(|i| 2 * ((i * 13 + 5) % 37) + i % 11)
            .collect::<Vec<i64>>(),
    )
}

#[test]
fn sum_matches_pandas() {
    assert_close(small().sum().unwrap(), "series.sum.small.sum");
    assert_close(large().sum().unwrap(), "series.sum.large.sum");
}

#[test]
fn mean_matches_pandas() {
    assert_close(small().mean().unwrap(), "series.mean.small.mean");
    assert_close(large().mean().unwrap(), "series.mean.large.mean");
}

#[test]
fn std_matches_pandas() {
    assert_close(small().std().unwrap(), "series.std.small.std");
    assert_close(large().std().unwrap(), "series.std.large.std");
}

#[test]
fn var_matches_pandas() {
    assert_close(small().var().unwrap(), "series.var.small.var");
    assert_close(large().var().unwrap(), "series.var.large.var");
}

#[test]
fn median_matches_pandas() {
    assert_close(small().median().unwrap(), "series.median.small.median");
    assert_close(large().median().unwrap(), "series.median.large.median");
}

#[test]
fn min_matches_pandas() {
    assert_close(as_f64(&small().min().unwrap()), "series.min.small.min");
    assert_close(as_f64(&large().min().unwrap()), "series.min.large.min");
}

#[test]
fn max_matches_pandas() {
    assert_close(as_f64(&small().max().unwrap()), "series.max.small.max");
    assert_close(as_f64(&large().max().unwrap()), "series.max.large.max");
}

#[test]
fn count_matches_pandas() {
    assert_close(small().count() as f64, "series.count.small.count");
    assert_close(large().count() as f64, "series.count.large.count");
}

#[test]
fn quantile_matches_pandas() {
    assert_close(small().quantile(0.25).unwrap(), "series.quantile.small.q25");
    assert_close(large().quantile(0.25).unwrap(), "series.quantile.large.q25");
    assert_close(large().quantile(0.9).unwrap(), "series.quantile.large.q90");
}

#[test]
fn corr_matches_pandas() {
    assert_close(corr_x().corr(&corr_y()).unwrap(), "series.corr.large.corr");
}

#[test]
fn abs_matches_pandas() {
    assert_close(small().abs().sum().unwrap(), "series.abs.small.sum");
}

#[test]
fn between_matches_pandas() {
    let inside = small().between(0.0, 3.0);
    assert_close(
        inside.iter().filter(|&&b| b).count() as f64,
        "series.between.small.count",
    );
}

#[test]
fn clip_matches_pandas() {
    assert_close(
        small().clip(0.0, 3.0).sum().unwrap(),
        "series.clip.small.sum",
    );
}

#[test]
fn drop_duplicates_matches_pandas() {
    assert_close(
        small().drop_duplicates("first").len() as f64,
        "series.drop_duplicates.small.len",
    );
}

#[test]
fn duplicated_matches_pandas() {
    let dup = small().duplicated("first");
    assert_close(
        dup.iter().filter(|&&b| b).count() as f64,
        "series.duplicated.small.count",
    );
}

#[test]
fn filter_matches_pandas() {
    let mask: Vec<bool> = (0..8).map(|i| i % 2 == 0).collect();
    assert_close(
        small().filter(&mask).unwrap().sum().unwrap(),
        "series.filter.small.sum",
    );
}

#[test]
fn isin_matches_pandas() {
    let hit = small().isin(&[Value::Int(4), Value::Int(0)]);
    assert_close(
        hit.iter().filter(|&&b| b).count() as f64,
        "series.isin.small.count",
    );
}

#[test]
fn map_matches_pandas() {
    let doubled = small().map(|v| match v.as_float() {
        Some(x) => Value::Float(x * 2.0 + 1.0),
        None => Value::Null,
    });
    assert_close(doubled.sum().unwrap(), "series.map.small.sum");
}

#[test]
fn mask_matches_pandas() {
    let cond: Vec<bool> = (0..8).map(|i| i % 3 == 0).collect();
    assert_close(
        small().mask(&cond, &Value::Int(0)).sum().unwrap(),
        "series.mask.small.sum",
    );
}

#[test]
fn nlargest_matches_pandas() {
    assert_close(
        small().nlargest(3).sum().unwrap(),
        "series.nlargest.small.sum",
    );
    assert_close(
        large().nlargest(5).sum().unwrap(),
        "series.nlargest.large.sum",
    );
}

#[test]
fn nsmallest_matches_pandas() {
    assert_close(
        small().nsmallest(3).sum().unwrap(),
        "series.nsmallest.small.sum",
    );
    assert_close(
        large().nsmallest(5).sum().unwrap(),
        "series.nsmallest.large.sum",
    );
}

#[test]
fn rank_matches_pandas() {
    // A null gets no rank (NaN in pandas); the other seven ranks sum to 28.
    let ranks = small().rank();
    assert_close(ranks.sum().unwrap(), "series.rank.small.sum");

    // The null stays null: count the Value::Null entries and check position 1.
    let nulls = ranks
        .values()
        .iter()
        .filter(|v| matches!(v, Value::Null))
        .count();
    assert_close(nulls as f64, "series.rank.small.nulls");
    assert!(matches!(ranks.values()[1], Value::Null));

    // The large case: the sum and the count of non-null ranks.
    let big = large().rank();
    assert_close(big.sum().unwrap(), "series.rank.large.sum");
    let ranked = big
        .values()
        .iter()
        .filter(|v| v.as_float().is_some())
        .count();
    assert_close(ranked as f64, "series.rank.large.count");
}

#[test]
fn replace_matches_pandas() {
    assert_close(
        small()
            .replace(&Value::Int(4), &Value::Int(40))
            .sum()
            .unwrap(),
        "series.replace.small.sum",
    );
}

#[test]
fn replace_map_matches_pandas() {
    let mapping: HashMap<Value, Value> = HashMap::from([
        (Value::Int(4), Value::Int(40)),
        (Value::Int(-1), Value::Int(-10)),
    ]);
    assert_close(
        small().replace_map(&mapping).sum().unwrap(),
        "series.replace_map.small.sum",
    );
}

#[test]
fn reset_index_matches_pandas() {
    // Labels 10..17 before the reset: label 7 exists only after it.
    let mut s = Series::with_index(small_values(), Index::Int((10..18).collect())).unwrap();
    s.reset_index();
    assert_close(
        as_f64(s.loc(&Value::Int(7)).unwrap()),
        "series.reset_index.small.loc7",
    );
}

#[test]
fn round_matches_pandas() {
    let s = Series::new(vec![1.234, 5.678, -2.3456, 0.1, 6.789]);
    assert_close(s.round(2).sum().unwrap(), "series.round.fixed.sum");
}

#[test]
fn set_name_matches_pandas() {
    let mut s = small();
    s.set_name("renamed");
    assert_eq!(s.name(), Some("renamed"));
    assert_close(s.sum().unwrap(), "series.set_name.small.sum");
}

#[test]
fn sort_matches_pandas() {
    assert_close(
        as_f64(small().sort(true).iloc(0).unwrap()),
        "series.sort.small.first",
    );
    assert_close(
        as_f64(small().sort(false).iloc(0).unwrap()),
        "series.sort.small.first_desc",
    );
}

#[test]
fn unique_matches_pandas() {
    // pandas counts the null as one unique value; so does frame.
    assert_close(small().unique().len() as f64, "series.unique.small.len");
}

#[test]
fn value_counts_matches_pandas() {
    let counts = large_keys().value_counts();
    assert_close(
        counts[&Value::Int(0)] as f64,
        "series.value_counts.large.key0",
    );
    assert_close(counts.len() as f64, "series.value_counts.large.keys");
}

#[test]
fn where_cond_matches_pandas() {
    let keep: Vec<bool> = (0..8).map(|i| i % 3 != 0).collect();
    assert_close(
        small().where_cond(&keep, &Value::Int(0)).sum().unwrap(),
        "series.where_cond.small.sum",
    );
}

// Moved from tests/it/reference.rs; the expected values are the fixture's.
// pandas 3.1.1 (skipna=True): pd.Series([1, None, 5]).sum() -> 6.0, .mean() -> 3.0
#[test]
fn series_sum_and_mean_skip_nulls() {
    let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(5)]);
    assert_close(s.sum().unwrap(), "series.sum.nulls.sum");
    assert_close(s.mean().unwrap(), "series.mean.nulls.mean");
}

// Moved from tests/it/reference.rs; pandas 3.1.1: pd.Series([1, 1, 2]).rank() -> [1.5, 1.5, 3.0]
#[test]
fn series_rank_average_ties() {
    let ranks = Series::new(vec![1i64, 1, 2]).rank();
    assert_close(ranks.len() as f64, "series.rank.ties.len");
    assert_close(as_f64(&ranks.values()[0]), "series.rank.ties.r0");
    assert_close(as_f64(&ranks.values()[1]), "series.rank.ties.r1");
    assert_close(as_f64(&ranks.values()[2]), "series.rank.ties.r2");
}
