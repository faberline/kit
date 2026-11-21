//! Reference tests for the window and null methods of `frame::frame/series`.
//! Each expected value is read from `fixtures/series_window_reference.tsv`, which
//! `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::{Series, Value};

const FIXTURE: &str = include_str!("../fixtures/series_window_reference.tsv");

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

/// A list value: comma separated, a null written as `nan`.
fn reference_list(key: &str) -> Vec<f64> {
    raw(key)
        .split(',')
        .map(|item| {
            if item == "nan" {
                f64::NAN
            } else {
                item.parse()
                    .unwrap_or_else(|_| panic!("fixture key {key}: bad number {item}"))
            }
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

fn assert_slice(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "{key}: length");
    for (pos, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let same = if e.is_nan() {
            a.is_nan()
        } else {
            (a - e).abs() <= 1e-9 * e.abs().max(1e-300)
        };
        assert!(same, "{key}: position {pos}: got {a}, expected {e}");
    }
}

/// The values of a series; a null becomes NaN.
fn to_vec(s: &Series) -> Vec<f64> {
    s.values()
        .iter()
        .map(|v| v.as_float().unwrap_or(f64::NAN))
        .collect()
}

fn assert_list(actual: &Series, key: &str) {
    assert_slice(&to_vec(actual), key);
}

/// Count, sum and position-weighted sum of the non-null values.
fn assert_large(actual: &Series, prefix: &str) {
    let all = to_vec(actual);
    let count = all.iter().filter(|x| !x.is_nan()).count() as f64;
    let sum: f64 = all.iter().filter(|x| !x.is_nan()).sum();
    let wsum: f64 = all
        .iter()
        .enumerate()
        .filter(|(_, x)| !x.is_nan())
        .map(|(i, x)| x * (i as f64 + 1.0))
        .sum();
    assert_close(count, &format!("{prefix}.count"));
    assert_close(sum, &format!("{prefix}.sum"));
    assert_close(wsum, &format!("{prefix}.wsum"));
}

fn bits(mask: Vec<bool>) -> Vec<f64> {
    mask.into_iter()
        .map(|b| if b { 1.0 } else { 0.0 })
        .collect()
}

/// The small case: 10 values, a null where i is 0, 3, 4 or 9.
fn small() -> Series {
    Series::new(
        (0..10i64)
            .map(|i| {
                if matches!(i, 0 | 3 | 4 | 9) {
                    Value::Null
                } else {
                    Value::Float(((i * 3) % 7 + 1) as f64 / 2.0)
                }
            })
            .collect::<Vec<Value>>(),
    )
}

/// The large case: 10,000 rows, a null where i % 7 == 3.
fn large() -> Series {
    Series::new(
        (0..10_000i64)
            .map(|i| {
                if i % 7 == 3 {
                    Value::Null
                } else {
                    Value::Float((((i * 37 + 11) % 101) + 1) as f64 / 4.0)
                }
            })
            .collect::<Vec<Value>>(),
    )
}

/// `apply` function: doubles a number and adds one; a null stays null.
fn scale_plus_one(v: &Value) -> Value {
    match v.as_float() {
        Some(x) => Value::Float(x * 2.0 + 1.0),
        None => Value::Null,
    }
}

#[test]
fn cumsum_matches_pandas() {
    assert_list(&small().cumsum(), "window.cumsum.small");
    assert_large(&large().cumsum(), "window.cumsum.large");
}

#[test]
fn cummax_matches_pandas() {
    assert_list(&small().cummax(), "window.cummax.small");
    assert_large(&large().cummax(), "window.cummax.large");
}

#[test]
fn cummin_matches_pandas() {
    assert_list(&small().cummin(), "window.cummin.small");
    assert_large(&large().cummin(), "window.cummin.large");
}

#[test]
fn cumprod_matches_pandas() {
    // Small case only: the large product overflows.
    assert_list(&small().cumprod(), "window.cumprod.small");
}

#[test]
fn diff_matches_pandas() {
    assert_list(&small().diff(1), "window.diff1.small");
    assert_large(&large().diff(1), "window.diff1.large");
    assert_list(&small().diff(2), "window.diff2.small");
    assert_large(&large().diff(2), "window.diff2.large");
}

#[test]
fn pct_change_matches_pandas() {
    assert_list(&small().pct_change(1), "window.pct_change1.small");
    assert_large(&large().pct_change(1), "window.pct_change1.large");
    assert_list(&small().pct_change(2), "window.pct_change2.small");
    assert_large(&large().pct_change(2), "window.pct_change2.large");
}

#[test]
fn shift_matches_pandas() {
    assert_list(&small().shift(2), "window.shift2.small");
    assert_large(&large().shift(2), "window.shift2.large");
    assert_list(&small().shift(-2), "window.shift_neg2.small");
    assert_large(&large().shift(-2), "window.shift_neg2.large");
}

#[test]
fn rolling_matches_pandas() {
    assert_list(&small().rolling(3).sum(), "rolling.sum.w3.small");
    assert_large(&large().rolling(3).sum(), "rolling.sum.w3.large");
    assert_list(
        &small().rolling(3).min_periods(2).sum(),
        "rolling.sum.w3mp2.small",
    );
    assert_large(
        &large().rolling(50).min_periods(40).sum(),
        "rolling.sum.w50.large",
    );
}

#[test]
fn rolling_sum_matches_pandas() {
    assert_list(&small().rolling(3).sum(), "rolling.sum.w3.small");
    assert_large(&large().rolling(3).sum(), "rolling.sum.w3.large");
    assert_list(
        &small().rolling(3).min_periods(2).sum(),
        "rolling.sum.w3mp2.small",
    );
    assert_large(
        &large().rolling(50).min_periods(40).sum(),
        "rolling.sum.w50.large",
    );
}

#[test]
fn rolling_mean_matches_pandas() {
    assert_list(&small().rolling(3).mean(), "rolling.mean.w3.small");
    assert_large(&large().rolling(3).mean(), "rolling.mean.w3.large");
    assert_list(
        &small().rolling(3).min_periods(2).mean(),
        "rolling.mean.w3mp2.small",
    );
    assert_large(
        &large().rolling(50).min_periods(40).mean(),
        "rolling.mean.w50.large",
    );
}

#[test]
fn rolling_min_matches_pandas() {
    assert_list(&small().rolling(3).min(), "rolling.min.w3.small");
    assert_large(&large().rolling(3).min(), "rolling.min.w3.large");
    assert_list(
        &small().rolling(3).min_periods(2).min(),
        "rolling.min.w3mp2.small",
    );
    assert_large(
        &large().rolling(50).min_periods(40).min(),
        "rolling.min.w50.large",
    );
}

#[test]
fn rolling_max_matches_pandas() {
    assert_list(&small().rolling(3).max(), "rolling.max.w3.small");
    assert_large(&large().rolling(3).max(), "rolling.max.w3.large");
    assert_list(
        &small().rolling(3).min_periods(2).max(),
        "rolling.max.w3mp2.small",
    );
    assert_large(
        &large().rolling(50).min_periods(40).max(),
        "rolling.max.w50.large",
    );
}

#[test]
fn rolling_std_matches_pandas() {
    assert_list(&small().rolling(3).std(), "rolling.std.w3.small");
    assert_large(&large().rolling(3).std(), "rolling.std.w3.large");
    assert_list(
        &small().rolling(3).min_periods(2).std(),
        "rolling.std.w3mp2.small",
    );
    assert_large(
        &large().rolling(50).min_periods(40).std(),
        "rolling.std.w50.large",
    );
}

#[test]
fn rolling_var_matches_pandas() {
    assert_list(&small().rolling(3).var(), "rolling.var.w3.small");
    assert_large(&large().rolling(3).var(), "rolling.var.w3.large");
    assert_list(
        &small().rolling(3).min_periods(2).var(),
        "rolling.var.w3mp2.small",
    );
    assert_large(
        &large().rolling(50).min_periods(40).var(),
        "rolling.var.w50.large",
    );
}

#[test]
fn ffill_matches_pandas() {
    assert_list(&small().ffill(), "null.ffill.small");
    assert_large(&large().ffill(), "null.ffill.large");
}

#[test]
fn bfill_matches_pandas() {
    assert_list(&small().bfill(), "null.bfill.small");
    assert_large(&large().bfill(), "null.bfill.large");
}

#[test]
fn fillna_matches_pandas() {
    assert_list(&small().fillna(Value::Float(-1.5)), "null.fillna.small");
    assert_large(&large().fillna(Value::Float(-1.5)), "null.fillna.large");
}

#[test]
fn apply_matches_pandas() {
    assert_list(&small().apply(scale_plus_one), "null.apply.small");
    assert_large(&large().apply(scale_plus_one), "null.apply.large");
}

#[test]
fn dropna_matches_pandas() {
    assert_list(&small().dropna(), "null.dropna.small");
    let clean = large().dropna();
    let total: f64 = to_vec(&clean).iter().sum();
    assert_close(clean.len() as f64, "null.dropna.large.count");
    assert_close(total, "null.dropna.large.sum");
}

#[test]
fn null_count_matches_pandas() {
    assert_close(small().null_count() as f64, "null.null_count.small");
    assert_close(large().null_count() as f64, "null.null_count.large");
}

#[test]
fn isna_matches_pandas() {
    assert_slice(&bits(small().isna()), "null.isna.small");
    let nulls = large().isna().iter().filter(|b| **b).count() as f64;
    assert_close(nulls, "null.isna.large.count");
}

#[test]
fn notna_matches_pandas() {
    assert_slice(&bits(small().notna()), "null.notna.small");
    let present = large().notna().iter().filter(|b| **b).count() as f64;
    assert_close(present, "null.notna.large.count");
}

#[test]
fn interpolate_matches_pandas() {
    assert_list(&small().interpolate(), "null.interpolate.small");
    assert_large(&large().interpolate(), "null.interpolate.large");
}

#[test]
fn interpolate_method_matches_pandas() {
    assert_list(
        &small().interpolate_method("linear"),
        "null.interpolate.small",
    );
    assert_list(&small().interpolate_method("ffill"), "null.ffill.small");
    assert_list(&small().interpolate_method("bfill"), "null.bfill.small");
    // Positions 3 and 4 only: the edges are not compared (see the spec of #265).
    let nearest = to_vec(&small().interpolate_method("nearest"));
    assert_slice(&nearest[3..5], "null.interpolate_nearest.small");
}
