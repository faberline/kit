//! Reference tests for the window methods of `frame::frame/ops/window`.
//! Each expected value is read from `fixtures/window_reference.tsv`, which
//! `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::{Series, Value};

const FIXTURE: &str = include_str!("../fixtures/window_reference.tsv");

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

fn assert_close(actual: f64, key: &str) {
    let expected = reference(key);
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-300),
        "{key}: got {actual}, expected {expected}"
    );
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

/// The small case with nulls: 10 values, a null where i is 0, 3, 4 or 9.
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

/// The large case with nulls: 10,000 rows, a null where i % 7 == 3.
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

/// The small Ewm case: 10 values, no nulls.
fn dense_small() -> Series {
    Series::new(
        (0..10i64)
            .map(|i| Value::Float(((i * 3) % 7 + 1) as f64 / 2.0))
            .collect::<Vec<Value>>(),
    )
}

/// The large Ewm case: 10,000 values, no nulls.
fn dense_large() -> Series {
    Series::new(
        (0..10_000i64)
            .map(|i| Value::Float((((i * 37 + 11) % 101) + 1) as f64 / 4.0))
            .collect::<Vec<Value>>(),
    )
}

/// An Ewm result for one constructor (`alpha`, `halflife` or `span`), one `adjust`
/// flag and one statistic (`mean`, `std` or `var`).
fn ewm_series(s: &Series, param: &str, value: f64, adjust: bool, stat: &str) -> Series {
    let window = match param {
        "alpha" => s.ewm_alpha(value),
        "halflife" => s.ewm_halflife(value),
        "span" => s.ewm_span(value),
        _ => unreachable!("unknown ewm constructor {param}"),
    };
    let window = window.adjust(adjust);
    match stat {
        "mean" => window.mean(),
        "std" => window.std(),
        "var" => window.var(),
        _ => unreachable!("unknown ewm statistic {stat}"),
    }
}

const EWM_PARAMS: [(&str, f64); 3] = [("alpha", 0.3), ("halflife", 4.0), ("span", 10.0)];

fn adjust_label(adjust: bool) -> &'static str {
    if adjust {
        "adj"
    } else {
        "noadj"
    }
}

#[test]
fn expanding_sum_matches_pandas() {
    assert_list(&small().expanding().sum(), "expanding.sum.small");
    assert_large(&large().expanding().sum(), "expanding.sum.large");
    assert_list(
        &small().expanding().min_periods(3).sum(),
        "expanding.sum.mp3.small",
    );
}

#[test]
fn expanding_mean_matches_pandas() {
    assert_list(&small().expanding().mean(), "expanding.mean.small");
    assert_large(&large().expanding().mean(), "expanding.mean.large");
    assert_list(
        &small().expanding().min_periods(3).mean(),
        "expanding.mean.mp3.small",
    );
}

#[test]
fn expanding_min_matches_pandas() {
    assert_list(&small().expanding().min(), "expanding.min.small");
    assert_large(&large().expanding().min(), "expanding.min.large");
}

#[test]
fn expanding_max_matches_pandas() {
    assert_list(&small().expanding().max(), "expanding.max.small");
    assert_large(&large().expanding().max(), "expanding.max.large");
}

#[test]
fn expanding_std_matches_pandas() {
    assert_list(&small().expanding().std(), "expanding.std.small");
    assert_large(&large().expanding().std(), "expanding.std.large");
}

#[test]
fn expanding_var_matches_pandas() {
    assert_list(&small().expanding().var(), "expanding.var.small");
    assert_large(&large().expanding().var(), "expanding.var.large");
}

#[test]
fn ewm_mean_matches_pandas() {
    for (param, value) in EWM_PARAMS {
        for adjust in [true, false] {
            let label = adjust_label(adjust);
            let key = format!("ewm.{param}.{label}.mean");
            assert_list(
                &ewm_series(&dense_small(), param, value, adjust, "mean"),
                &format!("{key}.small"),
            );
            assert_large(
                &ewm_series(&dense_large(), param, value, adjust, "mean"),
                &format!("{key}.large"),
            );
        }
    }
    let span_mp3 = dense_small().ewm_span(10.0).min_periods(3).mean();
    assert_list(&span_mp3, "ewm.span.min_periods3.mean.small");
}

#[test]
fn ewm_std_matches_pandas() {
    for (param, value) in EWM_PARAMS {
        for adjust in [true, false] {
            let label = adjust_label(adjust);
            let key = format!("ewm.{param}.{label}.std");
            assert_list(
                &ewm_series(&dense_small(), param, value, adjust, "std"),
                &format!("{key}.small"),
            );
            assert_large(
                &ewm_series(&dense_large(), param, value, adjust, "std"),
                &format!("{key}.large"),
            );
        }
    }
}

#[test]
fn ewm_var_matches_pandas() {
    for (param, value) in EWM_PARAMS {
        for adjust in [true, false] {
            let label = adjust_label(adjust);
            let key = format!("ewm.{param}.{label}.var");
            assert_list(
                &ewm_series(&dense_small(), param, value, adjust, "var"),
                &format!("{key}.small"),
            );
            assert_large(
                &ewm_series(&dense_large(), param, value, adjust, "var"),
                &format!("{key}.large"),
            );
        }
    }
}

#[test]
fn ewm_alpha_matches_pandas() {
    for adjust in [true, false] {
        let label = adjust_label(adjust);
        assert_list(
            &ewm_series(&dense_small(), "alpha", 0.3, adjust, "mean"),
            &format!("ewm.alpha.{label}.mean.small"),
        );
    }
}

#[test]
fn ewm_halflife_matches_pandas() {
    for adjust in [true, false] {
        let label = adjust_label(adjust);
        assert_list(
            &ewm_series(&dense_small(), "halflife", 4.0, adjust, "mean"),
            &format!("ewm.halflife.{label}.mean.small"),
        );
    }
}

#[test]
fn ewm_span_matches_pandas() {
    for adjust in [true, false] {
        let label = adjust_label(adjust);
        assert_list(
            &ewm_series(&dense_small(), "span", 10.0, adjust, "mean"),
            &format!("ewm.span.{label}.mean.small"),
        );
    }
}

#[test]
fn ewm_adjust_matches_pandas() {
    let adjusted = ewm_series(&dense_small(), "alpha", 0.3, true, "mean");
    let unadjusted = ewm_series(&dense_small(), "alpha", 0.3, false, "mean");
    assert_list(&adjusted, "ewm.alpha.adj.mean.small");
    assert_list(&unadjusted, "ewm.alpha.noadj.mean.small");
    // The flag must change the result, so that the test proves it acts.
    let differs = to_vec(&adjusted)
        .iter()
        .zip(to_vec(&unadjusted).iter())
        .any(|(a, b)| (a - b).abs() > 1e-9);
    assert!(differs, "adjust=true and adjust=false give the same values");
}
