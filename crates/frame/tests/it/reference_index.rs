//! Reference tests for the index and multi_index methods of `frame::frame`.
//! Each expected value is read from `fixtures/index_reference.tsv`, which
//! `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::{Index, MultiIndex, Value};

const FIXTURE: &str = include_str!("../fixtures/index_reference.tsv");

const SMALL_INT: [i64; 6] = [5, 3, 9, 3, 8, 1];
const SLICE_POS: [usize; 5] = [4, 0, 0, 5, 2];
const N: usize = 10_000;

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

/// A text value of the fixture, comma separated (level names).
fn reference_text(key: &str) -> Vec<String> {
    for line in FIXTURE.lines().skip(1) {
        let mut fields = line.split('\t');
        if fields.next() == Some(key) {
            let value = fields
                .next()
                .unwrap_or_else(|| panic!("fixture key {key}: no value column"));
            return value.split(',').map(str::to_string).collect();
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

/// An integer value as f64; any other variant is a test error.
fn int_of(value: &Value) -> f64 {
    match value {
        Value::Int(i) => *i as f64,
        other => panic!("expected an integer, got {other:?}"),
    }
}

fn ints(values: &[Value]) -> Vec<f64> {
    values.iter().map(int_of).collect()
}

fn positions(values: &[usize]) -> Vec<f64> {
    values.iter().map(|&p| p as f64).collect()
}

/// The big integer index input: `(i * 37 + 11) % 101` for 10,000 rows.
fn big_int() -> Vec<i64> {
    (0..N as i64).map(|i| (i * 37 + 11) % 101).collect()
}

/// The slice positions of the large case: `(k * 7919 + 3) % 10000` for 2,500 values.
fn big_pos() -> Vec<usize> {
    (0..2_500usize).map(|k| (k * 7919 + 3) % 10_000).collect()
}

/// The small 12-row MultiIndex with levels `a`, `b` and `c`.
fn small_mi() -> MultiIndex {
    let n = 12usize;
    MultiIndex::new(vec![
        ("a", (0..n).map(|i| Value::Int((i % 3) as i64)).collect()),
        (
            "b",
            (0..n)
                .map(|i| Value::Int(((i * 5 + 1) % 4) as i64))
                .collect(),
        ),
        (
            "c",
            (0..n).map(|i| Value::Int(((i * 7) % 5) as i64)).collect(),
        ),
    ])
    .unwrap()
}

/// The large 10,000-row MultiIndex with levels `a`, `b` and `c`.
fn large_mi() -> MultiIndex {
    MultiIndex::new(vec![
        ("a", (0..N).map(|i| Value::Int((i % 50) as i64)).collect()),
        (
            "b",
            (0..N)
                .map(|i| Value::Int(((i * 37 + 11) % 101) as i64))
                .collect(),
        ),
        (
            "c",
            (0..N).map(|i| Value::Int(((i * 13) % 7) as i64)).collect(),
        ),
    ])
    .unwrap()
}

/// The levels of a MultiIndex as f64 lists.
fn levels_of(mi: &MultiIndex) -> Vec<Vec<f64>> {
    (0..mi.nlevels())
        .map(|j| ints(mi.get_level_values(j).unwrap()))
        .collect()
}

/// Levels of a list of tuples (the output of `unique`): level j is entry j of each tuple.
fn tuple_levels(tuples: &[Vec<Value>]) -> Vec<Vec<f64>> {
    let nlevels = tuples.first().map_or(0, Vec::len);
    (0..nlevels)
        .map(|j| tuples.iter().map(|t| int_of(&t[j])).collect())
        .collect()
}

/// Check the levels and the names of a MultiIndex against the fixture keys `prefix`.
fn assert_multi(mi: &MultiIndex, prefix: &str) {
    let names = reference_text(&format!("{prefix}.names"));
    assert_eq!(mi.nlevels(), names.len(), "{prefix}: level count");
    assert_eq!(mi.names(), names.as_slice(), "{prefix}: names");
    for (j, level) in levels_of(mi).iter().enumerate() {
        assert_list(level, &format!("{prefix}.level{j}"));
    }
}

#[test]
fn index_range_matches_pandas() {
    assert_list(
        &ints(&Index::range(10).to_values()),
        "index.range.small.values",
    );
    let large = Index::range(N);
    assert_close(large.len() as f64, "index.range.large.len");
    assert_close(
        ints(&large.to_values()).iter().sum::<f64>(),
        "index.range.large.sum",
    );
}

#[test]
fn index_reset_matches_pandas() {
    let small = Index::int(SMALL_INT.to_vec()).reset();
    assert_list(&ints(&small.to_values()), "index.reset.small.values");
    let large = Index::int(big_int()).reset();
    assert_close(large.len() as f64, "index.reset.large.len");
    assert_close(
        ints(&large.to_values()).iter().sum::<f64>(),
        "index.reset.large.sum",
    );
}

#[test]
fn index_slice_matches_pandas() {
    let small = Index::int(SMALL_INT.to_vec()).slice(&SLICE_POS).unwrap();
    assert_list(&ints(&small.to_values()), "index.slice.small.int");
    // A range index keeps its positions as labels after a slice.
    let range = Index::range(10).slice(&[7, 2, 2, 9]).unwrap();
    assert_list(&ints(&range.to_values()), "index.slice.small.range");
    let large = ints(&Index::int(big_int()).slice(&big_pos()).unwrap().to_values());
    assert_close(large.len() as f64, "index.slice.large.len");
    assert_close(large.iter().sum::<f64>(), "index.slice.large.sum");
    assert_list(&large[..5], "index.slice.large.first5");
    assert_list(&large[large.len() - 5..], "index.slice.large.last5");
}

#[test]
fn multi_index_swaplevel_matches_pandas() {
    let result = small_mi().swaplevel(0, 2).unwrap();
    assert_multi(&result, "multi_index.swaplevel.small");
}

#[test]
fn multi_index_droplevel_matches_pandas() {
    let result = small_mi().droplevel(1).unwrap();
    assert_multi(&result, "multi_index.droplevel.small");
}

#[test]
fn multi_index_reorder_levels_matches_pandas() {
    let result = small_mi().reorder_levels(&[2, 0, 1]).unwrap();
    assert_multi(&result, "multi_index.reorder_levels.small");
}

#[test]
fn multi_index_slice_matches_pandas() {
    let result = small_mi().slice(&[11, 0, 4, 4, 7]).unwrap();
    assert_multi(&result, "multi_index.slice.small");
}

#[test]
fn multi_index_select_level_matches_pandas() {
    let small = small_mi().select_level(1, &Value::Int(2));
    assert_list(
        &positions(&small),
        "multi_index.select_level.small.positions",
    );
    let large = large_mi().select_level(2, &Value::Int(3));
    assert_close(large.len() as f64, "multi_index.select_level.large.len");
    assert_close(
        large.iter().map(|&p| p as f64).sum::<f64>(),
        "multi_index.select_level.large.sum",
    );
}

#[test]
fn multi_index_unique_matches_pandas() {
    let small = small_mi().unique();
    let small_levels = tuple_levels(&small);
    assert_eq!(
        small_levels.len(),
        reference_text("multi_index.unique.small.names").len()
    );
    assert_multi(
        &MultiIndex::new(vec![
            ("a", small.iter().map(|t| t[0].clone()).collect()),
            ("b", small.iter().map(|t| t[1].clone()).collect()),
            ("c", small.iter().map(|t| t[2].clone()).collect()),
        ])
        .unwrap(),
        "multi_index.unique.small",
    );
    let large = large_mi().unique();
    assert_close(large.len() as f64, "multi_index.unique.large.len");
    assert_list(
        &tuple_levels(&large)[0][..5],
        "multi_index.unique.large.first5_level0",
    );
}

#[test]
fn multi_index_argsort_matches_pandas() {
    let small = small_mi().argsort();
    assert_list(&positions(&small), "multi_index.argsort.small");
    let order = large_mi().argsort();
    let first = positions(&order[..5]);
    let last = positions(&order[order.len() - 5..]);
    assert_list(&first, "multi_index.argsort.large.first5");
    assert_list(&last, "multi_index.argsort.large.last5");
    let checksum: f64 = order
        .iter()
        .enumerate()
        .map(|(k, &r)| r as f64 * (k + 1) as f64)
        .sum();
    assert_close(checksum, "multi_index.argsort.large.checksum");
}
