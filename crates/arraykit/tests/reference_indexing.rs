//! Reference-value tests for `arraykit::array/indexing`.
//!
//! Expected values come from NumPy 2.4.6. The expression of each value is in
//! the `source` column of `fixtures/indexing_reference.tsv`, which
//! `tools/reference/arraykit/generate.py` writes. No Python runs in CI: the
//! fixture is the only source of the expected values.
//!
//! The inputs are the same integer formulas as in the generator, built with
//! the same operations, so the Rust side holds the same numbers.

use arraykit::array::indexing::AxisIndex;
use arraykit::array::NdArray;

const FIXTURE: &str = include_str!("fixtures/indexing_reference.tsv");

/// Relative tolerance of every check in this file.
const TOL: f64 = 1e-9;

/// Length of the large inputs.
const N: i64 = 10000;

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
    panic!("fixture key {key} is missing");
}

/// The expected list stored under `key`: comma separated numbers.
fn reference_list(key: &str) -> Vec<f64> {
    raw(key)
        .split(',')
        .map(|item| {
            item.parse()
                .unwrap_or_else(|_| panic!("fixture key {key}: bad number {item}"))
        })
        .collect()
}

/// The expected integer list stored under `key`: comma separated integers.
fn reference_ints(key: &str) -> Vec<i64> {
    raw(key)
        .split(',')
        .map(|item| {
            item.parse()
                .unwrap_or_else(|_| panic!("fixture key {key}: bad integer {item}"))
        })
        .collect()
}

/// Check one number with a relative tolerance. A zero expected value needs
/// an absolute error of at most 1e-12. A NaN expected value needs a NaN result.
fn check(actual: f64, expected: f64, rel_tol: f64, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}: got {actual}, expected NaN");
    } else if expected == 0.0 {
        assert!(actual.abs() <= 1e-12, "{label}: got {actual}, expected 0");
    } else {
        assert!(
            (actual - expected).abs() <= rel_tol * expected.abs(),
            "{label}: got {actual}, expected {expected}"
        );
    }
}

/// Check each element of `actual` against the fixture list `key`.
fn assert_close_list(actual: &[f64], key: &str, rel_tol: f64) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "{key}: length");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        check(*a, *e, rel_tol, &format!("{key}[{i}]"));
    }
}

/// Check the integer list `actual` against the fixture list `key`, exactly.
fn assert_ints(actual: &[i64], key: &str) {
    assert_eq!(actual, reference_ints(key).as_slice(), "{key}");
}

/// The dimensions as integers, to compare with the `.shape` key of the fixture.
fn dims_i64(dims: &[usize]) -> Vec<i64> {
    dims.iter().map(|&d| d as i64).collect()
}

/// The values and the shape of `actual` against the fixture keys `key` and `key.shape`.
fn check_array(actual: &NdArray<f64>, key: &str) {
    assert_close_list(actual.data(), key, TOL);
    assert_ints(&dims_i64(actual.dims()), &format!("{key}.shape"));
}

/// A large result against the fixture keys `key.len`, `key.sum` and `key.head`.
fn check_big(actual: &NdArray<f64>, key: &str) {
    assert_ints(&[actual.size() as i64], &format!("{key}.len"));
    let sum: f64 = actual.data().iter().sum();
    check(sum, reference_list(&format!("{key}.sum"))[0], TOL, key);
    assert_close_list(&actual.data()[..5], &format!("{key}.head"), TOL);
}

fn arr_f(data: Vec<f64>, dims: Vec<usize>) -> NdArray<f64> {
    NdArray::new(data, dims).unwrap()
}

/// The inputs of the small cases (the same values as in the generator).
fn a6() -> NdArray<f64> {
    arr_f(vec![3.0, -1.5, 3.0, 0.5, -1.5, 2.0], vec![6])
}

fn m23() -> NdArray<f64> {
    arr_f(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
}

/// The 4 x 6 grid with the value `i` at position `i` (0.0 .. 23.0).
fn g46() -> NdArray<f64> {
    arr_f((0..24).map(|i| i as f64).collect(), vec![4, 6])
}

/// The 2 x 3 x 4 cube with the value `i` at position `i` (0.0 .. 23.0).
fn x234() -> NdArray<f64> {
    arr_f((0..24).map(|i| i as f64).collect(), vec![2, 3, 4])
}

/// The large values: 10,000 values in multiples of 0.25, many ties.
fn fl_data() -> Vec<f64> {
    (0..N).map(|i| ((i * 37 + 11) % 101) as f64 / 4.0).collect()
}

/// The large input as a vector of 10,000 values.
fn fl_arr() -> NdArray<f64> {
    arr_f(fl_data(), vec![N as usize])
}

/// The large input as a 100 x 100 grid.
fn fl2() -> NdArray<f64> {
    arr_f(fl_data(), vec![100, 100])
}

/// A permutation of 0..10000 with out-of-order entries.
fn perm() -> Vec<usize> {
    (0..N).map(|i| ((i * 7919 + 13) % 10000) as usize).collect()
}

/// The large boolean mask: true where i % 3 == 0.
fn bool_large() -> Vec<bool> {
    (0..N).map(|i| i % 3 == 0).collect()
}

/// The 50 row indices of the large 2-D case.
fn rows50() -> Vec<usize> {
    (0..50).map(|i| ((i * 37 + 5) % 100) as usize).collect()
}

/// The 40 column indices of the large 2-D case.
fn cols40() -> Vec<usize> {
    (0..40).map(|i| ((i * 11 + 3) % 100) as usize).collect()
}

#[test]
fn fancy_index_matches_numpy() {
    check_array(
        &a6().fancy_index(&[4, 0, 0, 5]).unwrap(),
        "indexing.fancy_index.small",
    );
    check_array(
        &g46().fancy_index(&[3, 0, 3]).unwrap(),
        "indexing.fancy_index.2d",
    );
    check_big(
        &fl_arr().fancy_index(&perm()).unwrap(),
        "indexing.fancy_index.large",
    );
}

#[test]
fn bool_index_matches_numpy() {
    let mask = [true, false, true, true, false, true];
    check_array(
        &a6().bool_index(&mask).unwrap(),
        "indexing.bool_index.small",
    );
    let mask = [true, false, true, false, true, true];
    check_array(&m23().bool_index(&mask).unwrap(), "indexing.bool_index.2d");
    check_big(
        &fl_arr().bool_index(&bool_large()).unwrap(),
        "indexing.bool_index.large",
    );
}

#[test]
fn bool_index_array_matches_numpy() {
    let mask_2d: Vec<bool> = (0..24).map(|i| (i * 5 + 2) % 7 < 3).collect();
    let mask = NdArray::new(mask_2d, vec![4, 6]).unwrap();
    check_array(
        &g46().bool_index_array(&mask).unwrap(),
        "indexing.bool_index_array.2d",
    );

    let mask_large: Vec<bool> = (0..N).map(|i| i % 7 < 3).collect();
    let mask = NdArray::new(mask_large, vec![100, 100]).unwrap();
    check_big(
        &fl2().bool_index_array(&mask).unwrap(),
        "indexing.bool_index_array.large",
    );
}

#[test]
fn bool_set_matches_numpy() {
    let mut a = a6();
    a.bool_set(&[true, false, true, true, false, true], 9.5)
        .unwrap();
    check_array(&a, "indexing.bool_set.small");

    let mut a = fl_arr();
    a.bool_set(&bool_large(), -1.0).unwrap();
    check_big(&a, "indexing.bool_set.large");
}

#[test]
fn fancy_set_matches_numpy() {
    let mut a = a6();
    a.fancy_set(&[4, 0, 2], &[10.0, 20.0, 30.0]).unwrap();
    check_array(&a, "indexing.fancy_set.small");

    // A repeated index: the later value wins (index 1 gets 8.0).
    let mut a = a6();
    a.fancy_set(&[1, 1, 3], &[7.0, 8.0, 9.0]).unwrap();
    check_array(&a, "indexing.fancy_set.dup");

    let mut a = fl_arr();
    let values: Vec<f64> = (0..N).map(|i| (i % 13) as f64 / 2.0).collect();
    a.fancy_set(&perm(), &values).unwrap();
    check_big(&a, "indexing.fancy_set.large");
}

#[test]
fn advanced_index_matches_numpy() {
    let g = g46();
    check_array(
        &g.advanced_index(&[AxisIndex::Full, AxisIndex::Fancy(vec![5, 1, 1])])
            .unwrap(),
        "indexing.advanced_index.full_fancy",
    );
    check_array(
        &g.advanced_index(&[
            AxisIndex::Fancy(vec![3, 0]),
            AxisIndex::Fancy(vec![4, 2, 2]),
        ])
        .unwrap(),
        "indexing.advanced_index.fancy_fancy",
    );
    check_array(
        &g.advanced_index(&[AxisIndex::Single(2), AxisIndex::Full])
            .unwrap(),
        "indexing.advanced_index.single_full",
    );
    check_array(
        &g.advanced_index(&[AxisIndex::Full, AxisIndex::Single(3)])
            .unwrap(),
        "indexing.advanced_index.full_single",
    );
    check_array(
        &g.advanced_index(&[
            AxisIndex::Bool(vec![true, false, true, true]),
            AxisIndex::Full,
        ])
        .unwrap(),
        "indexing.advanced_index.bool_full",
    );
    check_array(
        &g.advanced_index(&[
            AxisIndex::Bool(vec![true, false, true, true]),
            AxisIndex::Fancy(vec![5, 0]),
        ])
        .unwrap(),
        "indexing.advanced_index.bool_fancy",
    );

    // All axes reduced: arraykit returns one element, NumPy a 0-d scalar.
    let single = g
        .advanced_index(&[AxisIndex::Single(1), AxisIndex::Single(4)])
        .unwrap();
    assert_close_list(single.data(), "indexing.advanced_index.single_single", TOL);
    assert_eq!(single.dims(), &[1]);

    check_array(
        &x234()
            .advanced_index(&[AxisIndex::Single(1), AxisIndex::Fancy(vec![2, 0])])
            .unwrap(),
        "indexing.advanced_index.3d_trailing",
    );
    check_array(
        &g.advanced_index(&[]).unwrap(),
        "indexing.advanced_index.empty",
    );

    let large = fl2()
        .advanced_index(&[AxisIndex::Fancy(rows50()), AxisIndex::Fancy(cols40())])
        .unwrap();
    check_big(&large, "indexing.advanced_index.large");
    assert_ints(
        &dims_i64(large.dims()),
        "indexing.advanced_index.large.shape",
    );
}

#[test]
fn take_matches_numpy() {
    check_array(&a6().take(&[5, 0, 0, 3], 0).unwrap(), "indexing.take.small");
    check_array(&g46().take(&[3, 1, 1], 0).unwrap(), "indexing.take.axis0");
    check_array(
        &g46().take(&[5, 0, 2, 2], 1).unwrap(),
        "indexing.take.axis1",
    );
    check_array(&x234().take(&[3, 0], 2).unwrap(), "indexing.take.3d_axis2");
    check_big(&fl_arr().take(&perm(), 0).unwrap(), "indexing.take.large");

    let large2d = fl2().take(&cols40(), 1).unwrap();
    check_big(&large2d, "indexing.take.large2d");
    assert_ints(&dims_i64(large2d.dims()), "indexing.take.large2d.shape");
}
