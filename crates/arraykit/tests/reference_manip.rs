//! Reference-value tests for `arraykit::array/manip`.
//!
//! Expected values come from NumPy 2.4.6. The expression of each value is in
//! the `source` column of `fixtures/manip_reference.tsv`, which
//! `tools/reference/arraykit/generate.py` writes. No Python runs in CI: the
//! fixture is the only source of the expected values.
//!
//! The inputs are the same integer formulas as in the generator, built with
//! the same operations, so the Rust side holds the same numbers.

use arraykit::array::NdArray;

const FIXTURE: &str = include_str!("fixtures/manip_reference.tsv");

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

/// A boolean result as 0 or 1, as the fixture stores it.
fn flag(value: bool) -> i64 {
    i64::from(value)
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

fn m23b() -> NdArray<f64> {
    arr_f(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![2, 3])
}

fn m13() -> NdArray<f64> {
    arr_f(vec![0.5, 1.5, 2.5], vec![1, 3])
}

fn m22() -> NdArray<f64> {
    arr_f(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
}

fn v3() -> NdArray<f64> {
    arr_f(vec![1.0, 2.0, 3.0], vec![3])
}

fn v2() -> NdArray<f64> {
    arr_f(vec![4.0, 5.0], vec![2])
}

fn v3b() -> NdArray<f64> {
    arr_f(vec![6.0, 7.0, 8.0], vec![3])
}

/// The 4 x 6 grid with the value `i` at position `i` (0.0 .. 23.0).
fn g46() -> NdArray<f64> {
    arr_f((0..24).map(|i| i as f64).collect(), vec![4, 6])
}

/// The small integer input with ties and negatives.
fn i6() -> NdArray<i64> {
    NdArray::new(vec![3, -2, 3, 0, -2, 5], vec![6]).unwrap()
}

/// The large integer input: 10,000 values, 101 distinct, many ties.
fn il_arr() -> NdArray<i64> {
    let data: Vec<i64> = (0..N).map(|i| ((i * 37 + 11) % 101) - 50).collect();
    NdArray::new(data, vec![N as usize]).unwrap()
}

/// The large float input: 10,000 values in multiples of 0.25, many ties.
fn fl_arr() -> NdArray<f64> {
    let data: Vec<f64> = (0..N).map(|i| ((i * 37 + 11) % 101) as f64 / 4.0).collect();
    arr_f(data, vec![N as usize])
}

/// The large boolean input: true where i % 3 == 0.
fn mask_arr() -> NdArray<bool> {
    let data: Vec<bool> = (0..N).map(|i| i % 3 == 0).collect();
    NdArray::new(data, vec![N as usize]).unwrap()
}

/// The sum of `(i + 1) * x[i]` over a list, in integers: the weighted sum of the fixture.
fn weighted_sum(values: &[i64]) -> i64 {
    values
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as i64 + 1) * x)
        .sum()
}

fn to_i64(values: &[usize]) -> Vec<i64> {
    values.iter().map(|&v| v as i64).collect()
}

#[test]
fn concatenate_matches_numpy() {
    let c = NdArray::concatenate(&[&v3(), &v2(), &v3b()], 0).unwrap();
    check_array(&c, "concatenate.1d");
    let c = NdArray::concatenate(&[&m23(), &m13()], 0).unwrap();
    check_array(&c, "concatenate.axis0");
    let c = NdArray::concatenate(&[&m23(), &m23b()], 1).unwrap();
    check_array(&c, "concatenate.axis1");
}

#[test]
fn stack_matches_numpy() {
    let s = NdArray::stack(&[&v3(), &v3b()], 0).unwrap();
    check_array(&s, "stack.axis0");
    let s = NdArray::stack(&[&v3(), &v3b()], 1).unwrap();
    check_array(&s, "stack.axis1");
    let s = NdArray::stack(&[&m23(), &m23b()], 2).unwrap();
    check_array(&s, "stack.2d.axis2");
}

#[test]
fn vstack_matches_numpy() {
    let s = NdArray::vstack(&[&m23(), &m13()]).unwrap();
    check_array(&s, "vstack.2d");
    let s = NdArray::vstack(&[&v3(), &v3b()]).unwrap();
    check_array(&s, "vstack.1d");
}

#[test]
fn hstack_matches_numpy() {
    let s = NdArray::hstack(&[&m23(), &m23b()]).unwrap();
    check_array(&s, "hstack.2d");
    let s = NdArray::hstack(&[&v3(), &v2(), &v3b()]).unwrap();
    check_array(&s, "hstack.1d");
}

#[test]
fn split_matches_numpy() {
    let pieces = a6().split(&[2, 4], 0).unwrap();
    assert_eq!(pieces.len(), 3);
    for (n, piece) in pieces.iter().enumerate() {
        check_array(piece, &format!("split.1d.{n}"));
    }
    let pieces = g46().split(&[1, 3], 0).unwrap();
    assert_eq!(pieces.len(), 3);
    for (n, piece) in pieces.iter().enumerate() {
        check_array(piece, &format!("split.axis0.{n}"));
    }
    let pieces = g46().split(&[2, 5], 1).unwrap();
    assert_eq!(pieces.len(), 3);
    for (n, piece) in pieces.iter().enumerate() {
        check_array(piece, &format!("split.axis1.{n}"));
    }
}

#[test]
fn repeat_matches_numpy() {
    check_array(&a6().repeat(3), "repeat.1d.r3");
    check_array(&m23().repeat(2), "repeat.2d.r2");
    assert_ints(&[a6().repeat(0).size() as i64], "repeat.r0.len");
}

#[test]
fn tile_matches_numpy() {
    check_array(&v3().tile(&[3]).unwrap(), "tile.1d.r3");
    check_array(&v3().tile(&[2, 2]).unwrap(), "tile.1d.r2x2");
    check_array(&m22().tile(&[2, 3]).unwrap(), "tile.2d.r2x3");
    check_array(&m22().tile(&[2]).unwrap(), "tile.2d.r2");
}

#[test]
fn sort_matches_numpy() {
    assert_ints(i6().sort().data(), "sort.small");
    let sorted = il_arr().sort();
    let data = sorted.data();
    let n = data.len();
    assert_ints(&data[..5], "sort.large.head");
    assert_ints(&data[n - 5..], "sort.large.tail");
    assert_ints(&[data.iter().sum()], "sort.large.sum");
    assert_ints(&[weighted_sum(data)], "sort.large.wsum");
}

#[test]
fn unique_matches_numpy() {
    // numpy.unique sorts the values; the result is [-2, 0, 3, 5].
    assert_ints(i6().unique().data(), "unique.small");
    let uniques = il_arr().unique();
    assert_ints(uniques.data(), "unique.large");
    assert_ints(&[uniques.size() as i64], "unique.large.len");
}

#[test]
fn argsort_matches_numpy() {
    // The stable order of ties: the same as numpy.argsort(kind="stable").
    assert_ints(&to_i64(&a6().argsort()), "argsort.small");
    let idx = to_i64(&fl_arr().argsort());
    let n = idx.len();
    assert_ints(&idx[..5], "argsort.large.head");
    assert_ints(&idx[n - 5..], "argsort.large.tail");
    assert_ints(&[n as i64], "argsort.large.len");
    assert_ints(&[weighted_sum(&idx)], "argsort.large.wsum");
}

#[test]
fn allclose_matches_numpy() {
    let a = arr_f(vec![1.0, 2.0, 3.0], vec![3]);
    let close = arr_f(vec![1.0 + 1e-9, 2.0 - 1e-9, 3.0], vec![3]);
    assert_ints(&[flag(a.allclose(&close, 1e-5, 1e-8))], "allclose.close");

    let far = arr_f(vec![1.0, 2.0, 3.1], vec![3]);
    assert_ints(&[flag(a.allclose(&far, 1e-5, 1e-8))], "allclose.far");

    let x = arr_f(vec![0.0, 1.0], vec![2]);
    let y = arr_f(vec![1e-9, 1.0], vec![2]);
    assert_ints(&[flag(x.allclose(&y, 0.0, 1e-8))], "allclose.atol");

    let x = arr_f(vec![100.0, 200.0], vec![2]);
    let y = arr_f(vec![100.0005, 200.0], vec![2]);
    assert_ints(&[flag(x.allclose(&y, 1e-5, 0.0))], "allclose.rtol");
}

#[test]
fn any_matches_numpy() {
    let b = NdArray::new(vec![false, true, false], vec![3]).unwrap();
    assert_ints(&[flag(b.any())], "any.mixed");
    let b = NdArray::new(vec![false, false], vec![2]).unwrap();
    assert_ints(&[flag(b.any())], "any.none");
    assert_ints(&[flag(mask_arr().any())], "any.large");
}

#[test]
fn all_matches_numpy() {
    let b = NdArray::new(vec![true, false, true], vec![3]).unwrap();
    assert_ints(&[flag(b.all())], "all.mixed");
    let b = NdArray::new(vec![true, true], vec![2]).unwrap();
    assert_ints(&[flag(b.all())], "all.every");
    assert_ints(&[flag(mask_arr().all())], "all.large");
}
