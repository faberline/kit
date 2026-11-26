//! Reference-value tests for `arraykit::array/search` and `arraykit::array/ops`.
//!
//! Expected values come from NumPy 2.4.6. The expression of each value is in
//! the `source` column of `fixtures/search_ops_reference.tsv`, which
//! `tools/reference/arraykit/generate.py` writes. No Python runs in CI: the
//! fixture is the only source of the expected values.
//!
//! The inputs are the same integer formulas as in the generator, built with
//! the same operations, so the Rust side holds the same numbers.

use arraykit::array::NdArray;

const FIXTURE: &str = include_str!("fixtures/search_ops_reference.tsv");

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

/// The expected single number stored under `key`.
fn scalar(key: &str) -> f64 {
    reference_list(key)[0]
}

/// Check one number with a relative tolerance. An infinite expected value
/// needs the same infinite result. A zero expected value needs an absolute
/// error of at most 1e-12. A NaN expected value needs a NaN result.
fn check(actual: f64, expected: f64, rel_tol: f64, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}: got {actual}, expected NaN");
    } else if expected.is_infinite() {
        assert_eq!(
            actual, expected,
            "{label}: got {actual}, expected {expected}"
        );
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
fn check_big(values: &[f64], key: &str) {
    assert_ints(&[values.len() as i64], &format!("{key}.len"));
    check(
        values.iter().sum::<f64>(),
        scalar(&format!("{key}.sum")),
        TOL,
        key,
    );
    assert_close_list(&values[..5], &format!("{key}.head"), TOL);
}

fn arr_f(data: Vec<f64>, dims: Vec<usize>) -> NdArray<f64> {
    NdArray::new(data, dims).unwrap()
}

fn to_i64(values: &[usize]) -> Vec<i64> {
    values.iter().map(|&v| v as i64).collect()
}

fn to_f64(values: &[usize]) -> Vec<f64> {
    values.iter().map(|&v| v as f64).collect()
}

/// The sum of `(i + 1) * x[i]`, in floats.
fn weighted_sum(values: &[f64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as f64 + 1.0) * x)
        .sum()
}

/// The sum of `(i + 1) * x[i]`, in integers.
fn weighted_sum_i64(values: &[i64]) -> i64 {
    values
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as i64 + 1) * x)
        .sum()
}

/// The inputs of the small cases (the same values as in the generator).
fn a6() -> NdArray<f64> {
    arr_f(vec![3.0, -1.5, 3.0, 0.5, -1.5, 2.0], vec![6])
}

fn b6() -> NdArray<f64> {
    arr_f(vec![0.5, 2.0, -4.0, 8.0, 0.25, -1.0], vec![6])
}

fn m23() -> NdArray<f64> {
    arr_f(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
}

fn m13() -> NdArray<f64> {
    arr_f(vec![0.5, 1.5, 2.5], vec![1, 3])
}

fn v3() -> NdArray<f64> {
    arr_f(vec![1.0, 2.0, 3.0], vec![3])
}

fn c21() -> NdArray<f64> {
    arr_f(vec![10.0, 20.0], vec![2, 1])
}

/// The 3 x 4 grid: `((i * 7 + 3) % 11) - 4` for i in 0..12 (has a tie).
fn gs() -> NdArray<f64> {
    let data: Vec<f64> = (0..12_i64)
        .map(|i| ((i * 7 + 3) % 11) as f64 - 4.0)
        .collect();
    arr_f(data, vec![3, 4])
}

/// The 2 x 3 x 4 grid: the same formula for i in 0..24.
fn xs() -> NdArray<f64> {
    let data: Vec<f64> = (0..24_i64)
        .map(|i| ((i * 7 + 3) % 11) as f64 - 4.0)
        .collect();
    arr_f(data, vec![2, 3, 4])
}

/// The large float input: 10,000 values in multiples of 0.25, many ties.
fn fl_data() -> Vec<f64> {
    (0..N).map(|i| ((i * 37 + 11) % 101) as f64 / 4.0).collect()
}

fn fl_arr() -> NdArray<f64> {
    arr_f(fl_data(), vec![N as usize])
}

/// The large divisor: `(((i * 53 + 7) % 97) + 1) / 8` for i in 0..10,000.
fn fd_arr() -> NdArray<f64> {
    let data: Vec<f64> = (0..N)
        .map(|i| (((i * 53 + 7) % 97) + 1) as f64 / 8.0)
        .collect();
    arr_f(data, vec![N as usize])
}

/// The large boolean input: true where i % 3 == 0.
fn mask_arr() -> NdArray<bool> {
    let data: Vec<bool> = (0..N).map(|i| i % 3 == 0).collect();
    NdArray::new(data, vec![N as usize]).unwrap()
}

#[test]
fn searchsorted_matches_numpy() {
    let s6 = arr_f(vec![-1.5, -1.5, 0.5, 2.0, 3.0, 3.0], vec![6]);
    let q8 = [-2.0, -1.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0];
    let left = s6.searchsorted(&q8, "left").unwrap();
    assert_ints(&to_i64(&left), "search.searchsorted.small.left");
    let right = s6.searchsorted(&q8, "right").unwrap();
    assert_ints(&to_i64(&right), "search.searchsorted.small.right");

    let mut sorted = fl_data();
    sorted.sort_by(f64::total_cmp);
    let sl = arr_f(sorted, vec![N as usize]);
    let ql: Vec<f64> = (0..2000_i64)
        .map(|i| ((i * 29 + 3) % 117) as f64 / 4.0 - 1.0)
        .collect();
    let left = sl.searchsorted(&ql, "left").unwrap();
    check_big(&to_f64(&left), "search.searchsorted.large.left");
    let right = sl.searchsorted(&ql, "right").unwrap();
    check_big(&to_f64(&right), "search.searchsorted.large.right");
}

#[test]
fn sort_axis_matches_numpy() {
    check_array(&a6().sort_axis(0).unwrap(), "search.sort_axis.1d");
    for axis in 0..2 {
        let key = format!("search.sort_axis.2d.axis{axis}");
        check_array(&gs().sort_axis(axis).unwrap(), &key);
    }
    for axis in 0..3 {
        let key = format!("search.sort_axis.3d.axis{axis}");
        check_array(&xs().sort_axis(axis).unwrap(), &key);
    }

    let sorted = fl_arr().sort_axis(0).unwrap();
    let data = sorted.data();
    let n = data.len();
    assert_close_list(&data[..5], "search.sort_axis.large.head", TOL);
    assert_close_list(&data[n - 5..], "search.sort_axis.large.tail", TOL);
    check(
        data.iter().sum::<f64>(),
        scalar("search.sort_axis.large.sum"),
        TOL,
        "search.sort_axis.large.sum",
    );
    check(
        weighted_sum(data),
        scalar("search.sort_axis.large.wsum"),
        TOL,
        "search.sort_axis.large.wsum",
    );
}

#[test]
fn argsort_axis_matches_numpy() {
    check_array(&a6().argsort_axis(0).unwrap(), "search.argsort_axis.1d");
    for axis in 0..2 {
        let key = format!("search.argsort_axis.2d.axis{axis}");
        check_array(&gs().argsort_axis(axis).unwrap(), &key);
    }
    for axis in 0..3 {
        let key = format!("search.argsort_axis.3d.axis{axis}");
        check_array(&xs().argsort_axis(axis).unwrap(), &key);
    }

    let idx = fl_arr().argsort_axis(0).unwrap();
    let data = idx.data();
    let n = data.len();
    assert_ints(&[n as i64], "search.argsort_axis.large.len");
    assert_close_list(&data[..5], "search.argsort_axis.large.head", TOL);
    assert_close_list(&data[n - 5..], "search.argsort_axis.large.tail", TOL);
    check(
        weighted_sum(data),
        scalar("search.argsort_axis.large.wsum"),
        TOL,
        "search.argsort_axis.large.wsum",
    );
}

#[test]
fn unique_full_matches_numpy() {
    let cases = [
        ("small", a6()),
        (
            "nan",
            arr_f(vec![1.0, f64::NAN, 0.5, f64::NAN, 1.0], vec![5]),
        ),
        ("2d", arr_f(vec![3.0, 1.0, 3.0, 2.0, 1.0, 0.0], vec![2, 3])),
    ];
    for (case, input) in &cases {
        let u = input.unique_full(true, true, true);
        assert_close_list(
            u.values.data(),
            &format!("search.unique_full.{case}.values"),
            TOL,
        );
        assert_ints(
            &to_i64(u.indices.as_ref().unwrap()),
            &format!("search.unique_full.{case}.indices"),
        );
        assert_ints(
            &to_i64(u.inverse.as_ref().unwrap()),
            &format!("search.unique_full.{case}.inverse"),
        );
        assert_ints(
            &to_i64(u.counts.as_ref().unwrap()),
            &format!("search.unique_full.{case}.counts"),
        );
    }

    let u = fl_arr().unique_full(true, true, true);
    assert_close_list(u.values.data(), "search.unique_full.large.values", TOL);
    assert_ints(
        &to_i64(u.indices.as_ref().unwrap()),
        "search.unique_full.large.indices",
    );
    assert_ints(
        &to_i64(u.counts.as_ref().unwrap()),
        "search.unique_full.large.counts",
    );
    let inverse = to_i64(u.inverse.as_ref().unwrap());
    assert_ints(
        &[inverse.iter().sum::<i64>()],
        "search.unique_full.large.inverse.sum",
    );
    assert_ints(
        &[weighted_sum_i64(&inverse)],
        "search.unique_full.large.inverse.wsum",
    );
}

#[test]
fn where_select_matches_numpy() {
    let cond = NdArray::new(vec![true, false, true, false, false, true], vec![6]).unwrap();
    let y = arr_f(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![6]);
    let r = NdArray::where_select(&cond, &a6(), &y).unwrap();
    check_array(&r, "search.where_select.small");

    let m = m23();
    let cond = NdArray::new(m.data().iter().map(|&v| v > 3.0).collect(), vec![2, 3]).unwrap();
    let neg: Vec<f64> = m.data().iter().map(|&v| -v).collect();
    let r = NdArray::where_select(&cond, &m, &arr_f(neg, vec![2, 3])).unwrap();
    check_array(&r, "search.where_select.2d");

    let cond = mask_arr();
    let neg: Vec<f64> = fl_data().iter().map(|&v| -v - 1.0).collect();
    let r = NdArray::where_select(&cond, &fl_arr(), &arr_f(neg, vec![N as usize])).unwrap();
    check_big(r.data(), "search.where_select.large");
}

#[test]
fn nonzero_matches_numpy() {
    let c = NdArray::new(vec![false, true, true, false, true], vec![5]).unwrap();
    let nz = NdArray::<f64>::nonzero(&c);
    assert_ints(&to_i64(&nz[0]), "search.nonzero.1d");

    let c = NdArray::new((0..12_i64).map(|i| i % 3 == 0).collect(), vec![3, 4]).unwrap();
    let nz = NdArray::<f64>::nonzero(&c);
    for (axis, indices) in nz.iter().enumerate() {
        assert_ints(&to_i64(indices), &format!("search.nonzero.2d.axis{axis}"));
    }

    let c = NdArray::new((0..24_i64).map(|i| i % 5 == 0).collect(), vec![2, 3, 4]).unwrap();
    let nz = NdArray::<f64>::nonzero(&c);
    for (axis, indices) in nz.iter().enumerate() {
        assert_ints(&to_i64(indices), &format!("search.nonzero.3d.axis{axis}"));
    }

    let nz = NdArray::<f64>::nonzero(&mask_arr());
    check_big(&to_f64(&nz[0]), "search.nonzero.large");

    let c = NdArray::new(vec![false; 4], vec![4]).unwrap();
    let nz = NdArray::<f64>::nonzero(&c);
    assert_ints(&[nz[0].len() as i64], "search.nonzero.none.len");
}

/// The four checks of one binary operation: the same shape, a row, a column,
/// an outer product and 10,000 values, against the fixture keys `ops.<name>.*`.
fn check_binary(name: &str, op: impl Fn(&NdArray<f64>, &NdArray<f64>) -> NdArray<f64>) {
    check_array(&op(&a6(), &b6()), &format!("ops.{name}.same"));
    check_array(&op(&m23(), &v3()), &format!("ops.{name}.row"));
    check_array(&op(&m23(), &c21()), &format!("ops.{name}.col"));
    check_array(&op(&c21(), &m13()), &format!("ops.{name}.outer"));
    check_big(
        op(&fl_arr(), &fd_arr()).data(),
        &format!("ops.{name}.large"),
    );
}

#[test]
fn add_matches_numpy() {
    check_binary("add", |a, b| a.add(b).unwrap());
}

#[test]
fn sub_matches_numpy() {
    check_binary("sub", |a, b| a.sub(b).unwrap());
}

#[test]
fn mul_matches_numpy() {
    check_binary("mul", |a, b| a.mul(b).unwrap());
}

#[test]
fn div_matches_numpy() {
    check_binary("div", |a, b| a.div(b).unwrap());
    // A zero divisor gives inf, -inf and NaN, as NumPy does.
    let zero = arr_f(vec![0.0, 0.0, 0.0], vec![3]);
    let r = arr_f(vec![1.0, -1.0, 0.0], vec![3]).div(&zero).unwrap();
    check_array(&r, "ops.div.zero");
}

#[test]
fn add_scalar_matches_numpy() {
    check_array(&a6().add_scalar(2.5), "ops.add_scalar.small");
    check_array(&m23().add_scalar(2.5), "ops.add_scalar.2d");
    check_big(fl_arr().add_scalar(1.25).data(), "ops.add_scalar.large");
}

#[test]
fn mul_scalar_matches_numpy() {
    check_array(&a6().mul_scalar(-0.5), "ops.mul_scalar.small");
    check_array(&m23().mul_scalar(-0.5), "ops.mul_scalar.2d");
    check_big(fl_arr().mul_scalar(-3.0).data(), "ops.mul_scalar.large");
}

#[test]
fn binary_op_rejects_incompatible_shapes() {
    // NumPy raises ValueError for (2, 3) + (4,); no fixture value is needed.
    let other = arr_f(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    assert!(m23().add(&other).is_err());
}
