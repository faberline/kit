//! Reference-value tests for `arraykit::array/window` and `arraykit::array/math`.
//!
//! Expected values come from NumPy 2.4.6. The expression of each value is in
//! the `source` column of `fixtures/window_math_reference.tsv`, which
//! `tools/reference/arraykit/generate.py` writes. No Python runs in CI: the
//! fixture is the only source of the expected values.
//!
//! The inputs are the same integer formulas as in the generator, built with
//! the same operations, so the Rust side holds the same numbers.

use arraykit::array::NdArray;

const FIXTURE: &str = include_str!("fixtures/window_math_reference.tsv");

/// Relative tolerance of every check in this file.
const TOL: f64 = 1e-9;

/// Length of the large input.
const N: usize = 10000;

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

/// The expected number stored under `key`.
fn reference(key: &str) -> f64 {
    let value = raw(key);
    value
        .parse()
        .unwrap_or_else(|_| panic!("fixture key {key}: bad number {value}"))
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

/// Check `actual` against the fixture value `key`.
fn assert_close(actual: f64, key: &str, rel_tol: f64) {
    check(actual, reference(key), rel_tol, key);
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

/// Check a long result `y` against the head, the tail and the sum of the
/// absolute values stored under `key` (`.head`, `.tail`, `.abssum`).
fn check_summary(y: &[f64], key: &str, rel_tol: f64) {
    let n = y.len();
    assert_close_list(&y[..5], &format!("{key}.head"), rel_tol);
    assert_close_list(&y[n - 5..], &format!("{key}.tail"), rel_tol);
    let abssum: f64 = y.iter().map(|v| v.abs()).sum();
    assert_close(abssum, &format!("{key}.abssum"), rel_tol);
}

/// Check the entries that arraykit sets itself: all NaN, or all 0.0.
fn check_prefix(d: &[f64], nan: bool) {
    assert!(
        d.iter().all(|&v| if nan { v.is_nan() } else { v == 0.0 }),
        "prefix must be {}",
        if nan { "NaN" } else { "0.0" }
    );
}

/// A one-dimensional array with the given values.
fn arr(data: Vec<f64>) -> NdArray<f64> {
    let n = data.len();
    NdArray::new(data, shape1(n)).unwrap()
}

/// The shape of a one-dimensional array of length n.
fn shape1(n: usize) -> Vec<usize> {
    vec![n]
}

/// The signal value at index i: the same integer formula as the generator.
fn sig(i: usize) -> f64 {
    ((i * 37 + 11) % 101) as f64 / 4.0
}

/// The signal values 0..n (ten values for n = 10, the large input for n = N).
fn signal(n: usize) -> Vec<f64> {
    (0..n).map(sig).collect()
}

/// The signed input: the signal minus 12.5.
fn signed(n: usize) -> Vec<f64> {
    (0..n).map(|i| sig(i) - 12.5).collect()
}

/// The positive input: the integer formula plus one, divided by four.
fn positive(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i * 37 + 11) % 101) + 1) as f64 / 4.0)
        .collect()
}

/// Integer input: the integer formula minus 50, times `scale`.
fn int_values(n: usize, scale: i64) -> Vec<i64> {
    (0..n)
        .map(|i| (((i * 37 + 11) % 101) as i64 - 50) * scale)
        .collect()
}

/// Check a rolling result `op` (the window of ten values and the window of
/// ten thousand values). The first `window - 1` entries are arraykit's own:
/// NaN when `nan_prefix`, else 0.0.
fn check_rolling_op(
    name: &str,
    nan_prefix: bool,
    op: impl Fn(&NdArray<f64>, usize) -> NdArray<f64>,
) {
    let small = arr(signal(10));
    let y = op(&small, 3);
    let d = y.data();
    check_prefix(&d[..2], nan_prefix);
    assert_close_list(&d[2..], &format!("window.{name}.w3"), TOL);

    let large = arr(signal(N));
    let y = op(&large, 50);
    let d = y.data();
    check_prefix(&d[..49], nan_prefix);
    check_summary(&d[49..], &format!("window.{name}.w50"), TOL);
}

/// Check an element-wise function on the small input (ten values) and on
/// the large input (summary of head, tail and absolute sum).
fn check_map(
    name: &str,
    small: Vec<f64>,
    large: Vec<f64>,
    op: impl Fn(&NdArray<f64>) -> NdArray<f64>,
) {
    let y = op(&arr(small));
    assert_close_list(y.data(), &format!("math.{name}.small"), TOL);
    let y = op(&arr(large));
    check_summary(y.data(), &format!("math.{name}.large"), TOL);
}

#[test]
fn rolling_sum_matches_numpy() {
    check_rolling_op("sum", false, |x, w| x.rolling(w).unwrap().sum());

    // min_periods 2 with window 4: entries from index 1 on are valid.
    let small = arr(signal(10));
    let y = small.rolling(4).unwrap().min_periods(2).sum();
    let d = y.data();
    check_prefix(&d[..1], false);
    assert_close_list(&d[1..], "window.sum.w4mp2", TOL);
}

#[test]
fn rolling_mean_matches_numpy() {
    check_rolling_op("mean", false, |x, w| x.rolling(w).unwrap().mean());

    let small = arr(signal(10));
    let y = small.rolling(4).unwrap().min_periods(2).mean();
    let d = y.data();
    check_prefix(&d[..1], false);
    assert_close_list(&d[1..], "window.mean.w4mp2", TOL);
}

#[test]
fn rolling_min_matches_numpy() {
    check_rolling_op("min", true, |x, w| x.rolling(w).unwrap().min());
}

#[test]
fn rolling_max_matches_numpy() {
    check_rolling_op("max", true, |x, w| x.rolling(w).unwrap().max());
}

#[test]
fn rolling_std_matches_numpy() {
    check_rolling_op("std", true, |x, w| x.rolling(w).unwrap().std());
}

#[test]
fn rolling_var_matches_numpy() {
    check_rolling_op("var", true, |x, w| x.rolling(w).unwrap().var());
}

#[test]
fn rolling_count_matches_numpy() {
    let small = arr(signal(10));
    let counts = small.rolling(4).unwrap().count();
    let got: Vec<i64> = counts.iter().map(|&c| c as i64).collect();
    assert_ints(&got, "window.count.w4");
}

#[test]
fn rolling_matches_numpy() {
    let small = arr(signal(10));
    let w1 = small.rolling(1).unwrap().sum();
    assert_close_list(w1.data(), "window.rolling.w1", TOL);

    let full = small.rolling(10).unwrap().sum();
    assert_close(full.data()[9], "window.rolling.full", TOL);

    // A window of zero is refused.
    assert!(small.rolling(0).is_err());
}

#[test]
fn diff_matches_numpy() {
    let small = arr(signal(10));
    assert_close_list(small.diff(1).unwrap().data(), "window.diff.n1", TOL);
    assert_close_list(small.diff(2).unwrap().data(), "window.diff.n2", TOL);

    let large = arr(signal(N));
    check_summary(large.diff(1).unwrap().data(), "window.diff.large", TOL);
}

#[test]
fn flip_matches_numpy() {
    let small = arr(signal(10));
    assert_close_list(small.flip().data(), "window.flip.small", TOL);

    let large = arr(signal(N));
    check_summary(large.flip().data(), "window.flip.large", TOL);
}

#[test]
fn roll_matches_numpy() {
    let small = arr(signal(10));
    assert_close_list(small.roll(3).data(), "window.roll.p3", TOL);
    assert_close_list(small.roll(-4).data(), "window.roll.m4", TOL);
    assert_close_list(small.roll(13).data(), "window.roll.p13", TOL);
}

#[test]
fn pad_matches_numpy() {
    let small = arr(signal(10));

    let p2 = small.pad(2).unwrap();
    assert_eq!(p2.dims(), &[14_usize]);
    assert_close_list(p2.data(), "window.pad.w2", TOL);

    let p0 = small.pad(0).unwrap();
    assert_eq!(p0.dims(), &[10_usize]);
    assert_close_list(p0.data(), "window.pad.w0", TOL);
}

#[test]
fn gradient_matches_numpy() {
    let small = arr(signal(10));
    assert_close_list(
        small.gradient().unwrap().data(),
        "window.gradient.small",
        TOL,
    );

    let large = arr(signal(N));
    check_summary(
        large.gradient().unwrap().data(),
        "window.gradient.large",
        TOL,
    );
}

#[test]
fn interp_matches_numpy() {
    let xp: Vec<f64> = (0..10).map(|i| 1.5 * i as f64).collect();
    let fp = signal(10);
    // Below the range, on a node, inside, at the end and above the range.
    let xq = [-1.0, 0.0, 0.3, 1.5, 4.4, 7.25, 13.5, 14.0, 20.0];
    let got = NdArray::<f64>::interp(&xq, &xp, &fp);
    assert_close_list(&got, "window.interp.small", TOL);

    let xp: Vec<f64> = (0..1001).map(|i| 0.5 * i as f64).collect();
    let fp = signal(1001);
    let xq: Vec<f64> = (0..N)
        .map(|j| ((j * 7919) % 5003) as f64 / 10.0 - 0.5)
        .collect();
    let got = NdArray::<f64>::interp(&xq, &xp, &fp);
    check_summary(&got, "window.interp.large", TOL);
}

#[test]
fn squeeze_matches_numpy() {
    let shape: Vec<usize> = vec![1, 3, 1, 2];
    let s = NdArray::<f64>::zeros(shape).squeeze();
    let dims: Vec<i64> = s.dims().iter().map(|&d| d as i64).collect();
    assert_ints(&dims, "window.squeeze.shape");

    // An all-ones shape gives shape [1]: arraykit has no 0-d array.
    let ones: Vec<usize> = vec![1, 1, 1];
    let o = NdArray::<f64>::zeros(ones).squeeze();
    let dims: Vec<i64> = o.dims().iter().map(|&d| d as i64).collect();
    assert_ints(&dims, "window.squeeze.ones");
}

#[test]
fn expand_dims_matches_numpy() {
    let shape: Vec<usize> = vec![2, 3];
    let m = NdArray::<f64>::zeros(shape);
    for axis in 0..3 {
        let dims: Vec<i64> = m
            .expand_dims(axis)
            .dims()
            .iter()
            .map(|&d| d as i64)
            .collect();
        assert_ints(&dims, &format!("window.expand_dims.a{axis}"));
    }
}

#[test]
fn rot90_matches_numpy() {
    // The 3 x 4 matrix with the values 0 to 11, row by row.
    let data: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let shape: Vec<usize> = vec![3, 4];
    let m = NdArray::new(data, shape).unwrap();
    for k in [0_usize, 1, 2, 3, 5] {
        let r = m.rot90(k).unwrap();
        assert_close_list(r.data(), &format!("window.rot90.k{k}.data"), TOL);
        let dims: Vec<i64> = r.dims().iter().map(|&d| d as i64).collect();
        assert_ints(&dims, &format!("window.rot90.k{k}.shape"));
    }
}

#[test]
fn sin_matches_numpy() {
    check_map("sin", signed(10), signed(N), |x| x.sin());
}

#[test]
fn cos_matches_numpy() {
    check_map("cos", signed(10), signed(N), |x| x.cos());
}

#[test]
fn tan_matches_numpy() {
    check_map("tan", signed(10), signed(N), |x| x.tan());
}

#[test]
fn exp_matches_numpy() {
    check_map("exp", signed(10), signed(N), |x| x.exp());
}

#[test]
fn sqrt_matches_numpy() {
    check_map("sqrt", positive(10), positive(N), |x| x.sqrt());
}

#[test]
fn log_matches_numpy() {
    check_map("log", positive(10), positive(N), |x| x.log());
}

#[test]
fn abs_matches_numpy() {
    check_map("abs", signed(10), signed(N), |x| x.abs());
}

#[test]
fn pow_matches_numpy() {
    check_map("pow.e25", positive(10), positive(N), |x| x.pow(2.5));
    check_map("pow.e3", signed(10), signed(N), |x| x.pow(3.0));
}

#[test]
fn clip_matches_numpy() {
    check_map("clip", signed(10), signed(N), |x| x.clip(-5.0, 5.0));
}

#[test]
fn linspace_matches_numpy() {
    let s11 = NdArray::<f64>::linspace(0.0, 1.0, 11);
    assert_close_list(s11.data(), "math.linspace.s11", TOL);

    let s = NdArray::<f64>::linspace(-5.0, 5.0, 10001);
    let d = s.data();
    assert_close_list(&d[..2], "math.linspace.s10001.first2", TOL);
    assert_close_list(&d[d.len() - 1..], "math.linspace.s10001.last", TOL);
    let abssum: f64 = d.iter().map(|v| v.abs()).sum();
    assert_close(abssum, "math.linspace.s10001.abssum", TOL);
}

#[test]
fn where_cond_matches_numpy() {
    // The condition is true where i is a multiple of 3; the other value is -x.
    let cond: Vec<bool> = (0..10).map(|i| i % 3 == 0).collect();
    let x = arr(signed(10));
    let neg = arr(signed(10).iter().map(|v| -v).collect());
    let y = x.where_cond(&cond, &neg).unwrap();
    assert_close_list(y.data(), "math.where_cond.small", TOL);

    let cond: Vec<bool> = (0..N).map(|i| i % 3 == 0).collect();
    let x = arr(signed(N));
    let neg = arr(signed(N).iter().map(|v| -v).collect());
    let y = x.where_cond(&cond, &neg).unwrap();
    check_summary(y.data(), "math.where_cond.large", TOL);
}

#[test]
fn abs_int_i32_matches_numpy() {
    let small: Vec<i32> = int_values(10, 1).iter().map(|&v| v as i32).collect();
    let y = NdArray::new(small, shape1(10)).unwrap().abs_int();
    let got: Vec<i64> = y.data().iter().map(|&v| v as i64).collect();
    assert_ints(&got, "math.abs_int.i32.small");

    let large: Vec<i32> = int_values(N, 1).iter().map(|&v| v as i32).collect();
    let y = NdArray::new(large, shape1(N)).unwrap().abs_int();
    let got: Vec<i64> = y.data().iter().map(|&v| v as i64).collect();
    assert_ints(&got[..5], "math.abs_int.i32.large.head");
    assert_ints(&got[N - 5..], "math.abs_int.i32.large.tail");
    assert_ints(&[got.iter().sum::<i64>()], "math.abs_int.i32.large.sum");
}

#[test]
fn abs_int_i64_matches_numpy() {
    // Values are scaled by 1e10, so they do not fit in i32.
    let small = int_values(10, 10_000_000_000);
    let y = NdArray::new(small, shape1(10)).unwrap().abs_int();
    assert_ints(y.data(), "math.abs_int.i64.small");

    let large = int_values(N, 10_000_000_000);
    let y = NdArray::new(large, shape1(N)).unwrap().abs_int();
    let got = y.data();
    assert_ints(&got[..5], "math.abs_int.i64.large.head");
    assert_ints(&got[N - 5..], "math.abs_int.i64.large.tail");
    assert_ints(&[got.iter().sum::<i64>()], "math.abs_int.i64.large.sum");
}
