//! Reference-value tests for `arraykit::array/stats`.
//!
//! Expected values come from NumPy 2.4.6 and SciPy 1.17.1. The expression of
//! each value is in the `source` column of `fixtures/stats_reference.tsv`,
//! which `tools/reference/arraykit/generate.py` writes. No Python runs in CI:
//! the fixture is the only source of the expected values.
//!
//! The inputs are the same integer formulas as in the generator, so the Rust
//! side rebuilds them exactly.

use arraykit::array::NdArray;

const FIXTURE: &str = include_str!("fixtures/stats_reference.tsv");

/// Relative tolerance of every check in this file.
const TOL: f64 = 1e-9;

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

/// The expected number stored under `key`. `nan` parses to NaN.
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

/// Check one number against its expected value with a relative tolerance.
/// A NaN expected value needs a NaN result.
fn check(actual: f64, expected: f64, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}: got {actual}, expected NaN");
        return;
    }
    assert!(
        (actual - expected).abs() <= TOL * expected.abs().max(1e-300),
        "{label}: got {actual}, expected {expected}"
    );
}

/// Check `actual` against the fixture value `key`.
fn assert_close(actual: f64, key: &str) {
    check(actual, reference(key), key);
}

/// Check each element of `actual` against the fixture list `key`.
fn assert_close_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "{key}: length");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        check(*a, *e, &format!("{key}[{i}]"));
    }
}

/// `n` values of `value(i)` for `i` in `0..n`.
fn formula(n: usize, value: impl Fn(usize) -> f64) -> Vec<f64> {
    (0..n).map(value).collect()
}

/// The 8-value input with a tie (4.0 three times) and a negative value.
const SMALL: [f64; 8] = [2.0, 4.0, 4.0, -3.0, 1.5, 0.5, 4.0, 6.0];
/// The 8-value input with only positive values, for the means of logs and reciprocals.
const SMALL_POS: [f64; 8] = [2.0, 4.0, 4.0, 1.5, 0.5, 4.0, 6.0, 3.0];

fn small() -> NdArray<f64> {
    NdArray::new(SMALL.to_vec(), vec![SMALL.len()]).unwrap()
}

fn small_pos() -> NdArray<f64> {
    NdArray::new(SMALL_POS.to_vec(), vec![SMALL_POS.len()]).unwrap()
}

/// 10,000 values in multiples of 0.25.
fn large() -> NdArray<f64> {
    let data = formula(10_000, |i| ((i * 37 + 11) % 101) as f64 / 4.0);
    NdArray::new(data, vec![10_000]).unwrap()
}

/// 10,000 positive values in multiples of 0.25.
fn large_pos() -> NdArray<f64> {
    let data = formula(10_000, |i| (((i * 37 + 11) % 101) + 1) as f64 / 4.0);
    NdArray::new(data, vec![10_000]).unwrap()
}

/// The 4 x 5 grid of the axis reductions.
fn grid() -> NdArray<f64> {
    let data = formula(20, |i| ((i * 37 + 11) % 101) as f64 / 4.0);
    NdArray::new(data, vec![4, 5]).unwrap()
}

/// The 3 x 20 matrix of `cov` and `corrcoef`.
fn mat() -> NdArray<f64> {
    let mut data = Vec::with_capacity(60);
    for i in 0..3 {
        for j in 0..20 {
            data.push(((i * 31 + j * 17 + i * j) % 23) as f64 / 7.0);
        }
    }
    NdArray::new(data, vec![3, 20]).unwrap()
}

/// The (name, input) pairs of the scalar tests: `small` and `large`.
fn cases() -> [(&'static str, NdArray<f64>); 2] {
    [("small", small()), ("large", large())]
}

fn cases_pos() -> [(&'static str, NdArray<f64>); 2] {
    [("small", small_pos()), ("large", large_pos())]
}

// ---- Scalar reductions -------------------------------------------------

// numpy: np.argmax(x) gives the first index of the largest value.
#[test]
fn argmax_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(
            arr.argmax().unwrap() as f64,
            &format!("stats.argmax.{case}"),
        );
    }
}

// numpy: np.argmin(x) gives the first index of the smallest value.
#[test]
fn argmin_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(
            arr.argmin().unwrap() as f64,
            &format!("stats.argmin.{case}"),
        );
    }
}

// numpy: np.argmax([3., 1., 3.]) = 0. The first of two equal maxima wins.
#[test]
fn argmax_tie_returns_first_index() {
    let arr = NdArray::new(vec![3.0f64, 1.0, 3.0], vec![3]).unwrap();
    assert_close(arr.argmax().unwrap() as f64, "stats.argmax.tie");
}

// numpy: np.argmin([1., 3., 1.]) = 0. The first of two equal minima wins.
#[test]
fn argmin_tie_returns_first_index() {
    let arr = NdArray::new(vec![1.0f64, 3.0, 1.0], vec![3]).unwrap();
    assert_close(arr.argmin().unwrap() as f64, "stats.argmin.tie");
}

// numpy: np.sum(np.array([1, 2, 3], dtype=np.int32)) = 6.
// numpy: np.sum(np.array([2147483647, 1], dtype=np.int32)) = 2147483648 with dtype int64.
// NumPy widens the int32 sum to int64. `checked_sum` keeps the type `i32` and
// returns an error on overflow. This is a deliberate difference from NumPy.
#[test]
fn checked_sum_matches_numpy() {
    let small = NdArray::new(vec![1i32, 2, 3], vec![3]).unwrap();
    assert_close(
        small.checked_sum().unwrap() as f64,
        "stats.checked_sum.int32.ok",
    );

    // NumPy's answer for the same input, widened to int64. This is the
    // value that `checked_sum` does not return: it errors instead.
    assert_eq!(
        reference("stats.checked_sum.int32.overflow.value"),
        2147483648.0
    );
    assert_eq!(raw("stats.checked_sum.int32.overflow.dtype"), "int64");

    let big = NdArray::new(vec![i32::MAX, 1], vec![2]).unwrap();
    assert!(
        big.checked_sum().is_err(),
        "checked_sum must error on i32 overflow"
    );
}

// scipy.stats.variation(x, ddof=1) = std(ddof=1) / mean. The inputs are positive.
#[test]
fn coefficient_of_variation_matches_scipy() {
    for (case, arr) in cases_pos() {
        assert_close(
            arr.coefficient_of_variation(),
            &format!("stats.coefficient_of_variation.{case}"),
        );
    }
}

// scipy.stats.gmean(x) for positive values.
#[test]
fn geometric_mean_matches_scipy() {
    for (case, arr) in cases_pos() {
        assert_close(
            arr.geometric_mean(),
            &format!("stats.geometric_mean.{case}"),
        );
    }
}

// scipy.stats.hmean(x) for positive values.
#[test]
fn harmonic_mean_matches_scipy() {
    for (case, arr) in cases_pos() {
        assert_close(arr.harmonic_mean(), &format!("stats.harmonic_mean.{case}"));
    }
}

// scipy.stats.iqr(x), linear interpolation.
#[test]
fn iqr_matches_scipy() {
    for (case, arr) in cases() {
        assert_close(arr.iqr().unwrap(), &format!("stats.iqr.{case}"));
    }
}

// scipy.stats.kurtosis(x, bias=False), Fisher definition.
#[test]
fn kurtosis_matches_scipy() {
    for (case, arr) in cases() {
        assert_close(arr.kurtosis(), &format!("stats.kurtosis.{case}"));
    }
}

// numpy: np.max(x).
#[test]
fn max_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.max().unwrap(), &format!("stats.max.{case}"));
    }
}

// numpy: np.min(x).
#[test]
fn min_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.min().unwrap(), &format!("stats.min.{case}"));
    }
}

// numpy: np.mean(x).
#[test]
fn mean_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.mean(), &format!("stats.mean.{case}"));
    }
}

// numpy: np.mean(x), computed in f64.
#[test]
fn mean_f64_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.mean_f64(), &format!("stats.mean_f64.{case}"));
    }
}

// numpy: np.median(x).
#[test]
fn median_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.median().unwrap(), &format!("stats.median.{case}"));
    }
}

// scipy.stats.mode(x, keepdims=False).mode. Of two values with the same count, the smallest wins.
#[test]
fn mode_matches_scipy() {
    for (case, arr) in cases() {
        assert_close(arr.mode().unwrap(), &format!("stats.mode.{case}"));
    }
    let tie = NdArray::new(vec![3.0f64, 1.0, 1.0, 3.0], vec![4]).unwrap();
    assert_close(tie.mode().unwrap(), "stats.mode.tie");
}

// scipy.stats.moment(x, moment=n): the mean of (x - mean)^n.
#[test]
fn moment_matches_scipy() {
    for (case, arr) in cases() {
        assert_close(arr.moment(2), &format!("stats.moment.2.{case}"));
        assert_close(arr.moment(3), &format!("stats.moment.3.{case}"));
    }
}

// numpy: np.percentile(x, q), linear interpolation.
#[test]
fn percentile_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(
            arr.percentile(25.0).unwrap(),
            &format!("stats.percentile.25.{case}"),
        );
        assert_close(
            arr.percentile(90.0).unwrap(),
            &format!("stats.percentile.90.{case}"),
        );
    }
}

// numpy: np.quantile(x, 0.9). The method is `percentile(q * 100)`.
#[test]
fn quantile_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(
            arr.quantile(0.9).unwrap(),
            &format!("stats.quantile.0.9.{case}"),
        );
    }
}

// scipy.stats.sem(x): the standard error of the mean with ddof=1.
#[test]
fn sem_matches_scipy() {
    for (case, arr) in cases() {
        assert_close(arr.sem(), &format!("stats.sem.{case}"));
    }
}

// scipy.stats.skew(x, bias=False).
#[test]
fn skew_matches_scipy() {
    for (case, arr) in cases() {
        assert_close(arr.skew(), &format!("stats.skew.{case}"));
    }
}

// numpy: np.std(x, ddof=1).
#[test]
fn std_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.std(), &format!("stats.std.{case}"));
    }
}

// numpy: np.std(x), ddof=0.
#[test]
fn std_pop_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.std_pop(), &format!("stats.std_pop.{case}"));
    }
}

// numpy: np.sum(x).
#[test]
fn sum_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.sum(), &format!("stats.sum.{case}"));
    }
}

// scipy.stats.trim_mean(x, 0.25).
#[test]
fn trim_mean_matches_scipy() {
    for (case, arr) in cases() {
        assert_close(arr.trim_mean(0.25), &format!("stats.trim_mean.0.25.{case}"));
    }
}

// numpy: np.var(x, ddof=1).
#[test]
fn var_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.var(), &format!("stats.var.{case}"));
    }
}

// numpy: np.var(x), ddof=0.
#[test]
fn var_pop_matches_numpy() {
    for (case, arr) in cases() {
        assert_close(arr.var_pop(), &format!("stats.var_pop.{case}"));
    }
}

// Empty input: np.mean([]) and np.var([], ddof=1) are NaN.
#[test]
fn empty_mean_and_var_are_nan() {
    let arr = NdArray::<f64>::new(vec![], vec![0]).unwrap();
    assert_close(arr.mean(), "stats.mean.empty");
    assert_close(arr.var(), "stats.var.empty");
}

// numpy: np.mean([1., 2.]) = 1.5 and np.var([1., 2., 3., 4.], ddof=1) = 1.6666666666666667.
#[test]
fn float_mean_of_one_and_two() {
    let a = NdArray::new(vec![1.0f64, 2.0], vec![2]).unwrap();
    assert_close(a.mean(), "stats.mean.one_two");

    let b = NdArray::new(vec![1.0f64, 2.0, 3.0, 4.0], vec![4]).unwrap();
    assert_close(b.var(), "stats.var.one_to_four");
}

// numpy: np.min and np.max return NaN when any input is NaN, in any position.
#[test]
fn min_max_with_nan_do_not_depend_on_order() {
    let inputs = [
        ("a", [1.0, f64::NAN, 3.0]),
        ("b", [f64::NAN, 1.0, 3.0]),
        ("c", [3.0, 1.0, f64::NAN]),
    ];
    for (name, values) in inputs {
        let arr = NdArray::new(values.to_vec(), vec![3]).unwrap();
        assert_close(arr.min().unwrap(), &format!("stats.min.nan.{name}"));
        assert_close(arr.max().unwrap(), &format!("stats.max.nan.{name}"));
    }
}

// ---- Lists and grids ---------------------------------------------------

// numpy: np.cumsum(x), and the first and last 5 values of the large input.
#[test]
fn cumsum_matches_numpy() {
    assert_close_list(small().cumsum().data(), "stats.cumsum.small");
    let c = large().cumsum();
    let data = c.data();
    assert_close_list(&data[..5], "stats.cumsum.large.head");
    assert_close_list(&data[data.len() - 5..], "stats.cumsum.large.tail");
}

// numpy: np.cumprod(x) of the small positive input.
#[test]
fn cumprod_matches_numpy() {
    assert_close_list(small_pos().cumprod().data(), "stats.cumprod.small");
}

// scipy.stats.zscore(x, ddof=1), which uses the sample standard deviation as `std` does.
#[test]
fn zscore_matches_scipy() {
    assert_close_list(small().zscore().data(), "stats.zscore.small");
    let z = large().zscore();
    let data = z.data();
    assert_close_list(&data[..5], "stats.zscore.large.head");
    assert_close_list(&data[data.len() - 5..], "stats.zscore.large.tail");
}

// numpy.histogram(x, bins=5): counts and bin edges.
#[test]
fn histogram_matches_numpy() {
    for (case, arr) in cases() {
        let (counts, edges) = arr.histogram(5);
        let counts: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        assert_close_list(&counts, &format!("stats.histogram.{case}.counts"));
        assert_close_list(&edges, &format!("stats.histogram.{case}.edges"));
    }
}

/// Check one axis reduction of the grid against `key.0` and `key.1`.
fn check_axis(result: impl Fn(usize) -> NdArray<f64>, key: &str) {
    assert_close_list(result(0).data(), &format!("{key}.0"));
    assert_close_list(result(1).data(), &format!("{key}.1"));
}

// numpy: np.sum(grid, axis=ax).
#[test]
fn sum_axis_matches_numpy() {
    check_axis(|ax| grid().sum_axis(ax).unwrap(), "stats.sum_axis");
}

// numpy: np.mean(grid, axis=ax).
#[test]
fn mean_axis_matches_numpy() {
    check_axis(|ax| grid().mean_axis(ax).unwrap(), "stats.mean_axis");
}

// numpy: np.max(grid, axis=ax).
#[test]
fn max_axis_matches_numpy() {
    check_axis(|ax| grid().max_axis(ax).unwrap(), "stats.max_axis");
}

// numpy: np.min(grid, axis=ax).
#[test]
fn min_axis_matches_numpy() {
    check_axis(|ax| grid().min_axis(ax).unwrap(), "stats.min_axis");
}

// numpy: np.var(grid, axis=ax, ddof=1).
#[test]
fn var_axis_matches_numpy() {
    check_axis(|ax| grid().var_axis(ax).unwrap(), "stats.var_axis");
}

// numpy: np.std(grid, axis=ax, ddof=1).
#[test]
fn std_axis_matches_numpy() {
    check_axis(|ax| grid().std_axis(ax).unwrap(), "stats.std_axis");
}

// numpy: np.cov(mat), rows are variables.
#[test]
fn cov_matches_numpy() {
    let c = mat().cov().unwrap();
    assert_close_list(c.data(), "stats.cov.matrix");
}

// numpy: np.corrcoef(mat), rows are variables.
#[test]
fn corrcoef_matches_numpy() {
    let c = mat().corrcoef().unwrap();
    assert_close_list(c.data(), "stats.corrcoef.matrix");
}
