//! Reference-value tests for `arraykit::array/ndarray` (the shape and value
//! functions) and `arraykit::array/random` (the distributions).
//!
//! The ndarray cases are checked against NumPy 2.4.6 values in
//! `fixtures/ndarray_reference.tsv`; the `source` column there is the NumPy
//! expression that made each value. The random cases are checked against the
//! SciPy closed forms (mean, variance, excess kurtosis) in the same fixture.
//! `KitRng` is not NumPy's generator, so its draws are compared by moments only.
//! No Python runs in CI: the fixture is the only source of the expected values.

use arraykit::array::{AxisSlice, KitRng, NdArray, Shape, SliceInfo};

const FIXTURE: &str = include_str!("fixtures/ndarray_reference.tsv");

/// Relative tolerance of every value check in this file.
const TOL: f64 = 1e-9;

/// Number of draws of each random case.
const DRAWS: usize = 100_000;

/// Seed of each random case.
const SEED: u64 = 20_261_123;

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
/// an absolute error of at most 1e-12.
fn check(actual: f64, expected: f64, rel_tol: f64, label: &str) {
    if expected == 0.0 {
        assert!(actual.abs() <= 1e-12, "{label}: got {actual}, expected 0");
    } else {
        assert!(
            (actual - expected).abs() <= rel_tol * expected.abs(),
            "{label}: got {actual}, expected {expected}"
        );
    }
}

/// Check each element of `actual` against the fixture list `key`.
fn assert_close_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "{key}: length");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        check(*a, *e, TOL, &format!("{key}[{i}]"));
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

/// The values and the shape of a result against the fixture keys `key` and `key.shape`.
fn check_values(data: &[f64], dims: &[usize], key: &str) {
    assert_close_list(data, key);
    assert_ints(&dims_i64(dims), &format!("{key}.shape"));
}

/// The values and the shape of `actual` against the fixture keys `key` and `key.shape`.
fn check_array(actual: &NdArray<f64>, key: &str) {
    check_values(actual.data(), actual.dims(), key);
}

/// Mean and variance of `samples` against the fixture row `key` = [mean, variance, excess kurtosis].
/// Mean: |m - mean| <= 5 * sqrt(var / n).
/// Variance: |v - var| <= 5 * sqrt(var^2 * (kurt + 2) / n).
fn assert_moments(samples: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(
        expected.len(),
        3,
        "{key}: needs mean, variance, excess kurtosis"
    );
    let (mean, var, kurt) = (expected[0], expected[1], expected[2]);
    let n = samples.len() as f64;
    let m = samples.iter().sum::<f64>() / n;
    let v = samples.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    let mean_tol = 5.0 * (var / n).sqrt();
    let var_tol = 5.0 * (var * var * (kurt + 2.0) / n).sqrt();
    assert!(
        (m - mean).abs() <= mean_tol,
        "{key}: mean got {m}, expected {mean} (tolerance {mean_tol})"
    );
    assert!(
        (v - var).abs() <= var_tol,
        "{key}: variance got {v}, expected {var} (tolerance {var_tol})"
    );
}

/// The 24 input values, the same integer formula as in the generator.
fn x24() -> Vec<f64> {
    (0..24u64)
        .map(|i| ((i * 37 + 11) % 101) as f64 / 4.0)
        .collect()
}

/// The 24 input values as a 1-D array.
fn x() -> NdArray<f64> {
    NdArray::new(x24(), vec![24]).unwrap()
}

#[test]
fn arange_matches_numpy() {
    let int = NdArray::arange(0i32, 10, 3).unwrap();
    let as_f64 = |d: &[i32]| d.iter().map(|&v| v as f64).collect::<Vec<f64>>();
    check_values(&as_f64(int.data()), int.dims(), "arange.int");

    let neg = NdArray::arange(10i32, 0, -3).unwrap();
    check_values(&as_f64(neg.data()), neg.dims(), "arange.negative_step");

    let float = NdArray::arange(5.0f64, 0.0, -1.5).unwrap();
    check_array(&float, "arange.float_step");

    let tenths = NdArray::arange(0.0f64, 1.0, 0.1).unwrap();
    check_array(&tenths, "arange.tenths");

    let empty = NdArray::arange(5i32, 5, 1).unwrap();
    assert_ints(&[empty.data().len() as i64], "arange.empty_len");
}

#[test]
fn reshape_matches_numpy() {
    let a = x()
        .reshape(vec![2, 3, 4])
        .unwrap()
        .reshape(vec![4, 6])
        .unwrap();
    check_array(&a, "reshape.3d_to_2d");

    let b = x().reshape(vec![4, 3, 2]).unwrap();
    check_array(&b, "reshape.to_3d");
}

#[test]
fn flatten_matches_numpy() {
    let a = x().reshape(vec![2, 3, 4]).unwrap().flatten();
    check_array(&a, "flatten.3d");
}

#[test]
fn transpose_matches_numpy() {
    let a = x().reshape(vec![4, 6]).unwrap().transpose();
    check_array(&a, "transpose.2d");

    // NumPy reverses all axes of a 3-D array: shape 4 x 3 x 2.
    let b = x().reshape(vec![2, 3, 4]).unwrap().transpose();
    check_array(&b, "transpose.3d");
}

#[test]
fn transpose_axes_matches_numpy() {
    let a = x().reshape(vec![2, 3, 4]).unwrap();
    let b = a.transpose_axes(&[1, 2, 0]).unwrap();
    check_array(&b, "transpose_axes.120");

    let c = a.transpose_axes(&[2, 0, 1]).unwrap();
    check_array(&c, "transpose_axes.201");
}

#[test]
fn diag_matches_numpy() {
    let x4 = NdArray::new(x24()[..4].to_vec(), vec![4]).unwrap();
    let build = NdArray::diag(&x4).unwrap();
    check_array(&build, "diag.build");

    let wide = x().reshape(vec![3, 8]).unwrap();
    check_array(&NdArray::diag(&wide).unwrap(), "diag.extract_wide");

    let tall = x().reshape(vec![8, 3]).unwrap();
    check_array(&NdArray::diag(&tall).unwrap(), "diag.extract_tall");
}

#[test]
fn slice_matches_numpy() {
    let m = x().reshape(vec![4, 6]).unwrap();

    // [1:4, ::2]
    let range_step = SliceInfo::new(vec![
        AxisSlice::Slice {
            start: Some(1),
            stop: Some(4),
            step: None,
        },
        AxisSlice::Slice {
            start: None,
            stop: None,
            step: Some(2),
        },
    ]);
    check_array(&m.slice(&range_step).unwrap(), "slice.range_step");

    // [::-1, 1]
    let reverse_index = SliceInfo::new(vec![
        AxisSlice::Slice {
            start: None,
            stop: None,
            step: Some(-1),
        },
        AxisSlice::Index(1),
    ]);
    check_array(&m.slice(&reverse_index).unwrap(), "slice.reverse_index");

    // [-3:, :-1]
    let negative = SliceInfo::new(vec![
        AxisSlice::Slice {
            start: Some(-3),
            stop: None,
            step: None,
        },
        AxisSlice::Slice {
            start: None,
            stop: Some(-1),
            step: None,
        },
    ]);
    check_array(&m.slice(&negative).unwrap(), "slice.negative");

    // [2]
    let row = SliceInfo::new(vec![AxisSlice::Index(2), AxisSlice::Full]);
    check_array(&m.slice(&row).unwrap(), "slice.row");

    // [1:20:4] on the 1-D input
    let one_d = SliceInfo::new(vec![AxisSlice::Slice {
        start: Some(1),
        stop: Some(20),
        step: Some(4),
    }]);
    check_array(&x().slice(&one_d).unwrap(), "slice.1d_step");

    // [:, 1:, ::3] on the 3-D input
    let m3 = x().reshape(vec![2, 3, 4]).unwrap();
    let three_d = SliceInfo::new(vec![
        AxisSlice::Full,
        AxisSlice::Slice {
            start: Some(1),
            stop: None,
            step: None,
        },
        AxisSlice::Slice {
            start: None,
            stop: None,
            step: Some(3),
        },
    ]);
    check_array(&m3.slice(&three_d).unwrap(), "slice.3d");
}

#[test]
fn broadcast_to_matches_numpy() {
    let first3 = NdArray::new(x24()[..3].to_vec(), vec![3]).unwrap();

    let row = first3.reshape(vec![1, 3]).unwrap();
    check_array(
        &row.broadcast_to(&Shape::new(vec![4, 3])).unwrap(),
        "broadcast_to.row",
    );

    let col = first3.reshape(vec![3, 1]).unwrap();
    check_array(
        &col.broadcast_to(&Shape::new(vec![3, 4])).unwrap(),
        "broadcast_to.col",
    );

    check_array(
        &first3.broadcast_to(&Shape::new(vec![2, 3])).unwrap(),
        "broadcast_to.vector",
    );

    let scalar = NdArray::new(vec![x24()[0]], vec![1, 1]).unwrap();
    check_array(
        &scalar.broadcast_to(&Shape::new(vec![2, 3, 2])).unwrap(),
        "broadcast_to.scalar_like",
    );
}

#[test]
fn next_f64_matches_scipy_uniform() {
    let mut rng = KitRng::new(SEED);
    let draws: Vec<f64> = (0..DRAWS).map(|_| rng.next_f64()).collect();
    assert!(
        draws.iter().all(|&u| (0.0..1.0).contains(&u)),
        "a draw is outside [0, 1)"
    );
    assert_moments(&draws, "random.uniform01");
}

#[test]
fn next_u64_matches_scipy_uniform() {
    let mut rng = KitRng::new(SEED);
    let scale = 18446744073709551616.0; // 2^64
    let draws: Vec<f64> = (0..DRAWS).map(|_| rng.next_u64() as f64 / scale).collect();
    assert_moments(&draws, "random.uniform01");
}

#[test]
fn uniform_matches_scipy() {
    let mut rng = KitRng::new(SEED);
    let draws: Vec<f64> = (0..DRAWS).map(|_| rng.uniform(2.0, 5.0)).collect();
    assert_moments(&draws, "random.uniform");
}

#[test]
fn normal_standard_matches_scipy() {
    let mut rng = KitRng::new(SEED);
    let draws: Vec<f64> = (0..DRAWS).map(|_| rng.normal_standard()).collect();
    assert_moments(&draws, "random.normal_standard");
}

#[test]
fn normal_matches_scipy() {
    let mut rng = KitRng::new(SEED);
    let draws: Vec<f64> = (0..DRAWS).map(|_| rng.normal(3.0, 2.0)).collect();
    assert_moments(&draws, "random.normal");
}

#[test]
fn randint_matches_scipy() {
    let mut rng = KitRng::new(SEED);
    let draws: Vec<i64> = (0..DRAWS).map(|_| rng.randint(-3, 7)).collect();
    assert!(
        draws.iter().all(|&v| (-3..=6).contains(&v)),
        "a draw is outside -3..=6"
    );
    let as_f64: Vec<f64> = draws.iter().map(|&v| v as f64).collect();
    assert_moments(&as_f64, "random.randint");
}

#[test]
fn bernoulli_matches_scipy() {
    let mut rng = KitRng::new(SEED);
    let draws: Vec<f64> = (0..DRAWS)
        .map(|_| if rng.bernoulli(0.3) { 1.0 } else { 0.0 })
        .collect();
    assert_moments(&draws, "random.bernoulli");
}

#[test]
fn binomial_matches_scipy() {
    // n = 20 takes the direct path (n <= 30).
    let mut rng = KitRng::new(SEED);
    let direct: Vec<f64> = (0..DRAWS).map(|_| rng.binomial(20, 0.3) as f64).collect();
    assert_moments(&direct, "random.binomial.n20");

    // n = 100, p = 0.3 takes the waiting-time path.
    let mut rng = KitRng::new(SEED);
    let waiting: Vec<f64> = (0..DRAWS).map(|_| rng.binomial(100, 0.3) as f64).collect();
    assert_moments(&waiting, "random.binomial.n100");

    // n = 100, p = 0.8 takes the flip path (p > 0.5).
    let mut rng = KitRng::new(SEED);
    let flip: Vec<f64> = (0..DRAWS).map(|_| rng.binomial(100, 0.8) as f64).collect();
    assert_moments(&flip, "random.binomial.n100_flip");
}

#[test]
fn poisson_matches_scipy() {
    // lambda = 4 takes Knuth's algorithm.
    let mut rng = KitRng::new(SEED);
    let small: Vec<f64> = (0..DRAWS).map(|_| rng.poisson(4.0) as f64).collect();
    assert_moments(&small, "random.poisson.l4");

    // lambda = 50 takes the normal approximation: its variance is about 1/12 above
    // lambda after rounding, which is inside the tolerance of the check.
    let mut rng = KitRng::new(SEED);
    let large: Vec<f64> = (0..DRAWS).map(|_| rng.poisson(50.0) as f64).collect();
    assert_moments(&large, "random.poisson.l50");
}

#[test]
fn choice_matches_scipy() {
    let mut rng = KitRng::new(SEED);
    let draws: Vec<f64> = (0..DRAWS).map(|_| rng.choice(7).unwrap() as f64).collect();
    assert_moments(&draws, "random.choice");
}

#[test]
fn shuffle_matches_scipy() {
    let mut rng = KitRng::new(SEED);
    let mut first: Vec<f64> = Vec::with_capacity(DRAWS);
    for _ in 0..DRAWS {
        let mut values: Vec<usize> = (0..10).collect();
        rng.shuffle(&mut values);
        let mut sorted = values.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..10).collect::<Vec<usize>>(),
            "shuffle is not a permutation"
        );
        first.push(values[0] as f64);
    }
    assert_moments(&first, "random.shuffle.position0");
}

#[test]
fn same_seed_gives_same_sequence() {
    let mut a = KitRng::new(SEED);
    let mut b = KitRng::new(SEED);
    let first_a: Vec<u64> = (0..5).map(|_| a.next_u64()).collect();
    let first_b: Vec<u64> = (0..5).map(|_| b.next_u64()).collect();
    assert_eq!(first_a, first_b);

    let s1 = KitRng::with_stream(SEED, 1).next_u64();
    let s2 = KitRng::with_stream(SEED, 2).next_u64();
    assert_ne!(s1, s2, "two streams of one seed must differ");
}
