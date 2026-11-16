use arraykit::NdArray;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Deterministic linear congruential generator. Fixed seed, no `rand` crate.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Build an `f64` array with `n` values and the given shape. Fixed seed 42.
fn make_f64(n: usize, dims: Vec<usize>) -> NdArray<f64> {
    let mut state = 42u64;
    let data: Vec<f64> = (0..n)
        .map(|_| (lcg(&mut state) % 1000) as f64 / 10.0)
        .collect();
    NdArray::new(data, dims).unwrap()
}

/// Build an `i64` array with `n` values. Fixed seed 42.
fn make_i64(n: usize) -> NdArray<i64> {
    let mut state = 42u64;
    let data: Vec<i64> = (0..n).map(|_| (lcg(&mut state) % 100_000) as i64).collect();
    NdArray::new(data, vec![n]).unwrap()
}

fn bench_matmul(c: &mut Criterion) {
    let a = make_f64(128 * 128, vec![128, 128]);
    let b = make_f64(128 * 128, vec![128, 128]);
    c.bench_function("matmul_128x128_f64", |bn| {
        bn.iter(|| black_box(&a).matmul(black_box(&b)).unwrap())
    });
}

fn bench_sum(c: &mut Criterion) {
    let a = make_f64(1_000_000, vec![1_000_000]);
    c.bench_function("sum_1m_f64", |bn| bn.iter(|| black_box(&a).sum()));
}

fn bench_sort(c: &mut Criterion) {
    let a = make_i64(100_000);
    c.bench_function("sort_100k_i64", |bn| bn.iter(|| black_box(&a).sort()));
}

criterion_group!(benches, bench_matmul, bench_sum, bench_sort);
criterion_main!(benches);
