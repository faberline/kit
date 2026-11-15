use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scikit::fft::{fft, Complex};
use scikit::stats::{describe, median, pearsonr};

/// Deterministic linear congruential generator. Fixed seed, no `rand` crate.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// `n` values in [0, 100) from a fixed seed.
fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| (lcg(&mut s) % 10_000) as f64 / 100.0)
        .collect()
}

fn bench_fft(c: &mut Criterion) {
    let input: Vec<Complex> = data(4096, 42)
        .into_iter()
        .map(|v| Complex::new(v, 0.0))
        .collect();
    c.bench_function("fft_4096", |b| b.iter(|| fft(black_box(&input))));
}

fn bench_stats(c: &mut Criterion) {
    let x = data(100_000, 42);
    let y = data(100_000, 7);
    c.bench_function("stats_describe_100k", |b| {
        b.iter(|| describe(black_box(&x)))
    });
    c.bench_function("stats_median_100k", |b| b.iter(|| median(black_box(&x))));
    c.bench_function("stats_pearsonr_100k", |b| {
        b.iter(|| pearsonr(black_box(&x), black_box(&y)))
    });
}

criterion_group!(benches, bench_fft, bench_stats);
criterion_main!(benches);
