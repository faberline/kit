use std::collections::HashMap;
use std::io::Write;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use frame::frame::io::read_csv;
use frame::frame::ops::JoinType;
use frame::frame::{DataFrame, Series};

/// Deterministic linear congruential generator. Fixed seed, no `rand` crate.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Build a frame with an integer `key` column (0..keys) and a float `val` column.
fn make_frame(rows: usize, keys: u64) -> DataFrame {
    let mut state = 42u64;
    let key: Vec<i64> = (0..rows).map(|_| (lcg(&mut state) % keys) as i64).collect();
    let val: Vec<f64> = (0..rows)
        .map(|_| (lcg(&mut state) % 1000) as f64 / 10.0)
        .collect();
    DataFrame::from_columns(vec![("key", Series::new(key)), ("val", Series::new(val))]).unwrap()
}

fn bench_groupby(c: &mut Criterion) {
    let df = make_frame(100_000, 100);
    c.bench_function("groupby_sum_100k", |b| {
        b.iter(|| {
            let mut aggs = HashMap::new();
            aggs.insert("val", "sum");
            black_box(&df).groupby(&["key"]).unwrap().agg(aggs).unwrap()
        })
    });
}

fn bench_join(c: &mut Criterion) {
    let left = make_frame(10_000, 1_000);
    let right_keys: Vec<i64> = (0..1000i64).collect();
    let right_vals: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let right = DataFrame::from_columns(vec![
        ("key", Series::new(right_keys)),
        ("rval", Series::new(right_vals)),
    ])
    .unwrap();
    c.bench_function("join_inner_10k_x_1k", |b| {
        b.iter(|| {
            black_box(&left)
                .join(black_box(&right), &["key"], JoinType::Inner)
                .unwrap()
        })
    });
}

fn bench_read_csv(c: &mut Criterion) {
    let mut state = 42u64;
    let mut text = String::from("a,b,c\n");
    for i in 0..50_000u64 {
        let x = (lcg(&mut state) % 1000) as f64 / 10.0;
        let n = lcg(&mut state) % 100;
        text.push_str(&format!("{},{},name{}\n", i, x, n));
    }
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(text.as_bytes()).unwrap();
    file.flush().unwrap();
    c.bench_function("read_csv_50k_rows", |b| {
        b.iter(|| read_csv(black_box(file.path())).unwrap())
    });
}

criterion_group!(benches, bench_groupby, bench_join, bench_read_csv);
criterion_main!(benches);
