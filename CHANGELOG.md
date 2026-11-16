# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

- Added crate metadata, the release checklist in docs/release.md and a packaging check in CI.
- Added SECURITY.md and a cargo audit step in CI.
- Added the stability promise and the public API test.
- Added the coverage job and the coverage ratchet in aw.toml.
- Added criterion benchmarks for arraykit matmul, sum and sort (`cargo bench -p arraykit`).
- Added criterion benchmarks for frame group-by, inner join and read_csv (`cargo bench -p frame`).
- Added criterion benchmarks for scikit fft and stats (`cargo bench -p scikit --features fft`).
