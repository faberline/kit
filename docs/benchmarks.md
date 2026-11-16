# Benchmarks

## Run the benchmarks

Each crate with a hot path has a Criterion benchmark:

```sh
cargo bench -p arraykit
cargo bench -p frame
cargo bench -p scikit
```

For a fast run, add `-- --quick`, for example `cargo bench -p arraykit -- --quick`.

## Regression gate

The CI job `test-bench` fails when a benchmark is more than 10 percent slower than the baseline. The baseline is the file `benches/baseline/baseline.json`. The gate is planned in #91 and may not be merged yet.

## Compare with NumPy, pandas and SciPy

Use the same machine and the same input sizes for both sides. Build in release mode. Run three warm-up calls, then take the median of at least 20 timed runs. Use one thread:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The Python side is `tools/bench-python/bench.py`. Run it by hand:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r tools/bench-python/requirements.txt
python3 tools/bench-python/bench.py
```

A person runs the script. CI never runs it.

## Status

| Function | kit crate | Python reference | Status |
|---|---|---|---|
| `matmul` | `arraykit` | `numpy.matmul` | not measured |
| `sum` | `arraykit` | `numpy.sum` | not measured |
| `sort` | `arraykit` | `numpy.sort` | not measured |
| `group-by` | `frame` | `pandas.DataFrame.groupby` | not measured |
| `join` | `frame` | `pandas.DataFrame.merge` | not measured |
| `CSV read` | `frame` | `pandas.read_csv` | not measured |
| `fft` | `scikit` | `scipy.fft.fft` | not measured |
| `stats` | `scikit` | `scipy.stats` | not measured |

The allowed statuses are `faster`, `slower` and `not measured`. A person changes a row only after a run of the script. The commit message then gives the date and the machine type. No time value is written in this file.
