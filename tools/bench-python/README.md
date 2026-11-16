# bench-python

`bench.py` times the same operations as the Rust benchmarks, with NumPy, pandas and SciPy. It prints one JSON object to stdout. The method is in [docs/benchmarks.md](../../docs/benchmarks.md).

## Run

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r tools/bench-python/requirements.txt
python3 tools/bench-python/bench.py
```

Run it on the same machine as `cargo bench`. Keep the other programs idle.

## Output

The JSON object has these keys:

- `matmul`, `sum`, `sort`, `groupby`, `join`, `csv`, `fft`, `stats`: the median time of one call, in nanoseconds (an integer).
- `python`: the Python version.
- `modules`: the versions of `numpy`, `pandas` and `scipy`.

If a module is missing, the script prints `{"error": "missing module <name>"}` to stderr and exits with status 2.

Do not run this in CI. Do not add a Python file under `crates/`.
