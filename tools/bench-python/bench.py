"""Time the same operations with NumPy, pandas and SciPy and print JSON.

A person runs this script by hand, on the same machine as the Rust benchmarks.
CI never runs it. See docs/benchmarks.md.
"""

import io
import json
import platform
import statistics
import sys
import time

try:
    import numpy as np
    import pandas as pd
    import scipy
    import scipy.fft
    import scipy.stats
except ImportError as exc:
    name = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
    print(json.dumps({"error": f"missing module {name}"}), file=sys.stderr)
    sys.exit(2)

SIZES = {
    "matmul": 256,
    "sum": 1_000_000,
    "sort": 1_000_000,
    "groupby": 100_000,
    "join": 100_000,
    "csv": 100_000,
    "fft": 4096,
    "stats": 100_000,
}


def median_ns(fn, warmup=3, runs=20):
    """Return the median time of fn() in nanoseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - start)
    return int(statistics.median(times))


def make_matmul():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((SIZES["matmul"], SIZES["matmul"]))
    b = rng.standard_normal((SIZES["matmul"], SIZES["matmul"]))
    return lambda: np.matmul(a, b)


def make_sum():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(SIZES["sum"])
    return lambda: np.sum(a)


def make_sort():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(SIZES["sort"])
    return lambda: np.sort(a)


def make_groupby():
    rng = np.random.default_rng(42)
    n = SIZES["groupby"]
    df = pd.DataFrame({"key": rng.integers(0, 100, n), "value": rng.standard_normal(n)})
    return lambda: df.groupby("key")["value"].sum()


def make_join():
    rng = np.random.default_rng(42)
    n = SIZES["join"]
    left = pd.DataFrame({"key": rng.integers(0, n, n), "a": rng.standard_normal(n)})
    right = pd.DataFrame({"key": rng.integers(0, n, n), "b": rng.standard_normal(n)})
    return lambda: left.merge(right, on="key")


def make_csv():
    rng = np.random.default_rng(42)
    n = SIZES["csv"]
    df = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.integers(0, 1000, n)})
    text = df.to_csv(index=False)
    return lambda: pd.read_csv(io.StringIO(text))


def make_fft():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(SIZES["fft"])
    return lambda: scipy.fft.fft(a)


def make_stats():
    rng = np.random.default_rng(42)
    n = SIZES["stats"]
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)

    def run():
        scipy.stats.pearsonr(x, y)
        np.mean(x)
        np.std(x)

    return run


BENCHES = {
    "matmul": make_matmul,
    "sum": make_sum,
    "sort": make_sort,
    "groupby": make_groupby,
    "join": make_join,
    "csv": make_csv,
    "fft": make_fft,
    "stats": make_stats,
}


def main():
    result = {name: median_ns(make()) for name, make in BENCHES.items()}
    result["python"] = platform.python_version()
    result["modules"] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
