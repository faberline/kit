"""Reference values for arraykit::array/stats (NumPy 2.4.6, SciPy 1.17.1).

Run from the repository root:
    .venv-reference/bin/python tools/reference/arraykit/generate.py
Check the committed fixture without writing it:
    .venv-reference/bin/python tools/reference/arraykit/generate.py --check

Every input is an integer formula that the Rust test rebuilds exactly, so no
random numbers are used.
"""

import os
import sys

import numpy
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

FIXTURE = "crates/arraykit/tests/fixtures/stats_reference.tsv"
N_LARGE = 10000

# Small input: a tie (4.0 three times) and a negative value.
SMALL = numpy.array([2.0, 4.0, 4.0, -3.0, 1.5, 0.5, 4.0, 6.0])
# Small positive input for the functions that need values above zero.
SMALL_POS = numpy.array([2.0, 4.0, 4.0, 1.5, 0.5, 4.0, 6.0, 3.0])
# Large input: 10,000 values in multiples of 0.25 (exact in binary).
LARGE = numpy.array([((i * 37 + 11) % 101) / 4.0 for i in range(N_LARGE)])
LARGE_POS = numpy.array([(((i * 37 + 11) % 101) + 1) / 4.0 for i in range(N_LARGE)])
# 4 x 5 grid for the axis reductions.
GRID = numpy.array([((i * 37 + 11) % 101) / 4.0 for i in range(20)]).reshape(4, 5)
# 3 variables x 20 observations for cov and corrcoef.
MAT = numpy.array(
    [[((i * 31 + j * 17 + i * j) % 23) / 7.0 for j in range(20)] for i in range(3)]
)

CASES = {"small": SMALL, "large": LARGE}
CASES_POS = {"small": SMALL_POS, "large": LARGE_POS}


def add(rows, key, value, source):
    rows.append((key, value, source))


def per_case(rows, name, source, fn, cases=CASES):
    for case, x in cases.items():
        add(rows, f"stats.{name}.{case}", fn(x), source)


def stats_rows():
    rows = []

    # Scalars over the small and large inputs.
    per_case(rows, "argmax", "numpy.argmax(x)", lambda x: int(numpy.argmax(x)))
    per_case(rows, "argmin", "numpy.argmin(x)", lambda x: int(numpy.argmin(x)))
    per_case(rows, "sum", "numpy.sum(x)", lambda x: float(numpy.sum(x)))
    per_case(rows, "max", "numpy.max(x)", lambda x: float(numpy.max(x)))
    per_case(rows, "min", "numpy.min(x)", lambda x: float(numpy.min(x)))
    per_case(rows, "mean", "numpy.mean(x)", lambda x: float(numpy.mean(x)))
    per_case(rows, "mean_f64", "numpy.mean(x)", lambda x: float(numpy.mean(x)))
    per_case(rows, "median", "numpy.median(x)", lambda x: float(numpy.median(x)))
    per_case(rows, "var", "numpy.var(x, ddof=1)", lambda x: float(numpy.var(x, ddof=1)))
    per_case(rows, "var_pop", "numpy.var(x)", lambda x: float(numpy.var(x)))
    per_case(rows, "std", "numpy.std(x, ddof=1)", lambda x: float(numpy.std(x, ddof=1)))
    per_case(rows, "std_pop", "numpy.std(x)", lambda x: float(numpy.std(x)))
    per_case(rows, "sem", "scipy.stats.sem(x)", lambda x: float(scipy.stats.sem(x)))
    per_case(rows, "iqr", "scipy.stats.iqr(x)", lambda x: float(scipy.stats.iqr(x)))
    per_case(rows, "skew", "scipy.stats.skew(x, bias=False)",
             lambda x: float(scipy.stats.skew(x, bias=False)))
    per_case(rows, "kurtosis", "scipy.stats.kurtosis(x, bias=False)",
             lambda x: float(scipy.stats.kurtosis(x, bias=False)))
    per_case(rows, "moment.2", "scipy.stats.moment(x, moment=2)",
             lambda x: float(scipy.stats.moment(x, moment=2)))
    per_case(rows, "moment.3", "scipy.stats.moment(x, moment=3)",
             lambda x: float(scipy.stats.moment(x, moment=3)))
    per_case(rows, "percentile.25", "numpy.percentile(x, 25)",
             lambda x: float(numpy.percentile(x, 25)))
    per_case(rows, "percentile.90", "numpy.percentile(x, 90)",
             lambda x: float(numpy.percentile(x, 90)))
    per_case(rows, "quantile.0.9", "numpy.quantile(x, 0.9)",
             lambda x: float(numpy.quantile(x, 0.9)))
    per_case(rows, "trim_mean.0.25", "scipy.stats.trim_mean(x, 0.25)",
             lambda x: float(scipy.stats.trim_mean(x, 0.25)))
    per_case(rows, "mode", "scipy.stats.mode(x, keepdims=False).mode",
             lambda x: float(scipy.stats.mode(x, keepdims=False).mode))
    per_case(rows, "coefficient_of_variation",
             "scipy.stats.variation(x, ddof=1)",
             lambda x: float(scipy.stats.variation(x, ddof=1)), CASES_POS)
    per_case(rows, "geometric_mean", "scipy.stats.gmean(x)",
             lambda x: float(scipy.stats.gmean(x)), CASES_POS)
    per_case(rows, "harmonic_mean", "scipy.stats.hmean(x)",
             lambda x: float(scipy.stats.hmean(x)), CASES_POS)

    # Ties: the first index wins for argmax and argmin; the smallest value wins for mode.
    add(rows, "stats.argmax.tie", int(numpy.argmax([3.0, 1.0, 3.0])),
        "numpy.argmax([3.0, 1.0, 3.0])")
    add(rows, "stats.argmin.tie", int(numpy.argmin([1.0, 3.0, 1.0])),
        "numpy.argmin([1.0, 3.0, 1.0])")
    add(rows, "stats.mode.tie",
        float(scipy.stats.mode(numpy.array([3.0, 1.0, 1.0, 3.0]), keepdims=False).mode),
        "scipy.stats.mode([3.0, 1.0, 1.0, 3.0], keepdims=False).mode")

    # Lists over the small input, and head and tail of the large input.
    add(rows, "stats.cumsum.small", numpy.cumsum(SMALL), "numpy.cumsum(x)")
    add(rows, "stats.cumsum.large.head", numpy.cumsum(LARGE)[:5], "numpy.cumsum(x)[:5]")
    add(rows, "stats.cumsum.large.tail", numpy.cumsum(LARGE)[-5:], "numpy.cumsum(x)[-5:]")
    add(rows, "stats.cumprod.small", numpy.cumprod(SMALL_POS), "numpy.cumprod(x)")
    zs_small = scipy.stats.zscore(SMALL, ddof=1)
    zs_large = scipy.stats.zscore(LARGE, ddof=1)
    add(rows, "stats.zscore.small", zs_small, "scipy.stats.zscore(x, ddof=1)")
    add(rows, "stats.zscore.large.head", zs_large[:5], "scipy.stats.zscore(x, ddof=1)[:5]")
    add(rows, "stats.zscore.large.tail", zs_large[-5:], "scipy.stats.zscore(x, ddof=1)[-5:]")

    # Histogram: counts and bin edges, five bins.
    for case, x in CASES.items():
        counts, edges = numpy.histogram(x, bins=5)
        add(rows, f"stats.histogram.{case}.counts", counts, "numpy.histogram(x, bins=5)[0]")
        add(rows, f"stats.histogram.{case}.edges", edges, "numpy.histogram(x, bins=5)[1]")

    # Axis reductions over the 4 x 5 grid: axis 0 gives 5 values, axis 1 gives 4.
    axis_fns = {
        "sum_axis": ("numpy.sum", lambda a, ax: numpy.sum(a, axis=ax)),
        "mean_axis": ("numpy.mean", lambda a, ax: numpy.mean(a, axis=ax)),
        "max_axis": ("numpy.max", lambda a, ax: numpy.max(a, axis=ax)),
        "min_axis": ("numpy.min", lambda a, ax: numpy.min(a, axis=ax)),
        "var_axis": ("numpy.var(ddof=1)", lambda a, ax: numpy.var(a, axis=ax, ddof=1)),
        "std_axis": ("numpy.std(ddof=1)", lambda a, ax: numpy.std(a, axis=ax, ddof=1)),
    }
    for name, (expr, fn) in axis_fns.items():
        for ax in (0, 1):
            add(rows, f"stats.{name}.{ax}", fn(GRID, ax), f"{expr}(grid, axis={ax})")

    # Matrices: 3 x 3 results, flattened row by row.
    add(rows, "stats.cov.matrix", numpy.cov(MAT).ravel(), "numpy.cov(mat).ravel()")
    add(rows, "stats.corrcoef.matrix", numpy.corrcoef(MAT).ravel(),
        "numpy.corrcoef(mat).ravel()")

    # Integer overflow in checked_sum: NumPy widens int32 sums to int64.
    i32 = numpy.array([2147483647, 1], dtype=numpy.int32)
    add(rows, "stats.checked_sum.int32.ok",
        int(numpy.sum(numpy.array([1, 2, 3], dtype=numpy.int32))),
        "numpy.sum(numpy.array([1, 2, 3], dtype=numpy.int32))")
    add(rows, "stats.checked_sum.int32.overflow.value", int(numpy.sum(i32)),
        "numpy.sum(numpy.array([2147483647, 1], dtype=numpy.int32))")
    add(rows, "stats.checked_sum.int32.overflow.dtype", str(numpy.sum(i32).dtype),
        "str(numpy.sum(numpy.array([2147483647, 1], dtype=numpy.int32)).dtype)")

    # Moved from tests/reference.rs: empty input, one-two means and NaN handling.
    add(rows, "stats.mean.empty", float(numpy.mean(numpy.array([]))), "numpy.mean([])")
    add(rows, "stats.var.empty", float(numpy.var(numpy.array([]), ddof=1)),
        "numpy.var([], ddof=1)")
    add(rows, "stats.mean.one_two", float(numpy.mean([1.0, 2.0])), "numpy.mean([1.0, 2.0])")
    add(rows, "stats.var.one_to_four", float(numpy.var([1.0, 2.0, 3.0, 4.0], ddof=1)),
        "numpy.var([1.0, 2.0, 3.0, 4.0], ddof=1)")
    nan_inputs = {
        "a": [1.0, float("nan"), 3.0],
        "b": [float("nan"), 1.0, 3.0],
        "c": [3.0, 1.0, float("nan")],
    }
    for name, values in nan_inputs.items():
        add(rows, f"stats.min.nan.{name}", float(numpy.min(values)),
            f"numpy.min({name} with nan)")
        add(rows, f"stats.max.nan.{name}", float(numpy.max(values)),
            f"numpy.max({name} with nan)")

    return rows


if __name__ == "__main__":
    common.main({FIXTURE: stats_rows})
