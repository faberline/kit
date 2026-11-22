"""Reference values for arraykit::array/stats and arraykit::array/linalg
(NumPy 2.4.6, SciPy 1.17.1). Each group writes its own fixture file.

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


# Linear algebra: arraykit::array/linalg and arraykit::array/linalg_decomp.
# Every input is built with the same plain loops as the Rust test, so both
# sides hold the same numbers. Products are summed in index order (no BLAS).
LINALG_FIXTURE = "crates/arraykit/tests/fixtures/linalg_reference.tsv"
N_LIN = 50


def plain_matmul(x, y):
    """Product of two lists of rows, summed in index order like the Rust test."""
    n, k, m = len(x), len(y), len(y[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for t in range(k):
                acc += x[i][t] * y[t][j]
            out[i][j] = acc
    return out


def well_matrix():
    """A: 50 x 50, well conditioned (50.0 on the diagonal)."""
    return [
        [((i * 31 + j * 17 + i * j) % 23) / 7.0 + (50.0 if i == j else 0.0)
         for j in range(N_LIN)]
        for i in range(N_LIN)
    ]


def cond_matrix():
    """M: 50 x 50, condition number about 1e10, built as H diag(d) H."""
    k = (1e10 - 1.0) / 49.0
    d = [1.0 / (1.0 + i * k) for i in range(N_LIN)]
    v = [float(i + 1) for i in range(N_LIN)]
    vtv = 0.0
    for item in v:
        vtv += item * item
    h = [
        [(1.0 if i == j else 0.0) - 2.0 * v[i] * v[j] / vtv for j in range(N_LIN)]
        for i in range(N_LIN)
    ]
    out = [[0.0] * N_LIN for _ in range(N_LIN)]
    for i in range(N_LIN):
        for j in range(N_LIN):
            acc = 0.0
            for t in range(N_LIN):
                acc += (h[i][t] * d[t]) * h[t][j]
            out[i][j] = acc
    return out


def rhs_vector():
    return [float((i % 9) - 4) for i in range(N_LIN)]


def tall_matrix():
    """L: 50 x 20, overdetermined, full rank."""
    return [
        [((i * 13 + j * 29 + i * j) % 19) / 5.0 + (5.0 if i == j else 0.0)
         for j in range(20)]
        for i in range(N_LIN)
    ]


def linalg_rows():
    rows = []
    a_list = well_matrix()
    a = numpy.array(a_list)
    b = numpy.array(rhs_vector())
    m_list = cond_matrix()
    m = numpy.array(m_list)
    s = a + a.T
    p = numpy.array(plain_matmul(a_list, [list(col) for col in zip(*a_list)]))
    l_list = tall_matrix()
    l_mat = numpy.array(l_list)

    def add_list(key, value, source, digits=12):
        add(rows, key, common.stable_value(value, digits), source)

    # Cases with condition number about 1e10 differ between BLAS and LAPACK builds
    # in the 11th digit, so they keep 6 digits. The Rust check allows 1e-5 for them.
    cond_digits = 6

    # Small inputs from the issue: x, y, two 2 x 3 and 3 x 2 matrices, and a 3 x 3.
    x = numpy.array([1.5, -2.0, 3.0, 0.5])
    y = numpy.array([2.0, 4.0, -1.0, 8.0])
    p23 = numpy.array([[1, 2, 3], [4, 5, 6]])
    q32 = numpy.array([[7, 8], [9, 10], [11, 12]])
    t3 = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])

    # solve: the first and last 5 of the solution, for the well and the cond 1e10 case.
    x_well = numpy.linalg.solve(a, b)
    add_list("linalg.solve.well.head", x_well[:5], "numpy.linalg.solve(a, b)[:5]")
    add_list("linalg.solve.well.tail", x_well[-5:], "numpy.linalg.solve(a, b)[-5:]")
    x_cond = numpy.linalg.solve(m, b)
    add_list("linalg.solve.cond.head", x_cond[:5], "numpy.linalg.solve(m, b)[:5]",
             digits=cond_digits)
    add_list("linalg.solve.cond.tail", x_cond[-5:], "numpy.linalg.solve(m, b)[-5:]",
             digits=cond_digits)
    add_list("linalg.solve.small2d",
             numpy.linalg.solve([[3.0, 1.0], [1.0, 2.0]],
                                [[9.0, 1.0], [8.0, 3.0]]).ravel(),
             "numpy.linalg.solve([[3, 1], [1, 2]], [[9, 1], [8, 3]]).ravel()")

    # inv: trace, first row and the small-scale value.
    add_list("linalg.inv.well.trace", float(numpy.trace(numpy.linalg.inv(a))),
             "numpy.trace(numpy.linalg.inv(a))")
    add_list("linalg.inv.well.row0_head", numpy.linalg.inv(a)[0, :5],
             "numpy.linalg.inv(a)[0, :5]")
    add_list("linalg.inv.cond.trace", float(numpy.trace(numpy.linalg.inv(m))),
             "numpy.trace(numpy.linalg.inv(m))", digits=cond_digits)
    add_list("linalg.inv.small_scale", float(numpy.linalg.inv([[1e-13]])[0][0]),
             "numpy.linalg.inv([[1e-13]])[0][0]")

    # det: log of the determinant (the 50 x 50 det is about 1e85), two small cases.
    add_list("linalg.det.well.logdet", float(numpy.linalg.slogdet(a)[1]),
             "numpy.linalg.slogdet(a)[1]")
    add_list("linalg.det.small_scale", float(numpy.linalg.det([[1e-7, 0.0], [0.0, 1e-7]])),
             "numpy.linalg.det([[1e-7, 0], [0, 1e-7]])")
    add_list("linalg.det.t3", float(numpy.linalg.det(t3)), "numpy.linalg.det(t3)")

    # eigh and eig: eigenvalues (sorted ascending for eigh and for the 2 x 2 eig).
    add_list("linalg.eigh.well.values", numpy.linalg.eigh(s)[0],
             "numpy.linalg.eigh(s)[0]")
    add_list("linalg.eigh.small", numpy.linalg.eigh([[2.0, 1.0], [1.0, 2.0]])[0],
             "numpy.linalg.eigh([[2, 1], [1, 2]])[0]")
    add_list("linalg.eig.triangular",
             numpy.sort(numpy.linalg.eig([[1.0, 1.0], [0.0, 2.0]])[0].real),
             "numpy.sort(numpy.linalg.eig([[1, 1], [0, 2]])[0].real)")

    # svd: all singular values, for the well case, the cond 1e10 case and a rank-1 case.
    add_list("linalg.svd.well.s", numpy.linalg.svd(a, compute_uv=False),
             "numpy.linalg.svd(a, compute_uv=False)")
    add_list("linalg.svd.cond.s", numpy.linalg.svd(m, compute_uv=False),
             "numpy.linalg.svd(m, compute_uv=False)", digits=cond_digits)
    add_list("linalg.svd.rank_deficient",
             numpy.linalg.svd([[0.0, 0.0], [1.0, 0.0]], compute_uv=False),
             "numpy.linalg.svd([[0, 0], [1, 0]], compute_uv=False)")

    # qr: absolute diagonal of R; cholesky: diagonal of the factor of P = A A^T.
    _, r = numpy.linalg.qr(a)
    add_list("linalg.qr.well.rdiag_abs", numpy.abs(numpy.diag(r)),
             "abs(numpy.diag(numpy.linalg.qr(a)[1]))")
    add_list("linalg.cholesky.well.diag", numpy.diag(numpy.linalg.cholesky(p)),
             "numpy.diag(numpy.linalg.cholesky(p))")

    # lstsq: the solution (20 values; head and tail), residual sum and rank.
    sol = numpy.linalg.lstsq(l_mat, b, rcond=None)
    add_list("linalg.lstsq.x.head", sol[0][:5], "numpy.linalg.lstsq(l, b, rcond=None)[0][:5]")
    add_list("linalg.lstsq.x.tail", sol[0][-5:], "numpy.linalg.lstsq(l, b, rcond=None)[0][-5:]")
    add_list("linalg.lstsq.residuals", float(sol[1][0]),
             "numpy.linalg.lstsq(l, b, rcond=None)[1][0]")
    add(rows, "linalg.lstsq.rank", int(sol[2]), "numpy.linalg.lstsq(l, b, rcond=None)[2]")

    # matmul, dot, inner, outer, trace.
    add_list("linalg.matmul.p23_q32", numpy.matmul(p23, q32).ravel(),
             "numpy.matmul(p23, q32).ravel()")
    add_list("linalg.dot.xy", float(numpy.dot(x, y)), "numpy.dot(x, y)")
    add_list("linalg.dot.p23_q32", numpy.dot(p23, q32).ravel(),
             "numpy.dot(p23, q32).ravel()")
    add_list("linalg.inner.xy", float(numpy.inner(x, y)), "numpy.inner(x, y)")
    add_list("linalg.outer.xy", numpy.outer(x, y).ravel(), "numpy.outer(x, y).ravel()")
    add_list("linalg.trace.t3", float(numpy.trace(t3)), "numpy.trace(t3)")

    # norm: vector orders, and the Frobenius norm of a 2 x 3 matrix.
    add_list("linalg.norm.x.none", float(numpy.linalg.norm(x)), "numpy.linalg.norm(x)")
    add_list("linalg.norm.x.2", float(numpy.linalg.norm(x, 2)), "numpy.linalg.norm(x, 2)")
    add_list("linalg.norm.x.1", float(numpy.linalg.norm(x, 1)), "numpy.linalg.norm(x, 1)")
    add_list("linalg.norm.x.inf", float(numpy.linalg.norm(x, numpy.inf)),
             "numpy.linalg.norm(x, numpy.inf)")
    add_list("linalg.norm.x.3", float(numpy.linalg.norm(x, 3)), "numpy.linalg.norm(x, 3)")
    add_list("linalg.norm.p23.none", float(numpy.linalg.norm(p23)), "numpy.linalg.norm(p23)")

    return rows


if __name__ == "__main__":
    common.main({FIXTURE: stats_rows, LINALG_FIXTURE: linalg_rows})
