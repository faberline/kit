"""Reference values for the scikit stats functions (docs/reference-coverage.md).

stats_rows writes crates/scikit/tests/fixtures/stats_reference.tsv. Each `source`
column is the SciPy or NumPy expression that made the value.

Run from the repository root:
    sh tools/reference/run.sh
"""

import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.integrate as ig
import scipy.interpolate as ip
import scipy.optimize as op
import scipy.signal as sg
import scipy.sparse as ssp
import scipy.spatial as spt
import scipy.special as sp
from scipy.spatial import distance as dist
import scipy.stats as st
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose as sm_decompose
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import pacf as sm_pacf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402


def stats_rows():
    rows = []

    def add(key, value, source):
        rows.append((key, value, source))

    add("lgamma_200", float(sp.gammaln(200)), "scipy.special.gammaln(200)")
    add("lgamma_1000", float(sp.gammaln(1000)), "scipy.special.gammaln(1000)")
    add("erf_0p5", float(sp.erf(0.5)), "scipy.special.erf(0.5)")
    add("erf_2", float(sp.erf(2.0)), "scipy.special.erf(2.0)")
    add("norm_cdf_1p96", float(st.norm.cdf(1.96)), "scipy.stats.norm.cdf(1.96)")
    add("norm_cdf_5p5_mu2_s3", float(st.norm.cdf(5.5, 2, 3)), "scipy.stats.norm.cdf(5.5, 2, 3)")
    add("t_cdf_2_df5", float(st.t.cdf(2.0, 5)), "scipy.stats.t.cdf(2.0, 5)")
    add("t_cdf_m1p5_df10", float(st.t.cdf(-1.5, 10)), "scipy.stats.t.cdf(-1.5, 10)")
    add("chi2_cdf_3_df4", float(st.chi2.cdf(3.0, 4)), "scipy.stats.chi2.cdf(3.0, 4)")
    add("chi2_cdf_10_df7", float(st.chi2.cdf(10.0, 7)), "scipy.stats.chi2.cdf(10.0, 7)")

    x = [((i % 20) * 10 + (i % 7) * 5) / 100.0 for i in range(400)]
    add(
        "t1_400_stat",
        float(st.ttest_1samp(x, 1.0).statistic),
        "scipy.stats.ttest_1samp(x, 1.0).statistic, x = [((i%20)*10+(i%7)*5)/100.0 for i in range(400)]",
    )
    add(
        "t1_400_p",
        float(st.ttest_1samp(x, 1.0).pvalue),
        "scipy.stats.ttest_1samp(x, 1.0).pvalue, same x",
    )

    s = [2.1, 3.4, 1.9, 5.6, 4.4, 3.8, 2.7, 4.9]
    add(
        "t1_8_stat",
        float(st.ttest_1samp(s, 3.0).statistic),
        "scipy.stats.ttest_1samp(s, 3.0).statistic, s = [2.1,3.4,1.9,5.6,4.4,3.8,2.7,4.9]",
    )
    add("t1_8_p", float(st.ttest_1samp(s, 3.0).pvalue), "scipy.stats.ttest_1samp(s, 3.0).pvalue")

    a = s
    b = [1.2, 2.8, 2.2, 3.1, 2.5, 3.3, 1.8]
    add(
        "ind_eq_stat",
        float(st.ttest_ind(a, b, equal_var=True).statistic),
        "scipy.stats.ttest_ind(a, b, equal_var=True).statistic, a = s, b = [1.2,2.8,2.2,3.1,2.5,3.3,1.8]",
    )
    add(
        "ind_eq_p",
        float(st.ttest_ind(a, b, equal_var=True).pvalue),
        "scipy.stats.ttest_ind(a, b, equal_var=True).pvalue",
    )
    add(
        "ind_welch_stat",
        float(st.ttest_ind(a, b, equal_var=False).statistic),
        "scipy.stats.ttest_ind(a, b, equal_var=False).statistic",
    )
    add(
        "ind_welch_p",
        float(st.ttest_ind(a, b, equal_var=False).pvalue),
        "scipy.stats.ttest_ind(a, b, equal_var=False).pvalue",
    )

    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    ys = [2.1, 2.9, 4.2, 3.8, 6.1, 5.9, 7.7, 8.4]
    add(
        "pearson_r",
        common.stable(float(st.pearsonr(xs, ys).statistic)),
        "scipy.stats.pearsonr(xs, ys).statistic, xs = [1,2,3,4,5,6,7,8], ys = [2.1,2.9,4.2,3.8,6.1,5.9,7.7,8.4]",
    )
    add("pearson_p", float(st.pearsonr(xs, ys).pvalue), "scipy.stats.pearsonr(xs, ys).pvalue")

    g1 = [6, 8, 4, 5, 3, 4]
    g2 = [8, 12, 9, 11, 6, 8]
    g3 = [13, 9, 11, 8, 7, 12]
    add(
        "anova_f",
        float(st.f_oneway(g1, g2, g3).statistic),
        "scipy.stats.f_oneway(g1, g2, g3).statistic, g1 = [6,8,4,5,3,4], g2 = [8,12,9,11,6,8], g3 = [13,9,11,8,7,12]",
    )
    add("anova_p", float(st.f_oneway(g1, g2, g3).pvalue), "scipy.stats.f_oneway(g1, g2, g3).pvalue")

    add(
        "chisq_stat",
        float(st.chisquare([16, 18, 16, 14, 12, 12], [16, 16, 16, 16, 16, 8]).statistic),
        "scipy.stats.chisquare([16,18,16,14,12,12], [16,16,16,16,16,8]).statistic",
    )
    add(
        "chisq_p",
        float(st.chisquare([16, 18, 16, 14, 12, 12], [16, 16, 16, 16, 16, 8]).pvalue),
        "scipy.stats.chisquare([16,18,16,14,12,12], [16,16,16,16,16,8]).pvalue",
    )

    add(
        "spearman_rho",
        float(st.spearmanr([1, 1, 1, 2], [1, 2, 3, 4]).statistic),
        "scipy.stats.spearmanr([1,1,1,2], [1,2,3,4]).statistic",
    )

    mx = [1, 2, 2, 3, 5, 6]
    my = [2, 3, 4, 4, 7, 8, 9]
    mw = st.mannwhitneyu(mx, my, use_continuity=False, method="asymptotic")
    add(
        "mw_u",
        float(mw.statistic),
        "scipy.stats.mannwhitneyu(x, y, use_continuity=False, method='asymptotic').statistic, x = [1,2,2,3,5,6], y = [2,3,4,4,7,8,9]",
    )
    add(
        "mw_p",
        float(mw.pvalue),
        "scipy.stats.mannwhitneyu(x, y, use_continuity=False, method='asymptotic').pvalue",
    )
    return rows


GRID_SYM = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
GRID_POS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
GRID_UNIT = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
GRID_UNI = [-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.99, 5.0, 6.0]
GRID_K = [0, 1, 2, 5, 10, 20, 30]

# Inputs of the grid rows: integer formulas that the Rust tests rebuild exactly.
X = [((i * 37 + 11) % 101) / 4.0 + 1.0 for i in range(500)]
Y = [((i * 53 + 7) % 97) / 4.0 + 2.0 for i in range(500)]
Z = [((i * 29 + 3) % 89) / 4.0 + 1.5 for i in range(500)]
M = X + [13.5] * 6
O = [((i * 7 + 3) % 23) + 10.0 for i in range(30)]
TAB33 = [float(((r * 5 + c * 3) % 11) + 8) for r in range(3) for c in range(4)]
TAB22 = [12.0, 5.0, 7.0, 19.0]


# The Rust tests compare these keys at 1e-12 or tighter, so they keep the full repr.
EXACT_PREFIXES = ("erf_", "ln_gamma_", "factorial_", "binomial_coef_")


def stats_grid_rows():
    rows = []

    def add(key, value, source):
        if not key.startswith(EXACT_PREFIXES):
            value = common.stable_value(value)
        rows.append((key, value, source))

    xa = np.array(X)

    # A. Descriptive functions on X.
    z = st.zscore(xa, ddof=1)
    add("skew", float(st.skew(X, bias=False)), "scipy.stats.skew(X, bias=False)")
    add(
        "kurtosis",
        float(st.kurtosis(X, fisher=True, bias=False)),
        "scipy.stats.kurtosis(X, fisher=True, bias=False)",
    )
    add("zscore_first5", z[:5], "scipy.stats.zscore(X, ddof=1)[:5]")
    add("zscore_last5", z[-5:], "scipy.stats.zscore(X, ddof=1)[-5:]")
    add("zscore_sumsq", float(np.sum(z**2)), "np.sum(scipy.stats.zscore(X, ddof=1)**2)")
    for k in (2, 3, 4):
        add(f"moment_{k}", float(st.moment(X, order=k)), f"scipy.stats.moment(X, order={k})")
    for k in (2, 3):
        add(f"raw_moment_{k}", float(np.mean(xa**k)), f"np.mean(np.array(X)**{k})")
    add("mode", float(st.mode(M).mode), "scipy.stats.mode(M).mode")
    add("median", float(np.median(xa)), "np.median(X)")
    add("iqr", float(st.iqr(X)), "scipy.stats.iqr(X)")
    for p in (0, 10, 25, 50, 90, 100):
        add(f"percentile_{p}", float(np.percentile(xa, p)), f"np.percentile(X, {p})")
    add("gmean", float(st.gmean(X)), "scipy.stats.gmean(X)")
    add("hmean", float(st.hmean(X)), "scipy.stats.hmean(X)")
    add("trim_0p1", float(st.trim_mean(X, 0.1)), "scipy.stats.trim_mean(X, 0.1)")
    add("trim_0p25", float(st.trim_mean(X, 0.25)), "scipy.stats.trim_mean(X, 0.25)")
    add("sem", float(st.sem(X)), "scipy.stats.sem(X)")
    add("variation", float(st.variation(X, ddof=1)), "scipy.stats.variation(X, ddof=1)")
    add("mad", float(st.median_abs_deviation(X)), "scipy.stats.median_abs_deviation(X)")
    add("describe_mean", float(np.mean(xa)), "np.mean(X)")
    add("describe_std", float(np.std(xa, ddof=1)), "np.std(X, ddof=1)")
    add("describe_min", float(np.min(xa)), "np.min(X)")
    add("describe_q1", float(np.percentile(xa, 25)), "np.percentile(X, 25)")
    add("describe_median", float(np.median(xa)), "np.median(X)")
    add("describe_q3", float(np.percentile(xa, 75)), "np.percentile(X, 75)")
    add("describe_max", float(np.max(xa)), "np.max(X)")
    add("describe_skew", float(st.skew(X, bias=False)), "scipy.stats.skew(X, bias=False)")
    add(
        "describe_kurt",
        float(st.kurtosis(X, fisher=True, bias=False)),
        "scipy.stats.kurtosis(X, fisher=True, bias=False)",
    )

    # B. Special functions on fixed arguments.
    for i, x in enumerate([-3.0, -1.0, -0.1, 0.1, 1.0, 3.0, 5.0]):
        add(f"erf_{i}", float(sp.erf(x)), f"scipy.special.erf({x!r})")
    for i, x in enumerate([0.5, 1.5, 2.5, 5.5, 10.5, 20.5]):
        add(f"gamma_fn_{i}", float(sp.gamma(x)), f"scipy.special.gamma({x!r})")
    for i, x in enumerate([0.5, 1.5, 10.5, 100.5, 500.5]):
        add(f"ln_gamma_{i}", float(sp.gammaln(x)), f"scipy.special.gammaln({x!r})")
    for i, (a, b) in enumerate([(0.5, 0.5), (2.0, 3.0), (10.5, 3.5), (50.0, 60.0)]):
        add(f"beta_fn_{i}", float(sp.beta(a, b)), f"scipy.special.beta({a!r}, {b!r})")
    for i, (s, x) in enumerate([(0.5, 0.1), (2.0, 1.0), (5.0, 3.0), (5.0, 12.0), (20.0, 15.0)]):
        add(
            f"lower_inc_gamma_{i}",
            float(sp.gammainc(s, x) * sp.gamma(s)),
            f"scipy.special.gammainc({s!r}, {x!r}) * scipy.special.gamma({s!r})",
        )
    for i, (a, b, x) in enumerate(
        [(2.0, 3.0, 0.4), (0.5, 0.5, 0.3), (10.0, 5.0, 0.7), (1.0, 1.0, 0.9), (30.0, 40.0, 0.45)]
    ):
        add(f"incomplete_beta_{i}", float(sp.betainc(a, b, x)), f"scipy.special.betainc({a!r}, {b!r}, {x!r})")
    for i, n in enumerate([0, 1, 5, 10, 20, 25]):
        add(f"factorial_{i}", float(math.factorial(n)), f"math.factorial({n})")
    for i, (n, k) in enumerate([(5, 2), (10, 0), (10, 10), (20, 10), (50, 25), (6, 7)]):
        add(f"binomial_coef_{i}", float(math.comb(n, k)), f"math.comb({n}, {k})")

    # C. Distributions on a grid that covers both tails (cdf, pdf, mean, variance).
    continuous = [
        ("normal", st.norm(1.5, 2.0), GRID_SYM, "scipy.stats.norm(1.5, 2.0)"),
        ("stdnormal", st.norm(0.0, 1.0), GRID_SYM, "scipy.stats.norm(0.0, 1.0)"),
        ("uniform", st.uniform(-3.0, 8.0), GRID_UNI, "scipy.stats.uniform(-3.0, 8.0)"),
        ("expon", st.expon(scale=2.0), GRID_POS, "scipy.stats.expon(scale=2.0)"),
        ("gamma", st.gamma(2.5, scale=1.5), GRID_POS, "scipy.stats.gamma(2.5, scale=1.5)"),
        ("beta", st.beta(2.0, 5.0), GRID_UNIT, "scipy.stats.beta(2.0, 5.0)"),
        ("t5", st.t(5.0), GRID_SYM, "scipy.stats.t(5.0)"),
        ("t30", st.t(30.0), GRID_SYM, "scipy.stats.t(30.0)"),
        ("chi2", st.chi2(4.0), GRID_POS, "scipy.stats.chi2(4.0)"),
        ("f", st.f(5.0, 10.0), GRID_POS, "scipy.stats.f(5.0, 10.0)"),
    ]
    for name, dist, grid, desc in continuous:
        for i, x in enumerate(grid):
            add(f"{name}_cdf_{i}", float(dist.cdf(x)), f"{desc}.cdf({x!r})")
            add(f"{name}_pdf_{i}", float(dist.pdf(x)), f"{desc}.pdf({x!r})")
        add(f"{name}_mean", float(dist.mean()), f"{desc}.mean()")
        add(f"{name}_var", float(dist.var()), f"{desc}.var()")
    discrete = [
        ("binom", st.binom(20, 0.3), "scipy.stats.binom(20, 0.3)"),
        ("poisson", st.poisson(4.5), "scipy.stats.poisson(4.5)"),
        ("geom", st.geom(0.2), "scipy.stats.geom(0.2)"),
    ]
    for name, dist, desc in discrete:
        for i, k in enumerate(GRID_K):
            add(f"{name}_pmf_{i}", float(dist.pmf(k)), f"{desc}.pmf({k})")
            add(f"{name}_cdf_{i}", float(dist.cdf(k)), f"{desc}.cdf({k})")
        add(f"{name}_mean", float(dist.mean()), f"{desc}.mean()")
        add(f"{name}_var", float(dist.var()), f"{desc}.var()")

    # D. Hypothesis tests on 500-value samples with ties.
    def test_pair(name, res, desc):
        stat = float(res.statistic)
        add(f"{name}_stat", stat, f"{desc}.statistic")
        add(f"{name}_p", float(res.pvalue), f"{desc}.pvalue")

    test_pair("ttest1", st.ttest_1samp(X, 13.0), "scipy.stats.ttest_1samp(X, 13.0)")
    test_pair(
        "ttest_ind_eq", st.ttest_ind(X, Y, equal_var=True), "scipy.stats.ttest_ind(X, Y, equal_var=True)"
    )
    test_pair(
        "ttest_ind_welch",
        st.ttest_ind(X, Y, equal_var=False),
        "scipy.stats.ttest_ind(X, Y, equal_var=False)",
    )
    test_pair("ttest_rel", st.ttest_rel(X, Y), "scipy.stats.ttest_rel(X, Y)")
    test_pair("anova", st.f_oneway(X, Y, Z), "scipy.stats.f_oneway(X, Y, Z)")
    expected = [sum(O) / 30.0] * 30
    test_pair("chisq", st.chisquare(O, expected), "scipy.stats.chisquare(O, [sum(O)/30.0]*30)")
    test_pair(
        "chi2c_3x4",
        st.chi2_contingency(np.array(TAB33).reshape(3, 4), correction=False),
        "scipy.stats.chi2_contingency(table 3x4, correction=False)",
    )
    test_pair(
        "chi2c_2x2",
        st.chi2_contingency(np.array(TAB22).reshape(2, 2), correction=False),
        "scipy.stats.chi2_contingency(table 2x2, correction=False)",
    )
    test_pair("pearson", st.pearsonr(X, Y), "scipy.stats.pearsonr(X, Y)")
    test_pair("spearman", st.spearmanr(X, Y), "scipy.stats.spearmanr(X, Y)")
    mw = st.mannwhitneyu(X, Y, use_continuity=False, method="asymptotic")
    add(
        "mw_stat",
        float(min(mw.statistic, 500 * 500 - mw.statistic)),
        "min(scipy.stats.mannwhitneyu(X, Y, use_continuity=False, method='asymptotic').statistic, 500*500 - that)",
    )
    add(
        "mw_p",
        float(mw.pvalue),
        "scipy.stats.mannwhitneyu(X, Y, use_continuity=False, method='asymptotic').pvalue",
    )
    zs = (xa - np.mean(xa)) / np.std(xa, ddof=1)
    ks_stat = float(st.kstest(zs, "norm").statistic)
    add("ks_stat", ks_stat, "scipy.stats.kstest(z, 'norm').statistic, z = (X - mean(X)) / std(X, ddof=1)")
    add(
        "ks_p_asym",
        float(st.kstwobign.sf((math.sqrt(500) + 0.12 + 0.11 / math.sqrt(500)) * ks_stat)),
        "scipy.stats.kstwobign.sf((sqrt(500) + 0.12 + 0.11/sqrt(500)) * ks_stat)",
    )
    return rows


def sample(n):
    """The shared input: x[i] = ((i * 37 + 11) % 101) / 4.0. Rust builds the same values."""
    return np.array([((i * 37 + 11) % 101) / 4.0 for i in range(n)], dtype=float)


def vec_rows(prefix, arr, source):
    """A 1-D array as rows: the whole list when it has at most 64 values.

    A longer array gives its first 5 values, its last 5 values, the sum of
    absolute values and its length, so that the fixture stays short.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= 64:
        return [(prefix, arr.tolist(), source)]
    return [
        (prefix + "_head", arr[:5].tolist(), source),
        (prefix + "_tail", arr[-5:].tolist(), source),
        (prefix + "_abs_sum", float(np.abs(arr).sum()), source),
        (prefix + "_len", float(len(arr)), source),
    ]


def fft_signal_rows():
    rows = []

    def add(prefix, arr, source):
        if np.ndim(arr) == 0:
            rows.append((prefix, float(arr), source))
        else:
            rows.extend(vec_rows(prefix, arr, source))

    def add_complex(prefix, z, source):
        add(prefix + "_re", np.real(z), source)
        add(prefix + "_im", np.imag(z), source)

    for n in (5, 8, 1000, 1024):
        x = sample(n)
        add_complex(f"fft_n{n}", np.fft.fft(x), f"numpy.fft.fft(sample({n}))")
        add_complex(f"ifft_n{n}", np.fft.ifft(x), f"numpy.fft.ifft(sample({n}))")
        add_complex(f"rfft_n{n}", np.fft.rfft(x), f"numpy.fft.rfft(sample({n}))")
        add(
            f"irfft_n{n}",
            np.fft.irfft(np.fft.rfft(x), n),
            f"numpy.fft.irfft(numpy.fft.rfft(sample({n})), {n})",
        )

    z8 = np.fft.fft(sample(8))
    add_complex("conj_n8", np.conj(z8), "numpy.conj(numpy.fft.fft(sample(8)))")
    add("norm_n8", np.abs(z8), "numpy.abs(numpy.fft.fft(sample(8)))")

    for n, d, tag in ((8, 0.5, "n8_d0p5"), (5, 0.1, "n5_d0p1"), (1000, 0.001, "n1000_d0p001")):
        add(f"fftfreq_{tag}", np.fft.fftfreq(n, d), f"numpy.fft.fftfreq({n}, {d})")
        add(f"rfftfreq_{tag}", np.fft.rfftfreq(n, d), f"numpy.fft.rfftfreq({n}, {d})")

    for n in (9, 64):
        add(f"hann_n{n}", sg.windows.hann(n, sym=True), f"scipy.signal.windows.hann({n}, sym=True)")
        add(f"hamming_n{n}", sg.windows.hamming(n, sym=True), f"scipy.signal.windows.hamming({n}, sym=True)")
        add(f"blackman_n{n}", sg.windows.blackman(n, sym=True), f"scipy.signal.windows.blackman({n}, sym=True)")
        for beta, tag in ((5.0, "beta5p0"), (8.6, "beta8p6")):
            add(
                f"kaiser_n{n}_{tag}",
                sg.windows.kaiser(n, beta, sym=True),
                f"scipy.signal.windows.kaiser({n}, {beta}, sym=True)",
            )

    for size, a, b in (("small", sample(7), [0.5, -1.0, 2.0]), ("long", sample(100), [0.25, 0.5, 0.25])):
        for mode in ("full", "same", "valid"):
            add(
                f"convolve_{mode}_{size}",
                sg.convolve(a, b, mode=mode, method="direct"),
                f"scipy.signal.convolve(sample({len(a)}), {b}, mode='{mode}', method='direct')",
            )
            add(
                f"correlate_{mode}_{size}",
                sg.correlate(a, b, mode=mode, method="direct"),
                f"scipy.signal.correlate(sample({len(a)}), {b}, mode='{mode}', method='direct')",
            )

    b_iir = [0.2, 0.3, 0.2]
    a_iir = [1.0, -0.5]
    add("lfilter_iir_64", sg.lfilter(b_iir, a_iir, sample(64)), "scipy.signal.lfilter([0.2, 0.3, 0.2], [1.0, -0.5], sample(64))")
    add("lfilter_iir_1000", sg.lfilter(b_iir, a_iir, sample(1000)), "scipy.signal.lfilter([0.2, 0.3, 0.2], [1.0, -0.5], sample(1000))")
    add(
        "lfilter_fir_100",
        sg.lfilter([0.25] * 4, [1.0], sample(100)),
        "scipy.signal.lfilter([0.25, 0.25, 0.25, 0.25], [1.0], sample(100))",
    )
    add("lfilter_a0_2", sg.lfilter([1.0, 2.0], [2.0, -1.0], sample(64)), "scipy.signal.lfilter([1.0, 2.0], [2.0, -1.0], sample(64))")


    sos4 = sg.butter(4, 0.3, btype="lowpass", output="sos")
    sos3 = sg.butter(3, 0.2, btype="highpass", output="sos")
    add(
        "sos_o4_wn0p3_low",
        sos4.flatten(),
        "scipy.signal.butter(4, 0.3, btype='lowpass', output='sos').flatten()",
    )
    add(
        "sosfilt_o4_wn0p3_low",
        sg.sosfilt(sos4, sample(64)),
        "scipy.signal.sosfilt(scipy.signal.butter(4, 0.3, btype='lowpass', output='sos'), sample(64))",
    )
    add(
        "sos_o3_wn0p2_high",
        sos3.flatten(),
        "scipy.signal.butter(3, 0.2, btype='highpass', output='sos').flatten()",
    )
    add(
        "sosfilt_o3_wn0p2_high",
        sg.sosfilt(sos3, sample(64)),
        "scipy.signal.sosfilt(scipy.signal.butter(3, 0.2, btype='highpass', output='sos'), sample(64))",
    )
    for order, w, btype in ((2, "0p3", "lowpass"), (4, "0p3", "lowpass"), (2, "0p2", "highpass"), (3, "0p2", "highpass")):
        wn = float(w.replace("p", "."))
        add(
            f"butter_o{order}_wn{w}_{btype}",
            sg.sosfilt(sg.butter(order, wn, btype=btype, output="sos"), sample(64)),
            f"scipy.signal.sosfilt(scipy.signal.butter({order}, {wn}, btype='{btype}', output='sos'), sample(64))",
        )

    d = sample(40)
    peak_cases = (
        ("all", {"prominence": 0}),
        ("height", {"prominence": 0, "height": 10.0}),
        ("distance", {"prominence": 0, "distance": 5}),
        ("prominence", {"prominence": 8.0}),
    )
    for name, kwargs in peak_cases:
        peaks, props = sg.find_peaks(d, **kwargs)
        opts = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        psrc = f"scipy.signal.find_peaks(sample(40), {opts})"
        add(f"peaks_{name}_index", peaks.astype(float), psrc + " peaks")
        add(f"peaks_{name}_height", d[peaks], psrc + " x[peaks]")
        add(f"peaks_{name}_prominence", props["prominences"], psrc + " prominences")

    x256 = sample(256)
    stft_src = (
        "numpy.fft.rfft(x[s:s+w] * scipy.signal.windows.hann(w, sym=True)) "
        "(kit stft: symmetric Hann, no scaling, no padding)"
    )
    for w, h, fs in ((64, 32, 100.0), (100, 25, 1000.0)):
        prefix = f"stft_w{w}_h{h}"
        starts = list(range(0, 256 - w + 1, h))
        win = sg.windows.hann(w, sym=True)
        zs = [np.fft.rfft(x256[s : s + w] * win) for s in starts]
        add(f"{prefix}_n_frames", float(len(starts)), stft_src)
        add(f"{prefix}_freqs", np.fft.rfftfreq(w, 1.0 / fs), stft_src)
        add(f"{prefix}_times", [(s + (w - 1) / 2) / fs for s in starts], stft_src)
        add_complex(f"{prefix}_frame0", zs[0], stft_src)
        add_complex(f"{prefix}_last", zs[-1], stft_src)
        add_complex(f"{prefix}_all", np.concatenate(zs), stft_src)

    return rows


def calc_rows():
    """Interpolation, optimization and integration rows (calc_reference.tsv).

    Each input is an integer formula that reference_calc.rs rebuilds exactly.
    """
    rows = []

    def add(key, value, source):
        rows.append((key, common.stable_value(value), source))

    # Interpolation: xs[i] = i * 0.5, ys[i] = ((i * 37 + 11) % 101) / 4.0, xq[k] = (k * 9 + 3) / 20.0.
    xs = [i * 0.5 for i in range(20)]
    ys = [((i * 37 + 11) % 101) / 4.0 for i in range(20)]
    xq = [(k * 9 + 3) / 20.0 for k in range(21)]
    add(
        "interp1d_linear",
        ip.interp1d(xs, ys, kind="linear")(xq),
        "scipy.interpolate.interp1d(xs, ys, kind='linear')(xq), xs[i]=i*0.5, ys[i]=((i*37+11)%101)/4.0, xq[k]=(k*9+3)/20.0",
    )
    add(
        "interp1d_cubic_natural",
        ip.CubicSpline(xs, ys, bc_type="natural")(xq),
        "scipy.interpolate.CubicSpline(xs, ys, bc_type='natural')(xq) (kit builds a natural spline; interp1d kind='cubic' is not-a-knot)",
    )

    # Grid for interp2d: xg[i] = i * 0.8, yg[j] = j * 0.5, z[i][j] = ((i*7 + j*13 + 3) % 17) / 2.0.
    xg = [i * 0.8 for i in range(6)]
    yg = [j * 0.5 for j in range(7)]
    z = np.array([[((i * 7 + j * 13 + 3) % 17) / 2.0 for j in range(7)] for i in range(6)])
    px = [((k * 13 + 2) % 39) / 10.0 for k in range(12)]
    py = [((k * 7 + 1) % 29) / 10.0 for k in range(12)]
    add(
        "interp2d_linear",
        ip.RegularGridInterpolator((xg, yg), z, method="linear")(np.column_stack([px, py])),
        "scipy.interpolate.RegularGridInterpolator((xg, yg), z, method='linear')(column_stack(px, py)), px[k]=((k*13+2)%39)/10.0, py[k]=((k*7+1)%29)/10.0",
    )

    # curve_fit: x[i] = i * 0.25, y[i] = 2.5*exp(-1.3*x) + 0.5 + 0.01*(((i*37+11)%101) - 50)/50.0.
    cx = [i * 0.25 for i in range(40)]
    cy = [2.5 * math.exp(-1.3 * x) + 0.5 + 0.01 * (((i * 37 + 11) % 101) - 50) / 50.0 for i, x in enumerate(cx)]
    fit = op.curve_fit(lambda x, a, b, c: a * np.exp(-b * x) + c, cx, cy, p0=[2.0, 1.0, 0.4], xtol=1e-12, ftol=1e-12)[0]
    # The iterative fit differs between BLAS builds at about 1e-10 relative, so keep 8 digits (test tolerance 1e-6).
    rows.append(("curve_fit_params", common.stable_value(fit, 8), "scipy.optimize.curve_fit(a*exp(-b*x)+c, x, y, p0=[2.0,1.0,0.4], xtol=1e-12, ftol=1e-12)[0], same x, y"))

    # linprog: minimise c.x subject to A_ub x <= b_ub, x >= 0 (HiGHS).
    lp_a = op.linprog([-1.0, -2.0], A_ub=[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], b_ub=[4.0, 3.0, 3.0], bounds=(0, None), method="highs")
    add("linprog_a_x", lp_a.x, "scipy.optimize.linprog([-1,-2], A_ub=[[1,1],[1,0],[0,1]], b_ub=[4,3,3], bounds=(0,None), method='highs').x")
    add("linprog_a_fun", float(lp_a.fun), "scipy.optimize.linprog(...same as linprog_a_x...).fun")
    lp_b = op.linprog(
        [-5.0, -4.0, -3.0],
        A_ub=[[2.0, 3.0, 1.0], [4.0, 1.0, 2.0], [3.0, 4.0, 2.0]],
        b_ub=[5.0, 11.0, 8.0],
        bounds=(0, None),
        method="highs",
    )
    add("linprog_b_x", lp_b.x, "scipy.optimize.linprog([-5,-4,-3], A_ub=[[2,3,1],[4,1,2],[3,4,2]], b_ub=[5,11,8], bounds=(0,None), method='highs').x")
    add("linprog_b_fun", float(lp_b.fun), "scipy.optimize.linprog(...same as linprog_b_x...).fun")

    # Rosenbrock and a convex bowl, from the given starts.
    def bowl(p):
        return (p[0] - 1) ** 2 + 2 * (p[1] + 2) ** 2 + 0.5 * p[0] * p[1]

    def bowl_grad(p):
        return np.array([2 * (p[0] - 1) + 0.5 * p[1], 4 * (p[1] + 2) + 0.5 * p[0]])

    rb = op.minimize(op.rosen, [-1.2, 1.0], jac=op.rosen_der, method="BFGS", options={"gtol": 1e-12})
    add("rosen_bfgs_x", rb.x, "scipy.optimize.minimize(rosen, [-1.2, 1.0], jac=rosen_der, method='BFGS', options={'gtol': 1e-12}).x")
    add("rosen_bfgs_fun", float(rb.fun), "same call, .fun")
    nm_opts = {"xatol": 1e-12, "fatol": 1e-12, "maxiter": 20000, "maxfev": 20000}
    rn = op.minimize(op.rosen, [-1.2, 1.0], method="Nelder-Mead", options=nm_opts)
    add("rosen_nm_x", rn.x, "scipy.optimize.minimize(rosen, [-1.2, 1.0], method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 20000, 'maxfev': 20000}).x")
    add("rosen_nm_fun", float(rn.fun), "same call, .fun")
    bb = op.minimize(bowl, [0.0, 0.0], jac=bowl_grad, method="BFGS", options={"gtol": 1e-12})
    add("bowl_bfgs_x", bb.x, "scipy.optimize.minimize((p0-1)**2 + 2*(p1+2)**2 + 0.5*p0*p1, [0,0], jac=grad, method='BFGS', options={'gtol': 1e-12}).x")
    bn = op.minimize(bowl, [0.0, 0.0], method="Nelder-Mead", options=nm_opts)
    add("bowl_nm_x", bn.x, "scipy.optimize.minimize((p0-1)**2 + 2*(p1+2)**2 + 0.5*p0*p1, [0,0], method='Nelder-Mead', options=nm_opts).x")

    # Roots: a cubic on [2, 3] and cos(x) - x on [0, 1].
    def fcub(x):
        return x**3 - 2 * x - 5

    def dcub(x):
        return 3 * x**2 - 2

    def gcos(x):
        return math.cos(x) - x

    def dgcos(x):
        return -math.sin(x) - 1

    eps4 = 4 * np.finfo(float).eps
    add("root_cubic_brentq", op.brentq(fcub, 2.0, 3.0, xtol=1e-14, rtol=eps4), "scipy.optimize.brentq(x**3-2x-5, 2, 3, xtol=1e-14, rtol=4*eps)")
    add("root_cubic_bisect", op.bisect(fcub, 2.0, 3.0, xtol=1e-14, rtol=eps4), "scipy.optimize.bisect(x**3-2x-5, 2, 3, xtol=1e-14, rtol=4*eps)")
    add("root_cubic_newton", op.newton(fcub, 2.0, fprime=dcub, tol=1e-14), "scipy.optimize.newton(x**3-2x-5, 2.0, fprime=3x**2-2, tol=1e-14)")
    add("root_cos_brentq", op.brentq(gcos, 0.0, 1.0, xtol=1e-14, rtol=eps4), "scipy.optimize.brentq(cos(x)-x, 0, 1, xtol=1e-14, rtol=4*eps)")
    add("root_cos_bisect", op.bisect(gcos, 0.0, 1.0, xtol=1e-14, rtol=eps4), "scipy.optimize.bisect(cos(x)-x, 0, 1, xtol=1e-14, rtol=4*eps)")
    add("root_cos_newton", op.newton(gcos, 0.5, fprime=dgcos, tol=1e-14), "scipy.optimize.newton(cos(x)-x, 0.5, fprime=-sin(x)-1, tol=1e-14)")

    # Quadrature of closed-form integrands.
    integrands = [
        ("quad_sin", np.sin, 0.0, math.pi, "sin(x) on [0, pi]"),
        ("quad_gauss", lambda x: np.exp(-x * x), 0.0, 2.0, "exp(-x*x) on [0, 2]"),
        ("quad_lorentz", lambda x: 1.0 / (1.0 + x * x), 0.0, 1.0, "1/(1+x*x) on [0, 1]"),
        ("quad_xlog", lambda x: x * np.log(x + 1.0), 0.0, 2.0, "x*log(x+1) on [0, 2]"),
        ("quad_damped", lambda x: np.exp(-x) * np.sin(x), 0.0, 10.0, "exp(-x)*sin(x) on [0, 10]"),
    ]
    for key, fn, a, b, desc in integrands:
        value = ig.quad(fn, a, b, epsabs=1e-13, epsrel=1e-13)[0]
        add(key, float(value), f"scipy.integrate.quad({desc}, epsabs=1e-13, epsrel=1e-13)[0]")
    for n in range(1, 6):
        add(f"fixed_quad_exp_n{n}", float(ig.fixed_quad(np.exp, 0.5, 2.5, n=n)[0]), f"scipy.integrate.fixed_quad(exp, 0.5, 2.5, n={n})[0]")
        add(
            f"fixed_quad_cos_n{n}",
            float(ig.fixed_quad(lambda x: np.cos(3 * x), 0.5, 2.5, n=n)[0]),
            f"scipy.integrate.fixed_quad(cos(3x), 0.5, 2.5, n={n})[0]",
        )

    # Data integrals: y[i] = ((i*37+11)%101)/4.0 for i in 0..41, dx = 0.25 (40 intervals, even).
    dy = np.array([((i * 37 + 11) % 101) / 4.0 for i in range(41)])
    add("trapezoid_data", float(ig.trapezoid(dy, dx=0.25)), "scipy.integrate.trapezoid(y, dx=0.25), y[i]=((i*37+11)%101)/4.0, 41 points")
    add("simpson_data", float(ig.simpson(dy, dx=0.25)), "scipy.integrate.simpson(y, dx=0.25), same y (even number of intervals only)")
    add("cumtrapz_data", ig.cumulative_trapezoid(dy, dx=0.25), "scipy.integrate.cumulative_trapezoid(y, dx=0.25), same y")

    # One fixed step of the system f(t, y) = [y1, -y0 + t] from t = 0.3, y = [1.0, 0.5], dt = 0.1.
    # No SciPy function does one fixed step: the textbook formulas, with NumPy.
    def f_sys(t, y):
        return np.array([y[1], -y[0] + t])

    t0, y0, dt = 0.3, np.array([1.0, 0.5]), 0.1
    add("euler_step", y0 + dt * f_sys(t0, y0), "textbook Euler: y + dt*f(t, y) (no SciPy equal)")
    k1 = f_sys(t0, y0)
    k2 = f_sys(t0 + dt / 2, y0 + dt / 2 * k1)
    k3 = f_sys(t0 + dt / 2, y0 + dt / 2 * k2)
    k4 = f_sys(t0 + dt, y0 + dt * k3)
    add("rk4_step", y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), "textbook RK4 step (no SciPy equal)")

    # Reference solutions over [0, 1] with DOP853 at a tight tolerance.
    add(
        "ivp_exp",
        ig.solve_ivp(lambda t, y: -y, (0.0, 1.0), [1.0], method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1],
        "scipy.integrate.solve_ivp(-y, (0, 1), [1.0], method='DOP853', rtol=1e-13, atol=1e-14).y[:, -1]",
    )
    add(
        "ivp_osc",
        ig.solve_ivp(lambda t, y: np.array([y[1], -y[0]]), (0.0, 1.0), [1.0, 0.0], method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1],
        "scipy.integrate.solve_ivp([y1, -y0], (0, 1), [1.0, 0.0], method='DOP853', rtol=1e-13, atol=1e-14).y[:, -1]",
    )
    return rows


def ts_y(n):
    """The shared series of the ts group. Rust builds the same values (scikit ts tests)."""
    return np.array(
        [10.0 + 0.05 * i + 3.0 * math.sin(2.0 * math.pi * i / 12.0) + ((i * 37 + 11) % 101) / 50.0 for i in range(n)],
        dtype=float,
    )


def ts_q(n):
    """Integer-built series for the Ljung-Box p-values: ((i*i*31 + i*17 + 7) % 103) / 10."""
    return np.array([((i * i * 31 + i * 17 + 7) % 103) / 10.0 for i in range(n)], dtype=float)


def ts_b(n):
    """The ARIMA input: a trend, a period-7 cycle, the integer ripple of ts_y and a period-5 step."""
    return np.array(
        [5.0 + 2.0 * math.sin(2.0 * math.pi * i / 7.0) + ((i * 37 + 11) % 101) / 25.0 + (i % 5) * 0.3 for i in range(n)],
        dtype=float,
    )


def nan_vec_rows(prefix, arr, source):
    """An array with NaN at both ends: its valid range (first, last, count), 5 values at each end, and the sum of absolute valid values."""
    arr = np.asarray(arr, dtype=float)
    idx = np.flatnonzero(~np.isnan(arr))
    first, last = int(idx[0]), int(idx[-1])
    assert np.all(np.isfinite(arr[first : last + 1])), "a NaN inside the valid range"
    return [
        (prefix + "_valid", [float(first), float(last), float(len(idx))], source),
        (prefix + "_head", arr[first : first + 5].tolist(), source),
        (prefix + "_tail", arr[last - 4 : last + 1].tolist(), source),
        (prefix + "_abs_sum", float(np.abs(arr[idx]).sum()), source),
    ]


def hw_multiplicative(y, alpha, beta, gamma, m):
    """The multiplicative component form (Hyndman and Athanasopoulos, section 8.3), as kit computes it: season divided by the level. Returns fitted, [level, trend] and the 14 forecasts."""
    n = len(y)
    fa = float(y[:m].mean())
    sa = float(y[m : 2 * m].mean())
    level = fa
    trend = (sa - fa) / m
    seas = list(y[:m] / fa)
    fitted = [(level + trend) * seas[0]]
    for i in range(1, n):
        prev = level
        k = i % m
        level = alpha * (y[i] / seas[k]) + (1 - alpha) * (prev + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
        seas[k] = gamma * (y[i] / level) + (1 - gamma) * seas[k]
        fitted.append((level + trend) * seas[(i + 1) % m])
    forecast = [(level + trend * h) * seas[(n + h - 1) % m] for h in range(1, 15)]
    return fitted, [level, trend], forecast


def ts_arima_css(ma, ar, c):
    """The conditional sum of squares of kit's CSS fit (compute_css in arima.rs)."""
    start = max(len(ar), len(ma))
    res = np.zeros(len(c))
    total = 0.0
    for t in range(start, len(c)):
        pred = sum(ar[j] * c[t - 1 - j] for j in range(len(ar))) + sum(ma[j] * res[t - 1 - j] for j in range(len(ma)))
        res[t] = c[t] - pred
        total += res[t] ** 2
    return total


def ts_rows():
    """Rows of the scikit::ts group (ts_reference.tsv): acf, arima, decompose, ops, resample and smoothing."""
    warnings.filterwarnings("ignore")
    rows = []

    def put(items):
        rows.extend((key, common.stable_value(value), source) for key, value, source in items)

    def add(key, value, source):
        rows.append((key, common.stable_value(value), source))

    # acf and Ljung-Box
    y = ts_y(500)
    put(vec_rows("acf_y500", sm_acf(y, nlags=24, adjusted=False, fft=False), "statsmodels.tsa.stattools.acf(y, nlags=24, adjusted=False, fft=False)"))
    put(vec_rows("pacf_y500", sm_pacf(y, nlags=24, method="ywm")[1:], "statsmodels.tsa.stattools.pacf(y, nlags=24, method='ywm')[1:]"))
    for series_name, x, lags in (("q500", ts_q(500), 5), ("q500", ts_q(500), 20), ("y500", y, 10)):
        r = acorr_ljungbox(x, lags=[lags], return_df=True)
        add(
            f"lb_{series_name}_lag{lags}",
            [float(r.lb_stat.iloc[0]), float(r.lb_pvalue.iloc[0])],
            f"statsmodels.stats.diagnostic.acorr_ljungbox({series_name}, lags=[{lags}]): lb_stat, lb_pvalue",
        )
    add("is_significant_p", [0.03, 0.03, 0.05, 0.0], "closed form: the p-values of the test")
    add("is_significant_alpha", [0.05, 0.01, 0.05, 0.05], "closed form: the significance levels of the test")
    add(
        "is_significant_expected",
        [float(p < a) for p, a in zip([0.03, 0.03, 0.05, 0.0], [0.05, 0.01, 0.05, 0.05])],
        "closed form: p_value < alpha (1 = significant)",
    )

    # arima: b = ts_b(120)
    b = ts_b(120)
    for d in (1, 2, 3):
        put(vec_rows(f"arima_diff{d}", np.diff(b, n=d), f"numpy.diff(b, n={d})"))

    def ar_part(series, p):
        mean = float(series.mean())
        phi = yule_walker(series - mean, order=p, method="mle")[0] if p > 0 else np.array([])
        return [float(v) for v in phi], mean

    for p, d, q in ((2, 0, 0), (1, 1, 0), (2, 1, 0), (1, 0, 1)):
        dd = np.diff(b, n=d)
        phi, mean = ar_part(dd, p)
        add(
            f"arima_fit_ar_{p}{d}{q}",
            phi + [mean],
            f"statsmodels.regression.linear_model.yule_walker(numpy.diff(b, {d}) - mean, order={p}, method='mle')[0], numpy.diff(b, {d}).mean()",
        )
    for p, d, q in ((0, 0, 1), (0, 1, 1), (1, 0, 1)):
        dd = np.diff(b, n=d)
        phi, mean = ar_part(dd, p)
        c = dd - dd.mean()
        theta = op.minimize(
            lambda m: ts_arima_css(m, phi, c),
            x0=np.zeros(q),
            method="Nelder-Mead",
            options=dict(xatol=1e-10, fatol=1e-14),
        ).x
        add(
            f"arima_fit_ma_{p}{d}{q}",
            [float(theta[0])],
            "scipy.optimize.minimize(css, Nelder-Mead) of the CSS of kit's model",
        )

    phi200, mean200 = ar_part(b, 2)
    res = ARIMA(b, order=(2, 0, 0), trend="c").smooth(np.r_[mean200, phi200, 1.0])
    fv = np.asarray(res.fittedvalues, dtype=float)
    put(vec_rows("arima_fitted_200", fv[2:], "statsmodels.tsa.arima.model.ARIMA(b, order=(2, 0, 0), trend='c').smooth(...).fittedvalues[2:]"))
    # d = 1 fitted values on the levels: fitted[t] = b[t-1] + one-step prediction of diff[t-1], for t >= p + 1
    for p in (1, 2):
        dd = np.diff(b, n=1)
        phi, mean = ar_part(dd, p)
        fv_diff = np.asarray(ARIMA(dd, order=(p, 0, 0), trend="c").smooth(np.r_[mean, phi, 1.0]).fittedvalues, dtype=float)
        levels = (b[:-1] + fv_diff)[p:]
        put(vec_rows(f"arima_fitted_{p}10", levels, f"b[t-1] + ARIMA(numpy.diff(b, 1), order=({p}, 0, 0), trend='c').smooth(...).fittedvalues[t-1], for t >= {p + 1}"))
    for p, d, q in ((2, 0, 0), (1, 1, 0), (2, 1, 0)):
        dd = np.diff(b, n=d)
        phi, mean = ar_part(dd, p)
        fc = np.asarray(ARIMA(dd, order=(p, 0, 0), trend="c").smooth(np.r_[mean, phi, 1.0]).forecast(5), dtype=float)
        if d == 1:
            fc = b[-1] + np.cumsum(fc)
        put([(f"arima_forecast_{p}{d}{q}", fc.tolist(), f"ARIMA(numpy.diff(b, {d}), order=({p}, 0, 0), trend='c').smooth(...).forecast(5){' cumulated from b[-1]' if d == 1 else ''}")])
        resid_res = ARIMA(dd, order=(p, 0, 0), trend="c").smooth(np.r_[mean, phi, 1.0])
        resid = dd[p:] - np.asarray(resid_res.fittedvalues, dtype=float)[p:]
        nn = len(resid)
        aic = nn * math.log((resid ** 2).sum() / nn) + 2 * (p + 1)
        add(
            f"arima_aic_{p}{d}{q}",
            aic,
            "closed form of ArimaModel::aic: n * ln(RSS / n) + 2 * (p + q + 1), RSS from statsmodels' one-step residuals",
        )

    # decompose: y = ts_y(500)
    for period in (12, 7):
        for model in ("additive", "multiplicative"):
            tag = f"{'add' if model == 'additive' else 'mul'}{period}"
            r = sm_decompose(y, model=model, period=period, two_sided=True)
            seasonal = np.asarray(r.seasonal, dtype=float)
            trend = np.asarray(r.trend, dtype=float)
            resid = np.asarray(r.resid, dtype=float)
            put(vec_rows(f"dec_{tag}_seasonal", seasonal[:period], f"statsmodels.tsa.seasonal.seasonal_decompose(y, model='{model}', period={period}, two_sided=True).seasonal[:{period}]"))
            put(nan_vec_rows(f"dec_{tag}_trend", trend, f"seasonal_decompose(y, model='{model}', period={period}, two_sided=True).trend"))
            put(nan_vec_rows(f"dec_{tag}_resid", resid, f"seasonal_decompose(y, model='{model}', period={period}, two_sided=True).resid"))
            ok = ~np.isnan(trend)
            rr = resid[ok]
            tr = trend[ok]
            det = tr + rr if model == "additive" else tr * rr
            rs = rr + seasonal[ok] if model == "additive" else rr * seasonal[ok]
            ts_strength = max(0.0, 1 - np.var(rr) / np.var(det))
            ss_strength = max(0.0, 1 - np.var(rr) / np.var(rs))
            add(
                f"dec_{tag}_strength",
                [float(ts_strength), float(ss_strength)],
                "closed form of the published definitions (Wang, Smith and Hyndman 2006): 1 - var(resid) / var(trend + resid) (multiplicative: trend * resid) and 1 - var(resid) / var(resid + seasonal)",
            )

    # ops: x = sample(60)
    x = sample(60)
    S = pd.Series(x)
    put(vec_rows("ewma_a0p3", S.ewm(alpha=0.3, adjust=False).mean().values, "pandas.Series.ewm(alpha=0.3, adjust=False).mean()"))
    put(vec_rows("ewma_span10", S.ewm(span=10, adjust=False).mean().values, "pandas.Series.ewm(span=10, adjust=False).mean()"))
    put(vec_rows("ewma_halflife3", S.ewm(halflife=3, adjust=False).mean().values, "pandas.Series.ewm(halflife=3, adjust=False).mean()"))
    put(vec_rows("expanding_mean", S.expanding().mean().values, "pandas.Series.expanding().mean()"))
    put(vec_rows("expanding_sum", S.expanding().sum().values, "pandas.Series.expanding().sum()"))
    put(vec_rows("expanding_min", S.expanding().min().values, "pandas.Series.expanding().min()"))
    put(vec_rows("expanding_max", S.expanding().max().values, "pandas.Series.expanding().max()"))
    put(vec_rows("expanding_std", S.expanding(min_periods=2).std(ddof=0).values, "pandas.Series.expanding(min_periods=2).std(ddof=0)"))
    put(vec_rows("lag3", pd.Series(sample(12)).shift(3).values, "pandas.Series.shift(3)"))
    put(vec_rows("lead2", pd.Series(sample(12)).shift(-2).values, "pandas.Series.shift(-2)"))

    # resample
    x58 = sample(58)
    groups = np.arange(58) // 4
    for name in ("mean", "sum", "min", "max", "first", "last", "median"):
        put(vec_rows(f"downsample_{name}", pd.Series(x58).groupby(groups).agg(name).values, f"pandas.Series.groupby(arange(58) // 4).agg('{name}')"))
    x5 = sample(5)
    n5, f3 = 5, 3
    ps = pd.Series(x5, index=np.arange(n5) * f3)
    put(vec_rows("upsample_linear", np.interp(np.arange((n5 - 1) * f3 + 1) / f3, np.arange(n5), x5), "numpy.interp(arange((n-1)*f+1) / f, arange(n), x)"))
    put(vec_rows("upsample_ffill", ps.reindex(range(n5 * f3), method="ffill").values, "pandas.Series.reindex(range(n*f), method='ffill')"))
    put(vec_rows("upsample_bfill", ps.reindex(range(n5 * f3), method="bfill").ffill().values, "pandas.Series.reindex(range(n*f), method='bfill').ffill()"))
    put(vec_rows("upsample_nearest", ps.reindex(range((n5 - 1) * f3 + 1), method="nearest").values, "pandas.Series.reindex(range((n-1)*f+1), method='nearest')"))
    x9 = sample(9)
    for k in (14, 5, 9):
        put(vec_rows(f"resample_to_len_{k}", np.interp(np.linspace(0, 8, k), np.arange(9), x9), f"numpy.interp(numpy.linspace(0, 8, {k}), arange(9), x)"))

    # smoothing: y = ts_y(60)
    y60 = ts_y(60)
    ses = SimpleExpSmoothing(y60, initialization_method="known", initial_level=float(y60[0])).fit(smoothing_level=0.3, optimized=False)
    put(vec_rows("ses_a0p3_fitted", np.asarray(ses.level, dtype=float), "statsmodels.tsa.holtwinters.SimpleExpSmoothing(y, initialization_method='known', initial_level=y[0]).fit(smoothing_level=0.3, optimized=False).level"))
    add("ses_a0p3_level", float(ses.level[-1]), "statsmodels: .level[-1]")
    put(vec_rows("ses_a0p3_forecast3", np.asarray(ses.forecast(3), dtype=float), "statsmodels: .forecast(3)"))

    holt_fit = Holt(
        y60[1:],
        initialization_method="known",
        initial_level=float(y60[0]),
        initial_trend=float(y60[1] - y60[0]),
    ).fit(smoothing_level=0.5, smoothing_trend=0.2, optimized=False)
    holt_fitted = np.r_[y60[0], np.asarray(holt_fit.level, dtype=float) + np.asarray(holt_fit.trend, dtype=float)]
    put(vec_rows("holt_a0p5_b0p2_fitted", holt_fitted, "y[0], then statsmodels Holt(y[1:]) .level + .trend"))
    add("holt_a0p5_b0p2_state", [float(holt_fit.level[-1]), float(holt_fit.trend[-1])], "statsmodels Holt(y[1:]): [.level[-1], .trend[-1]]")
    put(vec_rows("holt_a0p5_b0p2_forecast5", np.asarray(holt_fit.forecast(5), dtype=float), "statsmodels Holt(y[1:]): .forecast(5)"))

    m = 12
    fa = float(y60[:m].mean())
    sa = float(y60[m : 2 * m].mean())
    s0 = np.roll(y60[:m] - fa, -1)
    hw = ExponentialSmoothing(
        y60[1:],
        trend="add",
        seasonal="add",
        seasonal_periods=m,
        initialization_method="known",
        initial_level=fa,
        initial_trend=(sa - fa) / m,
        initial_seasonal=s0,
    ).fit(smoothing_level=0.4, smoothing_trend=0.1, smoothing_seasonal=0.3 * (1 - 0.4), optimized=False)
    hw_fitted = np.r_[y60[0] + (sa - fa) / m, np.asarray(hw.fittedvalues, dtype=float)[1:], np.asarray(hw.forecast(1), dtype=float)]
    put(vec_rows("hw_add_fitted", hw_fitted, "statsmodels ExponentialSmoothing(additive, known start, smoothing_seasonal=gamma*(1-alpha)).fittedvalues, with y[0] first"))
    add("hw_add_state", [float(hw.level[-1]), float(hw.trend[-1])], "statsmodels ExponentialSmoothing: [.level[-1], .trend[-1]]")
    hw_forecast = [float(hw.level[-1] + hw.trend[-1] * h + hw.season[-12 + (h - 1) % 12]) for h in range(1, 15)]
    put(vec_rows("hw_add_forecast14", hw_forecast, "closed form from statsmodels states: level + trend*h + season[-12 + (h-1)%12]"))

    mfit, mstate, mfc = hw_multiplicative(y60, 0.4, 0.1, 0.3, m)
    put(vec_rows("hw_mul_fitted", mfit, "numpy loop of the multiplicative component form (Hyndman and Athanasopoulos, section 8.3); statsmodels uses y / (level + trend) in the season, so it has no equal"))
    add("hw_mul_state", mstate, "numpy loop of the multiplicative component form: [level, trend]")
    put(vec_rows("hw_mul_forecast14", mfc, "numpy loop of the multiplicative component form: (level + trend*h) * season[(n+h-1)%12]"))
    return rows


def spatial_sparse_rows():
    """Rows of the scikit::spatial and scikit::sparse groups (spatial_sparse_reference.tsv).

    Rust (crates/scikit/tests/it/reference_spatial_sparse.rs) rebuilds every input
    with the integer formulas written here; the formulas must match exactly.
    """
    rows = []

    def put(items):
        rows.extend((key, common.stable_value(value), source) for key, value, source in items)

    def add(key, value, source):
        rows.append((key, common.stable_value(value), source))

    # Inputs (the same integer formulas are in reference_spatial_sparse.rs).
    P = np.array([[((i * 31 + d * 17 + i * d) % 23) / 7.0 for d in range(3)] for i in range(50)])
    H = np.array([[float((i * 13 + 5) % 17), float((i * 7 + 3) % 19)] for i in range(30)])
    T = np.array([[((i * 37 + d * 53 + i * d * 7) % 101) / 10.0 for d in range(3)] for i in range(40)])
    Q = np.array([[((k * 29 + d * 11 + k * d * 3) % 97) / 9.7 + 0.013 for d in range(3)] for k in range(5)])
    xv = np.array([((j * 5 + 3) % 11) / 4.0 for j in range(50)])
    A = np.array(
        [[float((i + 2 * j) % 9 + 1) if (i * 7 + j * 3) % 11 == 0 else 0.0 for j in range(50)] for i in range(50)]
    )
    B = np.array(
        [[float((3 * i + j) % 7 + 1) if (i * 5 + j * 2) % 13 == 0 else 0.0 for j in range(50)] for i in range(50)]
    )
    ALPHA = 2.5
    PSRC = "P[i][d] = ((i*31 + d*17 + i*d) % 23) / 7.0 (50 points, 3 dimensions)"

    # Distances from point 0 to points 1..49.
    put(vec_rows("euclidean_from0", [dist.euclidean(P[0], P[k]) for k in range(1, 50)], f"scipy.spatial.distance.euclidean(P[0], P[k]), {PSRC}"))
    put(vec_rows("manhattan_from0", [dist.cityblock(P[0], P[k]) for k in range(1, 50)], f"scipy.spatial.distance.cityblock(P[0], P[k]), {PSRC}"))
    put(vec_rows("chebyshev_from0", [dist.chebyshev(P[0], P[k]) for k in range(1, 50)], f"scipy.spatial.distance.chebyshev(P[0], P[k]), {PSRC}"))
    put(vec_rows("minkowski3_from0", [dist.minkowski(P[0], P[k], 3) for k in range(1, 50)], f"scipy.spatial.distance.minkowski(P[0], P[k], 3), {PSRC}"))
    put(vec_rows("cosine_from0", [dist.cosine(P[0], P[k]) for k in range(1, 50)], f"scipy.spatial.distance.cosine(P[0], P[k]), {PSRC}"))

    # Condensed and rectangular distance matrices.
    for name, metric in [("euclidean", "euclidean"), ("cityblock", "cityblock"), ("chebyshev", "chebyshev"), ("cosine", "cosine")]:
        put(vec_rows(f"pdist_{name}", dist.pdist(P, metric), f"scipy.spatial.distance.pdist(P, '{metric}'), {PSRC}"))
        put(vec_rows(f"cdist_{name}", dist.cdist(P[:20], P[20:], metric).ravel(), f"scipy.spatial.distance.cdist(P[:20], P[20:], '{metric}').ravel(), {PSRC}"))
    sq = dist.squareform(dist.pdist(P))
    put(vec_rows("squareform_euclidean", sq.ravel(), f"scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(P)).ravel(), {PSRC}"))
    put(vec_rows("squareform_row7", sq[7], f"scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(P))[7], {PSRC}"))

    # Convex hull of the 30 points H (2-D: volume is the area).
    hull = spt.ConvexHull(H)
    put(vec_rows("hull_vertices", [float(v) for v in sorted(hull.vertices)], "sorted(scipy.spatial.ConvexHull(H).vertices), H[i] = ((i*13+5) % 17, (i*7+3) % 19)"))
    add("hull_area", float(hull.volume), "scipy.spatial.ConvexHull(H).volume (the area in 2-D), H[i] = ((i*13+5) % 17, (i*7+3) % 19)")

    # Nearest neighbours and balls of the 40 points T for 5 query points Q.
    tree = spt.cKDTree(T)
    for k in range(5):
        d_all = np.linalg.norm(T - Q[k], axis=1)
        srt = np.sort(d_all)
        assert np.all(np.diff(srt[:6]) > 1e-6), f"query {k}: a tie among the 6 nearest distances"
        assert np.all(np.abs(d_all - 4.5) > 1e-6), f"query {k}: a distance lies on the radius 4.5"
        dd, ii = tree.query(Q[k], k=5)
        ball = sorted(tree.query_ball_point(Q[k], 4.5))
        assert len(ball) > 0, f"query {k}: the ball is empty"
        put(vec_rows(f"kdtree_query{k}_index", ii.astype(float), f"scipy.spatial.cKDTree(T).query(Q[{k}], k=5)[1]"))
        put(vec_rows(f"kdtree_query{k}_dist", dd, f"scipy.spatial.cKDTree(T).query(Q[{k}], k=5)[0]"))
        put(vec_rows(f"kdtree_ball{k}_index", np.array(ball, dtype=float), f"sorted(scipy.spatial.cKDTree(T).query_ball_point(Q[{k}], 4.5))"))
        put(vec_rows(f"kdtree_ball{k}_dist", np.sort(d_all[ball]), f"numpy.sort(numpy.linalg.norm(T[i] - Q[{k}])) for i in the sorted ball of query {k}"))

    # CSR and CSC of the 50 x 50 matrix A (B for the sum).
    for fmt, make in [("csr", ssp.csr_matrix), ("csc", ssp.csc_matrix)]:
        Am = make(A)
        Bm = make(B)
        s_add = Am + Bm
        put(vec_rows(f"{fmt}_dot", Am.dot(xv), f"scipy.sparse.{fmt}_matrix(A).dot(xv), A[i][j] = ((i+2*j) % 9 + 1) where (i*7 + j*3) % 11 == 0"))
        add(f"{fmt}_add_nnz", float(s_add.nnz), f"(scipy.sparse.{fmt}_matrix(A) + scipy.sparse.{fmt}_matrix(B)).nnz")
        add(f"{fmt}_add_sum", float(s_add.sum()), f"(scipy.sparse.{fmt}_matrix(A) + scipy.sparse.{fmt}_matrix(B)).sum()")
        put(vec_rows(f"{fmt}_add_row7", s_add.toarray()[7], f"(scipy.sparse.{fmt}_matrix(A) + scipy.sparse.{fmt}_matrix(B)).toarray()[7]"))
        s_scale = ALPHA * Am
        add(f"{fmt}_scale_sum", float(s_scale.sum()), f"(2.5 * scipy.sparse.{fmt}_matrix(A)).sum()")
        put(vec_rows(f"{fmt}_scale_row7", s_scale.toarray()[7], f"(2.5 * scipy.sparse.{fmt}_matrix(A)).toarray()[7]"))
        add(f"{fmt}_density", Am.nnz / (50 * 50), f"scipy.sparse.{fmt}_matrix(A).nnz / (50 * 50) (SciPy has no density method)")
    # The CSR transpose is the one extra row set of csr.
    AT = ssp.csr_matrix(A).T.tocsr()
    add("csr_transpose_nnz", float(AT.nnz), "scipy.sparse.csr_matrix(A).T.tocsr().nnz")
    put(vec_rows("csr_transpose_dot", AT.dot(xv), "scipy.sparse.csr_matrix(A).T.tocsr().dot(xv)"))
    put(vec_rows("csr_transpose_row7", AT.toarray()[7], "scipy.sparse.csr_matrix(A).T.tocsr().toarray()[7]"))
    return rows


FILTER_FIXTURE = "crates/scikit/tests/fixtures/filter_scipy_reference.tsv"

# The filters of the lfilter rows: name -> (b, a). F1 to F5 only; a single tap has no SciPy value.
FILTERS = {
    "f1": ([0.2, 0.3, 0.2], [1.0, -0.5]),
    "f2": ([0.0985, 0.2956, 0.2956, 0.0985], [1.0, -0.5772, 0.4218, -0.0563]),
    "f3": ([2.0, 1.0], [2.0, -0.5]),
    "f4": ([0.25, 0.25, 0.25, 0.25], [1.0]),
    "f5": ([1.0, 0.5], [1.0, -0.3, 0.1]),
}


def filter_rows():
    """Rows of the scikit::signal filter group (filter_scipy_reference.tsv): lfilter_zi and lfilter with a state."""
    warnings.filterwarnings("ignore")
    rows = []

    def add(key, value, source):
        rows.append((key, common.stable_value(value), source))

    x = sample(20)
    for name, (b, a) in FILTERS.items():
        zi = sg.lfilter_zi(b, a)
        add(f"lfilter_zi.{name}", zi.tolist(), "scipy.signal.lfilter_zi(b, a)")
        zi_x = zi * x[0]
        y, zf = sg.lfilter(b, a, x, zi=zi_x)
        add(f"lfilter_state.{name}.y", np.asarray(y, dtype=float).tolist(), "scipy.signal.lfilter(b, a, x, zi=scipy.signal.lfilter_zi(b, a) * x[0]), y")
        add(f"lfilter_state.{name}.zf", np.asarray(zf, dtype=float).tolist(), "scipy.signal.lfilter(b, a, x, zi=scipy.signal.lfilter_zi(b, a) * x[0]), zf")
    b1, a1 = FILTERS["f1"]
    y0, zf0 = sg.lfilter(b1, a1, x, zi=np.zeros(2))
    add("lfilter_state.f1.y_zero", np.asarray(y0, dtype=float).tolist(), "scipy.signal.lfilter(b, a, x, zi=numpy.zeros(2)), y")
    add("lfilter_state.f1.zf_zero", np.asarray(zf0, dtype=float).tolist(), "scipy.signal.lfilter(b, a, x, zi=numpy.zeros(2)), zf")

    # filtfilt (method='pad'): 64 values, the SciPy defaults unless the case names an option.
    xf = sample(64)
    ff_cases = [
        ("f1_default", "f1", {}, ""),
        ("f1_even", "f1", {"padtype": "even"}, ", padtype='even'"),
        ("f1_constant", "f1", {"padtype": "constant"}, ", padtype='constant'"),
        ("f1_odd10", "f1", {"padtype": "odd", "padlen": 10}, ", padtype='odd', padlen=10"),
        ("f1_none", "f1", {"padtype": None}, ", padtype=None"),
        ("f2_default", "f2", {}, ""),
        ("f3_default", "f3", {}, ""),
        ("f4_default", "f4", {}, ""),
        ("f5_default", "f5", {}, ""),
        ("f4_even30", "f4", {"padtype": "even", "padlen": 30}, ", padtype='even', padlen=30"),
    ]
    for name, filt, options, extra in ff_cases:
        b, a = FILTERS[filt]
        y = sg.filtfilt(b, a, xf, **options)
        add(f"filtfilt.{name}", np.asarray(y, dtype=float).tolist(), f"scipy.signal.filtfilt(b, a, x{extra})")
    # The FIR case of the deleted row filtfilt_fir_100: F4 on 100 values, the defaults.
    b4, a4 = FILTERS["f4"]
    y100 = sg.filtfilt(b4, a4, sample(100))
    add("filtfilt.f4_n100_default", np.asarray(y100, dtype=float).tolist(),
        "scipy.signal.filtfilt([0.25, 0.25, 0.25, 0.25], [1.0], sample(100))")
    return rows


common.main(
    {
        FILTER_FIXTURE: filter_rows,
        "crates/scikit/tests/fixtures/stats_reference.tsv": stats_rows,
        "crates/scikit/tests/fixtures/stats_grid_reference.tsv": stats_grid_rows,
        "crates/scikit/tests/fixtures/fft_signal_reference.tsv": fft_signal_rows,
        "crates/scikit/tests/fixtures/calc_reference.tsv": calc_rows,
        "crates/scikit/tests/fixtures/ts_reference.tsv": ts_rows,
        "crates/scikit/tests/fixtures/spatial_sparse_reference.tsv": spatial_sparse_rows,
    }
)
