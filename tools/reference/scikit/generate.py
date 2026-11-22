"""Reference values for the scikit stats functions (docs/reference-coverage.md).

stats_rows writes crates/scikit/tests/fixtures/stats_reference.tsv. Each `source`
column is the SciPy or NumPy expression that made the value.

Run from the repository root:
    sh tools/reference/run.sh
"""

import math
import os
import sys

import numpy as np
import scipy.signal as sg
import scipy.special as sp
import scipy.stats as st

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

    ff_note = "forward and backward sg.lfilter (kit filtfilt has no padding and a zero initial state)"
    x64 = sample(64)
    add(
        "filtfilt_iir_64",
        sg.lfilter(b_iir, a_iir, sg.lfilter(b_iir, a_iir, x64)[::-1])[::-1],
        ff_note + ", b = [0.2, 0.3, 0.2], a = [1.0, -0.5], sample(64)",
    )
    fir = [0.25] * 4
    x100 = sample(100)
    add(
        "filtfilt_fir_100",
        sg.lfilter(fir, [1.0], sg.lfilter(fir, [1.0], x100)[::-1])[::-1],
        ff_note + ", b = [0.25] * 4, a = [1.0], sample(100)",
    )

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


common.main(
    {
        "crates/scikit/tests/fixtures/stats_reference.tsv": stats_rows,
        "crates/scikit/tests/fixtures/stats_grid_reference.tsv": stats_grid_rows,
        "crates/scikit/tests/fixtures/fft_signal_reference.tsv": fft_signal_rows,
    }
)
