# scikit parity with SciPy

The reference library is SciPy (`scipy.stats`, `scipy.fft`, `scipy.signal`, `scipy.interpolate`, `scipy.optimize`, `scipy.spatial`, `scipy.sparse`, `scipy.integrate`). The `ts` module has no SciPy module: its names come from `statsmodels` or `pandas`, and each row says so.
The states are `has`, `partial` and `missing` (Table 2). Each feature is gated by a Cargo feature of the same name as its module.
A SciPy name of `none` means a kit helper or a kit type with no SciPy name. Its state is `has` when it works as its doc comment says.

## Table 1: public functions and types

## fft

| Function | SciPy name | State | Evidence |
|---|---|---|---|
| fft::fft | scipy.fft.fft | has | crates/scikit/src/fft/transform.rs; fixed in issue 52 |
| fft::ifft | scipy.fft.ifft | has | crates/scikit/src/fft/transform.rs; fixed in issue 52 |
| fft::rfft | scipy.fft.rfft | has | crates/scikit/src/fft/transform.rs; fixed in issue 52 |
| fft::irfft | scipy.fft.irfft | has | crates/scikit/src/fft/transform.rs; fixed in issue 52 (odd n) |
| fft::fftfreq | scipy.fft.fftfreq | has | crates/scikit/src/fft/transform.rs |
| fft::rfftfreq | scipy.fft.rfftfreq | has | crates/scikit/src/fft/transform.rs |
| fft::Complex | none (numpy complex) | has | crates/scikit/src/fft/transform.rs |
| fft::Complex::new | none (numpy complex) | has | crates/scikit/src/fft/transform.rs |
| fft::Complex::zero | none (numpy complex) | has | crates/scikit/src/fft/transform.rs |
| fft::Complex::norm | numpy.abs | has | crates/scikit/src/fft/transform.rs |
| fft::Complex::conj | numpy.conj | has | crates/scikit/src/fft/transform.rs |

## integrate

| Function | SciPy name | State | Evidence |
|---|---|---|---|
| integrate::solve_ivp | scipy.integrate.solve_ivp | partial | crates/scikit/src/integrate/ode.rs; three solvers only; no t_eval or events; fixed in issue 57 (RK45 error estimate) |
| integrate::euler | none (no SciPy Euler) | has | crates/scikit/src/integrate/ode.rs |
| integrate::rk4 | none (no SciPy RK4) | has | crates/scikit/src/integrate/ode.rs |
| integrate::OdeResult | scipy.integrate.OdeResult | has | crates/scikit/src/integrate/ode.rs |
| integrate::OdeSolver | none (method argument) | has | crates/scikit/src/integrate/ode.rs |
| integrate::quad | scipy.integrate.quad | partial | crates/scikit/src/integrate/quad.rs; adaptive Simpson, not QUADPACK; no infinite limits |
| integrate::trapezoid | scipy.integrate.trapezoid | has | crates/scikit/src/integrate/quad.rs |
| integrate::cumulative_trapezoid | scipy.integrate.cumulative_trapezoid | has | crates/scikit/src/integrate/quad.rs |
| integrate::simps | scipy.integrate.simpson | partial | crates/scikit/src/integrate/quad.rs; an odd interval count uses the trapezoid rule for the last interval |
| integrate::fixed_quad | scipy.integrate.fixed_quad | partial | crates/scikit/src/integrate/quad.rs; Gauss-Legendre orders 1 to 5 only; fixed in issue 53 |
| integrate::QuadResult | none (tuple in SciPy) | has | crates/scikit/src/integrate/quad.rs |

## interpolate

| Function | SciPy name | State | Evidence |
|---|---|---|---|
| interpolate::interp1d | scipy.interpolate.interp1d | partial | crates/scikit/src/interpolate/interp.rs; x must be sorted; kinds as in InterpKind only |
| interpolate::InterpKind | none (kind argument) | has | crates/scikit/src/interpolate/interp.rs |
| interpolate::CubicSpline | scipy.interpolate.CubicSpline | partial | crates/scikit/src/interpolate/interp.rs; natural boundary only |
| interpolate::CubicSpline::new | scipy.interpolate.CubicSpline | partial | crates/scikit/src/interpolate/interp.rs; natural boundary only |
| interpolate::CubicSpline::eval | CubicSpline.__call__ | partial | crates/scikit/src/interpolate/interp.rs; natural boundary only |
| interpolate::interp2d | scipy.interpolate.RegularGridInterpolator | partial | crates/scikit/src/interpolate/interp2d.rs; bilinear only; a query outside the grid is an error |
| interpolate::Interp2dError | none | has | crates/scikit/src/interpolate/interp2d.rs |

## optimize

| Function | SciPy name | State | Evidence |
|---|---|---|---|
| optimize::curve_fit | scipy.optimize.curve_fit | partial | crates/scikit/src/optimize/curve_fit.rs; Gauss-Newton only; no bounds, p0 or sigma |
| optimize::CurveFitResult | none (tuple in SciPy) | has | crates/scikit/src/optimize/curve_fit.rs |
| optimize::OptimizeError | none | has | crates/scikit/src/optimize/error.rs |
| optimize::linprog | scipy.optimize.linprog | partial | crates/scikit/src/optimize/linprog.rs; x >= 0 and A_ub only; no equality constraints or bounds |
| optimize::LinprogResult | scipy.optimize.OptimizeResult | has | crates/scikit/src/optimize/linprog.rs |
| optimize::nelder_mead | scipy.optimize.minimize(method='Nelder-Mead') | has | crates/scikit/src/optimize/minimize.rs |
| optimize::bfgs | scipy.optimize.minimize(method='BFGS') | has | crates/scikit/src/optimize/minimize.rs |
| optimize::MinimizeResult | scipy.optimize.OptimizeResult | has | crates/scikit/src/optimize/minimize.rs |
| optimize::brentq | scipy.optimize.brentq | has | crates/scikit/src/optimize/root.rs |
| optimize::newton | scipy.optimize.newton | has | crates/scikit/src/optimize/root.rs |
| optimize::bisect | scipy.optimize.bisect | has | crates/scikit/src/optimize/root.rs |

## signal

| Function | SciPy name | State | Evidence |
|---|---|---|---|
| signal::convolve | scipy.signal.convolve | has | crates/scikit/src/signal/convolve.rs |
| signal::correlate | scipy.signal.correlate | has | crates/scikit/src/signal/convolve.rs |
| signal::ConvolveMode | none (mode argument) | has | crates/scikit/src/signal/convolve.rs |
| signal::lfilter | scipy.signal.lfilter | has | crates/scikit/src/signal/filter.rs |
| signal::lfilter_zi | scipy.signal.lfilter_zi | has | crates/scikit/src/signal/filter.rs |
| signal::lfilter_with_state | scipy.signal.lfilter (zi argument) | has | crates/scikit/src/signal/filter.rs |
| signal::sosfilt | scipy.signal.sosfilt | has | crates/scikit/src/signal/filter.rs |
| signal::butter | scipy.signal.butter | has | crates/scikit/src/signal/filter.rs; fixed in issue 56 (odd order) |
| signal::FilterType | none (ftype argument) | has | crates/scikit/src/signal/filter.rs |
| signal::SosSection | none (sos rows) | has | crates/scikit/src/signal/filter.rs |
| signal::filtfilt | scipy.signal.filtfilt | partial | crates/scikit/src/signal/filtfilt.rs; no padding argument |
| signal::find_peaks | scipy.signal.find_peaks | partial | crates/scikit/src/signal/peaks.rs; PeakOptions fields only |
| signal::Peak | none | has | crates/scikit/src/signal/peaks.rs |
| signal::PeakOptions | scipy.signal.find_peaks (keyword arguments) | has | crates/scikit/src/signal/peaks.rs |
| signal::stft | scipy.signal.stft | partial | crates/scikit/src/signal/stft.rs; Hann window only; panics on a bad size |
| signal::StftResult | none (tuple in SciPy) | has | crates/scikit/src/signal/stft.rs |
| signal::hann | scipy.signal.windows.hann | has | crates/scikit/src/signal/windows.rs |
| signal::hamming | scipy.signal.windows.hamming | has | crates/scikit/src/signal/windows.rs |
| signal::blackman | scipy.signal.windows.blackman | has | crates/scikit/src/signal/windows.rs |
| signal::kaiser | scipy.signal.windows.kaiser | has | crates/scikit/src/signal/windows.rs |

## sparse

| Function | SciPy name | State | Evidence |
|---|---|---|---|
| sparse::CscMatrix | scipy.sparse.csc_matrix | has | crates/scikit/src/sparse/csc.rs |
| sparse::CscMatrix::new | scipy.sparse.csc_matrix((data, indices, indptr)) | has | crates/scikit/src/sparse/csc.rs |
| sparse::CscMatrix::from_dense | scipy.sparse.csc_matrix(dense) | has | crates/scikit/src/sparse/csc.rs |
| sparse::CscMatrix::nnz | csc_matrix.nnz | has | crates/scikit/src/sparse/csc.rs |
| sparse::CscMatrix::get | csc_matrix[i, j] | has | crates/scikit/src/sparse/csc.rs |
| sparse::CscMatrix::to_dense | csc_matrix.toarray | has | crates/scikit/src/sparse/csc.rs |
| sparse::CscMatrix::dot | csc_matrix @ vector | partial | crates/scikit/src/sparse/csc.rs; matrix times vector only |
| sparse::CscMatrix::to_csr | csc_matrix.tocsr | has | crates/scikit/src/sparse/csc.rs |
| sparse::CscMatrix::add | csc_matrix + csc_matrix | has | crates/scikit/src/sparse/csc.rs |
| sparse::CscMatrix::scale | csc_matrix * scalar | has | crates/scikit/src/sparse/csc.rs |
| sparse::CscMatrix::density | none (nnz / size) | has | crates/scikit/src/sparse/csc.rs |
| sparse::CsrMatrix | scipy.sparse.csr_matrix | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::new | scipy.sparse.csr_matrix((data, indices, indptr)) | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::from_dense | scipy.sparse.csr_matrix(dense) | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::from_coo | scipy.sparse.coo_matrix(...).tocsr | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::nnz | csr_matrix.nnz | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::get | csr_matrix[i, j] | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::to_dense | csr_matrix.toarray | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::dot | csr_matrix @ vector | partial | crates/scikit/src/sparse/csr.rs; matrix times vector only |
| sparse::CsrMatrix::add | csr_matrix + csr_matrix | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::scale | csr_matrix * scalar | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::transpose | csr_matrix.T | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::to_csc | csr_matrix.tocsc | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::eye | scipy.sparse.eye | has | crates/scikit/src/sparse/csr.rs |
| sparse::CsrMatrix::density | none (nnz / size) | has | crates/scikit/src/sparse/csr.rs |

## spatial

| Function | SciPy name | State | Evidence |
|---|---|---|---|
| spatial::euclidean | scipy.spatial.distance.euclidean | has | crates/scikit/src/spatial/distance.rs |
| spatial::manhattan | scipy.spatial.distance.cityblock | has | crates/scikit/src/spatial/distance.rs |
| spatial::chebyshev | scipy.spatial.distance.chebyshev | has | crates/scikit/src/spatial/distance.rs |
| spatial::minkowski | scipy.spatial.distance.minkowski | has | crates/scikit/src/spatial/distance.rs |
| spatial::cosine_distance | scipy.spatial.distance.cosine | has | crates/scikit/src/spatial/distance.rs |
| spatial::pdist | scipy.spatial.distance.pdist | partial | crates/scikit/src/spatial/distance.rs; metrics euclidean, cityblock, chebyshev and cosine only |
| spatial::cdist | scipy.spatial.distance.cdist | partial | crates/scikit/src/spatial/distance.rs; metrics euclidean, cityblock, chebyshev and cosine only |
| spatial::squareform | scipy.spatial.distance.squareform | partial | crates/scikit/src/spatial/distance.rs; condensed to square only |
| spatial::convex_hull | scipy.spatial.ConvexHull | partial | crates/scikit/src/spatial/hull.rs; 2-D only; returns indices in counter-clockwise order; no simplices |
| spatial::convex_hull_area | scipy.spatial.ConvexHull.volume | has | crates/scikit/src/spatial/hull.rs |
| spatial::KdTree | scipy.spatial.KDTree | has | crates/scikit/src/spatial/kdtree.rs |
| spatial::KdTree::new | scipy.spatial.KDTree(data) | has | crates/scikit/src/spatial/kdtree.rs |
| spatial::KdTree::query | scipy.spatial.KDTree.query | partial | crates/scikit/src/spatial/kdtree.rs; returns neighbor structs, not arrays |
| spatial::KdTree::query_ball | scipy.spatial.KDTree.query_ball_point | partial | crates/scikit/src/spatial/kdtree.rs; returns neighbor structs, not index lists |
| spatial::KdTree::len | KDTree.n | has | crates/scikit/src/spatial/kdtree.rs |
| spatial::KdTree::is_empty | none (kit helper) | has | crates/scikit/src/spatial/kdtree.rs |
| spatial::KdTree::dim | KDTree.m | has | crates/scikit/src/spatial/kdtree.rs |
| spatial::Neighbor | none (kit type) | has | crates/scikit/src/spatial/kdtree.rs |

## stats

| Function | SciPy name | State | Evidence |
|---|---|---|---|
| stats::skew | scipy.stats.skew | partial | crates/scikit/src/stats/descriptive.rs; sample-size adjusted; SciPy default bias=True is not adjusted |
| stats::kurtosis | scipy.stats.kurtosis | partial | crates/scikit/src/stats/descriptive.rs; sample-size adjusted; SciPy default bias=True is not adjusted |
| stats::zscore | scipy.stats.zscore | partial | crates/scikit/src/stats/descriptive.rs; sample std with ddof 1; SciPy default ddof 0 |
| stats::moment | scipy.stats.moment | has | crates/scikit/src/stats/descriptive.rs |
| stats::raw_moment | none (kit helper) | has | crates/scikit/src/stats/descriptive.rs |
| stats::mode | scipy.stats.mode | partial | crates/scikit/src/stats/descriptive.rs; first mode met on a tie; SciPy returns the smallest |
| stats::median | numpy.median | has | crates/scikit/src/stats/descriptive.rs |
| stats::iqr | scipy.stats.iqr | has | crates/scikit/src/stats/descriptive.rs |
| stats::percentile | numpy.percentile | has | crates/scikit/src/stats/descriptive.rs |
| stats::geometric_mean | scipy.stats.gmean | has | crates/scikit/src/stats/descriptive.rs |
| stats::harmonic_mean | scipy.stats.hmean | has | crates/scikit/src/stats/descriptive.rs |
| stats::trim_mean | scipy.stats.trim_mean | has | crates/scikit/src/stats/descriptive.rs |
| stats::sem | scipy.stats.sem | has | crates/scikit/src/stats/descriptive.rs |
| stats::variation | scipy.stats.variation | partial | crates/scikit/src/stats/descriptive.rs; sample std with ddof 1; SciPy default ddof 0 |
| stats::median_abs_deviation | scipy.stats.median_abs_deviation | has | crates/scikit/src/stats/descriptive.rs |
| stats::describe | scipy.stats.describe | partial | crates/scikit/src/stats/descriptive.rs; returns a struct, not a tuple |
| stats::DescribeResult | none (tuple in SciPy) | has | crates/scikit/src/stats/descriptive.rs |
| stats::ContinuousDistribution | scipy.stats.rv_continuous | has | crates/scikit/src/stats/distributions.rs |
| stats::DiscreteDistribution | scipy.stats.rv_discrete | has | crates/scikit/src/stats/distributions.rs |
| stats::Normal | scipy.stats.norm | has | crates/scikit/src/stats/distributions.rs |
| stats::Normal::new | scipy.stats.norm(loc, scale) | has | crates/scikit/src/stats/distributions.rs |
| stats::Normal::standard | scipy.stats.norm() | has | crates/scikit/src/stats/distributions.rs |
| stats::Uniform | scipy.stats.uniform | has | crates/scikit/src/stats/distributions.rs |
| stats::Uniform::new | scipy.stats.uniform(loc, scale) | has | crates/scikit/src/stats/distributions.rs |
| stats::Exponential | scipy.stats.expon | has | crates/scikit/src/stats/distributions.rs |
| stats::Exponential::new | scipy.stats.expon(scale) | has | crates/scikit/src/stats/distributions.rs |
| stats::Gamma | scipy.stats.gamma | has | crates/scikit/src/stats/distributions.rs |
| stats::Gamma::new | scipy.stats.gamma(a, scale) | has | crates/scikit/src/stats/distributions.rs |
| stats::Beta | scipy.stats.beta | has | crates/scikit/src/stats/distributions.rs |
| stats::Beta::new | scipy.stats.beta(a, b) | has | crates/scikit/src/stats/distributions.rs |
| stats::StudentT | scipy.stats.t | has | crates/scikit/src/stats/distributions.rs |
| stats::StudentT::new | scipy.stats.t(df) | has | crates/scikit/src/stats/distributions.rs |
| stats::ChiSquared | scipy.stats.chi2 | has | crates/scikit/src/stats/distributions.rs |
| stats::ChiSquared::new | scipy.stats.chi2(df) | has | crates/scikit/src/stats/distributions.rs |
| stats::FDistribution | scipy.stats.f | has | crates/scikit/src/stats/distributions.rs |
| stats::FDistribution::new | scipy.stats.f(dfn, dfd) | has | crates/scikit/src/stats/distributions.rs |
| stats::Binomial | scipy.stats.binom | has | crates/scikit/src/stats/distributions.rs |
| stats::Binomial::new | scipy.stats.binom(n, p) | has | crates/scikit/src/stats/distributions.rs |
| stats::Poisson | scipy.stats.poisson | has | crates/scikit/src/stats/distributions.rs |
| stats::Poisson::new | scipy.stats.poisson(mu) | has | crates/scikit/src/stats/distributions.rs |
| stats::Geometric | scipy.stats.geom | has | crates/scikit/src/stats/distributions.rs |
| stats::Geometric::new | scipy.stats.geom(p) | has | crates/scikit/src/stats/distributions.rs |
| stats::erf | scipy.special.erf | has | crates/scikit/src/stats/distributions.rs; fixed in issue 54 |
| stats::gamma_fn | scipy.special.gamma | has | crates/scikit/src/stats/distributions.rs |
| stats::ln_gamma | scipy.special.gammaln | has | crates/scikit/src/stats/distributions.rs; fixed in issue 54 |
| stats::beta_fn | scipy.special.beta | has | crates/scikit/src/stats/distributions.rs |
| stats::lower_incomplete_gamma | scipy.special.gammainc | has | crates/scikit/src/stats/distributions.rs; fixed in issue 54 |
| stats::incomplete_beta | scipy.special.betainc | has | crates/scikit/src/stats/distributions.rs |
| stats::factorial | scipy.special.factorial | has | crates/scikit/src/stats/distributions.rs |
| stats::binomial_coef | scipy.special.comb | has | crates/scikit/src/stats/distributions.rs |
| stats::TestResult | none (tuple in SciPy) | has | crates/scikit/src/stats/hypothesis.rs |
| stats::TestResult::new | none (kit helper) | has | crates/scikit/src/stats/hypothesis.rs |
| stats::TestResult::is_significant | none (kit helper) | has | crates/scikit/src/stats/hypothesis.rs |
| stats::ttest_1samp | scipy.stats.ttest_1samp | has | crates/scikit/src/stats/hypothesis.rs |
| stats::ttest_ind | scipy.stats.ttest_ind | has | crates/scikit/src/stats/hypothesis.rs |
| stats::ttest_rel | scipy.stats.ttest_rel | has | crates/scikit/src/stats/hypothesis.rs |
| stats::f_oneway | scipy.stats.f_oneway | has | crates/scikit/src/stats/hypothesis.rs |
| stats::chi2_contingency | scipy.stats.chi2_contingency | partial | crates/scikit/src/stats/hypothesis.rs; no Yates correction; SciPy applies it to 2x2 tables |
| stats::chisquare | scipy.stats.chisquare | has | crates/scikit/src/stats/hypothesis.rs |
| stats::pearsonr | scipy.stats.pearsonr | has | crates/scikit/src/stats/hypothesis.rs |
| stats::spearmanr | scipy.stats.spearmanr | has | crates/scikit/src/stats/hypothesis.rs; fixed in issue 55 (tie-averaged ranks) |
| stats::kstest_normal | scipy.stats.kstest(x, 'norm') | partial | crates/scikit/src/stats/hypothesis.rs; estimates mean and std from the data; asymptotic p-value |
| stats::mannwhitneyu | scipy.stats.mannwhitneyu | partial | crates/scikit/src/stats/hypothesis.rs; asymptotic p-value only; fixed in issue 55 (tie-averaged ranks) |

## ts

| Function | SciPy name | State | Evidence |
|---|---|---|---|
| ts::acf::acf | statsmodels.tsa.stattools.acf | has | crates/scikit/src/ts/acf.rs |
| ts::acf::pacf | statsmodels.tsa.stattools.pacf | has | crates/scikit/src/ts/acf.rs |
| ts::acf::pacf_from_acf | none (kit helper) | has | crates/scikit/src/ts/acf.rs |
| ts::acf::ljung_box | statsmodels.stats.diagnostic.acorr_ljungbox | has | crates/scikit/src/ts/acf.rs |
| ts::acf::LjungBoxResult | none (kit type) | has | crates/scikit/src/ts/acf.rs |
| ts::acf::LjungBoxResult::is_significant | none (kit helper) | has | crates/scikit/src/ts/acf.rs |
| ts::arima::ArimaModel | statsmodels.tsa.arima.model.ARIMA | partial | crates/scikit/src/ts/arima.rs; AR by Yule-Walker; MA by a simplified CSS search |
| ts::arima::ArimaOrder | none (order tuple) | has | crates/scikit/src/ts/arima.rs |
| ts::arima::ArimaOrder::new | none (order tuple) | has | crates/scikit/src/ts/arima.rs |
| ts::arima::ArimaModel::fit | ARIMA.fit | partial | crates/scikit/src/ts/arima.rs; MA by a simplified CSS search, not maximum likelihood |
| ts::arima::ArimaModel::fitted_values | ARIMA.fittedvalues | partial | crates/scikit/src/ts/arima.rs; MA by a simplified CSS search |
| ts::arima::ArimaModel::forecast | ARIMA.forecast | partial | crates/scikit/src/ts/arima.rs; MA by a simplified CSS search; fixed in issue 57 (d=2 forecast) |
| ts::arima::ArimaModel::aic | ARIMA.aic | partial | crates/scikit/src/ts/arima.rs; depends on the simplified MA fit |
| ts::arima::difference | numpy.diff (n times) | has | crates/scikit/src/ts/arima.rs |
| ts::decompose::seasonal_decompose | statsmodels.tsa.seasonal.seasonal_decompose | has | crates/scikit/src/ts/decompose.rs |
| ts::decompose::DecomposeModel | statsmodels model argument | has | crates/scikit/src/ts/decompose.rs |
| ts::decompose::DecomposeResult | statsmodels DecomposeResult | has | crates/scikit/src/ts/decompose.rs |
| ts::decompose::DecomposeResult::trend_strength | none (kit helper) | has | crates/scikit/src/ts/decompose.rs |
| ts::decompose::DecomposeResult::seasonal_strength | none (kit helper) | has | crates/scikit/src/ts/decompose.rs |
| ts::error::TsError | none (kit type) | has | crates/scikit/src/ts/error.rs |
| ts::ops::ewma | pandas.Series.ewm(alpha=) | has | crates/scikit/src/ts/ops.rs |
| ts::ops::ewma_span | pandas.Series.ewm(span=) | has | crates/scikit/src/ts/ops.rs |
| ts::ops::ewma_halflife | pandas.Series.ewm(halflife=) | has | crates/scikit/src/ts/ops.rs |
| ts::ops::expanding_mean | pandas.Series.expanding().mean | has | crates/scikit/src/ts/ops.rs |
| ts::ops::expanding_sum | pandas.Series.expanding().sum | has | crates/scikit/src/ts/ops.rs |
| ts::ops::expanding_std | pandas.Series.expanding().std | partial | crates/scikit/src/ts/ops.rs; population std; pandas default ddof 1 |
| ts::ops::expanding_min | pandas.Series.expanding().min | has | crates/scikit/src/ts/ops.rs |
| ts::ops::expanding_max | pandas.Series.expanding().max | has | crates/scikit/src/ts/ops.rs |
| ts::ops::lag | pandas.Series.shift | has | crates/scikit/src/ts/ops.rs |
| ts::ops::lead | pandas.Series.shift(-n) | has | crates/scikit/src/ts/ops.rs |
| ts::ops::SeriesTimeSeriesExt | none (kit trait) | has | crates/scikit/src/ts/ops.rs |
| ts::resample::AggMethod | none (agg argument) | has | crates/scikit/src/ts/resample.rs |
| ts::resample::InterpMethod | none (method argument) | has | crates/scikit/src/ts/resample.rs |
| ts::resample::downsample | pandas.Series.resample().agg | has | crates/scikit/src/ts/resample.rs |
| ts::resample::upsample | pandas.Series.resample().interpolate | partial | crates/scikit/src/ts/resample.rs; linear, nearest, ffill and bfill only |
| ts::resample::resample_to_len | scipy.signal.resample | partial | crates/scikit/src/ts/resample.rs; linear interpolation; SciPy uses the FFT |
| ts::smoothing::ses | statsmodels.tsa.holtwinters.SimpleExpSmoothing | partial | crates/scikit/src/ts/smoothing.rs; alpha is given; no fitting |
| ts::smoothing::SesResult | none (kit type) | has | crates/scikit/src/ts/smoothing.rs |
| ts::smoothing::SesResult::forecast | SimpleExpSmoothing.forecast | has | crates/scikit/src/ts/smoothing.rs |
| ts::smoothing::holt | statsmodels.tsa.holtwinters.Holt | partial | crates/scikit/src/ts/smoothing.rs; alpha and beta are given; no fitting |
| ts::smoothing::HoltResult | none (kit type) | has | crates/scikit/src/ts/smoothing.rs |
| ts::smoothing::HoltResult::forecast | Holt.forecast | has | crates/scikit/src/ts/smoothing.rs |
| ts::smoothing::HwSeasonal | none (kit type) | has | crates/scikit/src/ts/smoothing.rs |
| ts::smoothing::HoltWintersParams | none (kit type) | has | crates/scikit/src/ts/smoothing.rs |
| ts::smoothing::holt_winters | statsmodels.tsa.holtwinters.ExponentialSmoothing | partial | crates/scikit/src/ts/smoothing.rs; parameters are given; no fitting |
| ts::smoothing::HoltWintersResult | none (kit type) | has | crates/scikit/src/ts/smoothing.rs |
| ts::smoothing::HoltWintersResult::forecast | ExponentialSmoothing.forecast | has | crates/scikit/src/ts/smoothing.rs |

## scipy.stats functions kit lacks

| SciPy function | State | Later issue title |
|---|---|---|
| kruskal | missing | scikit stats: add kruskal |
| friedmanchisquare | missing | scikit stats: add friedmanchisquare |
| fisher_exact | missing | scikit stats: add fisher_exact |
| ks_2samp | missing | scikit stats: add ks_2samp |
| anderson | missing | scikit stats: add anderson |
| levene | missing | scikit stats: add levene |
| bartlett | missing | scikit stats: add bartlett |
| kendalltau | missing | scikit stats: add kendalltau |
| linregress | missing | scikit stats: add linregress |
| gaussian_kde | missing | scikit stats: add gaussian_kde |
| entropy | missing | scikit stats: add entropy |
| boxcox | missing | scikit stats: add boxcox |
| bootstrap | missing | scikit stats: add bootstrap |
| permutation_test | missing | scikit stats: add permutation_test |

Tables are maintained by hand; change a row in the same pull request that changes the code.
