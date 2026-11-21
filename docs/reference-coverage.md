# Reference coverage

Numeric rows: 410
Not numeric rows: 198

Each row is one `pub fn` or `pub const fn` that `tools/reference/inventory.sh` lists. A numeric function needs a reference value from a Python library (numpy, scipy, pandas or statsmodels, pinned in `tools/reference/requirements.txt`), a tolerance and a test. Status `not covered` means that no reference test exists yet. `-` means not written yet. `tools/reference/check_table.sh` checks this file.

## Numeric functions

| Crate | Function | Reference expression | Tolerance | Test | Status |
|---|---|---|---|---|---|
| arraykit | arraykit::array/indexing::advanced_index | - | - | - | not covered |
| arraykit | arraykit::array/indexing::bool_index | - | - | - | not covered |
| arraykit | arraykit::array/indexing::bool_index_array | - | - | - | not covered |
| arraykit | arraykit::array/indexing::bool_set | - | - | - | not covered |
| arraykit | arraykit::array/indexing::fancy_index | - | - | - | not covered |
| arraykit | arraykit::array/indexing::fancy_set | - | - | - | not covered |
| arraykit | arraykit::array/indexing::take | - | - | - | not covered |
| arraykit | arraykit::array/linalg::det | - | - | - | not covered |
| arraykit | arraykit::array/linalg::dot | - | - | - | not covered |
| arraykit | arraykit::array/linalg::inner | - | - | - | not covered |
| arraykit | arraykit::array/linalg::inv | - | - | - | not covered |
| arraykit | arraykit::array/linalg::matmul | - | - | - | not covered |
| arraykit | arraykit::array/linalg::norm | - | - | - | not covered |
| arraykit | arraykit::array/linalg::outer | - | - | - | not covered |
| arraykit | arraykit::array/linalg::solve | - | - | - | not covered |
| arraykit | arraykit::array/linalg::trace | - | - | - | not covered |
| arraykit | arraykit::array/linalg_decomp::cholesky | - | - | - | not covered |
| arraykit | arraykit::array/linalg_decomp::eig | - | - | - | not covered |
| arraykit | arraykit::array/linalg_decomp::eigh | - | - | - | not covered |
| arraykit | arraykit::array/linalg_decomp::lstsq | - | - | - | not covered |
| arraykit | arraykit::array/linalg_decomp::qr | - | - | - | not covered |
| arraykit | arraykit::array/linalg_decomp::svd | - | - | - | not covered |
| arraykit | arraykit::array/manip::all | - | - | - | not covered |
| arraykit | arraykit::array/manip::allclose | - | - | - | not covered |
| arraykit | arraykit::array/manip::any | - | - | - | not covered |
| arraykit | arraykit::array/manip::argsort | - | - | - | not covered |
| arraykit | arraykit::array/manip::concatenate | - | - | - | not covered |
| arraykit | arraykit::array/manip::hstack | - | - | - | not covered |
| arraykit | arraykit::array/manip::repeat | - | - | - | not covered |
| arraykit | arraykit::array/manip::sort | - | - | - | not covered |
| arraykit | arraykit::array/manip::split | - | - | - | not covered |
| arraykit | arraykit::array/manip::stack | - | - | - | not covered |
| arraykit | arraykit::array/manip::tile | - | - | - | not covered |
| arraykit | arraykit::array/manip::unique | - | - | - | not covered |
| arraykit | arraykit::array/manip::vstack | - | - | - | not covered |
| arraykit | arraykit::array/math::abs | - | - | - | not covered |
| arraykit | arraykit::array/math::abs_int | - | - | - | not covered |
| arraykit | arraykit::array/math::abs_int | - | - | - | not covered |
| arraykit | arraykit::array/math::clip | - | - | - | not covered |
| arraykit | arraykit::array/math::cos | - | - | - | not covered |
| arraykit | arraykit::array/math::exp | - | - | - | not covered |
| arraykit | arraykit::array/math::linspace | - | - | - | not covered |
| arraykit | arraykit::array/math::log | - | - | - | not covered |
| arraykit | arraykit::array/math::pow | - | - | - | not covered |
| arraykit | arraykit::array/math::sin | - | - | - | not covered |
| arraykit | arraykit::array/math::sqrt | - | - | - | not covered |
| arraykit | arraykit::array/math::tan | - | - | - | not covered |
| arraykit | arraykit::array/math::where_cond | - | - | - | not covered |
| arraykit | arraykit::array/ndarray::arange | - | - | - | not covered |
| arraykit | arraykit::array/ndarray::broadcast_to | - | - | - | not covered |
| arraykit | arraykit::array/ndarray::diag | - | - | - | not covered |
| arraykit | arraykit::array/ndarray::flatten | - | - | - | not covered |
| arraykit | arraykit::array/ndarray::reshape | - | - | - | not covered |
| arraykit | arraykit::array/ndarray::slice | - | - | - | not covered |
| arraykit | arraykit::array/ndarray::transpose | - | - | - | not covered |
| arraykit | arraykit::array/ndarray::transpose_axes | - | - | - | not covered |
| arraykit | arraykit::array/ops::add | - | - | - | not covered |
| arraykit | arraykit::array/ops::add_scalar | - | - | - | not covered |
| arraykit | arraykit::array/ops::div | - | - | - | not covered |
| arraykit | arraykit::array/ops::mul | - | - | - | not covered |
| arraykit | arraykit::array/ops::mul_scalar | - | - | - | not covered |
| arraykit | arraykit::array/ops::sub | - | - | - | not covered |
| arraykit | arraykit::array/random::bernoulli | - | - | - | not covered |
| arraykit | arraykit::array/random::binomial | - | - | - | not covered |
| arraykit | arraykit::array/random::choice | - | - | - | not covered |
| arraykit | arraykit::array/random::next_f64 | - | - | - | not covered |
| arraykit | arraykit::array/random::next_u64 | - | - | - | not covered |
| arraykit | arraykit::array/random::normal | - | - | - | not covered |
| arraykit | arraykit::array/random::normal_standard | - | - | - | not covered |
| arraykit | arraykit::array/random::poisson | - | - | - | not covered |
| arraykit | arraykit::array/random::randint | - | - | - | not covered |
| arraykit | arraykit::array/random::shuffle | - | - | - | not covered |
| arraykit | arraykit::array/random::uniform | - | - | - | not covered |
| arraykit | arraykit::array/search::argsort_axis | - | - | - | not covered |
| arraykit | arraykit::array/search::nonzero | - | - | - | not covered |
| arraykit | arraykit::array/search::searchsorted | - | - | - | not covered |
| arraykit | arraykit::array/search::sort_axis | - | - | - | not covered |
| arraykit | arraykit::array/search::unique_full | - | - | - | not covered |
| arraykit | arraykit::array/search::where_select | - | - | - | not covered |
| arraykit | arraykit::array/stats::argmax | `numpy.argmax(x)` | 1e-9 rel | `argmax_matches_numpy` | covered |
| arraykit | arraykit::array/stats::argmin | `numpy.argmin(x)` | 1e-9 rel | `argmin_matches_numpy` | covered |
| arraykit | arraykit::array/stats::checked_sum | `numpy.sum(x) for int32 (NumPy widens to int64; checked_sum errors instead)` | 1e-9 rel | `checked_sum_matches_numpy` | covered |
| arraykit | arraykit::array/stats::coefficient_of_variation | `scipy.stats.variation(x, ddof=1)` | 1e-9 rel | `coefficient_of_variation_matches_scipy` | covered |
| arraykit | arraykit::array/stats::corrcoef | `numpy.corrcoef(x)` | 1e-9 rel | `corrcoef_matches_numpy` | covered |
| arraykit | arraykit::array/stats::cov | `numpy.cov(x)` | 1e-9 rel | `cov_matches_numpy` | covered |
| arraykit | arraykit::array/stats::cumprod | `numpy.cumprod(x)` | 1e-9 rel | `cumprod_matches_numpy` | covered |
| arraykit | arraykit::array/stats::cumsum | `numpy.cumsum(x)` | 1e-9 rel | `cumsum_matches_numpy` | covered |
| arraykit | arraykit::array/stats::geometric_mean | `scipy.stats.gmean(x)` | 1e-9 rel | `geometric_mean_matches_scipy` | covered |
| arraykit | arraykit::array/stats::harmonic_mean | `scipy.stats.hmean(x)` | 1e-9 rel | `harmonic_mean_matches_scipy` | covered |
| arraykit | arraykit::array/stats::histogram | `numpy.histogram(x, bins=5)` | 1e-9 rel | `histogram_matches_numpy` | covered |
| arraykit | arraykit::array/stats::iqr | `scipy.stats.iqr(x)` | 1e-9 rel | `iqr_matches_scipy` | covered |
| arraykit | arraykit::array/stats::kurtosis | `scipy.stats.kurtosis(x, bias=False)` | 1e-9 rel | `kurtosis_matches_scipy` | covered |
| arraykit | arraykit::array/stats::max | `numpy.max(x)` | 1e-9 rel | `max_matches_numpy` | covered |
| arraykit | arraykit::array/stats::max_axis | `numpy.max(x, axis)` | 1e-9 rel | `max_axis_matches_numpy` | covered |
| arraykit | arraykit::array/stats::mean | `numpy.mean(x)` | 1e-9 rel | `mean_matches_numpy` | covered |
| arraykit | arraykit::array/stats::mean_axis | `numpy.mean(x, axis)` | 1e-9 rel | `mean_axis_matches_numpy` | covered |
| arraykit | arraykit::array/stats::mean_f64 | `numpy.mean(x)` | 1e-9 rel | `mean_f64_matches_numpy` | covered |
| arraykit | arraykit::array/stats::median | `numpy.median(x)` | 1e-9 rel | `median_matches_numpy` | covered |
| arraykit | arraykit::array/stats::min | `numpy.min(x)` | 1e-9 rel | `min_matches_numpy` | covered |
| arraykit | arraykit::array/stats::min_axis | `numpy.min(x, axis)` | 1e-9 rel | `min_axis_matches_numpy` | covered |
| arraykit | arraykit::array/stats::mode | `scipy.stats.mode(x, keepdims=False).mode` | 1e-9 rel | `mode_matches_scipy` | covered |
| arraykit | arraykit::array/stats::moment | `scipy.stats.moment(x, moment=n)` | 1e-9 rel | `moment_matches_scipy` | covered |
| arraykit | arraykit::array/stats::percentile | `numpy.percentile(x, q)` | 1e-9 rel | `percentile_matches_numpy` | covered |
| arraykit | arraykit::array/stats::quantile | `numpy.quantile(x, q)` | 1e-9 rel | `quantile_matches_numpy` | covered |
| arraykit | arraykit::array/stats::sem | `scipy.stats.sem(x)` | 1e-9 rel | `sem_matches_scipy` | covered |
| arraykit | arraykit::array/stats::skew | `scipy.stats.skew(x, bias=False)` | 1e-9 rel | `skew_matches_scipy` | covered |
| arraykit | arraykit::array/stats::std | `numpy.std(x, ddof=1)` | 1e-9 rel | `std_matches_numpy` | covered |
| arraykit | arraykit::array/stats::std_axis | `numpy.std(x, axis, ddof=1)` | 1e-9 rel | `std_axis_matches_numpy` | covered |
| arraykit | arraykit::array/stats::std_pop | `numpy.std(x)` | 1e-9 rel | `std_pop_matches_numpy` | covered |
| arraykit | arraykit::array/stats::sum | `numpy.sum(x)` | 1e-9 rel | `sum_matches_numpy` | covered |
| arraykit | arraykit::array/stats::sum_axis | `numpy.sum(x, axis)` | 1e-9 rel | `sum_axis_matches_numpy` | covered |
| arraykit | arraykit::array/stats::trim_mean | `scipy.stats.trim_mean(x, 0.25)` | 1e-9 rel | `trim_mean_matches_scipy` | covered |
| arraykit | arraykit::array/stats::var | `numpy.var(x, ddof=1)` | 1e-9 rel | `var_matches_numpy` | covered |
| arraykit | arraykit::array/stats::var_axis | `numpy.var(x, axis, ddof=1)` | 1e-9 rel | `var_axis_matches_numpy` | covered |
| arraykit | arraykit::array/stats::var_pop | `numpy.var(x)` | 1e-9 rel | `var_pop_matches_numpy` | covered |
| arraykit | arraykit::array/stats::zscore | `scipy.stats.zscore(x, ddof=1)` | 1e-9 rel | `zscore_matches_scipy` | covered |
| arraykit | arraykit::array/window::count | - | - | - | not covered |
| arraykit | arraykit::array/window::diff | - | - | - | not covered |
| arraykit | arraykit::array/window::expand_dims | - | - | - | not covered |
| arraykit | arraykit::array/window::flip | - | - | - | not covered |
| arraykit | arraykit::array/window::gradient | - | - | - | not covered |
| arraykit | arraykit::array/window::interp | - | - | - | not covered |
| arraykit | arraykit::array/window::max | - | - | - | not covered |
| arraykit | arraykit::array/window::mean | - | - | - | not covered |
| arraykit | arraykit::array/window::min | - | - | - | not covered |
| arraykit | arraykit::array/window::pad | - | - | - | not covered |
| arraykit | arraykit::array/window::roll | - | - | - | not covered |
| arraykit | arraykit::array/window::rolling | - | - | - | not covered |
| arraykit | arraykit::array/window::rot90 | - | - | - | not covered |
| arraykit | arraykit::array/window::squeeze | - | - | - | not covered |
| arraykit | arraykit::array/window::std | - | - | - | not covered |
| arraykit | arraykit::array/window::sum | - | - | - | not covered |
| arraykit | arraykit::array/window::var | - | - | - | not covered |
| frame | frame::frame/dataframe::apply | - | - | - | not covered |
| frame | frame::frame/dataframe::applymap | - | - | - | not covered |
| frame | frame::frame/dataframe::assign | - | - | - | not covered |
| frame | frame::frame/dataframe::concat | - | - | - | not covered |
| frame | frame::frame/dataframe::corr | - | - | - | not covered |
| frame | frame::frame/dataframe::describe | - | - | - | not covered |
| frame | frame::frame/dataframe::drop | - | - | - | not covered |
| frame | frame::frame/dataframe::drop_duplicates | - | - | - | not covered |
| frame | frame::frame/dataframe::dropna | - | - | - | not covered |
| frame | frame::frame/dataframe::dropna_all | - | - | - | not covered |
| frame | frame::frame/dataframe::dropna_subset | - | - | - | not covered |
| frame | frame::frame/dataframe::duplicated | - | - | - | not covered |
| frame | frame::frame/dataframe::fillna | - | - | - | not covered |
| frame | frame::frame/dataframe::fillna_column | - | - | - | not covered |
| frame | frame::frame/dataframe::filter | - | - | - | not covered |
| frame | frame::frame/dataframe::head | - | - | - | not covered |
| frame | frame::frame/dataframe::insert | - | - | - | not covered |
| frame | frame::frame/dataframe::isna | - | - | - | not covered |
| frame | frame::frame/dataframe::isna_all | - | - | - | not covered |
| frame | frame::frame/dataframe::isna_any | - | - | - | not covered |
| frame | frame::frame/dataframe::melt | - | - | - | not covered |
| frame | frame::frame/dataframe::pivot | - | - | - | not covered |
| frame | frame::frame/dataframe::query | - | - | - | not covered |
| frame | frame::frame/dataframe::rename | - | - | - | not covered |
| frame | frame::frame/dataframe::reset_index | - | - | - | not covered |
| frame | frame::frame/dataframe::sample | - | - | - | not covered |
| frame | frame::frame/dataframe::sample_seeded | - | - | - | not covered |
| frame | frame::frame/dataframe::select | - | - | - | not covered |
| frame | frame::frame/dataframe::set_index | - | - | - | not covered |
| frame | frame::frame/dataframe::sort_values | - | - | - | not covered |
| frame | frame::frame/dataframe::sort_values_by | - | - | - | not covered |
| frame | frame::frame/dataframe::tail | - | - | - | not covered |
| frame | frame::frame/index::range | - | - | - | not covered |
| frame | frame::frame/index::reset | - | - | - | not covered |
| frame | frame::frame/index::slice | - | - | - | not covered |
| frame | frame::frame/multi_index::argsort | - | - | - | not covered |
| frame | frame::frame/multi_index::droplevel | - | - | - | not covered |
| frame | frame::frame/multi_index::reorder_levels | - | - | - | not covered |
| frame | frame::frame/multi_index::select_level | - | - | - | not covered |
| frame | frame::frame/multi_index::slice | - | - | - | not covered |
| frame | frame::frame/multi_index::swaplevel | - | - | - | not covered |
| frame | frame::frame/multi_index::unique | - | - | - | not covered |
| frame | frame::frame/ops/groupby::agg | - | - | - | not covered |
| frame | frame::frame/ops/groupby::aggregate | - | - | - | not covered |
| frame | frame::frame/ops/groupby::count | - | - | - | not covered |
| frame | frame::frame/ops/groupby::filter_groups | - | - | - | not covered |
| frame | frame::frame/ops/groupby::first | - | - | - | not covered |
| frame | frame::frame/ops/groupby::last | - | - | - | not covered |
| frame | frame::frame/ops/groupby::max | - | - | - | not covered |
| frame | frame::frame/ops/groupby::mean | - | - | - | not covered |
| frame | frame::frame/ops/groupby::median | - | - | - | not covered |
| frame | frame::frame/ops/groupby::min | - | - | - | not covered |
| frame | frame::frame/ops/groupby::std | - | - | - | not covered |
| frame | frame::frame/ops/groupby::sum | - | - | - | not covered |
| frame | frame::frame/ops/groupby::transform | - | - | - | not covered |
| frame | frame::frame/ops/groupby::var | - | - | - | not covered |
| frame | frame::frame/ops/join::join | - | - | - | not covered |
| frame | frame::frame/ops/join::join | - | - | - | not covered |
| frame | frame::frame/ops/join::left_join | - | - | - | not covered |
| frame | frame::frame/ops/join::merge | - | - | - | not covered |
| frame | frame::frame/ops/join::outer_join | - | - | - | not covered |
| frame | frame::frame/ops/join::right_join | - | - | - | not covered |
| frame | frame::frame/ops/reshape::cross_tab | - | - | - | not covered |
| frame | frame::frame/ops/reshape::cross_tab | - | - | - | not covered |
| frame | frame::frame/ops/reshape::pivot_table | - | - | - | not covered |
| frame | frame::frame/ops/reshape::pivot_table | - | - | - | not covered |
| frame | frame::frame/ops/reshape::stack | - | - | - | not covered |
| frame | frame::frame/ops/reshape::stack | - | - | - | not covered |
| frame | frame::frame/ops/reshape::unstack | - | - | - | not covered |
| frame | frame::frame/ops/reshape::unstack | - | - | - | not covered |
| frame | frame::frame/ops/reshape::wide_to_long | - | - | - | not covered |
| frame | frame::frame/ops/reshape::wide_to_long | - | - | - | not covered |
| frame | frame::frame/ops/window::adjust | - | - | - | not covered |
| frame | frame::frame/ops/window::ewm_alpha | - | - | - | not covered |
| frame | frame::frame/ops/window::ewm_halflife | - | - | - | not covered |
| frame | frame::frame/ops/window::ewm_span | - | - | - | not covered |
| frame | frame::frame/ops/window::expanding | - | - | - | not covered |
| frame | frame::frame/ops/window::max | - | - | - | not covered |
| frame | frame::frame/ops/window::mean | - | - | - | not covered |
| frame | frame::frame/ops/window::mean | - | - | - | not covered |
| frame | frame::frame/ops/window::min | - | - | - | not covered |
| frame | frame::frame/ops/window::std | - | - | - | not covered |
| frame | frame::frame/ops/window::std | - | - | - | not covered |
| frame | frame::frame/ops/window::sum | - | - | - | not covered |
| frame | frame::frame/ops/window::var | - | - | - | not covered |
| frame | frame::frame/ops/window::var | - | - | - | not covered |
| frame | frame::frame/series::abs | s.abs().sum() | 1e-9 rel | abs_matches_pandas | covered |
| frame | frame::frame/series::between | s.between(0, 3).sum() | 1e-9 rel | between_matches_pandas | covered |
| frame | frame::frame/series::clip | s.clip(0, 3).sum() | 1e-9 rel | clip_matches_pandas | covered |
| frame | frame::frame/series::corr | x.corr(y) | 1e-9 rel | corr_matches_pandas | covered |
| frame | frame::frame/series::count | s.count() | 1e-9 rel | count_matches_pandas | covered |
| frame | frame::frame/series::drop_duplicates | len(s.drop_duplicates()) | 1e-9 rel | drop_duplicates_matches_pandas | covered |
| frame | frame::frame/series::duplicated | s.duplicated().sum() | 1e-9 rel | duplicated_matches_pandas | covered |
| frame | frame::frame/series::filter | s[mask].sum() | 1e-9 rel | filter_matches_pandas | covered |
| frame | frame::frame/series::isin | s.isin([4, 0]).sum() | 1e-9 rel | isin_matches_pandas | covered |
| frame | frame::frame/series::map | (s * 2 + 1).sum() | 1e-9 rel | map_matches_pandas | covered |
| frame | frame::frame/series::mask | s.mask(cond, 0).sum() | 1e-9 rel | mask_matches_pandas | covered |
| frame | frame::frame/series::max | s.max() | 1e-9 rel | max_matches_pandas | covered |
| frame | frame::frame/series::mean | s.mean() | 1e-9 rel | mean_matches_pandas | covered |
| frame | frame::frame/series::median | s.median() | 1e-9 rel | median_matches_pandas | covered |
| frame | frame::frame/series::min | s.min() | 1e-9 rel | min_matches_pandas | covered |
| frame | frame::frame/series::nlargest | s.nlargest(3).sum() | 1e-9 rel | nlargest_matches_pandas | covered |
| frame | frame::frame/series::nsmallest | s.nsmallest(3).sum() | 1e-9 rel | nsmallest_matches_pandas | covered |
| frame | frame::frame/series::quantile | s.quantile(0.25) | 1e-9 rel | quantile_matches_pandas | covered |
| frame | frame::frame/series::rank | s.rank().sum() | 1e-9 rel | rank_matches_pandas | covered |
| frame | frame::frame/series::replace | s.replace(4, 40).sum() | 1e-9 rel | replace_matches_pandas | covered |
| frame | frame::frame/series::replace_map | s.replace({4: 40, -1: -10}).sum() | 1e-9 rel | replace_map_matches_pandas | covered |
| frame | frame::frame/series::reset_index | s.reset_index(drop=True).loc[7] | 1e-9 rel | reset_index_matches_pandas | covered |
| frame | frame::frame/series::round | s.round(2).sum() | 1e-9 rel | round_matches_pandas | covered |
| frame | frame::frame/series::set_name | s.rename('renamed').sum() | 1e-9 rel | set_name_matches_pandas | covered |
| frame | frame::frame/series::sort | s.sort_values().iloc[0] | 1e-9 rel | sort_matches_pandas | covered |
| frame | frame::frame/series::std | s.std() | 1e-9 rel | std_matches_pandas | covered |
| frame | frame::frame/series::sum | s.sum() | 1e-9 rel | sum_matches_pandas | covered |
| frame | frame::frame/series::unique | len(s.unique()) | 1e-9 rel | unique_matches_pandas | covered |
| frame | frame::frame/series::value_counts | keys.value_counts()[0] | 1e-9 rel | value_counts_matches_pandas | covered |
| frame | frame::frame/series::var | s.var() | 1e-9 rel | var_matches_pandas | covered |
| frame | frame::frame/series::where_cond | s.where(cond, 0).sum() | 1e-9 rel | where_cond_matches_pandas | covered |
| frame | frame::frame/series/null_ops::apply | - | - | - | not covered |
| frame | frame::frame/series/null_ops::bfill | - | - | - | not covered |
| frame | frame::frame/series/null_ops::dropna | - | - | - | not covered |
| frame | frame::frame/series/null_ops::ffill | - | - | - | not covered |
| frame | frame::frame/series/null_ops::fillna | - | - | - | not covered |
| frame | frame::frame/series/null_ops::interpolate | - | - | - | not covered |
| frame | frame::frame/series/null_ops::interpolate_method | - | - | - | not covered |
| frame | frame::frame/series/null_ops::isna | - | - | - | not covered |
| frame | frame::frame/series/null_ops::notna | - | - | - | not covered |
| frame | frame::frame/series/null_ops::null_count | - | - | - | not covered |
| frame | frame::frame/series/window_ops::cummax | - | - | - | not covered |
| frame | frame::frame/series/window_ops::cummin | - | - | - | not covered |
| frame | frame::frame/series/window_ops::cumprod | - | - | - | not covered |
| frame | frame::frame/series/window_ops::cumsum | - | - | - | not covered |
| frame | frame::frame/series/window_ops::diff | - | - | - | not covered |
| frame | frame::frame/series/window_ops::max | - | - | - | not covered |
| frame | frame::frame/series/window_ops::mean | - | - | - | not covered |
| frame | frame::frame/series/window_ops::min | - | - | - | not covered |
| frame | frame::frame/series/window_ops::pct_change | - | - | - | not covered |
| frame | frame::frame/series/window_ops::rolling | - | - | - | not covered |
| frame | frame::frame/series/window_ops::shift | - | - | - | not covered |
| frame | frame::frame/series/window_ops::std | - | - | - | not covered |
| frame | frame::frame/series/window_ops::sum | - | - | - | not covered |
| frame | frame::frame/series/window_ops::var | - | - | - | not covered |
| scikit | scikit::fft/transform::conj | - | - | - | not covered |
| scikit | scikit::fft/transform::fft | - | - | - | not covered |
| scikit | scikit::fft/transform::fftfreq | - | - | - | not covered |
| scikit | scikit::fft/transform::ifft | - | - | - | not covered |
| scikit | scikit::fft/transform::irfft | - | - | - | not covered |
| scikit | scikit::fft/transform::norm | - | - | - | not covered |
| scikit | scikit::fft/transform::rfft | - | - | - | not covered |
| scikit | scikit::fft/transform::rfftfreq | - | - | - | not covered |
| scikit | scikit::integrate/ode::euler | - | - | - | not covered |
| scikit | scikit::integrate/ode::rk4 | - | - | - | not covered |
| scikit | scikit::integrate/ode::solve_ivp | - | - | - | not covered |
| scikit | scikit::integrate/quad::cumulative_trapezoid | - | - | - | not covered |
| scikit | scikit::integrate/quad::fixed_quad | - | - | - | not covered |
| scikit | scikit::integrate/quad::quad | - | - | - | not covered |
| scikit | scikit::integrate/quad::simps | - | - | - | not covered |
| scikit | scikit::integrate/quad::trapezoid | - | - | - | not covered |
| scikit | scikit::interpolate/interp::eval | - | - | - | not covered |
| scikit | scikit::interpolate/interp::interp1d | - | - | - | not covered |
| scikit | scikit::interpolate/interp2d::interp2d | - | - | - | not covered |
| scikit | scikit::linear_model::alpha | - | - | - | not covered |
| scikit | scikit::linear_model::coef | - | - | - | not covered |
| scikit | scikit::linear_model::coef | - | - | - | not covered |
| scikit | scikit::linear_model::fit | - | - | - | not covered |
| scikit | scikit::linear_model::fit | - | - | - | not covered |
| scikit | scikit::linear_model::intercept | - | - | - | not covered |
| scikit | scikit::linear_model::intercept | - | - | - | not covered |
| scikit | scikit::linear_model::predict | - | - | - | not covered |
| scikit | scikit::linear_model::predict | - | - | - | not covered |
| scikit | scikit::optimize/curve_fit::curve_fit | - | - | - | not covered |
| scikit | scikit::optimize/linprog::linprog | - | - | - | not covered |
| scikit | scikit::optimize/minimize::bfgs | - | - | - | not covered |
| scikit | scikit::optimize/minimize::nelder_mead | - | - | - | not covered |
| scikit | scikit::optimize/root::bisect | - | - | - | not covered |
| scikit | scikit::optimize/root::brentq | - | - | - | not covered |
| scikit | scikit::optimize/root::newton | - | - | - | not covered |
| scikit | scikit::signal/convolve::convolve | - | - | - | not covered |
| scikit | scikit::signal/convolve::correlate | - | - | - | not covered |
| scikit | scikit::signal/filter::butter | - | - | - | not covered |
| scikit | scikit::signal/filter::lfilter | - | - | - | not covered |
| scikit | scikit::signal/filter::sosfilt | - | - | - | not covered |
| scikit | scikit::signal/filtfilt::filtfilt | - | - | - | not covered |
| scikit | scikit::signal/peaks::find_peaks | - | - | - | not covered |
| scikit | scikit::signal/stft::stft | - | - | - | not covered |
| scikit | scikit::signal/windows::blackman | - | - | - | not covered |
| scikit | scikit::signal/windows::hamming | - | - | - | not covered |
| scikit | scikit::signal/windows::hann | - | - | - | not covered |
| scikit | scikit::signal/windows::kaiser | - | - | - | not covered |
| scikit | scikit::sparse/csc::add | - | - | - | not covered |
| scikit | scikit::sparse/csc::density | - | - | - | not covered |
| scikit | scikit::sparse/csc::dot | - | - | - | not covered |
| scikit | scikit::sparse/csc::scale | - | - | - | not covered |
| scikit | scikit::sparse/csr::add | - | - | - | not covered |
| scikit | scikit::sparse/csr::density | - | - | - | not covered |
| scikit | scikit::sparse/csr::dot | - | - | - | not covered |
| scikit | scikit::sparse/csr::scale | - | - | - | not covered |
| scikit | scikit::sparse/csr::transpose | - | - | - | not covered |
| scikit | scikit::spatial/distance::cdist | - | - | - | not covered |
| scikit | scikit::spatial/distance::chebyshev | - | - | - | not covered |
| scikit | scikit::spatial/distance::cosine_distance | - | - | - | not covered |
| scikit | scikit::spatial/distance::euclidean | - | - | - | not covered |
| scikit | scikit::spatial/distance::manhattan | - | - | - | not covered |
| scikit | scikit::spatial/distance::minkowski | - | - | - | not covered |
| scikit | scikit::spatial/distance::pdist | - | - | - | not covered |
| scikit | scikit::spatial/distance::squareform | - | - | - | not covered |
| scikit | scikit::spatial/hull::convex_hull | - | - | - | not covered |
| scikit | scikit::spatial/hull::convex_hull_area | - | - | - | not covered |
| scikit | scikit::spatial/kdtree::query | - | - | - | not covered |
| scikit | scikit::spatial/kdtree::query_ball | - | - | - | not covered |
| scikit | scikit::stats/descriptive::describe | - | - | - | not covered |
| scikit | scikit::stats/descriptive::geometric_mean | - | - | - | not covered |
| scikit | scikit::stats/descriptive::harmonic_mean | - | - | - | not covered |
| scikit | scikit::stats/descriptive::iqr | - | - | - | not covered |
| scikit | scikit::stats/descriptive::kurtosis | - | - | - | not covered |
| scikit | scikit::stats/descriptive::median | - | - | - | not covered |
| scikit | scikit::stats/descriptive::median_abs_deviation | - | - | - | not covered |
| scikit | scikit::stats/descriptive::mode | - | - | - | not covered |
| scikit | scikit::stats/descriptive::moment | - | - | - | not covered |
| scikit | scikit::stats/descriptive::percentile | - | - | - | not covered |
| scikit | scikit::stats/descriptive::raw_moment | - | - | - | not covered |
| scikit | scikit::stats/descriptive::sem | - | - | - | not covered |
| scikit | scikit::stats/descriptive::skew | - | - | - | not covered |
| scikit | scikit::stats/descriptive::trim_mean | - | - | - | not covered |
| scikit | scikit::stats/descriptive::variation | - | - | - | not covered |
| scikit | scikit::stats/descriptive::zscore | - | - | - | not covered |
| scikit | scikit::stats/distributions::beta_fn | - | - | - | not covered |
| scikit | scikit::stats/distributions::binomial_coef | - | - | - | not covered |
| scikit | scikit::stats/distributions::erf | - | - | - | not covered |
| scikit | scikit::stats/distributions::factorial | - | - | - | not covered |
| scikit | scikit::stats/distributions::gamma_fn | - | - | - | not covered |
| scikit | scikit::stats/distributions::incomplete_beta | - | - | - | not covered |
| scikit | scikit::stats/distributions::ln_gamma | - | - | - | not covered |
| scikit | scikit::stats/distributions::lower_incomplete_gamma | - | - | - | not covered |
| scikit | scikit::stats/distributions::standard | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::chi2_contingency | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::chisquare | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::f_oneway | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::is_significant | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::kstest_normal | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::mannwhitneyu | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::pearsonr | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::spearmanr | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::ttest_1samp | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::ttest_ind | - | - | - | not covered |
| scikit | scikit::stats/hypothesis::ttest_rel | - | - | - | not covered |
| scikit | scikit::ts/acf::acf | - | - | - | not covered |
| scikit | scikit::ts/acf::is_significant | - | - | - | not covered |
| scikit | scikit::ts/acf::ljung_box | - | - | - | not covered |
| scikit | scikit::ts/acf::pacf | - | - | - | not covered |
| scikit | scikit::ts/acf::pacf_from_acf | - | - | - | not covered |
| scikit | scikit::ts/arima::aic | - | - | - | not covered |
| scikit | scikit::ts/arima::difference | - | - | - | not covered |
| scikit | scikit::ts/arima::fit | - | - | - | not covered |
| scikit | scikit::ts/arima::fitted_values | - | - | - | not covered |
| scikit | scikit::ts/arima::forecast | - | - | - | not covered |
| scikit | scikit::ts/decompose::seasonal_decompose | - | - | - | not covered |
| scikit | scikit::ts/decompose::seasonal_strength | - | - | - | not covered |
| scikit | scikit::ts/decompose::trend_strength | - | - | - | not covered |
| scikit | scikit::ts/ops::ewma | - | - | - | not covered |
| scikit | scikit::ts/ops::ewma_halflife | - | - | - | not covered |
| scikit | scikit::ts/ops::ewma_span | - | - | - | not covered |
| scikit | scikit::ts/ops::expanding_max | - | - | - | not covered |
| scikit | scikit::ts/ops::expanding_mean | - | - | - | not covered |
| scikit | scikit::ts/ops::expanding_min | - | - | - | not covered |
| scikit | scikit::ts/ops::expanding_std | - | - | - | not covered |
| scikit | scikit::ts/ops::expanding_sum | - | - | - | not covered |
| scikit | scikit::ts/ops::lag | - | - | - | not covered |
| scikit | scikit::ts/ops::lead | - | - | - | not covered |
| scikit | scikit::ts/resample::downsample | - | - | - | not covered |
| scikit | scikit::ts/resample::resample_to_len | - | - | - | not covered |
| scikit | scikit::ts/resample::upsample | - | - | - | not covered |
| scikit | scikit::ts/smoothing::forecast | - | - | - | not covered |
| scikit | scikit::ts/smoothing::forecast | - | - | - | not covered |
| scikit | scikit::ts/smoothing::forecast | - | - | - | not covered |
| scikit | scikit::ts/smoothing::holt | - | - | - | not covered |
| scikit | scikit::ts/smoothing::holt_winters | - | - | - | not covered |
| scikit | scikit::ts/smoothing::ses | - | - | - | not covered |

## Not numeric

| Crate | Function | Reason |
|---|---|---|
| arraykit | arraykit::array/dtype::numpy_str | formatting |
| arraykit | arraykit::array/dtype::size | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::data | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::data_mut | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::dims | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::eye | constructor or conversion |
| arraykit | arraykit::array/ndarray::full | constructor or conversion |
| arraykit | arraykit::array/ndarray::get | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::identity | constructor or conversion |
| arraykit | arraykit::array/ndarray::ndim | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::new | constructor or conversion |
| arraykit | arraykit::array/ndarray::ones | constructor or conversion |
| arraykit | arraykit::array/ndarray::set | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::shape | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::size | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::view | accessor, returns stored data or metadata |
| arraykit | arraykit::array/ndarray::zeros | constructor or conversion |
| arraykit | arraykit::array/random::new | constructor or conversion |
| arraykit | arraykit::array/random::with_stream | constructor or conversion |
| arraykit | arraykit::array/shape::broadcast | shape, slice or view metadata |
| arraykit | arraykit::array/shape::broadcast_strides | shape, slice or view metadata |
| arraykit | arraykit::array/shape::contains | accessor, returns stored data or metadata |
| arraykit | arraykit::array/shape::dims | accessor, returns stored data or metadata |
| arraykit | arraykit::array/shape::ndim | accessor, returns stored data or metadata |
| arraykit | arraykit::array/shape::new | constructor or conversion |
| arraykit | arraykit::array/shape::offset | accessor, returns stored data or metadata |
| arraykit | arraykit::array/shape::size | accessor, returns stored data or metadata |
| arraykit | arraykit::array/shape::strides | accessor, returns stored data or metadata |
| arraykit | arraykit::array/shape::with_strides | constructor or conversion |
| arraykit | arraykit::array/slice::new | constructor or conversion |
| arraykit | arraykit::array/slice::normalize | shape, slice or view metadata |
| arraykit | arraykit::array/slice::normalize_all | shape, slice or view metadata |
| arraykit | arraykit::array/slice::output_shape | shape, slice or view metadata |
| arraykit | arraykit::array/slice::range | shape, slice or view metadata |
| arraykit | arraykit::array/slice::range_step | shape, slice or view metadata |
| arraykit | arraykit::array/slice::slices | accessor, returns stored data or metadata |
| arraykit | arraykit::array/view::data_ptr | accessor, returns stored data or metadata |
| arraykit | arraykit::array/view::dims | accessor, returns stored data or metadata |
| arraykit | arraykit::array/view::get | accessor, returns stored data or metadata |
| arraykit | arraykit::array/view::is_contiguous | accessor, returns stored data or metadata |
| arraykit | arraykit::array/view::ndim | accessor, returns stored data or metadata |
| arraykit | arraykit::array/view::reshape | shape, slice or view metadata |
| arraykit | arraykit::array/view::size | accessor, returns stored data or metadata |
| arraykit | arraykit::array/view::slice | shape, slice or view metadata |
| arraykit | arraykit::array/view::strides | accessor, returns stored data or metadata |
| arraykit | arraykit::array/view::to_owned | conversion between types or containers |
| arraykit | arraykit::array/view::transpose | shape, slice or view metadata |
| arraykit | arraykit::array/view::transpose_axes | shape, slice or view metadata |
| arraykit | arraykit::array/window::min_periods | accessor, returns stored data or metadata |
| arraykit | arraykit::array/window::new | constructor or conversion |
| frame | frame::frame/column::from_values | constructor or conversion |
| frame | frame::frame/column::get | accessor, returns stored data or metadata |
| frame | frame::frame/column::is_empty | accessor, returns stored data or metadata |
| frame | frame::frame/column::is_null | accessor, returns stored data or metadata |
| frame | frame::frame/column::len | accessor, returns stored data or metadata |
| frame | frame::frame/column::to_values | conversion between types or containers |
| frame | frame::frame/dataframe::col | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::columns | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::from_columns | constructor or conversion |
| frame | frame::frame/dataframe::from_map | constructor or conversion |
| frame | frame::frame/dataframe::from_records | constructor or conversion |
| frame | frame::frame/dataframe::get | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::get_columns | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::iloc | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::iloc_row | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::index | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::is_empty | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::iterrows | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::loc | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::loc_row | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::ncols | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::new | constructor or conversion |
| frame | frame::frame/dataframe::nrows | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::shape | accessor, returns stored data or metadata |
| frame | frame::frame/dataframe::to_dict | conversion between types or containers |
| frame | frame::frame/dataframe::to_records | conversion between types or containers |
| frame | frame::frame/index::get_label | accessor, returns stored data or metadata |
| frame | frame::frame/index::get_loc | accessor, returns stored data or metadata |
| frame | frame::frame/index::int | constructor or builder |
| frame | frame::frame/index::is_empty | accessor, returns stored data or metadata |
| frame | frame::frame/index::len | accessor, returns stored data or metadata |
| frame | frame::frame/index::string | constructor or builder |
| frame | frame::frame/index::to_map | conversion between types or containers |
| frame | frame::frame/index::to_values | conversion between types or containers |
| frame | frame::frame/io/binary::from_columnar_bytes | I/O: reads or writes a file or a string |
| frame | frame::frame/io/binary::load_columnar | I/O: reads or writes a file or a string |
| frame | frame::frame/io/binary::read_columnar | I/O: reads or writes a file or a string |
| frame | frame::frame/io/binary::read_columnar | I/O: reads or writes a file or a string |
| frame | frame::frame/io/binary::save_columnar | I/O: reads or writes a file or a string |
| frame | frame::frame/io/binary::to_columnar | I/O: reads or writes a file or a string |
| frame | frame::frame/io/binary::to_columnar_bytes | I/O: reads or writes a file or a string |
| frame | frame::frame/io/binary::write_columnar | I/O: reads or writes a file or a string |
| frame | frame::frame/io/csv::read_csv | I/O: reads or writes a file or a string |
| frame | frame::frame/io/csv::read_csv | I/O: reads or writes a file or a string |
| frame | frame::frame/io/csv::read_csv_with_options | I/O: reads or writes a file or a string |
| frame | frame::frame/io/csv::to_csv | I/O: reads or writes a file or a string |
| frame | frame::frame/io/csv::write_csv | I/O: reads or writes a file or a string |
| frame | frame::frame/io/csv::write_csv_with_options | I/O: reads or writes a file or a string |
| frame | frame::frame/io/json::from_json | I/O: reads or writes a file or a string |
| frame | frame::frame/io/json::read_json | I/O: reads or writes a file or a string |
| frame | frame::frame/io/json::read_json | I/O: reads or writes a file or a string |
| frame | frame::frame/io/json::read_json_str | I/O: reads or writes a file or a string |
| frame | frame::frame/io/json::save_json | I/O: reads or writes a file or a string |
| frame | frame::frame/io/json::save_json | I/O: reads or writes a file or a string |
| frame | frame::frame/io/json::to_json | I/O: reads or writes a file or a string |
| frame | frame::frame/io/json::write_json_str | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::add_sheet | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::from_string | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::get_sheet | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::iter | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::load | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::load_workbook | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::new | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::nsheets | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::save | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::save_workbook | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::sheet_names | I/O: reads or writes a file or a string |
| frame | frame::frame/io/workbook::to_string | I/O: reads or writes a file or a string |
| frame | frame::frame/multi_index::get_level_index | accessor, returns stored data or metadata |
| frame | frame::frame/multi_index::get_level_values | accessor, returns stored data or metadata |
| frame | frame::frame/multi_index::get_level_values_by_name | accessor, returns stored data or metadata |
| frame | frame::frame/multi_index::get_locs | accessor, returns stored data or metadata |
| frame | frame::frame/multi_index::get_tuple | accessor, returns stored data or metadata |
| frame | frame::frame/multi_index::is_empty | accessor, returns stored data or metadata |
| frame | frame::frame/multi_index::len | accessor, returns stored data or metadata |
| frame | frame::frame/multi_index::names | accessor, returns stored data or metadata |
| frame | frame::frame/multi_index::new | constructor or conversion |
| frame | frame::frame/multi_index::nlevels | accessor, returns stored data or metadata |
| frame | frame::frame/multi_index::to_group_map | conversion between types or containers |
| frame | frame::frame/ops/groupby::groupby | constructor or builder |
| frame | frame::frame/ops/groupby::groups | accessor, returns stored data or metadata |
| frame | frame::frame/ops/groupby::new | constructor or conversion |
| frame | frame::frame/ops/groupby::ngroups | accessor, returns stored data or metadata |
| frame | frame::frame/ops/window::min_periods | accessor, returns stored data or metadata |
| frame | frame::frame/ops/window::min_periods | accessor, returns stored data or metadata |
| frame | frame::frame/ops/window::new | constructor or conversion |
| frame | frame::frame/ops/window::with_alpha | constructor or conversion |
| frame | frame::frame/ops/window::with_halflife | constructor or conversion |
| frame | frame::frame/ops/window::with_span | constructor or conversion |
| frame | frame::frame/series::astype | conversion between types or containers |
| frame | frame::frame/series::column | accessor, returns stored data or metadata |
| frame | frame::frame/series::iloc | accessor, returns stored data or metadata |
| frame | frame::frame/series::iloc_many | accessor, returns stored data or metadata |
| frame | frame::frame/series::index | accessor, returns stored data or metadata |
| frame | frame::frame/series::is_empty | accessor, returns stored data or metadata |
| frame | frame::frame/series::len | accessor, returns stored data or metadata |
| frame | frame::frame/series::loc | accessor, returns stored data or metadata |
| frame | frame::frame/series::name | accessor, returns stored data or metadata |
| frame | frame::frame/series::new | constructor or conversion |
| frame | frame::frame/series::to_dict | conversion between types or containers |
| frame | frame::frame/series::to_f64 | conversion between types or containers |
| frame | frame::frame/series::to_i64 | conversion between types or containers |
| frame | frame::frame/series::values | accessor, returns stored data or metadata |
| frame | frame::frame/series::with_index | constructor or conversion |
| frame | frame::frame/series::with_name | constructor or conversion |
| frame | frame::frame/series/null_ops::has_nulls | accessor, returns stored data or metadata |
| frame | frame::frame/series/window_ops::min_periods | accessor, returns stored data or metadata |
| frame | frame::frame/value::as_bool | constructor or conversion |
| frame | frame::frame/value::as_float | constructor or conversion |
| frame | frame::frame/value::as_int | constructor or conversion |
| frame | frame::frame/value::as_str | constructor or conversion |
| frame | frame::frame/value::is_null | accessor, returns stored data or metadata |
| scikit | scikit::fft/transform::new | constructor or conversion |
| scikit | scikit::fft/transform::zero | constructor or builder |
| scikit | scikit::interpolate/interp::new | constructor or conversion |
| scikit | scikit::linear_model::new | constructor or conversion |
| scikit | scikit::linear_model::new | constructor or conversion |
| scikit | scikit::sparse/csc::from_dense | constructor or conversion |
| scikit | scikit::sparse/csc::get | accessor, returns stored data or metadata |
| scikit | scikit::sparse/csc::new | constructor or conversion |
| scikit | scikit::sparse/csc::nnz | accessor, returns stored data or metadata |
| scikit | scikit::sparse/csc::to_csr | conversion between types or containers |
| scikit | scikit::sparse/csc::to_dense | conversion between types or containers |
| scikit | scikit::sparse/csr::eye | constructor or conversion |
| scikit | scikit::sparse/csr::from_coo | constructor or conversion |
| scikit | scikit::sparse/csr::from_dense | constructor or conversion |
| scikit | scikit::sparse/csr::get | accessor, returns stored data or metadata |
| scikit | scikit::sparse/csr::new | constructor or conversion |
| scikit | scikit::sparse/csr::nnz | accessor, returns stored data or metadata |
| scikit | scikit::sparse/csr::to_csc | conversion between types or containers |
| scikit | scikit::sparse/csr::to_dense | conversion between types or containers |
| scikit | scikit::spatial/kdtree::dim | accessor, returns stored data or metadata |
| scikit | scikit::spatial/kdtree::is_empty | accessor, returns stored data or metadata |
| scikit | scikit::spatial/kdtree::len | accessor, returns stored data or metadata |
| scikit | scikit::spatial/kdtree::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/distributions::new | constructor or conversion |
| scikit | scikit::stats/hypothesis::new | constructor or conversion |
| scikit | scikit::ts/arima::new | constructor or conversion |
