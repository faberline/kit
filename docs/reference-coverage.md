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
| arraykit | arraykit::array/linalg::det | `numpy.linalg.slogdet(a)[1]` (log of the 50 x 50 det); `numpy.linalg.det(t3)`; `numpy.linalg.det([[1e-7, 0], [0, 1e-7]])` | 1e-9 rel | `det_matches_numpy`, `det_of_small_scale_matrix` | covered |
| arraykit | arraykit::array/linalg::dot | `numpy.dot(x, y)`; `numpy.dot(p23, q32).ravel()` | 1e-9 rel | `dot_matches_numpy` | covered |
| arraykit | arraykit::array/linalg::inner | `numpy.inner(x, y)` | 1e-9 rel | `inner_matches_numpy` | covered |
| arraykit | arraykit::array/linalg::inv | `numpy.linalg.inv(a)[0, :5]`; `numpy.trace(numpy.linalg.inv(a))`; `numpy.linalg.inv([[1e-13]])` | 1e-9 rel; 1e-5 rel for the cond 1e10 trace (the inverse of an ill-conditioned matrix loses digits) | `inv_matches_numpy`, `inv_cond_matches_numpy`, `inv_of_small_scale_matrix` | covered |
| arraykit | arraykit::array/linalg::matmul | `numpy.matmul(p23, q32).ravel()` | 1e-9 rel | `matmul_matches_numpy` | covered |
| arraykit | arraykit::array/linalg::norm | `numpy.linalg.norm(x)`, with ord 2, 1, inf and 3; `numpy.linalg.norm(p23)` (Frobenius, 2-D orders 1 and inf not tested: NumPy gives matrix norms, arraykit flattens) | 1e-9 rel | `norm_matches_numpy` | covered |
| arraykit | arraykit::array/linalg::outer | `numpy.outer(x, y).ravel()` | 1e-9 rel | `outer_matches_numpy` | covered |
| arraykit | arraykit::array/linalg::solve | `numpy.linalg.solve(a, b)[:5]` and `[-5:]`; `numpy.linalg.solve(m, b)` (cond 1e10); `numpy.linalg.solve([[3, 1], [1, 2]], [[9, 1], [8, 3]])` | 1e-9 rel; 1e-5 rel for the cond 1e10 case (a solve loses digits as the condition number grows) | `solve_matches_numpy`, `solve_cond_matches_numpy`, `solve_with_2d_right_hand_side` | covered |
| arraykit | arraykit::array/linalg::trace | `numpy.trace(t3)` | 1e-9 rel | `trace_matches_numpy` | covered |
| arraykit | arraykit::array/linalg_decomp::cholesky | `numpy.diag(numpy.linalg.cholesky(p))` for p = a a^T | 1e-9 rel | `cholesky_matches_numpy` | covered |
| arraykit | arraykit::array/linalg_decomp::eig | `numpy.sort(numpy.linalg.eig([[1, 1], [0, 2]])[0].real)` | 1e-9 rel | `eig_matches_numpy`, `eig_nonsymmetric_triangular` | covered |
| arraykit | arraykit::array/linalg_decomp::eigh | `numpy.linalg.eigh(s)[0]` for s = a + a^T; `numpy.linalg.eigh([[2, 1], [1, 2]])[0]` | 1e-9 rel | `eigh_matches_numpy`, `eigh_symmetric` | covered |
| arraykit | arraykit::array/linalg_decomp::lstsq | `numpy.linalg.lstsq(l, b, rcond=None)`: solution head and tail, `[1][0]` (residual sum), `[2]` (rank) | 1e-9 rel | `lstsq_matches_numpy` | covered |
| arraykit | arraykit::array/linalg_decomp::qr | `abs(numpy.diag(numpy.linalg.qr(a)[1]))` (signs of R differ between libraries) | 1e-9 rel | `qr_matches_numpy` | covered |
| arraykit | arraykit::array/linalg_decomp::svd | `numpy.linalg.svd(a, compute_uv=False)`; `numpy.linalg.svd(m, compute_uv=False)` (cond 1e10); `numpy.linalg.svd([[0, 0], [1, 0]], compute_uv=False)` | 1e-9 rel; 1e-5 rel for the cond 1e10 case | `svd_matches_numpy`, `svd_rank_deficient_has_no_zero_u_column` | covered |
| arraykit | arraykit::array/manip::all | `numpy.all(b)` on bool lists and on `i % 3 == 0` for 10,000 values | exact | `all_matches_numpy` | covered |
| arraykit | arraykit::array/manip::allclose | `numpy.allclose(a, b, rtol=1e-5, atol=1e-8)` and `atol`/`rtol` cases | exact (0 or 1) | `allclose_matches_numpy` | covered |
| arraykit | arraykit::array/manip::any | `numpy.any(b)` on bool lists and on `i % 3 == 0` for 10,000 values | exact | `any_matches_numpy` | covered |
| arraykit | arraykit::array/manip::argsort | `numpy.argsort(a, kind="stable")` on 6 values with ties and on 10,000 values (head, tail, length, weighted sum) | exact | `argsort_matches_numpy` | covered |
| arraykit | arraykit::array/manip::concatenate | `numpy.concatenate` on 1D, axis 0 and axis 1 inputs | 1e-9 rel | `concatenate_matches_numpy` | covered |
| arraykit | arraykit::array/manip::hstack | `numpy.hstack` on 2D and 1D inputs | 1e-9 rel | `hstack_matches_numpy` | covered |
| arraykit | arraykit::array/manip::repeat | `numpy.repeat(a, 3)`, `numpy.repeat(m, 2)`, `len(numpy.repeat(a, 0))` | 1e-9 rel | `repeat_matches_numpy` | covered |
| arraykit | arraykit::array/manip::sort | `numpy.sort` on 6 values and on 10,000 values (head, tail, sum, weighted sum) | exact | `sort_matches_numpy` | covered |
| arraykit | arraykit::array/manip::split | `numpy.split` on 1D, axis 0 and axis 1 inputs, each piece; empty and clamped pieces | 1e-9 rel | `split_matches_numpy`; `split_numpy.rs` | covered |
| arraykit | arraykit::array/manip::stack | `numpy.stack` on axis 0, 1 and 2 | 1e-9 rel | `stack_matches_numpy` | covered |
| arraykit | arraykit::array/manip::tile | `numpy.tile` with 1 and 2 reps, 1D and 2D | 1e-9 rel | `tile_matches_numpy` | covered |
| arraykit | arraykit::array/manip::unique | `numpy.unique` (sorted) on 6 values and on 10,000 values | exact | `unique_matches_numpy` | covered |
| arraykit | arraykit::array/manip::vstack | `numpy.vstack` on 2D and 1D inputs | 1e-9 rel | `vstack_matches_numpy` | covered |
| arraykit | arraykit::array/math::abs | `numpy.abs(x)` | 1e-9 rel | `abs_matches_numpy` | covered |
| arraykit | arraykit::array/math::abs_int | `numpy.abs(v)` on int32 | exact | `abs_int_i32_matches_numpy` | covered |
| arraykit | arraykit::array/math::abs_int | `numpy.abs(v)` on int64 | exact | `abs_int_i64_matches_numpy` | covered |
| arraykit | arraykit::array/math::clip | `numpy.clip(x, -5.0, 5.0)` | 1e-9 rel | `clip_matches_numpy` | covered |
| arraykit | arraykit::array/math::cos | `numpy.cos(x)` | 1e-9 rel | `cos_matches_numpy` | covered |
| arraykit | arraykit::array/math::exp | `numpy.exp(x)` | 1e-9 rel | `exp_matches_numpy` | covered |
| arraykit | arraykit::array/math::linspace | `numpy.linspace(0.0, 1.0, 11)`; `numpy.linspace(-5.0, 5.0, 10001)` (first two, last, abs-sum) | 1e-9 rel | `linspace_matches_numpy` | covered |
| arraykit | arraykit::array/math::log | `numpy.log(x)` | 1e-9 rel | `log_matches_numpy` | covered |
| arraykit | arraykit::array/math::pow | `numpy.power(x, 2.5)`; `numpy.power(x, 3.0)` | 1e-9 rel | `pow_matches_numpy` | covered |
| arraykit | arraykit::array/math::sin | `numpy.sin(x)` | 1e-9 rel | `sin_matches_numpy` | covered |
| arraykit | arraykit::array/math::sqrt | `numpy.sqrt(x)` | 1e-9 rel | `sqrt_matches_numpy` | covered |
| arraykit | arraykit::array/math::tan | `numpy.tan(x)` | 1e-9 rel | `tan_matches_numpy` | covered |
| arraykit | arraykit::array/math::where_cond | `numpy.where(c, x, -x)` | 1e-9 rel | `where_cond_matches_numpy` | covered |
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
| arraykit | arraykit::array/window::count | `numpy.minimum(numpy.arange(1, n + 1), w)` | exact | `rolling_count_matches_numpy` | covered |
| arraykit | arraykit::array/window::diff | `numpy.diff(x, 1)`; `numpy.diff(x, 2)` | 1e-9 rel | `diff_matches_numpy` | covered |
| arraykit | arraykit::array/window::expand_dims | `numpy.expand_dims(a, axis).shape` | exact | `expand_dims_matches_numpy` | covered |
| arraykit | arraykit::array/window::flip | `numpy.flip(x)` | 1e-9 rel | `flip_matches_numpy` | covered |
| arraykit | arraykit::array/window::gradient | `numpy.gradient(x)` | 1e-9 rel | `gradient_matches_numpy` | covered |
| arraykit | arraykit::array/window::interp | `numpy.interp(xq, xp, fp)` | 1e-9 rel | `interp_matches_numpy` | covered |
| arraykit | arraykit::array/window::max | `sliding_window_view(x, w).max(axis=1)` | 1e-9 rel; first w-1 entries: NaN | `rolling_max_matches_numpy` | covered |
| arraykit | arraykit::array/window::mean | `sliding_window_view(x, w).mean(axis=1)` | 1e-9 rel; first w-1 entries: 0.0 | `rolling_mean_matches_numpy` | covered |
| arraykit | arraykit::array/window::min | `sliding_window_view(x, w).min(axis=1)` | 1e-9 rel; first w-1 entries: NaN | `rolling_min_matches_numpy` | covered |
| arraykit | arraykit::array/window::pad | `numpy.pad(x, w)` | 1e-9 rel | `pad_matches_numpy` | covered |
| arraykit | arraykit::array/window::roll | `numpy.roll(x, shift)` | 1e-9 rel | `roll_matches_numpy` | covered |
| arraykit | arraykit::array/window::rolling | `sliding_window_view(x, 1).sum(axis=1)`; `numpy.sum(x)` | 1e-9 rel | `rolling_matches_numpy` | covered |
| arraykit | arraykit::array/window::rot90 | `numpy.rot90(m, k)` | 1e-9 rel (data); exact (shape) | `rot90_matches_numpy` | covered |
| arraykit | arraykit::array/window::squeeze | `numpy.atleast_1d(numpy.squeeze(a)).shape` | exact | `squeeze_matches_numpy` | covered |
| arraykit | arraykit::array/window::std | `sliding_window_view(x, w).std(axis=1, ddof=1)` | 1e-9 rel; first w-1 entries: NaN | `rolling_std_matches_numpy` | covered |
| arraykit | arraykit::array/window::sum | `sliding_window_view(x, w).sum(axis=1)` | 1e-9 rel; first w-1 entries: 0.0 | `rolling_sum_matches_numpy` | covered |
| arraykit | arraykit::array/window::var | `sliding_window_view(x, w).var(axis=1, ddof=1)` | 1e-9 rel; first w-1 entries: NaN | `rolling_var_matches_numpy` | covered |
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
| frame | frame::frame/ops/groupby::agg | `df.groupby('g')['v'].agg(name) for the nine names of agg` | 1e-9 rel | groupby_agg_matches_pandas | covered |
| frame | frame::frame/ops/groupby::aggregate | `df.groupby('g')['v'].agg(lambda x: x.max() - x.min())` | 1e-9 rel | groupby_aggregate_matches_pandas | covered |
| frame | frame::frame/ops/groupby::count | `df.groupby('g')['v'].count()` | 1e-9 rel | groupby_count_matches_pandas | covered |
| frame | frame::frame/ops/groupby::filter_groups | `df.groupby('g').filter(lambda x: x['v'].mean() > 12.5)` | 1e-9 rel | groupby_filter_groups_matches_pandas | covered |
| frame | frame::frame/ops/groupby::first | `df.groupby('g')['v'].first()` | 1e-9 rel | groupby_first_matches_pandas | covered |
| frame | frame::frame/ops/groupby::last | `df.groupby('g')['v'].last()` | 1e-9 rel | groupby_last_matches_pandas | covered |
| frame | frame::frame/ops/groupby::max | `df.groupby('g')['v'].max()` | 1e-9 rel | groupby_max_matches_pandas | covered |
| frame | frame::frame/ops/groupby::mean | `df.groupby('g')['v'].mean()` | 1e-9 rel | groupby_mean_matches_pandas | covered |
| frame | frame::frame/ops/groupby::median | `df.groupby('g')['v'].median()` | 1e-9 rel | groupby_median_matches_pandas | covered |
| frame | frame::frame/ops/groupby::min | `df.groupby('g')['v'].min()` | 1e-9 rel | groupby_min_matches_pandas | covered |
| frame | frame::frame/ops/groupby::std | `df.groupby('g')['v'].std()` | 1e-9 rel | groupby_std_matches_pandas | covered |
| frame | frame::frame/ops/groupby::sum | `df.groupby('g')['v'].sum()` | 1e-9 rel | groupby_sum_matches_pandas | covered |
| frame | frame::frame/ops/groupby::transform | `df.groupby('g')['v'].transform(lambda x: x / x.max())` | 1e-9 rel | groupby_transform_matches_pandas | covered |
| frame | frame::frame/ops/groupby::var | `df.groupby('g')['v'].var()` | 1e-9 rel | groupby_var_matches_pandas | covered |
| frame | frame::frame/ops/join::join | `left.merge(right, on='k', how=how)` | 1e-9 rel | join_matches_pandas | covered |
| frame | frame::frame/ops/join::join | `left.merge(right, on='k', how=how)` | 1e-9 rel | join_method_matches_pandas | covered |
| frame | frame::frame/ops/join::left_join | `left.merge(right, on='k', how='left')` | 1e-9 rel | left_join_matches_pandas | covered |
| frame | frame::frame/ops/join::merge | `left.merge(right, on='k')` | 1e-9 rel | merge_matches_pandas | covered |
| frame | frame::frame/ops/join::outer_join | `left.merge(right, on='k', how='outer')` | 1e-9 rel | outer_join_matches_pandas | covered |
| frame | frame::frame/ops/join::right_join | `left.merge(right, on='k', how='right')` | 1e-9 rel | right_join_matches_pandas | covered |
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
| frame | frame::frame/ops/window::adjust | `s.ewm(alpha=0.3, adjust=False).mean()` | 1e-9 rel | ewm_adjust_matches_pandas | covered |
| frame | frame::frame/ops/window::ewm_alpha | `s.ewm(alpha=0.3).mean()` | 1e-9 rel | ewm_alpha_matches_pandas | covered |
| frame | frame::frame/ops/window::ewm_halflife | `s.ewm(halflife=4.0).mean()` | 1e-9 rel | ewm_halflife_matches_pandas | covered |
| frame | frame::frame/ops/window::ewm_span | `s.ewm(span=10.0).mean()` | 1e-9 rel | ewm_span_matches_pandas | covered |
| frame | frame::frame/ops/window::expanding | `s.expanding().sum()` | 1e-9 rel | expanding_sum_matches_pandas | covered |
| frame | frame::frame/ops/window::max | `s.expanding().max()` | 1e-9 rel | expanding_max_matches_pandas | covered |
| frame | frame::frame/ops/window::mean | `s.expanding().mean()` | 1e-9 rel | expanding_mean_matches_pandas | covered |
| frame | frame::frame/ops/window::mean | `s.ewm(span=10.0).mean()` | 1e-9 rel | ewm_mean_matches_pandas | covered |
| frame | frame::frame/ops/window::min | `s.expanding().min()` | 1e-9 rel | expanding_min_matches_pandas | covered |
| frame | frame::frame/ops/window::std | `s.expanding().std()` | 1e-9 rel | expanding_std_matches_pandas | covered |
| frame | frame::frame/ops/window::std | `s.ewm(span=10.0).std()` | 1e-9 rel | ewm_std_matches_pandas | covered |
| frame | frame::frame/ops/window::sum | `s.expanding().sum()` | 1e-9 rel | expanding_sum_matches_pandas | covered |
| frame | frame::frame/ops/window::var | `s.expanding().var()` | 1e-9 rel | expanding_var_matches_pandas | covered |
| frame | frame::frame/ops/window::var | `s.ewm(span=10.0).var()` | 1e-9 rel | ewm_var_matches_pandas | covered |
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
| frame | frame::frame/series/null_ops::apply | s * 2 + 1 | 1e-9 rel | apply_matches_pandas | covered |
| frame | frame::frame/series/null_ops::bfill | s.bfill() | 1e-9 rel | bfill_matches_pandas | covered |
| frame | frame::frame/series/null_ops::dropna | s.dropna() | 1e-9 rel | dropna_matches_pandas | covered |
| frame | frame::frame/series/null_ops::ffill | s.ffill() | 1e-9 rel | ffill_matches_pandas | covered |
| frame | frame::frame/series/null_ops::fillna | s.fillna(-1.5) | 1e-9 rel | fillna_matches_pandas | covered |
| frame | frame::frame/series/null_ops::interpolate | s.interpolate() | 1e-9 rel | interpolate_matches_pandas | covered |
| frame | frame::frame/series/null_ops::interpolate_method | s.interpolate(method="linear" / "ffill" / "bfill" / "nearest") | 1e-9 rel | interpolate_method_matches_pandas | covered |
| frame | frame::frame/series/null_ops::isna | s.isna() | 1e-9 rel | isna_matches_pandas | covered |
| frame | frame::frame/series/null_ops::notna | s.notna() | 1e-9 rel | notna_matches_pandas | covered |
| frame | frame::frame/series/null_ops::null_count | s.isna().sum() | 1e-9 rel | null_count_matches_pandas | covered |
| frame | frame::frame/series/window_ops::cummax | s.cummax() | 1e-9 rel | cummax_matches_pandas | covered |
| frame | frame::frame/series/window_ops::cummin | s.cummin() | 1e-9 rel | cummin_matches_pandas | covered |
| frame | frame::frame/series/window_ops::cumprod | s.cumprod() | 1e-9 rel | cumprod_matches_pandas | covered |
| frame | frame::frame/series/window_ops::cumsum | s.cumsum() | 1e-9 rel | cumsum_matches_pandas | covered |
| frame | frame::frame/series/window_ops::diff | s.diff(1), s.diff(2) | 1e-9 rel | diff_matches_pandas | covered |
| frame | frame::frame/series/window_ops::max | s.rolling(3).max() | 1e-9 rel | rolling_max_matches_pandas | covered |
| frame | frame::frame/series/window_ops::mean | s.rolling(3).mean() | 1e-9 rel | rolling_mean_matches_pandas | covered |
| frame | frame::frame/series/window_ops::min | s.rolling(3).min() | 1e-9 rel | rolling_min_matches_pandas | covered |
| frame | frame::frame/series/window_ops::pct_change | s.pct_change(1), s.pct_change(2) | 1e-9 rel | pct_change_matches_pandas | covered |
| frame | frame::frame/series/window_ops::rolling | s.rolling(3).sum() | 1e-9 rel | rolling_matches_pandas | covered |
| frame | frame::frame/series/window_ops::shift | s.shift(2), s.shift(-2) | 1e-9 rel | shift_matches_pandas | covered |
| frame | frame::frame/series/window_ops::std | s.rolling(3).std() | 1e-9 rel | rolling_std_matches_pandas | covered |
| frame | frame::frame/series/window_ops::sum | s.rolling(3).sum() | 1e-9 rel | rolling_sum_matches_pandas | covered |
| frame | frame::frame/series/window_ops::var | s.rolling(3).var() | 1e-9 rel | rolling_var_matches_pandas | covered |
| scikit | scikit::fft/transform::conj | numpy.conj(numpy.fft.fft(sample(8))) | 1e-9 rel; exact zeros 1e-12 abs | conj_matches_numpy | covered |
| scikit | scikit::fft/transform::fft | numpy.fft.fft(sample(n)), n in 5, 8, 1000, 1024 | 1e-9 rel; exact zeros 1e-12 abs | fft_matches_numpy | covered |
| scikit | scikit::fft/transform::fftfreq | numpy.fft.fftfreq(n, d), (8, 0.5), (5, 0.1), (1000, 0.001) | 1e-9 rel; exact zeros 1e-12 abs | fftfreq_matches_numpy | covered |
| scikit | scikit::fft/transform::ifft | numpy.fft.ifft(sample(n)), n in 5, 8, 1000, 1024 | 1e-9 rel; exact zeros 1e-12 abs | ifft_matches_numpy | covered |
| scikit | scikit::fft/transform::irfft | numpy.fft.irfft(numpy.fft.rfft(sample(n)), n), n in 5, 8, 1000, 1024 | 1e-9 rel; exact zeros 1e-12 abs | irfft_matches_numpy | covered |
| scikit | scikit::fft/transform::norm | numpy.abs(numpy.fft.fft(sample(8))) | 1e-9 rel; exact zeros 1e-12 abs | norm_matches_numpy | covered |
| scikit | scikit::fft/transform::rfft | numpy.fft.rfft(sample(n)), n in 5, 8, 1000, 1024 | 1e-9 rel; exact zeros 1e-12 abs | rfft_matches_numpy | covered |
| scikit | scikit::fft/transform::rfftfreq | numpy.fft.rfftfreq(n, d), (8, 0.5), (5, 0.1), (1000, 0.001) | 1e-9 rel; exact zeros 1e-12 abs | rfftfreq_matches_numpy | covered |
| scikit | scikit::integrate/ode::euler | textbook Euler step y + dt*f(t, y) (no SciPy equal) | 1e-12 | euler_matches_formula | covered |
| scikit | scikit::integrate/ode::rk4 | textbook RK4 step (no SciPy equal) | 1e-12 | rk4_matches_formula | covered |
| scikit | scikit::integrate/ode::solve_ivp | `scipy.integrate.solve_ivp(-y, (0, 1), [1.0], method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]`; same with `[y1, -y0]` from `[1.0, 0.0]` | 5e-3 Euler, 1e-8 RK4, 1e-5 RK45 | solve_ivp_matches_scipy | covered |
| scikit | scikit::integrate/quad::cumulative_trapezoid | `scipy.integrate.cumulative_trapezoid(y, dx=0.25)`, y[i] = ((i*37+11)%101)/4.0 for 41 points | 1e-9 rel (abs below 1) | cumulative_trapezoid_matches_scipy | covered |
| scikit | scikit::integrate/quad::fixed_quad | `scipy.integrate.fixed_quad(f, 0.5, 2.5, n=k)[0]` for k = 1..5, f = exp and f = cos(3x) | 1e-9 rel (abs below 1) | fixed_quad_matches_scipy | covered |
| scikit | scikit::integrate/quad::quad | `scipy.integrate.quad(f, a, b, epsabs=1e-13, epsrel=1e-13)[0]` for sin on [0, pi], exp(-x*x) on [0, 2], 1/(1+x*x) on [0, 1], x*log(x+1) on [0, 2], exp(-x)*sin(x) on [0, 10] | 1e-9 rel (abs below 1) | quad_matches_scipy | covered |
| scikit | scikit::integrate/quad::simps | `scipy.integrate.simpson(y, dx=0.25)`, same y, 41 points (even number of intervals only) | 1e-9 rel (abs below 1) | simps_matches_scipy | covered |
| scikit | scikit::integrate/quad::trapezoid | `scipy.integrate.trapezoid(y, dx=0.25)`, same y, 41 points | 1e-9 rel (abs below 1) | trapezoid_matches_scipy | covered |
| scikit | scikit::interpolate/interp::eval | `scipy.interpolate.CubicSpline(xs, ys, bc_type="natural")(xq)` | 1e-9 rel (abs below 1) | eval_matches_scipy | covered |
| scikit | scikit::interpolate/interp::interp1d | `scipy.interpolate.interp1d(xs, ys, kind="linear")(xq)`; `CubicSpline(xs, ys, bc_type="natural")(xq)` for cubic (kit spline is natural) | 1e-9 rel (abs below 1) | interp1d_matches_scipy | covered |
| scikit | scikit::interpolate/interp2d::interp2d | `scipy.interpolate.RegularGridInterpolator((xg, yg), z, method="linear")` on a 6 x 7 grid | 1e-9 rel (abs below 1) | interp2d_matches_scipy | covered |
| scikit | scikit::linear_model::alpha | - | - | - | not covered |
| scikit | scikit::linear_model::coef | - | - | - | not covered |
| scikit | scikit::linear_model::coef | - | - | - | not covered |
| scikit | scikit::linear_model::fit | - | - | - | not covered |
| scikit | scikit::linear_model::fit | - | - | - | not covered |
| scikit | scikit::linear_model::intercept | - | - | - | not covered |
| scikit | scikit::linear_model::intercept | - | - | - | not covered |
| scikit | scikit::linear_model::predict | - | - | - | not covered |
| scikit | scikit::linear_model::predict | - | - | - | not covered |
| scikit | scikit::optimize/curve_fit::curve_fit | `scipy.optimize.curve_fit(a*exp(-b*x)+c, x, y, p0=[2.0, 1.0, 0.4], xtol=1e-12, ftol=1e-12)[0]`, 40 points | 1e-6 rel (abs below 1) | curve_fit_matches_scipy | covered |
| scikit | scikit::optimize/linprog::linprog | `scipy.optimize.linprog(c, A_ub, b_ub, bounds=(0, None), method="highs")` for two 2- and 3-variable problems: `.x` and `.fun` | 1e-9 rel (abs below 1) | linprog_matches_scipy | covered |
| scikit | scikit::optimize/minimize::bfgs | `scipy.optimize.minimize(rosen, [-1.2, 1.0], jac=rosen_der, method='BFGS', options={'gtol': 1e-12})` (`.x`, `.fun`); convex bowl from [0, 0] | 1e-6 rel (abs below 1) | bfgs_matches_scipy | covered |
| scikit | scikit::optimize/minimize::nelder_mead | `scipy.optimize.minimize(rosen, [-1.2, 1.0], method='Nelder-Mead', xatol=1e-12, fatol=1e-12)` (`.x`, `.fun`); convex bowl from [0, 0] | 1e-6 rel (abs below 1) | nelder_mead_matches_scipy | covered |
| scikit | scikit::optimize/root::bisect | `scipy.optimize.bisect(x**3-2x-5, 2, 3, xtol=1e-14, rtol=4*eps)`; `bisect(cos(x)-x, 0, 1, ...)` | 1e-9 rel (abs below 1) | bisect_matches_scipy | covered |
| scikit | scikit::optimize/root::brentq | `scipy.optimize.brentq(x**3-2x-5, 2, 3, xtol=1e-14, rtol=4*eps)`; `brentq(cos(x)-x, 0, 1, ...)` | 1e-9 rel (abs below 1) | brentq_matches_scipy | covered |
| scikit | scikit::optimize/root::newton | `scipy.optimize.newton(x**3-2x-5, 2.0, fprime=3x**2-2, tol=1e-14)`; `newton(cos(x)-x, 0.5, fprime=-sin(x)-1, tol=1e-14)` | 1e-9 rel (abs below 1) | newton_matches_scipy | covered |
| scikit | scikit::signal/convolve::convolve | scipy.signal.convolve(sample(n), b, mode, method='direct'), 3 modes, 2 cases | 1e-9 rel; exact zeros 1e-12 abs | convolve_matches_scipy | covered |
| scikit | scikit::signal/convolve::correlate | scipy.signal.correlate(sample(n), b, mode, method='direct'), 3 modes, 2 cases | 1e-9 rel; exact zeros 1e-12 abs | correlate_matches_scipy | covered |
| scikit | scikit::signal/filter::butter | scipy.signal.sosfilt(scipy.signal.butter(order, wn, btype=t, output='sos'), sample(64)) | 1e-9 rel; exact zeros 1e-12 abs | butter_matches_scipy | covered |
| scikit | scikit::signal/filter::lfilter | scipy.signal.lfilter(b, a, sample(n)), 4 cases | 1e-9 rel; exact zeros 1e-12 abs | lfilter_matches_scipy | covered |
| scikit | scikit::signal/filter::sosfilt | scipy.signal.sosfilt(sos, sample(64)), sos of butter(4, 0.3, lowpass) and butter(3, 0.2, highpass) | 1e-9 rel; exact zeros 1e-12 abs | sosfilt_matches_scipy | covered |
| scikit | scikit::signal/filtfilt::filtfilt | forward and backward scipy.signal.lfilter (scipy.signal.filtfilt differs: it pads and uses lfilter_zi) | 1e-9 rel; exact zeros 1e-12 abs | filtfilt_matches_scipy_lfilter | covered |
| scikit | scikit::signal/peaks::find_peaks | scipy.signal.find_peaks(sample(40), options), 4 option sets | 1e-9 rel; exact zeros 1e-12 abs | find_peaks_matches_scipy | covered |
| scikit | scikit::signal/stft::stft | numpy.fft.rfft of x[s:s+w] * scipy.signal.windows.hann(w, sym=True), 2 configs | 1e-9 rel; exact zeros 1e-12 abs | stft_matches_numpy_hann | covered |
| scikit | scikit::signal/windows::blackman | scipy.signal.windows.blackman(n, sym=True), n in 9, 64 | 1e-9 rel; exact zeros 1e-12 abs | blackman_matches_scipy | covered |
| scikit | scikit::signal/windows::hamming | scipy.signal.windows.hamming(n, sym=True), n in 9, 64 | 1e-9 rel; exact zeros 1e-12 abs | hamming_matches_scipy | covered |
| scikit | scikit::signal/windows::hann | scipy.signal.windows.hann(n, sym=True), n in 9, 64 | 1e-9 rel; exact zeros 1e-12 abs | hann_matches_scipy | covered |
| scikit | scikit::signal/windows::kaiser | scipy.signal.windows.kaiser(n, beta, sym=True), n in 9, 64, beta 5.0 and 8.6 | 1e-9 rel; exact zeros 1e-12 abs | kaiser_matches_scipy | covered |
| scikit | scikit::sparse/csc::add | `scipy.sparse.csc_matrix(A) + scipy.sparse.csc_matrix(B)`: `.nnz`, `.sum()`, `.toarray()[7]` | 1e-9 rel + 1e-12 abs | `csc_add_matches_scipy` | covered |
| scikit | scikit::sparse/csc::density | `scipy.sparse.csc_matrix(A).nnz / (50 * 50)` (SciPy has no density method) | 1e-9 rel + 1e-12 abs | `csc_density_matches_scipy` | covered |
| scikit | scikit::sparse/csc::dot | `scipy.sparse.csc_matrix(A).dot(xv)` | 1e-9 rel + 1e-12 abs | `csc_dot_matches_scipy` | covered |
| scikit | scikit::sparse/csc::scale | `2.5 * scipy.sparse.csc_matrix(A)`: `.sum()`, `.toarray()[7]` | 1e-9 rel + 1e-12 abs | `csc_scale_matches_scipy` | covered |
| scikit | scikit::sparse/csr::add | `scipy.sparse.csr_matrix(A) + scipy.sparse.csr_matrix(B)`: `.nnz`, `.sum()`, `.toarray()[7]` | 1e-9 rel + 1e-12 abs | `csr_add_matches_scipy` | covered |
| scikit | scikit::sparse/csr::density | `scipy.sparse.csr_matrix(A).nnz / (50 * 50)` (SciPy has no density method) | 1e-9 rel + 1e-12 abs | `csr_density_matches_scipy` | covered |
| scikit | scikit::sparse/csr::dot | `scipy.sparse.csr_matrix(A).dot(xv)` | 1e-9 rel + 1e-12 abs | `csr_dot_matches_scipy` | covered |
| scikit | scikit::sparse/csr::scale | `2.5 * scipy.sparse.csr_matrix(A)`: `.sum()`, `.toarray()[7]` | 1e-9 rel + 1e-12 abs | `csr_scale_matches_scipy` | covered |
| scikit | scikit::sparse/csr::transpose | `scipy.sparse.csr_matrix(A).T.tocsr()`: `.nnz`, `.dot(xv)`, `.toarray()[7]` | 1e-9 rel + 1e-12 abs | `csr_transpose_matches_scipy` | covered |
| scikit | scikit::spatial/distance::cdist | `scipy.spatial.distance.cdist(P[:20], P[20:], metric).ravel()` for euclidean, cityblock, chebyshev, cosine | 1e-9 rel + 1e-12 abs | `cdist_matches_scipy` | covered |
| scikit | scikit::spatial/distance::chebyshev | `scipy.spatial.distance.chebyshev(P[0], P[k])`, k = 1..49 | 1e-9 rel + 1e-12 abs | `chebyshev_matches_scipy` | covered |
| scikit | scikit::spatial/distance::cosine_distance | `scipy.spatial.distance.cosine(P[0], P[k])`, k = 1..49 | 1e-9 rel + 1e-12 abs | `cosine_distance_matches_scipy` | covered |
| scikit | scikit::spatial/distance::euclidean | `scipy.spatial.distance.euclidean(P[0], P[k])`, k = 1..49 | 1e-9 rel + 1e-12 abs | `euclidean_matches_scipy` | covered |
| scikit | scikit::spatial/distance::manhattan | `scipy.spatial.distance.cityblock(P[0], P[k])`, k = 1..49 | 1e-9 rel + 1e-12 abs | `manhattan_matches_scipy` | covered |
| scikit | scikit::spatial/distance::minkowski | `scipy.spatial.distance.minkowski(P[0], P[k], 3)`, k = 1..49 | 1e-9 rel + 1e-12 abs | `minkowski_matches_scipy` | covered |
| scikit | scikit::spatial/distance::pdist | `scipy.spatial.distance.pdist(P, metric)` for euclidean, cityblock, chebyshev, cosine | 1e-9 rel + 1e-12 abs | `pdist_matches_scipy` | covered |
| scikit | scikit::spatial/distance::squareform | `scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(P))`: flat and row 7 | 1e-9 rel + 1e-12 abs | `squareform_matches_scipy` | covered |
| scikit | scikit::spatial/hull::convex_hull | `sorted(scipy.spatial.ConvexHull(H).vertices)`, H = 30 integer-formula points | 1e-9 rel + 1e-12 abs | `convex_hull_matches_scipy` | covered |
| scikit | scikit::spatial/hull::convex_hull_area | `scipy.spatial.ConvexHull(H).volume` (the area in 2-D) | 1e-9 rel + 1e-12 abs | `convex_hull_area_matches_scipy` | covered |
| scikit | scikit::spatial/kdtree::query | `scipy.spatial.cKDTree(T).query(Q[k], k=5)`, k = 0..4 | 1e-9 rel + 1e-12 abs | `kdtree_query_matches_scipy` | covered |
| scikit | scikit::spatial/kdtree::query_ball | `scipy.spatial.cKDTree(T).query_ball_point(Q[k], 4.5)` (distances: `numpy.linalg.norm`), k = 0..4 | 1e-9 rel + 1e-12 abs | `kdtree_query_ball_matches_scipy` | covered |
| scikit | scikit::stats/descriptive::describe | np.mean, np.std(ddof=1), np.min, np.percentile, np.median, np.max; scipy.stats.skew, kurtosis (bias=False) | 1e-9 rel | describe_matches_numpy_scipy | covered |
| scikit | scikit::stats/descriptive::geometric_mean | scipy.stats.gmean(X) | 1e-9 rel | geometric_mean_matches_scipy | covered |
| scikit | scikit::stats/descriptive::harmonic_mean | scipy.stats.hmean(X) | 1e-9 rel | harmonic_mean_matches_scipy | covered |
| scikit | scikit::stats/descriptive::iqr | scipy.stats.iqr(X) | 1e-9 rel | iqr_matches_scipy | covered |
| scikit | scikit::stats/descriptive::kurtosis | scipy.stats.kurtosis(X, fisher=True, bias=False) | 1e-9 rel | kurtosis_matches_scipy | covered |
| scikit | scikit::stats/descriptive::median | np.median(X) | 1e-9 rel | median_matches_numpy | covered |
| scikit | scikit::stats/descriptive::median_abs_deviation | scipy.stats.median_abs_deviation(X) | 1e-9 rel | median_abs_deviation_matches_scipy | covered |
| scikit | scikit::stats/descriptive::mode | scipy.stats.mode(M).mode | 1e-9 rel | mode_matches_scipy | covered |
| scikit | scikit::stats/descriptive::moment | scipy.stats.moment(X, order=k), k = 2, 3, 4 | 1e-9 rel | moment_matches_scipy | covered |
| scikit | scikit::stats/descriptive::percentile | np.percentile(X, p), p in 0, 10, 25, 50, 90, 100 | 1e-9 rel | percentile_matches_numpy | covered |
| scikit | scikit::stats/descriptive::raw_moment | np.mean(X**k), k = 2, 3 | 1e-9 rel | raw_moment_matches_numpy | covered |
| scikit | scikit::stats/descriptive::sem | scipy.stats.sem(X) | 1e-9 rel | sem_matches_scipy | covered |
| scikit | scikit::stats/descriptive::skew | scipy.stats.skew(X, bias=False) | 1e-9 rel | skew_matches_scipy | covered |
| scikit | scikit::stats/descriptive::trim_mean | scipy.stats.trim_mean(X, 0.1), (X, 0.25) | 1e-9 rel | trim_mean_matches_scipy | covered |
| scikit | scikit::stats/descriptive::variation | scipy.stats.variation(X, ddof=1) | 1e-9 rel | variation_matches_scipy | covered |
| scikit | scikit::stats/descriptive::zscore | scipy.stats.zscore(X, ddof=1) | 1e-9 rel | zscore_matches_scipy | covered |
| scikit | scikit::stats/distributions::beta_fn | scipy.special.beta(a, b) | 1e-9 rel | beta_fn_matches_scipy | covered |
| scikit | scikit::stats/distributions::binomial_coef | math.comb(n, k) | 1e-12 rel | binomial_coef_matches_python | covered |
| scikit | scikit::stats/distributions::erf | scipy.special.erf(x) | 1e-12 rel + 1e-15 abs | erf_matches_scipy | covered |
| scikit | scikit::stats/distributions::factorial | math.factorial(n) | 1e-15 rel | factorial_matches_python | covered |
| scikit | scikit::stats/distributions::gamma_fn | scipy.special.gamma(x) | 1e-9 rel | gamma_fn_matches_scipy | covered |
| scikit | scikit::stats/distributions::incomplete_beta | scipy.special.betainc(a, b, x) | 1e-9 rel | incomplete_beta_matches_scipy | covered |
| scikit | scikit::stats/distributions::ln_gamma | scipy.special.gammaln(x) | 1e-12 rel | ln_gamma_matches_scipy | covered |
| scikit | scikit::stats/distributions::lower_incomplete_gamma | scipy.special.gammainc(s, x) * scipy.special.gamma(s) | 1e-9 rel | lower_incomplete_gamma_matches_scipy | covered |
| scikit | scikit::stats/distributions::standard | scipy.stats.norm(0.0, 1.0): cdf, pdf, mean, variance on a grid | 1e-9 rel + 1e-12 abs | normal_standard_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::chi2_contingency | scipy.stats.chi2_contingency(table, correction=False) | 1e-9 rel | chi2_contingency_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::chisquare | scipy.stats.chisquare(O, expected) | 1e-9 rel | chisquare_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::f_oneway | scipy.stats.f_oneway(X, Y, Z) | 1e-9 rel | f_oneway_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::is_significant | closed form: pvalue < alpha | exact | is_significant_follows_the_pvalue_rule | covered |
| scikit | scikit::stats/hypothesis::kstest_normal | scipy.stats.kstest(z, 'norm').statistic; scipy.stats.kstwobign.sf(...) for the p-value | 1e-9 rel (statistic); 1e-6 abs (p-value: one-term series) | kstest_normal_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::mannwhitneyu | scipy.stats.mannwhitneyu(X, Y, use_continuity=False, method='asymptotic') | 1e-9 rel | mannwhitneyu_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::pearsonr | scipy.stats.pearsonr(X, Y) | 1e-9 rel | pearsonr_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::spearmanr | scipy.stats.spearmanr(X, Y) | 1e-9 rel | spearmanr_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::ttest_1samp | scipy.stats.ttest_1samp(X, 13.0) | 1e-9 rel | ttest_1samp_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::ttest_ind | scipy.stats.ttest_ind(X, Y, equal_var=True / False) | 1e-9 rel | ttest_ind_matches_scipy | covered |
| scikit | scikit::stats/hypothesis::ttest_rel | scipy.stats.ttest_rel(X, Y) | 1e-9 rel | ttest_rel_matches_scipy | covered |
| scikit | scikit::ts/acf::acf | `statsmodels.tsa.stattools.acf(y, nlags=24, adjusted=False, fft=False)` | 1e-9 rel + 1e-12 abs | `acf_matches_statsmodels` | covered |
| scikit | scikit::ts/acf::is_significant | `closed form: p_value < alpha` | 1e-9 rel + 1e-12 abs | `is_significant_follows_the_pvalue_rule` | covered |
| scikit | scikit::ts/acf::ljung_box | `statsmodels.stats.diagnostic.acorr_ljungbox(x, lags=[k])`: lb_stat, lb_pvalue (the p-value uses the same tolerance) | 1e-9 rel + 1e-12 abs | `ljung_box_matches_statsmodels` | covered |
| scikit | scikit::ts/acf::pacf | `statsmodels.tsa.stattools.pacf(y, nlags=24, method='ywm')[1:]` | 1e-9 rel + 1e-12 abs | `pacf_matches_statsmodels` | covered |
| scikit | scikit::ts/acf::pacf_from_acf | `statsmodels.tsa.stattools.pacf(y, nlags=24, method='ywm')[1:]`, input: the acf of `acf` (not the 12-digit fixture) | 1e-9 rel + 1e-12 abs | `pacf_from_acf_matches_statsmodels` | covered |
| scikit | scikit::ts/arima::aic | closed form of `ArimaModel::aic`: n * ln(RSS / n) + 2 * (p + q + 1), RSS from statsmodels one-step residuals | 1e-9 rel + 1e-12 abs | `aic_matches_statsmodels_residuals` | covered |
| scikit | scikit::ts/arima::difference | `numpy.diff(b, n=d)`, d = 1, 2, 3 | 1e-9 rel + 1e-12 abs | `difference_matches_numpy` | covered |
| scikit | scikit::ts/arima::fit | AR and mean: `statsmodels.regression.linear_model.yule_walker(..., method='mle')`; MA: `scipy.optimize.minimize` of the CSS | AR: 1e-9 rel + 1e-12 abs; MA: 2e-3 abs (kit searches the MA term with a CSS step of 1e-3; the reference is the CSS optimum, not the exact-likelihood fit) | `fit_ar_matches_statsmodels`, `fit_ma_matches_css_optimum` | covered |
| scikit | scikit::ts/arima::fitted_values | `statsmodels.tsa.arima.model.ARIMA(..., trend='c').smooth(...).fittedvalues`; d = 0 only, d >= 1 is #312 (a defect) | 1e-9 rel + 1e-12 abs | `fitted_values_matches_statsmodels` | covered |
| scikit | scikit::ts/arima::forecast | `statsmodels.tsa.arima.model.ARIMA(..., trend='c').smooth(...).forecast(5)`; d = 1 cumulated from b[-1] | 1e-9 rel + 1e-12 abs | `forecast_matches_statsmodels` | covered |
| scikit | scikit::ts/decompose::seasonal_decompose | `statsmodels.tsa.seasonal.seasonal_decompose(y, model, period, two_sided=True)`: seasonal, trend, resid | 1e-9 rel + 1e-12 abs | `seasonal_decompose_matches_statsmodels` | covered |
| scikit | scikit::ts/decompose::seasonal_strength | closed form from the statsmodels parts: 1 - var(resid) / var(resid + seasonal) | 1e-9 rel + 1e-12 abs | `seasonal_strength_matches_closed_form` | covered |
| scikit | scikit::ts/decompose::trend_strength | closed form from the statsmodels parts: 1 - var(resid) / var(trend + resid) (Wang, Smith and Hyndman 2006) | 1e-9 rel + 1e-12 abs | `trend_strength_matches_closed_form` | covered |
| scikit | scikit::ts/ops::ewma | `pandas.Series.ewm(alpha=0.3, adjust=False).mean()` | 1e-9 rel + 1e-12 abs | `ewma_matches_pandas` | covered |
| scikit | scikit::ts/ops::ewma_halflife | `pandas.Series.ewm(halflife=3, adjust=False).mean()` | 1e-9 rel + 1e-12 abs | `ewma_halflife_matches_pandas` | covered |
| scikit | scikit::ts/ops::ewma_span | `pandas.Series.ewm(span=10, adjust=False).mean()` | 1e-9 rel + 1e-12 abs | `ewma_span_matches_pandas` | covered |
| scikit | scikit::ts/ops::expanding_max | `pandas.Series.expanding().max()` | 1e-9 rel + 1e-12 abs | `expanding_max_matches_pandas` | covered |
| scikit | scikit::ts/ops::expanding_mean | `pandas.Series.expanding().mean()` | 1e-9 rel + 1e-12 abs | `expanding_mean_matches_pandas` | covered |
| scikit | scikit::ts/ops::expanding_min | `pandas.Series.expanding().min()` | 1e-9 rel + 1e-12 abs | `expanding_min_matches_pandas` | covered |
| scikit | scikit::ts/ops::expanding_std | `pandas.Series.expanding(min_periods=2).std(ddof=0)` | 1e-9 rel + 1e-12 abs | `expanding_std_matches_pandas` | covered |
| scikit | scikit::ts/ops::expanding_sum | `pandas.Series.expanding().sum()` | 1e-9 rel + 1e-12 abs | `expanding_sum_matches_pandas` | covered |
| scikit | scikit::ts/ops::lag | `pandas.Series.shift(3)` | 1e-9 rel + 1e-12 abs | `lag_matches_pandas` | covered |
| scikit | scikit::ts/ops::lead | `pandas.Series.shift(-2)` | 1e-9 rel + 1e-12 abs | `lead_matches_pandas` | covered |
| scikit | scikit::ts/resample::downsample | `pandas.Series.groupby(arange(58) // 4).agg(name)`, 7 aggregations | 1e-9 rel + 1e-12 abs | `downsample_matches_pandas` | covered |
| scikit | scikit::ts/resample::resample_to_len | `numpy.interp(numpy.linspace(0, 8, k), arange(9), x)` | 1e-9 rel + 1e-12 abs | `resample_to_len_matches_numpy` | covered |
| scikit | scikit::ts/resample::upsample | `numpy.interp` (linear); `pandas.Series.reindex` (ffill, bfill, nearest) | 1e-9 rel + 1e-12 abs | `upsample_matches_numpy_pandas` | covered |
| scikit | scikit::ts/smoothing::forecast | `statsmodels.tsa.holtwinters.SimpleExpSmoothing(...).fit(...).forecast(3)` | 1e-9 rel + 1e-12 abs | `ses_forecast_matches_statsmodels` | covered |
| scikit | scikit::ts/smoothing::forecast | `statsmodels.tsa.holtwinters.Holt(y[1:], ...).fit(...).forecast(5)` | 1e-9 rel + 1e-12 abs | `holt_forecast_matches_statsmodels` | covered |
| scikit | scikit::ts/smoothing::forecast | additive: closed form from the statsmodels states (smoothing_seasonal = gamma * (1 - alpha)); multiplicative: numpy loop of the component form (statsmodels has another form) | 1e-9 rel + 1e-12 abs | `holt_winters_forecast_matches_statsmodels` | covered |
| scikit | scikit::ts/smoothing::holt | `statsmodels.tsa.holtwinters.Holt(y[1:], ...).fit(smoothing_level=0.5, smoothing_trend=0.2, optimized=False)` | 1e-9 rel + 1e-12 abs | `holt_matches_statsmodels` | covered |
| scikit | scikit::ts/smoothing::holt_winters | additive: `statsmodels.tsa.holtwinters.ExponentialSmoothing` with smoothing_seasonal = gamma * (1 - alpha); multiplicative: numpy loop of the component form (statsmodels has another form) | 1e-9 rel + 1e-12 abs | `holt_winters_matches_statsmodels` | covered |
| scikit | scikit::ts/smoothing::ses | `statsmodels.tsa.holtwinters.SimpleExpSmoothing(...).fit(smoothing_level=0.3, optimized=False)` | 1e-9 rel + 1e-12 abs | `ses_matches_statsmodels` | covered |

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
