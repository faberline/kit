# arraykit parity with NumPy

The reference library is NumPy. The states are `has` (same result as NumPy for the supported dtypes and shapes), `partial` (works only for some dtypes, shapes or axes; the phrase after the path says what is missing) and `missing` (not in arraykit).

Supported dtypes (`DType`): `Float32`, `Float64`, `Int32`, `Int64`, `Bool`. Complex dtypes, negative axes, `out=` and `dtype=` arguments are not supported.

Names marked `(SciPy)` are SciPy functions, not NumPy functions.

## Table 1: public functions and types

| Function | NumPy name | State | Evidence |
|---|---|---|---|
| `NdArray::new` | `numpy.array` | has | crates/arraykit/src/array/ndarray.rs |
| `zeros` | `numpy.zeros` | has | crates/arraykit/src/array/ndarray.rs |
| `ones` | `numpy.ones` | has | crates/arraykit/src/array/ndarray.rs |
| `full` | `numpy.full` | has | crates/arraykit/src/array/ndarray.rs |
| `arange` | `numpy.arange` | has | crates/arraykit/src/array/ndarray.rs |
| `eye` | `numpy.eye` | has | crates/arraykit/src/array/ndarray.rs |
| `identity` | `numpy.identity` | has | crates/arraykit/src/array/ndarray.rs |
| `diag` | `numpy.diag` | has | crates/arraykit/src/array/ndarray.rs |
| `shape` | `ndarray.shape` | has | crates/arraykit/src/array/ndarray.rs |
| `dims` | `ndarray.shape` | has | crates/arraykit/src/array/ndarray.rs |
| `ndim` | `ndarray.ndim` | has | crates/arraykit/src/array/ndarray.rs |
| `size` | `ndarray.size` | has | crates/arraykit/src/array/ndarray.rs |
| `data` | `ndarray.data` (buffer) | has | crates/arraykit/src/array/ndarray.rs |
| `data_mut` | `ndarray.data` (buffer, writable) | has | crates/arraykit/src/array/ndarray.rs |
| `get` | `ndarray.item` (index form) | has | crates/arraykit/src/array/ndarray.rs |
| `set` | `ndarray.__setitem__` (index form) | has | crates/arraykit/src/array/ndarray.rs |
| `reshape` | `numpy.reshape` | has | crates/arraykit/src/array/ndarray.rs |
| `flatten` | `numpy.ndarray.flatten` | has | crates/arraykit/src/array/ndarray.rs |
| `slice` | `ndarray.__getitem__` (slice form) | has | crates/arraykit/src/array/ndarray.rs |
| `transpose` | `numpy.transpose` | has | crates/arraykit/src/array/ndarray.rs |
| `transpose_axes` | `numpy.transpose` (axes) | has | crates/arraykit/src/array/ndarray.rs |
| `broadcast_to` | `numpy.broadcast_to` | has | crates/arraykit/src/array/ndarray.rs |
| `add` | `numpy.add` | has | crates/arraykit/src/array/ops.rs |
| `sub` | `numpy.subtract` | has | crates/arraykit/src/array/ops.rs |
| `mul` | `numpy.multiply` | has | crates/arraykit/src/array/ops.rs |
| `div` | `numpy.divide` | partial; integer division for `Int32` and `Int64` (truncates; a zero divisor panics) | crates/arraykit/src/array/ops.rs |
| `add_scalar` | `numpy.add` (scalar) | has | crates/arraykit/src/array/ops.rs |
| `mul_scalar` | `numpy.multiply` (scalar) | has | crates/arraykit/src/array/ops.rs |
| operators `+ - *` on `&NdArray` | `numpy` operators | has | crates/arraykit/src/array/ops.rs |
| operator `/` on `&NdArray` | `numpy` operator `/` | partial; integer division for `Int32` and `Int64` (truncates; a zero divisor panics) | crates/arraykit/src/array/ops.rs |
| `linspace` | `numpy.linspace` | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `sqrt` | `numpy.sqrt` | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `exp` | `numpy.exp` | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `log` | `numpy.log` | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `abs` | `numpy.abs` | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `abs_int` | `numpy.abs` (integer) | partial; only `i32` and `i64` | crates/arraykit/src/array/math.rs |
| `clip` | `numpy.clip` | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `where_cond` | `numpy.where` (two arrays) | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `pow` | `numpy.power` (scalar exponent) | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `sin` | `numpy.sin` | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `cos` | `numpy.cos` | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `tan` | `numpy.tan` | partial; `f32` and `f64` only | crates/arraykit/src/array/math.rs |
| `sum` | `numpy.sum` | has | crates/arraykit/src/array/stats.rs |
| `checked_sum` | `numpy.sum` (overflow checked) | has | crates/arraykit/src/array/stats.rs |
| `sum_axis` | `numpy.sum` (axis) | has | crates/arraykit/src/array/stats.rs |
| `mean` | `numpy.mean` | partial; integer division for `Int32` and `Int64` (use `mean_f64`) | crates/arraykit/src/array/stats.rs |
| `mean_f64` | `numpy.mean` (float result) | has | crates/arraykit/src/array/stats.rs |
| `mean_axis` | `numpy.mean` (axis) | partial; integer division for `Int32` and `Int64` | crates/arraykit/src/array/stats.rs |
| `min` | `numpy.min` | has | crates/arraykit/src/array/stats.rs |
| `max` | `numpy.max` | has | crates/arraykit/src/array/stats.rs |
| `min_axis` | `numpy.min` (axis) | has | crates/arraykit/src/array/stats.rs |
| `max_axis` | `numpy.max` (axis) | has | crates/arraykit/src/array/stats.rs |
| `var` | `numpy.var` (ddof=1) | partial; f32 and f64 only | crates/arraykit/src/array/stats.rs |
| `var_pop` | `numpy.var` (ddof=0) | partial; f32 and f64 only | crates/arraykit/src/array/stats.rs |
| `std` | `numpy.std` (ddof=1) | partial; f32 and f64 only | crates/arraykit/src/array/stats.rs |
| `std_pop` | `numpy.std` (ddof=0) | partial; f32 and f64 only | crates/arraykit/src/array/stats.rs |
| `var_axis` | `numpy.var` (axis, ddof=1) | partial; f32 and f64 only; an axis of length 1 gives 0 (NumPy gives nan) | crates/arraykit/src/array/stats.rs |
| `std_axis` | `numpy.std` (axis, ddof=1) | partial; f32 and f64 only; an axis of length 1 gives 0 (NumPy gives nan) | crates/arraykit/src/array/stats.rs |
| `argmin` | `numpy.argmin` | has | crates/arraykit/src/array/stats.rs |
| `argmax` | `numpy.argmax` | has | crates/arraykit/src/array/stats.rs |
| `cumsum` | `numpy.cumsum` | has | crates/arraykit/src/array/stats.rs |
| `cumprod` | `numpy.cumprod` | has | crates/arraykit/src/array/stats.rs |
| `median` | `numpy.median` | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `mode` | `scipy.stats.mode` (SciPy) | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `skew` | `scipy.stats.skew` (SciPy) | partial; `f64` only; bias-corrected only (SciPy default is `bias=True`) | crates/arraykit/src/array/stats.rs |
| `kurtosis` | `scipy.stats.kurtosis` (SciPy) | partial; `f64` only; bias-corrected only (SciPy default is `bias=True`) | crates/arraykit/src/array/stats.rs |
| `zscore` | `scipy.stats.zscore` (SciPy) | partial; `f64` only; `ddof=1` only (SciPy default is `ddof=0`) | crates/arraykit/src/array/stats.rs |
| `moment` | `scipy.stats.moment` (SciPy) | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `iqr` | `scipy.stats.iqr` (SciPy) | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `geometric_mean` | `scipy.stats.gmean` (SciPy) | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `harmonic_mean` | `scipy.stats.hmean` (SciPy) | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `trim_mean` | `scipy.stats.trim_mean` (SciPy) | partial; `f64` only; over-trim returns the median (SciPy raises) | crates/arraykit/src/array/stats.rs |
| `sem` | `scipy.stats.sem` (SciPy) | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `coefficient_of_variation` | `scipy.stats.variation` (SciPy) | partial; `f64` only; `ddof=1` and divides by `abs(mean)` (SciPy default is `ddof=0`, divides by `mean`) | crates/arraykit/src/array/stats.rs |
| `percentile` | `numpy.percentile` | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `quantile` | `numpy.quantile` | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `corrcoef` | `numpy.corrcoef` | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `cov` | `numpy.cov` | partial; `f64` only | crates/arraykit/src/array/stats.rs |
| `histogram` | `numpy.histogram` | partial; `f64` only; constant input gives one bin (NumPy gives `bins` bins) | crates/arraykit/src/array/stats.rs |
| `matmul` | `numpy.matmul` | has | crates/arraykit/src/array/linalg.rs |
| `dot` | `numpy.dot` | has | crates/arraykit/src/array/linalg.rs |
| `outer` | `numpy.outer` | has | crates/arraykit/src/array/linalg.rs |
| `inner` | `numpy.inner` | has | crates/arraykit/src/array/linalg.rs |
| `trace` | `numpy.trace` | has | crates/arraykit/src/array/linalg.rs |
| `det` | `numpy.linalg.det` | partial; `f64` only | crates/arraykit/src/array/linalg.rs |
| `inv` | `numpy.linalg.inv` | partial; `f64` only | crates/arraykit/src/array/linalg.rs |
| `solve` | `numpy.linalg.solve` | partial; `f64` only | crates/arraykit/src/array/linalg.rs |
| `norm` | `numpy.linalg.norm` | partial; `f64` only | crates/arraykit/src/array/linalg.rs |
| `qr` | `numpy.linalg.qr` | partial; `f64` only | crates/arraykit/src/array/linalg_decomp.rs |
| `cholesky` | `numpy.linalg.cholesky` | partial; `f64` only | crates/arraykit/src/array/linalg_decomp.rs |
| `eig` | `numpy.linalg.eig` | partial; `f64` only; complex eigenvalues return an error | crates/arraykit/src/array/linalg_decomp.rs |
| `eigh` | `numpy.linalg.eigh` | partial; `f64` only | crates/arraykit/src/array/linalg_decomp.rs |
| `svd` | `numpy.linalg.svd` | partial; `f64` only | crates/arraykit/src/array/linalg_decomp.rs |
| `lstsq` | `numpy.linalg.lstsq` | partial; `f64` only; `self` must be 2-D and `b` 1-D | crates/arraykit/src/array/linalg_decomp.rs |
| `QrResult` | result of `numpy.linalg.qr` | has | crates/arraykit/src/array/linalg_decomp.rs |
| `EigResult` | result of `numpy.linalg.eig` | has | crates/arraykit/src/array/linalg_decomp.rs |
| `SvdResult` | result of `numpy.linalg.svd` | has | crates/arraykit/src/array/linalg_decomp.rs |
| `LstsqResult` | result of `numpy.linalg.lstsq` | has | crates/arraykit/src/array/linalg_decomp.rs |
| `concatenate` | `numpy.concatenate` | has | crates/arraykit/src/array/manip.rs |
| `stack` | `numpy.stack` | has | crates/arraykit/src/array/manip.rs |
| `vstack` | `numpy.vstack` | has | crates/arraykit/src/array/manip.rs |
| `hstack` | `numpy.hstack` | has | crates/arraykit/src/array/manip.rs |
| `split` | `numpy.split` | has | crates/arraykit/src/array/manip.rs |
| `repeat` | `numpy.repeat` | has | crates/arraykit/src/array/manip.rs |
| `tile` | `numpy.tile` | has | crates/arraykit/src/array/manip.rs |
| `unique` | `numpy.unique` | partial; integer and bool only (no `f32`, `f64`); keeps first-seen order, not sorted | crates/arraykit/src/array/manip.rs |
| `sort` | `numpy.sort` | partial; integer and bool only (no `f32`, `f64`; use `sort_axis` for `f64`) | crates/arraykit/src/array/manip.rs |
| `argsort` | `numpy.argsort` | partial; NaN is not sorted last | crates/arraykit/src/array/manip.rs |
| `all` | `numpy.all` | partial; `bool` only | crates/arraykit/src/array/manip.rs |
| `any` | `numpy.any` | partial; `bool` only | crates/arraykit/src/array/manip.rs |
| `allclose` | `numpy.allclose` | has | crates/arraykit/src/array/manip.rs |
| `fancy_index` | `ndarray[int array]` | has | crates/arraykit/src/array/indexing.rs |
| `bool_index` | `ndarray[bool mask]` | has | crates/arraykit/src/array/indexing.rs |
| `bool_index_array` | `ndarray[bool array]` | has | crates/arraykit/src/array/indexing.rs |
| `bool_set` | `ndarray[bool mask] = value` | has | crates/arraykit/src/array/indexing.rs |
| `fancy_set` | `ndarray[int array] = values` | has | crates/arraykit/src/array/indexing.rs |
| `advanced_index` | `ndarray[mixed index]` | has | crates/arraykit/src/array/indexing.rs |
| `take` | `numpy.take` | has | crates/arraykit/src/array/indexing.rs |
| `IndexMode` | `numpy.take` (`mode=`) | has | crates/arraykit/src/array/indexing.rs |
| `AxisIndex` | index element of `advanced_index` | has | crates/arraykit/src/array/indexing.rs |
| `searchsorted` | `numpy.searchsorted` | partial; `f64` only; `side` is a `&str` | crates/arraykit/src/array/search.rs |
| `sort_axis` | `numpy.sort` (axis) | partial; `f64` only | crates/arraykit/src/array/search.rs |
| `argsort_axis` | `numpy.argsort` (axis) | partial; `f64` only | crates/arraykit/src/array/search.rs |
| `unique_full` | `numpy.unique` (with counts and indices) | partial; `f64` only | crates/arraykit/src/array/search.rs |
| `where_select` | `numpy.where` (three arguments) | partial; `f64` only | crates/arraykit/src/array/search.rs |
| `nonzero` | `numpy.nonzero` | has | crates/arraykit/src/array/search.rs |
| `UniqueResult` | result of `numpy.unique` | has | crates/arraykit/src/array/search.rs |
| `Rolling::new` | `pandas.DataFrame.rolling` | has | crates/arraykit/src/array/window.rs |
| `Rolling::min_periods` | `pandas.DataFrame.rolling` (`min_periods`) | has | crates/arraykit/src/array/window.rs |
| `Rolling::sum` | `pandas.DataFrame.rolling().sum` | partial; `0` before `min_periods` (pandas gives NaN) | crates/arraykit/src/array/window.rs |
| `Rolling::count` | `pandas.DataFrame.rolling().count` | partial; returns `Vec<usize>`; ignores `min_periods`; counts NaN as a value | crates/arraykit/src/array/window.rs |
| `Rolling::mean` | `pandas.DataFrame.rolling().mean` | partial; integer division for `Int32` and `Int64`; `0` before `min_periods` (pandas gives NaN) | crates/arraykit/src/array/window.rs |
| `Rolling::min` | `pandas.DataFrame.rolling().min` | partial; `f64` only | crates/arraykit/src/array/window.rs |
| `Rolling::max` | `pandas.DataFrame.rolling().max` | partial; `f64` only | crates/arraykit/src/array/window.rs |
| `Rolling::std` | `pandas.DataFrame.rolling().std` | partial; `f64` only | crates/arraykit/src/array/window.rs |
| `Rolling::var` | `pandas.DataFrame.rolling().var` | partial; `f64` only | crates/arraykit/src/array/window.rs |
| `rolling` | `pandas.DataFrame.rolling` | has | crates/arraykit/src/array/window.rs |
| `squeeze` | `numpy.squeeze` | has | crates/arraykit/src/array/window.rs |
| `expand_dims` | `numpy.expand_dims` | has | crates/arraykit/src/array/window.rs |
| `flip` | `numpy.flip` | has | crates/arraykit/src/array/window.rs |
| `roll` | `numpy.roll` | has | crates/arraykit/src/array/window.rs |
| `pad` | `numpy.pad` | has | crates/arraykit/src/array/window.rs |
| `diff` | `numpy.diff` | partial; `f64` only | crates/arraykit/src/array/window.rs |
| `interp` | `numpy.interp` | partial; `f64` only | crates/arraykit/src/array/window.rs |
| `gradient` | `numpy.gradient` | partial; `f64` only | crates/arraykit/src/array/window.rs |
| `rot90` | `numpy.rot90` | has | crates/arraykit/src/array/window.rs |
| `KitRng::new` | `numpy.random.Generator` (seed) | has | crates/arraykit/src/array/random.rs |
| `with_stream` | `numpy.random.Generator` (stream) | has | crates/arraykit/src/array/random.rs |
| `next_u64` | `numpy.random.Generator.integers` (raw bits) | has | crates/arraykit/src/array/random.rs |
| `next_f64` | `numpy.random.Generator.random` | has | crates/arraykit/src/array/random.rs |
| `uniform` | `numpy.random.Generator.uniform` | has | crates/arraykit/src/array/random.rs |
| `normal_standard` | `numpy.random.Generator.standard_normal` | has | crates/arraykit/src/array/random.rs |
| `normal` | `numpy.random.Generator.normal` | has | crates/arraykit/src/array/random.rs |
| `randint` | `numpy.random.Generator.integers` | has | crates/arraykit/src/array/random.rs |
| `bernoulli` | `numpy.random.Generator.random < p` | has | crates/arraykit/src/array/random.rs |
| `binomial` | `numpy.random.Generator.binomial` | has | crates/arraykit/src/array/random.rs |
| `poisson` | `numpy.random.Generator.poisson` | has | crates/arraykit/src/array/random.rs |
| `choice` | `numpy.random.Generator.choice` (index) | has | crates/arraykit/src/array/random.rs |
| `shuffle` | `numpy.random.Generator.shuffle` | has | crates/arraykit/src/array/random.rs |
| `RandomExt` (`rand_uniform`, `rand_normal`, `rand_int`, `rand_binomial`, `rand_poisson`) | `numpy.random.Generator` (array draws) | partial; `f64` result only | crates/arraykit/src/array/random.rs |
| `Shape::new` | `numpy.ndarray.shape` | has | crates/arraykit/src/array/shape.rs |
| `with_strides` | `numpy.ndarray.strides` | has | crates/arraykit/src/array/shape.rs |
| `ndim` | `numpy.ndarray.ndim` | has | crates/arraykit/src/array/shape.rs |
| `dims` | `numpy.ndarray.shape` | has | crates/arraykit/src/array/shape.rs |
| `strides` | `numpy.ndarray.strides` | has | crates/arraykit/src/array/shape.rs |
| `size` | `numpy.ndarray.size` | has | crates/arraykit/src/array/shape.rs |
| `contains` | index bounds check | has | crates/arraykit/src/array/shape.rs |
| `offset` | flat offset of an index | has | crates/arraykit/src/array/shape.rs |
| `broadcast` | `numpy.broadcast_shapes` | has | crates/arraykit/src/array/shape.rs |
| `broadcast_strides` | `numpy.broadcast_to` (strides) | has | crates/arraykit/src/array/shape.rs |
| `AxisSlice` | slice object of `numpy.s_` | has | crates/arraykit/src/array/slice.rs |
| `AxisSlice::range` | `slice(start, stop)` | has | crates/arraykit/src/array/slice.rs |
| `range_step` | `slice(start, stop, step)` | has | crates/arraykit/src/array/slice.rs |
| `normalize` | slice index normalization | has | crates/arraykit/src/array/slice.rs |
| `NormalizedSlice` | normalized slice object | has | crates/arraykit/src/array/slice.rs |
| `SliceInfo` | tuple of slice objects | has | crates/arraykit/src/array/slice.rs |
| `SliceInfo::new` | tuple of slice objects | has | crates/arraykit/src/array/slice.rs |
| `slices` | tuple of slice objects | has | crates/arraykit/src/array/slice.rs |
| `output_shape` | shape after slicing | has | crates/arraykit/src/array/slice.rs |
| `normalize_all` | slice index normalization (all axes) | has | crates/arraykit/src/array/slice.rs |
| `numpy_str` | `numpy.dtype.str` | has | crates/arraykit/src/array/dtype.rs |
| `DType` | `numpy.dtype` | partial; only `Float32`, `Float64`, `Int32`, `Int64`, `Bool` | crates/arraykit/src/array/dtype.rs |
| `ArrayElement` | element trait | has | crates/arraykit/src/array/dtype.rs |
| `RangeElement` | element trait for `arange` | has | crates/arraykit/src/array/dtype.rs |
| `ArrayError` | `numpy` exceptions | has | crates/arraykit/src/array/error.rs |
| `Result` | result alias | has | crates/arraykit/src/array/error.rs |
| `NdArray` | `numpy.ndarray` | has | crates/arraykit/src/array/ndarray.rs |
| `Shape` | `numpy.ndarray.shape` | has | crates/arraykit/src/array/shape.rs |
| `KitRng` | `numpy.random.Generator` | has | crates/arraykit/src/array/random.rs |
| `StatOps` | - (element trait for statistics) | has | crates/arraykit/src/array/stats.rs |
| `VarOps` | - (element trait for variance) | has | crates/arraykit/src/array/stats.rs |
| `NumericOps` | - (element trait for arithmetic) | has | crates/arraykit/src/array/ops.rs |
| `LinalgOps` | - (element trait for linear algebra) | has | crates/arraykit/src/array/linalg.rs |
| `FloatOps` | - (element trait for float math) | has | crates/arraykit/src/array/math.rs |

Random functions match NumPy in distribution, not in the drawn values for a seed.

## Table 2: NumPy functions that arraykit lacks

| NumPy function | State | Later issue title |
|---|---|---|
| `einsum` | missing | arraykit: add einsum |
| `fft` | missing in arraykit (`see scikit`) | arraykit: add fft |
| `meshgrid` | missing | arraykit: add meshgrid |
| `cross` | missing | arraykit: add cross |
| `kron` | missing | arraykit: add kron |
| `tril` | missing | arraykit: add tril |
| `triu` | missing | arraykit: add triu |
| `argpartition` | missing | arraykit: add argpartition |
| `partition` | missing | arraykit: add partition |
| `maximum.accumulate` (cumulative max) | missing | arraykit: add cummax |
| `isnan` | missing | arraykit: add isnan |
| `isfinite` | missing | arraykit: add isfinite |
| `nansum` | missing | arraykit: add nansum |
| `nanmean` | missing | arraykit: add nanmean |
| `linalg.matrix_power` | missing | arraykit: add matrix_power |
| `linalg.pinv` | missing | arraykit: add pinv |
| `linalg.eigvals` | missing | arraykit: add eigvals |
| `lexsort` | missing | arraykit: add lexsort |
| `bincount` | missing | arraykit: add bincount |
| `digitize` | missing | arraykit: add digitize |
| `convolve` | missing in arraykit (`see scikit`) | arraykit: add convolve |
| `correlate` | missing in arraykit (`see scikit`) | arraykit: add correlate |
| `linalg.matrix_rank` | missing | arraykit: add matrix_rank |
| `linalg.cond` | missing | arraykit: add cond |

Tables are maintained by hand; change a row in the same pull request that changes the code.
