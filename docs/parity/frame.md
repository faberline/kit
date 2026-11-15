# frame parity with pandas

The reference library is pandas. Each row is one public item of the `frame` crate (`pub fn`, reachable from the crate root).
The states are `has` (the same result as pandas for the cases the tests cover), `partial` (part only; the gap follows the path), and `missing` (Table 2).

Statistics skip Null; see the null policy in crates/frame/src/frame/mod.rs

A pandas name of `none` means a kit helper with no pandas name; its state is `has`.

## Table 1: public items

## DataFrame

| Function | pandas name | State | Evidence |
|---|---|---|---|
| DataFrame::new | DataFrame() | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::from_columns | DataFrame(dict) | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::from_map | DataFrame(dict of lists) | partial | crates/frame/src/frame/dataframe.rs; column order is the HashMap order |
| DataFrame::shape | DataFrame.shape | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::nrows | DataFrame.shape[0] | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::ncols | DataFrame.shape[1] | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::is_empty | DataFrame.empty | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::columns | DataFrame.columns | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::index | DataFrame.index | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::set_index | DataFrame.set_index | partial | crates/frame/src/frame/dataframe.rs; takes an Index, not a column name |
| DataFrame::get | DataFrame[name] | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::col | DataFrame[name] | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::get_columns | DataFrame[list] | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::select | DataFrame.filter(items=) | partial | crates/frame/src/frame/dataframe.rs; no like or regex argument |
| DataFrame::insert | DataFrame.insert | partial | crates/frame/src/frame/dataframe.rs; no loc argument; an existing name is replaced |
| DataFrame::drop | DataFrame.pop | partial | crates/frame/src/frame/dataframe.rs; removes and returns one column; no axis argument |
| DataFrame::rename | DataFrame.rename(columns=) | partial | crates/frame/src/frame/dataframe.rs; column names only |
| DataFrame::iloc_row | DataFrame.iloc[int] | partial | crates/frame/src/frame/dataframe.rs; returns a map, not a Series |
| DataFrame::iloc | DataFrame.iloc[list] | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::loc_row | DataFrame.loc[label] | partial | crates/frame/src/frame/dataframe.rs; returns a map, not a Series |
| DataFrame::loc | DataFrame.loc[list] | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::filter | DataFrame.loc[bool mask] | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::query | DataFrame.query | partial | crates/frame/src/frame/dataframe.rs; takes a closure, not an expression string |
| DataFrame::describe | DataFrame.describe | partial | crates/frame/src/frame/dataframe.rs; numeric columns only; no percentiles argument |
| DataFrame::sort_values | DataFrame.sort_values | partial | crates/frame/src/frame/dataframe.rs; one column; no na_position argument; nulls always last |
| DataFrame::sort_values_by | DataFrame.sort_values(by=, ascending=) | partial | crates/frame/src/frame/dataframe.rs; nulls always last; no na_position argument |
| DataFrame::head | DataFrame.head | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::tail | DataFrame.tail | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::reset_index | DataFrame.reset_index | partial | crates/frame/src/frame/dataframe.rs; drops the old index; it is not kept as a column |
| DataFrame::iterrows | DataFrame.iterrows | partial | crates/frame/src/frame/dataframe.rs; yields a map for each row, not a Series |
| DataFrame::isna | DataFrame.isna | partial | crates/frame/src/frame/dataframe.rs; returns a map of column to bool list, not a DataFrame |
| DataFrame::isna_any | DataFrame.isna().any(axis=1) | partial | crates/frame/src/frame/dataframe.rs; returns one bool per row |
| DataFrame::isna_all | DataFrame.isna().all(axis=1) | partial | crates/frame/src/frame/dataframe.rs; returns one bool per row |
| DataFrame::fillna | DataFrame.fillna(value) | partial | crates/frame/src/frame/dataframe.rs; scalar value only; no method argument |
| DataFrame::fillna_column | DataFrame[col].fillna(value) | partial | crates/frame/src/frame/dataframe.rs; one column; scalar value only |
| DataFrame::dropna | DataFrame.dropna() | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::dropna_all | DataFrame.dropna(how='all') | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::dropna_subset | DataFrame.dropna(subset=) | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::apply | DataFrame.apply | partial | crates/frame/src/frame/dataframe.rs; returns a map of results, not a DataFrame; no axis argument |
| DataFrame::applymap | DataFrame.map | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::concat | pd.concat(axis=0) | partial | crates/frame/src/frame/dataframe.rs; vertical only; columns must match in order; no ignore_index |
| DataFrame::from_records | DataFrame.from_records | partial | crates/frame/src/frame/dataframe.rs; column names sorted; a missing value is Null; no index argument |
| DataFrame::to_records | DataFrame.to_dict(orient='records') | partial | crates/frame/src/frame/dataframe.rs; returns maps, not ordered records |
| DataFrame::to_dict | DataFrame.to_dict(orient='list') | has | crates/frame/src/frame/dataframe.rs |
| DataFrame::duplicated | DataFrame.duplicated | partial | crates/frame/src/frame/dataframe.rs; keep is a string; any value other than first or last marks every copy |
| DataFrame::drop_duplicates | DataFrame.drop_duplicates | partial | crates/frame/src/frame/dataframe.rs; keep is a string; any value other than first or last drops every copy |
| DataFrame::corr | DataFrame.corr | partial | crates/frame/src/frame/dataframe.rs; Pearson only; a pair without variance gives 0.0, not NaN |
| DataFrame::pivot | DataFrame.pivot | partial | crates/frame/src/frame/dataframe.rs; no aggregation; a repeated index and column pair keeps the last value |
| DataFrame::melt | DataFrame.melt | partial | crates/frame/src/frame/dataframe.rs; fixed names variable and value; no var_name or value_name argument |
| DataFrame::sample | DataFrame.sample | partial | crates/frame/src/frame/dataframe.rs; fixed seed 42; no replace, frac or weights argument |
| DataFrame::sample_seeded | DataFrame.sample(random_state=) | partial | crates/frame/src/frame/dataframe.rs; no replace, frac or weights argument |
| DataFrame::assign | DataFrame.assign | partial | crates/frame/src/frame/dataframe.rs; takes Series, not callables; replaces existing columns |

## Index

| Function | pandas name | State | Evidence |
|---|---|---|---|
| Index::range | RangeIndex | partial | crates/frame/src/frame/index.rs; start is always 0; no step argument |
| Index::int | Index(int64) | has | crates/frame/src/frame/index.rs |
| Index::string | Index(object) | has | crates/frame/src/frame/index.rs |
| Index::len | len(Index) | has | crates/frame/src/frame/index.rs |
| Index::is_empty | Index.empty | has | crates/frame/src/frame/index.rs |
| Index::get_loc | Index.get_loc | partial | crates/frame/src/frame/index.rs; one label only; no slice or bool mask |
| Index::get_label | Index[pos] | has | crates/frame/src/frame/index.rs |
| Index::to_values | Index.tolist | has | crates/frame/src/frame/index.rs |
| Index::to_map | none (kit helper) | has | crates/frame/src/frame/index.rs |
| Index::slice | Index.take | partial | crates/frame/src/frame/index.rs; positions only; a range index becomes an integer index |
| Index::reset | RangeIndex(len) | partial | crates/frame/src/frame/index.rs; returns a range index of the same length |

## MultiIndex

| Function | pandas name | State | Evidence |
|---|---|---|---|
| MultiIndex::new | MultiIndex.from_arrays | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::nlevels | MultiIndex.nlevels | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::len | len(MultiIndex) | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::is_empty | MultiIndex.empty | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::names | MultiIndex.names | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::get_level_values | MultiIndex.get_level_values | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::get_level_index | MultiIndex._get_level_number | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::get_level_values_by_name | MultiIndex.get_level_values(name) | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::get_tuple | MultiIndex[pos] | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::get_locs | MultiIndex.get_locs | partial | crates/frame/src/frame/multi_index.rs; prefix match only; no slice or list |
| MultiIndex::select_level | MultiIndex.xs(value, level=) | partial | crates/frame/src/frame/multi_index.rs; returns row positions, not a frame |
| MultiIndex::swaplevel | MultiIndex.swaplevel | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::droplevel | MultiIndex.droplevel | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::reorder_levels | MultiIndex.reorder_levels | has | crates/frame/src/frame/multi_index.rs |
| MultiIndex::unique | MultiIndex.unique | partial | crates/frame/src/frame/multi_index.rs; returns tuples, not an index |
| MultiIndex::slice | MultiIndex.take | partial | crates/frame/src/frame/multi_index.rs; positions only |
| MultiIndex::to_group_map | MultiIndex.groupby | partial | crates/frame/src/frame/multi_index.rs; returns a map of tuples to positions |
| MultiIndex::argsort | MultiIndex.argsort | has | crates/frame/src/frame/multi_index.rs |

## Value

| Function | pandas name | State | Evidence |
|---|---|---|---|
| Value::is_null | pd.isna(value) | has | crates/frame/src/frame/value.rs |
| Value::as_int | none (kit helper) | has | crates/frame/src/frame/value.rs |
| Value::as_float | none (kit helper) | has | crates/frame/src/frame/value.rs |
| Value::as_str | none (kit helper) | has | crates/frame/src/frame/value.rs |
| Value::as_bool | none (kit helper) | has | crates/frame/src/frame/value.rs |

## Series

| Function | pandas name | State | Evidence |
|---|---|---|---|
| Series::new | pd.Series(data) | has | crates/frame/src/frame/series.rs |
| Series::with_name | pd.Series(data, name=) | has | crates/frame/src/frame/series.rs |
| Series::with_index | pd.Series(data, index=) | has | crates/frame/src/frame/series.rs |
| Series::name | Series.name | has | crates/frame/src/frame/series.rs |
| Series::set_name | Series.rename | has | crates/frame/src/frame/series.rs |
| Series::len | len(Series) | has | crates/frame/src/frame/series.rs |
| Series::is_empty | Series.empty | has | crates/frame/src/frame/series.rs |
| Series::index | Series.index | has | crates/frame/src/frame/series.rs |
| Series::iloc | Series.iloc[int] | has | crates/frame/src/frame/series.rs |
| Series::loc | Series.loc[label] | has | crates/frame/src/frame/series.rs |
| Series::iloc_many | Series.iloc[list] | has | crates/frame/src/frame/series.rs |
| Series::values | Series.values | has | crates/frame/src/frame/series.rs |
| Series::to_f64 | Series.astype(float) | partial | crates/frame/src/frame/series.rs; fails on Null and on text |
| Series::to_i64 | Series.astype(int) | partial | crates/frame/src/frame/series.rs; fails on Null and on text |
| Series::sum | Series.sum | partial | crates/frame/src/frame/series.rs; always returns a float; skips nulls |
| Series::mean | Series.mean | has | crates/frame/src/frame/series.rs |
| Series::min | Series.min | has | crates/frame/src/frame/series.rs |
| Series::max | Series.max | has | crates/frame/src/frame/series.rs |
| Series::count | Series.count | has | crates/frame/src/frame/series.rs |
| Series::filter | Series[bool mask] | has | crates/frame/src/frame/series.rs |
| Series::map | Series.map(func) | has | crates/frame/src/frame/series.rs |
| Series::reset_index | Series.reset_index | partial | crates/frame/src/frame/series.rs; drops the old index |
| Series::sort | Series.sort_values | partial | crates/frame/src/frame/series.rs; no na_position argument; nulls always last |
| Series::unique | Series.unique | has | crates/frame/src/frame/series.rs |
| Series::value_counts | Series.value_counts | partial | crates/frame/src/frame/series.rs; returns a map, not a sorted Series |
| Series::isna | Series.isna | partial | crates/frame/src/frame/series.rs; returns a bool list, not a Series |
| Series::notna | Series.notna | partial | crates/frame/src/frame/series.rs; returns a bool list, not a Series |
| Series::fillna | Series.fillna | partial | crates/frame/src/frame/series.rs; scalar value only; no method argument |
| Series::ffill | Series.ffill | has | crates/frame/src/frame/series.rs |
| Series::bfill | Series.bfill | has | crates/frame/src/frame/series.rs |
| Series::dropna | Series.dropna | has | crates/frame/src/frame/series.rs |
| Series::interpolate | Series.interpolate | partial | crates/frame/src/frame/series.rs; linear by default; no limit or direction argument |
| Series::interpolate_method | Series.interpolate(method=) | partial | crates/frame/src/frame/series.rs; methods linear, nearest, ffill and bfill only |
| Series::null_count | Series.isna().sum() | has | crates/frame/src/frame/series.rs |
| Series::has_nulls | Series.isna().any() | has | crates/frame/src/frame/series.rs |
| Series::apply | Series.apply | has | crates/frame/src/frame/series.rs |
| Series::cumsum | Series.cumsum | has | crates/frame/src/frame/series.rs |
| Series::cumprod | Series.cumprod | has | crates/frame/src/frame/series.rs |
| Series::cummax | Series.cummax | has | crates/frame/src/frame/series.rs |
| Series::cummin | Series.cummin | has | crates/frame/src/frame/series.rs |
| Series::shift | Series.shift | has | crates/frame/src/frame/series.rs |
| Series::diff | Series.diff | partial | crates/frame/src/frame/series.rs; periods must be 0 or more |
| Series::pct_change | Series.pct_change | partial | crates/frame/src/frame/series.rs; periods must be 0 or more; a zero previous value gives Null |
| Series::nlargest | Series.nlargest | has | crates/frame/src/frame/series.rs |
| Series::nsmallest | Series.nsmallest | has | crates/frame/src/frame/series.rs |
| Series::clip | Series.clip | partial | crates/frame/src/frame/series.rs; numeric values only; other values and Null are kept |
| Series::std | Series.std | partial | crates/frame/src/frame/series.rs; returns an error below 2 values; pandas gives NaN |
| Series::var | Series.var | partial | crates/frame/src/frame/series.rs; returns an error below 2 values; pandas gives NaN |
| Series::median | Series.median | has | crates/frame/src/frame/series.rs |
| Series::rank | Series.rank | partial | crates/frame/src/frame/series.rs; average method and ascending order only; no na_option |
| Series::duplicated | Series.duplicated | partial | crates/frame/src/frame/series.rs; keep is a string; any value other than first or last marks every copy |
| Series::drop_duplicates | Series.drop_duplicates | partial | crates/frame/src/frame/series.rs; keep is a string; any value other than first or last drops every copy |
| Series::isin | Series.isin | has | crates/frame/src/frame/series.rs |
| Series::between | Series.between | has | crates/frame/src/frame/series.rs |
| Series::replace | Series.replace(a, b) | has | crates/frame/src/frame/series.rs |
| Series::replace_map | Series.replace(dict) | has | crates/frame/src/frame/series.rs |
| Series::to_dict | Series.to_dict | has | crates/frame/src/frame/series.rs |
| Series::quantile | Series.quantile | partial | crates/frame/src/frame/series.rs; linear interpolation only |
| Series::corr | Series.corr | partial | crates/frame/src/frame/series.rs; returns 0.0 when a series has no variance; pandas gives NaN |
| Series::rolling | Series.rolling | partial | crates/frame/src/frame/series.rs; no center or closed argument |
| Series::astype | Series.astype | partial | crates/frame/src/frame/series.rs; dtype names int, float, str and bool only; another name keeps the data |
| Series::where_cond | Series.where | partial | crates/frame/src/frame/series.rs; scalar other only; no callable |
| Series::mask | Series.mask | partial | crates/frame/src/frame/series.rs; scalar other only; no callable |
| Series::abs | Series.abs | has | crates/frame/src/frame/series.rs |
| Series::round | Series.round | has | crates/frame/src/frame/series.rs |

## SeriesRolling

| Function | pandas name | State | Evidence |
|---|---|---|---|
| SeriesRolling::min_periods | Series.rolling(min_periods=) | has | crates/frame/src/frame/series.rs |
| SeriesRolling::sum | Series.rolling().sum | has | crates/frame/src/frame/series.rs |
| SeriesRolling::mean | Series.rolling().mean | has | crates/frame/src/frame/series.rs |
| SeriesRolling::min | Series.rolling().min | has | crates/frame/src/frame/series.rs |
| SeriesRolling::max | Series.rolling().max | has | crates/frame/src/frame/series.rs |
| SeriesRolling::std | Series.rolling().std | has | crates/frame/src/frame/series.rs |
| SeriesRolling::var | Series.rolling().var | has | crates/frame/src/frame/series.rs |

## GroupBy

| Function | pandas name | State | Evidence |
|---|---|---|---|
| GroupBy::new | DataFrame.groupby (object) | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::ngroups | GroupBy.ngroups | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::groups | GroupBy.groups | partial | crates/frame/src/frame/ops/groupby.rs; returns row positions per key, not labels |
| GroupBy::sum | GroupBy.sum | partial | crates/frame/src/frame/ops/groupby.rs; float result; non-numeric values are skipped |
| GroupBy::mean | GroupBy.mean | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::count | GroupBy.count | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::min | GroupBy.min | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::max | GroupBy.max | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::first | GroupBy.first | partial | crates/frame/src/frame/ops/groupby.rs; takes the first row even when its value is Null |
| GroupBy::last | GroupBy.last | partial | crates/frame/src/frame/ops/groupby.rs; takes the last row even when its value is Null |
| GroupBy::aggregate | GroupBy.agg(func) | partial | crates/frame/src/frame/ops/groupby.rs; one closure per group and column; no named aggregations |
| GroupBy::transform | GroupBy.transform | partial | crates/frame/src/frame/ops/groupby.rs; closure returns an optional value per row |
| GroupBy::filter_groups | GroupBy.filter | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::var | GroupBy.var | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::std | GroupBy.std | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::median | GroupBy.median | has | crates/frame/src/frame/ops/groupby.rs |
| GroupBy::agg | GroupBy.agg(dict) | partial | crates/frame/src/frame/ops/groupby.rs; nine names only: sum, mean, count, min, max, first, last, std, median |
| DataFrame::groupby | DataFrame.groupby | has | crates/frame/src/frame/ops/groupby.rs |

## Joins

| Function | pandas name | State | Evidence |
|---|---|---|---|
| join | pd.merge(how=) | partial | crates/frame/src/frame/ops/join.rs; all non-key right columns get the suffix _right |
| DataFrame::join | DataFrame.join(how=) | partial | crates/frame/src/frame/ops/join.rs; on only; all non-key right columns get the suffix _right |
| DataFrame::merge | DataFrame.merge(how='inner') | partial | crates/frame/src/frame/ops/join.rs; on only; all non-key right columns get the suffix _right |
| DataFrame::left_join | DataFrame.merge(how='left') | partial | crates/frame/src/frame/ops/join.rs; on only; all non-key right columns get the suffix _right |
| DataFrame::right_join | DataFrame.merge(how='right') | partial | crates/frame/src/frame/ops/join.rs; on only; all non-key right columns get the suffix _right |
| DataFrame::outer_join | DataFrame.merge(how='outer') | partial | crates/frame/src/frame/ops/join.rs; on only; all non-key right columns get the suffix _right |

## Reshape and AggFunc

| Function | pandas name | State | Evidence |
|---|---|---|---|
| reshape::pivot_table | pd.pivot_table | partial | crates/frame/src/frame/ops/reshape.rs; one aggregate function; numeric values only; no margins or fill_value |
| reshape::stack | DataFrame.stack | partial | crates/frame/src/frame/ops/reshape.rs; melts to long rows; no MultiIndex result |
| reshape::unstack | DataFrame.unstack | partial | crates/frame/src/frame/ops/reshape.rs; calls pivot; one value column |
| reshape::wide_to_long | pd.wide_to_long | partial | crates/frame/src/frame/ops/reshape.rs; suffixes come from the first stub only; sep is the only split |
| reshape::cross_tab | pd.crosstab | partial | crates/frame/src/frame/ops/reshape.rs; string categories give Null, not 0; counts are floats |
| DataFrame::pivot_table | DataFrame.pivot_table | partial | crates/frame/src/frame/ops/reshape.rs; one aggregate function; numeric values only; no margins or fill_value |
| DataFrame::stack | DataFrame.stack | partial | crates/frame/src/frame/ops/reshape.rs; melts to long rows; no MultiIndex result |
| DataFrame::unstack | DataFrame.unstack | partial | crates/frame/src/frame/ops/reshape.rs; calls pivot; one value column |
| DataFrame::wide_to_long | pd.wide_to_long | partial | crates/frame/src/frame/ops/reshape.rs; suffixes come from the first stub only; sep is the only split |
| DataFrame::cross_tab | pd.crosstab | partial | crates/frame/src/frame/ops/reshape.rs; string categories give Null, not 0; counts are floats |

## Expanding and Ewm

| Function | pandas name | State | Evidence |
|---|---|---|---|
| Expanding::new | Series.expanding | has | crates/frame/src/frame/ops/window.rs |
| Expanding::min_periods | Series.expanding(min_periods=) | has | crates/frame/src/frame/ops/window.rs |
| Expanding::sum | Series.expanding().sum | has | crates/frame/src/frame/ops/window.rs |
| Expanding::mean | Series.expanding().mean | has | crates/frame/src/frame/ops/window.rs |
| Expanding::min | Series.expanding().min | has | crates/frame/src/frame/ops/window.rs |
| Expanding::max | Series.expanding().max | has | crates/frame/src/frame/ops/window.rs |
| Expanding::std | Series.expanding().std | has | crates/frame/src/frame/ops/window.rs |
| Expanding::var | Series.expanding().var | has | crates/frame/src/frame/ops/window.rs |
| Ewm::with_span | Series.ewm(span=) | has | crates/frame/src/frame/ops/window.rs |
| Ewm::with_alpha | Series.ewm(alpha=) | has | crates/frame/src/frame/ops/window.rs |
| Ewm::with_halflife | Series.ewm(halflife=) | has | crates/frame/src/frame/ops/window.rs |
| Ewm::min_periods | Series.ewm(min_periods=) | has | crates/frame/src/frame/ops/window.rs |
| Ewm::adjust | Series.ewm(adjust=) | has | crates/frame/src/frame/ops/window.rs |
| Ewm::mean | Series.ewm().mean | has | crates/frame/src/frame/ops/window.rs |
| Ewm::std | Series.ewm().std | partial | crates/frame/src/frame/ops/window.rs; no bias argument |
| Ewm::var | Series.ewm().var | partial | crates/frame/src/frame/ops/window.rs; no bias argument |
| Series::expanding | Series.expanding | has | crates/frame/src/frame/ops/window.rs |
| Series::ewm_span | Series.ewm(span=) | has | crates/frame/src/frame/ops/window.rs |
| Series::ewm_alpha | Series.ewm(alpha=) | has | crates/frame/src/frame/ops/window.rs |
| Series::ewm_halflife | Series.ewm(halflife=) | has | crates/frame/src/frame/ops/window.rs |

## CSV

| Function | pandas name | State | Evidence |
|---|---|---|---|
| read_csv | pd.read_csv | partial | crates/frame/src/frame/io/csv.rs; no dtype, index_col or na_values argument |
| read_csv_with_options | pd.read_csv | partial | crates/frame/src/frame/io/csv.rs; options are delimiter, header and quote only |
| write_csv | DataFrame.to_csv(index=False) | partial | crates/frame/src/frame/io/csv.rs; no index column is written |
| write_csv_with_options | DataFrame.to_csv | partial | crates/frame/src/frame/io/csv.rs; options are delimiter, header and quote only; no index |
| DataFrame::read_csv | pd.read_csv | partial | crates/frame/src/frame/io/csv.rs; no dtype, index_col or na_values argument |
| DataFrame::to_csv | DataFrame.to_csv(index=False) | partial | crates/frame/src/frame/io/csv.rs; no index column is written |

## JSON

| Function | pandas name | State | Evidence |
|---|---|---|---|
| read_json | pd.read_json | partial, feature io-extra | crates/frame/src/frame/io/json.rs; records or columns only; nested objects flatten with a dot |
| read_json_str | pd.read_json (string) | partial, feature io-extra | crates/frame/src/frame/io/json.rs; records or columns only; nested objects flatten with a dot |
| write_json_str | DataFrame.to_json | partial, feature io-extra | crates/frame/src/frame/io/json.rs; orient records, columns or values; no index argument |
| save_json | DataFrame.to_json (file) | partial, feature io-extra | crates/frame/src/frame/io/json.rs; orient records, columns or values; no index argument |
| DataFrame::read_json | pd.read_json | partial, feature io-extra | crates/frame/src/frame/io/json.rs; records or columns only; nested objects flatten with a dot |
| DataFrame::from_json | pd.read_json (string) | partial, feature io-extra | crates/frame/src/frame/io/json.rs; records or columns only; nested objects flatten with a dot |
| DataFrame::to_json | DataFrame.to_json | partial, feature io-extra | crates/frame/src/frame/io/json.rs; orient records, columns or values; no index argument |
| DataFrame::save_json | DataFrame.to_json (file) | partial, feature io-extra | crates/frame/src/frame/io/json.rs; orient records, columns or values; no index argument |

## Sheets

| Function | pandas name | State | Evidence |
|---|---|---|---|
| Workbook::new | pd.ExcelWriter | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| Workbook::add_sheet | ExcelWriter sheet write | partial | crates/frame/src/frame/io/sheets.rs; kit text format; a name that exists is replaced |
| Workbook::get_sheet | pd.read_excel(sheet_name=) | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| Workbook::nsheets | len(ExcelFile.sheet_names) | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| Workbook::sheet_names | ExcelFile.sheet_names | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| Workbook::iter | none (kit helper) | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| Workbook::save | ExcelWriter.save | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| Workbook::load | pd.ExcelFile | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| Workbook::to_string | none (kit text format) | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| Workbook::from_string | none (kit text format) | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| save_workbook | ExcelWriter.save | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |
| load_workbook | pd.ExcelFile | partial | crates/frame/src/frame/io/sheets.rs; kit text format, not xlsx |

## Columnar

| Function | pandas name | State | Evidence |
|---|---|---|---|
| write_columnar | DataFrame.to_parquet | partial | crates/frame/src/frame/io/columnar.rs; kit binary format, not Parquet |
| read_columnar | pd.read_parquet | partial | crates/frame/src/frame/io/columnar.rs; kit binary format, not Parquet |
| save_columnar | DataFrame.to_parquet (file) | partial | crates/frame/src/frame/io/columnar.rs; kit binary format, not Parquet |
| load_columnar | pd.read_parquet (file) | partial | crates/frame/src/frame/io/columnar.rs; kit binary format, not Parquet |
| DataFrame::to_columnar | DataFrame.to_parquet | partial | crates/frame/src/frame/io/columnar.rs; kit binary format, not Parquet |
| DataFrame::read_columnar | pd.read_parquet | partial | crates/frame/src/frame/io/columnar.rs; kit binary format, not Parquet |
| DataFrame::to_columnar_bytes | DataFrame.to_parquet (bytes) | partial | crates/frame/src/frame/io/columnar.rs; kit binary format, not Parquet |
| DataFrame::from_columnar_bytes | pd.read_parquet (bytes) | partial | crates/frame/src/frame/io/columnar.rs; kit binary format, not Parquet |

## Table 2: pandas features that kit lacks

| pandas feature | State | Later issue title |
|---|---|---|
| merge_asof | missing | frame: add merge_asof |
| resample | missing | frame: add resample |
| to_parquet | missing | frame: add to_parquet |
| read_parquet | missing | frame: add read_parquet |
| read_excel | missing | frame: add read_excel |
| qcut | missing | frame: add qcut |
| cut | missing | frame: add cut |
| explode | missing | frame: add explode |
| eval | missing | frame: add eval |
| to_datetime | missing | frame: add to_datetime |
| DatetimeIndex | missing | frame: add DatetimeIndex |
| Categorical | missing | frame: add Categorical |
| DataFrame.rolling | missing | frame: add DataFrame.rolling |

Tables are maintained by hand; change a row in the same pull request that changes the code.
