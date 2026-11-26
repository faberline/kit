//! DataFrame - two-dimensional labeled data structure.

use super::error::{FrameError, Result};
use super::index::Index;
use super::series::Series;
use super::value::{cmp_nulls_last, Value};
use std::collections::HashMap;

/// A two-dimensional labeled data structure.
#[derive(Debug, Clone)]
pub struct DataFrame {
    /// Column data (stored as Series).
    columns: Vec<Series>,
    /// Column names.
    column_names: Vec<String>,
    /// Row index.
    index: Index,
}

impl DataFrame {
    /// Create an empty DataFrame.
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_names: Vec::new(),
            index: Index::Range(0),
        }
    }

    /// Create a DataFrame from columns.
    pub fn from_columns(columns: Vec<(&str, Series)>) -> Result<Self> {
        if columns.is_empty() {
            return Ok(Self::new());
        }

        let first_len = columns[0].1.len();
        for (_, series) in &columns {
            if series.len() != first_len {
                return Err(FrameError::ShapeMismatch {
                    expected: first_len,
                    actual: series.len(),
                });
            }
        }

        let column_names: Vec<String> = columns.iter().map(|(n, _)| n.to_string()).collect();
        let mut cols: Vec<Series> = columns.into_iter().map(|(_, s)| s).collect();

        // Set names on series
        for (col, name) in cols.iter_mut().zip(column_names.iter()) {
            col.set_name(name.clone());
        }

        Ok(Self {
            columns: cols,
            column_names,
            index: Index::Range(first_len),
        })
    }

    /// Create a DataFrame from a HashMap of column name -> values.
    pub fn from_map<V: Into<Value> + Clone>(data: HashMap<&str, Vec<V>>) -> Result<Self> {
        let columns: Vec<_> = data
            .into_iter()
            .map(|(name, values)| (name, Series::new(values)))
            .collect();
        Self::from_columns(columns)
    }

    /// Get the shape (rows, columns).
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Get number of rows.
    pub fn nrows(&self) -> usize {
        self.index.len()
    }

    /// Get number of columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Check if DataFrame is empty.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty() || self.index.is_empty()
    }

    /// Get column names.
    pub fn columns(&self) -> &[String] {
        &self.column_names
    }

    /// Get the row index.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Set the row index.
    pub fn set_index(&mut self, index: Index) -> Result<()> {
        if index.len() != self.nrows() {
            return Err(FrameError::ShapeMismatch {
                expected: self.nrows(),
                actual: index.len(),
            });
        }
        self.index = index;
        Ok(())
    }

    /// Get a column by name.
    pub fn get(&self, name: &str) -> Result<&Series> {
        let idx = self
            .column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| FrameError::ColumnNotFound(name.to_string()))?;
        Ok(&self.columns[idx])
    }

    /// Get a column by name (alias for get).
    pub fn col(&self, name: &str) -> Result<&Series> {
        self.get(name)
    }

    /// Get multiple columns.
    pub fn get_columns(&self, names: &[&str]) -> Result<DataFrame> {
        let columns: Result<Vec<_>> = names
            .iter()
            .map(|&name| Ok((name, self.get(name)?.clone())))
            .collect();
        DataFrame::from_columns(columns?)
    }

    /// Select columns (alias for get_columns, pandas-style).
    pub fn select(&self, names: &[&str]) -> Result<DataFrame> {
        self.get_columns(names)
    }

    /// Add a column to the DataFrame.
    pub fn insert(&mut self, name: &str, series: Series) -> Result<()> {
        if series.len() != self.nrows() && !self.columns.is_empty() {
            return Err(FrameError::ShapeMismatch {
                expected: self.nrows(),
                actual: series.len(),
            });
        }

        // If this is the first column, set the index size
        if self.columns.is_empty() {
            self.index = Index::Range(series.len());
        }

        // Check if column already exists
        if let Some(idx) = self.column_names.iter().position(|n| n == name) {
            // Replace existing column
            self.columns[idx] = series;
        } else {
            // Add new column
            self.column_names.push(name.to_string());
            self.columns.push(series);
        }

        Ok(())
    }

    /// Drop a column.
    pub fn drop(&mut self, name: &str) -> Result<Series> {
        let idx = self
            .column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| FrameError::ColumnNotFound(name.to_string()))?;

        self.column_names.remove(idx);
        Ok(self.columns.remove(idx))
    }

    /// Rename columns.
    pub fn rename(&mut self, mapping: HashMap<&str, &str>) {
        for name in &mut self.column_names {
            if let Some(&new_name) = mapping.get(name.as_str()) {
                *name = new_name.to_string();
            }
        }
    }

    // === Indexing ===

    /// Get row by position (iloc).
    pub fn iloc_row(&self, pos: usize) -> Result<HashMap<String, Value>> {
        if pos >= self.nrows() {
            return Err(FrameError::IndexOutOfBounds {
                index: pos,
                length: self.nrows(),
            });
        }

        let mut row = HashMap::new();
        for (name, col) in self.column_names.iter().zip(self.columns.iter()) {
            row.insert(name.clone(), col.iloc(pos)?.clone());
        }
        Ok(row)
    }

    /// Get rows by positions (iloc).
    pub fn iloc(&self, positions: &[usize]) -> Result<DataFrame> {
        let columns: Result<Vec<_>> = self
            .column_names
            .iter()
            .zip(self.columns.iter())
            .map(|(name, col)| Ok((name.as_str(), col.iloc_many(positions)?)))
            .collect();
        DataFrame::from_columns(columns?)
    }

    /// Get row by label (loc).
    pub fn loc_row(&self, label: &Value) -> Result<HashMap<String, Value>> {
        let pos = self.index.get_loc(label)?;
        self.iloc_row(pos)
    }

    /// Get rows by labels (loc).
    pub fn loc(&self, labels: &[Value]) -> Result<DataFrame> {
        let positions: Result<Vec<_>> = labels.iter().map(|l| self.index.get_loc(l)).collect();
        self.iloc(&positions?)
    }

    // === Filtering ===

    /// Filter rows by boolean mask.
    pub fn filter(&self, mask: &[bool]) -> Result<DataFrame> {
        if mask.len() != self.nrows() {
            return Err(FrameError::ShapeMismatch {
                expected: self.nrows(),
                actual: mask.len(),
            });
        }

        let positions: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        self.iloc(&positions)
    }

    /// Query with a predicate function.
    pub fn query<F>(&self, predicate: F) -> Result<DataFrame>
    where
        F: Fn(&HashMap<String, Value>) -> bool,
    {
        let mask: Vec<bool> = (0..self.nrows())
            .map(|i| {
                let row = self.iloc_row(i).unwrap_or_default();
                predicate(&row)
            })
            .collect();
        self.filter(&mask)
    }

    // === Aggregation ===

    /// Get descriptive statistics.
    pub fn describe(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut result = HashMap::new();

        for (name, col) in self.column_names.iter().zip(self.columns.iter()) {
            let mut stats = HashMap::new();

            // Statistics skip nulls (see the `# Null policy` section of the frame module).
            if let Ok(vals) = col.non_null_f64() {
                stats.insert("count".to_string(), vals.len() as f64);
                if vals.is_empty() {
                    stats.insert("sum".to_string(), 0.0);
                } else {
                    stats.insert("sum".to_string(), vals.iter().sum());
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    stats.insert("mean".to_string(), mean);
                    if let Ok(std) = col.std() {
                        stats.insert("std".to_string(), std);
                    }
                    if let Some(&min) = vals.iter().min_by(|a, b| a.total_cmp(b)) {
                        stats.insert("min".to_string(), min);
                    }
                    for (key, q) in [("25%", 0.25), ("50%", 0.5), ("75%", 0.75)] {
                        if let Ok(value) = col.quantile(q) {
                            stats.insert(key.to_string(), value);
                        }
                    }
                    if let Some(&max) = vals.iter().max_by(|a, b| a.total_cmp(b)) {
                        stats.insert("max".to_string(), max);
                    }
                }
            }

            result.insert(name.clone(), stats);
        }

        result
    }

    // === Sorting ===

    /// Sort by column.
    pub fn sort_values(&self, by: &str, ascending: bool) -> Result<DataFrame> {
        let col = self.get(by)?;
        let vals = col.values();

        let mut indices: Vec<usize> = (0..self.nrows()).collect();
        indices.sort_by(|&a, &b| cmp_nulls_last(&vals[a], &vals[b], ascending));

        self.iloc(&indices)
    }

    /// Sort by multiple columns.
    pub fn sort_values_by(&self, by: &[&str], ascending: &[bool]) -> Result<DataFrame> {
        if by.len() != ascending.len() {
            return Err(FrameError::InvalidOperation(
                "by and ascending must have same length".into(),
            ));
        }

        let cols: Result<Vec<_>> = by.iter().map(|&name| self.get(name)).collect();
        let cols = cols?;

        let mut indices: Vec<usize> = (0..self.nrows()).collect();
        indices.sort_by(|&a, &b| {
            for (col, &asc) in cols.iter().zip(ascending.iter()) {
                let va = col.iloc(a).unwrap();
                let vb = col.iloc(b).unwrap();
                let ord = cmp_nulls_last(va, vb, asc);
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });

        self.iloc(&indices)
    }

    // === Head/Tail ===

    /// Get first n rows.
    pub fn head(&self, n: usize) -> Result<DataFrame> {
        let n = n.min(self.nrows());
        self.iloc(&(0..n).collect::<Vec<_>>())
    }

    /// Get last n rows.
    pub fn tail(&self, n: usize) -> Result<DataFrame> {
        let n = n.min(self.nrows());
        let start = self.nrows().saturating_sub(n);
        self.iloc(&(start..self.nrows()).collect::<Vec<_>>())
    }

    /// Reset index to range.
    pub fn reset_index(&mut self) {
        self.index = Index::Range(self.nrows());
    }

    /// Iterate over rows as (index_label, row_data).
    pub fn iterrows(&self) -> impl Iterator<Item = (Value, HashMap<String, Value>)> + '_ {
        (0..self.nrows()).map(move |i| {
            let label = self.index.get_label(i).unwrap_or(Value::Int(i as i64));
            let row = self.iloc_row(i).unwrap_or_default();
            (label, row)
        })
    }

    // === Null handling (pandas-style) ===

    /// Return a boolean DataFrame indicating which values are null.
    pub fn isna(&self) -> HashMap<String, Vec<bool>> {
        self.column_names
            .iter()
            .zip(self.columns.iter())
            .map(|(name, col)| (name.clone(), col.isna()))
            .collect()
    }

    /// Return a boolean mask for rows with any null values.
    pub fn isna_any(&self) -> Vec<bool> {
        (0..self.nrows())
            .map(|i| {
                self.columns
                    .iter()
                    .any(|col| col.iloc(i).map(|v| v.is_null()).unwrap_or(false))
            })
            .collect()
    }

    /// Return a boolean mask for rows with all null values.
    pub fn isna_all(&self) -> Vec<bool> {
        (0..self.nrows())
            .map(|i| {
                self.columns
                    .iter()
                    .all(|col| col.iloc(i).map(|v| v.is_null()).unwrap_or(true))
            })
            .collect()
    }

    /// Fill null values in all columns with the given value.
    pub fn fillna(&self, value: Value) -> DataFrame {
        let columns: Vec<_> = self
            .column_names
            .iter()
            .zip(self.columns.iter())
            .map(|(name, col)| (name.as_str(), col.fillna(value.clone())))
            .collect();

        DataFrame::from_columns(columns).unwrap_or_else(|_| self.clone())
    }

    /// Fill null values in specific column with the given value.
    pub fn fillna_column(&self, column: &str, value: Value) -> Result<DataFrame> {
        let mut new_df = self.clone();
        let idx = self
            .column_names
            .iter()
            .position(|n| n == column)
            .ok_or_else(|| FrameError::ColumnNotFound(column.to_string()))?;

        new_df.columns[idx] = self.columns[idx].fillna(value);
        Ok(new_df)
    }

    /// Drop rows with any null values.
    ///
    /// # Example
    /// ```
    /// use frame::frame::{DataFrame, Series, Value};
    ///
    /// let a = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]);
    /// let b = Series::new(vec![Value::Int(4), Value::Int(5), Value::Int(6)]);
    /// let df = DataFrame::from_columns(vec![("a", a), ("b", b)]).unwrap();
    /// let clean = df.dropna().unwrap();
    /// assert_eq!(clean.nrows(), 2);
    /// ```
    pub fn dropna(&self) -> Result<DataFrame> {
        let mask: Vec<bool> = self.isna_any().iter().map(|&has_null| !has_null).collect();
        self.filter(&mask)
    }

    /// Drop rows with all null values.
    pub fn dropna_all(&self) -> Result<DataFrame> {
        let mask: Vec<bool> = self.isna_all().iter().map(|&all_null| !all_null).collect();
        self.filter(&mask)
    }

    /// Drop rows with null values in specific columns.
    pub fn dropna_subset(&self, columns: &[&str]) -> Result<DataFrame> {
        let col_indices: Result<Vec<_>> = columns
            .iter()
            .map(|&name| {
                self.column_names
                    .iter()
                    .position(|n| n == name)
                    .ok_or_else(|| FrameError::ColumnNotFound(name.to_string()))
            })
            .collect();

        let col_indices = col_indices?;

        let mask: Vec<bool> = (0..self.nrows())
            .map(|i| {
                !col_indices.iter().any(|&idx| {
                    self.columns[idx]
                        .iloc(i)
                        .map(|v| v.is_null())
                        .unwrap_or(false)
                })
            })
            .collect();

        self.filter(&mask)
    }

    /// Apply a function to each column.
    ///
    /// # Example
    /// ```
    /// use frame::frame::{DataFrame, Series, Value};
    ///
    /// let a = Series::new(vec![Value::Int(1), Value::Int(2)]);
    /// let df = DataFrame::from_columns(vec![("a", a)]).unwrap();
    /// let result = df.apply(|col| col.sum().unwrap_or(0.0));
    /// assert_eq!(result["a"], 3.0);
    /// ```
    pub fn apply<F, R>(&self, f: F) -> HashMap<String, R>
    where
        F: Fn(&Series) -> R,
    {
        self.column_names
            .iter()
            .zip(self.columns.iter())
            .map(|(name, col)| (name.clone(), f(col)))
            .collect()
    }

    /// Apply a function to each element in the DataFrame.
    pub fn applymap<F>(&self, f: F) -> DataFrame
    where
        F: Fn(&Value) -> Value,
    {
        let columns: Vec<_> = self
            .column_names
            .iter()
            .zip(self.columns.iter())
            .map(|(name, col)| (name.as_str(), col.apply(&f)))
            .collect();

        DataFrame::from_columns(columns).unwrap_or_else(|_| self.clone())
    }

    /// Concatenate DataFrames vertically (axis=0).
    pub fn concat(dfs: &[&DataFrame]) -> Result<DataFrame> {
        if dfs.is_empty() {
            return Ok(DataFrame::new());
        }

        let first = dfs[0];
        let col_names = first.columns();

        // Verify all DataFrames have the same columns
        for df in &dfs[1..] {
            if df.columns() != col_names {
                return Err(FrameError::InvalidOperation(
                    "all DataFrames must have the same columns".into(),
                ));
            }
        }

        // Concatenate each column
        let columns: Vec<_> = col_names
            .iter()
            .enumerate()
            .map(|(idx, name)| {
                let mut data: Vec<Value> = Vec::new();
                for df in dfs {
                    for val in df.columns[idx].values() {
                        data.push(val.clone());
                    }
                }
                (name.as_str(), Series::new(data))
            })
            .collect();

        DataFrame::from_columns(columns)
    }

    // === Advanced operations ===

    /// Create DataFrame from list of records (dicts).
    pub fn from_records(records: &[HashMap<String, Value>]) -> Result<DataFrame> {
        if records.is_empty() {
            return Ok(DataFrame::new());
        }

        // Get all column names
        let mut col_names: Vec<String> = records[0].keys().cloned().collect();
        col_names.sort();

        // Build columns
        let columns: Vec<_> = col_names
            .iter()
            .map(|name| {
                let data: Vec<Value> = records
                    .iter()
                    .map(|r| r.get(name).cloned().unwrap_or(Value::Null))
                    .collect();
                (name.as_str(), Series::new(data))
            })
            .collect();

        DataFrame::from_columns(columns)
    }

    /// Convert to list of records (dicts).
    pub fn to_records(&self) -> Vec<HashMap<String, Value>> {
        (0..self.nrows())
            .map(|i| self.iloc_row(i).unwrap_or_default())
            .collect()
    }

    /// Convert to dict of lists.
    pub fn to_dict(&self) -> HashMap<String, Vec<Value>> {
        self.column_names
            .iter()
            .zip(self.columns.iter())
            .map(|(name, col)| (name.clone(), col.values().to_vec()))
            .collect()
    }

    /// Check for duplicate rows.
    pub fn duplicated(&self, subset: Option<&[&str]>, keep: &str) -> Result<Vec<bool>> {
        let cols = match subset {
            Some(names) => {
                let cols: Result<Vec<_>> = names.iter().map(|&n| self.get(n)).collect();
                cols?
            }
            None => self.columns.iter().collect(),
        };

        let mut seen: HashMap<Vec<Value>, usize> = HashMap::new();
        let mut result = vec![false; self.nrows()];

        match keep {
            "first" => {
                for (i, slot) in result.iter_mut().enumerate() {
                    let key: Vec<Value> = cols
                        .iter()
                        .map(|c| c.iloc(i).cloned().unwrap_or(Value::Null))
                        .collect();
                    if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key) {
                        e.insert(i);
                    } else {
                        *slot = true;
                    }
                }
            }
            "last" => {
                for i in 0..self.nrows() {
                    let key: Vec<Value> = cols
                        .iter()
                        .map(|c| c.iloc(i).cloned().unwrap_or(Value::Null))
                        .collect();
                    if let Some(&prev_idx) = seen.get(&key) {
                        result[prev_idx] = true;
                    }
                    seen.insert(key, i);
                }
            }
            _ => {
                let mut counts: HashMap<Vec<Value>, usize> = HashMap::new();
                for i in 0..self.nrows() {
                    let key: Vec<Value> = cols
                        .iter()
                        .map(|c| c.iloc(i).cloned().unwrap_or(Value::Null))
                        .collect();
                    *counts.entry(key).or_insert(0) += 1;
                }
                for (i, slot) in result.iter_mut().enumerate() {
                    let key: Vec<Value> = cols
                        .iter()
                        .map(|c| c.iloc(i).cloned().unwrap_or(Value::Null))
                        .collect();
                    if counts.get(&key).map(|&c| c > 1).unwrap_or(false) {
                        *slot = true;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Drop duplicate rows.
    pub fn drop_duplicates(&self, subset: Option<&[&str]>, keep: &str) -> Result<DataFrame> {
        let mask: Vec<bool> = self.duplicated(subset, keep)?.iter().map(|&d| !d).collect();
        self.filter(&mask)
    }

    /// Compute correlation matrix.
    pub fn corr(&self) -> Result<DataFrame> {
        let numeric_cols: Vec<(&String, &Series)> = self
            .column_names
            .iter()
            .zip(self.columns.iter())
            .filter(|(_, s)| {
                s.values()
                    .iter()
                    .all(|v| v.is_null() || v.as_float().is_some())
            })
            .collect();

        if numeric_cols.is_empty() {
            return Ok(DataFrame::new());
        }

        let n = numeric_cols.len();
        let mut result_data: Vec<Vec<Value>> = vec![vec![Value::Null; n]; n];

        for (i, (_, col_i)) in numeric_cols.iter().enumerate() {
            for (j, (_, col_j)) in numeric_cols.iter().enumerate() {
                let corr = col_i.corr(col_j).unwrap_or(f64::NAN);
                result_data[i][j] = Value::Float(corr);
            }
        }

        let columns: Vec<_> = numeric_cols
            .iter()
            .enumerate()
            .map(|(i, (name, _))| {
                let data: Vec<Value> = result_data.iter().map(|row| row[i].clone()).collect();
                (name.as_str(), Series::new(data))
            })
            .collect();

        DataFrame::from_columns(columns)
    }

    /// Pivot table: reshape from long to wide format.
    ///
    /// - index: column to use as new index
    /// - columns: column to use for new column headers
    /// - values: column containing values
    pub fn pivot(&self, index: &str, columns: &str, values: &str) -> Result<DataFrame> {
        let idx_col = self.get(index)?;
        let col_col = self.get(columns)?;
        let val_col = self.get(values)?;

        // Get unique index and column values
        let unique_idx = idx_col.unique();
        let unique_cols = col_col.unique();

        // Build index -> row mapping
        let idx_map: HashMap<Value, usize> = unique_idx
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();

        // Build result
        let mut result_data: Vec<Vec<Value>> =
            vec![vec![Value::Null; unique_idx.len()]; unique_cols.len()];

        for i in 0..self.nrows() {
            let idx_val = idx_col.iloc(i)?;
            let col_val = col_col.iloc(i)?;
            let val = val_col.iloc(i)?;

            if let (Some(&row_idx), Some(col_idx)) = (
                idx_map.get(idx_val),
                unique_cols.iter().position(|c| c == col_val),
            ) {
                result_data[col_idx][row_idx] = val.clone();
            }
        }

        // Create columns. The names live in `names` until from_columns has copied them.
        let names: Vec<String> = unique_cols.iter().map(|v| v.to_string()).collect();
        let mut columns_vec: Vec<(&str, Series)> = vec![(index, Series::new(unique_idx))];
        for (i, name) in names.iter().enumerate() {
            columns_vec.push((name.as_str(), Series::new(result_data[i].clone())));
        }

        DataFrame::from_columns(columns_vec)
    }

    /// Melt: reshape from wide to long format.
    ///
    /// - id_vars: columns to keep as identifiers
    /// - value_vars: columns to unpivot
    pub fn melt(&self, id_vars: &[&str], value_vars: &[&str]) -> Result<DataFrame> {
        let value_cols: Vec<&str> = if value_vars.is_empty() {
            self.column_names
                .iter()
                .filter(|n| !id_vars.contains(&n.as_str()))
                .map(|s| s.as_str())
                .collect()
        } else {
            value_vars.to_vec()
        };

        let n_rows = self.nrows() * value_cols.len();
        let mut id_data: Vec<Vec<Value>> = vec![Vec::with_capacity(n_rows); id_vars.len()];
        let mut var_data: Vec<Value> = Vec::with_capacity(n_rows);
        let mut val_data: Vec<Value> = Vec::with_capacity(n_rows);

        for var_name in &value_cols {
            let var_col = self.get(var_name)?;
            for i in 0..self.nrows() {
                // Add ID values
                for (j, &id_name) in id_vars.iter().enumerate() {
                    let id_col = self.get(id_name)?;
                    id_data[j].push(id_col.iloc(i).cloned().unwrap_or(Value::Null));
                }
                // Add variable name and value
                var_data.push(Value::String(var_name.to_string()));
                val_data.push(var_col.iloc(i).cloned().unwrap_or(Value::Null));
            }
        }

        let mut columns_vec: Vec<(&str, Series)> = Vec::new();
        for (i, &id_name) in id_vars.iter().enumerate() {
            columns_vec.push((id_name, Series::new(std::mem::take(&mut id_data[i]))));
        }
        columns_vec.push(("variable", Series::new(var_data)));
        columns_vec.push(("value", Series::new(val_data)));

        DataFrame::from_columns(columns_vec)
    }

    /// Sample n distinct rows with the fixed seed 42, so the result is the same on every call.
    /// Use `sample_seeded` to choose the seed.
    pub fn sample(&self, n: usize) -> Result<DataFrame> {
        self.sample_seeded(n, 42)
    }

    /// Sample n distinct rows. The same seed gives the same rows in the same order.
    /// If n is larger than the row count, all rows are returned.
    pub fn sample_seeded(&self, n: usize, seed: u64) -> Result<DataFrame> {
        use std::collections::HashSet;

        let nrows = self.nrows();
        let n = n.min(nrows);
        if n == 0 {
            return Ok(DataFrame::new());
        }

        let mut state = seed;
        let mut seen: HashSet<usize> = HashSet::new();
        let mut indices: Vec<usize> = Vec::with_capacity(n);
        while indices.len() < n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = ((state >> 33) as usize) % nrows;
            if seen.insert(idx) {
                indices.push(idx);
            }
        }
        self.iloc(&indices)
    }

    /// Assign new columns.
    pub fn assign(&self, columns: Vec<(&str, Series)>) -> Result<DataFrame> {
        let mut df = self.clone();
        for (name, series) in columns {
            df.insert(name, series)?;
        }
        Ok(df)
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
