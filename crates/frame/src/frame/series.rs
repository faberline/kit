//! Series - one-dimensional labeled array.

use super::error::{FrameError, Result};
use super::index::Index;
use super::value::{cmp_nulls_last, Value};

mod null_ops;
mod window_ops;

/// A one-dimensional labeled array.
#[derive(Debug, Clone)]
pub struct Series {
    /// Series name.
    name: Option<String>,
    /// Data values.
    data: Vec<Value>,
    /// Row index.
    index: Index,
}

impl Series {
    /// Create a new Series from values.
    pub fn new<V: Into<Value>>(data: Vec<V>) -> Self {
        let data: Vec<Value> = data.into_iter().map(|v| v.into()).collect();
        let len = data.len();
        Self {
            name: None,
            data,
            index: Index::Range(len),
        }
    }

    /// Create a Series with a name.
    pub fn with_name<V: Into<Value>>(data: Vec<V>, name: impl Into<String>) -> Self {
        let mut series = Self::new(data);
        series.name = Some(name.into());
        series
    }

    /// Create a Series with a custom index.
    pub fn with_index<V: Into<Value>>(data: Vec<V>, index: Index) -> Result<Self> {
        let data: Vec<Value> = data.into_iter().map(|v| v.into()).collect();
        if data.len() != index.len() {
            return Err(FrameError::ShapeMismatch {
                expected: index.len(),
                actual: data.len(),
            });
        }
        Ok(Self {
            name: None,
            data,
            index,
        })
    }

    /// Get the series name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the series name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Get the length of the series.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the series is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the index.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Get value by positional index (iloc).
    pub fn iloc(&self, pos: usize) -> Result<&Value> {
        self.data.get(pos).ok_or(FrameError::IndexOutOfBounds {
            index: pos,
            length: self.len(),
        })
    }

    /// Get value by label (loc).
    pub fn loc(&self, label: &Value) -> Result<&Value> {
        let pos = self.index.get_loc(label)?;
        Ok(&self.data[pos])
    }

    /// Get multiple values by positions (iloc).
    pub fn iloc_many(&self, positions: &[usize]) -> Result<Series> {
        let data: Result<Vec<_>> = positions.iter().map(|&p| self.iloc(p).cloned()).collect();
        let new_index = self.index.slice(positions)?;
        Ok(Series {
            name: self.name.clone(),
            data: data?,
            index: new_index,
        })
    }

    /// Get the underlying data as a slice.
    pub fn values(&self) -> &[Value] {
        &self.data
    }

    /// Convert to Vec of f64 (for numeric operations).
    pub fn to_f64(&self) -> Result<Vec<f64>> {
        self.data
            .iter()
            .map(|v| {
                v.as_float()
                    .ok_or_else(|| FrameError::TypeMismatch("cannot convert to f64".into()))
            })
            .collect()
    }

    /// Convert the non-null values to Vec of f64. Nulls are skipped.
    pub(crate) fn non_null_f64(&self) -> Result<Vec<f64>> {
        self.data
            .iter()
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_float()
                    .ok_or_else(|| FrameError::TypeMismatch("cannot convert to f64".into()))
            })
            .collect()
    }

    /// Convert to Vec of i64 (for integer operations).
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        self.data
            .iter()
            .map(|v| {
                v.as_int()
                    .ok_or_else(|| FrameError::TypeMismatch("cannot convert to i64".into()))
            })
            .collect()
    }

    /// Sum of numeric values.
    pub fn sum(&self) -> Result<f64> {
        Ok(self.non_null_f64()?.into_iter().sum())
    }

    /// Mean of numeric values.
    pub fn mean(&self) -> Result<f64> {
        let vals = self.non_null_f64()?;
        if vals.is_empty() {
            return Err(FrameError::InvalidOperation("mean of empty series".into()));
        }
        Ok(vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// Minimum value.
    pub fn min(&self) -> Result<Value> {
        self.data
            .iter()
            .filter(|v| !v.is_null())
            .min()
            .cloned()
            .ok_or_else(|| FrameError::InvalidOperation("min of empty series".into()))
    }

    /// Maximum value.
    pub fn max(&self) -> Result<Value> {
        self.data
            .iter()
            .filter(|v| !v.is_null())
            .max()
            .cloned()
            .ok_or_else(|| FrameError::InvalidOperation("max of empty series".into()))
    }

    /// Count non-null values.
    pub fn count(&self) -> usize {
        self.data.iter().filter(|v| !v.is_null()).count()
    }

    /// Filter by boolean mask.
    pub fn filter(&self, mask: &[bool]) -> Result<Series> {
        if mask.len() != self.len() {
            return Err(FrameError::ShapeMismatch {
                expected: self.len(),
                actual: mask.len(),
            });
        }

        let positions: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        self.iloc_many(&positions)
    }

    /// Apply a function to each value.
    pub fn map<F>(&self, f: F) -> Series
    where
        F: Fn(&Value) -> Value,
    {
        Series {
            name: self.name.clone(),
            data: self.data.iter().map(f).collect(),
            index: self.index.clone(),
        }
    }

    /// Reset index to range.
    pub fn reset_index(&mut self) {
        self.index = Index::Range(self.len());
    }

    /// Sort values.
    pub fn sort(&self, ascending: bool) -> Series {
        let mut indexed: Vec<(usize, &Value)> = self.data.iter().enumerate().collect();
        indexed.sort_by(|a, b| cmp_nulls_last(a.1, b.1, ascending));

        let positions: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        // This unwrap is safe because positions come from valid indices
        self.iloc_many(&positions).unwrap()
    }

    /// Get unique values.
    pub fn unique(&self) -> Vec<Value> {
        let mut seen = std::collections::HashSet::new();
        self.data
            .iter()
            .filter(|v| seen.insert((*v).clone()))
            .cloned()
            .collect()
    }

    /// Count occurrences of each value.
    pub fn value_counts(&self) -> std::collections::HashMap<Value, usize> {
        let mut counts = std::collections::HashMap::new();
        for v in &self.data {
            *counts.entry(v.clone()).or_insert(0) += 1;
        }
        counts
    }

    // === Selection ===

    /// Get the n largest values.
    pub fn nlargest(&self, n: usize) -> Series {
        let sorted = self.sort(false);
        let positions: Vec<usize> = (0..n.min(sorted.len())).collect();
        sorted
            .iloc_many(&positions)
            .unwrap_or_else(|_| self.clone())
    }

    /// Get the n smallest values.
    pub fn nsmallest(&self, n: usize) -> Series {
        let sorted = self.sort(true);
        let positions: Vec<usize> = (0..n.min(sorted.len())).collect();
        sorted
            .iloc_many(&positions)
            .unwrap_or_else(|_| self.clone())
    }

    /// Clip values to a range.
    pub fn clip(&self, lower: f64, upper: f64) -> Series {
        let data: Vec<Value> = self
            .data
            .iter()
            .map(|v| {
                if let Some(val) = v.as_float() {
                    if val < lower {
                        Value::Float(lower)
                    } else if val > upper {
                        Value::Float(upper)
                    } else {
                        Value::Float(val)
                    }
                } else {
                    v.clone()
                }
            })
            .collect();

        Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        }
    }

    /// Standard deviation.
    pub fn std(&self) -> Result<f64> {
        let vals = self.non_null_f64()?;
        if vals.len() <= 1 {
            return Err(FrameError::InvalidOperation(
                "std requires at least 2 values".into(),
            ));
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let variance: f64 =
            vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
        Ok(variance.sqrt())
    }

    /// Variance.
    pub fn var(&self) -> Result<f64> {
        let vals = self.non_null_f64()?;
        if vals.len() <= 1 {
            return Err(FrameError::InvalidOperation(
                "var requires at least 2 values".into(),
            ));
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let variance: f64 =
            vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
        Ok(variance)
    }

    /// Median value.
    pub fn median(&self) -> Result<f64> {
        let mut vals = self.non_null_f64()?;
        if vals.is_empty() {
            return Err(FrameError::InvalidOperation(
                "median of empty series".into(),
            ));
        }
        vals.sort_by(|a, b| a.total_cmp(b));
        let mid = vals.len() / 2;
        if vals.len() % 2 == 0 {
            Ok((vals[mid - 1] + vals[mid]) / 2.0)
        } else {
            Ok(vals[mid])
        }
    }

    // === Advanced operations ===

    /// Rank values, 1-based, ascending. Tied values get the average of their ranks (pandas `method='average'`).
    pub fn rank(&self) -> Series {
        let n = self.len();
        let mut indexed: Vec<(usize, &Value)> = self.data.iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.cmp(b.1));

        let mut ranks = vec![Value::Null; n];
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            while end < n && indexed[end].1 == indexed[start].1 {
                end += 1;
            }
            // The 1-based positions start+1 ..= end form one tie group; they share their average.
            let rank = ((start + 1) + end) as f64 / 2.0;
            for (original_idx, _) in &indexed[start..end] {
                ranks[*original_idx] = Value::Float(rank);
            }
            start = end;
        }

        Series {
            name: self.name.clone(),
            data: ranks,
            index: self.index.clone(),
        }
    }

    /// Check if values are duplicated.
    pub fn duplicated(&self, keep: &str) -> Vec<bool> {
        let mut seen = std::collections::HashMap::new();
        let mut result = vec![false; self.len()];

        match keep {
            "first" => {
                for (i, v) in self.data.iter().enumerate() {
                    if seen.contains_key(v) {
                        result[i] = true;
                    } else {
                        seen.insert(v.clone(), i);
                    }
                }
            }
            "last" => {
                // First pass: mark all as duplicated
                for (i, v) in self.data.iter().enumerate() {
                    if let Some(&prev_idx) = seen.get(v) {
                        result[prev_idx] = true;
                    }
                    seen.insert(v.clone(), i);
                }
            }
            _ => {
                // Mark all duplicates
                let mut counts = std::collections::HashMap::new();
                for v in &self.data {
                    *counts.entry(v.clone()).or_insert(0) += 1;
                }
                for (i, v) in self.data.iter().enumerate() {
                    if counts.get(v).map(|&c| c > 1).unwrap_or(false) {
                        result[i] = true;
                    }
                }
            }
        }

        result
    }

    /// Drop duplicate values.
    pub fn drop_duplicates(&self, keep: &str) -> Series {
        let mask: Vec<bool> = self.duplicated(keep).iter().map(|&d| !d).collect();
        self.filter(&mask).unwrap_or_else(|_| self.clone())
    }

    /// Check if values are in a set.
    pub fn isin(&self, values: &[Value]) -> Vec<bool> {
        let set: std::collections::HashSet<_> = values.iter().collect();
        self.data.iter().map(|v| set.contains(v)).collect()
    }

    /// Check if values are between lower and upper (inclusive).
    pub fn between(&self, lower: f64, upper: f64) -> Vec<bool> {
        self.data
            .iter()
            .map(|v| {
                v.as_float()
                    .map(|f| f >= lower && f <= upper)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Replace values.
    pub fn replace(&self, to_replace: &Value, value: &Value) -> Series {
        let data: Vec<Value> = self
            .data
            .iter()
            .map(|v| {
                if v == to_replace {
                    value.clone()
                } else {
                    v.clone()
                }
            })
            .collect();

        Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        }
    }

    /// Replace multiple values.
    pub fn replace_map(&self, mapping: &std::collections::HashMap<Value, Value>) -> Series {
        let data: Vec<Value> = self
            .data
            .iter()
            .map(|v| mapping.get(v).cloned().unwrap_or_else(|| v.clone()))
            .collect();

        Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        }
    }

    /// Convert to dict (index -> value).
    pub fn to_dict(&self) -> std::collections::HashMap<Value, Value> {
        let mut result = std::collections::HashMap::new();
        for (i, v) in self.data.iter().enumerate() {
            let key = self.index.get_label(i).unwrap_or(Value::Int(i as i64));
            result.insert(key, v.clone());
        }
        result
    }

    /// Quantile (q between 0 and 1).
    pub fn quantile(&self, q: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&q) {
            return Err(FrameError::InvalidOperation(
                "q must be between 0 and 1".into(),
            ));
        }

        let mut vals = self.non_null_f64()?;
        if vals.is_empty() {
            return Err(FrameError::InvalidOperation(
                "quantile of empty series".into(),
            ));
        }

        vals.sort_by(|a, b| a.total_cmp(b));
        let idx = q * (vals.len() - 1) as f64;
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;

        if lower == upper {
            Ok(vals[lower])
        } else {
            let frac = idx - lower as f64;
            Ok(vals[lower] * (1.0 - frac) + vals[upper] * frac)
        }
    }

    /// Correlation with another series.
    pub fn corr(&self, other: &Series) -> Result<f64> {
        if self.len() != other.len() {
            return Err(FrameError::ShapeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let x = self.to_f64()?;
        let y = other.to_f64()?;
        let n = x.len() as f64;

        if n < 2.0 {
            return Err(FrameError::InvalidOperation(
                "need at least 2 values".into(),
            ));
        }

        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let std_x = var_x.sqrt();
        let std_y = var_y.sqrt();

        if std_x == 0.0 || std_y == 0.0 {
            return Ok(0.0);
        }

        Ok(cov / (std_x * std_y))
    }

    // === Type conversion ===

    /// Convert series to a specific type.
    pub fn astype(&self, dtype: &str) -> Series {
        let data: Vec<Value> = match dtype {
            "int" | "i64" => self
                .data
                .iter()
                .map(|v| v.as_int().map(Value::Int).unwrap_or(Value::Null))
                .collect(),
            "float" | "f64" => self
                .data
                .iter()
                .map(|v| v.as_float().map(Value::Float).unwrap_or(Value::Null))
                .collect(),
            "str" | "string" => self
                .data
                .iter()
                .map(|v| Value::String(v.to_string()))
                .collect(),
            "bool" => self
                .data
                .iter()
                .map(|v| v.as_bool().map(Value::Bool).unwrap_or(Value::Null))
                .collect(),
            _ => self.data.clone(),
        };

        Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        }
    }

    /// Conditional replacement: where condition is false, replace with other.
    pub fn where_cond(&self, cond: &[bool], other: &Value) -> Series {
        let data: Vec<Value> = self
            .data
            .iter()
            .zip(cond.iter())
            .map(|(v, &c)| if c { v.clone() } else { other.clone() })
            .collect();

        Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        }
    }

    /// Mask: where condition is true, replace with other.
    pub fn mask(&self, cond: &[bool], other: &Value) -> Series {
        let data: Vec<Value> = self
            .data
            .iter()
            .zip(cond.iter())
            .map(|(v, &c)| if c { other.clone() } else { v.clone() })
            .collect();

        Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        }
    }

    /// Absolute value.
    pub fn abs(&self) -> Series {
        let data: Vec<Value> = self
            .data
            .iter()
            .map(|v| match v {
                Value::Int(i) => Value::Int(i.abs()),
                Value::Float(f) => Value::Float(f.abs()),
                _ => v.clone(),
            })
            .collect();

        Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        }
    }

    /// Round values.
    pub fn round(&self, decimals: i32) -> Series {
        let factor = 10f64.powi(decimals);
        let data: Vec<Value> = self
            .data
            .iter()
            .map(|v| match v.as_float() {
                Some(f) => Value::Float((f * factor).round() / factor),
                None => v.clone(),
            })
            .collect();

        Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        }
    }
}

// Arithmetic operations for numeric Series
impl std::ops::Add for &Series {
    type Output = Result<Series>;

    fn add(self, other: &Series) -> Self::Output {
        if self.len() != other.len() {
            return Err(FrameError::ShapeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let data: Vec<Value> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| match (a.as_float(), b.as_float()) {
                (Some(x), Some(y)) => Value::Float(x + y),
                _ => Value::Null,
            })
            .collect();

        Ok(Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        })
    }
}

impl std::ops::Sub for &Series {
    type Output = Result<Series>;

    fn sub(self, other: &Series) -> Self::Output {
        if self.len() != other.len() {
            return Err(FrameError::ShapeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let data: Vec<Value> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| match (a.as_float(), b.as_float()) {
                (Some(x), Some(y)) => Value::Float(x - y),
                _ => Value::Null,
            })
            .collect();

        Ok(Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        })
    }
}

impl std::ops::Mul for &Series {
    type Output = Result<Series>;

    fn mul(self, other: &Series) -> Self::Output {
        if self.len() != other.len() {
            return Err(FrameError::ShapeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let data: Vec<Value> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| match (a.as_float(), b.as_float()) {
                (Some(x), Some(y)) => Value::Float(x * y),
                _ => Value::Null,
            })
            .collect();

        Ok(Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        })
    }
}

impl std::ops::Div for &Series {
    type Output = Result<Series>;

    fn div(self, other: &Series) -> Self::Output {
        if self.len() != other.len() {
            return Err(FrameError::ShapeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let data: Vec<Value> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| match (a.as_float(), b.as_float()) {
                (Some(x), Some(y)) if y != 0.0 => Value::Float(x / y),
                _ => Value::Null,
            })
            .collect();

        Ok(Series {
            name: self.name.clone(),
            data,
            index: self.index.clone(),
        })
    }
}

#[cfg(test)]
mod tests;
