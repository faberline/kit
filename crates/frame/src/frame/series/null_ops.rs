//! Null handling for Series (pandas-style).

use super::super::value::Value;
use super::Series;

impl Series {
    // === Null handling (pandas-style) ===

    /// Return a boolean mask indicating which values are null.
    ///
    /// # Example
    /// ```
    /// use frame::frame::{Series, Value};
    ///
    /// let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]);
    /// assert_eq!(s.isna(), vec![false, true, false]);
    /// ```
    pub fn isna(&self) -> Vec<bool> {
        self.values().iter().map(|v| v.is_null()).collect()
    }

    /// Return a boolean mask indicating which values are not null.
    pub fn notna(&self) -> Vec<bool> {
        self.values().iter().map(|v| !v.is_null()).collect()
    }

    /// Fill null values with the given value.
    ///
    /// # Example
    /// ```
    /// use frame::frame::{Series, Value};
    ///
    /// let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]);
    /// let filled = s.fillna(Value::Int(0));
    /// assert_eq!(filled.iloc(1).unwrap(), &Value::Int(0));
    /// ```
    pub fn fillna(&self, value: Value) -> Series {
        let data: Vec<Value> = self
            .values()
            .iter()
            .map(|v| {
                if v.is_null() {
                    value.clone()
                } else {
                    v.clone()
                }
            })
            .collect();

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    /// Fill null values using forward fill (propagate last valid value).
    pub fn ffill(&self) -> Series {
        let mut data = Vec::with_capacity(self.values().len());
        let mut last_valid: Option<Value> = None;

        for v in self.values() {
            if v.is_null() {
                data.push(last_valid.clone().unwrap_or(Value::Null));
            } else {
                last_valid = Some(v.clone());
                data.push(v.clone());
            }
        }

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    /// Fill null values using backward fill (propagate next valid value).
    pub fn bfill(&self) -> Series {
        let mut data = vec![Value::Null; self.values().len()];
        let mut next_valid: Option<Value> = None;

        for (i, v) in self.values().iter().enumerate().rev() {
            if v.is_null() {
                data[i] = next_valid.clone().unwrap_or(Value::Null);
            } else {
                next_valid = Some(v.clone());
                data[i] = v.clone();
            }
        }

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    /// Drop null values, returning a new Series.
    ///
    /// # Example
    /// ```
    /// use frame::frame::{Series, Value};
    ///
    /// let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]);
    /// let clean = s.dropna();
    /// assert_eq!(clean.len(), 2);
    /// ```
    pub fn dropna(&self) -> Series {
        let mask = self.notna();
        // filter will not fail because mask is same length as data
        self.filter(&mask).unwrap_or_else(|_| self.clone())
    }

    /// Interpolate missing values using linear interpolation.
    ///
    /// # Example
    /// ```
    /// use frame::frame::{Series, Value};
    ///
    /// let s = Series::new(vec![Value::Float(1.0), Value::Null, Value::Float(3.0)]);
    /// let filled = s.interpolate();
    /// assert_eq!(filled.iloc(1).unwrap(), &Value::Float(2.0));
    /// ```
    pub fn interpolate(&self) -> Series {
        self.interpolate_method("linear")
    }

    /// Interpolate missing values using the specified method.
    ///
    /// Methods:
    /// - "linear": Linear interpolation between valid values
    /// - "ffill": Forward fill (same as ffill())
    /// - "bfill": Backward fill (same as bfill())
    /// - "nearest": Use nearest valid value
    pub fn interpolate_method(&self, method: &str) -> Series {
        match method {
            "ffill" => self.ffill(),
            "bfill" => self.bfill(),
            "nearest" => self.interpolate_nearest(),
            _ => self.interpolate_linear(),
        }
    }

    /// Linear interpolation.
    ///
    /// Only interpolates Null values. Non-null values (including non-numeric) are preserved.
    fn interpolate_linear(&self) -> Series {
        let n = self.values().len();
        let mut result = self.values().to_vec();

        // Find valid numeric indices and values (non-null and numeric)
        let valid: Vec<(usize, f64)> = self
            .values()
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if v.is_null() {
                    None
                } else {
                    v.as_float().map(|f| (i, f))
                }
            })
            .collect();

        if valid.len() < 2 {
            // Not enough points to interpolate
            return Series::from_parts(self.name.clone(), result, self.index.clone());
        }

        // Only interpolate Null values, preserve non-null non-numeric values
        for (i, slot) in result.iter_mut().enumerate().take(n) {
            if slot.is_null() {
                // Find surrounding valid numeric points
                let prev = valid.iter().rev().find(|(idx, _)| *idx < i);
                let next = valid.iter().find(|(idx, _)| *idx > i);

                match (prev, next) {
                    (Some(&(i0, v0)), Some(&(i1, v1))) => {
                        // Linear interpolation
                        let t = (i - i0) as f64 / (i1 - i0) as f64;
                        *slot = Value::Float(v0 + t * (v1 - v0));
                    }
                    (Some(&(_, v)), None) => {
                        // Extrapolate forward (use last value)
                        *slot = Value::Float(v);
                    }
                    (None, Some(_)) => {
                        // A leading null has no earlier value: it stays null, as in pandas
                        // (linear interpolation, limit_direction "forward").
                    }
                    (None, None) => {
                        // No valid values
                    }
                }
            }
            // Non-null values (including non-numeric) are preserved as-is
        }

        Series::from_parts(self.name.clone(), result, self.index.clone())
    }

    /// Nearest value interpolation.
    ///
    /// Only interpolates Null values. Non-null values (including non-numeric) are preserved.
    fn interpolate_nearest(&self) -> Series {
        let n = self.values().len();
        let mut result = self.values().to_vec();

        // Find valid numeric indices and values (non-null and numeric)
        let valid: Vec<(usize, f64)> = self
            .values()
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if v.is_null() {
                    None
                } else {
                    v.as_float().map(|f| (i, f))
                }
            })
            .collect();

        if valid.is_empty() {
            return Series::from_parts(self.name.clone(), result, self.index.clone());
        }

        // Only interpolate Null values
        for (i, slot) in result.iter_mut().enumerate().take(n) {
            if slot.is_null() {
                // Find nearest valid numeric point
                let nearest = valid
                    .iter()
                    .min_by_key(|(idx, _)| (*idx as isize - i as isize).unsigned_abs())
                    .map(|(_, v)| *v);

                if let Some(v) = nearest {
                    *slot = Value::Float(v);
                }
            }
            // Non-null values are preserved as-is
        }

        Series::from_parts(self.name.clone(), result, self.index.clone())
    }

    /// Count null values.
    pub fn null_count(&self) -> usize {
        self.values().iter().filter(|v| v.is_null()).count()
    }

    /// Check if any values are null.
    pub fn has_nulls(&self) -> bool {
        self.values().iter().any(|v| v.is_null())
    }

    /// Apply a function to each value.
    ///
    /// # Example
    /// ```
    /// use frame::frame::{Series, Value};
    ///
    /// let s = Series::new(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    /// let doubled = s.apply(|v| Value::Float(v.as_float().unwrap_or(0.0) * 2.0));
    /// assert_eq!(doubled.iloc(2).unwrap(), &Value::Float(6.0));
    /// ```
    pub fn apply<F>(&self, f: F) -> Series
    where
        F: Fn(&Value) -> Value,
    {
        Series::from_parts(
            self.name.clone(),
            self.values().iter().map(f).collect(),
            self.index.clone(),
        )
    }
}
