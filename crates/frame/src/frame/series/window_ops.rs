//! Cumulative, shift, diff and rolling-window operations for Series.

use super::super::value::Value;
use super::Series;

impl Series {
    // === Cumulative operations ===

    /// Cumulative sum.
    pub fn cumsum(&self) -> Series {
        let mut acc = 0.0;
        let data: Vec<Value> = self
            .values()
            .iter()
            .map(|v| {
                if let Some(val) = v.as_float() {
                    acc += val;
                    Value::Float(acc)
                } else {
                    Value::Null
                }
            })
            .collect();

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    /// Cumulative product.
    pub fn cumprod(&self) -> Series {
        let mut acc = 1.0;
        let data: Vec<Value> = self
            .values()
            .iter()
            .map(|v| {
                if let Some(val) = v.as_float() {
                    acc *= val;
                    Value::Float(acc)
                } else {
                    Value::Null
                }
            })
            .collect();

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    /// Cumulative maximum.
    pub fn cummax(&self) -> Series {
        let mut max_val: Option<f64> = None;
        let data: Vec<Value> = self
            .values()
            .iter()
            .map(|v| {
                if let Some(val) = v.as_float() {
                    max_val = Some(match max_val {
                        None => val,
                        Some(m) => {
                            if val > m {
                                val
                            } else {
                                m
                            }
                        }
                    });
                    Value::Float(max_val.unwrap())
                } else {
                    Value::Null
                }
            })
            .collect();

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    /// Cumulative minimum.
    pub fn cummin(&self) -> Series {
        let mut min_val: Option<f64> = None;
        let data: Vec<Value> = self
            .values()
            .iter()
            .map(|v| {
                if let Some(val) = v.as_float() {
                    min_val = Some(match min_val {
                        None => val,
                        Some(m) => {
                            if val < m {
                                val
                            } else {
                                m
                            }
                        }
                    });
                    Value::Float(min_val.unwrap())
                } else {
                    Value::Null
                }
            })
            .collect();

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    /// Shift values by n periods.
    ///
    /// Positive n shifts down (introduces NaN at the beginning).
    /// Negative n shifts up (introduces NaN at the end).
    pub fn shift(&self, n: isize) -> Series {
        let len = self.values().len();
        let mut data = vec![Value::Null; len];

        if n >= 0 {
            let n = n as usize;
            // A shift past the end leaves every value null.
            if n < len {
                data[n..].clone_from_slice(&self.values()[..len - n]);
            }
        } else {
            let n = (-n) as usize;
            let m = len.saturating_sub(n);
            if m > 0 {
                data[..m].clone_from_slice(&self.values()[n..n + m]);
            }
        }

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    /// Compute difference between consecutive elements.
    ///
    /// First element will be Null.
    pub fn diff(&self, periods: usize) -> Series {
        let len = self.values().len();
        let mut data = vec![Value::Null; len];

        for (i, slot) in data.iter_mut().enumerate().take(len).skip(periods) {
            match (
                self.values()[i].as_float(),
                self.values()[i - periods].as_float(),
            ) {
                (Some(curr), Some(prev)) => {
                    *slot = Value::Float(curr - prev);
                }
                _ => {
                    *slot = Value::Null;
                }
            }
        }

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    /// Compute percentage change between consecutive elements.
    pub fn pct_change(&self, periods: usize) -> Series {
        let len = self.values().len();
        let mut data = vec![Value::Null; len];

        for (i, slot) in data.iter_mut().enumerate().take(len).skip(periods) {
            match (
                self.values()[i].as_float(),
                self.values()[i - periods].as_float(),
            ) {
                (Some(curr), Some(prev)) if prev != 0.0 => {
                    *slot = Value::Float((curr - prev) / prev);
                }
                _ => {
                    *slot = Value::Null;
                }
            }
        }

        Series::from_parts(self.name.clone(), data, self.index.clone())
    }

    // === Rolling window ===

    /// Create a rolling window.
    pub fn rolling(&self, window: usize) -> SeriesRolling<'_> {
        SeriesRolling {
            series: self,
            window,
            min_periods: window,
        }
    }
}

/// Rolling window for Series.
pub struct SeriesRolling<'a> {
    series: &'a Series,
    window: usize,
    min_periods: usize,
}

impl<'a> SeriesRolling<'a> {
    /// Set minimum periods.
    pub fn min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = min_periods;
        self
    }

    /// Rolling sum.
    pub fn sum(&self) -> Series {
        self.apply(|vals| vals.iter().sum())
    }

    /// Rolling mean.
    pub fn mean(&self) -> Series {
        self.apply(|vals| vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// Rolling min.
    pub fn min(&self) -> Series {
        self.apply(|vals| vals.iter().cloned().fold(f64::INFINITY, f64::min))
    }

    /// Rolling max.
    pub fn max(&self) -> Series {
        self.apply(|vals| vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }

    /// Rolling std.
    pub fn std(&self) -> Series {
        self.apply(|vals| {
            if vals.len() < 2 {
                return f64::NAN;
            }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var =
                vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
            var.sqrt()
        })
    }

    /// Rolling var.
    pub fn var(&self) -> Series {
        self.apply(|vals| {
            if vals.len() < 2 {
                return f64::NAN;
            }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64
        })
    }

    /// Apply a function to rolling windows.
    fn apply<F>(&self, f: F) -> Series
    where
        F: Fn(&[f64]) -> f64,
    {
        let vals: Vec<f64> = self
            .series
            .values()
            .iter()
            .map(|v| v.as_float().unwrap_or(f64::NAN))
            .collect();

        let n = vals.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            if i + 1 < self.min_periods {
                result.push(Value::Null);
            } else {
                let start = i.saturating_sub(self.window - 1);
                let window_vals: Vec<f64> = vals[start..=i]
                    .iter()
                    .filter(|x| !x.is_nan())
                    .cloned()
                    .collect();

                if window_vals.len() < self.min_periods {
                    result.push(Value::Null);
                } else {
                    result.push(Value::Float(f(&window_vals)));
                }
            }
        }

        Series::from_parts(self.series.name.clone(), result, self.series.index.clone())
    }
}
