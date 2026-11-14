//! DType system for array element types.
//!
//! Supports core numeric types (f32, f64, i32, i64) and boolean.

use std::fmt;

/// Data type descriptor for array elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}

impl DType {
    /// Size in bytes of this dtype.
    pub fn size(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Bool => 1,
        }
    }

    /// NumPy-compatible string representation.
    pub fn numpy_str(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Bool => "bool",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.numpy_str())
    }
}

/// Trait for types that can be stored in an NdArray.
pub trait ArrayElement: Clone + Copy + Default + PartialEq + fmt::Debug + 'static {
    const DTYPE: DType;

    fn zero() -> Self;
    fn one() -> Self;
}

impl ArrayElement for f32 {
    const DTYPE: DType = DType::Float32;
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl ArrayElement for f64 {
    const DTYPE: DType = DType::Float64;
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl ArrayElement for i32 {
    const DTYPE: DType = DType::Int32;
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl ArrayElement for i64 {
    const DTYPE: DType = DType::Int64;
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl ArrayElement for bool {
    const DTYPE: DType = DType::Bool;
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
}

/// Element types that `NdArray::arange` supports.
pub trait RangeElement: ArrayElement + PartialOrd {
    /// Number of elements of `arange(start, stop, step)`. `step` is not zero.
    fn range_len(start: Self, stop: Self, step: Self) -> usize;

    /// The i-th element: `start + i * step`.
    fn range_nth(start: Self, step: Self, i: usize) -> Self;
}

impl RangeElement for i32 {
    fn range_len(start: Self, stop: Self, step: Self) -> usize {
        let (a, b, s) = (start as i128, stop as i128, step as i128);
        if s > 0 && a < b {
            ((b - a + s - 1) / s) as usize
        } else if s < 0 && a > b {
            ((a - b + (-s) - 1) / (-s)) as usize
        } else {
            0
        }
    }

    fn range_nth(start: Self, step: Self, i: usize) -> Self {
        (start as i128 + i as i128 * step as i128) as i32
    }
}

impl RangeElement for i64 {
    fn range_len(start: Self, stop: Self, step: Self) -> usize {
        let (a, b, s) = (start as i128, stop as i128, step as i128);
        if s > 0 && a < b {
            ((b - a + s - 1) / s) as usize
        } else if s < 0 && a > b {
            ((a - b + (-s) - 1) / (-s)) as usize
        } else {
            0
        }
    }

    fn range_nth(start: Self, step: Self, i: usize) -> Self {
        (start as i128 + i as i128 * step as i128) as i64
    }
}

impl RangeElement for f32 {
    fn range_len(start: Self, stop: Self, step: Self) -> usize {
        let n = ((stop as f64 - start as f64) / step as f64).ceil();
        if n.is_finite() && n > 0.0 {
            n as usize
        } else {
            0
        }
    }

    fn range_nth(start: Self, step: Self, i: usize) -> Self {
        start + (i as f32) * step
    }
}

impl RangeElement for f64 {
    fn range_len(start: Self, stop: Self, step: Self) -> usize {
        let n = ((stop - start) / step).ceil();
        if n.is_finite() && n > 0.0 {
            n as usize
        } else {
            0
        }
    }

    fn range_nth(start: Self, step: Self, i: usize) -> Self {
        start + (i as f64) * step
    }
}
