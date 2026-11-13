//! # arraykit
//!
//! N-dimensional arrays (inspired by NumPy).

pub mod array;

// Re-export commonly used types
pub use array::{DType, NdArray, Shape};
