//! # arraykit
//!
//! N-dimensional arrays (NumPy-equivalent).

pub mod array;

// Re-export commonly used types
pub use array::{DType, NdArray, Shape};
