//! # scikit
//!
//! Scientific computing: statistics, FFT, signal processing,
//! interpolation, optimization, time-series analysis, spatial
//! algorithms, sparse matrices, numerical integration, and linear models.

#[cfg(feature = "stats")]
pub mod stats;

#[cfg(feature = "fft")]
pub mod fft;

#[cfg(feature = "signal")]
pub mod signal;

#[cfg(feature = "interpolate")]
pub mod interpolate;

#[cfg(feature = "optimize")]
pub mod optimize;

#[cfg(feature = "ts")]
pub mod ts;

#[cfg(feature = "spatial")]
pub mod spatial;

#[cfg(feature = "sparse")]
pub mod sparse;

#[cfg(feature = "integrate")]
pub mod integrate;

#[cfg(feature = "linear_model")]
pub mod linear_model;
