//! Common utilities for kit crates
//!
//! This crate provides shared functionality used across all kit crates.

pub mod error;
pub mod error_utils;
pub mod http;
pub mod utils;

pub use error::{KitError, Result};
pub use error_utils::{categorize_error, sanitize_error, sanitize_error_message, ErrorCategory};
pub use http::{HttpMethod, HttpRequestLike, HttpResponseLike, HttpStatus};
