//! DataFrame module (Pandas-like).
//!
//! # Null policy
//!
//! - Statistics (`sum`, `mean`, `std`, `var`, `median`, `quantile`, `describe`)
//!   skip `Value::Null`.
//! - A non-null value that is not numeric (a `String`) is `FrameError::TypeMismatch`.
//! - `sum` of no non-null values is `0.0`. `mean`, `std`, `var`, `median` and
//!   `quantile` of no non-null values are `FrameError::InvalidOperation`.
//! - Sort places nulls last when ascending and first when descending
//!   (later work; not yet in the code).
//! - Group-by drops rows whose key is null (later work; not yet in the code).
//! - CSV: an empty field is `Null` (later work; not yet in the code).

mod dataframe;
mod error;
mod index;
pub mod io;
pub mod multi_index;
pub mod ops;
mod series;
mod value;

pub use dataframe::DataFrame;
pub use error::{FrameError, Result};
pub use index::Index;
pub use io::{read_csv, write_csv, CsvOptions, Workbook};
pub use multi_index::MultiIndex;

#[cfg(feature = "io-extra")]
pub use io::{read_json, read_json_str, save_json, write_json_str, JsonOptions, JsonOrient};
pub use ops::{AggFunc, GroupBy, JoinType};
pub use series::Series;
pub use value::Value;
