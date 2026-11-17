//! IO operations for DataFrame.

pub mod binary;
mod csv;
pub mod workbook;

#[cfg(feature = "io-extra")]
pub mod json;

/// Old path of the `workbook` module. Kept so that existing code compiles.
pub mod sheets {
    pub use super::workbook::{load_workbook, save_workbook, Workbook};
}

/// Old path of the `binary` module. Kept so that existing code compiles.
pub mod columnar {
    pub use super::binary::{load_columnar, read_columnar, save_columnar, write_columnar};
}

pub use binary::{load_columnar, read_columnar, save_columnar, write_columnar};
pub use csv::{read_csv, read_csv_with_options, write_csv, write_csv_with_options, CsvOptions};
pub use workbook::Workbook;

#[cfg(feature = "io-extra")]
pub use json::{read_json, read_json_str, save_json, write_json_str, JsonOptions, JsonOrient};
