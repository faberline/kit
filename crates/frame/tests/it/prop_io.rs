//! Property tests for the CSV and JSON readers.

use frame::frame::io::{read_csv, read_csv_with_options, write_csv, CsvOptions};
use frame::{DataFrame, Series};
use proptest::prelude::*;
use std::io::Write;

/// Rows of an int, a float and a lower-case text cell (no comma, quote or newline).
fn rows_strategy() -> impl Strategy<Value = Vec<(i64, f64, String)>> {
    prop::collection::vec((any::<i64>(), -1.0e6f64..1.0e6f64, "[a-z]{1,8}"), 1..20)
}

fn temp_file_with(text: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(text.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn csv_reader_never_panics(text in ".{0,2000}") {
        let f = temp_file_with(&text);
        let _ = read_csv(f.path());
        let _ = read_csv_with_options(
            f.path(),
            CsvOptions {
                delimiter: ';',
                has_header: false,
                quote: '\'',
                skip_blank_lines: false,
            },
        );
    }

    #[test]
    fn csv_round_trip_keeps_shape(rows in rows_strategy()) {
        let ints: Vec<i64> = rows.iter().map(|r| r.0).collect();
        let floats: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let texts: Vec<String> = rows.iter().map(|r| r.2.clone()).collect();
        let df = DataFrame::from_columns(vec![
            ("c0", Series::new(ints)),
            ("c1", Series::new(floats)),
            ("c2", Series::new(texts)),
        ])
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.csv");
        write_csv(&df, &path).unwrap();
        let back = read_csv(&path).unwrap();

        prop_assert_eq!(back.nrows(), rows.len());
        prop_assert_eq!(back.ncols(), 3);
    }
}

#[cfg(feature = "io-extra")]
mod json {
    use super::rows_strategy;
    use frame::frame::io::read_json_str;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn json_reader_never_panics(text in ".{0,2000}") {
            let _ = read_json_str(&text);
        }

        #[test]
        fn json_records_round_trip_keeps_shape(rows in rows_strategy()) {
            let records: Vec<serde_json::Value> = rows
                .iter()
                .map(|(i, f, t)| serde_json::json!({"c0": i, "c1": f, "c2": t}))
                .collect();
            let text = serde_json::Value::Array(records).to_string();

            let df = read_json_str(&text).map_err(|e| TestCaseError::fail(e.to_string()))?;

            prop_assert_eq!(df.nrows(), rows.len());
            prop_assert_eq!(df.ncols(), 3);
        }
    }
}
