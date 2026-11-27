//! CSV rows whose field count differs from the header: a short row is an error, a long row drops the extra fields.

use frame::frame::io::{read_csv, read_csv_with_options, CsvOptions};
use frame::frame::{FrameError, Value};

/// A unique path in the temporary directory, one per test name.
fn temp_file(test: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kit_csv_short_row_{}_{}.tmp",
        test,
        std::process::id()
    ))
}

#[test]
fn short_row_is_shape_mismatch() {
    let path = temp_file("short_row_is_shape_mismatch");
    std::fs::write(&path, "a,b,c\n1,2\n").unwrap();
    let result = read_csv(&path);
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(
        result,
        Err(FrameError::ShapeMismatch {
            expected: 1,
            actual: 0
        })
    ));
}

#[test]
fn short_row_after_a_full_row_is_shape_mismatch() {
    let path = temp_file("short_row_after_a_full_row_is_shape_mismatch");
    std::fs::write(&path, "a,b,c\n1,2,3\n4,5\n").unwrap();
    let result = read_csv(&path);
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(
        result,
        Err(FrameError::ShapeMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn short_row_without_header_is_not_padded() {
    let path = temp_file("short_row_without_header_is_not_padded");
    std::fs::write(&path, "1,2,3\n4,5\n").unwrap();
    let options = CsvOptions {
        has_header: false,
        ..CsvOptions::default()
    };
    let result = read_csv_with_options(&path, options);
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(
        result,
        Err(FrameError::ShapeMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn long_row_drops_the_extra_field() {
    let path = temp_file("long_row_drops_the_extra_field");
    std::fs::write(&path, "a,b\n1,2,3\n").unwrap();
    let result = read_csv(&path);
    std::fs::remove_file(&path).unwrap();
    let df = result.unwrap();
    assert_eq!(df.shape(), (1, 2));
    assert!(matches!(df.get("a").unwrap().values(), [Value::Int(1)]));
    assert!(matches!(df.get("b").unwrap().values(), [Value::Int(2)]));
}
