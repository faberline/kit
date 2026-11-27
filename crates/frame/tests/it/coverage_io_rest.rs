//! Coverage tests for the rest of `crates/frame`: binary, CSV, workbook, `abs` and interpolation.
//! Tests only: every expected value is written out by hand from the current behavior of the code.

use frame::frame::io::{binary, write_csv, write_csv_with_options, CsvOptions, Workbook};
use frame::frame::{DataFrame, FrameError, Series, Value};

/// A unique path in the temporary directory, one per test name.
fn temp_file(test: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kit_coverage_io_rest_{}_{}.tmp",
        test,
        std::process::id()
    ))
}

// ---- binary -------------------------------------------------------------

#[test]
fn write_columnar_all_null_column_roundtrip() {
    let df =
        DataFrame::from_columns(vec![("a", Series::new(vec![Value::Null, Value::Null]))]).unwrap();
    let mut buf = Vec::new();
    binary::write_columnar(&df, &mut buf).unwrap();
    let back = binary::read_columnar(&mut buf.as_slice()).unwrap();
    assert_eq!(back.get("a").unwrap().values(), &[Value::Null, Value::Null]);
}

#[test]
fn write_columnar_mixed_string_column_writes_other_values_as_text() {
    let df = DataFrame::from_columns(vec![(
        "a",
        Series::new(vec![Value::String("a".into()), Value::Int(7)]),
    )])
    .unwrap();
    let mut buf = Vec::new();
    binary::write_columnar(&df, &mut buf).unwrap();
    let back = binary::read_columnar(&mut buf.as_slice()).unwrap();
    assert_eq!(
        back.get("a").unwrap().values(),
        &[Value::String("a".into()), Value::String("7".into())]
    );
}

#[test]
fn write_columnar_bool_column_with_null_roundtrip() {
    let df = DataFrame::from_columns(vec![(
        "a",
        Series::new(vec![Value::Bool(true), Value::Null]),
    )])
    .unwrap();
    let mut buf = Vec::new();
    binary::write_columnar(&df, &mut buf).unwrap();
    let back = binary::read_columnar(&mut buf.as_slice()).unwrap();
    assert_eq!(
        back.get("a").unwrap().values(),
        &[Value::Bool(true), Value::Null]
    );
}

#[test]
fn write_columnar_string_column_with_null_roundtrip() {
    let df = DataFrame::from_columns(vec![(
        "a",
        Series::new(vec![Value::String("x".into()), Value::Null]),
    )])
    .unwrap();
    let mut buf = Vec::new();
    binary::write_columnar(&df, &mut buf).unwrap();
    let back = binary::read_columnar(&mut buf.as_slice()).unwrap();
    assert_eq!(
        back.get("a").unwrap().values(),
        &[Value::String("x".into()), Value::Null]
    );
}

#[test]
fn read_columnar_bad_magic_is_invalid_operation() {
    let mut bytes: &[u8] = b"XXXX\x01";
    match binary::read_columnar(&mut bytes) {
        Err(FrameError::InvalidOperation(m)) => assert!(m.contains("bad magic"), "got: {m}"),
        other => panic!("expected InvalidOperation, got {other:?}"),
    }
}

#[test]
fn read_columnar_unsupported_version_is_invalid_operation() {
    let mut bytes: &[u8] = b"CCLF\x09";
    match binary::read_columnar(&mut bytes) {
        Err(FrameError::InvalidOperation(m)) => {
            assert_eq!(m, "unsupported version: 9")
        }
        other => panic!("expected InvalidOperation, got {other:?}"),
    }
}

#[test]
fn read_columnar_truncated_file_is_io_error() {
    let df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2]))]).unwrap();
    let mut buf = Vec::new();
    binary::write_columnar(&df, &mut buf).unwrap();
    let mut short: &[u8] = &buf[..buf.len() - 1];
    assert!(matches!(
        binary::read_columnar(&mut short),
        Err(FrameError::IoError(_))
    ));
}

// ---- CSV ----------------------------------------------------------------

#[test]
fn write_csv_bool_column_writes_true_and_false() {
    let df = DataFrame::from_columns(vec![(
        "b",
        Series::new(vec![Value::Bool(true), Value::Bool(false)]),
    )])
    .unwrap();
    let path = temp_file("write_csv_bool");
    write_csv(&df, &path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(text, "b\ntrue\nfalse\n");
}

#[test]
fn write_csv_without_header_writes_only_rows() {
    let df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2]))]).unwrap();
    let path = temp_file("write_csv_no_header");
    let options = CsvOptions {
        has_header: false,
        ..CsvOptions::default()
    };
    write_csv_with_options(&df, &path, options).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(text, "1\n2\n");
}

#[test]
fn dataframe_to_csv_and_read_csv_roundtrip() {
    let df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2]))]).unwrap();
    let path = temp_file("dataframe_csv_roundtrip");
    df.to_csv(&path).unwrap();
    let back = DataFrame::read_csv(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(back.nrows(), 2);
    assert_eq!(
        back.get("a").unwrap().values(),
        &[Value::Int(1), Value::Int(2)]
    );
}

// ---- workbook -----------------------------------------------------------

#[test]
fn workbook_default_is_empty() {
    assert_eq!(Workbook::default().nsheets(), 0);
}

#[test]
fn workbook_from_string_without_sheet_marker_is_empty() {
    let wb = Workbook::from_string("a,b\n1,2\n").unwrap();
    assert_eq!(wb.nsheets(), 0);
}

#[test]
fn workbook_sheet_without_rows_is_empty_frame() {
    let wb = Workbook::from_string("---SHEET:empty---\n").unwrap();
    assert_eq!(wb.get_sheet("empty").unwrap().nrows(), 0);
}

#[test]
fn workbook_blank_data_line_is_skipped() {
    let wb = Workbook::from_string("---SHEET:s---\na\n1\n\n2\n").unwrap();
    let df = wb.get_sheet("s").unwrap();
    assert_eq!(df.nrows(), 2);
    assert_eq!(
        df.get("a").unwrap().values(),
        &[Value::Int(1), Value::Int(2)]
    );
}

#[test]
fn workbook_short_row_is_padded_with_null() {
    let wb = Workbook::from_string("---SHEET:s---\na,b\n1\n").unwrap();
    let df = wb.get_sheet("s").unwrap();
    assert_eq!(df.get("a").unwrap().values(), &[Value::Int(1)]);
    assert_eq!(df.get("b").unwrap().values(), &[Value::Null]);
}

#[test]
fn workbook_null_token_reads_as_null() {
    let wb = Workbook::from_string("---SHEET:s---\na\nnull\n").unwrap();
    let df = wb.get_sheet("s").unwrap();
    assert_eq!(df.get("a").unwrap().values(), &[Value::Null]);
}

#[test]
fn workbook_format_null_and_bool_as_text() {
    let df = DataFrame::from_columns(vec![(
        "a",
        Series::new(vec![Value::Bool(true), Value::Null]),
    )])
    .unwrap();
    let mut wb = Workbook::new();
    wb.add_sheet("s", df);
    assert_eq!(wb.to_string().unwrap(), "---SHEET:s---\na\ntrue\n\n");
}

#[test]
fn workbook_format_quotes_comma_and_quote_characters() {
    let df = DataFrame::from_columns(vec![(
        "a",
        Series::new(vec![
            Value::String("a,b".into()),
            Value::String("say \"hi\"".into()),
        ]),
    )])
    .unwrap();
    let mut wb = Workbook::new();
    wb.add_sheet("s", df);
    assert_eq!(
        wb.to_string().unwrap(),
        "---SHEET:s---\na\n\"a,b\"\n\"say \"\"hi\"\"\"\n"
    );
}

// ---- series -------------------------------------------------------------

#[test]
fn abs_negative_float_becomes_positive() {
    let s = Series::new(vec![-2.5, 1.0]).abs();
    assert_eq!(s.values(), &[Value::Float(2.5), Value::Float(1.0)]);
}

// ---- interpolation ------------------------------------------------------

#[test]
fn interpolate_linear_with_one_number_keeps_nulls() {
    let s = Series::new(vec![Value::Null, Value::Float(2.0), Value::Null]);
    let out = s.interpolate_method("linear");
    assert_eq!(out.values(), &[Value::Null, Value::Float(2.0), Value::Null]);
}

#[test]
fn interpolate_linear_leading_null_stays_null() {
    let s = Series::new(vec![Value::Null, Value::Float(1.0), Value::Float(3.0)]);
    let out = s.interpolate_method("linear");
    assert_eq!(
        out.values(),
        &[Value::Null, Value::Float(1.0), Value::Float(3.0)]
    );
}

#[test]
fn interpolate_nearest_all_null_is_unchanged() {
    let s = Series::new(vec![Value::Null, Value::Null]);
    let out = s.interpolate_method("nearest");
    assert_eq!(out.values(), &[Value::Null, Value::Null]);
}
