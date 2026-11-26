//! Reference-value tests. Expected values come from pandas 2.2 / NumPy 1.26;
//! the expression that made each value is in the comment above it. CI does not run Python.

use frame::frame::io::{read_csv, write_csv};
use frame::frame::{Series, Value};
use std::io::Write;
use tempfile::NamedTempFile;

/// Maps the values of a series to `Some(i)` for `Value::Int(i)` and `None` for `Value::Null`.
fn ints(s: &Series) -> Vec<Option<i64>> {
    s.values()
        .iter()
        .map(|v| match v {
            Value::Int(i) => Some(*i),
            Value::Null => None,
            other => panic!("expected Int or Null, got {:?}", other),
        })
        .collect()
}

// pandas 2.2: pd.read_csv on 'a,b\n"l1\nl2",1\n' -> a = "l1\nl2", b = 1
#[test]
fn read_csv_quoted_newline_round_trips() {
    let mut src = NamedTempFile::new().unwrap();
    src.write_all(b"a,b\n\"l1\nl2\",1\n").unwrap();
    src.flush().unwrap();

    let df = read_csv(src.path()).unwrap();
    assert_eq!(df.nrows(), 1);
    assert_eq!(
        df.get("a").unwrap().values()[0],
        Value::String("l1\nl2".to_string())
    );
    assert_eq!(df.get("b").unwrap().values()[0], Value::Int(1));

    let out = NamedTempFile::new().unwrap();
    write_csv(&df, out.path()).unwrap();
    let again = read_csv(out.path()).unwrap();
    assert_eq!(again.nrows(), 1);
    assert_eq!(
        again.get("a").unwrap().values()[0],
        Value::String("l1\nl2".to_string())
    );
    assert_eq!(again.get("b").unwrap().values()[0], Value::Int(1));
}

// pandas 2.2: pd.read_csv(..., skip_blank_lines=False) on 'n\n1\n\n3\n' -> [1, NaN, 3]
#[test]
fn read_csv_one_column_blank_row_is_null() {
    let mut src = NamedTempFile::new().unwrap();
    src.write_all(b"n\n1\n\n3\n").unwrap();
    src.flush().unwrap();

    let df = read_csv(src.path()).unwrap();
    assert_eq!(ints(df.get("n").unwrap()), vec![Some(1), None, Some(3)]);
}

// pandas 2.2: pd.read_csv(..., encoding='utf-8-sig') on a UTF-8 BOM file -> columns [a, b]
#[test]
fn read_csv_strips_utf8_bom() {
    let mut src = NamedTempFile::new().unwrap();
    src.write_all(b"\xEF\xBB\xBFa,b\n1,2\n").unwrap();
    src.flush().unwrap();

    let df = read_csv(src.path()).unwrap();
    let cols: Vec<&str> = df.columns().iter().map(String::as_str).collect();
    assert_eq!(cols, ["a", "b"]);
}
