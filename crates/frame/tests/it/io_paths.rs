//! The old and the new I/O module paths both compile and give the same result.

use frame::frame::io::{binary, columnar, sheets, workbook};
use frame::frame::{DataFrame, Series};

fn sample() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2, 3])),
        ("b", Series::new(vec![4, 5, 6])),
    ])
    .unwrap()
}

#[test]
fn old_columnar_path_equals_new_binary_path() {
    let df = sample();
    let mut old = Vec::new();
    columnar::write_columnar(&df, &mut old).unwrap();
    let mut new = Vec::new();
    binary::write_columnar(&df, &mut new).unwrap();
    assert_eq!(old, new);
    assert_eq!(&old[..4], b"CCLF");
    let back = columnar::read_columnar(&mut old.as_slice()).unwrap();
    assert_eq!(back.nrows(), 3);
}

#[test]
fn old_sheets_path_equals_new_workbook_path() {
    let mut wb: sheets::Workbook = workbook::Workbook::new();
    wb.add_sheet("s1", sample());
    let path = std::env::temp_dir().join("kit_frame_io_paths_workbook.txt");
    sheets::save_workbook(&wb, &path).unwrap();
    let back = workbook::load_workbook(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.nsheets(), 1);
    assert_eq!(back.sheet_names(), vec!["s1"]);
}
