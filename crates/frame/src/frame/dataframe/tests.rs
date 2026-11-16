use super::*;

fn create_test_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2, 3, 4, 5])),
        ("b", Series::new(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
        ("c", Series::new(vec!["x", "y", "z", "x", "y"])),
    ])
    .unwrap()
}

#[test]
fn test_dataframe_basic() {
    let df = create_test_df();
    assert_eq!(df.shape(), (5, 3));
    assert_eq!(df.columns(), &["a", "b", "c"]);
}

#[test]
fn test_dataframe_get_column() {
    let df = create_test_df();
    let col = df.get("a").unwrap();
    assert_eq!(col.len(), 5);
    assert_eq!(col.iloc(0).unwrap(), &Value::Int(1));
}

#[test]
fn test_dataframe_iloc() {
    let df = create_test_df();
    let row = df.iloc_row(1).unwrap();
    assert_eq!(row.get("a").unwrap(), &Value::Int(2));
    assert_eq!(row.get("b").unwrap(), &Value::Float(20.0));
}

#[test]
fn test_dataframe_filter() {
    let df = create_test_df();
    let mask = vec![true, false, true, false, true];
    let filtered = df.filter(&mask).unwrap();
    assert_eq!(filtered.nrows(), 3);
}

fn nulls_df() -> DataFrame {
    DataFrame::from_columns(vec![
        (
            "a",
            Series::new(vec![Value::Int(3), Value::Null, Value::Int(1)]),
        ),
        (
            "b",
            Series::new(vec![Value::Int(10), Value::Int(20), Value::Int(30)]),
        ),
    ])
    .unwrap()
}

#[test]
fn sort_values_nulls_last() {
    let df = nulls_df();

    let asc = df.sort_values("a", true).unwrap();
    assert_eq!(
        asc.get("a").unwrap().values(),
        &[Value::Int(1), Value::Int(3), Value::Null]
    );
    assert_eq!(
        asc.get("b").unwrap().values(),
        &[Value::Int(30), Value::Int(10), Value::Int(20)]
    );

    let desc = df.sort_values("a", false).unwrap();
    assert_eq!(
        desc.get("a").unwrap().values(),
        &[Value::Int(3), Value::Int(1), Value::Null]
    );
    assert_eq!(
        desc.get("b").unwrap().values(),
        &[Value::Int(10), Value::Int(30), Value::Int(20)]
    );
}

#[test]
fn sort_values_by_nulls_last() {
    let df = nulls_df();
    let sorted = df.sort_values_by(&["a"], &[true]).unwrap();
    assert_eq!(
        sorted.get("a").unwrap().values(),
        &[Value::Int(1), Value::Int(3), Value::Null]
    );
}

#[test]
fn test_dataframe_sort() {
    let df = create_test_df();
    let sorted = df.sort_values("a", false).unwrap();
    assert_eq!(sorted.get("a").unwrap().iloc(0).unwrap(), &Value::Int(5));
}

#[test]
fn test_dataframe_head_tail() {
    let df = create_test_df();
    let head = df.head(2).unwrap();
    assert_eq!(head.nrows(), 2);
    let tail = df.tail(2).unwrap();
    assert_eq!(tail.nrows(), 2);
    assert_eq!(tail.get("a").unwrap().iloc(0).unwrap(), &Value::Int(4));
}

#[test]
fn test_dataframe_select() {
    let df = create_test_df();
    let selected = df.select(&["a", "c"]).unwrap();
    assert_eq!(selected.ncols(), 2);
    assert_eq!(selected.columns(), &["a", "c"]);
}

#[test]
fn test_dataframe_isna() {
    let df = DataFrame::from_columns(vec![
        (
            "a",
            Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]),
        ),
        (
            "b",
            Series::new(vec![Value::Null, Value::Int(2), Value::Int(3)]),
        ),
    ])
    .unwrap();

    let isna = df.isna();
    assert_eq!(isna.get("a").unwrap(), &vec![false, true, false]);
    assert_eq!(isna.get("b").unwrap(), &vec![true, false, false]);
}

#[test]
fn test_dataframe_isna_any() {
    let df = DataFrame::from_columns(vec![
        (
            "a",
            Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]),
        ),
        (
            "b",
            Series::new(vec![Value::Null, Value::Int(2), Value::Int(3)]),
        ),
    ])
    .unwrap();

    let mask = df.isna_any();
    assert_eq!(mask, vec![true, true, false]);
}

#[test]
fn test_dataframe_fillna() {
    let df = DataFrame::from_columns(vec![
        (
            "a",
            Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]),
        ),
        (
            "b",
            Series::new(vec![Value::Float(1.0), Value::Null, Value::Float(3.0)]),
        ),
    ])
    .unwrap();

    let filled = df.fillna(Value::Int(0));
    assert_eq!(filled.get("a").unwrap().iloc(1).unwrap(), &Value::Int(0));
    assert_eq!(filled.get("b").unwrap().iloc(1).unwrap(), &Value::Int(0));
}

#[test]
fn test_dataframe_dropna() {
    let df = DataFrame::from_columns(vec![
        (
            "a",
            Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]),
        ),
        (
            "b",
            Series::new(vec![Value::Int(1), Value::Int(2), Value::Null]),
        ),
    ])
    .unwrap();

    let clean = df.dropna().unwrap();
    assert_eq!(clean.nrows(), 1);
    assert_eq!(clean.get("a").unwrap().iloc(0).unwrap(), &Value::Int(1));
}

#[test]
fn test_dataframe_apply() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1.0, 2.0, 3.0])),
        ("b", Series::new(vec![10.0, 20.0, 30.0])),
    ])
    .unwrap();

    let sums = df.apply(|col| col.sum().unwrap_or(0.0));
    assert_eq!(*sums.get("a").unwrap(), 6.0);
    assert_eq!(*sums.get("b").unwrap(), 60.0);
}

#[test]
fn test_dataframe_applymap() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1.0, 2.0])),
        ("b", Series::new(vec![3.0, 4.0])),
    ])
    .unwrap();

    let doubled = df.applymap(|v| Value::Float(v.as_float().unwrap_or(0.0) * 2.0));
    assert_eq!(
        doubled.get("a").unwrap().iloc(0).unwrap(),
        &Value::Float(2.0)
    );
    assert_eq!(
        doubled.get("b").unwrap().iloc(1).unwrap(),
        &Value::Float(8.0)
    );
}

#[test]
fn test_dataframe_concat() {
    let df1 = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2])),
        ("b", Series::new(vec![3, 4])),
    ])
    .unwrap();

    let df2 = DataFrame::from_columns(vec![
        ("a", Series::new(vec![5, 6])),
        ("b", Series::new(vec![7, 8])),
    ])
    .unwrap();

    let combined = DataFrame::concat(&[&df1, &df2]).unwrap();
    assert_eq!(combined.nrows(), 4);
    assert_eq!(combined.get("a").unwrap().iloc(2).unwrap(), &Value::Int(5));
}

#[test]
fn test_dataframe_from_records() {
    let records = vec![
        {
            let mut m = HashMap::new();
            m.insert("a".to_string(), Value::Int(1));
            m.insert("b".to_string(), Value::Int(2));
            m
        },
        {
            let mut m = HashMap::new();
            m.insert("a".to_string(), Value::Int(3));
            m.insert("b".to_string(), Value::Int(4));
            m
        },
    ];

    let df = DataFrame::from_records(&records).unwrap();
    assert_eq!(df.nrows(), 2);
    assert_eq!(df.ncols(), 2);
}

#[test]
fn test_dataframe_to_dict() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2])),
        ("b", Series::new(vec![3, 4])),
    ])
    .unwrap();

    let dict = df.to_dict();
    assert_eq!(dict.get("a").unwrap().len(), 2);
}

#[test]
fn test_dataframe_duplicated() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 1, 2, 2])),
        ("b", Series::new(vec![1, 1, 2, 3])),
    ])
    .unwrap();

    let dups = df.duplicated(None, "first").unwrap();
    assert_eq!(dups, vec![false, true, false, false]);
}

#[test]
fn test_dataframe_drop_duplicates() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 1, 2])),
        ("b", Series::new(vec![1, 1, 2])),
    ])
    .unwrap();

    let unique = df.drop_duplicates(None, "first").unwrap();
    assert_eq!(unique.nrows(), 2);
}

#[test]
fn test_dataframe_corr() {
    let df = DataFrame::from_columns(vec![
        ("x", Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        ("y", Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0])),
    ])
    .unwrap();

    let corr = df.corr().unwrap();
    assert_eq!(corr.nrows(), 2);
    // Perfect correlation
    let x_y_corr = corr.get("y").unwrap().iloc(0).unwrap().as_float().unwrap();
    assert!(f64::abs(x_y_corr - 1.0) < 1e-10);
}

#[test]
fn test_dataframe_melt() {
    let df = DataFrame::from_columns(vec![
        ("id", Series::new(vec!["a", "b"])),
        ("x", Series::new(vec![1, 2])),
        ("y", Series::new(vec![3, 4])),
    ])
    .unwrap();

    let melted = df.melt(&["id"], &["x", "y"]).unwrap();
    assert_eq!(melted.nrows(), 4);
    assert_eq!(melted.ncols(), 3); // id, variable, value
}

#[test]
fn test_dataframe_sample() {
    let df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2, 3, 4, 5]))]).unwrap();

    let sample = df.sample(3).unwrap();
    assert_eq!(sample.nrows(), 3);
}

/// Reads an integer column as a Vec, for exact comparisons.
fn int_col(df: &DataFrame, name: &str) -> Vec<i64> {
    let s = df.get(name).unwrap();
    (0..s.len())
        .map(|i| match s.iloc(i).unwrap() {
            Value::Int(v) => *v,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

#[test]
fn test_dataframe_sample_seeded() {
    let df = DataFrame::from_columns(vec![("a", Series::new((0..10i64).collect::<Vec<i64>>()))])
        .unwrap();

    assert_eq!(
        int_col(&df.sample_seeded(3, 1).unwrap(), "a"),
        vec![4, 3, 6]
    );
    assert_eq!(
        int_col(&df.sample_seeded(3, 2).unwrap(), "a"),
        vec![0, 2, 6]
    );
    assert_eq!(
        int_col(&df.sample_seeded(3, 1).unwrap(), "a"),
        int_col(&df.sample_seeded(3, 1).unwrap(), "a")
    );
    assert_eq!(
        int_col(&df.sample(3).unwrap(), "a"),
        int_col(&df.sample_seeded(3, 42).unwrap(), "a")
    );
    assert_eq!(int_col(&df.sample(3).unwrap(), "a"), vec![4, 6, 8]);
    assert_eq!(
        int_col(&df.sample_seeded(10, 7).unwrap(), "a"),
        vec![8, 1, 3, 5, 9, 4, 6, 2, 0, 7]
    );
    assert_eq!(df.sample_seeded(0, 1).unwrap().nrows(), 0);
    assert_eq!(df.sample_seeded(99, 1).unwrap().nrows(), 10);
}

#[test]
fn test_dataframe_pivot_column_names() {
    let df = DataFrame::from_columns(vec![
        ("row", Series::new(vec!["a", "a", "b", "b"])),
        ("col", Series::new(vec!["x", "y", "x", "y"])),
        ("val", Series::new(vec![1.0, 2.0, 3.0, 4.0])),
    ])
    .unwrap();

    let pivoted = df.pivot("row", "col", "val").unwrap();
    let expected: Vec<String> = vec!["row".into(), "x".into(), "y".into()];
    assert_eq!(pivoted.columns(), expected.as_slice());
    assert_eq!(pivoted.nrows(), 2);
    assert_eq!(
        pivoted.get("x").unwrap().iloc(0).unwrap(),
        &Value::Float(1.0)
    );
    assert_eq!(
        pivoted.get("x").unwrap().iloc(1).unwrap(),
        &Value::Float(3.0)
    );
}

#[test]
fn test_dataframe_assign() {
    let df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2, 3]))]).unwrap();

    let df2 = df.assign(vec![("b", Series::new(vec![4, 5, 6]))]).unwrap();
    assert_eq!(df2.ncols(), 2);
    assert!(df2.get("b").is_ok());
}

#[test]
fn test_dataframe_describe() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        ("b", Series::new(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
    ])
    .unwrap();

    let desc = df.describe();
    assert!(desc.contains_key("a"));
    assert!(desc.contains_key("b"));
    assert_eq!(desc.get("a").unwrap().get("count"), Some(&5.0));
    assert_eq!(desc.get("a").unwrap().get("mean"), Some(&3.0));
}

#[test]
fn test_dataframe_query() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2, 3, 4, 5])),
        ("b", Series::new(vec![10, 20, 30, 40, 50])),
    ])
    .unwrap();

    let result = df
        .query(|row| row.get("a").and_then(|v| v.as_int()).unwrap_or(0) > 2)
        .unwrap();
    assert_eq!(result.nrows(), 3); // rows where a > 2
}

#[test]
fn test_dataframe_sort_values_by() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![2, 1, 2, 1])),
        ("b", Series::new(vec![20, 10, 10, 20])),
    ])
    .unwrap();

    let sorted = df.sort_values_by(&["a", "b"], &[true, true]).unwrap();
    assert_eq!(sorted.nrows(), 4);
    // Should be sorted by a then b
}

#[test]
fn test_dataframe_reset_index() {
    let mut df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2, 3]))]).unwrap();

    df.reset_index();
    // Index should be range
    assert_eq!(df.nrows(), 3);
}

#[test]
fn test_dataframe_iterrows() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2])),
        ("b", Series::new(vec![3, 4])),
    ])
    .unwrap();

    let rows: Vec<_> = df.iterrows().collect();
    assert_eq!(rows.len(), 2);
    assert!(rows[0].1.contains_key("a"));
    assert!(rows[0].1.contains_key("b"));
}

#[test]
fn test_dataframe_isna_all() {
    let df = DataFrame::from_columns(vec![
        (
            "a",
            Series::new(vec![Value::Null, Value::Int(2), Value::Null]),
        ),
        (
            "b",
            Series::new(vec![Value::Null, Value::Int(3), Value::Int(4)]),
        ),
    ])
    .unwrap();

    let mask = df.isna_all();
    assert_eq!(mask, vec![true, false, false]); // Only first row has all nulls
}

#[test]
fn test_dataframe_fillna_column() {
    let df = DataFrame::from_columns(vec![
        (
            "a",
            Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]),
        ),
        (
            "b",
            Series::new(vec![Value::Float(1.0), Value::Null, Value::Float(3.0)]),
        ),
    ])
    .unwrap();

    let filled = df.fillna_column("a", Value::Int(99)).unwrap();
    assert_eq!(filled.get("a").unwrap().iloc(1).unwrap(), &Value::Int(99));
    // b should still have null
    assert!(filled.get("b").unwrap().iloc(1).unwrap().is_null());
}

#[test]
fn test_dataframe_dropna_subset() {
    let df = DataFrame::from_columns(vec![
        (
            "a",
            Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]),
        ),
        (
            "b",
            Series::new(vec![Value::Int(1), Value::Int(2), Value::Null]),
        ),
    ])
    .unwrap();

    let clean = df.dropna_subset(&["a"]).unwrap();
    assert_eq!(clean.nrows(), 2); // Only row with null in 'a' dropped
}

#[test]
fn test_dataframe_col() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2, 3])),
        ("b", Series::new(vec![4, 5, 6])),
    ])
    .unwrap();

    let col_a = df.col("a").unwrap();
    assert_eq!(col_a.len(), 3);
}

#[test]
fn test_dataframe_rename() {
    let mut df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2, 3]))]).unwrap();

    let mut mapping = HashMap::new();
    mapping.insert("a", "x");
    df.rename(mapping);
    assert!(df.get("x").is_ok());
    assert!(df.get("a").is_err());
}

#[test]
fn test_dataframe_loc() {
    let mut df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2, 3])),
        ("b", Series::new(vec![4, 5, 6])),
    ])
    .unwrap();

    let _ = df.set_index(Index::String(vec!["x".into(), "y".into(), "z".into()]));
    let rows = df.loc(&[Value::String("y".into())]).unwrap();
    assert_eq!(rows.nrows(), 1);
    assert!(rows.get("a").is_ok());
}

#[test]
fn test_dataframe_set_index() {
    let mut df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2, 3]))]).unwrap();

    let _ = df.set_index(Index::String(vec!["x".into(), "y".into(), "z".into()]));
    assert_eq!(df.nrows(), 3);
}

#[test]
fn test_dataframe_to_records() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2])),
        ("b", Series::new(vec![3, 4])),
    ])
    .unwrap();

    let records = df.to_records();
    assert_eq!(records.len(), 2);
    assert!(records[0].contains_key("a"));
}

#[test]
fn test_dataframe_pivot() {
    let df = DataFrame::from_columns(vec![
        ("row", Series::new(vec!["a", "a", "b", "b"])),
        ("col", Series::new(vec!["x", "y", "x", "y"])),
        ("val", Series::new(vec![1.0, 2.0, 3.0, 4.0])),
    ])
    .unwrap();

    let pivoted = df.pivot("row", "col", "val").unwrap();
    assert!(pivoted.ncols() >= 2); // At least row + some value columns
}

#[test]
fn test_dataframe_from_map() {
    let mut data = HashMap::new();
    data.insert("a", vec![1, 2, 3]);
    data.insert("b", vec![4, 5, 6]);
    let df = DataFrame::from_map(data).unwrap();
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.ncols(), 2);
}

#[test]
fn test_dataframe_is_empty() {
    let df = DataFrame::from_columns(vec![]).unwrap();
    assert!(df.is_empty());

    let df2 = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2, 3]))]).unwrap();
    assert!(!df2.is_empty());
}

#[test]
fn test_dataframe_index() {
    let df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2, 3]))]).unwrap();
    assert_eq!(df.index().len(), 3);
}

#[test]
fn test_dataframe_get_columns() {
    let df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2, 3])),
        ("b", Series::new(vec![4, 5, 6])),
        ("c", Series::new(vec![7, 8, 9])),
    ])
    .unwrap();

    let subset = df.get_columns(&["a", "c"]).unwrap();
    assert_eq!(subset.ncols(), 2);
    assert!(subset.get("a").is_ok());
    assert!(subset.get("c").is_ok());
    assert!(subset.get("b").is_err());
}

#[test]
fn test_dataframe_insert() {
    let mut df = DataFrame::from_columns(vec![("a", Series::new(vec![1, 2, 3]))]).unwrap();

    df.insert("b", Series::new(vec![4, 5, 6])).unwrap();
    assert_eq!(df.ncols(), 2);
    assert!(df.get("b").is_ok());
}

#[test]
fn test_dataframe_drop() {
    let mut df = DataFrame::from_columns(vec![
        ("a", Series::new(vec![1, 2, 3])),
        ("b", Series::new(vec![4, 5, 6])),
    ])
    .unwrap();

    let dropped = df.drop("b").unwrap();
    assert_eq!(dropped.len(), 3);
    assert_eq!(df.ncols(), 1);
    assert!(df.get("b").is_err());
}
