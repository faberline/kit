//! Reference tests for the reshape functions of `frame::frame/ops/reshape`.
//! Each expected value is read from `fixtures/reshape_reference.tsv`, which
//! `tools/reference/frame/generate.py` writes from pandas 3.1.1.
//! The inputs below are the same integer formulas as in the generator.

use frame::frame::ops::reshape::{cross_tab, pivot_table, stack, unstack, wide_to_long};
use frame::frame::ops::AggFunc;
use frame::frame::{DataFrame, Series, Value};

const FIXTURE: &str = include_str!("../fixtures/reshape_reference.tsv");

fn raw(key: &str) -> &'static str {
    for line in FIXTURE.lines().skip(1) {
        let mut fields = line.split('\t');
        if fields.next() == Some(key) {
            return fields
                .next()
                .unwrap_or_else(|| panic!("fixture key {key}: no value column"));
        }
    }
    panic!("fixture key {key} not found");
}

fn reference(key: &str) -> f64 {
    let value = raw(key);
    value
        .parse()
        .unwrap_or_else(|_| panic!("fixture key {key}: bad number {value}"))
}

/// A list value of the fixture, comma separated; `nan` is a null.
fn reference_list(key: &str) -> Vec<f64> {
    raw(key)
        .split(',')
        .map(|item| {
            if item == "nan" {
                f64::NAN
            } else {
                item.parse()
                    .unwrap_or_else(|_| panic!("fixture key {key}: bad number {item}"))
            }
        })
        .collect()
}

/// A text list of the fixture, comma separated.
fn reference_text(key: &str) -> Vec<String> {
    raw(key).split(',').map(String::from).collect()
}

/// 1e-9 relative, as the table says.
fn assert_close(actual: f64, key: &str) {
    let expected = reference(key);
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-300),
        "{key}: got {actual}, expected {expected}"
    );
}

/// Each element of a list, in order, against the fixture list `key`; a NaN equals a NaN.
fn assert_list(actual: &[f64], key: &str) {
    let expected = reference_list(key);
    assert_eq!(actual.len(), expected.len(), "{key}: length");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{key}[{i}]: got {a}, expected NaN");
        } else {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1e-300),
                "{key}[{i}]: got {a}, expected {e}"
            );
        }
    }
}

fn assert_text(actual: &[String], key: &str) {
    assert_eq!(actual, reference_text(key).as_slice(), "{key}");
}

/// The values of a column in row order; a null is NaN.
fn floats(out: &DataFrame, column: &str) -> Vec<f64> {
    out.get(column)
        .unwrap_or_else(|_| panic!("output has no column {column}"))
        .values()
        .iter()
        .map(|v| v.as_float().unwrap_or(f64::NAN))
        .collect()
}

/// The values of a column in row order, as text.
fn texts(out: &DataFrame, column: &str) -> Vec<String> {
    out.get(column)
        .unwrap_or_else(|_| panic!("output has no column {column}"))
        .values()
        .iter()
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .collect()
}

/// The non-null count and the sum of a column; a null is skipped.
fn count_and_sum(out: &DataFrame, column: &str) -> (f64, f64) {
    let values: Vec<f64> = out
        .get(column)
        .unwrap_or_else(|_| panic!("output has no column {column}"))
        .values()
        .iter()
        .filter_map(Value::as_float)
        .collect();
    (values.len() as f64, values.iter().sum())
}

/// The small frame S: region, product and sales. It has no (w, b) pair.
fn small_s() -> DataFrame {
    DataFrame::from_columns(vec![
        ("region", Series::new(vec!["e", "e", "w", "e", "e", "w"])),
        ("product", Series::new(vec!["a", "b", "a", "a", "a", "a"])),
        (
            "sales",
            Series::new(vec![10.0, 20.0, 30.0, 40.0, 5.0, 15.0]),
        ),
    ])
    .unwrap()
}

/// The large frame BIG: 10,000 rows, `sales` null where i % 7 == 3.
fn big() -> DataFrame {
    let n = 10000usize;
    let region: Vec<Value> = (0..n).map(|i| Value::Int((i % 50) as i64)).collect();
    let product: Vec<Value> = (0..n)
        .map(|i| Value::String(format!("p{}", i % 4)))
        .collect();
    let sales: Vec<Value> = (0..n)
        .map(|i| {
            if i % 7 == 3 {
                Value::Null
            } else {
                Value::Float(((i * 37 + 11) % 101) as f64 / 4.0)
            }
        })
        .collect();
    DataFrame::from_columns(vec![
        ("region", Series::new(region)),
        ("product", Series::new(product)),
        ("sales", Series::new(sales)),
    ])
    .unwrap()
}

/// The wide frame W: id, a, and b with a null on the last row.
fn wide_w() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id", Series::new(vec![1i64, 2, 3])),
        ("a", Series::new(vec![1.5, 2.5, 3.5])),
        (
            "b",
            Series::new(vec![Value::Float(4.5), Value::Float(5.5), Value::Null]),
        ),
    ])
    .unwrap()
}

/// The large wide frame BIGW: 10,000 rows, id = i, and score_k null where (i + k) % 7 == 3.
fn wide_big() -> DataFrame {
    let n = 10000usize;
    let mut columns: Vec<(&str, Series)> = vec![(
        "id",
        Series::new((0..n).map(|i| Value::Int(i as i64)).collect::<Vec<Value>>()),
    )];
    let names = ["score_1", "score_2", "score_3"];
    for (k, name) in (1..=3).zip(names) {
        let values: Vec<Value> = (0..n)
            .map(|i| {
                if (i + k) % 7 == 3 {
                    Value::Null
                } else {
                    Value::Float(((i * 37 + k * 11) % 101) as f64 / 4.0)
                }
            })
            .collect();
        columns.push((name, Series::new(values)));
    }
    DataFrame::from_columns(columns).unwrap()
}

/// The long frame U for unstack: id, key and val. It has no (3, y) pair.
fn long_u() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id", Series::new(vec![1i64, 1, 2, 2, 3])),
        ("key", Series::new(vec!["x", "y", "x", "y", "x"])),
        ("val", Series::new(vec![1.5, 2.5, 3.5, 4.5, 5.5])),
    ])
    .unwrap()
}

/// The wide frame T for wide_to_long: id, score_1, score_2, bonus_1 and bonus_2.
fn wide_t() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id", Series::new(vec![1i64, 2, 3])),
        ("score_1", Series::new(vec![1.5, 2.5, 3.5])),
        ("score_2", Series::new(vec![4.5, 5.5, 6.5])),
        ("bonus_1", Series::new(vec![10.0, 20.0, 30.0])),
        ("bonus_2", Series::new(vec![40.0, 50.0, 60.0])),
    ])
    .unwrap()
}

/// The aggregates of the fixture keys, with their names.
const AGGS: [(&str, AggFunc); 7] = [
    ("sum", AggFunc::Sum),
    ("mean", AggFunc::Mean),
    ("count", AggFunc::Count),
    ("min", AggFunc::Min),
    ("max", AggFunc::Max),
    ("first", AggFunc::First),
    ("last", AggFunc::Last),
];

/// The small pivot table of one aggregate: the index and the columns a and b.
fn check_pivot_small(out: &DataFrame, name: &str) {
    assert_text(
        &texts(out, "region"),
        &format!("pivot_table.{name}.small.index"),
    );
    assert_list(&floats(out, "a"), &format!("pivot_table.{name}.small.a"));
    assert_list(&floats(out, "b"), &format!("pivot_table.{name}.small.b"));
}

/// The large pivot table of one aggregate: the row count, the non-null cells and their sum.
fn check_pivot_large(out: &DataFrame, name: &str) {
    assert_close(
        out.nrows() as f64,
        &format!("pivot_table.{name}.large.rows"),
    );
    let (mut cells, mut total) = (0.0, 0.0);
    for column in out.columns().iter().filter(|c| c.as_str() != "region") {
        let (count, sum) = count_and_sum(out, column);
        cells += count;
        total += sum;
    }
    assert_close(cells, &format!("pivot_table.{name}.large.cells"));
    assert_close(total, &format!("pivot_table.{name}.large.sum"));
}

/// The small cross tabulation: the index and the counts of a and b.
fn check_cross_small(out: &DataFrame) {
    assert_text(&texts(out, "region"), "cross_tab.small.index");
    assert_list(&floats(out, "a"), "cross_tab.small.a");
    assert_list(&floats(out, "b"), "cross_tab.small.b");
}

/// The large cross tabulation: the row count, the sum of all counts, and column p0.
fn check_cross_large(out: &DataFrame) {
    assert_close(out.nrows() as f64, "cross_tab.large.rows");
    let mut total = 0.0;
    for column in out.columns().iter().filter(|c| c.as_str() != "region") {
        total += count_and_sum(out, column).1;
    }
    assert_close(total, "cross_tab.large.sum");
    assert_close(count_and_sum(out, "p0").1, "cross_tab.large.p0");
}

/// The rows of a wide_to_long result, sorted by (id, t as text), as the fixture is.
fn sorted_long(out: &DataFrame) -> (Vec<f64>, Vec<String>, Vec<f64>, Vec<f64>) {
    let ids = floats(out, "id");
    let ts = texts(out, "t");
    let scores = floats(out, "score");
    let bonuses = floats(out, "bonus");
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by(|&a, &b| ids[a].total_cmp(&ids[b]).then(ts[a].cmp(&ts[b])));
    (
        order.iter().map(|&i| ids[i]).collect(),
        order.iter().map(|&i| ts[i].clone()).collect(),
        order.iter().map(|&i| scores[i]).collect(),
        order.iter().map(|&i| bonuses[i]).collect(),
    )
}

fn check_wide_to_long(out: &DataFrame) {
    let (ids, ts, scores, bonuses) = sorted_long(out);
    assert_list(&ids, "wide_to_long.small.id");
    assert_text(&ts, "wide_to_long.small.t");
    assert_list(&scores, "wide_to_long.small.score");
    assert_list(&bonuses, "wide_to_long.small.bonus");
}

fn check_wide_to_long_large(out: &DataFrame) {
    assert_close(out.nrows() as f64, "wide_to_long.large.rows");
    let (count, sum) = count_and_sum(out, "score");
    assert_close(count, "wide_to_long.large.score_count");
    assert_close(sum, "wide_to_long.large.score_sum");
    assert_close(count_and_sum(out, "id").1, "wide_to_long.large.id_sum");
}

fn check_stack_large(out: &DataFrame) {
    assert_close(out.nrows() as f64, "stack.large.rows");
    let (count, sum) = count_and_sum(out, "value");
    assert_close(count, "stack.large.value_count");
    assert_close(sum, "stack.large.value_sum");
    assert_close(count_and_sum(out, "id").1, "stack.large.id_sum");
}

fn check_stack_small(out: &DataFrame) {
    assert_list(&floats(out, "id"), "stack.small.id");
    assert_text(&texts(out, "variable"), "stack.small.variable");
    assert_list(&floats(out, "value"), "stack.small.value");
}

fn check_unstack(out: &DataFrame) {
    assert_list(&floats(out, "id"), "unstack.small.id");
    assert_list(&floats(out, "x"), "unstack.small.x");
    assert_list(&floats(out, "y"), "unstack.small.y");
}

#[test]
fn pivot_table_matches_pandas() {
    for (name, agg) in AGGS {
        let out = pivot_table(&small_s(), "sales", "region", "product", agg).unwrap();
        check_pivot_small(&out, name);
    }
    for (name, agg) in [
        ("sum", AggFunc::Sum),
        ("mean", AggFunc::Mean),
        ("count", AggFunc::Count),
    ] {
        let out = pivot_table(&big(), "sales", "region", "product", agg).unwrap();
        check_pivot_large(&out, name);
    }
}

#[test]
fn pivot_table_method_matches_pandas() {
    for (name, agg) in AGGS {
        let out = small_s()
            .pivot_table("sales", "region", "product", agg)
            .unwrap();
        check_pivot_small(&out, name);
    }
    for (name, agg) in [
        ("sum", AggFunc::Sum),
        ("mean", AggFunc::Mean),
        ("count", AggFunc::Count),
    ] {
        let out = big()
            .pivot_table("sales", "region", "product", agg)
            .unwrap();
        check_pivot_large(&out, name);
    }
}

#[test]
fn cross_tab_matches_pandas() {
    check_cross_small(&cross_tab(&small_s(), "region", "product").unwrap());
    check_cross_large(&cross_tab(&big(), "region", "product").unwrap());
}

#[test]
fn cross_tab_method_matches_pandas() {
    check_cross_small(&small_s().cross_tab("region", "product").unwrap());
    check_cross_large(&big().cross_tab("region", "product").unwrap());
}

#[test]
fn stack_matches_pandas() {
    check_stack_small(&stack(&wide_w(), &["a", "b"]).unwrap());
    check_stack_large(&stack(&wide_big(), &["score_1", "score_2", "score_3"]).unwrap());
}

#[test]
fn stack_method_matches_pandas() {
    check_stack_small(&wide_w().stack(&["a", "b"]).unwrap());
    check_stack_large(
        &wide_big()
            .stack(&["score_1", "score_2", "score_3"])
            .unwrap(),
    );
}

#[test]
fn unstack_matches_pandas() {
    check_unstack(&unstack(&long_u(), "id", "key", "val").unwrap());
}

#[test]
fn unstack_method_matches_pandas() {
    check_unstack(&long_u().unstack("id", "key", "val").unwrap());
}

#[test]
fn wide_to_long_matches_pandas() {
    let out = wide_to_long(&wide_t(), &["score", "bonus"], "id", "t", "_").unwrap();
    check_wide_to_long(&out);
    check_wide_to_long_large(&wide_to_long(&wide_big(), &["score"], "id", "t", "_").unwrap());
}

#[test]
fn wide_to_long_method_matches_pandas() {
    let out = wide_t()
        .wide_to_long(&["score", "bonus"], "id", "t", "_")
        .unwrap();
    check_wide_to_long(&out);
    check_wide_to_long_large(&wide_big().wide_to_long(&["score"], "id", "t", "_").unwrap());
}
