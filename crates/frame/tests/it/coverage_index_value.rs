//! Coverage tests for `frame::frame` index, multi-index and value code.
//! Tests only: every expected value was written by hand from the current
//! behavior of the code.

use frame::frame::{FrameError, Index, MultiIndex, Value};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn h(v: &Value) -> u64 {
    let mut s = DefaultHasher::new();
    v.hash(&mut s);
    s.finish()
}

fn err_text<T>(r: Result<T, FrameError>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

// ---------- Index ----------

#[test]
fn get_loc_range_out_of_bounds() {
    let r = Index::range(3);
    assert_eq!(
        err_text(r.get_loc(&Value::Int(9))),
        "index out of bounds: 9 (length: 3)"
    );
}

#[test]
fn get_loc_type_mismatch() {
    let r = Index::range(3);
    assert_eq!(
        err_text(r.get_loc(&Value::from("a"))),
        "type mismatch: index type mismatch: Range(3) vs String(\"a\")"
    );
    assert_eq!(
        err_text(Index::int(vec![10, 20]).get_loc(&Value::Null)),
        "type mismatch: index type mismatch: Int([10, 20]) vs Null"
    );
}

#[test]
fn get_loc_label_not_found() {
    assert_eq!(
        err_text(Index::int(vec![10, 20]).get_loc(&Value::Int(5))),
        "label not found in index: 5"
    );
    assert_eq!(
        err_text(Index::string(vec!["a".to_string()]).get_loc(&Value::from("z"))),
        "label not found in index: z"
    );
}

#[test]
fn get_label_each_kind() {
    let r = Index::range(3);
    assert_eq!(r.get_label(1).unwrap(), Value::Int(1));
    assert_eq!(
        Index::int(vec![10, 20]).get_label(1).unwrap(),
        Value::Int(20)
    );
    assert_eq!(
        Index::from(vec!["x", "y"]).get_label(0).unwrap(),
        Value::from("x")
    );
    assert_eq!(
        err_text(r.get_label(3)),
        "index out of bounds: 3 (length: 3)"
    );
}

#[test]
fn to_values_each_kind() {
    assert_eq!(
        Index::range(3).to_values(),
        vec![Value::Int(0), Value::Int(1), Value::Int(2)]
    );
    assert_eq!(
        Index::int(vec![7, 8]).to_values(),
        vec![Value::Int(7), Value::Int(8)]
    );
    assert_eq!(
        Index::from(vec!["x".to_string()]).to_values(),
        vec![Value::from("x")]
    );
}

#[test]
fn to_map_maps_labels_to_positions() {
    let m = Index::from(vec!["a", "b"]).to_map();
    assert_eq!(m.len(), 2);
    assert_eq!(m.get(&Value::from("b")), Some(&1));
}

#[test]
fn slice_and_reset() {
    let r = Index::range(3);
    assert_eq!(
        r.slice(&[2, 0]).unwrap().to_values(),
        vec![Value::Int(2), Value::Int(0)]
    );
    assert_eq!(
        err_text(r.slice(&[3])),
        "index out of bounds: 3 (length: 3)"
    );
    assert_eq!(r.reset().len(), 3);
}

#[test]
fn default_and_from_impls() {
    assert_eq!(Index::default().len(), 0);
    assert!(Index::default().is_empty());
    assert_eq!(Index::from(4usize).len(), 4);
    assert_eq!(
        Index::from(vec![1i64, 2]).to_values(),
        vec![Value::Int(1), Value::Int(2)]
    );
}

// ---------- MultiIndex ----------

fn mi() -> MultiIndex {
    MultiIndex::new(vec![
        ("a", vec![Value::Int(1), Value::Int(1), Value::Int(2)]),
        (
            "b",
            vec![Value::from("x"), Value::from("y"), Value::from("x")],
        ),
    ])
    .unwrap()
}

#[test]
fn new_shape_mismatch() {
    let e = MultiIndex::new(vec![
        ("a", vec![Value::Int(1), Value::Int(2)]),
        ("b", vec![Value::from("x")]),
    ])
    .err()
    .unwrap();
    assert_eq!(e.to_string(), "shape mismatch: expected 2, got 1");
}

#[test]
fn new_empty_levels() {
    assert!(MultiIndex::new(vec![]).unwrap().is_empty());
}

#[test]
fn level_lookup_errors() {
    let m = mi();
    assert_eq!(
        err_text(m.get_level_values(5)),
        "index out of bounds: 5 (length: 2)"
    );
    assert_eq!(
        err_text(m.get_level_values_by_name("zz")),
        "label not found in index: zz"
    );
    assert_eq!(
        err_text(m.get_tuple(3)),
        "index out of bounds: 3 (length: 3)"
    );
}

#[test]
fn select_level_unknown_level_is_empty() {
    let m = mi();
    assert_eq!(m.select_level(9, &Value::Int(1)), Vec::<usize>::new());
    assert_eq!(m.select_level(0, &Value::Int(1)), vec![0, 1]);
}

#[test]
fn swaplevel_droplevel_reorder_errors() {
    let m = mi();
    assert_eq!(
        err_text(m.swaplevel(0, 5)),
        "index out of bounds: 5 (length: 2)"
    );
    assert_eq!(
        err_text(m.droplevel(5)),
        "index out of bounds: 5 (length: 2)"
    );
    let one = m.droplevel(0).unwrap();
    assert_eq!(one.names(), vec!["b".to_string()]);
    assert_eq!(
        err_text(one.droplevel(0)),
        "invalid operation: cannot drop the last level"
    );
    assert_eq!(
        err_text(m.reorder_levels(&[0])),
        "invalid operation: order must have same length as number of levels"
    );
    assert_eq!(
        err_text(m.reorder_levels(&[0, 5])),
        "index out of bounds: 5 (length: 2)"
    );
    assert_eq!(
        err_text(m.slice(&[9])),
        "index out of bounds: 9 (length: 3)"
    );
}

#[test]
fn argsort_equal_rows_keep_order() {
    let eq = MultiIndex::new(vec![("a", vec![Value::Int(1), Value::Int(1)])]).unwrap();
    assert_eq!(eq.argsort(), vec![0, 1]);
    let m = mi();
    assert_eq!(m.argsort(), vec![0, 1, 2]);
    assert_eq!(m.get_locs(&[Value::Int(1)]), vec![0, 1]);
    assert_eq!(m.unique().len(), 3);
}

// ---------- Value ----------

#[test]
fn as_conversions_each_variant() {
    assert_eq!(Value::Float(2.9).as_int(), Some(2));
    assert_eq!(Value::Bool(true).as_int(), Some(1));
    assert_eq!(Value::Bool(false).as_int(), Some(0));
    assert_eq!(Value::from("a").as_int(), None);
    assert_eq!(Value::Int(2).as_bool(), Some(true));
    assert_eq!(Value::Int(0).as_bool(), Some(false));
    assert_eq!(Value::Float(1.0).as_bool(), None);
    assert_eq!(Value::Bool(true).as_float(), Some(1.0));
    assert_eq!(Value::Bool(false).as_float(), Some(0.0));
    assert_eq!(Value::Int(1).as_str(), None);
    assert_eq!(Value::Null.as_float(), None);
}

#[test]
fn ordering_across_types() {
    assert_eq!(Value::Null.cmp(&Value::Null), Ordering::Equal);
    assert_eq!(Value::Null.cmp(&Value::from("a")), Ordering::Less);
    assert_eq!(Value::Int(1).cmp(&Value::Null), Ordering::Greater);
    assert_eq!(
        Value::Bool(true).cmp(&Value::Bool(false)),
        Ordering::Greater
    );
    assert_eq!(Value::Float(1.5).cmp(&Value::Int(2)), Ordering::Less);
    assert_eq!(Value::Int(2).cmp(&Value::Float(1.5)), Ordering::Greater);
    assert_eq!(Value::Bool(true).cmp(&Value::Int(0)), Ordering::Less);
    assert_eq!(Value::Int(0).cmp(&Value::Bool(true)), Ordering::Greater);
    assert_eq!(Value::Int(5).cmp(&Value::from("a")), Ordering::Less);
    assert_eq!(Value::from("a").cmp(&Value::Int(5)), Ordering::Greater);
    assert_eq!(Value::from("a").cmp(&Value::Float(1.0)), Ordering::Greater);
    assert_eq!(Value::Float(1.0).cmp(&Value::from("a")), Ordering::Less);
    assert_eq!(Value::Bool(true).cmp(&Value::from("a")), Ordering::Less);
    assert_eq!(Value::from("a").cmp(&Value::Bool(true)), Ordering::Greater);

    let a = Value::Int(3);
    let b = Value::Null;
    assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
}

#[test]
fn hash_bool_and_equality() {
    assert_eq!(h(&Value::Bool(true)), h(&Value::Bool(true)));
    assert_ne!(h(&Value::Bool(true)), h(&Value::Bool(false)));
    assert_ne!(Value::Null, Value::Bool(false));
    assert_ne!(Value::from("a"), Value::Int(1));
}

#[test]
fn from_impls_and_display() {
    assert_eq!(Value::from(1.5f32), Value::Float(1.5));
    assert_eq!(Value::from(7i32), Value::Int(7));
    assert_eq!(Value::from(String::from("s")), Value::from("s"));
    assert_eq!(Value::from(Some(3i64)), Value::Int(3));
    assert_eq!(Value::from(None::<i64>), Value::Null);
    assert_eq!(Value::from(Some("a")), Value::from("a"));
    assert_eq!(Value::default(), Value::Null);

    assert_eq!(Value::Null.to_string(), "null");
    assert_eq!(Value::Bool(true).to_string(), "true");
    assert_eq!(Value::Int(-3).to_string(), "-3");
    assert_eq!(Value::Float(2.5).to_string(), "2.5");
    assert_eq!(Value::from("hi").to_string(), "hi");
}
