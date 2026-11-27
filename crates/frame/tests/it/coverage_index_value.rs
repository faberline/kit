//! Coverage tests for `index.rs`, `multi_index.rs` and `value.rs` of
//! `frame::frame`: every error branch and every conversion or ordering arm.

use frame::frame::{FrameError, Index, MultiIndex, Value};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn h(v: &Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    v.hash(&mut hasher);
    hasher.finish()
}

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

// ---- Index ----

#[test]
fn get_loc_range_out_of_bounds() {
    let r = Index::range(3);
    let err = r.get_loc(&Value::Int(9)).err().unwrap();
    assert_eq!(err.to_string(), "index out of bounds: 9 (length: 3)");
}

#[test]
fn get_loc_type_mismatch() {
    let r = Index::range(3);
    let err = r.get_loc(&Value::from("a")).err().unwrap();
    assert_eq!(
        err.to_string(),
        "type mismatch: index type mismatch: Range(3) vs String(\"a\")"
    );
    let err = Index::int(vec![10, 20])
        .get_loc(&Value::Null)
        .err()
        .unwrap();
    assert_eq!(
        err.to_string(),
        "type mismatch: index type mismatch: Int([10, 20]) vs Null"
    );
}

#[test]
fn get_loc_label_not_found() {
    let err = Index::int(vec![10, 20])
        .get_loc(&Value::Int(5))
        .err()
        .unwrap();
    assert_eq!(err.to_string(), "label not found in index: 5");
    let err = Index::string(vec!["a".to_string()])
        .get_loc(&Value::from("z"))
        .err()
        .unwrap();
    assert_eq!(err.to_string(), "label not found in index: z");
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
    let err = r.get_label(3).err().unwrap();
    assert_eq!(err.to_string(), "index out of bounds: 3 (length: 3)");
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
    let err = r.slice(&[3]).err().unwrap();
    assert_eq!(err.to_string(), "index out of bounds: 3 (length: 3)");
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

// ---- MultiIndex ----

#[test]
fn new_shape_mismatch() {
    let err: FrameError = MultiIndex::new(vec![
        ("a", vec![Value::Int(1), Value::Int(2)]),
        ("b", vec![Value::from("x")]),
    ])
    .err()
    .unwrap();
    assert_eq!(err.to_string(), "shape mismatch: expected 2, got 1");
}

#[test]
fn new_empty_levels() {
    assert!(MultiIndex::new(Vec::new()).unwrap().is_empty());
}

#[test]
fn level_lookup_errors() {
    let m = mi();
    let err = m.get_level_values(5).err().unwrap();
    assert_eq!(err.to_string(), "index out of bounds: 5 (length: 2)");
    let err = m.get_level_values_by_name("zz").err().unwrap();
    assert_eq!(err.to_string(), "label not found in index: zz");
    let err = m.get_tuple(3).err().unwrap();
    assert_eq!(err.to_string(), "index out of bounds: 3 (length: 3)");
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
    let err = m.swaplevel(0, 5).err().unwrap();
    assert_eq!(err.to_string(), "index out of bounds: 5 (length: 2)");
    let err = m.droplevel(5).err().unwrap();
    assert_eq!(err.to_string(), "index out of bounds: 5 (length: 2)");

    let dropped = m.droplevel(0).unwrap();
    assert_eq!(dropped.names(), ["b".to_string()]);
    let err = dropped.droplevel(0).err().unwrap();
    assert_eq!(
        err.to_string(),
        "invalid operation: cannot drop the last level"
    );

    let err = m.reorder_levels(&[0]).err().unwrap();
    assert_eq!(
        err.to_string(),
        "invalid operation: order must have same length as number of levels"
    );
    let err = m.reorder_levels(&[0, 5]).err().unwrap();
    assert_eq!(err.to_string(), "index out of bounds: 5 (length: 2)");
    let err = m.slice(&[9]).err().unwrap();
    assert_eq!(err.to_string(), "index out of bounds: 9 (length: 3)");
}

#[test]
fn argsort_equal_rows_keep_order() {
    let equal = MultiIndex::new(vec![("a", vec![Value::Int(1), Value::Int(1)])]).unwrap();
    assert_eq!(equal.argsort(), vec![0, 1]);

    let m = mi();
    assert_eq!(m.argsort(), vec![0, 1, 2]);
    assert_eq!(m.get_locs(&[Value::Int(1)]), vec![0, 1]);
    assert_eq!(m.unique().len(), 3);
}

// ---- Value ----

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
    assert_eq!(Value::Int(1).as_str(), None);
    assert_eq!(Value::Null.as_float(), None);
}

#[test]
fn as_float_bool_false() {
    assert_eq!(Value::Bool(false).as_float(), Some(0.0));
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
    let b = Value::from("z");
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

#[test]
fn multi_index_accessors_and_group_map() {
    let d = MultiIndex::default();
    assert_eq!(d.nlevels(), 0);
    assert_eq!(d.len(), 0);
    assert!(d.is_empty());
    assert_eq!(
        format!("{:?}", d),
        "MultiIndex { names: [], levels: [], length: 0 }"
    );

    let m = mi();
    assert_eq!(m.nlevels(), 2);
    assert_eq!(m.len(), 3);
    assert!(!m.is_empty());
    assert_eq!(m.names(), &["a".to_string(), "b".to_string()]);
    assert_eq!(m.get_level_index("b").unwrap(), 1);
    assert_eq!(
        m.get_level_index("zz").unwrap_err().to_string(),
        "label not found in index: zz"
    );
    assert_eq!(m.clone().len(), 3);

    let g = m.to_group_map();
    assert_eq!(g.len(), 3);
    assert_eq!(g[&vec![Value::Int(1), Value::from("x")]], vec![0]);
    assert_eq!(g[&vec![Value::Int(1), Value::from("y")]], vec![1]);
    assert_eq!(g[&vec![Value::Int(2), Value::from("x")]], vec![2]);
}
