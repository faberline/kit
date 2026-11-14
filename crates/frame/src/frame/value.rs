//! Value type for DataFrame cells.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A value that can be stored in a DataFrame cell.
#[derive(Debug, Clone, Default)]
pub enum Value {
    /// Null/missing value.
    #[default]
    Null,
    /// Boolean value.
    Bool(bool),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit floating point.
    Float(f64),
    /// String value.
    String(String),
}

impl Value {
    /// Check if the value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Try to get as i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            Value::Float(v) => Some(*v as i64),
            Value::Bool(v) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Try to get as f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Int(v) => Some(*v as f64),
            Value::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Try to get as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            Value::Int(v) => Some(*v != 0),
            _ => None,
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
                (*a as f64).to_bits() == b.to_bits()
            }
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Must agree with `PartialEq`: `Int(a) == Float(b)` when
        // `(a as f64).to_bits() == b.to_bits()`, so Int and Float share a tag
        // and hash the same f64 bits.
        match self {
            Value::Null => 0u8.hash(state),
            Value::Bool(v) => {
                1u8.hash(state);
                v.hash(state);
            }
            Value::Int(v) => {
                2u8.hash(state);
                (*v as f64).to_bits().hash(state);
            }
            Value::Float(v) => {
                2u8.hash(state);
                v.to_bits().hash(state);
            }
            Value::String(v) => {
                3u8.hash(state);
                v.hash(state);
            }
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => Ordering::Less,
            (_, Value::Null) => Ordering::Greater,
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => a.total_cmp(b),
            (Value::String(a), Value::String(b)) => a.cmp(b),
            (Value::Int(a), Value::Float(b)) => (*a as f64).total_cmp(b),
            (Value::Float(a), Value::Int(b)) => a.total_cmp(&(*b as f64)),
            // Type ordering: Null < Bool < Int < Float < String
            (Value::Bool(_), _) => Ordering::Less,
            (_, Value::Bool(_)) => Ordering::Greater,
            (Value::Int(_), _) => Ordering::Less,
            (_, Value::Int(_)) => Ordering::Greater,
            (Value::Float(_), _) => Ordering::Less,
            (_, Value::Float(_)) => Ordering::Greater,
        }
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Int(v as i64)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}

impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Value::Float(v as f64)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_string())
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => v.into(),
            None => Value::Null,
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(v) => write!(f, "{}", v),
            Value::Int(v) => write!(f, "{}", v),
            Value::Float(v) => write!(f, "{}", v),
            Value::String(v) => write!(f, "{}", v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_conversions() {
        assert_eq!(Value::from(42i32).as_int(), Some(42));
        assert_eq!(Value::from(2.5f64).as_float(), Some(2.5));
        assert_eq!(Value::from("hello").as_str(), Some("hello"));
        assert_eq!(Value::from(true).as_bool(), Some(true));
    }

    #[test]
    fn test_value_ordering() {
        let null = Value::Null;
        let int = Value::Int(10);
        let float = Value::Float(10.5);
        let string = Value::String("hello".to_string());

        assert!(null < int);
        assert!(int < float);
        assert!(float < string);
    }
}

#[cfg(test)]
mod hash_tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn h(v: &Value) -> u64 {
        let mut s = DefaultHasher::new();
        v.hash(&mut s);
        s.finish()
    }

    #[test]
    fn int_and_float_hash_equal() {
        assert_eq!(Value::Int(2), Value::Float(2.0));
        assert_eq!(h(&Value::Int(2)), h(&Value::Float(2.0)));
    }

    #[test]
    fn different_values_hash_differently_in_practice() {
        assert_ne!(h(&Value::Int(2)), h(&Value::Int(3)));
        assert_ne!(h(&Value::String("a".into())), h(&Value::String("b".into())));
    }

    #[test]
    fn float_nan_is_unchanged() {
        assert_eq!(Value::Float(f64::NAN), Value::Float(f64::NAN));
        assert_eq!(h(&Value::Float(f64::NAN)), h(&Value::Float(f64::NAN)));
    }
}
