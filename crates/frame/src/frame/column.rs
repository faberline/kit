//! Typed column with a null mask.

use super::value::Value;

/// A column of one type. The second vector is the null mask: `true` means null.
/// Both vectors have the same length. A null slot holds a placeholder
/// (`false`, `0`, `0.0`, `""`) that nobody may read as a value.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    Bool(Vec<bool>, Vec<bool>),
    Int(Vec<i64>, Vec<bool>),
    Float(Vec<f64>, Vec<bool>),
    Str(Vec<String>, Vec<bool>),
}

impl Column {
    /// Build a column from cell values. The type follows the non-null values.
    pub fn from_values(values: &[Value]) -> Column {
        let mask: Vec<bool> = values.iter().map(Value::is_null).collect();
        if values.iter().all(Value::is_null) {
            return Column::Float(vec![0.0; values.len()], mask);
        }
        let all = |pred: fn(&Value) -> bool| values.iter().filter(|v| !v.is_null()).all(pred);
        if all(|v| matches!(v, Value::Bool(_))) {
            let data = values
                .iter()
                .map(|v| matches!(v, Value::Bool(true)))
                .collect();
            Column::Bool(data, mask)
        } else if all(|v| matches!(v, Value::Int(_))) {
            let data = values
                .iter()
                .map(|v| if let Value::Int(i) = v { *i } else { 0 })
                .collect();
            Column::Int(data, mask)
        } else if all(|v| matches!(v, Value::Int(_) | Value::Float(_))) {
            let data = values.iter().map(|v| v.as_float().unwrap_or(0.0)).collect();
            Column::Float(data, mask)
        } else {
            let data = values
                .iter()
                .map(|v| {
                    if v.is_null() {
                        String::new()
                    } else {
                        v.to_string()
                    }
                })
                .collect();
            Column::Str(data, mask)
        }
    }

    /// Return the cell values of the column. A null slot gives `Value::Null`.
    pub fn to_values(&self) -> Vec<Value> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }

    /// Return the number of slots, null or not.
    pub fn len(&self) -> usize {
        match self {
            Column::Bool(d, _) => d.len(),
            Column::Int(d, _) => d.len(),
            Column::Float(d, _) => d.len(),
            Column::Str(d, _) => d.len(),
        }
    }

    /// Return `true` when the column has no slots.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the value at `i`, or `Value::Null`. Panics when `i >= len`.
    pub fn get(&self, i: usize) -> Value {
        if self.is_null(i) {
            return Value::Null;
        }
        match self {
            Column::Bool(d, _) => Value::Bool(d[i]),
            Column::Int(d, _) => Value::Int(d[i]),
            Column::Float(d, _) => Value::Float(d[i]),
            Column::Str(d, _) => Value::String(d[i].clone()),
        }
    }

    /// Return `true` when slot `i` is null. Panics when `i >= len`.
    pub fn is_null(&self, i: usize) -> bool {
        match self {
            Column::Bool(_, m) | Column::Int(_, m) | Column::Float(_, m) | Column::Str(_, m) => {
                m[i]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_round_trip() {
        let input = vec![Value::Bool(true), Value::Null, Value::Bool(false)];
        let c = Column::from_values(&input);
        assert_eq!(
            c,
            Column::Bool(vec![true, false, false], vec![false, true, false])
        );
        assert_eq!(c.to_values(), input);
    }

    #[test]
    fn int_round_trip() {
        let input = vec![Value::Int(1), Value::Null, Value::Int(-3)];
        let c = Column::from_values(&input);
        assert!(matches!(c, Column::Int(_, _)));
        assert_eq!(c.to_values(), input);
        assert!(c.is_null(1));
        assert!(!c.is_null(0));
    }

    #[test]
    fn float_round_trip() {
        let input = vec![Value::Float(1.5), Value::Float(f64::NAN), Value::Null];
        let c = Column::from_values(&input);
        assert!(matches!(c, Column::Float(_, _)));
        assert_eq!(c.to_values(), input);
        // NaN is a value, not a null; only the mask decides null.
        assert!(!c.is_null(1));
        assert!(c.is_null(2));
    }

    #[test]
    fn str_round_trip() {
        let input = vec![
            Value::String("a".to_string()),
            Value::Null,
            Value::String(String::new()),
        ];
        let c = Column::from_values(&input);
        assert!(matches!(c, Column::Str(_, _)));
        assert_eq!(c.to_values(), input);
    }

    #[test]
    fn mixed_int_float_becomes_float() {
        let input = vec![Value::Int(1), Value::Float(2.5), Value::Null];
        let c = Column::from_values(&input);
        assert_eq!(
            c,
            Column::Float(vec![1.0, 2.5, 0.0], vec![false, false, true])
        );
    }

    #[test]
    fn mixed_with_string_becomes_str() {
        let input = vec![
            Value::Int(1),
            Value::String("x".to_string()),
            Value::Bool(true),
            Value::Float(1.5),
            Value::Null,
        ];
        let c = Column::from_values(&input);
        assert!(matches!(c, Column::Str(_, _)));
        assert_eq!(
            c.to_values(),
            vec![
                Value::String("1".to_string()),
                Value::String("x".to_string()),
                Value::String("true".to_string()),
                Value::String("1.5".to_string()),
                Value::Null,
            ]
        );
    }

    #[test]
    fn bool_with_int_becomes_str() {
        let c = Column::from_values(&[Value::Bool(true), Value::Int(2)]);
        assert!(matches!(c, Column::Str(_, _)));
        assert_eq!(
            c.to_values(),
            vec![
                Value::String("true".to_string()),
                Value::String("2".to_string()),
            ]
        );
    }

    #[test]
    fn empty_and_all_null() {
        let empty = Column::from_values(&[]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let nulls = Column::from_values(&[Value::Null, Value::Null]);
        assert_eq!(nulls.len(), 2);
        assert!(nulls.is_null(0));
        assert!(nulls.is_null(1));
        assert_eq!(nulls.get(0), Value::Null);
    }

    #[test]
    fn get_and_len() {
        let c = Column::from_values(&[Value::Int(1), Value::Null, Value::Int(-3)]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.get(0), Value::Int(1));
        assert_eq!(c.get(1), Value::Null);
        assert_eq!(c.get(2), Value::Int(-3));
    }

    #[test]
    fn null_is_not_the_placeholder() {
        let c = Column::from_values(&[Value::Int(0), Value::Null]);
        assert_eq!(c.get(0), Value::Int(0));
        assert_eq!(c.get(1), Value::Null);
    }

    #[test]
    fn exported_from_crate_root() {
        let _ = crate::Column::from_values(&[]);
    }
}
