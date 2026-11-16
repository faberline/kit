use super::*;

#[test]
fn test_series_basic() {
    let s = Series::new(vec![1, 2, 3, 4, 5]);
    assert_eq!(s.len(), 5);
    assert_eq!(s.iloc(0).unwrap(), &Value::Int(1));
    assert_eq!(s.iloc(4).unwrap(), &Value::Int(5));
}

#[test]
fn test_series_with_name() {
    let s = Series::with_name(vec![1.0, 2.0, 3.0], "values");
    assert_eq!(s.name(), Some("values"));
}

#[test]
fn test_series_sum_mean() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(s.sum().unwrap(), 15.0);
    assert_eq!(s.mean().unwrap(), 3.0);
}

#[test]
fn test_series_filter() {
    let s = Series::new(vec![1, 2, 3, 4, 5]);
    let mask = vec![true, false, true, false, true];
    let filtered = s.filter(&mask).unwrap();
    assert_eq!(filtered.len(), 3);
    assert_eq!(filtered.iloc(0).unwrap(), &Value::Int(1));
    assert_eq!(filtered.iloc(1).unwrap(), &Value::Int(3));
    assert_eq!(filtered.iloc(2).unwrap(), &Value::Int(5));
}

#[test]
fn test_series_arithmetic() {
    let a = Series::new(vec![1.0, 2.0, 3.0]);
    let b = Series::new(vec![10.0, 20.0, 30.0]);
    let sum = (&a + &b).unwrap();
    assert_eq!(sum.iloc(0).unwrap(), &Value::Float(11.0));
    assert_eq!(sum.iloc(2).unwrap(), &Value::Float(33.0));
}

#[test]
fn sort_puts_nulls_last() {
    let s = Series::new(vec![Value::Int(3), Value::Null, Value::Int(1)]);
    assert_eq!(
        s.sort(true).values(),
        &[Value::Int(1), Value::Int(3), Value::Null]
    );
    assert_eq!(
        s.sort(false).values(),
        &[Value::Int(3), Value::Int(1), Value::Null]
    );
}

#[test]
fn test_series_sort() {
    let s = Series::new(vec![3, 1, 4, 1, 5, 9, 2, 6]);
    let sorted = s.sort(true);
    assert_eq!(sorted.iloc(0).unwrap(), &Value::Int(1));
    assert_eq!(sorted.iloc(1).unwrap(), &Value::Int(1));
    assert_eq!(sorted.iloc(2).unwrap(), &Value::Int(2));
}

#[test]
fn test_series_unique() {
    let s = Series::new(vec![1, 2, 2, 3, 3, 3]);
    let unique = s.unique();
    assert_eq!(unique.len(), 3);
}

#[test]
fn test_series_isna() {
    let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3), Value::Null]);
    let mask = s.isna();
    assert_eq!(mask, vec![false, true, false, true]);
}

#[test]
fn test_series_fillna() {
    let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3), Value::Null]);
    let filled = s.fillna(Value::Int(0));
    assert_eq!(filled.iloc(0).unwrap(), &Value::Int(1));
    assert_eq!(filled.iloc(1).unwrap(), &Value::Int(0));
    assert_eq!(filled.iloc(2).unwrap(), &Value::Int(3));
    assert_eq!(filled.iloc(3).unwrap(), &Value::Int(0));
}

#[test]
fn test_series_ffill() {
    let s = Series::new(vec![Value::Int(1), Value::Null, Value::Null, Value::Int(4)]);
    let filled = s.ffill();
    assert_eq!(filled.iloc(1).unwrap(), &Value::Int(1));
    assert_eq!(filled.iloc(2).unwrap(), &Value::Int(1));
}

#[test]
fn test_series_bfill() {
    let s = Series::new(vec![Value::Null, Value::Null, Value::Int(3), Value::Int(4)]);
    let filled = s.bfill();
    assert_eq!(filled.iloc(0).unwrap(), &Value::Int(3));
    assert_eq!(filled.iloc(1).unwrap(), &Value::Int(3));
}

#[test]
fn test_series_dropna() {
    let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3), Value::Null]);
    let clean = s.dropna();
    assert_eq!(clean.len(), 2);
    assert_eq!(clean.iloc(0).unwrap(), &Value::Int(1));
    assert_eq!(clean.iloc(1).unwrap(), &Value::Int(3));
}

#[test]
fn test_series_apply() {
    let s = Series::new(vec![1.0, 2.0, 3.0]);
    let doubled = s.apply(|v| Value::Float(v.as_float().unwrap_or(0.0) * 2.0));
    assert_eq!(doubled.iloc(0).unwrap(), &Value::Float(2.0));
    assert_eq!(doubled.iloc(2).unwrap(), &Value::Float(6.0));
}

#[test]
fn test_series_cumsum() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0]);
    let result = s.cumsum();
    assert_eq!(result.iloc(0).unwrap(), &Value::Float(1.0));
    assert_eq!(result.iloc(1).unwrap(), &Value::Float(3.0));
    assert_eq!(result.iloc(2).unwrap(), &Value::Float(6.0));
    assert_eq!(result.iloc(3).unwrap(), &Value::Float(10.0));
}

#[test]
fn test_series_cumprod() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0]);
    let result = s.cumprod();
    assert_eq!(result.iloc(0).unwrap(), &Value::Float(1.0));
    assert_eq!(result.iloc(1).unwrap(), &Value::Float(2.0));
    assert_eq!(result.iloc(2).unwrap(), &Value::Float(6.0));
    assert_eq!(result.iloc(3).unwrap(), &Value::Float(24.0));
}

#[test]
fn test_series_shift() {
    let s = Series::new(vec![1, 2, 3, 4, 5]);
    let shifted = s.shift(2);
    assert!(shifted.iloc(0).unwrap().is_null());
    assert!(shifted.iloc(1).unwrap().is_null());
    assert_eq!(shifted.iloc(2).unwrap(), &Value::Int(1));
    assert_eq!(shifted.iloc(4).unwrap(), &Value::Int(3));
}

#[test]
fn test_series_diff() {
    let s = Series::new(vec![1.0, 3.0, 6.0, 10.0]);
    let result = s.diff(1);
    assert!(result.iloc(0).unwrap().is_null());
    assert_eq!(result.iloc(1).unwrap(), &Value::Float(2.0));
    assert_eq!(result.iloc(2).unwrap(), &Value::Float(3.0));
    assert_eq!(result.iloc(3).unwrap(), &Value::Float(4.0));
}

#[test]
fn test_series_pct_change() {
    let s = Series::new(vec![100.0, 110.0, 121.0]);
    let result = s.pct_change(1);
    assert!(result.iloc(0).unwrap().is_null());
    assert!(f64::abs(result.iloc(1).unwrap().as_float().unwrap() - 0.1) < 1e-10);
    assert!(f64::abs(result.iloc(2).unwrap().as_float().unwrap() - 0.1) < 1e-10);
}

#[test]
fn test_series_nlargest_nsmallest() {
    let s = Series::new(vec![3, 1, 4, 1, 5, 9, 2, 6]);
    let largest = s.nlargest(3);
    assert_eq!(largest.len(), 3);
    assert_eq!(largest.iloc(0).unwrap(), &Value::Int(9));
    assert_eq!(largest.iloc(1).unwrap(), &Value::Int(6));
    assert_eq!(largest.iloc(2).unwrap(), &Value::Int(5));

    let smallest = s.nsmallest(3);
    assert_eq!(smallest.iloc(0).unwrap(), &Value::Int(1));
}

#[test]
fn test_series_clip() {
    let s = Series::new(vec![1.0, 5.0, 10.0, 15.0]);
    let clipped = s.clip(3.0, 12.0);
    assert_eq!(clipped.iloc(0).unwrap(), &Value::Float(3.0));
    assert_eq!(clipped.iloc(1).unwrap(), &Value::Float(5.0));
    assert_eq!(clipped.iloc(2).unwrap(), &Value::Float(10.0));
    assert_eq!(clipped.iloc(3).unwrap(), &Value::Float(12.0));
}

#[test]
fn test_series_std_var() {
    let s = Series::new(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    let variance = s.var().unwrap();
    let std_dev = s.std().unwrap();
    assert!(f64::abs(variance - 4.571428571428571) < 1e-10);
    assert!(f64::abs(std_dev - 2.138089935299395) < 1e-10);
}

#[test]
fn test_series_median() {
    let s = Series::new(vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    assert_eq!(s.median().unwrap(), 5.0);

    let s2 = Series::new(vec![1.0, 3.0, 5.0, 7.0]);
    assert_eq!(s2.median().unwrap(), 4.0);
}

#[test]
fn test_series_rank() {
    let s = Series::new(vec![3, 1, 4, 1, 5]);
    let ranked = s.rank();
    // Values: 3, 1, 4, 1, 5 -> the two 1s share ranks 1 and 2 (average 1.5)
    assert_eq!(ranked.iloc(0).unwrap(), &Value::Float(3.0));
    assert_eq!(ranked.iloc(1).unwrap(), &Value::Float(1.5));
    assert_eq!(ranked.iloc(2).unwrap(), &Value::Float(4.0));
    assert_eq!(ranked.iloc(3).unwrap(), &Value::Float(1.5));
    assert_eq!(ranked.iloc(4).unwrap(), &Value::Float(5.0));
}

#[test]
fn test_series_rank_ties_average() {
    let ranked = Series::new(vec![1, 1, 2]).rank();
    assert_eq!(ranked.iloc(0).unwrap(), &Value::Float(1.5));
    assert_eq!(ranked.iloc(1).unwrap(), &Value::Float(1.5));
    assert_eq!(ranked.iloc(2).unwrap(), &Value::Float(3.0));
}

#[test]
fn test_series_rank_no_ties() {
    let ranked = Series::new(vec![3, 1, 2]).rank();
    assert_eq!(ranked.iloc(0).unwrap(), &Value::Float(3.0));
    assert_eq!(ranked.iloc(1).unwrap(), &Value::Float(1.0));
    assert_eq!(ranked.iloc(2).unwrap(), &Value::Float(2.0));
}

#[test]
fn test_series_rank_all_tied() {
    let ranked = Series::new(vec![5, 5, 5]).rank();
    assert_eq!(ranked.iloc(0).unwrap(), &Value::Float(2.0));
    assert_eq!(ranked.iloc(1).unwrap(), &Value::Float(2.0));
    assert_eq!(ranked.iloc(2).unwrap(), &Value::Float(2.0));
}

#[test]
fn test_series_duplicated() {
    let s = Series::new(vec![1, 2, 2, 3, 3, 3]);
    let dups = s.duplicated("first");
    assert_eq!(dups, vec![false, false, true, false, true, true]);
}

#[test]
fn test_series_drop_duplicates() {
    let s = Series::new(vec![1, 2, 2, 3, 3, 3]);
    let unique = s.drop_duplicates("first");
    assert_eq!(unique.len(), 3);
}

#[test]
fn test_series_isin() {
    let s = Series::new(vec![1, 2, 3, 4, 5]);
    let mask = s.isin(&[Value::Int(2), Value::Int(4)]);
    assert_eq!(mask, vec![false, true, false, true, false]);
}

#[test]
fn test_series_between() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask = s.between(2.0, 4.0);
    assert_eq!(mask, vec![false, true, true, true, false]);
}

#[test]
fn test_series_replace() {
    let s = Series::new(vec![1, 2, 3, 2, 1]);
    let replaced = s.replace(&Value::Int(2), &Value::Int(99));
    assert_eq!(replaced.iloc(1).unwrap(), &Value::Int(99));
    assert_eq!(replaced.iloc(3).unwrap(), &Value::Int(99));
}

#[test]
fn test_series_quantile() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(s.quantile(0.0).unwrap(), 1.0);
    assert_eq!(s.quantile(0.5).unwrap(), 3.0);
    assert_eq!(s.quantile(1.0).unwrap(), 5.0);
}

#[test]
fn test_series_corr() {
    let x = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    let corr = x.corr(&y).unwrap();
    assert!(f64::abs(corr - 1.0) < 1e-10); // Perfect correlation
}

// Phase 2 interpolation tests
#[test]
fn test_series_interpolate_linear() {
    let s = Series::new(vec![
        Value::Float(1.0),
        Value::Null,
        Value::Float(3.0),
        Value::Null,
        Value::Float(5.0),
    ]);
    let filled = s.interpolate();
    assert_eq!(filled.iloc(1).unwrap(), &Value::Float(2.0));
    assert_eq!(filled.iloc(3).unwrap(), &Value::Float(4.0));
}

#[test]
fn test_series_interpolate_nearest() {
    let s = Series::new(vec![
        Value::Float(1.0),
        Value::Null,
        Value::Null,
        Value::Float(10.0),
    ]);
    let filled = s.interpolate_method("nearest");
    // Position 1 is closer to 0 (1.0), position 2 is closer to 3 (10.0)
    assert!(filled.iloc(1).unwrap().as_float().is_some());
    assert!(filled.iloc(2).unwrap().as_float().is_some());
}

#[test]
fn test_series_null_count() {
    let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3), Value::Null]);
    assert_eq!(s.null_count(), 2);
    assert!(s.has_nulls());
}

#[test]
fn test_series_no_nulls() {
    let s = Series::new(vec![1, 2, 3]);
    assert_eq!(s.null_count(), 0);
    assert!(!s.has_nulls());
}

#[test]
fn test_series_cummax() {
    let s = Series::new(vec![1.0, 3.0, 2.0, 5.0, 4.0]);
    let result = s.cummax();
    assert_eq!(result.iloc(0).unwrap(), &Value::Float(1.0));
    assert_eq!(result.iloc(1).unwrap(), &Value::Float(3.0));
    assert_eq!(result.iloc(2).unwrap(), &Value::Float(3.0));
    assert_eq!(result.iloc(3).unwrap(), &Value::Float(5.0));
    assert_eq!(result.iloc(4).unwrap(), &Value::Float(5.0));
}

#[test]
fn test_series_cummin() {
    let s = Series::new(vec![5.0, 3.0, 4.0, 1.0, 2.0]);
    let result = s.cummin();
    assert_eq!(result.iloc(0).unwrap(), &Value::Float(5.0));
    assert_eq!(result.iloc(1).unwrap(), &Value::Float(3.0));
    assert_eq!(result.iloc(2).unwrap(), &Value::Float(3.0));
    assert_eq!(result.iloc(3).unwrap(), &Value::Float(1.0));
    assert_eq!(result.iloc(4).unwrap(), &Value::Float(1.0));
}

#[test]
fn test_series_value_counts() {
    let s = Series::new(vec!["a", "b", "a", "c", "a", "b"]);
    let counts = s.value_counts();
    assert_eq!(counts.get(&Value::String("a".into())), Some(&3));
    assert_eq!(counts.get(&Value::String("b".into())), Some(&2));
    assert_eq!(counts.get(&Value::String("c".into())), Some(&1));
}

#[test]
fn test_series_notna() {
    let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]);
    let result = s.notna();
    assert_eq!(result, vec![true, false, true]);
}

#[test]
fn test_series_values() {
    let s = Series::new(vec![1, 2, 3]);
    let vals = s.values();
    assert_eq!(vals.len(), 3);
    assert_eq!(vals[0], Value::Int(1));
}

#[test]
fn test_series_to_f64() {
    let s = Series::new(vec![1.0, 2.0, 3.0]);
    let vals = s.to_f64().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_series_to_i64() {
    let s = Series::new(vec![1, 2, 3]);
    let vals = s.to_i64().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn test_series_iloc_many() {
    let s = Series::new(vec![10, 20, 30, 40, 50]);
    let result = s.iloc_many(&[0, 2, 4]).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.iloc(0).unwrap(), &Value::Int(10));
    assert_eq!(result.iloc(1).unwrap(), &Value::Int(30));
    assert_eq!(result.iloc(2).unwrap(), &Value::Int(50));
}

#[test]
fn test_series_set_name() {
    let mut s = Series::new(vec![1, 2, 3]);
    s.set_name("my_series");
    assert_eq!(s.name(), Some("my_series"));
}

#[test]
fn test_series_with_index() {
    let s = Series::with_index(
        vec![1, 2, 3],
        Index::String(vec!["a".into(), "b".into(), "c".into()]),
    )
    .unwrap();
    assert_eq!(s.index().len(), 3);
}

#[test]
fn test_series_astype() {
    let s = Series::new(vec![1, 2, 3]);
    let floats = s.astype("float");
    assert_eq!(floats.iloc(0).unwrap(), &Value::Float(1.0));
}

#[test]
fn test_series_round() {
    let s = Series::new(vec![1.234, 2.567, 3.891]);
    let rounded = s.round(1);
    assert_eq!(rounded.iloc(0).unwrap(), &Value::Float(1.2));
    assert_eq!(rounded.iloc(1).unwrap(), &Value::Float(2.6));
}

#[test]
fn test_series_mask() {
    let s = Series::new(vec![1, 2, 3, 4, 5]);
    let masked = s.mask(&[false, true, false, true, false], &Value::Int(0));
    assert_eq!(masked.iloc(0).unwrap(), &Value::Int(1));
    assert_eq!(masked.iloc(1).unwrap(), &Value::Int(0));
    assert_eq!(masked.iloc(2).unwrap(), &Value::Int(3));
    assert_eq!(masked.iloc(3).unwrap(), &Value::Int(0));
}

#[test]
fn test_series_where_cond() {
    let s = Series::new(vec![1, 2, 3, 4, 5]);
    let result = s.where_cond(&[true, false, true, false, true], &Value::Int(-1));
    assert_eq!(result.iloc(0).unwrap(), &Value::Int(1));
    assert_eq!(result.iloc(1).unwrap(), &Value::Int(-1));
    assert_eq!(result.iloc(2).unwrap(), &Value::Int(3));
}

#[test]
fn test_series_replace_map() {
    let s = Series::new(vec![1, 2, 3, 2, 1]);
    let mut replacements = std::collections::HashMap::new();
    replacements.insert(Value::Int(1), Value::Int(10));
    replacements.insert(Value::Int(2), Value::Int(20));
    let result = s.replace_map(&replacements);
    assert_eq!(result.iloc(0).unwrap(), &Value::Int(10));
    assert_eq!(result.iloc(1).unwrap(), &Value::Int(20));
    assert_eq!(result.iloc(2).unwrap(), &Value::Int(3)); // unchanged
}

#[test]
fn test_series_rolling() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = s.rolling(3).mean();
    // First two values should be null (not enough data)
    assert!(result.iloc(0).unwrap().is_null());
    assert!(result.iloc(1).unwrap().is_null());
    // Third value: mean(1, 2, 3) = 2.0
    assert_eq!(result.iloc(2).unwrap(), &Value::Float(2.0));
}

#[test]
fn test_series_min_periods() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = s.rolling(3).min_periods(2).mean();
    // With min_periods=2, we should get results starting from index 1
    assert!(result.iloc(0).unwrap().is_null()); // Only 1 value
    assert!(!result.iloc(1).unwrap().is_null()); // 2 values >= min_periods
}

#[test]
fn test_series_loc() {
    let s = Series::with_index(
        vec![10, 20, 30],
        Index::String(vec!["a".into(), "b".into(), "c".into()]),
    )
    .unwrap();

    let val = s.loc(&Value::String("b".into())).unwrap();
    assert_eq!(val, &Value::Int(20));
}

#[test]
fn test_series_reset_index() {
    let mut s = Series::with_index(
        vec![10, 20, 30],
        Index::String(vec!["a".into(), "b".into(), "c".into()]),
    )
    .unwrap();

    s.reset_index();
    assert_eq!(s.index().len(), 3);
    // Index should now be Range type
}

#[test]
fn shift_by_more_than_len_gives_all_null() {
    let s = Series::new(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    let forward = s.shift(5);
    let backward = s.shift(-5);
    assert_eq!(forward.len(), 3);
    assert_eq!(backward.len(), 3);
    for i in 0..3 {
        assert!(forward.iloc(i).unwrap().is_null());
        assert!(backward.iloc(i).unwrap().is_null());
    }
}
