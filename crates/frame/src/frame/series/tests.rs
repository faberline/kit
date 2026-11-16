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

#[test]
fn min_and_max_skip_nulls() {
    let s = Series::new(vec![
        Value::Int(4),
        Value::Null,
        Value::Int(2),
        Value::Int(9),
    ]);
    assert_eq!(s.min().unwrap(), Value::Int(2));
    assert_eq!(s.max().unwrap(), Value::Int(9));
}

#[test]
fn min_and_max_of_only_nulls_fail() {
    let s = Series::new(vec![Value::Null, Value::Null]);
    assert!(s.min().is_err());
    assert!(s.max().is_err());
}

#[test]
fn count_ignores_nulls() {
    let s = Series::new(vec![Value::Int(1), Value::Null, Value::Int(3)]);
    assert_eq!(s.count(), 2);
}

#[test]
fn map_keeps_name_and_applies_function() {
    let s = Series::with_name(vec![1, 2, 3], "n");
    let doubled = s.map(|v| match v {
        Value::Int(x) => Value::Int(x * 2),
        other => other.clone(),
    });
    assert_eq!(doubled.name(), Some("n"));
    assert_eq!(
        doubled.values(),
        &[Value::Int(2), Value::Int(4), Value::Int(6)]
    );
}

#[test]
fn to_dict_maps_labels_to_values() {
    let s = Series::new(vec![10, 20]);
    let d = s.to_dict();
    assert_eq!(d.len(), 2);
    assert_eq!(d.get(&Value::Int(0)), Some(&Value::Int(10)));
    assert_eq!(d.get(&Value::Int(1)), Some(&Value::Int(20)));
}

#[test]
fn arithmetic_rejects_length_mismatch() {
    let a = Series::new(vec![1.0, 2.0]);
    let b = Series::new(vec![1.0]);
    assert!((&a + &b).is_err());
    assert!((&a - &b).is_err());
    assert!((&a * &b).is_err());
    assert!((&a / &b).is_err());
}

#[test]
fn division_by_zero_gives_null() {
    let a = Series::new(vec![1.0, 4.0]);
    let b = Series::new(vec![0.0, 2.0]);
    let q = (&a / &b).unwrap();
    assert_eq!(q.iloc(0).unwrap(), &Value::Null);
    assert_eq!(q.iloc(1).unwrap(), &Value::Float(2.0));
}

#[test]
fn with_index_rejects_length_mismatch() {
    assert!(Series::with_index(vec![1, 2], Index::Range(3)).is_err());
}

#[test]
fn is_empty_reports_zero_length() {
    assert!(Series::new(Vec::<i32>::new()).is_empty());
    assert!(!Series::new(vec![1]).is_empty());
}

#[test]
fn loc_finds_label_and_rejects_missing_label() {
    let s = Series::new(vec![1, 2]);
    assert_eq!(s.loc(&Value::Int(1)).unwrap(), &Value::Int(2));
    assert!(s.loc(&Value::Int(9)).is_err());
}

#[test]
fn iloc_many_keeps_order_and_rejects_out_of_bounds() {
    let s = Series::with_name(vec![10, 20, 30], "x");
    let picked = s.iloc_many(&[2, 0]).unwrap();
    assert_eq!(picked.name(), Some("x"));
    assert_eq!(picked.values(), &[Value::Int(30), Value::Int(10)]);
    assert!(s.iloc_many(&[0, 5]).is_err());
}

#[test]
fn numeric_conversions_reject_text() {
    let text = Series::new(vec![Value::String("a".to_string())]);
    assert!(text.to_f64().is_err());
    assert!(text.to_i64().is_err());
    assert_eq!(
        Series::new(vec![1, 2]).to_i64().unwrap(),
        vec![1_i64, 2_i64]
    );
}

#[test]
fn reductions_reject_text_and_all_null_input() {
    let text = Series::new(vec![Value::String("a".to_string())]);
    assert!(text.sum().is_err());
    assert!(text.mean().is_err());
    let nulls = Series::new(vec![Value::Null]);
    assert!(nulls.mean().is_err());
}

#[test]
fn filter_rejects_mask_of_wrong_length() {
    let s = Series::new(vec![1, 2]);
    assert!(s.filter(&[true]).is_err());
}

#[test]
fn clip_bounds_numbers_and_keeps_text() {
    let s = Series::new(vec![
        Value::Float(1.0),
        Value::Float(5.0),
        Value::Float(10.0),
    ]);
    assert_eq!(
        s.clip(2.0, 8.0).values(),
        &[Value::Float(2.0), Value::Float(5.0), Value::Float(8.0)]
    );
    let text = Series::new(vec![Value::String("x".to_string())]);
    assert_eq!(
        text.clip(0.0, 1.0).values(),
        &[Value::String("x".to_string())]
    );
}

#[test]
fn std_and_var_use_sample_formula() {
    let s = Series::new(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert!((s.var().unwrap() - 32.0 / 7.0).abs() < 1e-12);
    assert!((s.std().unwrap() - (32.0_f64 / 7.0).sqrt()).abs() < 1e-12);
}

#[test]
fn std_and_var_need_two_values() {
    let s = Series::new(vec![1.0]);
    assert!(s.std().is_err());
    assert!(s.var().is_err());
}

#[test]
fn median_odd_even_and_empty() {
    assert_eq!(Series::new(vec![3.0, 1.0, 2.0]).median().unwrap(), 2.0);
    assert_eq!(Series::new(vec![1.0, 2.0, 3.0, 4.0]).median().unwrap(), 2.5);
    assert!(Series::new(Vec::<f64>::new()).median().is_err());
    assert!(Series::new(vec![Value::Null]).median().is_err());
}

#[test]
fn duplicated_supports_last_and_all() {
    let s = Series::new(vec![1, 2, 1, 3, 2]);
    assert_eq!(s.duplicated("last"), vec![true, true, false, false, false]);
    assert_eq!(s.duplicated("first"), vec![false, false, true, false, true]);
    assert_eq!(s.duplicated("all"), vec![true, true, true, false, true]);
}

#[test]
fn drop_duplicates_keeps_first_occurrence() {
    let s = Series::new(vec![1, 2, 1, 3]);
    assert_eq!(
        s.drop_duplicates("first").values(),
        &[Value::Int(1), Value::Int(2), Value::Int(3)]
    );
}

#[test]
fn quantile_interpolates_and_rejects_bad_input() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(s.quantile(0.0).unwrap(), 1.0);
    assert_eq!(s.quantile(0.5).unwrap(), 2.5);
    assert_eq!(s.quantile(1.0).unwrap(), 4.0);
    assert!(s.quantile(1.5).is_err());
    assert!(Series::new(vec![Value::Null]).quantile(0.5).is_err());
}

#[test]
fn corr_handles_linear_constant_and_bad_input() {
    let x = Series::new(vec![1.0, 2.0, 3.0]);
    let y = Series::new(vec![2.0, 4.0, 6.0]);
    assert!((x.corr(&y).unwrap() - 1.0).abs() < 1e-12);

    let flat = Series::new(vec![1.0, 1.0, 1.0]);
    assert_eq!(flat.corr(&y).unwrap(), 0.0);

    assert!(Series::new(vec![1.0])
        .corr(&Series::new(vec![2.0]))
        .is_err());
    assert!(x.corr(&Series::new(vec![1.0])).is_err());
    let text = Series::new(vec![Value::String("a".to_string()); 3]);
    assert!(text.corr(&y).is_err());
}

#[test]
fn astype_converts_each_dtype() {
    assert_eq!(
        Series::new(vec![Value::Int(3)]).astype("int").values(),
        &[Value::Int(3)]
    );
    assert_eq!(
        Series::new(vec![Value::Int(2), Value::Bool(true)])
            .astype("f64")
            .values(),
        &[Value::Float(2.0), Value::Float(1.0)]
    );
    assert_eq!(
        Series::new(vec![Value::Int(7)]).astype("str").values(),
        &[Value::String("7".to_string())]
    );
    assert_eq!(
        Series::new(vec![Value::Bool(false)])
            .astype("bool")
            .values(),
        &[Value::Bool(false)]
    );
    assert_eq!(
        Series::new(vec![Value::Int(5)]).astype("unknown").values(),
        &[Value::Int(5)]
    );
}

#[test]
fn where_cond_keeps_true_and_replaces_false() {
    let s = Series::new(vec![1, 2, 3]);
    assert_eq!(
        s.where_cond(&[true, false, true], &Value::Int(0)).values(),
        &[Value::Int(1), Value::Int(0), Value::Int(3)]
    );
}

#[test]
fn abs_keeps_non_numbers() {
    let s = Series::new(vec![
        Value::Int(-2),
        Value::Float(-1.5),
        Value::String("t".to_string()),
    ]);
    assert_eq!(
        s.abs().values(),
        &[
            Value::Int(2),
            Value::Float(1.5),
            Value::String("t".to_string())
        ]
    );
}

#[test]
fn round_rounds_numbers_and_keeps_text() {
    let s = Series::new(vec![Value::Float(1.2345), Value::String("t".to_string())]);
    let r = s.round(2);
    assert!(matches!(r.iloc(0).unwrap(), Value::Float(f) if (f - 1.23).abs() < 1e-12));
    assert_eq!(r.iloc(1).unwrap(), &Value::String("t".to_string()));
}

#[test]
fn sub_and_mul_compute_elementwise() {
    let a = Series::new(vec![5.0, 3.0]);
    let b = Series::new(vec![1.0, 1.0]);
    assert_eq!(
        (&a - &b).unwrap().values(),
        &[Value::Float(4.0), Value::Float(2.0)]
    );
    let c = Series::new(vec![2.0, 3.0]);
    let d = Series::new(vec![4.0, 5.0]);
    assert_eq!(
        (&c * &d).unwrap().values(),
        &[Value::Float(8.0), Value::Float(15.0)]
    );
}

#[test]
fn arithmetic_gives_null_for_null_operands() {
    let a = Series::new(vec![Value::Null, Value::Int(1)]);
    let b = Series::new(vec![Value::Int(1), Value::Int(2)]);
    assert_eq!(
        (&a + &b).unwrap().values(),
        &[Value::Null, Value::Float(3.0)]
    );
    assert_eq!(
        (&a - &b).unwrap().values(),
        &[Value::Null, Value::Float(-1.0)]
    );
    assert_eq!(
        (&a * &b).unwrap().values(),
        &[Value::Null, Value::Float(2.0)]
    );
}
