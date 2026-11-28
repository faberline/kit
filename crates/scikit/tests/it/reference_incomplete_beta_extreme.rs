//! Reference test for `scikit::stats::incomplete_beta` with a tiny `a` or a huge `b`.
//! Each expected value is read from `fixtures/incomplete_beta_extreme_reference.tsv`,
//! which holds mpmath 1.4.1 values at 80 digits. The key is `a|b|x`.

use scikit::stats::incomplete_beta;

const FIXTURE: &str = include_str!("../fixtures/incomplete_beta_extreme_reference.tsv");

const ROWS: usize = 11;
const TOL: f64 = 1e-12;

fn parse_number(key: &str, text: &str) -> f64 {
    text.parse()
        .unwrap_or_else(|_| panic!("fixture key {key}: bad number {text}"))
}

#[test]
fn incomplete_beta_extreme_matches_mpmath() {
    let mut checked = 0;
    for line in FIXTURE.lines().skip(1) {
        let mut fields = line.split('\t');
        let key = fields.next().expect("fixture line has no key");
        let value = fields
            .next()
            .unwrap_or_else(|| panic!("fixture key {key}: no value column"));

        let parts: Vec<&str> = key.split('|').collect();
        assert_eq!(parts.len(), 3, "fixture key {key}: want a|b|x");
        let a = parse_number(key, parts[0]);
        let b = parse_number(key, parts[1]);
        let x = parse_number(key, parts[2]);
        let expected = parse_number(key, value);

        let actual = incomplete_beta(a, b, x);
        assert!(
            (actual - expected).abs() <= TOL,
            "{key}: got {actual}, expected {expected}"
        );
        checked += 1;
    }
    assert_eq!(checked, ROWS, "the fixture must keep all {ROWS} rows");
}
