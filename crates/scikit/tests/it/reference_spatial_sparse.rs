//! Reference-value tests for the `spatial` and `sparse` rows of `docs/reference-coverage.md`.
//!
//! The expected values are in `fixtures/spatial_sparse_reference.tsv`. The generator
//! `tools/reference/scikit/generate.py` writes them from NumPy 2.4.6 and SciPy
//! 1.17.1; its `source` column names the SciPy expression of each value. The
//! inputs come from the integer formulas below, so Rust and Python build the
//! same values.

use scikit::sparse::{CscMatrix, CsrMatrix};
use scikit::spatial::{
    cdist, chebyshev, convex_hull, convex_hull_area, cosine_distance, euclidean, manhattan,
    minkowski, pdist, squareform, KdTree,
};

const FIXTURE: &str = include_str!("../fixtures/spatial_sparse_reference.tsv");

/// Relative tolerance of every reference value (the table's `1e-9 rel`).
const REL_TOL: f64 = 1e-9;

/// Absolute part of every tolerance (the table's `1e-12 abs`).
const ABS_FLOOR: f64 = 1e-12;

/// The values of the fixture line `key`. Panics when the key is missing.
fn reference(key: &str) -> Vec<f64> {
    let line = FIXTURE
        .lines()
        .find(|l| l.split('\t').next() == Some(key))
        .unwrap_or_else(|| panic!("fixture key `{key}` not found"));
    line.split('\t')
        .nth(1)
        .unwrap_or_else(|| panic!("fixture key `{key}` has no value column"))
        .split(',')
        .map(|v| {
            v.parse::<f64>()
                .unwrap_or_else(|e| panic!("bad value in `{key}`: {e}"))
        })
        .collect()
}

/// Whether the fixture has a line `key`.
fn has_key(key: &str) -> bool {
    FIXTURE.lines().any(|l| l.split('\t').next() == Some(key))
}

/// Every element of `actual` is within `REL_TOL * |e| + ABS_FLOOR` of the
/// fixture value `e` of `key`, and the lengths are equal.
fn assert_close(actual: &[f64], key: &str) {
    let expected = reference(key);
    assert_eq!(actual.len(), expected.len(), "length of `{key}`");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = REL_TOL * e.abs() + ABS_FLOOR;
        assert!(
            (a - e).abs() <= tol,
            "`{key}`[{i}] = {a}, expected {e} (tol {tol})"
        );
    }
}

/// A long list. The fixture keeps the whole list under `prefix` when it is
/// short; otherwise it keeps `prefix_head`, `prefix_tail`, `prefix_abs_sum`
/// and `prefix_len`, and those are compared.
fn assert_list(actual: &[f64], prefix: &str) {
    if has_key(prefix) {
        assert_close(actual, prefix);
        return;
    }
    assert_close(&actual[..5], &format!("{prefix}_head"));
    assert_close(&actual[actual.len() - 5..], &format!("{prefix}_tail"));
    let abs_sum: f64 = actual.iter().map(|v| v.abs()).sum();
    assert_close(&[abs_sum], &format!("{prefix}_abs_sum"));
    assert_close(&[actual.len() as f64], &format!("{prefix}_len"));
}

/// The generator's `P`: 50 points in 3 dimensions.
fn points_p() -> Vec<Vec<f64>> {
    (0..50)
        .map(|i| {
            (0..3)
                .map(|d| ((i * 31 + d * 17 + i * d) % 23) as f64 / 7.0)
                .collect()
        })
        .collect()
}

/// The generator's `H`: 30 points in 2 dimensions.
fn points_h() -> Vec<(f64, f64)> {
    (0..30)
        .map(|i| (((i * 13 + 5) % 17) as f64, ((i * 7 + 3) % 19) as f64))
        .collect()
}

/// The generator's `T`: 40 points in 3 dimensions for the KD-tree.
fn points_t() -> Vec<Vec<f64>> {
    (0..40)
        .map(|i| {
            (0..3)
                .map(|d| ((i * 37 + d * 53 + i * d * 7) % 101) as f64 / 10.0)
                .collect()
        })
        .collect()
}

/// The generator's `Q`: 5 query points in 3 dimensions.
fn queries() -> Vec<Vec<f64>> {
    (0..5)
        .map(|k| {
            (0..3)
                .map(|d| ((k * 29 + d * 11 + k * d * 3) % 97) as f64 / 9.7 + 0.013)
                .collect()
        })
        .collect()
}

/// The generator's `A`: 50 x 50 dense, row-major.
fn dense_a() -> Vec<f64> {
    (0..50)
        .flat_map(|i| {
            (0..50).map(move |j| {
                if (i * 7 + j * 3) % 11 == 0 {
                    ((i + 2 * j) % 9 + 1) as f64
                } else {
                    0.0
                }
            })
        })
        .collect()
}

/// The generator's `B`: 50 x 50 dense, row-major.
fn dense_b() -> Vec<f64> {
    (0..50)
        .flat_map(|i| {
            (0..50).map(move |j| {
                if (i * 5 + j * 2) % 13 == 0 {
                    ((3 * i + j) % 7 + 1) as f64
                } else {
                    0.0
                }
            })
        })
        .collect()
}

/// The generator's `xv`: a vector of 50 values.
fn xvec() -> Vec<f64> {
    (0..50).map(|j| ((j * 5 + 3) % 11) as f64 / 4.0).collect()
}

#[test]
fn euclidean_matches_scipy() {
    let p = points_p();
    let got: Vec<f64> = (1..50).map(|k| euclidean(&p[0], &p[k])).collect();
    assert_list(&got, "euclidean_from0");
}

#[test]
fn manhattan_matches_scipy() {
    let p = points_p();
    let got: Vec<f64> = (1..50).map(|k| manhattan(&p[0], &p[k])).collect();
    assert_list(&got, "manhattan_from0");
}

#[test]
fn chebyshev_matches_scipy() {
    let p = points_p();
    let got: Vec<f64> = (1..50).map(|k| chebyshev(&p[0], &p[k])).collect();
    assert_list(&got, "chebyshev_from0");
}

#[test]
fn minkowski_matches_scipy() {
    let p = points_p();
    let got: Vec<f64> = (1..50).map(|k| minkowski(&p[0], &p[k], 3.0)).collect();
    assert_list(&got, "minkowski3_from0");
}

#[test]
fn cosine_distance_matches_scipy() {
    let p = points_p();
    let got: Vec<f64> = (1..50).map(|k| cosine_distance(&p[0], &p[k])).collect();
    assert_list(&got, "cosine_from0");
}

#[test]
fn pdist_matches_scipy() {
    let p = points_p();
    for name in ["euclidean", "cityblock", "chebyshev", "cosine"] {
        let got = pdist(&p, name);
        assert_list(&got, &format!("pdist_{name}"));
    }
}

#[test]
fn cdist_matches_scipy() {
    let p = points_p();
    for name in ["euclidean", "cityblock", "chebyshev", "cosine"] {
        let got = cdist(&p[..20], &p[20..], name);
        assert_list(&got, &format!("cdist_{name}"));
    }
}

#[test]
fn squareform_matches_scipy() {
    let p = points_p();
    let sq = squareform(&pdist(&p, "euclidean"));
    assert_list(&sq, "squareform_euclidean");
    assert_close(&sq[7 * 50..8 * 50], "squareform_row7");
}

#[test]
fn convex_hull_matches_scipy() {
    let mut idx = convex_hull(&points_h());
    idx.sort_unstable();
    let got: Vec<f64> = idx.iter().map(|&i| i as f64).collect();
    assert_close(&got, "hull_vertices");
}

#[test]
fn convex_hull_area_matches_scipy() {
    let area = convex_hull_area(&points_h());
    assert_close(&[area], "hull_area");
}

#[test]
fn kdtree_query_matches_scipy() {
    let tree = KdTree::new(&points_t());
    let q = queries();
    for (k, point) in q.iter().enumerate() {
        let nn = tree.query(point, 5);
        let index: Vec<f64> = nn.iter().map(|n| n.index as f64).collect();
        let dist: Vec<f64> = nn.iter().map(|n| n.distance).collect();
        assert_close(&index, &format!("kdtree_query{k}_index"));
        assert_close(&dist, &format!("kdtree_query{k}_dist"));
    }
}

#[test]
fn kdtree_query_ball_matches_scipy() {
    let tree = KdTree::new(&points_t());
    let q = queries();
    for (k, point) in q.iter().enumerate() {
        let ball = tree.query_ball(point, 4.5);
        let mut index: Vec<f64> = ball.iter().map(|n| n.index as f64).collect();
        index.sort_by(|a, b| a.partial_cmp(b).expect("finite index"));
        let dist: Vec<f64> = ball.iter().map(|n| n.distance).collect();
        assert_close(&index, &format!("kdtree_ball{k}_index"));
        assert_close(&dist, &format!("kdtree_ball{k}_dist"));
    }
}

#[test]
fn csr_dot_matches_scipy() {
    let a = CsrMatrix::from_dense(&dense_a(), 50, 50);
    assert_list(&a.dot(&xvec()), "csr_dot");
}

#[test]
fn csr_add_matches_scipy() {
    let a = CsrMatrix::from_dense(&dense_a(), 50, 50);
    let b = CsrMatrix::from_dense(&dense_b(), 50, 50);
    let s = a.add(&b);
    let dense = s.to_dense();
    assert_close(&[s.nnz() as f64], "csr_add_nnz");
    assert_close(&[dense.iter().sum::<f64>()], "csr_add_sum");
    assert_list(&dense[7 * 50..8 * 50], "csr_add_row7");
}

#[test]
fn csr_scale_matches_scipy() {
    let a = CsrMatrix::from_dense(&dense_a(), 50, 50);
    let dense = a.scale(2.5).to_dense();
    assert_close(&[dense.iter().sum::<f64>()], "csr_scale_sum");
    assert_list(&dense[7 * 50..8 * 50], "csr_scale_row7");
}

#[test]
fn csr_transpose_matches_scipy() {
    let a = CsrMatrix::from_dense(&dense_a(), 50, 50);
    let t = a.transpose();
    assert_close(&[t.nnz() as f64], "csr_transpose_nnz");
    assert_list(&t.dot(&xvec()), "csr_transpose_dot");
    assert_list(&t.to_dense()[7 * 50..8 * 50], "csr_transpose_row7");
}

#[test]
fn csr_density_matches_scipy() {
    let a = CsrMatrix::from_dense(&dense_a(), 50, 50);
    assert_close(&[a.density()], "csr_density");
}

#[test]
fn csc_dot_matches_scipy() {
    let a = CscMatrix::from_dense(&dense_a(), 50, 50);
    assert_list(&a.dot(&xvec()), "csc_dot");
}

#[test]
fn csc_add_matches_scipy() {
    let a = CscMatrix::from_dense(&dense_a(), 50, 50);
    let b = CscMatrix::from_dense(&dense_b(), 50, 50);
    let s = a.add(&b);
    let dense = s.to_dense();
    assert_close(&[s.nnz() as f64], "csc_add_nnz");
    assert_close(&[dense.iter().sum::<f64>()], "csc_add_sum");
    assert_list(&dense[7 * 50..8 * 50], "csc_add_row7");
}

#[test]
fn csc_scale_matches_scipy() {
    let a = CscMatrix::from_dense(&dense_a(), 50, 50);
    let dense = a.scale(2.5).to_dense();
    assert_close(&[dense.iter().sum::<f64>()], "csc_scale_sum");
    assert_list(&dense[7 * 50..8 * 50], "csc_scale_row7");
}

#[test]
fn csc_density_matches_scipy() {
    let a = CscMatrix::from_dense(&dense_a(), 50, 50);
    assert_close(&[a.density()], "csc_density");
}
