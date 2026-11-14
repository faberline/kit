//! Root re-export check for arraykit. Fails to compile if a re-export is removed.

use arraykit::{DType, NdArray, Shape};

fn takes<T>() {}

#[test]
fn arraykit_root_reexports_exist() {
    takes::<DType>();
    takes::<Shape>();
    takes::<NdArray<f64>>();
}
