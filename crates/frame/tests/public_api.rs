//! Root re-export check for frame. Fails to compile if a re-export is removed.

use frame::{DataFrame, Series};

fn takes<T>() {}

#[test]
fn frame_root_reexports_exist() {
    takes::<DataFrame>();
    takes::<Series>();
}
