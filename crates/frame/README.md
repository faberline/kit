# frame

DataFrame and Series for Rust, inspired by pandas. Each cell holds a typed `Value`.

- `DataFrame`, `Series`, `Value`, `Index`, `MultiIndex`, `FrameError`: construction, indexing, selection, null handling, sorting and statistics.
- Group-by, joins, pivot, stack and unstack, rolling, expanding and EWM windows (`frame::frame::ops`).
- CSV, binary and workbook files, and JSON with the Cargo feature `io-extra` (`frame::frame::io`).

## Example

```rust
use frame::{DataFrame, Series};

let df = DataFrame::from_columns(vec![
    ("a", Series::new(vec![1_i64, 2, 3])),
])
.unwrap();
assert_eq!(df.shape(), (3, 1));
```

## Test

```bash
cargo test -p frame
```
