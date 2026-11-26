#[cfg(all(feature = "fft", feature = "stats"))]
mod bench_smoke;
#[cfg(feature = "fft")]
mod fft;
#[cfg(all(
    feature = "fft",
    feature = "signal",
    feature = "integrate",
    feature = "ts"
))]
mod reference_numeric;
#[cfg(feature = "ts")]
mod reference_ts;
#[cfg(feature = "stats")]
mod stats;

#[cfg(feature = "stats")]
mod reference_stats;
#[cfg(feature = "stats")]
mod reference_stats_grid;

#[cfg(all(feature = "fft", feature = "signal"))]
mod reference_fft_signal;

#[cfg(all(feature = "interpolate", feature = "optimize", feature = "integrate"))]
mod reference_calc;

#[cfg(all(feature = "spatial", feature = "sparse"))]
mod reference_spatial_sparse;

#[cfg(feature = "signal")]
mod reference_filter_scipy;

#[cfg(feature = "linear_model")]
mod reference_linear_model;
