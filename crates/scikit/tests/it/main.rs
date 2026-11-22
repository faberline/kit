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
