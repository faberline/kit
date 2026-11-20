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
