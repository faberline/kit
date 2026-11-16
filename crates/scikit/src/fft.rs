//! FFT module: radix-2 for powers of two and a direct DFT for other lengths.
//!
//! - **fft / ifft**: Complex FFT and inverse
//! - **rfft / irfft**: Real-valued FFT
//! - **fftfreq / rfftfreq**: Frequency bins

mod transform;

pub use transform::{fft, fftfreq, ifft, irfft, rfft, rfftfreq, Complex};
