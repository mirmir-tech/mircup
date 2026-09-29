//! Matrix products: Apple Accelerate on macOS, where it reaches the AMX
//! matrix units, and the portable `gemm` crate elsewhere.

#[cfg(target_os = "macos")]
mod accelerate;
#[cfg(not(target_os = "macos"))]
mod portable;

#[cfg(target_os = "macos")]
pub use accelerate::multiply;
#[cfg(not(target_os = "macos"))]
pub use portable::multiply;
