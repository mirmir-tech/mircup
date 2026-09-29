# Mircup Agent Notes

Mircup is the standalone CPU execution layer of the Mirmir project, next to
`mirtal` (Metal) and `mircuda` (CUDA). It must not depend on libmir, Mirmir,
model, tokenizer, runtime, or server crates.

## Hard Rules

- Rust source files must never exceed 250 lines; split large modules by focused
  responsibility.
- Split modules only as `module/mod.rs` plus focused files such as
  `module/feature.rs`. Tests use `module/tests.rs`.
- Nothing model-specific belongs here: no tensor names, architectures, or
  checkpoint layouts. Operations take shapes and values only.
- Activations are `f32`. Accumulate at least in `f32`; do not silently lower
  precision for speed.
- Shape and data mismatches caused by callers return typed `Error` values.
  Panics are reserved for internal invariants of this crate.
- Keep every `unsafe` block inside `linear/backend/` with a `SAFETY` comment
  proving the strided views stay inside their slices. `linear/gemm.rs` checks
  dimensions and bounds before any backend runs.
- Matrix products use Apple Accelerate on macOS, which reaches the AMX units,
  and the `gemm` crate elsewhere. Do not replace either without measuring
  against the upstream PyTorch CPU runtime on the same machine.
- `fastmath` holds vectorisable `exp` and `erf`; keep their error bounds
  pinned by tests when changing them.
- Use `thiserror` conversions and `?`.
- Keep nightly rustfmt and clippy clean with all targets.
- Do not add Python or PyTorch to any runtime or build path.

## Checks

`make check` runs formatting, Clippy with warnings denied, and the tests.
