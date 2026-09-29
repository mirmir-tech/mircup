# mircup

Mircup is a model-agnostic Rust execution layer for CPUs. It provides the
tensor primitives `libmir` uses when no accelerator is present, while keeping
model, tokenizer, and application policy out of this crate.

## What it provides

- dense row-major `f32` tensors decoded from `F32`, `F16`, and `BF16`
  checkpoint payloads;
- multithreaded linear maps backed by `gemm` and Rayon;
- layer normalisation with and without bias;
- exact GELU and ReLU;
- rotate-half rotary position embeddings;
- bidirectional attention with length masks and banded windows;
- token embedding tables that stay in their 16-bit encoding until read.

## Add it to a project

```toml
[dependencies]
mircup = "0.1.0"
```

## Checks

```sh
make check
```
