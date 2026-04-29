# Continue Here - hipfire-gfx1151-opt

## Goal

Make `lemon-mlx-engine` on gfx1151/Strix Halo much closer to hipfire for local Qwen-class LLM inference, while preserving correctness and MLX model compatibility.

The current concrete target is:

- Model family: Qwen3.5-0.8B.
- Lemon artifact: `mlx-community/Qwen3.5-0.8B-MLX-4bit`.
- hipfire artifact: `qwen3.5:0.8b`.
- Prompt: `benchmarks/prompts/qwen35_add_raw.py`.
- Latest result: Lemon `125.86 tok/s`, hipfire `229.5 tok/s`.

## Current State

Branch: `hipfire-gfx1151-opt`

Recent commits before this handover:

```text
5868105 Cache Qwen35 decode debug flags
d5bd8d1 Add model artifact audit to qwen bench
42a21a1 Optimize MLX Qwen decode on gfx1151
```

Current work adds:

- Correct HIP GDN launch sizing.
- HIP GDN parity harness.
- Opt-in full-model Qwen3.5 decode routing through common GDN/HIP GDN.
- Reproducibility docs and artifacts in this directory.

## Architecture Notes

### Qwen3.5 decode path

Relevant files:

- `src/llm/models/qwen35_moe.cpp`
- `src/llm/models/qwen35.cpp`
- `src/common/gated_delta.cpp`
- `include/mlx-lm/common/quantized_linear.h`

Qwen3.5 has a hybrid stack:

- Linear/GDN layers.
- Full attention layers.
- MoE or dense MLP projections depending on model variant.
- A large tied lm-head projection through `model.embed_tokens.weight`.

The decode fast path in Qwen3.5 fuses or reduces launches for:

- input projection packing,
- decode conv+silu,
- GDN recurrence,
- gated norm,
- final output projection.

The old compiled decode recurrence remains the default. The common GDN route is opt-in:

```bash
LEMON_MLX_GDN_ENABLE_HIP=1
```

or:

```bash
LEMON_MLX_QWEN35_USE_COMMON_GDN=1
```

### HIP GDN bug and fix

MLX custom CUDA/HIP kernels take `grid` in total threads. The prior GDN code passed block counts:

```cpp
{1, (Dv + 3) / 4, B * Hv}
```

with a threadgroup:

```cpp
{32, 4, 1}
```

That launched only one X lane per block. The reduction then read uninitialized shared memory and produced NaNs/repeated-token attractors.

The fixed launch is:

```cpp
{32, ((Dv + 3) / 4) * 4, B * Hv}
```

with the same threadgroup:

```cpp
{32, 4, 1}
```

### Quantized linear path

`include/mlx-lm/common/quantized_linear.h` routes registered quantized weights through:

```cpp
mx::quantized_matmul(x, w, scales, biases, true, group_size, bits)
```

The remaining performance gap is dominated by quantized GEMV/QMV:

- tied lm-head projection is about `1.3 ms/token` in the timing profile,
- most layer projections are tens to low hundreds of microseconds each,
- exact artifact mismatch means hipfire has a strong format advantage via MQ4G256/FWHT.

## Environment

Known working environment:

- ROCm enabled build in `build-gfx1151`.
- GPU backend uses HIP/ROCm, not Vulkan, for the current benchmark.
- `ldd build-gfx1151/bench` links `libamdhip64`, `librocblas`, `libhiprand`, `libhiprtc`, `libhipblaslt`, and `librocrand`.

The benchmark scripts warn that the AMD GPU can be in a low-power state. Treat single-run numbers as directional unless using repeated fresh processes.

## Quick Start

Build:

```bash
cmake --build build-gfx1151 --target bench test_gdn_hip_parity -j 8
```

Run GDN parity:

```bash
./build-gfx1151/test_gdn_hip_parity
```

Expected:

```text
qwen35_0p8b_decode    status=ok
qwen35_large_decode   status=ok
qwen35_0p8b_prefill4  status=ok
```

Run default correctness smoke:

```bash
RUNS=1 TIMEOUT_SECONDS=240 scripts/smoke_qwen35_0p8b_correctness.sh
```

Run HIP GDN correctness smoke:

```bash
RUNS=1 \
TIMEOUT_SECONDS=240 \
ALLOW_EXPERIMENTAL_HIP_GDN=1 \
LEMON_MLX_GDN_ENABLE_HIP=1 \
REQUIRE_NATIVE_DEQUANT_HASH_MATCH=0 \
scripts/smoke_qwen35_0p8b_correctness.sh
```

Run canonical hipfire comparison:

```bash
RUNS=3 \
TIMEOUT_SECONDS=240 \
ALLOW_EXPERIMENTAL_HIP_GDN=1 \
LEMON_MLX_GDN_ENABLE_HIP=1 \
scripts/bench_qwen35_0p8b_vs_hipfire.sh
```

Run QMV column sweep:

```bash
RUNS=2 \
TIMEOUT_SECONDS=240 \
QWEN35_MAX_TOKENS=56 \
ALLOW_EXPERIMENTAL_HIP_GDN=1 \
LEMON_MLX_GDN_ENABLE_HIP=1 \
scripts/study_qmv_cols_gfx1151.sh
```

Run quantized-shape timing profile:

```bash
RUNS=1 \
MAX_TOKENS=5 \
PROFILE_TIMING=1 \
ALLOW_EXPERIMENTAL_HIP_GDN=1 \
LEMON_MLX_GDN_ENABLE_HIP=1 \
scripts/profile_quant_shapes.sh
```

## Important Environment Flags

```bash
LEMON_MLX_GDN_ENABLE_HIP=1
```

Enables the opt-in common/HIP GDN route.

```bash
LEMON_MLX_GDN_DISABLE_HIP=1
```

Forces the compiled MLX recurrence. The smoke scripts set this unless `ALLOW_EXPERIMENTAL_HIP_GDN=1`.

```bash
LEMON_MLX_QWEN35_KEEP_QUANTIZED=1
```

Keeps Qwen3.5 MLX 4-bit weights quantized instead of dequantizing.

```bash
LEMON_MLX_DEQUANTIZE_WEIGHTS=1
```

Forces dequantized load. Useful for correctness comparison, slow for perf.

```bash
MLX_ROCM_QMV_COLS_PER_BLOCK=16|32|64
```

Overrides QMV cols per block. Current default is effectively 64 for dominant Qwen3.5 decode shapes and remains best or tied-best.

```bash
MLX_ROCM_QMV_ENABLE_TILED=1
MLX_ROCM_QMV_TILE_N=8
```

Opt-in tiled QMV. Treat as experimental because prior testing found possible hangs or wrong logits on gfx1151.

## Reproducibility Artifacts

Artifacts are committed under:

```text
hipfire-gfx1151-opt/artifacts/
```

Most useful files:

- `qwen35_0p8b_vs_hipfire_20260429T193927Z/findings.md`
- `qwen35_0p8b_vs_hipfire_20260429T193927Z/aggregate.tsv`
- `qwen35_0p8b_vs_hipfire_20260429T193927Z/artifacts.tsv`
- `qwen35_0p8b_correctness_20260429T194016Z/aggregate.tsv`
- `qwen35_0p8b_correctness_20260429T194035Z/aggregate.tsv`
- `quant_profile_shapes_20260429T194109Z/quant_profile_family_totals.tsv`
- `qmv_cols_gfx1151_20260429T194222Z/aggregate.tsv`

No artifact copied here is over 100 MB.

## Next Engineering Steps

1. Build an exact artifact path.
   - Either load hipfire MQ4G256/FWHT in MLX or generate an MLX artifact from the same quantized weights.
   - This removes the biggest ambiguity in MLX-vs-hipfire comparison.

2. Attack tied lm-head QMV.
   - It is the largest single measured decode cost at about `1.3 ms/token`.
   - Options:
     - specialize vocab projection for top-k/argmax instead of full logits,
     - port hipfire-style rotated/MQ4 kernel,
     - add a persistent or split-vocab reduction kernel for large `N`.

3. Revisit tiled QMV safely.
   - The `TILE_N=8` single checks were coherent and modestly faster in short runs.
   - Need a dedicated parity harness and repetition gate before defaulting it.

4. Add full timing for GDN and non-linear ops.
   - Current quant profile only times quantized embedding/linear operations.
   - Need equivalent timing for RMSNorm, conv+silu, attention, GDN, and sampling.

5. Continue full-model benchmarking with fresh processes.
   - Avoid running GPU benchmarks in parallel.
   - Use `timeout`.
   - Keep logs and prompt hashes.

6. Only after HIP/ROCm path is stable, return to Vulkan/RTX 4090 portability.
   - Current branch is still ROCm/HIP-centered.
   - Vulkan/4090 work needs its own backend feature branch or a clean MLX backend plan.
