# hipfire-gfx1151-opt Changelog

## 2026-04-29 - HIP GDN parity and Qwen3.5 decode speedup

### Added

- Added `examples/test_gdn_hip_parity.cpp`, a standalone parity harness for the common Gated Delta Net recurrence.
  - Compares the compiled MLX recurrence against the opt-in HIP recurrence.
  - Covers Qwen3.5-0.8B decode shape, larger Qwen3.5 GDN decode shape, and T=4 prefill.
  - Uses bf16 bounded-random inputs and fails if output/state max absolute error exceeds `5e-2`.
- Added branch-local reproducibility artifacts under `hipfire-gfx1151-opt/artifacts/`.
  - `qwen35_0p8b_vs_hipfire_20260429T193927Z/`
  - `qwen35_0p8b_correctness_20260429T194016Z/`
  - `qwen35_0p8b_correctness_20260429T194035Z/`
  - `quant_profile_shapes_20260429T194109Z/`
  - `qmv_cols_gfx1151_20260429T194222Z/`

### Changed

- Fixed the HIP GDN custom-kernel launch in `src/common/gated_delta.cpp`.
  - MLX custom CUDA/HIP kernels expect `grid` in total thread counts, not block counts.
  - Previous launch used `{1, ceil(Dv/4), B*Hv}` with a `{32,4,1}` block, creating one X lane per block and causing reductions over uninitialized shared memory.
  - New launch uses `{32, ceil(Dv/4)*4, B*Hv}`.
- Re-enabled a controlled Qwen3.5/Qwen3.5-MoE decode route through the common GDN implementation.
  - `src/llm/models/qwen35.cpp`
  - `src/llm/models/qwen35_moe.cpp`
  - Activated by `LEMON_MLX_GDN_ENABLE_HIP=1` or `LEMON_MLX_QWEN35_USE_COMMON_GDN=1`.
  - Default path stays on the previously validated compiled decode recurrence.
- Updated `CMakeLists.txt` to build the new `test_gdn_hip_parity` example.

### Fixed

- Fixed the invalid HIP GDN result that previously produced attractor output such as repeated `!` when routed through full Qwen3.5 decode.
- Verified the HIP GDN kernel now matches the compiled recurrence:

```text
qwen35_0p8b_decode    y_diff=4.882812e-04  state_diff=9.765625e-04  status=ok
qwen35_large_decode   y_diff=9.765625e-04  state_diff=9.765625e-04  status=ok
qwen35_0p8b_prefill4  y_diff=9.765625e-04  state_diff=2.929688e-03  status=ok
```

### Benchmarks

- Canonical Qwen3.5-0.8B 4-bit vs hipfire, HIP GDN enabled:
  - Lemon MLX: `125.86 tok/s` median decode, 3/3 ok.
  - hipfire: `229.5 tok/s` median decode, 3/3 ok.
  - Lemon is now `0.548x` hipfire on the comparable prompt, up from the prior `~105.47 tok/s`.
- Artifact audit:
  - Source BF16 HF: `Qwen/Qwen3.5-0.8B`, md5 `7d61ccb37ef5d4ad330fec4ad0e06934`, no quant.
  - Lemon MLX 4-bit: `mlx-community/Qwen3.5-0.8B-MLX-4bit`, md5 `439d1796c1a3ce24ef83681eabd656db`, affine int4 group64.
  - hipfire: `qwen3.5:0.8b`, md5 `0769fdeaa08e82bd6ed555e3f151d04b`, MQ4G256/FWHT-rotated.
  - Exact artifact match remains `0`; these are same source-family, not the same quant artifact.
- Correctness smoke, default path:
  - BF16: `28.2849 tok/s`, 1/1 ok.
  - 4-bit dequantized: `28.4 tok/s`, 1/1 ok.
  - 4-bit native: `106.13 tok/s`, 1/1 ok.
  - Native/dequantized strict hash match passed.
- Correctness smoke, HIP GDN enabled and native/dequant hash equality disabled:
  - BF16: `29.0429 tok/s`, 1/1 ok.
  - 4-bit dequantized: `95.63 tok/s`, 1/1 ok.
  - 4-bit native: `124.22 tok/s`, 1/1 ok.
- QMV cols sweep with HIP GDN enabled:
  - Qwen3.5 default: `127.900 tok/s`, 2/2 ok.
  - Qwen3.5 cols 64: `127.090 tok/s`, 2/2 ok.
  - Qwen3.5 cols 32: `122.870 tok/s`, 2/2 ok.
  - Qwen3.5 cols 16: `116.280 tok/s`, 2/2 ok.
  - Current default 64-column behavior remains best or tied-best.
- Quant profile with timing enabled:
  - Dominant decode cost is still quantized linear work.
  - Qwen3.5 tied lm-head projection: `~1299 us` per decode call.
  - GDN output projections: `~119 us` average.
  - GDN input projections: `~100 us` average.
  - MLP gate/up projections: `~86 us` average.
  - MLP down projections: `~69 us` average.

### Known Issues

- Lemon MLX still trails hipfire on Qwen3.5-0.8B 4-bit decode: `125.86` vs `229.5 tok/s`.
- The comparison is not exact-artifact-equal:
  - MLX uses affine int4 group64 safetensors.
  - hipfire uses MQ4G256/FWHT-rotated weights.
  - Exact apples-to-apples requires an MQ4 loader/converter path in MLX or a hipfire-compatible artifact reader.
- HIP GDN enabled changes greedy continuation hashes relative to the dequantized path.
  - Outputs are coherent and pass substring/repetition gates.
  - Strict native/dequantized byte-hash parity should remain disabled when benchmarking HIP GDN.
- Tiled ROCm QMV remains opt-in.
  - `MLX_ROCM_QMV_ENABLE_TILED=1 MLX_ROCM_QMV_TILE_N=8` gave a coherent single run around `127.48 tok/s` for 56 tokens and `132.81 tok/s` for 32 tokens.
  - Prior work found tiled QMV could hang or return wrong logits on gfx1151, so it is not safe as default yet.
- No Jupyter notebook exists in this repo at this checkpoint; experiment artifacts are TSV/log files instead.

## Earlier branch commits

### `42a21a1 Optimize MLX Qwen decode on gfx1151`

- Added initial Qwen3.5 decode optimizations and benchmark support.

### `d5bd8d1 Add model artifact audit to qwen bench`

- Added `scripts/audit_model_artifacts.py`.
- Updated `scripts/bench_qwen35_0p8b_vs_hipfire.sh` to write artifact metadata and support `REQUIRE_EXACT_ARTIFACT_MATCH=1`.

### `5868105 Cache Qwen35 decode debug flags`

- Cached Qwen3.5 environment/debug flags in `src/llm/models/qwen35_moe.cpp`.
- Passed correctness but did not materially improve speed.
