# Handover - hipfire-gfx1151-opt

## Executive Summary

The MLX Qwen3.5-0.8B 4-bit path is now materially faster and more coherent than at the start of this continuation.

Main accomplishment:

- Fixed the experimental HIP GDN kernel launch.
- Added a parity harness proving HIP GDN matches the compiled MLX recurrence.
- Routed Qwen3.5 decode through the common/HIP GDN path behind an opt-in flag.
- Improved Lemon MLX canonical decode from about `105 tok/s` to `125.86 tok/s`.
- Preserved default correctness behavior and captured reproducible artifacts.

Remaining gap:

- hipfire still reaches `229.5 tok/s` on its `qwen3.5:0.8b` artifact.
- Lemon reaches `125.86 tok/s` on `mlx-community/Qwen3.5-0.8B-MLX-4bit`.
- The comparison is not exact-artifact-equal because hipfire uses MQ4G256/FWHT and MLX uses affine int4 group64.

## What Changed

### Code

- `src/common/gated_delta.cpp`
  - Fixed HIP custom-kernel launch grid from block counts to total thread counts.
  - This removed NaN parity failures and repeated-token attractors.

- `src/llm/models/qwen35_moe.cpp`
  - Added opt-in decode route through `gated_delta_update(...)`.
  - Enabled when `LEMON_MLX_GDN_ENABLE_HIP=1` or `LEMON_MLX_QWEN35_USE_COMMON_GDN=1`.
  - Default compiled recurrence remains unchanged.

- `src/llm/models/qwen35.cpp`
  - Same opt-in common GDN route for non-MoE Qwen3.5.

- `examples/test_gdn_hip_parity.cpp`
  - New standalone parity test.

- `CMakeLists.txt`
  - Adds `test_gdn_hip_parity`.

### Documentation and artifacts

- `hipfire-gfx1151-opt/CHANGELOG.md`
- `hipfire-gfx1151-opt/CONTINUE_HERE.md`
- `hipfire-gfx1151-opt/SESSION_LOG_2026-04-29.md`
- `hipfire-gfx1151-opt/HANDOVER.md`
- `hipfire-gfx1151-opt/artifacts/...`

## Verification Completed

Build:

```bash
cmake --build build-gfx1151 --target bench test_gdn_hip_parity -j 8
```

Parity:

```bash
./build-gfx1151/test_gdn_hip_parity
```

Result:

```text
qwen35_0p8b_decode    status=ok
qwen35_large_decode   status=ok
qwen35_0p8b_prefill4  status=ok
```

Canonical benchmark:

```bash
RUNS=3 TIMEOUT_SECONDS=240 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/bench_qwen35_0p8b_vs_hipfire.sh
```

Result:

```text
lemon   3/3 ok  median 125.86 tok/s
hipfire 3/3 ok  median 229.5 tok/s
```

Default smoke:

```bash
RUNS=1 TIMEOUT_SECONDS=240 scripts/smoke_qwen35_0p8b_correctness.sh
```

Result:

```text
bf16               1/1 ok  28.2849 tok/s
4bit dequantized   1/1 ok  28.4 tok/s
4bit native        1/1 ok  106.13 tok/s
```

HIP GDN smoke:

```bash
RUNS=1 TIMEOUT_SECONDS=240 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 REQUIRE_NATIVE_DEQUANT_HASH_MATCH=0 scripts/smoke_qwen35_0p8b_correctness.sh
```

Result:

```text
bf16               1/1 ok  29.0429 tok/s
4bit dequantized   1/1 ok  95.63 tok/s
4bit native        1/1 ok  124.22 tok/s
```

QMV cols sweep:

```bash
RUNS=2 TIMEOUT_SECONDS=240 QWEN35_MAX_TOKENS=56 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/study_qmv_cols_gfx1151.sh
```

Result:

```text
default 127.900 tok/s
64      127.090 tok/s
32      122.870 tok/s
16      116.280 tok/s
```

## Critical Caveats

- Do not claim exact parity with hipfire yet. The artifacts differ.
- Do not enable tiled QMV by default yet.
- Do not require native/dequantized output hash equality with HIP GDN enabled.
- Do not run multiple GPU benchmarks in parallel; previous runs were invalidated that way.
- The machine has had SSH-drop/hang concerns in the broader project history; use `timeout` around every benchmark.

## Best Next Task

The next agent should focus on the quantized matmul gap, specifically the tied lm-head and MQ4/FWHT artifact mismatch.

Recommended next sequence:

1. Add a small deterministic quantized-matmul parity harness for the dominant shapes:
   - `[1x1x1024] @ [248320x128]`
   - `[1x1x1024] @ [8224x128]`
   - `[1x1x2048] @ [1024x256]`
   - `[1x1x3584] @ [1024x448]`
2. Test `MLX_ROCM_QMV_ENABLE_TILED=1 MLX_ROCM_QMV_TILE_N=8` across 3 to 5 fresh processes with:
   - output hash stability,
   - token hash stability,
   - repetition gates,
   - timing profile.
3. If tiled QMV is coherent, isolate why prior tiled paths hung or produced wrong logits.
4. If tiled QMV is not robust, build a hipfire-inspired special kernel for large-N decode projection.
5. In parallel, design an MQ4G256/FWHT loader/converter so the MLX benchmark can run the exact same quantized artifact as hipfire.

## One-command continuation

```bash
cd /home/reckon/projects/lemon-mlx-engine
cmake --build build-gfx1151 --target bench test_gdn_hip_parity -j 8
./build-gfx1151/test_gdn_hip_parity
RUNS=3 TIMEOUT_SECONDS=240 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/bench_qwen35_0p8b_vs_hipfire.sh
```

## Suggested gnhf prompt

```text
Continue /home/reckon/projects/lemon-mlx-engine on branch hipfire-gfx1151-opt. Read hipfire-gfx1151-opt/CONTINUE_HERE.md, HANDOVER.md, CHANGELOG.md, and SESSION_LOG_2026-04-29.md first. The HIP GDN kernel launch was fixed and Qwen3.5-0.8B MLX 4-bit improved to 125.86 tok/s vs hipfire 229.5 tok/s. Continue by optimizing the remaining quantized QMV/GEMV bottleneck, especially tied lm-head [1x1x1024] @ [248320x128], while preserving correctness. Add parity harnesses for QMV tiled/warp paths before enabling any fast path. Use fresh-process benchmarks with prompt hashes, artifact audits, timeout guards, and committed artifacts. Do not run GPU benchmarks in parallel.
```
