# Session Log - 2026-04-29

## Scope

This log captures the continuation of the Lemon MLX vs hipfire optimization work on branch `hipfire-gfx1151-opt`.

The session resumed from a compacted context after earlier work had already:

- cloned and built `lemon-mlx-engine`,
- benchmarked Qwen3.5-0.8B MLX 4-bit against hipfire,
- added artifact auditing,
- found that MLX and hipfire were not using exact-identical quant artifacts,
- tried and reverted several slower or invalid experiments,
- identified experimental HIP GDN as a promising but incorrect path.

## User Requests

Requests visible in the active continuation context:

> "Proceed"

> "Fully comprehensively summarize what has been accomplished, files and scripts touched, next steps etc so that we can easily continue this in a new session. Be comprehensive and:
>
> 1. Update/create these documentation files:
>    - feat(or branch)_name/CHANGELOG.md - version history with features, fixes, known issues
>    - feat(or branch)_name/CONTINUE_HERE.md - quick start, architecture, commands, environment
>    - feat(or branch)_name/SESSION_LOG_<DATE>.md - full session transcript including:
>      * All user requests (quoted)
>      * Technical decisions made and rationale
>      * Errors encountered and fixes
>      * Benchmark/test results
>      * Commands reference
>      * Git history from session
>
> 2. Keep things organized and reuse existing files, scripts and docs as needed, always update the current Jupyter notebook with scripts and visualizations if we are running experiments
>
> 3. Commit using semantic commit with appropriate title and all changed files including artifacts. If large use lfs, if >100mb don't add them. Use --no-verify even if tests fail.
>
> 4. Prepare a handover document that based on all the above allows the next agent to fully continue the work"

Relevant prior user requests from the compacted session context:

> "clone, fork, Build and run this https://github.com/lemonade-sdk/lemon-mlx-engine. Then benchmark its performance vs ours in hipfire and then proceed to fully optimize it using all the learnings we have from hipfire. Basically i want fully functional mlx running on this machine and being able to use mlx-lm to run any mlx supported llm. Reason about this step by step and do a good job. We will want a rust version of mlx-lm or some form of runner that is built in rust and will use hipfire as the inspiration for it"

> "Let's run the same model on both hipfire and mlx-engine i.e. qwen3.5-0.8b-4bit"

> "Let's fix it, basically proceed until mlx-engine is atleast as performant and cohesive as hipfire"

> "create a command that i can give gnhf to run autonomously and fix this"

> "Check what the gnhf agent did"

> "WE need to test the same exact models also why are we so far off yet we have hipfire prior art ?"

> "Proceed"

## Technical Decisions and Rationale

### 1. Stop guessing and add a HIP GDN parity harness

Earlier full-model experiments showed:

- common MLX GDN path was coherent and roughly same speed as the hand-written compiled decode path,
- `LEMON_MLX_GDN_ENABLE_HIP=1` was faster-looking but produced invalid attractor output,
- direct performance numbers were meaningless until HIP GDN correctness was proven.

Decision:

- Add `examples/test_gdn_hip_parity.cpp`.
- Compare compiled MLX GDN against HIP GDN directly.
- Cover both current 0.8B shapes and larger Qwen3.5 shapes.

Rationale:

- A standalone recurrence parity harness isolates the kernel from tokenizer/model/sampling noise.
- It avoids wasting time on full-model benchmark runs with an invalid kernel.

### 2. Fix MLX custom-kernel launch semantics

Initial parity result:

```text
qwen35_0p8b_decode    y_diff=nan  state_diff=nan  status=fail
qwen35_large_decode   y_diff=nan  state_diff=nan  status=fail
qwen35_0p8b_prefill4  y_diff=nan  state_diff=nan  status=fail
```

Investigation found MLX custom CUDA/HIP kernels define `grid` in total threads, not blocks. The existing GDN call passed:

```cpp
{1, (Dv + 3) / 4, B * Hv}
```

with:

```cpp
{32, 4, 1}
```

This produced a block with only one active X lane, then reductions read uninitialized shared memory.

Fix:

```cpp
{32, ((Dv + 3) / 4) * 4, B * Hv}
```

with the same threadgroup:

```cpp
{32, 4, 1}
```

### 3. Keep HIP GDN opt-in

Decision:

- The fixed HIP GDN route is available with `LEMON_MLX_GDN_ENABLE_HIP=1`.
- Default path remains the existing compiled decode recurrence.

Rationale:

- Default smoke with strict native/dequantized output hash equality remains valuable.
- HIP GDN is now coherent but can change greedy continuation hashes, so it should remain explicit until more model/prompt coverage lands.

### 4. Preserve exact artifact audit

Decision:

- Continue reporting exact artifact mismatch in the hipfire comparison.

Rationale:

- Lemon MLX and hipfire are not currently running identical quantized weights:
  - Lemon MLX: affine int4 group64 safetensors.
  - hipfire: MQ4G256/FWHT-rotated MQ4.
- The gap cannot be fully interpreted without that caveat.

### 5. No notebook update

The repo currently has no `.ipynb` files. No notebook was created because the user asked to update the current notebook if one exists while running experiments. Experiment outputs were preserved as TSV/log artifacts instead.

## Errors Encountered and Fixes

### HIP GDN all-NaN parity failure

Command:

```bash
./build-gfx1151/test_gdn_hip_parity
```

Failure:

```text
qwen35_0p8b_decode    y_diff=nan  state_diff=nan  status=fail
qwen35_large_decode   y_diff=nan  state_diff=nan  status=fail
qwen35_0p8b_prefill4  y_diff=nan  state_diff=nan  status=fail
```

Cause:

- Wrong custom-kernel grid semantics.

Fix:

- Change GDN HIP launch grid to total threads.

Retest:

```text
qwen35_0p8b_decode    y_diff=4.882812e-04  state_diff=9.765625e-04  status=ok
qwen35_large_decode   y_diff=9.765625e-04  state_diff=9.765625e-04  status=ok
qwen35_0p8b_prefill4  y_diff=9.765625e-04  state_diff=2.929688e-03  status=ok
```

### Strict native/dequantized output hash mismatch under HIP GDN

Command:

```bash
RUNS=1 TIMEOUT_SECONDS=240 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/smoke_qwen35_0p8b_correctness.sh
```

Failure:

```text
native/dequantized output hash mismatch on run 1: native=26c903070b130ae6 dequantized=f68df19f930bcc8e
```

Interpretation:

- Both outputs were coherent and passed substring/repetition checks.
- HIP GDN changes greedy continuation versus dequantized path, likely due small numerical differences.

Fix for benchmarking:

```bash
REQUIRE_NATIVE_DEQUANT_HASH_MATCH=0
```

The default non-HIP path still passes strict native/dequantized hash equality.

### Tiled QMV remains unsafe as default

Single checks:

```bash
MLX_ROCM_QMV_ENABLE_TILED=1 MLX_ROCM_QMV_TILE_N=8
```

were coherent:

- `132.81 tok/s` for a 32-token single run.
- `127.48 tok/s` for a 56-token single run.

But previous work found tiled QMV could hang or return wrong logits on gfx1151. It remains opt-in until a dedicated parity and stress harness exists.

## Benchmark and Test Results

### GDN parity

Command:

```bash
./build-gfx1151/test_gdn_hip_parity
```

Result:

```text
qwen35_0p8b_decode    B=1 T=1 Hk=16 Hv=16 Dk=128 Dv=128 y_diff=4.882812e-04 state_diff=9.765625e-04 status=ok
qwen35_large_decode   B=1 T=1 Hk=16 Hv=64 Dk=192 Dv=128 y_diff=9.765625e-04 state_diff=9.765625e-04 status=ok
qwen35_0p8b_prefill4  B=1 T=4 Hk=16 Hv=16 Dk=128 Dv=128 y_diff=9.765625e-04 state_diff=2.929688e-03 status=ok
```

### Direct A/B smoke

Default compiled recurrence:

```text
decode_tok_s=97.22
status=ok
```

HIP GDN route:

```text
decode_tok_s=127.10
status=ok
```

Both used `mlx-community/Qwen3.5-0.8B-MLX-4bit`, the same prompt, 32 max tokens, and native quantized weights.

### Canonical Lemon vs hipfire benchmark

Command:

```bash
timeout 900 env RUNS=3 TIMEOUT_SECONDS=240 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/bench_qwen35_0p8b_vs_hipfire.sh
```

Output artifact:

```text
hipfire-gfx1151-opt/artifacts/qwen35_0p8b_vs_hipfire_20260429T193927Z/
```

Aggregate:

```text
engine   model                                   runs ok_runs median_decode_tok_s avg_decode_tokens
lemon    mlx-community/Qwen3.5-0.8B-MLX-4bit    3    3       125.86              56.00
hipfire  qwen3.5:0.8b                            3    3       229.5               56.00
```

Artifact audit:

```text
source  Qwen/Qwen3.5-0.8B                   md5=7d61ccb37ef5d4ad330fec4ad0e06934  quant=none
lemon   mlx-community/Qwen3.5-0.8B-MLX-4bit md5=439d1796c1a3ce24ef83681eabd656db  quant=affine int4 group64
hipfire qwen3.5:0.8b                         md5=0769fdeaa08e82bd6ed555e3f151d04b  quant=MQ4G256/FWHT
exact_artifact_match=0
```

### Default smoke

Command:

```bash
timeout 900 env RUNS=1 TIMEOUT_SECONDS=240 scripts/smoke_qwen35_0p8b_correctness.sh
```

Artifact:

```text
hipfire-gfx1151-opt/artifacts/qwen35_0p8b_correctness_20260429T194016Z/
```

Aggregate:

```text
qwen35_0p8b_bf16             1/1 ok  28.2849 tok/s
qwen35_0p8b_4bit_dequantized 1/1 ok  28.4 tok/s
qwen35_0p8b_4bit_native      1/1 ok  106.13 tok/s
```

### HIP GDN smoke

Command:

```bash
timeout 900 env RUNS=1 TIMEOUT_SECONDS=240 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 REQUIRE_NATIVE_DEQUANT_HASH_MATCH=0 scripts/smoke_qwen35_0p8b_correctness.sh
```

Artifact:

```text
hipfire-gfx1151-opt/artifacts/qwen35_0p8b_correctness_20260429T194035Z/
```

Aggregate:

```text
qwen35_0p8b_bf16             1/1 ok  29.0429 tok/s
qwen35_0p8b_4bit_dequantized 1/1 ok  95.63 tok/s
qwen35_0p8b_4bit_native      1/1 ok  124.22 tok/s
```

### Quant profile timing

Command:

```bash
timeout 600 env RUNS=1 MAX_TOKENS=5 PROFILE_TIMING=1 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/profile_quant_shapes.sh
```

Artifact:

```text
hipfire-gfx1151-opt/artifacts/quant_profile_shapes_20260429T194109Z/
```

Highlights:

```text
lm_head_tied_embedding decode: 5 calls, 6.497670 ms total, 1299.534 us avg
gdn_output_projection decode: 72 calls, 8.556172 ms total, 118.836 us avg
gdn_input_projection decode: 72 calls, 7.175948 ms total, 99.666 us avg
mlp_gate_up_projection decode: 96 calls, 8.276653 ms total, 86.215 us avg
mlp_down_projection decode: 96 calls, 6.628403 ms total, 69.046 us avg
```

### QMV cols sweep

Command:

```bash
timeout 900 env RUNS=2 TIMEOUT_SECONDS=240 QWEN35_MAX_TOKENS=56 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/study_qmv_cols_gfx1151.sh
```

Artifact:

```text
hipfire-gfx1151-opt/artifacts/qmv_cols_gfx1151_20260429T194222Z/
```

Qwen3.5 result:

```text
default 127.900 tok/s 2/2 ok
64      127.090 tok/s 2/2 ok
32      122.870 tok/s 2/2 ok
16      116.280 tok/s 2/2 ok
```

Qwen3 result:

```text
default 154.950 tok/s 2/2 ok
64      154.160 tok/s 2/2 ok
32      151.200 tok/s 2/2 ok
16      147.180 tok/s 2/2 ok
```

## Commands Reference

Build:

```bash
cmake --build build-gfx1151 --target bench test_gdn_hip_parity -j 8
```

GDN parity:

```bash
./build-gfx1151/test_gdn_hip_parity
```

Single default model run:

```bash
timeout 240 env -u LEMON_MLX_DEQUANTIZE_WEIGHTS LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 LEMON_MLX_GDN_DISABLE_HIP=1 ./build-gfx1151/bench mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt-file benchmarks/prompts/qwen35_add_raw.py --raw --max-tokens 32 --warmup-decode-steps 1 --temperature 0 --top-p 1 --expect-substring 'return x + y' --expect-prefix '    return x + y' --fail-on-attractor --print-output
```

Single HIP GDN model run:

```bash
timeout 240 env -u LEMON_MLX_DEQUANTIZE_WEIGHTS LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 LEMON_MLX_GDN_ENABLE_HIP=1 ./build-gfx1151/bench mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt-file benchmarks/prompts/qwen35_add_raw.py --raw --max-tokens 32 --warmup-decode-steps 1 --temperature 0 --top-p 1 --expect-substring 'return x + y' --expect-prefix '    return x + y' --fail-on-attractor --print-output
```

Canonical benchmark:

```bash
RUNS=3 TIMEOUT_SECONDS=240 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/bench_qwen35_0p8b_vs_hipfire.sh
```

Default smoke:

```bash
RUNS=1 TIMEOUT_SECONDS=240 scripts/smoke_qwen35_0p8b_correctness.sh
```

HIP GDN smoke:

```bash
RUNS=1 TIMEOUT_SECONDS=240 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 REQUIRE_NATIVE_DEQUANT_HASH_MATCH=0 scripts/smoke_qwen35_0p8b_correctness.sh
```

QMV cols sweep:

```bash
RUNS=2 TIMEOUT_SECONDS=240 QWEN35_MAX_TOKENS=56 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/study_qmv_cols_gfx1151.sh
```

Quant timing profile:

```bash
RUNS=1 MAX_TOKENS=5 PROFILE_TIMING=1 ALLOW_EXPERIMENTAL_HIP_GDN=1 LEMON_MLX_GDN_ENABLE_HIP=1 scripts/profile_quant_shapes.sh
```

Tiled QMV smoke only:

```bash
timeout 240 env -u LEMON_MLX_DEQUANTIZE_WEIGHTS LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 LEMON_MLX_GDN_ENABLE_HIP=1 MLX_ROCM_QMV_ENABLE_TILED=1 MLX_ROCM_QMV_TILE_N=8 ./build-gfx1151/bench mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt-file benchmarks/prompts/qwen35_add_raw.py --raw --max-tokens 56 --warmup-decode-steps 1 --temperature 0 --top-p 1 --expect-substring 'return x + y' --max-token-freq 0.40 --max-token-run 16 --attractor-min-tokens 16
```

## Git History From Session

Branch before this handover:

```text
5868105 Cache Qwen35 decode debug flags
d5bd8d1 Add model artifact audit to qwen bench
42a21a1 Optimize MLX Qwen decode on gfx1151
6db801d origin/main Merge pull request #19 from bong-water-water-bong/feat/cache-mlx-source
```

Uncommitted work prepared for this handover:

```text
M  CMakeLists.txt
M  src/common/gated_delta.cpp
M  src/llm/models/qwen35.cpp
M  src/llm/models/qwen35_moe.cpp
A  examples/test_gdn_hip_parity.cpp
A  hipfire-gfx1151-opt/*
```

Semantic commit planned:

```text
feat: enable correct HIP GDN path for Qwen35
```

## Files and Scripts Touched

Code:

- `CMakeLists.txt`
- `src/common/gated_delta.cpp`
- `src/llm/models/qwen35.cpp`
- `src/llm/models/qwen35_moe.cpp`
- `examples/test_gdn_hip_parity.cpp`

Scripts used:

- `scripts/bench_qwen35_0p8b_vs_hipfire.sh`
- `scripts/smoke_qwen35_0p8b_correctness.sh`
- `scripts/study_qmv_cols_gfx1151.sh`
- `scripts/profile_quant_shapes.sh`
- `scripts/audit_model_artifacts.py`

Docs/artifacts:

- `hipfire-gfx1151-opt/CHANGELOG.md`
- `hipfire-gfx1151-opt/CONTINUE_HERE.md`
- `hipfire-gfx1151-opt/HANDOVER.md`
- `hipfire-gfx1151-opt/SESSION_LOG_2026-04-29.md`
- `hipfire-gfx1151-opt/artifacts/`

## Next Steps

1. Add deterministic QMV parity and timing harnesses for the dominant shapes.
2. Stress-test tiled QMV with `TILE_N=8`; only promote if coherent across fresh-process runs.
3. Specialize tied lm-head projection, possibly by top-k/argmax or split-vocab reduction.
4. Implement exact MQ4G256/FWHT compatibility or conversion to remove artifact mismatch.
5. Extend profiler to non-quantized kernels and sampling.
6. Keep all claims tied to prompt md5, binary md5, artifact audit, and fresh-process benchmark logs.
