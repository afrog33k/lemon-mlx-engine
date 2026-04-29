# Next Steps - QMV Optimization (2026-04-29)

## Summary

Tested QMV optimization options for lemon-mlx-engine on gfx1151. Profiled both lemon-mlx-engine and hipfire to identify bottlenecks. Found that `MLX_ROCM_QMV_COLS_PER_BLOCK=64` provides a ~2% improvement, but the fundamental gap is due to group_size difference (64 vs 256).

## Benchmark Results

Average decode speed (4 runs, Qwen3.5-0.8B 4-bit, 32 tokens):

| Setting | Speed (tok/s) | Range |
|---------|---------------|-------|
| cols64 | 126.0 | 121.8 - 128.2 |
| tiled8 (TILE_N=8) | 124.1 | 120.5 - 131.0 |
| default | 123.4 | 118.6 - 129.1 |

## Profiling Analysis

### Lemon MLX Bottleneck
```
lm_head_tied_embedding: 5 calls, 6.497670 ms total, 1299.534 us avg
Shape: [1x1x1024] @ [248320x128] (M=1, N=248320 vocab, K=1024)
```

This is the **single largest decode cost** in lemon-mlx-engine.

### Hipfire Profile
Top kernels (1 step, ctx=32):
```
1. gemv_hfq4g256_residual    48 calls   645.2us   13.44us/call   (20.7%)
2. fused_qkvza_hfq4g256      18 calls   490.4us   27.24us/call   (15.7%)
3. fused_rmsnorm_mq_rotate   48 calls   437.5us    9.11us/call   (14.1%)
...
```

**Key finding: lm_head projection is NOT in the top kernels!** This means hipfire's lm_head is either:
- Very fast (<50 us per call)
- Profiled under a different name
- Fused with another operation

## Root Cause: group_size Difference

| Aspect | Lemon MLX | Hipfire |
|--------|-----------|----------|
| Format | affine int4 | MQ4G256 (FWHT-rotated) |
| Group size | 64 | 256 |
| Scale/bias loads | 4x more | 1x |
| Kernel | MLX QMV | Custom MQ4 kernel |
| lm_head time | 1297 us/call | <50 us (estimated) |

**The key difference is group_size:**
1. **Group size 256**: 4x fewer scale/bias loads = better memory bandwidth
2. For lm_head with N=248320, K=1024:
   - MLX (gs=64): 16 groups/row × 248320 columns = ~4M scale/bias loads
   - Hipfire (gs=256): 4 groups/row × 248320 columns = ~1M scale/bias loads

## Performance Gap Breakdown

- lemon-mlx-engine: ~126 tok/s (with cols64)
- hipfire: ~229 tok/s
- Gap: ~1.8x

The lm_head alone accounts for ~6.5 ms per 5 steps (1.3 ms/step) in lemon, which is ~20% of total decode time. If hipfire's lm_head is 25x faster (50 us vs 1297 us), that alone would explain a ~20% speedup.

The remaining 60% gap comes from other projections also benefiting from MQ4 format.

## Recommended Next Steps

### Option 1: Re-quantize model with group_size=256 (high impact, medium effort)
- **Status**: MLX kernel support added (commit 65b40a9)
- Remaining: Re-quantize Qwen3.5-0.8B model with group_size=256
- Requires access to FP32 source weights
- **Estimated impact**: 20-40% overall speedup (lm_head 3-4x faster)

### Option 2: Add MQ4G256/FWHT support to MLX (high impact, high effort)
- Implement FWHT rotation in MLX
- Add MQ4 dequantization kernel
- Would enable exact artifact parity with hipfire
- **Estimated impact**: 1.5-2x speedup

### Option 3: Specialized tied lm-head kernel (medium impact, medium effort)
- For M=1, very large N (vocab), consider top-k/argmax
- Avoid computing full logits for greedy sampling
- **Estimated impact**: 30-50% speedup on lm_head for greedy

### Option 4: Accept current constraints
- Document that hipfire's MQ4 format is fundamentally faster
- Focus on improving MLX within its format constraints
- cols64 setting already provides ~2% improvement

## Environment Variables for Performance

For best performance on gfx1151, set:
```bash
export LEMON_MLX_GDN_ENABLE_HIP=1
export MLX_ROCM_QMV_COLS_PER_BLOCK=64
```

## Verification

Run correctness check:
```bash
./verify_cols64_correctness.sh
```

Run performance comparison:
```bash
./final_qmv_comparison.sh
```

## Profile Both Engines

To see the detailed bottleneck analysis:

Lemon MLX:
```bash
LEMON_MLX_QUANT_PROFILE_TIMING=1 MAX_TOKENS=5 ./build-gfx1151/bench \
  mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "..." \
  --raw --max-tokens 5 --warmup-decode-steps 1 --temperature 0 --top-p 1
```

Hipfire:
```bash
cargo run --release --example profile_qwen35_mq4 --features deltanet -- \
  ~/.hipfire/models/qwen3.5-0.8b.mq4 --profile-steps 5
```
