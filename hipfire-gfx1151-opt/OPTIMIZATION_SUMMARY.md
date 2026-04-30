# Performance Optimization Summary

## Current Status (2026-04-30)

**hipfire group256: 127.1 tok/s**
**llama.cpp Q4_0: 195.2 tok/s**
**Gap: 35%**

## Completed Work

1. ✅ group_size=256 support (24% improvement over group64)
2. ✅ TILE_N=8 for Rdna35 (2% improvement)
3. ✅ Fair comparison with Q4_0 (apples-to-apples)
4. ❌ Shared memory padding (no improvement - regression)
5. ❌ AMD intrinsics experiment (not yet integrated)

## Key Findings

### llama.cpp Advantages

1. **AMD Hardware Intrinsics**
   - Uses `__builtin_amdgcn_perm` for fast table lookups
   - Significantly faster dequantization than our generic code

2. **MMQ-Style Architecture**
   - 8 warps per block (we also do this)
   - Sophisticated tiling: `MMQ_TILE_NE_K = 32` tuned to avoid bank conflicts
   - Different tile sizes per quantization type

3. **Vector Dot Product Optimization**
   - `vecdotq.cuh` with heavy unrolling
   - Partial sum pre-computation
   - Optimized memory access patterns

### hipfire Current State

- Already uses multiple warps (256 threads = 8 warps)
- QMV tiled kernel is well-optimized for L2 cache reuse
- Dequantization is generic (no AMD-specific optimizations)

## Why the 35% Gap Exists

The gap is **NOT** from:
- ❌ Group size (256 vs 32 - we're actually more efficient here)
- ❌ Number of warps (we use 8, same as llama.cpp)
- ❌ Basic algorithm (both are matmul-quantized)

The gap **IS** from:
- ✅ AMD intrinsics for dequantization (major factor)
- ✅ Tile size tuning per quantization type
- ✅ Data layout optimization (MMQ blocking)
- ✅ Vector operations (dp4a/mma utilization)

## What It Would Take to Match llama.cpp

### Required Changes (35% improvement needed)

1. **AMD Intrinsics Dequantization** (10-15%)
   - Replace generic unpacking with `__builtin_amdgcn_perm`
   - Files: `qdequant.hpp`, `qmv_tiled_kernel.hip`
   - Estimated effort: 2-3 days

2. **MMQ-Style Kernel** (10-15%)
   - Implement MMQ blocking and tiling
   - Add partial sum optimization
   - Files: New `qmv_mmq.hip`
   - Estimated effort: 5-7 days

3. **Per-Type Tile Tuning** (5-10%)
   - Different TILE_N for group256 vs group64
   - Optimize BSK based on quantization
   - Estimated effort: 1-2 days

**Total: 8-12 days of focused kernel development work**

## Alternative Approaches

1. **Port llama.cpp MMQ directly**
   - Pros: Proven performance
   - Cons: Different architecture, may not fit MLX's design

2. **Wait for MLX community optimizations**
   - MLX is actively developed
   - ROCm backend is maturing
   - Cons: Uncertain timeline

3. **Accept current performance**
   - 127 tok/s is still quite good
   - Focus on other optimizations (speculative decoding, etc.)
   - Pros: Faster iteration on other features

## Recommendation

Given the scope of work (8-12 days), recommend:
1. Document current status
2. Consider if 127 tok/s is acceptable for current use case
3. If matching llama.cpp is critical, allocate dedicated kernel development time
4. Otherwise, focus on other optimizations that may be easier

## Files Created

- `LLAMACPP_MATCH_PLAN.md` - Detailed implementation plan
- `qmv_amd_intrinsics.hip` - AMD intrinsic utilities (not yet integrated)
- `qmv_optimized.hip` - Optimized kernel prototype (not yet integrated)
- `FUSION_ANALYSIS.md` - Operator fusion analysis
- `PERF_COMPARISON_Q4_0.md` - Fair benchmark comparison
