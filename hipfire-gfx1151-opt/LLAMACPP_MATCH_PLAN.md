# Plan: Match llama.cpp Performance

## Current Gap Analysis

**llama.cpp Q4_0: 195.2 tok/s**
**hipfire group256: 127.1 tok/s**
**Gap: 53% slower (need 35% improvement)**

## Key Differences Found

### 1. Hardware Intrinsics (llama.cpp advantage)
```cpp
// llama.cpp uses AMD-specific permute for dequantization
uint32_t v_even = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
```
We use generic HIP code - missing AMD-specific optimizations.

### 2. Tile Size Optimization
- llama.cpp: Different tile sizes per quantization type
- hipfire: Fixed TILE_N=8 for all operations

### 3. Data Layout
- llama.cpp: Padding for bank conflicts (MMQ_TILE_NE_K = 32)
- hipfire: Basic BSK=512

### 4. Vector Dot Product
- llama.cpp: Highly optimized vecdotq with unrolling
- hipfire: Basic loop

## Implementation Plan

### Phase 1: AMD Intrinsics Dequantization (Expected: 10-15%)

Create new HIP kernel using AMD intrinsics:
```cpp
__device__ __forceinline__ float4 dequant_q4_amdgcn(
    const uint32_t packed,
    const float scale) {
    // Use __builtin_amdgcn_perm for fast lookup
    // Similar to llama.cpp vecdotq.cuh
}
```

**Files to create:**
- `mlx/backend/rocm/quantized/qmv_amd_intrinsics.hip`

### Phase 2: Optimized Tile Sizes (Expected: 5-10%)

Add per-quantization-type tuning:
- Q4_0: TILE_N based on MMQ_DP4A_TXS_Q4_0 pattern
- group256: Different TILE_N than group64

### Phase 3: MMQ-Style Tiled MatMul (Expected: 10-15%)

Implement MMQ-style tiled matrix multiply:
- 8 warps per block (vs our current 1 warp per column)
- Shared memory blocking
- Optimized for K-dimension tiling

### Phase 4: Partial Sum Optimization (Expected: 5%)

Pre-compute partial sums in shared memory to reduce redundant calculations.

## Success Criteria

- Phase 1: 140-146 tok/s (+10-15%)
- Phase 2: 147-160 tok/s (+5-10% more)
- Phase 3: 162-184 tok/s (+10-15% more)
- Phase 4: 170-195 tok/s (+5% more)

**Target: 195+ tok/s to match llama.cpp**

## Risk Assessment

- **High risk**: Phase 3 (MMQ rewrite) - Significant kernel changes
- **Medium risk**: Phase 1 (AMD intrinsics) - Needs careful validation
- **Low risk**: Phase 2 (Tile tuning) - Easy to A/B test

## Next Step

Start with **Phase 1** - Add AMD intrinsic dequantization as an alternative kernel.
Create a simple benchmark to validate before integrating.
