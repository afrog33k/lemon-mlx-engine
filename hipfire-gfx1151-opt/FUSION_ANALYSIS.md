# Optimization Status - Next Steps

## Current Performance (Qwen3.5-0.8B, Strix Halo gfx1151)

| Engine | Quantization | Decode tok/s | Relative |
|--------|--------------|--------------|----------|
| llama.cpp Q4_0 | 195.2 | 1.00x (baseline) |
| hipfire group256 | 127.1 | 0.65x |

**Gap: 35%**

## Operator Fusion Analysis

### Current Attention Flow (qwen3.cpp)

```cpp
// 1. QKV projection (single linear)
auto qkv = linear_fwd(x, wqkv_weight_);

// 2. Split Q, K, V
auto queries = mx::slice(qkv, ...);
auto keys = mx::slice(qkv, ...);
auto values = mx::slice(qkv, ...);

// 3. RMSNorm (separate for Q and K)
queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);

// 4. Transpose
queries = mx::transpose(queries, ...);
keys = mx::transpose(keys, ...);

// 5. RoPE (separate for Q and K)
queries = mx::fast::rope(queries, ...);
keys = mx::fast::rope(keys, ...);

// 6. SDPA
auto output = sdpa(queries, keys, values, ...);
```

### Fusion Opportunity

**RMSNorm + Transpose + RoPE** could be fused into a single kernel:
- Current: 3 kernel launches × 2 (Q and K) = 6 launches
- Fused: 1 kernel launch × 2 (Q and K) = 2 launches

**Expected benefit: 10-20%** from:
- Reduced kernel launch overhead
- Eliminated intermediate memory writes/reads
- Better cache utilization

### Implementation Challenge

This requires adding a new MLX fast primitive and ROCm HIP kernel. The file structure:
- `mlx/fast.h` - Declare new function
- `mlx/fast.cpp` - CPU fallback
- `mlx/backend/rocm/rms_norm_rope.hip` - New HIP kernel
- `mlx/backend/rocm/fast.cpp` - ROCm dispatch

This is a significant change to MLX itself.

## Alternative: Simpler Optimizations

### 1. QMV Shared Memory Padding (5-10%)
Add padding to `x_shared` array to avoid bank conflicts:
```cpp
// Current: __shared__ float x_shared[BSK];
// Proposed: __shared__ float x_shared[BSK + PADDING];
```

### 2. Study llama.cpp MMQ (10-15%)
Reverse-engineer what llama.cpp does differently:
- Better data layouts
- Partial sum computation
- More aggressive unrolling

### 3. Activation Fusion (5-10%)
Fuse SiLU/GELU with linear projections in MLP

## Recommended Approach

1. **Start with QMV shared memory padding** (quick win, 5-10%)
2. **Profile to confirm bottleneck location**
3. **Then tackle RMSNorm+RoPE fusion** (requires MLX changes)

## Files Referenced

- `src/llm/models/qwen3.cpp` - Attention implementation
- `mlx/backend/rocm/rms_norm.hip` - RMSNorm kernel
- `mlx/backend/rocm/quantized/qmv_tiled_kernel.hip` - QMV kernel
- `mlx/fast.h` - Fast operations interface
