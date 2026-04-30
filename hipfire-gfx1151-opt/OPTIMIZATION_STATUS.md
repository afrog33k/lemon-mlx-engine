# Performance Optimization Status - hipfire vs llama.cpp

## Current Status (2026-04-30)

### Performance Numbers (Qwen3.5-0.8B, Strix Halo iGPU gfx1151)

| Engine | Quantization | Decode tok/s | Relative |
|--------|--------------|--------------|----------|
| **llama.cpp** | Q4_K_M | **180.3** | 1.00x (baseline) |
| **hipfire** | group256, TILE_N=8 | **135.9** | 0.75x |
| **hipfire** | group256, TILE_N=16 | 124.7 | 0.69x |
| **hipfire** | group64 | 106.6 | 0.59x |

### Progress

- **Starting point**: 106.6 tok/s (group64)
- **After group256**: 132.7 tok/s (+24%)
- **After TILE_N=8**: 135.9 tok/s (+2.4%, +27% total)
- **Remaining gap**: 25% slower than llama.cpp

## Completed Optimizations

### 1. group_size=256 Support ✓
- 4x fewer scale/bias loads
- 24% performance improvement
- Fully functional and validated

### 2. TILE_N=8 for Rdna35 ✓
- Targeted optimization for gfx1151 (Strix Halo)
- ~14% improvement over TILE_N=16
- Applied via `patches/apply_rdna35_tile_n.py`

## Optimization Roadmap

### High Priority (10-20% each)

1. **Operator Fusion**
   - Fuse RMSNorm + RoPE in attention
   - Fuse SiLU/GELU activations with matmul
   - Expected: 10-20% improvement
   - Status: Not started

2. **Shared Memory Optimization**
   - Add padding to avoid bank conflicts
   - Improve data layout for coalesced access
   - Expected: 5-10% improvement
   - Status: Not started

3. **Kernel-Level Optimization**
   - Study llama.cpp MMQ implementation
   - Vectorized dot product optimization
   - Partial sum computation
   - Expected: 10-15% improvement
   - Status: Analysis phase

### Medium Priority (3-5% each)

4. **Wave Utilization**
   - Reduce idle threads
   - Better thread organization
   - Expected: 3-5% improvement

5. **Memory Access Patterns**
   - Improve L2 cache hit rate
   - Reduce global memory transactions
   - Expected: 3-5% improvement

## Next Steps

1. **Profile with rocprof** - Find actual bottlenecks
2. **Implement RMSNorm+RoPE fusion** - Highest ROI
3. **Study llama.cpp MMQ** - Copy proven optimizations
4. **Shared memory padding** - Relatively easy win

## Tools Created

- `scripts/optimize_tile_n.py` - TILE_N benchmarking
- `patches/apply_rdna35_tile_n.py` - Rdna35 optimization script
- `patches/mlx-rocm-rdna35-tile-n8.patch` - Reference patch

## References

- llama.cpp MMQ: `ggml/src/ggml-cuda/mmq.cuh`
- hipfire QMV: `mlx/backend/rocm/quantized/qmv_tiled_kernel.hip`
- Arch tuning: `mlx/backend/rocm/device/config.h`
