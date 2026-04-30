# Performance Comparison - hipfire vs llama.cpp (Apples-to-Apples)

## Test Setup
- Model: Qwen3.5-0.8B
- GPU: Strix Halo iGPU (gfx1151, Rdna35)
- Prompt: "Hello world, this is a test" (7 tokens)
- Generation: 64 tokens
- Temperature: 0 (deterministic)

## Results (Q4_0 vs group256 - Apples to Apples)

| Engine | Quantization | Decode tok/s | Relative |
|--------|--------------|--------------|----------|
| **llama.cpp** | Q4_0 | **195.2** | 1.00x (baseline) |
| **hipfire** | group256, TILE_N=8 | **127.1** | 0.65x |

**Gap: 35% slower**

This is a fair comparison:
- **Q4_0**: Basic 4-bit quantization with symmetric scales (no K-quants)
- **group256**: Affine int4 with symmetric+asymmetric scales, group_size=256

Both use similar quantization approaches. The 35% gap is due to:
1. Kernel optimization differences
2. Operator fusion
3. Memory access patterns

## Previous (Unfair) Comparison - Q4_K_M

| Engine | Quantization | Decode tok/s | Relative |
|--------|--------------|--------------|----------|
| **llama.cpp** | Q4_K_M | **180.3** | 1.00x |
| **hipfire** | group256, TILE_N=8 | **135.9** | 0.75x |

Q4_K_M is more sophisticated (variable group sizes, super-blocks) so this wasn't a fair comparison.

## Optimization Roadmap

### High Priority (10-20% each)

1. **Operator Fusion**
   - Fuse RMSNorm + RoPE in attention
   - Fuse SiLU/GELU activations with matmul
   - Expected: 10-20% improvement
   - Target: 140-152 tok/s

2. **Shared Memory Optimization**
   - Add padding to avoid bank conflicts (32 banks on RDNA)
   - Improve data layout for coalesced access
   - Expected: 5-10% improvement
   - Target: 133-139 tok/s

3. **Kernel-Level Optimization**
   - Study llama.cpp MMQ implementation
   - Vectorized dot product optimization
   - Partial sum computation
   - Expected: 10-15% improvement
   - Target: 140-146 tok/s

### Combined Potential

If we implement all three:
- Best case: 127 × 1.15 × 1.10 × 1.10 = **177 tok/s** (~91% of llama.cpp)
- Realistic: 127 × 1.10 × 1.05 × 1.08 = **158 tok/s** (~81% of llama.cpp)

## Next Steps

1. Profile actual bottlenecks with rocprof
2. Implement RMSNorm+RoPE fusion (highest ROI, 10-20%)
3. Add shared memory padding (5-10%)
4. Study llama.cpp MMQ for kernel optimizations

## Files

- Benchmark script: `scripts/optimize_tile_n.py`
- Rdna35 patch: `patches/apply_rdna35_tile_n.py`
- Q4_0 model: `/tmp/qwen_q4_0/Qwen_Qwen3.5-0.8B-Q4_0.gguf`
