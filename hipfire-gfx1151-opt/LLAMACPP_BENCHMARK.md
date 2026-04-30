# llama.cpp vs hipfire Benchmark - Qwen3.5-0.8B

## Hardware
- GPU: AMD Radeon RX 7900 XTX (gfx1151, RDNA3)
- Driver: ROCm 7.1.0

## Test Setup
- Model: Qwen3.5-0.8B
- Prompt: 63 tokens (Python Fibonacci function explanation question)
- Generation: 128 tokens
- Temperature: 0 (deterministic)
- 3 runs per configuration

## Results

### Decode Performance (tokens/second)

| Engine | Quantization | Decode tok/s | Speedup vs baseline |
|--------|--------------|--------------|---------------------|
| **llama.cpp** | Q4_K_M (GGUF) | **180.3** | **1.00x** (baseline) |
| **hipfire** | group_size=256 (affine int4) | **132.7** | 0.74x |
| **hipfire** | group_size=64 (affine int4) | **106.6** | 0.59x |

### Key Findings

1. **llama.cpp is 35% faster than hipfire group256**
   - llama.cpp Q4_K_M: 180.3 tok/s
   - hipfire group256: 132.7 tok/s
   - This is surprising given the similar quantization schemes

2. **hipfire group256 is 24% faster than group64**
   - Confirms the kernel optimization works correctly
   - 4x fewer scale/bias loads provides measurable benefit

3. **llama.cpp's performance advantage likely comes from:**
   - More mature codebase (years of optimization)
   - Q4_K_M's sophisticated mixed quantization (super-blocks, varying group sizes)
   - Heavier kernel optimization (CUDA/HIP code highly tuned)

### Quantization Details

| Aspect | hipfire group256 | llama.cpp Q4_K_M |
|--------|------------------|------------------|
| Type | Affine int4 | K-quants |
| Group size | Fixed 256 | Variable (32/64/256) |
| Scales per row | 12 (3072/256) | ~10-15 average |
| File size | 589 MB | 520 MB |

### Possible hipfire Optimizations

1. **Kernel-level optimizations**
   - Better memory access patterns
   - Improved wave utilization
   - Reduce register pressure

2. **Quantization improvements**
   - Implement K-quants (Q4_K_M equivalent)
   - Mixed group sizes per layer
   - Better scale/bias initialization

3. **Operator fusion**
   - Fuse RMSNorm + RoPE
   - Fuse element-wise operations
   - Better KV cache handling

## Conclusion

The group_size=256 work successfully improved hipfire performance by 24% over group_size=64.
However, llama.cpp still maintains a 35% lead due to more advanced quantization and
mature optimization.

**Next steps to close the gap:**
1. Profile hipfire kernels to identify bottlenecks
2. Implement Q4_K_M-style mixed quantization
3. Study llama.cpp's HIP kernels for optimization techniques
4. Consider fusing more operators

## Data

```
=== hipfire group256 ===
Run 1: 132.36 tok/s
Run 2: 132.35 tok/s
Run 3: 133.53 tok/s
Average: 132.7 tok/s

=== hipfire group64 ===
Run 1: 105.85 tok/s
Run 2: 107.54 tok/s
Run 3: 106.28 tok/s
Average: 106.6 tok/s

=== llama.cpp Q4_K_M ===
Run 1: 180.44 tok/s
Run 2: 180.42 tok/s
Run 3: 180.12 tok/s
Average: 180.3 tok/s
```

## Files

- llama.cpp build: `/home/reckon/projects/ai/mlx/third_party/llama.cpp/build-hip/`
- llama.cpp binary: `bin/llama-cli`
- hipfire build: `/home/reckon/projects/lemon-mlx-engine/build-gfx1151/`
- hipfire binary: `bench`
- Benchmark prompt: `/tmp/bench_prompt.txt`

## Date

2026-04-30
