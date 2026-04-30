# Group Size 256 Support - COMPLETE ✓

## Status: FULLY FUNCTIONAL (2026-04-30)

All work completed successfully. The group_size=256 optimization is now working
and providing measurable performance improvements.

## Completed Work

### 1. MLX C++ Kernel Support ✓
- File: `build-gfx1151/_deps/mlx-src/mlx/backend/rocm/quantized/qmm.hip`
- Changes:
  - Added `gs256` type alias for group_size=256
  - Added `case 256:` to DISPATCH_GROUP_SIZE macro
  - Added group_size=256 to all tiled QMV launch paths
- Status: **PATCHED AND WORKING**

### 2. MLX Python Support ✓
- File: `patches/mlx-vulkan-group256.patch`
- Change: Modified `mlx/ops.cpp` to accept group_size=256
- Enables: `mx.quantize(weights, group_size=256, bits=4)`
- Status: **PATCHED AND WORKING**

### 3. Working Model ✓
- Script: `scripts/dequantize_requantize_final.py`
- Location: `/home/reckon/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-group256-final/`
- Method: Dequantize MLX-community gs=64 model, re-quantize with gs=256
- Key insight: Use MLX's native `mx.load()` and `mx.save_safetensors()` to handle BF16 correctly
- Status: **WORKING MODEL WITH CORRECT OUTPUT**

### 4. Validation Tests ✓
- CPU quantize/dequantize roundtrip: **PASSED**
  - group64 mean error: 0.007714
  - group256 mean error: 0.009285 (1.2x - expected)
- Shape validation: **PASSED**
- Model output quality: **CORRECT** (no repetitive tokens)

## Performance Results

### hipfire Internal Comparison

| Configuration | Decode tok/s | Speedup |
|---------------|--------------|---------|
| group_size=256 | **132.7** | **1.24x** |
| group_size=64 | 106.6 | 1.00x (baseline) |

**Result: 24% performance improvement from group_size=256**

### llama.cpp Comparison (for context)

| Engine | Quantization | Decode tok/s |
|--------|--------------|--------------|
| llama.cpp | Q4_K_M (GGUF) | **180.3** |
| hipfire | group_size=256 | 132.7 |
| hipfire | group_size=64 | 106.6 |

**llama.cpp is 35% faster** due to:
- More mature codebase with years of optimization
- Q4_K_M's sophisticated mixed quantization
- Highly tuned HIP kernels

## How The BF16 Issue Was Solved

The problem: MLX-community models store scales/biases in BF16 format, which
was being incorrectly interpreted as float16.

The solution: Use MLX's native functions instead of manual conversion:
```python
# Load using MLX (handles BF16 automatically)
ref_model = mx.load(str(ref_path / "model.safetensors"))

# Dequantize with group_size=64
w_fp32 = mx.dequantize(packed, scales, biases, group_size=64, bits=4)

# Re-quantize with group_size=256
q256 = mx.quantize(w_fp32, group_size=256, bits=4)

# Save using MLX (handles BF16 automatically)
mx.save_safetensors(str(output_file), new_model)
```

## Files Added/Modified

- `patches/mlx-vulkan-group256.patch` - MLX Python patch
- `scripts/dequantize_requantize_final.py` - **Working conversion script**
- `scripts/test_group256_kernel.py` - Validation tests
- `build-gfx1151/_deps/mlx-src/mlx/backend/rocm/quantized/qmm.hip` - Kernel patch (manual)
- `LLAMACPP_BENCHMARK.md` - Benchmark comparison with llama.cpp
- `GROUP256_SUMMARY.md` - This file

## Next Steps (Future Optimizations)

To close the gap with llama.cpp:

1. **Kernel optimization**
   - Profile and optimize memory access patterns
   - Improve wave utilization
   - Reduce register pressure

2. **Advanced quantization**
   - Implement Q4_K_M-style mixed quantization
   - Variable group sizes per layer
   - Better scale/bias initialization

3. **Operator fusion**
   - Fuse RMSNorm + RoPE
   - Fuse element-wise operations
   - Better KV cache handling
