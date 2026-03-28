# Session Summary — Qwen3-Next ROCm Optimization

## What we accomplished

### 1. Fixed Qwen3-Next model (was producing garbage output)

**Root causes found and fixed:**
- **Q/K scaling bug**: q got `1/sqrt(Dk)` instead of `1/Dk`, k got nothing instead of `1/sqrt(Dk)`
- **Norm weight +1**: safetensors stores direct multipliers (~1.0), our `rms_norm` uses weight directly — adding +1 doubled all norm outputs
- **SSM mask**: returned `ones({B,S})` instead of `nullopt` during decode

**Clean rewrite** from Swift reference (Qwen3Next.swift + GatedDelta.swift):
- New `gated_delta.{h,cpp}` — separate gated delta ops module
- Faithful 1:1 port with no compiled kernels initially, then re-optimized

### 2. Performance Optimization Chain

| Step | Prompt | Gen | Key Change |
|------|--------|-----|------------|
| Start (garbage) | N/A | 0.4 tok/s | Broken model |
| Correct output | N/A | ~6.5 tok/s | Q/K fix + norm fix |
| Compiled kernels | — | 13.5 tok/s | Fused ops, T=1 fast paths |
| 4-bit gather QMV | — | 12.0 tok/s | Enable warp-shared for 4-bit MoE (18.6x kernel speedup) |
| Allocator: stream sync | — | 18.9 tok/s | hipStreamSync instead of hipDeviceSync for APU |
| Expert-grouped prefill | 6.1 tok/s | 18.9 tok/s | Group by expert in gather QMV |
| **Warmup pass** | **80 tok/s** | 18.9 tok/s | Prime allocator cache (eliminates 2.2s cold start) |
| **WMMA kernel** | **117 tok/s** | 18.9 tok/s | rocWMMA 16x16x16 bf16→f32 tiles |

### 3. Upstream Merge

Merged NripeshN/mlx `rocm-support` branch into `lemon-mlx-core-amd`:
- Flash attention kernel, allocator redesign, bfloat16 math overloads
- QMV vectorization, depthwise conv1d, event sync improvements
- Resolved 8 merge conflicts (SDPA, QMM, matmul, rocBLAS, etc.)
- Fixed JIT "File name too long" crash (comgr temp file naming)

## Repositories & Commits

### lemon-mlx-engine (main branch)
```
43cf63a Add warmup pass to prime GPU allocator cache (23x prompt speedup)
99f082b Add fused HIP kernel for GDN recurrence + mega-fused decode path
533e59a Optimize Qwen3-Next: compiled kernels, T=1 fast paths, fused ops
fd61773 Fix Qwen3-Next: correct Q/K scaling, remove wrong norm +1, fix SSM mask
70e5b41 Rewrite Qwen3-Next as faithful 1:1 port of Swift reference
```

### lemon-mlx-core-amd (rocm-support branch)
```
e35d6aae WMMA prefill kernel: support non-aligned M, sort unsorted indices
a66e273b Add WMMA-accelerated prefill kernel for GatherQMM on RDNA 3/3.5/4
c9167d22 Allocator: prefer hipExtMallocWithFlags for APU, fallback to hipMallocManaged
0ec6b45f Add expert-grouped prefill kernel for GatherQMM (3.4x prompt speedup)
780b4feb Prefer shared-memory QMV over noshared variant for decode
b1300b92 Optimize ROCm allocator for integrated GPUs (APU)
5ffb8636 Enable 4-bit fast gather QMV dispatch for MoE decode
d30fe29e Merge upstream NripeshN/mlx rocm-support with ROCm optimizations
```

## vs llama.cpp Targets (Strix Halo gfx1151)

| Metric | llama.cpp | Our MLX | Gap |
|--------|-----------|---------|-----|
| Prompt (pp512) | 344 tok/s | 117 tok/s | 2.9x |
| Generation (tg128) | 24.2 tok/s | 18.9 tok/s | 1.3x |

## Remaining Work

1. **GPU-only token batching for MoE** — current SwitchGLU expands all tokens to individual M=1 gather_qmm calls (600 dispatches). Batching per-expert on GPU would reduce to ~60-100 larger GEMMs.
2. **hipBLASLt integration** — replace rocBLAS with hipBLASLt for dense attention GEMMs (~70ms overhead).
3. **Larger WMMA tile sizes** — 32x32, 64x64 for better compute density.
4. **Allocator pool pre-warming** — reduce cold-start allocation overhead without the warmup pass.
5. **Qwen3-8B decode regression** — 17→11 tok/s after upstream merge, needs investigation.

## PR for NripeshN/mlx

Create at: https://github.com/NripeshN/mlx/compare/rocm-support...lemonade-sdk:lemon-mlx-core-amd:rocm-support

**Title:** ROCm: WMMA prefill, 4-bit MoE dispatch, APU allocator optimizations

**Base:** rocm-support → **Head:** lemonade-sdk:rocm-support
