#!/usr/bin/env python3
"""
Standalone numerical trace for Qwen3-Next layer 0 GDN.
Loads quantized 4-bit affine weights from safetensors, dequantizes manually,
and computes the same values that the C++ debug output prints.

Uses only numpy + struct (no mlx/torch).
"""

import struct
import json
import numpy as np
import os

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community-Qwen3-Coder-Next-4bit/snapshots/main"
)
SHARD = os.path.join(MODEL_DIR, "model-00001-of-00009.safetensors")
TOKEN_ID = 9707  # "Hello"

# --- Safetensors reader ---

def read_safetensors(path):
    """Read safetensors file, return dict of name -> numpy array."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        data_start = 8 + header_size

        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str = info["dtype"]
            shape = info["shape"]
            offsets = info["data_offsets"]
            start, end = offsets

            dtype_map = {
                "BF16": np.uint16,  # read as uint16, convert later
                "F16": np.float16,
                "F32": np.float32,
                "U32": np.uint32,
                "I32": np.int32,
            }
            np_dtype = dtype_map[dtype_str]
            f.seek(data_start + start)
            raw = f.read(end - start)
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            if dtype_str == "BF16":
                # Convert bf16 to float32: shift left by 16 bits
                arr = bf16_to_f32(arr)
            tensors[name] = arr
        return tensors


def bf16_to_f32(arr_u16):
    """Convert bf16 (stored as uint16) to float32."""
    # bf16 is the upper 16 bits of float32
    as_u32 = arr_u16.astype(np.uint32) << 16
    return as_u32.view(np.float32)


# --- 4-bit affine dequantization ---

def dequantize_4bit_affine(packed_u32, scales, biases, group_size=64):
    """
    Dequantize 4-bit affine quantized weights.

    packed_u32: [rows, cols_packed] uint32 - each uint32 holds 8 x 4-bit values
    scales: [rows, num_groups] float32
    biases: [rows, num_groups] float32
    group_size: number of elements per group (default 64)

    Returns: [rows, cols_packed * 8] float32
    """
    rows, cols_packed = packed_u32.shape
    total_cols = cols_packed * 8  # 8 nibbles per uint32

    # Unpack 4-bit values from uint32
    # Each uint32 contains 8 x 4-bit values, LSB first
    unpacked = np.zeros((rows, total_cols), dtype=np.float32)
    for i in range(8):
        nibble = (packed_u32 >> (4 * i)) & 0xF
        unpacked[:, i::8] = nibble.astype(np.float32)

    # Apply scales and biases per group
    num_groups = total_cols // group_size
    result = np.zeros_like(unpacked)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        s = scales[:, g:g+1]  # [rows, 1]
        b = biases[:, g:g+1]  # [rows, 1]
        result[:, start:end] = unpacked[:, start:end] * s + b

    return result


# --- RMS Norm ---

def rms_norm(x, weight, eps=1e-6):
    """RMS normalization: x * weight / sqrt(mean(x^2) + eps)."""
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * weight / np.sqrt(ms + eps)


# --- Quantized matmul (simulated) ---

def quantized_matmul_4bit(x, packed_u32, scales, biases, group_size=64):
    """
    Simulate mx::quantized_matmul for 4-bit affine.

    x: [1, hidden_size] float32  (input)
    packed_u32: [out_features, cols_packed] uint32
    scales: [out_features, num_groups] float32
    biases: [out_features, num_groups] float32

    Returns: [1, out_features] float32  (x @ dequantized_weight.T)
    """
    # Dequantize the weight matrix
    w_dequant = dequantize_4bit_affine(packed_u32, scales, biases, group_size)
    # matmul: x @ W.T
    return x @ w_dequant.T


def dbg(label, arr):
    """Print debug info matching the C++ dbg() format."""
    flat = arr.flatten().astype(np.float32)
    mean = np.mean(flat)
    std = np.std(flat)
    n = min(4, len(flat))
    vals = ",".join(f"{flat[i]:.6g}" for i in range(n))
    print(f"  PY  {label}: mean={mean:.6g} std={std:.6g} [{vals}]")


def main():
    print("Loading safetensors...")
    tensors = read_safetensors(SHARD)

    print(f"\nAvailable keys (embed + layer0 GDN):")
    for k in sorted(tensors.keys()):
        if "embed_tokens" in k or ("layers.0." in k and ("in_proj_qkvz" in k or "input_layernorm" in k)):
            print(f"  {k}: shape={tensors[k].shape} dtype={tensors[k].dtype}")

    # Step 1: Dequantize embedding and look up token
    print(f"\n=== Step 1: Embedding lookup for token {TOKEN_ID} ===")
    embed_packed = tensors["model.embed_tokens.weight"]
    embed_scales = tensors["model.embed_tokens.scales"]
    embed_biases = tensors["model.embed_tokens.biases"]

    # Dequantize the full embedding table (or just the row we need)
    # For efficiency, just dequantize row TOKEN_ID
    row_packed = embed_packed[TOKEN_ID:TOKEN_ID+1]
    row_scales = embed_scales[TOKEN_ID:TOKEN_ID+1]
    row_biases = embed_biases[TOKEN_ID:TOKEN_ID+1]
    embedding = dequantize_4bit_affine(row_packed, row_scales, row_biases, group_size=64)
    # embedding shape: [1, 2048]

    dbg("embedding (raw)", embedding)
    print(f"  embedding first 8: {embedding[0, :8]}")

    # Step 2: Apply input_layernorm (RMS norm)
    print(f"\n=== Step 2: Input LayerNorm ===")
    ln_weight = tensors["model.layers.0.input_layernorm.weight"]
    normed = rms_norm(embedding, ln_weight, eps=1e-6)
    dbg("input (after layernorm)", normed)
    print(f"  normed first 8: {normed[0, :8]}")

    # Step 3: Quantized matmul with in_proj_qkvz
    print(f"\n=== Step 3: in_proj_qkvz projection ===")
    qkvz_packed = tensors["model.layers.0.linear_attn.in_proj_qkvz.weight"]
    qkvz_scales = tensors["model.layers.0.linear_attn.in_proj_qkvz.scales"]
    qkvz_biases = tensors["model.layers.0.linear_attn.in_proj_qkvz.biases"]

    print(f"  qkvz weight shape: {qkvz_packed.shape}")
    print(f"  qkvz scales shape: {qkvz_scales.shape}")
    print(f"  qkvz biases shape: {qkvz_biases.shape}")

    mixed_qkvz = quantized_matmul_4bit(normed, qkvz_packed, qkvz_scales, qkvz_biases, group_size=64)
    dbg("qkvz_proj", mixed_qkvz)
    print(f"  qkvz first 8: {mixed_qkvz[0, :8]}")

    # Step 4: Compare with C++ output
    print("\n" + "="*70)
    print("=== COMPARISON WITH C++ OUTPUT ===")
    print("="*70)
    print()
    print("C++ GDN input: mean=0.0229131 std=2.05292 [0.570031,1.77881,0.567847,-1.67379]")
    print(f"PY  input:     mean={np.mean(normed):.6g} std={np.std(normed):.6g} [{normed[0,0]:.6g},{normed[0,1]:.6g},{normed[0,2]:.6g},{normed[0,3]:.6g}]")
    print()
    print("C++ GDN qkvz_proj: mean=0.242528 std=4.45337 [-5.12161,3.32916,1.46951,1.38724]")
    print(f"PY  qkvz_proj:     mean={np.mean(mixed_qkvz):.6g} std={np.std(mixed_qkvz):.6g} [{mixed_qkvz[0,0]:.6g},{mixed_qkvz[0,1]:.6g},{mixed_qkvz[0,2]:.6g},{mixed_qkvz[0,3]:.6g}]")

    # Check if they match
    cpp_input_vals = [0.570031, 1.77881, 0.567847, -1.67379]
    py_input_vals = [normed[0, i] for i in range(4)]
    input_close = all(abs(c - p) < 0.01 for c, p in zip(cpp_input_vals, py_input_vals))

    cpp_qkvz_vals = [-5.12161, 3.32916, 1.46951, 1.38724]
    py_qkvz_vals = [mixed_qkvz[0, i] for i in range(4)]
    qkvz_close = all(abs(c - p) < 0.1 for c, p in zip(cpp_qkvz_vals, py_qkvz_vals))

    print()
    if input_close:
        print("MATCH: input values match between C++ and Python")
    else:
        print("MISMATCH: input values DIFFER between C++ and Python!")
        print(f"  C++ first 4: {cpp_input_vals}")
        print(f"  PY  first 4: {py_input_vals}")
        print(f"  Diffs:       {[abs(c-p) for c,p in zip(cpp_input_vals, py_input_vals)]}")

    if qkvz_close:
        print("MATCH: qkvz_proj values match between C++ and Python")
    else:
        print("MISMATCH: qkvz_proj values DIFFER between C++ and Python!")
        print(f"  C++ first 4: {cpp_qkvz_vals}")
        print(f"  PY  first 4: {py_qkvz_vals}")
        print(f"  Diffs:       {[abs(c-p) for c,p in zip(cpp_qkvz_vals, py_qkvz_vals)]}")


if __name__ == "__main__":
    main()
