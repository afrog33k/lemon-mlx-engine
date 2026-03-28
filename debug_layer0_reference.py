#!/usr/bin/env python3
"""
Reference script: loads Qwen3-Next model weights and runs layer 0 (GDN)
to produce intermediate values for comparison with C++ debug output.

Usage:
    python3 debug_layer0_reference.py
"""

import json
import numpy as np
from pathlib import Path

MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--mlx-community-Qwen3-Coder-Next-4bit/snapshots/main"

def load_config():
    with open(MODEL_DIR / "config.json") as f:
        return json.load(f)

def load_weights_mlx():
    """Load weights using mlx's safetensors loader."""
    import mlx.core as mx
    import mlx.nn as nn

    # Find all safetensors files
    st_files = sorted(MODEL_DIR.glob("*.safetensors"))
    weights = {}
    for f in st_files:
        w = mx.load(str(f))
        weights.update(w)
    return weights

def l2norm(x, axis=-1, eps=1e-6):
    """L2 normalization matching HuggingFace reference (fla library)."""
    inv_norm = 1.0 / np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm

def rms_norm(x, weight, eps=1e-6):
    """RMS normalization."""
    variance = np.mean(x * x, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def softplus(x):
    return np.log(1.0 + np.exp(x))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dequantize_weight(w_quant, scales, biases, group_size=64, bits=4):
    """Dequantize a 4-bit quantized weight matrix (MLX format)."""
    # w_quant: [out_features, in_features // pack_factor] uint32
    # scales: [out_features, num_groups]
    # biases: [out_features, num_groups]
    out_features = scales.shape[0]
    num_groups = scales.shape[1]
    in_features = num_groups * group_size

    pack_factor = 32 // bits  # 8 values per uint32 for 4-bit

    result = np.zeros((out_features, in_features), dtype=np.float32)

    for row in range(out_features):
        for g in range(num_groups):
            scale = float(scales[row, g])
            bias = float(biases[row, g])
            col_start = g * group_size

            for j in range(group_size):
                col = col_start + j
                pack_idx = col // pack_factor
                bit_offset = (col % pack_factor) * bits

                packed_val = int(w_quant[row, pack_idx])
                val = (packed_val >> bit_offset) & ((1 << bits) - 1)

                result[row, col] = val * scale + bias

    return result

def linear_forward_quantized(x, weight, scales, biases, group_size=64, bits=4):
    """Forward pass for a quantized linear layer using MLX for dequant."""
    import mlx.core as mx

    # Use MLX's quantized matmul
    x_mx = mx.array(x)
    w_mx = mx.array(weight)
    s_mx = mx.array(scales)
    b_mx = mx.array(biases)

    result = mx.quantized_matmul(x_mx, w_mx, s_mx, b_mx, group_size=group_size, bits=bits)
    mx.eval(result)
    return np.array(result, dtype=np.float32)

def main():
    cfg = load_config()
    print("=== Model Config ===")
    for k in ['hidden_size', 'num_attention_heads', 'num_key_value_heads',
              'linear_num_key_heads', 'linear_num_value_heads',
              'linear_key_head_dim', 'linear_value_head_dim',
              'linear_conv_kernel_dim', 'intermediate_size',
              'num_experts', 'num_experts_per_tok', 'num_hidden_layers',
              'rms_norm_eps', 'vocab_size', 'full_attention_interval',
              'head_dim', 'partial_rotary_factor', 'decoder_sparse_step']:
        print(f"  {k}: {cfg.get(k, 'NOT FOUND')}")

    # Derive layer types
    interval = cfg.get('full_attention_interval', 4)
    num_layers = cfg['num_hidden_layers']
    layer_types = [
        'linear_attention' if bool((i + 1) % interval) else 'full_attention'
        for i in range(num_layers)
    ]
    print(f"\n  Layer 0 type: {layer_types[0]}")
    assert layer_types[0] == 'linear_attention', "Layer 0 should be GDN!"

    # Load weights
    print("\n=== Loading weights ===")
    import mlx.core as mx

    weights = load_weights_mlx()

    # Print available layer 0 weight keys
    layer0_keys = sorted([k for k in weights.keys() if k.startswith('model.layers.0.')])
    print(f"Layer 0 weight keys ({len(layer0_keys)}):")
    for k in layer0_keys:
        w = weights[k]
        print(f"  {k}: shape={w.shape} dtype={w.dtype}")

    # Get embedding
    embed_w = weights['model.embed_tokens.weight']

    # Tokenize "Hello" - use the tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    tokens = tokenizer.encode("Hello", add_special_tokens=False)
    print(f"\n=== Input tokens: {tokens} ===")

    # Embed
    input_ids = mx.array([tokens])  # [1, seq_len]
    h = mx.take(embed_w, input_ids, axis=0)
    mx.eval(h)
    h_np = np.array(h, dtype=np.float32)
    print(f"Embedding: shape={h_np.shape} mean={h_np.mean():.6f} std={h_np.std():.6f}")
    print(f"  first 4: {h_np[0, 0, :4]}")

    # === Layer 0: GDN ===
    print("\n=== Layer 0 (GDN) ===")

    # Input layernorm
    ln_w = weights['model.layers.0.input_layernorm.weight']
    eps = cfg['rms_norm_eps']

    normed = mx.fast.rms_norm(h, ln_w, eps)
    mx.eval(normed)
    normed_np = np.array(normed, dtype=np.float32)
    print(f"input (after layernorm): mean={normed_np.mean():.6f} std={normed_np.std():.6f}")
    print(f"  first 4: {normed_np[0, 0, :4]}")

    # GDN params
    nk = cfg['linear_num_key_heads']      # 16
    nv = cfg['linear_num_value_heads']     # 32
    dk = cfg['linear_key_head_dim']        # 128
    dv = cfg['linear_value_head_dim']      # 128
    v_per_k = nv // nk                     # 2
    conv_kernel = cfg['linear_conv_kernel_dim']  # 4
    key_dim = dk * nk                      # 2048
    value_dim = dv * nv                    # 4096
    conv_dim = key_dim * 2 + value_dim     # 8192

    print(f"\nGDN dims: nk={nk} nv={nv} dk={dk} dv={dv} v_per_k={v_per_k}")
    print(f"  key_dim={key_dim} value_dim={value_dim} conv_dim={conv_dim}")

    # Projections (quantized)
    prefix = 'model.layers.0.linear_attn.'

    # qkvz projection
    qkvz_w = weights[prefix + 'in_proj_qkvz.weight']
    qkvz_s = weights[prefix + 'in_proj_qkvz.scales']
    qkvz_b = weights[prefix + 'in_proj_qkvz.biases']

    mixed_qkvz = mx.quantized_matmul(normed, qkvz_w, qkvz_s, qkvz_b, group_size=64, bits=4)
    mx.eval(mixed_qkvz)
    mixed_qkvz_np = np.array(mixed_qkvz, dtype=np.float32)
    print(f"\nqkvz_proj: mean={mixed_qkvz_np.mean():.6f} std={mixed_qkvz_np.std():.6f}")
    print(f"  first 4: {mixed_qkvz_np[0, 0, :4]}")

    # ba projection
    ba_w = weights[prefix + 'in_proj_ba.weight']
    ba_s = weights[prefix + 'in_proj_ba.scales']
    ba_b = weights[prefix + 'in_proj_ba.biases']

    mixed_ba = mx.quantized_matmul(normed, ba_w, ba_s, ba_b, group_size=64, bits=4)
    mx.eval(mixed_ba)
    mixed_ba_np = np.array(mixed_ba, dtype=np.float32)
    print(f"ba_proj: mean={mixed_ba_np.mean():.6f} std={mixed_ba_np.std():.6f}")
    print(f"  first 4: {mixed_ba_np[0, 0, :4]}")

    # fix_query_key_value_ordering (matching HuggingFace reference)
    B, S = 1, len(tokens)

    qkvz = mixed_qkvz_np.reshape(B, S, nk, -1)
    ba = mixed_ba_np.reshape(B, S, nk, -1)

    # Split qkvz: [q(dk), k(dk), v(v_per_k*dv), z(v_per_k*dv)]
    split_sizes_qkvz = [dk, dk, v_per_k * dv, v_per_k * dv]
    idx = 0
    q = qkvz[:, :, :, idx:idx+split_sizes_qkvz[0]]; idx += split_sizes_qkvz[0]
    k = qkvz[:, :, :, idx:idx+split_sizes_qkvz[1]]; idx += split_sizes_qkvz[1]
    v = qkvz[:, :, :, idx:idx+split_sizes_qkvz[2]]; idx += split_sizes_qkvz[2]
    z = qkvz[:, :, :, idx:idx+split_sizes_qkvz[3]]

    # Split ba: [b(v_per_k), a(v_per_k)]
    b_val = ba[:, :, :, :v_per_k].reshape(B, S, nv)
    a_val = ba[:, :, :, v_per_k:].reshape(B, S, nv)

    # Reshape v, z
    v = v.reshape(B, S, -1, dv)
    z = z.reshape(B, S, -1, dv)

    # Flatten q, k, v for conv
    q_flat = q.reshape(B, S, -1)  # [B, S, key_dim]
    k_flat = k.reshape(B, S, -1)  # [B, S, key_dim]
    v_flat = v.reshape(B, S, -1)  # [B, S, value_dim]

    mixed_qkv = np.concatenate([q_flat, k_flat, v_flat], axis=-1)  # [B, S, conv_dim]
    print(f"\nmixed_qkv (pre-conv): shape={mixed_qkv.shape}")

    # Conv1d (causal, depthwise)
    conv_w = weights[prefix + 'conv1d.weight']
    mx.eval(conv_w)
    conv_w_np = np.array(conv_w, dtype=np.float32)
    print(f"conv1d weight: shape={conv_w_np.shape}")

    # For prefill without cache: pad left by (kernel-1), apply conv, take first S outputs
    # HF reference: F.pad(mixed_qkv.transpose(1,2), (kernel-1, 0)) then conv1d then silu
    # mixed_qkv is [B, S, C], transpose to [B, C, S], pad left, conv, take [:,:,:S]
    mixed_qkv_t = mixed_qkv.transpose(0, 2, 1)  # [B, C, S]
    padded = np.pad(mixed_qkv_t, ((0,0), (0,0), (conv_kernel-1, 0)), mode='constant')

    # Depthwise conv1d: each channel independently
    # conv_w shape in MLX: [C, 1, K] (out_channels, in_channels/groups, kernel_size)
    # In numpy: for each channel c, output[c, t] = sum_k(input[c, t+k] * weight[c, 0, k])
    conv_out_t = np.zeros_like(mixed_qkv_t)
    for c in range(conv_dim):
        for t in range(S):
            for kk in range(conv_kernel):
                conv_out_t[0, c, t] += padded[0, c, t + kk] * conv_w_np[c, 0, kk]

    # Apply silu
    conv_out_t = silu(conv_out_t)
    conv_out = conv_out_t.transpose(0, 2, 1)  # back to [B, S, C]
    print(f"conv_out: mean={conv_out.mean():.6f} std={conv_out.std():.6f}")
    print(f"  first 4: {conv_out[0, 0, :4]}")

    # Split conv output
    q_conv = conv_out[:, :, :key_dim].reshape(B, S, nk, dk)
    k_conv = conv_out[:, :, key_dim:2*key_dim].reshape(B, S, nk, dk)
    v_conv = conv_out[:, :, 2*key_dim:].reshape(B, S, nv, dv)

    # L2 normalization (HuggingFace reference, correct)
    q_l2 = l2norm(q_conv, axis=-1, eps=1e-6)
    k_l2 = l2norm(k_conv, axis=-1, eps=1e-6)

    print(f"\nq (L2 normed, CORRECT): mean={q_l2.mean():.6f} std={q_l2.std():.6f}")
    print(f"  first 4: {q_l2[0, 0, 0, :4]}")
    print(f"k (L2 normed, CORRECT): mean={k_l2.mean():.6f} std={k_l2.std():.6f}")
    print(f"  first 4: {k_l2[0, 0, 0, :4]}")

    # C++ bug reproduction: RMS norm with wrong weight for q
    inv_scale = dk ** -0.5
    q_cpp_weight = inv_scale * inv_scale  # 1/D -- THIS IS THE BUG
    k_cpp_weight = inv_scale              # 1/sqrt(D) -- this is correct

    q_rms_buggy = rms_norm(q_conv, np.full(dk, q_cpp_weight), 1e-6)
    k_rms_correct = rms_norm(k_conv, np.full(dk, k_cpp_weight), 1e-6)

    print(f"\nq (C++ buggy RMS norm): mean={q_rms_buggy.mean():.6f} std={q_rms_buggy.std():.6f}")
    print(f"  first 4: {q_rms_buggy[0, 0, 0, :4]}")
    print(f"k (C++ RMS norm, correct): mean={k_rms_correct.mean():.6f} std={k_rms_correct.std():.6f}")
    print(f"  first 4: {k_rms_correct[0, 0, 0, :4]}")

    # Verify k matches
    print(f"\nk L2 vs k RMS match: {np.allclose(k_l2, k_rms_correct, atol=1e-4)}")
    print(f"q L2 vs q RMS(buggy) match: {np.allclose(q_l2, q_rms_buggy, atol=1e-4)}")
    print(f"q ratio (l2/buggy): {(q_l2.std() / q_rms_buggy.std()):.4f} (should be ~{dk**0.5:.4f} = sqrt({dk}))")

    # Compute beta and g
    beta = sigmoid(b_val)
    g_exp = -np.exp(np.array(weights[prefix + 'A_log'], dtype=np.float32)) * softplus(
        a_val + np.array(weights[prefix + 'dt_bias'], dtype=np.float32)
    )

    print(f"\nbeta: mean={beta.mean():.6f} std={beta.std():.6f}")
    print(f"g (decay logits): mean={g_exp.mean():.6f} std={g_exp.std():.6f}")

    # Repeat q, k heads to match v heads
    if nv // nk > 1:
        q_l2_rep = np.repeat(q_l2, v_per_k, axis=2)
        k_l2_rep = np.repeat(k_l2, v_per_k, axis=2)
    else:
        q_l2_rep = q_l2
        k_l2_rep = k_l2

    # Run GDN recurrence (torch fallback style)
    # State: [B, Hv, Dv, Dk]
    state = np.zeros((B, nv, dv, dk), dtype=np.float32)
    g_decay = np.exp(g_exp)  # actual decay factors

    ys = []
    for t in range(S):
        q_t = q_l2_rep[:, t, :, :]   # [B, Hv, Dk]
        k_t = k_l2_rep[:, t, :, :]   # [B, Hv, Dk]
        v_t = v_conv[:, t, :, :]     # [B, Hv, Dv]
        g_t = g_decay[:, t, :]       # [B, Hv]
        beta_t = beta[:, t, :]       # [B, Hv]

        # Expand decay
        decay = g_t[:, :, None, None]  # [B, Hv, 1, 1]

        # State update
        s = state * decay
        kv_mem = np.sum(s * k_t[:, :, None, :], axis=-1)  # [B, Hv, Dv]
        delta = (v_t - kv_mem) * beta_t[:, :, None]       # [B, Hv, Dv]
        s = s + k_t[:, :, None, :] * delta[:, :, :, None]  # [B, Hv, Dv, Dk]
        y = np.sum(s * q_t[:, :, None, :], axis=-1)        # [B, Hv, Dv]

        ys.append(y[:, None, :, :])
        state = s

    ssm_out = np.concatenate(ys, axis=1)  # [B, S, Hv, Dv]
    print(f"\nssm_out (CORRECT, L2 norm): mean={ssm_out.mean():.6f} std={ssm_out.std():.6f}")
    print(f"  first 4: {ssm_out[0, 0, 0, :4]}")

    # Also compute with buggy q
    if nv // nk > 1:
        q_buggy_rep = np.repeat(q_rms_buggy, v_per_k, axis=2)
    else:
        q_buggy_rep = q_rms_buggy

    state_buggy = np.zeros((B, nv, dv, dk), dtype=np.float32)
    ys_buggy = []
    for t in range(S):
        q_t = q_buggy_rep[:, t, :, :]
        k_t = k_l2_rep[:, t, :, :]   # k is correct in C++
        v_t = v_conv[:, t, :, :]
        g_t = g_decay[:, t, :]
        beta_t = beta[:, t, :]

        decay = g_t[:, :, None, None]
        s = state_buggy * decay
        kv_mem = np.sum(s * k_t[:, :, None, :], axis=-1)
        delta = (v_t - kv_mem) * beta_t[:, :, None]
        s = s + k_t[:, :, None, :] * delta[:, :, :, None]
        y = np.sum(s * q_t[:, :, None, :], axis=-1)

        ys_buggy.append(y[:, None, :, :])
        state_buggy = s

    ssm_out_buggy = np.concatenate(ys_buggy, axis=1)
    print(f"ssm_out (BUGGY, C++ q norm): mean={ssm_out_buggy.mean():.6f} std={ssm_out_buggy.std():.6f}")
    print(f"  first 4: {ssm_out_buggy[0, 0, 0, :4]}")

    print(f"\nssm_out ratio (correct/buggy): {ssm_out.std() / ssm_out_buggy.std():.4f}")
    print(f"  Expected ratio: sqrt({dk}) = {dk**0.5:.4f}")

    # Gated norm (RMSNormGated)
    norm_w = weights[prefix + 'norm.weight']
    mx.eval(norm_w)
    norm_w_np = np.array(norm_w, dtype=np.float32)

    # norm(ssm_out, z): rms_norm(ssm_out) * silu(z)
    gated_correct = rms_norm(ssm_out, norm_w_np, eps) * silu(z.astype(np.float32))
    gated_buggy = rms_norm(ssm_out_buggy, norm_w_np, eps) * silu(z.astype(np.float32))

    print(f"\ngated_norm (CORRECT): mean={gated_correct.mean():.6f} std={gated_correct.std():.6f}")
    print(f"gated_norm (BUGGY):   mean={gated_buggy.mean():.6f} std={gated_buggy.std():.6f}")

    # Output projection
    out_w = weights[prefix + 'out_proj.weight']
    out_s = weights[prefix + 'out_proj.scales']
    out_b = weights[prefix + 'out_proj.biases']

    gated_flat_correct = mx.array(gated_correct.reshape(B, S, -1).astype(np.float32))
    gated_flat_buggy = mx.array(gated_buggy.reshape(B, S, -1).astype(np.float32))

    gdn_out_correct = mx.quantized_matmul(gated_flat_correct, out_w, out_s, out_b, group_size=64, bits=4)
    gdn_out_buggy = mx.quantized_matmul(gated_flat_buggy, out_w, out_s, out_b, group_size=64, bits=4)
    mx.eval(gdn_out_correct, gdn_out_buggy)

    gdn_out_correct_np = np.array(gdn_out_correct, dtype=np.float32)
    gdn_out_buggy_np = np.array(gdn_out_buggy, dtype=np.float32)

    print(f"\ngdn_output (CORRECT): mean={gdn_out_correct_np.mean():.6f} std={gdn_out_correct_np.std():.6f}")
    print(f"  first 4: {gdn_out_correct_np[0, 0, :4]}")
    print(f"gdn_output (BUGGY):   mean={gdn_out_buggy_np.mean():.6f} std={gdn_out_buggy_np.std():.6f}")
    print(f"  first 4: {gdn_out_buggy_np[0, 0, :4]}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"BUG: C++ q normalization uses RMS norm with weight = 1/D = {1/dk:.6f}")
    print(f"     Should use L2 norm, equivalent to RMS norm with weight = 1/sqrt(D) = {1/dk**0.5:.6f}")
    print(f"     This scales q DOWN by an extra factor of sqrt({dk}) = {dk**0.5:.4f}")
    print(f"     Making ssm_out ~{dk**0.5:.0f}x too small")
    print(f"\nFIX in qwen3_next.cpp line 508:")
    print(f"  BEFORE: auto q_norm_w = mx::full({{head_k_dim_}}, inv_scale * inv_scale, ...);")
    print(f"  AFTER:  auto q_norm_w = mx::full({{head_k_dim_}}, inv_scale, ...);")

if __name__ == "__main__":
    main()
