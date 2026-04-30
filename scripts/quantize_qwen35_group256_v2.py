#!/usr/bin/env python3.12
"""
Re-quantize Qwen3.5-0.8B from group_size=64 to group_size=256.
This version produces output matching MLX-community model format.
"""

# Set LD_LIBRARY_PATH BEFORE importing MLX
import os
os.environ['LD_LIBRARY_PATH'] = "/home/reckon/projects/mlx-vulkan/python/mlx/lib64:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['MLX_DEFAULT_DEVICE'] = 'cpu'

import sys
import json
from pathlib import Path
import struct
import numpy as np
import torch
from safetensors.torch import load_file, save_file
import mlx.core as mx

def quantize_model_group256():
    source_path = Path("/home/reckon/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17")
    output_path = Path("/home/reckon/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-group256")
    output_path.mkdir(exist_ok=True)

    # Load reference MLX-community model to get exact key structure
    ref_path = Path("/home/reckon/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/main/model.safetensors")

    with open(ref_path, 'rb') as f:
        ref_header_len = struct.unpack('<Q', f.read(8))[0]
        ref_header_json = f.read(ref_header_len).decode('utf-8')
        ref_header = json.loads(ref_header_json)
        ref_data = f.read()

    print(f"Reference model has {len(ref_header)} tensors")
    ref_keys = set(ref_header.keys())

    # Load FP32 model
    sf_file = source_path / "model.safetensors-00001-of-00001.safetensors"
    fp32_weights = load_file(str(sf_file))

    # Build new model following exact reference structure
    new_header = {}
    new_data_parts = []
    current_offset = 0
    quantized_base_keys = set()  # Track which base keys we've already quantized

    def add_tensor(name, array, dtype_str):
        nonlocal current_offset
        # Convert to appropriate dtype
        if dtype_str == 'U32':
            data = np.array(array, dtype=np.uint32).tobytes()
        elif dtype_str == 'BF16':
            data = np.array(array, dtype=np.float16).tobytes()
        elif dtype_str == 'F32':
            data = np.array(array, dtype=np.float32).tobytes()
        elif dtype_str == 'F16':
            data = np.array(array, dtype=np.float16).tobytes()
        elif dtype_str == 'I16':
            data = np.array(array, dtype=np.int16).tobytes()
        else:
            data = array.tobytes() if hasattr(array, 'tobytes') else bytes(array)

        shape = list(array.shape)
        data_size = len(data)

        new_header[name] = {
            'dtype': dtype_str,
            'shape': shape,
            'data_offsets': [current_offset, current_offset + data_size]
        }
        new_data_parts.append(data)
        current_offset += data_size

    # Process keys in reference order
    for key in sorted(ref_header.keys()):
        if key == '__metadata__':
            # Copy metadata as-is
            new_header[key] = ref_header[key]
            continue

        ref_spec = ref_header[key]
        orig_dtype = ref_spec['dtype']
        orig_shape = ref_spec['shape']

        # Check if this is a quantized layer (has scales/biases)
        if key.endswith('.scales') or key.endswith('.biases') or key.endswith('.weight'):
            # Extract base name
            if key.endswith('.weight'):
                base_key = key[:-7]  # Remove ".weight"
            elif key.endswith('.scales'):
                base_key = key[:-7]
            elif key.endswith('.biases'):
                base_key = key[:-7]
            else:
                base_key = key

            # Check if this layer should be quantized (Linear or Embedding)
            is_quantizable = (
                any(x in base_key for x in ['embed_tokens', 'gate_proj', 'down_proj', 'up_proj',
                      'in_proj', 'out_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']) or
                (base_key.endswith('.mlp') or base_key.endswith('.self_attn') or
                 (base_key.endswith('.linear_attn')))
            ) and len(orig_shape) == 2

            if is_quantizable:
                # Only process on .weight key to avoid re-quantizing
                if not key.endswith('.weight'):
                    # Skip .scales and .biases - they were added during .weight processing
                    continue

                # Check if we already processed this base key
                if base_key in quantized_base_keys:
                    continue

                # Load FP32 weight and quantize with group_size=256
                weight_key = base_key + '.weight'

                # Map MLX key to torch key
                # MLX: language_model.model.* → FP32: model.language_model.*
                torch_key = weight_key
                if torch_key.startswith('language_model.model.'):
                    # Swap the prefix: language_model.model. → model.language_model.
                    rest = torch_key[len('language_model.model.'):]  # e.g., embed_tokens.weight
                    torch_key = 'model.language_model.' + rest

                if torch_key in fp32_weights:
                    w_fp32 = fp32_weights[torch_key]
                    if w_fp32.dtype == torch.bfloat16:
                        w_np = w_fp32.float().numpy()
                    else:
                        w_np = w_fp32.numpy()

                    # Quantize with group_size=256
                    if 'embed_tokens' in base_key:
                        # Embedding: vocab x hidden - quantize as-is (no transpose)
                        vocab_size, embed_dim = w_np.shape
                        print(f"  Quantizing {base_key}: Embedding {vocab_size} x {embed_dim}")
                        w_mx = mx.array(w_np)  # [vocab, hidden]
                        quantized = mx.quantize(w_mx, group_size=256, bits=4)
                        packed_w, scales, biases = quantized
                    else:
                        # Linear: [out, in] - need to transpose for quantize
                        print(f"  Quantizing {base_key}: shape={w_np.shape}")
                        w_mx = mx.array(w_np)  # Already [out, in]
                        quantized = mx.quantize(w_mx, group_size=256, bits=4)
                        packed_w, scales, biases = quantized

                    # Add packed weight
                    add_tensor(base_key + '.weight', mx.array(packed_w), 'U32')
                    add_tensor(base_key + '.scales', mx.array(scales), 'BF16')
                    add_tensor(base_key + '.biases', mx.array(biases), 'BF16')

                    # Mark as processed
                    quantized_base_keys.add(base_key)

                    # Skip original scales/biases keys since we added them above
                    continue

        # For non-quantized tensors or scales/biases already handled
        # Skip keys that were added during quantization (weight/scales/biases of quantized layers)
        if key not in new_header:
            # Copy from reference data (layernorm, etc.)
            data_start, data_end = ref_spec['data_offsets']
            data_bytes = ref_data[data_start:data_end]
            data_size = len(data_bytes)

            # Create entry with NEW offsets (not the reference offsets)
            new_header[key] = {
                'dtype': ref_spec['dtype'],
                'shape': ref_spec['shape'],
                'data_offsets': [current_offset, current_offset + data_size]
            }
            new_data_parts.append(data_bytes)
            current_offset += data_size
        else:
            # Key already exists - verify it's from quantization
            # This should be weight/scales/biases that were quantized above
            pass

    # Metadata is already copied above

    # Combine all data
    all_data = b''.join(new_data_parts)

    # Write safetensors
    new_header_json = json.dumps(new_header, separators=(",", ":")).encode('utf-8')
    new_header_len = len(new_header_json)

    output_file = output_path / "model.safetensors"
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<Q', new_header_len))
        f.write(new_header_json)
        f.write(all_data)

    print(f"\nSaved {len(new_header)} tensors to {output_file}")

    # Copy and update config - use MLX-community config as base
    mlx_community_config = Path("/home/reckon/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/main/config.json")
    new_config = output_path / "config.json"

    with open(mlx_community_config) as f:
        config = json.load(f)

    # Update quantization
    config['quantization'] = {'group_size': 256, 'bits': 4, 'mode': 'affine'}
    config['quantization_config'] = {'group_size': 256, 'bits': 4, 'mode': 'affine'}

    with open(new_config, 'w') as f:
        json.dump(config, f, indent=2)

    # Copy other files from MLX-community (tokenizer, etc.)
    import shutil
    mlx_community_path = Path("/home/reckon/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/main")
    for f_name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json',
                    'special_tokens_map.json', 'vocab.json', 'merges.txt']:
        src = mlx_community_path / f_name
        if src.exists():
            shutil.copy(src, output_path / f_name)

    print(f"Model saved to {output_path}")
    print(f"  - model.safetensors ({len(new_header)} tensors)")
    print(f"  - config.json (group_size=256)")

    return output_path


if __name__ == "__main__":
    quantize_model_group256()
