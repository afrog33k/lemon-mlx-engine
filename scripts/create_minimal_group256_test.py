#!/usr/bin/env python3.12
"""
Create a minimal test model with group_size=256 for GPU kernel validation.
Uses synthetic weights with known values to verify correctness.
"""

# Set LD_LIBRARY_PATH BEFORE importing MLX
import os
os.environ['LD_LIBRARY_PATH'] = "/home/reckon/projects/mlx-vulkan/python/mlx/lib64:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['MLX_DEFAULT_DEVICE'] = 'cpu'

import json
from pathlib import Path
import struct
import numpy as np
import mlx.core as mx

def create_minimal_test_model():
    """Create a minimal Qwen-like model with group_size=256 quantization."""

    output_path = Path("/home/reckon/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-group256-test")
    output_path.mkdir(exist_ok=True, parents=True)

    # Reference model structure
    ref_path = Path("/home/reckon/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/main")

    # Load reference to get exact structure
    with open(ref_path / "model.safetensors", 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_len).decode('utf-8')
        ref_header = json.loads(header_json)
        ref_data = f.read()

    print(f"Reference model has {len(ref_header)} tensors")

    # Build new model
    new_header = {}
    new_data_parts = []
    current_offset = 0

    def add_tensor(name, array, dtype_str):
        nonlocal current_offset
        if dtype_str == 'U32':
            data = np.array(array, dtype=np.uint32).tobytes()
        elif dtype_str == 'BF16':
            data = np.array(array, dtype=np.float16).tobytes()
        elif dtype_str == 'F32':
            data = np.array(array, dtype=np.float32).tobytes()
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

    # Create test weights with known values
    # Use simple pattern: weight[i,j] = (i + j) * small_factor
    # This makes validation easier

    processed_keys = set()

    # Process a small subset of keys for testing
    # Skip embed_tokens since hidden_dim=128 is not divisible by 256
    test_keys = [
        'language_model.model.layers.0.mlp.gate_proj.weight',
        'language_model.model.layers.0.self_attn.q_proj.weight',
    ]

    for key in sorted(ref_header.keys()):
        if key == '__metadata__':
            new_header[key] = ref_header[key]
            continue

        ref_spec = ref_header[key]
        orig_shape = ref_spec['shape']
        orig_dtype = ref_spec['dtype']

        # Check if this is a quantized layer we want to test
        base_key = None
        if key.endswith('.weight'):
            base_key = key[:-7]  # Remove ".weight"
            if base_key + '.scales' in ref_header:
                # This is a quantized weight
                if key in test_keys:
                    print(f"  Creating test weights for {key}")

                    # Get original unpacked shape
                    # Packed shape [out, hidden/8] -> Original [out, hidden]
                    packed_cols = orig_shape[1]
                    hidden_dim = packed_cols * 8  # 4-bit = 8 values per uint32

                    # Create synthetic weights with known pattern
                    out_dim = orig_shape[0]
                    out_indices = np.arange(out_dim).reshape(-1, 1)
                    in_indices = np.arange(hidden_dim).reshape(1, -1)
                    w_test = np.sin(out_indices * 0.01) * np.sin(in_indices * 0.01) * 0.1

                    # Quantize with group_size=256
                    w_mx = mx.array(w_test)
                    q256 = mx.quantize(w_mx, group_size=256, bits=4)
                    packed_w, scales, biases = q256

                    add_tensor(key, mx.array(packed_w), 'U32')
                    add_tensor(base_key + '.scales', mx.array(scales), 'BF16')
                    add_tensor(base_key + '.biases', mx.array(biases), 'BF16')

                    processed_keys.add(base_key)
                    processed_keys.add(base_key + '.scales')
                    processed_keys.add(base_key + '.biases')
                    processed_keys.add(key)
                    continue

        # Copy non-quantized tensors or skip quantized ones we already processed
        if key not in processed_keys:
            # For other quantized layers, just copy reference data
            data_start, data_end = ref_spec['data_offsets']
            data_bytes = ref_data[data_start:data_end]
            data_size = len(data_bytes)

            new_header[key] = {
                'dtype': ref_spec['dtype'],
                'shape': ref_spec['shape'],
                'data_offsets': [current_offset, current_offset + data_size]
            }
            new_data_parts.append(data_bytes)
            current_offset += data_size

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

    # Copy config (update for group_size=256)
    with open(ref_path / "config.json") as f:
        config = json.load(f)

    config['quantization'] = {'group_size': 256, 'bits': 4, 'mode': 'affine'}
    config['quantization_config'] = {'group_size': 256, 'bits': 4, 'mode': 'affine'}

    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files
    import shutil
    for f_name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json',
                    'special_tokens_map.json']:
        src = ref_path / f_name
        if src.exists():
            shutil.copy(src, output_path / f_name)

    print(f"Test model saved to {output_path}")
    return output_path


if __name__ == "__main__":
    create_minimal_test_model()
