#!/usr/bin/env python3.12
"""
Convert Qwen3.5-0.8B from group_size=64 to group_size=256.
This version dequantizes the existing MLX model and re-quantizes.
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
import mlx.core as mx

def convert_model():
    ref_path = Path("/home/reckon/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/main")
    output_path = Path("/home/reckon/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-group256")
    output_path.mkdir(exist_ok=True)

    # Load reference model
    model_file = ref_path / "model.safetensors"
    with open(model_file, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)
        ref_data = f.read()

    print(f"Reference model has {len(header)} tensors")

    # Build new model
    new_header = {}
    new_data_parts = []
    current_offset = 0
    processed_base_keys = set()

    def add_tensor(name, array, dtype_str):
        nonlocal current_offset
        if dtype_str == 'U32':
            data = np.array(array, dtype=np.uint32).tobytes()
        elif dtype_str == 'BF16':
            data = np.array(array, dtype=np.float16).tobytes()
        elif dtype_str == 'F32':
            data = np.array(array, dtype=np.float32).tobytes()
        elif dtype_str == 'F16':
            data = np.array(array, dtype=np.float16).tobytes()
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

    def load_tensor(key):
        spec = header[key]
        start, end = spec['data_offsets']
        tensor_bytes = ref_data[start:end]
        dtype = spec['dtype']
        shape = spec['shape']

        if dtype == 'U32':
            return np.frombuffer(tensor_bytes, dtype=np.uint32).reshape(shape)
        elif dtype == 'BF16':
            return np.frombuffer(tensor_bytes, dtype=np.float16).reshape(shape)
        elif dtype == 'F32':
            return np.frombuffer(tensor_bytes, dtype=np.float32).reshape(shape)
        elif dtype == 'F16':
            return np.frombuffer(tensor_bytes, dtype=np.float16).reshape(shape)
        else:
            raise ValueError(f'Unknown dtype: {dtype}')

    # Process keys in reference order
    for key in sorted(header.keys()):
        if key == '__metadata__':
            new_header[key] = header[key]
            continue

        ref_spec = header[key]

        # Check if this is a quantized layer
        if key.endswith('.scales') or key.endswith('.biases') or key.endswith('.weight'):
            if key.endswith('.weight'):
                base_key = key[:-7]
            elif key.endswith('.scales'):
                base_key = key[:-7]
            elif key.endswith('.biases'):
                base_key = key[:-7]
            else:
                base_key = key

            # Only process on .weight key
            if not key.endswith('.weight'):
                continue

            # Check if already processed
            if base_key in processed_base_keys:
                continue

            # Check if this layer has scales/biases (is quantized)
            scales_key = base_key + '.scales'
            biases_key = base_key + '.biases'

            if scales_key in header and biases_key in header:
                # Dequantize and re-quantize with group_size=256
                print(f"  Converting {base_key}")

                packed = load_tensor(base_key + '.weight')
                scales = load_tensor(scales_key)
                biases = load_tensor(biases_key)

                # Dequantize using MLX's built-in function
                packed_mx = mx.array(packed)
                scales_mx = mx.array(scales)
                biases_mx = mx.array(biases)
                dequant = mx.dequantize(packed_mx, scales_mx, biases_mx, group_size=64, bits=4)

                # Re-quantize with group_size=256
                w_mx = mx.array(dequant)
                quantized = mx.quantize(w_mx, group_size=256, bits=4)
                packed_w, scales_new, biases_new = quantized

                add_tensor(base_key + '.weight', mx.array(packed_w), 'U32')
                add_tensor(base_key + '.scales', mx.array(scales_new), 'BF16')
                add_tensor(base_key + '.biases', mx.array(biases_new), 'BF16')

                processed_base_keys.add(base_key)
                continue

        # Non-quantized tensor - copy as-is
        if key not in new_header:
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

    # Copy and update config
    with open(ref_path / "config.json") as f:
        config = json.load(f)

    # Update quantization
    config['quantization'] = {'group_size': 256, 'bits': 4, 'mode': 'affine'}
    config['quantization_config'] = {'group_size': 256, 'bits': 4, 'mode': 'affine'}

    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Copy other files
    import shutil
    for f_name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json',
                   'special_tokens_map.json', 'vocab.json', 'merges.txt']:
        src = ref_path / f_name
        if src.exists():
            shutil.copy(src, output_path / f_name)

    print(f"Model saved to {output_path}")
    print(f"  - model.safetensors ({len(new_header)} tensors)")
    print(f"  - config.json (group_size=256)")

    return output_path


if __name__ == "__main__":
    convert_model()
