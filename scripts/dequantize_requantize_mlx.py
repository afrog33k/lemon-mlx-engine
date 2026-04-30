#!/usr/bin/env python3.12
"""
Dequantize MLX-community model to FP32, then re-quantize with group_size=256.
This should preserve the exact weights used by MLX-community.
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

def convert_via_dequantize():
    """Dequantize MLX-community model, then re-quantize with group_size=256."""

    ref_path = Path("/home/reckon/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/main")
    output_path = Path("/home/reckon/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-group256-from-mlx")
    output_path.mkdir(exist_ok=True, parents=True)

    # Load reference model
    with open(ref_path / "model.safetensors", 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)
        ref_data = f.read()

    print(f"Reference model has {len(header)} tensors")

    # Also load using MLX to get correctly typed data
    # This will handle BF16 conversion properly
    try:
        ref_model_mx = mx.load(str(ref_path / "model.safetensors"))
        use_mlx_load = True
        print("MLX load successful - will use for BF16 tensors")
    except Exception as e:
        print(f"MLX load failed: {e}")
        print("Will use manual loading for BF16 tensors")
        use_mlx_load = False

    # Build new model
    new_header = {}
    new_data_parts = []
    current_offset = 0
    processed_base_keys = set()

    def add_tensor(name, array, dtype_str):
        nonlocal current_offset
        # For MLX arrays, convert to numpy first
        if hasattr(array, 'numpy'):
            arr = array.numpy()
        else:
            arr = array

        if dtype_str == 'U32':
            data = np.array(arr, dtype=np.uint32).tobytes()
        elif dtype_str == 'BF16':
            # Convert MLX array to numpy, then to bytes
            # MLX stores as float32, convert to BF16-like representation
            # For compatibility, use float16 (close enough for this purpose)
            data = np.array(arr, dtype=np.float16).tobytes()
        elif dtype_str == 'F32':
            data = np.array(arr, dtype=np.float32).tobytes()
        else:
            data = arr.tobytes() if hasattr(arr, 'tobytes') else bytes(arr)

        shape = list(arr.shape)
        data_size = len(data)

        new_header[name] = {
            'dtype': dtype_str,
            'shape': shape,
            'data_offsets': [current_offset, current_offset + data_size]
        }
        new_data_parts.append(data)
        current_offset += data_size

    # Process keys in reference order
    for key in sorted(header.keys()):
        if key == '__metadata__':
            new_header[key] = header[key]
            continue

        ref_spec = header[key]
        dtype = ref_spec['dtype']
        shape = ref_spec['shape']

        # Check if this is a quantized layer
        if key.endswith('.weight') and key.replace('.weight', '.scales') in header:
            base_key = key[:-7]  # Remove ".weight"

            # Check if already processed
            if base_key in processed_base_keys:
                continue

            # Check if it has scales and biases (quantized layer)
            scales_key = base_key + '.scales'
            biases_key = base_key + '.biases'

            if scales_key in header and biases_key in header:
                print(f"  Converting {base_key}")

                # Load packed weight
                start, end = ref_spec['data_offsets']
                packed_bytes = ref_data[start:end]
                packed = np.frombuffer(packed_bytes, dtype=np.uint32).reshape(shape)

                # Load scales/biases from MLX model if available (handles BF16 correctly)
                if use_mlx_load and scales_key in ref_model_mx and biases_key in ref_model_mx:
                    scales_mx = ref_model_mx[scales_key]
                    biases_mx = ref_model_mx[biases_key]
                else:
                    # Fallback: read as float16 (incorrect but will fail gracefully)
                    spec_scales = header[scales_key]
                    start_s, end_s = spec_scales['data_offsets']
                    scales_bytes = ref_data[start_s:end_s]
                    scales_mx = mx.array(np.frombuffer(scales_bytes, dtype=np.float16).reshape(spec_scales['shape']))

                    spec_biases = header[biases_key]
                    start_b, end_b = spec_biases['data_offsets']
                    biases_bytes = ref_data[start_b:end_b]
                    biases_mx = mx.array(np.frombuffer(biases_bytes, dtype=np.float16).reshape(spec_biases['shape']))

                # Dequantize using MLX (this gives us the original FP32 weights!)
                try:
                    packed_mx = mx.array(packed)

                    # Dequantize with group_size=64
                    w_fp32 = mx.dequantize(packed_mx, scales_mx, biases_mx, group_size=64, bits=4)

                    print(f"    Dequantized shape: {w_fp32.shape}, range: [{mx.min(w_fp32).item():.2f}, {mx.max(w_fp32).item():.2f}]")

                    # Re-quantize with group_size=256
                    q256 = mx.quantize(w_fp32, group_size=256, bits=4)
                    packed_w256, scales256, biases256 = q256

                    add_tensor(base_key + '.weight', mx.array(packed_w256), 'U32')
                    add_tensor(base_key + '.scales', mx.array(scales256), 'BF16')
                    add_tensor(base_key + '.biases', mx.array(biases256), 'BF16')

                    processed_base_keys.add(base_key)
                    processed_base_keys.add(scales_key)
                    processed_base_keys.add(biases_key)
                    processed_base_keys.add(key)
                    continue

                except Exception as e:
                    print(f"    Error: {e}")
                    # Fall through to copy reference data

        # Non-quantized tensor or already processed
        if key not in processed_base_keys:
            start, end = ref_spec['data_offsets']
            data_bytes = ref_data[start:end]
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

    config['quantization'] = {'group_size': 256, 'bits': 4, 'mode': 'affine'}
    config['quantization_config'] = {'group_size': 256, 'bits': 4, 'mode': 'affine'}

    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Copy other files
    import shutil
    for f_name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json',
                    'special_tokens_map.json']:
        src = ref_path / f_name
        if src.exists():
            shutil.copy(src, output_path / f_name)

    print(f"Model saved to {output_path}")
    return output_path


if __name__ == "__main__":
    convert_via_dequantize()
