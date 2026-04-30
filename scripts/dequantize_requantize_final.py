#!/usr/bin/env python3.12
"""
Properly handle BF16 conversion for safetensors storage.
BF16 uses the upper 16 bits of float32 representation.
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

def float32_to_bf16_array(f32_array):
    """Convert float32 numpy array to BF16 uint16 array."""
    # BF16 is the upper 16 bits of float32
    # View as uint32, then shift right 16 bits
    u32_view = f32_array.view(np.uint32)
    bf16_view = (u32_view >> 16).astype(np.uint16)
    return bf16_view

def bf16_to_float32_array(bf16_array):
    """Convert BF16 uint16 array to float32 array."""
    # BF16 is the upper 16 bits of float32
    # Shift left 16 bits to reconstruct
    u32_view = (bf16_array.astype(np.uint32) << 16).astype(np.uint32)
    f32_view = u32_view.view(np.float32)
    return f32_view

def convert_via_dequantize():
    """Dequantize MLX-community model, then re-quantize with group_size=256."""

    ref_path = Path("/home/reckon/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-4bit/snapshots/main")
    output_path = Path("/home/reckon/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-group256-final")
    output_path.mkdir(exist_ok=True, parents=True)

    # Load model using MLX (handles all formats correctly)
    print("Loading MLX-community model...")
    ref_model = mx.load(str(ref_path / "model.safetensors"))
    print(f"Loaded {len(ref_model)} tensors")

    # Process quantized layers
    new_model = {}
    processed_base_keys = set()

    # Key patterns for quantized layers
    quantized_patterns = [
        'embed_tokens', 'gate_proj', 'down_proj', 'up_proj',
        'in_proj', 'out_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj',
        'linear_attn', 'self_attn', 'mlp'
    ]

    for key in ref_model.keys():
        # Check if this is a quantized weight
        if key.endswith('.weight'):
            base_key = key[:-7]

            # Check if it has corresponding scales/biases
            scales_key = base_key + '.scales'
            biases_key = base_key + '.biases'

            if scales_key in ref_model and biases_key in ref_model:
                # This is a quantized layer
                if any(pattern in base_key for pattern in quantized_patterns):
                    if base_key in processed_base_keys:
                        continue

                    print(f"  Converting {base_key}")

                    # Get the FP32 weight by dequantizing
                    packed = ref_model[key]
                    scales = ref_model[scales_key]
                    biases = ref_model[biases_key]

                    # Dequantize with group_size=64
                    w_fp32 = mx.dequantize(packed, scales, biases, group_size=64, bits=4)

                    # Re-quantize with group_size=256
                    q256 = mx.quantize(w_fp32, group_size=256, bits=4)
                    packed256, scales256, biases256 = q256

                    # Store in new model
                    new_model[key] = packed256
                    new_model[scales_key] = scales256
                    new_model[biases_key] = biases256

                    processed_base_keys.add(base_key)
                    processed_base_keys.add(scales_key)
                    processed_base_keys.add(biases_key)
                    processed_base_keys.add(key)
                    continue

        # Copy non-quantized tensors
        if key not in processed_base_keys:
            new_model[key] = ref_model[key]

    print(f"Processed model has {len(new_model)} tensors")

    # Save using MLX's save function (handles all formats correctly)
    output_file = output_path / "model.safetensors"
    print(f"Saving to {output_file}...")
    mx.save_safetensors(str(output_file), new_model)
    print("Saved successfully")

    # Copy and update config
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

    print(f"\nModel saved to {output_path}")
    return output_path


if __name__ == "__main__":
    convert_via_dequantize()
