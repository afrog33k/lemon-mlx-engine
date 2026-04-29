#!/usr/bin/env python3.12
"""
Re-quantize Qwen3.5-0.8B from group_size=64 to group_size=256.

This requires:
1. FP32 source model (Qwen/Qwen3.5-0.8B)
2. MLX with group_size=256 support (patched)
"""

# Set LD_LIBRARY_PATH BEFORE importing MLX
import os
os.environ['LD_LIBRARY_PATH'] = "/home/reckon/projects/mlx-vulkan/python/mlx/lib64:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['MLX_DEFAULT_DEVICE'] = 'cpu'

import sys
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import QuantizedLinear, QuantizedEmbedding
from mlx.utils import tree_flatten
import torch  # For bfloat16 support

def convert_model_to_group256(
    source_model_path: str,
    output_path: str,
):
    """Convert a model from group_size=64 to group_size=256."""

    print(f"Loading model from: {source_model_path}")

    # Load FP32 model using safetensors
    from safetensors import safe_open
    import numpy as np

    # Find all safetensors files
    source_path = Path(source_model_path)
    safetensors_files = list(source_path.glob("*.safetensors"))

    if not safetensors_files:
        # Check for sharded safetensors
        index_file = source_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            # Get unique filenames
            safetensors_files = sorted(set(
                source_path / Path(path).name
                for path in index.get("weight_map", {}).values()
            ))
            print(f"Found {len(safetensors_files)} sharded safetensors files")
        else:
            raise ValueError(f"No safetensors found in {source_path}")

    # Load all weights
    weights = {}
    for sf_file in safetensors_files:
        print(f"  Loading {sf_file.name}...")
        try:
            # Use torch backend to handle bfloat16
            from safetensors.torch import load_file as load_torch
            import torch
            torch_weights = load_torch(str(sf_file))
            for key, value in torch_weights.items():
                # Convert to MLX array via numpy
                if value.dtype == torch.bfloat16:
                    # Convert bfloat16 to float32 first
                    weights[key] = mx.array(value.float().numpy())
                else:
                    weights[key] = mx.array(value.numpy())
        except Exception as e:
            print(f"    Error loading {sf_file.name}: {e}")
            # Fallback to numpy (doesn't support bfloat16)
            try:
                from safetensors.numpy import load_file as load_numpy
                np_weights = load_numpy(str(sf_file))
                for key, value in np_weights.items():
                    weights[key] = mx.array(value)
            except Exception as e2:
                print(f"    Fallback also failed: {e2}")

    print(f"Loaded {len(weights)} tensors")

    # Now re-quantize Linear and Embedding layers with group_size=256
    print("\nRe-quantizing with group_size=256...")

    quantized_weights = {}
    quantized_config = {
        "quantization": {
            "group_size": 256,
            "bits": 4,
            "mode": "affine"
        }
    }

    # Group weights by layer
    from collections import defaultdict
    layer_weights = defaultdict(dict)

    for key, value in weights.items():
        # Extract layer name (e.g., "model.layers.0.mlp.gate_proj" from "model.layers.0.mlp.gate_proj.weight")
        if key.endswith(".weight"):
            base_key = key[:-7]  # Remove ".weight"
            layer_weights[base_key]["weight"] = value
        elif key.endswith(".bias"):
            base_key = key[:-5]
            layer_weights[base_key]["bias"] = value
        else:
            # Keep non-weight tensors as-is
            quantized_weights[key] = value

    # Quantize each layer
    for layer_name, layer_data in layer_weights.items():
        weight = layer_data["weight"]
        bias = layer_data.get("bias")

        # Skip 1D weights (LayerNorm, etc.) - keep them as-is
        if len(weight.shape) == 1:
            quantized_weights[f"{layer_name}.weight"] = weight
            if bias is not None:
                quantized_weights[f"{layer_name}.bias"] = bias
            continue

        # For conv/linear 2D+ weights
        is_embedding = "embed" in layer_name.lower() and len(weight.shape) == 2

        if is_embedding:
            # For embeddings, use mx.quantize directly
            vocab_size, embed_dim = weight.shape
            print(f"  {layer_name}: Embedding {vocab_size} x {embed_dim}")

            # Quantize using mx.quantize
            quantized_result = mx.quantize(weight, group_size=256, bits=4)
            packed_w, scales, biases = quantized_result

            quantized_weights[f"{layer_name}.weight"] = packed_w
            quantized_weights[f"{layer_name}.scales"] = scales
            quantized_weights[f"{layer_name}.biases"] = biases

        else:
            # For Linear layers - transpose to [out, in] format if needed
            if len(weight.shape) == 2:
                out_features, in_features = weight.shape
                has_bias = bias is not None
                print(f"  {layer_name}: Linear {in_features} -> {out_features}")

                # Use mx.quantize directly to get packed weights
                quantized_result = mx.quantize(weight, group_size=256, bits=4)
                packed_w, scales, biases = quantized_result

                quantized_weights[f"{layer_name}.weight"] = packed_w
                quantized_weights[f"{layer_name}.scales"] = scales
                quantized_weights[f"{layer_name}.biases"] = biases
                if has_bias:
                    quantized_weights[f"{layer_name}.bias"] = bias
            else:
                # Keep higher dimensional weights as-is
                quantized_weights[f"{layer_name}.weight"] = weight
                if bias is not None:
                    quantized_weights[f"{layer_name}.bias"] = bias

    # Add non-weight tensors
    for key, value in weights.items():
        if key not in quantized_weights:
            quantized_weights[key] = value

    print(f"\nTotal quantized tensors: {len(quantized_weights)}")

    # Save the quantized model
    print(f"\nSaving to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Save as safetensors
    from safetensors.numpy import save_file

    # Convert MLX arrays to numpy, and convert scales/biases to BF16
    numpy_weights = {}
    for key, value in quantized_weights.items():
        if isinstance(value, mx.array):
            arr = np.array(value)
            # Convert scales and biases to BF16 to match MLX format
            if key.endswith('.scales') or key.endswith('.biases'):
                # F32 -> BF16
                import numpy as np
                arr = arr.astype(np.float16)
            numpy_weights[key] = arr
        else:
            numpy_weights[key] = value

    output_file = os.path.join(output_path, "model.safetensors")
    save_file(numpy_weights, output_file)

    # Save config
    config_path = os.path.join(output_path, "config.json")
    if (source_path / "config.json").exists():
        with open(source_path / "config.json") as f:
            config = json.load(f)
        # Add quantization config
        config["quantization_config"] = quantized_config["quantization"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Save generation config
    if (source_path / "generation_config.json").exists():
        import shutil
        shutil.copy(source_path / "generation_config.json", output_path / "generation_config.json")

    print(f"Done! Model saved to {output_path}")
    return output_path


if __name__ == "__main__":
    source_model = "/home/reckon/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
    output_model = "/home/reckon/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-group256"

    convert_model_to_group256(source_model, output_model)
