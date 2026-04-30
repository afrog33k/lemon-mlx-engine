#!/usr/bin/env python3.12
"""
Systematically test TILE_N values for QMV kernel optimization.

The TILE_N parameter controls how many output columns are processed per block.
Higher values can improve L2 cache reuse but may reduce occupancy.
"""
import subprocess
import os
import sys

# Model path
MODEL = "/home/reckon/.cache/huggingface/hub/Qwen3.5-0.8B-MLX-4bit-group256-final"
BENCH = "/home/reckon/projects/lemon-mlx-engine/build-gfx1151/bench"
PROMPT = "Hello world, this is a test."

# Test different TILE_N values
TILE_N_VALUES = [4, 8, 12, 16, 20, 24, 28, 32]

print(f"{'TILE_N':<10} {'Decode (tok/s)':<20} {'Speedup':<10}")
print("-" * 40)

baseline = None

for tile_n in TILE_N_VALUES:
    os.environ['MLX_ROCM_QMV_TILE_N'] = str(tile_n)
    
    result = subprocess.run(
        [BENCH, MODEL, "--prompt", PROMPT, "--max-tokens", "64", "--runs", "2", "--raw"],
        capture_output=True,
        text=True
    )
    
    # Parse output for decode speed
    decode_speed = None
    for line in result.stdout.split('\n'):
        if '\t' in line:
            parts = line.split('\t')
            if len(parts) >= 8:
                try:
                    decode_speed = float(parts[7])
                    break
                except ValueError:
                    pass
    
    if decode_speed:
        if baseline is None:
            baseline = decode_speed
            speedup = 1.0
        else:
            speedup = decode_speed / baseline
        
        print(f"{tile_n:<10} {decode_speed:<20.2f} {speedup:<10.2%}")
    else:
        print(f"{tile_n:<10} ERROR")

print(f"\nBest TILE_N saved to MLX_ROCM_QMV_TILE_N environment variable")
