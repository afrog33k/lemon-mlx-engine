#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-gfx1151}"
BENCH_BIN="$BUILD_DIR/bench"
PROMPT_FILE="benchmarks/prompts/qwen35_add_raw.py"

prompt="$(cat "$PROMPT_FILE")"
while [[ "$prompt" == *$'\n' ]]; do
    prompt="${prompt%$'\n'}"
done

echo "Testing tiled QMV (TILE_N=8) correctness"
echo "=========================================="

for run in 1 2 3; do
    echo "Run $run:"
    output=$(timeout 60 env -u LEMON_MLX_DEQUANTIZE_WEIGHTS \
        LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 \
        LEMON_MLX_GDN_ENABLE_HIP=1 \
        MLX_ROCM_QMV_ENABLE_TILED=1 \
        MLX_ROCM_QMV_TILE_N=8 \
        "$BENCH_BIN" "mlx-community/Qwen3.5-0.8B-MLX-4bit" \
        --prompt "$prompt" \
        --raw \
        --max-tokens 32 \
        --warmup-decode-steps 1 \
        --temperature 0 \
        --top-p 1 \
        --expect-substring "return x + y" \
        --max-token-freq 0.40 \
        --max-token-run 16 \
        --attractor-min-tokens 16 \
        2>&1)
    
    tok_s=$(echo "$output" | awk '$1 != "model" && NF >= 10 {print $8; found=1; exit} END {if (!found) print "nan"}')
    status=$(echo "$output" | awk '$1 != "model" && NF >= 10 {print $16; found=1; exit} END {if (!found) print "fail"}')
    output_preview=$(echo "$output" | awk '$1 != "model" && NF >= 10 {print $17; found=1; exit} END {if (!found) print "no_output"}')
    
    echo "  Speed: $tok_s tok/s"
    echo "  Status: $status"
    echo "  Output: $output_preview"
    echo ""
done
