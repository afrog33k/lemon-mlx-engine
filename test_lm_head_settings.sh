#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-gfx1151}"
BENCH_BIN="$BUILD_DIR/bench"
PROMPT_FILE="benchmarks/prompts/qwen35_add_raw.py"

# Read prompt
prompt="$(cat "$PROMPT_FILE")"
while [[ "$prompt" == *$'\n' ]]; do
    prompt="${prompt%$'\n'}"
done

echo "Testing QMV cols_per_block for tied lm-head optimization"
echo "=========================================================="

settings=(
    "default"
    "MLX_ROCM_QMV_COLS_PER_BLOCK=32"
    "MLX_ROCM_QMV_COLS_PER_BLOCK=64"
    "MLX_ROCM_QMV_COLS_PER_BLOCK=128"
)

for setting in "${settings[@]}"; do
    echo ""
    echo "Setting: $setting"
    echo "---------"
    
    # Parse the setting
    if [[ "$setting" == "default" ]]; then
        env_vars=""
    else
        env_vars="$setting"
    fi
    
    # Run a quick test
    for run in 1 2; do
        echo "  Run $run:"
        timeout 60 env -u LEMON_MLX_DEQUANTIZE_WEIGHTS \
            LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 \
            LEMON_MLX_GDN_ENABLE_HIP=1 \
            ${env_vars:+$env_vars} \
            MAX_TOKENS=8 \
            PROFILE_TIMING=1 \
            "$BENCH_BIN" "mlx-community/Qwen3.5-0.8B-MLX-4bit" \
            --prompt "$prompt" \
            --raw \
            --max-tokens 8 \
            --warmup-decode-steps 1 \
            --temperature 0 \
            --top-p 1 \
            2>&1 | grep -E "decode_tok_s|lm_head" || true
    done
done
