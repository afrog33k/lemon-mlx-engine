#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-gfx1151}"
BENCH_BIN="$BUILD_DIR/bench"
PROMPT_FILE="benchmarks/prompts/qwen35_add_raw.py"
MAX_TOKENS=32
RUNS=2
TIMEOUT=120

echo "Testing QMV settings for tied lm-head optimization"
echo "===================================================="

# Read prompt
prompt="$(cat "$PROMPT_FILE")"
while [[ "$prompt" == *$'\n' ]]; do
    prompt="${prompt%$'\n'}"
done

settings=(
    "default"
    "cols_per_block=32"
    "cols_per_block=64"
    "cols_per_block=128"
)

for setting in "${settings[@]}"; do
    echo ""
    echo "Testing: $setting"
    echo "-------------------------"
    
    case "$setting" in
        default)
            env_vars=""
            ;;
        cols_per_block=*)
            cols="${setting#*=}"
            env_vars="MLX_ROCM_QMV_COLS_PER_BLOCK=$cols"
            ;;
        *)
            env_vars=""
            ;;
    esac
    
    # Run with timing profile
    for run in $(seq 1 $RUNS); do
        echo "  Run $run:"
        timeout "$TIMEOUT" env -u LEMON_MLX_DEQUANTIZE_WEIGHTS \
            LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 \
            LEMON_MLX_GDN_ENABLE_HIP=1 \
            ${env_vars:+$env_vars} \
            PROFILE_TIMING=1 \
            MAX_TOKENS=5 \
            "$BENCH_BIN" "mlx-community/Qwen3.5-0.8B-MLX-4bit" \
            --prompt "$prompt" \
            --raw \
            --max-tokens "$MAX_TOKENS" \
            --warmup-decode-steps 1 \
            --temperature 0 \
            --top-p 1 \
            2>&1 | grep -E "decode_tok_s|lm_head_tied_embedding" || true
    done
done
