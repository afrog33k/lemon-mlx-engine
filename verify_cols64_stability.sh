#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-gfx1151}"
BENCH_BIN="$BUILD_DIR/bench"
PROMPT_FILE="benchmarks/prompts/qwen35_add_raw.py"
RUNS=5
MAX_TOKENS=32

prompt="$(cat "$PROMPT_FILE")"
while [[ "$prompt" == *$'\n' ]]; do
    prompt="${prompt%$'\n'}"
done

echo "Verifying MLX_ROCM_QMV_COLS_PER_BLOCK=64 stability"
echo "==================================================="

total_speed=0
valid_runs=0

for run in $(seq 1 $RUNS); do
    result=$(timeout 90 env -u LEMON_MLX_DEQUANTIZE_WEIGHTS \
        LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 \
        LEMON_MLX_GDN_ENABLE_HIP=1 \
        MLX_ROCM_QMV_COLS_PER_BLOCK=64 \
        "$BENCH_BIN" "mlx-community/Qwen3.5-0.8B-MLX-4bit" \
        --prompt "$prompt" \
        --raw \
        --max-tokens "$MAX_TOKENS" \
        --warmup-decode-steps 1 \
        --temperature 0 \
        --top-p 1 \
        2>&1 | awk '$1 != "model" && NF >= 10 {print $8; found=1; exit} END {if (!found) print "nan"}')
    
    if [[ "$result" != "nan" ]]; then
        echo "  Run $run: $result tok/s"
        total_speed=$(awk "BEGIN {print $total_speed + $result}")
        valid_runs=$((valid_runs + 1))
    fi
done

if [[ $valid_runs -gt 0 ]]; then
    avg=$(awk "BEGIN {print $total_speed / $valid_runs}")
    echo ""
    echo "Average: $avg tok/s ($valid_runs runs)"
fi
