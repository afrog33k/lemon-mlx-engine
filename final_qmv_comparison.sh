#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-gfx1151}"
BENCH_BIN="$BUILD_DIR/bench"
PROMPT_FILE="benchmarks/prompts/qwen35_add_raw.py"
RUNS=4
MAX_TOKENS=32

prompt="$(cat "$PROMPT_FILE")"
while [[ "$prompt" == *$'\n' ]]; do
    prompt="${prompt%$'\n'}"
done

echo "Final QMV comparison"
echo "===================="

declare -A results

settings=(
    "default: "
    "cols64: MLX_ROCM_QMV_COLS_PER_BLOCK=64"
    "tiled8: MLX_ROCM_QMV_ENABLE_TILED=1 MLX_ROCM_QMV_TILE_N=8"
)

for key_setting in "${settings[@]}"; do
    key="${key_setting%%:*}"
    env="${key_setting#*:}"
    
    echo ""
    echo "Testing: $key"
    
    total_speed=0
    valid_runs=0
    speeds=()
    
    for run in $(seq 1 $RUNS); do
        output=$(timeout 90 env -u LEMON_MLX_DEQUANTIZE_WEIGHTS \
            LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 \
            LEMON_MLX_GDN_ENABLE_HIP=1 \
            ${env:+$env} \
            "$BENCH_BIN" "mlx-community/Qwen3.5-0.8B-MLX-4bit" \
            --prompt "$prompt" \
            --raw \
            --max-tokens "$MAX_TOKENS" \
            --warmup-decode-steps 1 \
            --temperature 0 \
            --top-p 1 \
            2>&1)
        
        tok_s=$(echo "$output" | awk -F'\t' '$1 != "model" && NF >= 10 {print $8; found=1; exit} END {if (!found) print "0"}')
        
        # Use awk for floating point comparison
        is_valid=$(awk "BEGIN {print ($tok_s > 50) ? 1 : 0}")
        if [[ "$is_valid" -eq 1 ]]; then
            speeds+=("$tok_s")
            total_speed=$(awk "BEGIN {print $total_speed + $tok_s}")
            valid_runs=$((valid_runs + 1))
            echo "  Run $run: $tok_s tok/s"
        fi
    done
    
    if [[ $valid_runs -gt 0 ]]; then
        avg=$(awk "BEGIN {print $total_speed / $valid_runs}")
        min=$(printf "%s\n" "${speeds[@]}" | sort -n | head -1)
        max=$(printf "%s\n" "${speeds[@]}" | sort -n | tail -1)
        echo "  Average: $avg tok/s (range: $min - $max, $valid_runs runs)"
        results[$key]="$avg"
    fi
done

echo ""
echo "=== Summary (sorted by speed) ==="
for key in "default" "cols64" "tiled8"; do
    if [[ -n "${results[$key]:-}" ]]; then
        printf "%-10s %6.1f tok/s\n" "$key:" "${results[$key]}"
    fi
done | awk -F' ' '{print $2, $1}' | sort -rn | awk '{print $2, $1}'
