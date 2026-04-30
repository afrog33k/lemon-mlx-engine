#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-gfx1151}"
BENCH_BIN="$BUILD_DIR/bench"
PROMPT_FILE="benchmarks/prompts/qwen35_add_raw.py"
RUNS=3
MAX_TOKENS=32

prompt="$(cat "$PROMPT_FILE")"
while [[ "$prompt" == *$'\n' ]]; do
    prompt="${prompt%$'\n'}"
done

echo "Comprehensive QMV comparison"
echo "=============================="

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
    echo "---------"
    
    total_speed=0
    valid_runs=0
    all_ok=1
    
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
            --expect-substring "return x + y" \
            --max-token-freq 0.40 \
            --max-token-run 16 \
            --attractor-min-tokens 16 \
            2>&1)
        
        tok_s=$(echo "$output" | awk '$1 != "model" && NF >= 10 {print $8; found=1; exit} END {if (!found) print "nan"}')
        status=$(echo "$output" | awk '$1 != "model" && NF >= 10 {print $16; found=1; exit} END {if (!found) print "unknown"}')
        output_preview=$(echo "$output" | awk '$1 != "model" && NF >= 10 {print $17; found=1; exit} END {if (!found) print "no_output"}')
        
        if [[ "$tok_s" != "nan" && "$status" == "ok" ]]; then
            echo "  Run $run: $tok_s tok/s - $output_preview"
            total_speed=$(awk "BEGIN {print $total_speed + $tok_s}")
            valid_runs=$((valid_runs + 1))
        else
            echo "  Run $run: FAILED (status=$status)"
            all_ok=0
        fi
    done
    
    if [[ $valid_runs -gt 0 ]]; then
        avg=$(awk "BEGIN {print $total_speed / $valid_runs}")
        echo "  Average: $avg tok/s ($valid_runs runs)"
        if [[ $all_ok -eq 1 ]]; then
            echo "  All runs: OK"
        fi
        results[$key]="$avg|$valid_runs|$all_ok"
    fi
done

echo ""
echo "=== Summary ==="
for key in "default" "cols64" "tiled8"; do
    if [[ -n "${results[$key]:-}" ]]; then
        IFS='|' read -r avg runs ok <<< "${results[$key]}"
        status_str=$([ "$ok" -eq 1 ] && echo "OK" || echo "MIXED")
        printf "%-10s %6.1f tok/s  (%d runs, %s)\n" "$key:" "$avg" "$runs" "$status_str"
    fi
done
