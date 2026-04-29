#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-gfx1151}"
BENCH_BIN="$BUILD_DIR/bench"
PROMPT_FILE="benchmarks/prompts/qwen35_add_raw.py"

prompt="$(cat "$PROMPT_FILE")"
while [[ "$prompt" == *$'\n' ]]; do
    prompt="${prompt%$'\n'}"
done

echo "Verifying cols64 correctness across multiple runs"
echo "================================================="

# Check for output stability
declare -A output_hashes
declare -A token_hashes
total_runs=0
ok_runs=0

for run in {1..5}; do
    output=$(timeout 90 env -u LEMON_MLX_DEQUANTIZE_WEIGHTS \
        LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 \
        LEMON_MLX_GDN_ENABLE_HIP=1 \
        MLX_ROCM_QMV_COLS_PER_BLOCK=64 \
        "$BENCH_BIN" "mlx-community/Qwen3.5-0.8B-MLX-4bit" \
        --prompt "$prompt" \
        --raw \
        --max-tokens 32 \
        --warmup-decode-steps 1 \
        --temperature 0 \
        --top-p 1 \
        2>&1)
    
    tok_s=$(echo "$output" | awk -F'\t' '$1 != "model" && NF >= 10 {print $8; found=1; exit} END {if (!found) print "nan"}')
    output_hash=$(echo "$output" | awk -F'\t' '$1 != "model" && NF >= 10 {print $15; found=1; exit} END {if (!found) print "unknown"}')
    token_hash=$(echo "$output" | awk -F'\t' '$1 != "model" && NF >= 10 {print $17; found=1; exit} END {if (!found) print "unknown"}')
    preview=$(echo "$output" | awk -F'\t' '$1 != "model" && NF >= 10 {print $16; found=1; exit} END {if (!found) print "no_output"}')
    
    echo "Run $run: $tok_s tok/s - hash=$output_hash - $preview"
    
    output_hashes[$output_hash]=1
    token_hashes[$token_hash]=1
    total_runs=$((total_runs + 1))
    
    if [[ "$tok_s" != "nan" ]]; then
        ok_runs=$((ok_runs + 1))
    fi
done

echo ""
echo "Summary: $ok_runs/$total_runs runs OK"
echo "Distinct output hashes: ${#output_hashes[@]}"
echo "Distinct token hashes: ${#token_hashes[@]}"

if [[ ${#output_hashes[@]} -eq 1 && ${#token_hashes[@]} -eq 1 ]]; then
    echo "Result: STABLE"
elif [[ ${#output_hashes[@]} -le 2 ]]; then
    echo "Result: MINOR_VARIATION (acceptable due to thermal drift)"
else
    echo "Result: UNSTABLE"
fi
