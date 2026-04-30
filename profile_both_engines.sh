#!/usr/bin/env bash
set -euo pipefail

# Profile both engines with timing enabled
BUILD_DIR="${BUILD_DIR:-build-gfx1151}"
BENCH_BIN="$BUILD_DIR/bench"
PROMPT_FILE="benchmarks/prompts/qwen35_add_raw.py"
HIPFIRE_ROOT="${HIPFIRE_ROOT:-/home/reckon/projects/amd-llm}"

prompt="$(cat "$PROMPT_FILE")"
while [[ "$prompt" == *$'\n' ]]; do
    prompt="${prompt%$'\n'}"
done

echo "Profiling lemon-mlx-engine with timing"
echo "======================================"

timeout 90 env -u LEMON_MLX_DEQUANTIZE_WEIGHTS \
    LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 \
    LEMON_MLX_GDN_ENABLE_HIP=1 \
    MLX_ROCM_QMV_COLS_PER_BLOCK=64 \
    MAX_TOKENS=5 \
    PROFILE_TIMING=1 \
    "$BENCH_BIN" "mlx-community/Qwen3.5-0.8B-MLX-4bit" \
    --prompt "$prompt" \
    --raw \
    --max-tokens 5 \
    --warmup-decode-steps 1 \
    --temperature 0 \
    --top-p 1 \
    2>&1 | tee /tmp/lemon_profile.log

echo ""
echo "======================================"
echo "Profiling hipfire with timing"

# Check if hipfire is available
if command -v hipfire >/dev/null 2>&1; then
    cd "$HIPFIRE_ROOT"
    timeout 90 hipfire bench qwen3.5:0.8b --runs 1 "$prompt" 2>&1 | tee /tmp/hipfire_profile.log || true
    cd -
else
    echo "hipfire not found in PATH"
fi

echo ""
echo "======================================"
echo "Lemon timing breakdown:"
grep "quant_profile" /tmp/lemon_profile.log 2>/dev/null || echo "No profile data found"

