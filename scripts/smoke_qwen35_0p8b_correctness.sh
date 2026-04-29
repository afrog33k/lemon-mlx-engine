#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-gfx1151}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bench}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/benchmarks/prompts/qwen35_add_raw.py}"
OUT_BASE="${OUT_BASE:-$ROOT_DIR/.codeinsight+research}"
RUNS="${RUNS:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
MAX_TOKENS="${MAX_TOKENS:-32}"
DECODE_WARMUP_STEPS="${DECODE_WARMUP_STEPS:-1}"
JOBS="${JOBS:-8}"
RUN_NATIVE_QUANTIZED="${RUN_NATIVE_QUANTIZED:-1}"
REQUIRE_NATIVE_DEQUANT_HASH_MATCH="${REQUIRE_NATIVE_DEQUANT_HASH_MATCH:-1}"

QWEN35_BF16_MODEL="${QWEN35_BF16_MODEL:-Qwen/Qwen3.5-0.8B}"
QWEN35_4BIT_MODEL="${QWEN35_4BIT_MODEL:-mlx-community/Qwen3.5-0.8B-MLX-4bit}"
EXPECT_SUBSTRING="${EXPECT_SUBSTRING:-return x + y}"
EXPECT_PREFIX="${EXPECT_PREFIX:-    return x + y}"
STOP_AFTER_SUBSTRING="${STOP_AFTER_SUBSTRING:-}"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "missing build directory: $BUILD_DIR" >&2
    echo "configure a gfx1151 build first, or set BUILD_DIR" >&2
    exit 1
fi

cmake --build "$BUILD_DIR" --target bench -j "$JOBS"

if [[ ! -x "$BENCH_BIN" ]]; then
    echo "missing bench binary: $BENCH_BIN" >&2
    exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "missing prompt file: $PROMPT_FILE" >&2
    exit 1
fi

if [[ "${ALLOW_EXPERIMENTAL_HIP_GDN:-0}" != "1" ]]; then
    export LEMON_MLX_GDN_DISABLE_HIP=1
fi
if [[ "${ALLOW_UNSTABLE_TILED_QMV:-0}" != "1" ]]; then
    unset MLX_ROCM_QMV_ENABLE_TILED
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/qwen35_0p8b_correctness_$stamp"
mkdir -p "$OUT_DIR"

prompt_md5="$(md5sum "$PROMPT_FILE" | awk '{print $1}')"
bench_md5="$(md5sum "$BENCH_BIN" | awk '{print $1}')"

cat > "$OUT_DIR/metadata.txt" <<EOF_META
date_utc=$stamp
build_dir=$BUILD_DIR
bench_bin=$BENCH_BIN
bench_md5=$bench_md5
prompt_file=$PROMPT_FILE
prompt_md5=$prompt_md5
runs=$RUNS
timeout_seconds=$TIMEOUT_SECONDS
max_tokens=$MAX_TOKENS
decode_warmup_steps=$DECODE_WARMUP_STEPS
expect_substring=$EXPECT_SUBSTRING
expect_prefix=$EXPECT_PREFIX
stop_after_substring=$STOP_AFTER_SUBSTRING
qwen35_bf16_model=$QWEN35_BF16_MODEL
qwen35_4bit_model=$QWEN35_4BIT_MODEL
lemon_mlx_gdn_enable_hip=${LEMON_MLX_GDN_ENABLE_HIP:-}
lemon_mlx_gdn_disable_hip=${LEMON_MLX_GDN_DISABLE_HIP:-}
mlx_rocm_qmv_enable_tiled=${MLX_ROCM_QMV_ENABLE_TILED:-}
allow_unstable_tiled_qmv=${ALLOW_UNSTABLE_TILED_QMV:-0}
lemon_mlx_dequantize_weights=${LEMON_MLX_DEQUANTIZE_WEIGHTS:-}
lemon_mlx_qwen35_keep_quantized=${LEMON_MLX_QWEN35_KEEP_QUANTIZED:-}
run_native_quantized=$RUN_NATIVE_QUANTIZED
require_native_dequant_hash_match=$REQUIRE_NATIVE_DEQUANT_HASH_MATCH
EOF_META

{
    echo "[free -h]"
    free -h || true
    echo
    echo "[rocm-smi --showmeminfo vram]"
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showmeminfo vram || true
    else
        echo "rocm-smi not found"
    fi
} > "$OUT_DIR/system_before.txt"

summary="$OUT_DIR/summary.tsv"
printf 'case\tmodel\tquant_mode\tprompt_md5\tbench_md5\trun\texit_code\tstatus\tdecode_tok_s\tdecode_tokens\toutput_hash\ttoken_hash\tlog\n' > "$summary"

run_case() {
    local name="$1"
    local model="$2"
    local quant_mode="${3:-default}"
    local run
    local case_failed=0
    for ((run = 1; run <= RUNS; run += 1)); do
        local log="$OUT_DIR/${name}_run${run}.log"
        local stop_args=()
        if [[ -n "$STOP_AFTER_SUBSTRING" ]]; then
            stop_args=(--stop-after-substring "$STOP_AFTER_SUBSTRING")
        fi
        local expect_prefix_args=()
        if [[ -n "$EXPECT_PREFIX" ]]; then
            expect_prefix_args=(--expect-prefix "$EXPECT_PREFIX")
        fi
        local env_args=()
        case "$quant_mode" in
native_quantized)
                env_args=(env -u LEMON_MLX_DEQUANTIZE_WEIGHTS LEMON_MLX_QWEN35_KEEP_QUANTIZED=1)
                ;;
            dequantized)
                env_args=(env -u LEMON_MLX_QWEN35_KEEP_QUANTIZED LEMON_MLX_DEQUANTIZE_WEIGHTS=1)
                ;;
            default)
                env_args=()
                ;;
            *)
                echo "unknown quant_mode: $quant_mode" >&2
                return 1
                ;;
        esac
        set +e
        timeout "$TIMEOUT_SECONDS" "${env_args[@]}" "$BENCH_BIN" "$model" \
            --prompt-file "$PROMPT_FILE" \
            --raw \
            --max-tokens "$MAX_TOKENS" \
            --warmup-decode-steps "$DECODE_WARMUP_STEPS" \
            --temperature 0 \
            --top-p 1 \
            --expect-substring "$EXPECT_SUBSTRING" \
            "${expect_prefix_args[@]}" \
            "${stop_args[@]}" \
            --fail-on-attractor \
            --print-output \
            > "$log" 2>&1
        local rc=$?
        set -e

        local parsed
        parsed="$(awk -F '\t' '$1 != "model" && NF >= 16 {print $15 "\t" $8 "\t" $6 "\t" $14 "\t" (NF >= 17 ? $17 : "0000000000000000"); found=1; exit} END {if (!found) print "no_tsv\tnan\t0\t0000000000000000\t0000000000000000"}' "$log")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$name" "$model" "$quant_mode" "$prompt_md5" "$bench_md5" "$run" "$rc" "$parsed" "$log" >> "$summary"

        if [[ "$rc" -ne 0 ]]; then
            echo "$name run $run failed with exit code $rc; see $log" >&2
            case_failed=1
        fi
    done
    return "$case_failed"
}

any_failed=0
run_case "qwen35_0p8b_bf16" "$QWEN35_BF16_MODEL" "default" || any_failed=1
run_case "qwen35_0p8b_4bit_dequantized" "$QWEN35_4BIT_MODEL" "dequantized" || any_failed=1
if [[ "$RUN_NATIVE_QUANTIZED" != "0" ]]; then
    run_case "qwen35_0p8b_4bit_native" "$QWEN35_4BIT_MODEL" "native_quantized" || any_failed=1
fi

aggregate="$OUT_DIR/aggregate.tsv"
awk -F '\t' '
NR == 1 { next }
{
    key = $1 "\t" $2 "\t" $3
    cases[key] = 1
    runs[key] += 1
    if ($7 == 0 && $8 == "ok") ok[key] += 1
    if ($9 != "nan") {
        speed_count[key] += 1
        speed[key, speed_count[key]] = $9 + 0
    }
    tokens[key] += $10
    hashes[key, $11] = 1
    token_hashes[key, $12] = 1
}
END {
    print "case\tmodel\tquant_mode\truns\tok_runs\tmedian_decode_tok_s\tavg_decode_tokens\tdistinct_output_hashes\tdistinct_token_hashes"
    for (key in cases) {
        n = speed_count[key]
        median = "nan"
        if (n > 0) {
            for (i = 1; i <= n; ++i) {
                for (j = i + 1; j <= n; ++j) {
                    if (speed[key, j] < speed[key, i]) {
                        tmp = speed[key, i]
                        speed[key, i] = speed[key, j]
                        speed[key, j] = tmp
                    }
                }
            }
            if (n % 2) {
                median = speed[key, int(n / 2) + 1]
            } else {
                median = (speed[key, n / 2] + speed[key, n / 2 + 1]) / 2
            }
        }
        distinct = 0
        for (hk in hashes) {
            split(hk, parts, SUBSEP)
            if (parts[1] == key) distinct += 1
        }
        distinct_tokens = 0
        for (hk in token_hashes) {
            split(hk, parts, SUBSEP)
            if (parts[1] == key) distinct_tokens += 1
        }
        avg_tokens = runs[key] ? tokens[key] / runs[key] : 0
        split(key, fields, "\t")
        printf "%s\t%s\t%s\t%d\t%d\t%s\t%.2f\t%d\t%d\n",
            fields[1], fields[2], fields[3], runs[key], ok[key] + 0,
            median, avg_tokens, distinct, distinct_tokens
    }
}' "$summary" > "$aggregate"

if [[ "$RUN_NATIVE_QUANTIZED" != "0" && "$REQUIRE_NATIVE_DEQUANT_HASH_MATCH" != "0" ]]; then
    if ! awk -F '\t' '
        NR == 1 { next }
        $1 == "qwen35_0p8b_4bit_dequantized" && $7 == 0 && $8 == "ok" {
            dequant[$6] = $11
            dequant_tokens[$6] = $12
        }
        $1 == "qwen35_0p8b_4bit_native" && $7 == 0 && $8 == "ok" {
            native[$6] = $11
            native_tokens[$6] = $12
        }
        END {
            checked = 0
            for (run in native) {
                if (!(run in dequant)) {
                    printf "missing dequantized output hash for run %s\n", run > "/dev/stderr"
                    exit 1
                }
                checked = 1
                if (native[run] != dequant[run]) {
                    printf "native/dequantized output hash mismatch on run %s: native=%s dequantized=%s\n",
                        run, native[run], dequant[run] > "/dev/stderr"
                    exit 1
                }
                if (native_tokens[run] != dequant_tokens[run]) {
                    printf "native/dequantized token hash mismatch on run %s: native=%s dequantized=%s\n",
                        run, native_tokens[run], dequant_tokens[run] > "/dev/stderr"
                    exit 1
                }
            }
            if (!checked) {
                print "no successful native/dequantized hash pairs found" > "/dev/stderr"
                exit 1
            }
        }
    ' "$summary"; then
        any_failed=1
    fi
fi

{
    echo "[free -h]"
    free -h || true
    echo
    echo "[rocm-smi --showmeminfo vram]"
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showmeminfo vram || true
    else
        echo "rocm-smi not found"
    fi
} > "$OUT_DIR/system_after.txt"

echo "wrote $summary"
echo "wrote $aggregate"
exit "$any_failed"
