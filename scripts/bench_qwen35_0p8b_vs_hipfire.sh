#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-gfx1151}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bench}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/benchmarks/prompts/qwen35_add_raw.py}"
OUT_BASE="${OUT_BASE:-$ROOT_DIR/.codeinsight+research}"
HIPFIRE_ROOT="${HIPFIRE_ROOT:-/home/reckon/projects/amd-llm}"

RUNS="${RUNS:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
MAX_TOKENS_WAS_SET="${MAX_TOKENS+x}"
MAX_TOKENS="${MAX_TOKENS:-56}"
DECODE_WARMUP_STEPS="${DECODE_WARMUP_STEPS:-1}"
if [[ -z "${DECODE_TOKEN_BUDGETS+x}" ]]; then
    if [[ -z "$MAX_TOKENS_WAS_SET" && "$MAX_TOKENS" == "56" ]]; then
        # Hipfire can emit a few extra tokens around stop/EOT handling while
        # still running the same requested decode budget. Treat that as a valid
        # perf row, but keep the window bounded so early-stop failures still
        # surface.
        DECODE_TOKEN_BUDGETS="52,53,54,55,56,57,58,59,60"
    else
        DECODE_TOKEN_BUDGETS="$(python3 - "$MAX_TOKENS" <<'PY'
import sys
max_tokens = int(sys.argv[1])
lo = max(1, max_tokens - 4)
hi = max_tokens + 4
print(",".join(str(v) for v in range(lo, hi + 1)))
PY
)"
    fi
fi
JOBS="${JOBS:-8}"

LEMON_4BIT_MODEL="${LEMON_4BIT_MODEL:-mlx-community/Qwen3.5-0.8B-MLX-4bit}"
HIPFIRE_MODEL="${HIPFIRE_MODEL:-qwen3.5:0.8b}"
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3.5-0.8B}"
EXPECT_SUBSTRING="${EXPECT_SUBSTRING:-return x + y}"
EXPECT_PREFIX="${EXPECT_PREFIX:-}"
ATTRACTOR_MIN_TOKENS="${ATTRACTOR_MIN_TOKENS:-16}"
MAX_TOKEN_FREQ="${MAX_TOKEN_FREQ:-0.40}"
MAX_TOKEN_RUN="${MAX_TOKEN_RUN:-16}"
MIN_UNIQUE_TOKEN_RATIO="${MIN_UNIQUE_TOKEN_RATIO:-}"
REQUIRE_STABLE_LEMON_HASH="${REQUIRE_STABLE_LEMON_HASH:-1}"
REQUIRE_TOKEN_BUDGET_MATCH="${REQUIRE_TOKEN_BUDGET_MATCH:-1}"
REQUIRE_EXACT_ARTIFACT_MATCH="${REQUIRE_EXACT_ARTIFACT_MATCH:-0}"

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

declare -a HIPFIRE_CMD
if [[ -n "${HIPFIRE_BIN:-}" ]]; then
    HIPFIRE_CMD=("$HIPFIRE_BIN")
elif command -v hipfire >/dev/null 2>&1; then
    HIPFIRE_CMD=(hipfire)
elif command -v bun >/dev/null 2>&1 && [[ -f "$HIPFIRE_ROOT/cli/index.ts" ]]; then
    HIPFIRE_CMD=(bun "$HIPFIRE_ROOT/cli/index.ts")
else
    echo "missing hipfire command; install hipfire, set HIPFIRE_BIN, or set HIPFIRE_ROOT" >&2
    exit 1
fi

if [[ "${ALLOW_EXPERIMENTAL_HIP_GDN:-0}" != "1" ]]; then
    export LEMON_MLX_GDN_DISABLE_HIP=1
fi
if [[ "${ALLOW_UNSTABLE_TILED_QMV:-0}" != "1" ]]; then
    unset MLX_ROCM_QMV_ENABLE_TILED
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/qwen35_0p8b_vs_hipfire_$stamp"
mkdir -p "$OUT_DIR"

artifact_audit="$OUT_DIR/artifacts.tsv"
if [[ -x "$ROOT_DIR/scripts/audit_model_artifacts.py" || -f "$ROOT_DIR/scripts/audit_model_artifacts.py" ]]; then
    python3 "$ROOT_DIR/scripts/audit_model_artifacts.py" \
        --source "$SOURCE_MODEL" \
        --lemon "$LEMON_4BIT_MODEL" \
        --hipfire "$HIPFIRE_MODEL" \
        > "$artifact_audit"
else
    echo "missing artifact audit script: $ROOT_DIR/scripts/audit_model_artifacts.py" >&2
    exit 1
fi
exact_artifact_match="$(awk -F '\t' '$1 == "exact_artifact_match" {print $2; found=1; exit} END {if (!found) print "unknown"}' "$artifact_audit")"

prompt_arg_file="$OUT_DIR/prompt_arg.txt"
python3 - "$PROMPT_FILE" "$prompt_arg_file" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1]).read_bytes()
# hipfire bench accepts the prompt as a CLI argument, so use the exact bytes
# that shell argv can preserve and feed those same bytes to lemon via --prompt.
while src.endswith(b"\n"):
    src = src[:-1]
pathlib.Path(sys.argv[2]).write_bytes(src)
PY

prompt="$(<"$prompt_arg_file")"
prompt_file_md5="$(md5sum "$PROMPT_FILE" | awk '{print $1}')"
prompt_arg_md5="$(md5sum "$prompt_arg_file" | awk '{print $1}')"
bench_md5="$(md5sum "$BENCH_BIN" | awk '{print $1}')"
hipfire_ref="${HIPFIRE_CMD[*]}"
if [[ -f "$HIPFIRE_ROOT/cli/index.ts" ]]; then
    hipfire_ref_md5="$(md5sum "$HIPFIRE_ROOT/cli/index.ts" | awk '{print $1}')"
else
    hipfire_ref_md5="unknown"
fi

cat > "$OUT_DIR/metadata.txt" <<EOF_META
date_utc=$stamp
build_dir=$BUILD_DIR
bench_bin=$BENCH_BIN
bench_md5=$bench_md5
prompt_file=$PROMPT_FILE
prompt_file_md5=$prompt_file_md5
prompt_arg_file=$prompt_arg_file
prompt_arg_md5=$prompt_arg_md5
runs=$RUNS
timeout_seconds=$TIMEOUT_SECONDS
max_tokens=$MAX_TOKENS
decode_warmup_steps=$DECODE_WARMUP_STEPS
decode_token_budgets=$DECODE_TOKEN_BUDGETS
lemon_4bit_model=$LEMON_4BIT_MODEL
hipfire_model=$HIPFIRE_MODEL
source_model=$SOURCE_MODEL
artifact_audit=$artifact_audit
exact_artifact_match=$exact_artifact_match
hipfire_cmd=$hipfire_ref
hipfire_ref_md5=$hipfire_ref_md5
expect_prefix=$EXPECT_PREFIX
expect_substring=$EXPECT_SUBSTRING
attractor_min_tokens=$ATTRACTOR_MIN_TOKENS
max_token_freq=$MAX_TOKEN_FREQ
max_token_run=$MAX_TOKEN_RUN
min_unique_token_ratio=$MIN_UNIQUE_TOKEN_RATIO
require_stable_lemon_hash=$REQUIRE_STABLE_LEMON_HASH
require_token_budget_match=$REQUIRE_TOKEN_BUDGET_MATCH
require_exact_artifact_match=$REQUIRE_EXACT_ARTIFACT_MATCH
lemon_mlx_gdn_enable_hip=${LEMON_MLX_GDN_ENABLE_HIP:-}
lemon_mlx_gdn_disable_hip=${LEMON_MLX_GDN_DISABLE_HIP:-}
mlx_rocm_qmv_enable_tiled=${MLX_ROCM_QMV_ENABLE_TILED:-}
allow_unstable_tiled_qmv=${ALLOW_UNSTABLE_TILED_QMV:-0}
lemon_native_dequantize_weights=unset
lemon_mlx_qwen35_keep_quantized=1
fresh_process_per_run=1
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
    echo
    echo "[possible gpu users]"
    ps -eo pid,comm,args | awk 'NR == 1 || /llama-server|vllm|sglang|text-generation-launcher|ollama|mlx|target\/release\/examples\/daemon|target\/release\/examples\/bench/ { print }' || true
} > "$OUT_DIR/system_before.txt"

summary="$OUT_DIR/summary.tsv"
printf 'engine\tmodel\tprompt_arg_md5\tbinary_md5\trun\texit_code\tstatus\tdecode_tok_s\tdecode_tokens\toutput_hash\ttoken_hash\tlog\n' > "$summary"

run_lemon_native() {
    local run="$1"
    local log="$OUT_DIR/lemon_native_run${run}.log"
    local prefix_args=()
    local unique_args=()
    if [[ -n "$EXPECT_PREFIX" ]]; then
        prefix_args=(--expect-prefix "$EXPECT_PREFIX")
    fi
    if [[ -n "$MIN_UNIQUE_TOKEN_RATIO" ]]; then
        unique_args=(--min-unique-token-ratio "$MIN_UNIQUE_TOKEN_RATIO")
    fi
    set +e
    timeout "$TIMEOUT_SECONDS" env -u LEMON_MLX_DEQUANTIZE_WEIGHTS LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 "$BENCH_BIN" "$LEMON_4BIT_MODEL" \
        --prompt "$prompt" \
        --raw \
        --max-tokens "$MAX_TOKENS" \
        --warmup-decode-steps "$DECODE_WARMUP_STEPS" \
        --temperature 0 \
        --top-p 1 \
        --expect-substring "$EXPECT_SUBSTRING" \
        "${prefix_args[@]}" \
        --max-token-freq "$MAX_TOKEN_FREQ" \
        --max-token-run "$MAX_TOKEN_RUN" \
        --attractor-min-tokens "$ATTRACTOR_MIN_TOKENS" \
        "${unique_args[@]}" \
        --print-output \
        > "$log" 2>&1
    local rc=$?
    set -e

    local parsed
    parsed="$(awk -F '\t' '$1 != "model" && NF >= 16 {print $15 "\t" $8 "\t" $6 "\t" $14 "\t" (NF >= 17 ? $17 : "0000000000000000"); found=1; exit} END {if (!found) print "no_tsv\tnan\t0\t0000000000000000\t0000000000000000"}' "$log")"
    printf 'lemon\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$LEMON_4BIT_MODEL" "$prompt_arg_md5" "$bench_md5" "$run" "$rc" "$parsed" "$log" >> "$summary"
    return "$rc"
}

run_hipfire() {
    local run="$1"
    local log="$OUT_DIR/hipfire_run${run}.log"
    set +e
    timeout "$TIMEOUT_SECONDS" "${HIPFIRE_CMD[@]}" bench "$HIPFIRE_MODEL" --runs 1 "$prompt" > "$log" 2>&1
    local rc=$?
    set -e

    local decode tokens status
    decode="$(awk '$1 == "Decode" && $2 == "tok/s" {print $3; found=1; exit} END {if (!found) print "nan"}' "$log")"
    tokens="$(awk 'match($0, /decode [0-9.]+ tok\/s \(([0-9]+) tok\)/, m) {print m[1]; found=1; exit} END {if (!found) print 0}' "$log")"
    status="ok"
    if [[ "$rc" -ne 0 || "$decode" == "nan" ]]; then
        status="fail"
    fi
    printf 'hipfire\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$HIPFIRE_MODEL" "$prompt_arg_md5" "$hipfire_ref_md5" "$run" "$rc" "$status" "$decode" "$tokens" "not_captured" "not_captured" "$log" >> "$summary"
    return "$rc"
}

any_failed=0
if [[ "$REQUIRE_EXACT_ARTIFACT_MATCH" == "1" && "$exact_artifact_match" != "1" ]]; then
    echo "exact artifact match required but audit reported exact_artifact_match=$exact_artifact_match" >&2
    echo "see $artifact_audit" >&2
    exit 1
fi
for ((run = 1; run <= RUNS; run += 1)); do
    run_lemon_native "$run" || any_failed=1
    run_hipfire "$run" || any_failed=1
done

aggregate="$OUT_DIR/aggregate.tsv"
awk -F '\t' '
NR == 1 { next }
{
    key = $1 "\t" $2
    cases[key] = 1
    runs[key] += 1
    if ($6 == 0 && $7 == "ok") ok[key] += 1
    if ($8 != "nan") {
        speed_count[key] += 1
        speed[key, speed_count[key]] = $8 + 0
    }
    token_value = $9 + 0
    tokens[key] += token_value
    if (!(key in min_tokens) || token_value < min_tokens[key]) min_tokens[key] = token_value
    if (!(key in max_tokens) || token_value > max_tokens[key]) max_tokens[key] = token_value
    hashes[key, $10] = 1
    token_hashes[key, $11] = 1
}
END {
    print "engine\tmodel\truns\tok_runs\tmedian_decode_tok_s\tavg_decode_tokens\tmin_decode_tokens\tmax_decode_tokens\tdistinct_output_hashes\tdistinct_token_hashes"
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
        printf "%s\t%s\t%d\t%d\t%s\t%.2f\t%d\t%d\t%d\t%d\n",
            fields[1], fields[2], runs[key], ok[key] + 0, median, avg_tokens,
            min_tokens[key] + 0, max_tokens[key] + 0, distinct, distinct_tokens
    }
}' "$summary" > "$aggregate"

findings="$OUT_DIR/findings.md"
awk -F '\t' \
    -v title="Qwen3.5 0.8B 4-bit vs hipfire" \
    -v prompt_arg_md5="$prompt_arg_md5" \
    -v prompt_file_md5="$prompt_file_md5" \
    -v bench_md5="$bench_md5" \
    -v hipfire_ref_md5="$hipfire_ref_md5" \
    -v prompt_arg_file="$prompt_arg_file" '
BEGIN {
    print "# " title
    print ""
    print "- prompt_arg_md5: `" prompt_arg_md5 "`"
    print "- prompt_file_md5: `" prompt_file_md5 "`"
    print "- prompt_arg_file: `" prompt_arg_file "`"
    print "- lemon_bench_md5: `" bench_md5 "`"
    print "- hipfire_ref_md5: `" hipfire_ref_md5 "`"
    print ""
    print "| engine | model | ok/runs | median decode tok/s | avg decode tokens | output hashes | token hashes |"
    print "|---|---|---:|---:|---:|---:|---:|"
}
NR == 1 { next }
{
    printf "| %s | %s | %s/%s | %s | %.2f | %s | %s |\n",
        $1, $2, $4, $3, $5, $6, $9, $10
    if ($1 == "lemon") lemon_median = $5 + 0
    if ($1 == "hipfire") hipfire_median = $5 + 0
}
END {
    print ""
    if (lemon_median > 0 && hipfire_median > 0) {
        printf "Lemon median decode speed is %.3fx hipfire (%.2f / %.2f tok/s).\n",
            lemon_median / hipfire_median, lemon_median, hipfire_median
    }
}
' "$aggregate" > "$findings"

{
    echo ""
    echo "## Artifact Audit"
    echo ""
    awk -F '\t' '
    NR == 1 {
        print "| label | kind | format | primary md5 | size bytes | quant |"
        print "|---|---|---|---|---:|---|"
        next
    }
    $1 == "exact_artifact_match" {
        exact = $2
        next
    }
    NF >= 11 {
        printf "| %s | %s | %s | `%s` | %s | `%s` |\n", $1, $2, $5, $7, $8, $10
    }
    END {
        print ""
        print "exact_artifact_match: `" (exact == "" ? "unknown" : exact) "`"
        if (exact != "1") {
            print ""
            print "These runs are same-source-family, not same exact artifact: hipfire uses MQ4G256/FWHT-rotated weights while MLX uses its native quantized safetensors format."
        }
    }' "$artifact_audit"
} >> "$findings"

if [[ "$REQUIRE_STABLE_LEMON_HASH" == "1" ]]; then
    lemon_hash_status="$(awk -F '\t' '
        NR == 1 { next }
        $1 == "lemon" {
            runs += 1
            if ($6 == 0 && $7 == "ok") ok += 1
            hashes[$10] = 1
            token_hashes[$11] = 1
        }
        END {
            distinct = 0
            for (h in hashes) distinct += 1
            distinct_tokens = 0
            for (h in token_hashes) distinct_tokens += 1
            printf "%d\t%d\t%d\t%d\n", runs, ok, distinct, distinct_tokens
        }
    ' "$summary")"
    IFS=$'\t' read -r lemon_runs lemon_ok lemon_distinct_hashes lemon_distinct_token_hashes <<< "$lemon_hash_status"
    if [[ "$lemon_runs" -gt 0 && ("$lemon_ok" -ne "$lemon_runs" || "$lemon_distinct_hashes" -ne 1 || "$lemon_distinct_token_hashes" -ne 1) ]]; then
        echo "lemon native stability check failed: runs=$lemon_runs ok=$lemon_ok distinct_output_hashes=$lemon_distinct_hashes distinct_token_hashes=$lemon_distinct_token_hashes" >&2
        any_failed=1
    fi
fi

if [[ "$REQUIRE_TOKEN_BUDGET_MATCH" == "1" ]]; then
    token_budget_status="$(awk -F '\t' -v allowed_csv="$DECODE_TOKEN_BUDGETS" '
        BEGIN {
            n = split(allowed_csv, raw, /[ ,]+/)
            for (i = 1; i <= n; ++i) {
                if (raw[i] != "") allowed[raw[i] + 0] = 1
            }
        }
        NR == 1 { next }
        $6 == 0 && $7 == "ok" {
            checked += 1
            if (!(($9 + 0) in allowed)) {
                mismatched += 1
                details = details sprintf("%s/%s/run%s=%s ", $1, $2, $5, $9)
            }
        }
        END {
            printf "%d\t%d\t%s\n", checked, mismatched + 0, details
        }
    ' "$summary")"
    IFS=$'\t' read -r checked_budget_rows mismatched_budget_rows budget_mismatch_details <<< "$token_budget_status"
    if [[ "$checked_budget_rows" -gt 0 && "$mismatched_budget_rows" -ne 0 ]]; then
        echo "decode token-budget match failed: allowed=$DECODE_TOKEN_BUDGETS mismatches=$budget_mismatch_details" >&2
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
echo "wrote $findings"
exit "$any_failed"
