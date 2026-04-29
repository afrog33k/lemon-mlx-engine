#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-gfx1151}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bench}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/benchmarks/prompts/qwen35_add_raw.py}"
OUT_BASE="${OUT_BASE:-$ROOT_DIR/.codeinsight+research}"

RUNS="${RUNS:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
DECODE_WARMUP_STEPS="${DECODE_WARMUP_STEPS:-1}"
JOBS="${JOBS:-8}"
SUPPORTED_QMV_COL_VARIANTS="4 8 16 32 64"
VARIANTS="${VARIANTS:-default 16 32 64}"
VARIANT_ORDER_POLICY="${VARIANT_ORDER_POLICY:-rotate}"
CASES="${CASES:-qwen35 qwen3}"
MAX_TOKEN_FREQ="${MAX_TOKEN_FREQ:-0.40}"
MAX_TOKEN_RUN="${MAX_TOKEN_RUN:-16}"
ATTRACTOR_MIN_TOKENS="${ATTRACTOR_MIN_TOKENS:-16}"
EXPECT_SUBSTRING="${EXPECT_SUBSTRING:-return x + y}"

QWEN35_MODEL="${QWEN35_MODEL:-mlx-community/Qwen3.5-0.8B-MLX-4bit}"
QWEN35_MAX_TOKENS="${QWEN35_MAX_TOKENS:-56}"
QWEN3_MODEL="${QWEN3_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
QWEN3_MAX_TOKENS="${QWEN3_MAX_TOKENS:-128}"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "missing build directory: $BUILD_DIR" >&2
    echo "configure a gfx1151 build first, or set BUILD_DIR" >&2
    exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "missing prompt file: $PROMPT_FILE" >&2
    exit 1
fi

normalize_words() {
    printf '%s\n' "$1" | tr ',' ' '
}

variants_for_run() {
    local run="$1"
    local variants_text
    variants_text="$(normalize_words "$VARIANTS" | xargs)"

    if [[ "$VARIANT_ORDER_POLICY" == "fixed" ]]; then
        printf '%s\n' $variants_text
        return
    fi

    if [[ "$VARIANT_ORDER_POLICY" != "rotate" ]]; then
        echo "unsupported VARIANT_ORDER_POLICY: $VARIANT_ORDER_POLICY" >&2
        echo "supported policies: rotate fixed" >&2
        exit 1
    fi

    local variants=($variants_text)
    local count="${#variants[@]}"
    if [[ "$count" == "0" ]]; then
        return
    fi

    local offset=$(( (run - 1) % count ))
    local idx
    local pos
    for ((pos = 0; pos < count; pos += 1)); do
        idx=$(( (pos + offset) % count ))
        printf '%s\n' "${variants[$idx]}"
    done
}

case_model() {
    case "$1" in
        qwen35) printf '%s\n' "$QWEN35_MODEL" ;;
        qwen3) printf '%s\n' "$QWEN3_MODEL" ;;
        *) return 1 ;;
    esac
}

case_max_tokens() {
    case "$1" in
        qwen35) printf '%s\n' "$QWEN35_MAX_TOKENS" ;;
        qwen3) printf '%s\n' "$QWEN3_MAX_TOKENS" ;;
        *) return 1 ;;
    esac
}

for case_name in $(normalize_words "$CASES"); do
    case_model "$case_name" >/dev/null || {
        echo "unknown case in CASES: $case_name" >&2
        exit 1
    }
done

for variant in $(normalize_words "$VARIANTS"); do
    if [[ "$variant" == "default" ]]; then
        continue
    fi
    supported=0
    for supported_variant in $SUPPORTED_QMV_COL_VARIANTS; do
        if [[ "$variant" == "$supported_variant" ]]; then
            supported=1
            break
        fi
    done
    if [[ "$supported" != "1" ]]; then
        echo "unsupported variant in VARIANTS: $variant" >&2
        echo "supported variants: default $SUPPORTED_QMV_COL_VARIANTS" >&2
        exit 1
    fi
done

case "$VARIANT_ORDER_POLICY" in
    rotate|fixed) ;;
    *)
        echo "unsupported VARIANT_ORDER_POLICY: $VARIANT_ORDER_POLICY" >&2
        echo "supported policies: rotate fixed" >&2
        exit 1
        ;;
esac

cmake --build "$BUILD_DIR" --target bench -j "$JOBS"

if [[ ! -x "$BENCH_BIN" ]]; then
    echo "missing bench binary: $BENCH_BIN" >&2
    exit 1
fi

if [[ "${ALLOW_EXPERIMENTAL_HIP_GDN:-0}" != "1" ]]; then
    export LEMON_MLX_GDN_DISABLE_HIP=1
fi
if [[ "${ALLOW_UNSTABLE_TILED_QMV:-0}" != "1" ]]; then
    unset MLX_ROCM_QMV_ENABLE_TILED
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/qmv_cols_gfx1151_$stamp"
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
decode_warmup_steps=$DECODE_WARMUP_STEPS
cases=$(normalize_words "$CASES" | xargs)
variants=$(normalize_words "$VARIANTS" | xargs)
variant_order_policy=$VARIANT_ORDER_POLICY
supported_qmv_col_variants=default $SUPPORTED_QMV_COL_VARIANTS
qwen35_model=$QWEN35_MODEL
qwen35_max_tokens=$QWEN35_MAX_TOKENS
qwen3_model=$QWEN3_MODEL
qwen3_max_tokens=$QWEN3_MAX_TOKENS
expect_substring=$EXPECT_SUBSTRING
attractor_min_tokens=$ATTRACTOR_MIN_TOKENS
max_token_freq=$MAX_TOKEN_FREQ
max_token_run=$MAX_TOKEN_RUN
lemon_mlx_gdn_enable_hip=${LEMON_MLX_GDN_ENABLE_HIP:-}
lemon_mlx_gdn_disable_hip=${LEMON_MLX_GDN_DISABLE_HIP:-}
allow_unstable_tiled_qmv=${ALLOW_UNSTABLE_TILED_QMV:-0}
mlx_rocm_qmv_enable_tiled=${MLX_ROCM_QMV_ENABLE_TILED:-}
parent_mlx_rocm_qmv_cols_per_block=${MLX_ROCM_QMV_COLS_PER_BLOCK:-}
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
} > "$OUT_DIR/system_before.txt"

summary="$OUT_DIR/summary.tsv"
printf 'case\tvariant\tcols_per_block\tmodel\tprompt_md5\tbinary_md5\trun\texit_code\tstatus\tdecode_tok_s\tdecode_tokens\toutput_hash\ttoken_hash\tunique_token_ratio\tmax_token_freq\tmax_token_run\tlog\n' > "$summary"
variant_order="$OUT_DIR/variant_order.tsv"
printf 'run\tcase\tposition\tvariant\n' > "$variant_order"

run_one() {
    local case_name="$1"
    local variant="$2"
    local run="$3"
    local model
    local max_tokens
    local log
    model="$(case_model "$case_name")"
    max_tokens="$(case_max_tokens "$case_name")"
    log="$OUT_DIR/${case_name}_${variant}_run${run}.log"

    local env_unset=(-u LEMON_MLX_DEQUANTIZE_WEIGHTS -u MLX_ROCM_QMV_COLS_PER_BLOCK)
    local env_set=()
    local cols_value="default"
    if [[ "$case_name" == "qwen35" ]]; then
        env_set+=(LEMON_MLX_QWEN35_KEEP_QUANTIZED=1)
    fi
    if [[ "$variant" != "default" ]]; then
        env_set+=(MLX_ROCM_QMV_COLS_PER_BLOCK="$variant")
        cols_value="$variant"
    fi

    set +e
    timeout "$TIMEOUT_SECONDS" env "${env_unset[@]}" "${env_set[@]}" "$BENCH_BIN" "$model" \
        --prompt-file "$PROMPT_FILE" \
        --raw \
        --max-tokens "$max_tokens" \
        --warmup-decode-steps "$DECODE_WARMUP_STEPS" \
        --temperature 0 \
        --top-p 1 \
        --expect-substring "$EXPECT_SUBSTRING" \
        --max-token-freq "$MAX_TOKEN_FREQ" \
        --max-token-run "$MAX_TOKEN_RUN" \
        --attractor-min-tokens "$ATTRACTOR_MIN_TOKENS" \
        > "$log" 2>&1
    local rc=$?
    set -e

    local parsed
    parsed="$(awk -F '\t' '$1 != "model" && NF >= 16 {print $15 "\t" $8 "\t" $6 "\t" $14 "\t" (NF >= 17 ? $17 : "0000000000000000") "\t" $11 "\t" $12 "\t" $13; found=1; exit} END {if (!found) print "no_tsv\tnan\t0\t0000000000000000\t0000000000000000\tnan\tnan\t0"}' "$log")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$case_name" "$variant" "$cols_value" "$model" "$prompt_md5" "$bench_md5" "$run" "$rc" "$parsed" "$log" >> "$summary"

    return "$rc"
}

any_failed=0
for run in $(seq 1 "$RUNS"); do
    for case_name in $(normalize_words "$CASES"); do
        variant_position=0
        for variant in $(variants_for_run "$run"); do
            variant_position=$((variant_position + 1))
            printf '%s\t%s\t%s\t%s\n' "$run" "$case_name" "$variant_position" "$variant" >> "$variant_order"
            run_one "$case_name" "$variant" "$run" || any_failed=1
        done
    done
done

aggregate="$OUT_DIR/aggregate.tsv"
python3 - "$summary" "$aggregate" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

summary_path, aggregate_path = sys.argv[1:3]
groups = defaultdict(list)
with open(summary_path, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        groups[(row["case"], row["variant"], row["cols_per_block"], row["model"])].append(row)

with open(aggregate_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow([
        "case",
        "variant",
        "cols_per_block",
        "model",
        "runs",
        "ok_runs",
        "median_decode_tok_s",
        "min_decode_tokens",
        "max_decode_tokens",
        "stable_output_hash",
        "output_hashes",
        "stable_token_hash",
        "token_hashes",
        "status",
    ])
    for key, rows in sorted(groups.items()):
        speeds = []
        tokens = []
        hashes = set()
        token_hashes = set()
        ok_runs = 0
        for row in rows:
            if row["exit_code"] == "0" and row["status"] == "ok":
                ok_runs += 1
                try:
                    speeds.append(float(row["decode_tok_s"]))
                    tokens.append(int(row["decode_tokens"]))
                except ValueError:
                    pass
                hashes.add(row["output_hash"])
                token_hashes.add(row["token_hash"])
        stable_output = len(hashes) == 1 and ok_runs == len(rows)
        stable_tokens = len(token_hashes) == 1 and ok_runs == len(rows)
        stable = stable_output and stable_tokens
        median_speed = statistics.median(speeds) if speeds else float("nan")
        status = "ok" if stable else "unstable_or_failed"
        writer.writerow([
            *key,
            len(rows),
            ok_runs,
            f"{median_speed:.3f}" if median_speed == median_speed else "nan",
            min(tokens) if tokens else 0,
            max(tokens) if tokens else 0,
            1 if stable_output else 0,
            ",".join(sorted(hashes)) if hashes else "",
            1 if stable_tokens else 0,
            ",".join(sorted(token_hashes)) if token_hashes else "",
            status,
        ])
PY

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

unstable_count="$(awk -F '\t' '$1 != "case" && $14 != "ok" {count += 1} END {print count + 0}' "$aggregate")"

echo "wrote $summary"
echo "wrote $aggregate"
echo "wrote $variant_order"
if [[ "$unstable_count" != "0" ]]; then
    echo "one or more QMV column variants failed or produced unstable output; see $aggregate" >&2
    exit 1
fi
exit "$any_failed"
