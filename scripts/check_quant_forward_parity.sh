#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-gfx1151}"
DIAG_BIN="${DIAG_BIN:-$BUILD_DIR/diagnose}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/benchmarks/prompts/qwen35_add_raw.py}"
OUT_BASE="${OUT_BASE:-$ROOT_DIR/.codeinsight+research}"

RUNS="${RUNS:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
STEPS="${STEPS:-8}"
JOBS="${JOBS:-8}"
REQUIRE_STABLE_TOKEN_HASH="${REQUIRE_STABLE_TOKEN_HASH:-1}"
CASES="${CASES:-qwen35 qwen3}"

QWEN35_MODEL="${QWEN35_MODEL:-mlx-community/Qwen3.5-0.8B-MLX-4bit}"
QWEN3_MODEL="${QWEN3_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
QWEN35_MAX_DIFF="${QWEN35_MAX_DIFF:-0.15}"
QWEN3_MAX_DIFF="${QWEN3_MAX_DIFF:-1.0}"

normalize_words() {
    printf '%s\n' "$1" | tr ',' ' '
}

case_model() {
    case "$1" in
        qwen35) printf '%s\n' "$QWEN35_MODEL" ;;
        qwen3) printf '%s\n' "$QWEN3_MODEL" ;;
        *) return 1 ;;
    esac
}

case_max_diff() {
    case "$1" in
        qwen35) printf '%s\n' "$QWEN35_MAX_DIFF" ;;
        qwen3) printf '%s\n' "$QWEN3_MAX_DIFF" ;;
        *) return 1 ;;
    esac
}

case_label() {
    case "$1" in
        qwen35) printf '%s\n' "qwen35_0p8b_4bit_native_vs_dequant" ;;
        qwen3) printf '%s\n' "qwen3_0p6b_4bit_native_vs_dequant" ;;
        *) return 1 ;;
    esac
}

for case_name in $(normalize_words "$CASES"); do
    case_model "$case_name" >/dev/null || {
        echo "unknown case in CASES: $case_name" >&2
        echo "supported cases: qwen35 qwen3" >&2
        exit 1
    }
done

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "missing build directory: $BUILD_DIR" >&2
    echo "configure a gfx1151 build first, or set BUILD_DIR" >&2
    exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "missing prompt file: $PROMPT_FILE" >&2
    exit 1
fi

cmake --build "$BUILD_DIR" --target diagnose -j "$JOBS"

if [[ ! -x "$DIAG_BIN" ]]; then
    echo "missing diagnose binary: $DIAG_BIN" >&2
    exit 1
fi

if [[ "${ALLOW_EXPERIMENTAL_HIP_GDN:-0}" != "1" ]]; then
    export LEMON_MLX_GDN_DISABLE_HIP=1
fi
if [[ "${ALLOW_UNSTABLE_TILED_QMV:-0}" != "1" ]]; then
    unset MLX_ROCM_QMV_ENABLE_TILED
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_BASE/quant_forward_parity_$stamp"
mkdir -p "$OUT_DIR"

prompt_md5="$(md5sum "$PROMPT_FILE" | awk '{print $1}')"
diagnose_md5="$(md5sum "$DIAG_BIN" | awk '{print $1}')"

cat > "$OUT_DIR/metadata.txt" <<EOF_META
date_utc=$stamp
build_dir=$BUILD_DIR
diagnose_bin=$DIAG_BIN
diagnose_md5=$diagnose_md5
prompt_file=$PROMPT_FILE
prompt_md5=$prompt_md5
runs=$RUNS
timeout_seconds=$TIMEOUT_SECONDS
steps=$STEPS
require_stable_token_hash=$REQUIRE_STABLE_TOKEN_HASH
cases=$(normalize_words "$CASES" | xargs)
qwen35_model=$QWEN35_MODEL
qwen3_model=$QWEN3_MODEL
qwen35_max_diff=$QWEN35_MAX_DIFF
qwen3_max_diff=$QWEN3_MAX_DIFF
lemon_mlx_gdn_enable_hip=${LEMON_MLX_GDN_ENABLE_HIP:-}
lemon_mlx_gdn_disable_hip=${LEMON_MLX_GDN_DISABLE_HIP:-}
mlx_rocm_qmv_enable_tiled=${MLX_ROCM_QMV_ENABLE_TILED:-}
allow_unstable_tiled_qmv=${ALLOW_UNSTABLE_TILED_QMV:-0}
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
printf 'case\tmodel\tprompt_md5\tdiagnose_md5\trun\texit_code\tstatus\tprompt_tokens\tsteps_observed\tworst_max_diff\tworst_mean_diff\tmin_dequant_margin\tworst_diff_over_margin\tsame_argmax\tfirst_mismatch_step\tmax_allowed\tdequant_token_hash\tquant_token_hash\tlog\n' > "$summary"

run_case() {
    local name="$1"
    local model="$2"
    local max_diff="$3"
    local run
    local case_failed=0

    for ((run = 1; run <= RUNS; run += 1)); do
        local log="$OUT_DIR/${name}_run${run}.log"
        set +e
        timeout "$TIMEOUT_SECONDS" "$DIAG_BIN" "$model" \
            --qforward-compare \
            --qforward-prompt-file "$PROMPT_FILE" \
            --qforward-steps "$STEPS" \
            --qforward-max-diff "$max_diff" \
            > "$log" 2>&1
        local rc=$?
        set -e

        local parsed
        parsed="$(python3 - "$log" <<'PY'
import re
import sys

log_path = sys.argv[1]
text = open(log_path, "r", encoding="utf-8", errors="replace").read().splitlines()
fields = {
    "status": "no_status",
    "prompt_tokens": "0",
    "steps_observed": "0",
    "worst_max_diff": "nan",
    "worst_mean_diff": "nan",
    "min_dequant_margin": "nan",
    "worst_diff_over_margin": "nan",
    "same_argmax": "0",
    "first_mismatch_step": "-2",
    "max_allowed": "nan",
    "dequant_token_hash": "0000000000000000",
    "quant_token_hash": "0000000000000000",
}
dequant_tokens = []
quant_tokens = []
for line in text:
    match = re.search(
        r"QFORWARD_STEP step=(\d+) dequant_token=(\d+).* quant_token=(\d+).* same_argmax=([01])",
        line,
    )
    if match:
        fields["steps_observed"] = str(max(int(fields["steps_observed"]), int(match.group(1)) + 1))
        dequant_tokens.append(match.group(2))
        quant_tokens.append(match.group(3))
    match = re.search(r"QFORWARD_PROMPT_TOKENS = (\d+)", line)
    if match:
        fields["prompt_tokens"] = match.group(1)
    match = re.search(r"QFORWARD_WORST_MAX_DIFF = ([^ ]+)", line)
    if match:
        fields["worst_max_diff"] = match.group(1)
    match = re.search(r"QFORWARD_WORST_MEAN_DIFF = ([^ ]+)", line)
    if match:
        fields["worst_mean_diff"] = match.group(1)
    match = re.search(r"QFORWARD_MIN_DEQUANT_MARGIN = ([^ ]+)", line)
    if match:
        fields["min_dequant_margin"] = match.group(1)
    match = re.search(r"QFORWARD_WORST_DIFF_OVER_MARGIN = ([^ ]+)", line)
    if match:
        fields["worst_diff_over_margin"] = match.group(1)
    match = re.search(
        r"QFORWARD_STATUS = (\S+) same_argmax=(\d+) "
        r"first_mismatch_step=([-0-9]+) max_allowed=([^ ]+)",
        line,
    )
    if match:
        fields["status"] = match.group(1)
        fields["same_argmax"] = match.group(2)
        fields["first_mismatch_step"] = match.group(3)
        fields["max_allowed"] = match.group(4)

if dequant_tokens:
    import hashlib
    fields["dequant_token_hash"] = hashlib.sha256(",".join(dequant_tokens).encode()).hexdigest()[:16]
    fields["quant_token_hash"] = hashlib.sha256(",".join(quant_tokens).encode()).hexdigest()[:16]

print(
    "\t".join(
        fields[key]
        for key in [
            "status",
            "prompt_tokens",
            "steps_observed",
            "worst_max_diff",
            "worst_mean_diff",
            "min_dequant_margin",
            "worst_diff_over_margin",
            "same_argmax",
            "first_mismatch_step",
            "max_allowed",
            "dequant_token_hash",
            "quant_token_hash",
        ]
    )
)
PY
)"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$name" "$model" "$prompt_md5" "$diagnose_md5" "$run" "$rc" "$parsed" "$log" >> "$summary"

        if [[ "$rc" -ne 0 ]]; then
            echo "$name run $run failed with exit code $rc; see $log" >&2
            case_failed=1
        fi
    done

    return "$case_failed"
}

any_failed=0
for case_name in $(normalize_words "$CASES"); do
    run_case "$(case_label "$case_name")" "$(case_model "$case_name")" "$(case_max_diff "$case_name")" || any_failed=1
done

aggregate="$OUT_DIR/aggregate.tsv"
python3 - "$summary" "$aggregate" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

summary_path, aggregate_path = sys.argv[1:3]
rows_by_case = defaultdict(list)
with open(summary_path, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows_by_case[(row["case"], row["model"])].append(row)

with open(aggregate_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow([
        "case",
        "model",
        "runs",
        "ok_runs",
        "median_worst_max_diff",
        "max_worst_max_diff",
        "max_worst_mean_diff",
        "min_dequant_margin",
        "max_diff_over_margin",
        "all_same_argmax",
        "distinct_dequant_token_hashes",
        "distinct_quant_token_hashes",
        "all_token_hashes_stable",
        "steps_observed",
        "prompt_tokens",
    ])
    for (case, model), rows in sorted(rows_by_case.items()):
        diffs = []
        mean_diffs = []
        margins = []
        diff_over_margins = []
        for row in rows:
            try:
                diffs.append(float(row["worst_max_diff"]))
            except ValueError:
                pass
            try:
                mean_diffs.append(float(row["worst_mean_diff"]))
            except ValueError:
                pass
            try:
                margins.append(float(row["min_dequant_margin"]))
            except ValueError:
                pass
            try:
                diff_over_margins.append(float(row["worst_diff_over_margin"]))
            except ValueError:
                pass
        ok_runs = sum(
            1
            for row in rows
            if row["exit_code"] == "0"
            and row["status"] == "ok"
            and row["same_argmax"] == "1"
        )
        prompt_tokens = rows[0]["prompt_tokens"] if rows else "0"
        steps_observed = rows[0]["steps_observed"] if rows else "0"
        dequant_hashes = {
            row["dequant_token_hash"]
            for row in rows
            if row["dequant_token_hash"] != "0000000000000000"
        }
        quant_hashes = {
            row["quant_token_hash"]
            for row in rows
            if row["quant_token_hash"] != "0000000000000000"
        }
        stable_hashes = int(
            len(dequant_hashes) == 1
            and len(quant_hashes) == 1
            and dequant_hashes == quant_hashes
        )
        writer.writerow([
            case,
            model,
            len(rows),
            ok_runs,
            f"{statistics.median(diffs):.6g}" if diffs else "nan",
            f"{max(diffs):.6g}" if diffs else "nan",
            f"{max(mean_diffs):.6g}" if mean_diffs else "nan",
            f"{min(margins):.6g}" if margins else "nan",
            f"{max(diff_over_margins):.6g}" if diff_over_margins else "nan",
            1 if ok_runs == len(rows) else 0,
            len(dequant_hashes),
            len(quant_hashes),
            stable_hashes,
            steps_observed,
            prompt_tokens,
        ])
PY

if [[ "$REQUIRE_STABLE_TOKEN_HASH" != "0" ]]; then
    if ! awk -F '\t' '
        NR == 1 { next }
        $13 != "1" {
            printf "unstable or mismatched token hashes for %s: distinct_dequant=%s distinct_quant=%s\n",
                $1, $11, $12 > "/dev/stderr"
            exit 1
        }
    ' "$aggregate"; then
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
