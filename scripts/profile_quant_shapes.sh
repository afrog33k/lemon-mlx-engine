#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-gfx1151}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bench}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/benchmarks/prompts/qwen35_add_raw.py}"
OUT_BASE="${OUT_BASE:-$ROOT_DIR/.codeinsight+research}"

RUNS="${RUNS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
MAX_TOKENS="${MAX_TOKENS:-32}"
DECODE_WARMUP_STEPS="${DECODE_WARMUP_STEPS:-1}"
JOBS="${JOBS:-8}"

QWEN35_MODEL="${QWEN35_MODEL:-mlx-community/Qwen3.5-0.8B-MLX-4bit}"
QWEN3_MODEL="${QWEN3_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
EXPECT_SUBSTRING="${EXPECT_SUBSTRING:-return x + y}"
STOP_AFTER_SUBSTRING="${STOP_AFTER_SUBSTRING:-$EXPECT_SUBSTRING}"
ATTRACTOR_MIN_TOKENS="${ATTRACTOR_MIN_TOKENS:-16}"
MAX_TOKEN_FREQ="${MAX_TOKEN_FREQ:-0.40}"
MAX_TOKEN_RUN="${MAX_TOKEN_RUN:-16}"
PROFILE_TIMING="${PROFILE_TIMING:-0}"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "missing build directory: $BUILD_DIR" >&2
    echo "configure a gfx1151 build first, or set BUILD_DIR" >&2
    exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "missing prompt file: $PROMPT_FILE" >&2
    exit 1
fi

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
OUT_DIR="$OUT_BASE/quant_profile_shapes_$stamp"
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
qwen35_model=$QWEN35_MODEL
qwen3_model=$QWEN3_MODEL
expect_substring=$EXPECT_SUBSTRING
stop_after_substring=$STOP_AFTER_SUBSTRING
attractor_min_tokens=$ATTRACTOR_MIN_TOKENS
max_token_freq=$MAX_TOKEN_FREQ
max_token_run=$MAX_TOKEN_RUN
lemon_mlx_gdn_enable_hip=${LEMON_MLX_GDN_ENABLE_HIP:-}
lemon_mlx_gdn_disable_hip=${LEMON_MLX_GDN_DISABLE_HIP:-}
profile_timing=$PROFILE_TIMING
mlx_rocm_qmv_enable_tiled=${MLX_ROCM_QMV_ENABLE_TILED:-}
allow_unstable_tiled_qmv=${ALLOW_UNSTABLE_TILED_QMV:-0}
qwen35_native_dequantize_weights=unset
qwen35_native_keep_quantized=1
qwen3_native_dequantize_weights=unset
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

bench_summary="$OUT_DIR/bench_summary.tsv"
profile_rows="$OUT_DIR/quant_profile.tsv"
printf 'case\tmodel\tprompt_md5\tbinary_md5\trun\texit_code\tstatus\tdecode_tok_s\tdecode_tokens\toutput_hash\tlog\n' > "$bench_summary"
printf 'case\tmodel\trun\ttable\tphase\tweight\tcalls\ttotal_ms\tavg_us\tlhs_shape\trhs_shape\tsequence_len\tlog\n' > "$profile_rows"

run_case() {
    local case_name="$1"
    local model="$2"
    local run="$3"
    shift 3
    local log="$OUT_DIR/${case_name}_run${run}.log"
    local stop_args=()
    local profile_env=(LEMON_MLX_QUANT_PROFILE=1)
    if [[ -n "$STOP_AFTER_SUBSTRING" ]]; then
        stop_args=(--stop-after-substring "$STOP_AFTER_SUBSTRING")
    fi
    if [[ "$PROFILE_TIMING" == "1" ]]; then
        profile_env+=(LEMON_MLX_QUANT_PROFILE_TIMING=1)
    fi

    set +e
    timeout "$TIMEOUT_SECONDS" env "$@" "${profile_env[@]}" "$BENCH_BIN" "$model" \
        --prompt-file "$PROMPT_FILE" \
        --raw \
        --max-tokens "$MAX_TOKENS" \
        --warmup-decode-steps "$DECODE_WARMUP_STEPS" \
        --temperature 0 \
        --top-p 1 \
        --expect-substring "$EXPECT_SUBSTRING" \
        "${stop_args[@]}" \
        --max-token-freq "$MAX_TOKEN_FREQ" \
        --max-token-run "$MAX_TOKEN_RUN" \
        --attractor-min-tokens "$ATTRACTOR_MIN_TOKENS" \
        --print-output \
        > "$log" 2>&1
    local rc=$?
    set -e

    local parsed
    parsed="$(awk -F '\t' '$1 != "model" && NF >= 16 {print $15 "\t" $8 "\t" $6 "\t" $14; found=1; exit} END {if (!found) print "no_tsv\tnan\t0\t0000000000000000"}' "$log")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$case_name" "$model" "$prompt_md5" "$bench_md5" "$run" "$rc" "$parsed" "$log" >> "$bench_summary"

    python3 - "$case_name" "$model" "$run" "$log" >> "$profile_rows" <<'PY'
import re
import sys

case_name, model, run, log_path = sys.argv[1:5]
text = open(log_path, "r", encoding="utf-8", errors="replace").read().splitlines()
pattern = re.compile(
    r'^\[lemon-mlx\] quant_profile (\S+) calls=(\d+) shape="(\[[^\]]+\]) @ (\[[^\]]+\])"'
    r'(?: total_ms=([0-9.eE+-]+) avg_us=([0-9.eE+-]+))?'
    r'(?: weight="([^"]+)")?'
)

def dims(shape):
    inner = shape.strip()[1:-1]
    if not inner:
        return []
    return [int(part) for part in inner.split("x")]

rows = []
for line in text:
    match = pattern.search(line)
    if not match:
        continue
    table, calls, lhs, rhs, total_ms, avg_us, weight = match.groups()
    lhs_dims = dims(lhs)
    if len(lhs_dims) >= 3:
        sequence_len = lhs_dims[-2]
    elif len(lhs_dims) == 2:
        sequence_len = lhs_dims[-1]
    else:
        sequence_len = 0
    phase = "decode" if sequence_len == 1 else "prefill"
    rows.append((table, phase, weight or "<unknown>", int(calls), total_ms or "nan", avg_us or "nan", lhs, rhs, sequence_len))

rows.sort(key=lambda row: (-row[3], row[0], row[2], row[6], row[7]))
for table, phase, weight, calls, total_ms, avg_us, lhs, rhs, sequence_len in rows:
    print(
        f"{case_name}\t{model}\t{run}\t{table}\t{phase}\t{weight}\t{calls}\t{total_ms}\t{avg_us}\t"
        f"{lhs}\t{rhs}\t{sequence_len}\t{log_path}"
    )
PY

    return "$rc"
}

any_failed=0
for ((run = 1; run <= RUNS; run += 1)); do
    run_case "qwen35_0p8b_native4" "$QWEN35_MODEL" "$run" \
        -u LEMON_MLX_DEQUANTIZE_WEIGHTS LEMON_MLX_QWEN35_KEEP_QUANTIZED=1 || any_failed=1
    run_case "qwen3_0p6b_native4" "$QWEN3_MODEL" "$run" \
        -u LEMON_MLX_DEQUANTIZE_WEIGHTS || any_failed=1
done

aggregate="$OUT_DIR/quant_profile_aggregate.tsv"
phase_totals="$OUT_DIR/quant_profile_phase_totals.tsv"
family_totals="$OUT_DIR/quant_profile_family_totals.tsv"
python3 - "$profile_rows" "$aggregate" "$phase_totals" "$family_totals" <<'PY'
import csv
import sys
from collections import defaultdict

profile_path, aggregate_path, phase_totals_path, family_totals_path = sys.argv[1:5]
totals = defaultdict(lambda: {"calls": 0, "total_ms": 0.0, "has_timing": False})
phase_totals = defaultdict(lambda: {"calls": 0, "total_ms": 0.0, "has_timing": False})
family_totals = defaultdict(lambda: {"calls": 0, "total_ms": 0.0, "has_timing": False})

def weight_family(table, weight):
    if table.startswith("embedding_"):
        return "token_embedding"
    if weight.endswith("embed_tokens.weight"):
        return "lm_head_tied_embedding"
    if ".linear_attn." in weight:
        if ".in_proj_" in weight:
            return "gdn_input_projection"
        if ".out_proj." in weight:
            return "gdn_output_projection"
        return "gdn_other"
    if ".self_attn." in weight:
        if ".qkv_proj." in weight or ".q_proj." in weight or ".k_proj." in weight or ".v_proj." in weight:
            return "attention_input_projection"
        if ".o_proj." in weight:
            return "attention_output_projection"
        return "attention_other"
    if ".mlp." in weight:
        if ".gate_up_proj." in weight or ".gate_proj." in weight or ".up_proj." in weight:
            return "mlp_gate_up_projection"
        if ".down_proj." in weight:
            return "mlp_down_projection"
        return "mlp_other"
    if ".shared_expert." in weight:
        return "shared_expert_projection"
    if ".switch_mlp." in weight:
        return "switch_expert_projection"
    if weight.endswith(".gate.weight") or weight.endswith("shared_expert_gate.weight"):
        return "router_or_gate_projection"
    return "other"

with open(profile_path, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        key = (
            row["case"],
            row["model"],
            row["table"],
            row["phase"],
            row["weight"],
            row["lhs_shape"],
            row["rhs_shape"],
            row["sequence_len"],
        )
        calls = int(row["calls"])
        totals[key]["calls"] += calls
        phase_key = (
            row["case"],
            row["model"],
            row["table"],
            row["phase"],
        )
        phase_totals[phase_key]["calls"] += calls
        family_key = (
            row["case"],
            row["model"],
            row["table"],
            row["phase"],
            weight_family(row["table"], row["weight"]),
        )
        family_totals[family_key]["calls"] += calls
        try:
            total_ms = float(row["total_ms"])
        except ValueError:
            total_ms = float("nan")
        if total_ms == total_ms:
            totals[key]["total_ms"] += total_ms
            totals[key]["has_timing"] = True
            phase_totals[phase_key]["total_ms"] += total_ms
            phase_totals[phase_key]["has_timing"] = True
            family_totals[family_key]["total_ms"] += total_ms
            family_totals[family_key]["has_timing"] = True

with open(aggregate_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow([
        "case",
        "model",
        "table",
        "phase",
        "weight",
        "calls",
        "total_ms",
        "avg_us",
        "lhs_shape",
        "rhs_shape",
        "sequence_len",
    ])
    for key, stats in sorted(totals.items(), key=lambda item: (-item[1]["calls"], item[0])):
        case, model, table, phase, weight, lhs, rhs, sequence_len = key
        calls = stats["calls"]
        if stats["has_timing"]:
            total_ms = stats["total_ms"]
            avg_us = (total_ms * 1000.0 / calls) if calls else 0.0
            total_ms_text = f"{total_ms:.6f}"
            avg_us_text = f"{avg_us:.3f}"
        else:
            total_ms_text = "nan"
            avg_us_text = "nan"
        writer.writerow([
            case, model, table, phase, weight, calls, total_ms_text, avg_us_text,
            lhs, rhs, sequence_len,
        ])

with open(phase_totals_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow([
        "case",
        "model",
        "table",
        "phase",
        "calls",
        "total_ms",
        "avg_us",
    ])
    for key, stats in sorted(
        phase_totals.items(),
        key=lambda item: (
            not item[1]["has_timing"],
            -item[1]["total_ms"],
            -item[1]["calls"],
            item[0],
        ),
    ):
        case, model, table, phase = key
        calls = stats["calls"]
        if stats["has_timing"]:
            total_ms = stats["total_ms"]
            avg_us = (total_ms * 1000.0 / calls) if calls else 0.0
            total_ms_text = f"{total_ms:.6f}"
            avg_us_text = f"{avg_us:.3f}"
        else:
            total_ms_text = "nan"
            avg_us_text = "nan"
        writer.writerow([
            case, model, table, phase, calls, total_ms_text, avg_us_text,
        ])

with open(family_totals_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow([
        "case",
        "model",
        "table",
        "phase",
        "family",
        "calls",
        "total_ms",
        "avg_us",
    ])
    for key, stats in sorted(
        family_totals.items(),
        key=lambda item: (
            not item[1]["has_timing"],
            -item[1]["total_ms"],
            -item[1]["calls"],
            item[0],
        ),
    ):
        case, model, table, phase, family = key
        calls = stats["calls"]
        if stats["has_timing"]:
            total_ms = stats["total_ms"]
            avg_us = (total_ms * 1000.0 / calls) if calls else 0.0
            total_ms_text = f"{total_ms:.6f}"
            avg_us_text = f"{avg_us:.3f}"
        else:
            total_ms_text = "nan"
            avg_us_text = "nan"
        writer.writerow([
            case, model, table, phase, family, calls, total_ms_text, avg_us_text,
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

echo "wrote $bench_summary"
echo "wrote $profile_rows"
echo "wrote $aggregate"
echo "wrote $phase_totals"
echo "wrote $family_totals"
exit "$any_failed"
