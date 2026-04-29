// Diagnostic tool: checks each stage of the inference pipeline for numerical issues
// Usage: ./diagnose <model_path>

#include <mlx-lm/llm/llm_factory.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx/mlx.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

namespace mx = mlx::core;
namespace fs = std::filesystem;

static void print_stats(const char* label, const mx::array& arr) {
    auto flat = mx::reshape(arr, {-1});
    auto f = mx::astype(flat, mx::float32);
    mx::eval(f);

    auto data = f.data<float>();
    int total = f.size();
    double sum = 0, abs_sum = 0;
    float mn = data[0], mx_val = data[0];
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < total; i++) {
        float v = data[i];
        if (std::isnan(v)) { nan_count++; continue; }
        if (std::isinf(v)) { inf_count++; continue; }
        if (v < mn) mn = v;
        if (v > mx_val) mx_val = v;
        sum += v;
        abs_sum += std::fabs(v);
    }
    int valid = total - nan_count - inf_count;
    float mean = valid > 0 ? sum / valid : 0;
    float amean = valid > 0 ? abs_sum / valid : 0;

    std::cerr << std::fixed << std::setprecision(6);
    std::cerr << "[DIAG] " << std::setw(35) << std::left << label
              << " shape=" << arr.shape()
              << " min=" << mn << " max=" << mx_val
              << " mean=" << mean << " |mean|=" << amean;
    if (nan_count) std::cerr << " NaN=" << nan_count;
    if (inf_count) std::cerr << " Inf=" << inf_count;
    std::cerr << std::endl;
}

static void print_vals(const char* label, const mx::array& arr, int n = 10) {
    auto flat = mx::reshape(arr, {-1});
    auto f = mx::astype(flat, mx::float32);
    mx::eval(f);
    auto data = f.data<float>();
    int count = std::min(n, (int)f.size());
    std::cerr << "[VALS] " << label << ": [";
    for (int i = 0; i < count; i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << std::fixed << std::setprecision(4) << data[i];
    }
    std::cerr << "]" << std::endl;
}

// Find safetensors files in model directory
static std::vector<std::string> find_safetensors(const std::string& dir) {
    std::vector<std::string> files;
    for (auto& entry : fs::directory_iterator(dir)) {
        auto name = entry.path().filename().string();
        if (name.find(".safetensors") != std::string::npos && name.find(".index") == std::string::npos) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

static std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct ForwardLogits {
    mx::array last_logits = mx::array(0.0f);
    int token_count = 0;
    int argmax_token = -1;
    float argmax_logit = 0.0f;
    int runner_up_token = -1;
    float runner_up_logit = 0.0f;
    float argmax_margin = 0.0f;
    std::string argmax_text;
};

static ForwardLogits run_forward_logits(
    mlx_lm::ModelContext& ctx,
    std::vector<mlx_lm::KVCache>& cache,
    const std::vector<int>& tokens)
{
    if (tokens.empty()) {
        throw std::runtime_error("forward input has zero tokens");
    }

    auto token_array = mx::array(
        tokens.data(), {1, static_cast<int>(tokens.size())}, mx::int32);
    mx::eval(token_array);
    auto out = ctx.call_fn(mlx_lm::LMInput::Text(token_array), &cache, nullptr);
    mx::eval(out.logits);

    auto last = mx::slice(
        out.logits,
        {0, out.logits.shape(1) - 1, 0},
        {1, out.logits.shape(1), out.logits.shape(2)});
    last = mx::reshape(mx::astype(last, mx::float32), {-1});
    mx::eval(last);

    auto next = mx::argmax(last);
    mx::eval(next);
    int next_id = static_cast<int>(next.item<uint32_t>());

    auto data = last.data<float>();
    float runner_up_logit = -std::numeric_limits<float>::infinity();
    int runner_up_token = -1;
    for (int i = 0; i < last.size(); ++i) {
        if (i == next_id) {
            continue;
        }
        float value = data[i];
        if (value > runner_up_logit) {
            runner_up_logit = value;
            runner_up_token = i;
        }
    }

    ForwardLogits result;
    result.last_logits = last;
    result.token_count = static_cast<int>(tokens.size());
    result.argmax_token = next_id;
    result.argmax_logit = data[next_id];
    result.runner_up_token = runner_up_token;
    result.runner_up_logit = runner_up_logit;
    result.argmax_margin = data[next_id] - runner_up_logit;
    result.argmax_text = ctx.decode_fn({next_id});
    return result;
}

static bool run_qforward_compare(
    const std::string& model_path,
    const std::string& prompt,
    int steps,
    float max_diff)
{
    if (!std::getenv("LEMON_MLX_GDN_ENABLE_HIP") &&
        !std::getenv("LEMON_MLX_GDN_DISABLE_HIP")) {
        setenv("LEMON_MLX_GDN_DISABLE_HIP", "1", 1);
    }

    std::cerr << "\n--- TEST: dequantized forward vs native quantized forward ---" << std::endl;
    std::cerr << "[DIAG] qforward prompt_bytes=" << prompt.size() << std::endl;
    std::cerr << "[DIAG] qforward steps=" << steps << std::endl;

    mlx_lm::QuantizedWeightRegistry::instance().clear();
    unsetenv("LEMON_MLX_QWEN35_KEEP_QUANTIZED");
    setenv("LEMON_MLX_DEQUANTIZE_WEIGHTS", "1", 1);
    std::cerr << "[DIAG] Loading dequantized reference model" << std::endl;
    auto deq_ctx = mlx_lm::load_llm(model_path);
    auto prompt_tokens = deq_ctx.encode_fn(prompt);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("prompt encoded to zero tokens");
    }
    auto deq_cache = deq_ctx.new_cache_fn({});
    auto deq = run_forward_logits(deq_ctx, deq_cache, prompt_tokens);

    mlx_lm::QuantizedWeightRegistry::instance().clear();
    unsetenv("LEMON_MLX_DEQUANTIZE_WEIGHTS");
    setenv("LEMON_MLX_QWEN35_KEEP_QUANTIZED", "1", 1);
    std::cerr << "[DIAG] Loading native quantized model" << std::endl;
    auto quant_ctx = mlx_lm::load_llm(model_path);
    auto quant_tokens = quant_ctx.encode_fn(prompt);
    if (quant_tokens != prompt_tokens) {
        throw std::runtime_error("dequantized and quantized tokenizers produced different prompt ids");
    }
    auto quant_cache = quant_ctx.new_cache_fn({});
    auto quant = run_forward_logits(quant_ctx, quant_cache, prompt_tokens);

    float worst_max = 0.0f;
    float worst_mean = 0.0f;
    float min_dequant_margin = std::numeric_limits<float>::infinity();
    float worst_diff_over_margin = 0.0f;
    bool same_argmax = true;
    int first_mismatch_step = -1;
    std::vector<int> next_input;

    for (int step = 0; step < steps; ++step) {
        auto diff = mx::abs(mx::subtract(deq.last_logits, quant.last_logits));
        mx::eval(diff);
        float observed_max = mx::max(diff).item<float>();
        float observed_mean = mx::mean(diff).item<float>();
        worst_max = std::max(worst_max, observed_max);
        worst_mean = std::max(worst_mean, observed_mean);
        min_dequant_margin = std::min(min_dequant_margin, deq.argmax_margin);
        float margin_denom = std::max(deq.argmax_margin, 1e-6f);
        worst_diff_over_margin = std::max(worst_diff_over_margin, observed_max / margin_denom);
        bool step_same = deq.argmax_token == quant.argmax_token;
        if (!step_same && first_mismatch_step < 0) {
            first_mismatch_step = step;
        }
        same_argmax = same_argmax && step_same;

        std::cerr << "[DIAG] QFORWARD_STEP step=" << step
                  << " dequant_token=" << deq.argmax_token
                  << " dequant_logit=" << deq.argmax_logit
                  << " dequant_text=\"" << deq.argmax_text << "\""
                  << " quant_token=" << quant.argmax_token
                  << " quant_logit=" << quant.argmax_logit
                  << " quant_text=\"" << quant.argmax_text << "\""
                  << " dequant_margin=" << deq.argmax_margin
                  << " quant_margin=" << quant.argmax_margin
                  << " max_diff=" << observed_max
                  << " mean_diff=" << observed_mean
                  << " diff_over_dequant_margin=" << (observed_max / margin_denom)
                  << " same_argmax=" << (step_same ? "1" : "0")
                  << std::endl;

        next_input.assign(1, deq.argmax_token);
        deq = run_forward_logits(deq_ctx, deq_cache, next_input);
        quant = run_forward_logits(quant_ctx, quant_cache, next_input);
    }

    bool within_diff = worst_max <= max_diff;
    bool ok = same_argmax && within_diff;

    std::cerr << "[DIAG] QFORWARD_PROMPT_TOKENS = " << prompt_tokens.size() << std::endl;
    std::cerr << "[DIAG] QFORWARD_WORST_MAX_DIFF = " << worst_max << std::endl;
    std::cerr << "[DIAG] QFORWARD_WORST_MEAN_DIFF = " << worst_mean << std::endl;
    std::cerr << "[DIAG] QFORWARD_MIN_DEQUANT_MARGIN = " << min_dequant_margin << std::endl;
    std::cerr << "[DIAG] QFORWARD_WORST_DIFF_OVER_MARGIN = " << worst_diff_over_margin << std::endl;
    std::cerr << "[DIAG] QFORWARD_STATUS = " << (ok ? "ok" : "fail")
              << " same_argmax=" << (same_argmax ? "1" : "0")
              << " first_mismatch_step=" << first_mismatch_step
              << " max_allowed=" << max_diff << std::endl;
    return ok;
}

struct QMatmulTestResult {
    std::string prefix;
    float max_diff_ones = 0.0f;
    float max_diff_random = 0.0f;
    float max_diff_batch = 0.0f;
    float worst_diff = 0.0f;
};

static std::pair<int, int> read_quantization_defaults(const std::string& model_path) {
    int group_size = 32;
    int bits = 4;
    std::ifstream cfg(model_path + "/config.json");
    if (cfg.is_open()) {
        auto j = nlohmann::json::parse(cfg);
        if (j.contains("quantization")) {
            auto& q = j["quantization"];
            if (q.contains("group_size")) group_size = q["group_size"];
            if (q.contains("bits")) bits = q["bits"];
        }
    }
    return {group_size, bits};
}

static std::vector<std::string> discover_qmatmul_prefixes(
    const std::unordered_map<std::string, mx::array>& weights)
{
    std::vector<std::string> prefixes;
    for (const auto& [key, _] : weights) {
        const std::string suffix = ".scales";
        if (key.size() <= suffix.size() ||
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        auto prefix = key.substr(0, key.size() - suffix.size());
        if (weights.count(prefix + ".weight") && weights.count(prefix + ".biases")) {
            prefixes.push_back(prefix);
        }
    }
    std::sort(prefixes.begin(), prefixes.end());
    return prefixes;
}

static QMatmulTestResult run_qmatmul_probe(
    const std::string& prefix,
    const mx::array& w,
    const mx::array& scales,
    const mx::array& biases,
    int group_size,
    int bits,
    bool verbose)
{
    mx::eval(w); mx::eval(scales); mx::eval(biases);

    if (verbose) {
        std::cerr << "[DIAG] prefix=" << prefix << std::endl;
        std::cerr << "[DIAG] weight=" << w.shape() << " " << w.dtype()
                  << " scales=" << scales.shape() << " biases=" << biases.shape() << std::endl;
        print_stats("scales", scales);
        print_stats("biases", biases);
        std::cerr << "[DIAG] bits=" << bits << " group_size=" << group_size << std::endl;
    }

    auto deq = mx::dequantize(w, scales, biases, group_size, bits);
    mx::eval(deq);
    if (verbose) {
        print_stats("dequantized_weight", deq);
        print_vals("dequant row0", deq, 20);
    }

    int hidden = deq.shape(1);
    QMatmulTestResult result;
    result.prefix = prefix;

    auto x1 = mx::ones({1, 1, hidden}, mx::bfloat16);
    mx::eval(x1);
    auto ref1 = mx::matmul(mx::astype(x1, mx::float32), mx::transpose(mx::astype(deq, mx::float32)));
    mx::eval(ref1);
    auto qmm1 = mx::quantized_matmul(x1, w, scales, biases, true, group_size, bits);
    mx::eval(qmm1);
    auto qmm1_f = mx::astype(qmm1, mx::float32);
    mx::eval(qmm1_f);
    auto d1 = mx::abs(mx::subtract(ref1, qmm1_f));
    mx::eval(d1);
    result.max_diff_ones = mx::max(d1).item<float>();

    if (verbose) {
        print_stats("REF(ones)", ref1);
        print_vals("REF(ones)", ref1, 20);
        print_stats("QMM(ones)", qmm1_f);
        print_vals("QMM(ones)", qmm1_f, 20);
        print_stats("DIFF(ones)", d1);
        std::cerr << "[DIAG] MAX DIFF(ones) = " << result.max_diff_ones << std::endl;
    }

    auto x2 = mx::astype(mx::random::normal({1, 1, hidden}), mx::bfloat16);
    mx::eval(x2);
    auto ref2 = mx::matmul(mx::astype(x2, mx::float32), mx::transpose(mx::astype(deq, mx::float32)));
    mx::eval(ref2);
    auto qmm2 = mx::quantized_matmul(x2, w, scales, biases, true, group_size, bits);
    mx::eval(qmm2);
    auto qmm2_f = mx::astype(qmm2, mx::float32);
    mx::eval(qmm2_f);
    auto d2 = mx::abs(mx::subtract(ref2, qmm2_f));
    mx::eval(d2);
    result.max_diff_random = mx::max(d2).item<float>();

    if (verbose) {
        print_stats("DIFF(random)", d2);
        std::cerr << "[DIAG] MAX DIFF(random) = " << result.max_diff_random << std::endl;
    }

    auto x3 = mx::astype(mx::random::normal({1, 3, hidden}), mx::bfloat16);
    mx::eval(x3);
    auto ref3 = mx::matmul(mx::astype(x3, mx::float32), mx::transpose(mx::astype(deq, mx::float32)));
    mx::eval(ref3);
    auto qmm3 = mx::quantized_matmul(x3, w, scales, biases, true, group_size, bits);
    mx::eval(qmm3);
    auto qmm3_f = mx::astype(qmm3, mx::float32);
    mx::eval(qmm3_f);
    auto d3 = mx::abs(mx::subtract(ref3, qmm3_f));
    mx::eval(d3);
    result.max_diff_batch = mx::max(d3).item<float>();

    if (verbose) {
        print_stats("DIFF(batch=3)", d3);
        std::cerr << "[DIAG] MAX DIFF(batch=3) = " << result.max_diff_batch << std::endl;
    }

    result.worst_diff = std::max(result.max_diff_ones,
                         std::max(result.max_diff_random, result.max_diff_batch));
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> [--qmatmul-only] [--qmatmul-scan-all]"
                  << " [--qmatmul-sanitize]"
                  << " [--qmatmul-prefix NAME] [--qmatmul-max-diff N]"
                  << " [--qforward-compare] [--qforward-prompt TEXT]"
                  << " [--qforward-prompt-file PATH] [--qforward-steps N]"
                  << " [--qforward-max-diff N]"
                  << std::endl;
        return argc < 2 ? 1 : 0;
    }

    std::string model_path = argv[1];
    bool qmatmul_only = false;
    bool qmatmul_scan_all = false;
    bool qmatmul_sanitize = false;
    std::string qmatmul_prefix;
    float qmatmul_max_diff = std::numeric_limits<float>::infinity();
    bool qforward_compare = false;
    std::string qforward_prompt = "def add(x, y):\n    ";
    std::string qforward_prompt_file;
    int qforward_steps = 8;
    float qforward_max_diff = std::numeric_limits<float>::infinity();
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--qmatmul-only") {
            qmatmul_only = true;
        } else if (flag == "--qmatmul-scan-all") {
            qmatmul_scan_all = true;
        } else if (flag == "--qmatmul-sanitize") {
            qmatmul_sanitize = true;
        } else if (flag == "--qmatmul-prefix") {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --qmatmul-prefix" << std::endl;
                return 1;
            }
            qmatmul_prefix = argv[++i];
        } else if (flag == "--qmatmul-max-diff") {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --qmatmul-max-diff" << std::endl;
                return 1;
            }
            qmatmul_max_diff = std::stof(argv[++i]);
        } else if (flag == "--qforward-compare") {
            qforward_compare = true;
        } else if (flag == "--qforward-prompt") {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --qforward-prompt" << std::endl;
                return 1;
            }
            qforward_prompt = argv[++i];
        } else if (flag == "--qforward-prompt-file") {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --qforward-prompt-file" << std::endl;
                return 1;
            }
            qforward_prompt_file = argv[++i];
        } else if (flag == "--qforward-steps") {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --qforward-steps" << std::endl;
                return 1;
            }
            qforward_steps = std::stoi(argv[++i]);
        } else if (flag == "--qforward-max-diff") {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --qforward-max-diff" << std::endl;
                return 1;
            }
            qforward_max_diff = std::stof(argv[++i]);
        } else if (flag == "--help" || flag == "-h") {
            std::cerr << "Usage: " << argv[0]
                      << " <model_path> [--qmatmul-only] [--qmatmul-scan-all]"
                      << " [--qmatmul-sanitize]"
                      << " [--qmatmul-prefix NAME] [--qmatmul-max-diff N]"
                      << " [--qforward-compare] [--qforward-prompt TEXT]"
                      << " [--qforward-prompt-file PATH] [--qforward-steps N]"
                      << " [--qforward-max-diff N]"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "unknown flag: " << flag << std::endl;
            return 1;
        }
    }
    if (qmatmul_scan_all && !qmatmul_prefix.empty()) {
        std::cerr << "--qmatmul-scan-all cannot be combined with --qmatmul-prefix" << std::endl;
        return 1;
    }
    if (qforward_compare && qmatmul_only) {
        std::cerr << "--qforward-compare cannot be combined with --qmatmul-only" << std::endl;
        return 1;
    }
    if (!qforward_prompt_file.empty()) {
        qforward_prompt = read_text_file(qforward_prompt_file);
    }
    if (qforward_steps < 1) {
        std::cerr << "--qforward-steps must be >= 1" << std::endl;
        return 1;
    }
    std::cerr << "=== INFERENCE PIPELINE DIAGNOSTICS ===" << std::endl;
    std::cerr << "Loading model: " << model_path << std::endl;

    std::optional<mlx_lm::ModelContext> ctx;
    if ((!qmatmul_only && !qforward_compare) || qmatmul_sanitize) {
        ctx.emplace(mlx_lm::load_llm(model_path));
        std::cerr << "Model loaded.\n" << std::endl;
    } else if (qforward_compare) {
        std::cerr << "Skipping default model load for qforward comparison.\n" << std::endl;
    } else {
        std::cerr << "Skipping full model load for qmatmul-only diagnostic.\n" << std::endl;
    }

    if (qforward_compare) {
        return run_qforward_compare(
            model_path, qforward_prompt, qforward_steps, qforward_max_diff) ? 0 : 2;
    }

    // =========================================================
    // TEST 1: Basic GPU ops sanity check
    // =========================================================
    if (!qmatmul_only) {
        std::cerr << "--- TEST 1: Basic GPU ops ---" << std::endl;
        auto a = mx::ones({4, 4}, mx::float32);
        auto b = mx::full({4, 4}, 2.0f, mx::float32);
        auto c = mx::matmul(a, b);
        mx::eval(c);
        print_stats("matmul(ones, 2*ones) expect=8", c);
        print_vals("matmul result", c, 8);

        // bf16 matmul
        auto a_bf = mx::ones({4, 4}, mx::bfloat16);
        auto b_bf = mx::full({4, 4}, 2.0f, mx::bfloat16);
        auto c_bf = mx::matmul(a_bf, b_bf);
        mx::eval(c_bf);
        print_stats("bf16 matmul expect=8", c_bf);
    }

    // =========================================================
    // TEST 2: Quantized matmul — compare GPU vs dequant+matmul
    // =========================================================
    std::cerr << "\n--- TEST 2: quantized_matmul vs dequant ---" << std::endl;
    float qmatmul_worst_diff = 0.0f;
    {
        auto st_files = find_safetensors(model_path);
        if (st_files.empty()) {
            std::cerr << "[DIAG] No safetensors found!" << std::endl;
            qmatmul_worst_diff = std::numeric_limits<float>::infinity();
        } else {
            auto [group_size, bits] = read_quantization_defaults(model_path);
            std::cerr << "[DIAG] bits=" << bits << " group_size=" << group_size << std::endl;
            bool found_any = false;
            std::vector<std::string> candidate_prefixes;
            if (!qmatmul_prefix.empty()) {
                candidate_prefixes.push_back(qmatmul_prefix);
            } else if (!qmatmul_scan_all) {
                candidate_prefixes = {
                    "model.layers.0.self_attn.q_proj",
                    "model.layers.0.linear_attn.in_proj_qkv",
                    "language_model.model.layers.0.self_attn.q_proj",
                    "language_model.model.layers.0.linear_attn.in_proj_qkv",
                    "language_model.model.layers.0.mlp.gate_proj",
                };
            }

            for (auto& f : st_files) {
                auto [ww, meta] = mx::load_safetensors(f);
                if (qmatmul_sanitize) {
                    if (!ctx.has_value()) {
                        throw std::runtime_error("--qmatmul-sanitize requires a loaded model context");
                    }
                    ww = ctx->sanitize_fn(std::move(ww));
                }
                auto prefixes = qmatmul_scan_all ? discover_qmatmul_prefixes(ww) : candidate_prefixes;
                for (const auto& prefix : prefixes) {
                    auto w_it = ww.find(prefix + ".weight");
                    auto s_it = ww.find(prefix + ".scales");
                    auto b_it = ww.find(prefix + ".biases");
                    if (w_it == ww.end() || s_it == ww.end() || b_it == ww.end()) {
                        continue;
                    }
                    found_any = true;
                    bool verbose = !qmatmul_scan_all;
                    auto result = run_qmatmul_probe(
                        prefix, w_it->second, s_it->second, b_it->second,
                        group_size, bits, verbose);
                    qmatmul_worst_diff = std::max(qmatmul_worst_diff, result.worst_diff);
                    if (qmatmul_scan_all) {
                        std::cerr << "[DIAG] QMATMUL_ROW"
                                  << " prefix=" << result.prefix
                                  << " ones=" << result.max_diff_ones
                                  << " random=" << result.max_diff_random
                                  << " batch3=" << result.max_diff_batch
                                  << " worst=" << result.worst_diff
                                  << " status=" << ((result.worst_diff <= qmatmul_max_diff) ? "ok" : "fail")
                                  << std::endl;
                    } else {
                        std::cerr << "[DIAG] QMATMUL_WORST_DIFF = "
                                  << qmatmul_worst_diff << std::endl;
                        break;
                    }
                }
                if (found_any && !qmatmul_scan_all) break;
            }

            if (!found_any) {
                std::cerr << "[DIAG] quantized weight prefix not found" << std::endl;
                qmatmul_worst_diff = std::numeric_limits<float>::infinity();
            }
        }
    }
    if (qmatmul_only) {
        if (!(qmatmul_worst_diff <= qmatmul_max_diff)) {
            std::cerr << "[DIAG] QMATMUL_STATUS = fail"
                      << " worst_diff=" << qmatmul_worst_diff
                      << " max_allowed=" << qmatmul_max_diff << std::endl;
            return 2;
        }
        std::cerr << "[DIAG] QMATMUL_STATUS = ok"
                  << " worst_diff=" << qmatmul_worst_diff
                  << " max_allowed=" << qmatmul_max_diff << std::endl;
        return 0;
    }

    // =========================================================
    // TEST 3: RMS Norm
    // =========================================================
    std::cerr << "\n--- TEST 3: RMS Norm ---" << std::endl;
    {
        auto x = mx::array({1.0f, 2.0f, 3.0f, 4.0f}, {1, 1, 4});
        auto w = mx::ones({4});
        mx::eval(x); mx::eval(w);
        auto normed = mx::fast::rms_norm(x, w, 1e-6f);
        mx::eval(normed);
        print_stats("rms_norm([1,2,3,4])", normed);
        print_vals("rms_norm([1,2,3,4]) expect≈[.365,.730,1.095,1.461]", normed, 4);

        // bf16
        auto x_bf = mx::astype(mx::random::normal({1, 3, 4096}), mx::bfloat16);
        auto w_bf = mx::ones({4096}, mx::bfloat16);
        mx::eval(x_bf); mx::eval(w_bf);
        auto normed_bf = mx::fast::rms_norm(x_bf, w_bf, 1e-6f);
        mx::eval(normed_bf);
        print_stats("rms_norm(rand bf16 4096)", normed_bf);
    }

    // =========================================================
    // TEST 4: RoPE
    // =========================================================
    std::cerr << "\n--- TEST 4: RoPE ---" << std::endl;
    {
        auto x = mx::ones({1, 1, 1, 128}, mx::bfloat16);
        mx::eval(x);
        auto r0 = mx::fast::rope(x, 128, false, 1000000.0f, 1.0f, 0);
        mx::eval(r0);
        print_stats("rope(ones, off=0)", r0);
        print_vals("rope(ones, off=0)", r0, 20);

        auto r100 = mx::fast::rope(x, 128, false, 1000000.0f, 1.0f, 100);
        mx::eval(r100);
        print_stats("rope(ones, off=100)", r100);
        print_vals("rope(ones, off=100)", r100, 20);
    }

    // =========================================================
    // TEST 5: Full forward pass
    // =========================================================
    std::cerr << "\n--- TEST 5: Full forward pass ---" << std::endl;
    {
        int tok_val = 1;
        auto tok = mx::array(&tok_val, {1, 1}, mx::int32);
        mx::eval(tok);
        auto cache = ctx->new_cache_fn({});
        mlx_lm::LMInput::Text text(tok);
        auto out = ctx->call_fn(text, &cache, nullptr);
        mx::eval(out.logits);
        print_stats("logits(token=1)", out.logits);
        print_vals("logits(token=1)", out.logits, 30);

        // Top-10
        auto last = mx::reshape(out.logits, {-1});
        auto sorted = mx::argsort(last);
        mx::eval(sorted);
        auto lf = mx::astype(last, mx::float32);
        mx::eval(lf);
        auto si = sorted.data<uint32_t>();
        auto ld = lf.data<float>();
        int V = last.shape(0);
        std::cerr << "[DIAG] Top-10:" << std::endl;
        for (int i = 0; i < 10 && i < V; i++) {
            int idx = si[V - 1 - i];
            std::cerr << "  token=" << idx << " logit=" << std::setprecision(4) << ld[idx] << std::endl;
        }

        // Decode step 2
        int next = si[V - 1];
        auto tok2 = mx::array(&next, {1, 1}, mx::int32);
        mx::eval(tok2);
        mlx_lm::LMInput::Text text2(tok2);
        auto out2 = ctx->call_fn(text2, &cache, nullptr);
        mx::eval(out2.logits);
        print_stats("logits(step2)", out2.logits);
    }

    // =========================================================
    // TEST 6: dequantize sanity
    // =========================================================
    std::cerr << "\n--- TEST 6: dequantize() sanity ---" << std::endl;
    {
        uint32_t packed = 0x76543210;
        auto w = mx::array(&packed, {1, 1}, mx::uint32);
        auto s = mx::array({1.0f}, {1, 1});
        auto b = mx::array({0.0f}, {1, 1});
        mx::eval(w); mx::eval(s); mx::eval(b);
        auto deq = mx::dequantize(w, s, b, 8, 4);
        mx::eval(deq);
        print_stats("dequant([0..7],s=1,b=0)", deq);
        print_vals("dequant expect=[0,1,2,3,4,5,6,7]", deq, 8);
    }

    // =========================================================
    // TEST 6b: Warmup pass (same as chat.cpp)
    // =========================================================
    std::cerr << "\n--- TEST 6b: Warmup pass ---" << std::endl;
    {
        mlx_lm::GenerateParameters warmup_params;
        warmup_params.max_tokens = 1;
        warmup_params.temperature = 0.0f;
        auto warmup_cache = ctx->new_cache_fn(warmup_params);
        auto dummy_tokens = mx::reshape(mx::array({1}), {1, 1});
        mlx_lm::LMInput::Text warmup_text(dummy_tokens);
        auto warmup_out = ctx->call_fn(warmup_text, &warmup_cache, nullptr);
        mx::eval(warmup_out.logits);
        print_stats("warmup logits", warmup_out.logits);
        std::cerr << "[DIAG] Warmup complete" << std::endl;
    }

    // =========================================================
    // TEST 7: End-to-end generation with token tracing (POST-WARMUP)
    // =========================================================
    std::cerr << "\n--- TEST 7: Token-level generation trace ---" << std::endl;
    {
        // Encode a prompt the same way chat.cpp does
        std::string prompt = "What is 2+2?";
        auto prompt_tokens = ctx->encode_fn(prompt);
        std::cerr << "[DIAG] encode(\"" << prompt << "\") = [";
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            if (i > 0) std::cerr << ", ";
            std::cerr << prompt_tokens[i];
        }
        std::cerr << "] (" << prompt_tokens.size() << " tokens)" << std::endl;

        // Decode each token back to verify tokenizer round-trip
        std::cerr << "[DIAG] Token-by-token decode:" << std::endl;
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            auto decoded = ctx->decode_fn({prompt_tokens[i]});
            std::cerr << "  token " << prompt_tokens[i] << " -> \"" << decoded << "\"" << std::endl;
        }

        // Also try chat template if available
        if (ctx->apply_chat_template_fn) {
            std::vector<std::unordered_map<std::string, std::string>> messages = {
                {{"role", "user"}, {"content", prompt}}
            };
            if (ctx->template_extra_context) {
                (*ctx->template_extra_context)["enable_thinking"] = false;
            }
            auto tmpl_tokens = ctx->apply_chat_template_fn(messages);
            std::cerr << "[DIAG] Chat template tokens (" << tmpl_tokens.size() << "): [";
            for (size_t i = 0; i < tmpl_tokens.size() && i < 50; i++) {
                if (i > 0) std::cerr << ", ";
                std::cerr << tmpl_tokens[i];
            }
            if (tmpl_tokens.size() > 50) std::cerr << "...";
            std::cerr << "]" << std::endl;

            // Decode the full template back to text
            auto tmpl_text = ctx->decode_fn(tmpl_tokens);
            std::cerr << "[DIAG] Chat template decoded: \"" << tmpl_text << "\"" << std::endl;

            // Now do generation step by step
            auto tok_arr = mx::array(tmpl_tokens.data(), {1, static_cast<int>(tmpl_tokens.size())}, mx::int32);
            mx::eval(tok_arr);
            auto cache = ctx->new_cache_fn({});
            mlx_lm::LMInput::Text text(tok_arr);

            // Prefill
            auto out = ctx->call_fn(text, &cache, nullptr);
            mx::eval(out.logits);
            print_stats("prefill logits", out.logits);

            // Generate 20 tokens with argmax
            std::cerr << "[DIAG] Generating 20 tokens (argmax):" << std::endl;
            std::string full_output;
            for (int step = 0; step < 20; step++) {
                auto last = mx::slice(out.logits, {0, out.logits.shape(1)-1, 0},
                    {1, out.logits.shape(1), out.logits.shape(2)});
                last = mx::reshape(last, {-1});
                auto next_id_arr = mx::argmax(last);
                mx::eval(next_id_arr);
                int next_id = next_id_arr.item<uint32_t>();

                auto decoded = ctx->decode_fn({next_id});
                std::cerr << "  step=" << step << " token=" << next_id << " text=\"" << decoded << "\"" << std::endl;
                full_output += decoded;

                // Feed back
                auto next_tok = mx::array(&next_id, {1, 1}, mx::int32);
                mx::eval(next_tok);
                mlx_lm::LMInput::Text next_text(next_tok);
                out = ctx->call_fn(next_text, &cache, nullptr);
                mx::eval(out.logits);
            }
            std::cerr << "[DIAG] Full output (argmax): \"" << full_output << "\"" << std::endl;

            // Now test with random::categorical (what the sampler actually uses)
            std::cerr << "\n[DIAG] Generating 20 tokens (categorical T=0.7):" << std::endl;
            {
                auto tok_arr2 = mx::array(tmpl_tokens.data(), {1, static_cast<int>(tmpl_tokens.size())}, mx::int32);
                mx::eval(tok_arr2);
                auto cache2 = ctx->new_cache_fn({});
                mlx_lm::LMInput::Text text2(tok_arr2);
                auto out2 = ctx->call_fn(text2, &cache2, nullptr);
                mx::eval(out2.logits);

                std::string cat_output;
                float inv_temp = 1.0f / 0.7f;
                for (int step = 0; step < 20; step++) {
                    auto last2 = mx::slice(out2.logits, {0, out2.logits.shape(1)-1, 0},
                        {1, out2.logits.shape(1), out2.logits.shape(2)});
                    last2 = mx::reshape(last2, {1, -1});
                    auto scaled = mx::multiply(last2, mx::array(inv_temp));
                    auto sampled = mx::random::categorical(scaled);
                    mx::eval(sampled);
                    int next_id2 = sampled.item<uint32_t>();

                    auto decoded2 = ctx->decode_fn({next_id2});
                    std::cerr << "  step=" << step << " token=" << next_id2 << " text=\"" << decoded2 << "\"" << std::endl;
                    cat_output += decoded2;

                    auto next_tok2 = mx::array(&next_id2, {1, 1}, mx::int32);
                    mx::eval(next_tok2);
                    mlx_lm::LMInput::Text next_text2(next_tok2);
                    out2 = ctx->call_fn(next_text2, &cache2, nullptr);
                    mx::eval(out2.logits);
                }
                std::cerr << "[DIAG] Full output (categorical): \"" << cat_output << "\"" << std::endl;
            }
            // Now test via the actual generate_text path (same as chat.cpp raw mode)
            std::cerr << "\n[DIAG] Testing via generate_text (chat.cpp path):" << std::endl;
            {
                auto tok_arr3 = mx::array(tmpl_tokens.data(), {static_cast<int>(tmpl_tokens.size())}, mx::int32);
                mx::eval(tok_arr3);
                mlx_lm::LMInput lm_input(tok_arr3);

                mlx_lm::GenerateParameters gen_params;
                gen_params.max_tokens = 20;
                gen_params.temperature = 0.7f;
                gen_params.top_p = 0.9f;

                std::set<int> eos_set;
                if (ctx->eos_token_ids.has_value()) {
                    for (int id : ctx->eos_token_ids.value()) eos_set.insert(id);
                }

                std::string gen_output;
                auto info = mlx_lm::generate_text(
                    *ctx, lm_input, gen_params, eos_set,
                    [&](const std::string& text, int token) {
                        std::cerr << "  token=" << token << " text=\"" << text << "\"" << std::endl;
                        gen_output += text;
                        return mlx_lm::GenerateDisposition::more;
                    });
                std::cerr << "[DIAG] generate_text output: \"" << gen_output << "\"" << std::endl;
                std::cerr << "[DIAG] " << info.summary() << std::endl;
            }
        } else {
            std::cerr << "[DIAG] No chat template available" << std::endl;
        }
    }

    // =========================================================
    // TEST 8: mx::random::categorical sanity
    // =========================================================
    std::cerr << "\n--- TEST 8: random::categorical ---" << std::endl;
    {
        // Simple test: categorical with known logits
        auto logits = mx::array({-100.0f, -100.0f, 10.0f, -100.0f, -100.0f}, {1, 5});
        mx::eval(logits);
        for (int i = 0; i < 5; i++) {
            auto s = mx::random::categorical(logits);
            mx::eval(s);
            std::cerr << "[DIAG] categorical([..., 10, ...]) = " << s.item<uint32_t>() << " (expect 2)" << std::endl;
        }

        // Test with larger vocab (like the actual model)
        auto big_logits = mx::full({1, 151936}, -100.0f);
        // Set token 17 to high value
        // Can't easily index-set, so create via where
        auto indices = mx::arange(151936);
        auto mask = mx::equal(indices, mx::array(17));
        big_logits = mx::where(mx::reshape(mask, {1, -1}), mx::array(10.0f), big_logits);
        mx::eval(big_logits);
        for (int i = 0; i < 3; i++) {
            auto s = mx::random::categorical(big_logits);
            mx::eval(s);
            std::cerr << "[DIAG] categorical(peak@17, V=151936) = " << s.item<uint32_t>() << " (expect 17)" << std::endl;
        }

        // Test with temperature-scaled real logits
        std::cerr << "[DIAG] Testing categorical with real model logits..." << std::endl;
        {
            auto tok_arr = mx::array(std::vector<int>{151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271}.data(),
                {1, 19}, mx::int32);
            mx::eval(tok_arr);
            auto cache_t8 = ctx->new_cache_fn({});
            mlx_lm::LMInput::Text text_t8(tok_arr);
            auto out_t8 = ctx->call_fn(text_t8, &cache_t8, nullptr);
            mx::eval(out_t8.logits);

            auto last_logits = mx::slice(out_t8.logits, {0, 18, 0}, {1, 19, 151936});
            last_logits = mx::reshape(last_logits, {1, -1});
            mx::eval(last_logits);

            // Argmax for reference
            auto am = mx::argmax(last_logits, -1);
            mx::eval(am);
            std::cerr << "[DIAG] argmax = " << am.item<uint32_t>() << std::endl;

            // Categorical with T=0.7
            float inv_temp = 1.0f / 0.7f;
            auto scaled = mx::multiply(last_logits, mx::array(inv_temp));
            mx::eval(scaled);
            for (int i = 0; i < 5; i++) {
                auto s = mx::random::categorical(scaled);
                mx::eval(s);
                std::cerr << "[DIAG] categorical(T=0.7) = " << s.item<uint32_t>() << std::endl;
            }
        }
    }

    std::cerr << "\n=== DIAGNOSTICS COMPLETE ===" << std::endl;
    return 0;
}
