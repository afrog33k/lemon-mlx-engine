// Non-interactive benchmark runner for lemon-mlx-engine.

#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/model_container.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx-lm/llm/llm_factory.h>
#include <mlx/mlx.h>

#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace mx = mlx::core;

struct Args {
    std::string model;
    std::string prompt = "Write a Python function that adds two numbers.";
    std::string prompt_file;
    int max_tokens = 64;
    int runs = 1;
    int ctx_size = 0;
    int kv_bits = 0;
    int kv_group_size = 64;
    float temperature = 0.0f;
    float top_p = 1.0f;
    bool raw = false;
    bool no_think = false;
    bool no_warmup = false;
    int warmup_decode_steps = 1;
    bool trace = false;
    bool print_output = false;
    std::string expect_substring;
    std::string expect_prefix;
    std::string stop_after_substring;
    double min_unique_token_ratio = 0.0;
    double max_token_freq = 1.0;
    int max_token_run = 0;
    int attractor_min_tokens = 16;
};

static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <model_id_or_directory> [options]\n"
              << "  --prompt TEXT          Prompt text\n"
              << "  --prompt-file PATH     Read prompt from file\n"
              << "  --max-tokens N         Decode token limit (default: 64)\n"
              << "  --runs N               Runs after model load/warmup (default: 1)\n"
              << "  --ctx-size N           Pre-allocate KV cache tokens\n"
              << "  --kv-bits N            KV quantization bits, 4 or 8\n"
              << "  --kv-group-size N      KV quantization group size (default: 64)\n"
              << "  --temperature T        Sampling temperature (default: 0)\n"
              << "  --top-p P              Nucleus top-p (default: 1)\n"
              << "  --raw                  Skip chat template\n"
              << "  --no-think             Disable Qwen-style thinking in chat template\n"
              << "  --no-warmup            Skip warmup forwards\n"
              << "  --warmup-decode-steps N\n"
              << "                         Extra one-token decode forwards during warmup (default: 1)\n"
              << "  --trace                Print benchmark phase trace to stderr\n"
              << "  --print-output         Print full generated output to stderr\n"
              << "  --expect-substring S   Fail if generated output does not contain S\n"
              << "  --expect-prefix S      Fail if generated output does not start with S\n"
              << "  --stop-after-substring S\n"
              << "                         Stop generation after the decoded output contains S\n"
              << "  --min-unique-token-ratio R\n"
              << "                         Fail if unique generated tokens / total is below R\n"
              << "  --max-token-freq R     Fail if the most common generated token exceeds R\n"
              << "  --max-token-run N      Fail if any generated token repeats more than N times in a row\n"
              << "  --attractor-min-tokens N\n"
              << "                         Only apply repetition thresholds after N generated tokens\n"
              << "  --fail-on-attractor    Apply coherence-gate-style repetition thresholds\n";
}

static Args parse_args(int argc, char** argv) {
    Args args;
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        usage(argv[0]);
        std::exit(argc < 2 ? 1 : 0);
    }
    args.model = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (flag == "--prompt") {
            args.prompt = need_value("--prompt");
        } else if (flag == "--prompt-file") {
            args.prompt_file = need_value("--prompt-file");
        } else if (flag == "--max-tokens") {
            args.max_tokens = std::stoi(need_value("--max-tokens"));
        } else if (flag == "--runs") {
            args.runs = std::stoi(need_value("--runs"));
        } else if (flag == "--ctx-size") {
            args.ctx_size = std::stoi(need_value("--ctx-size"));
        } else if (flag == "--kv-bits") {
            args.kv_bits = std::stoi(need_value("--kv-bits"));
        } else if (flag == "--kv-group-size") {
            args.kv_group_size = std::stoi(need_value("--kv-group-size"));
        } else if (flag == "--temperature") {
            args.temperature = std::stof(need_value("--temperature"));
        } else if (flag == "--top-p") {
            args.top_p = std::stof(need_value("--top-p"));
        } else if (flag == "--raw") {
            args.raw = true;
        } else if (flag == "--no-think") {
            args.no_think = true;
        } else if (flag == "--no-warmup") {
            args.no_warmup = true;
        } else if (flag == "--warmup-decode-steps") {
            args.warmup_decode_steps = std::stoi(need_value("--warmup-decode-steps"));
        } else if (flag == "--trace") {
            args.trace = true;
        } else if (flag == "--print-output") {
            args.print_output = true;
        } else if (flag == "--expect-substring") {
            args.expect_substring = need_value("--expect-substring");
        } else if (flag == "--expect-prefix") {
            args.expect_prefix = need_value("--expect-prefix");
        } else if (flag == "--stop-after-substring") {
            args.stop_after_substring = need_value("--stop-after-substring");
        } else if (flag == "--min-unique-token-ratio") {
            args.min_unique_token_ratio = std::stod(need_value("--min-unique-token-ratio"));
        } else if (flag == "--max-token-freq") {
            args.max_token_freq = std::stod(need_value("--max-token-freq"));
        } else if (flag == "--max-token-run") {
            args.max_token_run = std::stoi(need_value("--max-token-run"));
        } else if (flag == "--attractor-min-tokens") {
            args.attractor_min_tokens = std::stoi(need_value("--attractor-min-tokens"));
        } else if (flag == "--fail-on-attractor") {
            args.min_unique_token_ratio = 0.30;
            args.max_token_freq = 0.40;
            args.max_token_run = 16;
        } else {
            throw std::runtime_error("unknown flag: " + flag);
        }
    }
    if (args.runs < 1) {
        args.runs = 1;
    }
    if (args.warmup_decode_steps < 0) {
        args.warmup_decode_steps = 0;
    }
    return args;
}

static std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open prompt file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static uint64_t fnv1a64(const std::string& text) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : text) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t fnv1a64_tokens(const std::vector<int>& tokens) {
    uint64_t h = 1469598103934665603ULL;
    for (int token : tokens) {
        uint32_t value = static_cast<uint32_t>(token);
        for (int byte = 0; byte < 4; ++byte) {
            h ^= static_cast<unsigned char>((value >> (byte * 8)) & 0xff);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

static bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

static std::string preview(const std::string& text, size_t limit = 240) {
    std::string out;
    out.reserve(std::min(text.size(), limit));
    for (char c : text) {
        if (out.size() >= limit) {
            break;
        }
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

struct TokenStats {
    int total = 0;
    int unique = 0;
    int max_count = 0;
    int max_run = 0;
    double unique_ratio = 1.0;
    double max_freq = 0.0;
};

static TokenStats token_stats(const std::vector<int>& tokens) {
    TokenStats stats;
    stats.total = static_cast<int>(tokens.size());
    if (tokens.empty()) {
        return stats;
    }

    std::unordered_map<int, int> counts;
    counts.reserve(tokens.size());
    int current_token = tokens.front();
    int current_run = 0;
    for (int token : tokens) {
        int count = ++counts[token];
        stats.max_count = std::max(stats.max_count, count);
        if (token == current_token) {
            ++current_run;
        } else {
            current_token = token;
            current_run = 1;
        }
        stats.max_run = std::max(stats.max_run, current_run);
    }

    stats.unique = static_cast<int>(counts.size());
    stats.unique_ratio = static_cast<double>(stats.unique) / static_cast<double>(stats.total);
    stats.max_freq = static_cast<double>(stats.max_count) / static_cast<double>(stats.total);
    return stats;
}

static std::string status_for_run(
    const Args& args,
    const std::string& output,
    const TokenStats& stats)
{
    std::vector<std::string> failures;
    if (!args.expect_substring.empty() &&
        output.find(args.expect_substring) == std::string::npos) {
        failures.push_back("missing_expected_substring");
    }
    if (!args.expect_prefix.empty() &&
        output.rfind(args.expect_prefix, 0) != 0) {
        failures.push_back("missing_expected_prefix");
    }
    if (stats.total == 0) {
        failures.push_back("zero_tokens");
    }
    if (stats.total >= args.attractor_min_tokens) {
        if (args.min_unique_token_ratio > 0.0 &&
            stats.unique_ratio < args.min_unique_token_ratio) {
            failures.push_back("low_unique_ratio");
        }
        if (args.max_token_freq < 1.0 &&
            stats.max_freq > args.max_token_freq) {
            failures.push_back("high_token_freq");
        }
        if (args.max_token_run > 0 && stats.max_run > args.max_token_run) {
            failures.push_back("long_token_run");
        }
    }

    if (failures.empty()) {
        return "ok";
    }
    std::ostringstream ss;
    for (size_t i = 0; i < failures.size(); ++i) {
        if (i) {
            ss << ',';
        }
        ss << failures[i];
    }
    return ss.str();
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (!args.prompt_file.empty()) {
            args.prompt = read_file(args.prompt_file);
        }
        args.trace = args.trace || env_enabled("LEMON_MLX_BENCH_TRACE");

        auto trace_start = std::chrono::steady_clock::now();
        auto trace = [&](const std::string& message) {
            if (!args.trace) {
                return;
            }
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - trace_start).count();
            std::cerr << "[bench-trace +" << std::fixed << std::setprecision(3)
                      << elapsed << "s] " << message << std::endl;
        };

        trace("load_llm begin: " + args.model);
        auto ctx = mlx_lm::load_llm(args.model);
        trace("load_llm done");
        if (ctx.template_extra_context) {
            (*ctx.template_extra_context)["enable_thinking"] = !args.no_think;
        }

        mlx_lm::GenerateParameters params;
        params.max_tokens = args.max_tokens;
        params.temperature = args.temperature;
        params.top_p = args.top_p;
        if (args.ctx_size > 0) {
            params.ctx_size = args.ctx_size;
        }
        if (args.kv_bits > 0) {
            params.kv_bits = args.kv_bits;
            params.kv_group_size = args.kv_group_size;
        }

        trace("tokenize begin");
        std::vector<int> tokens;
        if (!args.raw && ctx.apply_chat_template_fn) {
            std::vector<std::unordered_map<std::string, std::string>> messages = {
                {{"role", "user"}, {"content", args.prompt}},
            };
            tokens = ctx.apply_chat_template_fn(messages);
        } else {
            tokens = ctx.encode_fn(args.prompt);
        }
        trace("tokenize done: " + std::to_string(tokens.size()) + " tokens");

        if (!args.no_warmup) {
            trace("warmup cache begin");
            mlx_lm::GenerateParameters warmup_params;
            warmup_params.max_tokens = 1;
            warmup_params.temperature = 0.0f;
            auto warmup_cache = ctx.new_cache_fn(warmup_params);
            trace("warmup call_fn begin");
            int warmup_len = std::max(1, static_cast<int>(tokens.size()));
            std::vector<int> dummy_tokens(static_cast<size_t>(warmup_len), 1);
            auto dummy = mx::array(dummy_tokens.data(), {1, warmup_len}, mx::int32);
            auto out = ctx.call_fn(mlx_lm::LMInput::Text(dummy), &warmup_cache, nullptr);
            trace("warmup call_fn returned; eval logits begin");
            mx::eval(out.logits);
            trace("warmup eval logits done");
            for (int step = 0; step < args.warmup_decode_steps; ++step) {
                trace("warmup decode call_fn begin: step " + std::to_string(step + 1));
                int dummy_token = 1;
                auto decode_dummy = mx::array(&dummy_token, {1, 1}, mx::int32);
                auto decode_out = ctx.call_fn(
                    mlx_lm::LMInput::Text(decode_dummy), &warmup_cache, nullptr);
                trace("warmup decode call_fn returned; eval logits begin");
                mx::eval(decode_out.logits);
                trace("warmup decode eval logits done");
            }
        } else {
            trace("warmup skipped");
        }
        mlx_lm::QuantizedWeightRegistry::instance().reset_profile();
        trace("profile counters reset after warmup");

        std::set<int> eos;
        if (ctx.eos_token_ids) {
            eos.insert(ctx.eos_token_ids->begin(), ctx.eos_token_ids->end());
        }

        auto token_array = mx::array(tokens.data(), {static_cast<int>(tokens.size())}, mx::int32);
        mlx_lm::LMInput input(token_array);

        std::cout << "model\tprompt_hash\tprompt_tokens\tmax_tokens\trun\tdecode_tokens\t"
                  << "prompt_tok_s\tdecode_tok_s\tprompt_s\tdecode_s\t"
                  << "unique_token_ratio\tmax_token_freq\tmax_token_run\toutput_hash\tstatus\tpreview\t"
                  << "token_hash\n";

        bool any_failed = false;
        for (int run = 1; run <= args.runs; ++run) {
            std::string output;
            std::vector<int> generated_tokens;
            mlx_lm::NaiveStreamingDetokenizer detokenizer;
            trace("generate run " + std::to_string(run) + " begin");
            auto info = mlx_lm::generate(
                ctx,
                input,
                params,
                eos,
                [&](int token) {
                    generated_tokens.push_back(token);
                    detokenizer.append(token);
                    if (auto text = detokenizer.next(ctx.decode_fn)) {
                        output += *text;
                        if (!args.stop_after_substring.empty() &&
                            output.find(args.stop_after_substring) != std::string::npos) {
                            return mlx_lm::GenerateDisposition::stop;
                        }
                    }
                    return mlx_lm::GenerateDisposition::more;
                });
            trace("generate run " + std::to_string(run) + " done");

            auto stats = token_stats(generated_tokens);
            auto status = status_for_run(args, output, stats);
            any_failed = any_failed || status != "ok";
            if (args.print_output) {
                std::cerr << "[bench-output run=" << run << "] " << output << std::endl;
            }

            std::cout << args.model << '\t'
                      << std::hex << std::setw(16) << std::setfill('0') << fnv1a64(args.prompt)
                      << std::dec << std::setfill(' ') << '\t'
                      << info.prompt_token_count << '\t'
                      << args.max_tokens << '\t'
                      << run << '\t'
                      << info.generation_token_count << '\t'
                      << info.prompt_tokens_per_second() << '\t'
                      << info.tokens_per_second() << '\t'
                      << info.prompt_time << '\t'
                      << info.generation_time << '\t'
                      << stats.unique_ratio << '\t'
                      << stats.max_freq << '\t'
                      << stats.max_run << '\t'
                      << std::hex << std::setw(16) << std::setfill('0') << fnv1a64(output)
                      << std::dec << std::setfill(' ') << '\t'
                      << status << '\t'
                      << preview(output) << '\t'
                      << std::hex << std::setw(16) << std::setfill('0') << fnv1a64_tokens(generated_tokens)
                      << std::dec << std::setfill(' ') << '\n';
        }

        return any_failed ? 2 : 0;
    } catch (const std::exception& e) {
        std::cerr << "bench error: " << e.what() << std::endl;
        return 1;
    }
}
