// Copyright © 2025 — Ported to C++
// QuantizedLinear — quantized weight storage and registry-based dispatch.
//
// Matches Swift's QuantizedLinear: keeps weights packed as uint32 and uses
// mx::quantized_matmul at inference time instead of dequantizing at load time.
#pragma once

#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlx_lm {

// Quantization metadata for a single weight.
struct QuantizationInfo {
    mlx::core::array scales;
    std::optional<mlx::core::array> biases;
    int group_size;
    int bits;
    std::string name;
};

// Global registry mapping weight array addresses to quantization metadata.
//
// At load time, quantized weights are NOT dequantized. Instead, the packed
// uint32 weight is stored in the model's member array as-is, and the
// corresponding scales/biases/group_size/bits are registered here.
//
// At inference time, linear_forward() checks this registry: if the weight
// has an entry, it uses mx::quantized_matmul; otherwise, regular mx::matmul.
class QuantizedWeightRegistry {
public:
    static QuantizedWeightRegistry& instance() {
        static auto* reg = new QuantizedWeightRegistry();
        return *reg;
    }

    void register_weight(const mlx::core::array* weight_ptr,
                         mlx::core::array scales,
                         std::optional<mlx::core::array> biases,
                         int group_size, int bits,
                         std::string name = {}) {
        registry_.insert_or_assign(
            weight_ptr,
            QuantizationInfo{
                std::move(scales),
                std::move(biases),
                group_size,
                bits,
                std::move(name)});
    }

    const QuantizationInfo* find(const mlx::core::array* weight_ptr) const {
        auto it = registry_.find(weight_ptr);
        return (it != registry_.end()) ? &it->second : nullptr;
    }

    void clear() { registry_.clear(); }
    size_t size() const { return registry_.size(); }

    void reset_profile() {
        quantized_linear_shapes_.clear();
        dense_linear_shapes_.clear();
        quantized_embedding_shapes_.clear();
        dense_embedding_shapes_.clear();
    }

    void record_linear(
        bool quantized,
        const std::string& name,
        const mlx::core::array& x,
        const mlx::core::array& w)
    {
        if (!profile_enabled_) return;
        auto& table = quantized ? quantized_linear_shapes_ : dense_linear_shapes_;
        table[profile_key(name, x, w)].calls++;
    }

    void record_embedding(
        bool quantized,
        const std::string& name,
        const mlx::core::array& w,
        const mlx::core::array& indices)
    {
        if (!profile_enabled_) return;
        auto& table = quantized ? quantized_embedding_shapes_ : dense_embedding_shapes_;
        table[profile_key(name, indices, w)].calls++;
    }

    template <typename Fn>
    mlx::core::array profile_array_eval(
        bool quantized,
        bool embedding,
        const std::string& name,
        const mlx::core::array& lhs,
        const mlx::core::array& rhs,
        Fn&& fn)
    {
        auto result = fn();
        if (!timing_enabled_) return result;

        // Diagnostic timing mode intentionally materializes each op independently.
        mlx::core::synchronize();
        auto start = std::chrono::steady_clock::now();
        mlx::core::eval(result);
        mlx::core::synchronize();
        auto stop = std::chrono::steady_clock::now();
        const auto elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

        auto& table = embedding
            ? (quantized ? quantized_embedding_shapes_ : dense_embedding_shapes_)
            : (quantized ? quantized_linear_shapes_ : dense_linear_shapes_);
        table[profile_key(name, lhs, rhs)].total_ns += static_cast<uint64_t>(elapsed_ns);
        return result;
    }

private:
    struct QuantProfileStats {
        size_t calls = 0;
        uint64_t total_ns = 0;
    };

    QuantizedWeightRegistry()
        : profile_enabled_(std::getenv("LEMON_MLX_QUANT_PROFILE") != nullptr
                           || std::getenv("LEMON_MLX_QUANT_PROFILE_TIMING") != nullptr),
          timing_enabled_(std::getenv("LEMON_MLX_QUANT_PROFILE_TIMING") != nullptr)
    {
        if (profile_enabled_ || timing_enabled_) {
            std::atexit([]() {
                QuantizedWeightRegistry::instance().print_profile();
            });
        }
    }

    static std::string shape_to_string(const mlx::core::array& a) {
        std::ostringstream ss;
        ss << '[';
        for (int i = 0; i < a.ndim(); ++i) {
            if (i) ss << 'x';
            ss << a.shape(i);
        }
        ss << ']';
        return ss.str();
    }

    static std::string shape_key(
        const mlx::core::array& lhs,
        const mlx::core::array& rhs)
    {
        return shape_to_string(lhs) + " @ " + shape_to_string(rhs);
    }

    static std::string profile_key(
        const std::string& name,
        const mlx::core::array& lhs,
        const mlx::core::array& rhs)
    {
        const std::string weight_name = name.empty() ? "<unknown>" : name;
        return weight_name + "\t" + shape_key(lhs, rhs);
    }

    static std::pair<std::string, std::string> split_profile_key(const std::string& key) {
        auto tab = key.find('\t');
        if (tab == std::string::npos) return {"<unknown>", key};
        return {key.substr(0, tab), key.substr(tab + 1)};
    }

    void print_table(
        const char* name,
        const std::unordered_map<std::string, QuantProfileStats>& table) const
    {
        size_t total = 0;
        uint64_t total_ns = 0;
        for (const auto& [_, stats] : table) {
            total += stats.calls;
            total_ns += stats.total_ns;
        }
        std::cerr << "[lemon-mlx] quant_profile " << name
                  << " total=" << total
                  << " distinct_shapes=" << table.size();
        if (timing_enabled_) {
            std::cerr << " total_ms=" << (static_cast<double>(total_ns) / 1e6);
        }
        std::cerr << '\n';

        std::vector<std::pair<std::string, QuantProfileStats>> rows(table.begin(), table.end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            if (a.second.calls != b.second.calls) return a.second.calls > b.second.calls;
            return a.first < b.first;
        });

        for (const auto& [key, stats] : rows) {
            auto [weight_name, shape] = split_profile_key(key);
            std::cerr << "[lemon-mlx] quant_profile " << name
                      << " calls=" << stats.calls
                      << " shape=\"" << shape << "\"";
            if (timing_enabled_) {
                const double total_ms = static_cast<double>(stats.total_ns) / 1e6;
                const double avg_us = stats.calls
                    ? (static_cast<double>(stats.total_ns) / 1e3 / static_cast<double>(stats.calls))
                    : 0.0;
                std::cerr << " total_ms=" << total_ms
                          << " avg_us=" << avg_us;
            }
            std::cerr << " weight=\"" << weight_name << "\"";
            std::cerr << '\n';
        }
    }

    void print_profile() const {
        print_table("linear_quantized", quantized_linear_shapes_);
        print_table("linear_dense", dense_linear_shapes_);
        print_table("embedding_quantized", quantized_embedding_shapes_);
        print_table("embedding_dense", dense_embedding_shapes_);
    }

    std::unordered_map<const mlx::core::array*, QuantizationInfo> registry_;
    bool profile_enabled_ = false;
    bool timing_enabled_ = false;
    std::unordered_map<std::string, QuantProfileStats> quantized_linear_shapes_;
    std::unordered_map<std::string, QuantProfileStats> dense_linear_shapes_;
    std::unordered_map<std::string, QuantProfileStats> quantized_embedding_shapes_;
    std::unordered_map<std::string, QuantProfileStats> dense_embedding_shapes_;
};

// Quantization-aware linear forward pass.
//
// If the weight is registered as quantized, uses mx::quantized_matmul.
// Otherwise, falls back to regular mx::matmul(x, transpose(w)).
// Matches Swift's QuantizedLinear.callAsFunction / Linear.callAsFunction.
//
// Each model's static linear_fwd() should delegate to this function.
inline mlx::core::array linear_forward(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const mlx::core::array* bias = nullptr)
{
    namespace mx = mlx::core;

    auto* qi = QuantizedWeightRegistry::instance().find(&w);
    auto& registry = QuantizedWeightRegistry::instance();
    const std::string name = qi ? qi->name : std::string();
    registry.record_linear(qi != nullptr, name, x, w);

    if (qi) {
        return registry.profile_array_eval(qi != nullptr, false, name, x, w, [&]() {
            auto result = mx::quantized_matmul(
              x, w, qi->scales, qi->biases,
              /*transpose=*/true, qi->group_size, qi->bits);
            if (bias) result = mx::add(result, *bias);
            return result;
        });
    }

    // Non-quantized path: use fused addmm when bias is present.
    // addmm computes D = beta*C + alpha*(A @ B) in a single kernel.
    return registry.profile_array_eval(qi != nullptr, false, name, x, w, [&]() {
        if (bias) {
            return mx::addmm(*bias, x, mx::transpose(w));
        }
        return mx::matmul(x, mx::transpose(w));
    });
}

inline mlx::core::array embedding_forward(
    const mlx::core::array& w,
    const mlx::core::array& indices)
{
    namespace mx = mlx::core;

    auto* qi = QuantizedWeightRegistry::instance().find(&w);
    auto& registry = QuantizedWeightRegistry::instance();
    const std::string name = qi ? qi->name : std::string();
    registry.record_embedding(qi != nullptr, name, w, indices);
    if (!qi) {
        return registry.profile_array_eval(qi != nullptr, true, name, indices, w, [&]() {
            return mx::take(w, indices, 0);
        });
    }

    return registry.profile_array_eval(qi != nullptr, true, name, indices, w, [&]() {
        auto packed_rows = mx::take(w, indices, 0);
        auto scale_rows = mx::take(qi->scales, indices, 0);
        std::optional<mx::array> bias_rows;
        if (qi->biases.has_value()) {
            bias_rows = mx::take(qi->biases.value(), indices, 0);
        }
        return mx::dequantize(
            packed_rows, scale_rows, bias_rows, qi->group_size, qi->bits);
    });
}

} // namespace mlx_lm
