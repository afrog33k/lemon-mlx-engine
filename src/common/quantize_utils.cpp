// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/quantize_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace mlx_lm {

static std::optional<Quantization> quantization_for_prefix(
    const PerLayerQuantization& plq,
    const std::string& prefix)
{
    auto lookup_override = [&](const std::string& key) -> std::optional<Quantization> {
        auto it = plq.per_layer.find(key);
        if (it == plq.per_layer.end()) return std::nullopt;
        if (it->second.tag == QuantizationOptionTag::Skip) return std::nullopt;
        return it->second.quantization;
    };

    auto quant = lookup_override(prefix);
    if (quant.has_value()) return quant;

    // Some VLM/text models sanitize "language_model.model.*" weights to
    // "model.*" before quantized weight registration. Their config.json
    // keeps the original unsanitized quantization keys.
    if (prefix.find("model.") == 0) {
        quant = lookup_override("language_model." + prefix);
        if (quant.has_value()) return quant;

        quant = lookup_override("model.language_model." + prefix.substr(6));
        if (quant.has_value()) return quant;
    }

    return plq.default_quantization;
}

void register_quantized_weights(
    std::unordered_map<std::string, mx::array>& weights,
    const BaseConfiguration& config,
    const std::unordered_map<std::string, mx::array*>& weight_map)
{
    if (!config.per_layer_quantization.has_value()) return;

    auto& plq = config.per_layer_quantization.value();
    if (!plq.default_quantization.has_value()) return;

    int default_group_size = plq.default_quantization->group_size;
    int default_bits = plq.default_quantization->bits;

    auto& reg = QuantizedWeightRegistry::instance();

    // Collect prefixes that have .scales (indicating quantized weights)
    std::vector<std::string> prefixes;
    for (auto& [key, _] : weights) {
        const std::string suffix = ".scales";
        if (key.size() > suffix.size() &&
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
            auto prefix = key.substr(0, key.size() - suffix.size());
            if (weights.count(prefix + ".weight")) {
                prefixes.push_back(prefix);
            }
        }
    }

    for (auto& prefix : prefixes) {
        auto weight_key = prefix + ".weight";
        auto scales_key = prefix + ".scales";
        auto biases_key = prefix + ".biases";

        // Check per-layer quantization overrides
        int group_size = default_group_size;
        int bits = default_bits;
        auto layer_quant = quantization_for_prefix(plq, prefix);
        if (layer_quant.has_value()) {
            group_size = layer_quant->group_size;
            bits = layer_quant->bits;
        }

        // Get scales and optional biases
        auto& scales = weights.at(scales_key);
        std::optional<mx::array> biases;
        auto biases_it = weights.find(biases_key);
        if (biases_it != weights.end()) {
            biases = biases_it->second;
        }

        // Qwen3/Qwen3.5 have been updated to call embedding_forward(), so they can keep
        // embeddings packed and dequantize only selected token rows. Other
        // model ports still call mx::take() directly on the embedding array,
        // so they need the legacy dequantized embedding tensor.
        bool is_embedding = (prefix.find("embed") != std::string::npos);

        if (is_embedding) {
            if (config.model_type != "qwen3" && config.model_type != "qwen3_5") {
                auto& weight = weights.at(weight_key);
                if (biases.has_value()) {
                    weight = mx::dequantize(weight, scales, biases.value(), group_size, bits);
                } else {
                    weight = mx::dequantize(weight, scales, std::nullopt, group_size, bits);
                }
                weights.erase(scales_key);
                if (biases_it != weights.end()) {
                    weights.erase(biases_it);
                }
                continue;
            }
            auto wm_it = weight_map.find(weight_key);
            if (wm_it == weight_map.end()) {
                continue;
            }
            mx::array* member_ptr = wm_it->second;
            reg.register_weight(member_ptr, scales, biases, group_size, bits, weight_key);
        } else {
            // Find the model's member array address for this weight
            auto wm_it = weight_map.find(weight_key);
            if (wm_it == weight_map.end()) {
                // Not in weight_map — can't register, skip
                continue;
            }
            mx::array* member_ptr = wm_it->second;
            reg.register_weight(member_ptr, scales, biases, group_size, bits, weight_key);
        }

        // Remove scales/biases from the weight map so they don't get
        // loaded as regular weights
        weights.erase(scales_key);
        if (biases_it != weights.end()) {
            weights.erase(biases_it);
        }
    }
}

// Legacy dequantize-at-load-time (kept for reference/fallback)
std::unordered_map<std::string, mx::array> dequantize_weights(
    std::unordered_map<std::string, mx::array> weights,
    const BaseConfiguration& config)
{
    if (!config.per_layer_quantization.has_value()) return weights;

    auto& plq = config.per_layer_quantization.value();
    if (!plq.default_quantization.has_value()) return weights;

    int default_group_size = plq.default_quantization->group_size;
    int default_bits = plq.default_quantization->bits;

    std::vector<std::string> prefixes;
    for (auto& [key, _] : weights) {
        const std::string suffix = ".scales";
        if (key.size() > suffix.size() &&
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
            auto prefix = key.substr(0, key.size() - suffix.size());
            if (weights.count(prefix + ".weight")) {
                prefixes.push_back(prefix);
            }
        }
    }

    for (auto& prefix : prefixes) {
        auto weight_key = prefix + ".weight";
        auto scales_key = prefix + ".scales";
        auto biases_key = prefix + ".biases";

        auto& weight = weights.at(weight_key);
        auto& scales = weights.at(scales_key);

        int group_size = default_group_size;
        int bits = default_bits;
        auto layer_quant = quantization_for_prefix(plq, prefix);
        if (layer_quant.has_value()) {
            group_size = layer_quant->group_size;
            bits = layer_quant->bits;
        }

        auto biases_it = weights.find(biases_key);
        if (biases_it != weights.end()) {
            weight = mx::dequantize(weight, scales, biases_it->second,
                                    group_size, bits);
            weights.erase(biases_it);
        } else {
            weight = mx::dequantize(weight, scales, std::nullopt,
                                    group_size, bits);
        }

        weights.erase(scales_key);
    }

    return weights;
}

} // namespace mlx_lm
