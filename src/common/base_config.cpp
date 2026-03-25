// Copyright © 2025 Apple Inc. — Ported to C++

#include <mlx-lm/common/base_config.h>
#include <set>
#include <string>

namespace mlx_lm {

BaseConfiguration parse_base_configuration(const nlohmann::json& config) {
    BaseConfiguration base;
    base.model_type = config.value("model_type", std::string(""));

    if (config.contains("eos_token_id")) {
        IntOrIntArray eos;
        from_json(config["eos_token_id"], eos);
        base.eos_token_ids = eos;
    }

    if (config.contains("quantization")) {
        const auto& q_json = config["quantization"];

        Quantization default_quant;
        default_quant.group_size = q_json.value("group_size", 64);
        default_quant.bits = q_json.value("bits", 4);

        PerLayerQuantization plq;
        plq.default_quantization = default_quant;

        // Known non-layer keys to skip
        static const std::set<std::string> skip_keys = {
            "group_size", "bits", "mode",
            "quant_method", "linear_class", "quantization_mode"
        };

        for (auto& [key, value] : q_json.items()) {
            if (skip_keys.count(key)) continue;

            if (value.is_boolean()) {
                if (!value.get<bool>()) {
                    plq.per_layer[key] = QuantizationOption::skip();
                }
            } else if (value.is_object()) {
                Quantization layer_quant;
                from_json(value, layer_quant);
                plq.per_layer[key] = QuantizationOption::quantize(layer_quant);
            }
        }

        base.per_layer_quantization = plq;
    }

    return base;
}

} // namespace mlx_lm
