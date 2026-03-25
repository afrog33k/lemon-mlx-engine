// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <variant>

namespace mlx_lm {

// StringOrNumber — a value that can be either a string or a float in JSON.
// Used for rope_scaling and similar mixed-type config fields.
struct StringOrNumber {
    std::variant<std::string, float> value;

    bool is_string() const { return std::holds_alternative<std::string>(value); }
    bool is_float() const { return std::holds_alternative<float>(value); }

    const std::string& as_string() const { return std::get<std::string>(value); }
    float as_float() const { return std::get<float>(value); }

    static StringOrNumber from_string(const std::string& s) { return {s}; }
    static StringOrNumber from_float(float f) { return {f}; }

    bool operator==(const StringOrNumber& other) const { return value == other.value; }
};

inline void from_json(const nlohmann::json& j, StringOrNumber& v) {
    if (j.is_string()) {
        v.value = j.get<std::string>();
    } else if (j.is_number()) {
        v.value = j.get<float>();
    }
}

inline void to_json(nlohmann::json& j, const StringOrNumber& v) {
    if (v.is_string()) {
        j = v.as_string();
    } else {
        j = v.as_float();
    }
}

} // namespace mlx_lm
